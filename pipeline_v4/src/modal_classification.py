
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from pyproj import Transformer
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

_TRANSFORMER_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)


class ServingContractError(ValueError):
    """Raised when production cannot reproduce the classifier training inputs."""


@dataclass(frozen=True)
class TripServingContext:
    """GPS-level inputs that routing rows cannot faithfully represent.

    Snap distances follow the training generator contract: one minimum candidate
    distance per effective GPS ping, with the configured search buffer used when
    no candidate exists.
    """

    raw_ping_count: int
    effective_ping_count: int
    snap_dist_drive: tuple[float, ...]
    snap_dist_walk: tuple[float, ...]

    @property
    def pct_pings_conserved(self) -> float:
        return (
            100.0 * self.effective_ping_count / self.raw_ping_count
            if self.raw_ping_count > 0 else 0.0
        )

    @classmethod
    def from_trip(cls, trip, raw_ping_count, drive_buffer_m=150.0, walk_buffer_m=50.0):
        def minima(column, fallback):
            if column not in trip:
                raise ServingContractError(f"Falta {column!r} en los pings efectivos del viaje.")
            values = []
            for candidates in trip[column]:
                candidates = list(candidates) if candidates is not None else []
                values.append(float(min(candidates)) if candidates else float(fallback))
            return tuple(values)

        return cls(
            raw_ping_count=int(raw_ping_count),
            effective_ping_count=int(len(trip)),
            snap_dist_drive=minima("drive_dists", drive_buffer_m),
            snap_dist_walk=minima("walk_dists", walk_buffer_m),
        )

class InfrastructureProximityCache:
    """Run-scoped, bounded cache for exact coordinate proximity flags."""

    def __init__(self, max_entries=200_000):
        self.max_entries = int(max_entries)
        self._values = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get_many(self, keys):
        found = {}
        with self._lock:
            for key in keys:
                if key in self._values:
                    self.hits += 1
                    self._values.move_to_end(key)
                    found[key] = self._values[key]
                else:
                    self.misses += 1
        return found

    def put_many(self, values):
        with self._lock:
            for key, value in values.items():
                self._values[key] = value
                self._values.move_to_end(key)
                if len(self._values) > self.max_entries:
                    self._values.popitem(last=False)
                    self.evictions += 1

    def stats(self):
        with self._lock:
            return {
                "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions, "entries": len(self._values),
                "max_entries": self.max_entries,
            }


def calcular_cercania_infraestructura(
    df, subway_routes, bus_routes, *, proximity_cache: InfrastructureProximityCache | None = None,
):
    """
    Calcula la proximidad espacial (distancia métrica exacta) de cada ping GPS
    a las líneas físicas del metro y rutas de autobús oficiales.
    """
    if df.empty:
        return df.copy()
        
    coordinate_keys = list(zip(df["longitude"].tolist(), df["latitude"].tolist()))
    unique_coordinates = list(dict.fromkeys(coordinate_keys))
    namespace = (id(subway_routes), id(bus_routes))
    cache_keys = [(*namespace, *coordinate) for coordinate in unique_coordinates]
    known_by_cache_key = proximity_cache.get_many(cache_keys) if proximity_cache is not None else {}
    known = {
        coordinate: known_by_cache_key[cache_key]
        for coordinate, cache_key in zip(unique_coordinates, cache_keys)
        if cache_key in known_by_cache_key
    }
    missing_keys = [key for key in unique_coordinates if key not in known]

    if not missing_keys:
        df["near_subway_line"] = [known[key][0] for key in coordinate_keys]
        df["near_bus_route"] = [known[key][1] for key in coordinate_keys]
        return df

    # 1. Proyectar temporalmente sólo coordenadas exactas aún no evaluadas usando UTM directo
    missing_lons = np.array([k[0] for k in missing_keys])
    missing_lats = np.array([k[1] for k in missing_keys])
    xs, ys = _TRANSFORMER_TO_UTM.transform(missing_lons, missing_lats)
    pts_geom = shapely.points(xs, ys)
    
    subway_proj = subway_routes if subway_routes.crs == "EPSG:32614" else subway_routes.to_crs("EPSG:32614")
    bus_proj = bus_routes if bus_routes.crs == "EPSG:32614" else bus_routes.to_crs("EPSG:32614")
    
    RADIO_METRO = 50.0
    RADIO_BUS = 20.0
    
    # 2. Búsqueda Espacial Exacta por STRtree (bounding box broad-phase) + Distancia Métrica Shapely Exacta
    near_metro = np.zeros(len(pts_geom), dtype=int)
    if subway_proj is not None and len(subway_proj) > 0:
        metro_boxes = shapely.box(xs - RADIO_METRO, ys - RADIO_METRO, xs + RADIO_METRO, ys + RADIO_METRO)
        metro_matches = subway_proj.sindex.query(metro_boxes, predicate="intersects")
        if len(metro_matches) > 0 and len(metro_matches[0]) > 0:
            pt_indices = metro_matches[0]
            geom_indices = metro_matches[1]
            subway_geoms = subway_proj.geometry.to_numpy()[geom_indices]
            dists_metro = shapely.distance(pts_geom[pt_indices], subway_geoms)
            valid_m = dists_metro < RADIO_METRO
            if np.any(valid_m):
                near_metro[pt_indices[valid_m]] = 1
        
    near_bus = np.zeros(len(pts_geom), dtype=int)
    if bus_proj is not None and len(bus_proj) > 0:
        bus_boxes = shapely.box(xs - RADIO_BUS, ys - RADIO_BUS, xs + RADIO_BUS, ys + RADIO_BUS)
        bus_matches = bus_proj.sindex.query(bus_boxes, predicate="intersects")
        if len(bus_matches) > 0 and len(bus_matches[0]) > 0:
            pt_indices = bus_matches[0]
            geom_indices = bus_matches[1]
            bus_geoms = bus_proj.geometry.to_numpy()[geom_indices]
            dists_bus = shapely.distance(pts_geom[pt_indices], bus_geoms)
            valid_b = dists_bus < RADIO_BUS
            if np.any(valid_b):
                near_bus[pt_indices[valid_b]] = 1

    computed = {
        key: (int(metro_value), int(bus_value))
        for key, metro_value, bus_value in zip(missing_keys, near_metro, near_bus)
    }
    if proximity_cache is not None:
        proximity_cache.put_many({(*namespace, *key): value for key, value in computed.items()})
    values = {**known, **computed}
    df['near_subway_line'] = [values[key][0] for key in coordinate_keys]
    df['near_bus_route'] = [values[key][1] for key in coordinate_keys]
    
    return df

class PriorModeClassifier:
    """
    Clasificador de Prior Modal y Heurísticas de Filtrado.
    Calcula priors heurísticos y realiza el descarte/poda de hipótesis imposibles
    utilizando variables baratas derivadas directamente del GPS crudo (sin ruteo).
    """
    def __init__(self, max_walk_speed=22.0, max_walk_dist=15.0):
        self.max_walk_speed = max_walk_speed
        self.max_walk_dist = max_walk_dist


    def prune_impossible_hypotheses(self, df_trip, near_subway, near_bus):
        """
        Remove physically impossible hypotheses before routing.
        
        Return the simplified candidate modes that require routing:
        - 'Caminar' when physically feasible;
        - 'Metro' when physically feasible and near the Metro network;
        - 'Carro' as the shared road-motorized hypothesis.
        
        Bus is not routed independently because it uses the same road graph.
        The later serving step separates Carro from Bus when the road hypothesis wins.
        """
        max_speed = df_trip['Speed [km/h]'].max()
        total_dist_km = df_trip['dis lineal [m]'].sum() / 1000.0
        
        candidates = []
        
        # 1. Walking feasibility.
        if max_speed <= self.max_walk_speed and total_dist_km <= self.max_walk_dist:
            candidates.append('Caminar')
            
        # 2. Metro feasibility.
        if near_subway.any() and total_dist_km > 1.0:
            candidates.append('Metro')
            
        # 3. Shared road-motorized feasibility (Carro / Bus).
        if max_speed > 3.0 or total_dist_km > 0.5:
            candidates.append('Carro')
            
        # Always return at least one candidate.
        if not candidates:
            candidates = ['Carro']
            
        return candidates

    def predict_candidates(self, df_trip, near_subway, near_bus):
        """Deprecated compatibility alias."""
        return self.prune_impossible_hypotheses(df_trip, near_subway, near_bus)


class BayesianRouteEvaluator:
    """
    Evaluador Bayesiano Posterior de Rutas.
    Clasifica las hipótesis de ruteo completadas utilizando métricas físicas
    corregidas (velocidad de calle, distancia topológica) y cercanía a infraestructura,
    contrastadas contra las matrices probabilísticas del paper.
    """
    def __init__(self):
        self.modos = ['Carro', 'Bus', 'Metro', 'Caminar']
        
        # Intentar cargar cachés precomputadas de infraestructura para optimización espacial
        import pickle
        from pathlib import Path
        
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parents[1]  # pipeline_v4/src/ -> dos niveles arriba
        cache_dir = project_root / "Inputs" / "Infrastructure" / "Cache_Optimizado"
        
        cache_file_drive = cache_dir / "drive_infra_cache.pkl"
        cache_file_walk = cache_dir / "walk_infra_cache.pkl"
        
        if cache_file_drive.exists() and cache_file_walk.exists():
            try:
                with open(cache_file_drive, 'rb') as f:
                    self.drive_infra_cache = pickle.load(f)
                with open(cache_file_walk, 'rb') as f:
                    self.walk_infra_cache = pickle.load(f)
            except Exception:
                self.drive_infra_cache = None
                self.walk_infra_cache = None
        else:
            self.drive_infra_cache = None
            self.walk_infra_cache = None
        
        # Matrices de Probabilidad Condicional originales del artículo de investigación
        self.Cercania = np.array([
            [0.0396, 0.0099, 0.9354, 0.0150],
            [0.0098, 0.0613, 0.0098, 0.9191],
            [0.0231, 0.0099, 0.0122, 0.9547],
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ])

        self.Velocidad = np.array([
            [0.0457, 0.0099, 0.8942, 0.0501],
            [0.7629, 0.0199, 0.0099, 0.2073],
            [0.2317, 0.0099, 0.7464, 0.0120],
            [0.8739, 0.0098, 0.1065, 0.0098],
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ])

        self.Distancia = np.array([
            [0.4232, 0.4261, 0.1132, 0.0375],
            [0.0099, 0.7101, 0.0707, 0.2093],
            [0.7451, 0.2252, 0.0173, 0.0124],
            [0.1364, 0.5562, 0.2896, 0.0178],
            [0.0099, 0.9563, 0.0099, 0.0239],
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ])

        self.Velprom = np.array([
            [0.3825, 0.0725, 0.0222, 0.5228],
            [0.7861, 0.1943, 0.0098, 0.0098],
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        ])

    def evaluate_completed_route_with_matrices(self, df_routed, mode_hypothesis, subway_routes, bus_routes):
        """
        Evalúa una hipótesis de ruta ya calculada a través de las matrices bayesianas del paper.
        Retorna la distribución de probabilidad a posteriori para los 4 modos en esta hipótesis.
        """
        if df_routed.empty:
            return pd.Series([0.0, 0.0, 0.0, 0.0], index=self.modos)

        df_eval = df_routed.copy()
        
        # Intentar usar el caché precalculado si está cargado y el ruteo contiene nodos válidos
        if self.drive_infra_cache is not None and self.walk_infra_cache is not None:
            is_walk = (str(mode_hypothesis).lower() == 'caminar')
            cache = self.walk_infra_cache if is_walk else self.drive_infra_cache
            
            near_subway_list = []
            near_bus_list = []
            
            if 'start_node' not in df_eval.columns or 'end_node' not in df_eval.columns or (df_eval['start_node'] == 'N/A').all():
                df_eval = calcular_cercania_infraestructura(df_eval, subway_routes, bus_routes)
            else:
                for u, v in zip(df_eval['start_node'], df_eval['end_node']):
                    res = cache.get((u, v))
                    if res is None:
                        res = cache.get((v, u), (0, 0))
                    near_subway_list.append(res[0])
                    near_bus_list.append(res[1])
                df_eval['near_subway_line'] = near_subway_list
                df_eval['near_bus_route'] = near_bus_list
        else:
            # Fallback a sjoin_nearest en runtime si no hay caché
            df_eval = calcular_cercania_infraestructura(df_eval, subway_routes, bus_routes)

        # 1. Índice de Cercanía: 0 (Metro), 1 (Bus), 2 (Ninguno)
        idx_c = np.where(df_eval['near_subway_line'] == 1, 0,
                np.where(df_eval['near_bus_route'] == 1, 1, 2))

        # 2. Índice de Velocidad por punto corregido
        idx_v = np.digitize(df_eval['Speed [km/h]'].fillna(0), bins=[6.001, 20.001, 80.001])

        # 3. Índice de Distancia física acumulada de ruteo
        total_dist_km = df_eval['distance_m'].sum() / 1000.0
        idx_d = np.digitize([total_dist_km], bins=[1.0, 6.001, 10.001, 18.001])[0]
        idx_d_arr = np.repeat(idx_d, len(df_eval))

        # 4. Índice de Velocidad promedio de ruta
        avg_speed_trip = df_eval['Speed [km/h]'].mean()
        idx_vp = np.digitize([avg_speed_trip], bins=[6.001])[0]
        idx_vp_arr = np.repeat(idx_vp, len(df_eval))

        # 5. Multiplicación Bayesiana
        P_unnorm = (self.Cercania[idx_c] * 
                    self.Velocidad[idx_v] * 
                    self.Distancia[idx_d_arr] * 
                    self.Velprom[idx_vp_arr])

        # 6. Normalización por puntos
        suma_puntos = P_unnorm.sum(axis=1, keepdims=True)
        suma_puntos[suma_puntos == 0] = 1
        P_norm = P_unnorm / suma_puntos

        # 7. Votación acumulada de viaje
        total_votes = P_norm.sum(axis=0)
        total_votes_normalized = total_votes / (total_votes.sum() + 1e-9)
        
        return pd.Series(total_votes_normalized, index=self.modos)

    def _resolve_car_vs_bus(self, df_routed, subway_routes, bus_routes, threshold_bus=0.70):
        """
        Separate Carro from Bus after the road-motorized hypothesis has won.
        
        Use high-fidelity contextual and topological variables: proximity to bus
        routes, stop and speed patterns, and Bayesian probabilities evaluated on
        the routed road hypothesis.
        """
        df_eval = df_routed.copy()
        
        # Reuse the precomputed cache when valid routed nodes are available.
        if self.drive_infra_cache is not None:
            cache = self.drive_infra_cache
            near_subway_list = []
            near_bus_list = []
            
            if 'start_node' not in df_eval.columns or 'end_node' not in df_eval.columns or (df_eval['start_node'] == 'N/A').all():
                df_eval = calcular_cercania_infraestructura(df_eval, subway_routes, bus_routes)
            else:
                for u, v in zip(df_eval['start_node'], df_eval['end_node']):
                    res = cache.get((u, v))
                    if res is None:
                        res = cache.get((v, u), (0, 0))
                    near_subway_list.append(res[0])
                    near_bus_list.append(res[1])
                df_eval['near_subway_line'] = near_subway_list
                df_eval['near_bus_route'] = near_bus_list
        else:
            df_eval = calcular_cercania_infraestructura(df_eval, subway_routes, bus_routes)
            
        prob_vector_road = self.evaluate_completed_route_with_matrices(df_eval, 'Carro', subway_routes, bus_routes)
        
        # Fraction of the route near the bus network.
        fraction_near_bus = df_eval['near_bus_route'].mean() if not df_eval.empty else 0.0
        
        if prob_vector_road['Bus'] > prob_vector_road['Carro'] and fraction_near_bus >= threshold_bus:
            return 'Bus'
        else:
            return 'Carro'

    def select_final_mode(self, hypotheses, subway_routes, bus_routes):
        """
        Evaluate completed routing hypotheses and select the highest-probability mode.

        Each candidate is scored on its routed geometry. If the road hypothesis
        wins, a final Carro-versus-Bus step avoids routing the same graph twice.
        """
        best_mode = None
        best_probability = -1.0
        best_df = None
        diagnostic_probs = {}

        # Evaluate routed candidates (Caminar, Metro, shared road hypothesis).
        for mode, df_routed in hypotheses.items():
            prob_vector = self.evaluate_completed_route_with_matrices(df_routed, mode, subway_routes, bus_routes)
            # Score each mode under its own routed hypothesis.
            prob_self = prob_vector[mode]
            diagnostic_probs[mode] = prob_vector.to_dict()
            
            if prob_self > best_probability:
                best_probability = prob_self
                best_mode = mode
                best_df = df_routed

        # Final Carro-versus-Bus classification for the road hypothesis.
        if best_mode == 'Carro' and 'Carro' in hypotheses:
            resolved_mode = self._resolve_car_vs_bus(hypotheses['Carro'], subway_routes, bus_routes)
            if resolved_mode == 'Bus':
                best_mode = 'Bus'
                best_df = best_df.copy()
                best_df['modo_transporte'] = 'Bus'

        return best_mode, best_df, best_probability, diagnostic_probs

    def evaluate_hypothesis(self, df_routed, mode_hypothesis, subway_routes, bus_routes):
        """Deprecated compatibility alias."""
        return self.evaluate_completed_route_with_matrices(df_routed, mode_hypothesis, subway_routes, bus_routes)

    def select_best_hypothesis(self, hypotheses, subway_routes, bus_routes):
        """Deprecated compatibility alias."""
        return self.select_final_mode(hypotheses, subway_routes, bus_routes)


try:
    from . import random_forest_contract
    from .random_forest_contract import (
        BUS_PROBABILITY_THRESHOLD, MIN_EFFECTIVE_PINGS, MIN_PCT_CONSERVED,
        N1_FEATURES, N2_FEATURES, N3_FEATURES, RF_FEATURES,
    )
except ImportError:
    import random_forest_contract
    from random_forest_contract import (
        BUS_PROBABILITY_THRESHOLD, MIN_EFFECTIVE_PINGS, MIN_PCT_CONSERVED,
        N1_FEATURES, N2_FEATURES, N3_FEATURES, RF_FEATURES,
    )


class RandomForestRouteEvaluator:
    """Inferencia estricta del clasificador jerárquico ML v4 de 52 variables."""

    def __init__(self, model_path=None):
        self.classifier_name = "random_forest"
        self.model_version = "ML_v4_52"
        self.modos = ["Carro", "Bus", "Metro", "Caminar"]
        self.feature_cols_v4 = list(RF_FEATURES)
        self.feature_cols_new = list(RF_FEATURES)
        self.feature_cols = list(RF_FEATURES)
        self.n1_features = list(RF_FEATURES)
        self.n2_features = list(RF_FEATURES)
        self.n3_features = list(RF_FEATURES)
        self.bus_threshold = 0.50
        self.metro_threshold = getattr(random_forest_contract, "METRO_PROBABILITY_THRESHOLD", 0.30)
        self.raw_counts = self._load_raw_counts()
        project_root = Path(__file__).resolve().parents[2]
        self.model_path = Path(model_path) if model_path else project_root / "pipeline_v4" / "calibration_and_diagnostics" / "modal_classification" / "artifacts" / "random_forest_modal.pkl"
        self.loaded_from_disk = False
        self._load_model()

    @staticmethod
    def _trip_key(caid, num_trip):
        try:
            num_trip = str(int(float(num_trip)))
        except (TypeError, ValueError):
            num_trip = str(num_trip).strip()
        return f"{str(caid).strip()}_{num_trip}"

    def _load_raw_counts(self):
        raw_counts = {}
        raw_csv = Path(__file__).resolve().parents[2] / "Inputs" / "GPS User Data" / "Datos de MATLAB GPS.csv"
        if not raw_csv.exists():
            return raw_counts
        try:
            data = pd.read_csv(raw_csv, usecols=["caid", "num_trip"])
            for (caid, num_trip), count in data.groupby(["caid", "num_trip"]).size().items():
                raw_counts[self._trip_key(caid, num_trip)] = int(count)
        except Exception as exc:
            print(
                f"[RandomForestRouteEvaluator] Unable to preload raw counts: {exc}",
                flush=True,
            )
        return raw_counts

    def _load_model(self):
        import pickle

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Production RF model not found at {self.model_path}. "
                "Inference does not train or write models; run the training utility explicitly."
            )
        try:
            with self.model_path.open("rb") as handle:
                saved = pickle.load(handle)
            if saved.get("feature_cols_v4") != list(RF_FEATURES) or saved.get("feature_cols_new") != list(RF_FEATURES):
                raise ValueError("The PKL feature order does not match the official 52-feature contract.")
            for name in ("clf_n1", "clf_n2", "clf_n3"):
                classifier = saved.get(name)
                if classifier is None or getattr(classifier, "n_features_in_", None) != 52:
                    raise ValueError(f"{name} is not a compatible 52-feature classifier.")
                names = list(getattr(classifier, "feature_names_in_", []))
                if names and names != list(RF_FEATURES):
                    raise ValueError(f"The internal feature order for {name} is incompatible.")
                setattr(self, name, classifier)
            self.clf = self.clf_n1
            self.loaded_from_disk = True
            print("[RandomForestRouteEvaluator] ML v4 model loaded.", flush=True)
        except Exception as exc:
            raise RuntimeError(f"Could not load a compatible RF PKL from {self.model_path}: {exc}") from exc

    @staticmethod
    def _speeds(frame):
        return frame["Speed [km/h]"].fillna(0.0).to_numpy(dtype=float)

    @staticmethod
    def _speed_stats(prefix, speeds):
        if len(speeds) == 0:
            return {f"{prefix}_{name}": 0.0 for name in (
                "mean_speed", "max_speed", "p25_speed", "p50_speed", "p75_speed",
                "max_speed_diff", "mean_speed_diff"
            )}
        diffs = np.abs(np.diff(speeds))
        return {
            f"{prefix}_mean_speed": float(np.mean(speeds)),
            f"{prefix}_max_speed": float(np.max(speeds)),
            f"{prefix}_p25_speed": float(np.percentile(speeds, 25)),
            f"{prefix}_p50_speed": float(np.percentile(speeds, 50)),
            f"{prefix}_p75_speed": float(np.percentile(speeds, 75)),
            f"{prefix}_max_speed_diff": float(np.max(diffs)) if len(diffs) else 0.0,
            f"{prefix}_mean_speed_diff": float(np.mean(diffs)) if len(diffs) else 0.0,
        }

    @staticmethod
    def _max_run(values):
        best = current = 0
        for value in values:
            current = current + 1 if value else 0
            best = max(best, current)
        return float(best)

    @staticmethod
    def _window_values_fast(timestamps_series, values_arr, min_points=3, window_sec=180, step_sec=30, agg='mean'):
        if timestamps_series is None or len(timestamps_series) == 0:
            return []
        ts = pd.to_datetime(timestamps_series)
        if ts.isna().all():
            return []
        ts_ns = ts.astype('datetime64[ns]').astype('int64').to_numpy()
        vals = np.asarray(values_arr, dtype=float)
        if len(ts_ns) <= 1:
            return []
        if not np.all(ts_ns[:-1] <= ts_ns[1:]):
            sort_order = np.argsort(ts_ns, kind='stable')
            ts_ns = ts_ns[sort_order]
            vals = vals[sort_order]
        start_ns = ts_ns[0]
        max_ns = ts_ns[-1]
        window_ns = int(window_sec * 1_000_000_000)
        step_ns = int(step_sec * 1_000_000_000)
        res = []
        curr_ns = start_ns
        while curr_ns + window_ns <= max_ns:
            end_ns = curr_ns + window_ns
            i_start = np.searchsorted(ts_ns, curr_ns, side='left')
            i_end = np.searchsorted(ts_ns, end_ns, side='right')
            count = i_end - i_start
            if count >= min_points:
                slice_val = vals[i_start:i_end]
                val = np.sum(slice_val) if agg == 'sum' else np.mean(slice_val)
                res.append(float(val))
            curr_ns += step_ns
        return res

    def _window_values(self, frame, value_fn):
        if frame is None or frame.empty or "local_timestamp" not in frame:
            return []
        timestamps = pd.to_datetime(frame["local_timestamp"])
        start, maximum = timestamps.min(), timestamps.max()
        values = []
        while start + pd.Timedelta(minutes=3) <= maximum:
            mask = (timestamps >= start) & (timestamps <= start + pd.Timedelta(minutes=3))
            if int(mask.sum()) >= 3:
                values.append(float(value_fn(frame.loc[mask])))
            start += pd.Timedelta(seconds=30)
        return values

    @staticmethod
    def _require_columns(frame, columns, hypothesis):
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            raise ServingContractError(
                f"Hypothesis {hypothesis!r} does not satisfy the training contract; "
                f"missing columns: {missing}"
            )

    def extract_features(self, hypotheses, serving_context=None):
        if not isinstance(serving_context, TripServingContext):
            raise ServingContractError(
                "ML inference requires TripServingContext with GPS pings and real snapping distances."
            )
        hyps = {str(key).lower(): value for key, value in hypotheses.items()}
        drive = hyps.get("carro") if hyps.get("carro") is not None else hyps.get("bus")
        walk, metro = hyps.get("caminar"), hyps.get("metro")
        row = {}

        if drive is not None and not drive.empty:
            self._require_columns(
                drive,
                {"Speed [km/h]", "highway", "near_bus_route", "near_subway_line"},
                "Carro",
            )
            speeds = self._speeds(drive)
            row.update(self._speed_stats("drive", speeds))
            row["drive_std_speed"] = float(np.std(speeds))
            row["drive_stop_frac"] = float(np.mean(speeds < 2.0))
            highways = drive["highway"].fillna("unclassified").astype(str)
            row["drive_highway_motorway_frac"] = float(np.mean(highways.str.contains("motorway|trunk|primary"))) if len(highways) else 0.0
            row["drive_highway_residential_frac"] = float(np.mean(highways.str.contains("residential"))) if len(highways) else 0.0
            row["drive_near_bus_frac"] = float(np.mean(drive["near_bus_route"] == 1))
            row["drive_near_metro_frac"] = float(np.mean(drive["near_subway_line"] == 1))
        else:
            for feature in RF_FEATURES:
                if feature.startswith("drive_"):
                    row[feature] = 0.0

        if walk is not None and not walk.empty:
            self._require_columns(walk, {"Speed [km/h]", "highway"}, "Caminar")
            speeds = self._speeds(walk)
            row.update(self._speed_stats("walk", speeds))
            row["walk_std_speed"] = float(np.std(speeds))
            highways = walk["highway"].fillna("unclassified").astype(str)
            row["walk_highway_footway_frac"] = float(np.mean(highways.str.contains("footway|pedestrian|steps|path|living_street"))) if len(highways) else 0.0
        else:
            for feature in RF_FEATURES:
                if feature.startswith("walk_"):
                    row[feature] = 0.0

        if metro is not None and not metro.empty:
            self._require_columns(metro, {"Speed [km/h]", "near_subway_line"}, "Metro")
            row.update(self._speed_stats("metro", self._speeds(metro)))
            row["metro_near_metro_frac"] = float(np.mean(metro["near_subway_line"] == 1))
        else:
            for feature in RF_FEATURES:
                if feature.startswith("metro_"):
                    row[feature] = 0.0

        for network in ("drive", "walk"):
            values = np.asarray(getattr(serving_context, f"snap_dist_{network}"), dtype=float)
            if len(values) != serving_context.effective_ping_count or not np.isfinite(values).all():
                raise ServingContractError(
                    f"snap_dist_{network} debe contener una distancia finita por ping efectivo."
                )
            row[f"mean_snap_dist_{network}"] = float(np.mean(values))
            row[f"max_snap_dist_{network}"] = float(np.max(values))
            row[f"std_snap_dist_{network}"] = float(np.std(values))

        row["drive_near_bus_drift_decay"] = row["drive_near_bus_frac"] * np.exp(-row["mean_snap_dist_drive"] / 15.0)
        row["drive_near_bus_high_drift"] = row["drive_near_bus_frac"] * (1.0 - np.exp(-row["mean_snap_dist_drive"] / 15.0))

        row.update({"drive_num_stops": 0.0, "drive_mean_stop_duration": 0.0,
                    "drive_mean_stop_interval": 0.0, "drive_std_stop_interval": 0.0})
        if drive is not None and not drive.empty:
            speeds = self._speeds(drive)
            stop_mask = speeds < 2.0
            row["drive_num_stops"] = float(np.sum(stop_mask))
            if "local_timestamp" in drive and np.any(stop_mask):
                elapsed = pd.to_datetime(drive["local_timestamp"]).diff().dt.total_seconds().fillna(0.0).to_numpy()
                row["drive_mean_stop_duration"] = float(np.sum(elapsed[stop_mask]) / np.sum(stop_mask))
            if "distance_m" in drive and np.sum(stop_mask) > 1:
                intervals = np.diff(drive["distance_m"].fillna(0.0).cumsum().to_numpy()[stop_mask])
                row["drive_mean_stop_interval"] = float(np.mean(intervals))
                row["drive_std_stop_interval"] = float(np.std(intervals))

        if metro is not None and not metro.empty and "local_timestamp" in metro:
            vals_m = (metro.get("near_subway_line", 0) == 1).to_numpy(dtype=float)
            metro_windows = self._window_values_fast(metro["local_timestamp"], vals_m, agg='mean')
        else:
            metro_windows = []

        if drive is not None and not drive.empty and "local_timestamp" in drive:
            bus_vals = (drive.get("near_bus_route", 0) == 1).to_numpy(dtype=float)
            bus_windows = self._window_values_fast(drive["local_timestamp"], bus_vals, agg='mean')
            stop_vals = (self._speeds(drive) < 2.0).astype(float)
            stop_windows = self._window_values_fast(drive["local_timestamp"], stop_vals, agg='sum')
        else:
            bus_windows, stop_windows = [], []

        if walk is not None and not walk.empty and "local_timestamp" in walk:
            w_speeds = self._speeds(walk)
            walk_regime = ((w_speeds >= 2.0) & (w_speeds <= 6.0)).astype(float)
            walk_windows = self._window_values_fast(walk["local_timestamp"], walk_regime, agg='mean')
        else:
            walk_windows = []

        row["metro_win_near_metro_max"] = max(metro_windows, default=0.0)
        row["metro_win_near_metro_p90"] = float(np.percentile(metro_windows, 90)) if metro_windows else 0.0
        row["metro_win_near_metro_consec_run"] = self._max_run(np.array(metro_windows) > 0.7)
        row["drive_win_near_bus_max"] = max(bus_windows, default=0.0)
        row["drive_win_near_bus_p90"] = float(np.percentile(bus_windows, 90)) if bus_windows else 0.0
        row["drive_win_near_bus_consec_run"] = self._max_run(np.array(bus_windows) > 0.7)
        row["drive_win_stops_max"] = max(stop_windows, default=0.0)
        row["drive_win_stops_consec_run"] = self._max_run(np.array(stop_windows) >= 1)
        row["walk_win_walk_regime_max"] = max(walk_windows, default=0.0)
        row["walk_win_walk_regime_consec_run"] = self._max_run(np.array(walk_windows) > 0.7)
        return {feature: float(row.get(feature, 0.0)) for feature in RF_FEATURES}

    def select_final_mode(self, hypotheses, subway_routes=None, bus_routes=None, *, serving_context=None):
        if not hypotheses:
            return None, None, 0.0, {}
        if not isinstance(serving_context, TripServingContext):
            raise ServingContractError("Falta TripServingContext para aplicar el guardrail GPS.")
        any_frame = next(iter(hypotheses.values()))
        min_pings = getattr(random_forest_contract, "MIN_EFFECTIVE_PINGS", MIN_EFFECTIVE_PINGS)
        min_pct = 100.0 * getattr(random_forest_contract, "MIN_PCT_CONSERVED", MIN_PCT_CONSERVED)
        if (serving_context.effective_ping_count < min_pings or
                serving_context.pct_pings_conserved < min_pct):
            return "Calidad insuficiente", None, 0.0, {mode: 0.0 for mode in self.modos}

        extracted = self.extract_features(hypotheses, serving_context=serving_context)
        missing = [name for name in set(self.n1_features + self.n2_features + self.n3_features) if name not in extracted]
        if missing:
            raise ValueError(f"Required inference features are missing: {sorted(missing)}")
        features = pd.DataFrame([extracted], columns=RF_FEATURES)
        x_n1 = features.loc[:, self.n1_features]
        pred_n1 = int(self.clf_n1.predict(x_n1)[0])
        prob_n1 = self.clf_n1.predict_proba(x_n1)[0]
        has_metro_candidate = any(str(k).lower() == "metro" for k in hypotheses.keys())
        if pred_n1 == 0:
            mode, diagnostics = "Caminar", {"Caminar": float(prob_n1[0]), "Metro": 0.0, "Bus": 0.0, "Carro": 0.0}
        else:
            x_n2 = features.loc[:, self.n2_features]
            prob_n2 = self.clf_n2.predict_proba(x_n2)[0]
            metro_threshold = getattr(self, "metro_threshold", getattr(random_forest_contract, "METRO_PROBABILITY_THRESHOLD", 0.30))
            pred_n2 = int(prob_n2[1] >= metro_threshold) if has_metro_candidate else 0
            if pred_n2 == 1:
                mode = "Metro"
                diagnostics = {"Caminar": float(prob_n1[0]), "Metro": float(prob_n1[1] * prob_n2[1]), "Bus": 0.0, "Carro": 0.0}
            else:
                x_n3 = features.loc[:, self.n3_features]
                prob_n3 = self.clf_n3.predict_proba(x_n3)[0]
                pred_n3 = int(prob_n3[1] >= self.bus_threshold)
                mode = "Bus" if pred_n3 else "Carro"
                p_surface = prob_n2[0] if has_metro_candidate else 1.0
                p_metro = prob_n2[1] if has_metro_candidate else 0.0
                diagnostics = {"Caminar": float(prob_n1[0]), "Metro": float(prob_n1[1] * p_metro),
                               "Bus": float(prob_n1[1] * p_surface * prob_n3[1]),
                               "Carro": float(prob_n1[1] * p_surface * prob_n3[0])}
        total = sum(diagnostics.values())
        diagnostics = {key: value / total for key, value in diagnostics.items()} if total else diagnostics
        hyps = {str(key).lower(): value for key, value in hypotheses.items()}
        if mode == "Caminar":
            selected = hyps.get("caminar")
        elif mode == "Metro":
            selected = hyps.get("metro")
        else:
            selected = hyps.get("carro") if hyps.get("carro") is not None else hyps.get("bus")
        if selected is None:
            selected = any_frame
        else:
            selected = selected.copy()
        selected["modo_transporte"] = mode
        return mode, selected, diagnostics.get(mode, 0.0), diagnostics

    def evaluate_with_contract(self, hypotheses, subway_routes=None, bus_routes=None, *, serving_context=None):
        mode, selected, probability, probabilities = self.select_final_mode(
            hypotheses, subway_routes, bus_routes, serving_context=serving_context
        )
        accepted = mode not in {None, "Calidad insuficiente"}
        return {
            "final_class": mode,
            "probabilities": probabilities,
            "selected_route": selected,
            "selected_probability": probability,
            "classifier": self.classifier_name,
            "model_version": self.model_version,
            "quality_status": "accepted" if accepted else "rejected",
            "rejection_reason": None if accepted else ("quality_guardrail" if mode == "Calidad insuficiente" else "no_hypotheses"),
        }


class HybridRouteEvaluator(RandomForestRouteEvaluator):
    """Official hierarchical hybrid modal classifier: GB / RF / Extra Trees."""

    def __init__(self, model_path=None):
        self.classifier_name = "hybrid"
        self.model_version = "hybrid_v1"
        self.modos = ["Carro", "Bus", "Metro", "Caminar"]
        self.feature_cols_v4 = list(RF_FEATURES)
        self.feature_cols_new = list(RF_FEATURES)
        self.feature_cols = list(RF_FEATURES)
        self.n1_features = list(N1_FEATURES)
        self.n2_features = list(N2_FEATURES)
        self.n3_features = list(N3_FEATURES)
        self.bus_threshold = BUS_PROBABILITY_THRESHOLD
        self.metro_threshold = getattr(random_forest_contract, "METRO_PROBABILITY_THRESHOLD", 0.30)
        self.raw_counts = self._load_raw_counts()
        project_root = Path(__file__).resolve().parents[2]
        self.model_path = Path(model_path) if model_path else project_root / "pipeline_v4" / "calibration_and_diagnostics" / "modal_classification" / "artifacts" / "modal_classifier_hybrid_v1.pkl"
        self.loaded_from_disk = False
        self._load_hybrid_model()

    def _load_hybrid_model(self):
        import pickle
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Official hybrid model not found at {self.model_path}. "
                "Inference does not train or overwrite models."
            )
        try:
            with self.model_path.open("rb") as handle:
                saved = pickle.load(handle)
            contract = saved.get("model_contract", {})
            expected = {"n1": self.n1_features, "n2": self.n2_features, "n3": self.n3_features}
            for level, features in expected.items():
                declared = contract.get(level, {}).get("features")
                if declared != features:
                    raise ValueError(f"Incompatible feature order in {level}: expected {len(features)} features.")
                classifier = saved.get(f"clf_{level}")
                if classifier is None or getattr(classifier, "n_features_in_", None) != len(features):
                    raise ValueError(f"clf_{level} does not accept the {len(features)} contract features.")
                names = list(getattr(classifier, "feature_names_in_", []))
                if names and names != features:
                    raise ValueError(f"The internal feature order for clf_{level} is incompatible.")
                setattr(self, f"clf_{level}", classifier)
            self.bus_threshold = BUS_PROBABILITY_THRESHOLD
            self.clf = self.clf_n1
            self.loaded_from_disk = True
            print("[HybridRouteEvaluator] Hybrid model loaded.", flush=True)
        except Exception as exc:
            raise RuntimeError(f"Could not load a compatible hybrid model from {self.model_path}: {exc}") from exc


class GuardrailedBayesianRouteEvaluator(BayesianRouteEvaluator):
    """Historical Bayes backend with the same quality guardrail as ML models."""

    def __init__(self):
        super().__init__()
        self.classifier_name = "bayes"
        self.model_version = "bayes_matrices_v1"
        self.raw_counts = RandomForestRouteEvaluator._load_raw_counts(self)

    _trip_key = staticmethod(RandomForestRouteEvaluator._trip_key)

    def select_final_mode(self, hypotheses, subway_routes, bus_routes, *, serving_context=None):
        if not hypotheses:
            return None, None, 0.0, {}
        if not isinstance(serving_context, TripServingContext):
            raise ServingContractError("Falta TripServingContext para aplicar el guardrail GPS.")
        min_pings = getattr(random_forest_contract, "MIN_EFFECTIVE_PINGS", MIN_EFFECTIVE_PINGS)
        min_pct = 100.0 * getattr(random_forest_contract, "MIN_PCT_CONSERVED", MIN_PCT_CONSERVED)
        if (serving_context.effective_ping_count < min_pings or
                serving_context.pct_pings_conserved < min_pct):
            return "Calidad insuficiente", None, 0.0, {mode: 0.0 for mode in self.modos}
        return super().select_final_mode(hypotheses, subway_routes, bus_routes)

    def evaluate_with_contract(self, hypotheses, subway_routes, bus_routes, *, serving_context=None):
        mode, selected, probability, probabilities = self.select_final_mode(
            hypotheses, subway_routes, bus_routes, serving_context=serving_context
        )
        accepted = mode not in {None, "Calidad insuficiente"}
        return {
            "final_class": mode, "probabilities": probabilities,
            "selected_route": selected, "selected_probability": probability,
            "classifier": self.classifier_name, "model_version": self.model_version,
            "quality_status": "accepted" if accepted else "rejected",
            "rejection_reason": None if accepted else ("quality_guardrail" if mode == "Calidad insuficiente" else "no_hypotheses"),
        }


def create_modal_evaluator(classifier=None, enable_bayes_fallback=False):
    if classifier is None:
        try:
            from . import config
        except ImportError:
            import config
        classifier = config.MODAL_CLASSIFIER
    classifier = str(classifier).strip().lower()
    if classifier in {"bayes", "bayesian"}:
        return GuardrailedBayesianRouteEvaluator()
    if classifier == "hybrid":
        return HybridRouteEvaluator()
    if classifier != "random_forest":
        raise ValueError(f"Clasificador modal desconocido: {classifier!r}")
    try:
        return RandomForestRouteEvaluator()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        if enable_bayes_fallback:
            print(
                f"[ModalClassification] RF unavailable; Bayesian fallback enabled: {exc}",
                flush=True,
            )
            return BayesianRouteEvaluator()
        raise RuntimeError(
            "ML V4 initialization failed and the Bayes fallback is disabled. "
            f"Detalle: {exc}"
        ) from exc

