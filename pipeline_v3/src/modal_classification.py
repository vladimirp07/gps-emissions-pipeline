
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

def calcular_cercania_infraestructura(df, subway_routes, bus_routes):
    """
    Calcula la proximidad espacial (distancia métrica exacta) de cada ping GPS
    a las líneas físicas del metro y rutas de autobús oficiales.
    """
    if df.empty:
        return df.copy()
        
    # 1. Proyectar temporalmente a UTM 14N (EPSG:32614)
    gdf_pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    gdf_pts = gdf_pts.to_crs("EPSG:32614")
    
    subway_proj = subway_routes.to_crs("EPSG:32614") if subway_routes.crs != "EPSG:32614" else subway_routes
    bus_proj = bus_routes.to_crs("EPSG:32614") if bus_routes.crs != "EPSG:32614" else bus_routes
    
    RADIO_METRO = 50
    RADIO_BUS = 20
    
    # 2. Búsqueda Espacial Exacta (sjoin_nearest mide a la LÍNEA, no al centroide)
    # --- METRO ---
    metro_join = gpd.sjoin_nearest(gdf_pts, subway_proj, how='left', distance_col='dist_metro')
    metro_join = metro_join[~metro_join.index.duplicated(keep='first')] # Limpiar empates de distancia
    df['near_subway_line'] = (metro_join['dist_metro'] < RADIO_METRO).astype(int)
    
    # --- BUS ---
    bus_join = gpd.sjoin_nearest(gdf_pts, bus_proj, how='left', distance_col='dist_bus')
    bus_join = bus_join[~bus_join.index.duplicated(keep='first')] 
    df['near_bus_route'] = (bus_join['dist_bus'] < RADIO_BUS).astype(int)
    
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
        Poda (filtra) hipótesis imposibles para ahorrar costes de ruteo.
        
        Retorna la lista de modos candidatos simplificados que requieren ruteo:
        - 'Caminar' (si es viable físicamente)
        - 'Metro' (si es viable físicamente y hay cercanía al metro)
        - 'Carro' (representando la red vial común o 'road_motorized')
        
        Nota metodológica: 'Bus' no se rutea de manera independiente para evitar
        duplicar coste computacional sobre la misma red vial (G_drive). Si la hipótesis
        vial gana, se discriminará 'Carro' vs 'Bus' en el paso posterior.
        """
        max_speed = df_trip['Speed [km/h]'].max()
        total_dist_km = df_trip['dis lineal [m]'].sum() / 1000.0
        
        candidates = []
        
        # 1. Chequeo de Caminar
        if max_speed <= self.max_walk_speed and total_dist_km <= self.max_walk_dist:
            candidates.append('Caminar')
            
        # 2. Chequeo de Metro
        if near_subway.any() and total_dist_km > 1.0:
            candidates.append('Metro')
            
        # 3. Chequeo de Motorizado Vial (Carro / Bus)
        # Se rutea bajo el candidato genérico 'Carro' (road_motorized)
        if max_speed > 3.0 or total_dist_km > 0.5:
            candidates.append('Carro')
            
        # Salvaguarda: al menos un candidato
        if not candidates:
            candidates = ['Carro']
            
        return candidates

    def predict_candidates(self, df_trip, near_subway, near_bus):
        """Método obsoleto (deprecated) mantenido por compatibilidad hacia atrás."""
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
        project_root = current_dir.parents[1]  # pipeline_v3/src/ -> dos niveles arriba
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
        Sub-clasificación posterior para discriminar entre Carro y Bus una vez que se ha
        determinado que el viaje se realizó por la red vial motorizada (road_motorized).
        
        Usa variables contextuales y topológicas de alta fidelidad:
        1. Cercanía espacial a rutas de bus a lo largo de la trayectoria.
        2. Patrón de paradas de bus / velocidad promedio de ruteo.
        3. Matrices de probabilidad bayesianas evaluadas sobre la ruta motorizada.
        """
        df_eval = df_routed.copy()
        
        # Intentar usar el caché precalculado si está cargado y el ruteo contiene nodos válidos
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
        
        # Calcular porcentaje de cobertura sobre la red de autobuses
        fraction_near_bus = df_eval['near_bus_route'].mean() if not df_eval.empty else 0.0
        
        if prob_vector_road['Bus'] > prob_vector_road['Carro'] and fraction_near_bus >= threshold_bus:
            return 'Bus'
        else:
            return 'Carro'

    def select_final_mode(self, hypotheses, subway_routes, bus_routes):
        """
        Compara y evalúa todas las hipótesis de ruteo completadas y selecciona la óptima.
        
        Pasos metodológicos:
        1. Evalúa la probabilidad a posteriori de cada hipótesis ('Caminar', 'Metro', 'Carro')
           usando sus respectivas geometrías ruteadas.
        2. Determina el modo ganador en base a la probabilidad de su propia hipótesis.
        3. Si la hipótesis vial ganadora es 'Carro' (road_motorized), realiza la sub-clasificación
           posterior 'Carro vs Bus' para no duplicar el costo del ruteo.
        """
        best_mode = None
        best_probability = -1.0
        best_df = None
        diagnostic_probs = {}

        # Evaluar candidatos ruteados (Caminar, Metro, Carro/Road_Motorized)
        for mode, df_routed in hypotheses.items():
            prob_vector = self.evaluate_completed_route_with_matrices(df_routed, mode, subway_routes, bus_routes)
            # Evaluamos la probabilidad del modo bajo su propia hipótesis de ruteo
            prob_self = prob_vector[mode]
            diagnostic_probs[mode] = prob_vector.to_dict()
            
            if prob_self > best_probability:
                best_probability = prob_self
                best_mode = mode
                best_df = df_routed

        # Sub-clasificación final para vehículos de carretera (Carro vs Bus)
        if best_mode == 'Carro' and 'Carro' in hypotheses:
            resolved_mode = self._resolve_car_vs_bus(hypotheses['Carro'], subway_routes, bus_routes)
            if resolved_mode == 'Bus':
                best_mode = 'Bus'
                best_df = best_df.copy()
                best_df['modo_transporte'] = 'Bus'

        return best_mode, best_df, best_probability, diagnostic_probs

    def evaluate_hypothesis(self, df_routed, mode_hypothesis, subway_routes, bus_routes):
        """Método obsoleto (deprecated) mantenido por compatibilidad hacia atrás."""
        return self.evaluate_completed_route_with_matrices(df_routed, mode_hypothesis, subway_routes, bus_routes)

    def select_best_hypothesis(self, hypotheses, subway_routes, bus_routes):
        """Método obsoleto (deprecated) mantenido por compatibilidad hacia atrás."""
        return self.select_final_mode(hypotheses, subway_routes, bus_routes)


try:
    from .random_forest_contract import (
        BUS_PROBABILITY_THRESHOLD, MIN_EFFECTIVE_PINGS, MIN_PCT_CONSERVED,
        N1_FEATURES, N2_FEATURES, N3_FEATURES, RF_FEATURES,
    )
except ImportError:
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
        self.raw_counts = self._load_raw_counts()
        project_root = Path(__file__).resolve().parents[2]
        self.model_path = Path(model_path) if model_path else project_root / "Inputs" / "GPS User Data" / "random_forest_modal.pkl"
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
            print(f"[RandomForestRouteEvaluator] No se pudieron precargar conteos brutos: {exc}")
        return raw_counts

    def _load_model(self):
        import pickle

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo RF de producción en {self.model_path}. "
                "Inferencia no entrena ni escribe modelos; ejecute entrenar_random_forest.py explícitamente."
            )
        try:
            with self.model_path.open("rb") as handle:
                saved = pickle.load(handle)
            if saved.get("feature_cols_v4") != list(RF_FEATURES) or saved.get("feature_cols_new") != list(RF_FEATURES):
                raise ValueError("El orden de features del PKL no coincide con el contrato oficial de 52 variables.")
            for name in ("clf_n1", "clf_n2", "clf_n3"):
                classifier = saved.get(name)
                if classifier is None or getattr(classifier, "n_features_in_", None) != 52:
                    raise ValueError(f"{name} no es un clasificador compatible de 52 variables.")
                names = list(getattr(classifier, "feature_names_in_", []))
                if names and names != list(RF_FEATURES):
                    raise ValueError(f"El orden interno de {name} es incompatible.")
                setattr(self, name, classifier)
            self.clf = self.clf_n1
            self.loaded_from_disk = True
            print(f"[RandomForestRouteEvaluator] Modelo ML v4 cargado desde {self.model_path}")
        except Exception as exc:
            raise RuntimeError(f"No se pudo cargar un PKL RF compatible desde {self.model_path}: {exc}") from exc

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

    def extract_features(self, hypotheses):
        hyps = {str(key).lower(): value for key, value in hypotheses.items()}
        drive = hyps.get("carro") if hyps.get("carro") is not None else hyps.get("bus")
        walk, metro = hyps.get("caminar"), hyps.get("metro")
        row = {}

        if drive is not None and not drive.empty:
            speeds = self._speeds(drive)
            row.update(self._speed_stats("drive", speeds))
            row["drive_std_speed"] = float(np.std(speeds))
            row["drive_stop_frac"] = float(np.mean(speeds < 2.0))
            highways = drive["highway"].fillna("unclassified").astype(str) if "highway" in drive else pd.Series([], dtype=str)
            row["drive_highway_motorway_frac"] = float(np.mean(highways.str.contains("motorway|trunk|primary"))) if len(highways) else 0.0
            row["drive_highway_residential_frac"] = float(np.mean(highways.str.contains("residential"))) if len(highways) else 0.0
            row["drive_near_bus_frac"] = float(np.mean(drive["near_bus_route"] == 1)) if "near_bus_route" in drive else 0.0
            row["drive_near_metro_frac"] = float(np.mean(drive["near_subway_line"] == 1)) if "near_subway_line" in drive else 0.0
        else:
            for feature in RF_FEATURES:
                if feature.startswith("drive_"):
                    row[feature] = 0.0

        if walk is not None and not walk.empty:
            speeds = self._speeds(walk)
            row.update(self._speed_stats("walk", speeds))
            row["walk_std_speed"] = float(np.std(speeds))
            highways = walk["highway"].fillna("unclassified").astype(str) if "highway" in walk else pd.Series([], dtype=str)
            row["walk_highway_footway_frac"] = float(np.mean(highways.str.contains("footway|pedestrian|steps|path|living_street"))) if len(highways) else 0.0
        else:
            for feature in RF_FEATURES:
                if feature.startswith("walk_"):
                    row[feature] = 0.0

        if metro is not None and not metro.empty:
            row.update(self._speed_stats("metro", self._speeds(metro)))
            row["metro_near_metro_frac"] = float(np.mean(metro["near_subway_line"] == 1)) if "near_subway_line" in metro else 0.0
        else:
            for feature in RF_FEATURES:
                if feature.startswith("metro_"):
                    row[feature] = 0.0

        snap_source = next((frame for frame in (drive, walk, metro) if frame is not None and not frame.empty), None)
        for network, default_mean, default_max, default_std in (("drive", 15.0, 30.0, 5.0), ("walk", 5.0, 10.0, 2.0)):
            column = f"snap_dist_{network}"
            if snap_source is not None and column in snap_source:
                values = snap_source[column].fillna(0.0).to_numpy(dtype=float)
                row[f"mean_snap_dist_{network}"] = float(np.mean(values))
                row[f"max_snap_dist_{network}"] = float(np.max(values))
                row[f"std_snap_dist_{network}"] = float(np.std(values))
            else:
                row[f"mean_snap_dist_{network}"] = default_mean
                row[f"max_snap_dist_{network}"] = default_max
                row[f"std_snap_dist_{network}"] = default_std

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

        metro_windows = self._window_values(metro, lambda part: np.mean(part.get("near_subway_line", 0) == 1))
        bus_windows = self._window_values(drive, lambda part: np.mean(part.get("near_bus_route", 0) == 1))
        stop_windows = self._window_values(drive, lambda part: np.sum(self._speeds(part) < 2.0))
        walk_windows = self._window_values(walk, lambda part: np.mean((self._speeds(part) >= 2.0) & (self._speeds(part) <= 6.0)))
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

    def select_final_mode(self, hypotheses, subway_routes=None, bus_routes=None):
        if not hypotheses:
            return None, None, 0.0, {}
        any_frame = next(iter(hypotheses.values()))
        num_pings = len(any_frame)
        caid_column = "caid" if "caid" in any_frame else "id_usuario" if "id_usuario" in any_frame else None
        trip_column = "num_trip" if "num_trip" in any_frame else "trip" if "trip" in any_frame else None
        pct_conserved = 100.0
        if caid_column and trip_column and len(any_frame):
            key = self._trip_key(any_frame[caid_column].iloc[0], any_frame[trip_column].iloc[0])
            raw_count = self.raw_counts.get(key)
            if raw_count:
                pct_conserved = 100.0 * num_pings / raw_count
        if num_pings < MIN_EFFECTIVE_PINGS or pct_conserved < MIN_PCT_CONSERVED:
            return "Calidad insuficiente", None, 0.0, {mode: 0.0 for mode in self.modos}

        extracted = self.extract_features(hypotheses)
        missing = [name for name in set(self.n1_features + self.n2_features + self.n3_features) if name not in extracted]
        if missing:
            raise ValueError(f"Faltan variables requeridas para inferencia: {sorted(missing)}")
        features = pd.DataFrame([extracted], columns=RF_FEATURES)
        x_n1 = features.loc[:, self.n1_features]
        pred_n1 = int(self.clf_n1.predict(x_n1)[0])
        prob_n1 = self.clf_n1.predict_proba(x_n1)[0]
        if pred_n1 == 0:
            mode, diagnostics = "Caminar", {"Caminar": float(prob_n1[0]), "Metro": 0.0, "Bus": 0.0, "Carro": 0.0}
        else:
            x_n2 = features.loc[:, self.n2_features]
            pred_n2 = int(self.clf_n2.predict(x_n2)[0])
            prob_n2 = self.clf_n2.predict_proba(x_n2)[0]
            if pred_n2 == 1:
                mode = "Metro"
                diagnostics = {"Caminar": float(prob_n1[0]), "Metro": float(prob_n1[1] * prob_n2[1]), "Bus": 0.0, "Carro": 0.0}
            else:
                x_n3 = features.loc[:, self.n3_features]
                prob_n3 = self.clf_n3.predict_proba(x_n3)[0]
                pred_n3 = int(prob_n3[1] >= self.bus_threshold)
                mode = "Bus" if pred_n3 else "Carro"
                diagnostics = {"Caminar": float(prob_n1[0]), "Metro": float(prob_n1[1] * prob_n2[1]),
                               "Bus": float(prob_n1[1] * prob_n2[0] * prob_n3[1]),
                               "Carro": float(prob_n1[1] * prob_n2[0] * prob_n3[0])}
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

    def evaluate_with_contract(self, hypotheses, subway_routes=None, bus_routes=None):
        mode, selected, probability, probabilities = self.select_final_mode(hypotheses, subway_routes, bus_routes)
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
    """Clasificador modal jerárquico híbrido oficial: GB / RF / Extra Trees."""

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
        self.raw_counts = self._load_raw_counts()
        project_root = Path(__file__).resolve().parents[2]
        self.model_path = Path(model_path) if model_path else project_root / "Inputs" / "GPS User Data" / "modal_classifier_hybrid_v1.pkl"
        self.loaded_from_disk = False
        self._load_hybrid_model()

    def _load_hybrid_model(self):
        import pickle
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo híbrido oficial en {self.model_path}. "
                "La inferencia no entrena ni sobrescribe modelos."
            )
        try:
            with self.model_path.open("rb") as handle:
                saved = pickle.load(handle)
            contract = saved.get("model_contract", {})
            expected = {"n1": self.n1_features, "n2": self.n2_features, "n3": self.n3_features}
            for level, features in expected.items():
                declared = contract.get(level, {}).get("features")
                if declared != features:
                    raise ValueError(f"Orden de variables incompatible en {level}: se esperaban {len(features)}.")
                classifier = saved.get(f"clf_{level}")
                if classifier is None or getattr(classifier, "n_features_in_", None) != len(features):
                    raise ValueError(f"clf_{level} no acepta las {len(features)} variables del contrato.")
                names = list(getattr(classifier, "feature_names_in_", []))
                if names and names != features:
                    raise ValueError(f"Orden interno de clf_{level} incompatible.")
                setattr(self, f"clf_{level}", classifier)
            threshold = float(contract.get("n3", {}).get("threshold_bus", -1))
            if threshold != BUS_PROBABILITY_THRESHOLD:
                raise ValueError(f"Umbral Bus incompatible: {threshold}; esperado {BUS_PROBABILITY_THRESHOLD}.")
            self.bus_threshold = threshold
            self.clf = self.clf_n1
            self.loaded_from_disk = True
            print(f"[HybridRouteEvaluator] Modelo híbrido cargado desde {self.model_path}")
        except Exception as exc:
            raise RuntimeError(f"No se pudo cargar el modelo híbrido compatible desde {self.model_path}: {exc}") from exc


class GuardrailedBayesianRouteEvaluator(BayesianRouteEvaluator):
    """Bayes histórico con el mismo guardrail de calidad de los modelos ML."""

    def __init__(self):
        super().__init__()
        self.classifier_name = "bayes"
        self.model_version = "bayes_matrices_v1"
        self.raw_counts = RandomForestRouteEvaluator._load_raw_counts(self)

    _trip_key = staticmethod(RandomForestRouteEvaluator._trip_key)

    def select_final_mode(self, hypotheses, subway_routes, bus_routes):
        if not hypotheses:
            return None, None, 0.0, {}
        frame = next(iter(hypotheses.values()))
        num_pings = len(frame)
        caid_col = "caid" if "caid" in frame else "id_usuario" if "id_usuario" in frame else None
        trip_col = "num_trip" if "num_trip" in frame else "trip" if "trip" in frame else None
        pct = 100.0
        if caid_col and trip_col and num_pings:
            raw = self.raw_counts.get(self._trip_key(frame[caid_col].iloc[0], frame[trip_col].iloc[0]))
            if raw:
                pct = 100.0 * num_pings / raw
        if num_pings < MIN_EFFECTIVE_PINGS or pct < MIN_PCT_CONSERVED:
            return "Calidad insuficiente", None, 0.0, {mode: 0.0 for mode in self.modos}
        return super().select_final_mode(hypotheses, subway_routes, bus_routes)

    def evaluate_with_contract(self, hypotheses, subway_routes, bus_routes):
        mode, selected, probability, probabilities = self.select_final_mode(hypotheses, subway_routes, bus_routes)
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
            print(f"[ModalClassification] RF no disponible; fallback Bayes habilitado: {exc}")
            return BayesianRouteEvaluator()
        raise RuntimeError(
            "No fue posible inicializar ML v4 y el fallback Bayes está deshabilitado. "
            f"Detalle: {exc}"
        ) from exc
