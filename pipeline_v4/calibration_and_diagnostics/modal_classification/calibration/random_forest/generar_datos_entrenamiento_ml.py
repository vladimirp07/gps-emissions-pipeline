import os
import sys
import time
import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely.wkt
from pathlib import Path
import multiprocessing
import argparse
import json

# Agregar la raiz del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))
from pipeline_v4.src.routing import get_candidates_vectorized, complete_route_v2_optimized
from pipeline_v4.src.modal_classification import PriorModeClassifier, calcular_cercania_infraestructura
from pipeline_v4.src import config
from pipeline_v4.src.random_forest_contract import MIN_EFFECTIVE_PINGS, MIN_PCT_CONSERVED

EXPANDED_CACHE_NAME = "datos_entrenamiento_ml_expanded.pkl"
EXPANDED_ROUTES_DIR = "cache_rutas_completas_expanded"


def trip_key(caid, trip_id):
    try:
        trip_id = str(int(float(trip_id)))
    except (TypeError, ValueError):
        trip_id = str(trip_id).strip()
    return f"{str(caid).strip()}_{trip_id}"


def temporal_downsample(df, max_points):
    """Muestreo cronológico uniforme, preservando extremos, sin descartar el viaje."""
    if max_points <= 0 or len(df) <= max_points:
        return df.copy().reset_index(drop=True)
    indices = np.unique(np.linspace(0, len(df) - 1, max_points, dtype=int))
    return df.iloc[indices].copy().reset_index(drop=True)

def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367.0 * c
    return km

def apply_spatial_filter(df, min_dist_m):
    if len(df) <= 2:
        return df.copy()
    kept_indices = [0]
    last_idx = 0
    lats = df['latitude'].to_numpy()
    lons = df['longitude'].to_numpy()
    for i in range(1, len(df) - 1):
        d = haversine_np(lons[last_idx], lats[last_idx], lons[i], lats[i]) * 1000.0
        if d >= min_dist_m:
            kept_indices.append(i)
            last_idx = i
    kept_indices.append(len(df) - 1)
    kept_indices = sorted(list(set(kept_indices)))
    return df.iloc[kept_indices].copy().reset_index(drop=True)

# Niveles de degradacion
def degradar_nivel_1(df):
    indices = np.arange(0, len(df), 20)
    if len(df)-1 not in indices:
        indices = np.append(indices, len(df)-1)
    return df.iloc[indices].copy().reset_index(drop=True)
    
def degradar_nivel_2(df):
    df_copy = df.copy()
    n = len(df_copy)
    if n > 300:
        mid = n // 2
        gap_size = min(240, int(n * 0.4))
        half_gap = gap_size // 2
        drop_range = range(mid - half_gap, mid + half_gap)
        df_gap = df_copy.drop(index=drop_range).reset_index(drop=True)
    else:
        df_gap = df_copy
    indices = np.arange(0, len(df_gap), 90)
    if len(df_gap) > 0:
        if len(df_gap)-1 not in indices:
            indices = np.append(indices, len(df_gap)-1)
        return df_gap.iloc[indices].copy().reset_index(drop=True)
    return df_gap
    
def degradar_nivel_3(df):
    df_copy = df.copy()
    n = len(df_copy)
    if n > 800:
        mid = n // 2
        gap_size = min(720, int(n * 0.5))
        half_gap = gap_size // 2
        drop_range = range(mid - half_gap, mid + half_gap)
        df_gap = df_copy.drop(index=drop_range).reset_index(drop=True)
    elif n > 100:
        mid = n // 2
        gap_size = int(n * 0.4)
        half_gap = gap_size // 2
        drop_range = range(mid - half_gap, mid + half_gap)
        df_gap = df_copy.drop(index=drop_range).reset_index(drop=True)
    else:
        df_gap = df_copy
    indices = np.arange(0, len(df_gap), 360)
    if len(df_gap) > 0:
        if len(df_gap)-1 not in indices:
            indices = np.append(indices, len(df_gap)-1)
        return df_gap.iloc[indices].copy().reset_index(drop=True)
    return df_gap

# Variables globales para los procesos hijos
G_drive_global = None
G_walk_global = None
ig_drive_global = None
ig_walk_global = None
map_drive_global = None
map_walk_global = None
edges_drive_global = None
edges_walk_global = None
subway_routes_global = None
bus_routes_global = None
geometry_metro_proj_global = None
metro_union_global = None
bus_union_global = None
prior_classifier_global = None

def init_worker():
    global G_drive_global, G_walk_global, ig_drive_global, ig_walk_global
    global map_drive_global, map_walk_global, edges_drive_global, edges_walk_global
    global subway_routes_global, bus_routes_global, geometry_metro_proj_global
    global metro_union_global, bus_union_global, prior_classifier_global
    
    os.environ['PHYSICS_FACTOR'] = '2.0'
    
    # Cargar grafos
    with open(config.FILE_GRAFO, 'rb') as f:
        G_drive_global = pickle.load(f)
    with open(config.FILE_GRAFO_WALK, 'rb') as f:
        G_walk_global = pickle.load(f)
    with open(config.FILE_CACHE_IG_DRIVE, 'rb') as f:
        ig_drive_global, map_drive_global = pickle.load(f)
    with open(config.FILE_CACHE_IG_WALK, 'rb') as f:
        ig_walk_global, map_walk_global = pickle.load(f)
        
    edges_drive_global = gpd.read_parquet(config.FILE_CACHE_EDGES_DRIVE)
    edges_walk_global = gpd.read_parquet(config.FILE_CACHE_EDGES_WALK)
    
    subway_df = pd.read_csv(config.FILE_METRO)
    if 'WKT' in subway_df.columns:
        subway_df['geometry'] = subway_df['WKT'].apply(shapely.wkt.loads)
    elif 'geometry' in subway_df.columns:
        subway_df['geometry'] = subway_df['geometry'].apply(shapely.wkt.loads)
    subway_routes_global = gpd.GeoDataFrame(subway_df, geometry='geometry', crs="EPSG:4326")
    geometry_metro_proj_global = subway_routes_global.to_crs("EPSG:32614")
    
    bus_routes_global = gpd.read_file(config.FILE_BUS)
    
    metro_union_global = geometry_metro_proj_global.unary_union
    bus_union_global = bus_routes_global.to_crs("EPSG:32614").unary_union
    
    prior_classifier_global = PriorModeClassifier(max_walk_speed=22.0, max_walk_dist=15.0)

def process_single_task(task_args):
    caid, trip_id, modo_real, df_trip_data, deg_name, max_route_pings = task_args
    
    # Reconstruir dataframe
    df_trip = temporal_downsample(pd.DataFrame(df_trip_data), max_route_pings)
    
    SPATIAL_FILTER_M = 15.0
    WALK_BUFFER_M = 50.0
    DRIVE_BUFFER_M = 150.0
    
    degradaciones_funcs = {
        'Raw': lambda df: df.copy(),
        'L1': degradar_nivel_1,
        'L2': degradar_nivel_2,
        'L3': degradar_nivel_3
    }
    
    deg_func = degradaciones_funcs[deg_name]
    df_deg = deg_func(df_trip)
    df_deg = apply_spatial_filter(df_deg, SPATIAL_FILTER_M)
    
    if len(df_deg) < 2:
        return task_args[:3] + (deg_name,), [], ["menos_de_2_pings_tras_degradacion_y_filtro_espacial"]
        
    # Recalcular distancias y velocidades
    df_deg['lon_prev'] = df_deg['longitude'].shift(1)
    df_deg['lat_prev'] = df_deg['latitude'].shift(1)
    df_deg['dis lineal [m]'] = haversine_np(df_deg['longitude'], df_deg['latitude'], df_deg['lon_prev'], df_deg['lat_prev']) * 1000.0
    df_deg['dis lineal [m]'] = df_deg['dis lineal [m]'].fillna(0.0)
    
    df_deg['time_prev'] = df_deg['local_timestamp'].shift(1)
    df_deg['dt_sec'] = (df_deg['local_timestamp'] - df_deg['time_prev']).dt.total_seconds().fillna(0.0)
    df_deg['Speed [km/h]'] = np.where(
        df_deg['dt_sec'] > 0,
        (df_deg['dis lineal [m]'] / 1000.0) / (df_deg['dt_sec'] / 3600.0),
        0.0
    )
    
    gdf_pts = gpd.GeoDataFrame(df_deg, geometry=gpd.points_from_xy(df_deg['longitude'], df_deg['latitude']), crs="EPSG:4326")
    gdf_pts_proj = gdf_pts.to_crs("EPSG:32614")
    
    dist_metro = gdf_pts_proj.distance(metro_union_global)
    near_subway = dist_metro < 50.0
    
    dist_bus = gdf_pts_proj.distance(bus_union_global)
    near_bus = dist_bus < 50.0
    
    candidatos = prior_classifier_global.prune_impossible_hypotheses(df_deg, near_subway, near_bus)
    
    resultados = []
    errores = []
    
    for modo_hip in candidatos:
        try:
            # Directorio de cache para Routed Routes DB
            cache_dir = Path(config.GPS_DIR) / EXPANDED_ROUTES_DIR
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Nombre de archivo único
            cache_file = cache_dir / f"{caid}_{trip_id}_{deg_name}_{modo_hip.lower()}.pkl"
            
            legacy_cache_file = Path(config.GPS_DIR) / "cache_rutas_completas" / cache_file.name
            if cache_file.exists():
                # Cargar de cache
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                df_routed = cached_data["df_routed"]
                snap_d_drive = cached_data["snap_d_drive"]
                snap_d_walk = cached_data["snap_d_walk"]
            elif legacy_cache_file.exists() and len(pd.DataFrame(df_trip_data)) <= max_route_pings:
                with open(legacy_cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                df_routed = cached_data["df_routed"]
                snap_d_drive = cached_data["snap_d_drive"]
                snap_d_walk = cached_data["snap_d_walk"]
                with open(cache_file, "wb") as f:
                    pickle.dump(cached_data, f)
            else:
                # Ejecutar ruteo completo
                edges_act = edges_walk_global if (modo_hip.lower() == 'caminar') else edges_drive_global
                gdf_pts_proj_edges = gdf_pts.to_crs(edges_act.crs)
                
                drive_ids, drive_dists = get_candidates_vectorized(
                    edges_drive_global, gdf_pts_proj_edges, buffer_m=DRIVE_BUFFER_M
                )
                walk_ids, walk_dists = get_candidates_vectorized(
                    edges_walk_global, gdf_pts_proj_edges, buffer_m=WALK_BUFFER_M
                )
                
                df_deg['drive_ids'] = drive_ids
                df_deg['drive_dists'] = drive_dists
                df_deg['walk_ids'] = walk_ids
                df_deg['walk_dists'] = walk_dists
                
                snap_d_drive = np.array([min(d) if len(d) > 0 else DRIVE_BUFFER_M for d in drive_dists], dtype=np.float32)
                snap_d_walk = np.array([min(w) if len(w) > 0 else WALK_BUFFER_M for w in walk_dists], dtype=np.float32)
                
                df_routed = complete_route_v2_optimized(
                    id=caid,
                    registros_person=df_deg,
                    G_drive=G_drive_global,
                    G_walk=G_walk_global,
                    ig_drive=ig_drive_global,
                    ig_walk=ig_walk_global,
                    map_drive=map_drive_global,
                    map_walk=map_walk_global,
                    geometry_metro=geometry_metro_proj_global
                )
                
                # Intentar calcular cercanía a infraestructura antes de guardar en caché
                is_failed = df_routed.empty or (
                    df_routed['ruteo_fallido'].all() if 'ruteo_fallido' in df_routed.columns else False
                )
                if not is_failed:
                    df_routed = calcular_cercania_infraestructura(df_routed, subway_routes_global, bus_routes_global)
                
                # Guardar en cache
                with open(cache_file, "wb") as f:
                    pickle.dump({
                        "df_routed": df_routed,
                        "snap_d_drive": snap_d_drive,
                        "snap_d_walk": snap_d_walk
                    }, f)
                    
            is_failed = df_routed.empty or (
                df_routed['ruteo_fallido'].all() if 'ruteo_fallido' in df_routed.columns else False
            )
            if is_failed:
                continue
                
            total_dist_km = df_routed['distance_m'].sum() / 1000.0
            avg_speed = df_routed['Speed [km/h]'].mean()
            
            # Indexación/binning
            idx_c = np.where(df_routed['near_subway_line'] == 1, 0,
                             np.where(df_routed['near_bus_route'] == 1, 1, 2))
            idx_v = np.digitize(df_routed['Speed [km/h]'].fillna(0.0).to_numpy(), bins=[6.001, 20.001, 80.001])
            idx_d = np.digitize([total_dist_km], bins=[1.0, 6.001, 10.001, 18.001])[0]
            idx_d_arr = np.repeat(idx_d, len(df_routed))
            idx_vp = np.digitize([avg_speed], bins=[6.001])[0]
            idx_vp_arr = np.repeat(idx_vp, len(df_routed))
            
            real_mode_mapped = modo_real.strip().lower()
            if real_mode_mapped == 'caminar':
                label = 'Caminar'
            elif real_mode_mapped == 'carro':
                label = 'Carro'
            elif real_mode_mapped == 'bus':
                label = 'Bus'
            elif real_mode_mapped == 'metro':
                label = 'Metro'
            else:
                label = modo_real.capitalize()
                
            num_stops = 0
            mean_stop_dur = 0.0
            mean_stop_int = 0.0
            std_stop_int = 0.0
            
            if modo_hip.lower() in ['carro', 'bus']:
                try:
                    df_routed['local_timestamp'] = pd.to_datetime(df_routed['local_timestamp'])
                    df_routed['dt_sec'] = df_routed['local_timestamp'].diff().dt.total_seconds().fillna(0.0)
                    
                    stop_mask = df_routed['Speed [km/h]'] < 2.0
                    n_s = int(stop_mask.sum())
                    if n_s > 0:
                        num_stops = n_s
                        total_stop_duration = float(df_routed.loc[stop_mask, 'dt_sec'].sum())
                        mean_stop_dur = total_stop_duration / n_s
                        
                        cum_dist = df_routed['distance_m'].cumsum()
                        stop_dists = cum_dist[stop_mask].values
                        intervals = np.diff(stop_dists) if len(stop_dists) > 1 else np.array([])
                        mean_stop_int = float(np.mean(intervals)) if len(intervals) > 0 else 0.0
                        std_stop_int = float(np.std(intervals)) if len(intervals) > 0 else 0.0
                except Exception:
                    pass
            
            # Calcular variables locales multiescala
            win_duration = pd.Timedelta(minutes=3)
            step_duration = pd.Timedelta(seconds=30)
            
            ts_g = pd.to_datetime(df_routed["local_timestamp"])
            t_min = ts_g.min()
            t_max = ts_g.max()
            
            local_features = {}
            win_val_list = []
            win_stops_list = []
            
            start_time = t_min
            while start_time + win_duration <= t_max:
                end_time = start_time + win_duration
                mask = (ts_g >= start_time) & (ts_g <= end_time)
                
                if mask.sum() >= 3:
                    df_slice = df_routed[mask]
                    speeds_slice = df_slice["Speed [km/h]"].fillna(0.0).values
                    
                    if modo_hip.lower() == 'metro':
                        near_metro_col = df_slice["near_subway_line"] if "near_subway_line" in df_slice.columns else pd.Series([0]*len(df_slice))
                        win_val_list.append(np.mean(near_metro_col == 1))
                    elif modo_hip.lower() in ['carro', 'bus']:
                        near_bus_col = df_slice["near_bus_route"] if "near_bus_route" in df_slice.columns else pd.Series([0]*len(df_slice))
                        win_val_list.append(np.mean(near_bus_col == 1))
                        win_stops_list.append(np.sum(speeds_slice < 2.0))
                    elif modo_hip.lower() == 'caminar':
                        win_val_list.append(np.mean((speeds_slice >= 2.0) & (speeds_slice <= 6.0)))
                        
                start_time += step_duration
                
            def helper_consec_run(bool_arr):
                if len(bool_arr) == 0: return 0
                max_run, curr_run = 0, 0
                for v in bool_arr:
                    if v:
                        curr_run += 1
                        if curr_run > max_run: max_run = curr_run
                    else:
                        curr_run = 0
                return max_run

            if modo_hip.lower() == 'metro':
                local_features["metro_win_near_metro_max"] = float(np.max(win_val_list)) if len(win_val_list) > 0 else 0.0
                local_features["metro_win_near_metro_p90"] = float(np.percentile(win_val_list, 90)) if len(win_val_list) > 0 else 0.0
                local_features["metro_win_near_metro_consec_run"] = float(helper_consec_run(np.array(win_val_list) > 0.7)) if len(win_val_list) > 0 else 0.0
            elif modo_hip.lower() in ['carro', 'bus']:
                local_features["drive_win_near_bus_max"] = float(np.max(win_val_list)) if len(win_val_list) > 0 else 0.0
                local_features["drive_win_near_bus_p90"] = float(np.percentile(win_val_list, 90)) if len(win_val_list) > 0 else 0.0
                local_features["drive_win_near_bus_consec_run"] = float(helper_consec_run(np.array(win_val_list) > 0.7)) if len(win_val_list) > 0 else 0.0
                local_features["drive_win_stops_max"] = float(np.max(win_stops_list)) if len(win_stops_list) > 0 else 0.0
                local_features["drive_win_stops_consec_run"] = float(helper_consec_run(np.array(win_stops_list) >= 1)) if len(win_stops_list) > 0 else 0.0
                
                # Calcular las 6 nuevas variables
                speeds = df_routed['Speed [km/h]'].fillna(0.0)
                smoothed_speed = speeds.rolling(window=5, center=True, min_periods=1).median().fillna(0.0).values
                cum_dist = df_routed['distance_m'].fillna(0.0).cumsum().values
                total_dist_km = cum_dist[-1] / 1000.0 if len(cum_dist) > 0 else 0.0
                
                state = 0
                num_cycles = 0
                for s in smoothed_speed:
                    if state == 0:
                        if s > 15.0:
                            state = 1
                    elif state == 1:
                        if s < 2.0:
                            state = 2
                    elif state == 2:
                        if s > 15.0:
                            num_cycles += 1
                            state = 1
                stop_cycles_per_km = num_cycles / total_dist_km if total_dist_km > 0.01 else 0.0
                
                is_stop = smoothed_speed < 2.0
                stop_indices = []
                for i in range(len(is_stop)):
                    if is_stop[i] and (i == 0 or not is_stop[i-1]):
                        stop_indices.append(i)
                        
                if len(stop_indices) > 1:
                    stop_dists = cum_dist[stop_indices]
                    spacings = np.diff(stop_dists)
                    median_stop_spacing_m = np.median(spacings)
                    mean_space = np.mean(spacings)
                    std_space = np.std(spacings)
                    cv_stop_spacing = std_space / mean_space if mean_space > 0.0 else 0.0
                else:
                    median_stop_spacing_m = 0.0
                    cv_stop_spacing = 0.0
                    
                timestamps = pd.to_datetime(df_routed['local_timestamp']).values
                restart_times = []
                for stop_idx in stop_indices:
                    restart_idx = None
                    for j in range(stop_idx + 1, len(smoothed_speed)):
                        if smoothed_speed[j] > 15.0:
                            restart_idx = j
                            break
                    if restart_idx is not None:
                        dur = (timestamps[restart_idx] - timestamps[stop_idx]) / np.timedelta64(1, 's')
                        restart_times.append(dur)
                        
                median_restart_time_s = np.median(restart_times) if len(restart_times) > 0 else 0.0
                p90_restart_time_s = np.percentile(restart_times, 90) if len(restart_times) > 0 else 0.0
                
                W = len(win_stops_list)
                max_run = helper_consec_run(np.array(win_stops_list) >= 1)
                stop_pattern_persistence = max_run / W if W > 0 else 0.0
                
                local_features["stop_cycles_per_km"] = float(stop_cycles_per_km)
                local_features["median_stop_spacing_m"] = float(median_stop_spacing_m)
                local_features["cv_stop_spacing"] = float(cv_stop_spacing)
                local_features["median_restart_time_s"] = float(median_restart_time_s)
                local_features["p90_restart_time_s"] = float(p90_restart_time_s)
                local_features["stop_pattern_persistence"] = float(stop_pattern_persistence)
            elif modo_hip.lower() == 'caminar':
                local_features["walk_win_walk_regime_max"] = float(np.max(win_val_list)) if len(win_val_list) > 0 else 0.0
                local_features["walk_win_walk_regime_consec_run"] = float(helper_consec_run(np.array(win_val_list) > 0.7)) if len(win_val_list) > 0 else 0.0

            resultados.append({
                'trip_id': f"{caid}_{trip_id}-{label}_{deg_name}_{modo_hip.lower()}",
                'label': label,
                'modo_hipotesis': modo_hip.lower(),
                'degradacion': deg_name,
                'idx_c': np.array(idx_c, dtype=np.int32),
                'idx_v': np.array(idx_v, dtype=np.int32),
                'idx_d_arr': np.array(idx_d_arr, dtype=np.int32),
                'idx_vp_arr': np.array(idx_vp_arr, dtype=np.int32),
                'speed_raw': df_routed['Speed [km/h]'].fillna(0.0).to_numpy(dtype=np.float32),
                'highway_raw': list(df_routed['highway'].fillna('unclassified').values),
                'snap_dist_drive': snap_d_drive, # Feature de Snapping Road
                'snap_dist_walk': snap_d_walk,   # Feature de Snapping Walk
                'num_stops': num_stops,
                'mean_stop_duration': mean_stop_dur,
                'mean_stop_interval': mean_stop_int,
                'std_stop_interval': std_stop_int,
                'local_features': local_features
            })
        except Exception as exc:
            errores.append(f"{modo_hip.lower()}: {type(exc).__name__}: {exc}")
            continue
            
    if not resultados and not errores:
        errores.append("ninguna_hipotesis_ruteada_valida")
    return task_args[:3] + (deg_name,), resultados, errores

def main():
    print("=== INICIANDO EXTRACCION DE DATOS EN PARALELO PARA RANDOM FOREST / OPTUNA ===")
    
    parser = argparse.ArgumentParser(description="Genera el caché ML ampliado sin sobrescribir el baseline.")
    parser.add_argument("--output", default=EXPANDED_CACHE_NAME)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-route-pings", type=int, default=1200)
    parser.add_argument("--audit-json", default="auditoria_dataset_ml_expanded.json")
    args = parser.parse_args()
    if not 1 <= args.workers <= 3:
        parser.error("--workers debe estar entre 1 y 3")
    if Path(args.output).name == "datos_entrenamiento_ml.pkl":
        parser.error("La salida ampliada no puede sobrescribir el caché baseline")

    clean_matlab_path = config.GPS_DIR / "Datos de MATLAB GPS Limpios.csv"
    if not clean_matlab_path.exists():
        print(f"Error: No se encontro {clean_matlab_path}")
        sys.exit(1)
        
    df_all = pd.read_csv(clean_matlab_path)
    df_all = df_all.drop(columns=['trip'], errors='ignore')
    df_all = df_all.rename(columns={'lat': 'latitude', 'lon': 'longitude', 'num_trip': 'trip'})
    df_all['local_timestamp'] = pd.to_datetime(df_all['Timestamp'])
    df_all['modo_transporte'] = df_all['mode_of_transport'].str.strip().str.lower()
    
    raw_counts_df = pd.read_csv(config.GPS_DIR / "Datos de MATLAB GPS.csv", usecols=["caid", "num_trip"])
    raw_counts = {trip_key(caid, trip): int(count) for (caid, trip), count in raw_counts_df.groupby(["caid", "num_trip"]).size().items()}

    selected_keys = []
    audit = {"selected": {}, "excluded": {}, "scenario_failures": {}, "configuration": vars(args)}
    valid_modes = {"caminar", "carro", "bus", "metro"}
    for (caid, trip_id), df_trip in df_all.groupby(["caid", "trip"]):
        key = trip_key(caid, trip_id)
        modes = sorted({str(mode).strip().lower() for mode in df_trip["modo_transporte"].dropna() if str(mode).strip()})
        if len(modes) > 1:
            audit["excluded"][key] = "etiqueta_mixta"
            continue
        if not modes or modes[0] not in valid_modes:
            audit["excluded"][key] = "etiqueta_vacia_o_invalida"
            continue
        raw_n = raw_counts.get(key, 0)
        pct_conserved = 100.0 * len(df_trip) / raw_n if raw_n else 0.0
        if len(df_trip) < MIN_EFFECTIVE_PINGS:
            audit["excluded"][key] = f"calidad_insuficiente_pings:{len(df_trip)}<{MIN_EFFECTIVE_PINGS}"
            continue
        if pct_conserved < MIN_PCT_CONSERVED:
            audit["excluded"][key] = f"calidad_insuficiente_conservacion:{pct_conserved:.2f}%<{MIN_PCT_CONSERVED}%"
            continue
        audit["selected"][key] = {"mode": modes[0], "clean_pings": len(df_trip), "raw_pings": raw_n,
                                  "pct_conserved": pct_conserved, "downsampled": len(df_trip) > args.max_route_pings}
        selected_keys.append(((caid, trip_id, modes[0]), df_trip))
        
    print(f"Seleccionados {len(selected_keys)} viajes para ruteo.")
    
    # Preparar los argumentos de tareas individuales (240 tareas en total)
    tasks = []
    for (caid, trip_id, modo_real), df_trip in selected_keys:
        # Convertir dataframe a dict para evitar problemas de comunicación entre procesos
        df_trip_dict = df_trip.to_dict(orient='list')
        # Añadir timestamp parseado como lista de strings para serializar
        df_trip_dict['local_timestamp'] = df_trip['local_timestamp'].tolist()
        
        for deg_name in ['Raw', 'L1', 'L2', 'L3']:
            tasks.append((caid, trip_id, modo_real, df_trip_dict, deg_name, args.max_route_pings))
            
    print(f"Total de tareas a procesar en paralelo: {len(tasks)}")
    
    # Determinar CPUs a utilizar (limitar a un máximo seguro de 3 para no congelar la compu)
    num_cpus = min(args.workers, 3)
    print(f"Utilizando {num_cpus} núcleo(s)...")
    
    t0 = time.time()
    
    # Arrancar pool de procesos
    pool = multiprocessing.Pool(processes=num_cpus, initializer=init_worker)
    
    resultados_totales = []
    completed = 0
    
    # Procesar tareas
    for task_meta, res_list, errors in pool.imap_unordered(process_single_task, tasks):
        if res_list:
            resultados_totales.extend(res_list)
        if errors:
            caid, trip_id, _, deg_name = task_meta
            audit["scenario_failures"][f"{trip_key(caid, trip_id)}:{deg_name}"] = errors
        completed += 1
        if completed % 20 == 0 or completed == len(tasks):
            elapsed = time.time() - t0
            print(f" -> Procesados {completed}/{len(tasks)} escenarios ({elapsed:.1f}s, Muestras en cache: {len(resultados_totales)})")
            
    pool.close()
    pool.join()
    
    # Guardar
    output_pkl = config.GPS_DIR / Path(args.output).name
    print(f"\nGuardando {len(resultados_totales)} muestras en caché de Machine Learning: {output_pkl}")
    with open(output_pkl, 'wb') as f:
        pickle.dump(resultados_totales, f)

    audit["elapsed_seconds"] = time.time() - t0
    audit["records"] = len(resultados_totales)
    audit["physical_trips_with_results"] = len({item["trip_id"].split("-", 1)[0] for item in resultados_totales})
    audit_path = config.GPS_DIR / Path(args.audit_json).name
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"Auditoría guardada en: {audit_path}")
        
    print(f"Caché generado con éxito en {time.time() - t0:.1f} segundos.")

if __name__ == '__main__':
    # Necesario para multiprocessing en Windows
    multiprocessing.freeze_support()
    main()

