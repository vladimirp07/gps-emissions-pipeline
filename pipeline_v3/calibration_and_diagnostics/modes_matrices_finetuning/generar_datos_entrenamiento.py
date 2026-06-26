import os
import sys
import time
import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
import shapely.wkt
from pathlib import Path

# Agregar la raiz del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from pipeline_v3.src.routing import get_candidates_vectorized, complete_route_v1_optimized
from pipeline_v3.src.bayes_classifier import PriorModeClassifier, calcular_cercania_infraestructura
from pipeline_v3.src import config

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

def main():
    print("=== INICIANDO EXTRACCION DE DATOS DE ENTRENAMIENTO PARA OPTUNA ===")
    
    # Parametros optimos calibrados (Escenario 8 Combined Optimal)
    SPATIAL_FILTER_M = 15.0
    WALK_BUFFER_M = 50.0
    DRIVE_BUFFER_M = 150.0
    PHYSICS_FACTOR = 2.0
    
    os.environ['PHYSICS_FACTOR'] = str(PHYSICS_FACTOR)
    
    # Cargar recursos
    print("Cargando redes e infraestructura...")
    with open(config.FILE_GRAFO, 'rb') as f:
        G_drive = pickle.load(f)
    with open(config.FILE_GRAFO_WALK, 'rb') as f:
        G_walk = pickle.load(f)
    with open(config.FILE_CACHE_IG_DRIVE, 'rb') as f:
        ig_drive, map_drive = pickle.load(f)
    with open(config.FILE_CACHE_IG_WALK, 'rb') as f:
        ig_walk, map_walk = pickle.load(f)
    edges_drive = gpd.read_parquet(config.FILE_CACHE_EDGES_DRIVE)
    edges_walk = gpd.read_parquet(config.FILE_CACHE_EDGES_WALK)
    
    subway_df = pd.read_csv(config.FILE_METRO)
    if 'WKT' in subway_df.columns:
        subway_df['geometry'] = subway_df['WKT'].apply(shapely.wkt.loads)
    elif 'geometry' in subway_df.columns:
        subway_df['geometry'] = subway_df['geometry'].apply(shapely.wkt.loads)
    subway_routes = gpd.GeoDataFrame(subway_df, geometry='geometry', crs="EPSG:4326")
    geometry_metro_proj = subway_routes.to_crs("EPSG:32614")
    
    bus_routes = gpd.read_file(config.FILE_BUS)
    
    # Cargar dataset de MATLAB limpio
    clean_matlab_path = config.GPS_DIR / "Datos de MATLAB GPS Limpios.csv"
    print(f"Cargando dataset de MATLAB limpio: {clean_matlab_path}")
    if not clean_matlab_path.exists():
        print(f"Error: No se encontro el dataset depurado en {clean_matlab_path}. Corre primero depurar_datos_matlab.py.")
        sys.exit(1)
        
    df_all = pd.read_csv(clean_matlab_path)
    
    # Estandarizar nombres de columnas para el ruteador
    df_all = df_all.rename(columns={'lat': 'latitude', 'lon': 'longitude', 'num_trip': 'trip'})
    df_all['local_timestamp'] = pd.to_datetime(df_all['Timestamp'])
    df_all['modo_transporte'] = df_all['mode_of_transport'].str.strip().str.lower()
    
    # Clasificador prior para poda heuristica previa al ruteo
    prior_classifier = PriorModeClassifier(max_walk_speed=22.0, max_walk_dist=15.0)
    
    degradaciones = {
        'Raw': lambda df: df.copy(),
        'L1': degradar_nivel_1,
        'L2': degradar_nivel_2,
        'L3': degradar_nivel_3
    }
    
    # Agrupar por usuario, viaje y modo real
    groups = df_all.groupby(['caid', 'trip', 'modo_transporte'])
    print(f"Total de viajes unicos a evaluar: {len(groups)}")
    
    registros_entrenamiento = []
    start_time = time.time()
    processed_trips = 0
    
    for (caid, trip_id, modo_real), df_trip in groups:
        df_trip = df_trip.sort_values(by='local_timestamp').reset_index(drop=True)
        
        # Calcular distancias y velocidades lineales crudas para el prior
        df_trip['lon_prev'] = df_trip['longitude'].shift(1)
        df_trip['lat_prev'] = df_trip['latitude'].shift(1)
        df_trip['dis lineal [m]'] = haversine_np(df_trip['longitude'], df_trip['latitude'], df_trip['lon_prev'], df_trip['lat_prev']) * 1000.0
        df_trip['dis lineal [m]'] = df_trip['dis lineal [m]'].fillna(0.0)
        
        df_trip['time_prev'] = df_trip['local_timestamp'].shift(1)
        df_trip['dt_sec'] = (df_trip['local_timestamp'] - df_trip['time_prev']).dt.total_seconds().fillna(0.0)
        df_trip['Speed [km/h]'] = np.where(
            df_trip['dt_sec'] > 0,
            (df_trip['dis lineal [m]'] / 1000.0) / (df_trip['dt_sec'] / 3600.0),
            0.0
        )
        
        for deg_name, deg_func in degradaciones.items():
            df_deg = deg_func(df_trip)
            
            # Aplicar filtro espacial optimizado (15m)
            df_deg = apply_spatial_filter(df_deg, SPATIAL_FILTER_M)
            
            if len(df_deg) < 2:
                continue
                
            # Proximidad espacial cruda (para la heuristica del Prior)
            gdf_pts = gpd.GeoDataFrame(df_deg, geometry=gpd.points_from_xy(df_deg['longitude'], df_deg['latitude']), crs="EPSG:4326")
            gdf_pts_proj = gdf_pts.to_crs("EPSG:32614")
            
            metro_union = geometry_metro_proj.unary_union
            dist_metro = gdf_pts_proj.distance(metro_union)
            near_subway = dist_metro < 50.0
            
            bus_union = bus_routes.to_crs("EPSG:32614").unary_union
            dist_bus = gdf_pts_proj.distance(bus_union)
            near_bus = dist_bus < 50.0
            
            # Podar hipotesis imposibles para acelerar el proceso
            candidatos = prior_classifier.prune_impossible_hypotheses(df_deg, near_subway, near_bus)
            
            # Ruterar unicamente los candidatos viables
            for modo_hip in candidatos:
                try:
                    edges_act = edges_walk if (modo_hip.lower() == 'caminar') else edges_drive
                    gdf_pts_proj_edges = gdf_pts.to_crs(edges_act.crs)
                    
                    df_deg['drive_ids'], df_deg['drive_dists'] = get_candidates_vectorized(
                        edges_drive, gdf_pts_proj_edges, buffer_m=DRIVE_BUFFER_M
                    )
                    df_deg['walk_ids'], df_deg['walk_dists'] = get_candidates_vectorized(
                        edges_walk, gdf_pts_proj_edges, buffer_m=WALK_BUFFER_M
                    )
                    
                    # Llamada al Ruteador principal
                    df_routed = complete_route_v1_optimized(
                        id=caid,
                        registros_person=df_deg,
                        G_drive=G_drive,
                        G_walk=G_walk,
                        ig_drive=ig_drive,
                        ig_walk=ig_walk,
                        map_drive=map_drive,
                        map_walk=map_walk,
                        geometry_metro=geometry_metro_proj
                    )
                    
                    is_failed = df_routed.empty or (
                        df_routed['ruteo_fallido'].all() if 'ruteo_fallido' in df_routed.columns else False
                    )
                    
                    if is_failed:
                        continue
                        
                    # Medir proximidad sobre la ruta ya completada en el grafo
                    df_routed = calcular_cercania_infraestructura(df_routed, subway_routes, bus_routes)
                    
                    # Extraer las variables necesarias para el clasificador bayesiano
                    total_dist_km = df_routed['distance_m'].sum() / 1000.0
                    avg_speed = df_routed['Speed [km/h]'].mean()
                    
                    velocidades = df_routed['Speed [km/h]'].fillna(0.0).tolist()
                    near_subway_list = df_routed['near_subway_line'].fillna(0).tolist()
                    near_bus_list = df_routed['near_bus_route'].fillna(0).tolist()
                    
                    registros_entrenamiento.append({
                        'caid': caid,
                        'trip_id': trip_id,
                        'modo_real': modo_real,
                        'modo_hipotesis': modo_hip.lower(),
                        'degradacion': deg_name,
                        'distancia_ruteada_km': total_dist_km,
                        'velocidad_promedio_kmh': avg_speed,
                        'velocidades': velocidades,
                        'near_subway_line': near_subway_list,
                        'near_bus_route': near_bus_list
                    })
                    
                except Exception:
                    continue
                    
        processed_trips += 1
        if processed_trips % 20 == 0:
            elapsed = time.time() - start_time
            print(f" -> Procesados {processed_trips}/{len(groups)} viajes (Tiempo transcurrido: {elapsed:.1f}s, Muestras guardadas: {len(registros_entrenamiento)})")
            
    # Guardar los datos en formato pickle para conservar las estructuras de listas
    df_training = pd.DataFrame(registros_entrenamiento)
    output_pkl = config.GPS_DIR / "datos_entrenamiento_optuna.pkl"
    
    print(f"\nGuardando {len(df_training)} muestras de entrenamiento ruteadas en: {output_pkl}")
    with open(output_pkl, 'wb') as f:
        pickle.dump(df_training, f)
        
    print(f"Generacion completada con exito. Tiempo de ejecucion: {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()
