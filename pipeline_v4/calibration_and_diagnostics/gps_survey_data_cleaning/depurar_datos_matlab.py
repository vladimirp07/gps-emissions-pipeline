import os
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import shapely.wkt
from pathlib import Path

# Agregar la raiz del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))
from pipeline_v4.src import config

def haversine_np(lon1, lat1, lon2, lat2):
    # Radio de la Tierra en km
    R = 6367.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = R * c
    return km

def main():
    print("=== INICIANDO DEPURACION DE DATOS DE MATLAB ===")
    
    input_path = config.GPS_DIR / "Datos de MATLAB GPS.csv"
    output_path = config.GPS_DIR / "Datos de MATLAB GPS Limpios.csv"
    
    print(f"Cargando dataset original desde: {input_path}")
    if not input_path.exists():
        print(f"Error: No se encontro el archivo en {input_path}")
        sys.exit(1)
        
    df = pd.read_csv(input_path)
    print(f"Dataset cargado. Total de pings: {len(df):,}")
    
    # Crear columnas temporales de procesamiento sin modificar las del dataset original
    df['Timestamp_parsed'] = pd.to_datetime(df['Timestamp'])
    df['modo_transporte'] = df['mode_of_transport'].str.strip().str.lower()
    
    # 1. Descartar viajes con más del 2% de duplicados en timestamps (indica viajes corruptos o fusionados)
    print("Analizando tasa de duplicación de timestamps por viaje...")
    trips_to_discard = []
    for (caid, trip_id), group in df.groupby(['caid', 'num_trip']):
        total_rows = len(group)
        unique_ts = group['Timestamp_parsed'].nunique()
        dup_rate = (total_rows - unique_ts) / total_rows
        if dup_rate > 0.02:
            print(f"  -> Viaje corrupto descartado: Usuario {caid}, Viaje {trip_id} | Pings: {total_rows}, Duplicación: {dup_rate:.2%}")
            trips_to_discard.append((caid, trip_id))
            
    for caid, trip_id in trips_to_discard:
        df = df[~((df['caid'] == caid) & (df['num_trip'] == trip_id))]
        
    print(f"Total de viajes descartados por duplicidad/corrupción: {len(trips_to_discard)}")
    print(f"Pings restantes tras descarte: {len(df):,}")
    
    # Deduplicar timestamps por usuario y viaje (quedarse con el primero de cada segundo para minorías <= 2%)
    print("Deduplicando registros con el mismo timestamp por usuario y viaje...")
    df = df.drop_duplicates(subset=['caid', 'num_trip', 'Timestamp_parsed'], keep='first').reset_index(drop=True)
    print(f"Pings después de la deduplicación de timestamps: {len(df):,}")
    
    # Ordenar cronologicamente por usuario, viaje y timestamp para asegurar calculos correctos
    df = df.sort_values(by=['caid', 'num_trip', 'Timestamp_parsed']).reset_index(drop=True)
    
    # Calcular distancias, tiempos y velocidades instantaneas
    print("Calculando distancias, tiempos y velocidades instantaneas...")
    df['lon_prev'] = df.groupby(['caid', 'num_trip'])['lon'].shift(1)
    df['lat_prev'] = df.groupby(['caid', 'num_trip'])['lat'].shift(1)
    df['time_prev'] = df.groupby(['caid', 'num_trip'])['Timestamp_parsed'].shift(1)
    
    df['dist_m'] = haversine_np(df['lon'], df['lat'], df['lon_prev'], df['lat_prev']) * 1000.0
    df['dt_sec'] = (df['Timestamp_parsed'] - df['time_prev']).dt.total_seconds()
    
    df['speed_kmh'] = np.where(
        df['dt_sec'] > 0,
        (df['dist_m'] / 1000.0) / (df['dt_sec'] / 3600.0),
        0.0
    )
    
    # Rellenar nulos
    df['dist_m'] = df['dist_m'].fillna(0.0)
    df['dt_sec'] = df['dt_sec'].fillna(0.0)
    df['speed_kmh'] = df['speed_kmh'].fillna(0.0)
    
    # Auditoria espacial para el Metro (Metrorrey)
    print("Cargando infraestructura del metro para auditoria espacial...")
    subway_df = pd.read_csv(config.FILE_METRO)
    if 'WKT' in subway_df.columns:
        subway_df['geometry'] = subway_df['WKT'].apply(shapely.wkt.loads)
    elif 'geometry' in subway_df.columns:
        subway_df['geometry'] = subway_df['geometry'].apply(shapely.wkt.loads)
    gdf_subway = gpd.GeoDataFrame(subway_df, geometry='geometry', crs="EPSG:4326")
    gdf_subway_proj = gdf_subway.to_crs("EPSG:32614")  # Proyeccion UTM para Monterrey
    
    print("Calculando distancias a vias de metro...")
    df_metro = df[df['modo_transporte'] == 'metro'].copy()
    if not df_metro.empty:
        gdf_metro = gpd.GeoDataFrame(
            df_metro, 
            geometry=gpd.points_from_xy(df_metro['lon'], df_metro['lat']), 
            crs="EPSG:4326"
        )
        gdf_metro_proj = gdf_metro.to_crs("EPSG:32614")
        
        metro_lines_union = gdf_subway_proj.unary_union
        gdf_metro_proj['dist_to_tracks_m'] = gdf_metro_proj.geometry.distance(metro_lines_union)
        df['dist_to_tracks_m'] = df.index.map(gdf_metro_proj['dist_to_tracks_m'])
    else:
        df['dist_to_tracks_m'] = np.nan

    # Regla espacial del metro: mas de 300 metros de cualquier via de Metrorrey
    df['anomalia_metro_vias'] = (df['modo_transporte'] == 'metro') & (df['dist_to_tracks_m'] > 300.0)
    
    df['es_anomalo'] = df['anomalia_metro_vias']
    df['anomalia_velocidad'] = False
    df['anomalia_glitch'] = False
    
    # Evaluar y depurar a nivel de viaje (Trip-Level) con validación secuencial de trayectorias
    print("Evaluando y depurando anomalías de velocidad y viajes...")
    trip_groups = df.groupby(['caid', 'num_trip'])
    
    viajes_descartados = set()
    viajes_prune_count = 0
    puntos_eliminados_prune = 0
    puntos_eliminados_individuales = 0
    puntos_eliminados_static_ends = 0
    
    keep_indices = []
    
    for (caid, trip_id), group in trip_groups:
        total_pings = len(group)
        indices = group.index.to_numpy()
        
        lats = group['lat'].to_numpy()
        lons = group['lon'].to_numpy()
        times = group['Timestamp_parsed'].to_numpy()
        modes = group['modo_transporte'].to_numpy()
        es_anomalo_trip = group['es_anomalo'].to_numpy().copy()
        
        # Validación secuencial de trayectoria: detecta bloques de glitches de cualquier tamaño en una sola pasada
        last_valid_idx = -1
        for i in range(total_pings):
            if not es_anomalo_trip[i]:
                last_valid_idx = i
                break
                
        if last_valid_idx != -1:
            for i in range(last_valid_idx + 1, total_pings):
                if es_anomalo_trip[i]:
                    continue
                    
                lat1, lon1, t1 = lats[last_valid_idx], lons[last_valid_idx], times[last_valid_idx]
                lat2, lon2, t2 = lats[i], lons[i], times[i]
                mode2 = modes[i]
                
                dist = haversine_np(lon1, lat1, lon2, lat2) * 1000.0
                dt = (t2 - t1) / np.timedelta64(1, 's')
                
                speed = (dist / 1000.0) / (dt / 3600.0) if dt > 0 else 0.0
                
                is_anom = False
                if mode2 == 'carro' and speed > 160.0:
                    is_anom = True
                elif mode2 == 'bus' and speed > 110.0:
                    is_anom = True
                elif mode2 == 'metro' and speed > 110.0:
                    is_anom = True
                elif mode2 == 'caminar' and speed > 30.0:
                    is_anom = True
                elif speed > 250.0:
                    is_anom = True
                    
                if is_anom:
                    es_anomalo_trip[i] = True
                else:
                    last_valid_idx = i
                    
        # Propagar es_anomalo al dataframe original para estadísticas finales
        df.loc[indices, 'es_anomalo'] = es_anomalo_trip
        
        # Criterio A: Descarte completo de viaje si supera el 30% de pings anómalos
        if (es_anomalo_trip.sum() / total_pings) > 0.30:
            viajes_descartados.add((caid, trip_id))
            continue
            
        modo_viaje = group['modo_transporte'].iloc[0]
        
        # 1. Poda de Caminatas Vehicularizadas (Fase 3 de Caminar)
        if modo_viaje == 'caminar':
            df_raw_trip = group.copy()
            df_raw_trip['lon_prev'] = df_raw_trip['lon'].shift(1)
            df_raw_trip['lat_prev'] = df_raw_trip['lat'].shift(1)
            df_raw_trip['time_prev'] = df_raw_trip['Timestamp_parsed'].shift(1)
            df_raw_trip['dist_m'] = haversine_np(df_raw_trip['lon'], df_raw_trip['lat'], df_raw_trip['lon_prev'], df_raw_trip['lat_prev']) * 1000.0
            df_raw_trip['dt_sec'] = (df_raw_trip['Timestamp_parsed'] - df_raw_trip['time_prev']).dt.total_seconds()
            df_raw_trip['speed_kmh'] = np.where(df_raw_trip['dt_sec'] > 0, (df_raw_trip['dist_m']/1000.0)/(df_raw_trip['dt_sec']/3600.0), 0.0)
            
            anomalous_indices = df_raw_trip[df_raw_trip['speed_kmh'] > 30.0].index
            if not anomalous_indices.empty:
                first_anom_idx = anomalous_indices[0]
                pos_in_group = group.index.get_loc(first_anom_idx)
                
                # Conservar solo los puntos previos al primer punto de velocidad vehicular
                group_valid = group.iloc[:pos_in_group]
                viajes_prune_count += 1
                puntos_eliminados_prune += (total_pings - len(group_valid))
            else:
                group_valid = group
        else:
            group_valid = group

        # 2. Filtrar glitches individuales (Fase 2)
        valid_mask_in_group_valid = ~df.loc[group_valid.index, 'es_anomalo']
        df_valid = group_valid[valid_mask_in_group_valid].copy()
        
        # Sumar los glitches eliminados en esta fase
        glitches_removed = len(group_valid) - len(df_valid)
        puntos_eliminados_individuales += glitches_removed
        
        if len(df_valid) < 2:
            viajes_descartados.add((caid, trip_id))
            continue
            
        # 3. Recorte de Extremos Estáticos (Trim-to-Motion)
        if modo_viaje == 'caminar':
            motion_threshold = 2.0  # km/h para caminar
        else:
            motion_threshold = 5.0  # km/h para carro, bus, metro
            
        motion_pings = df_valid[df_valid['speed_kmh'] >= motion_threshold]
        
        if not motion_pings.empty:
            first_motion_idx = motion_pings.index[0]
            last_motion_idx = motion_pings.index[-1]
            
            pings_before_trim = len(df_valid)
            df_valid = df_valid.loc[first_motion_idx:last_motion_idx]
            pings_after_trim = len(df_valid)
            
            puntos_eliminados_extremos = pings_before_trim - pings_after_trim
            if puntos_eliminados_extremos > 0:
                puntos_eliminados_static_ends += puntos_eliminados_extremos
        else:
            viajes_descartados.add((caid, trip_id))
            continue
            
        if len(df_valid) < 2:
            viajes_descartados.add((caid, trip_id))
            continue
            
        keep_indices.extend(df_valid.index)
                
    # Filtrar el DataFrame
    df_cleaned = df.loc[keep_indices].copy()
    
    # Volver a ordenar cronologicamente por caid, num_trip, Timestamp_parsed
    df_cleaned = df_cleaned.sort_values(by=['caid', 'num_trip', 'Timestamp_parsed']).reset_index(drop=True)
    
    # Eliminar columnas de procesamiento para dejar exactamente el mismo formato original
    cols_to_drop = [
        'Timestamp_parsed', 'modo_transporte', 'lon_prev', 'lat_prev', 'time_prev',
        'dist_m', 'dt_sec', 'speed_kmh', 'dist_to_tracks_m', 'anomalia_velocidad',
        'anomalia_metro_vias', 'anomalia_glitch', 'es_anomalo'
    ]
    df_cleaned_output = df_cleaned.drop(columns=cols_to_drop, errors='ignore')
    
    # Guardar en CSV
    print(f"Guardando dataset depurado en: {output_path}")
    df_cleaned_output.to_csv(output_path, index=False)
    
    # Calcular y reportar estadisticas
    total_original_pings = len(df)
    total_cleaned_pings = len(df_cleaned_output)
    total_removed_pings = total_original_pings - total_cleaned_pings
    
    total_original_trips = len(trip_groups)
    total_removed_trips = len(viajes_descartados)
    
    pings_viajes_descartados = df[df.set_index(['caid', 'num_trip']).index.isin(viajes_descartados)].shape[0]
    
    print("\n=== RESUMEN DE LA DEPURACION ===")
    print(f"Pings Originales: {total_original_pings:,}")
    print(f"Pings Limpios Guardados: {total_cleaned_pings:,}")
    print(f"Total de Pings Eliminados: {total_removed_pings:,} ({total_removed_pings/total_original_pings*100:.2f}%)")
    print(f"  - Por descarte de viajes completos (>30% anomalos o vacios): {pings_viajes_descartados:,}")
    print(f"  - Por poda de caminata (seccion vehicular eliminada): {puntos_eliminados_prune:,}")
    print(f"  - Por recorte de extremos estaticos (Trim-to-Motion): {puntos_eliminados_static_ends:,}")
    print(f"  - Por glitches individuales en viajes validos: {puntos_eliminados_individuales:,}")
    print(f"Viajes Originales: {total_original_trips:,}")
    print(f"Viajes Completamente Eliminados: {total_removed_trips:,} ({total_removed_trips/total_original_trips*100:.2f}%)")
    print(f"Viajes Peatonales Podados (Truncados): {viajes_prune_count:,}")
    print("=================================\n")

if __name__ == '__main__':
    main()

