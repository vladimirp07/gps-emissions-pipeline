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
from pipeline_v3.src import config

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

    # Identificar anomalias a nivel de punto
    print("Evaluando anomalias a nivel de punto...")
    df['anomalia_velocidad'] = False
    
    # Reglas de velocidad maxima por modo de transporte
    df.loc[(df['modo_transporte'] == 'carro') & (df['speed_kmh'] > 160.0), 'anomalia_velocidad'] = True
    df.loc[(df['modo_transporte'] == 'bus') & (df['speed_kmh'] > 110.0), 'anomalia_velocidad'] = True
    df.loc[(df['modo_transporte'] == 'metro') & (df['speed_kmh'] > 110.0), 'anomalia_velocidad'] = True
    # Nota: El umbral de velocidad para caminar se define en 30 km/h por sobreestimacion de Haversine
    df.loc[(df['modo_transporte'] == 'caminar') & (df['speed_kmh'] > 30.0), 'anomalia_velocidad'] = True
    
    # Regla espacial del metro: mas de 300 metros de cualquier via de Metrorrey
    df['anomalia_metro_vias'] = (df['modo_transporte'] == 'metro') & (df['dist_to_tracks_m'] > 300.0)
    
    # Glitch general: saltos de posicion que requieran velocidad superior a 250 km/h
    df['anomalia_glitch'] = (df['speed_kmh'] > 250.0)
    
    # Marcador total de anomalia
    df['es_anomalo'] = df['anomalia_velocidad'] | df['anomalia_metro_vias'] | df['anomalia_glitch']
    
    # Evaluar y depurar a nivel de viaje (Trip-Level)
    print("Evaluando y depurando a nivel de viaje...")
    trip_groups = df.groupby(['caid', 'num_trip'])
    
    viajes_descartados = set()
    viajes_prune_count = 0
    puntos_eliminados_prune = 0
    puntos_eliminados_individuales = 0
    
    keep_indices = []
    
    for (caid, trip_id), group in trip_groups:
        total_pings = len(group)
        anomalous_pings = group['es_anomalo'].sum()
        pct_anomalous = (anomalous_pings / total_pings) * 100.0
        
        # Criterio A: Descarte completo si mas del 30% de los pings son anomalos
        if pct_anomalous > 30.0:
            viajes_descartados.add((caid, trip_id))
            continue
            
        modo_viaje = group['modo_transporte'].iloc[0]
        
        if modo_viaje == 'caminar':
            # Criterio B: Poda de viaje caminar en el primer punto de transicion a vehiculo (> 30 km/h)
            anomalous_indices = group[group['speed_kmh'] > 30.0].index
            if not anomalous_indices.empty:
                first_anom_idx = anomalous_indices[0]
                pos_in_group = group.index.get_loc(first_anom_idx)
                
                # Conservar solo los puntos previos al primer punto de velocidad vehicular
                group_valid = group.iloc[:pos_in_group]
                
                # Descartar el viaje completo si la porcion peatonal valida es muy corta
                if len(group_valid) < 2:
                    viajes_descartados.add((caid, trip_id))
                else:
                    viajes_prune_count += 1
                    puntos_eliminados_prune += (total_pings - len(group_valid))
                    
                    # Eliminar glitches individuales dentro de la porcion valida de caminar
                    valid_indices = group_valid[~group_valid['es_anomalo']].index
                    puntos_eliminados_individuales += (len(group_valid) - len(valid_indices))
                    
                    if len(valid_indices) < 2:
                        viajes_descartados.add((caid, trip_id))
                    else:
                        keep_indices.extend(valid_indices)
            else:
                # No hay velocidad vehicular, remover glitches individuales y verificar tamaño
                valid_indices = group[~group['es_anomalo']].index
                puntos_eliminados_individuales += (total_pings - len(valid_indices))
                
                if len(valid_indices) < 2:
                    viajes_descartados.add((caid, trip_id))
                else:
                    keep_indices.extend(valid_indices)
        else:
            # Para modos no peatonales: remover glitches y pings anomalos individuales
            valid_indices = group[~group['es_anomalo']].index
            puntos_eliminados_individuales += (total_pings - len(valid_indices))
            
            if len(valid_indices) < 2:
                viajes_descartados.add((caid, trip_id))
            else:
                keep_indices.extend(valid_indices)
                
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
    print(f"  - Por glitches individuales en viajes validos: {puntos_eliminados_individuales:,}")
    print(f"Viajes Originales: {total_original_trips:,}")
    print(f"Viajes Completamente Eliminados: {total_removed_trips:,} ({total_removed_trips/total_original_trips*100:.2f}%)")
    print(f"Viajes Peatonales Podados (Truncados): {viajes_prune_count:,}")
    print("=================================\n")

if __name__ == '__main__':
    main()
