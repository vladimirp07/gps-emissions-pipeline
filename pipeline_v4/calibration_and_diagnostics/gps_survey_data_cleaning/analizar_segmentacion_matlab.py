import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path for importing pipeline_v4 modules
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from pipeline_v4.src.segmentation import assign_trips, apply_spatial_filter, haversine_vectorized

def run_segmentation_analysis():
    print("=== INICIANDO ANÁLISIS DE SEGMENTACIÓN MATRICIAL (VERASET VS MANUAL) ===")
    
    # 1. Rutas de archivos
    gps_dir = PROJECT_ROOT / "Inputs" / "GPS User Data"
    input_path = gps_dir / "Datos de MATLAB GPS.csv"
    output_json_path = PROJECT_ROOT / "pipeline_v4" / "calibration_and_diagnostics" / "gps_survey_data_cleaning" / "resultados_segmentacion.json"
    
    if not input_path.exists():
        print(f"Error: No se encontró el archivo {input_path}")
        sys.exit(1)
        
    print(f"Cargando dataset de MATLAB desde: {input_path}")
    df = pd.read_csv(input_path, dtype={"altitude": str, "course": str, "hacc": str})
    print(f"Cargado exitosamente. Filas: {len(df):,}")
    
    # Preprocesar columnas requeridas para la máquina de estados
    df['local_timestamp'] = pd.to_datetime(df['Timestamp'])
    df['latitude'] = df['lat']
    df['longitude'] = df['lon']
    df['date'] = df['local_timestamp'].dt.date
    df['original_index'] = df.index
    df['mode_clean'] = df['mode_of_transport'].str.strip().str.lower()
    
    # Ordenar por usuario y tiempo cronológico
    df = df.sort_values(by=['caid', 'local_timestamp']).reset_index(drop=True)
    
    # 2. Segmentar día por día y usuario por usuario (emulando el Orchestrator)
    print("Ejecutando máquina de estados de Veraset usuario-día por usuario-día...")
    segmented_dfs = []
    grouped = df.groupby(['caid', 'date'])
    
    for (user, dt), group in grouped:
        group_sorted = group.sort_values('local_timestamp').copy()
        
        # Filtro espacial de 15 metros
        group_filtered = apply_spatial_filter(group_sorted, min_dist_m=15.0)
        
        if not group_filtered.empty:
            group_segmented = assign_trips(group_filtered)
            segmented_dfs.append(group_segmented)
            
    df_segmented = pd.concat(segmented_dfs, ignore_index=True)
    print(f"Segmentación completada. Registros resultantes tras filtro espacial: {len(df_segmented):,}")
    
    # Mapear los identificadores de viaje asignados por el algoritmo de vuelta al df original
    df_segmented_map = df_segmented.set_index('original_index')['trip']
    df['algo_trip'] = df['original_index'].map(df_segmented_map)
    # Rellenar los puntos filtrados mediante propagación temporal dentro de cada usuario-día
    df['algo_trip'] = df.groupby(['caid', 'date'])['algo_trip'].ffill().bfill()
    
    # Calcular velocidades instantáneas geodésicas en el df original para la validación física
    print("Calculando velocidades geodésicas instantáneas...")
    df['lon_prev'] = df.groupby(['caid'])['longitude'].shift(1)
    df['lat_prev'] = df.groupby(['caid'])['latitude'].shift(1)
    df['time_prev'] = df.groupby(['caid'])['local_timestamp'].shift(1)
    
    df['dist_m'] = haversine_vectorized(df['latitude'], df['longitude'], df['lat_prev'], df['lon_prev']) * 1000.0
    df['dt_sec'] = (df['local_timestamp'] - df['time_prev']).dt.total_seconds().fillna(0.0)
    df['speed_kmh'] = np.where(df['dt_sec'] > 0, (df['dist_m'] / 1000.0) / (df['dt_sec'] / 3600.0), 0.0)
    
    df['dist_m'] = df['dist_m'].fillna(0.0)
    df['speed_kmh'] = df['speed_kmh'].fillna(0.0)
    
    # ------------------ ANÁLISIS 1: PARADAS EN VIAJES MANUALES ------------------
    print("\n--- Analizando Paradas en Viajes Manuales ---")
    manual_trips = df.groupby(['caid', 'num_trip'])
    total_manual_trips = 0
    trips_with_any_stop = 0
    trips_with_stop_gt_5min = 0
    stop_durations = []
    
    for (user, num_trip), group in manual_trips:
        total_manual_trips += 1
        stop_points = group[group['algo_trip'] < 0]
        if not stop_points.empty:
            trips_with_any_stop += 1
            stop_time_sec = stop_points['dt_sec'].sum()
            stop_durations.append(stop_time_sec / 60.0) # en minutos
            if stop_time_sec >= 300.0:
                trips_with_stop_gt_5min += 1
                
    print(f"Viajes manuales totales: {total_manual_trips}")
    print(f"Viajes con algún punto de parada detectado: {trips_with_any_stop} ({trips_with_any_stop/total_manual_trips*100:.2f}%)")
    print(f"Viajes con paradas acumuladas >= 5 min: {trips_with_stop_gt_5min} ({trips_with_stop_gt_5min/total_manual_trips*100:.2f}%)")
    
    # ------------------ ANÁLISIS 2: CRUCE Y COINCIDENCIA DE INICIO/FIN ------------------
    print("\n--- Analizando Coincidencia de Inicio/Fin de Viajes ---")
    # Intervalos de viajes manuales
    manual_intervals = []
    for (user, num_trip), group in df.groupby(['caid', 'num_trip']):
        t_start = group['local_timestamp'].min()
        t_end = group['local_timestamp'].max()
        manual_intervals.append({
            'caid': user,
            'num_trip': num_trip,
            't_start_m': t_start,
            't_end_m': t_end,
            'duration_m_min': (t_end - t_start).total_seconds() / 60.0,
            'mode': group['mode_clean'].iloc[0]
        })
    df_m_intervals = pd.DataFrame(manual_intervals)
    
    # Intervalos de viajes automáticos (diarios)
    auto_intervals = []
    for (user, dt, trip_id), group in df_segmented[df_segmented['trip'] > 0].groupby(['caid', 'date', 'trip']):
        t_start = group['local_timestamp'].min()
        t_end = group['local_timestamp'].max()
        auto_intervals.append({
            'caid': user,
            'date': dt,
            'trip': trip_id,
            't_start_a': t_start,
            't_end_a': t_end,
            'duration_a_min': (t_end - t_start).total_seconds() / 60.0
        })
    df_a_intervals = pd.DataFrame(auto_intervals)
    
    matches = []
    unmatched_manual = 0
    
    for idx_m, row_m in df_m_intervals.iterrows():
        user = row_m['caid']
        t_sm = row_m['t_start_m']
        t_em = row_m['t_end_m']
        
        user_autos = df_a_intervals[df_a_intervals['caid'] == user]
        
        best_match = None
        max_overlap = 0.0
        best_iou = 0.0
        best_dt_start = 0.0
        best_dt_end = 0.0
        
        for idx_a, row_a in user_autos.iterrows():
            t_sa = row_a['t_start_a']
            t_ea = row_a['t_end_a']
            
            # Compute overlap
            overlap_start = max(t_sm, t_sa)
            overlap_end = min(t_em, t_ea)
            overlap_sec = max(0.0, (overlap_end - overlap_start).total_seconds())
            
            if overlap_sec > 0:
                union_start = min(t_sm, t_sa)
                union_end = max(t_em, t_ea)
                union_sec = (union_end - union_start).total_seconds()
                iou = overlap_sec / union_sec if union_sec > 0 else 0.0
                
                if overlap_sec > max_overlap:
                    max_overlap = overlap_sec
                    best_iou = iou
                    best_match = row_a['trip']
                    best_dt_start = (t_sa - t_sm).total_seconds() / 60.0
                    best_dt_end = (t_ea - t_em).total_seconds() / 60.0
                    
        if best_match is not None:
            matches.append({
                'caid': user,
                'num_trip': row_m['num_trip'],
                'trip': best_match,
                'iou': best_iou,
                'dt_start_min': best_dt_start,
                'dt_end_min': best_dt_end,
                'abs_dt_start_min': abs(best_dt_start),
                'abs_dt_end_min': abs(best_dt_end),
                'duration_m_min': row_m['duration_m_min'],
                'mode': row_m['mode']
            })
        else:
            unmatched_manual += 1
            matches.append({
                'caid': user,
                'num_trip': row_m['num_trip'],
                'trip': np.nan,
                'iou': 0.0,
                'dt_start_min': np.nan,
                'dt_end_min': np.nan,
                'abs_dt_start_min': np.nan,
                'abs_dt_end_min': np.nan,
                'duration_m_min': row_m['duration_m_min'],
                'mode': row_m['mode']
            })
            
    df_matches = pd.DataFrame(matches)
    df_valid_matches = df_matches.dropna(subset=['trip'])
    
    print(f"Viajes manuales emparejados con viajes del algoritmo: {len(df_valid_matches)} ({len(df_valid_matches)/len(df_matches)*100:.2f}%)")
    print(f"Viajes manuales no emparejados (parada pura según algoritmo): {unmatched_manual} ({unmatched_manual/len(df_matches)*100:.2f}%)")
    print(f"IoU Promedio (emparejados): {df_valid_matches['iou'].mean():.4f}")
    print(f"IoU Mediana (emparejados): {df_valid_matches['iou'].median():.4f}")
    print(f"Emparejamientos Buenos (IoU >= 0.50): {(df_matches['iou'] >= 0.50).sum()} ({(df_matches['iou'] >= 0.50).mean()*100:.2f}%)")
    print(f"Emparejamientos Excelentes (IoU >= 0.80): {(df_matches['iou'] >= 0.80).sum()} ({(df_matches['iou'] >= 0.80).mean()*100:.2f}%)")
    
    # ------------------ ANÁLISIS 3: VALIDACIÓN LÓGICA DE MODOS ------------------
    print("\n--- Validando Físicamente los Modos de Transporte ---")
    speed_stats = []
    for mode in df['mode_clean'].unique():
        mode_df = df[df['mode_clean'] == mode]
        total_pings = len(mode_df)
        if total_pings == 0:
            continue
            
        mean_spd = mode_df['speed_kmh'].mean()
        median_spd = mode_df['speed_kmh'].median()
        max_spd = mode_df['speed_kmh'].max()
        p95_spd = mode_df['speed_kmh'].quantile(0.95)
        p99_spd = mode_df['speed_kmh'].quantile(0.99)
        
        # Limitar anomalias
        if mode == 'caminar':
            ex_6 = (mode_df['speed_kmh'] > 6.0).sum()
            ex_10 = (mode_df['speed_kmh'] > 10.0).sum()
            ex_30 = (mode_df['speed_kmh'] > 30.0).sum()
            pct_ex = ex_30 / total_pings * 100.0
            anomaly_info = {
                'exceed_6_pct': float(ex_6/total_pings*100),
                'exceed_10_pct': float(ex_10/total_pings*100),
                'exceed_30_pct': float(pct_ex),
                'exceed_30_count': int(ex_30)
            }
        elif mode == 'carro':
            ex_120 = (mode_df['speed_kmh'] > 120.0).sum()
            ex_160 = (mode_df['speed_kmh'] > 160.0).sum()
            pct_ex = ex_160 / total_pings * 100.0
            anomaly_info = {
                'exceed_120_pct': float(ex_120/total_pings*100),
                'exceed_160_pct': float(pct_ex),
                'exceed_160_count': int(ex_160)
            }
        elif mode == 'bus':
            ex_80 = (mode_df['speed_kmh'] > 80.0).sum()
            ex_110 = (mode_df['speed_kmh'] > 110.0).sum()
            pct_ex = ex_110 / total_pings * 100.0
            anomaly_info = {
                'exceed_80_pct': float(ex_80/total_pings*100),
                'exceed_110_pct': float(pct_ex),
                'exceed_110_count': int(ex_110)
            }
        elif mode == 'metro':
            ex_90 = (mode_df['speed_kmh'] > 90.0).sum()
            ex_110 = (mode_df['speed_kmh'] > 110.0).sum()
            pct_ex = ex_110 / total_pings * 100.0
            anomaly_info = {
                'exceed_90_pct': float(ex_90/total_pings*100),
                'exceed_110_pct': float(pct_ex),
                'exceed_110_count': int(ex_110)
            }
        else:
            anomaly_info = {}
            
        speed_stats.append({
            'mode': mode,
            'total_pings': total_pings,
            'mean_speed_kmh': mean_spd,
            'median_speed_kmh': median_spd,
            'max_speed_kmh': max_spd,
            'p95_speed_kmh': p95_spd,
            'p99_speed_kmh': p99_spd,
            'anomalies': anomaly_info
        })
        
    # Paradas largas registradas en tránsito
    stop_transit_stats = []
    for mode in ['caminar', 'carro', 'bus', 'metro']:
        mode_all = df[df['mode_clean'] == mode]
        mode_stop = mode_all[mode_all['algo_trip'] < 0]
        
        tot_sec = mode_all['dt_sec'].sum()
        stp_sec = mode_stop['dt_sec'].sum()
        pct_stp = stp_sec / tot_sec * 100.0 if tot_sec > 0 else 0.0
        
        stop_transit_stats.append({
            'mode': mode,
            'total_hours': float(tot_sec/3600.0),
            'stop_hours': float(stp_sec/3600.0),
            'pct_in_stops': float(pct_stp)
        })
        
    # 4. Guardar resultados estructurados
    summary_results = {
        'total_manual_trips': total_manual_trips,
        'trips_with_any_stop': trips_with_any_stop,
        'pct_trips_with_any_stop': trips_with_any_stop / total_manual_trips * 100.0,
        'trips_with_stop_gt_5min': trips_with_stop_gt_5min,
        'pct_trips_with_stop_gt_5min': trips_with_stop_gt_5min / total_manual_trips * 100.0,
        'matching': {
            'matched_count': len(df_valid_matches),
            'unmatched_count': unmatched_manual,
            'pct_matched': (len(df_matches) - unmatched_manual) / len(df_matches) * 100.0,
            'iou_mean': float(df_valid_matches['iou'].mean()),
            'iou_median': float(df_valid_matches['iou'].median()),
            'iou_gt_05_count': int((df_matches['iou'] >= 0.50).sum()),
            'iou_gt_05_pct': float((df_matches['iou'] >= 0.50).mean() * 100.0),
            'iou_gt_08_count': int((df_matches['iou'] >= 0.80).sum()),
            'iou_gt_08_pct': float((df_matches['iou'] >= 0.80).mean() * 100.0),
            'dt_start_abs_mean_min': float(df_valid_matches['abs_dt_start_min'].mean()),
            'dt_start_abs_median_min': float(df_valid_matches['abs_dt_start_min'].median()),
            'dt_end_abs_mean_min': float(df_valid_matches['abs_dt_end_min'].mean()),
            'dt_end_abs_median_min': float(df_valid_matches['abs_dt_end_min'].median())
        },
        'speed_validation_by_mode': speed_stats,
        'stops_in_transit_by_mode': stop_transit_stats
    }
    
    # Encoder JSON personalizado
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)
            
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=4, cls=NpEncoder)
        
    print(f"Resultados estructurados guardados en: {output_json_path}")
    print("=== ANÁLISIS COMPLETADO EXITOSAMENTE ===")

if __name__ == '__main__':
    run_segmentation_analysis()

