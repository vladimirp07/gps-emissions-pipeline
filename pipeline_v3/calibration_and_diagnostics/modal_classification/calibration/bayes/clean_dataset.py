import sys
import pickle
import numpy as np
from pathlib import Path

# Force stdout/stderr to handle encodings gracefully on Windows consoles
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
except AttributeError:
    pass

# Configurar rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from pipeline_v3.src import config

def main():
    print("=== DEPURACION DE DATOS FISICAMENTE IMPOSIBLES ===")
    gps_dir = config.GPS_DIR
    dataset_path = gps_dir / "datos_entrenamiento_optuna.pkl"
    
    if not dataset_path.exists():
        print(f"Error: No se encontro el archivo de entrenamiento en {dataset_path}")
        sys.exit(1)
        
    with open(dataset_path, 'rb') as f:
        records = pickle.load(f)
        
    print(f"Muestras totales iniciales: {len(records)}")
    
    # Agrupar muestras por viaje original (caid y num_trip) para descartar el viaje completo
    # trip_id formato: caid_trip-label_deg_hip
    trips_data = {}
    for r in records:
        trip_key = r['trip_id'].split('-')[0] # Obtiene caid_trip
        if trip_key not in trips_data:
            trips_data[trip_key] = []
        trips_data[trip_key].append(r)
        
    trips_to_remove = set()
    reasons = {}
    
    for trip_key, samples in trips_data.items():
        # Tomar una muestra representativa para revisar el label real y propiedades físicas
        # ya que todos los samples del mismo viaje comparten el label real y trayectoria física original.
        first_sample = samples[0]
        label = first_sample['label']
        
        if label.lower() == 'bus':
            # 1. Filtro de Velocidad Extrema (percentil 95 > 80 km/h)
            # Analizamos todas las velocidades de los pings ruteados
            speeds = []
            for s in samples:
                if 'speed_raw' in s and len(s['speed_raw']) > 0:
                    speeds.extend(s['speed_raw'])
            
            if len(speeds) > 0:
                p95_speed = np.percentile(speeds, 95)
                max_speed = np.max(speeds)
                if p95_speed > 80.0:
                    trips_to_remove.add(trip_key)
                    reasons[trip_key] = f"Velocidad extrema de Bus (P95: {p95_speed:.1f} km/h, Max: {max_speed:.1f} km/h)"
                    continue
            
            # 2. Filtro de Red Faltante (0% cercanía a rutas de autobús)
            # idx_c: Cercanía (0: Metro, 1: Bus, 2: Ninguno)
            total_points = 0
            bus_points = 0
            for s in samples:
                if 'idx_c' in s and len(s['idx_c']) > 0:
                    total_points += len(s['idx_c'])
                    bus_points += np.sum(s['idx_c'] == 1)
            
            if total_points > 0:
                proximity_pct = (bus_points / total_points) * 100.0
                if proximity_pct == 0.0:
                    trips_to_remove.add(trip_key)
                    reasons[trip_key] = f"Sin cobertura de red de Autobus (0% cercania)"
                    
    # Filtrar registros
    cleaned_records = [r for r in records if r['trip_id'].split('-')[0] not in trips_to_remove]
    
    print("\n--- Reporte de Depuracion ---")
    if trips_to_remove:
        print(f"Viajes descartados ({len(trips_to_remove)}):")
        for trip_key in sorted(list(trips_to_remove)):
            print(f"  * {trip_key}: {reasons[trip_key]}")
    else:
        print("No se encontraron viajes anomalos que violen los limites fisicos.")
        
    print(f"\nMuestras eliminadas: {len(records) - len(cleaned_records)}")
    print(f"Muestras limpias guardadas: {len(cleaned_records)}")
    
    # Sobrescribir el archivo pickle con los datos limpios
    with open(dataset_path, 'wb') as f:
        pickle.dump(cleaned_records, f)
        
    print("Depuracion completada con exito! ✅")

if __name__ == '__main__':
    main()
