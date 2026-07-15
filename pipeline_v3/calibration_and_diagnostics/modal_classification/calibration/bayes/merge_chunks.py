import sys
import pickle
import glob
from pathlib import Path

# Force stdout/stderr to handle encodings gracefully on Windows consoles
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
except AttributeError:
    pass # Older Python versions

# Configurar rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from pipeline_v3.src import config

def main():
    print("=== MERGIENDO CHUNKS DE ENTRENAMIENTO ===")
    gps_dir = config.GPS_DIR
    chunk_files = sorted(glob.glob(str(gps_dir / "datos_entrenamiento_optuna_chunk_*.pkl")))
    
    if not chunk_files:
        print("No se encontraron archivos de chunks para mergir.")
        return
        
    combined_records = []
    for cf in chunk_files:
        print(f"Cargando chunk: {Path(cf).name}")
        with open(cf, 'rb') as f:
            records = pickle.load(f)
            combined_records.extend(records)
            
    output_pkl = gps_dir / "datos_entrenamiento_optuna.pkl"
    print(f"\nGuardando {len(combined_records)} muestras totales en: {output_pkl}")
    with open(output_pkl, 'wb') as f:
        pickle.dump(combined_records, f)
        
    # Limpiar archivos temporales
    for cf in chunk_files:
        try:
            Path(cf).unlink()
            print(f"Eliminado archivo temporal: {Path(cf).name}")
        except Exception as e:
            print(f"Error al eliminar {Path(cf).name}: {e}")
            
    print("\nProceso de fusion completado con exito! [OK]")

if __name__ == '__main__':
    main()
