import os
import sys
import subprocess
import time
from pathlib import Path

# Force stdout/stderr to handle encodings gracefully
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
except AttributeError:
    pass # Older Python versions

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ejecutar la generación de datos de entrenamiento en paralelo.")
    parser.add_argument("--num-chunks", type=int, default=2, help="Número de chunks (procesos en paralelo) a ejecutar.")
    parser.add_argument("--trips-per-mode", type=int, default=2, help="Número de viajes por modo a procesar (bajo para pruebas).")
    args = parser.parse_args()

    num_chunks = args.num_chunks
    trips_per_mode = args.trips_per_mode

    print("=== INICIANDO PIPELINE DE GENERACION EN PARALELO ===")
    print(f"Configuración: num_chunks={num_chunks}, trips_per_mode={trips_per_mode}")
    
    script_path = Path(__file__).parent / "generar_datos_entrenamiento.py"
    
    processes = []
    start_time = time.time()
    
    for chunk_id in range(num_chunks):
        cmd = [
            sys.executable,
            str(script_path),
            "--balanced",
            "--trips-per-mode", str(trips_per_mode),
            "--num-chunks", str(num_chunks),
            "--chunk-id", str(chunk_id)
        ]
        print(f"Lanzando Chunk {chunk_id}/{num_chunks}...")
        # Use errors="replace" to handle any system-specific non-UTF-8 characters in subprocess streams
        p = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            encoding="utf-8", 
            errors="replace"
        )
        processes.append((chunk_id, p))
        
    print("\nTodos los subprocesos han sido lanzados. Monitoreando ejecucion...")
    
    # Wait for all processes to finish and print their status
    completed = 0
    errors = []
    
    while completed < num_chunks:
        time.sleep(2)
        completed = 0
        for chunk_id, p in processes:
            poll = p.poll()
            if poll is not None:
                completed += 1
                
        print(f"  -> Chunks completados: {completed}/{num_chunks} (Tiempo transcurrido: {time.time() - start_time:.1f}s)", end="\r")
        
    print("\n\nTodos los procesos terminaron. Analizando resultados...")
    
    for chunk_id, p in processes:
        out, err = p.communicate()
        if p.returncode != 0:
            print(f"[ERROR] Chunk {chunk_id} fallo con codigo {p.returncode}!")
            print(f"--- Errores del Chunk {chunk_id} ---")
            print(err)
            errors.append(chunk_id)
        else:
            print(f"[OK] Chunk {chunk_id} completado con exito.")
            # Print the last few lines of the output for context
            lines = out.strip().split("\n")
            summary_lines = [l for l in lines if "muestras de entrenamiento" in l or "Generacion completada" in l or "PROCESANDO" in l]
            for l in summary_lines:
                print(f"   [{chunk_id}]: {l}")
                
    if errors:
        print("\n[ERROR] La generacion en paralelo fallo en algunos chunks. Fusion cancelada.")
        sys.exit(1)
        
    print("\n Fusion de los chunks...")
    merge_script = Path(__file__).parent / "merge_chunks.py"
    res = subprocess.run(
        [sys.executable, str(merge_script)], 
        capture_output=True, 
        text=True, 
        encoding="utf-8", 
        errors="replace"
    )
    print(res.stdout)
    if res.stderr:
        print("Errores en la fusion:")
        print(res.stderr)
        
    print("\n Depuracion de datos fisicamente imposibles...")
    clean_script = Path(__file__).parent / "clean_dataset.py"
    res_clean = subprocess.run(
        [sys.executable, str(clean_script)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    print(res_clean.stdout)
    if res_clean.stderr:
        print("Errores en la depuracion:")
        print(res_clean.stderr)
        
    print(f"\n[OK] PIPELINE TERMINADO CON EXITO! Tiempo total: {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()
