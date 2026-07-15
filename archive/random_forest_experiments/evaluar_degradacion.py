import sys
import pickle
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
except AttributeError:
    pass

PROJECT_ROOT = Path(r"C:\Users\Eydan\OneDrive\Escritorio\ITESM\MAITEC Lab\Eventos Masivos\GPS_Emissions_Project_Pipeline-v2.0")
sys.path.append(str(PROJECT_ROOT))

from pipeline_v3.src import config

MODOS = ["Carro", "Bus", "Metro", "Caminar"]
MODE_TO_IDX = {m: i for i, m in enumerate(MODOS)}

def main():
    pkl_path = config.GPS_DIR / "datos_entrenamiento_optuna.pkl"
    with open(pkl_path, "rb") as f:
        data_cache = pickle.load(f)

    print("=====================================================================")
    print("      EVALUACIÓN DE CLASIFICACIÓN POR NIVEL DE DEGRADACIÓN GPS")
    print("                (LÓGICA DE DECISIÓN DE PRODUCCIÓN)")
    print("=====================================================================")
    print(f"Caché cargado: {len(data_cache)} hipótesis en total.")

    # Cargar matrices optimizadas
    json_path = Path(__file__).parent / "matrices_optimas.json"
    if json_path.exists():
        print(f"Cargando matrices optimizadas desde {json_path.name}...")
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        m = meta["matrices"]
        Cercania = np.array(m["Cercania"])
        Velocidad = np.array(m["Velocidad"])
        Distancia = np.array(m["Distancia"])
        Velprom = np.array(m["Velprom"])
    else:
        print("[WARN] No se encontró matrices_optimas.json. Usando matrices del paper...")
        Cercania = np.array([
            [0.0661, 0.0058, 0.9007, 0.0274],
            [0.0004, 0.0619, 0.0011, 0.9366],
            [0.0522, 0.0084, 0.0321, 0.9073],
        ])
        Velocidad = np.array([
            [0.0955, 0.0022, 0.8176, 0.0847],
            [0.6656, 0.0560, 0.0002, 0.2782],
            [0.1846, 0.0028, 0.8001, 0.0125],
            [0.8505, 0.0001, 0.1485, 0.0010],
        ])
        Distancia = np.array([
            [0.6026, 0.2618, 0.1159, 0.0196],
            [0.0003, 0.8000, 0.1287, 0.0711],
            [0.7892, 0.1721, 0.0100, 0.0287],
            [0.2254, 0.4666, 0.3019, 0.0061],
            [0.0087, 0.9083, 0.0001, 0.0830],
        ])
        Velprom = np.array([
            [0.3011, 0.1768, 0.0732, 0.4489],
            [0.7003, 0.2994, 0.0003, 0.0001],
        ])

    # Agrupar hipótesis por viaje físico único
    # trip_id tiene formato: caid_trip_degradation_hypothesis
    trip_groups = {}
    for item in data_cache:
        parts = item["trip_id"].split("_")
        base_trip_id = f"{parts[0]}_{parts[1]}"
        deg = item.get("degradacion", "Raw")
        
        # Agrupar por viaje y nivel de degradación
        group_key = f"{base_trip_id}_{deg}"
        if group_key not in trip_groups:
            trip_groups[group_key] = {
                "label": item["label"],
                "degradacion": deg,
                "records": []
            }
        trip_groups[group_key]["records"].append(item)

    # Evaluar por degradación
    deg_groups = {"Raw": [], "L1": [], "L2": [], "L3": []}
    for group_key, info in trip_groups.items():
        deg = info["degradacion"]
        if deg in deg_groups:
            deg_groups[deg].append(info)

    print("\n--- RESULTADOS DE CLASIFICACIÓN POR DEGRADACIÓN ---")
    print("| Nivel GPS   | Viajes   | Correctos | Balanced Accuracy | Carro (Acc) | Bus (Acc) | Metro (Acc) | Caminar (Acc) |")
    print("|-------------|----------|-----------|-------------------|-------------|-----------|-------------|---------------|")

    THRESHOLD_BUS = 0.70 # Exigir al menos 70% de cobertura en la ruta de bus

    for deg_name in ["Raw", "L1", "L2", "L3"]:
        groups = deg_groups[deg_name]
        if not groups:
            print(f"| {deg_name:<11} | 0        | 0         | 0.00%             | N/A         | N/A       | N/A         | N/A           |")
            continue

        y_true = []
        y_pred = []

        for group in groups:
            label = group["label"]
            
            # Simular clasificación de producción
            scores = {}
            probs_by_mode = {}
            idx_c_by_mode = {}
            
            for record in group["records"]:
                modo_hip = record["modo_hipotesis"].lower()
                
                idx_c = record["idx_c"]
                idx_v = record["idx_v"]
                idx_d = record["idx_d_arr"]
                idx_vp = record["idx_vp_arr"]
                
                # APLICAR EL MISMO FILTRADO QUE EN EL ENTRENAMIENTO
                if modo_hip == "metro":
                    mask = (idx_c == 0) & (idx_v > 0)
                    if mask.any():
                        idx_c, idx_v, idx_d = idx_c[mask], idx_v[mask], idx_d[mask]
                        idx_vp = np.ones_like(idx_c)
                elif modo_hip in ["carro", "bus"]:
                    mask = (idx_v > 0)
                    if mask.any():
                        idx_c, idx_v, idx_d = idx_c[mask], idx_v[mask], idx_d[mask]
                        idx_vp = np.ones_like(idx_c)
                        
                if len(idx_c) == 0:
                    continue
                
                P_unnorm = Cercania[idx_c] * Velocidad[idx_v] * Distancia[idx_d] * Velprom[idx_vp]
                row_sums = P_unnorm.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0.0] = 1.0
                P_norm = P_unnorm / row_sums
                
                mean_probs = P_norm.mean(axis=0)
                probs_by_mode[modo_hip] = mean_probs
                idx_c_by_mode[modo_hip] = idx_c
                
                if modo_hip == "caminar":
                    scores["Caminar"] = mean_probs[3]
                elif modo_hip == "metro":
                    scores["Metro"] = mean_probs[2]
                elif modo_hip in ["carro", "bus"]:
                    scores["Carro"] = mean_probs[0]
                    
            if not scores:
                pred = "Caminar"
            else:
                # 1. Determinar el ganador de hipótesis
                best_mode = max(scores, key=scores.get)
                pred = best_mode
                
                # 2. Si el ganador es la red vial, resolver Carro vs Bus
                if best_mode == "Carro":
                    road_hip = None
                    for hip in ["carro", "bus"]:
                        if hip in probs_by_mode:
                            road_hip = hip
                            break
                    if road_hip is not None:
                        road_probs = probs_by_mode[road_hip]
                        road_idx_c = idx_c_by_mode[road_hip]
                        
                        # Fracción de puntos cerca de ruta de autobús (idx_c == 1)
                        fraction_near_bus = np.mean(road_idx_c == 1)
                        
                        if road_probs[1] > road_probs[0] and fraction_near_bus >= THRESHOLD_BUS:
                            pred = "Bus"
                        else:
                            pred = "Carro"
                            
                            pass
            y_true.append(MODE_TO_IDX[label])
            y_pred.append(MODE_TO_IDX[pred])

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        total = len(y_true)
        corrects = int(np.sum(y_true == y_pred))
        bal_acc = balanced_accuracy_score(y_true, y_pred)

        # Calcular precisión por modo
        mode_accs = {}
        for m_name in MODOS:
            m_idx = MODE_TO_IDX[m_name]
            mask = (y_true == m_idx)
            if np.sum(mask) > 0:
                mode_accs[m_name] = f"{np.sum(y_pred[mask] == m_idx) / np.sum(mask):.1%}"
            else:
                mode_accs[m_name] = "N/A"

        print(f"| {deg_name:<11} | {total:<8} | {corrects:<9} | {bal_acc:.2%}            | {mode_accs['Carro']:<11} | {mode_accs['Bus']:<9} | {mode_accs['Metro']:<11} | {mode_accs['Caminar']:<13} |")
    print("=====================================================================")

if __name__ == '__main__':
    main()
