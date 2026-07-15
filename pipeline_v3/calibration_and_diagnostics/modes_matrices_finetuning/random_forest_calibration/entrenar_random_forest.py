import os
import sys
import pickle
import argparse
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

# Workaround de compatibilidad de Pickle
try:
    import numpy._core.numeric
except ModuleNotFoundError:
    import numpy.core as numpy_core
    import numpy.core.numeric as numpy_core_numeric
    sys.modules['numpy._core'] = numpy_core
    sys.modules['numpy._core.numeric'] = numpy_core_numeric

# Configurar rutas del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))
from pipeline_v3.src import config
from pipeline_v3.src.random_forest_contract import (
    RF_FEATURES, RF_HYPERPARAMETERS, TRAINING_SCENARIOS, TRAINING_TRIPS,
)

def parse_trip_key(trip_id_str):
    parts = trip_id_str.split("-")
    caid_trip = parts[0]
    rest = parts[1]
    subparts = rest.split("_")
    label = subparts[0]
    deg = subparts[1]
    modo_hip = subparts[2]
    return caid_trip, label, deg, modo_hip

def main():
    parser = argparse.ArgumentParser(description="Entrena ML v4 sin sobrescribir artefactos salvo la salida explícita.")
    parser.add_argument("--input-cache", default="datos_entrenamiento_ml.pkl")
    parser.add_argument("--output-model", default="random_forest_modal.pkl")
    parser.add_argument("--metrics-json")
    parser.add_argument("--confusion-plot")
    parser.add_argument("--expected-trips", type=int)
    parser.add_argument("--expected-scenarios", type=int)
    args = parser.parse_args()
    started_at = time.perf_counter()

    pkl_path = config.GPS_DIR / Path(args.input_cache).name
    clean_csv_path = config.GPS_DIR / "Datos de MATLAB GPS Limpios.csv"
    output_pkl = config.GPS_DIR / Path(args.output_model).name
    
    print(f"Cargando caché de entrenamiento: {pkl_path}")
    if not pkl_path.exists():
        print(f"Error: No existe el archivo {pkl_path}. Ejecuta generar_datos_entrenamiento_ml.py primero.")
        sys.exit(1)
        
    with open(pkl_path, "rb") as f:
        data_cache = pickle.load(f)
        
    print("Estableciendo mapeo de viajes canónicos...")
    df_clean = pd.read_csv(clean_csv_path)
    canonical_modes = {}
    mixed_trips = []
    empty_trips = []
    
    for (caid, num_trip), sub in df_clean.groupby(['caid', 'num_trip']):
        try:
            trip_num_str = str(int(float(num_trip)))
        except:
            trip_num_str = str(num_trip).strip()
        key = f"{str(caid).strip()}_{trip_num_str}"
        modes = sub['mode_of_transport'].dropna().unique()
        modes = [m.strip().lower() for m in modes if str(m).strip()]
        
        if len(modes) == 0:
            canonical_modes[key] = "empty_label"
            empty_trips.append(key)
        elif len(modes) > 1:
            canonical_modes[key] = "mixed_label"
            mixed_trips.append(key)
        else:
            canonical_modes[key] = modes[0]
            
    print(f"Total trayectorias físicas únicas en MATLAB: {len(canonical_modes)}")
    print(f"Viajes excluidos por etiquetas mixtas ({len(mixed_trips)})")
    
    MODOS = ["Carro", "Bus", "Metro", "Caminar"]
    MODE_TO_IDX = {m.lower(): i for i, m in enumerate(MODOS)}
    
    trips_dict = {}
    for item in data_cache:
        caid_trip, label, deg, modo_hip = parse_trip_key(item["trip_id"])
        canonical_label = canonical_modes.get(caid_trip)
        
        # Excluir viajes mixtos y vacíos
        if not canonical_label or canonical_label in ["mixed_label", "empty_label"]:
            continue
            
        key = (caid_trip, canonical_label, deg)
        if key not in trips_dict:
            trips_dict[key] = {}
        trips_dict[key][modo_hip] = item
        
    print(f"Agrupados en {len(trips_dict)} instancias de viaje canónicas únicas en caché.")
    
    rows = []
    for (caid_trip, label, deg), hyps in trips_dict.items():
        row = {
            "caid_trip": caid_trip,
            "label": label,
            "deg": deg,
        }
        
        # 1. Drive
        drive_hyp = hyps.get("carro") or hyps.get("bus")
        if drive_hyp:
            speeds = drive_hyp["speed_raw"]
            row["drive_mean_speed"] = np.mean(speeds)
            row["drive_max_speed"] = np.max(speeds)
            row["drive_std_speed"] = np.std(speeds)
            row["drive_stop_frac"] = np.mean(speeds < 2.0)
            row["drive_p25_speed"] = np.percentile(speeds, 25) if len(speeds) > 0 else 0.0
            row["drive_p50_speed"] = np.percentile(speeds, 50) if len(speeds) > 0 else 0.0
            row["drive_p75_speed"] = np.percentile(speeds, 75) if len(speeds) > 0 else 0.0
            row["drive_max_speed_diff"] = np.max(np.abs(np.diff(speeds))) if len(speeds) > 1 else 0.0
            row["drive_mean_speed_diff"] = np.mean(np.abs(np.diff(speeds))) if len(speeds) > 1 else 0.0
            row["drive_highway_motorway_frac"] = np.mean([1 if any(w in str(h) for w in ["motorway", "trunk", "primary"]) else 0 for h in drive_hyp["highway_raw"]]) if len(drive_hyp["highway_raw"]) > 0 else 0.0
            row["drive_highway_residential_frac"] = np.mean([1 if "residential" in str(h) else 0 for h in drive_hyp["highway_raw"]]) if len(drive_hyp["highway_raw"]) > 0 else 0.0
            row["drive_near_bus_frac"] = np.mean(drive_hyp["idx_c"] == 1) if len(drive_hyp["idx_c"]) > 0 else 0.0
            row["drive_near_metro_frac"] = np.mean(drive_hyp["idx_c"] == 0) if len(drive_hyp["idx_c"]) > 0 else 0.0
            
            row["drive_num_stops"] = float(drive_hyp.get("num_stops", 0.0))
            row["drive_mean_stop_duration"] = float(drive_hyp.get("mean_stop_duration", 0.0))
            row["drive_mean_stop_interval"] = float(drive_hyp.get("mean_stop_interval", 0.0))
            row["drive_std_stop_interval"] = float(drive_hyp.get("std_stop_interval", 0.0))
            
            lf = drive_hyp.get("local_features", {})
            row["drive_win_near_bus_max"] = lf.get("drive_win_near_bus_max", 0.0)
            row["drive_win_near_bus_p90"] = lf.get("drive_win_near_bus_p90", 0.0)
            row["drive_win_near_bus_consec_run"] = lf.get("drive_win_near_bus_consec_run", 0.0)
            row["drive_win_stops_max"] = lf.get("drive_win_stops_max", 0.0)
            row["drive_win_stops_consec_run"] = lf.get("drive_win_stops_consec_run", 0.0)
        else:
            for c in ["drive_mean_speed", "drive_max_speed", "drive_std_speed", "drive_stop_frac", "drive_p25_speed", "drive_p50_speed", "drive_p75_speed", "drive_max_speed_diff", "drive_mean_speed_diff", "drive_highway_motorway_frac", "drive_highway_residential_frac", "drive_near_bus_frac", "drive_near_metro_frac", "drive_num_stops", "drive_mean_stop_duration", "drive_mean_stop_interval", "drive_std_stop_interval", "drive_win_near_bus_max", "drive_win_near_bus_p90", "drive_win_near_bus_consec_run", "drive_win_stops_max", "drive_win_stops_consec_run"]:
                row[c] = 0.0
                
        # 2. Walk
        walk_hyp = hyps.get("caminar")
        if walk_hyp:
            speeds = walk_hyp["speed_raw"]
            row["walk_mean_speed"] = np.mean(speeds)
            row["walk_max_speed"] = np.max(speeds)
            row["walk_std_speed"] = np.std(speeds)
            row["walk_p25_speed"] = np.percentile(speeds, 25) if len(speeds) > 0 else 0.0
            row["walk_p50_speed"] = np.percentile(speeds, 50) if len(speeds) > 0 else 0.0
            row["walk_p75_speed"] = np.percentile(speeds, 75) if len(speeds) > 0 else 0.0
            row["walk_max_speed_diff"] = np.max(np.abs(np.diff(speeds))) if len(speeds) > 1 else 0.0
            row["walk_mean_speed_diff"] = np.mean(np.abs(np.diff(speeds))) if len(speeds) > 1 else 0.0
            row["walk_highway_footway_frac"] = np.mean([1 if any(w in str(h) for w in ["footway", "pedestrian", "steps", "path", "living_street"]) else 0 for h in walk_hyp["highway_raw"]]) if len(walk_hyp["highway_raw"]) > 0 else 0.0
            
            lf = walk_hyp.get("local_features", {})
            row["walk_win_walk_regime_max"] = lf.get("walk_win_walk_regime_max", 0.0)
            row["walk_win_walk_regime_consec_run"] = lf.get("walk_win_walk_regime_consec_run", 0.0)
        else:
            for c in ["walk_mean_speed", "walk_max_speed", "walk_std_speed", "walk_p25_speed", "walk_p50_speed", "walk_p75_speed", "walk_max_speed_diff", "walk_mean_speed_diff", "walk_highway_footway_frac", "walk_win_walk_regime_max", "walk_win_walk_regime_consec_run"]:
                row[c] = 0.0
                
        # 3. Metro
        metro_hyp = hyps.get("metro")
        if metro_hyp:
            speeds = metro_hyp["speed_raw"]
            row["metro_mean_speed"] = np.mean(speeds)
            row["metro_max_speed"] = np.max(speeds)
            row["metro_p25_speed"] = np.percentile(speeds, 25) if len(speeds) > 0 else 0.0
            row["metro_p50_speed"] = np.percentile(speeds, 50) if len(speeds) > 0 else 0.0
            row["metro_p75_speed"] = np.percentile(speeds, 75) if len(speeds) > 0 else 0.0
            row["metro_max_speed_diff"] = np.max(np.abs(np.diff(speeds))) if len(speeds) > 1 else 0.0
            row["metro_mean_speed_diff"] = np.mean(np.abs(np.diff(speeds))) if len(speeds) > 1 else 0.0
            row["metro_near_metro_frac"] = np.mean(metro_hyp["idx_c"] == 0) if len(metro_hyp["idx_c"]) > 0 else 0.0
            
            lf = metro_hyp.get("local_features", {})
            row["metro_win_near_metro_max"] = lf.get("metro_win_near_metro_max", 0.0)
            row["metro_win_near_metro_p90"] = lf.get("metro_win_near_metro_p90", 0.0)
            row["metro_win_near_metro_consec_run"] = lf.get("metro_win_near_metro_consec_run", 0.0)
        else:
            for c in ["metro_mean_speed", "metro_max_speed", "metro_p25_speed", "metro_p50_speed", "metro_p75_speed", "metro_max_speed_diff", "metro_mean_speed_diff", "metro_near_metro_frac", "metro_win_near_metro_max", "metro_win_near_metro_p90", "metro_win_near_metro_consec_run"]:
                row[c] = 0.0
                
        any_hyp = hyps.get("carro") or hyps.get("bus") or hyps.get("caminar") or hyps.get("metro")
        if any_hyp and "snap_dist_drive" in any_hyp:
            row["mean_snap_dist_drive"] = np.mean(any_hyp["snap_dist_drive"])
            row["max_snap_dist_drive"] = np.max(any_hyp["snap_dist_drive"])
            row["std_snap_dist_drive"] = np.std(any_hyp["snap_dist_drive"])
            row["mean_snap_dist_walk"] = np.mean(any_hyp["snap_dist_walk"])
            row["max_snap_dist_walk"] = np.max(any_hyp["snap_dist_walk"])
            row["std_snap_dist_walk"] = np.std(any_hyp["snap_dist_walk"])
        else:
            row["mean_snap_dist_drive"] = 150.0
            row["max_snap_dist_drive"] = 150.0
            row["std_snap_dist_drive"] = 0.0
            row["mean_snap_dist_walk"] = 50.0
            row["max_snap_dist_walk"] = 50.0
            row["std_snap_dist_walk"] = 0.0
            
        row["drive_near_bus_drift_decay"] = row["drive_near_bus_frac"] * np.exp(-row["mean_snap_dist_drive"] / 15.0)
        row["drive_near_bus_high_drift"] = row["drive_near_bus_frac"] * (1.0 - np.exp(-row["mean_snap_dist_drive"] / 15.0))
        
        rows.append(row)
        
    df_features = pd.DataFrame(rows)

    physical_trip_count = df_features["caid_trip"].nunique()
    expected_trips = args.expected_trips
    expected_scenarios = args.expected_scenarios
    if pkl_path.name == "datos_entrenamiento_ml.pkl":
        expected_trips = TRAINING_TRIPS if expected_trips is None else expected_trips
        expected_scenarios = TRAINING_SCENARIOS if expected_scenarios is None else expected_scenarios
    if expected_trips is not None and physical_trip_count != expected_trips:
        raise RuntimeError(
            f"Se esperaban {expected_trips} viajes y se obtuvieron {physical_trip_count}."
        )
    if expected_scenarios is not None and len(df_features) != expected_scenarios:
        raise RuntimeError(f"Se esperaban {expected_scenarios} escenarios y se obtuvieron {len(df_features)}.")
    
    # Contrato único de 52 variables de producción.
    feature_cols_v4 = list(RF_FEATURES)
    
    X_v4 = df_features[feature_cols_v4].fillna(0.0)
    y = df_features["label"].map(MODE_TO_IDX)
    groups = df_features["caid_trip"]
    
    # Evaluar por 5-fold CV canónico
    gkf = GroupKFold(n_splits=5)
    cv_scores = []
    y_true_all = []
    y_pred_all = []
    
    print("\n" + "="*70)
    print("  EVALUACIÓN CASCADA JERÁRQUICA CANÓNICA OFICIAL (52 VARIABLES)")
    print("="*70)
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df_features, y, groups)):
        df_train = df_features.iloc[train_idx].fillna(0.0)
        df_val = df_features.iloc[val_idx].fillna(0.0)
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        
        # N1 (Caminar vs Motorizado)
        clf_n1 = RandomForestClassifier(**RF_HYPERPARAMETERS["n1"])
        clf_n1.fit(df_train[feature_cols_v4], (y_train != MODE_TO_IDX["caminar"]).astype(int))
        
        # N2 (Metro vs Road)
        motor_mask = (y_train != MODE_TO_IDX["caminar"])
        clf_n2 = RandomForestClassifier(**RF_HYPERPARAMETERS["n2"])
        clf_n2.fit(df_train[motor_mask][feature_cols_v4], (y_train[motor_mask] == MODE_TO_IDX["metro"]).astype(int))
        
        # N3 (Carro vs Bus)
        road_mask = (y_train == MODE_TO_IDX["carro"]) | (y_train == MODE_TO_IDX["bus"])
        clf_n3 = RandomForestClassifier(**RF_HYPERPARAMETERS["n3"])
        clf_n3.fit(df_train[road_mask][feature_cols_v4], (y_train[road_mask] == MODE_TO_IDX["bus"]).astype(int))
        
        # Inferencia
        pred_n1 = clf_n1.predict(df_val[feature_cols_v4])
        pred_n2 = clf_n2.predict(df_val[feature_cols_v4])
        pred_n3 = clf_n3.predict(df_val[feature_cols_v4])
        
        pred_fold = []
        for idx in range(len(df_val)):
            if pred_n1[idx] == 0:
                pred_fold.append(MODE_TO_IDX["caminar"])
            else:
                if pred_n2[idx] == 1:
                    pred_fold.append(MODE_TO_IDX["metro"])
                else:
                    if pred_n3[idx] == 1:
                        pred_fold.append(MODE_TO_IDX["bus"])
                    else:
                        pred_fold.append(MODE_TO_IDX["carro"])
                        
        fold_acc = balanced_accuracy_score(y_val, pred_fold)
        cv_scores.append(fold_acc)
        y_true_all.extend(y_val)
        y_pred_all.extend(pred_fold)
        
    print("-"*70)
    print(f"  Promedio CV Balanced Accuracy : {np.mean(cv_scores)*100:.2f}% ± {np.std(cv_scores)*100:.2f}%")
    print(f"  Macro F1-Score                : {f1_score(y_true_all, y_pred_all, average='macro')*100:.2f}%")
    print("="*70)

    cm = confusion_matrix(y_true_all, y_pred_all, labels=list(range(len(MODOS))))
    recalls = recall_score(y_true_all, y_pred_all, labels=list(range(len(MODOS))), average=None, zero_division=0)
    metrics = {
        "input_cache": pkl_path.name,
        "physical_trips": int(physical_trip_count),
        "scenarios": int(len(df_features)),
        "feature_count": len(feature_cols_v4),
        "fold_balanced_accuracy": [float(score) for score in cv_scores],
        "balanced_accuracy_mean": float(np.mean(cv_scores)),
        "balanced_accuracy_std": float(np.std(cv_scores)),
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro")),
        "recall_by_class": {mode: float(value) for mode, value in zip(MODOS, recalls)},
        "confusion_matrix": cm.tolist(),
    }
    
    # Generar y guardar gráfico de la matriz de confusión
    try:
        import matplotlib.pyplot as plt
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Crear figura con 2 subplots verticales
        fig, axes = plt.subplots(2, 1, figsize=(8, 12))
        
        try:
            import seaborn as sns
            sns_available = True
        except ImportError:
            sns_available = False
            
        # Plot 1: Absoluta
        if sns_available:
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=[m.capitalize() for m in MODOS], 
                        yticklabels=[m.capitalize() for m in MODOS],
                        cbar=True, annot_kws={"size": 12}, ax=axes[0])
        else:
            im0 = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            fig.colorbar(im0, ax=axes[0])
            tick_marks = np.arange(len(MODOS))
            axes[0].set_xticks(tick_marks)
            axes[0].set_xticklabels([m.capitalize() for m in MODOS], rotation=45)
            axes[0].set_yticks(tick_marks)
            axes[0].set_yticklabels([m.capitalize() for m in MODOS])
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    axes[0].text(j, i, format(cm[i, j], 'd'),
                                 horizontalalignment="center",
                                 color="white" if cm[i, j] > thresh else "black")
        axes[0].set_ylabel('Clase Real')
        axes[0].set_xlabel('Predicción del Modelo')
        axes[0].set_title('Matriz de Confusión Absoluta')
        
        # Plot 2: Normalizada
        if sns_available:
            sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Oranges", vmin=0.0, vmax=1.0,
                        xticklabels=[m.capitalize() for m in MODOS], 
                        yticklabels=[m.capitalize() for m in MODOS],
                        cbar=True, annot_kws={"size": 12}, ax=axes[1])
        else:
            im1 = axes[1].imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Oranges, vmin=0.0, vmax=1.0)
            fig.colorbar(im1, ax=axes[1])
            tick_marks = np.arange(len(MODOS))
            axes[1].set_xticks(tick_marks)
            axes[1].set_xticklabels([m.capitalize() for m in MODOS], rotation=45)
            axes[1].set_yticks(tick_marks)
            axes[1].set_yticklabels([m.capitalize() for m in MODOS])
            thresh = 0.5
            for i in range(cm_norm.shape[0]):
                for j in range(cm_norm.shape[1]):
                    axes[1].text(j, i, format(cm_norm[i, j], '.2f'),
                                 horizontalalignment="center",
                                 color="white" if cm_norm[i, j] > thresh else "black")
        axes[1].set_ylabel('Clase Real')
        axes[1].set_xlabel('Predicción del Modelo')
        axes[1].set_title('Matriz de Confusión Normalizada (0 a 1)')
        
        plt.suptitle('Matriz de Confusión - Random Forest ML v4 Oficial (Dataset Canónico)', fontsize=13, y=0.99)
        plt.tight_layout()
        
        plot_path = PROJECT_ROOT / args.confusion_plot if args.confusion_plot else PROJECT_ROOT / "matriz_confusion_random_forest.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Gráfico de matriz de confusión guardado exitosamente en: {plot_path}")
    except Exception as e:
        print(f"Advertencia al generar gráfico de matriz de confusión: {e}")
        
    # Entrenamiento de los modelos finales para producción
    print("\nEntrenando modelos finales oficiales en todos los datos canónicos...")
    
    clf_n1_final = RandomForestClassifier(**RF_HYPERPARAMETERS["n1"])
    clf_n1_final.fit(X_v4, (y != MODE_TO_IDX["caminar"]).astype(int))
    
    motor_mask_final = (y != MODE_TO_IDX["caminar"])
    clf_n2_final = RandomForestClassifier(**RF_HYPERPARAMETERS["n2"])
    clf_n2_final.fit(X_v4[motor_mask_final], (y[motor_mask_final] == MODE_TO_IDX["metro"]).astype(int))
    
    road_mask_final = (y == MODE_TO_IDX["carro"]) | (y == MODE_TO_IDX["bus"])
    clf_n3_final = RandomForestClassifier(**RF_HYPERPARAMETERS["n3"])
    clf_n3_final.fit(X_v4[road_mask_final], (y[road_mask_final] == MODE_TO_IDX["bus"]).astype(int))
    
    print(f"\nGuardando modelos cascada unificados en: {output_pkl}")
    with open(output_pkl, "wb") as f:
        pickle.dump({
            "clf_n1": clf_n1_final,
            "clf_n2": clf_n2_final,
            "clf_n3": clf_n3_final,
            "feature_cols_v4": feature_cols_v4,
            "feature_cols_new": feature_cols_v4
        }, f)

    metrics["elapsed_seconds"] = float(time.perf_counter() - started_at)
    metrics["output_model"] = output_pkl.name
    if args.metrics_json:
        metrics_path = config.GPS_DIR / Path(args.metrics_json).name
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Métricas guardadas en: {metrics_path}")
        
    print("Entrenamiento y serialización finalizados exitosamente!")

if __name__ == "__main__":
    main()
