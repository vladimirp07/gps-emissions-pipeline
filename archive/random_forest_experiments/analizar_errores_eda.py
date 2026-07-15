import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

# Compatibilidad de Pickle entre diferentes versiones de NumPy (1.x vs 2.x)
try:
    import numpy._core.numeric
except ModuleNotFoundError:
    import numpy.core as numpy_core
    import numpy.core.numeric as numpy_core_numeric
    sys.modules['numpy._core'] = numpy_core
    sys.modules['numpy._core.numeric'] = numpy_core_numeric

try:
    import numpy.core.numeric
except ModuleNotFoundError:
    import numpy._core as numpy_core
    import numpy._core.numeric as numpy_core_numeric
    sys.modules['numpy.core'] = numpy_core
    sys.modules['numpy.core.numeric'] = numpy_core_numeric

# Agregar la raíz del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))
from pipeline_v3.src import config

MODOS = ["Carro", "Bus", "Metro", "Caminar"]
MODE_TO_IDX = {m: i for i, m in enumerate(MODOS)}

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
    pkl_path = config.GPS_DIR / "datos_entrenamiento_ml.pkl"
    if not pkl_path.exists():
        print(f"Error: No se encontró la caché en {pkl_path}")
        sys.exit(1)
        
    with open(pkl_path, "rb") as f:
        data_cache = pickle.load(f)
        
    trips_dict = {}
    for item in data_cache:
        caid_trip, label, deg, modo_hip = parse_trip_key(item["trip_id"])
        key = (caid_trip, label, deg)
        if key not in trips_dict:
            trips_dict[key] = {}
        trips_dict[key][modo_hip] = item

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
            row["drive_highway_motorway_frac"] = np.mean([1 if any(w in str(h) for w in ["motorway", "trunk", "primary"]) else 0 for h in drive_hyp["highway_raw"]])
            row["drive_highway_residential_frac"] = np.mean([1 if "residential" in str(h) else 0 for h in drive_hyp["highway_raw"]])
            row["drive_near_bus_frac"] = np.mean(drive_hyp["idx_c"] == 1)
            row["drive_near_metro_frac"] = np.mean(drive_hyp["idx_c"] == 0)
        else:
            row["drive_mean_speed"] = 0.0
            row["drive_max_speed"] = 0.0
            row["drive_std_speed"] = 0.0
            row["drive_stop_frac"] = 0.0
            row["drive_p25_speed"] = 0.0
            row["drive_p50_speed"] = 0.0
            row["drive_p75_speed"] = 0.0
            row["drive_max_speed_diff"] = 0.0
            row["drive_mean_speed_diff"] = 0.0
            row["drive_highway_motorway_frac"] = 0.0
            row["drive_highway_residential_frac"] = 0.0
            row["drive_near_bus_frac"] = 0.0
            row["drive_near_metro_frac"] = 0.0

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
            row["walk_highway_footway_frac"] = np.mean([1 if any(w in str(h) for w in ["footway", "pedestrian", "steps", "path", "living_street"]) else 0 for h in walk_hyp["highway_raw"]])
        else:
            row["walk_mean_speed"] = 0.0
            row["walk_max_speed"] = 0.0
            row["walk_std_speed"] = 0.0
            row["walk_p25_speed"] = 0.0
            row["walk_p50_speed"] = 0.0
            row["walk_p75_speed"] = 0.0
            row["walk_max_speed_diff"] = 0.0
            row["walk_mean_speed_diff"] = 0.0
            row["walk_highway_footway_frac"] = 0.0

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
            row["metro_near_metro_frac"] = np.mean(metro_hyp["idx_c"] == 0)
        else:
            row["metro_mean_speed"] = 0.0
            row["metro_max_speed"] = 0.0
            row["metro_p25_speed"] = 0.0
            row["metro_p50_speed"] = 0.0
            row["metro_p75_speed"] = 0.0
            row["metro_max_speed_diff"] = 0.0
            row["metro_mean_speed_diff"] = 0.0
            row["metro_near_metro_frac"] = 0.0
            
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
        # 5. GPS Drift and Bus Route interaction features
        row["drive_near_bus_drift_decay"] = row["drive_near_bus_frac"] * np.exp(-row["mean_snap_dist_drive"] / 15.0)
        row["drive_near_bus_high_drift"] = row["drive_near_bus_frac"] * (1.0 - np.exp(-row["mean_snap_dist_drive"] / 15.0))
        
        rows.append(row)
        
    df_features = pd.DataFrame(rows)
    feature_cols = [
        "drive_mean_speed", "drive_max_speed", "drive_std_speed", "drive_stop_frac",
        "drive_p25_speed", "drive_p50_speed", "drive_p75_speed",
        "drive_max_speed_diff", "drive_mean_speed_diff",
        "drive_highway_motorway_frac", "drive_highway_residential_frac", "drive_near_bus_frac", "drive_near_metro_frac",
        "drive_near_bus_drift_decay", "drive_near_bus_high_drift",
        "walk_mean_speed", "walk_max_speed", "walk_std_speed", "walk_highway_footway_frac",
        "walk_p25_speed", "walk_p50_speed", "walk_p75_speed",
        "walk_max_speed_diff", "walk_mean_speed_diff",
        "metro_mean_speed", "metro_max_speed", "metro_near_metro_frac",
        "metro_p25_speed", "metro_p50_speed", "metro_p75_speed",
        "metro_max_speed_diff", "metro_mean_speed_diff",
        "mean_snap_dist_drive", "max_snap_dist_drive", "std_snap_dist_drive",
        "mean_snap_dist_walk", "max_snap_dist_walk", "std_snap_dist_walk"
    ]
    
    X = df_features[feature_cols].fillna(0.0)
    y = df_features["label"].map(MODE_TO_IDX)
    groups = df_features["caid_trip"]
    
    gkf = GroupKFold(n_splits=5)
    df_features_cv = df_features.copy()
    df_features_cv["pred"] = ""
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_leaf=8, random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        df_features_cv.iloc[val_idx, df_features_cv.columns.get_loc("pred")] = [MODOS[pred] for pred in y_pred]
        
    df_features_cv["correct"] = df_features_cv["label"] == df_features_cv["pred"]
    
    print("=== ANÁLISIS GLOBAL DE CONFUSIONES EN CROSS-VALIDATION ===")
    confusions = df_features_cv[~df_features_cv["correct"]].groupby(["label", "pred"]).size().reset_index(name="counts")
    print(confusions.to_string(index=False))
    
    print("\n=== COMPARATIVA DE CARACTERÍSTICAS (EDA) ===")
    
    # Caso 1: Real Carro predicho como Bus vs predicho como Carro
    car_trips = df_features_cv[df_features_cv["label"] == "Carro"]
    print("\n--- CASO: Real Carro predicho como Bus vs predicho como Carro ---")
    car_correct_vs_bus_error = car_trips[car_trips["pred"].isin(["Carro", "Bus"])].groupby("pred")[[
        "drive_mean_speed", "drive_stop_frac", "drive_near_bus_frac", "drive_highway_motorway_frac", "mean_snap_dist_drive"
    ]].mean()
    print(car_correct_vs_bus_error.to_string())
    
    # Caso 2: Real Bus predicho como Carro vs predicho como Bus
    bus_trips = df_features_cv[df_features_cv["label"] == "Bus"]
    print("\n--- CASO: Real Bus predicho como Carro vs predicho como Bus ---")
    bus_correct_vs_car_error = bus_trips[bus_trips["pred"].isin(["Carro", "Bus"])].groupby("pred")[[
        "drive_mean_speed", "drive_stop_frac", "drive_near_bus_frac", "drive_highway_motorway_frac", "mean_snap_dist_drive"
    ]].mean()
    print(bus_correct_vs_car_error.to_string())
    
    # Caso 3: Real Carro predicho como Caminar (4 casos)
    print("\n--- CASO: Real Carro predicho como Caminar ---")
    car_error_caminar = car_trips[car_trips["pred"] == "Caminar"][[
        "caid_trip", "deg", "drive_mean_speed", "walk_mean_speed", "mean_snap_dist_drive", "mean_snap_dist_walk"
    ]]
    print(car_error_caminar.to_string(index=False))
    
    # Caso 4: Real Caminar predicho como Carro (2 casos)
    walk_trips = df_features_cv[df_features_cv["label"] == "Caminar"]
    print("\n--- CASO: Real Caminar predicho como Carro ---")
    walk_error_car = walk_trips[walk_trips["pred"] == "Carro"][[
        "caid_trip", "deg", "drive_mean_speed", "walk_mean_speed", "mean_snap_dist_drive", "mean_snap_dist_walk"
    ]]
    print(walk_error_car.to_string(index=False))

    df_features_cv.to_csv(config.GPS_DIR / "auditoria_errores_cv.csv", index=False)
    print(f"\nArchivo de auditoría guardado en: {config.GPS_DIR / 'auditoria_errores_cv.csv'}")

if __name__ == "__main__":
    main()
