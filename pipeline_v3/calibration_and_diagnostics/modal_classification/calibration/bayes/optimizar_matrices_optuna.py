"""
optimizar_matrices_optuna.py
============================
Fase B del sistema de calibración desacoplada de matrices bayesianas.

Este script consume el caché precomputado `datos_entrenamiento_optuna.pkl`
(generado por `generar_datos_entrenamiento.py`) y utiliza Optimización Bayesiana
(Optuna + TPE Sampler) para calibrar las 56 probabilidades que componen las
4 matrices condicionales del clasificador modal `BayesianRouteEvaluator`.
"""

import os
import sys

# Force stdout/stderr to handle encodings gracefully
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')
except AttributeError:
    pass

import json
import re
import pickle
import argparse
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ── Configuración de rutas ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(PROJECT_ROOT))

from pipeline_v3.src import config

CLASSIFIER_PATH = PROJECT_ROOT / "pipeline_v3" / "src" / "modal_classification.py"
DEFAULT_PKL_PATH = config.GPS_DIR / "datos_entrenamiento_ml.pkl"
DEFAULT_OUTPUT_JSON = Path(__file__).parent / "matrices_optimas.json"

# ── Mapeo de modos a índice de columna (fijo, debe coincidir con BayesianRouteEvaluator) ──
MODOS = ["Carro", "Bus", "Metro", "Caminar"]
MODE_TO_IDX = {m: i for i, m in enumerate(MODOS)}

# ── Matrices baseline del artículo de investigación (para comparación) ─────
BASELINE_CERCANIA = np.array([
    [0.0661, 0.0058, 0.9007, 0.0274],
    [0.0004, 0.0619, 0.0011, 0.9366],
    [0.0522, 0.0084, 0.0321, 0.9073],
])
BASELINE_VELOCIDAD = np.array([
    [0.0955, 0.0022, 0.8176, 0.0847],
    [0.6656, 0.0560, 0.0002, 0.2782],
    [0.1846, 0.0028, 0.8001, 0.0125],
    [0.8505, 0.0001, 0.1485, 0.0010],
])
BASELINE_DISTANCIA = np.array([
    [0.6026, 0.2618, 0.1159, 0.0196],
    [0.0003, 0.8000, 0.1287, 0.0711],
    [0.7892, 0.1721, 0.0100, 0.0287],
    [0.2254, 0.4666, 0.3019, 0.0061],
    [0.0087, 0.9083, 0.0001, 0.0830],
])
BASELINE_VELPROM = np.array([
    [0.3011, 0.1768, 0.0732, 0.4489],
    [0.7003, 0.2994, 0.0003, 0.0001],
])


# ═══════════════════════════════════════════════════════════════════════════
# Funciones auxiliares
# ═══════════════════════════════════════════════════════════════════════════

def softmax_rows(logits_flat: np.ndarray, n_rows: int, n_cols: int = 4) -> np.ndarray:
    mat = logits_flat.reshape(n_rows, n_cols)
    mat = mat - mat.max(axis=1, keepdims=True)
    exp_mat = np.exp(mat)
    probs = exp_mat / exp_mat.sum(axis=1, keepdims=True)
    # Suavizado de Laplace (clipeo a [0.01, 0.99] y re-normalización para evitar colapsos)
    probs = np.clip(probs, 0.01, 0.99)
    return probs / probs.sum(axis=1, keepdims=True)


def build_vectorized_cache(data_cache: list) -> list:
    # Agrupar por trip_group_id (caid_trip_degradation)
    groups = {}
    for item in data_cache:
        parts = item["trip_id"].split("_")
        group_key = f"{parts[0]}_{parts[1]}_{parts[2]}"
        if group_key not in groups:
            groups[group_key] = {
                "label": item["label"],
                "degradacion": item.get("degradacion", "Raw"),
                "hypotheses": []
            }
        
        # Aplicar filtrado inteligente en memoria
        label = item["label"]
        idx_c = item["idx_c"].astype(np.int32)
        idx_v = item["idx_v"].astype(np.int32)
        idx_d = item["idx_d_arr"].astype(np.int32)
        idx_vp = item["idx_vp_arr"].astype(np.int32)
        
        modo_hip = item["modo_hipotesis"].lower()
        
        # Filtrado de paradas y transbordos
        if label == "Metro":
            # Para Metro: solo conservar puntos cerca de la vía y en movimiento
            mask = (idx_c == 0) & (idx_v > 0)
            if mask.any():
                idx_c, idx_v, idx_d = idx_c[mask], idx_v[mask], idx_d[mask]
                idx_vp = np.ones_like(idx_c)
        elif label in ["Carro", "Bus"]:
            # Para Carro y Bus: eliminar paradas estáticas
            mask = (idx_v > 0)
            if mask.any():
                idx_c, idx_v, idx_d = idx_c[mask], idx_v[mask], idx_d[mask]
                idx_vp = np.ones_like(idx_c)
                
        if len(idx_c) > 0:
            groups[group_key]["hypotheses"].append({
                "modo_hipotesis": modo_hip,
                "idx_c": idx_c,
                "idx_v": idx_v,
                "idx_d": idx_d,
                "idx_vp": idx_vp
            })
            
    # Filtrar grupos vacíos
    return [g for g in groups.values() if len(g["hypotheses"]) > 0]


def evaluate_matrices_vectorized(
    vcache: list,
    Cercania: np.ndarray,
    Velocidad: np.ndarray,
    Distancia: np.ndarray,
    Velprom: np.ndarray,
    metric: str = "balanced_accuracy",
) -> tuple[float, float]:
    if not vcache:
        return 0.0, 0.0

    y_true = []
    y_pred = []
    
    # Usamos el umbral óptimo de cobertura de bus del 80%
    THRESHOLD_BUS = 0.80

    for trip in vcache:
        label = trip["label"]
        label_idx = MODE_TO_IDX[label]
        
        scores = {}
        probs_by_mode = {}
        idx_c_by_mode = {}
        
        for hyp in trip["hypotheses"]:
            modo_hip = hyp["modo_hipotesis"]
            idx_c = hyp["idx_c"]
            idx_v = hyp["idx_v"]
            idx_d = hyp["idx_d"]
            idx_vp = hyp["idx_vp"]
            
            # Multiplicación bayesiana punto a punto
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
            best_mode = max(scores, key=scores.get)
            pred = best_mode
            
            if best_mode == "Carro":
                road_hip = None
                for hip in ["carro", "bus"]:
                    if hip in probs_by_mode:
                        road_hip = hip
                        break
                if road_hip is not None:
                    road_probs = probs_by_mode[road_hip]
                    road_idx_c = idx_c_by_mode[road_hip]
                    
                    fraction_near_bus = np.mean(road_idx_c == 1)
                    if road_probs[1] > road_probs[0] and fraction_near_bus >= THRESHOLD_BUS:
                        pred = "Bus"
                    else:
                        pred = "Carro"
                        
        y_true.append(label_idx)
        y_pred.append(MODE_TO_IDX[pred])
        
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    bal_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)
    f1      = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    
    if metric == "f1":
        return f1, bal_acc
    return bal_acc, f1


def format_matrix(mat: np.ndarray, name: str, row_labels: list[str]) -> str:
    col_header = "         " + "  ".join(f"{m:>8}" for m in MODOS)
    lines = [f"\n  {name}:", col_header]
    for i, row in enumerate(mat):
        row_str = "  ".join(f"{v:8.4f}" for v in row)
        lines.append(f"    [{row_labels[i]:>20}]  {row_str}")
    return "\n".join(lines)


def print_matrices_report(Cercania, Velocidad, Distancia, Velprom, primary_score, secondary_score,
                          label="ÓPTIMAS", metric="balanced_accuracy"):
    primary_name   = "Balanced Accuracy" if metric == "balanced_accuracy" else "Macro F1-Score"
    secondary_name = "Macro F1-Score"    if metric == "balanced_accuracy" else "Balanced Accuracy"

    print(f"\n{'═'*70}")
    print(f"  MATRICES {label}")
    print(f"  {primary_name}:   {primary_score:.4f} ({primary_score*100:.2f}%)")
    print(f"  {secondary_name}: {secondary_score:.4f}")
    print(f"{'═'*70}")

    print(format_matrix(Cercania, "CERCANÍA (3×4)",
        ["Cerca Metro (idx=0)", "Cerca Bus (idx=1)", "Sin Infraest. (idx=2)"]))
    print(format_matrix(Velocidad, "VELOCIDAD (4×4)",
        ["≤ 6 km/h", "6–20 km/h", "20–80 km/h", "> 80 km/h"]))
    print(format_matrix(Distancia, "DISTANCIA (5×4)",
        ["≤ 1 km", "1–6 km", "6–10 km", "10–18 km", "> 18 km"]))
    print(format_matrix(Velprom, "VEL. PROMEDIO (2×4)",
        ["Vprom ≤ 6 km/h", "Vprom > 6 km/h"]))
    print(f"{'═'*70}")


class EarlyStoppingCallback:
    def __init__(self, patience: int = 300, min_delta: float = 1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self._best_value      = float("inf")
        self._trials_no_improve = 0

    def __call__(self, study, trial):
        current_value = study.best_value
        if current_value < self._best_value - self.min_delta:
            self._best_value        = current_value
            self._trials_no_improve = 0
        else:
            self._trials_no_improve += 1

        if self._trials_no_improve >= self.patience:
            print(
                f"\n  ⏹️  Early stopping activado: sin mejora en los últimos "
                f"{self.patience} trials (mejor loss={self._best_value:.6f})."
            )
            study.stop()


def build_objective(vcache: list, metric: str = "balanced_accuracy", prior_anchored: bool = False, delta_range: float = 1.5, l2_strength: float = 0.001):
    # Convert baseline probabilities to log-space priors (clipped to avoid ln(0))
    c_priors = np.log(np.clip(BASELINE_CERCANIA, 1e-5, 1.0))
    v_priors = np.log(np.clip(BASELINE_VELOCIDAD, 1e-5, 1.0))
    d_priors = np.log(np.clip(BASELINE_DISTANCIA, 1e-5, 1.0))
    vp_priors = np.log(np.clip(BASELINE_VELPROM, 1e-5, 1.0))

    def objective(trial):
        if prior_anchored:
            c_deltas = np.array([
                trial.suggest_float(f"c_d_{r}_{k}", -delta_range, delta_range)
                for r in range(3) for k in range(4)
            ]).reshape(3, 4)
            v_deltas = np.array([
                trial.suggest_float(f"v_d_{r}_{k}", -delta_range, delta_range)
                for r in range(4) for k in range(4)
            ]).reshape(4, 4)
            d_deltas = np.array([
                trial.suggest_float(f"d_d_{r}_{k}", -delta_range, delta_range)
                for r in range(5) for k in range(4)
            ]).reshape(5, 4)
            vp_deltas = np.array([
                trial.suggest_float(f"vp_d_{r}_{k}", -delta_range, delta_range)
                for r in range(2) for k in range(4)
            ]).reshape(2, 4)

            Cercania  = softmax_rows(c_priors + c_deltas,  n_rows=3)
            Velocidad = softmax_rows(v_priors + v_deltas,  n_rows=4)
            Distancia = softmax_rows(d_priors + d_deltas,  n_rows=5)
            Velprom   = softmax_rows(vp_priors + vp_deltas, n_rows=2)
        else:
            LOGIT_MIN, LOGIT_MAX = -6.0, 6.0
            c_logits = np.array([
                trial.suggest_float(f"c_r{r}_c{k}", LOGIT_MIN, LOGIT_MAX)
                for r in range(3) for k in range(4)
            ])
            v_logits = np.array([
                trial.suggest_float(f"v_r{r}_c{k}", LOGIT_MIN, LOGIT_MAX)
                for r in range(4) for k in range(4)
            ])
            d_logits = np.array([
                trial.suggest_float(f"d_r{r}_c{k}", LOGIT_MIN, LOGIT_MAX)
                for r in range(5) for k in range(4)
            ])
            vp_logits = np.array([
                trial.suggest_float(f"vp_r{r}_c{k}", LOGIT_MIN, LOGIT_MAX)
                for r in range(2) for k in range(4)
            ])

            Cercania  = softmax_rows(c_logits,  n_rows=3)
            Velocidad = softmax_rows(v_logits,  n_rows=4)
            Distancia = softmax_rows(d_logits,  n_rows=5)
            Velprom   = softmax_rows(vp_logits, n_rows=2)
            
            # Convert priors to flat array for L2 calculation
            priors_flat = np.concatenate([c_priors.flatten(), v_priors.flatten(), d_priors.flatten(), vp_priors.flatten()])
            logits_flat = np.concatenate([c_logits, v_logits, d_logits, vp_logits])
            l2_reg = l2_strength * np.sum((logits_flat - priors_flat) ** 2)

        if prior_anchored:
            l2_reg = l2_strength * (
                np.sum(c_deltas ** 2) +
                np.sum(v_deltas ** 2) +
                np.sum(d_deltas ** 2) +
                np.sum(vp_deltas ** 2)
            )

        primary_score, _ = evaluate_matrices_vectorized(
            vcache, Cercania, Velocidad, Distancia, Velprom, metric=metric
        )
        return (1.0 - primary_score) + l2_reg

    return objective


def extract_best_matrices(best_params: dict, prior_anchored: bool = False) -> tuple:
    if prior_anchored:
        c_priors = np.log(np.clip(BASELINE_CERCANIA, 1e-5, 1.0))
        v_priors = np.log(np.clip(BASELINE_VELOCIDAD, 1e-5, 1.0))
        d_priors = np.log(np.clip(BASELINE_DISTANCIA, 1e-5, 1.0))
        vp_priors = np.log(np.clip(BASELINE_VELPROM, 1e-5, 1.0))

        c_deltas = np.array([best_params[f"c_d_{r}_{k}"] for r in range(3) for k in range(4)]).reshape(3, 4)
        v_deltas = np.array([best_params[f"v_d_{r}_{k}"] for r in range(4) for k in range(4)]).reshape(4, 4)
        d_deltas = np.array([best_params[f"d_d_{r}_{k}"] for r in range(5) for k in range(4)]).reshape(5, 4)
        vp_deltas = np.array([best_params[f"vp_d_{r}_{k}"] for r in range(2) for k in range(4)]).reshape(2, 4)

        Cercania  = softmax_rows(c_priors + c_deltas,  n_rows=3)
        Velocidad = softmax_rows(v_priors + v_deltas,  n_rows=4)
        Distancia = softmax_rows(d_priors + d_deltas,  n_rows=5)
        Velprom   = softmax_rows(vp_priors + vp_deltas, n_rows=2)
    else:
        c_logits  = np.array([best_params[f"c_r{r}_c{k}"]  for r in range(3) for k in range(4)])
        v_logits  = np.array([best_params[f"v_r{r}_c{k}"]  for r in range(4) for k in range(4)])
        d_logits  = np.array([best_params[f"d_r{r}_c{k}"]  for r in range(5) for k in range(4)])
        vp_logits = np.array([best_params[f"vp_r{r}_c{k}"] for r in range(2) for k in range(4)])

        Cercania  = softmax_rows(c_logits,  n_rows=3)
        Velocidad = softmax_rows(v_logits,  n_rows=4)
        Distancia = softmax_rows(d_logits,  n_rows=5)
        Velprom   = softmax_rows(vp_logits, n_rows=2)
    return Cercania, Velocidad, Distancia, Velprom


def _matrix_to_python_literal(mat: np.ndarray, indent: int = 8) -> str:
    pad = " " * indent
    rows = []
    for row in mat:
        vals = ", ".join(f"{v:.4f}" for v in row)
        rows.append(f"{pad}[{vals}],")
    return "\n".join(rows)


def update_classifier_matrices(Cercania: np.ndarray, Velocidad: np.ndarray,
                                Distancia: np.ndarray, Velprom: np.ndarray) -> bool:
    if not CLASSIFIER_PATH.exists():
        print(f"  [ERROR] No se encontró modal_classification.py en: {CLASSIFIER_PATH}")
        return False

    source = CLASSIFIER_PATH.read_text(encoding="utf-8")

    def replace_matrix_block(source_text: str, attr_name: str, new_mat: np.ndarray) -> str:
        new_rows = _matrix_to_python_literal(new_mat, indent=12)
        pattern = (
            rf"(self\.{attr_name}\s*=\s*np\.array\(\[)"
            rf"[\s\S]*?"
            rf"(\s*\]\))"
        )
        replacement = rf"\g<1>\n{new_rows}\n        \2"
        updated, count = re.subn(pattern, replacement, source_text, count=1)
        if count == 0:
            print(f"  [WARN] No se pudo localizar self.{attr_name} en el archivo.")
        return updated

    source = replace_matrix_block(source, "Cercania",  Cercania)
    source = replace_matrix_block(source, "Velocidad", Velocidad)
    source = replace_matrix_block(source, "Distancia", Distancia)
    source = replace_matrix_block(source, "Velprom",   Velprom)

    backup_path = CLASSIFIER_PATH.with_suffix(".py.bak")
    if CLASSIFIER_PATH.exists():
        if backup_path.exists():
            backup_path.unlink()
        CLASSIFIER_PATH.rename(backup_path)
    CLASSIFIER_PATH.write_text(source, encoding="utf-8")
    print(f"  ✅ modal_classification.py actualizado correctamente.")
    print(f"  📦 Backup guardado en: {backup_path.name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Optimización Bayesiana (Optuna) de matrices del clasificador modal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=DEFAULT_PKL_PATH,
        help="Ruta al archivo .pkl precomputado.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5000,
        help="Número máximo de trials de optimización.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Tiempo máximo de optimización en segundos.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Ruta del archivo JSON de salida.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="calibracion_matrices_bayesianas",
        help="Nombre del estudio Optuna.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="URL de almacenamiento (sqlite:///optuna_study.db).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Número de workers paralelos.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria.",
    )
    parser.add_argument(
        "--update-classifier",
        action="store_true",
        default=False,
        help="Actualiza las matrices en bayes_classifier.py.",
    )
    parser.add_argument(
        "--baseline-comparison",
        action="store_true",
        default=False,
        help="Evalúa las matrices originales antes de optimizar.",
    )
    parser.add_argument(
        "--no-pruning",
        action="store_true",
        default=False,
        help="Desactiva el MedianPruner.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="balanced_accuracy",
        choices=["balanced_accuracy", "f1"],
        help="Métrica a optimizar.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Trials consecutivos sin mejora para early stopping (0=desactivado).",
    )
    parser.add_argument(
        "--prior-anchored",
        action="store_true",
        default=False,
        help="Optimiza desviaciones en log-probabilidad alrededor del prior físico del paper.",
    )
    parser.add_argument(
        "--delta-range",
        type=float,
        default=1.5,
        help="Rango máximo de desviación del logit en modo --prior-anchored.",
    )
    parser.add_argument(
        "--l2-strength",
        type=float,
        default=0.001,
        help="Coeficiente de regularización L2 (0.0 para desactivar).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción de viajes para el conjunto de prueba (0.0 a 1.0) para división por viajes.",
    )
    args = parser.parse_args()

    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner, NopPruner
    except ImportError:
        print("\n[ERROR] Optuna no está instalado.")
        print("  Instálalo con:  pip install optuna scikit-learn")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    metric_display = "Balanced Accuracy" if args.metric == "balanced_accuracy" else "Macro F1-Score"

    print("=" * 70)
    print("  OPTIMIZACIÓN BAYESIANA DE MATRICES — Fase B (Optuna TPE)")
    print("=" * 70)
    print(f"  Métrica objetivo: {metric_display}")
    print(f"  Early Stopping:   {f'activado (patience={args.patience})' if args.patience > 0 else 'desactivado'}")

    print(f"\n▶ Cargando caché precomputado: {args.pkl}")
    if not args.pkl.exists():
        print(f"\n[ERROR] No se encontró el archivo: {args.pkl}")
        sys.exit(1)

    with open(args.pkl, "rb") as f:
        data_cache_raw = pickle.load(f)

    # Filtrar con el dataset canónico
    clean_csv_path = config.GPS_DIR / "Datos de MATLAB GPS Limpios.csv"
    print(f"  [Canónico] Cargando mapeo de exclusión desde: {clean_csv_path}")
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
            
    print(f"  [Canónico] Trayectorias físicas totales: {len(canonical_modes)}")
    print(f"  [Canónico] Excluyendo {len(mixed_trips)} viajes mixtos y {len(empty_trips)} vacíos.")
    
    data_cache = []
    for item in data_cache_raw:
        parts = item["trip_id"].split("-")
        caid_trip = parts[0]
        canonical_label = canonical_modes.get(caid_trip)
        
        # Excluir si no tiene etiqueta canónica o es mixto/vacío
        if not canonical_label or canonical_label in ["mixed_label", "empty_label"]:
            continue
            
        # Reemplazar la etiqueta del item con la etiqueta canónica correcta
        item["label"] = canonical_label.capitalize()
        data_cache.append(item)

    print(f"  ✅ [Canónico] Cargadas {len(data_cache):,} muestras de entrenamiento filtradas.")

    labels = [t["label"] for t in data_cache]
    from collections import Counter
    dist = Counter(labels)
    print(f"  📊 Distribución de clases:")
    for mode in MODOS:
        print(f"     {mode:>10}: {dist.get(mode, 0):>5} muestras")

    # ── División Entrenamiento / Prueba ──────────────────────────────────────
    if args.test_size > 0:
        # Agrupar registros por viaje base para evitar fuga de datos
        trip_groups = {}
        for item in data_cache:
            base_id = item["trip_id"].split("-")[0]
            if base_id not in trip_groups:
                trip_groups[base_id] = []
            trip_groups[base_id].append(item)
            
        unique_trips = list(trip_groups.keys())
        # Ordenar para consistencia antes de barajar
        unique_trips.sort()
        np.random.seed(args.seed)
        np.random.shuffle(unique_trips)
        
        n_total_trips = len(unique_trips)
        n_test = max(1, int(n_total_trips * args.test_size))
        n_train = n_total_trips - n_test
        
        train_trips = set(unique_trips[:n_train])
        test_trips = set(unique_trips[n_train:])
        
        train_cache = []
        val_cache = []
        for base_id, items in trip_groups.items():
            if base_id in train_trips:
                train_cache.extend(items)
            else:
                val_cache.extend(items)
                
        print(f"\n▶ División Train/Test Split (para evitar fuga de datos):")
        print(f"   Total viajes: {n_total_trips}  |  Entrenamiento: {len(train_trips)} viajes  |  Validación: {len(test_trips)} viajes")
    else:
        train_cache = data_cache
        val_cache = []
        print(f"\n▶ Utilizando dataset completo para optimización (sin split).")

    print(f"\n▶ Pre-procesando caché vectorizado...")
    import time
    t_vec = time.time()
    train_vcache = build_vectorized_cache(train_cache)
    val_vcache = build_vectorized_cache(val_cache) if val_cache else None
    
    if not train_vcache:
        print("[ERROR] No se pudieron extraer viajes válidos para entrenamiento.")
        sys.exit(1)
        
    n_train_pts = sum(sum(len(h["idx_c"]) for h in t["hypotheses"]) for t in train_vcache)
    n_train_trips = len(train_vcache)
    print(f"  ✅ Cache de Entrenamiento listo: {n_train_trips:,} viajes, {n_train_pts:,} pings.")
    
    if val_vcache:
        n_val_pts = sum(sum(len(h["idx_c"]) for h in t["hypotheses"]) for t in val_vcache)
        n_val_trips = len(val_vcache)
        print(f"  ✅ Cache de Validación listo: {n_val_trips:,} viajes, {n_val_pts:,} pings.")

    if args.baseline_comparison:
        print("\n▶ Evaluando matrices baseline del paper...")
        base_train_p, base_train_s = evaluate_matrices_vectorized(
            train_vcache,
            BASELINE_CERCANIA, BASELINE_VELOCIDAD,
            BASELINE_DISTANCIA, BASELINE_VELPROM,
            metric=args.metric,
        )
        print(f"   [Baseline] Entrenamiento: {base_train_p:.4f}")
        
        if val_vcache:
            base_val_p, base_val_s = evaluate_matrices_vectorized(
                val_vcache,
                BASELINE_CERCANIA, BASELINE_VELOCIDAD,
                BASELINE_DISTANCIA, BASELINE_VELPROM,
                metric=args.metric,
            )
            print(f"   [Baseline] Validación:    {base_val_p:.4f}")

    sampler = TPESampler(seed=args.seed)
    pruner  = NopPruner() if args.no_pruning else MedianPruner(n_startup_trials=50, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=args.storage is not None,
    )

    objective_fn = build_objective(
        train_vcache, 
        metric=args.metric, 
        prior_anchored=args.prior_anchored, 
        delta_range=args.delta_range,
        l2_strength=args.l2_strength
    )

    callbacks = []
    if args.patience > 0:
        callbacks.append(EarlyStoppingCallback(patience=args.patience))

    print(f"\n▶ Iniciando optimización:")
    print(f"   Métrica:    {metric_display}")
    print(f"   Modo:       {'Prior-Anchored (delta=' + str(args.delta_range) + ')' if args.prior_anchored else 'Sin restricciones'}")
    print(f"   Trials:     {args.trials:,}")
    print(f"   Timeout:    {args.timeout or 'Sin límite'} s")
    print()

    t0 = time.time()
    study.optimize(
        objective_fn,
        n_trials=args.trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
        callbacks=callbacks if callbacks else None,
    )

    elapsed = time.time() - t0
    print(f"\n  ⏱️  Optimización completada en {elapsed:.2f} s")
    print(f"  🏆 Mejor valor (loss): {study.best_value:.6f}")

    Cercania, Velocidad, Distancia, Velprom = extract_best_matrices(
        study.best_params, 
        prior_anchored=args.prior_anchored
    )
    best_train_p, best_train_s = evaluate_matrices_vectorized(
        train_vcache, Cercania, Velocidad, Distancia, Velprom, metric=args.metric
    )
    if val_vcache:
        best_val_p, best_val_s = evaluate_matrices_vectorized(
            val_vcache, Cercania, Velocidad, Distancia, Velprom, metric=args.metric
        )
    else:
        best_val_p, best_val_s = 0.0, 0.0

    print_matrices_report(
        Cercania, Velocidad, Distancia, Velprom,
        best_train_p, best_train_s,
        label="ÓPTIMAS (Evaluadas en Entrenamiento)",
        metric=args.metric,
    )

    if val_vcache:
        primary_name = "Balanced Accuracy" if args.metric == "balanced_accuracy" else "Macro F1-Score"
        secondary_name = "Macro F1-Score" if args.metric == "balanced_accuracy" else "Balanced Accuracy"
        print(f"\n{'═'*70}")
        print(f"  RENDIMIENTO EN EL CONJUNTO DE VALIDACIÓN (NO ENTRENADO)")
        print(f"  {primary_name}:   {best_val_p:.4f} ({best_val_p*100:.2f}%)")
        print(f"  {secondary_name}: {best_val_s:.4f}")
        print(f"{'═'*70}")

    if args.baseline_comparison:
        primary_name = "Balanced Accuracy" if args.metric == "balanced_accuracy" else "Macro F1-Score"
        print(f"\n  📊 Comparación vs. Baseline del Paper (Métrica: {primary_name}):")
        print(f"     Entrenamiento: {base_train_p:.4f} → {best_train_p:.4f}  (Δ {best_train_p - base_train_p:+.4f})")
        if val_vcache:
            print(f"     Validación:    {base_val_p:.4f} → {best_val_p:.4f}  (Δ {best_val_p - base_val_p:+.4f})")

    output_data = {
        "study_name":         args.study_name,
        "timestamp":          datetime.datetime.now().isoformat(),
        "n_trials_completed": len(study.trials),
        "best_loss":          float(study.best_value),
        "metric_optimized":   args.metric,
        "balanced_accuracy":  float(best_train_p if args.metric == "balanced_accuracy" else best_train_s),
        "macro_f1":           float(best_train_p if args.metric == "f1" else best_train_s),
        "val_balanced_accuracy": float(best_val_p if args.metric == "balanced_accuracy" else best_val_s) if val_vcache else None,
        "val_macro_f1":          float(best_val_p if args.metric == "f1" else best_val_s) if val_vcache else None,
        "matrices": {
            "Cercania":  Cercania.tolist(),
            "Velocidad": Velocidad.tolist(),
            "Distancia": Distancia.tolist(),
            "Velprom":   Velprom.tolist(),
        }
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Matrices óptimas guardadas en: {args.output_json}")

    if args.update_classifier:
        print(f"\n▶ Actualizando modal_classification.py con las matrices óptimas...")
        success = update_classifier_matrices(Cercania, Velocidad, Distancia, Velprom)
        if not success:
            print("  [ERROR] No se pudo actualizar el clasificador automáticamente.")
    else:
        print(f"\n  ℹ️  Para aplicar las matrices óptimas al clasificador, ejecuta con --update-classifier")

    print(f"\n{'═'*70}")
    print("  OPTIMIZACIÓN COMPLETADA ✅")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
