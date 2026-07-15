"""
optimizar_matrices_optuna.py
============================
Fase B del sistema de calibración desacoplada de matrices bayesianas.

Este script consume el caché precomputado `datos_entrenamiento_optuna.pkl`
(generado por `generar_datos_entrenamiento.py`) y utiliza Optimización Bayesiana
(Optuna + TPE Sampler) para calibrar las 56 probabilidades que componen las
4 matrices condicionales del clasificador modal `BayesianRouteEvaluator`.

Arquitectura:
  - 56 logits libres (rango [-6, 6]) → Softmax por fila → Matrices probabilísticas
  - Evaluación vectorizada total: un solo pass NumPy sobre todos los viajes simultáneamente
  - Métrica objetivo: configurable via --metric (balanced_accuracy | f1)
  - Early stopping: para la búsqueda si no hay mejora en N trials consecutivos
  - Salida: JSON con matrices óptimas + opción de actualizar bayes_classifier.py

MEJORAS v2 (respecto a la versión anterior):
  1. [MÉTRICA] --metric permite elegir entre balanced_accuracy y f1 como objetivo.
  2. [EARLY STOPPING] EarlyStoppingCallback detiene Optuna si no hay mejora en
     --patience trials, ahorrando tiempo y cómputo.
  3. [VECTORIZACIÓN] evaluate_matrices_vectorized() elimina el bucle Python por
     viaje y ejecuta un único pass NumPy (np.add.reduceat), reduciendo el tiempo
     de evaluación de segundos a milisegundos para grandes datasets.

Uso:
  # Optimización estándar (5,000 trials, balanced accuracy)
  python pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/optimizar_matrices_optuna.py

  # Usar F1-Score macro como métrica objetivo
  python ... --metric f1

  # Early stopping si no hay mejora en 300 trials consecutivos
  python ... --patience 300

  # Con más trials y timeout de seguridad
  python ... --trials 10000 --timeout 300

  # Persistir el estudio en SQLite (permite pausar/reanudar)
  python ... --storage sqlite:///optuna_study.db --study-name calibracion_v1

  # Actualizar automáticamente bayes_classifier.py con las matrices óptimas
  python ... --update-classifier

  # Comparar matrices actuales del paper vs las optimizadas
  python ... --baseline-comparison
"""

import os
import sys
import json
import re
import pickle
import argparse
import datetime
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ── Configuración de rutas ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT))

from pipeline_v3.src import config

CLASSIFIER_PATH = PROJECT_ROOT / "pipeline_v3" / "src" / "modal_classification.py"
DEFAULT_PKL_PATH = config.GPS_DIR / "datos_entrenamiento_optuna.pkl"
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
    """
    Convierte un vector plano de logits en una matriz de probabilidades
    aplicando softmax independiente por fila (numerically stable).

    Args:
        logits_flat: Vector 1D de longitud (n_rows * n_cols).
        n_rows:      Número de filas de la matriz resultante.
        n_cols:      Número de columnas (default 4 = número de modos).

    Returns:
        np.ndarray de shape (n_rows, n_cols) con filas que suman 1.0.
    """
    mat = logits_flat.reshape(n_rows, n_cols)
    # Substracción del máximo por fila para estabilidad numérica
    mat = mat - mat.max(axis=1, keepdims=True)
    exp_mat = np.exp(mat)
    return exp_mat / exp_mat.sum(axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════
# MEJORA 3: Evaluación vectorizada (sin bucle Python por viaje)
# ═══════════════════════════════════════════════════════════════════════════

def build_vectorized_cache(data_cache: list) -> dict:
    """
    Pre-aplana todos los viajes en arrays NumPy concatenados para permitir
    evaluación matricial masiva sin bucles Python en runtime.

    Llama a esta función UNA SOLA VEZ al cargar el caché. El resultado se
    pasa al closure de la función objetivo para ser reutilizado en los 56
    parámetros de cada trial de Optuna.

    Returns:
        dict con las siguientes claves:
          - idx_c_all    : np.ndarray int32, todos los índices de cercanía concatenados.
          - idx_v_all    : np.ndarray int32, todos los índices de velocidad concatenados.
          - idx_d_all    : np.ndarray int32, todos los índices de distancia concatenados.
          - idx_vp_all   : np.ndarray int32, todos los índices de velprom concatenados.
          - trip_lengths : np.ndarray int64, longitud de cada viaje (puntos GPS).
          - reduceat_idx : np.ndarray int64, índices de inicio de cada viaje (para np.add.reduceat).
          - y_true       : np.ndarray int64, etiqueta real de cada viaje (0-3).
    """
    idx_c_list, idx_v_list, idx_d_list, idx_vp_list = [], [], [], []
    y_true_list = []
    trip_lengths = []

    for trip in data_cache:
        label_idx = MODE_TO_IDX.get(trip["label"])
        if label_idx is None:
            continue  # Modo desconocido, saltar

        n_pts = len(trip["idx_c"])
        if n_pts == 0:
            continue

        idx_c_list.append(trip["idx_c"].astype(np.int32))
        idx_v_list.append(trip["idx_v"].astype(np.int32))
        idx_d_list.append(trip["idx_d_arr"].astype(np.int32))
        idx_vp_list.append(trip["idx_vp_arr"].astype(np.int32))
        y_true_list.append(label_idx)
        trip_lengths.append(n_pts)

    # Concatenar todos los puntos de todos los viajes
    idx_c_all  = np.concatenate(idx_c_list)
    idx_v_all  = np.concatenate(idx_v_list)
    idx_d_all  = np.concatenate(idx_d_list)
    idx_vp_all = np.concatenate(idx_vp_list)

    trip_lengths = np.array(trip_lengths, dtype=np.int64)
    # Índices de inicio de cada viaje dentro del array concatenado (para reduceat)
    reduceat_idx = np.concatenate([[0], np.cumsum(trip_lengths[:-1])]).astype(np.int64)

    return {
        "idx_c_all":    idx_c_all,
        "idx_v_all":    idx_v_all,
        "idx_d_all":    idx_d_all,
        "idx_vp_all":   idx_vp_all,
        "trip_lengths": trip_lengths,
        "reduceat_idx": reduceat_idx,
        "y_true":       np.array(y_true_list, dtype=np.int64),
    }


def evaluate_matrices_vectorized(
    vcache: dict,
    Cercania: np.ndarray,
    Velocidad: np.ndarray,
    Distancia: np.ndarray,
    Velprom: np.ndarray,
    metric: str = "balanced_accuracy",
) -> tuple[float, float]:
    """
    Evalúa un conjunto de matrices bayesianas sobre el dataset cacheado usando
    un ÚNICO pass NumPy vectorizado sobre todos los puntos de todos los viajes.

    En lugar de un bucle Python (O(n_trips) llamadas), ejecuta:
      1. Cuatro lookups vectorizados en matrices (O(1) en tiempo NumPy).
      2. Un producto Hadamard de shape (total_puntos, 4).
      3. Una normalización por fila.
      4. np.add.reduceat() para sumar votos por viaje sin bucles.
      5. argmax() sobre cada viaje para obtener la predicción.

    Args:
        vcache:  Diccionario generado por build_vectorized_cache().
        metric:  'balanced_accuracy' o 'f1' (métrica primaria a retornar primero).

    Returns:
        Tuple (métrica_primaria, métrica_secundaria)
        Si metric='balanced_accuracy': (bal_acc, f1)
        Si metric='f1':                (f1, bal_acc)
    """
    idx_c   = vcache["idx_c_all"]
    idx_v   = vcache["idx_v_all"]
    idx_d   = vcache["idx_d_all"]
    idx_vp  = vcache["idx_vp_all"]
    ra_idx  = vcache["reduceat_idx"]
    y_true  = vcache["y_true"]

    if len(y_true) == 0:
        return 0.0, 0.0

    # ── Paso 1: Lookup vectorizado (shape: total_puntos × 4) ───────────────
    P_unnorm = (
        Cercania[idx_c]  *
        Velocidad[idx_v] *
        Distancia[idx_d] *
        Velprom[idx_vp]
    )

    # ── Paso 2: Normalización por punto (evitar divisón por cero) ──────────
    row_sums = P_unnorm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    P_norm = P_unnorm / row_sums

    # ── Paso 3: Sumar votos por viaje usando reduceat (sin bucle) ──────────
    # np.add.reduceat suma segmentos contiguos definidos por los índices de inicio.
    # Resultado shape: (n_trips, 4)
    trip_votes = np.add.reduceat(P_norm, ra_idx, axis=0)

    # ── Paso 4: Predicción por argmax en los votos de cada viaje ───────────
    y_pred = np.argmax(trip_votes, axis=1)

    # ── Paso 5: Métricas ───────────────────────────────────────────────────
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1      = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if metric == "f1":
        return f1, bal_acc
    return bal_acc, f1


# ── Función de evaluación con bucle (fallback / compatibilidad) ────────────

def evaluate_matrices(data_cache: list, Cercania: np.ndarray, Velocidad: np.ndarray,
                      Distancia: np.ndarray, Velprom: np.ndarray) -> tuple[float, float]:
    """
    Evaluación con bucle Python (conservada para compatibilidad y como fallback).
    Para datasets grandes, preferir evaluate_matrices_vectorized().

    Returns:
        Tuple (balanced_accuracy, macro_f1)
    """
    y_true = []
    y_pred = []

    for trip in data_cache:
        label_idx = MODE_TO_IDX.get(trip["label"])
        if label_idx is None:
            continue

        idx_c  = trip["idx_c"]
        idx_v  = trip["idx_v"]
        idx_d  = trip["idx_d_arr"]
        idx_vp = trip["idx_vp_arr"]

        P_unnorm = (Cercania[idx_c] *
                    Velocidad[idx_v] *
                    Distancia[idx_d] *
                    Velprom[idx_vp])

        row_sums = P_unnorm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        P_norm = P_unnorm / row_sums

        total_votes = P_norm.sum(axis=0)
        prediction  = int(np.argmax(total_votes))

        y_pred.append(prediction)
        y_true.append(label_idx)

    if not y_true:
        return 0.0, 0.0

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1      = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return bal_acc, f1


def format_matrix(mat: np.ndarray, name: str, row_labels: list[str]) -> str:
    """Formatea una matriz numpy para impresión legible en consola."""
    col_header = "         " + "  ".join(f"{m:>8}" for m in MODOS)
    lines = [f"\n  {name}:", col_header]
    for i, row in enumerate(mat):
        row_str = "  ".join(f"{v:8.4f}" for v in row)
        lines.append(f"    [{row_labels[i]:>20}]  {row_str}")
    return "\n".join(lines)


def print_matrices_report(Cercania, Velocidad, Distancia, Velprom, primary_score, secondary_score,
                          label="ÓPTIMAS", metric="balanced_accuracy"):
    """Imprime un reporte formateado completo de las matrices calibradas."""
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


# ═══════════════════════════════════════════════════════════════════════════
# MEJORA 2: Early Stopping Callback
# ═══════════════════════════════════════════════════════════════════════════

class EarlyStoppingCallback:
    """
    Callback de Optuna que detiene la optimización cuando no se observa
    ninguna mejora en la función objetivo durante `patience` trials consecutivos.

    Útil para evitar continuar la búsqueda una vez que TPE ha convergido,
    reduciendo el tiempo de cómputo sin sacrificar calidad.

    Uso:
        early_stop = EarlyStoppingCallback(patience=300, min_delta=1e-5)
        study.optimize(objective, callbacks=[early_stop])

    Args:
        patience:  Número máximo de trials consecutivos sin mejora antes de detener.
        min_delta: Mejora mínima requerida en la función objetivo para contar como progreso.
                   Evita que mejoras de ruido microscópico reinicien el contador.
    """

    def __init__(self, patience: int = 300, min_delta: float = 1e-5):
        self.patience   = patience
        self.min_delta  = min_delta
        self._best_value      = float("inf")
        self._trials_no_improve = 0

    def __call__(self, study, trial):
        """Invocado por Optuna al terminar cada trial."""
        current_value = study.best_value  # Mejor valor GLOBAL hasta ahora

        if current_value < self._best_value - self.min_delta:
            # Hay una mejora real: reiniciar el contador
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


# ═══════════════════════════════════════════════════════════════════════════
# MEJORA 1 + 3: Función objetivo de Optuna (usa vectorización + métrica configurable)
# ═══════════════════════════════════════════════════════════════════════════

def build_objective(vcache: dict, metric: str = "balanced_accuracy"):
    """
    Factoría que crea la función objetivo de Optuna usando el caché vectorizado
    capturado en el closure.

    MEJORAS aplicadas:
      - Vectorización: usa evaluate_matrices_vectorized() en lugar del bucle.
      - Métrica configurable: 'balanced_accuracy' o 'f1' como función de pérdida.

    Espacio de búsqueda: 56 logits en [-6.0, 6.0]
      - Cercanía:  3 filas × 4 cols = 12 logits  (parámetros c_r{i}_c{j})
      - Velocidad: 4 filas × 4 cols = 16 logits  (parámetros v_r{i}_c{j})
      - Distancia: 5 filas × 4 cols = 20 logits  (parámetros d_r{i}_c{j})
      - Velprom:   2 filas × 4 cols =  8 logits  (parámetros vp_r{i}_c{j})

    Args:
        vcache: Diccionario precomputado por build_vectorized_cache().
        metric: 'balanced_accuracy' (default) o 'f1'. Determina qué se minimiza.
    """
    LOGIT_MIN, LOGIT_MAX = -6.0, 6.0

    def objective(trial):
        # --- Cercanía (3×4 = 12 logits) ---
        c_logits = np.array([
            trial.suggest_float(f"c_r{r}_c{k}", LOGIT_MIN, LOGIT_MAX)
            for r in range(3) for k in range(4)
        ])
        # --- Velocidad (4×4 = 16 logits) ---
        v_logits = np.array([
            trial.suggest_float(f"v_r{r}_c{k}", LOGIT_MIN, LOGIT_MAX)
            for r in range(4) for k in range(4)
        ])
        # --- Distancia (5×4 = 20 logits) ---
        d_logits = np.array([
            trial.suggest_float(f"d_r{r}_c{k}", LOGIT_MIN, LOGIT_MAX)
            for r in range(5) for k in range(4)
        ])
        # --- Velprom (2×4 = 8 logits) ---
        vp_logits = np.array([
            trial.suggest_float(f"vp_r{r}_c{k}", LOGIT_MIN, LOGIT_MAX)
            for r in range(2) for k in range(4)
        ])

        Cercania  = softmax_rows(c_logits,  n_rows=3)
        Velocidad = softmax_rows(v_logits,  n_rows=4)
        Distancia = softmax_rows(d_logits,  n_rows=5)
        Velprom   = softmax_rows(vp_logits, n_rows=2)

        # Evaluación vectorizada: retorna (métrica_primaria, métrica_secundaria)
        primary_score, _ = evaluate_matrices_vectorized(
            vcache, Cercania, Velocidad, Distancia, Velprom, metric=metric
        )
        return 1.0 - primary_score  # Optuna minimiza → convertir a pérdida

    return objective


def extract_best_matrices(best_params: dict) -> tuple:
    """Reconstruye las 4 matrices óptimas a partir del diccionario de parámetros del mejor trial."""
    c_logits  = np.array([best_params[f"c_r{r}_c{k}"]  for r in range(3) for k in range(4)])
    v_logits  = np.array([best_params[f"v_r{r}_c{k}"]  for r in range(4) for k in range(4)])
    d_logits  = np.array([best_params[f"d_r{r}_c{k}"]  for r in range(5) for k in range(4)])
    vp_logits = np.array([best_params[f"vp_r{r}_c{k}"] for r in range(2) for k in range(4)])

    Cercania  = softmax_rows(c_logits,  n_rows=3)
    Velocidad = softmax_rows(v_logits,  n_rows=4)
    Distancia = softmax_rows(d_logits,  n_rows=5)
    Velprom   = softmax_rows(vp_logits, n_rows=2)

    return Cercania, Velocidad, Distancia, Velprom


# ═══════════════════════════════════════════════════════════════════════════
# Actualización automática de bayes_classifier.py
# ═══════════════════════════════════════════════════════════════════════════

def _matrix_to_python_literal(mat: np.ndarray, indent: int = 8) -> str:
    """Convierte una matriz numpy a su representación como literal Python."""
    pad = " " * indent
    rows = []
    for row in mat:
        vals = ", ".join(f"{v:.4f}" for v in row)
        rows.append(f"{pad}[{vals}],")
    return "\n".join(rows)


def update_classifier_matrices(Cercania: np.ndarray, Velocidad: np.ndarray,
                                Distancia: np.ndarray, Velprom: np.ndarray) -> bool:
    """
    Reescribe las matrices de probabilidad en bayes_classifier.py usando
    expresiones regulares para localizar cada bloque de asignación.

    Returns:
        True si la actualización fue exitosa, False en caso de error.
    """
    if not CLASSIFIER_PATH.exists():
        print(f"  [ERROR] No se encontró modal_classification.py en: {CLASSIFIER_PATH}")
        return False

    source = CLASSIFIER_PATH.read_text(encoding="utf-8")

    def replace_matrix_block(source_text: str, attr_name: str, new_mat: np.ndarray) -> str:
        """Reemplaza el bloque np.array([...]) de un atributo self.<attr_name>."""
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

    # Escribir de vuelta con un backup previo
    backup_path = CLASSIFIER_PATH.with_suffix(".py.bak")
    CLASSIFIER_PATH.rename(backup_path)
    CLASSIFIER_PATH.write_text(source, encoding="utf-8")
    print(f"  ✅ modal_classification.py actualizado correctamente.")
    print(f"  📦 Backup guardado en: {backup_path.name}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Punto de entrada principal
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Optimización Bayesiana (Optuna) de matrices del clasificador modal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pkl",
        type=Path,
        default=DEFAULT_PKL_PATH,
        help="Ruta al archivo .pkl precomputado (datos_entrenamiento_optuna.pkl).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5000,
        help="Número máximo de trials de optimización a ejecutar.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Tiempo máximo de optimización en segundos (None = sin límite).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="Ruta del archivo JSON de salida con las matrices óptimas.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="calibracion_matrices_bayesianas",
        help="Nombre del estudio Optuna (útil para SQLite).",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="URL de almacenamiento Optuna (ej. 'sqlite:///optuna_study.db'). "
             "None = en memoria (no persistente).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Número de workers paralelos para Optuna (-1 = todos los cores).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad del TPE Sampler.",
    )
    parser.add_argument(
        "--update-classifier",
        action="store_true",
        default=False,
        help="Si se activa, reescribe las matrices en bayes_classifier.py con los valores óptimos.",
    )
    parser.add_argument(
        "--baseline-comparison",
        action="store_true",
        default=False,
        help="Evalúa las matrices originales del paper como punto de referencia antes de optimizar.",
    )
    parser.add_argument(
        "--no-pruning",
        action="store_true",
        default=False,
        help="Desactiva el MedianPruner de Optuna (más exploración, más lento).",
    )
    # ── MEJORA 1: Selección de métrica ─────────────────────────────────────
    parser.add_argument(
        "--metric",
        type=str,
        default="balanced_accuracy",
        choices=["balanced_accuracy", "f1"],
        help=(
            "Métrica a optimizar como función objetivo de pérdida.\n"
            "  balanced_accuracy: Igual peso a cada clase (recomendado para datasets desbalanceados).\n"
            "  f1: F1-Score macro, mayor penalización en clases con bajo recall (Metro, Caminar)."
        ),
    )
    # ── MEJORA 2: Early Stopping ────────────────────────────────────────────
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help=(
            "Trials consecutivos sin mejora antes de detener la optimización (early stopping). "
            "0 = desactivado (comportamiento original). Recomendado: 300-500."
        ),
    )
    args = parser.parse_args()

    # ── Importar Optuna (aquí para dar un error claro si no está instalado) ──
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
    if args.patience > 0:
        print(f"  Early Stopping:   activado (patience={args.patience} trials)")
    else:
        print(f"  Early Stopping:   desactivado")
    print(f"  Evaluación:       vectorizada (np.add.reduceat)")

    # ── 1. Cargar caché precomputado ──────────────────────────────────────
    print(f"\n▶ Cargando caché precomputado: {args.pkl}")
    if not args.pkl.exists():
        print(f"\n[ERROR] No se encontró el archivo: {args.pkl}")
        print("  Asegúrate de haber ejecutado primero:")
        print("    python pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py")
        sys.exit(1)

    with open(args.pkl, "rb") as f:
        data_cache = pickle.load(f)

    print(f"  ✅ Cargadas {len(data_cache):,} muestras de entrenamiento.")

    # Resumen del dataset cargado
    labels = [t["label"] for t in data_cache]
    from collections import Counter
    dist = Counter(labels)
    print(f"  📊 Distribución de clases:")
    for mode in MODOS:
        print(f"     {mode:>10}: {dist.get(mode, 0):>5} muestras")

    # ── MEJORA 3: Pre-aplanar caché para evaluación vectorizada ──────────
    print(f"\n▶ Pre-procesando caché vectorizado...")
    import time
    t_vec = time.time()
    vcache = build_vectorized_cache(data_cache)
    n_total_pts = len(vcache["idx_c_all"])
    n_trips     = len(vcache["y_true"])
    print(f"  ✅ Cache vectorizado listo: {n_trips:,} viajes válidos, "
          f"{n_total_pts:,} puntos GPS totales ({time.time()-t_vec:.3f}s)")

    # ── 2. Evaluación baseline (opcional) ────────────────────────────────
    if args.baseline_comparison:
        print("\n▶ Evaluando matrices baseline del paper...")
        base_primary, base_secondary = evaluate_matrices_vectorized(
            vcache,
            BASELINE_CERCANIA, BASELINE_VELOCIDAD,
            BASELINE_DISTANCIA, BASELINE_VELPROM,
            metric=args.metric,
        )
        print_matrices_report(
            BASELINE_CERCANIA, BASELINE_VELOCIDAD,
            BASELINE_DISTANCIA, BASELINE_VELPROM,
            base_primary, base_secondary,
            label="BASELINE (Paper Original)",
            metric=args.metric,
        )

    # ── 3. Configurar y ejecutar Optuna ──────────────────────────────────
    sampler = TPESampler(seed=args.seed, multivariate=True, group=True)
    pruner  = NopPruner() if args.no_pruning else MedianPruner(n_startup_trials=50, n_warmup_steps=0)

    load_if_exists = args.storage is not None
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=load_if_exists,
    )

    # ── MEJORA 1+3: Función objetivo con métrica configurable + vectorización
    objective_fn = build_objective(vcache, metric=args.metric)

    # ── MEJORA 2: Configurar callbacks (early stopping opcional) ──────────
    callbacks = []
    if args.patience > 0:
        callbacks.append(EarlyStoppingCallback(patience=args.patience))

    print(f"\n▶ Iniciando optimización:")
    print(f"   Métrica:    {metric_display}")
    print(f"   Trials:     {args.trials:,}")
    print(f"   Timeout:    {args.timeout or 'Sin límite'} s")
    print(f"   Workers:    {args.n_jobs}")
    print(f"   Storage:    {args.storage or 'En memoria (no persistente)'}")
    print(f"   Pruning:    {'Desactivado (NopPruner)' if args.no_pruning else 'MedianPruner'}")
    print(f"   Semilla:    {args.seed}")
    print(f"   Patience:   {args.patience if args.patience > 0 else 'Sin early stopping'}")
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
    print(f"  📈 Trials completados: {len(study.trials):,}")
    print(f"  🏆 Mejor valor (loss): {study.best_value:.6f}")

    # ── 4. Reconstruir matrices óptimas ──────────────────────────────────
    Cercania, Velocidad, Distancia, Velprom = extract_best_matrices(study.best_params)
    best_primary, best_secondary = evaluate_matrices_vectorized(
        vcache, Cercania, Velocidad, Distancia, Velprom, metric=args.metric
    )

    print_matrices_report(
        Cercania, Velocidad, Distancia, Velprom,
        best_primary, best_secondary,
        label="ÓPTIMAS",
        metric=args.metric,
    )

    # Comparación numérica si se calculó el baseline
    if args.baseline_comparison:
        delta_primary   = best_primary   - base_primary
        delta_secondary = best_secondary - base_secondary
        primary_name    = "Balanced Accuracy" if args.metric == "balanced_accuracy" else "Macro F1-Score"
        secondary_name  = "Macro F1-Score"    if args.metric == "balanced_accuracy" else "Balanced Accuracy"
        print(f"\n  📊 Mejora vs. Baseline del Paper:")
        print(f"     {primary_name}:   {base_primary:.4f} → {best_primary:.4f}  (Δ {delta_primary:+.4f})")
        print(f"     {secondary_name}: {base_secondary:.4f} → {best_secondary:.4f}  (Δ {delta_secondary:+.4f})")

    # ── 5. Guardar resultados en JSON ─────────────────────────────────────
    output_data = {
        "study_name":         args.study_name,
        "timestamp":          datetime.datetime.now().isoformat(),
        "n_trials_completed": len(study.trials),
        "best_loss":          float(study.best_value),
        "metric_optimized":   args.metric,
        "balanced_accuracy":  float(best_primary if args.metric == "balanced_accuracy" else best_secondary),
        "macro_f1":           float(best_primary if args.metric == "f1" else best_secondary),
        "patience_used":      args.patience,
        "sampler":            "TPESampler (multivariate=True)",
        "pruner":             "NopPruner" if args.no_pruning else "MedianPruner",
        "logit_range":        [-6.0, 6.0],
        "matrices": {
            "Cercania":  Cercania.tolist(),
            "Velocidad": Velocidad.tolist(),
            "Distancia": Distancia.tolist(),
            "Velprom":   Velprom.tolist(),
        },
        "bins": {
            "Cercania_rows":  ["Cerca Metro (dist<50m)", "Cerca Bus (dist<50m)", "Sin infraestructura"],
            "Velocidad_rows": ["<=6 km/h", "6-20 km/h", "20-80 km/h", ">80 km/h"],
            "Distancia_rows": ["<=1 km", "1-6 km", "6-10 km", "10-18 km", ">18 km"],
            "Velprom_rows":   ["Vprom<=6 km/h", "Vprom>6 km/h"],
            "columns":        MODOS,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n  💾 Matrices óptimas guardadas en: {args.output_json}")

    # ── 6. Actualizar bayes_classifier.py (opcional) ──────────────────────
    if args.update_classifier:
        print(f"\n▶ Actualizando modal_classification.py con las matrices óptimas...")
        success = update_classifier_matrices(Cercania, Velocidad, Distancia, Velprom)
        if not success:
            print("  [ERROR] No se pudo actualizar el clasificador automáticamente.")
            print("  Copia manualmente las matrices desde el JSON de salida.")
    else:
        print(
            f"\n  ℹ️  Para aplicar las matrices óptimas al clasificador, ejecuta con --update-classifier"
            f"\n  o copia los valores desde: {args.output_json}"
        )

    print(f"\n{'═'*70}")
    print("  OPTIMIZACIÓN COMPLETADA ✅")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
