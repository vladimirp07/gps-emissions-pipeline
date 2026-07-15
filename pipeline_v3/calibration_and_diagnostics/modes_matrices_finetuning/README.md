# Calibración y Sintonización de Modelos de Clasificación Modal

Este directorio contiene las herramientas y scripts diseñados para optimizar y calibrar los motores de predicción del modo de transporte. Con el fin de evitar confusión, el módulo está dividido en dos metodologías independientes:

---

## 📁 1. Calibración Bayesiana (`bayesian_calibration/`)
Contiene los desarrollos dedicados a optimizar las matrices de probabilidad condicional del artículo de investigación para el clasificador a posteriori `BayesianRouteEvaluator`.

*   **`generar_datos_entrenamiento.py`**: Rutea y discretiza el dataset GPS en índices para evaluación rápida en memoria.
*   **`optimizar_matrices_optuna.py`**: Corre búsquedas hiperparamétricas con Optuna sobre el caché discretizado para maximizar el Balanced Accuracy probabilístico.
*   **`playground_calibracion.ipynb`**: Notebook de experimentación e inferencia interactiva bayesiana.
*   **`matrices_optimas.json`**: Estructura resultante conteniendo las matrices numéricas calibradas listas para producción.

---

## 📁 2. Entrenamiento de Random Forest (`random_forest_calibration/`)
Contiene la implementación oficial de producción **ML v4** basada en ensambles de árboles de decisión en cascada jerárquica.

*   **`entrenar_random_forest.py`**: Script de entrenamiento final y serialización de la cascada (`clf_n1`, `clf_n2`, `clf_n3`) usando el contrato único de 52 variables. El artefacto actual corresponde a 66 viajes físicos de etiqueta única y 260 escenarios; los viajes mixtos se excluyen.
*   **`generar_datos_entrenamiento_ml.py`**: Extrae las características cinemáticas, espaciales y multiescala temporales de los tramos ruteados para armar la caché de entrenamiento.
*   **`playground_random_forest.ipynb`**: Notebook oficial que reconstruye los 66 viajes/260 escenarios, calcula conservación real, ejecuta GroupKFold por viaje y prueba inferencia y guardrail.
*   **Contrato compartido**: `pipeline_v3/src/random_forest_contract.py` fija orden, hiperparámetros y guardrails oficiales.
*   **Trabajo futuro**: regenerar el caché para los 124 viajes de etiqueta única y comparar bajo el mismo protocolo. Los análisis EDA/degradación anteriores se conservan únicamente en `archive/random_forest_experiments/`.
