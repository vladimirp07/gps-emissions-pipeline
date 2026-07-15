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

## 📁 2. Clasificador modal jerárquico híbrido (`random_forest_calibration/`)
Contiene la versión oficial: N1 Gradient Boosting (16 variables), N2 Random Forest (52) y N3 Extra Trees (25, umbral Bus 0.50).

*   **Dataset oficial**: 114 viajes físicos de etiqueta única y 445 escenarios; los viajes mixtos se excluyen y las degradaciones permanecen agrupadas.
*   **`generar_datos_entrenamiento_ml.py`**: Extrae las características cinemáticas, espaciales y multiescala temporales de los tramos ruteados para armar la caché de entrenamiento.
*   **`notebooks/playground_modal_classifier.ipynb`**: Reproduce la cascada híbrida, validación agrupada, métricas por degradación, inferencia, guardrail y selección de backend.
*   **Contrato compartido**: `random_forest_contract.py` fija `N1_FEATURES`, `N2_FEATURES`, `N3_FEATURES`, hiperparámetros y guardrail.
*   **Rollback**: `random_forest_modal.pkl` conserva ML v4 de tres Random Forest. Bayes continúa disponible. `MODAL_CLASSIFIER=hybrid|random_forest|bayes` controla la fábrica.
*   **Compromiso medido**: en el entorno de producción, BA 88.86%, recall Bus 81.25%, Caminar 80.56%, Carro 93.62% y Metro 100%; mejora Bus a cambio de menor recall Caminar y más Carro→Bus frente al rollback.
