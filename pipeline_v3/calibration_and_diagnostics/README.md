# Calibracion y Diagnosticos del Pipeline (calibration_and_diagnostics)

Esta carpeta unifica todos los modulos, scripts, notebooks y reportes tecnicos dedicados a la calibracion, diagnostico de calidad y sintonizacion hiperparametrica de los algoritmos de map matching, ruteo y clasificacion modal de la tercera version del pipeline.

El objetivo de este espacio es aislar las tareas de calibracion del codigo de produccion principal (`src/` y `orchestrator.ipynb`), permitiendo realizar auditorias de datos y pruebas de sensibilidad sin alterar la estructura operativa del pipeline.

---

## Modulos y Subdirectorios

La carpeta esta organizada en tres subdirectorios de trabajo especializados:

### 1. [routing_algorithm_calibration](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/routing_algorithm_calibration)
Modulo encargado de realizar pruebas de sensibilidad sobre el ruteador frente a baches de senal GPS y ruido espacial (jitter).
* **Entregables Clave:** Reporte comparativo de intervenciones de ruteo ([hallazgos_intervenciones_ruteo.md](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/routing_algorithm_calibration/hallazgos_intervenciones_ruteo.md)), collages visuales de trayectorias (`scen_*`) y notebooks orquestadores.
* **Calibracion de Produccion:** Define la parametrizacion optima (filtro de 15m, snapping buffer peatonal de 50m y factor de fisica de 2.0x) utilizada en el pipeline principal.

### 2. [gps_survey_data_cleaning](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/gps_survey_data_cleaning)
Modulo dedicado al diagnostico de calidad y depuracion automatica de los datos declarados manualmente en las encuestas de MATLAB.
* **Entregables Clave:** Propuesta de depuracion ([diagnostic_report.md](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/gps_survey_data_cleaning/diagnostic_report.md)) y script de limpieza ([depurar_datos_matlab.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/gps_survey_data_cleaning/depurar_datos_matlab.py)).
* **Impacto en los Datos:** Genera el archivo limpio y compatible `Datos de MATLAB GPS Limpios.csv` removiendo glitches GPS e incongruencias de velocidad.

### 3. [modes_matrices_finetuning](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning)
Modulo diseñado para el ajuste fino de las matrices de probabilidad condicional del evaluador de Bayes (`BayesianRouteEvaluator`).
* **Entregables Clave:** Script de generacion del dataset de entrenamiento ruteado ([generar_datos_entrenamiento.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py)) e instrucciones de integracion con el optimizador Optuna.
* **Impacto en el Proceso:** Desacopla la busqueda en grafos de la evaluacion estadistica, permitiendo optimizaciones rapidas sin coste de Dijkstra en runtime.

---

## Politica de Control de Archivos en Git

* **Archivos Trackeados:** Unicamente se mantendran en control de versiones los scripts de ejecucion (`*.py`), los notebooks orquestadores (`*.ipynb`) y la documentacion tecnica de soporte (`*.md`).
* **Archivos Excluidos (.gitignore):** Todos los datasets binarios generados (`*.pkl`), las tablas de metricas CSV resultantes (`*.csv`) y los graficos de diagnostico PNG (`*.png` o carpetas `scen_*/`) se encuentran explicitamente excluidos de Git para evitar sobrecargar el repositorio del proyecto.
