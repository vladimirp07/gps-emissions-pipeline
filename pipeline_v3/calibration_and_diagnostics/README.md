# Módulos de Calibración, Diagnóstico y Sintonización (calibration_and_diagnostics)

Este directorio agrupa los módulos, scripts y especificaciones técnicas dedicados a la calibración de parámetros espaciales, validación de la calidad de los datos de entrada y optimización probabilística del pipeline v3. Su diseño permite aislar estas actividades experimentales del flujo operativo de producción (`src/` y `orchestrator.ipynb`), facilitando auditorías de datos y análisis de sensibilidad sin alterar la integridad del sistema principal.

---
modes_matrices_finetuning
## 1. Arquitectura y Módulos Componentes

El directorio se divide en tres subsistemas funcionales especializados:

### 1.1. [routing_algorithm_calibration](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/routing_algorithm_calibration)
* **Función:** Evaluación de la sensibilidad del algoritmo de map matching ante degradación y ruido (jitter) en la señal GPS.
* **Componentes Clave:**
  * Reporte de intervenciones de ruteo: [hallazgos_intervenciones_ruteo.md](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/routing_algorithm_calibration/hallazgos_intervenciones_ruteo.md)
  * Visualización de escenarios geométricos (`scen_*`) y cuadernos de control.
* **Resultado de Calibración:** Validación de la configuración operativa óptima (Escenario 8: filtro espacial de 15m, snapping peatonal de 50m, snapping vehicular de 150m y factor de física de 2.0x).

### 1.2. [gps_survey_data_cleaning](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/gps_survey_data_cleaning)
* **Función:** Análisis cualitativo y depuración algorítmica de los registros manuales provenientes de encuestas de MATLAB.
* **Componentes Clave:**
  * Diagnóstico de calidad de datos: [diagnostic_report.md](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/gps_survey_data_cleaning/diagnostic_report.md)
  * Script de automatización de limpieza: [depurar_datos_matlab.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/gps_survey_data_cleaning/depurar_datos_matlab.py)
* **Flujo de Salida:** Generación del dataset saneado `Datos de MATLAB GPS Limpios.csv` libre de inconsistencias cronológicas y picos de velocidad físicamente imposibles.

### 1.3. [modes_matrices_finetuning](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning)
* **Función:** Calibración del clasificador modal jerárquico híbrido oficial y de las alternativas Random Forest y Bayes.
* **Componentes Clave:**
  * Generador de datos de entrenamiento: [generar_datos_entrenamiento.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py)
  * Guía técnica para sintonización con Optuna: [README.md](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/README.md)
* **Flujo de Salida:** El artefacto oficial `modal_classifier_hybrid_v1.pkl` usa 114 viajes/445 escenarios. El Random Forest anterior se conserva como rollback y Bayes como alternativa; la selección ocurre con `MODAL_CLASSIFIER`.

---

## 2. Política de Gestión de Archivos en el Repositorio (Git Policy)

Con el objetivo de mantener la eficiencia del repositorio y evitar el almacenamiento de archivos binarios pesados o resultados de simulación redundantes, se establece el siguiente control de versiones:
* **Archivos Controlados (Trackeados):** Únicamente se integran scripts (`*.py`), cuadernos de Jupyter (`*.ipynb`) y especificaciones técnicas o reportes en formato Markdown (`*.md`).
* **Archivos Excluidos (.gitignore):** Se omiten de forma estricta los archivos de datos serializados (`*.pkl`), conjuntos de datos intermedios o finales en formato plano (`*.csv`) y representaciones visuales generadas (`*.png`, y carpetas tipo `scen_*/`). Estos elementos deben permanecer alojados localmente o en sistemas de almacenamiento en la nube designados.
