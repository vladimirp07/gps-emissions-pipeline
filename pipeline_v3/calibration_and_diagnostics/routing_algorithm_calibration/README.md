# Calibracion del Algoritmo de Ruteo (routing_algorithm_calibration)

Esta carpeta contiene los scripts, notebooks, reportes y recursos visuales dedicados a la calibracion sistematica y analisis de sensibilidad del algoritmo de ruteo y map matching (Dijkstra optimizado) de la tercera version del pipeline.

El objetivo de esta calibracion es encontrar los parametros optimos (filtros espaciales de jitter, radios de snapping y tolerancia fisica) que maximicen la tasa de exito del algoritmo y minimicen la desviacion geometrica del trazado de rutas bajo diferentes niveles de calidad de senal GPS.

---

## Estructura de la Carpeta

* [ejecutar_experimentos.ipynb](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/routing_algorithm_calibration/ejecutar_experimentos.ipynb): Notebook orquestador interactivo que importa y ejecuta la bateria completa de experimentos de ruteo modularizada en `scratch/run_routing_experiments.py`.
* [comparacion_resolucion.ipynb](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/routing_algorithm_calibration/comparacion_resolucion.ipynb): Notebook de diagnostico enfocado en medir el intervalo temporal ($\Delta t$) de muestreo entre los datos locales de encuesta (MATLAB) y los datos globales agregados (Veraset).
* [hallazgos_intervenciones_ruteo.md](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/routing_algorithm_calibration/hallazgos_intervenciones_ruteo.md): Reporte tecnico formal que detalla el rendimiento, exito de map matching y errores de distancia geodesica para cada uno de los 8 escenarios de intervencion fisica y espacial evaluados.
* `scen_1_base/` a `scen_8_combined_optimal/`: Directorios locales creados para almacenar de forma estructurada los graficos y collages de degradacion de rutas (Raw, L1, L2, L3) generados por los experimentos. *(Nota: Estas carpetas y sus archivos PNG estan excluidos de Git en el .gitignore para evitar la saturacion del repositorio).*
* `metricas_experimentos_intervencion.csv`: Tabla de datos consolidada con los tiempos de ejecucion, distancias y precision de todos los experimentos corridos. *(Ignorado en Git).*

---

## Parametros Calibrados (Escenario 8 - Combined Optimal)

Tras completar las pruebas de sensibilidad sobre 8 viajes representativos (de Carro, Caminar, Bus y Metro), se determino la siguiente politica de parametros como la mas robusta para produccion, logrando una tasa de exito global del **100%**:

1. **Filtro Espacial Dinamico (15 metros):** Remocion de pings GPS sucesivos en el preprocesamiento cuya distancia sea menor a 15 metros, lo cual filtra el jitter estatico y optimiza el tiempo de ejecucion en un ~20%.
2. **Buffers de Snapping Diferenciados:**
   * **Caminar:** Buffer maximo de snapping restringido a **50m** sobre `edges_walk` para evitar que los peatonales salten a autopistas.
   * **Vehicular (Carro / Bus):** Buffer maximo de snapping de **150m** sobre `edges_drive`.
3. **Multiplicador de Limites Fisicos (2.0x):** Penalizacion relajada de velocidad maxima en Dijkstra para permitir que el vehiculo rebase temporalmente el limite vial y evitar abortos de ruteo ante gaps de senal largos que exigen desvios por calles de un solo sentido.

---

## Instrucciones para Ejecutar la Bateria de Experimentos

Para volver a generar los collages visuales y reconstruir el archivo de metricas consolidadas, puedes abrir el notebook orquestador [ejecutar_experimentos.ipynb](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/routing_algorithm_calibration/ejecutar_experimentos.ipynb) o ejecutar el script de python en terminal directamente desde la raiz del proyecto:

```bash
python scratch/run_routing_experiments.py
```
