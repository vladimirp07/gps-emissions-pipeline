# Archivo de Experimentos de Random Forest (Histórico)

Este directorio contiene informes, análisis cinemáticos, gráficos de trayectoria y tablas comparativas de desempeño generados durante las iteraciones de calibración del clasificador Random Forest (versiones experimentales de 55 variables con adición de paradas de autobús urbano y análisis de la frontera Carro-Bus).

### Contenido:
- `analisis_bus_features.md`: Análisis físico de las variables de autobús descartadas.
- `auditoria_viajes_bus_persistentes.md`: Reporte forense individual de errores persistentes bajo el dataset canónico.
- `comparacion_ml_v4_bus_features.csv`: Comparación de métricas globales del experimento de variables de bus.
- `comparacion_viajes_bus_persistentes.csv`: Auditoría cinemática de los cuatro viajes objetivo.
- `errores_carro_bus_ml_v4.csv`: Predicciones out-of-fold del baseline de Nivel 3.
- `*_analysis.png`: Series de velocidad y trayectorias espaciales graficadas para el diagnóstico.
- `analizar_errores_eda.py` y `evaluar_degradacion.py`: scripts diagnósticos anteriores al contrato único de producción.
- `matriz_confusion_random_forest.png`: resultado previo conservado únicamente como evidencia histórica.
- `random_forest_reconciliation.md`: auditoría que fundamentó la consolidación de 52 variables.

> [!WARNING]
> **NO UTILIZAR ESTOS ARCHIVOS EN PRODUCCIÓN.** El clasificador oficial es **ML v4 de 52 variables**, entrenado con el caché reproducible disponible de 66 viajes físicos de etiqueta única y 260 escenarios. Las seis variables experimentales de Bus están descartadas. Los 124 viajes canónicos completos quedan como regeneración futura.
