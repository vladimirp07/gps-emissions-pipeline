# GPS Trajectory Processing and Emissions Estimation Pipeline

`pipeline_v4_production` procesa trayectorias GPS, reconstruye rutas sobre redes urbanas, clasifica el modo de transporte y calcula emisiones por subsegmento. La versión actual está preparada para producción controlada y no depende de notebooks para la inferencia.

## Flujo

```text
GPS crudo
  → limpieza y segmentación
  → map-matching y reconstrucción de ruta
  → clasificación modal jerárquica
  → cálculo de emisiones
  → outputs trazables por viaje y subsegmento
```

Los contratos entre módulos conservan el ID físico del viaje y el orden temporal. Las distancias del ruteo alimentan directamente emisiones, cuya convención operativa provisional es:

```text
tasa [g/km] × distancia [km] = emisión [g]
```

La unidad de las tablas MOVES sigue registrada como `WARNING` pendiente de confirmar contra la fuente original.

## Clasificador modal de producción

El backend predeterminado es el **clasificador modal jerárquico híbrido**:

1. N1 — Gradient Boosting: Caminar vs. Motorizado, 16 variables.
2. N2 — Random Forest: Metro vs. Superficie, 52 variables.
3. N3 — Extra Trees: Carro vs. Bus, 25 variables y umbral Bus `0.50`.

Fue entrenado con 114 viajes físicos de etiqueta única y 445 escenarios Raw/L1/L2/L3, manteniendo todas las degradaciones del mismo viaje en el mismo fold. Los viajes mixtos están excluidos.

El guardrail exige al menos 15 pings efectivos y 30% conservado. Si no se cumple, la salida es `Calidad insuficiente`.

El clasificador se selecciona sin modificar el orquestador:

```powershell
$env:MODAL_CLASSIFIER = "hybrid"        # predeterminado
$env:MODAL_CLASSIFIER = "random_forest" # rollback
$env:MODAL_CLASSIFIER = "bayes"         # alternativa histórica
```

También puede configurarse mediante `MODAL_CLASSIFIER` en `pipeline_v4/src/config.py`. La inferencia carga artefactos existentes; no entrena ni sobrescribe modelos.

## Estructura del repositorio

```text
pipeline_v4/
├── orchestrator.ipynb
├── src/
│   ├── config.py
│   ├── segmentation.py
│   ├── routing.py
│   ├── modal_classification.py
│   ├── random_forest_contract.py
│   ├── pipeline_contracts.py
│   └── emissions.py
└── calibration_and_diagnostics/
    ├── routing_algorithm_calibration/
    ├── gps_survey_data_cleaning/
    └── modal_classification/
        ├── notebooks/playground_modal_classifier.ipynb
        ├── calibration/random_forest/
        ├── calibration/hybrid/
        ├── calibration/bayes/
        ├── tests_local/       # ignorado por Git
        ├── reports_local/     # ignorado por Git
        ├── figures_local/     # ignorado por Git
        ├── artifacts_local/   # ignorado por Git
        └── archive/

tests/
├── routing/
├── modal_classification/
├── emissions/
└── integration/
```

Los datos, modelos y tablas se mantienen en `Inputs/`. Los resultados finales se escriben en `Outputs/`; smoke tests, reportes locales y figuras generadas no se versionan.

## Ejecución

Requisitos de referencia: Python 3.12 y scikit-learn 1.5.2. Las dependencias geoespaciales principales son pandas, NumPy, GeoPandas, Shapely, NetworkX, igraph y PyArrow.

1. Prepare los datos y cachés requeridos bajo `Inputs/`.
2. Revise rutas y backend en `pipeline_v4/src/config.py`.
3. Ejecute el orquestador:

```powershell
jupyter nbconvert --to notebook --execute pipeline_v4/orchestrator.ipynb
```

Para reproducir la validación modal y sus matrices OOF:

```powershell
jupyter nbconvert --to notebook --execute pipeline_v4/calibration_and_diagnostics/modal_classification/notebooks/playground_modal_classifier.ipynb
```

Las matrices absoluta y normalizada se guardan localmente en `figures_local/`.

## Pruebas

```powershell
py -3.12 -m pytest tests -q
```

La suite cubre contratos de ruteo, selección de los tres clasificadores, guardrail, carga de modelos, emisiones en gramos e integración end-to-end.

## Documentación y reproducibilidad

- `PIPELINE_V4_PRODUCTION.md`: guía operativa y contratos.
- `production_v4_manifest.json`: versiones, hashes, configuración y estado de pruebas.
- `Inputs/GPS User Data/modal_classifier_hybrid_v1.manifest.json`: contrato y metadata del modelo híbrido.
- `archive/`: experimentos y resultados históricos que no forman parte del flujo activo.

No se deben versionar reportes de auditoría, figuras generadas, smoke outputs ni artefactos locales.
