# pipeline_v4_production

**pipeline_v4_production — READY FOR CONTROLLED PRODUCTION**

El código de producción reside en `pipeline_v4/` y la versión lógica se fija en `config.PIPELINE_RELEASE`.

## Flujo

`GPS → segmentación → map-matching multi-hipótesis → clasificador modal jerárquico híbrido → lookup MOVES → emisiones por subsegmento`.

## Contratos

- Ruteo entrega `physical_trip_id`, timestamps, hipótesis de red, nodos/aristas, `osmid`, metros, km/h, duración y estados explícitos de ruteo/snapping.
- Clasificación recibe hipótesis ruteadas y entrega clase, probabilidades, backend, versión, calidad y motivo de rechazo mediante `evaluate_with_contract`.
- Emisiones recibe modo, `osmid`, metros, km/h, vía y hora. Entrega tasas `g/km`, distancia calculada en km, totales en g y estado del lookup.

Los validadores centrales están en `pipeline_v4/src/pipeline_contracts.py`.

## Clasificador modal

- Predeterminado: `hybrid` — N1 Gradient Boosting (16), N2 Random Forest (52), N3 Extra Trees (25), umbral Bus 0.50.
- Rollback: `random_forest`.
- Alternativa: `bayes`.

Cambiar en `config.py` o con `MODAL_CLASSIFIER=hybrid|random_forest|bayes`.

## Guardrail

Requiere al menos 15 pings efectivos y 30% conservado. En otro caso devuelve `Calidad insuficiente` con causa `quality_guardrail`.

## Ejecución

El runtime no depende del playground. El orquestador usa únicamente `create_modal_evaluator(config.MODAL_CLASSIFIER)`. Para validación rápida:

```powershell
py -3.12 -m pytest tests -q
jupyter nbconvert --to notebook --execute pipeline_v4/calibration_and_diagnostics/modal_classification/notebooks/playground_modal_classifier.ipynb
```

## Limitaciones

- WARNING: se asume provisionalmente que la lookup MOVES está en `g/km`; falta confirmación contra la exportación original.
- El router legacy no conserva distancia exacta de snapping por subsegmento y lo marca `not_recorded_by_legacy_router`.
- Los PKL deben cargarse con scikit-learn 1.5.2.

## Estado del módulo

- Estado: READY FOR CONTROLLED PRODUCTION
- Bloquea producción: No
- Acción recomendada: operar con monitoreo de lookup y confirmar unidades MOVES.

