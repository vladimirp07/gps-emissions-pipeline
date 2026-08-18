# pipeline_v4_production

**pipeline_v4_production — READY FOR CONTROLLED PRODUCTION**

El código de producción reside en `pipeline_v4/` y la versión lógica se fija en `config.PIPELINE_RELEASE`.

La baseline oficial de ruteo es V2 optimizado. Conserva exactamente las
decisiones y salidas de V2 previamente congeladas, con `n_jobs=2`, lookahead
acotado a 10 pings, separación por componentes, preservación de endpoints V2
y rollback V1.

## Flujo

`GPS → segmentación → map-matching multi-hipótesis → clasificador modal jerárquico híbrido → lookup MOVES → emisiones por subsegmento`.

## Contratos

- Ruteo entrega `physical_trip_id`, timestamps, hipótesis de red, nodos/aristas, `osmid`, metros, km/h, duración y estados explícitos de ruteo/snapping.
- Clasificación recibe hipótesis ruteadas y entrega clase, probabilidades, backend, versión, calidad y motivo de rechazo mediante `evaluate_with_contract`.
- Emisiones recibe modo, `osmid`, metros, km/h, vía y hora. Entrega tasas `g/km`, distancia calculada en km, totales en g y estado del lookup.

Los validadores centrales están en `pipeline_v4/src/pipeline_contracts.py`.

## Baseline de ruteo V2

La implementación productiva elimina trabajo redundante mediante un índice
comprimido de aristas incidentes, snapping conjunto de endpoints, caché
determinista y acotada de shortest paths, ensamblaje vectorizado de candidatos,
caché de proximidad a infraestructura por ejecución y caché lazy/thread-safe de
atributos de aristas. La comparación congelada del 18 de agosto de 2026 confirmó
igualdad exacta de candidatos, rutas, WKT, distancias, componentes, estados,
modos, features, emisiones, ledger, summary y detailed.

El generador productivo usa `sjoin_nearest`: devuelve la arista más cercana y
sus empates exactos dentro del radio configurado, limitada por `max_cands`; no
representa una consulta K-nearest. La evaluación diagnóstica de K=3, selección
adaptativa, horizontes cortos, beam search y secuencias globales no encontró una
alternativa con mejora suficiente y sin regresiones. Esas implementaciones se
conservan exclusivamente en `calibration_and_diagnostics`; producción mantiene
nearest-only/ties.

## Clasificador modal

- Predeterminado: `hybrid` — N1 Gradient Boosting (16), N2 Random Forest (52), N3 Extra Trees (25), umbral Bus 0.50.
- Rollback: `random_forest`.
- Alternativa: `bayes`.

Cambiar en `config.py` o con `MODAL_CLASSIFIER=hybrid|random_forest|bayes`.

## Guardrail

Requiere al menos 15 pings efectivos y 30% conservado. En otro caso devuelve `Calidad insuficiente` con causa `quality_guardrail`.

## Ejecución

El runtime no depende del playground. El orquestador usa únicamente `create_modal_evaluator(config.MODAL_CLASSIFIER)`. Para validación rápida:

El entorno productivo requiere Python 3.12 y scikit-learn 1.5.2. Las versiones
se fijan en `requirements-production.txt` y el workflow las valida en runtime
antes de cargar el artifact. Los timestamps de entrada deben representar UTC;
los timestamps naive se interpretan explícitamente como UTC. La cobertura de
adquisición se declara como intervalo semiabierto para eliminar sólo días
locales de borde realmente truncados.

```powershell
py -3.12 -m pytest tests -q
jupyter nbconvert --to notebook --execute pipeline_v4/calibration_and_diagnostics/modal_classification/notebooks/playground_modal_classifier.ipynb
```

## Limitaciones

- El router legacy no conserva distancia exacta de snapping por subsegmento y lo marca `not_recorded_by_legacy_router`.
- `max_cands` limita resultados nearest/ties; no garantiza K aristas cercanas.
- Candidatos adicionales pueden mejorar rutas aisladas, pero la selección local
  puede deteriorar la consistencia posterior. Cualquier evaluación futura de
  secuencias deberá conservar exactamente el estado productivo de rollback y
  nodos penalizados.
- Los PKL deben cargarse con scikit-learn 1.5.2.

## Estado del módulo

- Estado: READY FOR CONTROLLED PRODUCTION
- Bloquea producción: No
- Acción recomendada: mantener V2 optimizado con nearest-only/ties y monitoreo
  normal de calidad, clasificación y emisiones.

