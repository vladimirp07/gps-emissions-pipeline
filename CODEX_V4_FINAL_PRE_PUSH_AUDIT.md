# CODEX V4 Final Pre-Push Audit

Fecha: 2026-08-24  
Branch: `main` (`HEAD 4c87eea`, sin cambios staged)  
Alcance: estado real del working tree, no conclusiones previas.

## 1. Repository State

No hay cambios staged. No existe `.gemini/` dentro del repositorio, no hay archivos `.gemini` tracked y tampoco se encontró `.gemini` en el directorio padre inmediato.

### INTENTIONAL PRODUCTION CHANGE

- `.gitignore`
- `notebooks/GPS_preprocessing_and_pipeline_v4.ipynb`
- `pipeline_v4/diagnostics/quality_report.py`
- `pipeline_v4/preprocessing/workflow.py`
- `pipeline_v4/src/config.py`
- `pipeline_v4/src/modal_classification.py`
- `pipeline_v4/src/output_schema.py`
- `pipeline_v4/src/production_workflow.py`
- `pipeline_v4/src/random_forest_contract.py`
- `pipeline_v4/src/run_workflow.py`
- `diagnostics/performance_parallelism/run_lab_benchmark.py` (untracked)

### INTENTIONAL TEST CHANGE

- `tests/integration/test_production_environment.py`
- `tests/integration/test_production_output_schemas.py`
- `tests/integration/test_run_scoped_outputs.py`
- `tests/modal_classification/test_serving_contract.py`
- `tests/integration/test_execution_profiles_and_equivalence.py` (untracked)
- `tests/integration/test_restartability_and_checkpoints.py` (untracked)
- `tests/modal_classification/test_candidate_gating_comprehensive.py` (untracked)

### DOCUMENTATION

Todos estos Markdown raíz están untracked:

`BLOCK1B_MODAL_ROBUSTNESS_AUDIT.md`, `BLOCK1C_GROUND_TRUTH_AND_MODEL_PROVENANCE.md`, `BLOCK1D_FINAL_OOF_MODAL_VALIDATION.md`, `BLOCK1_FINAL_SCIENTIFIC_VALIDATION.md`, `BLOCK1_SCIENTIFIC_CLOSEOUT_REPORT.md`, `BLOCK2B_FINAL_MEMORY_RECOVERY_VALIDATION.md`, `BLOCK2_MEMORY_SAFE_RECOVERABLE_PRODUCTION_REPORT.md`, `BLOCK3_CRASH_RECOVERY_NOTE.md`, `BLOCK3_LAB_BENCHMARK_GUIDE.md`, `BLOCK3_PARALLELISM_PERFORMANCE_ARCHITECTURE.md`, `BLOCK4_OBSERVABILITY_COMPATIBILITY_FINAL_FREEZE.md`, `CODEX_PRODUCTION_V4_FINAL_RELEASE_AUDIT.md`, `CODEX_PRODUCTION_V4_PERFORMANCE_INTEGRITY_AUDIT.md`, `CODEX_SPARSE_GPS_FINAL_APPROVAL.md`, `CODEX_SPARSE_GPS_FINAL_CANDIDATE_REVIEW.md`, `CODEX_SPARSE_GPS_FINAL_PRODUCTION_APPROVAL.md`, `FINAL_OUTPUT_AUDIT.md`, `FINAL_PRODUCTION_PUBLIC_RELEASE_REVIEW.md`, `GUIA_CONFIGURACION_ENTORNO_VIRTUAL.md`, `N1_WALKING_FINAL_SANITY_CHECK.md`, `N1_WALKING_PARTICIPANT_VALIDATION_FINAL.md`, `NEW_100K_SHORT_QUALITY_MODAL_DIAGNOSTIC.md`, `PRODUCTION_BOUNDED_BATCHING_FINAL_REVIEW.md`, `PRODUCTION_LARGE_RUN_ARCHITECTURE_AUDIT.md`, `PRODUCTION_LOCAL_KERNEL_CRASH_DIAGNOSTIC.md`, `PRODUCTION_MODAL_CLASSIFIERS_FINAL_FREEZE_REVIEW.md`, `PRODUCTION_MODAL_DISTRIBUTION_CAUSAL_REVIEW.md`, `PRODUCTION_MODAL_FINAL_FREEZE_CHECK.md`, `PRODUCTION_MODAL_FINAL_REAL_DATA_DIAGNOSTIC.md`, `PRODUCTION_N1_WALKING_IMPROVEMENT_REVIEW.md`, `PRODUCTION_NOTEBOOK_GITHUB_DISCREPANCY_REVIEW.md`, `PRODUCTION_SPARSE_GPS_FINAL_BLOCKERS_RESOLVED.md`, `PRODUCTION_SPARSE_GPS_FINAL_FIX_READY_FOR_CODEX.md`, `PRODUCTION_SPARSE_GPS_FINAL_RESOLUTION.md`, `PRODUCTION_V4_FINAL_PERFORMANCE_OPTIMIZATION_REVIEW.md`, `PRODUCTION_V4_OOM_ROOT_CAUSE.md`, `PRODUCTION_V4_PERFORMANCE_ABLATION.md`, `PRODUCTION_V4_PERFORMANCE_CODEX_FIX_REVIEW.md`, `PRODUCTION_WALKING_GROUND_TRUTH_RECOVERY_AND_HYBRID_REVIEW.md`, `SPARSE_GPS_GUARDRAIL_DECISION.md`, `V4_FINAL_FREEZE_SANITY_CHECK.md`, `V4_FINAL_REAL_PRODUCTION_DOMAIN_VALIDATION.md`, `V4_FINAL_SCIENTIFIC_DOMAIN_FORENSIC_CHECK.md`, `V4_SPARSE_DEGRADATION_ARITHMETIC_SANITY_CHECK.md`.

Varios contienen enlaces absolutos `file:///C:/Users/Eydan/...`; no deben entrar indiscriminadamente al baseline.

### DIAGNOSTIC / SCRATCH

- `dense_trip_regression_baseline_vs_candidate.csv`
- `ground_truth_sparse_feature_audit.csv`
- `ground_truth_sparse_validation.csv`
- `sparse_gps_strict_emissions_gate_verification.csv`
- `sparse_gps_threshold_sweep_two_tier_FINAL.csv`
- `diagnostics/block1_final_oof_validation/bus_threshold_oof_sweep.png`
- `diagnostics/block1_final_oof_validation/metro_threshold_oof_sweep.png`
- `pipeline_v4/calibration_and_diagnostics/modal_classification/artifacts/modal_classifier_hybrid_v1_baseline.pkl` (rollback/diagnóstico, no serving)

### GENERATED OUTPUT

- `Outputs/` completo, ahora ignored.
- `Inputs.zip` (165,917,030 bytes), ahora ignored.
- `__pycache__/`, `.pytest_cache/`, `*.tmp`, outputs de benchmark y checkpoints, ignored.

El notebook mide 8,029 bytes, tiene 8 celdas, 0 outputs y 0 execution counts. No hay outputs embebidos grandes. No hay Parquet/CSV/PNG staged. No hay candidatos recomendados para commit mayores de 50 MB.

## 2. Scientific Contract — PASS

Verificación ejecutable directa:

- artefacto oficial: `modal_classifier_hybrid_v1.pkl`;
- SHA256: `f38c0e36f6039ec396e9b30fc0d1b6595b817cee6d4e578facb8fbdb330cd1dd`;
- `T_metro == 0.30`;
- `T_bus == 0.50`;
- `MIN_EFFECTIVE_PINGS == 8`;
- `MIN_PCT_CONSERVED == 0.30`;
- comparación operativa: 30% porque `pct_pings_conserved` se almacena en escala 0–100;
- MOVES distance rate unit: `g/km`.

El entry point oficial falla cerrado si el backend no es `hybrid` o si cambia el hash. `modal_classifier_hybrid_v1_baseline.pkl` no es seleccionado por serving.

El trace sparse es GPS real: `effective_ping_count = len(trip)` después de segmentación y filtro espacial. No deriva de edges, `speed_raw` ni interpolación. Una regresión L3 de 3 pings demuestra rechazo antes de routing/modal serving; otra demuestra que 120 subsegmentos routed no pueden convertir 5 pings en un viaje admisible.

Walking/Metro nunca pasan el predicado vehicular. Car/Bus requieren `modal_usable`, geometría complete/partial estricta, distancia positiva, ratio 0.50–2.0, gap <=100 m, uncovered <=20% y failed rows <=20%. Routing fallido no produce MOVES. Tier 2 no cambia la clase modal; solo decide elegibilidad de emisiones.

## 3. Memory & Recovery — PASS con riesgo residual documentado

PASS:

- ventanas de user-days acotadas;
- `LOCAL_SAFE` restaurado al batch validado de 32 (no 500);
- checkpoints routing/modal, emissions y ledger;
- schemas mínimos preservados incluso con checkpoints válidos de cero filas;
- identidad de input, metadata, límites, recursos, classifier/hash, thresholds, router y MOVES antes de aceptar resume;
- recomputar routing invalida emisiones previas;
- Parquet y markers usan `.tmp -> os.replace` atómico;
- detailed y summary se escriben por chunks con `PyArrow ParquetWriter`;
- fallo detailed queda explícito y no marca el manifest oficial como completed;
- `run_production(..., resume_run_dir=...)` reanuda el run oficial y reutiliza preprocessing validado.

Riesgo residual MEDIUM: el contrato de retorno conserva `routes` y `individual_emissions` completos en memoria y consolida los frames finales con `pd.concat`. La duplicación histórica adicional de canonicalización del detailed de ~4.4 GiB fue eliminada, pero la arquitectura no es streaming end-to-end/O(1). No se cambió esta API en el freeze.

## 4. Parallel Architecture — PASS

Perfiles presentes: `LOCAL_SAFE`, `LAB_THREADING_MODERATE`, `LAB_THREADING_AGGRESSIVE`, `LAB_PROCESS_TEST`, `CUSTOM`.

El perfil solo propaga backend, workers, batch y metadata de ejecución. Selección de usuarios/fechas/input permanece separada. Resultados se reordenan por task index. El initializer y worker process son top-level; cada worker carga graphs una vez y no recibe graphs por task. El pool persistente se cierra por context manager. Backend desconocido, workers/batch no positivos y process con recursos científicos in-memory fallan cerrado. No hay switch automático oculto.

No se declara backend lab óptimo. El benchmark usa los mismos primeros usuarios/fechas, directorios aislados, `resume=False`, process 4 antes de 6/8 y detiene scaling si la ganancia es <5%. Se corrigió la métrica para usar `pipeline_manifest.user_days` real, no número de viajes, y la paridad ahora hashea ledger/routes/emissions científicas.

## 5. Output Integrity — PASS

- `trip_ledger.parquet`: una fila por `physical_trip_id`, duplicados rechazados.
- `routes_emissions_summary.parquet`: incluye `physical_trip_id`.
- `routes_emissions_detailed.parquet`: link-level, conserva lineage y auditoría.
- viajes fallidos/rechazados permanecen trazables en ledger;
- emissions se filtran por IDs `emissions_usable=True`, modo Car/Bus y geometría no fallida;
- no hay filas de emisiones sin `physical_trip_id`;
- joins usan `_output_row_id` para emisiones y `physical_trip_id` para ledger.

## 6. Environment & Portability — PASS

Runtime verificado: Python 3.12.6, scikit-learn 1.5.2. El pickle carga con warnings convertidos a error y no emitió warnings.

El process pool de Windows usa el mismo `sys.executable` del padre y targets importables top-level. Código de producción sin rutas `C:\Users\Eydan`, claves, tokens, passwords o directorios temporales privados. Notebook usa rutas relativas a `ROOT`. Los warnings LF/CRLF son de autocrlf, no un cambio lógico.

Manifest oficial toma valores runtime: run ID, input/AGEB, resume contract, rango solicitado/real, supplied-users Parquet + SHA256, classifier + SHA256, thresholds, gate, backend, requested/effective workers, batch, executable Python, Git commit/dirty, status y unidad MOVES confirmada.

## 7. Test Results

Suite completa final:

```text
69 passed
0 failed
1 warning
```

Warning visible: `PytestCacheWarning` porque Windows deniega escritura en `.pytest_cache`. No afecta código ni artefactos; no se ocultó.

Smoke cubierto: segmentación, cuatro modos, guardrail sparse, MOVES vehicular, lineage, ledger, summary, detailed, manifest, lectura Parquet, failure injection y resume sin recomputación.

## 8. Issues Found

### BLOCKER

Ninguno pendiente.

### HIGH — corregidos

1. Notebook con `IndentationError`.
2. Checkpoints stale aceptados sin identidad de workload/ciencia.
3. Emisiones stale aceptables después de recomputar routing.
4. Entry point oficial incapaz de reanudar un run existente.
5. Manifest exterior podía reportar completed tras fallo detailed.
6. `LOCAL_SAFE` aumentó accidentalmente batch 32 -> 500.
7. Producción permitía classifier/hash distinto del freeze.

### MEDIUM — corregidos

1. Reemplazo Parquet eliminaba destino antes de rename en Windows.
2. Checkpoints válidos vacíos forzaban recomputación.
3. Harness medía viajes/s pero lo rotulaba user-days/s.
4. Paridad del harness comparaba solo conteos.
5. Manifest decía que unidad MOVES seguía pendiente.

### MEDIUM — residual

- Consolidación final mantiene routes/emissions completos en memoria; no reintroduce la canonicalización monolítica del detailed, pero no es streaming integral.

### LOW — corregidos

- backend/worker/batch inválidos no fallaban temprano;
- definición duplicada de directorios output;
- outputs y `Inputs.zip` no estaban cubiertos globalmente por gitignore;
- log final decía “success” aunque el status pudiera indicar fallo detailed.

## 9. Fixes Applied

Correcciones mínimas en checkpoint identity/atomicidad, resume oficial, manifests, freeze enforcement, batch LOCAL_SAFE, schemas vacíos, notebook, benchmark y tests de regresión. No hubo retraining, recalibración ni cambio de clases/umbrales científicos.

## 10. Files Recommended for Commit

- `.gitignore`
- `notebooks/GPS_preprocessing_and_pipeline_v4.ipynb`
- `pipeline_v4/diagnostics/quality_report.py`
- `pipeline_v4/preprocessing/workflow.py`
- `pipeline_v4/src/config.py`
- `pipeline_v4/src/modal_classification.py`
- `pipeline_v4/src/output_schema.py`
- `pipeline_v4/src/production_workflow.py`
- `pipeline_v4/src/random_forest_contract.py`
- `pipeline_v4/src/run_workflow.py`
- `diagnostics/performance_parallelism/run_lab_benchmark.py`
- los siete tests clasificados como INTENTIONAL TEST CHANGE;
- `CODEX_V4_FINAL_PRE_PUSH_AUDIT.md`.

Revisar el staging exacto con `git diff --cached --name-status` antes del push.

## 11. Files Recommended NOT to Commit

- `Inputs.zip`;
- `Outputs/` completo;
- cinco CSV diagnósticos raíz;
- dos PNG threshold sweep;
- `modal_classifier_hybrid_v1_baseline.pkl`;
- caches, checkpoints, `.tmp` y resultados de benchmark;
- todos los informes Markdown anteriores listados en §1, salvo selección manual consciente. Contienen duplicación, conclusiones históricas y varios paths privados absolutos.

No ejecutar `git add -A` en este working tree.

## 12. Final Verdict

`SAFE TO PUSH AFTER MINOR CLEANUP`

La limpieza pendiente es únicamente staging selectivo de la lista §10. No queda blocker científico, de recovery, schema, environment o Windows identificado por esta auditoría. El benchmark de scaling lab continúa pendiente y no es blocker.

