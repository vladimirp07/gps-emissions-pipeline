# Auditoría de reconciliación de Random Forest modal

Fecha de auditoría: 2026-07-15  
Alcance: inspección de solo lectura del código, notebooks, documentación, historial Git y artefactos PKL disponibles. No se ejecutó entrenamiento, no se modificó el orquestador y no se sobrescribió ningún PKL.

## Resumen ejecutivo

La implementación y el modelo desplegado son inequívocamente de **52 variables**, no de 49. La misma lista ordenada de 52 aparece en entrenamiento, inferencia, notebook y PKL; los tres clasificadores de la cascada reciben las 52 variables completas.

La denominación histórica “49 variables” sólo aparece en texto descriptivo del notebook, dos documentos archivados y el docstring de una prueba. La supuesta lista de 49 incluida en el propio notebook contiene 52 elementos y su salida guardada imprime `Total variables oficiales: 52`. No existe en el historial Git alcanzable, en los respaldos localizados, en scripts archivados ni en ningún PKL disponible una lista material de exactamente 49 nombres. Por ello no es posible identificar con evidencia tres variables concretas que deban retirarse.

Los “124 viajes canónicos” son los 124 viajes con etiqueta única entre 139 viajes físicos del CSV limpio. El generador aplica límites de número de pings y sólo deja 66 de esos 124; también deja 12 viajes mixtos, posteriormente excluidos por entrenamiento. El caché tiene por eso 78 viajes físicos totales, pero el entrenamiento canónico utiliza únicamente 66 viajes y 260 escenarios viaje-degradación.

El modelo desplegado fue serializado con scikit-learn 1.5.2, usa 52 variables y conserva tamaños bootstrap correspondientes a 260 filas en N1, 232 en N2 y 208 en N3. Esos tamaños coinciden exactamente con el dataset reconstruido por el script actual: 260 escenarios canónicos, 232 motorizados y 208 Carro/Bus. La evidencia indica que el modelo corresponde al caché canónico filtrado actual, no a 124 viajes completos ni a una versión de 49 variables.

## 1. Inventario exacto de las 52 variables

### 1.1 Fuentes y consumo

- **Generador de caché:** `pipeline_v4/calibration_and_diagnostics/modes_matrices_finetuning/random_forest_calibration/generar_datos_entrenamiento_ml.py`. Produce las series base ruteadas y, dentro de `local_features`, las variables de ventanas y paradas.
- **Reconstrucción para entrenamiento:** `entrenar_random_forest.py`, líneas 91–193. Agrega las series del caché y deriva las variables escalares que entran al modelo.
- **Lista oficial de entrenamiento:** `entrenar_random_forest.py`, líneas 200–220.
- **Inferencia:** `pipeline_v4/src/modal_classification.py`, lista en líneas 409–428 y extracción en `extract_features`, líneas 674–942.
- **Notebook:** `playground_random_forest.ipynb`, celdas 4 y 5.
- **PKL:** `Inputs/GPS User Data/random_forest_modal.pkl`, claves `feature_cols_v4` y `feature_cols_new`.

Las columnas “Entr.”, “Inf.”, “NB” y “PKL” indican presencia en entrenamiento, inferencia, notebook y modelo serializado. Todas están presentes en los cuatro lugares y en el mismo orden.

| # | Variable | Generación/origen | Cascada | Entr. | Inf. | NB | PKL |
|---:|---|---|---|:---:|:---:|:---:|:---:|
| 1 | `drive_mean_speed` | agregado de `speed_raw` de hipótesis vial | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 2 | `drive_max_speed` | agregado de `speed_raw` vial | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 3 | `drive_std_speed` | agregado de `speed_raw` vial | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 4 | `drive_stop_frac` | fracción de `speed_raw < 2 km/h` | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 5 | `drive_p25_speed` | percentil de `speed_raw` vial | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 6 | `drive_p50_speed` | percentil de `speed_raw` vial | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 7 | `drive_p75_speed` | percentil de `speed_raw` vial | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 8 | `drive_max_speed_diff` | máximo de diferencias consecutivas de velocidad | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 9 | `drive_mean_speed_diff` | media de diferencias consecutivas de velocidad | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 10 | `drive_highway_motorway_frac` | fracción OSM motorway/trunk/primary | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 11 | `drive_highway_residential_frac` | fracción OSM residential | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 12 | `drive_near_bus_frac` | fracción `idx_c == 1`/cercanía a ruta Bus | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 13 | `drive_near_metro_frac` | fracción `idx_c == 0`/cercanía a Metro | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 14 | `drive_near_bus_drift_decay` | `drive_near_bus_frac * exp(-mean_snap_dist_drive/15)` | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 15 | `drive_near_bus_high_drift` | complemento de decaimiento por drift | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 16 | `drive_num_stops` | detector de paradas del generador | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 17 | `drive_mean_stop_duration` | detector de paradas del generador | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 18 | `drive_mean_stop_interval` | detector de paradas del generador | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 19 | `drive_std_stop_interval` | detector de paradas del generador | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 20 | `walk_mean_speed` | agregado de `speed_raw` peatonal | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 21 | `walk_max_speed` | agregado de `speed_raw` peatonal | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 22 | `walk_std_speed` | agregado de `speed_raw` peatonal | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 23 | `walk_highway_footway_frac` | fracción OSM peatonal | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 24 | `walk_p25_speed` | percentil de velocidad peatonal | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 25 | `walk_p50_speed` | percentil de velocidad peatonal | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 26 | `walk_p75_speed` | percentil de velocidad peatonal | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 27 | `walk_max_speed_diff` | máximo de diferencias de velocidad peatonal | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 28 | `walk_mean_speed_diff` | media de diferencias de velocidad peatonal | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 29 | `metro_mean_speed` | agregado de `speed_raw` de hipótesis Metro | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 30 | `metro_max_speed` | agregado de `speed_raw` de hipótesis Metro | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 31 | `metro_near_metro_frac` | fracción `idx_c == 0` | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 32 | `metro_p25_speed` | percentil de velocidad Metro | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 33 | `metro_p50_speed` | percentil de velocidad Metro | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 34 | `metro_p75_speed` | percentil de velocidad Metro | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 35 | `metro_max_speed_diff` | máximo de diferencias de velocidad Metro | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 36 | `metro_mean_speed_diff` | media de diferencias de velocidad Metro | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 37 | `mean_snap_dist_drive` | agregado de `snap_dist_drive` | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 38 | `max_snap_dist_drive` | agregado de `snap_dist_drive` | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 39 | `std_snap_dist_drive` | agregado de `snap_dist_drive` | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 40 | `mean_snap_dist_walk` | agregado de `snap_dist_walk` | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 41 | `max_snap_dist_walk` | agregado de `snap_dist_walk` | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 42 | `std_snap_dist_walk` | agregado de `snap_dist_walk` | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 43 | `metro_win_near_metro_max` | ventanas multiescala, generador líneas 314–361 | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 44 | `metro_win_near_metro_p90` | ventanas multiescala | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 45 | `metro_win_near_metro_consec_run` | persistencia de ventanas Metro | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 46 | `drive_win_near_bus_max` | ventanas multiescala, generador líneas 362–365 | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 47 | `drive_win_near_bus_p90` | ventanas multiescala | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 48 | `drive_win_near_bus_consec_run` | persistencia de ventanas Bus | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 49 | `drive_win_stops_max` | ventanas multiescala de paradas | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 50 | `drive_win_stops_consec_run` | persistencia de ventanas con paradas | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 51 | `walk_win_walk_regime_max` | ventanas multiescala de régimen peatonal | N1, N2, N3 | Sí | Sí | Sí | Sí |
| 52 | `walk_win_walk_regime_consec_run` | persistencia de régimen peatonal | N1, N2, N3 | Sí | Sí | Sí | Sí |

### 1.2 Uso por nivel de cascada

No hay selección diferenciada de variables por nivel:

- N1 ajusta y predice con `feature_cols_v4` completo: 52.
- N2 ajusta y predice con `feature_cols_v4` completo: 52.
- N3 ajusta con `feature_cols_v4`; en inferencia usa `feature_cols_new`, pero en el PKL `feature_cols_new == feature_cols_v4`: 52.

Aunque varias variables tengan semántica específica de Metro, Bus o Caminar, el código entrega todas a los tres bosques.

## 2. Reconciliación histórica 49 vs. 52

### 2.1 Evidencia encontrada

| Fuente | Afirmación textual | Lista material | Resultado verificable |
|---|---|---|---|
| `playground_random_forest.ipynb` | “49 variables” | Sí | La lista tiene 52; la salida guardada dice 52 |
| `archive/random_forest_experiments/README.md` | baseline 49, experimento 55 | No | Sólo narrativa |
| `auditoria_viajes_bus_persistentes.md` | control 49 + seis Bus = 55 | No | Sólo narrativa y métricas |
| `test_random_forest_official.py` | docstring “49 variables” | No | Las aserciones exigen 52 |
| `entrenar_random_forest.py` | 52 variables | Sí | 52 nombres |
| `modal_classification.py` | lista de producción | Sí | 52 nombres |
| `random_forest_modal.pkl` | esquema del modelo | Sí | Dos listas iguales de 52 |
| `analizar_errores_eda.py` | lista EDA reducida | Sí | 38 nombres, no 49 |

Los seis features experimentales Bus son:

1. `stop_cycles_per_km`
2. `median_stop_spacing_m`
3. `cv_stop_spacing`
4. `median_restart_time_s`
5. `p90_restart_time_s`
6. `stop_pattern_persistence`

El generador y la inferencia todavía los calculan, pero no aparecen en ninguna de las listas de 52 usadas por los modelos. Por tanto, la aritmética implementada es **52 baseline + 6 experimentales = 58**, no 49 + 6 = 55.

### 2.2 Historial Git y archivos históricos

- El historial de `main` y `origin/main` no contiene los scripts Random Forest actuales; aparecen sin seguimiento en el worktree.
- Los únicos archivos históricos de clasificación modal presentes en commits son implementaciones bayesianas.
- No hay ramas ni tags adicionales con una implementación RF anterior.
- La búsqueda incluyó `archive/`, `scratch/`, `Respaldos temporales/`, `Temporary outputs/`, notebooks, Markdown, JSON, CSV, archivos `.bak` y PKL.
- Ningún archivo disponible contiene una lista material de 49 nombres.

### 2.3 Las tres variables de diferencia

**No es posible determinar cuáles serían esas tres variables con evidencia del repositorio.** No existe un conjunto histórico de 49 contra el cual calcular `52 − 49`. Seleccionar las posiciones 50–52, tres variables de ventanas o cualquier otro trío sería una suposición sin respaldo.

La evidencia sí demuestra que “49” es, en los artefactos disponibles, una etiqueta documental desactualizada o un error de conteo: la celda rotulada como lista de 49 contiene exactamente la lista actual de 52 y su propia ejecución guardada contabiliza 52.

## 3. Auditoría del dataset

### 3.1 Universo canónico

`Datos de MATLAB GPS Limpios.csv` contiene:

| Concepto | Viajes físicos |
|---|---:|
| Total | 139 |
| Etiqueta única/canónica | 124 |
| Etiqueta mixta | 15 |
| Etiqueta vacía | 0 |

Distribución de los 124 canónicos:

| Clase | Viajes |
|---|---:|
| Carro | 91 |
| Caminar | 15 |
| Bus | 12 |
| Metro | 6 |

### 3.2 Por qué el caché sólo representa 78 viajes

El generador aplica estos límites antes de crear tareas:

| Clase | Pings mínimos | Pings máximos |
|---|---:|---:|
| Caminar | 100 | 600 |
| Carro | 300 | 1,200 |
| Bus | 50 | 5,000 |
| Metro | 100 | 5,000 |

El resultado del filtro es:

- 84 grupos `(caid, trip, modo)` elegibles.
- 78 viajes físicos distintos.
- 66 viajes canónicos: 41 Carro, 12 Bus, 7 Caminar y 6 Metro.
- 12 viajes mixtos también pasan el filtro y quedan escritos en el caché.
- Los 58 viajes canónicos restantes son rechazados exclusivamente por los límites de pings: 50 Carro y 8 Caminar.
- No hay viajes canónicos elegibles ausentes completamente del caché; los 66 elegibles aparecen.

Así se reconcilian los 124 canónicos: **66 incluidos + 58 descartados por longitud = 124**. Los 78 del caché no son “78 canónicos”: son **66 canónicos + 12 mixtos**.

### 3.3 Escenarios reales Raw/L1/L2/L3

El caché `datos_entrenamiento_ml.pkl` contiene 482 registros de hipótesis, pero un registro no equivale a un escenario: cada escenario puede tener entre una y cinco hipótesis ruteadas.

| Nivel | Escenarios físicos totales | Escenarios canónicos usados |
|---|---:|---:|
| Raw | 77 | 65 |
| L1 | 78 | 66 |
| L2 | 77 | 65 |
| L3 | 76 | 64 |
| **Total** | **308** | **260** |

Sólo dos viajes carecen de alguna degradación completa en el caché:

- `CHH_20`: sólo tiene L1; faltan Raw, L2 y L3.
- `EAAL_8`: falta L3.

Distribución del número de hipótesis por escenario: 166 escenarios con una hipótesis, 122 con dos, 11 con tres, 6 con cuatro y 3 con cinco.

### 3.4 Reproducibilidad de métricas

El caché actual sí contiene las 260 filas canónicas que muestra el notebook guardado. Sus soportes de validación también suman 260:

- Carro: 160 escenarios.
- Bus: 48.
- Metro: 24.
- Caminar: 28.

Por ello, con el mismo entorno y semillas, el caché puede en principio reproducir la evaluación guardada del notebook: `GroupKFold(5)`, Balanced Accuracy 86.80% y Macro F1 86.99%. Esta auditoría no reentrenó el modelo para volver a calcularlas.

No puede reproducir como evaluación de “124 viajes canónicos” porque sólo contiene 66 de ellos. Tampoco puede reconciliar directamente las métricas archivadas del experimento Bus:

- Las matrices archivadas suman 312 observaciones, frente a 260 escenarios canónicos actuales.
- Los reportes mencionan `StratifiedGroupKFold` de 20 particiones y Balanced Accuracy alrededor de 77–78%, mientras el notebook usa `GroupKFold(5)` y reporta 86.80%.
- No se conserva el script ni el dataset exacto que generaron las 312 predicciones archivadas.

## 4. Auditoría del modelo desplegado

Archivo: `Inputs/GPS User Data/random_forest_modal.pkl`  
Tamaño: 490,661 bytes  
Marca de tiempo del sistema de archivos: 2026-07-15 04:00:47, hora local. Esta fecha no está embebida como metadata en el diccionario.

### 4.1 Metadata disponible

El diccionario sólo contiene:

- `clf_n1`
- `clf_n2`
- `clf_n3`
- `feature_cols_v4`
- `feature_cols_new`

No contiene hash de dataset, lista de viajes, fecha de entrenamiento, commit, métricas, versión de NumPy/Python ni manifiesto de generación.

El pickle contiene `_sklearn_version = 1.5.2`. Al cargarlo con scikit-learn 1.8.0 se emite `InconsistentVersionWarning`; por tanto, la versión demostrable de serialización es **scikit-learn 1.5.2**.

### 4.2 Features e hiperparámetros

Los tres modelos tienen `n_features_in_ = 52`, `feature_names_in_` con los 52 nombres de la sección 1 y 100 árboles.

Parámetros comunes:

- `bootstrap=True`
- `criterion='gini'`
- `class_weight='balanced'`
- `max_depth=7`
- `max_features='sqrt'`
- `min_samples_split=2`
- `n_estimators=100`
- `random_state=42`
- `oob_score=False`
- `ccp_alpha=0.0`
- `n_jobs=None`

Diferencia por nivel:

- N1: `min_samples_leaf=4`.
- N2: `min_samples_leaf=4`.
- N3: `min_samples_leaf=2`.

### 4.3 Correspondencia con el dataset

Los índices bootstrap conservados permiten recuperar el tamaño de ajuste de cada bosque:

| Modelo | Filas de ajuste guardadas | Filas esperadas del caché actual |
|---|---:|---:|
| N1: Caminar vs. motorizado | 260 | 260 escenarios canónicos |
| N2: Metro vs. superficie | 232 | 260 − 28 Caminar = 232 |
| N3: Carro vs. Bus | 208 | 160 Carro + 48 Bus = 208 |

La coincidencia exacta, junto con la lista y los hiperparámetros idénticos al script, demuestra que el modelo es compatible con el dataset canónico filtrado reconstruido desde el caché actual. No demuestra criptográficamente identidad de cada fila porque el PKL carece de hash/manifiesto, pero descarta que sea un modelo entrenado sobre los 124 viajes completos o sobre 49 variables.

## 5. Auditoría de `playground_random_forest.ipynb`

### 5.1 Partes reproducibles con los datos actuales

- Lectura del CSV limpio y obtención de 139 viajes, 15 mixtos y 124 de etiqueta única.
- Exclusión de la lista de 15 viajes mixtos.
- Lectura del caché y reconstrucción de 260 escenarios canónicos.
- Extracción de las 52 variables.
- Evaluación jerárquica con `GroupKFold(5)` y semilla 42, siempre que se use un entorno compatible.
- Carga del PKL y una inferencia de demostración basada en sus 52 features.

Las salidas guardadas son internamente coherentes con el caché actual: DataFrame `(260, 57)`, 52 variables y matriz de confusión con soporte total 260.

### 5.2 Placeholders y simplificaciones

- `pct_conserved` se fija a `99.0` para todas las filas. El notebook define rutas a los CSV bruto y limpio, pero no calcula la conservación real.
- El guardrail se demuestra con casos sintéticos; no se valida sobre los 260 escenarios reales.
- `num_pings` se toma de la hipótesis vial. Para escenarios sin hipótesis Carro/Bus queda en cero, aunque pueda existir una hipótesis peatonal o Metro.
- El notebook no genera el caché ni valida que proceda de la misma versión del ruteo; parte de un PKL preexistente.
- No registra versiones del entorno, hash del caché, hash del modelo ni commit.
- La resolución `Path("../../../..").resolve()` depende de ejecutar el notebook con su directorio como working directory.

### 5.3 Resultados que no pueden reproducirse con la evidencia actual

- Una evaluación realmente basada en los 124 viajes canónicos completos.
- Una versión material de 49 variables.
- La comparación histórica 49 vs. 55.
- Las matrices archivadas de 312 observaciones y los 20 folds del experimento Bus.
- El porcentaje conservado y el comportamiento real del guardrail reportado viaje por viaje.
- Una garantía bit a bit del modelo sin recrear el entorno scikit-learn 1.5.2 y sin un manifiesto de datos.

## 6. Conclusión y recomendación

La evidencia disponible permite afirmar que la versión ejecutable y desplegada actual es de 52 variables, pero no permite convertir esa observación en una decisión científica de que esas 52 deban ser las oficiales. Tampoco permite reconstruir qué tres variables componían la diferencia histórica, porque nunca se conservó una lista real de 49 ni el dataset/script de la comparación reportada.

Antes de oficializar 49 o 52 se necesita regenerar un dataset con manifiesto a partir de los 124 viajes canónicos, definir explícitamente un candidato de 49 variables con justificación trazable y compararlo contra las 52 bajo el mismo esquema de validación agrupada.

**Recomendación inequívoca: C. No hay evidencia suficiente y es necesario regenerar el dataset y comparar 49 vs. 52.**

