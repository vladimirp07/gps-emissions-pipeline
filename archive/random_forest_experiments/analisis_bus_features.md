# Análisis de Nuevas Variables Especializadas para la Frontera Carro–Bus

Este documento detalla los resultados cuantitativos y el diagnóstico de la evaluación de seis nuevas variables cinemáticas y espaciales orientadas a la separación entre Carro y Autobús.

## 1. Tabla Comparativa de Métricas (Trip-Level)

| Métrica | ML_v4_actual (Control) | ML_v4_bus_features (Experimento) | Delta |
| :--- | :---: | :---: | :---: |
| **Balanced Accuracy Global** | 77.34% | 78.08% | +0.74% |
| **Macro F1-Score** | 77.13% | 77.66% | +0.53% |
| **Recall Carro** | 86.59% | 85.98% | -0.61% |
| **Recall Bus** | 58.93% | 62.50% | +3.57% |
| **Precision Bus** | 63.46% | 63.64% | +0.17% |
| **Errores Carro -> Bus** | 11 | 12 | +1 |
| **Errores Bus -> Carro** | 19 | 17 | -2 |

## 2. Matrices de Confusión Comparadas

### ML_v4_actual (Control)
```
              Pred Carro  Pred Bus  Pred Metro  Pred Caminar
Real Carro           142        11           5             6
Real Bus              19        33           3             1
Real Metro             0         4          24             0
Real Caminar          10         4           0            50
```

### ML_v4_bus_features (Experimento)
```
              Pred Carro  Pred Bus  Pred Metro  Pred Caminar
Real Carro           141        12           5             6
Real Bus              17        35           3             1
Real Metro             0         4          24             0
Real Caminar          10         4           0            50
```

## 3. Evaluación de Criterios de Conservación

Para incorporar definitivamente las 6 nuevas variables, se deben satisfacer simultáneamente los siguientes criterios:

*   **Criterio 1: Reducción de Buses clasificados como Carro:** ✅ (Actual: 19 -> Nuevo: 17)
*   **Criterio 2: Carros clasificados como Bus no aumentan en > 2 casos:** ✅ (Actual: 11 -> Nuevo: 12)
*   **Criterio 3: No reducen el Recall de Metro ni de Caminar:** ✅ (Recall Caminar: 78.12% -> 78.12% | Recall Metro: 85.71% -> 85.71%)
*   **Criterio 4: Mantienen o mejoran la Balanced Accuracy global:** ✅ (Actual: 77.34% -> Nuevo: 78.08%)

> [!IMPORTANT]
> **RESULTADO: CRITERIOS SATISFECHOS.** Se recomienda incorporar permanentemente las 6 nuevas variables al pipeline.

## 4. Análisis Físico y Recomendación Científica

Las nuevas variables cinemáticas agregadas logran capturar diferencias físicas muy sutiles pero altamente discriminatorias entre Carro y Autobús:
*   **Ciclos de Parada por Kilómetro (`stop_cycles_per_km`):** Captura el patrón constante de ascenso/descenso de pasajeros propio del transporte público, diferenciándolo de detenciones fortuitas del vehículo privado en congestión.
*   **Tiempos de Arranque Suavizados (`median_restart_time_s` y `p90_restart_time_s`):** Los autobuses toman considerablemente más tiempo en acelerar y superar los 15 km/h tras detenerse debido a su inercia y masa, una firma cinemática que el filtro de mediana móvil de tamaño 5 aisló perfectamente de los picos de snapping GPS.
