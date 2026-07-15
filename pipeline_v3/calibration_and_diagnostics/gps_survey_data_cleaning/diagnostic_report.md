# Reporte Consolidado de Diagnóstico, Limpieza y Segmentación (Dataset de MATLAB GPS)

Este reporte documenta el análisis exploratorio de datos (EDA), los resultados del diagnóstico de calidad física y espacial, y la validación de la segmentación temporal realizada sobre los datos de encuesta de MATLAB (`Datos de MATLAB GPS.csv`), etiquetados manualmente por los participantes. 

El objetivo es establecer y detallar la metodología de depuración para eliminar glitches de GPS, viajes mal clasificados y colas de inactividad estática, garantizando datos limpios y físicamente realizables para la calibración del clasificador bayesiano en Optuna y el cálculo preciso de emisiones.

---

## 1. Criterios de Calidad de Datos y Limpieza Física

Para la depuración de los datos de MATLAB, aplicamos cuatro niveles de limpieza implementados en el script [depurar_datos_matlab.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/gps_survey_data_cleaning/depurar_datos_matlab.py):

*   **Identificación y Descarte de Viajes Corruptos por Duplicidad (Fase 1):**
    *   Si un viaje presenta más del **2% de sus marcas de tiempo duplicadas**, se cataloga como corrupto (fusión de trayectorias distintas de diferentes dispositivos o fallas de volcado del logger) y se **elimina por completo**.
    *   Para viajes válidos con duplicidad menor al 2%, se conserva únicamente el primer registro cronológico, eliminando los repetidos.
*   **Validación Secuencial de Trayectoria (Fase 2 - Path Validation):**
    *   Para detectar glitches individuales y bloques de teletransportación sin enmascaramiento, se establece el primer punto físicamente válido como ancla (`last_valid`).
    *   Para cada punto sucesivo, se calcula la distancia y tiempo transcurrido **con respecto al `last_valid`** en lugar del predecesor inmediato. Si la velocidad instantánea excede el límite físico de su modo de transporte (o el límite global de glitch de **250 km/h**), el punto se marca como anómalo y el ancla **NO** se actualiza. Si es coherente, el punto se acepta y se convierte en el nuevo `last_valid`.
    *   **Límites de velocidad física por modo:**
        *   **Caminar:** Velocidades superiores a **30 km/h** (este umbral holgado se establece porque el cálculo geodésico Haversine directo punto a punto sobreestima la velocidad real debido al jitter de alta frecuencia del receptor de GPS de 1 Hz).
        *   **Metro y Autobús (Bus):** Velocidades superiores a **110 km/h** (límite máximo operativo urbano).
        *   **Automóvil (Carro):** Velocidades superiores a **160 km/h**.
    *   **Inconsistencia Espacial de Infraestructura (Caso Metro):** Pings clasificados como `metro` que se ubican a más de **300 metros** de la línea oficial de Metrorrey ([lineas_metrorrey.csv](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/Infrastructure/lineas_metrorrey.csv)) se descartan.
    *   Si un viaje tiene más del **30% de pings anómalos**, se asume una clasificación errónea general y se descarta todo el viaje.
*   **Poda de Caminatas Vehicularizadas (Fase 3 - Walking Trip Pruning):**
    *   Para los viajes de Caminar que no superan el 30% de error total, pero que en algún punto muestran velocidades de vehículo (>30 km/h): se corta el viaje en ese instante exacto. Se conserva la primera parte (caminata lógica) y se elimina el tramo vehicular posterior (donde el usuario abordó un autobús o automóvil pero olvidó detener el registro de caminar).
*   **Recorte de Extremos Estáticos (Fase 4 - Trim-to-Motion):**
    *   *El Problema:* Los usuarios activaban el log antes de salir de casa o lo dejaban encendido horas después de llegar y estacionarse, generando largos bloques de pings estáticos al inicio y fin del viaje.
    *   *Solución:* Se recorta el viaje eliminando los pings estacionarios en los bordes. Escaneamos desde el inicio hacia adelante y desde el final hacia atrás para conservar el viaje únicamente entre el primer y último ping que superen un umbral de movimiento mínimo:
        *   **Caminar:** Umbral de movimiento de **2.0 km/h**.
        *   **Carro / Bus / Metro:** Umbral de movimiento de **5.0 km/h**.
    *   Si ningún ping del viaje supera el umbral (viaje 100% estático) o la porción restante tiene menos de 2 pings, el viaje se descarta por completo.

---

## 2. Resultados del Proceso de Depuración Física

El proceso de depuración sistemática fue ejecutado sobre el dataset original de MATLAB GPS, arrojando los siguientes resultados estadísticos:

### Resumen de Pings y Viajes:
*   **Pings Crudos en el CSV Original:** 398,043 pings (289 viajes)
*   **Pings en Viajes Corruptos Descartados (Fase 1, Tasa Duplicados >2%):** 79,413 pings (32 viajes descartados)
*   **Pings tras deduplicación de registros (≤2% duplicados):** 317,699 pings (257 viajes restantes)
*   **Pings Limpios Guardados Finales:** **177,520 pings** (139 viajes restantes)
*   **Total de Pings Eliminados en Segunda Fase:** 140,179 pings (44.12% de los pings restantes)
    *   *Por descarte de viajes completos (inactividad total, vacíos o >30% de error):* 129,121 pings
    *   *Por poda de caminatas vehicularizadas:* 129,731 pings
    *   *Por recorte de extremos estáticos (Trim-to-Motion):* 1,947 pings
    *   *Por glitches individuales en trayectos válidos:* 1,435 pings
*   **Viajes Completamente Eliminados:** 118 viajes (45.91% de los viajes que pasaron la duplicidad). Esto remueve una enorme cantidad de registros basura que consistían puramente en usuarios estacionados sin movimiento.
*   **Viajes Peatonales Truncados/Podados:** 127 viajes.

### Auditoría de Velocidades Máximas por Modo de Transporte:

| Modo de Transporte | Velocidad Máxima (Original Crudo) | Velocidad Máxima (Con Limpieza) | Velocidad Promedio Limpia | Estado de Limpieza |
| :--- | :---: | :---: | :---: | :--- |
| **Caminar** | 2,952.60 km/h | **29.94 km/h** | 2.85 km/h | **100% Físicamente Coherente** |
| **Carro** | 29,746.20 km/h | **158.47 km/h** | 31.58 km/h | **100% Físicamente Coherente** |
| **Autobús (Bus)** | 85,296.72 km/h | **109.96 km/h** | 24.04 km/h | **100% Físicamente Coherente** |
| **Metro** | 4,003.02 km/h | **109.09 km/h** | 31.05 km/h | **100% Físicamente Coherente** |

---

## 3. Diagnóstico de Segmentación Temporal (Veraset State Machine vs. Manual)

Comparamos de manera independiente los registros de viaje manuales de MATLAB frente a la salida del algoritmo de segmentación temporal de Veraset (máquina de estados con $v < 3\text{ km/h}$ y $d < 100\text{ m}$ por más de 5 minutos, definido en [segmentation.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/src/segmentation.py)):

### 3.1 Presencia de Paradas en los Viajes Manuales
*   **Viajes manuales con paradas cortas en extremos:** **73.36% (212 viajes)** contienen al menos un ping de parada en sus límites, reflejando el desfase humano menor al encender o apagar la encuesta.
*   **Viajes manuales con paradas largas internas (>= 5 min):** **20.07% (58 viajes)** de los 289 contienen paradas acumuladas duraderas.
*   **Sesgo de inactividad:** Para ese 20% de viajes, la inactividad interna promedio fue de **37.45 minutos** (máximo extremo de 492 minutos). Esto demuestra el gran sesgo de inactividad que existe en el dataset manual.

### 3.2 Coincidencia Temporal y Desfases
*   **Tasa de Emparejamiento:** **99.31%** de los viajes manuales coinciden temporalmente con al menos una sección de viaje del algoritmo.
*   **IoU (Intersección sobre Unión):** IoU Mediana de **87.58%**, demostrando una alineación temporal muy alta en trayectos regulares.
*   **Desfase en Tiempos de Inicio:** Mediana de **0.00 minutos** (coincidencia perfecta en el arranque en la mayoría de los casos).
*   **Desfase en Tiempos de Fin:** Mediana de **0.28 minutos (17 segundos)**, lo cual demuestra que el final de los viajes está sumamente sincronizado en condiciones normales.

---

## 4. Problemática del Trip Splitting e Integración con Optuna

Durante las fases previas del proyecto, se analizó la posibilidad de realizar **Trip Splitting** (dividir un viaje manual en múltiples segmentos activos cuando se detectara una parada intermedia superior a 5 minutos). Sin embargo, se identificaron limitaciones de diseño y se decidió tomar la siguiente ruta para la calibración con Optuna:

### 4.1 Descarte del Trip Splitting
*   **Ausencia de Etiquetas de Modo:** Si dividimos un viaje manual que originalmente está etiquetado como `Carro` en tres segmentos (Viaje A, Parada de 10 min, Viaje B), no tenemos garantía de que las etiquetas manuales del modo de transporte sigan siendo válidas o uniformes para cada sub-segmento. Por ejemplo, el usuario pudo haber tomado un autobús para el segundo segmento (Viaje B). 
*   Dividir el viaje arbitrariamente y propagar la etiqueta original generaría **falso ground truth**, contaminando el conjunto de entrenamiento de Optuna.

### 4.2 Solución Implementada: Exclusión de Paradas en la Calibración de Optuna
*   Dado que las paradas estacionarias intermedias y prolongadas se manejan en producción mediante la máquina de estados espacial y temporal de [segmentation.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/src/segmentation.py) (que asigna IDs negativos a periodos inmóviles), el clasificador bayesiano ([BayesianRouteEvaluator](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/src/bayes_classifier.py)) no clasifica "paradas", sino que determina el modo de transporte activo (`Carro`, `Bus`, `Metro`, `Caminar`).
*   Por lo tanto, la base de entrenamiento para la sintonización hiperparamétrica en Optuna **no incluye ni evalúa el estado de parada**. Se extrae únicamente la información de movimiento activo de los viajes depurados para optimizar las 56 probabilidades de las matrices condicionales, minimizando el impacto del sesgo de quietud sobre los parámetros físicos de velocidad y distancia.

---

## 5. El Bug Histórico de la Base de Datos Encontrado

Durante la auditoría del pipeline, se identificó un bug crítico en la lógica previa de limpieza que invalidaba el filtrado de velocidades extremas:

1.  **El bug del tiempo cero (`dt_sec = 0.0`):** Cuando dos pings en el archivo `Datos de MATLAB GPS.csv` tenían exactamente la misma marca de tiempo (`Timestamp`), la diferencia temporal se calculaba como cero. El código anterior interceptaba esta división por cero y le asignaba al ping una velocidad de `0.0 km/h`.
2.  **La teletransportación oculta:** Esto permitía que saltos espaciales gigantescos (causados por logger freezes o por registros encimados de dos dispositivos con el mismo ID de viaje) se guardaran en el dataset "limpio" con una velocidad ficticia de 0 km/h.
3.  **Ruptura del ruteo:** Cuando el motor de map matching ordenaba cronológicamente estos puntos sobre la red vial, las velocidades reales de miles de km/h se recalculavam en el grafo de Dijkstra, provocando fallas masivas de ruteo y descalibración de las matrices.
4.  **Solución:** Ahora se evalúa la tasa de duplicación por viaje, descartando viajes corruptos (>2% duplicados) y conservando únicamente el primer ping por segundo en viajes con tasas menores a 2%.

---

## 6. Sintonización Desacoplada con Optuna

Para entrenar Optuna en menos de 5 segundos evitando la latencia de 1-2 segundos que introduce el cálculo Dijkstra por viaje, implementamos una **arquitectura desacoplada**:

1.  **Fase A (Offline - [generar_datos_entrenamiento.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modal_classification/generar_datos_entrenamiento.py)):** Carga los datos limpios de MATLAB, realiza el map matching y ruteo una única vez, discretiza las variables de producción y guarda los índices correspondientes en el archivo serializado `datos_entrenamiento_optuna.pkl`.
2.  **Fase B (Online - [optimizar_matrices_optuna.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modal_classification/optimizar_matrices_optuna.py)):** Carga el caché precomputado y ejecuta la optimización de Optuna en memoria mediante lookups rápidos de NumPy (`probs_by_mode = Cercania[idx_c] * Velocidad[idx_v]...`).

### Instrucciones de ejecución
Para regenerar la base de entrenamiento con la sintonización balanceada de 15 viajes por modo (con la nueva limpieza de extremos estáticos activa), ejecuta en tu terminal:

```powershell
# 1. Limpieza de datos (Fase 1 a 4)
python pipeline_v3/calibration_and_diagnostics/gps_survey_data_cleaning/depurar_datos_matlab.py

# 2. Generación de caché desacoplado para Optuna (15 viajes balanceados por modo)
python pipeline_v3/calibration_and_diagnostics/modal_classification/generar_datos_entrenamiento.py --balanced --trips-per-mode 15
```

