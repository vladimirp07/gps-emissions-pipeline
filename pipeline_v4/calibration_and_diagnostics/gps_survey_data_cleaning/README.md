# Depuración y Limpieza del Dataset de MATLAB

Esta carpeta está destinada a albergar los scripts y notebooks para realizar la depuración sistemática del dataset manual de **MATLAB GPS** (`Datos de MATLAB GPS.csv`). 

Dado que la clasificación original de los modos de transporte fue realizada de forma manual, es altamente factible que contenga incongruencias físicas e incongruencias espaciales que afecten negativamente la calibración del algoritmo de ruteo y las matrices probabilísticas del pipeline.

---

## Plan de Depuración y Auditoría Física

A continuación se detallan las incongruencias específicas que se auditarán y corregirán viaje por viaje, modo por modo:

### 1. Incoherencias de Velocidad Física (Límites por Modo)
Filtrar pings sucesivos cuya velocidad geodésica instantánea ($v = \Delta d / \Delta t$) exceda los límites físicos del modo de transporte asignado:
* **Caminar / Peatonal:** Velocidades instantáneas $> 22\text{ km/h}$ (límite humano máximo de sprint).
* **Metro:** Velocidades instantáneas $> 110\text{ km/h}$ (velocidad máxima operativa de Metrorrey).
* **Parada (Estático):** Velocidades de desplazamiento $> 4\text{ km/h}$ mientras se declare que el usuario está detenido.
* **Vehicular (Carro / Bus):** Velocidades instantáneas $> 160\text{ km/h}$.

### 2. Dispersión Espacial de Infraestructura Exclusiva (Caso Metro)
* **Metro fuera de Vías:** Detectar viajes etiquetados como `metro` cuyos pings GPS se encuentren a una distancia geodésica superior a un radio de tolerancia (ej. $> 300\text{ metros}$) respecto a la geometría oficial de las líneas de Metrorrey (`lineas_metrorrey.csv`).
* **Justificación:** Si un ping de metro está a kilómetros de las vías del tren, es un error de clasificación manual (el usuario probablemente iba en carro o camión por una avenida paralela).

### 3. Saltos Absurdos y Teletransportación (GPS Bounces)
* **Saltos de Posición:** Filtrar pings individuales que muestren un pico de velocidad extremo (ej. $> 250\text{ km/h}$) seguido de un retorno inmediato a la velocidad normal anterior. Estos representan rebotes de señal por efecto cañón (edificios altos) o glitches del sensor del teléfono.

### 4. Coherencia Temporal y Duración Mínima
* **Viajes Huérfanos:** Identificar y remover registros de viajes que contengan menos de 5 pings en total o cuya duración acumulada sea inferior a 1 minuto, ya que no aportan información representativa para el ruteo.

---

## Resultados de la Depuración (Dataset de MATLAB GPS)

El proceso de depuración sistemática fue ejecutado sobre el dataset original [Datos de MATLAB GPS.csv](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/GPS%20User%20Data/Datos%20de%20MATLAB%20GPS.csv) utilizando el script de automatización [depurar_datos_matlab.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v4/calibration_and_diagnostics/gps_survey_data_cleaning/depurar_datos_matlab.py).

Los resultados de esta limpieza y remoción sistemática de anomalías se resumen a continuación:

### Estadísticas Generales de Limpieza
* **Pings Originales:** 398,043
* **Pings Limpios Guardados:** 240,708 (conservados para análisis de ruteo y emisiones)
* **Total de Pings Eliminados:** 157,335 (39.53% del dataset original)
  * **Por descarte de viajes completos:** 52,738 pings (viajes con >30% de anomalías o que quedaron con menos de 2 pings).
  * **Por poda de caminatas (Walking Trip Pruning):** 102,157 pings (tramo vehicular detectado y truncado a velocidades peatonales >30 km/h).
  * **Por glitches individuales en viajes válidos:** 2,440 pings (picos aislados de velocidad >250 km/h o metro a >300 metros de las vías).
* **Viajes Originales:** 289
* **Viajes Completamente Eliminados:** 27 (9.34% del total de viajes)
* **Viajes Peatonales Podados (Truncados):** 119 viajes de Caminar (donde el participante abordó un vehículo pero olvidó cambiar la etiqueta manual).

El dataset depurado resultante se guardó localmente en [Datos de MATLAB GPS Limpios.csv](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/GPS%20User%20Data/Datos%20de%20MATLAB%20GPS%20Limpios.csv) manteniendo el formato exacto de columnas para asegurar compatibilidad total con el pipeline.

---

## Entregables Esperados en esta Carpeta

* `notebooks/`: Notebooks interactivos de diagnóstico visual para graficar la distribución de incongruencias.
* `src/`: Scripts modulares para automatizar la limpieza y exportar el dataset purgado manteniendo el mismo esquema de columnas original (`caid`, `num_trip`, `lat`, `lon`, `Timestamp`, `mode_of_transport`).

