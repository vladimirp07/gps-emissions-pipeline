# Diagnostico de Calidad de Datos del Dataset de MATLAB

Este reporte documenta los resultados de la auditoria fisica y espacial realizada sobre los datos de encuesta de MATLAB (`Datos de MATLAB GPS.csv`), la cual fue etiquetada de forma manual por los participantes de la encuesta. 

El objetivo es cuantificar y localizar las incongruencias fisicas (velocidades imposibles por modo, glitches de teletransportacion) y espaciales (metro operando fuera de la red ferroviaria) para fundamentar un proceso de depuracion de datos en el futuro.

---

## 1. Criterios de Calidad de Datos Aplicados

Para identificar los puntos anomalos, definimos tres tipos de inconsistencias:

* **Inconsistencia de Velocidad por Modo (Anomalia de Velocidad):**
  * **Caminar:** Velocidades instantaneas superiores a **15 km/h** (limite de trote/carrera ligera para una clasificacion peatonal).
  * **Parada (Detenido):** Desplazamientos que reportan velocidades instantaneas superiores a **4 km/h** (lo que indica que el usuario se estaba moviendo significativamente a pesar de reportarse estatico).
  * **Metro / Autobus:** Velocidades instantaneas superiores a **110 km/h** (limite de velocidad operativa urbana).
  * **Automovil (Carro):** Velocidades instantaneas superiores a **160 km/h**.
* **Inconsistencia Espacial de Infraestructura (Metro fuera de vias):**
  * Pings clasificados como `metro` que se ubican a mas de **300 metros** de distancia de la linea geometrica oficial de Metrorrey (`lineas_metrorrey.csv`).
* **Teletransportacion General (GPS Glitch):**
  * Saltos de posicion que requieran velocidades instantaneas superiores a **250 km/h** en cualquier modo de transporte.

---

## 2. Resultados del Diagnostico por Modo de Transporte

A continuacion se presenta la proporcion de pings anomalos detectados, asi como la distancia y tiempo afectados por estas inconsistencias:

| Modo de Transporte | Pings Totales | Pings Anomalos | % Pings Anomalos | % Distancia Afectada | % Tiempo Afectado | Detalles de la Anomalia |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Caminar** | 184,860 | 16,039 | 8.68% | 47.33% | 1.03% | 8.7% exceden vel. max, 0.0% glitches GPS |
| **Carro** | 167,230 | 449 | 0.27% | 12.82% | 0.05% | 0.3% exceden vel. max, 0.3% glitches GPS |
| **Bus** | 34,705 | 2,631 | 7.58% | 98.55% | 0.92% | 7.6% exceden vel. max, 7.6% glitches GPS |
| **Metro** | 8,674 | 39 | 0.45% | 7.12% | 0.03% | 0.4% exceden vel. max, 0.0% glitches GPS |

---

## 2.1 Distribucion de Velocidades en el Modo Caminar (Anomalias Peatonales)

Dado el alto porcentaje de distancia afectada en los datos etiquetados manualmente como `Caminar`, realizamos un analisis de distribucion acumulada de velocidades instantaneas sobre los **184,860 pings peatonales** para identificar el tipo de error en la encuesta:

| Umbral de Velocidad | Pings que lo superan | % Pings Peatonales | Distancia Acumulada (km) | % Distancia de Caminar | Tipo de Incongruencia / Diagnostico |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **> 6.0 km/h** | 20,911 | 11.31% | 269.75 km | 52.24% | Trote, carrera o transito vehicular |
| **> 10.0 km/h** | 16,803 | 9.09% | 248.77 km | 48.18% | Velocidad de carrera continua o vehiculo |
| **> 15.0 km/h** | 16,039 | 8.68% | 244.40 km | 47.33% | Sprint imposible o vehiculo motorizado (Carro/Bus) |
| **> 25.0 km/h** | 15,598 | 8.44% | 238.28 km | 46.15% | Transito vehicular urbano indudable |
| **> 50.0 km/h** | 2,015 | 1.09% | 90.98 km | 17.62% | Transito vehicular en avenidas/autopistas |

* **Conclusiones del analisis de distribucion:**
  1. El **46.15% de la distancia total** declarada como caminata ocurrió a mas de **25 km/h**, lo cual es biologicamente imposible para un peaton.
  2. Esto demuestra que los usuarios iniciaron la grabacion del viaje a pie y posteriormente se subieron a un carro o autobus olvidando cambiar la etiqueta del modo en la aplicacion.
  3. Calibrar las matrices de probabilidad o el ruteo basandose en que el usuario caminaba a estas velocidades introduce un sesgo inaceptable en los modelos de emisiones.

---

## 3. Diagnostico de Calidad a Nivel de Viaje (Trip-Level)

Muchas veces un ping individual no es solo un error del GPS, sino que indica que **todo el viaje fue clasificado de manera incorrecta**. 

Definimos que si un viaje tiene **mas del 30% de sus pings marcados como anomalos**, la clasificacion manual del viaje es erronea en su totalidad y el viaje completo deberia descartarse o reclasificarse.

* **Cantidad Total de Viajes Auditados:** 335 viajes
* **Viajes Incorrectos Detectados (>30% de error):** 10 viajes
* **Porcentaje de Viajes a Eliminar/Reclasificar:** **2.99%** del total de viajes.
* **Cantidad de Pings Involucrados en Viajes Incorrectos:** 3,929 pings (representa el **0.99%** del dataset completo).

### Ejemplos de Viajes Criticos con Clasificacion Erronea:
| Usuario (caid) | ID Viaje | Modo Declarado | Pings Totales | Pings Anomalos | % Anomalos | Recomendacion |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| CHH | 13 | Bus | 1,488 | 1,243 | 83.5% | Eliminar viaje completo |
| CHH | 25 | Bus | 1,268 | 712 | 56.2% | Eliminar viaje completo |
| DEDO | 13 | Caminar | 70 | 32 | 45.7% | Eliminar viaje completo |
| EJH | 2 | Caminar | 84 | 38 | 45.2% | Eliminar viaje completo |
| DEPO | 2 | Caminar | 146 | 62 | 42.5% | Eliminar viaje completo |
| EJH | 1 | Caminar | 62 | 26 | 41.9% | Eliminar viaje completo |
| DEDO | 3 | Caminar | 169 | 70 | 41.4% | Eliminar viaje completo |
| FAA | 2 | Caminar | 187 | 68 | 36.4% | Eliminar viaje completo |

---

## 3.1 Tratamiento Propuesto para los Viajes Peatonales Erroneos

Para resolver el problema del **47% de la distancia de caminar contaminada** con velocidad vehicular, aplicaremos las siguientes tres reglas de limpieza y curacion de datos:

1. **Descarte Completo por Umbral (Viajes Mal Clasificados):**
   * Descartar los viajes completos marcados como `Caminar` que posean mas de un 30% de puntos anomalos (como `DEDO-13` o `EJH-2`). Estos viajes completos representan fallas sistematicas del encuestado al reportar su modo.
2. **Segmentacion y Division de Viajes Mixtos (Trip Splitting):**
   * Si un viaje peatonal inicia a velocidad normal ($<6\text{ km/h}$) pero cambia a velocidades sostenidas de vehiculo ($>20\text{ km/h}$) durante mas de 3 minutos para luego volver a detenerse, el viaje debe dividirse.
   * El tramo vehicular debe ser reetiquetado como `carro` o `bus` mediante clasificacion probabilistica espacial, o eliminarse de forma aislada para salvar los tramos peatonales correctos de inicio y fin.
3. **Filtro de Glitch GPS Aislado (Spike Filtering):**
   * Pings peatonales aislados que superan los 15 km/h durante un solo segundo pero regresan inmediatamente a $<5\text{ km/h}$ representan rebotes de señal. Estos puntos se deben eliminar de forma individual y su posicion debe interpolarse linealmente entre los pings vecinos.

---

## 4. Resumen y Plan de Limpieza de Datos

### Proporcion de Datos Potencialmente Eliminables/Corregibles:
1. **Limpieza a Nivel de Puntos (Glitch GPS):** Podemos remover un **4.81%** de los pings individuales del dataset sin alterar el resto de las secuencias de viaje, eliminando el ruido y picos de teletransportacion.
2. **Limpieza a Nivel de Viajes (Misclassified Trips):** Deberiamos depurar el **2.99%** de los viajes completos que fueron etiquetados con el modo equivocado (como el caso de caminar a velocidades de autopista o el metro transitando lejos de las vias ferroviarias).

### Pasos Futuros para la Depuracion:
1. **Descarte de Gaps y Teleportacion:** Programar un script en esta carpeta para filtrar puntos donde $v > 250\text{ km/h}$.
2. **Criterio de Proximidad Vial:** Para los viajes de Metro, corregir la etiqueta a `carro` o `bus` si el viaje ocurre lejos de las vias, o bien removerlo si no es posible discernir el modo real.
3. **Suavizado de Velocidad Peatonal:** Reclasificar tramos marcados como `caminar` a `bus`/`carro` si la velocidad es consistentemente alta (e.g. $> 15\text{ km/h}$) durante mas de 3 pings seguidos.
