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
| CHH | 13 | Bus | 1488 | 1243 | 83.5% | Eliminar viaje completo |
| CHH | 25 | Bus | 1268 | 712 | 56.2% | Eliminar viaje completo |
| DEDO | 13 | Caminar | 70 | 32 | 45.7% | Eliminar viaje completo |
| EJH | 2 | Caminar | 84 | 38 | 45.2% | Eliminar viaje completo |
| DEPO | 2 | Caminar | 146 | 62 | 42.5% | Eliminar viaje completo |
| EJH | 1 | Caminar | 62 | 26 | 41.9% | Eliminar viaje completo |
| DEDO | 3 | Caminar | 169 | 70 | 41.4% | Eliminar viaje completo |
| FAA | 2 | Caminar | 187 | 68 | 36.4% | Eliminar viaje completo |

---

## 4. Resumen y Plan de Limpieza de Datos

### Proporcion de Datos Potencialmente Eliminables/Corregibles:
1. **Limpieza a Nivel de Puntos (Glitch GPS):** Podemos remover un **4.81%** de los pings individuales del dataset sin alterar el resto de las secuencias de viaje, eliminando el ruido y picos de teletransportacion.
2. **Limpieza a Nivel de Viajes (Misclassified Trips):** Deberiamos depurar el **2.99%** de los viajes completos que fueron etiquetados con el modo equivocado (como el caso critico del usuario `GAR` en metro que se encuentra a kilometros de las vias reales, o velocidades de peatones imposibles).

### Pasos Futuros para la Depuracion:
1. **Descarte de Gaps y Teleportacion:** Programar un script en esta carpeta para filtrar puntos donde $v > 250\text{ km/h}$.
2. **Criterio de Proximidad Vial:** Para los viajes de Metro, corregir la etiqueta a `carro` o `bus` si el viaje ocurre lejos de las vias, o bien removerlo si no es posible discernir el modo real.
3. **Suavizado de Velocidad Peatonal:** Reclasificar tramos marcados como `caminar` a `bus`/`carro` si la velocidad es consistentemente alta (e.g. $> 15\text{ km/h}$) durante mas de 3 pings seguidos.
