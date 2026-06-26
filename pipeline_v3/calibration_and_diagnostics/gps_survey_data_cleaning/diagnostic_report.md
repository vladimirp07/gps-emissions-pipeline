# Propuesta de Limpieza y Diagnostico de Calidad de Datos del Dataset de MATLAB

Este reporte documenta la propuesta de limpieza y los resultados del diagnostico de calidad fisica y espacial realizado sobre los datos de encuesta de MATLAB (`Datos de MATLAB GPS.csv`), etiquetados de forma manual por los participantes. 

El objetivo es establecer y justificar una metodologia de depuracion para eliminar glitches de GPS y viajes mal clasificados, garantizando la compatibilidad del formato original y eliminando el sesgo que introducen los datos erroneos en la calibracion de matrices de transicion y modelos de emisiones.

---

## 1. Criterios de Calidad de Datos Establecidos

Para la depuracion de los datos de MATLAB, aplicamos dos niveles de limpieza: **Limpieza a Nivel de Puntos (Glitch GPS)** y **Limpieza a Nivel de Viajes (Misclassified Trips)**. Definimos los siguientes criterios especificos:

* **Inconsistencia de Velocidad por Modo (Limites Fisicos):**
  * **Caminar:** Velocidades instantaneas superiores a **30 km/h**.
    * *Nota de Calibracion:* Aunque 30 km/h es una velocidad vehicular, se establece este umbral para caminar porque el calculo de velocidad mediante distancia Haversine directa punto a punto sobreestima la velocidad real debido al ruido de zig-zag o jitter intrinseco del GPS de alta frecuencia. En un desarrollo posterior del pipeline, se debera implementar un preprocesamiento de suavizado (como Filtro de Kalman o medias moviles) para afinar esta velocidad, pero por ahora los 30 km/h actuan como umbral vehicular definitivo.
  * **Metro / Autobus (Bus):** Velocidades instantaneas superiores a **110 km/h** (limite maximo operativo en la zona urbana).
  * **Automovil (Carro):** Velocidades instantaneas superiores a **160 km/h** (limite razonable de transito en autopista con margen de error de GPS).
* **Inconsistencia Espacial de Infraestructura (Metro fuera de vias):**
  * Pings clasificados como `metro` que se ubican a mas de **300 metros** de distancia de la linea geometrica oficial de Metrorrey (`lineas_metrorrey.csv`).
* **Teletransportacion General (GPS Glitch):**
  * Saltos de posicion individuales que requieran velocidades instantaneas superiores a **250 km/h** en cualquier modo de transporte.
* **Exclusion de Filtro para Paradas:**
  * No se aplica ningun filtro estatico de velocidad o distancia sobre el modo "parada" o paradas fisicas en este dataset, ya que la maquina de estados de nuestro algoritmo de ruteo y clasificacion se encargara de identificar y tratar las paradas de forma dinamica.

---

## 2. Analisis Fisico por Modo de Transporte (Datos Originales)

A continuacion se presenta la proporcion de pings anomalos detectados originalmente en el dataset de MATLAB usando los criterios basicos de velocidad y espacio:

| Modo de Transporte | Pings Totales | Pings Anomalos | % Pings Anomalos | % Distancia Afectada | % Tiempo Afectado | Detalles de la Anomalia |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Caminar** | 184,860 | 14,802 | 8.01% | 44.77% | 0.95% | 8.0% exceden vel. max (>30 km/h), 0.0% glitches GPS |
| **Carro** | 167,230 | 449 | 0.27% | 12.82% | 0.05% | 0.3% exceden vel. max, 0.3% glitches GPS |
| **Bus** | 34,705 | 2,631 | 7.58% | 98.55% | 0.92% | 7.6% exceden vel. max, 7.6% glitches GPS |
| **Metro** | 8,674 | 39 | 0.45% | 7.12% | 0.03% | 0.4% exceden vel. max, 0.4% fuera de vias (>300m) |

---

## 2.1 Distribucion de Velocidades en el Modo Caminar

Dado el alto impacto espacial en los datos de Caminar, la distribucion acumulada de velocidades instantaneas sobre los **184,860 pings peatonales** revela lo siguiente:

| Umbral de Velocidad | Pings que lo superan | % Pings Peatonales | Distancia Acumulada (km) | % Distancia de Caminar | Tipo de Incongruencia / Diagnostico |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **> 6.0 km/h** | 20,911 | 11.31% | 269.75 km | 52.24% | Trote o carrera ligera |
| **> 10.0 km/h** | 16,803 | 9.09% | 248.77 km | 48.18% | Velocidad de carrera continua o vehiculo |
| **> 15.0 km/h** | 16,039 | 8.68% | 244.40 km | 47.33% | Sprint humano limite o vehiculo |
| **> 25.0 km/h** | 15,598 | 8.44% | 238.28 km | 46.15% | Transito vehicular urbano indudable |
| **> 30.0 km/h (Umbral)**| 14,802 | 8.01% | 231.15 km | 44.77% | **Vehiculo motorizado indudable (Carro/Bus)** |
| **> 50.0 km/h** | 2,015 | 1.09% | 90.98 km | 17.62% | Transito vehicular en avenidas/autopistas |

* **Conclusion clave:** El **44.77% de la distancia total** declarada bajo la etiqueta de Caminar ocurrio a velocidades superiores a 30 km/h (y el 46.15% a mas de 25 km/h). Esto confirma que casi la mitad de los datos peatonales corresponden en realidad a trayectos vehiculares (el participante abordo un autobus o automovil pero olvido apagar la grabacion o cambiar la etiqueta manual del viaje).

---

## 3. Estrategia de Depuracion Aplicada

Para limpiar el dataset sin introducir suposiciones de reetiquetado (lo que podria sesgar el modelo), se aplica una estrategia de **poda y eliminacion**:

### 3.1 Limpieza a Nivel de Viajes (Misclassified Trips)
* **Descarte por Fraccion de Error:** Si un viaje (agrupado por `caid` y `num_trip`) presenta **mas del 30% de sus pings originales como anomalos** (por excesos de velocidad en carro/bus/metro o distancia del metro a las vias), se asume que la clasificacion manual de todo el viaje fue incorrecta y se **elimina el viaje completo**.
* **Descarte de Viajes Vacios o Minimos:** Si tras aplicar la limpieza de puntos un viaje queda con menos de **2 pings**, se elimina por completo para evitar errores de ruteo.

### 3.2 Limpieza a Nivel de Puntos y Poda de Viajes Peatonales (Walking Trip Pruning)
* **Poda de Caminatas Vehiculares:** Para los viajes de Caminar que no superan el 30% de error total, pero que en algun punto muestran velocidades de vehiculo (>30 km/h):
  * Se identifica cronologicamente el **primer punto** del viaje donde la velocidad es superior a **30 km/h**.
  * Se realiza una **poda** (truncado) en ese instante exacto: se **conserva la primera parte** del viaje (trayecto peatonal logico) y se **elimina ese punto y todos los subsecuentes** (segmento donde el participante abordo un vehiculo).
  * Si la porcion valida restante tiene menos de 2 pings, el viaje se descarta por completo.
* **Filtro de Glitch GPS Aislado:**
  * En viajes validos de cualquier modo (incluyendo la porcion valida de las caminatas), los pings individuales marcados como anomalos (por ejemplo, glitches aislados de teletransportacion >250 km/h) son eliminados individualmente.

---

## 4. Implementacion del Codigo de Depuracion

El proceso ha sido automatizado en el script [depurar_datos_matlab.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/gps_survey_data_cleaning/depurar_datos_matlab.py). 

Este codigo:
1. Lee los datos originales en `Inputs/GPS User Data/Datos de MATLAB GPS.csv`.
2. Procesa cronologicamente cada viaje para detectar anomalias puntuales y de trayecto.
3. Aplica los descartes de viajes y la poda de trayectos peatonales vehicularizados.
4. Elimina las columnas auxiliares creadas para el procesamiento.
5. Exporta el archivo depurado a `Inputs/GPS User Data/Datos de MATLAB GPS Limpios.csv` en el **mismo formato exacto** (mismas columnas, tipos y orden) para asegurar compatibilidad total con el pipeline principal.
