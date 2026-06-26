# Ajuste y Sintonizacion de Matrices de Bayes (Finetuning)

Esta carpeta contiene los scripts y la documentacion para realizar la optimizacion bayesiana de los parametros del clasificador posterior de modos de transporte (`BayesianRouteEvaluator`) utilizando **Optuna**.

El objetivo es ajustar los valores de las matrices de probabilidad condicional (`Cercania`, `Velocidad`, `Distancia` y `Velprom`) para maximizar la precision del clasificador sobre viajes completos, evaluando tanto los datos crudos como los datos bajo escenarios de degradacion temporal/espacial.

---

## Estrategia de Desacoplamiento de Cómputo (Optuna Veloz)

Correr el algoritmo de map matching y busqueda en grafos (`complete_route_v1_optimized`) es computacionalmente costoso y toma entre 1 y 2 segundos por viaje. Si ejecutaramos el ruteador dentro de cada iteracion (trial) de Optuna, optimizar el clasificador requeriria de decenas de horas.

Para resolver esto de forma limpia y eficiente, implementamos una estrategia de **desacoplamiento**:

1. **Pre-ruteo y Extraccion (Paso Pesado - Una Sola Vez):**
   El script [generar_datos_entrenamiento.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py) se ejecuta una sola vez. Este script:
   * Carga los datos de MATLAB limpios de [Datos de MATLAB GPS Limpios.csv](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/GPS%20User%20Data/Datos%20de%20MATLAB%20GPS%20Limpios.csv).
   * Genera las degradaciones (Raw, L1, L2, L3) para todos los viajes del dataset.
   * Filtra espacialmente (15m) y extrae las hipotesis de modo viables.
   * Ejecuta el ruteador con la configuracion optima calibrada (Scenario 8: buffers de 50m peatonal, 150m vehicular y factor de fisica 2.0x).
   * Mide la cercania espacial de la ruta reconstruida respecto a la infraestructura de transporte.
   * Extrae y guarda todas las variables fisicas evaluadas por el clasificador de Bayes (distancia ruteada, velocidades instantaneas del recorrido y arreglos de cercania) en un archivo binario ligero [datos_entrenamiento_optuna.pkl](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/GPS%20User%20Data/datos_entrenamiento_optuna.pkl).

2. **Optimizacion Matematica (Paso Rapido - Optuna):**
   El archivo binario resultante contiene un DataFrame de Pandas donde cada registro ya posee el ruteo resuelto y los arreglos de entrada listos. El desarrollador de Optuna simplemente carga este archivo. Dado que no hay calculo de Dijkstra en el bucle, la evaluacion Bayesiana de las matrices se reduce a multiplicaciones matriciales directas en NumPy. **Esto permite correr 10,000 trials de Optuna en menos de 5 segundos.**

---

## Instrucciones de Uso

### Paso 1: Generar los Datos de Entrenamiento Ruteados
Ejecuta el script de extraccion en la terminal para crear el archivo pickle con todos los viajes y sus hipotesis ruteadas bajo las 4 degradaciones de datos:

```bash
python pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py
```

### Paso 2: Cargar y Optimizar con Optuna
En el notebook o script de Optuna, tu companero puede cargar los datos y definir el bucle de optimizacion directamente de la siguiente manera:

```python
import pickle
import pandas as pd
import numpy as np

# 1. Cargar las hipotesis pre-ruteadas
with open('Inputs/GPS User Data/datos_entrenamiento_optuna.pkl', 'rb') as f:
    df_train = pickle.load(f)

# 2. Definir la funcion objetivo de Optuna
def objective(trial):
    # Proponer nuevos valores para las matrices (ej. bins o probabilidades)
    # Ejemplo: proponer bins para la velocidad
    bin_v1 = trial.suggest_float('bin_v1', 4.0, 8.0)
    bin_v2 = trial.suggest_float('bin_v2', 15.0, 25.0)
    
    # Crear matrices experimentales a partir de las propuestas
    # ...
    
    # Evaluar la precision de clasificacion de forma puramente vectorial
    # utilizando los datos cargados en 'df_train' (distancia, velocidades, cercanias)
    # sin llamar al ruteador
    
    # Retornar el accuracy o f1-score general
    return accuracy
```

---

## Archivos en esta Carpeta

* [generar_datos_entrenamiento.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py): Script de ruteo sistematico de hipotesis y empaquetamiento de muestras.
* `README.md`: Documentacion tecnica e instrucciones de integracion con Optuna.
