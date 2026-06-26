# Ajuste y Sintonizacion de Matrices de Bayes (Finetuning)

Esta carpeta contiene la documentacion y los scripts de automatizacion para la optimizacion de las matrices de probabilidad condicional del evaluador de hipotesis del clasificador posterior de modos de transporte (`BayesianRouteEvaluator`) mediante algoritmos de optimizacion bayesiana (Optuna).

El objetivo principal es calibrar las matrices de probabilidad condicional (`Cercania`, `Velocidad`, `Distancia` y `Velprom`) para maximizar la precision del clasificador sobre viajes completos, evaluando tanto los datos crudos como los datos bajo escenarios de degradacion temporal y espacial.

---

## Arquitectura de Computo Desacoplada (Optimizacion Eficiente)

La ejecucion del algoritmo de map matching y busqueda en grafos (`complete_route_v1_optimized`) es computacionalmente costosa, requiriendo entre 1 y 2 segundos por consulta. Ejecutar el ruteador dentro de la funcion objetivo del bucle de Optuna resultaria en tiempos de ejecucion del orden de decenas de horas.

Para mitigar esto, se implementa una arquitectura de ejecucion desacoplada:

1. **Fase de Extraccion y Pre-ruteo (Procesamiento Offline):**
   El script [generar_datos_entrenamiento.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py) se ejecuta de manera previa y por unica vez. Este proceso:
   * Carga el dataset depurado de [Datos de MATLAB GPS Limpios.csv](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/GPS%20User%20Data/Datos%20de%20MATLAB%20GPS%20Limpios.csv).
   * Genera las degradaciones temporales y espaciales correspondientes (Raw, L1, L2, L3) para toda la muestra.
   * Aplica la resolucion espacial minima (15m) y extrae las hipotesis de modo viables mediante el prior clasificador.
   * Resuelve el ruteo de cada hipotesis factible utilizando la parametrizacion optima calibrada (Escenario 8: buffers de snapping de 50m peatonal, 150m vehicular y factor de fisica 2.0x).
   * Mide la cercania a la infraestructura y extrae los vectores de variables fisicas del trayecto completado.
   * Serializa esta informacion en el archivo pickle [datos_entrenamiento_optuna.pkl](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/GPS%20User%20Data/datos_entrenamiento_optuna.pkl).

2. **Fase de Optimizacion Matematica (Bucle Online - Optuna):**
   El archivo serializado resultante se carga en el script de optimizacion. Debido a que las distancias de ruteo, las velocidades y las variables de cercania ya estan precomputadas para cada hipotesis, la evaluacion Bayesiana se reduce a multiplicaciones matriciales basicas utilizando NumPy. Esta simplificacion permite evaluar hasta 10,000 combinaciones (trials) de Optuna en un tiempo inferior a 5 segundos.

---

## Guia de Ejecucion y Consumo

### Paso 1: Generacion del Dataset de Entrenamiento Ruteado
Para generar el archivo pickle con las muestras de entrenamiento ruteadas, ejecute el script de extraccion desde la raiz del proyecto:

```bash
python pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py
```

### Paso 2: Consumo en la Funcion Objetivo de Optuna
El script de optimizacion debe cargar los datos serializados e implementar la evaluacion bayesiana dentro del bucle de la siguiente manera:

```python
import pickle
import pandas as pd
import numpy as np

# 1. Cargar las hipotesis de ruteo precomputadas
with open('Inputs/GPS User Data/datos_entrenamiento_optuna.pkl', 'rb') as f:
    df_train = pickle.load(f)

# 2. Estructurar la funcion objetivo para la busqueda hiperparametrica
def objective(trial):
    # Proponer limites (bins) para las variables fisicas
    bin_v1 = trial.suggest_float('bin_v1', 4.0, 8.0)
    bin_v2 = trial.suggest_float('bin_v2', 15.0, 25.0)
    
    # Configurar las matrices de probabilidad condicional experimentales
    # ...
    
    # Calcular las probabilidades posteriores de forma puramente vectorial
    # utilizando los arreglos de velocidad y cercania preexistentes en df_train
    
    # Retornar la metrica de desempeno del clasificador (Accuracy, F1-Score)
    return accuracy
```

---

## Recursos en el Directorio

* [generar_datos_entrenamiento.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py): Script para la generacion del dataset precomputado.
* `README.md`: Documentacion de integracion con el optimizador.
