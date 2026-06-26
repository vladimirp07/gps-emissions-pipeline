# Módulo de Sintonización Hiperparamétrica del Evaluador Bayesiano (Finetuning)

Este directorio provee los recursos y la especificación técnica para la optimización y calibración de las matrices de probabilidad condicional que gobiernan la clasificación modal de trayectos a través del componente `BayesianRouteEvaluator`. La calibración se fundamenta en técnicas de Optimización Bayesiana implementadas mediante el framework `Optuna`.

---

## 1. Objetivos del Módulo
* Calibrar de manera óptima las variables físicas multimodales (`Cercanía`, `Velocidad`, `Distancia` y `Velprom`).
* Maximizar la precisión de clasificación (`Accuracy` y `F1-Score`) en la inferencia del modo de transporte.
* Validar la robustez del modelo frente a escenarios controlados de degradación espacial y temporal de la señal GPS original.

---

## 2. Arquitectura de Cómputo Desacoplada (Decoupled Architecture)

El proceso de emparejamiento de mapas (map matching) y ruteo a través de la función `complete_route_v1_optimized` introduce una latencia aproximada de 1 a 2 segundos por consulta debido a la complejidad geométrica y topológica de la búsqueda en grafos. Integrar el motor de ruteo directamente en la función objetivo del ciclo de optimización de `Optuna` es inviable en términos de rendimiento computacional, pues implicaría tiempos de ejecución acumulados de decenas de horas.

Para solventar esta restricción de rendimiento, se implementa una arquitectura desacoplada estructurada en dos fases secuenciales:

### Fase A: Pre-procesamiento y Ruteo Offline (Extracción de Features)
Se ejecuta de forma previa y por única vez a través del script [generar_datos_entrenamiento.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py). Las operaciones realizadas son:
1. **Carga de Datos de Referencia:** Lectura del conjunto de datos depurado proveniente de [Datos de MATLAB GPS Limpios.csv](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/GPS%20User%20Data/Datos%20de%20MATLAB%20GPS%20Limpios.csv).
2. **Degradación Controlada:** Simulación y estructuración de los niveles de degradación de señal (Raw, L1, L2, L3).
3. **Inferencia de Hipótesis:** Aplicación del clasificador a priori para determinar hipótesis viables de modo de transporte.
4. **Ruteo y Extracción Geométrica:** Ejecución de ruteos aplicando los parámetros validados del Escenario 8 (`SPATIAL_FILTER_M=15.0`, `WALK_BUFFER_M=50.0`, `DRIVE_BUFFER_M=150.0`, y factor físico `PHYSICS_FACTOR=2.0`).
5. **Serialización:** Persistencia de las distancias calculadas, velocidades y perfiles de proximidad a infraestructura en formato binario mediante el archivo pickle [datos_entrenamiento_optuna.pkl](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/GPS%20User%20Data/datos_entrenamiento_optuna.pkl).

### Fase B: Optimización Hiperparamétrica Online (Optuna)
El proceso de optimización consume de forma exclusiva el archivo serializado [datos_entrenamiento_optuna.pkl](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/GPS%20User%20Data/datos_entrenamiento_optuna.pkl).
* Dado que las métricas físicas y espaciales ya han sido calculadas e indexadas en la estructura de datos, la evaluación probabilística del clasificador bayesiano se reduce a operaciones aritméticas vectorizadas implementadas en `NumPy`.
* Esta optimización algorítmica permite realizar la simulación de hasta 10,000 iteraciones (trials) de Optuna en un intervalo de tiempo inferior a 5 segundos.

---

## 3. Protocolo de Ejecución

### Generación del Dataset Precomputado
La ejecución se inicia desde la raíz del repositorio empleando la siguiente instrucción en la interfaz de comandos:

```bash
python pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py
```

### Integración en la Función Objetivo de Optuna
El script encargado de la búsqueda hiperparamétrica debe implementar la lectura de los datos estructurados y realizar el cálculo de la verosimilitud de la siguiente forma:

```python
import pickle
import pandas as pd
import numpy as np

# Carga de las hipótesis precomputadas
with open('Inputs/GPS User Data/datos_entrenamiento_optuna.pkl', 'rb') as f:
    df_train = pickle.load(f)

def objective(trial):
    # Proposición de umbrales para las variables físicas del clasificador
    bin_v1 = trial.suggest_float('bin_v1', 4.0, 8.0)
    bin_v2 = trial.suggest_float('bin_v2', 15.0, 25.0)
    
    # Configuración e instanciación de las matrices de probabilidad de prueba
    # ...
    
    # Cálculo vectorizado de las probabilidades a posteriori
    # ...
    
    # Retorno de la métrica de desempeño de interés (e.g., F1-Score general)
    return f1_score
```

---

## 4. Componentes y Referencias de Archivos
* [generar_datos_entrenamiento.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py): Script encargado del procesamiento offline de ruteo y generación del conjunto de entrenamiento.
* [datos_entrenamiento_optuna.pkl](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/GPS%20User%20Data/datos_entrenamiento_optuna.pkl) (Ignorado por Git): Dataset binario serializado que actúa como entrada del proceso de optimización.
