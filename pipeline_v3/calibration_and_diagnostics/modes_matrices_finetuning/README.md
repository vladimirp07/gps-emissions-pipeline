# Módulo de Sintonización Hiperparamétrica del Evaluador Bayesiano (Finetuning)

Este directorio provee los recursos y la especificación técnica para la optimización y calibración de las matrices de probabilidad condicional que gobiernan la clasificación modal de trayectos a través del componente `BayesianRouteEvaluator`. La calibración se fundamenta en técnicas de Optimización Bayesiana implementadas mediante el framework `Optuna`.

---

## 1. Objetivos del Módulo
* Calibrar de manera óptima las probabilidades condicionales asociadas a las variables físicas (`Cercanía`, `Velocidad`, `Distancia` y `Velprom`).
* Maximizar la precisión de clasificación (`Balanced Accuracy` y `F1-Score`) en la inferencia del modo de transporte.
* Validar la robustez del modelo frente a escenarios controlados de degradación espacial y temporal de la señal GPS original.

---

## 2. Arquitectura de Cómputo Desacoplada (Decoupled Architecture)

El proceso de emparejamiento de mapas (map matching) y ruteo a través de la función `complete_route_v1_optimized` introduce una latencia aproximada de 1 a 2 segundos por consulta debido a la complejidad geométrica y topológica de la búsqueda en grafos. Integrar el motor de ruteo directamente en la función objetivo del ciclo de optimización de `Optuna` es inviable en términos de rendimiento computacional, pues implicaría tiempos de ejecución acumulados de decenas de horas.

Para solventar esta restricción de rendimiento, se implementa una arquitectura desacoplada estructurada en dos fases secuenciales:

### Fase A: Pre-procesamiento, Ruteo e Indexación (Extracción de Índices)
Se ejecuta de forma previa y por única vez a través del script [generar_datos_entrenamiento.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py). Las operaciones realizadas son:
1. **Carga de Datos de Referencia:** Lectura del conjunto de datos depurado proveniente de [Datos de MATLAB GPS Limpios.csv](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/GPS%20User%20Data/Datos%20de%20MATLAB%20GPS%20Limpios.csv).
2. **Degradación Controlada:** Simulación y estructuración de los niveles de degradación de señal (Raw, L1, L2, L3).
3. **Inferencia de Hipótesis:** Aplicación del clasificador a priori para determinar hipótesis viables de modo de transporte.
4. **Ruteo y Extracción Geométrica:** Ejecución del ruteador oficial bajo la parametrización de producción (Escenario 8: buffers de snapping de 50m peatonal, 150m vehicular y factor de física por defecto 2.0x).
5. **Indexación y Discretización (Binning):** Los valores físicos resultantes se mapean a sus correspondientes índices enteros de fila (`idx`) basándose en los límites de bins de producción.
6. **Serialización:** Persistencia del conjunto de entrenamiento discretizado en formato binario mediante el archivo pickle [datos_entrenamiento_optuna.pkl](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/GPS%20User%20Data/datos_entrenamiento_optuna.pkl).

### Fase B: Optimización Hiperparamétrica Online (Optuna)
El proceso de optimización consume de forma exclusiva el archivo serializado [datos_entrenamiento_optuna.pkl](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/GPS%20User%20Data/datos_entrenamiento_optuna.pkl).
* Dado que las métricas físicas y espaciales ya han sido discretizadas y convertidas a índices de fila en el caché, la evaluación probabilística del clasificador bayesiano se reduce a un lookup de matriz indexado en NumPy (`Velocidad[idx_v]`) y multiplicaciones básicas.
* Al saltarse el cálculo de ruteo de grafos y la lógica de `np.digitize`, se permite realizar la simulación de hasta 10,000 iteraciones (trials) de Optuna en un intervalo de tiempo inferior a 5 segundos.

---

## 3. Protocolo de Ejecución

### Generación del Dataset Precomputado
La ejecución se inicia desde la raíz del repositorio. Se puede emplear el parámetro `--limit` para procesar una cantidad limitada de viajes de prueba para validación de flujo rápida:

```bash
# Ejecución completa
python pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py

# Ejecución de prueba (ej. procesar solo los primeros 2 viajes)
python pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py --limit 2
```

### Esquema de Datos del Caché (`datos_entrenamiento_optuna.pkl`)
El pickle contiene una lista de diccionarios con el siguiente esquema para cada viaje ruteado:
```python
{
    "trip_id": str,          # Identificador del viaje y su hipótesis
    "label": str,            # Modo de transporte real ('Carro', 'Bus', 'Metro', 'Caminar')
    "idx_c": np.ndarray,      # Array de enteros (shape: N,) con valores de bin en [0, 1, 2]
    "idx_v": np.ndarray,      # Array de enteros (shape: N,) con valores de bin en [0, 1, 2, 3]
    "idx_d_arr": np.ndarray,  # Array de enteros (shape: N,) con valores de bin en [0, 1, 2, 3, 4]
    "idx_vp_arr": np.ndarray  # Array de enteros (shape: N,) con valores de bin en [0, 1]
}
```

### Integración en la Función Objetivo de Optuna
El script de optimización debe cargar las hipótesis precomputadas y evaluar la verosimilitud empleando indexación directa sobre las propuestas de matrices probabilísticas:

```python
import pickle
import numpy as np
from sklearn.metrics import balanced_accuracy_score

# 1. Cargar las hipótesis precomputadas
with open('Inputs/GPS User Data/datos_entrenamiento_optuna.pkl', 'rb') as f:
    data_cache = pickle.load(f)

def objective(trial):
    # Proponer 56 logits libres (ej. de -5.0 a 5.0) para estructurar las matrices
    # ...
    # Aplicar Softmax por fila para obtener las matrices probabilísticas:
    # Cercania (3x4), Velocidad (4x4), Distancia (5x4), Velprom (2x4)
    
    y_true = []
    y_pred = []
    
    for trip in data_cache:
        # Lookup vectorial instantáneo
        P_unnorm = (Cercania[trip['idx_c']] * 
                    Velocidad[trip['idx_v']] * 
                    Distancia[trip['idx_d_arr']] * 
                    Velprom[trip['idx_vp_arr']])
        
        # Normalización por fila (evitando divisiones por cero)
        suma_puntos = P_unnorm.sum(axis=1, keepdims=True)
        suma_puntos[suma_puntos == 0] = 1.0
        P_norm = P_unnorm / suma_puntos
        
        # Votación acumulada del viaje y predicción
        total_votes = P_norm.sum(axis=0)
        prediction = np.argmax(total_votes)
        
        y_pred.append(prediction)
        y_true.append(mode_to_idx(trip['label']))
        
    score = balanced_accuracy_score(y_true, y_pred)
    return 1.0 - score
```

---

## 4. Componentes y Referencias de Archivos
* [generar_datos_entrenamiento.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/calibration_and_diagnostics/modes_matrices_finetuning/generar_datos_entrenamiento.py): Script encargado del procesamiento offline de ruteo, discretización y generación del conjunto de entrenamiento.
* [datos_entrenamiento_optuna.pkl](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/Inputs/GPS%20User%20Data/datos_entrenamiento_optuna.pkl) (Ignorado por Git): Dataset binario serializado indexado que actúa como entrada del proceso de optimización en memoria.
