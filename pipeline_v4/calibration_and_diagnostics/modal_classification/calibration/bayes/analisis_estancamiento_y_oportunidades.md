# Análisis de Estancamiento y Oportunidades: Calibración con Optuna

Este documento presenta una evaluación honesta, matemática y de arquitectura sobre el estado actual de la calibración del clasificador bayesiano utilizando **Optuna**, explicando por qué los resultados históricos del 100% eran en realidad un espejismo técnico, las limitaciones estructurales del modelo actual y propuestas "fuera de la caja" para superar la barrera del 90% en todas las categorías.

---

## 1. El Mito del "100% de Precisión" en Versiones Anteriores (Espejismo Técnico)

Históricamente, se reportó que el clasificador bayesiano lograba un **100% de precisión en tres categorías y un rendimiento muy alto en la restante**. Es fundamental entender por qué esto ocurría y por qué en el entorno de pruebas actual y realista la precisión se ha sincerado:

1.  **Fuga de Información por Ping (Ping-Level Data Leakage):**
    *   En las pruebas anteriores de Alejandro, la división entre el conjunto de entrenamiento y el de validación se realizaba **a nivel de puntos individuales (pings)** de GPS.
    *   Dado que un viaje está compuesto por pings consecutivos en el tiempo (que comparten casi idéntica velocidad, cercanía e infraestructura), mezclar aleatoriamente pings del viaje 1 en entrenamiento y pings del mismo viaje 1 en validación provoca que el modelo evalúe puntos de los cuales ya conoce su contexto inmediato. Esto infla artificialmente la precisión al 100%.
    *   **En la versión actual:** Hemos corregido esto en el orquestador y en el playground aplicando un **Trip-Level Split**. Los viajes de un usuario en validación son completamente invisibles para el entrenamiento.
2.  **Omisión de la Degradación de Señal:**
    *   Anteriormente solo se evaluaban trayectorias ideales y continuas (Raw 1Hz). 
    *   Ahora, para garantizar robustez en producción ante la pérdida de señal real de Veraset, el dataset de entrenamiento incluye **niveles de degradación de señal (L1, L2 y L3)** con gaps y muestreos de hasta 6 minutos. Clasificar correctamente un viaje con solo 5 pings espaciados en el tiempo es físicamente más complejo y reduce los scores perfectos.
3.  **El Bug del Tiempo Cero (`dt_sec = 0.0`):**
    *   Las velocidades absurdas causadas por duplicidad no se limpiaban correctamente y se les asignaba $0.0\text{ km/h}$. Al no existir validación secuencial ni recorte de extremos estáticos, el dataset de entrenamiento estaba lleno de ruido que el clasificador "memorizaba".

---

## 2. Limitaciones Estructurales del Clasificador Bayesiano (Por qué estamos estancados)

Aunque Optuna busque incansablemente el óptimo matemático, el clasificador [BayesianRouteEvaluator](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v4/src/bayes_classifier.py#L89) actual tiene limitantes matemáticas insalvables:

1.  **Suposición "Naive" de Independencia de Variables:**
    *   El modelo asume que las variables `Velocidad` (velocidad instantánea ping a ping) y `Velprom` (velocidad promedio del viaje) son estadísticamente independientes.
    *   **La realidad:** Ambas variables están sumamente correlacionadas. Multiplicar las probabilidades individuales de variables correlacionadas en la regla de Bayes provoca un efecto de **"double counting"**, sesgando e inflando artificialmente las probabilidades de ciertos modos (como Carro sobre Autobús).
2.  **Baja Capacidad del Modelo (Solo 56 Parámetros):**
    *   Las matrices son tablas de búsqueda fija de bins:
        *   Cercanía: $3 \times 4$
        *   Velocidad: $4 \times 4$
        *   Distancia: $5 \times 4$
        *   Velprom: $2 \times 4$
    *   Esto da un total de **56 probabilidades libres** (o logits) a optimizar por Optuna. Un modelo lineal simple tiene más capacidad de representación. Tratar de separar dinámicas complejas (como diferenciar un auto en tráfico pesado de un camión o autobús) usando solo 56 logits es matemáticamente inviable si el dataset contiene ruido realista.
3.  **El Dilema de los Semáforos y Paradas en Avenidas:**
    *   Aunque el entrenamiento excluya las paradas largas de 5 minutos, los vehículos siguen deteniéndose en semáforos o intersecciones (velocidad = $0.0\text{ km/h}$, bin 0 de velocidad).
    *   Al evaluar ping a ping, estos pings de velocidad cero votan masivamente por `Caminar`, restando votos al modo vehicular real (`Carro` o `Bus`) en viajes cortos con semáforos frecuentes.
4.  **Traslape de Cercanías:**
    *   Un automóvil circulando por una avenida paralela a la línea del metro o sobre una ruta de autobús activa pings con cercanía positiva a infraestructura (`idx_c = 0` o `idx_c = 1`). Bayes no sabe de geometría fina, solo de bins, lo que confunde constantemente las predicciones entre `Carro`, `Bus` y `Metro`.

---

## 3. Propuestas Fuera de la Caja (Out-of-the-Box) para Alcanzar > 90%

Para romper el techo de cristal de Optuna y lograr el rendimiento deseado, es necesario discutir estas propuestas de re-diseño del clasificador:

### Propuesta A: Reemplazar Bayes por un Modelo de Machine Learning Ligero
*   **La Idea:** Mantener la arquitectura desacoplada (extracción de variables offline), pero en lugar de alimentar la regla de Bayes con las 4 matrices, entrenar un modelo **Random Forest** o **LightGBM** ligero sobre las mismas variables indexadas del caché.
*   **Por qué funciona:** Los árboles de decisión manejan de forma nativa la no-linealidad, las interacciones complejas de variables (ej. *si está cerca del metro Y la velocidad es >50 km/h, entonces es Metro*) y no asumen independencia de variables.
*   **Implementación:** El modelo en memoria se puede guardar como un objeto serializado pequeño y la predicción tarda microsegundos, manteniendo la alta velocidad que logramos con Optuna.

### Propuesta B: Incorporar la "Distancia de Snapping" al Grafo como Variable
*   **La Idea:** Agregar una nueva variable condicional a la clasificación: **la distancia geodésica entre el punto GPS original y el arco vial al que se snappeó**.
*   **Por qué funciona:** Los automóviles y autobuses viajan estrictamente sobre las calles (snapping distance $< 5\text{ metros}$ en promedio). Los peatones caminan sobre banquetas, plazas o dentro de edificios, registrando offsets de snapping mucho mayores ($15\text{ a }50\text{ metros}$). Esta es la variable más potente para separar de tajo `Caminar` de los modos motorizados, y actualmente no la estamos usando en el clasificador.

### Propuesta C: Modelo Secuencial (Cadenas de Markov / HMM)
*   **La Idea:** Dejar de tratar cada ping de forma aislada para la votación acumulada. Implementar una matriz de transición de estados donde la probabilidad de pasar de `Carro` a `Caminar` en un segundo sea bajísima, a menos que se detecte una parada de transferencia.
*   **Por qué funciona:** Si un auto se detiene en un semáforo (velocidad $0\text{ km/h}$), el modelo secuencial mantendrá la hipótesis de `Carro` porque la probabilidad de transición a `Caminar` a mitad de una avenida es casi nula. Esto elimina el ruido que causan las detenciones cortas en la votación global.

### Propuesta D: Suavizado Temporal de Predicciones y Ventanas Móviles
*   **La Idea:** Aplicar un filtro de promedio móvil o votación por mayoría en ventanas temporales de pings antes de calcular el voto del viaje.
*   **Por qué funciona:** Elimina picos transitorios y glitches de velocidad que logran evadir los filtros de depuración.

