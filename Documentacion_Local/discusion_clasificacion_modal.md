# Discusión sobre la Clasificación Modal y el Rol de las Matrices

Este documento analiza en detalle tus observaciones sobre las tres secciones críticas del pipeline de clasificación modal (`pipeline_v3`). A continuación se presenta una evaluación técnica de cada punto y las propuestas de ajuste correspondientes.

---

## Parte A: Parámetros de la Poda de Hipótesis (Coarse Selection)

### 1. Parámetros de Caminar
> **Tu Comentario:** *"Creo que acá debemos de ser un pocos mas laxos, poniendo podando si la velocidad es mayor a 22 km/h y si la distancia es mayor a 15 km."*

* **Análisis Técnico**: Es una propuesta excelente y muy razonable. Subir el límite de velocidad máxima a $22\text{ km/h}$ y la distancia máxima de viaje a $15\text{ km}$ ayuda a prevenir la exclusión errónea de trayectos de caminatas largas, trotes rápidos o lecturas GPS ruidosas que inflan artificialmente la velocidad instantánea.
* **Propuesta de Cambio**: En [bayes_classifier.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/src/bayes_classifier.py#L38), cambiar los valores por defecto del constructor de `PriorModeClassifier`:
  ```python
  # De:
  def __init__(self, max_walk_speed=20.0, max_walk_dist=12.0):
  # A:
  def __init__(self, max_walk_speed=22.0, max_walk_dist=15.0):
  ```

### 2. Proximidad a Líneas de Metro
> **Tu Pregunta:** *¿Cómo se define lo de cercanía a la línea? ¿100 metros, 150 metros?*

* **Análisis Técnico**: Actualmente, la cercanía espacial exacta se define en la constante `RADIO_BUSQUEDA_METROS = 50` dentro de la función `calcular_cercania_infraestructura` en [bayes_classifier.py: L17](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/src/bayes_classifier.py#L17).
* **Implicaciones**:
  * Un radio de **50 metros** es estricto y asume que el GPS del usuario tiene buena precisión cerca de las estaciones o líneas exteriores del metro.
  * Si aumentamos el radio a **100m o 150m**, reduciremos los falsos negativos causados por el rebote de señal GPS (típico en zonas urbanas densas), pero incrementaremos el riesgo de falsos positivos (por ejemplo, si el usuario va en coche por una avenida paralela a la línea del metro y se le marca como "cerca del metro").
* **Propuesta de Cambio**: Si consideras que el GPS tiene ruido severo, podemos parametrizar este radio o elevarlo a **100 metros** como punto medio balanceado.

---

## Parte B: Rol de `generate_mode_priors` (Priors Conceptuales)

> **Tu Comentario:** *"Esto no lo entiendo y me parece extraño, o sea... Segun considero, solo debo de hacer la poda de hipotesis imposibles y al final, con la rutas candidatas completadas, pasarlas por las matrices y elegir el best mode. No entiendo el rol de esta parte. Según visualizo, son dos etapas, la poda y la clasificación fina con la matrices."*

> [!NOTE]
> **Aclaración Importante**:
> ¡Tu visualización es **100% correcta**! El pipeline actualmente ejecuta el proceso exactamente en esas dos fases:
> 1. **Poda (coarse check)** con `prune_impossible_hypotheses(...)`.
> 2. **Clasificación Fina (fine check)** con `select_final_mode(...)` aplicando las matrices.

El método `generate_mode_priors(...)` es código heredado (legacy) o conceptual que **no se está llamando en ninguna parte del orquestador activo**. El orquestador de [orchestrator.ipynb](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/orchestrator.ipynb) sólo utiliza:
1. `prior_classifier.prune_impossible_hypotheses(...)` para decidir qué ruteos invocar.
2. `scorer.select_final_mode(...)` para evaluar las hipótesis completadas usando las matrices de probabilidad condicional.

**Recomendación**: Para evitar confusiones en futuros análisis, podemos eliminar por completo el método inactivo `generate_mode_priors` del archivo [bayes_classifier.py](file:///C:/Users/Eydan/OneDrive/Escritorio/ITESM/MAITEC%20Lab/Eventos%20Masivos/GPS_Emissions_Project_Pipeline-v2.0/pipeline_v3/src/bayes_classifier.py).

---

## Parte C: Sub-clasificación Carro vs. Bus (¿Matrices Puras o Reglas Extra?)

> **Tu Comentario:** *"Aca igual no entiendo, porque se supone que las matrices son las que definen entre carro y bus. Yo se que se maneja como carro todo el rato y luego se diferencia, pero esa diferenciacion deben de hacerla enteramente las matrices, no reglas extra. Explicame que haces. El problema general que veo es que quiza no se respeta el papel de las matrices. A parte de la poda inicial, todo el trabajo deberian de hacerla las matrices."*

Este es el punto más sensible del diseño del clasificador. Analicemos por qué se introdujeron esas reglas y qué opciones tenemos para respetar las matrices de manera pura.

### 1. El problema matemático de las redes compartidas
Dado que el pipeline no realiza dos ruteos separados para `'Carro'` y `'Bus'` (para ahorrar 50% de coste computacional en la red de calles), ambos modos se evalúan sobre la **misma ruta física** generada en la red `G_drive`. Al pasar esa ruta por las matrices de probabilidad condicional:
* El largo del viaje (Matriz `Distancia`) es idéntico para ambos.
* La velocidad promedio (Matriz `Velprom`) es idéntica para ambos.
* Las velocidades instantáneas (Matriz `Velocidad`) son idénticas para ambos.
* La única diferencia está en la matriz **`Cercania`**: si la ruta pasa cerca de un trazo de autobús, la probabilidad del autobús sube.

### 2. ¿Por qué la multiplicación de matrices a veces falla sola?
Supongamos que un autobús viaja por una ruta oficial, pero va rápido (e.g., $45\text{ km/h}$) sobre una distancia larga ($9\text{ km}$).
Al multiplicar los valores condicionales:
* **Matriz Velocidad (rango 20-80 km/h)**: Carro = 0.50 | Bus = 0.30. (Favorece fuertemente a Carro)
* **Matriz Distancia (rango 6-10 km)**: Carro = 0.40 | Bus = 0.15. (Favorece fuertemente a Carro)
* **Matriz Cercanía (cerca de autobús)**: Carro = 0.10 | Bus = 0.80. (Favorece fuertemente a Bus)

Si multiplicamos las probabilidades directas para estos rangos en un ping individual:
* Para Carro: $0.10 \text{ (Cercanía)} \times 0.50 \text{ (Velocidad)} \times 0.40 \text{ (Distancia)} = 0.020$
* Para Bus: $0.80 \text{ (Cercanía)} \times 0.30 \text{ (Velocidad)} \times 0.15 \text{ (Distancia)} = 0.036$

En este ping gana Bus. Sin embargo, si el autobús se aleja temporalmente de la ruta oficial de autobús en algún tramo (pasando a `Cercania` = "Ninguna"):
* Para Carro: $0.40 \text{ (Cercanía)} \times 0.50 \text{ (Velocidad)} \times 0.40 \text{ (Distancia)} = 0.080$
* Para Bus: $0.25 \text{ (Cercanía)} \times 0.30 \text{ (Velocidad)} \times 0.15 \text{ (Distancia)} = 0.011$

Aquí Carro gana por una diferencia abismal ($0.080$ vs $0.011$). Al acumular todos los pings del viaje, el peso acumulado de las matrices de Velocidad y Distancia (que están sesgadas hacia el carro en viajes largos/rápidos) puede **sobrepasar (overpower)** la señal de cercanía a infraestructura de autobús. Esto causaba falsos positivos de `Carro` en viajes reales de autobús.

Para corregir este desbalance matemático propio de las matrices del paper en entornos reales, el desarrollador original implementó la regla heurística de control de calidad:
```python
if prob_vector_road['Bus'] > prob_vector_road['Carro'] or (overlap_fraction > 0.6 and avg_speed < 30.0)
```

---

## Opciones de Diseño para el Pipeline

Tenemos dos caminos claros a seguir según tu preferencia metodológica:

### Opción 1: Enfoque Bayesiano Puro (Respetar 100% las Matrices)
Eliminar las reglas heurísticas adicionales del código. La decisión final vial se tomará **estrictamente** comparando cuál probabilidad resultante de las matrices es mayor:
```python
# Modificación en _resolve_car_vs_bus
if prob_vector_road['Bus'] > prob_vector_road['Carro']:
    return 'Bus'
else:
    return 'Carro'
```
* **Ventaja**: Pureza metodológica absoluta según el diseño probabilístico del artículo científico.
* **Riesgo**: Posible incremento de falsos positivos en carros (clasificar autobuses rápidos o de larga distancia como autos privados) debido al sesgo natural de las matrices de velocidad y distancia acumulada.

### Opción 2: Enfoque Híbrido Calibrado (Mantener Reglas de Seguridad)
Mantener la regla actual de solapamiento físico (`overlap_fraction > 0.6`) y velocidad promedio (`avg_speed < 30 km/h`) como un "filtro de auditoría" para rescatar trayectos que espacialmente son autobuses obvios pero que las velocidades de tránsito rápido clasificaron como automóviles.

---

### ¿Cómo te gustaría proceder?
Dime cuál opción de diseño prefieres para la Parte C (Pura vs. Híbrida) y qué límites finales prefieres establecer para la Parte A (Caminar y Metro), y con gusto actualizaré el código del clasificador para ti en el siguiente paso.
