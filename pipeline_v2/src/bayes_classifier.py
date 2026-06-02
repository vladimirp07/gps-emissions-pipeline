import numpy as np
import pandas as pd
import geopandas as gpd

def calcular_cercania_infraestructura(df, subway_routes, bus_routes):
    """
    Calcula la proximidad espacial (distancia métrica exacta) de cada ping GPS
    a las líneas físicas del metro y rutas de autobús oficiales.
    """
    if df.empty:
        return df.copy()
        
    # 1. Proyectar temporalmente a UTM 14N (EPSG:32614)
    gdf_pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
    gdf_pts = gdf_pts.to_crs("EPSG:32614")
    
    RADIO_BUSQUEDA_METROS = 150
    
    # 2. Búsqueda Espacial Exacta (sjoin_nearest mide a la LÍNEA, no al centroide)
    # --- METRO ---
    metro_join = gpd.sjoin_nearest(gdf_pts, subway_routes, how='left', distance_col='dist_metro')
    metro_join = metro_join[~metro_join.index.duplicated(keep='first')] # Limpiar empates de distancia
    df['near_subway_line'] = (metro_join['dist_metro'] < RADIO_BUSQUEDA_METROS).astype(int)
    
    # --- BUS ---
    bus_join = gpd.sjoin_nearest(gdf_pts, bus_routes, how='left', distance_col='dist_bus')
    bus_join = bus_join[~bus_join.index.duplicated(keep='first')] 
    df['near_bus_route'] = (bus_join['dist_bus'] < RADIO_BUSQUEDA_METROS).astype(int)
    
    return df

class PriorModeClassifier:
    """
    Clasificador de Prior Modal y Heurísticas de Filtrado.
    Calcula priors heurísticos y realiza el descarte/poda de hipótesis imposibles
    utilizando variables baratas derivadas directamente del GPS crudo (sin ruteo).
    """
    def __init__(self, max_walk_speed=25.0, max_walk_dist=12.0):
        self.max_walk_speed = max_walk_speed
        self.max_walk_dist = max_walk_dist

    def generate_mode_priors(self, df_trip, near_subway, near_bus):
        """
        Calcula una distribución de probabilidad a priori simplificada para cada modo
        (Caminar, Metro, Carro, Bus) basándose en estadísticas rápidas del GPS crudo.
        
        Este es el 'Prior' conceptual en nuestro flujo Bayesiano Coarse-to-Fine.
        """
        max_speed = df_trip['Speed [km/h]'].max()
        total_dist_km = df_trip['dis lineal [m]'].sum() / 1000.0
        
        # Inicializar priors uniformes / base
        priors = {'Caminar': 0.25, 'Metro': 0.25, 'Carro': 0.25, 'Bus': 0.25}
        
        # Ajustes heurísticos rápidos
        if max_speed > self.max_walk_speed:
            priors['Caminar'] = 0.0
            
        if not near_subway.any() or total_dist_km <= 1.0:
            priors['Metro'] = 0.0
            
        if max_speed <= 3.0 and total_dist_km <= 0.5:
            # Si casi no hay movimiento, disminuye probabilidad de motorizados
            priors['Carro'] = 0.05
            priors['Bus'] = 0.05
            
        # Normalizar vector
        sum_priors = sum(priors.values())
        if sum_priors > 0:
            priors = {k: v / sum_priors for k, v in priors.items()}
        else:
            priors = {'Caminar': 0.0, 'Metro': 0.0, 'Carro': 0.5, 'Bus': 0.5}
            
        return priors

    def prune_impossible_hypotheses(self, df_trip, near_subway, near_bus):
        """
        Poda (filtra) hipótesis imposibles para ahorrar costes de ruteo.
        
        Retorna la lista de modos candidatos simplificados que requieren ruteo:
        - 'Caminar' (si es viable físicamente)
        - 'Metro' (si es viable físicamente y hay cercanía al metro)
        - 'Carro' (representando la red vial común o 'road_motorized')
        
        Nota metodológica: 'Bus' no se rutea de manera independiente para evitar
        duplicar coste computacional sobre la misma red vial (G_drive). Si la hipótesis
        vial gana, se discriminará 'Carro' vs 'Bus' en el paso posterior.
        """
        max_speed = df_trip['Speed [km/h]'].max()
        total_dist_km = df_trip['dis lineal [m]'].sum() / 1000.0
        
        candidates = []
        
        # 1. Chequeo de Caminar
        if max_speed <= self.max_walk_speed and total_dist_km <= self.max_walk_dist:
            candidates.append('Caminar')
            
        # 2. Chequeo de Metro
        if near_subway.any() and total_dist_km > 1.0:
            candidates.append('Metro')
            
        # 3. Chequeo de Motorizado Vial (Carro / Bus)
        # Se rutea bajo el candidato genérico 'Carro' (road_motorized)
        if max_speed > 3.0 or total_dist_km > 0.5:
            candidates.append('Carro')
            
        # Salvaguarda: al menos un candidato
        if not candidates:
            candidates = ['Carro']
            
        return candidates

    def predict_candidates(self, df_trip, near_subway, near_bus):
        """Método obsoleto (deprecated) mantenido por compatibilidad hacia atrás."""
        return self.prune_impossible_hypotheses(df_trip, near_subway, near_bus)


class BayesianRouteEvaluator:
    """
    Evaluador Bayesiano Posterior de Rutas.
    Clasifica las hipótesis de ruteo completadas utilizando métricas físicas
    corregidas (velocidad de calle, distancia topológica) y cercanía a infraestructura,
    contrastadas contra las matrices probabilísticas del paper.
    """
    def __init__(self):
        self.modos = ['Carro', 'Bus', 'Metro', 'Caminar']
        
        # Matrices de Probabilidad Condicional originales del artículo de investigación
        self.Cercania = np.array([
            [0.10, 0.10, 0.80, 0.00],  # Cerca de estación de metro (índice 0)
            [0.10, 0.80, 0.00, 0.10],  # Cerca de ruta de autobús (índice 1)
            [0.40, 0.25, 0.05, 0.30]   # Sin infraestructura cerca (índice 2)
        ])

        self.Velocidad = np.array([
            [0.05, 0.10, 0.15, 0.60],  # Velocidad instantánea <= 6.0 km/h
            [0.47, 0.38, 0.05, 0.10],  # 6.0 < Velocidad <= 20.0 km/h
            [0.50, 0.30, 0.20, 0.00],  # 20.0 < Velocidad <= 80.0 km/h
            [1.00, 0.00, 0.00, 0.00]   # Velocidad > 80.0 km/h
        ])

        self.Distancia = np.array([
            [0.10, 0.20, 0.30, 0.40],  # Distancia de viaje <= 1.0 km
            [0.25, 0.25, 0.30, 0.20],  # 1.0 < Distancia <= 6.0 km
            [0.40, 0.15, 0.25, 0.20],  # 6.0 < Distancia <= 10.0 km
            [0.60, 0.30, 0.00, 0.10],  # 10.0 < Distancia <= 18.0 km
            [0.40, 0.40, 0.00, 0.20]   # Distancia > 18.0 km
        ])

        self.Velprom = np.array([
            [0.10, 0.10, 0.20, 0.60],  # Velocidad promedio viaje <= 6.0 km/h
            [0.40, 0.25, 0.25, 0.10]   # Velocidad promedio viaje > 6.0 km/h
        ])

    def evaluate_completed_route_with_matrices(self, df_routed, mode_hypothesis, subway_routes, bus_routes):
        """
        Evalúa una hipótesis de ruta ya calculada a través de las matrices bayesianas del paper.
        Retorna la distribución de probabilidad a posteriori para los 4 modos en esta hipótesis.
        """
        if df_routed.empty:
            return pd.Series([0.0, 0.0, 0.0, 0.0], index=self.modos)

        df_eval = df_routed.copy()
        # Calculamos proximidad en base a la línea ruteada físicamente
        df_eval = calcular_cercania_infraestructura(df_eval, subway_routes, bus_routes)

        # 1. Índice de Cercanía: 0 (Metro), 1 (Bus), 2 (Ninguno)
        idx_c = np.where(df_eval['near_subway_line'] == 1, 0,
                np.where(df_eval['near_bus_route'] == 1, 1, 2))

        # 2. Índice de Velocidad por punto corregido
        idx_v = np.digitize(df_eval['Speed [km/h]'].fillna(0), bins=[6.001, 20.001, 80.001])

        # 3. Índice de Distancia física acumulada de ruteo
        total_dist_km = df_eval['distance_m'].sum() / 1000.0
        idx_d = np.digitize([total_dist_km], bins=[1.0, 6.001, 10.001, 18.001])[0]
        idx_d_arr = np.repeat(idx_d, len(df_eval))

        # 4. Índice de Velocidad promedio de ruta
        avg_speed_trip = df_eval['Speed [km/h]'].mean()
        idx_vp = np.digitize([avg_speed_trip], bins=[6.001])[0]
        idx_vp_arr = np.repeat(idx_vp, len(df_eval))

        # 5. Multiplicación Bayesiana
        P_unnorm = (self.Cercania[idx_c] * 
                    self.Velocidad[idx_v] * 
                    self.Distancia[idx_d_arr] * 
                    self.Velprom[idx_vp_arr])

        # 6. Normalización por puntos
        suma_puntos = P_unnorm.sum(axis=1, keepdims=True)
        suma_puntos[suma_puntos == 0] = 1
        P_norm = P_unnorm / suma_puntos

        # 7. Votación acumulada de viaje
        total_votes = P_norm.sum(axis=0)
        total_votes_normalized = total_votes / (total_votes.sum() + 1e-9)
        
        return pd.Series(total_votes_normalized, index=self.modos)

    def _resolve_car_vs_bus(self, df_routed, subway_routes, bus_routes):
        """
        Sub-clasificación posterior para discriminar entre Carro y Bus una vez que se ha
        determinado que el viaje se realizó por la red vial motorizada (road_motorized).
        
        Usa variables contextuales y topológicas de alta fidelidad:
        1. Cercanía espacial a rutas de bus a lo largo de la trayectoria.
        2. Patrón de paradas de bus / velocidad promedio de ruteo.
        3. Matrices de probabilidad bayesianas evaluadas sobre la ruta motorizada.
        """
        # Calcular proximidad en base a la línea ruteada físicamente primero
        df_eval = df_routed.copy()
        df_eval = calcular_cercania_infraestructura(df_eval, subway_routes, bus_routes)
        
        prob_vector_road = self.evaluate_completed_route_with_matrices(df_eval, 'Carro', subway_routes, bus_routes)
        
        # Detección heurística de paradas/patrón de velocidad baja recurrente
        # (Los autobuses paran constantemente y circulan a menor velocidad media en zonas de infraestructura)
        avg_speed = df_eval['Speed [km/h]'].mean()
        overlap_fraction = df_eval['near_bus_route'].mean()
        
        # Si la probabilidad bayesiana de Bus es superior a la de Carro,
        # o si hay un alto solapamiento con la red de bus y velocidades moderadas:
        if prob_vector_road['Bus'] > prob_vector_road['Carro'] or (overlap_fraction > 0.6 and avg_speed < 30.0):
            return 'Bus'
        else:
            return 'Carro'

    def select_final_mode(self, hypotheses, subway_routes, bus_routes):
        """
        Compara y evalúa todas las hipótesis de ruteo completadas y selecciona la óptima.
        
        Pasos metodológicos:
        1. Evalúa la probabilidad a posteriori de cada hipótesis ('Caminar', 'Metro', 'Carro')
           usando sus respectivas geometrías ruteadas.
        2. Determina el modo ganador en base a la probabilidad de su propia hipótesis.
        3. Si la hipótesis vial ganadora es 'Carro' (road_motorized), realiza la sub-clasificación
           posterior 'Carro vs Bus' para no duplicar el costo del ruteo.
        """
        best_mode = None
        best_probability = -1.0
        best_df = None
        diagnostic_probs = {}

        # Evaluar candidatos ruteados (Caminar, Metro, Carro/Road_Motorized)
        for mode, df_routed in hypotheses.items():
            prob_vector = self.evaluate_completed_route_with_matrices(df_routed, mode, subway_routes, bus_routes)
            # Evaluamos la probabilidad del modo bajo su propia hipótesis de ruteo
            prob_self = prob_vector[mode]
            diagnostic_probs[mode] = prob_vector.to_dict()
            
            if prob_self > best_probability:
                best_probability = prob_self
                best_mode = mode
                best_df = df_routed

        # Sub-clasificación final para vehículos de carretera (Carro vs Bus)
        if best_mode == 'Carro' and 'Carro' in hypotheses:
            resolved_mode = self._resolve_car_vs_bus(hypotheses['Carro'], subway_routes, bus_routes)
            if resolved_mode == 'Bus':
                best_mode = 'Bus'
                best_df = best_df.copy()
                best_df['modo_transporte'] = 'Bus'

        return best_mode, best_df, best_probability, diagnostic_probs

    def evaluate_hypothesis(self, df_routed, mode_hypothesis, subway_routes, bus_routes):
        """Método obsoleto (deprecated) mantenido por compatibilidad hacia atrás."""
        return self.evaluate_completed_route_with_matrices(df_routed, mode_hypothesis, subway_routes, bus_routes)

    def select_best_hypothesis(self, hypotheses, subway_routes, bus_routes):
        """Método obsoleto (deprecated) mantenido por compatibilidad hacia atrás."""
        return self.select_final_mode(hypotheses, subway_routes, bus_routes)
