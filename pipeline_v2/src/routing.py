import time
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gpd
from pyproj import Transformer
from shapely.ops import substring
from shapely import wkt
from shapely.geometry import Point, LineString
from .segmentation import haversine_vectorized

# --- TRANSFORMATION TRANSFORMERS ---
TRANSFORMER_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)
TRANSFORMER_TO_WGS = Transformer.from_crs("EPSG:32614", "EPSG:4326", always_xy=True)

def get_candidates_vectorized(edges_gdf, gdf_points, buffer_m=150, max_cands=12):
    """
    Asignación espacial VECTORIZADA de alto rendimiento.
    """
    # 1. Spatial Join Nearest (Búsqueda optimizada por R-Tree)
    joined = gpd.sjoin_nearest(
        gdf_points, 
        edges_gdf, 
        how='left', 
        max_distance=buffer_m, 
        distance_col='dist_exacta'
    )

    # SAFEGUARD 1: Si no encontró absolutamente nada (fuera de radio)
    if 'index_right' not in joined.columns or joined['index_right'].isna().all():
        empty_series = pd.Series([[] for _ in range(len(gdf_points))], index=gdf_points.index)
        return empty_series, empty_series
    
    joined = joined.dropna(subset=['index_right'])
    
    # SAFEGUARD 2: Si tras borrar NaNs queda vacío
    if joined.empty:
        empty_series = pd.Series([[] for _ in range(len(gdf_points))], index=gdf_points.index)
        return empty_series, empty_series
    
    # 3. Filtrar para mantener solo los mejores 'max_cands' por punto
    joined = joined.sort_values('dist_exacta')
    joined = joined.groupby(level=0).head(max_cands)
    
    # 4. Estructurar la salida: Combinamos nodos 'u' y 'v' de la arista
    def format_candidates(group):
        ids = []
        dists = []
        for _, row in group.iterrows():
            ids.extend([row['u'], row['v']])
            # La distancia al nodo es la distancia a la arista (estimación conservadora)
            dists.extend([row['dist_exacta'], row['dist_exacta']])
        return pd.Series({'ids': ids, 'dists': dists})

    result_df = joined.groupby(level=0).apply(format_candidates)
    
    # 5. Reindexar para no perder puntos GPS originales
    result_df = result_df.reindex(gdf_points.index)
    
    # Limpieza final de nulos en las listas
    result_df['ids'] = result_df['ids'].apply(lambda x: x if isinstance(x, list) else [])
    result_df['dists'] = result_df['dists'].apply(lambda x: x if isinstance(x, list) else [])
    
    return result_df['ids'], result_df['dists']


def _obtener_tramo_metro(lon1, lat1, lon2, lat2, df_metro_proj):
    """
    Proyecta dos puntos GPS a la vía real del metro, elimina el ruido lateral
    y extrae la LineString de la ruta exacta y su distancia en metros.
    """
    # 1. Proyectar a UTM 14N
    x_start, y_start = TRANSFORMER_TO_UTM.transform(lon1, lat1)
    x_end, y_end = TRANSFORMER_TO_UTM.transform(lon2, lat2)
    pt_start = Point(x_start, y_start)
    pt_end = Point(x_end, y_end)
    
    # 2. Identificar la línea de metro más cercana al origen
    distancias_start = df_metro_proj.distance(pt_start)
    linea_mas_cercana_idx = distancias_start.idxmin()
    nearest_line = df_metro_proj.loc[linea_mas_cercana_idx, 'geometry']
    
    # 3. FILTRO DE RUIDO (Snap to track)
    d_start = nearest_line.project(pt_start)
    d_end = nearest_line.project(pt_end)
    
    # 4. Extraer el segmento de la curva original
    distancia_m = abs(d_end - d_start) # ¡Distancia exacta sobre las vías!
    
    if d_start <= d_end:
        sub_line_utm = substring(nearest_line, d_start, d_end)
    else:
        sub_line_utm = substring(nearest_line, d_end, d_start)
        # Invertir coordenadas para mantener la dirección correcta del vehículo
        sub_line_utm = LineString(list(sub_line_utm.coords)[::-1])
        
    # 5. Retornar a WGS84 para Kepler.gl
    x_coords, y_coords = sub_line_utm.xy
    lons, lats = TRANSFORMER_TO_WGS.transform(x_coords, y_coords)
    sub_line_wgs = LineString(zip(lons, lats))
    
    return sub_line_wgs.wkt, distancia_m


def preparar_pares_candidatos(candidatos_origen, candidatos_destino):
    if not candidatos_origen or not candidatos_destino: return []
    pares = []
    for id_org, dist_org in candidatos_origen:
        for id_dest, dist_dest in candidatos_destino:
            pares.append({'origen': id_org, 'destino': id_dest, 'costo_espacial': dist_org + dist_dest})
    return sorted(pares, key=lambda x: x['costo_espacial'])


def complete_route(id, registros_person, 
                G_drive, G_walk,
                ig_drive, ig_walk,      
                map_drive, map_walk,    
                geometry_metro=None):
    """
    Realiza el map-matching con iGraph usando una ventana dinámica (while loop)
    y escudos de amnesia temporal/spatial skip.
    Implementa un sistema de Strikes para evitar Deadlocks por anclas corruptas (Rollback).
    Física estricta: Si CUALQUIER arista rompe la velocidad máxima, se descarta el ruteo.
    """ 
    rpc_list = []

    # Safe fallback if input is too small
    n_registros = len(registros_person)
    columnas_esperadas = ['caid', 'trip', 'modo_transporte', 'distance_m', 'highway', 'ruteo_fallido', 'corregido_espacialmente', 'Speed [km/h]', 'geometry', 'local_timestamp', 'latitude', 'longitude', 'start_node', 'end_node', 'osmid', 'flag_auditoria']
    
    if n_registros < 2:
        return pd.DataFrame(columns=columnas_esperadas)

    trips_arr = registros_person['trip'].to_numpy()
    lats_arr = registros_person['lat_ruteo'].to_numpy() if 'lat_ruteo' in registros_person.columns else registros_person['latitude'].to_numpy()
    lons_arr = registros_person['lon_ruteo'].to_numpy() if 'lon_ruteo' in registros_person.columns else registros_person['longitude'].to_numpy()

    drive_ids_arr = registros_person['drive_ids'].to_numpy() 
    drive_dists_arr = registros_person['drive_dists'].to_numpy()
    walk_ids_arr = registros_person['walk_ids'].to_numpy()
    walk_dists_arr = registros_person['walk_dists'].to_numpy()
    
    timestamps_list = registros_person['local_timestamp'].tolist() 
    
    if 'modo_transporte' in registros_person.columns:
        modos_arr = registros_person['modo_transporte'].to_numpy()
    else:
        modos_arr = np.array(['Carro'] * n_registros)

    nodo_final_anterior = None
    trip_anterior = None
    limites_kmh = {'Caminar': 22.0, 'Bus': 100, 'Metro': 100.0, 'Carro': 150.0, 'Parada': 4.0}
    limites_kmhlazy = {'Caminar': 4.5, 'Bus': 20.0, 'Metro': 35.0, 'Carro': 35.0, 'Parada': 3.0}

    origen_idx = 0
    destino_idx = 1
    strikes = 0
    nodos_envenenados = set() # NUEVO: Memoria para evitar bucles infinitos en el rollback

    while destino_idx < n_registros:
        trip_id = trips_arr[origen_idx]
        trip_dest = trips_arr[destino_idx]
        
        modo_actual = modos_arr[origen_idx]
        max_speed_kmh = limites_kmh.get(modo_actual, 160.0)
        
        # Paradas
        if trip_id <= 0:
            lat_org, lon_org = lats_arr[origen_idx], lons_arr[origen_idx]
            geom_wkt = f'POINT ({lon_org} {lat_org})'
            rpc_list.append({
                'caid': id, 'trip': trip_id,         
                'latitude': lat_org, 'longitude': lon_org, 
                'Speed [km/h]': 0.0, 
                'local_timestamp': timestamps_list[origen_idx],
                'start_node': 'N/A', 'end_node': 'N/A', 
                'osmid': 'N/A', 'highway': 'parada_inactiva', 
                'geometry': geom_wkt, 'distance_m': 0.0, 
                'modo_transporte': modo_actual,
                'ruteo_fallido': False,
                'corregido_espacialmente': False,
                'flag_auditoria': 'None',
                'idx_origen': origen_idx,       # NUEVO: Rollback Tracking
                'idx_destino': destino_idx,     # NUEVO: Rollback Tracking
                'nodo_final': None              # NUEVO: Rollback Tracking
            })
            nodo_final_anterior = None
            trip_anterior = trip_id
            origen_idx = destino_idx
            destino_idx += 1
            strikes = 0
            continue

        # =============================================================
        # INYECTA EL BLOQUE DEL METRO AQUÍ MISMO
        # =============================================================

        # BYPASS TOPOLÓGICO PARA EL METRO (Snap to Track Geometry)
        if str(modo_actual).title() == 'Metro':
            lat1, lon1 = lats_arr[origen_idx], lons_arr[origen_idx]
            lat2, lon2 = lats_arr[destino_idx], lons_arr[destino_idx]
            
            time_real = (timestamps_list[destino_idx] - timestamps_list[origen_idx]).total_seconds()
            
            try:
                # 1. Ejecutamos tu función mágica de ajuste a las vías
                geom_wkt, distancia_m = _obtener_tramo_metro(lon1, lat1, lon2, lat2, geometry_metro)
                vel_metro = (distancia_m / 1000.0) / (time_real / 3600.0) if time_real > 0 else 0
                flag = 'Metro_Topologico (Track Snapped)'
            except Exception:
                # 2. Airbag de seguridad: Si falla, trazamos línea recta
                dist_km = haversine_vectorized(lat1, lon1, lat2, lon2)
                distancia_m = dist_km * 1000.0
                vel_metro = dist_km / (time_real / 3600.0) if time_real > 0 else 0
                geom_wkt = f'LINESTRING ({lon1} {lat1}, {lon2} {lat2})'
                flag = 'Metro_Bypass_Haversine'
            
            rpc_list.append({
                'caid': id, 'trip': trip_id,         
                'latitude': lat2, 'longitude': lon2, 
                'Speed [km/h]': vel_metro, 
                'local_timestamp': timestamps_list[destino_idx],
                'start_node': 'Metro_Track', 'end_node': 'Metro_Track', 
                'osmid': 'Metro', 'highway': 'railway', 
                'geometry': geom_wkt, 'distance_m': distancia_m, 
                'modo_transporte': modo_actual,
                'ruteo_fallido': False,
                'corregido_espacialmente': True,
                'flag_auditoria': flag,
                'idx_origen': origen_idx,
                'idx_destino': destino_idx,
                'nodo_final': None 
            })
            
            nodo_final_anterior = None
            trip_anterior = trip_id
            origen_idx = destino_idx
            destino_idx += 1
            strikes = 0
            continue
        # =============================================================


        es_peaton = (str(modo_actual).lower() == 'caminar')
        G_actual = G_walk if es_peaton else G_drive
        ig_actual = ig_walk if es_peaton else ig_drive 
        map_actual = map_walk if es_peaton else map_drive 
        
        lat1, lon1 = lats_arr[origen_idx], lons_arr[origen_idx]
        lat2, lon2 = lats_arr[destino_idx], lons_arr[destino_idx]
        
        # Calculamos distancia con haversine
        distancia_haversine_km = haversine_vectorized(lat1, lon1, lat2, lon2)

        # -------------------------------------------------------------
        # Escudo 1: Spatial Skip (Burbuja de Incertidumbre)
        # -------------------------------------------------------------
        if distancia_haversine_km < 0.010:
            geom_wkt = f'POINT ({lon2} {lat2})'
            rpc_list.append({
                'caid': id, 'trip': trip_dest,         
                'latitude': lat2, 'longitude': lon2, 
                'Speed [km/h]': 0.0, 
                'local_timestamp': timestamps_list[destino_idx],
                'start_node': 'N/A', 'end_node': 'N/A', 
                'osmid': 'N/A', 'highway': 'Fallback: Spatial_Skip (Ruido)', 
                'geometry': geom_wkt, 'distance_m': 0.0, 
                'modo_transporte': modos_arr[destino_idx],
                'ruteo_fallido': True,
                'corregido_espacialmente': False,
                'flag_auditoria': 'Spatial_Skip',
                'idx_origen': origen_idx,       # NUEVO: Rollback Tracking
                'idx_destino': destino_idx,     # NUEVO: Rollback Tracking
                'nodo_final': nodo_final_anterior # NUEVO: Rollback Tracking
            })
            destino_idx += 1
            strikes = 0
            continue
            
        # -------------------------------------------------------------
        # RUTEO TOPOLÓGICO Y SELECCIÓN DE CANDIDATOS (CON ROLLBACK)
        # -------------------------------------------------------------
        # NUEVO: ROLLBACK - 1. Limpiamos la lista negra si cambiamos de viaje
        if trip_id != trip_anterior:
            nodos_envenenados.clear()

        # NUEVO: ROLLBACK - 2. Generamos la lista de candidatos
        if es_peaton:
            cands_base_org = [c for c in zip(walk_ids_arr[origen_idx], walk_dists_arr[origen_idx]) if c[0] not in nodos_envenenados]
            cands_dest = [c for c in zip(walk_ids_arr[destino_idx], walk_dists_arr[destino_idx]) if c[0] not in nodos_envenenados]
        else:
            cands_base_org = [c for c in zip(drive_ids_arr[origen_idx], drive_dists_arr[origen_idx]) if c[0] not in nodos_envenenados]
            cands_dest = [c for c in zip(drive_ids_arr[destino_idx], drive_dists_arr[destino_idx]) if c[0] not in nodos_envenenados]
        
        # ¡CORRECCIÓN VITAL! Mantenemos el ancla estrictamente para no romper la línea visual.
        # Solo se libera si es el inicio de un viaje o venimos de una Amnesia Definitiva.
        if trip_id == trip_anterior and nodo_final_anterior is not None:
            cands_org = [(nodo_final_anterior, 0.0)]
        else:
            cands_org = cands_base_org
            
        pares_evaluacion = preparar_pares_candidatos(cands_org, cands_dest)

        time_real = (timestamps_list[destino_idx] - timestamps_list[origen_idx]).total_seconds()
        delta_t_horas = time_real / 3600.0 if time_real > 0 else 0
        current_time = timestamps_list[origen_idx]
        
        best_route = None
        time_calc_best = 0
        ruta_exitosa = False
        flag_auditoria = 'None'
        candidatos_validos = []
        
        rutas_rechazadas_por_velocidad = 0 
        
        if pares_evaluacion:
            for par in pares_evaluacion:
                u, v = par['origen'], par['destino']
                if u == v: continue
            
                try:
                    ig_u, ig_v = map_actual[u], map_actual[v]
                    route_ig = ig_actual.get_shortest_paths(ig_u, to=ig_v, weights='length', output='vpath')[0]
                    if len(route_ig) < 2: continue
                    route = [ig_actual.vs[nx_id]['_nx_name'] for nx_id in route_ig]
                
                    distancia_m = sum(G_actual[route[n]][route[n+1]][0].get('length', 0) for n in range(len(route)-1))
                    distancia_grafo_km = distancia_m / 1000.0
                    vel_ruta_kmh = distancia_grafo_km / delta_t_horas if delta_t_horas > 0 else 0
                
                    # Nivel 1: Lazy Selection
                    limite_distancia = max(distancia_haversine_km * 1.4, distancia_haversine_km + 0.25)
                    
                    if (distancia_grafo_km <= limite_distancia) and (vel_ruta_kmh <= limites_kmhlazy.get(modo_actual, 60.0)):
                        best_route = route
                        time_calc_best = sum(G_actual[best_route[n]][best_route[n+1]][0].get('travel_time', 0) for n in range(len(best_route)-1))
                        flag_auditoria = 'Nivel1_Lazy'
                        ruta_exitosa = True
                        break
                    
                    # ---------------------------------------------------------
                    # FILTRO DE CIRCUIDAD (ANTI-DESVÍOS MASIVOS)
                    # ---------------------------------------------------------
                    ratio_desvio = distancia_grafo_km / (distancia_haversine_km + 0.0001)

                    if ratio_desvio > 5.0 and distancia_grafo_km > 1.2:
                        rutas_rechazadas_por_velocidad += 1 
                        continue 
                
                    # Nivel 2: Validación Física
                    if vel_ruta_kmh <= limites_kmh.get(modo_actual, 160.0):
                        candidatos_validos.append({'ruta': route, 'dist': distancia_grafo_km, 'vel': vel_ruta_kmh})
                    else:
                        rutas_rechazadas_por_velocidad += 1 
                
                except Exception:
                    continue

        if not ruta_exitosa and candidatos_validos:
            candidatos_validos.sort(key=lambda x: x['dist'])
            ganador = candidatos_validos[0]
            best_route = ganador['ruta']
            time_calc_best = sum(G_actual[best_route[n]][best_route[n+1]][0].get('travel_time', 0) for n in range(len(best_route)-1))
            flag_auditoria = 'Nivel2_Exhaustivo'
            ruta_exitosa = True

        # -------------------------------------------------------------
        # Escudo 2 ESTRICTO: Amnesia Temporal 
        # (Se descarta si CUALQUIER tramo de la ruta viola la física)
        # -------------------------------------------------------------
        velocidad_excedida_subsegmento = False
        if ruta_exitosa:
            try:
                attrs = G_actual[best_route[0]][best_route[1]][0]
                maxspeed_osm = attrs.get('maxspeed', 40)
                if isinstance(maxspeed_osm, list): maxspeed_osm = maxspeed_osm[0]
                limite_calle = float(str(maxspeed_osm).split()[0]) if str(maxspeed_osm).replace('.','',1).isdigit() else 40.0
            except:
                limite_calle = 40.0
            
            # NUEVO: Fix Lógico para permitir rebase de límite de calle (margen de 50%) sin sobrepasar límite global
            v_techo = limite_calle * 1.5 if modo_actual not in ['Caminar', 'Parada'] else max_speed_kmh
            techo_final = min(v_techo, max_speed_kmh) if modo_actual not in ['Caminar', 'Parada'] else max_speed_kmh
            
            # Validación global del viaje
            distancia_m_best = sum(G_actual[best_route[n]][best_route[n+1]][0].get('length', 0) for n in range(len(best_route)-1))
            vel_final_ruta = (distancia_m_best / 1000.0) / delta_t_horas if delta_t_horas > 0 else 0
            
            if vel_final_ruta > techo_final:
                velocidad_excedida_subsegmento = True
            else:
                # Validación local (sub-segmento por sub-segmento)
                for i in range(len(best_route)-1):
                    u_test, v_test = best_route[i], best_route[i+1]
                    edge_test = G_actual[u_test][v_test][0]
                    l_test = edge_test.get('length', 0)
                    t_ideal_test = edge_test.get('travel_time', 1) 
                    
                    t_alloc_test = time_real * (t_ideal_test / time_calc_best) if time_calc_best > 0 else (time_real / (len(best_route)-1))
                    vel_local = (l_test / 1000.0) / (t_alloc_test / 3600.0) if t_alloc_test > 0 else 0
                    
                    if vel_local > techo_final:
                        velocidad_excedida_subsegmento = True
                        break # ¡Rompe la física! Abortamos todo este ruteo
                    
                    
        if not ruta_exitosa or velocidad_excedida_subsegmento:
            strikes += 1
            salto_dinamico = strikes
            
            if velocidad_excedida_subsegmento:
                razon = 'Fisica_Rota_Subsegmento'
            elif rutas_rechazadas_por_velocidad > 0 and not ruta_exitosa:
                razon = 'Fisica_Rota_Nivel2 (>160kmh)'
            else:
                razon = 'OSM_Desconectado (Topologia)'
            
            # ---------------------------------------------------------
            # NUEVO: ROLLBACK LOGIC (Amnesia Hacia Atrás) - FIX GAPS
            # ---------------------------------------------------------
            if strikes == 2 and 'Fisica_Rota' in razon:
                tiene_exito_previo = any(not t.get('ruteo_fallido', True) for t in rpc_list if t.get('trip') == trip_id)
                
                if tiene_exito_previo:
                    # 1. Identificamos el ID de destino del último viaje exitoso
                    ultimo_idx_destino = next(t.get('idx_destino') for t in reversed(rpc_list) if not t.get('ruteo_fallido', True) and t.get('trip') == trip_id)
                    
                    # 2. Hacemos pop de TODA la ruta (todas las calles) que pertenezcan a ese intento
                    tramo_malo = None
                    while len(rpc_list) > 0 and rpc_list[-1].get('trip') == trip_id and rpc_list[-1].get('idx_destino', 0) >= ultimo_idx_destino:
                        tramo = rpc_list.pop()
                        if not tramo.get('ruteo_fallido', True):
                            tramo_malo = tramo # El último en salir (el primero de la ruta) tiene nuestro origen original
                    
                    # --- FIX CRÍTICO: El Paracaídas del Rollback ---
                    # Si el ruido de Veraset rompió la secuencia y no encontramos el tramo,
                    # abortamos el rollback para evitar el TypeError y forzamos rendición.
                    if tramo_malo is None:
                        # Forzamos los strikes al máximo para que entre al bloque de "Rendición"
                        strikes = 99 
                        # Rompemos el ciclo del Rollback y dejamos que el código siga
                        # hacia la lógica de "Amnesia_Definitiva" que está justo abajo.
                    else:
                        # 3. Envenenamos el ancla que nos metió en este problema
                        if nodo_final_anterior is not None:
                            nodos_envenenados.add(nodo_final_anterior)
                        
                        # 4. Retrocedemos el origen al paso anterior
                        origen_idx = tramo_malo['idx_origen']
                        
                        # 5. Restauramos el ancla al paso anterior (quedando limpios del error)
                        if len(rpc_list) > 0 and rpc_list[-1].get('trip') == trip_id:
                            nodo_final_anterior = rpc_list[-1].get('nodo_final')
                        else:
                            nodo_final_anterior = None
                        
                        # 6. Reiniciamos strikes y reintentamos el puente
                        strikes = 0
                        continue        
            # ---------------------------------------------------------
            
            if (destino_idx + salto_dinamico) >= n_registros or strikes > 20:
                geom_wkt = f'POINT ({lon2} {lat2})'
                rpc_list.append({
                    'caid': id, 'trip': trip_dest,         
                    'latitude': lat2, 'longitude': lon2, 
                    'Speed [km/h]': 0.0, 
                    'local_timestamp': timestamps_list[destino_idx],
                    'start_node': 'N/A', 'end_node': 'N/A', 
                    'osmid': 'N/A', 'highway': f'Rendicion: {razon}', 
                    'geometry': geom_wkt, 'distance_m': 0.0, 
                    'modo_transporte': modos_arr[destino_idx],
                    'ruteo_fallido': True,
                    'corregido_espacialmente': False,
                    'flag_auditoria': f'Amnesia_Definitiva ({razon})',
                    'idx_origen': origen_idx,       # NUEVO: Rollback Tracking
                    'idx_destino': destino_idx,     # NUEVO: Rollback Tracking
                    'nodo_final': nodo_final_anterior # NUEVO: Rollback Tracking
                })
                # BREAK DEADLOCK
                origen_idx = destino_idx
                destino_idx += 1
                strikes = 0
                nodo_final_anterior = None
            else:
                geom_wkt = f'POINT ({lon2} {lat2})'
                rpc_list.append({
                    'caid': id, 'trip': trip_dest,         
                    'latitude': lat2, 'longitude': lon2, 
                    'Speed [km/h]': 0.0, 
                    'local_timestamp': timestamps_list[destino_idx],
                    'start_node': 'N/A', 'end_node': 'N/A', 
                    'osmid': 'N/A', 'highway': 'Lookahead_Skip', 
                    'geometry': geom_wkt, 'distance_m': 0.0, 
                    'modo_transporte': modos_arr[destino_idx],
                    'ruteo_fallido': True,
                    'corregido_espacialmente': False,
                    'flag_auditoria': f'Lookahead_Skip ({razon})',
                    'idx_origen': origen_idx,       # NUEVO: Rollback Tracking
                    'idx_destino': destino_idx,     # NUEVO: Rollback Tracking
                    'nodo_final': nodo_final_anterior # NUEVO: Rollback Tracking
                })
                
                # Amnesia normal con salto dinámico (Adaptive Lookahead)
                destino_idx += salto_dinamico
                
            trip_anterior = trip_id
            continue

        # -------------------------------------------------------------
        # 3. Éxito Topológico
        # -------------------------------------------------------------
        for i in range(len(best_route)-1):
            u, v = best_route[i], best_route[i+1]
            edge_data = G_actual[u][v][0] 
            
            l_row = edge_data.get('length', 0)
            t_ideal = edge_data.get('travel_time', 1) 
            time_alloc = time_real * (t_ideal / time_calc_best) if time_calc_best > 0 else (time_real / (len(best_route)-1))
            
            speed_kph = (l_row / 1000.0) / (time_alloc / 3600.0) if time_alloc > 0 else 0
            
            if 'geometry' in edge_data: geom = edge_data['geometry'].wkt
            else: geom = f"LINESTRING ({G_actual.nodes[u]['x']} {G_actual.nodes[u]['y']}, {G_actual.nodes[v]['x']} {G_actual.nodes[v]['y']})"

            rpc_list.append({
                'caid': id, 'trip': trip_id,    
                'latitude': G_actual.nodes[u]['y'], 'longitude': G_actual.nodes[u]['x'],
                'Speed [km/h]': speed_kph,
                'local_timestamp': current_time,
                'start_node': u, 'end_node': v,
                'osmid': str(edge_data.get('osmid', 'N/A')),
                'highway': edge_data.get('highway', 'unclassified'), 
                'geometry': geom, 'distance_m': l_row,
                'modo_transporte': modo_actual,
                'ruteo_fallido': False, 
                'corregido_espacialmente': False,
                'flag_auditoria': flag_auditoria,
                'idx_origen': origen_idx,       # NUEVO: Rollback Tracking
                'idx_destino': destino_idx,     # NUEVO: Rollback Tracking
                'nodo_final': best_route[-1]    # NUEVO: Rollback Tracking
            })
            current_time = current_time + pd.Timedelta(seconds=time_alloc)
            
        nodo_final_anterior = best_route[-1]
        trip_anterior = trip_id
        
        # Avanzamos ambos punteros y REINICIAMOS strikes
        origen_idx = destino_idx
        destino_idx += 1
        strikes = 0

    # ---------------------------------------------------------
    # FORMATO DE TIEMPO PARA VISUALIZACIÓN EN KEPLER.GL
    # ---------------------------------------------------------
    
    res_df = pd.DataFrame(rpc_list)
    if res_df.empty:
        return pd.DataFrame(columns=columnas_esperadas)
    
    # Asegurar que las columnas existan
    for col in columnas_esperadas:
        if col not in res_df.columns:
            res_df[col] = None

    # Crear columna string pura compatible con el Time Playback de Kepler
    if 'local_timestamp' in res_df.columns:
        res_df['local_timestamp'] = pd.to_datetime(res_df['local_timestamp'], errors='coerce')
        res_df['kepler_time'] = res_df['local_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
    # NUEVO: Limpieza final - Removemos las columnas de tracking que usamos para el rollback
    columnas_a_borrar = ['idx_origen', 'idx_destino', 'nodo_final']
    res_df.drop(columns=[c for c in columnas_a_borrar if c in res_df.columns], inplace=True, errors='ignore')

    return res_df


class RouteHypothesisEvaluator:
    """
    Orquesta la ejecución de complete_route bajo múltiples hipótesis de transporte.
    """
    def __init__(self, G_drive, G_walk, ig_drive, ig_walk, map_drive, map_walk, geometry_metro):
        self.G_drive = G_drive
        self.G_walk = G_walk
        self.ig_drive = ig_drive
        self.ig_walk = ig_walk
        self.map_drive = map_drive
        self.map_walk = map_walk
        self.geometry_metro = geometry_metro

    def evaluate(self, id_user, df_trip, mode_candidates):
        hypotheses = {}
        for mode in mode_candidates:
            df_mock = df_trip.copy()
            df_mock['modo_transporte'] = mode
            
            # complete_route decide internamente qué red usar según 'modo_transporte'
            df_routed = complete_route(
                id=id_user,
                registros_person=df_mock,
                G_drive=self.G_drive,
                G_walk=self.G_walk,
                ig_drive=self.ig_drive,
                ig_walk=self.ig_walk,
                map_drive=self.map_drive,
                map_walk=self.map_walk,
                geometry_metro=self.geometry_metro
            )
            if not df_routed.empty:
                hypotheses[mode] = df_routed
        return hypotheses
