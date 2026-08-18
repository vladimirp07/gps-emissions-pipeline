"""Archived progressive V2 experiment.

This is not a production router.  It is retained only for historical
reproducibility of the MATLAB calibration campaigns.
"""
import collections
import os
import numpy as np
import pandas as pd
from shapely import wkt
from pipeline_v4.src.routing import (
    _finalize_routing_contract,
    _obtener_tramo_metro,
    get_ig_cache,
    haversine_vectorized,
    preparar_pares_candidatos,
)
def complete_route_v2_progressive_legacy(id, registros_person, 
                G_drive, G_walk,
                ig_drive, ig_walk,      
                map_drive, map_walk,    
                geometry_metro=None):
    """
    Versión Alternativa V2 - Optimizada con Consulta Progresiva y Traducción Diferida.
    
    Cambios implementados:
    1. Consulta Progresiva: Evalúa primero el candidato más cercano por separado. Si
       satisface el criterio Lazy, se usa inmediatamente y se evita calcular el Dijkstra
       para los otros 11 destinos.
    2. Traducción Diferida: Mantiene las rutas como secuencias de IDs enteros de iGraph durante
       las fases de validación. Traduce a nombres de NetworkX (strings) únicamente la ruta ganadora.
    3. Caché de iGraph: Utiliza un diccionario plano con claves de tuplas de enteros para búsquedas O(1) nativas.
    """ 
    from collections import defaultdict
    rpc_list = []

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
    nodos_envenenados = set() 

    # Obtenemos los cachés de aristas planos para iGraph
    ig_cache_drive, ig_names_drive = get_ig_cache(G_drive, ig_drive, map_drive)
    ig_cache_walk, ig_names_walk = get_ig_cache(G_walk, ig_walk, map_walk)

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

        # BYPASS TOPOLÓGICO PARA EL METRO (Snap to Track Geometry)
        if str(modo_actual).title() == 'Metro':
            lat1, lon1 = lats_arr[origen_idx], lons_arr[origen_idx]
            lat2, lon2 = lats_arr[destino_idx], lons_arr[destino_idx]
            
            time_real = (timestamps_list[destino_idx] - timestamps_list[origen_idx]).total_seconds()
            
            try:
                geom_wkt, distancia_m = _obtener_tramo_metro(lon1, lat1, lon2, lat2, geometry_metro)
                vel_metro = (distancia_m / 1000.0) / (time_real / 3600.0) if time_real > 0 else 0
                flag = 'Metro_Topologico (Track Snapped)'
            except Exception:
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

        es_peaton = (str(modo_actual).lower() == 'caminar')
        G_actual = G_walk if es_peaton else G_drive
        ig_actual = ig_walk if es_peaton else ig_drive 
        map_actual = map_walk if es_peaton else map_drive 
        ig_cache_actual = ig_cache_walk if es_peaton else ig_cache_drive
        ig_node_names = ig_names_walk if es_peaton else ig_names_drive
        
        lat1, lon1 = lats_arr[origen_idx], lons_arr[origen_idx]
        lat2, lon2 = lats_arr[destino_idx], lons_arr[destino_idx]
        
        distancia_haversine_km = haversine_vectorized(lat1, lon1, lat2, lon2)

        # Escudo 1: Spatial Skip
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
                'idx_origen': origen_idx,       
                'idx_destino': destino_idx,     
                'nodo_final': nodo_final_anterior 
            })
            destino_idx += 1
            strikes = 0
            continue
            
        if trip_id != trip_anterior:
            nodos_envenenados.clear()

        # Generamos candidatos
        if es_peaton:
            cands_base_org = [c for c in zip(walk_ids_arr[origen_idx], walk_dists_arr[origen_idx]) if c[0] not in nodos_envenenados]
            cands_dest = [c for c in zip(walk_ids_arr[destino_idx], walk_dists_arr[destino_idx]) if c[0] not in nodos_envenenados]
        else:
            cands_base_org = [c for c in zip(drive_ids_arr[origen_idx], drive_dists_arr[origen_idx]) if c[0] not in nodos_envenenados]
            cands_dest = [c for c in zip(drive_ids_arr[destino_idx], drive_dists_arr[destino_idx]) if c[0] not in nodos_envenenados]
        
        if trip_id == trip_anterior and nodo_final_anterior is not None:
            cands_org = [(nodo_final_anterior, 0.0)]
        else:
            cands_org = cands_base_org
            
        pares_evaluacion = preparar_pares_candidatos(cands_org, cands_dest)

        time_real = (timestamps_list[destino_idx] - timestamps_list[origen_idx]).total_seconds()
        delta_t_horas = time_real / 3600.0 if time_real > 0 else 0
        current_time = timestamps_list[origen_idx]
        
        best_path_ig = None
        best_tramos_ig = None
        best_time_calc = 0.0
        ruta_exitosa = False
        flag_auditoria = 'None'
        candidatos_validos = []
        rutas_rechazadas_por_velocidad = 0 
        
        if pares_evaluacion:
            # OPTIMIZACIÓN A: Consulta Progresiva
            # Primero evaluamos el candidato de menor costo espacial (el más cercano) por separado
            primer_par = pares_evaluacion[0]
            u1, v1 = primer_par['origen'], primer_par['destino']
            
            primer_camino_lazy = False
            if u1 in map_actual and v1 in map_actual and u1 != v1:
                try:
                    ig_u1 = map_actual[u1]
                    ig_v1 = map_actual[v1]
                    path_ig = ig_actual.get_shortest_paths(ig_u1, to=ig_v1, weights='length', output='vpath')[0]
                    
                    if len(path_ig) >= 2:
                        distancia_m = 0.0
                        time_calc_best = 0.0
                        valid_path = True
                        tramos_list = []
                        
                        for n in range(len(path_ig)-1):
                            edge_key = (path_ig[n], path_ig[n+1])
                            edge_vals = ig_cache_actual.get(edge_key)
                            if edge_vals is None:
                                valid_path = False
                                break
                            distancia_m += edge_vals[0]
                            time_calc_best += edge_vals[1]
                            tramos_list.append((path_ig[n], path_ig[n+1], edge_vals[0], edge_vals[1]))
                        
                        if valid_path:
                            distancia_grafo_km = distancia_m / 1000.0
                            vel_ruta_kmh = distancia_grafo_km / delta_t_horas if delta_t_horas > 0 else 0
                            limite_distancia = max(distancia_haversine_km * 1.4, distancia_haversine_km + 0.25)
                            
                            # Si pasa el filtro Lazy, lo tomamos y rompemos sin calcular nada más
                            if (distancia_grafo_km <= limite_distancia) and (vel_ruta_kmh <= limites_kmhlazy.get(modo_actual, 60.0)):
                                best_path_ig = path_ig
                                best_tramos_ig = tramos_list
                                best_time_calc = time_calc_best
                                flag_auditoria = 'Nivel1_Lazy'
                                ruta_exitosa = True
                                primer_camino_lazy = True
                except Exception:
                    pass
            
            if not primer_camino_lazy:
                # Si el primer candidato falló o no fue Lazy, corremos Bulk Query para el resto
                org_to_dest = defaultdict(list)
                for par in pares_evaluacion:
                    u, v = par['origen'], par['destino']
                    if u == v: continue
                    org_to_dest[u].append(v)
                
                bulk_paths = {}
                for u, dest_list in org_to_dest.items():
                    try:
                        if u not in map_actual: continue
                        ig_u = map_actual[u]
                        valid_dests = [v for v in dest_list if v in map_actual]
                        if not valid_dests: continue
                        
                        ig_dests = [map_actual[v] for v in valid_dests]
                        paths_ig_list = ig_actual.get_shortest_paths(ig_u, to=ig_dests, weights='length', output='vpath')
                        
                        for v, path_ig in zip(valid_dests, paths_ig_list):
                            if len(path_ig) >= 2:
                                bulk_paths[(u, v)] = path_ig
                    except Exception:
                        continue
                
                # Evaluamos las rutas en orden de cercanía
                for par in pares_evaluacion:
                    u, v = par['origen'], par['destino']
                    path_ig = bulk_paths.get((u, v))
                    if path_ig is None: continue
                    
                    distancia_m = 0.0
                    time_calc_best = 0.0
                    valid_path = True
                    tramos_list = []
                    
                    for n in range(len(path_ig)-1):
                        edge_key = (path_ig[n], path_ig[n+1])
                        edge_vals = ig_cache_actual.get(edge_key)
                        if edge_vals is None:
                            valid_path = False
                            break
                        distancia_m += edge_vals[0]
                        time_calc_best += edge_vals[1]
                        tramos_list.append((path_ig[n], path_ig[n+1], edge_vals[0], edge_vals[1]))
                    
                    if not valid_path: continue
                    
                    distancia_grafo_km = distancia_m / 1000.0
                    vel_ruta_kmh = distancia_grafo_km / delta_t_horas if delta_t_horas > 0 else 0
                    
                    # Nivel 1: Lazy Selection
                    limite_distancia = max(distancia_haversine_km * 1.4, distancia_haversine_km + 0.25)
                    if (distancia_grafo_km <= limite_distancia) and (vel_ruta_kmh <= limites_kmhlazy.get(modo_actual, 60.0)):
                        best_path_ig = path_ig
                        best_tramos_ig = tramos_list
                        best_time_calc = time_calc_best
                        flag_auditoria = 'Nivel1_Lazy'
                        ruta_exitosa = True
                        break
                    
                    # Filtro de Circuidad
                    ratio_desvio = distancia_grafo_km / (distancia_haversine_km + 0.0001)
                    if ratio_desvio > 5.0 and distancia_grafo_km > 1.2:
                        rutas_rechazadas_por_velocidad += 1 
                        continue 
                
                    # Nivel 2: Validación Física
                    if vel_ruta_kmh <= limites_kmh.get(modo_actual, 160.0):
                        candidatos_validos.append({
                            'path_ig': path_ig, 
                            'tramos': tramos_list,
                            'dist': distancia_grafo_km, 
                            'time': time_calc_best, 
                            'vel': vel_ruta_kmh
                        })
                    else:
                        rutas_rechazadas_por_velocidad += 1 

        if not ruta_exitosa and candidatos_validos:
            candidatos_validos.sort(key=lambda x: x['dist'])
            ganador = candidatos_validos[0]
            best_path_ig = ganador['path_ig']
            best_tramos_ig = ganador['tramos']
            best_time_calc = ganador['time']
            flag_auditoria = 'Nivel2_Exhaustivo'
            ruta_exitosa = True

        # Escudo 2 ESTRICTO: Validación Física de Subsegmentos
        velocidad_excedida_subsegmento = False
        if ruta_exitosa:
            # OPTIMIZACIÓN B: Traducción Diferida - Convertimos a NetworkX strings únicamente la ruta ganadora final
            best_route = [ig_node_names[nx_id] for nx_id in best_path_ig]
            
            try:
                attrs = G_actual[best_route[0]][best_route[1]][0]
                maxspeed_osm = attrs.get('maxspeed', 40)
                if isinstance(maxspeed_osm, list): maxspeed_osm = maxspeed_osm[0]
                limite_calle = float(str(maxspeed_osm).split()[0]) if str(maxspeed_osm).replace('.','',1).isdigit() else 40.0
            except:
                limite_calle = 40.0
            
            physics_factor = float(os.environ.get('PHYSICS_FACTOR', '2.0'))
            v_techo = limite_calle * physics_factor if modo_actual not in ['Caminar', 'Parada'] else max_speed_kmh
            techo_final = min(v_techo, max_speed_kmh) if modo_actual not in ['Caminar', 'Parada'] else max_speed_kmh
            
            distancia_m_best = sum(t[2] for t in best_tramos_ig)
            vel_final_ruta = (distancia_m_best / 1000.0) / delta_t_horas if delta_t_horas > 0 else 0
            
            if vel_final_ruta > techo_final:
                velocidad_excedida_subsegmento = True
            else:
                for u_test_ig, v_test_ig, l_test, t_ideal_test in best_tramos_ig:
                    t_alloc_test = time_real * (t_ideal_test / best_time_calc) if best_time_calc > 0 else (time_real / len(best_tramos_ig))
                    vel_local = (l_test / 1000.0) / (t_alloc_test / 3600.0) if t_alloc_test > 0 else 0
                    
                    if vel_local > techo_final:
                        velocidad_excedida_subsegmento = True
                        break 
                    
        if not ruta_exitosa or velocidad_excedida_subsegmento:
            strikes += 1
            salto_dinamico = strikes
            
            if velocidad_excedida_subsegmento:
                razon = 'Fisica_Rota_Subsegmento'
            elif rutas_rechazadas_por_velocidad > 0 and not ruta_exitosa:
                razon = 'Fisica_Rota_Nivel2 (>160kmh)'
            else:
                razon = 'OSM_Desconectado (Topologia)'
            
            # ROLLBACK LOGIC
            if strikes == 2 and 'Fisica_Rota' in razon:
                tiene_exito_previo = any(not t.get('ruteo_fallido', True) for t in rpc_list if t.get('trip') == trip_id)
                
                if tiene_exito_previo:
                    ultimo_idx_destino = next(t.get('idx_destino') for t in reversed(rpc_list) if not t.get('ruteo_fallido', True) and t.get('trip') == trip_id)
                    
                    tramo_malo = None
                    while len(rpc_list) > 0 and rpc_list[-1].get('trip') == trip_id and rpc_list[-1].get('idx_destino', 0) >= ultimo_idx_destino:
                        tramo = rpc_list.pop()
                        if not tramo.get('ruteo_fallido', True):
                            tramo_malo = tramo 
                    
                    if tramo_malo is None:
                        strikes = 99 
                    else:
                        if nodo_final_anterior is not None:
                            nodos_envenenados.add(nodo_final_anterior)
                        
                        origen_idx = tramo_malo['idx_origen']
                        
                        if len(rpc_list) > 0 and rpc_list[-1].get('trip') == trip_id:
                            nodo_final_anterior = rpc_list[-1].get('nodo_final')
                        else:
                            nodo_final_anterior = None
                        
                        strikes = 0
                        continue        
            
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
                    'idx_origen': origen_idx,       
                    'idx_destino': destino_idx,     
                    'nodo_final': nodo_final_anterior 
                })
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
                    'idx_origen': origen_idx,       
                    'idx_destino': destino_idx,     
                    'nodo_final': nodo_final_anterior 
                })
                destino_idx += salto_dinamico
                
            trip_anterior = trip_id
            continue

        # Éxito Topológico
        for i in range(len(best_route)-1):
            u, v = best_route[i], best_route[i+1]
            edge_data = G_actual[u][v][0] 
            
            l_row = edge_data.get('length', 0)
            t_ideal = edge_data.get('travel_time', 1) 
            time_alloc = time_real * (t_ideal / best_time_calc) if best_time_calc > 0 else (time_real / (len(best_route)-1))
            
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
                'idx_origen': origen_idx,       
                'idx_destino': destino_idx,     
                'nodo_final': best_route[-1]    
            })
            current_time = current_time + pd.Timedelta(seconds=time_alloc)
            
        nodo_final_anterior = best_route[-1]
        trip_anterior = trip_id
        
        origen_idx = destino_idx
        destino_idx += 1
        strikes = 0

    res_df = pd.DataFrame(rpc_list)
    if res_df.empty:
        return pd.DataFrame(columns=columnas_esperadas)
    
    for col in columnas_esperadas:
        if col not in res_df.columns:
            res_df[col] = None

    if 'local_timestamp' in res_df.columns:
        res_df['local_timestamp'] = pd.to_datetime(res_df['local_timestamp'], errors='coerce')
        res_df['kepler_time'] = res_df['local_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
    columnas_a_borrar = ['idx_origen', 'idx_destino', 'nodo_final']
    res_df.drop(columns=[c for c in columnas_a_borrar if c in res_df.columns], inplace=True, errors='ignore')

    res_df = _finalize_routing_contract(res_df)
    return res_df

