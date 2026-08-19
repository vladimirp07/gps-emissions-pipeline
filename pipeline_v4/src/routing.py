import os
import time
import pickle
import builtins
from dataclasses import dataclass
from threading import Lock
import pandas as pd
import numpy as np
import networkx as nx
import geopandas as gpd
from pyproj import Transformer
from shapely.ops import substring
from shapely import wkt
from shapely.geometry import Point, LineString
from .segmentation import haversine_vectorized
from .pipeline_contracts import ROUTING_REQUIRED_COLUMNS
from .endpoint_routing import (
    ENDPOINT_PATCH_VERSION,
    attach_real_edge_endpoint_segments,
    expand_endpoint_candidates,
    explicitly_reject_failed_geometry,
    normalized_mode,
)

# --- TRANSFORMATION TRANSFORMERS ---
TRANSFORMER_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)
TRANSFORMER_TO_WGS = Transformer.from_crs("EPSG:32614", "EPSG:4326", always_xy=True)


@dataclass(frozen=True)
class CandidateEdgeIndex:
    """Lightweight candidate table with a spatial index bound to its lifetime."""

    edges: gpd.GeoDataFrame
    source_positions: np.ndarray
    spatial_index: object
    build_seconds: float
    resource_key: str


def build_candidate_edge_index(edges_gdf, *, resource_key):
    """Build the nearest-edge search resource once without changing row order."""
    geometry_column = edges_gdf.geometry.name
    required = ["u", "v", geometry_column]
    missing = [column for column in required if column not in edges_gdf.columns]
    if missing:
        raise ValueError(f"Candidate edge table is missing columns: {missing}")
    candidate_edges = edges_gdf.loc[:, required].copy()
    position_dtype = np.int32 if len(candidate_edges) <= np.iinfo(np.int32).max else np.int64
    source_positions = np.arange(len(candidate_edges), dtype=position_dtype)
    started = time.perf_counter()
    spatial_index = candidate_edges.sindex
    build_seconds = time.perf_counter() - started
    return CandidateEdgeIndex(
        edges=candidate_edges,
        source_positions=source_positions,
        spatial_index=spatial_index,
        build_seconds=build_seconds,
        resource_key=str(resource_key),
    )


def _finalize_routing_contract(res_df):
    """Normaliza la salida de cualquier variante del router activo."""
    if res_df.empty:
        return res_df
    res_df = res_df.copy()
    res_df['local_timestamp'] = pd.to_datetime(res_df['local_timestamp'], errors='coerce')
    res_df['physical_trip_id'] = res_df['caid'].astype(str) + '_' + res_df['trip'].astype(str)
    res_df['network_hypothesis'] = res_df['modo_transporte'].astype(str)
    res_df['duration_s'] = res_df.groupby('physical_trip_id')['local_timestamp'].diff().dt.total_seconds().fillna(0.0)
    if 'snap_distance_m' not in res_df:
        res_df['snap_distance_m'] = np.nan
    if 'snapping_quality_status' not in res_df:
        res_df['snapping_quality_status'] = 'not_recorded_by_legacy_router'
    res_df['routing_status'] = np.where(res_df['ruteo_fallido'].fillna(False), 'fallback_or_failed', 'success')
    return res_df

def get_candidates_vectorized(edges_gdf, gdf_points, buffer_m=150, max_cands=12):
    """
    Asignación espacial VECTORIZADA de alto rendimiento.

    Known routing candidate-generation limitation: ``sjoin_nearest`` retains
    the nearest edge(s), including exact ties; ``max_cands`` only caps those
    returned rows and is intentionally not a true K-nearest-neighbor request.
    Production behavior is frozen pending a separately calibrated methodology.

    Only ``u``, ``v`` and the active geometry are carried into the spatial
    join.  Keeping unrelated OSM edge attributes here previously multiplied
    memory use without affecting candidate IDs or distances.
    """
    if isinstance(edges_gdf, CandidateEdgeIndex):
        candidate_edges = edges_gdf.edges
    else:
        geometry_column = edges_gdf.geometry.name
        required = ["u", "v", geometry_column]
        missing = [column for column in required if column not in edges_gdf.columns]
        if missing:
            raise ValueError(f"Candidate edge table is missing columns: {missing}")
        # Compatibility path for isolated API callers. Production supplies a
        # CandidateEdgeIndex, so this per-call view and index build are avoided.
        candidate_edges = edges_gdf.loc[:, required]
    geometry_column = candidate_edges.geometry.name
    required = ["u", "v", geometry_column]
    missing = [column for column in required if column not in candidate_edges.columns]
    if missing:
        raise ValueError(f"Candidate edge table is missing columns: {missing}")
    candidate_points = gdf_points.loc[:, [gdf_points.geometry.name]]

    # 1. Spatial Join Nearest (Búsqueda optimizada por R-Tree)
    joined = gpd.sjoin_nearest(
        candidate_points,
        candidate_edges,
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
    
    # 4. Expandir u/v en el mismo orden row-major del ensamblado histórico.
    # ``tolist`` preserves native Python scalar types, duplicates, and tie order.
    point_index = joined.index.to_numpy().repeat(2)
    candidate_ids = joined[['u', 'v']].to_numpy(copy=False).reshape(-1).tolist()
    candidate_distances = joined['dist_exacta'].to_numpy(copy=False).repeat(2).tolist()
    ids_grouped = pd.Series(candidate_ids, index=point_index, dtype=object).groupby(
        level=0, sort=False
    ).agg(list)
    distances_grouped = pd.Series(candidate_distances, index=point_index, dtype=object).groupby(
        level=0, sort=False
    ).agg(list)

    # 5. Reindexar sin perder puntos GPS sin candidato.
    ids = ids_grouped.reindex(gdf_points.index)
    distances = distances_grouped.reindex(gdf_points.index)
    missing_ids = ids.isna()
    missing_distances = distances.isna()
    if missing_ids.any():
        ids.loc[missing_ids] = pd.Series(
            [[] for _ in range(int(missing_ids.sum()))], index=ids.index[missing_ids], dtype=object
        )
    if missing_distances.any():
        distances.loc[missing_distances] = pd.Series(
            [[] for _ in range(int(missing_distances.sum()))],
            index=distances.index[missing_distances], dtype=object,
        )
    return ids, distances


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
            pares.append({
                'origen': id_org,
                'destino': id_dest,
                'costo_espacial': dist_org + dist_dest,
                # Present only for calibration IDs; native production node IDs
                # have no rank metadata and retain the exact historical path.
                'origin_candidate_rank': getattr(id_org, 'edge_rank', None),
                'destination_candidate_rank': getattr(id_dest, 'edge_rank', None),
            })
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
        # Metro topology bypass.
        # =============================================================

        # Snap Metro endpoints to track geometry.
        if str(modo_actual).title() == 'Metro':
            lat1, lon1 = lats_arr[origen_idx], lons_arr[origen_idx]
            lat2, lon2 = lats_arr[destino_idx], lons_arr[destino_idx]
            
            time_real = (timestamps_list[destino_idx] - timestamps_list[origen_idx]).total_seconds()
            
            try:
                # 1. Snap the observed endpoints to the Metro track.
                geom_wkt, distancia_m = _obtener_tramo_metro(lon1, lat1, lon2, lat2, geometry_metro)
                vel_metro = (distancia_m / 1000.0) / (time_real / 3600.0) if time_real > 0 else 0
                flag = 'Metro_Topologico (Track Snapped)'
            except Exception:
                # 2. Fall back to a straight line if track snapping fails.
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
            physics_factor = float(os.environ.get('PHYSICS_FACTOR', '2.0'))
            v_techo = limite_calle * physics_factor if modo_actual not in ['Caminar', 'Parada'] else max_speed_kmh
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
                        break  # Reject the route when a subsegment violates the speed ceiling.
                    
                    
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

    res_df = _finalize_routing_contract(res_df)
        
    # NUEVO: Limpieza final - Removemos las columnas de tracking que usamos para el rollback
    columnas_a_borrar = ['idx_origen', 'idx_destino', 'nodo_final']
    res_df.drop(columns=[c for c in columnas_a_borrar if c in res_df.columns], inplace=True, errors='ignore')

    res_df = _finalize_routing_contract(res_df)
    return res_df


_EDGE_CACHE = {}
_EDGE_CACHE_LOCK = Lock()

def get_edge_cache(G):
    g_id = id(G)
    if g_id not in _EDGE_CACHE:
        with _EDGE_CACHE_LOCK:
            if g_id not in _EDGE_CACHE:
                cache = {}
                for u in G.nodes():
                    for v in G[u]:
                        if 0 in G[u][v]:
                            d = G[u][v][0]
                            cache[(u, v)] = (d.get('length', 0.0), d.get('travel_time', 1.0))
                        elif G[u][v]:
                            first_key = list(G[u][v].keys())[0]
                            d = G[u][v][first_key]
                            cache[(u, v)] = (d.get('length', 0.0), d.get('travel_time', 1.0))
                _EDGE_CACHE[g_id] = cache
    return _EDGE_CACHE[g_id]

def complete_route_v1_optimized(id, registros_person, 
                G_drive, G_walk,
                ig_drive, ig_walk,      
                map_drive, map_walk,    
                geometry_metro=None,
                *,
                max_lookahead_skipped_pings=None):
    """
    Optimized V1-compatible routing implementation.
    
    Uses batched single-source/multi-target iGraph queries, a flat edge-attribute
    cache, and reuse of path weight and length calculations during physical
    validation and final row assembly.
    """ 
    from collections import OrderedDict, defaultdict
    rpc_list = []
    bounded_recovery = max_lookahead_skipped_pings is not None

    def observed_path_distance_m(start_idx, end_idx):
        if end_idx <= start_idx:
            return 0.0
        return float(sum(
            haversine_vectorized(
                lats_arr[idx], lons_arr[idx], lats_arr[idx + 1], lons_arr[idx + 1]
            ) * 1000.0
            for idx in range(start_idx, end_idx)
        ))

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
    nodos_envenenados = set() 

    # Scope is one trip/hypothesis.  The full key records every routing input
    # even though network and weight are constant within the current call.
    path_cache = OrderedDict()
    path_cache_limit = max(256, min(4096, n_registros * 32))
    path_cache_hits = 0
    path_cache_misses = 0
    path_cache_evictions = 0
    def fetch_path_records(pairs, ig_actual, map_actual, cache_actual):
        nonlocal path_cache_hits, path_cache_misses, path_cache_evictions
        org_to_dest = defaultdict(list)
        for par in pairs:
            u, v = par['origen'], par['destino']
            if u != v:
                org_to_dest[u].append(v)
        records = {}
        for u, dest_list in org_to_dest.items():
            try:
                if u not in map_actual:
                    continue
                ig_u = map_actual[u]
                valid_dests = [v for v in dest_list if v in map_actual]
                if not valid_dests:
                    continue
                network_key = (builtins.id(ig_actual), bool(ig_actual.is_directed()), 'length')
                missing = []
                missing_keys = set()
                for v in valid_dests:
                    key = (*network_key, ig_u, map_actual[v])
                    if key in path_cache:
                        path_cache_hits += 1
                        path_cache.move_to_end(key)
                    elif key not in missing_keys:
                        path_cache_misses += 1
                        missing.append(v)
                        missing_keys.add(key)
                if missing:
                    paths_ig = ig_actual.get_shortest_paths(
                        ig_u, to=[map_actual[v] for v in missing],
                        weights='length', output='vpath',
                    )
                    for v, path_ig in zip(missing, paths_ig):
                        key = (*network_key, ig_u, map_actual[v])
                        record = None
                        if len(path_ig) >= 2:
                            route = [ig_actual.vs[nx_id]['_nx_name'] for nx_id in path_ig]
                            distancia_m = 0.0
                            travel_time = 0.0
                            valid_path = True
                            tramos_list = []
                            for n in range(len(route) - 1):
                                edge_vals = cache_actual.get((route[n], route[n + 1]))
                                if edge_vals is None:
                                    valid_path = False
                                    break
                                distancia_m += edge_vals[0]
                                travel_time += edge_vals[1]
                                tramos_list.append((route[n], route[n + 1], edge_vals[0], edge_vals[1]))
                            if valid_path:
                                record = (route, tramos_list, distancia_m, travel_time)
                        path_cache[key] = record
                        path_cache.move_to_end(key)
                        if len(path_cache) > path_cache_limit:
                            path_cache.popitem(last=False)
                            path_cache_evictions += 1
                for v in valid_dests:
                    key = (*network_key, ig_u, map_actual[v])
                    record = path_cache.get(key)
                    if record is not None:
                        path_cache.move_to_end(key)
                        records[(u, v)] = record
            except Exception:
                continue
        return records

    # Build each network cache only if this hypothesis actually uses it.
    cache_drive = None
    cache_walk = None

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
        if es_peaton:
            if cache_walk is None:
                cache_walk = get_edge_cache(G_walk)
            cache_actual = cache_walk
        else:
            if cache_drive is None:
                cache_drive = get_edge_cache(G_drive)
            cache_actual = cache_drive
        
        lat1, lon1 = lats_arr[origen_idx], lons_arr[origen_idx]
        lat2, lon2 = lats_arr[destino_idx], lons_arr[destino_idx]
        
        distancia_haversine_km = haversine_vectorized(lat1, lon1, lat2, lon2)

        # Escudo 1: Spatial Skip (Burbuja de Incertidumbre)
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
            
        # Limpiamos la lista negra si cambiamos de viaje
        if trip_id != trip_anterior:
            nodos_envenenados.clear()

        # Generamos la lista de candidatos
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
        
        best_route = None
        best_tramos = None
        best_time_calc = 0.0
        best_origin_candidate_rank = None
        best_destination_candidate_rank = None
        ruta_exitosa = False
        flag_auditoria = 'None'
        candidatos_validos = []
        rutas_rechazadas_por_velocidad = 0 
        
        if pares_evaluacion:
            bulk_paths = fetch_path_records(
                pares_evaluacion, ig_actual, map_actual, cache_actual
            )
            # Evaluate in the unchanged global candidate order.
            for par in pares_evaluacion:
                u, v = par['origen'], par['destino']
                record = bulk_paths.get((u, v))
                if record is None: continue
                route, tramos_list, distancia_m, time_calc_best = record

                distancia_grafo_km = distancia_m / 1000.0
                vel_ruta_kmh = distancia_grafo_km / delta_t_horas if delta_t_horas > 0 else 0

                # Nivel 1: Lazy Selection
                limite_distancia = max(distancia_haversine_km * 1.4, distancia_haversine_km + 0.25)
                if (distancia_grafo_km <= limite_distancia and
                        vel_ruta_kmh <= limites_kmhlazy.get(modo_actual, 60.0)):
                    best_route = route
                    best_tramos = tramos_list
                    best_time_calc = time_calc_best
                    best_origin_candidate_rank = par['origin_candidate_rank']
                    best_destination_candidate_rank = par['destination_candidate_rank']
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
                        'ruta': route, 
                        'tramos': tramos_list,
                        'dist': distancia_grafo_km, 
                        'time': time_calc_best, 
                        'vel': vel_ruta_kmh,
                        'origin_candidate_rank': par['origin_candidate_rank'],
                        'destination_candidate_rank': par['destination_candidate_rank'],
                    })
                else:
                    rutas_rechazadas_por_velocidad += 1 

        if not ruta_exitosa and candidatos_validos:
            candidatos_validos.sort(key=lambda x: x['dist'])
            ganador = candidatos_validos[0]
            best_route = ganador['ruta']
            best_tramos = ganador['tramos']
            best_time_calc = ganador['time']
            best_origin_candidate_rank = ganador['origin_candidate_rank']
            best_destination_candidate_rank = ganador['destination_candidate_rank']
            flag_auditoria = 'Nivel2_Exhaustivo'
            ruta_exitosa = True

        # Escudo 2 ESTRICTO: Amnesia Temporal (Física de Subsegmentos)
        velocidad_excedida_subsegmento = False
        if ruta_exitosa:
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
            
            distancia_m_best = sum(t[2] for t in best_tramos)
            vel_final_ruta = (distancia_m_best / 1000.0) / delta_t_horas if delta_t_horas > 0 else 0
            
            if vel_final_ruta > techo_final:
                velocidad_excedida_subsegmento = True
            else:
                for u_test, v_test, l_test, t_ideal_test in best_tramos:
                    t_alloc_test = time_real * (t_ideal_test / best_time_calc) if best_time_calc > 0 else (time_real / len(best_tramos))
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
            
            # ROLLBACK LOGIC (Amnesia Hacia Atrás)
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
            
            proposed_destination = destino_idx + salto_dinamico
            proposed_skipped_pings = max(proposed_destination - origen_idx - 1, 0)
            recovery_bound_exceeded = (
                bounded_recovery
                and proposed_skipped_pings > int(max_lookahead_skipped_pings)
            )
            definitive_failure = (
                proposed_destination >= n_registros
                or strikes > 20
                or recovery_bound_exceeded
            )
            if definitive_failure:
                geom_wkt = f'POINT ({lon2} {lat2})'
                unresolved_skipped_pings = max(destino_idx - origen_idx - 1, 0)
                unresolved_elapsed_seconds = float(
                    (timestamps_list[destino_idx] - timestamps_list[origen_idx]).total_seconds()
                )
                unresolved_observed_distance_m = observed_path_distance_m(origen_idx, destino_idx)
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
                    'flag_auditoria': (
                        f'Transition_Unresolved_Lookahead_Bound ({razon})'
                        if recovery_bound_exceeded else f'Amnesia_Definitiva ({razon})'
                    ),
                    'idx_origen': origen_idx,       
                    'idx_destino': destino_idx,     
                    'nodo_final': nodo_final_anterior,
                    'routing_event': 'transition_unresolved' if bounded_recovery else 'legacy_definitive_failure',
                    'lookahead_skipped_pings': unresolved_skipped_pings,
                    'lookahead_elapsed_seconds': unresolved_elapsed_seconds,
                    'lookahead_observed_distance_m': unresolved_observed_distance_m,
                    'lookahead_origin_ping': origen_idx,
                    'lookahead_resume_ping': destino_idx,
                    'lookahead_failure_reason': razon,
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
                    'nodo_final': nodo_final_anterior,
                    'routing_event': 'lookahead_skip' if bounded_recovery else None,
                    'lookahead_skipped_pings': max(destino_idx - origen_idx - 1, 0),
                    'lookahead_elapsed_seconds': float(
                        (timestamps_list[destino_idx] - timestamps_list[origen_idx]).total_seconds()
                    ),
                    'lookahead_observed_distance_m': observed_path_distance_m(origen_idx, destino_idx),
                    'lookahead_origin_ping': origen_idx,
                    'lookahead_resume_ping': destino_idx,
                    'lookahead_failure_reason': razon,
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

            output_row = {
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
            }
            if (best_origin_candidate_rank is not None or
                    best_destination_candidate_rank is not None):
                output_row['selected_origin_candidate_rank'] = best_origin_candidate_rank
                output_row['selected_destination_candidate_rank'] = best_destination_candidate_rank
                output_row['routing_origin_ping'] = origen_idx
                output_row['routing_destination_ping'] = destino_idx
            rpc_list.append(output_row)
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
        
    if bounded_recovery:
        unresolved = res_df.get(
            'routing_event', pd.Series(None, index=res_df.index)
        ).eq('transition_unresolved')
        res_df['route_component_id'] = unresolved.cumsum().shift(fill_value=0).astype(int) + 1
        # The unresolved marker belongs to the uncovered interval, not to a
        # valid route component. Keep its preceding component for ordering;
        # valid rows after it receive the next component ID.
        res_df['max_lookahead_skipped_pings'] = int(max_lookahead_skipped_pings)
    else:
        # Preserve the historical V1 output schema exactly. These audit fields
        # belong only to the bounded recovery contract activated by V2.
        v2_recovery_columns = [
            'routing_event', 'lookahead_skipped_pings',
            'lookahead_elapsed_seconds', 'lookahead_observed_distance_m',
            'lookahead_origin_ping', 'lookahead_resume_ping',
            'lookahead_failure_reason',
        ]
        res_df.drop(
            columns=[column for column in v2_recovery_columns if column in res_df.columns],
            inplace=True, errors='ignore',
        )

    columnas_a_borrar = ['idx_origen', 'idx_destino', 'nodo_final']
    res_df.drop(columns=[c for c in columnas_a_borrar if c in res_df.columns], inplace=True, errors='ignore')

    res_df = _finalize_routing_contract(res_df)
    res_df.attrs["shortest_path_cache"] = {
        "hits": path_cache_hits,
        "misses": path_cache_misses,
        "evictions": path_cache_evictions,
        "max_entries": path_cache_limit,
        "final_entries": len(path_cache),
    }
    return res_df


_IG_EDGE_CACHE = {}
_IG_NODE_NAMES = {}

def get_ig_cache(G, ig_actual, map_actual):
    g_id = id(G)
    if g_id not in _IG_EDGE_CACHE:
        cache_actual = get_edge_cache(G)
        ig_cache_actual = {}
        for (u, v), vals in cache_actual.items():
            if u in map_actual and v in map_actual:
                ig_cache_actual[(map_actual[u], map_actual[v])] = vals
        _IG_EDGE_CACHE[g_id] = ig_cache_actual
        
        ig_node_names = [v['_nx_name'] for v in ig_actual.vs]
        _IG_NODE_NAMES[g_id] = ig_node_names
        
    return _IG_EDGE_CACHE[g_id], _IG_NODE_NAMES[g_id]

def complete_route_v2_progressive_legacy(*args, **kwargs):
    """Compatibility entry point for the archived progressive experiment."""
    import warnings
    warnings.warn(
        "complete_route_v2_progressive_legacy is calibration-only; "
        "production routers are v1 and endpoint-preserving v2.",
        DeprecationWarning,
        stacklevel=2,
    )
    from pipeline_v4.calibration_and_diagnostics.legacy_routing.complete_route_v2_progressive_legacy import (
        complete_route_v2_progressive_legacy as legacy_router,
    )
    return legacy_router(*args, **kwargs)

def complete_route_v2_optimized(
    id,
    registros_person,
    G_drive,
    G_walk,
    ig_drive,
    ig_walk,
    map_drive,
    map_walk,
    geometry_metro,
    edges_drive=None,
    edges_walk=None,
    candidate_edges_drive=None,
    candidate_edges_walk=None,
    incident_edges_drive=None,
    incident_edges_walk=None,
    max_lookahead_skipped_pings=None,
):
    """V1 stable routing plus independently validated endpoint preservation.

    Metro remains byte-methodologically equivalent to V1: the historical
    endpoint patch was not validated for rail and must not attach road edges
    to a Metro hypothesis.  Road and Walking require their real edge table so
    missing endpoint information cannot be silently replaced by placeholders.
    """
    if max_lookahead_skipped_pings is None:
        from pipeline_v4.src import config as production_config
        max_lookahead_skipped_pings = production_config.MAX_LOOKAHEAD_SKIPPED_PINGS
    mode = normalized_mode(registros_person)
    if mode == "metro":
        result = complete_route_v1_optimized(
            id, registros_person, G_drive, G_walk, ig_drive, ig_walk,
            map_drive, map_walk, geometry_metro,
            max_lookahead_skipped_pings=max_lookahead_skipped_pings,
        ).copy()
        if not result.empty:
            result["router_version"] = "v2"
            result["endpoint_patch_version"] = ENDPOINT_PATCH_VERSION
            result["endpoint_start_status"] = "not_applicable_metro"
            result["endpoint_end_status"] = "not_applicable_metro"
        return result

    walking = mode in {"caminar", "walking", "walk"}
    edges = edges_walk if walking else edges_drive
    candidate_edges = candidate_edges_walk if walking else candidate_edges_drive
    incident_edge_index = incident_edges_walk if walking else incident_edges_drive
    if edges is None:
        raise ValueError(
            "complete_route_v2_optimized requires the real edge table for "
            f"{'Walking' if walking else 'Road'} endpoint preservation"
        )
    expanded, endpoint_meta = expand_endpoint_candidates(
        registros_person,
        edges,
        get_candidates_vectorized,
        walking=walking,
        candidate_edges=candidate_edges,
    )
    result = complete_route_v1_optimized(
        id, expanded, G_drive, G_walk, ig_drive, ig_walk,
        map_drive, map_walk, geometry_metro,
        max_lookahead_skipped_pings=max_lookahead_skipped_pings,
    )
    path_cache_stats = result.attrs.get("shortest_path_cache", {}).copy()
    result, _ = attach_real_edge_endpoint_segments(
        result, expanded, edges, incident_edge_index=incident_edge_index,
        projected_endpoint_points=endpoint_meta["_projected_endpoint_points"],
    )
    result = explicitly_reject_failed_geometry(result)
    if not result.empty:
        result["router_version"] = "v2"
        result["endpoint_patch_version"] = ENDPOINT_PATCH_VERSION
        result = _finalize_routing_contract(result)
        result.attrs["shortest_path_cache"] = path_cache_stats
    return result


class RouteHypothesisEvaluator:
    """
    Orquesta la ejecución de complete_route bajo múltiples hipótesis de transporte.
    """
    def __init__(
        self,
        G_drive,
        G_walk,
        ig_drive,
        ig_walk,
        map_drive,
        map_walk,
        geometry_metro,
        *,
        router_version="v1",
        edges_drive=None,
        edges_walk=None,
        candidate_edges_drive=None,
        candidate_edges_walk=None,
        incident_edges_drive=None,
        incident_edges_walk=None,
    ):
        self.G_drive = G_drive
        self.G_walk = G_walk
        self.ig_drive = ig_drive
        self.ig_walk = ig_walk
        self.map_drive = map_drive
        self.map_walk = map_walk
        self.geometry_metro = geometry_metro
        self.router_version = str(router_version).strip().lower()
        self.edges_drive = edges_drive
        self.edges_walk = edges_walk
        self.candidate_edges_drive = candidate_edges_drive
        self.candidate_edges_walk = candidate_edges_walk
        self.incident_edges_drive = incident_edges_drive
        self.incident_edges_walk = incident_edges_walk
        if self.router_version not in {"v1", "v2"}:
            raise ValueError(f"Unsupported ROUTER_VERSION: {router_version}")

    def evaluate(self, id_user, df_trip, mode_candidates):
        hypotheses = {}
        for mode in mode_candidates:
            df_mock = df_trip.copy()
            df_mock['modo_transporte'] = mode
            
            router = complete_route_v2_optimized if self.router_version == "v2" else complete_route_v1_optimized
            kwargs = {}
            if self.router_version == "v2":
                kwargs = {
                    "edges_drive": self.edges_drive,
                    "edges_walk": self.edges_walk,
                    "candidate_edges_drive": self.candidate_edges_drive,
                    "candidate_edges_walk": self.candidate_edges_walk,
                    "incident_edges_drive": self.incident_edges_drive,
                    "incident_edges_walk": self.incident_edges_walk,
                }
            df_routed = router(
                id=id_user,
                registros_person=df_mock,
                G_drive=self.G_drive,
                G_walk=self.G_walk,
                ig_drive=self.ig_drive,
                ig_walk=self.ig_walk,
                map_drive=self.map_drive,
                map_walk=self.map_walk,
                geometry_metro=self.geometry_metro,
                **kwargs,
            )
            if not df_routed.empty:
                hypotheses[mode] = df_routed
        return hypotheses
