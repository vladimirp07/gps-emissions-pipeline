import numpy as np
import pandas as pd
import geopandas as gpd


LOCAL_TIMEZONE = "America/Monterrey"


def _coverage_boundary(value, timezone=LOCAL_TIMEZONE):
    """Normalize a supplied acquisition boundary to the production timezone.

    Raw naive boundaries follow the production input contract and represent
    UTC.  Aware boundaries retain their declared instant.
    """
    if value is None:
        return None
    boundary = pd.Timestamp(value)
    if boundary.tzinfo is None:
        boundary = boundary.tz_localize("UTC")
    return boundary.tz_convert(timezone)


def classify_local_day_coverage(dates, coverage_start=None, coverage_end=None,
                                timezone=LOCAL_TIMEZONE):
    """Classify local dates against a half-open acquisition interval.

    Completeness depends only on declared input coverage, never on how often a
    user emitted pings.  Unknown coverage is retained by the caller.
    """
    start = _coverage_boundary(coverage_start, timezone)
    end = _coverage_boundary(coverage_end, timezone)
    if start is not None and end is not None and end <= start:
        raise ValueError("coverage_end must be later than coverage_start")

    statuses = {}
    for date in sorted(set(dates)):
        day_start = pd.Timestamp(date).tz_localize(timezone)
        day_end = (pd.Timestamp(date) + pd.DateOffset(days=1)).tz_localize(timezone)
        partial_start = start is not None and day_start < start < day_end
        partial_end = end is not None and day_start < end < day_end
        if partial_start and partial_end:
            status = "partial_start_end"
        elif partial_start:
            status = "partial_start"
        elif partial_end:
            status = "partial_end"
        elif start is not None and end is not None and start <= day_start and end >= day_end:
            status = "complete"
        else:
            status = "unknown"
        statuses[date] = status
    return statuses

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia Haversine en kilómetros de forma vectorizada.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return 6371.0 * c  

def delete_ids_with_few_rows(df, id_list, threshold=2):
    """
    Filtra los IDs de usuario que tienen menos registros que el umbral definido.
    """
    counts = df['caid'].value_counts()
    keep_ids = counts[counts >= threshold].index
    return df[df['caid'].isin(keep_ids)].copy()

def preprocess_gps_frame(
    df, user_ids=None, num_users=None, *, coverage_start=None,
    coverage_end=None, timezone=LOCAL_TIMEZONE,
):
    """Apply the production GPS cleaning rules to an in-memory frame.

    ``user_ids`` is the integration point used by the preprocessing workflow.
    Keeping the externally supplied user list outside the methodological
    transformations preserves the legacy first-N behavior of
    :func:`preprocess_gps_data`.
    """
    df = df.copy()

    if user_ids is not None:
        usuarios_unicos = list(user_ids)
        df = df[df['caid'].isin(usuarios_unicos)].copy()
    elif num_users is not None:
        usuarios_unicos = list(df['caid'].unique()[:num_users])
        df = df[df['caid'].isin(usuarios_unicos)].copy()
    else:
        usuarios_unicos = list(df['caid'].unique())

    if df.empty:
        return df, usuarios_unicos

    col_tiempo = 'utc_timestamp' if 'utc_timestamp' in df.columns else 'date'
    try:
        df[col_tiempo] = pd.to_datetime(df[col_tiempo], unit='s')
    except (ValueError, TypeError):
        df[col_tiempo] = pd.to_datetime(df[col_tiempo])

    # Production input contract: raw timestamps represent UTC. Naive values
    # are therefore localized to UTC before conversion to local civil time.
    if df[col_tiempo].dt.tz is None:
        df[col_tiempo] = df[col_tiempo].dt.tz_localize('UTC').dt.tz_convert(timezone)
    else:
        df[col_tiempo] = df[col_tiempo].dt.tz_convert(timezone)

    df[col_tiempo] = df[col_tiempo].dt.tz_localize(None)
    df = df.rename(columns={col_tiempo: 'local_timestamp'})

    print(f"[Preprocessing] Records in the selected time window: {len(df):,}", flush=True)
    df = df.sort_values(by=['caid', 'local_timestamp'])

    # Downsampling de 10 segundos con last para coordenadas (evitar sintéticas) y mediana para velocidad/distancia
    df['time_bucket'] = df['local_timestamp'].dt.floor('10s')
    agg_dict = {col: 'median' if col in ['Speed [km/h]', 'dis lineal [m]'] else 'last'
                for col in df.columns if col not in ['caid', 'time_bucket']}
    df = df.groupby(['caid', 'time_bucket'], as_index=False).agg(agg_dict)
    df = df.drop(columns=['time_bucket'])

    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

    # Geofencing Monterrey (48km)
    ref_lat, ref_lon = 25.6866, -100.3161
    threshold_distance = 48.0
    df['Distance'] = haversine_vectorized(ref_lat, ref_lon, df['latitude'].values, df['longitude'].values)
    df = df[df['Distance'] < threshold_distance]
    df = delete_ids_with_few_rows(df, df['caid'].unique(), threshold=2)

    # Retain unknown days. Only declared acquisition/slicing boundaries may
    # prove that a local edge day is truncated.
    df['date'] = df['local_timestamp'].dt.date
    day_statuses = classify_local_day_coverage(
        df['date'].unique(), coverage_start=coverage_start,
        coverage_end=coverage_end, timezone=timezone,
    )
    df['day_completeness_status'] = df['date'].map(day_statuses)
    excluded_statuses = {'partial_start', 'partial_end', 'partial_start_end'}
    df = df[~df['day_completeness_status'].isin(excluded_statuses)].copy()

    df = df.sort_values(by=['caid', 'local_timestamp'])
    df['lat_end'] = df.groupby(['caid', 'date'])['latitude'].shift(-1)
    df['lon_end'] = df.groupby(['caid', 'date'])['longitude'].shift(-1)
    df['local_timestamp_end'] = df.groupby(['caid', 'date'])['local_timestamp'].shift(-1)
    df = df.dropna(subset=['lat_end', 'lon_end', 'local_timestamp_end'])

    # Re-cálculo de vectores de distancia física y velocidad lineal
    df['dis lineal [m]'] = haversine_vectorized(
        df['latitude'].values, df['longitude'].values,
        df['lat_end'].values, df['lon_end'].values
    ) * 1000.0

    df['travel time_sec'] = (df['local_timestamp_end'] - df['local_timestamp']).dt.total_seconds()
    df['Speed [km/h]'] = np.where(
        df['travel time_sec'] > 0,
        (df['dis lineal [m]'] / 1000.0) / (df['travel time_sec'] / 3600.0),
        0.0
    )
    # Filtro extremo de seguridad física
    df = df[df['Speed [km/h]'] <= 150.0]
    df = df.drop(columns=['lat_end', 'lon_end', 'local_timestamp_end', 'travel time_sec', 'Distance'], errors='ignore')
    df = df.reset_index(drop=True)
    df.attrs['day_completeness'] = {
        'coverage_start_local': str(_coverage_boundary(coverage_start, timezone)) if coverage_start is not None else None,
        'coverage_end_local': str(_coverage_boundary(coverage_end, timezone)) if coverage_end is not None else None,
        'status_by_date': {str(date): status for date, status in day_statuses.items()},
        'excluded_dates': [str(date) for date, status in day_statuses.items() if status in excluded_statuses],
    }
    return df, usuarios_unicos


def preprocess_gps_data(file_path, num_users=3):
    """
    Preprocesa los datos GPS originales aplicando huso horario local de Monterrey,
    downsampling estabilizado de 10s, geofencing regional y cálculo de vectores físicos.
    """
    print("[Preprocessing] Loading and preprocessing raw GPS data...", flush=True)
    df = pd.read_parquet(file_path)
    return preprocess_gps_frame(df, num_users=num_users)

def assign_trips(df):
    """
    Particiona la trayectoria en paradas estacionarias (trip <= 0)
    y fases de viaje/movimiento activo (trip > 0) usando un filtro paso-bajo.
    """
    df = df.copy()
    df = df.sort_values(by=['caid', 'local_timestamp'])
    
    if 'date' not in df.columns:
        df['date'] = df['local_timestamp'].dt.date
        
    # Inicialización de distancias y velocidad si no existieran
    if 'dis lineal [m]' not in df.columns or 'Speed [km/h]' not in df.columns:
        df['lat_end'] = df.groupby(['caid', 'date'])['latitude'].shift(-1)
        df['lon_end'] = df.groupby(['caid', 'date'])['longitude'].shift(-1)
        df['ts_end'] = df.groupby(['caid', 'date'])['local_timestamp'].shift(-1)
        
        mask_next = df['lat_end'].notna()
        if 'dis lineal [m]' not in df.columns:
            df['dis lineal [m]'] = 0.0
            if mask_next.any():
                df.loc[mask_next, 'dis lineal [m]'] = haversine_vectorized(
                    df.loc[mask_next, 'latitude'].values, df.loc[mask_next, 'longitude'].values,
                    df.loc[mask_next, 'lat_end'].values, df.loc[mask_next, 'lon_end'].values
                ) * 1000.0
        
        if 'Speed [km/h]' not in df.columns:
            df['Speed [km/h]'] = 0.0
            dt_sec = (df['ts_end'] - df['local_timestamp']).dt.total_seconds()
            mask_speed = mask_next & (dt_sec > 0)
            if mask_speed.any():
                df.loc[mask_speed, 'Speed [km/h]'] = (df.loc[mask_speed, 'dis lineal [m]'] / 1000.0) / (dt_sec[mask_speed] / 3600.0)
        
        df = df.drop(columns=['lat_end', 'lon_end', 'ts_end'], errors='ignore')

    if 'travel time' not in df.columns:
        df['travel time'] = df.groupby(['caid', 'date'])['local_timestamp'].shift(-1) - df['local_timestamp']
        
    df['Speed [km/h]'] = df['Speed [km/h]'].fillna(0.0)
    df['dis lineal [m]'] = df['dis lineal [m]'].fillna(0.0)
    df['travel time'] = df['travel time'].fillna(pd.Timedelta(seconds=0))

    speeds = df['Speed [km/h]'].tolist()
    travel_times = df['travel time'].dt.total_seconds().tolist() 
    distances = df['dis lineal [m]'].tolist()
    users = df['caid'].tolist()

    trips = []
    trip_counter = 0
    stop_counter = 0
    
    STOP_SPEED = 3.0  # km/h
    STOP_TIME = 300   # 5 minutos acumulados
    T_MAX_TELEPORT = 1800 # 30 minutos sin datos indican corte/parada forzada
    accumulated_stop_time = 0
    current_user = None

    for i in range(len(speeds)):
        if i == 0 or users[i] != current_user:
            current_user = users[i]
            trip_counter = 0
            stop_counter = 0
            accumulated_stop_time = 0
            previous_trip = None
        else:
            previous_trip = trips[-1]
        speed = speeds[i]
        tt = travel_times[i]
        dist = distances[i]

        # 1. Filtro Anti-Teletransportación
        if tt > T_MAX_TELEPORT:
            stop_counter -= 1
            current_trip = stop_counter
            accumulated_stop_time = 0
        else:
            # 2. Quietud
            if speed < STOP_SPEED and dist < 100:
                accumulated_stop_time += tt
                if accumulated_stop_time > STOP_TIME:
                    if previous_trip is None or previous_trip > 0:
                        stop_counter -= 1
                        current_trip = stop_counter
                    else:
                        current_trip = previous_trip
                else:
                    if previous_trip is None:
                        # A short quiet spell at the beginning is provisionally
                        # part of a mobile run.  It must consume the positive ID
                        # just like a movement-start does; otherwise a later
                        # stop/gap followed by movement reuses ``trip == 1`` and
                        # groupby("trip") silently joins discontiguous runs.
                        trip_counter += 1
                        current_trip = trip_counter
                    else:
                        current_trip = previous_trip
            # 3. Movimiento
            else:
                accumulated_stop_time = 0 
                if previous_trip is None or previous_trip < 0:
                    trip_counter += 1
                    current_trip = trip_counter
                else:
                    current_trip = previous_trip
                    
        trips.append(current_trip)
        
    df['trip'] = trips
    return df

def apply_spatial_filter(df, min_dist_m=15.0):
    """
    Elimina pings sucesivos cuya distancia Haversine al ping guardado anterior
    sea menor a min_dist_m metros.
    """
    if len(df) <= 2:
        return df.copy()
    
    kept_indices = [0]
    last_idx = 0
    lats = df['latitude'].to_numpy()
    lons = df['longitude'].to_numpy()
    
    for i in range(1, len(df) - 1):
        d = haversine_vectorized(lats[last_idx], lons[last_idx], lats[i], lons[i]) * 1000.0
        if d >= min_dist_m:
            kept_indices.append(i)
            last_idx = i
            
    kept_indices.append(len(df) - 1)
    kept_indices = sorted(list(set(kept_indices)))
    
    return df.iloc[kept_indices].copy().reset_index(drop=True)
