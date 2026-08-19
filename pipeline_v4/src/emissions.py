import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from .pipeline_contracts import validate_emissions_input
from . import config

def calculate_emissions(df_rutas_in, file_moves_path):
    """
    Match routed trajectory segments to MOVES rates and calculate emission
    density and total mass for every configured pollutant.
    """
    df_rutas = df_rutas_in.copy()
    
    if df_rutas.empty:
        return df_rutas

    contract_errors = validate_emissions_input(df_rutas)
    if contract_errors:
        raise ValueError(f"Incompatible emissions input: {contract_errors}")
        
    print("[Emissions] Starting emissions calculation...", flush=True)
    if config.EMISSION_RATE_DISTANCE_UNIT != 'g/km':
        raise RuntimeError("Production V4 requires explicit g/km rates.")
    
    COLUMNA_MODO = 'modo_transporte' 
    COLUMNA_VELOCIDAD = 'Speed [km/h]'
    COLUMNA_DISTANCIA = 'distance_m'
    
    # Load the MOVES rate table.
    df_emisiones_raw = pd.read_parquet(file_moves_path)
    
    # Criteria pollutants and greenhouse-gas fields defined by the contract.
    POLLUTANTS = ['CO', 'CO2', 'CO2_Equiv', 'HC', 'NOx', 'PM10', 'PM25']
    
    OSM_TO_MOVES_ROAD = {
        'motorway': 4, 'motorway_link': 4, 'trunk': 4, 'trunk_link': 4, 
        'primary': 4, 'primary_link': 4, 'secondary': 5, 'secondary_link': 5, 
        'tertiary': 5, 'tertiary_link': 5, 'unclassified': 5, 'residential': 5,
        'routing_error': 5, 'parada_inactiva': 5
    }
    
    # 1. Prepare lookup variables.
    df_rutas['orden_original'] = range(len(df_rutas))
    df_rutas['local_timestamp'] = pd.to_datetime(df_rutas['local_timestamp'])
    
    # MOVES month, hour, and day type (5=weekday, 2=weekend).
    df_rutas['Month'] = df_rutas['local_timestamp'].dt.month
    df_rutas['Hour'] = df_rutas['local_timestamp'].dt.hour + 1
    df_rutas['Day'] = np.where(df_rutas['local_timestamp'].dt.dayofweek < 5, 5, 2)
    
    mapped_road = df_rutas['highway'].map(OSM_TO_MOVES_ROAD)
    df_rutas['road_lookup_status'] = np.where(mapped_road.isna(), 'default_road_5', 'mapped')
    df_rutas['Road'] = mapped_road.fillna(5).astype(int)
    
    # MOVES source ID (0=non-motorized/stop, 21=car, 42=bus).
    condiciones = [
        df_rutas['trip'] < 0,
        df_rutas[COLUMNA_MODO].astype(str).str.contains('(?i)carro|auto', na=False),
        df_rutas[COLUMNA_MODO].astype(str).str.contains('(?i)bus', na=False)
    ]
    opciones = [0, 21, 42]
    df_rutas['Source'] = np.select(condiciones, opciones, default=0)
    
    # MOVES speed-bin classification after converting km/h to mph.
    df_rutas['avg_speed_mph'] = df_rutas[COLUMNA_VELOCIDAD] * 0.621371
    bins = [0, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, float('inf')]
    df_rutas['SpeedBin'] = pd.cut(df_rutas['avg_speed_mph'], bins=bins, labels=False, right=False).fillna(0).astype(int) + 1
    
    # 2. Join with MOVES emission rates.
    merge_cols = ['Day', 'Hour', 'Road', 'Source', 'SpeedBin']
    cols_needed = merge_cols + POLLUTANTS
    df_emisiones_raw = df_emisiones_raw[cols_needed].astype({
        'Day': int, 'Hour': int, 'Road': int, 'Source': int, 'SpeedBin': int
    })
    
    df_emisiones = df_emisiones_raw.groupby(merge_cols, as_index=False)[POLLUTANTS].mean()
    
    df_motorizados = df_rutas[df_rutas['Source'] > 0].copy()
    df_no_motorizados = df_rutas[df_rutas['Source'] == 0].copy()
    
    # Cruce de Nivel 1 (Exacto)
    df_motorizados = pd.merge(df_motorizados, df_emisiones, on=merge_cols, how='left')
    df_motorizados['emission_lookup_status'] = np.where(
        df_motorizados[POLLUTANTS[0]].notna(), 'exact', 'pending_imputation'
    )
    
    # Cascaded imputation for missing lookup records.
    mask_faltantes = df_motorizados[POLLUTANTS[0]].isna()
    if mask_faltantes.any():
        promedios_sb = df_emisiones.groupby(['Source', 'SpeedBin'])[POLLUTANTS].mean().reset_index()
        promedios_source = df_emisiones.groupby('Source')[POLLUTANTS].mean().reset_index()
        
        # Level 2: impute from the speed-bin curve.
        fuentes_unicas = df_emisiones['Source'].unique()
        idx_completo = pd.MultiIndex.from_product([fuentes_unicas, range(1, 17)], names=['Source', 'SpeedBin'])
        df_curvas = promedios_sb.set_index(['Source', 'SpeedBin']).reindex(idx_completo).groupby(level='Source').ffill().groupby(level='Source').bfill().reset_index()
        df_curvas = df_curvas.rename(columns={p: f"{p}_curve" for p in POLLUTANTS})
        
        df_motorizados = pd.merge(df_motorizados, df_curvas, on=['Source', 'SpeedBin'], how='left')
        
        # Level 3: global imputation by vehicle type.
        df_global = promedios_source.rename(columns={p: f"{p}_global" for p in POLLUTANTS})
        df_motorizados = pd.merge(df_motorizados, df_global, on='Source', how='left')
        
        # Aplicar imputaciones en cascada
        for p in POLLUTANTS:
            curve_available = df_motorizados[p].isna() & df_motorizados[f"{p}_curve"].notna()
            source_available = df_motorizados[p].isna() & ~curve_available & df_motorizados[f"{p}_global"].notna()
            df_motorizados.loc[curve_available, 'emission_lookup_status'] = 'imputed_speed_curve'
            df_motorizados.loc[source_available, 'emission_lookup_status'] = 'imputed_source'
            df_motorizados[p] = df_motorizados[p].fillna(df_motorizados[f"{p}_curve"])
            df_motorizados[p] = df_motorizados[p].fillna(df_motorizados[f"{p}_global"])
            missing_absolute = df_motorizados[p].isna()
            df_motorizados.loc[missing_absolute, 'emission_lookup_status'] = 'missing_zero_fallback'
            df_motorizados[p] = df_motorizados[p].fillna(0.0)
            
        # Remove auxiliary imputation columns.
        cols_drop = [f"{p}_curve" for p in POLLUTANTS] + [f"{p}_global" for p in POLLUTANTS]
        df_motorizados = df_motorizados.drop(columns=[c for c in cols_drop if c in df_motorizados.columns], errors='ignore')

    # Unir motorizados y no motorizados
    for p in POLLUTANTS: 
        df_no_motorizados[p] = 0.0
    df_no_motorizados['emission_lookup_status'] = 'not_applicable_non_motorized'
        
    df_final = pd.concat([df_motorizados, df_no_motorizados], ignore_index=True).sort_values('orden_original').reset_index(drop=True)
    
    # 3. Calculate total mass.
    df_final['distance_km_calc'] = df_final[COLUMNA_DISTANCIA] / 1000.0
    OCUPACION_MEDIA_BUS = 25
    
    for p in POLLUTANTS:
        df_final[f"Densidad_{p}_g_km"] = df_final[p]
        # Prorate Bus mass using the configured average occupancy (25 people).
        df_final[f"Total_{p}_g"] = np.where(
            df_final['Source'] == 42, 
            (df_final[f"Densidad_{p}_g_km"] * df_final['distance_km_calc']) / OCUPACION_MEDIA_BUS,
            df_final[f"Densidad_{p}_g_km"] * df_final['distance_km_calc']
        )

    # Unidades y aliases de nomenclatura exigidos por el contrato externo.
    df_final['distance_unit'] = 'm'
    df_final['emission_rate_unit'] = config.EMISSION_RATE_DISTANCE_UNIT
    df_final['emission_total_unit'] = config.EMISSION_TOTAL_UNIT
    df_final['Densidad_CO2e_g_km'] = df_final['Densidad_CO2_Equiv_g_km']
    df_final['Total_CO2e_g'] = df_final['Total_CO2_Equiv_g']
    df_final['Densidad_PM2.5_g_km'] = df_final['Densidad_PM25_g_km']
    df_final['Total_PM2.5_g'] = df_final['Total_PM25_g']
        
    df_final['fecha_kepler'] = df_final['local_timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    print("[Emissions] Emissions calculation complete.", flush=True)
    
    return df_final
