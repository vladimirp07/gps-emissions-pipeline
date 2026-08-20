import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from pyproj import Transformer

from pipeline_v4.src.modal_classification import (
    calcular_cercania_infraestructura,
    InfrastructureProximityCache,
)


def _legacy_calcular_cercania(df, subway_routes, bus_routes):
    if df.empty:
        return df.copy()
    coordinate_keys = list(zip(df["longitude"].tolist(), df["latitude"].tolist()))
    unique_coordinates = list(dict.fromkeys(coordinate_keys))
    missing = pd.DataFrame(unique_coordinates, columns=["longitude", "latitude"])
    gdf_pts = gpd.GeoDataFrame(
        missing,
        geometry=gpd.points_from_xy(missing.longitude, missing.latitude),
        crs="EPSG:4326",
    ).to_crs("EPSG:32614")

    subway_proj = subway_routes.to_crs("EPSG:32614") if subway_routes.crs != "EPSG:32614" else subway_routes
    bus_proj = bus_routes.to_crs("EPSG:32614") if bus_routes.crs != "EPSG:32614" else bus_routes

    RADIO_METRO = 50.0
    RADIO_BUS = 20.0

    if len(subway_proj) > 0:
        m_join = gpd.sjoin_nearest(gdf_pts, subway_proj, how="left", distance_col="dist_metro")
        m_join = m_join[~m_join.index.duplicated(keep="first")]
        near_metro = (m_join["dist_metro"] < RADIO_METRO).astype(int).to_numpy()
    else:
        near_metro = np.zeros(len(gdf_pts), dtype=int)

    if len(bus_proj) > 0:
        b_join = gpd.sjoin_nearest(gdf_pts, bus_proj, how="left", distance_col="dist_bus")
        b_join = b_join[~b_join.index.duplicated(keep="first")]
        near_bus = (b_join["dist_bus"] < RADIO_BUS).astype(int).to_numpy()
    else:
        near_bus = np.zeros(len(gdf_pts), dtype=int)

    computed = {
        key: (int(m), int(b))
        for key, m, b in zip(unique_coordinates, near_metro, near_bus)
    }
    df_out = df.copy()
    df_out["near_subway_line"] = [computed[key][0] for key in coordinate_keys]
    df_out["near_bus_route"] = [computed[key][1] for key in coordinate_keys]
    return df_out


def test_strtree_proximity_exact_threshold_adversarial():
    # Construct multiple geometry types and orientations
    # 1. Horizontal line
    line_h = shapely.linestrings([(370000, 2840000), (370500, 2840000)])
    # 2. Diagonal MultiLineString
    mls = shapely.multilinestrings([
        ([(371000, 2841000), (371300, 2841400)]),
        ([(371300, 2841400), (371600, 2841400)]),
    ])
    # 3. Duplicate LineString
    line_dup = shapely.linestrings([(370000, 2840000), (370500, 2840000)])

    subway = gpd.GeoDataFrame(geometry=[line_h, mls, line_dup], crs="EPSG:32614")
    bus = gpd.GeoDataFrame(geometry=[line_h, mls], crs="EPSG:32614")

    t_to_wgs = Transformer.from_crs("EPSG:32614", "EPSG:4326", always_xy=True)

    # Test Metro distances: 49.0, 49.9, 49.99, 50.0, 50.01, 51.0
    metro_dists = [49.0, 49.9, 49.99, 50.0, 50.01, 51.0]
    # Test Bus distances: 19.0, 19.9, 19.99, 20.0, 20.01, 21.0
    bus_dists = [19.0, 19.9, 19.99, 20.0, 20.01, 21.0]

    # Create points displaced from horizontal line
    pts = []
    for d in metro_dists:
        pts.append(shapely.Point(370250.0, 2840000.0 + d))
    for d in bus_dists:
        pts.append(shapely.Point(370250.0, 2840000.0 + d))

    # Also add diagonal query points
    for d in [15.0, 19.99, 20.01, 45.0, 49.99, 50.01, 100.0]:
        pts.append(shapely.Point(371150.0 - 0.8 * d, 2841200.0 + 0.6 * d))

    lons, lats = t_to_wgs.transform([p.x for p in pts], [p.y for p in pts])
    df = pd.DataFrame({"longitude": lons, "latitude": lats})

    cache = InfrastructureProximityCache()
    res_opt = calcular_cercania_infraestructura(df.copy(), subway, bus, proximity_cache=cache)
    res_leg = _legacy_calcular_cercania(df.copy(), subway, bus)

    assert res_opt["near_subway_line"].tolist() == res_leg["near_subway_line"].tolist()
    assert res_opt["near_bus_route"].tolist() == res_leg["near_bus_route"].tolist()


def test_strtree_proximity_empty_infrastructure():
    empty_subway = gpd.GeoDataFrame(geometry=[], crs="EPSG:32614")
    empty_bus = gpd.GeoDataFrame(geometry=[], crs="EPSG:32614")
    df = pd.DataFrame({
        "longitude": [-100.31, -100.32],
        "latitude": [25.68, 25.69],
    })
    res_opt = calcular_cercania_infraestructura(df.copy(), empty_subway, empty_bus)
    res_leg = _legacy_calcular_cercania(df.copy(), empty_subway, empty_bus)

    assert res_opt["near_subway_line"].tolist() == res_leg["near_subway_line"].tolist()
    assert res_opt["near_bus_route"].tolist() == res_leg["near_bus_route"].tolist()
    assert (res_opt["near_subway_line"] == 0).all()
    assert (res_opt["near_bus_route"] == 0).all()
