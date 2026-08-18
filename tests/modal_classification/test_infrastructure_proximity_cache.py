from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from pipeline_v4.src.modal_classification import (
    InfrastructureProximityCache,
    calcular_cercania_infraestructura,
)


def test_proximity_cache_preserves_exact_flags_and_is_network_scoped():
    points = pd.DataFrame({
        "longitude": [-100.30, -100.30, -100.31],
        "latitude": [25.68, 25.68, 25.69],
    })
    subway = gpd.GeoDataFrame(
        geometry=[LineString([(-100.30, 25.67), (-100.30, 25.70)])], crs="EPSG:4326",
    ).to_crs("EPSG:32614")
    bus = gpd.GeoDataFrame(
        geometry=[LineString([(-100.32, 25.68), (-100.28, 25.68)])], crs="EPSG:4326",
    ).to_crs("EPSG:32614")
    expected = calcular_cercania_infraestructura(points.copy(), subway, bus)
    cache = InfrastructureProximityCache(max_entries=10)
    first = calcular_cercania_infraestructura(
        points.copy(), subway, bus, proximity_cache=cache,
    )
    second = calcular_cercania_infraestructura(
        points.copy(), subway, bus, proximity_cache=cache,
    )
    pd.testing.assert_frame_equal(expected, first, check_exact=True)
    pd.testing.assert_frame_equal(expected, second, check_exact=True)
    assert cache.stats()["hits"] == 2

    other_bus = bus.copy()
    calcular_cercania_infraestructura(
        points.copy(), subway, other_bus, proximity_cache=cache,
    )
    assert cache.stats()["misses"] == 4
