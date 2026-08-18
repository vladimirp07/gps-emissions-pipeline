from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import LineString

from pipeline_v4.src.endpoint_routing import (
    attach_real_edge_endpoint_segments,
    build_incident_edge_index,
    explicitly_reject_failed_geometry,
)


def _fixture():
    lon0, lat0 = -100.3161, 25.6866
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)
    x0, y0 = transformer.transform(lon0, lat0)
    edges = gpd.GeoDataFrame(
        {"u": [1, 2], "v": [2, 3]},
        geometry=[
            LineString([(x0 - 100, y0), (x0 + 100, y0)]),
            LineString([(x0 + 100, y0), (x0 + 300, y0)]),
        ],
        crs="EPSG:32614",
    )
    input_frame = pd.DataFrame(
        {
            "latitude": [lat0 + 0.00005, lat0 + 0.00005],
            "longitude": [lon0 - 0.0005, lon0 + 0.0024],
            "local_timestamp": pd.to_datetime(["2026-01-01 08:00", "2026-01-01 08:05"]),
        }
    )
    core_wgs = gpd.GeoSeries([edges.geometry.iloc[0]], crs=edges.crs).to_crs("EPSG:4326").iloc[0]
    route = pd.DataFrame(
        {
            "caid": ["u1"],
            "trip": [1],
            "latitude": [lat0],
            "longitude": [lon0],
            "Speed [km/h]": [20.0],
            "local_timestamp": [pd.Timestamp("2026-01-01 08:01")],
            "start_node": [1],
            "end_node": [2],
            "osmid": ["core"],
            "highway": ["residential"],
            "geometry": [core_wgs.wkt],
            "distance_m": [200.0],
            "modo_transporte": ["Carro"],
            "ruteo_fallido": [False],
            "flag_auditoria": ["None"],
        }
    )
    return input_frame, route, edges


def test_v2_adds_real_edge_segments_without_straight_connectors():
    input_frame, route, edges = _fixture()
    result, meta = attach_real_edge_endpoint_segments(route, input_frame, edges)
    added = result[result.flag_auditoria.astype(str).str.startswith("Endpoint_Partial_Snap_")]
    assert meta["segments_added"] == 2
    assert set(added.endpoint_role) == {"start", "end"}
    assert (added.endpoint_uncovered_m > 0).all()
    for row in added.itertuples():
        segment = gpd.GeoSeries([wkt.loads(row.geometry)], crs="EPSG:4326").to_crs(edges.crs).iloc[0]
        assert segment.difference(edges.geometry.buffer(0.05).unary_union).length == pytest.approx(0.0, abs=0.1)


def test_v2_preserves_core_route_and_direction_contract():
    input_frame, route, edges = _fixture()
    result, _ = attach_real_edge_endpoint_segments(route, input_frame, edges)
    core = result[result.flag_auditoria == "None"]
    pd.testing.assert_series_equal(core.iloc[0][route.columns], route.iloc[0], check_names=False)
    start = result[result.endpoint_role == "start"].iloc[0]
    end = result[result.endpoint_role == "end"].iloc[0]
    assert start.start_node == "PartialSnap_START" and start.end_node == route.start_node.iloc[0]
    assert end.start_node == route.end_node.iloc[-1] and end.end_node == "PartialSnap_END"


def test_incident_edge_index_is_exactly_equal_to_legacy_scan():
    input_frame, route, edges = _fixture()
    legacy, legacy_meta = attach_real_edge_endpoint_segments(route, input_frame, edges)
    indexed, indexed_meta = attach_real_edge_endpoint_segments(
        route, input_frame, edges, incident_edge_index=build_incident_edge_index(edges),
    )
    pd.testing.assert_frame_equal(legacy, indexed, check_exact=True)
    assert legacy_meta == indexed_meta


def test_impossible_endpoint_is_explicitly_uncovered():
    input_frame, route, edges = _fixture()
    input_frame.loc[0, ["longitude", "latitude"]] = [-100.0, 26.0]
    result, meta = attach_real_edge_endpoint_segments(route, input_frame, edges, max_snap_m=100.0)
    assert meta["segments_added"] == 1
    assert set(result.endpoint_start_status) == {"uncovered_rejected_distance"}
    assert not result.flag_auditoria.astype(str).eq("Endpoint_Partial_Snap_start").any()


def test_failed_geometry_is_not_presented_as_route():
    _, route, _ = _fixture()
    route.loc[0, "ruteo_fallido"] = True
    rejected = explicitly_reject_failed_geometry(route)
    assert rejected.geometry.isna().all()
    assert rejected.distance_m.eq(0.0).all()
    assert rejected.flag_auditoria.eq("Explicit_Rejection_Unrecoverable").all()
