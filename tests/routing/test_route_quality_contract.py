from __future__ import annotations

import pandas as pd

from pipeline_v4.src.route_quality import attach_route_quality, evaluate_route_quality


def _trip():
    return pd.DataFrame({
        "latitude": [25.6700, 25.6710, 25.6720],
        "longitude": [-100.3100, -100.3090, -100.3080],
    })


def _route(distance=300.0):
    return pd.DataFrame({
        "geometry": [
            "LINESTRING (-100.3100 25.6700, -100.3090 25.6710)",
            "LINESTRING (-100.3090 25.6710, -100.3080 25.6720)",
        ],
        "distance_m": [distance / 2, distance / 2],
        "ruteo_fallido": [False, False],
        "route_component_id": [1, 1],
        "endpoint_start_uncovered_m": [0.0, 0.0],
        "endpoint_end_uncovered_m": [0.0, 0.0],
    })


def test_complete_route_is_emissions_eligible():
    metrics = evaluate_route_quality(_route(), _trip())

    assert metrics["route_completeness_status"] == "complete"
    assert metrics["emissions_eligible"] is True
    assert metrics["route_component_count"] == 1


def test_explicit_unresolved_transition_creates_partial_components():
    route = _route()
    unresolved = route.iloc[[0]].copy()
    unresolved["geometry"] = None
    unresolved["distance_m"] = 0.0
    unresolved["ruteo_fallido"] = True
    unresolved["routing_event"] = "transition_unresolved"
    unresolved["lookahead_observed_distance_m"] = 40.0
    unresolved["route_component_id"] = 1
    route.loc[1, "route_component_id"] = 2
    route = pd.concat([route.iloc[[0]], unresolved, route.iloc[[1]]], ignore_index=True)

    metrics = evaluate_route_quality(route, _trip())

    assert metrics["route_completeness_status"] == "partial"
    assert metrics["emissions_eligible"] is True
    assert metrics["unresolved_transition_count"] == 1
    assert metrics["route_component_count"] == 2
    assert metrics["total_uncovered_distance_m"] >= 40.0


def test_extreme_ratio_is_failed_and_not_emissions_eligible():
    metrics = evaluate_route_quality(_route(distance=2000.0), _trip())

    assert metrics["route_completeness_status"] == "failed"
    assert metrics["emissions_eligible"] is False


def test_quality_fields_are_attached_to_route_rows():
    route = _route()
    metrics = evaluate_route_quality(route, _trip())
    enriched = attach_route_quality(route, metrics)

    assert enriched.route_completeness_status.eq("complete").all()
    assert enriched.emissions_scope.eq("complete_trip").all()
