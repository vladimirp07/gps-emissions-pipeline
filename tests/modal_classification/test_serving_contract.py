import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from pipeline_v4.src.modal_classification import (
    ServingContractError, TripServingContext, create_modal_evaluator,
)


def _route(rows=100):
    return pd.DataFrame({
        "caid": ["VERASET_USER"] * rows,
        "trip": [1] * rows,
        "Speed [km/h]": np.linspace(10.0, 30.0, rows),
        "local_timestamp": pd.date_range("2026-08-18 08:00", periods=rows, freq="5s"),
        "highway": ["primary"] * rows,
        "distance_m": [20.0] * rows,
        "near_bus_route": [1] * rows,
        "near_subway_line": [0] * rows,
    })


def test_guardrail_cannot_count_routing_subsegments_as_gps_pings():
    evaluator = create_modal_evaluator("hybrid")
    context = TripServingContext(5, 5, (12.0,) * 5, (7.0,) * 5)
    mode, selected, _, _ = evaluator.select_final_mode(
        {"Carro": _route(120)}, serving_context=context
    )
    assert mode == "Calidad insuficiente"
    assert selected is None


def test_conserved_ping_fraction_contract_uses_thirty_percent_without_scale_drift():
    from pipeline_v4.src.random_forest_contract import MIN_PCT_CONSERVED

    assert MIN_PCT_CONSERVED == 0.30
    evaluator = create_modal_evaluator("hybrid")
    rejected = TripServingContext(40, 8, (12.0,) * 8, (7.0,) * 8)  # 20%
    accepted = TripServingContext(20, 8, (12.0,) * 8, (7.0,) * 8)  # 40%
    mode, selected, _, _ = evaluator.select_final_mode(
        {"Carro": _route(20)}, serving_context=rejected
    )
    assert mode == "Calidad insuficiente"
    assert selected is None
    mode, selected, _, _ = evaluator.select_final_mode(
        {"Carro": _route(20)}, serving_context=accepted
    )
    assert mode != "Calidad insuficiente"
    assert selected is not None


def test_sparse_pings_between_8_and_14_are_admitted_and_evaluated():
    evaluator = create_modal_evaluator("hybrid")
    context = TripServingContext(10, 10, (12.0,) * 10, (7.0,) * 10)
    mode, selected, _, _ = evaluator.select_final_mode(
        {"Carro": _route(20)}, serving_context=context
    )
    assert mode != "Calidad insuficiente"
    assert selected is not None


def test_snap_features_follow_training_semantics_not_placeholders():
    evaluator = create_modal_evaluator("hybrid")
    drive = (1.0, 2.0, 150.0) + (10.0,) * 17
    walk = (3.0, 4.0, 50.0) + (8.0,) * 17
    context = TripServingContext(20, 20, drive, walk)
    features = evaluator.extract_features({"Carro": _route(30)}, serving_context=context)
    assert features["mean_snap_dist_drive"] == pytest.approx(np.mean(drive))
    assert features["max_snap_dist_drive"] == 150.0
    assert features["mean_snap_dist_walk"] == pytest.approx(np.mean(walk))
    assert features["max_snap_dist_walk"] == 50.0


def test_missing_routed_infrastructure_features_fails_visibly():
    evaluator = create_modal_evaluator("hybrid")
    context = TripServingContext(20, 20, (10.0,) * 20, (5.0,) * 20)
    incomplete = _route(30).drop(columns="near_bus_route")
    with pytest.raises(ServingContractError, match="near_bus_route"):
        evaluator.extract_features({"Carro": incomplete}, serving_context=context)


def test_promoted_n1_contract_features_and_no_footway_dependency():
    from pipeline_v4.src.random_forest_contract import N1_FEATURES
    evaluator = create_modal_evaluator("hybrid")
    assert len(evaluator.n1_features) == 15
    assert "walk_highway_footway_frac" not in evaluator.n1_features
    assert evaluator.clf_n1.n_features_in_ == 15
    assert list(evaluator.clf_n1.feature_names_in_) == list(N1_FEATURES)


def test_metro_probability_threshold_and_candidate_guard():
    from pipeline_v4.src.random_forest_contract import METRO_PROBABILITY_THRESHOLD
    evaluator = create_modal_evaluator("hybrid")
    assert evaluator.metro_threshold == METRO_PROBABILITY_THRESHOLD
    assert evaluator.metro_threshold == 0.30

    # Trip without Metro candidate cannot predict Metro even if surface features are ambiguous
    context = TripServingContext(20, 20, (10.0,) * 20, (25.0,) * 20)
    route_drive = _route(40)
    mode, _, _, diags = evaluator.select_final_mode(
        {"Carro": route_drive}, serving_context=context
    )
    assert mode in {"Carro", "Bus"}
    assert diags["Metro"] == 0.0


def test_three_input_pings_are_rejected_before_routing_or_modal_serving(monkeypatch):
    from pipeline_v4.src import production_workflow

    trip = pd.DataFrame({
        "caid": ["L3"] * 3,
        "trip": [1] * 3,
        "local_timestamp": pd.date_range("2026-08-18 08:00", periods=3, freq="30s"),
        "latitude": [25.68, 25.681, 25.682],
        "longitude": [-100.31, -100.311, -100.312],
    })

    monkeypatch.setattr(production_workflow, "assign_trips", lambda frame: frame.assign(trip=1))
    monkeypatch.setattr(production_workflow, "apply_spatial_filter", lambda frame, **_: frame)
    monkeypatch.setattr(
        production_workflow, "calcular_cercania_infraestructura",
        lambda frame, *args, **kwargs: frame.assign(near_subway_line=0, near_bus_route=0),
    )
    monkeypatch.setattr(
        production_workflow, "get_candidates_vectorized",
        lambda *args, **kwargs: ([[]] * 3, [[10.0]] * 3),
    )

    class RoutingMustNotRun:
        def __init__(self, *args, **kwargs):
            pass

        def evaluate(self, *args, **kwargs):
            pytest.fail("L3-like sparse input reached routing")

    monkeypatch.setattr(production_workflow, "RouteHypothesisEvaluator", RoutingMustNotRun)
    resources = {
        name: object() for name in (
            "graph_drive", "graph_walk", "ig_drive", "ig_walk", "map_drive", "map_walk",
            "candidate_edges_drive", "candidate_edges_walk", "incident_edges_drive",
            "incident_edges_walk", "subway_routes", "bus_routes",
        )
    }
    resources["edges_drive"] = SimpleNamespace(crs="EPSG:32614")
    resources["edges_walk"] = SimpleNamespace(crs="EPSG:32614")
    resources["modal_evaluator"] = SimpleNamespace(
        select_final_mode=lambda *args, **kwargs: pytest.fail("Sparse input reached modal serving")
    )

    result = production_workflow.process_user_day("L3", trip, resources)
    assert result.routes.empty
    assert result.trip_ledger.loc[0, "effective_ping_count"] == 3
    assert result.trip_ledger.loc[0, "processing_status"] == "quality_rejected"

