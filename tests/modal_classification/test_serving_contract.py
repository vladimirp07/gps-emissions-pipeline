import numpy as np
import pandas as pd
import pytest

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

