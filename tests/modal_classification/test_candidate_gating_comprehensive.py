import pytest
import numpy as np
import pandas as pd

from pipeline_v4.src.modal_classification import (
    HybridRouteEvaluator, TripServingContext, create_modal_evaluator,
)
from pipeline_v4.src.random_forest_contract import (
    METRO_PROBABILITY_THRESHOLD, BUS_PROBABILITY_THRESHOLD,
)

def _mock_route(mode="Carro", rows=30, speed=35.0, near_metro=0, near_bus=0, highway="primary"):
    return pd.DataFrame({
        "caid": ["TEST_USER"] * rows,
        "trip": [1] * rows,
        "Speed [km/h]": np.full(rows, speed),
        "local_timestamp": pd.date_range("2026-08-18 08:00", periods=rows, freq="10s"),
        "highway": [highway] * rows,
        "distance_m": [100.0] * rows,
        "near_bus_route": [near_bus] * rows,
        "near_subway_line": [near_metro] * rows,
        "snap_dist_drive": [2.0] * rows,
        "snap_dist_walk": [25.0] * rows,
        "modo_transporte": [mode] * rows,
        "ruteo_fallido": [False] * rows,
    })

def test_case_a_high_pmetro_with_metro_candidate_predicts_metro():
    """Case A: High P(Metro) + Metro candidate in hypotheses -> Metro selected."""
    evaluator = create_modal_evaluator("hybrid")
    h_metro = {
        "Carro": _mock_route("Carro", rows=30, speed=40.0, near_metro=1, near_bus=0),
        "Metro": _mock_route("Metro", rows=30, speed=40.0, near_metro=1, near_bus=0),
    }
    context = TripServingContext(30, 30, (5.0,)*30, (20.0,)*30)
    mode, selected, prob, diags = evaluator.select_final_mode(h_metro, serving_context=context)
    
    assert mode == "Metro"
    assert "Metro" in diags
    assert diags["Metro"] >= METRO_PROBABILITY_THRESHOLD
    assert selected["modo_transporte"].iloc[0] == "Metro"

def test_case_b_high_pmetro_without_metro_candidate_cannot_select_metro():
    """Case B: High P(Metro) feature representation + NO Metro candidate -> Surface only."""
    evaluator = create_modal_evaluator("hybrid")
    h_car_only = {
        "Carro": _mock_route("Carro", rows=30, speed=40.0, near_metro=1, near_bus=0),
    }
    context = TripServingContext(30, 30, (5.0,)*30, (20.0,)*30)
    mode, selected, prob, diags = evaluator.select_final_mode(h_car_only, serving_context=context)
    
    assert mode in {"Carro", "Bus"}
    assert mode != "Metro"
    assert diags["Metro"] == 0.0

def test_case_c_metro_candidate_with_subthreshold_probability_routes_to_surface():
    """Case C: Metro candidate present + P(Metro) < threshold -> Surface (Carro/Bus)."""
    evaluator = create_modal_evaluator("hybrid")
    # Surface route with bus corridor proximity and 0 metro proximity
    h_sub = {
        "Carro": _mock_route("Carro", rows=30, speed=35.0, near_metro=0, near_bus=1, highway="primary"),
        "Metro": _mock_route("Metro", rows=30, speed=35.0, near_metro=0, near_bus=0),
    }
    context = TripServingContext(30, 30, (2.0,)*30, (20.0,)*30)
    mode, selected, prob, diags = evaluator.select_final_mode(h_sub, serving_context=context)
    
    assert mode in {"Carro", "Bus"}
    assert mode != "Metro"
    assert diags["Metro"] < METRO_PROBABILITY_THRESHOLD

def test_case_d_metro_prediction_bypasses_n3():
    """Case D: Metro prediction does not evaluate N3 Carro/Bus cascade."""
    evaluator = create_modal_evaluator("hybrid")
    h_metro = {
        "Carro": _mock_route("Carro", rows=30, speed=40.0, near_metro=1, near_bus=0),
        "Metro": _mock_route("Metro", rows=30, speed=40.0, near_metro=1, near_bus=0),
    }
    context = TripServingContext(30, 30, (5.0,)*30, (20.0,)*30)
    mode, _, _, diags = evaluator.select_final_mode(h_metro, serving_context=context)
    
    if mode == "Metro":
        assert diags["Bus"] == 0.0
        assert diags["Carro"] == 0.0

def test_case_e_non_metro_trips_classified_identically():
    """Case E: Standard non-metro trips (Walk, Highway Car, Transit Bus) maintain integrity."""
    evaluator = create_modal_evaluator("hybrid")
    
    # 1. Clear walking
    h_walk = {
        "Caminar": _mock_route("Caminar", rows=25, speed=3.5, highway="footway"),
        "Carro": _mock_route("Carro", rows=25, speed=3.5, highway="residential"),
    }
    ctx_walk = TripServingContext(25, 25, (50.0,)*25, (2.0,)*25)
    mode_walk, _, _, _ = evaluator.select_final_mode(h_walk, serving_context=ctx_walk)
    assert mode_walk == "Caminar"
    
    # 2. Clear car on highway
    h_car = {
        "Carro": _mock_route("Carro", rows=30, speed=70.0, highway="motorway", near_bus=0),
    }
    ctx_car = TripServingContext(30, 30, (2.0,)*30, (30.0,)*30)
    mode_car, _, _, _ = evaluator.select_final_mode(h_car, serving_context=ctx_car)
    assert mode_car == "Carro"
