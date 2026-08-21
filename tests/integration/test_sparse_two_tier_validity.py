"""Comprehensive tests for two-tier validity, sparse GPS recovery, and strict emissions handoff."""
import pytest
import pandas as pd
import numpy as np
import shapely.geometry

from pipeline_v4.src.modal_classification import (
    TripServingContext, create_modal_evaluator,
)
from pipeline_v4.src.production_workflow import (
    DayProcessingResult, LEDGER_COLUMNS, _ensure_ledger_schema,
)
from pipeline_v4.src.output_schema import (
    LEDGER_OUTPUT_COLUMNS, project_ledger, project_detailed,
)
from pipeline_v4.src.route_quality import (
    evaluate_route_quality, attach_route_quality, is_strict_emissions_usable,
)


def _dummy_route(rows=10, mode="Carro", failed=False, speed=40.0):
    lines = [shapely.geometry.LineString([(0, i), (0, i+1)]) for i in range(rows)]
    return pd.DataFrame({
        "caid": ["USER_TEST"] * rows,
        "trip": [1] * rows,
        "physical_trip_id": ["USER_TEST_2026-08-18_1"] * rows,
        "latitude": np.linspace(25.67, 25.671, rows),
        "longitude": np.linspace(-100.31, -100.31, rows),
        "lat_ruteo": np.linspace(25.67, 25.671, rows),
        "lon_ruteo": np.linspace(-100.31, -100.31, rows),
        "Speed [km/h]": [speed] * rows,
        "local_timestamp": pd.date_range("2026-08-18 08:00", periods=rows, freq="15s"),
        "highway": ["primary"] * rows,
        "distance_m": [11.1] * rows,
        "geometry": lines,
        "near_bus_route": [0] * rows,
        "near_subway_line": [0] * rows,
        "osmid": [100 + i for i in range(rows)],
        "start_node": [1000 + i for i in range(rows)],
        "end_node": [1001 + i for i in range(rows)],
        "modo_transporte": [mode] * rows,
        "ruteo_fallido": [failed] * rows,
        "route_component_id": [0] * rows,
        "flag_auditoria": ["Nivel1_Lazy"] * rows,
    })


def test_guardrail_rejects_below_threshold_and_admits_at_threshold():
    evaluator = create_modal_evaluator("hybrid")
    # Below 8 pings: 7 pings -> Rejected
    ctx_7 = TripServingContext(7, 7, (10.0,) * 7, (5.0,) * 7)
    mode_7, sel_7, _, _ = evaluator.select_final_mode({"Carro": _dummy_route(7)}, serving_context=ctx_7)
    assert mode_7 == "Calidad insuficiente"
    assert sel_7 is None

    # At threshold: 8 pings -> Admitted
    ctx_8 = TripServingContext(8, 8, (10.0,) * 8, (5.0,) * 8)
    mode_8, sel_8, _, _ = evaluator.select_final_mode({"Carro": _dummy_route(8)}, serving_context=ctx_8)
    assert mode_8 != "Calidad insuficiente"
    assert sel_8 is not None


def test_walking_and_metro_excluded_from_emissions_usable():
    walk_route = _dummy_route(10, mode="Caminar", speed=4.0)
    quality = evaluate_route_quality(walk_route, walk_route)
    
    assert is_strict_emissions_usable("Caminar", True, quality) is False
    assert is_strict_emissions_usable("Metro", True, quality) is False


def test_failed_geometry_car_and_bus_excluded_from_emissions_usable():
    bad_route = _dummy_route(10, mode="Carro")
    gps_trip = _dummy_route(10, mode="Carro")
    gps_trip.loc[5, "lat_ruteo"] = 28.5  # 300 km jump -> ratio < 0.25 -> failed
    gps_trip.loc[5, "latitude"] = 28.5
    quality = evaluate_route_quality(bad_route, gps_trip)
    
    assert quality["route_completeness_status"] == "failed"
    assert is_strict_emissions_usable("Carro", True, quality) is False
    assert is_strict_emissions_usable("Bus", True, quality) is False


def test_valid_partial_car_and_bus_included_when_geometry_passes():
    route = _dummy_route(10, mode="Carro")
    quality = evaluate_route_quality(route, route)
    assert quality["route_completeness_status"] == "complete"
    assert is_strict_emissions_usable("Carro", True, quality) is True
    assert is_strict_emissions_usable("Bus", True, quality) is True


def test_modal_usable_and_emissions_usable_persisted_in_ledger_schema():
    assert "modal_usable" in LEDGER_COLUMNS
    assert "emissions_usable" in LEDGER_COLUMNS
    assert "modal_usable" in LEDGER_OUTPUT_COLUMNS
    assert "emissions_usable" in LEDGER_OUTPUT_COLUMNS
    
    empty_df = pd.DataFrame([{
        "user_id": "U1", "trip_id": 1, "physical_trip_id": "U1_1",
        "processing_status": "success", "classification_success": True,
        "modal_usable": True, "emissions_usable": False,
    }])
    projected = project_ledger(empty_df)
    assert "modal_usable" in projected.columns
    assert "emissions_usable" in projected.columns
    assert bool(projected["modal_usable"].iloc[0]) is True
    assert bool(projected["emissions_usable"].iloc[0]) is False


def test_routing_status_and_processing_status_semantics():
    quality_failed = {
        "route_completeness_status": "failed",
        "route_gps_ratio": 0.1,
        "max_continuity_gap_m": 0.0,
        "uncovered_fraction": 0.0,
        "failed_row_fraction": 0.0,
        "reconstructed_distance_m": 100.0,
    }
    final_mode = "Carro"
    modal_usable = bool(final_mode not in {None, "Calidad insuficiente"})
    emissions_usable = is_strict_emissions_usable(final_mode, modal_usable, quality_failed)
    
    row = {
        "processing_status": "success",
        "failure_reason": "post_routing_quality_failed",
        "route_success": True,
        "final_mode": final_mode,
        "classification_success": modal_usable,
        "modal_usable": modal_usable,
        "emissions_usable": emissions_usable,
        "route_completeness_status": quality_failed["route_completeness_status"],
    }
    
    assert row["processing_status"] == "success"
    assert row["failure_reason"] == "post_routing_quality_failed"
    assert row["classification_success"] is True
    assert row["modal_usable"] is True
    assert row["emissions_usable"] is False


def test_strict_emissions_handoff_predicate_cases():
    """Verify the 5 exact predicate cases:
    A. valid Car -> emissions_usable = True -> enters emissions
    B. Car with gap > 100m -> emissions_usable = False -> excluded from emissions
    C. Bus with route_gps_ratio > 2.0 -> emissions_usable = False -> excluded from emissions
    D. Walking with perfect geometry -> modal_usable = True, emissions_usable = False -> excluded
    E. Metro with perfect geometry -> modal_usable = True, emissions_usable = False -> excluded
    """
    # Case A: Valid Car
    q_valid_car = {
        "route_completeness_status": "complete", "reconstructed_distance_m": 500.0,
        "route_gps_ratio": 1.05, "max_continuity_gap_m": 15.0,
        "uncovered_fraction": 0.01, "failed_row_fraction": 0.0,
    }
    assert is_strict_emissions_usable("Carro", True, q_valid_car) is True

    # Case B: Car with gap > 100m
    q_gap_car = {
        "route_completeness_status": "partial", "reconstructed_distance_m": 500.0,
        "route_gps_ratio": 1.05, "max_continuity_gap_m": 180.0,
        "uncovered_fraction": 0.05, "failed_row_fraction": 0.0,
    }
    assert is_strict_emissions_usable("Carro", True, q_gap_car) is False

    # Case C: Bus with ratio > 2.0
    q_ratio_bus = {
        "route_completeness_status": "partial", "reconstructed_distance_m": 2500.0,
        "route_gps_ratio": 2.50, "max_continuity_gap_m": 20.0,
        "uncovered_fraction": 0.05, "failed_row_fraction": 0.0,
    }
    assert is_strict_emissions_usable("Bus", True, q_ratio_bus) is False

    # Case D: Walking with perfect geometry
    q_walk = {
        "route_completeness_status": "complete", "reconstructed_distance_m": 300.0,
        "route_gps_ratio": 1.00, "max_continuity_gap_m": 0.0,
        "uncovered_fraction": 0.0, "failed_row_fraction": 0.0,
    }
    assert is_strict_emissions_usable("Caminar", True, q_walk) is False

    # Case E: Metro with perfect geometry
    q_metro = {
        "route_completeness_status": "complete", "reconstructed_distance_m": 5000.0,
        "route_gps_ratio": 1.00, "max_continuity_gap_m": 0.0,
        "uncovered_fraction": 0.0, "failed_row_fraction": 0.0,
    }
    assert is_strict_emissions_usable("Metro", True, q_metro) is False


def test_production_emissions_handoff_integration_pipeline(monkeypatch, tmp_path):
    """Real integration test for the actual production emissions handoff.
    Exercises run_pipeline_v4 end-to-end with the 5 required cases:
    A. valid Car partial/complete: passes strict gate -> reaches calculate_emissions
    B. Car with gap >100m: emissions_usable=False -> no emission rows
    C. Bus with route_gps_ratio >2.0: emissions_usable=False -> no emission rows
    D. Walking with perfect geometry: modal_usable=True, emissions_usable=False -> no emission rows
    E. Metro with perfect geometry: modal_usable=True, emissions_usable=False -> no emission rows
    This test fails if any non-eligible trip enters calculate_emissions.
    """
    from pipeline_v4.src import production_workflow

    users = ["user_A", "user_B", "user_C", "user_D", "user_E"]
    gps = pd.DataFrame({
        "caid": users,
        "user_id": users,
        "local_timestamp": pd.to_datetime(["2026-08-18 08:00"] * 5),
        "latitude": [25.68] * 5,
        "longitude": [-100.31] * 5,
    })
    metadata = pd.DataFrame({
        "user_id": users,
        "routing_eligible": [True] * 5,
        "home_lat": [25.67] * 5,
        "home_lon": [-100.32] * 5,
        "home_ageb": ["A", "B", "C", "D", "E"],
        "home_confidence": [0.9] * 5,
        "home_quality_flag": ["reliable"] * 5,
        "processing_status": ["ready_for_pipeline"] * 5,
    })

    test_cases = {
        "user_A": {  # Case A: Valid Car complete -> passes strict gate
            "mode": "Carro", "quality": {
                "route_completeness_status": "complete", "reconstructed_distance_m": 500.0,
                "route_gps_ratio": 1.05, "max_continuity_gap_m": 15.0,
                "uncovered_fraction": 0.01, "failed_row_fraction": 0.0,
            }
        },
        "user_B": {  # Case B: Car with gap > 100m -> fails strict gate
            "mode": "Carro", "quality": {
                "route_completeness_status": "partial", "reconstructed_distance_m": 500.0,
                "route_gps_ratio": 1.05, "max_continuity_gap_m": 180.0,
                "uncovered_fraction": 0.05, "failed_row_fraction": 0.0,
            }
        },
        "user_C": {  # Case C: Bus with ratio > 2.0 -> fails strict gate
            "mode": "Bus", "quality": {
                "route_completeness_status": "partial", "reconstructed_distance_m": 2500.0,
                "route_gps_ratio": 2.50, "max_continuity_gap_m": 20.0,
                "uncovered_fraction": 0.05, "failed_row_fraction": 0.0,
            }
        },
        "user_D": {  # Case D: Walking with perfect geometry -> modal_usable=True, emissions_usable=False
            "mode": "Caminar", "quality": {
                "route_completeness_status": "complete", "reconstructed_distance_m": 300.0,
                "route_gps_ratio": 1.00, "max_continuity_gap_m": 0.0,
                "uncovered_fraction": 0.0, "failed_row_fraction": 0.0,
            }
        },
        "user_E": {  # Case E: Metro with perfect geometry -> modal_usable=True, emissions_usable=False
            "mode": "Metro", "quality": {
                "route_completeness_status": "complete", "reconstructed_distance_m": 5000.0,
                "route_gps_ratio": 1.00, "max_continuity_gap_m": 0.0,
                "uncovered_fraction": 0.0, "failed_row_fraction": 0.0,
            }
        },
    }

    def fake_process_user_day(user_id, day_frame, resources):
        cfg = test_cases[user_id]
        mode = cfg["mode"]
        qual = cfg["quality"]
        physical_trip_id = f"{user_id}_2026-08-18_1"
        
        route = pd.DataFrame({
            "caid": [user_id], "trip": [1], "local_timestamp": [day_frame.local_timestamp.iloc[0]],
            "physical_trip_id": [physical_trip_id],
            "distance_m": [qual["reconstructed_distance_m"]],
            "modo_transporte": [mode], "osmid": ["101"],
            "Speed [km/h]": [35.0 if mode in {"Carro", "Bus"} else 5.0],
            "highway": ["primary"],
            "ruteo_fallido": [False],
            "route_completeness_status": [qual["route_completeness_status"]],
        })
        
        modal_usable = bool(mode not in {None, "Calidad insuficiente"})
        emissions_usable = is_strict_emissions_usable(mode, modal_usable, qual)
        
        ledger_row = {
            "user_id": user_id, "trip_id": 1, "physical_trip_id": physical_trip_id,
            "raw_ping_count": 20, "effective_ping_count": 20, "pct_pings_conserved": 100.0,
            "hypotheses_attempted": mode, "hypotheses_successful": mode,
            "hypotheses_attempted_count": 1, "hypotheses_successful_count": 1,
            "route_success": True, "final_mode": mode,
            "classification_success": modal_usable,
            "modal_usable": modal_usable,
            "emissions_usable": emissions_usable,
            "emissions_success": False,
            "processing_status": "emissions_pending" if emissions_usable else "success",
            "failure_reason": None if (emissions_usable or mode in {"Caminar", "Metro"}) else "post_routing_quality_failed",
            "pre_routing_quality_status": "passed",
        }
        ledger_row.update(qual)
        return production_workflow.DayProcessingResult(route, pd.DataFrame([ledger_row]))

    trips_entering_emissions = []

    def mock_calculate_emissions(emission_routes, moves_file):
        for trip_id in emission_routes["physical_trip_id"].unique():
            trips_entering_emissions.append(str(trip_id))
            # Strict assertion: Non-eligible trips must NEVER enter calculate_emissions
            assert "user_A" in trip_id, f"Ineligible trip {trip_id} erroneously entered calculate_emissions!"
        return emission_routes.assign(Total_CO2_g=42.0)

    monkeypatch.setattr(production_workflow, "process_user_day", fake_process_user_day)
    monkeypatch.setattr(production_workflow, "calculate_emissions", mock_calculate_emissions)

    result = production_workflow.run_pipeline_v4(
        gps, tmp_path, metadata, n_jobs=1, resources={"fixture": True}, output_mode="detailed"
    )

    # 1. Verify emissions handoff received ONLY Case A
    assert trips_entering_emissions == ["user_A_2026-08-18_1"]
    
    # 2. Verify ledger fields for all 5 cases
    ledger = result.trip_ledger.set_index("user_id")
    
    # Case A: Valid Car
    row_a = ledger.loc["user_A"]
    assert bool(row_a["modal_usable"]) is True
    assert bool(row_a["emissions_usable"]) is True
    assert str(row_a["processing_status"]) == "success"
    assert bool(row_a["emissions_success"]) is True

    # Case B: Car with gap > 100m
    row_b = ledger.loc["user_B"]
    assert bool(row_b["modal_usable"]) is True
    assert bool(row_b["emissions_usable"]) is False
    assert str(row_b["processing_status"]) == "success"
    assert bool(row_b["emissions_success"]) is False

    # Case C: Bus with ratio > 2.0
    row_c = ledger.loc["user_C"]
    assert bool(row_c["modal_usable"]) is True
    assert bool(row_c["emissions_usable"]) is False
    assert str(row_c["processing_status"]) == "success"
    assert bool(row_c["emissions_success"]) is False

    # Case D: Walking with perfect geometry
    row_d = ledger.loc["user_D"]
    assert bool(row_d["modal_usable"]) is True
    assert bool(row_d["emissions_usable"]) is False
    assert str(row_d["processing_status"]) == "success"
    assert bool(row_d["emissions_success"]) is False

    # Case E: Metro with perfect geometry
    row_e = ledger.loc["user_E"]
    assert bool(row_e["modal_usable"]) is True
    assert bool(row_e["emissions_usable"]) is False
    assert str(row_e["processing_status"]) == "success"
    assert bool(row_e["emissions_success"]) is False

    # 3. Verify individual emissions artifact has rows ONLY for Case A
    assert len(result.individual_emissions) == 1
    assert result.individual_emissions.iloc[0]["user_id"] == "user_A"
    assert result.individual_emissions.iloc[0]["physical_trip_id"] == "user_A_2026-08-18_1"

