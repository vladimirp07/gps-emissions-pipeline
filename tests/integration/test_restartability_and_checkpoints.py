import json
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

from pipeline_v4.src import production_workflow
from pipeline_v4.src.output_schema import (
    DETAILED_COLUMNS,
    SUMMARY_COLUMNS,
    project_detailed,
    project_summary,
    write_detailed_output_streaming,
    write_summary_output_streaming,
    write_parquet_atomic,
)
from pipeline_v4.src.production_workflow import _validate_parquet_checkpoint


@pytest.fixture
def mini_gps_fixture():
    gps = pd.DataFrame({
        "caid": [101, 101, 202, 202],
        "user_id": [101, 101, 202, 202],
        "local_timestamp": pd.to_datetime([
            "2026-01-01 08:00:00", "2026-01-01 08:00:10",
            "2026-01-01 09:00:00", "2026-01-01 09:00:10",
        ]),
        "latitude": [25.68, 25.681, 25.69, 25.691],
        "longitude": [-100.31, -100.311, -100.30, -100.301],
        "Speed [km/h]": [20.0, 25.0, 30.0, 35.0],
        "dis lineal [m]": [10.0, 10.0, 15.0, 15.0],
        "trip": [1, 1, 1, 1],
        "travel time": [10, 10, 10, 10],
    })
    metadata = pd.DataFrame({
        "user_id": [101, 202],
        "home_lat": [25.67, 25.66],
        "home_lon": [-100.32, -100.33],
        "home_ageb": ["190390001001", "190390001002"],
        "home_quality_flag": ["reliable", "probable"],
        "home_inventory_status": ["eligible", "eligible"],
        "routing_eligible": [True, True],
        "processing_status": ["ready_for_pipeline", "ready_for_pipeline"],
    })
    return gps, metadata


def test_stage_checkpoints_created_and_resumed_without_recomputation(monkeypatch, tmp_path, mini_gps_fixture):
    gps, metadata = mini_gps_fixture
    routing_call_count = 0
    emissions_call_count = 0

    def fake_process_user_day(user_id, day_frame, resources):
        nonlocal routing_call_count
        routing_call_count += 1
        routes = pd.DataFrame({
            "caid": [user_id, user_id],
            "trip": [1, 1],
            "local_timestamp": day_frame.local_timestamp.values,
            "latitude": day_frame.latitude.values,
            "longitude": day_frame.longitude.values,
            "physical_trip_id": [f"{user_id}_2026-01-01_1", f"{user_id}_2026-01-01_1"],
            "distance_m": [100.0, 150.0],
            "modo_transporte": ["Carro", "Carro"],
            "osmid": ["101", "102"],
            "highway": ["primary", "secondary"],
            "geometry": ["LINESTRING(...)", "LINESTRING(...)"],
            "ruteo_fallido": [False, False],
            "emissions_eligible": [True, True],
        })
        ledger = pd.DataFrame([{
            "user_id": user_id,
            "trip_id": 1,
            "physical_trip_id": f"{user_id}_2026-01-01_1",
            "processing_status": "emissions_pending",
            "failure_reason": None,
            "raw_ping_count": 20,
            "effective_ping_count": 20,
            "pct_pings_conserved": 100.0,
            "pre_routing_quality_status": "passed",
            "hypotheses_attempted": "Carro",
            "hypotheses_successful": "Carro",
            "hypotheses_attempted_count": 1,
            "hypotheses_successful_count": 1,
            "route_success": True,
            "final_mode": "Carro",
            "classification_success": True,
            "modal_usable": True,
            "emissions_eligible": True,
            "emissions_usable": True,
            "emissions_success": False,
        }])
        return production_workflow.DayProcessingResult(routes, ledger)

    def fake_calculate_emissions(routes, moves_file):
        nonlocal emissions_call_count
        emissions_call_count += 1
        return routes.assign(
            Total_CO2_g=10.0, Densidad_CO2_g_km=150.0,
            road_lookup_status="success", emission_lookup_status="success",
        )

    monkeypatch.setattr(production_workflow, "process_user_day", fake_process_user_day)
    monkeypatch.setattr(production_workflow, "calculate_emissions", fake_calculate_emissions)

    run_dir = tmp_path / "run_test_resume"
    fixture_resources = {"fixture": True}

    # --- Run 1: Initial full run ---
    result1 = production_workflow.run_pipeline_v4(
        gps, run_dir, metadata, n_jobs=1, output_mode="both", resources=fixture_resources, resume=True
    )
    assert routing_call_count == 2
    assert emissions_call_count == 1
    assert (run_dir / "checkpoints" / "routed_trajectories.parquet").exists()
    assert (run_dir / "checkpoints" / "emissions_results.parquet").exists()
    assert (run_dir / "stages" / "stage_routing_modal.done").exists()
    assert (run_dir / "stages" / "stage_emissions.done").exists()
    assert (run_dir / "routes_emissions_detailed.parquet").exists()
    assert (run_dir / "routes_emissions_summary.parquet").exists()
    assert (run_dir / "trip_ledger.parquet").exists()

    # --- Run 2: Resume with existing checkpoints ---
    # Should skip routing and emissions completely
    result2 = production_workflow.run_pipeline_v4(
        gps, run_dir, metadata, n_jobs=1, output_mode="both", resources=fixture_resources, resume=True
    )
    assert routing_call_count == 2  # Proves routing was NOT recomputed
    assert emissions_call_count == 1  # Proves emissions was NOT recomputed
    assert len(result2.routes) == len(result1.routes)
    assert len(result2.individual_emissions) == len(result1.individual_emissions)
    assert len(result2.trip_ledger) == len(result1.trip_ledger)

    # A changed scientific workload must invalidate both stages, even when old
    # markers and readable parquet files remain in the output directory.
    result3 = production_workflow.run_pipeline_v4(
        gps, run_dir, metadata, n_jobs=1, limit_users=1, output_mode="both",
        resources=fixture_resources, resume=True,
    )
    assert routing_call_count == 3
    assert emissions_call_count == 2
    assert len(result3.trip_ledger) == 1


def test_failure_injection_during_detailed_output_preserves_checkpoints(monkeypatch, tmp_path, mini_gps_fixture):
    gps, metadata = mini_gps_fixture

    def fake_process_user_day(user_id, day_frame, resources):
        routes = pd.DataFrame({
            "caid": [user_id], "trip": [1], "local_timestamp": day_frame.local_timestamp.values[:1],
            "latitude": [25.68], "longitude": [-100.31],
            "physical_trip_id": [f"{user_id}_2026-01-01_1"],
            "distance_m": [100.0], "modo_transporte": ["Carro"], "osmid": ["101"],
            "highway": ["primary"], "geometry": ["LINESTRING(...)"], "ruteo_fallido": [False],
            "emissions_eligible": [True],
        })
        ledger = pd.DataFrame([{
            "user_id": user_id, "trip_id": 1, "physical_trip_id": f"{user_id}_2026-01-01_1",
            "processing_status": "emissions_pending", "failure_reason": None,
            "raw_ping_count": 20, "effective_ping_count": 20, "pct_pings_conserved": 100.0,
            "pre_routing_quality_status": "passed", "hypotheses_attempted": "Carro",
            "hypotheses_successful": "Carro", "hypotheses_attempted_count": 1,
            "hypotheses_successful_count": 1, "route_success": True, "final_mode": "Carro",
            "classification_success": True, "modal_usable": True, "emissions_eligible": True,
            "emissions_usable": True, "emissions_success": False,
        }])
        return production_workflow.DayProcessingResult(routes, ledger)

    monkeypatch.setattr(production_workflow, "process_user_day", fake_process_user_day)
    monkeypatch.setattr(production_workflow, "calculate_emissions", lambda routes, _: routes.assign(Total_CO2_g=10.0))

    # Inject failure into write_detailed_output_streaming
    def failing_writer(*args, **kwargs):
        raise MemoryError("Simulated injected OOM during detailed chunk 3")

    monkeypatch.setattr(production_workflow, "write_detailed_output_streaming", failing_writer)

    run_dir = tmp_path / "run_injected_failure"
    fixture_resources = {"fixture": True}
    result = production_workflow.run_pipeline_v4(
        gps, run_dir, metadata, n_jobs=1, output_mode="both", resources=fixture_resources
    )

    # 1. Checkpoints and ledger MUST exist and be intact
    assert (run_dir / "checkpoints" / "routed_trajectories.parquet").exists()
    assert (run_dir / "checkpoints" / "emissions_results.parquet").exists()
    assert (run_dir / "trip_ledger.parquet").exists()
    assert (run_dir / "routes_emissions_summary.parquet").exists()
    
    # 2. Detailed parquet should NOT exist in corrupt state
    assert not (run_dir / "routes_emissions_detailed.parquet").exists()
    assert not (run_dir / "routes_emissions_detailed.parquet.tmp").exists()

    # 3. Manifest should record scientific computation complete with detailed failure status
    manifest = json.loads((run_dir / "pipeline_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "scientific_computation_complete_detailed_failed"
    assert manifest["detailed_output"]["generated"] is False
    assert "Simulated injected OOM" in manifest["detailed_output"]["error"]

    # 4. Now restore valid writer and resume: should regenerate detailed output from checkpoints without re-running routing!
    monkeypatch.undo()
    monkeypatch.setattr(production_workflow, "process_user_day", lambda *a, **k: pytest.fail("Should not re-route"))
    monkeypatch.setattr(production_workflow, "calculate_emissions", lambda *a, **k: pytest.fail("Should not re-calc emissions"))

    resumed_result = production_workflow.run_pipeline_v4(
        gps, run_dir, metadata, n_jobs=1, output_mode="both", resources=fixture_resources, resume=True
    )
    assert (run_dir / "routes_emissions_detailed.parquet").exists()
    assert resumed_result.output_artifacts["detailed_output"]["generated"] is True


def test_checkpoint_validation_rejects_corrupted_files(tmp_path):
    # Valid file
    valid_file = tmp_path / "valid.parquet"
    pd.DataFrame({"physical_trip_id": ["t1"], "modo_transporte": ["Carro"]}).to_parquet(valid_file)
    assert _validate_parquet_checkpoint(valid_file, min_rows=1, required_columns=["physical_trip_id", "modo_transporte"]) is True
    
    # Missing required column
    assert _validate_parquet_checkpoint(valid_file, min_rows=1, required_columns=["missing_column"]) is False

    # Corrupt binary file
    corrupt_file = tmp_path / "corrupt.parquet"
    corrupt_file.write_bytes(b"CORRUPTED_PARQUET_HEADER_DATA")
    assert _validate_parquet_checkpoint(corrupt_file) is False

    # Empty 0-byte file
    empty_file = tmp_path / "empty.parquet"
    empty_file.write_bytes(b"")
    assert _validate_parquet_checkpoint(empty_file) is False

    # Non-existent file
    missing_file = tmp_path / "does_not_exist.parquet"
    assert _validate_parquet_checkpoint(missing_file) is False


def test_atomic_parquet_replacement_uses_os_replace(monkeypatch, tmp_path):
    target = tmp_path / "checkpoint.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(target)
    calls = []
    real_replace = production_workflow.os.replace

    def spy_replace(source, destination):
        calls.append((Path(source), Path(destination)))
        return real_replace(source, destination)

    monkeypatch.setattr("pipeline_v4.src.output_schema.os.replace", spy_replace)
    write_parquet_atomic(pd.DataFrame({"value": [2]}), target)
    assert calls == [(target.with_suffix(".parquet.tmp"), target)]
    assert pd.read_parquet(target).value.tolist() == [2]


def test_valid_empty_stage_checkpoints_resume_without_recomputation(monkeypatch, tmp_path):
    gps = pd.DataFrame({
        "caid": [101], "local_timestamp": pd.to_datetime(["2026-01-01 08:00"]),
        "latitude": [25.68], "longitude": [-100.31],
    })
    metadata = pd.DataFrame({"user_id": [101], "routing_eligible": [True]})
    calls = 0

    def rejected_day(user_id, day_frame, resources):
        nonlocal calls
        calls += 1
        ledger = pd.DataFrame([{
            "user_id": user_id, "trip_id": 1, "physical_trip_id": "101_2026-01-01_1",
            "processing_status": "quality_rejected", "failure_reason": "quality_guardrail",
            "raw_ping_count": 3, "effective_ping_count": 3, "pct_pings_conserved": 100.0,
            "classification_success": False, "route_success": False,
        }])
        return production_workflow.DayProcessingResult(pd.DataFrame(), ledger)

    monkeypatch.setattr(production_workflow, "process_user_day", rejected_day)
    resources = {"fixture": True}
    output = tmp_path / "empty_valid"
    production_workflow.run_pipeline_v4(
        gps, output, metadata, n_jobs=1, resources=resources, resume=True,
    )
    production_workflow.run_pipeline_v4(
        gps, output, metadata, n_jobs=1, resources=resources, resume=True,
    )
    assert calls == 1
    assert "modo_transporte" in pd.read_parquet(
        output / "checkpoints" / "routed_trajectories.parquet"
    ).columns
    assert "physical_trip_id" in pd.read_parquet(
        output / "checkpoints" / "emissions_results.parquet"
    ).columns
