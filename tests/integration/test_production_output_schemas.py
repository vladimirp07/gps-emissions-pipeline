import json

import pandas as pd

from pipeline_v4.src import production_workflow
from pipeline_v4.src.output_schema import DETAILED_COLUMNS, LEDGER_OUTPUT_COLUMNS, SUMMARY_COLUMNS


def _fake_day_result(user_id, day_frame, resources):
    physical_trip_id = f"{user_id}_2026-01-01_1"
    routes = pd.DataFrame({
        "caid": [user_id], "trip": [1], "physical_trip_id": [physical_trip_id],
        "local_timestamp": [day_frame.local_timestamp.iloc[0]],
        "latitude": [25.68], "longitude": [-100.31], "Speed [km/h]": [36.0],
        "duration_s": [10.0], "distance_m": [100.0], "osmid": ["1"],
        "highway": ["residential"], "geometry": ["LINESTRING (-100.31 25.68, -100.30 25.69)"],
        "modo_transporte": ["Carro"], "route_component_id": [1],
        "route_completeness_status": ["complete"], "emissions_eligible": [True],
    })
    ledger = pd.DataFrame([{
        "user_id": user_id, "trip_id": 1, "physical_trip_id": physical_trip_id,
        "processing_status": "emissions_pending", "failure_reason": None,
        "raw_ping_count": 20, "effective_ping_count": 20, "pct_pings_conserved": 100.0,
        "hypotheses_attempted": "Carro;Bus", "hypotheses_successful": "Carro",
        "hypotheses_attempted_count": 2, "hypotheses_successful_count": 1,
        "route_success": True, "final_mode": "Carro", "classification_success": True,
        "route_component_count": 1, "route_completeness_status": "complete",
        "emissions_eligible": True, "emissions_success": False,
    }])
    return production_workflow.DayProcessingResult(routes, ledger)


def _run(monkeypatch, tmp_path, mode):
    gps = pd.DataFrame({
        "caid": [101], "local_timestamp": pd.to_datetime(["2026-01-01 08:00"]),
        "latitude": [25.68], "longitude": [-100.31],
    })
    metadata = pd.DataFrame({
        "user_id": [101], "routing_eligible": [True], "processing_status": ["ready_for_pipeline"],
        "home_lat": [25.67], "home_lon": [-100.32], "home_ageb": ["A"],
    })
    monkeypatch.setattr(production_workflow, "process_user_day", _fake_day_result)
    monkeypatch.setattr(
        production_workflow, "calculate_emissions",
        lambda routes, _: routes.assign(
            Densidad_CO_g_km=1.0, Total_CO_g=2.0,
            Densidad_CO2_g_km=3.0, Total_CO2_g=4.0,
        ),
    )
    return production_workflow.run_pipeline_v4(
        gps, tmp_path, metadata, n_jobs=1, resources={"fixture": True},
        output_mode=mode, reuse_modal_evaluator=False,
    )


def test_summary_schema_is_canonical_english_and_matches_detailed(monkeypatch, tmp_path):
    result = _run(monkeypatch, tmp_path, "both")
    summary = pd.read_parquet(tmp_path / "routes_emissions_summary.parquet")
    detailed = pd.read_parquet(tmp_path / "routes_emissions_detailed.parquet")
    ledger = pd.read_parquet(tmp_path / "trip_ledger.parquet")
    manifest = json.loads((tmp_path / "pipeline_manifest.json").read_text(encoding="utf-8"))

    assert tuple(summary.columns) == SUMMARY_COLUMNS
    assert tuple(detailed.columns) == DETAILED_COLUMNS
    assert tuple(ledger.columns) == LEDGER_OUTPUT_COLUMNS
    expected_kepler_time = summary.loc[0, "local_timestamp"].strftime("%Y-%m-%d %H:%M:%S")
    assert summary.loc[0, "kepler_time"] == expected_kepler_time
    assert detailed.loc[0, "kepler_time"] == expected_kepler_time
    assert "fecha_kepler" not in summary and "fecha_kepler" not in detailed
    assert "kepler_time" not in result.routes
    assert "kepler_time" not in result.individual_emissions
    assert "kepler_time" not in result.trip_ledger
    assert not {"caid", "trip", "modo_transporte", "Densidad_CO2_g_km", "Total_CO2_g"} & set(summary)
    assert {"user_id", "trip_id", "uncovered_distance_m", "pre_routing_quality_status"} <= set(ledger)
    assert summary.loc[0, "co2_total_g"] == detailed.loc[0, "co2_total_g"] == 4.0
    assert summary.loc[0, "transport_mode"] == detailed.loc[0, "transport_mode"] == "Carro"
    assert manifest["output_mode"] == "both"
    assert manifest["summary_output"]["rows"] == manifest["detailed_output"]["rows"] == 1
    assert result.output_artifacts["trip_ledger"]["filename"] == "trip_ledger.parquet"


def test_output_modes_write_only_requested_product(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path / "summary", "summary")
    _run(monkeypatch, tmp_path / "detailed", "detailed")
    assert (tmp_path / "summary" / "routes_emissions_summary.parquet").exists()
    assert not (tmp_path / "summary" / "routes_emissions_detailed.parquet").exists()
    assert (tmp_path / "detailed" / "routes_emissions_detailed.parquet").exists()
    assert not (tmp_path / "detailed" / "routes_emissions_summary.parquet").exists()
    assert (tmp_path / "summary" / "trip_ledger.parquet").exists()
    assert (tmp_path / "detailed" / "trip_ledger.parquet").exists()


def test_modal_evaluator_is_loaded_once_and_reused_for_all_user_days(monkeypatch, tmp_path):
    gps = pd.DataFrame({
        "caid": [101, 101],
        "local_timestamp": pd.to_datetime(["2026-01-01 08:00", "2026-01-02 08:00"]),
        "latitude": [25.68, 25.68], "longitude": [-100.31, -100.31],
    })
    metadata = pd.DataFrame({"user_id": [101], "routing_eligible": [True]})
    evaluator = object()
    loads, seen = [], []

    def load_once(_):
        loads.append(True)
        return evaluator

    def fake_process(user_id, day_frame, resources):
        seen.append(resources["modal_evaluator"])
        return production_workflow.DayProcessingResult(pd.DataFrame(), pd.DataFrame())

    monkeypatch.setattr(production_workflow, "create_modal_evaluator", load_once)
    monkeypatch.setattr(production_workflow, "process_user_day", fake_process)
    production_workflow.run_pipeline_v4(
        gps, tmp_path, metadata, n_jobs=1, resources={"fixture": True},
        output_mode="summary",
    )
    assert len(loads) == 1
    assert seen == [evaluator, evaluator]
