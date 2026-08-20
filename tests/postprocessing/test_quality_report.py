import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline_v4.diagnostics.quality_report import (
    QualityReportConfig,
    build_quality_population,
    compute_modal_metrics,
    generate_quality_report,
    generate_quality_report_if_enabled,
    load_run_outputs,
)


POLLUTANT_COLUMNS = {
    "co_g_km": 1.0, "co_total_g": 0.1,
    "co2_g_km": 100.0, "co2_total_g": 10.0,
    "co2e_g_km": 101.0, "co2e_total_g": 10.1,
    "hc_g_km": 0.5, "hc_total_g": 0.05,
    "nox_g_km": 0.7, "nox_total_g": 0.07,
    "pm10_g_km": 0.1, "pm10_total_g": 0.01,
    "pm25_g_km": 0.05, "pm25_total_g": 0.005,
}


def _ledger() -> pd.DataFrame:
    rows = [
        {
            "user_id": "u1", "trip_id": 1, "physical_trip_id": "u1_2026-01-01_1",
            "route_completeness_status": "complete", "processing_status": "success",
            "final_mode": "Carro", "reconstructed_distance_m": 300.0,
            "gps_distance_m": 280.0, "route_gps_ratio": 300 / 280,
            "uncovered_fraction": 0.01, "max_continuity_gap_m": 0.0,
            "route_component_count": 1, "emissions_eligible": True, "emissions_success": True,
        },
        {
            "user_id": "u1", "trip_id": 2, "physical_trip_id": "u1_2026-01-01_2",
            "route_completeness_status": "partial", "processing_status": "success",
            "final_mode": "Bus", "reconstructed_distance_m": 200.0,
            "gps_distance_m": 150.0, "route_gps_ratio": 200 / 150,
            "uncovered_fraction": 0.10, "max_continuity_gap_m": 120.0,
            "route_component_count": 1, "emissions_eligible": True, "emissions_success": True,
        },
        {
            "user_id": "u2", "trip_id": 1, "physical_trip_id": "u2_2026-01-01_1",
            "route_completeness_status": "failed", "processing_status": "routing_failed",
            "final_mode": "Metro", "reconstructed_distance_m": 9000.0,
            "gps_distance_m": 100.0, "route_gps_ratio": 90.0,
            "uncovered_fraction": 0.9, "max_continuity_gap_m": 5000.0,
            "route_component_count": 2, "emissions_eligible": False, "emissions_success": False,
        },
        {
            "user_id": "u3", "trip_id": 1, "physical_trip_id": "u3_2026-01-01_1",
            "route_completeness_status": None, "processing_status": "quality_rejected",
            "final_mode": None, "reconstructed_distance_m": 0.0,
            "gps_distance_m": 0.0, "route_gps_ratio": None,
            "uncovered_fraction": None, "max_continuity_gap_m": None,
            "route_component_count": 0, "emissions_eligible": False, "emissions_success": False,
        },
    ]
    return pd.DataFrame(rows)


def _detailed() -> pd.DataFrame:
    rows = []
    specifications = [
        ("u1", 1, "u1_2026-01-01_1", "Carro", [100.0, 200.0], [25.0, 30.0], 21),
        ("u1", 2, "u1_2026-01-01_2", "Bus", [80.0, 120.0], [18.0, 20.0], 42),
    ]
    for user, trip, physical, mode, distances, speeds, source in specifications:
        for index, (distance, speed) in enumerate(zip(distances, speeds)):
            row = {
                "user_id": user, "trip_id": trip, "physical_trip_id": physical,
                "local_timestamp": pd.Timestamp("2026-01-01 08:00:00") + pd.Timedelta(minutes=index),
                "latitude": 25.0 + index * 0.001, "longitude": -100.0,
                "speed_kmh": speed, "duration_s": 60.0, "distance_m": distance,
                "transport_mode": mode, "routing_failed": False, "route_component_id": 1,
                "route_completeness_status": "complete" if trip == 1 else "partial",
                "emissions_eligible": True, "emissions_success": True,
                "moves_source_type": source, "moves_speed_bin": 4,
                "moves_road_type": 5, "road_lookup_status": "mapped",
                "emission_lookup_status": "exact",
            }
            row.update(POLLUTANT_COLUMNS)
            rows.append(row)
    return pd.DataFrame(rows)


def _write_run(root: Path, *, omit_pm25: bool = False) -> Path:
    run = root / "2026-01-01_test_run"
    pipeline = run / "pipeline"
    pipeline.mkdir(parents=True)
    ledger = _ledger()
    detailed = _detailed()
    if omit_pm25:
        detailed = detailed.drop(columns=["pm25_g_km", "pm25_total_g"])
    summary = detailed.drop(columns=["physical_trip_id", "routing_failed", "route_component_id"])
    ledger.to_parquet(pipeline / "trip_ledger.parquet", index=False)
    detailed.to_parquet(pipeline / "routes_emissions_detailed.parquet", index=False)
    summary.to_parquet(pipeline / "routes_emissions_summary.parquet", index=False)
    (run / "manifest.json").write_text('{"status": "completed", "output_mode": "both"}', encoding="utf-8")
    return run


def test_report_disabled_does_not_touch_filesystem(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert generate_quality_report_if_enabled(False, run_path=missing) is None
    assert not missing.exists()


def test_canonical_notebook_documents_safe_results_report_controls():
    root = Path(__file__).resolve().parents[2]
    notebook = json.loads(
        (root / "notebooks" / "GPS_preprocessing_and_pipeline_v4.ipynb").read_text(encoding="utf-8")
    )
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "## Results Report" in source
    assert "ENABLE_EXTENDED_QUALITY_REPORT = False" in source
    assert "QUALITY_REPORT_RUN_PATH = None" in source
    assert "QUALITY_REPORT_OUTPUT_DIR = None" in source
    assert "QUALITY_REPORT_OVERWRITE = False" in source
    assert "generate_quality_report(" in source
    readme = (root / "pipeline_v4" / "diagnostics" / "README.md").read_text(encoding="utf-8")
    assert "Pipeline-valid routes" in readme
    assert "Plot-quality routes" in readme
    assert "Analyze an existing run" in readme


def test_loading_run_by_root_and_pipeline_path(tmp_path):
    run = _write_run(tmp_path)
    from_root = load_run_outputs(run)
    from_pipeline = load_run_outputs(run / "pipeline")
    from_files = load_run_outputs(
        None,
        ledger_path=run / "pipeline" / "trip_ledger.parquet",
        detailed_path=run / "pipeline" / "routes_emissions_detailed.parquet",
        summary_path=run / "pipeline" / "routes_emissions_summary.parquet",
        manifest_path=run / "manifest.json",
    )
    assert len(from_root.ledger) == len(from_pipeline.ledger) == 4
    assert len(from_files.ledger) == 4
    assert from_root.detailed is not None
    assert from_root.summary is not None


def test_complete_partial_filtering_and_failed_exclusion():
    trips, valid, plot_quality, _, metadata = build_quality_population(_ledger())
    assert set(valid["route_completeness_status"]) == {"complete", "partial"}
    assert set(valid["physical_trip_id"]) == {"u1_2026-01-01_1", "u1_2026-01-01_2"}
    assert len(plot_quality) == 2
    assert metadata["retention_percent"] == 100.0
    assert len(trips) == 4


def test_failed_exclusion_cannot_be_disabled():
    with pytest.raises(ValueError, match="must remain excluded"):
        build_quality_population(_ledger(), QualityReportConfig(exclude_failed=False))


def test_distribution_relative_tail_filter_is_explicit():
    rows = []
    for index in range(24):
        row = _ledger().iloc[0].to_dict()
        row["physical_trip_id"] = f"u1_2026-01-01_{index + 10}"
        row["trip_id"] = index + 10
        row["route_gps_ratio"] = 0.95 + (index % 5) * 0.025
        rows.append(row)
    extreme = rows[-1].copy()
    extreme["physical_trip_id"] = "u1_2026-01-01_999"
    extreme["trip_id"] = 999
    extreme["route_gps_ratio"] = 3.5
    rows.append(extreme)
    _, valid, plot_quality, _, metadata = build_quality_population(pd.DataFrame(rows))
    assert len(valid) == 25
    assert len(plot_quality) == 24
    assert metadata["route_gps_ratio_filter"]["applied"] is True
    excluded = valid.loc[~valid["plot_quality"]]
    assert excluded.iloc[0]["plot_exclusion_reason"] == "extreme_route_gps_ratio_outer_fence"


def test_mode_counts_use_one_row_per_physical_trip():
    _, valid, _, _, _ = build_quality_population(_ledger())
    metrics, table = compute_modal_metrics(valid)
    assert metrics == {"Bus": 1, "Car": 1}
    assert table.set_index("mode").loc["Car", "trips"] == 1


def test_report_uses_ledger_trip_distance_and_aggregates_emissions(tmp_path):
    run = _write_run(tmp_path)
    report = generate_quality_report(run, show_figures=False, save_figures=False)
    distance = report.tables["trip_distance_by_mode"].set_index("physical_trip_id")
    assert distance.loc["u1_2026-01-01_1", "distance_km"] == pytest.approx(0.3)
    assert distance.loc["u1_2026-01-01_2", "distance_km"] == pytest.approx(0.2)
    co = report.tables["emissions_by_pollutant"].set_index("pollutant").loc["CO"]
    assert co["total_mass_g"] == pytest.approx(0.4)
    by_mode = report.tables["emissions_by_mode"].set_index("mode")
    assert by_mode.loc["Car", "CO"] == pytest.approx(0.2)
    assert by_mode.loc["Bus", "CO"] == pytest.approx(0.2)


def test_missing_optional_pollutant_is_handled(tmp_path):
    run = _write_run(tmp_path, omit_pm25=True)
    report = generate_quality_report(run, show_figures=False, save_figures=False)
    pollutants = set(report.tables["emissions_by_pollutant"]["pollutant"])
    assert "PM2.5" not in pollutants
    assert "PM2.5" not in report.metrics["emissions"]["available_pollutants"]


def test_output_directory_markdown_json_and_machine_tables(tmp_path):
    run = _write_run(tmp_path)
    report = generate_quality_report(run, show_figures=False, save_figures=False)
    assert report.markdown_path.exists()
    assert report.metrics_path.exists()
    assert (report.output_dir / "tables" / "mode_summary.csv").exists()
    assert (report.output_dir / "tables" / "mode_summary.parquet").exists()
    markdown = report.markdown_path.read_text(encoding="utf-8")
    assert "## 13. Quality Verdict" in markdown
    assert "Pipeline-valid routes" in markdown
    with pytest.raises(FileExistsError, match="not empty"):
        generate_quality_report(run, show_figures=False, save_figures=False)


def test_all_five_requested_figures_are_generated(tmp_path):
    run = _write_run(tmp_path)
    report = generate_quality_report(run, show_figures=False, save_figures=True)
    assert len(report.figure_paths) == 5
    assert all(path.exists() and path.stat().st_size > 0 for path in report.figure_paths)
