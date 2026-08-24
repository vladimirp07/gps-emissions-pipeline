from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from pipeline_v4.preprocessing.workflow import PreprocessingResult
from pipeline_v4.src import run_workflow
from pipeline_v4.src.production_workflow import PipelineV4Result


def _fake_preprocessing(source_path, ageb_path, output_dir, **kwargs):
    output_dir.mkdir(parents=True, exist_ok=True)
    supplied = pd.DataFrame({"user_id": ["U1"]})
    metadata = pd.DataFrame({"user_id": ["U1"], "processing_status": ["ready_for_pipeline"]})
    gps = pd.DataFrame({"caid": ["U1"]})
    supplied.to_parquet(output_dir / "supplied_users.parquet", index=False)
    metadata.to_parquet(output_dir / "user_home_metadata.parquet", index=False)
    if kwargs.get("save_preprocessed_gps", True):
        gps.to_parquet(output_dir / "preprocessed_gps.parquet", index=False)
    return PreprocessingResult(supplied, metadata, gps, output_dir)


def _fake_pipeline(gps, output_dir, metadata, **kwargs):
    output_dir.mkdir(parents=True, exist_ok=True)
    routes = pd.DataFrame({"physical_trip_id": ["U1_2026-08-18_1"]})
    emissions = routes.copy()
    ledger = pd.DataFrame({"physical_trip_id": ["U1_2026-08-18_1"], "processing_status": ["success"]})
    routes.to_parquet(output_dir / "routes_emissions_summary.parquet", index=False)
    ledger.to_parquet(output_dir / "trip_ledger.parquet", index=False)
    return PipelineV4Result(routes, emissions, ledger, output_dir)


def test_two_runs_never_overwrite_and_manifest_is_complete(monkeypatch, tmp_path):
    monkeypatch.setattr(run_workflow, "run_preprocessing", _fake_preprocessing)
    monkeypatch.setattr(run_workflow, "run_pipeline_v4", _fake_pipeline)
    source, ageb = tmp_path / "input.parquet", tmp_path / "ageb.geojson"
    source.touch(); ageb.touch()
    fixed = datetime(2026, 8, 18, 5, 14, 23, tzinfo=ZoneInfo("America/Monterrey"))
    first = run_workflow.run_production(source, ageb, output_root=tmp_path / "runs", now=fixed)
    second = run_workflow.run_production(source, ageb, output_root=tmp_path / "runs", now=fixed)
    assert first.run_id == "2026-08-18_051423_input"
    assert second.run_id == "2026-08-18_051423_input_02"
    assert first.run_dir != second.run_dir
    assert (first.run_dir / "pipeline" / "trip_ledger.parquet").exists()
    assert (first.run_dir / "figures").is_dir()
    manifest = json.loads((first.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["moves_rate_unit_status"] == "confirmed"
    assert manifest["moves_distance_rate_unit"] == "g/km"
    assert manifest["parameters"]["home"] == {
        "night_start": 22, "night_end": 5, "min_nights": 3,
    }
    assert manifest["execution"]["n_jobs"] == 2
    assert manifest["execution"]["execution_profile"] == "LOCAL_SAFE"
    assert manifest["execution"]["backend"] == "threading"
    assert manifest["execution"]["user_day_batch_size"] == 32
    assert manifest["execution"]["limit_users"] is None
    assert manifest["execution"]["limit_days_per_user"] is None
    assert manifest["output_mode"] == "summary"
    assert manifest["run_label"] == "input"
    assert manifest["supplied_users_artifact"] == "preprocessing/supplied_users.parquet"
    assert len(manifest["hashes"]["supplied_users_sha256"]) == 64


def test_save_preprocessed_gps_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(run_workflow, "run_preprocessing", _fake_preprocessing)
    monkeypatch.setattr(run_workflow, "run_pipeline_v4", _fake_pipeline)
    source, ageb = tmp_path / "input.parquet", tmp_path / "ageb.geojson"
    source.touch(); ageb.touch()
    result = run_workflow.run_production(
        source, ageb, output_root=tmp_path / "runs", save_preprocessed_gps=False
    )
    assert not (result.run_dir / "preprocessing" / "preprocessed_gps.parquet").exists()


def test_limit_users_uses_first_source_order_for_both_stages(monkeypatch, tmp_path):
    seen = {}

    monkeypatch.setattr(
        run_workflow,
        "supplied_user_ids",
        lambda *a, **k: pd.DataFrame({"user_id": ["u_first", "u_second", "u_third"]}),
    )

    def _spy_preprocessing(source_path, ageb_path, output_dir, **kwargs):
        seen["preprocessing_user_ids"] = kwargs.get("user_ids")
        return _fake_preprocessing(source_path, ageb_path, output_dir, **kwargs)

    def _spy_pipeline(gps, output_dir, metadata, **kwargs):
        seen["pipeline_limit_users"] = kwargs.get("limit_users")
        return _fake_pipeline(gps, output_dir, metadata, **kwargs)

    monkeypatch.setattr(run_workflow, "run_preprocessing", _spy_preprocessing)
    monkeypatch.setattr(run_workflow, "run_pipeline_v4", _spy_pipeline)
    source, ageb = tmp_path / "input.parquet", tmp_path / "ageb.geojson"
    source.touch(); ageb.touch()
    run_workflow.run_production(
        source, ageb, output_root=tmp_path / "runs", limit_users=2
    )
    assert seen["preprocessing_user_ids"] == ["u_first", "u_second"]
    assert seen["pipeline_limit_users"] == 2


def test_notebook_code_cells_compile():
    notebook_path = Path(__file__).resolve().parents[2] / "notebooks" / "GPS_preprocessing_and_pipeline_v4.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"notebook-cell-{index}", "exec")


def test_outer_manifest_does_not_report_completed_after_detailed_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(run_workflow, "run_preprocessing", _fake_preprocessing)

    def failed_detailed(gps, output_dir, metadata, **kwargs):
        result = _fake_pipeline(gps, output_dir, metadata, **kwargs)
        return PipelineV4Result(
            result.routes, result.individual_emissions, result.trip_ledger,
            result.output_dir,
            {"status": "scientific_computation_complete_detailed_failed",
             "detailed_output": {"generated": False, "error": "injected"}},
        )

    monkeypatch.setattr(run_workflow, "run_pipeline_v4", failed_detailed)
    source, ageb = tmp_path / "input.parquet", tmp_path / "ageb.geojson"
    source.touch(); ageb.touch()
    result = run_workflow.run_production(
        source, ageb, output_root=tmp_path / "runs", output_mode="both"
    )
    assert result.manifest["status"] == "scientific_computation_complete_detailed_failed"


def test_official_entry_point_resumes_existing_run_without_reprocessing(monkeypatch, tmp_path):
    calls = {"preprocessing": 0, "pipeline": 0}

    def counted_preprocessing(*args, **kwargs):
        calls["preprocessing"] += 1
        return _fake_preprocessing(*args, **kwargs)

    def counted_pipeline(*args, **kwargs):
        calls["pipeline"] += 1
        assert kwargs["resume"] is True
        return _fake_pipeline(*args, **kwargs)

    monkeypatch.setattr(run_workflow, "run_preprocessing", counted_preprocessing)
    monkeypatch.setattr(run_workflow, "run_pipeline_v4", counted_pipeline)
    source, ageb = tmp_path / "input.parquet", tmp_path / "ageb.geojson"
    source.touch(); ageb.touch()
    first = run_workflow.run_production(source, ageb, output_root=tmp_path / "runs")
    resumed = run_workflow.run_production(source, ageb, resume_run_dir=first.run_dir)
    assert resumed.run_id == first.run_id
    assert resumed.run_dir == first.run_dir
    assert calls == {"preprocessing": 1, "pipeline": 2}
