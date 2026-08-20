from datetime import datetime
import json
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
    assert manifest["moves_rate_unit_status"] == "pending_external_confirmation"
    assert manifest["parameters"]["home"] == {
        "night_start": 22, "night_end": 5, "min_nights": 3,
    }
    assert manifest["execution"] == {
        "n_jobs": 2, "limit_users": None, "limit_days_per_user": None,
        "user_day_batch_size": 32,
    }
    assert manifest["output_mode"] == "summary"
    assert manifest["run_label"] == "input"


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
        lambda source, config: pd.DataFrame({"user_id": ["U3", "U1", "U2"]}),
    )

    def capture_preprocessing(source_path, ageb_path, output_dir, **kwargs):
        seen["preprocessing_user_ids"] = kwargs["user_ids"]
        return _fake_preprocessing(source_path, ageb_path, output_dir, **kwargs)

    def capture_pipeline(gps, output_dir, metadata, **kwargs):
        seen["pipeline_limit_users"] = kwargs["limit_users"]
        return _fake_pipeline(gps, output_dir, metadata, **kwargs)

    monkeypatch.setattr(run_workflow, "run_preprocessing", capture_preprocessing)
    monkeypatch.setattr(run_workflow, "run_pipeline_v4", capture_pipeline)
    source, ageb = tmp_path / "input.parquet", tmp_path / "ageb.geojson"
    source.touch(); ageb.touch()

    run_workflow.run_production(source, ageb, output_root=tmp_path / "runs", limit_users=2)

    assert seen["preprocessing_user_ids"] == ["U3", "U1"]
    assert seen["pipeline_limit_users"] == 2
