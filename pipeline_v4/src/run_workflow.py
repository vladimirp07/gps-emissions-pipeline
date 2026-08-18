"""Lightweight run-scoped orchestration for the canonical production notebook."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Collection
from zoneinfo import ZoneInfo

import sklearn
import pandas as pd

from pipeline_v4.preprocessing.gps_home_sampling.workflow import HomeConfig
from pipeline_v4.preprocessing.workflow import PreprocessingConfig, PreprocessingResult, run_preprocessing
from pipeline_v4.src import config
from pipeline_v4.src.production_workflow import PipelineV4Result, run_pipeline_v4
from pipeline_v4.src.random_forest_contract import (
    BUS_PROBABILITY_THRESHOLD, MIN_EFFECTIVE_PINGS, MIN_PCT_CONSERVED,
)


PROJECT_TIMEZONE = "America/Monterrey"
REQUIRED_PYTHON = (3, 12)
REQUIRED_SCIKIT_LEARN = "1.5.2"


def validate_production_environment(*, python_version=None, sklearn_version=None):
    """Fail early when the persisted classifier cannot be reproduced safely."""
    detected_python = tuple(python_version or sys.version_info[:2])
    detected_sklearn = str(sklearn_version or sklearn.__version__)
    errors = []
    if detected_python != REQUIRED_PYTHON:
        errors.append(
            f"Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]} is required; "
            f"detected {detected_python[0]}.{detected_python[1]}"
        )
    if detected_sklearn != REQUIRED_SCIKIT_LEARN:
        errors.append(
            f"scikit-learn {REQUIRED_SCIKIT_LEARN} is required by the production "
            f"classifier artifact; detected {detected_sklearn}"
        )
    if errors:
        raise RuntimeError("Incompatible production environment: " + "; ".join(errors))
    return {
        "python": f"{detected_python[0]}.{detected_python[1]}",
        "scikit_learn": detected_sklearn,
        "status": "compatible",
    }


@dataclass(frozen=True)
class ProductionRunResult:
    run_id: str
    run_dir: Path
    preprocessing: PreprocessingResult
    pipeline: PipelineV4Result
    manifest: dict


def _safe_label(value: str | None) -> str:
    if not value:
        return ""
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in str(value).strip())
    return "_".join(filter(None, normalized.split("_")))


def _input_run_label(source_path) -> str:
    """Derive the run label from the supplied input filename."""
    label = _safe_label(Path(source_path).stem)
    if not label:
        raise ValueError("source_path must have a usable filename for the run ID")
    return label


def generate_run_id(run_label=None, *, output_root=None, now=None):
    root = Path(output_root or config.OUTPUTS_DIR / "runs")
    instant = now or datetime.now(ZoneInfo(PROJECT_TIMEZONE))
    base = instant.strftime("%Y-%m-%d_%H%M%S")
    label = _safe_label(run_label)
    if label:
        base = f"{base}_{label}"
    candidate, suffix = base, 2
    while (root / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def _sha256(path):
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata():
    root = config.PROJECT_ROOT
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")


def run_production(
    source_path,
    ageb_path,
    *,
    output_root=None,
    preprocessing_config=None,
    home_config=None,
    user_ids: Collection[object] | None = None,
    save_preprocessed_gps=True,
    n_jobs=2,
    limit_users=None,
    limit_days_per_user=None,
    resources=None,
    output_mode="summary",
    show_progress=True,
    now=None,
):
    """Run supplied sample -> preprocessing/home -> pipeline_v4 in a unique directory."""
    environment = validate_production_environment()
    root = Path(output_root or config.OUTPUTS_DIR / "runs")
    input_label = _input_run_label(source_path)
    run_id = generate_run_id(input_label, output_root=root, now=now)
    run_dir = root / run_id
    preprocessing_dir = run_dir / "preprocessing"
    pipeline_dir = run_dir / "pipeline"
    figures_dir = run_dir / "figures"
    for directory in (preprocessing_dir, pipeline_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=False if directory == preprocessing_dir else True)
    start = datetime.now(ZoneInfo(PROJECT_TIMEZONE))
    pre_cfg = preprocessing_config or PreprocessingConfig()
    home_cfg = home_config or HomeConfig(min_nights=config.HOME_MIN_NIGHTS)
    commit, dirty = _git_metadata()
    artifact = config.FILE_MODAL_HYBRID if config.MODAL_CLASSIFIER == "hybrid" else config.FILE_MODAL_RANDOM_FOREST
    manifest = {
        "run_id": run_id, "run_label": input_label,
        "start_timestamp": start.isoformat(), "end_timestamp": None, "status": "running",
        "input_path": str(Path(source_path).resolve()),
        "ageb_path": str(Path(ageb_path).resolve()),
        "supplied_users": None, "ready_for_pipeline_users": None,
        "trips": None, "successful_trips": None,
        "classifier_backend": config.MODAL_CLASSIFIER,
        "classifier_artifact": str(Path(artifact).resolve()),
        "python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__,
        "production_environment_validation": environment,
        "git_commit": commit, "git_dirty": dirty,
        "parameters": {
            "home": {
                "night_start": home_cfg.night_start,
                "night_end": home_cfg.night_end,
                "min_nights": home_cfg.min_nights,
            },
            "segmentation": {
                "downsampling_seconds": 10, "spatial_filter_m": 15.0,
                "stop_speed_kmh": 3.0, "stop_time_s": 300, "gap_threshold_s": 1800,
            },
            "routing": {
                "walk_buffer_m": 50.0, "drive_buffer_m": 150.0,
                "physics_factor": float(os.environ.get("PHYSICS_FACTOR", "2.0")),
                "router_version": config.ROUTER_VERSION,
                "max_lookahead_skipped_pings": config.MAX_LOOKAHEAD_SKIPPED_PINGS,
            },
            "classification": {
                "classifier_name": config.MODAL_CLASSIFIER,
                "bus_threshold": BUS_PROBABILITY_THRESHOLD,
                "min_effective_pings": MIN_EFFECTIVE_PINGS,
                "min_pct_pings_conserved": MIN_PCT_CONSERVED,
            },
        },
        "preprocessing_config": asdict(pre_cfg),
        "save_preprocessed_gps": bool(save_preprocessed_gps),
        "output_mode": output_mode,
        "execution": {
            "n_jobs": int(n_jobs),
            "limit_users": limit_users,
            "limit_days_per_user": limit_days_per_user,
        },
        "moves_rate_unit_status": "pending_external_confirmation",
        "hashes": {
            "classifier_artifact_sha256": _sha256(artifact),
            "config_py_sha256": _sha256(config.PROJECT_ROOT / "pipeline_v4" / "src" / "config.py"),
            "random_forest_contract_sha256": _sha256(
                config.PROJECT_ROOT / "pipeline_v4" / "src" / "random_forest_contract.py"
            ),
            "routing_py_sha256": _sha256(config.PROJECT_ROOT / "pipeline_v4" / "src" / "routing.py"),
            "endpoint_routing_py_sha256": _sha256(
                config.PROJECT_ROOT / "pipeline_v4" / "src" / "endpoint_routing.py"
            ),
            "route_quality_py_sha256": _sha256(
                config.PROJECT_ROOT / "pipeline_v4" / "src" / "route_quality.py"
            ),
        },
    }
    _write_manifest(run_dir / "manifest.json", manifest)
    try:
        print("[Run 1/2] Starting GPS preprocessing, home metadata, and AGEB attachment...", flush=True)
        preprocessing = run_preprocessing(
            source_path, ageb_path, preprocessing_dir, config=pre_cfg,
            home_config=home_cfg, user_ids=user_ids,
            save_preprocessed_gps=save_preprocessed_gps,
        )
        print(
            f"[Run 1/2] Preprocessing complete for {len(preprocessing.supplied_users)} supplied user(s).",
            flush=True,
        )
        print("[Run 2/2] Starting routing, modal inference, emissions, and output generation...", flush=True)
        pipeline = run_pipeline_v4(
            preprocessing.preprocessed_gps, pipeline_dir, preprocessing.user_metadata,
            n_jobs=n_jobs, limit_users=limit_users,
            limit_days_per_user=limit_days_per_user, resources=resources,
            output_mode=output_mode, show_progress=show_progress,
        )
        failures = int((pipeline.trip_ledger.processing_status != "success").sum())
        metadata = preprocessing.user_metadata
        routing_eligible = (
            metadata["routing_eligible"].fillna(False).astype(bool)
            if "routing_eligible" in metadata.columns
            else metadata.processing_status.eq("ready_for_pipeline")
        )
        home_inventory_eligible = (
            metadata["home_eligible_for_inventory"].fillna(False).astype(bool)
            if "home_eligible_for_inventory" in metadata.columns
            else metadata.get("home_quality_flag", pd.Series(index=metadata.index, dtype=object))
                .isin(("reliable", "probable"))
        )
        manifest.update({
            "end_timestamp": datetime.now(ZoneInfo(PROJECT_TIMEZONE)).isoformat(),
            "status": "completed_with_trip_failures" if failures else "completed",
            "supplied_users": int(len(preprocessing.supplied_users)),
            "ready_for_pipeline_users": int(routing_eligible.sum()),
            "home_eligible_for_inventory_users": int(home_inventory_eligible.sum()),
            "trips": int(len(pipeline.trip_ledger)),
            "successful_trips": int(pipeline.trip_ledger.processing_status.eq("success").sum()),
            "trip_status_counts": pipeline.trip_ledger.processing_status.value_counts().to_dict(),
            "route_completeness_counts": (
                pipeline.trip_ledger["route_completeness_status"].value_counts(dropna=False).to_dict()
                if "route_completeness_status" in pipeline.trip_ledger.columns else {"unknown": int(len(pipeline.trip_ledger))}
            ),
            "pipeline_outputs": pipeline.output_artifacts,
        })
        print("[Run 2/2] Pipeline processing complete.", flush=True)
    except Exception as exc:
        manifest.update({
            "end_timestamp": datetime.now(ZoneInfo(PROJECT_TIMEZONE)).isoformat(),
            "status": "failed", "failure": f"{type(exc).__name__}: {exc}",
        })
        _write_manifest(run_dir / "manifest.json", manifest)
        raise
    _write_manifest(run_dir / "manifest.json", manifest)
    print(f"[Run] Completed successfully: {run_dir}", flush=True)
    return ProductionRunResult(run_id, run_dir, preprocessing, pipeline, manifest)
