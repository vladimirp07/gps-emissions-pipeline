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
from pipeline_v4.preprocessing.workflow import (
    PreprocessingConfig,
    PreprocessingResult,
    run_preprocessing,
    supplied_user_ids,
)
from pipeline_v4.src import config
from pipeline_v4.src.production_workflow import PipelineV4Result, run_pipeline_v4
from pipeline_v4.src.random_forest_contract import (
    BUS_PROBABILITY_THRESHOLD, METRO_PROBABILITY_THRESHOLD,
    MIN_EFFECTIVE_PINGS, MIN_PCT_CONSERVED, MIN_PCT_CONSERVED_PERCENT,
)


PROJECT_TIMEZONE = "America/Monterrey"
REQUIRED_PYTHON = (3, 12)
REQUIRED_SCIKIT_LEARN = "1.5.2"
OFFICIAL_CLASSIFIER_SHA256 = "f38c0e36f6039ec396e9b30fc0d1b6595b817cee6d4e578facb8fbdb330cd1dd"


def validate_production_environment(
    *, python_version=None, sklearn_version=None,
    modal_classifier=None, classifier_hash=None,
):
    """Fail early when the persisted classifier cannot be reproduced safely."""
    detected_python = tuple(python_version or sys.version_info[:2])
    detected_sklearn = str(sklearn_version or sklearn.__version__)
    detected_classifier = str(modal_classifier or config.MODAL_CLASSIFIER).strip().lower()
    detected_hash = classifier_hash or _sha256(config.FILE_MODAL_HYBRID)
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
    if detected_classifier != "hybrid":
        errors.append(
            f"The frozen production classifier must be 'hybrid'; detected {detected_classifier!r}"
        )
    if detected_hash != OFFICIAL_CLASSIFIER_SHA256:
        errors.append(
            "The frozen production classifier SHA256 does not match "
            f"{OFFICIAL_CLASSIFIER_SHA256}; detected {detected_hash}"
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


def _file_identity(path):
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _jsonable(value):
    return json.loads(json.dumps(value, default=str))


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
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _load_preprocessing_checkpoint(preprocessing_dir: Path) -> PreprocessingResult | None:
    paths = {
        "supplied": preprocessing_dir / "supplied_users.parquet",
        "metadata": preprocessing_dir / "user_home_metadata.parquet",
        "gps": preprocessing_dir / "preprocessed_gps.parquet",
    }
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        return None
    try:
        supplied = pd.read_parquet(paths["supplied"])
        metadata = pd.read_parquet(paths["metadata"])
        gps = pd.read_parquet(paths["gps"])
        if "user_id" not in supplied or "user_id" not in metadata or "caid" not in gps:
            return None
        return PreprocessingResult(supplied, metadata, gps, preprocessing_dir)
    except Exception:
        return None


def run_production(
    source_path,
    ageb_path,
    output_root=None,
    preprocessing_config=None,
    home_config=None,
    user_ids=None,
    save_preprocessed_gps=True,
    n_jobs=None,
    limit_users=None,
    limit_days_per_user=None,
    resources=None,
    output_mode="summary",
    show_progress=True,
    user_day_batch_size=None,
    execution_profile=None,
    n_jobs_override=None,
    backend_override=None,
    batch_size_override=None,
    backend=None,
    now=None,
    resume_run_dir=None,
):
    """Run supplied sample -> preprocessing/home -> pipeline_v4 in a unique directory."""
    environment = validate_production_environment()
    root = Path(output_root or config.OUTPUTS_DIR / "runs")
    input_label = _input_run_label(source_path)
    if resume_run_dir is None:
        run_id = generate_run_id(input_label, output_root=root, now=now)
        run_dir = root / run_id
    else:
        run_dir = Path(resume_run_dir).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume_run_dir does not exist: {run_dir}")
        run_id = run_dir.name
    preprocessing_dir = run_dir / "preprocessing"
    pipeline_dir = run_dir / "pipeline"
    figures_dir = run_dir / "figures"
    for directory in (preprocessing_dir, pipeline_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=resume_run_dir is not None or directory != preprocessing_dir)
    start = datetime.now(ZoneInfo(PROJECT_TIMEZONE))
    pre_cfg = preprocessing_config or PreprocessingConfig()
    home_cfg = home_config or HomeConfig(min_nights=config.HOME_MIN_NIGHTS)
    effective_user_ids = user_ids
    if limit_users is not None:
        if user_ids is None:
            # Select by first appearance in the supplied source.  Preprocessing
            # sorts rows by user for its transformations, so applying the limit
            # only after preprocessing would silently change the user's choice.
            effective_user_ids = supplied_user_ids(source_path, pre_cfg, limit_users=limit_users)["user_id"].head(limit_users).tolist()
        else:
            effective_user_ids = list(dict.fromkeys(user_ids))[:limit_users]
    commit, dirty = _git_metadata()
    artifact = config.FILE_MODAL_HYBRID if config.MODAL_CLASSIFIER == "hybrid" else config.FILE_MODAL_RANDOM_FOREST

    # Resolve execution profile
    profile_info = config.resolve_execution_profile(
        execution_profile,
        n_jobs_override=n_jobs_override if n_jobs_override is not None else n_jobs,
        backend_override=backend_override or backend,
        batch_size_override=batch_size_override if batch_size_override is not None else user_day_batch_size,
    )
    resume_contract = {
        "input": _file_identity(source_path),
        "ageb": _file_identity(ageb_path),
        "preprocessing_config": _jsonable(asdict(pre_cfg)),
        "home_config": _jsonable(asdict(home_cfg)),
        "user_ids": _jsonable(list(user_ids)) if user_ids is not None else None,
        "limit_users": limit_users,
        "limit_days_per_user": limit_days_per_user,
        "classifier_backend": config.MODAL_CLASSIFIER,
        "classifier_artifact_sha256": _sha256(artifact),
    }
    if resume_run_dir is not None:
        existing_manifest_path = run_dir / "manifest.json"
        try:
            existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Cannot safely resume without a readable manifest: {existing_manifest_path}"
            ) from exc
        if existing_manifest.get("resume_contract") != resume_contract:
            raise ValueError("resume_run_dir is incompatible with the current input or scientific workload")

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
        "resume_contract": resume_contract,
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
                "metro_threshold": METRO_PROBABILITY_THRESHOLD,
                "bus_threshold": BUS_PROBABILITY_THRESHOLD,
                "min_effective_pings": MIN_EFFECTIVE_PINGS,
                "min_pct_pings_conserved": MIN_PCT_CONSERVED,
                "min_pct_pings_conserved_percent": MIN_PCT_CONSERVED_PERCENT,
            },
        },
        "preprocessing_config": asdict(pre_cfg),
        "save_preprocessed_gps": bool(save_preprocessed_gps),
        "output_mode": output_mode,
        "execution": {
            "n_jobs": profile_info["effective_n_jobs"],
            "execution_profile": profile_info["execution_profile"],
            "backend": profile_info["backend"],
            "requested_n_jobs": profile_info["requested_n_jobs"],
            "effective_n_jobs": profile_info["effective_n_jobs"],
            "user_day_batch_size": profile_info["user_day_batch_size"],
            "guardrail_warning": profile_info["guardrail_warning"],
            "internal_thread_limits": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "1"),
            },
            "hostname": platform.node(),
            "python_executable": sys.executable,
            "system_platform": sys.platform,
            "total_ram_gb": profile_info["total_ram_gb"],
            "available_ram_gb": profile_info["available_ram_gb"],
            "limit_users": limit_users,
            "limit_days_per_user": limit_days_per_user,
        },
        "moves_rate_unit_status": "confirmed",
        "moves_distance_rate_unit": config.EMISSION_RATE_DISTANCE_UNIT,
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
        preprocessing = (
            _load_preprocessing_checkpoint(preprocessing_dir)
            if resume_run_dir is not None else None
        )
        if preprocessing is None:
            print("[Run 1/2] Starting GPS preprocessing, home metadata, and AGEB attachment...", flush=True)
            preprocessing = run_preprocessing(
                source_path, ageb_path, preprocessing_dir, config=pre_cfg,
                home_config=home_cfg, user_ids=effective_user_ids,
                save_preprocessed_gps=save_preprocessed_gps,
            )
        else:
            print("[Run 1/2] Resuming from validated preprocessing artifacts.", flush=True)
        print(
            f"[Run 1/2] Preprocessing complete for {len(preprocessing.supplied_users)} supplied user(s).",
            flush=True,
        )
        print("[Run 2/2] Starting routing, modal inference, emissions, and output generation...", flush=True)
        pipeline = run_pipeline_v4(
            preprocessing.preprocessed_gps, pipeline_dir, preprocessing.user_metadata,
            limit_users=limit_users,
            limit_days_per_user=limit_days_per_user, resources=resources,
            output_mode=output_mode, show_progress=show_progress,
            execution_profile=profile_info["execution_profile"],
            n_jobs_override=profile_info["effective_n_jobs"],
            backend_override=profile_info["backend"],
            batch_size_override=profile_info["user_day_batch_size"],
            resume=True,
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
            "status": pipeline.output_artifacts.get(
                "status", "completed_with_trip_failures" if failures else "completed"
            ),
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
            "supplied_users_artifact": "preprocessing/supplied_users.parquet",
            "date_range": {
                "requested_start": pre_cfg.start_date,
                "requested_end": pre_cfg.end_date,
                "actual_start": (
                    pd.to_datetime(preprocessing.preprocessed_gps["local_timestamp"]).min().isoformat()
                    if "local_timestamp" in preprocessing.preprocessed_gps and not preprocessing.preprocessed_gps.empty else None
                ),
                "actual_end": (
                    pd.to_datetime(preprocessing.preprocessed_gps["local_timestamp"]).max().isoformat()
                    if "local_timestamp" in preprocessing.preprocessed_gps and not preprocessing.preprocessed_gps.empty else None
                ),
            },
        })
        manifest["hashes"]["supplied_users_sha256"] = _sha256(
            preprocessing_dir / "supplied_users.parquet"
        )
        print("[Run 2/2] Pipeline processing complete.", flush=True)
    except Exception as exc:
        manifest.update({
            "end_timestamp": datetime.now(ZoneInfo(PROJECT_TIMEZONE)).isoformat(),
            "status": "failed", "failure": f"{type(exc).__name__}: {exc}",
        })
        _write_manifest(run_dir / "manifest.json", manifest)
        raise
    _write_manifest(run_dir / "manifest.json", manifest)
    print(f"[Run] Finished with status {manifest['status']}: {run_dir}", flush=True)
    return ProductionRunResult(run_id, run_dir, preprocessing, pipeline, manifest)
