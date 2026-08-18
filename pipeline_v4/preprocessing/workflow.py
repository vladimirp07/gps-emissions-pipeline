"""Reproducible handoff from raw VeraSet GPS to ``pipeline_v4``.

This module composes existing home-inference and production-cleaning code.  It
does not perform routing, modal classification, emissions, or inventory work.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Collection

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline_v4.preprocessing.gps_home_sampling.workflow import (
    HomeConfig,
    assign_ageb,
    infer_home_locations,
    read_ageb,
    standardize_gps,
)
from pipeline_v4.src.segmentation import preprocess_gps_frame


@dataclass(frozen=True)
class PreprocessingConfig:
    start_date: str | None = None
    end_date: str | None = None
    # Half-open UTC acquisition interval. Naive values follow the raw input
    # contract and are interpreted as UTC.
    coverage_start: str | None = None
    coverage_end: str | None = None
    user_column: str = "caid"
    min_preprocessed_records: int = 2
    accepted_home_quality: tuple[str, ...] = ("reliable", "probable")


@dataclass(frozen=True)
class PreprocessingResult:
    supplied_users: pd.DataFrame
    user_metadata: pd.DataFrame
    preprocessed_gps: pd.DataFrame
    output_dir: Path


def _timestamp_column(columns: Collection[str]) -> str:
    for name in ("utc_timestamp", "local_timestamp", "timestamp", "datetime", "date"):
        if name in columns:
            return name
    raise KeyError("No timestamp column was found in the raw GPS source")


def _as_utc(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def _utc_boundary(value: str) -> pd.Timestamp:
    boundary = pd.Timestamp(value)
    return boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")


def effective_coverage(config: PreprocessingConfig):
    """Intersect declared acquisition coverage with any requested UTC slice."""
    starts = []
    ends = []
    if config.coverage_start is not None:
        starts.append(_utc_boundary(config.coverage_start))
    if config.coverage_end is not None:
        ends.append(_utc_boundary(config.coverage_end))
    if config.start_date is not None:
        starts.append(_utc_boundary(config.start_date))
    if config.end_date is not None:
        ends.append(_utc_boundary(config.end_date) + pd.Timedelta(days=1))
    start = max(starts) if starts else None
    end = min(ends) if ends else None
    if start is not None and end is not None and end <= start:
        raise ValueError("Effective input coverage is empty or inverted")
    return start, end


def supplied_user_ids(
    source_path: str | Path,
    config: PreprocessingConfig,
    user_ids: Collection[object] | None = None,
) -> pd.DataFrame:
    """Return every externally supplied user, without sampling or replacement."""
    source = Path(source_path)
    columns = pq.ParquetFile(source).schema.names
    if config.user_column not in columns:
        raise KeyError(f"Missing user column: {config.user_column}")
    if user_ids is None:
        # ``unique`` preserves first appearance and the original scalar dtype.
        selected = pd.read_parquet(source, columns=[config.user_column])[config.user_column].dropna().unique().tolist()
        input_source = "all_users_in_supplied_dataset"
    else:
        selected = list(dict.fromkeys(user_ids))
        input_source = "explicit_user_list"
    if not selected:
        raise ValueError("The supplied sample contains no user IDs")
    return pd.DataFrame({
        "user_id": selected,
        "input_order": np.arange(1, len(selected) + 1),
        "input_source": input_source,
    })


def load_supplied_gps(
    source_path: str | Path,
    selected_users: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Load the supplied users without choosing, substituting, or resampling."""
    source = Path(source_path)
    ids = selected_users["user_id"].tolist()
    try:
        raw = pd.read_parquet(source, filters=[(config.user_column, "in", ids)])
    except (TypeError, ValueError, NotImplementedError):
        raw = pd.read_parquet(source)
        raw = raw[raw[config.user_column].isin(ids)].copy()
    timestamp = _timestamp_column(raw.columns)
    if config.start_date is not None or config.end_date is not None:
        observed = _as_utc(raw[timestamp])
        mask = observed.notna()
        if config.start_date is not None:
            mask &= observed >= _utc_boundary(config.start_date)
        if config.end_date is not None:
            mask &= observed < _utc_boundary(config.end_date) + pd.Timedelta(days=1)
        raw = raw.loc[mask].copy()
    return raw


def _build_user_metadata(
    selected: pd.DataFrame,
    canonical_raw: pd.DataFrame,
    prepared: pd.DataFrame,
    homes: pd.DataFrame,
    config: PreprocessingConfig,
    home_config: HomeConfig,
) -> pd.DataFrame:
    observations = canonical_raw.assign(
        observed_date=canonical_raw["timestamp_local"].dt.date,
        is_night=(canonical_raw["timestamp_local"].dt.hour >= home_config.night_start)
        | (canonical_raw["timestamp_local"].dt.hour < home_config.night_end),
    )
    raw_stats = observations.groupby("user_id", as_index=False).agg(
        raw_records=("user_id", "size"),
        days_observed=("observed_date", "nunique"),
    )
    night_stats = observations.loc[observations.is_night].groupby("user_id", as_index=False).agg(
        nights_with_observations=("observed_date", "nunique")
    )
    raw_stats = raw_stats.merge(night_stats, on="user_id", how="left")
    if prepared.empty:
        prepared_stats = pd.DataFrame(columns=["user_id", "preprocessed_records", "preprocessed_days"])
    else:
        prepared_stats = prepared.groupby("caid", as_index=False).agg(
            preprocessed_records=("caid", "size"),
            preprocessed_days=("date", "nunique"),
        ).rename(columns={"caid": "user_id"})
    master = selected.merge(raw_stats, on="user_id", how="left")
    master = master.merge(homes, on="user_id", how="left")
    master = master.merge(prepared_stats, on="user_id", how="left")
    for column in ("raw_records", "days_observed", "nights_with_observations", "preprocessed_records", "preprocessed_days"):
        master[column] = master[column].fillna(0).astype(int)
    has_gps = master.preprocessed_records >= config.min_preprocessed_records
    has_home = master.home_lat.notna() & master.home_lon.notna()
    accepted_home = master.home_quality_flag.isin(config.accepted_home_quality)
    # Mobility processing and residential suitability are independent
    # contracts.  Home quality is metadata for later spatial/inventory use and
    # must not prevent technically valid GPS from reaching routing.
    master["routing_eligible"] = has_gps.astype(bool)
    master["routing_eligibility_reason"] = np.where(
        has_gps, "gps_qc_passed", "insufficient_preprocessed_gps"
    )
    master["home_eligible_for_inventory"] = (has_home & accepted_home).astype(bool)
    master["home_inventory_status"] = np.select(
        [~has_home, ~accepted_home],
        ["home_not_inferred", "home_requires_review"],
        default="home_eligible",
    )
    # Retain the established status field as the pipeline handoff status so
    # existing consumers continue to work.
    master["processing_status"] = np.where(
        has_gps, "ready_for_pipeline", "no_valid_gps"
    )
    master["quality_flags"] = master.apply(
        lambda row: ";".join(filter(None, [
            str(row.get("home_quality_flag", "")) if pd.notna(row.get("home_quality_flag")) else "home_missing",
            "ageb_missing" if pd.isna(row.get("home_ageb")) else "",
            "gps_insufficient" if row["preprocessed_records"] < config.min_preprocessed_records else "",
        ])), axis=1,
    )
    return master


def run_preprocessing(
    source_path: str | Path,
    ageb_path: str | Path,
    output_dir: str | Path,
    config: PreprocessingConfig | None = None,
    home_config: HomeConfig | None = None,
    user_ids: Collection[object] | None = None,
    save_preprocessed_gps: bool = True,
) -> PreprocessingResult:
    """Run QC, home inference, AGEB assignment and the pipeline handoff."""
    cfg = config or PreprocessingConfig()
    home_cfg = home_config or HomeConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    selected = supplied_user_ids(source_path, cfg, user_ids)
    print("[Preprocessing] Loading selected GPS records...", flush=True)
    raw = load_supplied_gps(source_path, selected, cfg)
    canonical_input = raw
    if cfg.user_column not in ("user_id", "caid", "device_id", "id"):
        canonical_input = raw.rename(columns={cfg.user_column: "user_id"})
    print("[Preprocessing] Standardizing timestamps and coordinates...", flush=True)
    canonical = standardize_gps(canonical_input, source_kind="raw", timezone=home_cfg.timezone)
    print("[Preprocessing] Inferring home locations...", flush=True)
    homes = infer_home_locations(canonical, home_cfg)
    if homes.empty:
        homes = pd.DataFrame(columns=[
            "user_id", "home_lat", "home_lon", "n_nights_observed",
            "n_nights_home_detected", "home_confidence", "home_quality_flag",
        ])
    if homes[["home_lat", "home_lon"]].notna().all(axis=1).any():
        print("[Preprocessing] Attaching home AGEB values...", flush=True)
        homes, _ = assign_ageb(homes, read_ageb(ageb_path))
    else:
        print("[Preprocessing] No attachable home locations; leaving home AGEB values empty.", flush=True)
        homes["home_ageb"] = pd.NA

    print("[Preprocessing] Cleaning GPS records and preparing routing input...", flush=True)
    pipeline_input = raw.copy()
    if "caid" not in pipeline_input.columns:
        pipeline_input["caid"] = pipeline_input[cfg.user_column]
    coverage_start, coverage_end = effective_coverage(cfg)
    prepared, _ = preprocess_gps_frame(
        pipeline_input, user_ids=selected.user_id.tolist(),
        coverage_start=coverage_start, coverage_end=coverage_end,
        timezone=home_cfg.timezone,
    )
    day_completeness = prepared.attrs.get("day_completeness", {})
    if "user_id" not in prepared.columns:
        prepared["user_id"] = prepared["caid"]
    master = _build_user_metadata(selected, canonical, prepared, homes, cfg, home_cfg)
    ready_ids = master.loc[master.routing_eligible, "user_id"]
    prepared = prepared[prepared.caid.isin(ready_ids)].copy()

    print("[Preprocessing] Writing preprocessing artifacts...", flush=True)
    selected.to_parquet(output / "supplied_users.parquet", index=False)
    master.to_parquet(output / "user_home_metadata.parquet", index=False)
    if save_preprocessed_gps:
        prepared.to_parquet(output / "preprocessed_gps.parquet", index=False)
    manifest = {
        "stage": "preprocessing",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(Path(source_path).resolve()),
        "ageb_path": str(Path(ageb_path).resolve()),
        "config": asdict(cfg),
        "home_config": asdict(home_cfg),
        "input_coverage": day_completeness,
        "supplied_users": int(len(selected)),
        "ready_users": int(master.routing_eligible.sum()),
        "home_eligible_for_inventory_users": int(master.home_eligible_for_inventory.sum()),
        "preprocessed_rows": int(len(prepared)),
        "preprocessed_gps_saved": bool(save_preprocessed_gps),
    }
    (output / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print("[Preprocessing] Complete.", flush=True)
    return PreprocessingResult(selected, master, prepared, output)


def attach_user_metadata(frame: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Attach residence fields after pipeline calculations, without affecting them."""
    user_column = "caid" if "caid" in frame.columns else "user_id"
    fields = [
        "user_id", "home_lat", "home_lon", "home_ageb", "home_confidence",
        "home_quality_flag", "days_observed", "n_nights_observed", "processing_status",
        "routing_eligible", "routing_eligibility_reason",
        "home_eligible_for_inventory", "home_inventory_status",
    ]
    available = [column for column in fields if column in metadata.columns]
    right = metadata[available].drop_duplicates("user_id")
    if user_column == "caid":
        right["caid"] = right["user_id"]
        if "user_id" in frame.columns:
            right = right.drop(columns="user_id")
    return frame.merge(right, on=user_column, how="left", validate="many_to_one")
