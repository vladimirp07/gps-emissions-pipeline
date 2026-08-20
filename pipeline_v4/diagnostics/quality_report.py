"""Extended quality reporting for completed pipeline_v4 runs.

This module is deliberately downstream of production.  It reads persisted
summary/detailed/ledger artifacts and never invokes preprocessing, routing,
modal classification, or emissions calculation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


PIPELINE_VALID_STATUSES = ("complete", "partial")
POLLUTANTS = {
    "CO": ("co_total_g", "co_g_km"),
    "CO2": ("co2_total_g", "co2_g_km"),
    "CO2e": ("co2e_total_g", "co2e_g_km"),
    "HC": ("hc_total_g", "hc_g_km"),
    "NOx": ("nox_total_g", "nox_g_km"),
    "PM10": ("pm10_total_g", "pm10_g_km"),
    "PM2.5": ("pm25_total_g", "pm25_g_km"),
}
MODE_ORDER = ("Walking", "Metro", "Bus", "Car")
MODE_COLORS = {
    "Walking": "#4C8064",
    "Metro": "#686C9F",
    "Bus": "#B77B35",
    "Car": "#4F6F8F",
}
_MODE_ALIASES = {
    "caminar": "Walking", "walking": "Walking", "walk": "Walking",
    "metro": "Metro", "subway": "Metro",
    "bus": "Bus", "autobus": "Bus", "autobús": "Bus",
    "carro": "Car", "car": "Car", "auto": "Car", "automovil": "Car",
    "automóvil": "Car",
}
_TRIP_DATE_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_-?\d+$")


@dataclass(frozen=True)
class QualityReportConfig:
    """Explicit analytical configuration; it does not alter pipeline outputs."""

    include_complete: bool = True
    include_partial: bool = True
    exclude_failed: bool = True
    route_ratio_outer_fence_iqr: float = 3.0
    min_trips_for_tail_filter: int = 20
    min_segment_displacement_m: float = 5.0
    visual_upper_quantile: float = 0.99
    distance_rtol: float = 1e-9
    distance_atol_m: float = 1e-6
    emissions_rtol: float = 1e-9
    emissions_atol_g: float = 1e-9
    figure_size: tuple[float, float] = (11.0, 6.5)
    dpi: int = 220

    @property
    def valid_statuses(self) -> tuple[str, ...]:
        statuses: list[str] = []
        if self.include_complete:
            statuses.append("complete")
        if self.include_partial:
            statuses.append("partial")
        return tuple(statuses)


@dataclass(frozen=True)
class RunOutputs:
    run_path: Path
    artifact_dir: Path
    manifest: dict[str, Any]
    ledger: pd.DataFrame
    summary: pd.DataFrame | None
    detailed: pd.DataFrame | None
    paths: dict[str, Path | None]


@dataclass
class QualityReportResult:
    run_path: Path
    output_dir: Path
    markdown_path: Path
    metrics_path: Path
    verdict: str
    verdict_reasons: list[str]
    metrics: dict[str, Any]
    tables: dict[str, pd.DataFrame] = field(repr=False)
    figure_paths: list[Path] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def _resolve_artifacts(
    run_path: str | Path | None,
    *,
    summary_path: str | Path | None = None,
    detailed_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> tuple[Path, Path, dict[str, Path | None]]:
    explicit = {
        "summary": Path(summary_path).resolve() if summary_path else None,
        "detailed": Path(detailed_path).resolve() if detailed_path else None,
        "ledger": Path(ledger_path).resolve() if ledger_path else None,
        "manifest": Path(manifest_path).resolve() if manifest_path else None,
    }
    if run_path is None and not explicit["ledger"]:
        raise ValueError("run_path or ledger_path is required")

    supplied = Path(run_path).resolve() if run_path is not None else explicit["ledger"].parent
    if supplied.is_file():
        known = {
            "routes_emissions_summary.parquet": "summary",
            "routes_emissions_detailed.parquet": "detailed",
            "trip_ledger.parquet": "ledger",
            "manifest.json": "manifest",
            "pipeline_manifest.json": "manifest",
        }
        kind = known.get(supplied.name)
        if kind and explicit[kind] is None:
            explicit[kind] = supplied
        artifact_dir = supplied.parent
        run_root = artifact_dir.parent if artifact_dir.name == "pipeline" else artifact_dir
    elif (supplied / "pipeline").is_dir():
        run_root, artifact_dir = supplied, supplied / "pipeline"
    else:
        artifact_dir = supplied
        run_root = supplied.parent if supplied.name == "pipeline" else supplied

    defaults = {
        "summary": artifact_dir / "routes_emissions_summary.parquet",
        "detailed": artifact_dir / "routes_emissions_detailed.parquet",
        "ledger": artifact_dir / "trip_ledger.parquet",
    }
    for kind, candidate in defaults.items():
        if explicit[kind] is None and candidate.exists():
            explicit[kind] = candidate
    if explicit["manifest"] is None:
        candidates = (run_root / "manifest.json", artifact_dir / "pipeline_manifest.json")
        explicit["manifest"] = next((path for path in candidates if path.exists()), None)
    if explicit["ledger"] is None or not explicit["ledger"].exists():
        raise FileNotFoundError("trip_ledger.parquet was not found for the requested run")
    for kind in ("summary", "detailed"):
        if explicit[kind] is not None and not explicit[kind].exists():
            raise FileNotFoundError(explicit[kind])
    return run_root, artifact_dir, explicit


def load_run_outputs(
    run_path: str | Path | None,
    *,
    summary_path: str | Path | None = None,
    detailed_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> RunOutputs:
    """Load each persisted artifact at most once; raw GPS is never required."""
    root, artifact_dir, paths = _resolve_artifacts(
        run_path,
        summary_path=summary_path,
        detailed_path=detailed_path,
        ledger_path=ledger_path,
        manifest_path=manifest_path,
    )
    ledger = pd.read_parquet(paths["ledger"])
    required = {"physical_trip_id", "route_completeness_status", "processing_status"}
    missing = sorted(required - set(ledger.columns))
    if missing:
        raise ValueError(f"Trip ledger is missing required columns: {missing}")
    if ledger["physical_trip_id"].dropna().duplicated().any():
        raise ValueError("Trip ledger contains duplicate physical_trip_id values")
    summary = pd.read_parquet(paths["summary"]) if paths["summary"] else None
    detailed = pd.read_parquet(paths["detailed"]) if paths["detailed"] else None
    manifest: dict[str, Any] = {}
    if paths["manifest"]:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    return RunOutputs(root, artifact_dir, manifest, ledger, summary, detailed, paths)


def _canonical_mode(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    return _MODE_ALIASES.get(text.casefold(), text)


def _mode_order(values: Iterable[str]) -> list[str]:
    present = list(dict.fromkeys(str(value) for value in values if pd.notna(value)))
    return [mode for mode in MODE_ORDER if mode in present] + sorted(
        mode for mode in present if mode not in MODE_ORDER and mode != "Unknown"
    )


def _numeric(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _quantiles(series: pd.Series, quantiles=(0.5, 0.9, 0.95, 0.99)) -> dict[str, float | None]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {
        f"p{int(q * 100):02d}": (float(clean.quantile(q)) if len(clean) else None)
        for q in quantiles
    }


def _outer_fences(series: pd.Series, multiplier: float) -> tuple[float, float] | None:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 4:
        return None
    q1, q3 = clean.quantile([0.25, 0.75])
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= 0:
        return None
    return float(q1 - multiplier * iqr), float(q3 + multiplier * iqr)


def build_quality_population(
    ledger: pd.DataFrame, config: QualityReportConfig | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build contract-valid and transparent plot-quality trip populations."""
    cfg = config or QualityReportConfig()
    if not cfg.exclude_failed:
        raise ValueError("Failed routes must remain excluded from quality-report figures")
    trips = ledger.copy()
    trips["transport_mode"] = trips.get("final_mode", pd.Series(None, index=trips.index)).map(_canonical_mode)
    trips["distance_m"] = _numeric(trips, "reconstructed_distance_m")
    trips["distance_km"] = trips["distance_m"] / 1000.0
    valid_mask = trips["route_completeness_status"].isin(cfg.valid_statuses)
    valid = trips.loc[valid_mask].copy()
    valid["plot_quality"] = True
    valid["plot_exclusion_reason"] = "included"

    structural_bad = (
        ~np.isfinite(valid["distance_m"])
        | valid["distance_m"].le(0)
        | ~np.isfinite(_numeric(valid, "route_gps_ratio"))
        | _numeric(valid, "route_gps_ratio").le(0)
        | ~np.isfinite(_numeric(valid, "uncovered_fraction"))
        | _numeric(valid, "uncovered_fraction").lt(0)
        | _numeric(valid, "uncovered_fraction").gt(1)
        | ~np.isfinite(_numeric(valid, "max_continuity_gap_m"))
        | _numeric(valid, "max_continuity_gap_m").lt(0)
    )
    valid.loc[structural_bad, ["plot_quality", "plot_exclusion_reason"]] = [
        False, "malformed_or_missing_quality_metric"
    ]

    ratio_fences = None
    eligible_for_tail = valid[valid["plot_quality"]]
    if len(eligible_for_tail) >= cfg.min_trips_for_tail_filter:
        ratio_fences = _outer_fences(
            eligible_for_tail["route_gps_ratio"], cfg.route_ratio_outer_fence_iqr
        )
        if ratio_fences:
            low, high = ratio_fences
            extreme = valid["plot_quality"] & ~_numeric(valid, "route_gps_ratio").between(low, high)
            valid.loc[extreme, ["plot_quality", "plot_exclusion_reason"]] = [
                False, "extreme_route_gps_ratio_outer_fence"
            ]

    plot_quality = valid.loc[valid["plot_quality"]].copy()
    filter_summary = (
        valid["plot_exclusion_reason"].value_counts(dropna=False)
        .rename_axis("population_or_reason").reset_index(name="trips")
    )
    filter_summary = pd.concat([
        pd.DataFrame([
            {"population_or_reason": "all_physical_trips", "trips": len(trips)},
            {"population_or_reason": "pipeline_valid", "trips": len(valid)},
            {"population_or_reason": "plot_quality", "trips": len(plot_quality)},
        ]),
        filter_summary,
    ], ignore_index=True)
    metadata = {
        "pipeline_valid_statuses": list(cfg.valid_statuses),
        "pipeline_valid_trips": int(len(valid)),
        "plot_quality_trips": int(len(plot_quality)),
        "retention_percent": float(100.0 * len(plot_quality) / len(valid)) if len(valid) else 0.0,
        "route_gps_ratio_filter": {
            "method": "Tukey outer fence (Q1/Q3 ± 3×IQR); disabled for small samples",
            "iqr_multiplier": cfg.route_ratio_outer_fence_iqr,
            "minimum_sample": cfg.min_trips_for_tail_filter,
            "lower": ratio_fences[0] if ratio_fences else None,
            "upper": ratio_fences[1] if ratio_fences else None,
            "applied": bool(ratio_fences),
        },
    }
    return trips, valid, plot_quality, filter_summary, metadata


def compute_routing_metrics(
    trips: pd.DataFrame, valid: pd.DataFrame, plot_quality: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    status = trips["route_completeness_status"]
    processing = trips["processing_status"]
    ratio = _numeric(valid, "route_gps_ratio")
    uncovered = _numeric(valid, "uncovered_fraction")
    gap = _numeric(valid, "max_continuity_gap_m")
    metrics = {
        "total_trips": int(len(trips)),
        "complete": int(status.eq("complete").sum()),
        "partial": int(status.eq("partial").sum()),
        "failed": int(status.eq("failed").sum()),
        "quality_rejected": int(processing.eq("quality_rejected").sum()),
        "routing_failed": int(processing.eq("routing_failed").sum()),
        "route_gps_ratio": _quantiles(ratio),
        "uncovered_fraction": _quantiles(uncovered),
        "continuity_gap_m": _quantiles(gap),
        "pipeline_valid": int(len(valid)),
        "plot_quality": int(len(plot_quality)),
    }
    row: dict[str, Any] = {
        "total_trips": metrics["total_trips"],
        "complete": metrics["complete"],
        "partial": metrics["partial"],
        "failed": metrics["failed"],
        "quality_rejected": metrics["quality_rejected"],
        "routing_failed": metrics["routing_failed"],
        "pipeline_valid": metrics["pipeline_valid"],
        "plot_quality": metrics["plot_quality"],
    }
    for prefix, values in (
        ("route_gps_ratio", metrics["route_gps_ratio"]),
        ("uncovered_fraction", metrics["uncovered_fraction"]),
        ("continuity_gap_m", metrics["continuity_gap_m"]),
    ):
        row.update({f"{prefix}_{key}": value for key, value in values.items()})
    return metrics, pd.DataFrame([row])


def compute_modal_metrics(valid: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    grouped = valid.groupby("transport_mode", dropna=False, sort=False)
    rows = []
    total_distance = valid["distance_km"].sum(min_count=1)
    for mode, group in grouped:
        distance = group["distance_km"].sum(min_count=1)
        rows.append({
            "mode": mode,
            "trips": int(group["physical_trip_id"].nunique()),
            "trip_share_percent": float(100 * group["physical_trip_id"].nunique() / len(valid)) if len(valid) else 0.0,
            "distance_km": float(distance) if pd.notna(distance) else None,
            "routed_activity_share_percent": (
                float(100 * distance / total_distance) if pd.notna(total_distance) and total_distance > 0 else None
            ),
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        order = {mode: idx for idx, mode in enumerate(MODE_ORDER)}
        table["_order"] = table["mode"].map(order).fillna(len(order))
        table = table.sort_values(["_order", "mode"]).drop(columns="_order").reset_index(drop=True)
    metrics = {
        str(row["mode"]): int(row["trips"])
        for row in table.to_dict("records")
    }
    return metrics, table


def compute_modal_funnel_metrics(trips: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Calculate the modal funnel and failure breakdown across all physical trips."""
    total = len(trips)
    pre_status = trips.get("pre_routing_quality_status", pd.Series(dtype=object))
    quality_passed = int(pre_status.eq("passed").sum())
    quality_rejected = int(pre_status.eq("rejected").sum())

    def contains_mode(series: pd.Series, mode_name: str) -> int:
        return int(series.fillna("").astype(str).str.split(";").apply(lambda items: mode_name in items).sum())

    att_series = trips.get("hypotheses_attempted", pd.Series(dtype=object))
    succ_series = trips.get("hypotheses_successful", pd.Series(dtype=object))

    walk_att = contains_mode(att_series, "Caminar")
    walk_succ = contains_mode(succ_series, "Caminar")
    metro_att = contains_mode(att_series, "Metro")
    metro_succ = contains_mode(succ_series, "Metro")
    car_att = contains_mode(att_series, "Carro")
    car_succ = contains_mode(succ_series, "Carro")

    final_modes_raw = trips.get("final_mode", pd.Series(dtype=object)).value_counts(dropna=False).to_dict()
    final_modes = {str(k): int(v) for k, v in final_modes_raw.items() if pd.notna(k)}

    funnel_rows = [
        {"funnel_stage": "1. Physical trips detected", "trip_count": total, "share_percent": 100.0},
        {"funnel_stage": "2. Pre-routing quality passed", "trip_count": quality_passed, "share_percent": float(100.0 * quality_passed / total) if total else 0.0},
        {"funnel_stage": "2b. Pre-routing quality rejected", "trip_count": quality_rejected, "share_percent": float(100.0 * quality_rejected / total) if total else 0.0},
        {"funnel_stage": "3a. Walking hypothesis attempted", "trip_count": walk_att, "share_percent": float(100.0 * walk_att / total) if total else 0.0},
        {"funnel_stage": "3b. Walking hypothesis routed", "trip_count": walk_succ, "share_percent": float(100.0 * walk_succ / total) if total else 0.0},
        {"funnel_stage": "4a. Metro hypothesis attempted", "trip_count": metro_att, "share_percent": float(100.0 * metro_att / total) if total else 0.0},
        {"funnel_stage": "4b. Metro hypothesis routed", "trip_count": metro_succ, "share_percent": float(100.0 * metro_succ / total) if total else 0.0},
        {"funnel_stage": "5a. Road hypothesis attempted", "trip_count": car_att, "share_percent": float(100.0 * car_att / total) if total else 0.0},
        {"funnel_stage": "5b. Road hypothesis routed", "trip_count": car_succ, "share_percent": float(100.0 * car_succ / total) if total else 0.0},
        {"funnel_stage": "6a. Final Mode: Walking", "trip_count": int(final_modes.get("Caminar", 0)), "share_percent": float(100.0 * final_modes.get("Caminar", 0) / total) if total else 0.0},
        {"funnel_stage": "6b. Final Mode: Metro", "trip_count": int(final_modes.get("Metro", 0)), "share_percent": float(100.0 * final_modes.get("Metro", 0) / total) if total else 0.0},
        {"funnel_stage": "6c. Final Mode: Bus", "trip_count": int(final_modes.get("Bus", 0)), "share_percent": float(100.0 * final_modes.get("Bus", 0) / total) if total else 0.0},
        {"funnel_stage": "6d. Final Mode: Car", "trip_count": int(final_modes.get("Carro", 0)), "share_percent": float(100.0 * final_modes.get("Carro", 0) / total) if total else 0.0},
    ]
    funnel_table = pd.DataFrame(funnel_rows)

    reasons = trips.get("failure_reason", pd.Series(dtype=object)).dropna().value_counts().reset_index()
    if not reasons.empty:
        reasons.columns = ["failure_reason", "trip_count"]
        reasons["share_percent"] = (100.0 * reasons["trip_count"] / total).round(2)
    else:
        reasons = pd.DataFrame(columns=["failure_reason", "trip_count", "share_percent"])

    funnel_metrics = {
        "physical_trips": total,
        "pre_routing_quality_passed": quality_passed,
        "pre_routing_quality_rejected": quality_rejected,
        "hypotheses_attempted": {"Walking": walk_att, "Metro": metro_att, "Road": car_att},
        "hypotheses_successful": {"Walking": walk_succ, "Metro": metro_succ, "Road": car_succ},
        "final_modes": final_modes,
        "failure_reasons": reasons.set_index("failure_reason")["trip_count"].to_dict() if not reasons.empty else {},
    }
    return funnel_metrics, funnel_table, reasons


def _extract_trip_date(identifier: Any) -> str | None:
    match = _TRIP_DATE_RE.search(str(identifier))
    return match.group(1) if match else None


def _attach_physical_trip_ids(frame: pd.DataFrame, ledger: pd.DataFrame) -> pd.DataFrame:
    if "physical_trip_id" in frame and frame["physical_trip_id"].notna().any():
        return frame.copy()
    result = frame.copy()
    required = {"user_id", "trip_id", "local_timestamp"}
    if not required.issubset(result.columns) or not {"user_id", "trip_id", "physical_trip_id"}.issubset(ledger.columns):
        result["physical_trip_id"] = pd.NA
        return result
    lookup = ledger[["user_id", "trip_id", "physical_trip_id"]].copy()
    lookup["_date"] = lookup["physical_trip_id"].map(_extract_trip_date)
    lookup["_user"] = lookup["user_id"].astype(str)
    lookup["_trip"] = lookup["trip_id"].astype(str)
    lookup = lookup.dropna(subset=["_date"]).drop_duplicates(["_user", "_trip", "_date"], keep=False)
    result["_user"] = result["user_id"].astype(str)
    result["_trip"] = result["trip_id"].astype(str)
    result["_date"] = pd.to_datetime(result["local_timestamp"], errors="coerce").dt.strftime("%Y-%m-%d")
    result = result.merge(
        lookup[["_user", "_trip", "_date", "physical_trip_id"]],
        on=["_user", "_trip", "_date"], how="left", validate="many_to_one",
    )
    return result.drop(columns=["_user", "_trip", "_date"])


def _emission_rows(outputs: RunOutputs) -> pd.DataFrame:
    source = outputs.detailed if outputs.detailed is not None else outputs.summary
    if source is None:
        return pd.DataFrame()
    frame = _attach_physical_trip_ids(source, outputs.ledger)
    if "transport_mode" in frame:
        frame["transport_mode"] = frame["transport_mode"].map(_canonical_mode)
    else:
        frame["transport_mode"] = "Unknown"
    emission_columns = [column for pair in POLLUTANTS.values() for column in pair]
    available = [column for column in emission_columns if column in frame]
    if available:
        has_emission_data = frame[available].notna().any(axis=1)
        frame = frame.loc[has_emission_data].copy()
    return frame


def compute_emissions_metrics(
    outputs: RunOutputs, valid: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    ledger = outputs.ledger
    valid_ids = set(valid["physical_trip_id"].dropna().astype(str))
    eligible = ledger.get("emissions_eligible", pd.Series(False, index=ledger.index)).fillna(False).astype(bool)
    success = ledger.get("emissions_success", pd.Series(False, index=ledger.index)).fillna(False).astype(bool)
    valid_mask = ledger["physical_trip_id"].astype(str).isin(valid_ids)
    eligible_ids = set(ledger.loc[valid_mask & eligible, "physical_trip_id"].astype(str))
    successful_ids = set(ledger.loc[valid_mask & success, "physical_trip_id"].astype(str))
    emissions = _emission_rows(outputs)
    if not emissions.empty:
        emissions = emissions[emissions["physical_trip_id"].astype(str).isin(eligible_ids)].copy()

    distance_by_id = valid.set_index(valid["physical_trip_id"].astype(str))["distance_km"]
    mode_by_id = valid.set_index(valid["physical_trip_id"].astype(str))["transport_mode"]
    pollutant_rows: list[dict[str, Any]] = []
    mode_pollutants: dict[str, dict[str, float]] = {}
    rate_checks: list[dict[str, Any]] = []
    available_pollutants: list[str] = []
    for pollutant, (total_column, rate_column) in POLLUTANTS.items():
        if total_column not in emissions:
            continue
        numeric_totals = pd.to_numeric(emissions[total_column], errors="coerce")
        if not numeric_totals.notna().any():
            continue
        available_pollutants.append(pollutant)
        totals = numeric_totals
        by_trip = totals.groupby(emissions["physical_trip_id"].astype(str)).sum(min_count=1)
        by_trip = by_trip.reindex(sorted(successful_ids), fill_value=0.0)
        total_mass = float(by_trip.sum(min_count=1)) if len(by_trip) else 0.0
        routed_distance = float(distance_by_id.reindex(by_trip.index).sum(min_count=1)) if len(by_trip) else 0.0
        quant = _quantiles(by_trip, (0.5, 0.9, 0.95))
        pollutant_rows.append({
            "pollutant": pollutant,
            "unit": "g",
            "trips": int(len(by_trip)),
            "total_mass_g": total_mass,
            "mean_g_per_trip": float(by_trip.mean()) if len(by_trip) else None,
            "median_g_per_trip": quant["p50"],
            "p90_g_per_trip": quant["p90"],
            "p95_g_per_trip": quant["p95"],
            "total_reconstructed_distance_km": routed_distance,
            "mass_g_per_routed_km": total_mass / routed_distance if routed_distance > 0 else None,
        })
        mode_frame = pd.DataFrame({"mass": by_trip, "mode": mode_by_id.reindex(by_trip.index)})
        for mode, group in mode_frame.groupby("mode", dropna=False):
            mode_pollutants.setdefault(str(mode), {})[pollutant] = float(group["mass"].sum())
        if rate_column in emissions:
            raw = emissions[rate_column]
            numeric = pd.to_numeric(raw, errors="coerce")
            for mode, index in emissions.groupby("transport_mode", dropna=False).groups.items():
                values = numeric.loc[index]
                original = raw.loc[index]
                finite = np.isfinite(values)
                rate_checks.append({
                    "pollutant": pollutant,
                    "mode": str(mode),
                    "rows": int(len(values)),
                    "non_numeric": int((original.notna() & values.isna()).sum()),
                    "nan": int(values.isna().sum()),
                    "inf": int((values.notna() & ~finite).sum()),
                    "negative": int((values < 0).sum()),
                    "zero": int((values == 0).sum()),
                    "zero_percent": float(100 * (values == 0).sum() / len(values)) if len(values) else 0.0,
                })

    mode_rows: list[dict[str, Any]] = []
    successful_distance = float(distance_by_id.reindex(sorted(successful_ids)).sum(min_count=1)) if successful_ids else 0.0
    for mode in _mode_order(valid["transport_mode"]):
        ids = set(valid.loc[valid["transport_mode"].eq(mode), "physical_trip_id"].astype(str)) & successful_ids
        mode_emissions = emissions[emissions["physical_trip_id"].astype(str).isin(ids)] if not emissions.empty else emissions
        row: dict[str, Any] = {
            "mode": mode,
            "trips": len(ids),
            "distance_km": float(distance_by_id.reindex(sorted(ids)).sum(min_count=1)) if ids else 0.0,
            "emission_rows": int(len(mode_emissions)),
        }
        row["routed_activity_share_percent"] = (
            100.0 * row["distance_km"] / successful_distance if successful_distance > 0 else None
        )
        row.update({
            pollutant: (
                mode_pollutants.get(mode, {}).get(pollutant, 0.0)
                if pollutant in available_pollutants else None
            )
            for pollutant in POLLUTANTS
        })
        mode_rows.append(row)

    lookup_rows: list[dict[str, Any]] = []
    if not emissions.empty and "emission_lookup_status" in emissions:
        lookup_rows = (
            emissions["emission_lookup_status"].fillna("missing").value_counts()
            .rename_axis("emission_lookup_status").reset_index(name="rows").to_dict("records")
        )
    lookup_table = pd.DataFrame(lookup_rows, columns=["emission_lookup_status", "rows"])
    source_table = pd.DataFrame(columns=["moves_source_type", "rows"])
    if not emissions.empty and "moves_source_type" in emissions:
        source_table = (
            emissions["moves_source_type"].fillna("missing").value_counts()
            .rename_axis("moves_source_type").reset_index(name="rows")
        )
    speed_bin_table = pd.DataFrame(columns=["moves_speed_bin", "rows"])
    if not emissions.empty and "moves_speed_bin" in emissions:
        speed_bin_table = (
            emissions["moves_speed_bin"].fillna("missing").value_counts()
            .rename_axis("moves_speed_bin").reset_index(name="rows")
        )
    road_type_table = pd.DataFrame(columns=["moves_road_type", "rows"])
    if not emissions.empty and "moves_road_type" in emissions:
        road_type_table = (
            emissions["moves_road_type"].fillna("missing").value_counts()
            .rename_axis("moves_road_type").reset_index(name="rows")
        )
    road_lookup_table = pd.DataFrame(columns=["road_lookup_status", "rows"])
    if not emissions.empty and "road_lookup_status" in emissions:
        road_lookup_table = (
            emissions["road_lookup_status"].fillna("missing").value_counts()
            .rename_axis("road_lookup_status").reset_index(name="rows")
        )
    temporal_table = pd.DataFrame(columns=["moves_month", "moves_day_type", "moves_hour", "rows"])
    temporal_columns = [
        column for column in ("moves_month", "moves_day_type", "moves_hour")
        if column in emissions
    ]
    if temporal_columns:
        temporal_table = (
            emissions.groupby(temporal_columns, dropna=False).size()
            .reset_index(name="rows")
        )

    matched_statuses = {"exact", "imputed_speed_curve", "imputed_source"}
    missing_statuses = {"missing", "missing_zero_fallback", "pending_imputation"}
    lookup_status = emissions.get("emission_lookup_status", pd.Series(dtype=object)).fillna("missing")
    metrics = {
        "valid_routed_trips": int(len(valid)),
        "eligible_trips": int(len(eligible_ids)),
        "successful_trips": int(len(successful_ids)),
        "trips_without_emissions": int(len(valid_ids - successful_ids)),
        "success_rate_percent": float(100 * len(successful_ids) / len(eligible_ids)) if eligible_ids else None,
        "emission_rows": int(len(emissions)),
        "available_pollutants": available_pollutants,
        "lookup": {
            "available": "emission_lookup_status" in emissions,
            "matched_or_imputed_rows": int(lookup_status.isin(matched_statuses).sum()),
            "missing_factor_rows": int(lookup_status.isin(missing_statuses).sum()),
            "non_motorized_not_applicable_rows": int(lookup_status.eq("not_applicable_non_motorized").sum()),
            "unmatched_dimension_detail": "not persisted by the production schema",
        },
    }
    tables = {
        "emissions_by_pollutant": pd.DataFrame(pollutant_rows),
        "emissions_by_mode": pd.DataFrame(mode_rows),
        "emission_rate_checks": pd.DataFrame(rate_checks),
        "emission_lookup_coverage": lookup_table,
        "moves_source_distribution": source_table,
        "moves_speed_bin_distribution": speed_bin_table,
        "moves_road_type_distribution": road_type_table,
        "road_lookup_coverage": road_lookup_table,
        "moves_temporal_distribution": temporal_table,
    }
    return metrics, tables


def compute_segment_sinuosity(
    detailed: pd.DataFrame | None,
    plot_trip_ids: set[str],
    config: QualityReportConfig | None = None,
) -> pd.DataFrame:
    """Segment network-distance/displacement metric.

    The current detailed schema stores routed observations and per-row network
    distance. Consecutive rows are paired only inside the same physical trip
    and route component. In the current output, coordinates are reconstructed
    route observations rather than raw GPS pings. Displacements below 5 m are
    omitted, matching the stable-denominator rule; extreme-value truncation is
    intentionally not applied here.
    """
    cfg = config or QualityReportConfig()
    columns = ["physical_trip_id", "transport_mode", "segment_sinuosity", "gps_displacement_m", "network_distance_m"]
    if detailed is None:
        return pd.DataFrame(columns=columns)
    required = {"physical_trip_id", "latitude", "longitude", "distance_m"}
    if not required.issubset(detailed.columns):
        return pd.DataFrame(columns=columns)
    frame = detailed[detailed["physical_trip_id"].astype(str).isin(plot_trip_ids)].copy()
    if "routing_failed" in frame:
        frame = frame[~frame["routing_failed"].fillna(True).astype(bool)]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    mode = frame.get("transport_mode", pd.Series("Unknown", index=frame.index))
    frame["transport_mode"] = mode.map(_canonical_mode)
    sort_columns = [column for column in ("physical_trip_id", "route_component_id", "local_timestamp") if column in frame]
    frame = frame.sort_values(sort_columns, kind="stable")
    group_columns = ["physical_trip_id"]
    if "route_component_id" in frame:
        group_columns.append("route_component_id")
    grouped = frame.groupby(group_columns, dropna=False, sort=False)
    next_lat = grouped["latitude"].shift(-1)
    next_lon = grouped["longitude"].shift(-1)
    lat1 = np.radians(_numeric(frame, "latitude"))
    lat2 = np.radians(pd.to_numeric(next_lat, errors="coerce"))
    delta_lat = lat2 - lat1
    delta_lon = np.radians(pd.to_numeric(next_lon, errors="coerce") - _numeric(frame, "longitude"))
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    displacement = 6_371_000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = _numeric(frame, "distance_m")
    stable = np.isfinite(displacement) & np.isfinite(distance) & displacement.ge(cfg.min_segment_displacement_m) & distance.ge(0)
    result = frame.loc[stable, ["physical_trip_id", "transport_mode"]].copy()
    result["gps_displacement_m"] = displacement.loc[stable]
    result["network_distance_m"] = distance.loc[stable]
    result["segment_sinuosity"] = result["network_distance_m"] / result["gps_displacement_m"]
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=["segment_sinuosity"]).reset_index(drop=True)


def compute_routed_speeds(detailed: pd.DataFrame | None, plot_trip_ids: set[str]) -> pd.DataFrame:
    columns = ["physical_trip_id", "transport_mode", "speed_kmh"]
    if detailed is None or not {"physical_trip_id", "speed_kmh"}.issubset(detailed.columns):
        return pd.DataFrame(columns=columns)
    frame = detailed[detailed["physical_trip_id"].astype(str).isin(plot_trip_ids)].copy()
    if "routing_failed" in frame:
        frame = frame[~frame["routing_failed"].fillna(True).astype(bool)]
    frame["speed_kmh"] = _numeric(frame, "speed_kmh")
    mode = frame.get("transport_mode", pd.Series("Unknown", index=frame.index))
    frame["transport_mode"] = mode.map(_canonical_mode)
    valid = np.isfinite(frame["speed_kmh"]) & frame["speed_kmh"].ge(0)
    return frame.loc[valid, columns].reset_index(drop=True)


def _distance_consistency(
    detailed: pd.DataFrame | None, valid: pd.DataFrame, cfg: QualityReportConfig
) -> pd.DataFrame:
    columns = ["physical_trip_id", "ledger_distance_m", "segment_distance_m", "absolute_difference_m", "consistent"]
    if detailed is None or not {"physical_trip_id", "distance_m"}.issubset(detailed.columns):
        return pd.DataFrame(columns=columns)
    frame = detailed[detailed["physical_trip_id"].notna()].copy()
    if "routing_failed" in frame:
        frame = frame[~frame["routing_failed"].fillna(True).astype(bool)]
    summed = _numeric(frame, "distance_m").groupby(frame["physical_trip_id"].astype(str)).sum(min_count=1)
    ledger_distance = valid.set_index(valid["physical_trip_id"].astype(str))["distance_m"]
    comparison = pd.concat([ledger_distance.rename("ledger_distance_m"), summed.rename("segment_distance_m")], axis=1, join="inner").reset_index(names="physical_trip_id")
    comparison["absolute_difference_m"] = (comparison["ledger_distance_m"] - comparison["segment_distance_m"]).abs()
    comparison["consistent"] = np.isclose(
        comparison["ledger_distance_m"], comparison["segment_distance_m"],
        rtol=cfg.distance_rtol, atol=cfg.distance_atol_m,
    )
    return comparison[columns]


def _summary_detailed_emission_consistency(outputs: RunOutputs, cfg: QualityReportConfig) -> pd.DataFrame:
    rows = []
    if outputs.summary is None or outputs.detailed is None:
        return pd.DataFrame(columns=["pollutant", "summary_total_g", "detailed_total_g", "absolute_difference_g", "consistent"])
    summary = outputs.summary
    detailed = outputs.detailed
    for pollutant, (column, _) in POLLUTANTS.items():
        if column not in summary or column not in detailed:
            continue
        summary_values = pd.to_numeric(summary[column], errors="coerce")
        detailed_values = pd.to_numeric(detailed[column], errors="coerce")
        if not summary_values.notna().any() and not detailed_values.notna().any():
            continue
        left = float(summary_values.sum(min_count=1))
        right = float(detailed_values.sum(min_count=1))
        rows.append({
            "pollutant": pollutant,
            "summary_total_g": left,
            "detailed_total_g": right,
            "absolute_difference_g": abs(left - right),
            "consistent": bool(np.isclose(left, right, rtol=cfg.emissions_rtol, atol=cfg.emissions_atol_g)),
        })
    return pd.DataFrame(rows)


def _quality_warnings(
    trips: pd.DataFrame,
    valid: pd.DataFrame,
    plot_quality: pd.DataFrame,
    emissions_metrics: Mapping[str, Any],
    emissions_tables: Mapping[str, pd.DataFrame],
    distance_consistency: pd.DataFrame,
    emission_consistency: pd.DataFrame,
    cfg: QualityReportConfig,
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []

    def add(level: str, code: str, message: str) -> None:
        warnings.append({"level": level, "code": code, "message": message})

    if len(valid) == 0:
        add("ERROR", "NO_VALID_ROUTES", "No complete or partial routes are available.")
    if len(valid) < cfg.min_trips_for_tail_filter:
        add("INFO", "SMALL_SAMPLE", "The valid-route sample is too small for distribution-relative tail filtering.")
    failed = int(trips["route_completeness_status"].eq("failed").sum())
    if failed > len(valid):
        add("WARNING", "FAILED_MAJORITY", "Failed routes outnumber complete and partial routes.")
    partial = int(trips["route_completeness_status"].eq("partial").sum())
    complete = int(trips["route_completeness_status"].eq("complete").sum())
    if partial > complete and partial > 0:
        add("WARNING", "PARTIAL_MAJORITY", "Partial routes outnumber complete routes.")
    exclusion_reasons = valid.loc[~valid["plot_quality"], "plot_exclusion_reason"].value_counts()
    malformed_excluded = int(exclusion_reasons.get("malformed_or_missing_quality_metric", 0))
    extreme_excluded = int(exclusion_reasons.get("extreme_route_gps_ratio_outer_fence", 0))
    if malformed_excluded:
        add("ERROR", "MALFORMED_ROUTE_QUALITY", f"{malformed_excluded} pipeline-valid route(s) have malformed or missing quality metrics.")
    if extreme_excluded:
        add("WARNING", "EXTREME_ROUTE_GPS_TAIL", f"{extreme_excluded} pipeline-valid route(s) lie outside the dataset-relative route/GPS outer fence.")
    if not distance_consistency.empty and (~distance_consistency["consistent"]).any():
        add("ERROR", "DISTANCE_MISMATCH", "Ledger trip distance differs from summed valid detailed rows.")
    if not emission_consistency.empty and (~emission_consistency["consistent"]).any():
        add("ERROR", "EMISSION_OUTPUT_MISMATCH", "Summary and detailed emission totals disagree.")
    rate_checks = emissions_tables.get("emission_rate_checks", pd.DataFrame())
    if not rate_checks.empty:
        malformed = rate_checks[[column for column in ("non_numeric", "inf", "negative") if column in rate_checks]].sum().sum()
        if malformed:
            add("ERROR", "MALFORMED_EMISSION_RATE", "Non-numeric, infinite, or negative emission rates were detected.")
        if rate_checks.get("nan", pd.Series(dtype=float)).sum() > 0:
            add("WARNING", "MISSING_EMISSION_RATE", "NaN emission rates were detected in persisted emission rows.")
    if emissions_metrics["lookup"]["missing_factor_rows"]:
        add("WARNING", "EMISSION_LOOKUP_FAILURE", "One or more emission rows used a missing-factor fallback.")
    if emissions_metrics["eligible_trips"] and emissions_metrics["successful_trips"] < emissions_metrics["eligible_trips"]:
        add("WARNING", "EMISSIONS_INCOMPLETE", "Not every emission-eligible trip has successful emissions.")
    return warnings


def _verdict(warnings: list[dict[str, str]]) -> tuple[str, list[str]]:
    errors = [item["message"] for item in warnings if item["level"] == "ERROR"]
    cautions = [item["message"] for item in warnings if item["level"] == "WARNING"]
    if errors:
        return "REVIEW REQUIRED", errors
    if cautions:
        return "PASS WITH WARNINGS", cautions
    return "PASS", ["No operational error or warning condition was detected."]


def _prepare_output_dir(path: Path, overwrite: bool) -> tuple[Path, Path]:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Quality report directory is not empty: {path}. Pass overwrite=True explicitly to replace managed files."
        )
    figures = path / "figures"
    tables = path / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    return figures, tables


def _write_tables(tables: Mapping[str, pd.DataFrame], directory: Path) -> None:
    for name, table in tables.items():
        if table is None:
            continue
        table.to_csv(directory / f"{name}.csv", index=False)
        table.to_parquet(directory / f"{name}.parquet", index=False)


def _plot_context():
    import matplotlib as mpl
    return mpl.rc_context({
        "font.family": "Times New Roman",
        "font.size": 15,
        "axes.titlesize": 20,
        "axes.titleweight": "semibold",
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#4A4A4A",
        "axes.grid": True,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.65,
    })


def _finish_figure(
    fig, path: Path, cfg: QualityReportConfig, show: bool, save: bool
) -> None:
    import matplotlib.pyplot as plt
    fig.tight_layout()
    if save:
        fig.savefig(path, dpi=cfg.dpi, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _draw_horizontal_distribution(ax, data: pd.DataFrame, value: str, modes: list[str], *, violin: bool, x_limit: float | None = None) -> None:
    arrays = [pd.to_numeric(data.loc[data["transport_mode"].eq(mode), value], errors="coerce").dropna().to_numpy() for mode in modes]
    positions = np.arange(1, len(modes) + 1)
    if violin:
        usable = [(pos, mode, values) for pos, mode, values in zip(positions, modes, arrays) if len(values) >= 2]
        if usable:
            parts = ax.violinplot([item[2] for item in usable], positions=[item[0] for item in usable], vert=False, showmedians=True, showextrema=False)
            for body, (_, mode, _) in zip(parts["bodies"], usable):
                body.set_facecolor(MODE_COLORS.get(mode, "#7A7A7A"))
                body.set_edgecolor("#4A4A4A")
                body.set_alpha(0.78)
            if "cmedians" in parts:
                parts["cmedians"].set_color("#303030")
    else:
        usable = [(pos, mode, values) for pos, mode, values in zip(positions, modes, arrays) if len(values)]
        if usable:
            boxes = ax.boxplot(
                [item[2] for item in usable], positions=[item[0] for item in usable], vert=False,
                patch_artist=True, showfliers=False, showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#303030", "markersize": 6},
                medianprops={"color": "#303030", "linewidth": 1.5},
                whiskerprops={"color": "#555555"}, capprops={"color": "#555555"},
            )
            for box, (_, mode, _) in zip(boxes["boxes"], usable):
                box.set_facecolor(MODE_COLORS.get(mode, "#7A7A7A"))
                box.set_alpha(0.78)
    ax.set_yticks(positions, modes)
    if x_limit is not None and np.isfinite(x_limit) and x_limit > 0:
        ax.set_xlim(left=0, right=x_limit)


def generate_figures(
    valid: pd.DataFrame,
    plot_quality: pd.DataFrame,
    sinuosity: pd.DataFrame,
    speeds: pd.DataFrame,
    figures_dir: Path,
    *,
    config: QualityReportConfig,
    show_figures: bool,
    save_figures: bool,
) -> tuple[list[Path], dict[str, Any]]:
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    visual_notes: dict[str, Any] = {}
    with _plot_context():
        modes = _mode_order(valid["transport_mode"])
        if modes:
            counts = valid["transport_mode"].value_counts().reindex(modes, fill_value=0)
            shares = counts / counts.sum() * 100
            fig, ax = plt.subplots(figsize=config.figure_size)
            colors = [MODE_COLORS.get(mode, "#7A7A7A") for mode in modes]
            bars = ax.barh(modes, shares, color=colors, edgecolor="white", linewidth=0.8)
            ax.invert_yaxis()
            ax.set_title("Modal Share of Valid Routed Trips")
            ax.set_xlabel("Share of pipeline-valid trips (%)")
            ax.set_ylabel("Transportation mode")
            for bar, mode in zip(bars, modes):
                ax.text(bar.get_width() + max(shares.max() * 0.012, 0.25), bar.get_y() + bar.get_height()/2, f"{bar.get_width():.1f}%  (N={counts[mode]:,})", va="center", fontsize=14)
            ax.set_xlim(0, max(100, shares.max() * 1.2))
            path = figures_dir / "01_modal_share_valid_routed_trips.png"
            _finish_figure(fig, path, config, show_figures, save_figures)
            if save_figures: paths.append(path)

        distance_modes = _mode_order(plot_quality["transport_mode"])
        if distance_modes and plot_quality["distance_km"].notna().any():
            fig, ax = plt.subplots(figsize=config.figure_size)
            _draw_horizontal_distribution(ax, plot_quality, "distance_km", distance_modes, violin=False)
            ax.set_title("Trip Distance by Transportation Mode")
            ax.set_xlabel("Reconstructed trip distance (km); median = line, mean = diamond")
            ax.set_ylabel("Transportation mode")
            path = figures_dir / "02_trip_distance_by_mode.png"
            _finish_figure(fig, path, config, show_figures, save_figures)
            if save_figures: paths.append(path)

        sin_modes = _mode_order(sinuosity["transport_mode"]) if not sinuosity.empty else []
        if sin_modes:
            upper = float(sinuosity["segment_sinuosity"].quantile(config.visual_upper_quantile))
            fig, ax = plt.subplots(figsize=config.figure_size)
            _draw_horizontal_distribution(ax, sinuosity, "segment_sinuosity", sin_modes, violin=False, x_limit=upper)
            ax.axvline(1.0, color="#8B4A45", linestyle="--", linewidth=1.2, label="Network distance = straight-line displacement")
            ax.set_title("Segment-Level Routing Sinuosity")
            ax.set_xlabel("Network distance / reconstructed-observation displacement")
            ax.set_ylabel("Transportation mode")
            ax.legend(frameon=False, loc="best")
            visual_notes["sinuosity_x_axis"] = f"Displayed through the p{config.visual_upper_quantile*100:g} value ({upper:.3g}); all finite stable-denominator rows remain in tables."
            path = figures_dir / "03_segment_level_routing_sinuosity.png"
            _finish_figure(fig, path, config, show_figures, save_figures)
            if save_figures: paths.append(path)

        speed_modes = _mode_order(speeds["transport_mode"]) if not speeds.empty else []
        if speed_modes:
            upper = float(speeds["speed_kmh"].quantile(config.visual_upper_quantile))
            visual_notes["speed_x_axis"] = f"Displayed through the p{config.visual_upper_quantile*100:g} value ({upper:.3g} km/h); no rows were deleted for visual clipping."
            fig, ax = plt.subplots(figsize=config.figure_size)
            _draw_horizontal_distribution(ax, speeds, "speed_kmh", speed_modes, violin=False, x_limit=upper)
            ax.set_title("Routed Speed Distribution by Transportation Mode")
            ax.set_xlabel("Routed speed (km/h); visual axis clipped at dataset p99")
            ax.set_ylabel("Transportation mode")
            path = figures_dir / "04_routed_speed_boxplots.png"
            _finish_figure(fig, path, config, show_figures, save_figures)
            if save_figures: paths.append(path)

            fig, ax = plt.subplots(figsize=config.figure_size)
            _draw_horizontal_distribution(ax, speeds, "speed_kmh", speed_modes, violin=True, x_limit=upper)
            ax.set_title("Routed Speed Density by Transportation Mode")
            ax.set_xlabel("Routed speed (km/h); visual axis clipped at dataset p99")
            ax.set_ylabel("Transportation mode")
            path = figures_dir / "05_routed_speed_violin.png"
            _finish_figure(fig, path, config, show_figures, save_figures)
            if save_figures: paths.append(path)
    return paths, visual_notes


def _markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    if frame is None or frame.empty:
        return "_Not available for this run._"
    view = frame.head(max_rows).copy()
    for column in view:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
        else:
            view[column] = view[column].fillna("").astype(str).str.replace("|", "\\|", regex=False)
    headers = [str(column) for column in view.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None))
    if len(frame) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(frame)} rows._")
    return "\n".join(lines)


def _write_markdown(
    path: Path,
    outputs: RunOutputs,
    metrics: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    figure_paths: list[Path],
    warnings: list[dict[str, str]],
    verdict: str,
    verdict_reasons: list[str],
) -> None:
    routing = metrics["routing"]
    filtering = metrics["quality_filter"]
    emissions = metrics["emissions"]
    visual = metrics.get("visual_clipping", {})
    warning_text = "\n".join(
        f"- **{item['level']} — {item['code']}**: {item['message']}" for item in warnings
    ) or "- No operational warnings."
    figures = "\n".join(f"- [{figure.name}](figures/{figure.name})" for figure in figure_paths) or "- No figures were generated."
    reasons = "\n".join(f"- {reason}" for reason in verdict_reasons)
    text = f"""# Extended Pipeline V4 Quality Report

## 1. Run Summary

- Run: `{outputs.run_path.name}`
- Users: {metrics['run']['users']:,}
- Physical trips: {routing['total_trips']:,}
- Source artifacts: ledger={outputs.paths['ledger'].name}; detailed={'available' if outputs.detailed is not None else 'not available'}; summary={'available' if outputs.summary is not None else 'not available'}.

## 2. Routing Outcomes

{_markdown_table(tables['routing_quality_summary'])}

Pipeline-valid routes are exactly the production statuses `complete` and `partial`. Failed, quality-rejected, and routing-failed trips are not used in valid-route figures.

## 3. Route Quality

Route/GPS ratio is reconstructed route distance divided by observed GPS distance at trip level. Uncovered fraction, continuity gaps, and component counts come directly from the production ledger; this report does not recalculate or change completeness.

{_markdown_table(tables['route_quality_distribution'])}

## 4. Plot Population / Filtering

- Pipeline-valid routes: {filtering['pipeline_valid_trips']:,}
- Plot-quality routes: {filtering['plot_quality_trips']:,}
- Retained: {filtering['retention_percent']:.2f}%
- Criterion: finite, structurally valid quality fields plus {filtering['route_gps_ratio_filter']['method']}.
- Computed route/GPS bounds: {filtering['route_gps_ratio_filter']['lower']} to {filtering['route_gps_ratio_filter']['upper']}.

{_markdown_table(tables['quality_filter_summary'])}

This analytical subset does not change `emissions_eligible`, completeness, routing, classification, emissions, or persisted production outputs.

## 5. Modal Distribution & Pipeline Funnel

### Active Transportation Modes (Pipeline-Valid Routes)

{_markdown_table(tables['mode_summary'])}

### Modal Funnel (All Physical Movements)

{_markdown_table(tables['modal_funnel_summary'])}

### Pipeline Failure & Rejection Reasons

{_markdown_table(tables['failure_reason_summary'])}

## 6. Trip Distance

Trip distance is read once per physical trip from ledger `reconstructed_distance_m`; segment emission rows are never double-counted.

{_markdown_table(tables['trip_distance_by_mode'])}

## 7. Segment-Level Sinuosity

Definition: current-row reconstructed network distance divided by Haversine displacement to the next persisted routed observation within the same physical trip and component. The formula and 5 m denominator guard use the stable route-segment definition; the current output coordinates are reconstructed route observations—not raw GPS pings—so this is route-segment geometry sinuosity rather than raw-GPS routing error. Extreme-value truncation is not applied.

Rows available: {len(tables['segment_sinuosity']):,}. {visual.get('sinuosity_x_axis', 'No visual clipping was required or detailed output was unavailable.')}

## 8. Routed Speed

Uses persisted routed `speed_kmh` from valid detailed route rows. Boxplot and violin use the same population. {visual.get('speed_x_axis', 'Detailed routed speeds were unavailable.')}

## 9. Emissions Diagnostic

- Valid routed trips: {emissions['valid_routed_trips']:,}
- Emission-eligible trips: {emissions['eligible_trips']:,}
- Emission-successful trips: {emissions['successful_trips']:,}
- Trips without successful emissions: {emissions['trips_without_emissions']:,}
- Success rate among eligible trips: {emissions['success_rate_percent'] if emissions['success_rate_percent'] is not None else 'not applicable'}

### Emissions by mode

{_markdown_table(tables['emissions_by_mode'])}

### Emissions by pollutant

{_markdown_table(tables['emissions_by_pollutant'])}

Walking and other Source 0 rows retain the existing production treatment (`not_applicable_non_motorized` and zero mass); no MOVES unit or factor is reinterpreted.

### Distance and output consistency

{_markdown_table(tables['distance_consistency_summary'])}

{_markdown_table(tables['emission_output_consistency'])}

There is no independently persisted trip-level emission total. When both summary and detailed outputs exist, their pollutant totals are compared exactly within numerical tolerance; segment totals are otherwise the authoritative persisted mass.

## 10. Emission Lookup Coverage

{_markdown_table(tables['emission_lookup_coverage'])}

### Source ID distribution

{_markdown_table(tables['moves_source_distribution'])}

### Speed-bin distribution

{_markdown_table(tables['moves_speed_bin_distribution'])}

### Road-type distribution

{_markdown_table(tables['moves_road_type_distribution'])}

### Road lookup status

{_markdown_table(tables['road_lookup_coverage'])}

### Temporal dimensions

{_markdown_table(tables['moves_temporal_distribution'])}

The production schema persists final lookup statuses and chosen dimensions, but not a separate reason for each unmatched temporal/speed/road dimension. Those unavailable causes are not inferred.

### Emission-rate data checks

{_markdown_table(tables['emission_rate_checks'])}

Checks flag NaN, infinity, non-numeric, and negative values. Zero frequency is reported without treating non-motorized zero rates as an error; no scientific high-rate threshold is invented.

## 11. Data Quality Warnings

{warning_text}

## 12. Generated Figures

{figures}

## 13. Quality Verdict

**{verdict}**

{reasons}

This verdict addresses operational consistency and run anomalies only; it is not a scientific validation of routing or emissions methodology.
"""
    path.write_text(text, encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def generate_quality_report(
    run_path: str | Path | None,
    output_dir: str | Path | None = None,
    *,
    show_figures: bool = True,
    save_figures: bool = True,
    overwrite: bool = False,
    config: QualityReportConfig | None = None,
    summary_path: str | Path | None = None,
    detailed_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> QualityReportResult:
    """Generate figures, machine-readable tables, JSON metrics, and Markdown."""
    started = time.perf_counter()
    cfg = config or QualityReportConfig()
    outputs = load_run_outputs(
        run_path,
        summary_path=summary_path,
        detailed_path=detailed_path,
        ledger_path=ledger_path,
        manifest_path=manifest_path,
    )
    destination = Path(output_dir).resolve() if output_dir else outputs.run_path / "quality_report"
    trips, valid, plot_quality, filter_summary, filter_metadata = build_quality_population(outputs.ledger, cfg)
    routing_metrics, routing_table = compute_routing_metrics(trips, valid, plot_quality)
    modal_metrics, mode_table = compute_modal_metrics(valid)
    funnel_metrics, funnel_table, reasons_table = compute_modal_funnel_metrics(trips)
    emissions_metrics, emissions_tables = compute_emissions_metrics(outputs, valid)
    plot_ids = set(plot_quality["physical_trip_id"].dropna().astype(str))
    sinuosity = compute_segment_sinuosity(outputs.detailed, plot_ids, cfg)
    speed_source = outputs.detailed
    if speed_source is None and outputs.summary is not None:
        speed_source = _attach_physical_trip_ids(outputs.summary, outputs.ledger)
    speeds = compute_routed_speeds(speed_source, plot_ids)
    distance_consistency = _distance_consistency(outputs.detailed, valid, cfg)
    emission_consistency = _summary_detailed_emission_consistency(outputs, cfg)
    distance_summary = pd.DataFrame([{
        "trips_compared": int(len(distance_consistency)),
        "consistent_trips": int(distance_consistency.get("consistent", pd.Series(dtype=bool)).sum()),
        "discrepant_trips": int((~distance_consistency.get("consistent", pd.Series(dtype=bool))).sum()),
        "max_absolute_difference_m": float(distance_consistency["absolute_difference_m"].max()) if not distance_consistency.empty else None,
        "rtol": cfg.distance_rtol,
        "atol_m": cfg.distance_atol_m,
    }])
    route_distribution = pd.DataFrame([
        {"metric": "route_gps_ratio", **routing_metrics["route_gps_ratio"]},
        {"metric": "uncovered_fraction", **routing_metrics["uncovered_fraction"]},
        {"metric": "max_continuity_gap_m", **routing_metrics["continuity_gap_m"]},
    ])
    trip_distance = plot_quality[[
        column for column in ("physical_trip_id", "user_id", "trip_id", "transport_mode", "distance_km", "route_completeness_status", "route_gps_ratio", "uncovered_fraction", "max_continuity_gap_m")
        if column in plot_quality
    ]].copy()
    tables: dict[str, pd.DataFrame] = {
        "routing_quality_summary": routing_table,
        "route_quality_distribution": route_distribution,
        "quality_filter_summary": filter_summary,
        "mode_summary": mode_table,
        "modal_funnel_summary": funnel_table,
        "failure_reason_summary": reasons_table,
        "trip_distance_by_mode": trip_distance,
        "segment_sinuosity": sinuosity,
        "routed_speed": speeds,
        "distance_consistency": distance_consistency,
        "distance_consistency_summary": distance_summary,
        "emission_output_consistency": emission_consistency,
        **emissions_tables,
    }
    warnings = _quality_warnings(
        trips, valid, plot_quality, emissions_metrics, emissions_tables,
        distance_consistency, emission_consistency, cfg,
    )
    verdict, verdict_reasons = _verdict(warnings)
    figures_dir, tables_dir = _prepare_output_dir(destination, overwrite)
    figure_paths: list[Path] = []
    visual_notes: dict[str, Any] = {}
    if save_figures or show_figures:
        figure_paths, visual_notes = generate_figures(
            valid, plot_quality, sinuosity, speeds, figures_dir,
            config=cfg, show_figures=show_figures, save_figures=save_figures,
        )
    metrics: dict[str, Any] = {
        "run": {
            "name": outputs.run_path.name,
            "path": str(outputs.run_path),
            "users": int(outputs.ledger.get("user_id", pd.Series(dtype=object)).nunique()),
            "manifest_status": outputs.manifest.get("status"),
            "detailed_available": outputs.detailed is not None,
            "summary_available": outputs.summary is not None,
        },
        "routing": routing_metrics,
        "quality_filter": filter_metadata,
        "modes": modal_metrics,
        "modal_funnel": funnel_metrics,
        "emissions": emissions_metrics,
        "segment_sinuosity": {"rows": int(len(sinuosity)), **_quantiles(sinuosity.get("segment_sinuosity", pd.Series(dtype=float)))},
        "routed_speed_kmh": {"rows": int(len(speeds)), **_quantiles(speeds.get("speed_kmh", pd.Series(dtype=float)))},
        "visual_clipping": visual_notes,
        "warnings": warnings,
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "configuration": {
            "valid_statuses": list(cfg.valid_statuses),
            "route_ratio_outer_fence_iqr": cfg.route_ratio_outer_fence_iqr,
            "min_trips_for_tail_filter": cfg.min_trips_for_tail_filter,
            "min_segment_displacement_m": cfg.min_segment_displacement_m,
            "visual_upper_quantile": cfg.visual_upper_quantile,
        },
    }
    _write_tables(tables, tables_dir)
    metrics_path = destination / "quality_metrics.json"
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    markdown_path = destination / "quality_report.md"
    _write_markdown(markdown_path, outputs, metrics, tables, figure_paths, warnings, verdict, verdict_reasons)
    elapsed = time.perf_counter() - started
    metrics["generation_seconds"] = elapsed
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2), encoding="utf-8")
    return QualityReportResult(
        outputs.run_path, destination, markdown_path, metrics_path,
        verdict, verdict_reasons, metrics, tables, figure_paths, elapsed,
    )


def generate_quality_report_if_enabled(
    enabled: bool,
    *,
    run_path: str | Path | None,
    **kwargs,
) -> QualityReportResult | None:
    """Return immediately when disabled, without touching the run or filesystem."""
    if not enabled:
        return None
    return generate_quality_report(run_path=run_path, **kwargs)


def display_quality_report_summary(report: QualityReportResult) -> None:
    """Print one compact notebook-friendly summary block."""
    metrics = report.metrics
    routing = metrics["routing"]
    filtering = metrics["quality_filter"]
    emissions = metrics["emissions"]
    modes = metrics["modes"]
    ratio = routing["route_gps_ratio"]
    uncovered = routing["uncovered_fraction"]
    lines = [
        "Extended Quality Report",
        "",
        f"Run: {metrics['run']['name']}",
        f"Users: {metrics['run']['users']:,}",
        f"Physical trips: {routing['total_trips']:,}",
        "",
        "Routing:",
        f"Complete: {routing['complete']:,}",
        f"Partial: {routing['partial']:,}",
        f"Failed: {routing['failed']:,}",
        f"Quality rejected: {routing['quality_rejected']:,}",
        "",
        "Plot-quality retention:",
        f"Valid routes: {filtering['pipeline_valid_trips']:,}",
        f"Used in plots: {filtering['plot_quality_trips']:,}",
        f"Retention: {filtering['retention_percent']:.2f}%",
        "",
        "Modes:",
        *[f"{mode}: {modes.get(mode, 0):,}" for mode in MODE_ORDER],
        "",
        "Emissions:",
        f"Eligible trips: {emissions['eligible_trips']:,}",
        f"Successful: {emissions['successful_trips']:,}",
        f"Success rate: {emissions['success_rate_percent']:.2f}%" if emissions["success_rate_percent"] is not None else "Success rate: not applicable",
        "",
        "Routing quality:",
        f"Median route/GPS ratio: {ratio['p50']}",
        f"P95 route/GPS ratio: {ratio['p95']}",
        f"Median uncovered fraction: {uncovered['p50']}",
        f"P95 uncovered fraction: {uncovered['p95']}",
        "",
        f"Verdict: {report.verdict}",
        f"Report: {report.markdown_path}",
    ]
    print("\n".join(lines))
