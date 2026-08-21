"""Canonical English production-output schemas at the pipeline boundary."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


OUTPUT_MODES = frozenset({"summary", "detailed", "both"})

SUMMARY_COLUMNS = (
    "user_id", "trip_id",
    "local_timestamp", "kepler_time", "latitude", "longitude",
    "speed_kmh", "duration_s", "distance_m",
    "osmid", "highway", "geometry",
    "transport_mode",
    "co_g_km", "co_total_g",
    "co2_g_km", "co2_total_g",
    "co2e_g_km", "co2e_total_g",
    "hc_g_km", "hc_total_g",
    "nox_g_km", "nox_total_g",
    "pm10_g_km", "pm10_total_g",
    "pm25_g_km", "pm25_total_g",
    "home_lat", "home_lon", "home_ageb",
)

DETAILED_COLUMNS = (
    "user_id", "trip_id", "physical_trip_id",
    "local_timestamp", "kepler_time", "latitude", "longitude", "speed_kmh", "duration_s", "distance_m",
    "osmid", "highway", "geometry", "transport_mode", "network_hypothesis",
    "hypotheses_attempted", "hypotheses_successful", "hypotheses_attempted_count",
    "hypotheses_successful_count", "classification_success", "processing_status", "failure_reason",
    "pre_routing_quality_status",
    "routing_failed", "routing_status", "audit_flag", "route_component_id",
    "endpoint_role", "endpoint_start_status", "endpoint_end_status",
    "endpoint_start_uncovered_m", "endpoint_end_uncovered_m", "endpoint_uncovered_m",
    "routing_event", "lookahead_skipped_pings", "lookahead_elapsed_seconds",
    "lookahead_observed_distance_m", "lookahead_origin_ping", "lookahead_resume_ping",
    "lookahead_failure_reason", "unresolved_transition_count",
    "route_component_count", "failed_row_count", "failed_row_fraction", "failure_cluster_count",
    "max_continuity_gap_m", "uncovered_distance_m", "uncovered_fraction", "route_gps_ratio",
    "reconstructed_distance_m", "gps_distance_m", "route_coverage_fraction",
    "endpoint_start_error_m", "endpoint_end_error_m", "route_completeness_status",
    "emissions_eligible", "emissions_success", "emissions_scope",
    "near_subway_line", "near_bus_route", "router_version", "endpoint_patch_version",
    "moves_month", "moves_hour", "moves_day_type", "moves_road_type", "moves_source_type",
    "avg_speed_mph", "moves_speed_bin", "road_lookup_status", "emission_lookup_status",
    "co_g_km", "co_total_g", "co2_g_km", "co2_total_g", "co2e_g_km", "co2e_total_g",
    "hc_g_km", "hc_total_g", "nox_g_km", "nox_total_g", "pm10_g_km", "pm10_total_g",
    "pm25_g_km", "pm25_total_g",
    "home_lat", "home_lon", "home_ageb", "home_quality_status", "home_inventory_status",
)

LEDGER_OUTPUT_COLUMNS = (
    "user_id", "trip_id", "physical_trip_id",
    "raw_ping_count", "effective_ping_count", "pct_pings_conserved",
    "pre_routing_quality_status",
    "hypotheses_attempted", "hypotheses_successful", "hypotheses_attempted_count",
    "hypotheses_successful_count", "route_success",
    "route_component_count", "failed_row_count", "failed_row_fraction", "failure_cluster_count",
    "max_continuity_gap_m", "uncovered_distance_m", "uncovered_fraction", "route_gps_ratio",
    "endpoint_start_error_m", "endpoint_end_error_m", "unresolved_transition_count",
    "reconstructed_distance_m", "gps_distance_m", "route_coverage_fraction",
    "route_completeness_status",
    "final_mode", "classification_success", "modal_usable",
    "emissions_eligible", "emissions_usable", "emissions_success",
    "processing_status", "failure_reason",
)

_DIRECT_RENAMES = {
    "caid": "user_id", "trip": "trip_id", "Speed [km/h]": "speed_kmh",
    "modo_transporte": "transport_mode", "ruteo_fallido": "routing_failed",
    "flag_auditoria": "audit_flag", "total_uncovered_distance_m": "uncovered_distance_m",
    "Month": "moves_month", "Hour": "moves_hour", "Day": "moves_day_type",
    "Road": "moves_road_type", "Source": "moves_source_type", "SpeedBin": "moves_speed_bin",
    "home_quality_flag": "home_quality_status",
}
_EMISSION_ALIASES = {
    "co_g_km": ("Densidad_CO_g_km",), "co_total_g": ("Total_CO_g",),
    "co2_g_km": ("Densidad_CO2_g_km",), "co2_total_g": ("Total_CO2_g",),
    "co2e_g_km": ("Densidad_CO2e_g_km", "Densidad_CO2_Equiv_g_km"),
    "co2e_total_g": ("Total_CO2e_g", "Total_CO2_Equiv_g"),
    "hc_g_km": ("Densidad_HC_g_km",), "hc_total_g": ("Total_HC_g",),
    "nox_g_km": ("Densidad_NOx_g_km",), "nox_total_g": ("Total_NOx_g",),
    "pm10_g_km": ("Densidad_PM10_g_km",), "pm10_total_g": ("Total_PM10_g",),
    "pm25_g_km": ("Densidad_PM2.5_g_km", "Densidad_PM25_g_km"),
    "pm25_total_g": ("Total_PM2.5_g", "Total_PM25_g"),
}


def validate_output_mode(output_mode: str) -> str:
    normalized = str(output_mode).strip().lower()
    if normalized not in OUTPUT_MODES:
        raise ValueError(f"output_mode must be one of {sorted(OUTPUT_MODES)}, got {output_mode!r}")
    return normalized


def _promote_alias(result: pd.DataFrame, target: str, sources: Iterable[str]) -> None:
    if target not in result:
        for source in sources:
            if source in result:
                result[target] = result[source]
                break


def canonicalize_english_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Translate legacy names only at the production-output boundary."""
    result = frame.copy()
    for legacy, canonical in _DIRECT_RENAMES.items():
        _promote_alias(result, canonical, (legacy,))
    for canonical, aliases in _EMISSION_ALIASES.items():
        _promote_alias(result, canonical, aliases)
    aliases_to_drop = set(_DIRECT_RENAMES).union(
        alias for aliases in _EMISSION_ALIASES.values() for alias in aliases
    )
    return result.drop(columns=[column for column in aliases_to_drop if column in result], errors="ignore")


def add_kepler_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the legacy-compatible visualization helper from the canonical time.

    ``local_timestamp`` remains the authoritative temporal field. Kepler.gl
    visualizations historically consume a timezone-free ``YYYY-MM-DD HH:MM:SS``
    string, so this helper is derived from it at the output boundary.
    """
    result = frame.copy()
    if "local_timestamp" in result:
        result["kepler_time"] = pd.to_datetime(
            result["local_timestamp"], errors="coerce"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        result["kepler_time"] = pd.NA
    return result


def project_summary(emissions: pd.DataFrame) -> pd.DataFrame:
    result = add_kepler_time(canonicalize_english_columns(emissions))
    return result.reindex(columns=SUMMARY_COLUMNS)


def project_detailed(
    routes: pd.DataFrame, emissions: pd.DataFrame, ledger: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Attach emission audit values to all route rows without dropping failures."""
    base = routes.copy()
    base["_output_row_id"] = range(len(base))
    if not emissions.empty and "_output_row_id" in emissions:
        emission_fields = set(_DIRECT_RENAMES).union(
            alias for aliases in _EMISSION_ALIASES.values() for alias in aliases
        ).union({"emission_lookup_status", "road_lookup_status", "Month", "Hour", "Day", "Road", "Source", "SpeedBin", "avg_speed_mph"})
        fields = ["_output_row_id", *[column for column in emission_fields if column in emissions]]
        base = base.merge(emissions[fields], on="_output_row_id", how="left", suffixes=("", "_emission"))
    base = base.drop(columns="_output_row_id", errors="ignore")
    result = add_kepler_time(canonicalize_english_columns(base))
    if ledger is not None and not ledger.empty and "physical_trip_id" in result:
        ledger_fields = [
            "physical_trip_id", "hypotheses_attempted", "hypotheses_successful",
            "hypotheses_attempted_count", "hypotheses_successful_count",
            "classification_success", "processing_status", "failure_reason",
            "pre_routing_quality_status", "emissions_success",
        ]
        available = [column for column in ledger_fields if column in ledger]
        result = result.merge(
            ledger[available].drop_duplicates("physical_trip_id"),
            on="physical_trip_id", how="left", suffixes=("", "_ledger"),
        )
    return result.reindex(columns=DETAILED_COLUMNS)


def project_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    result = canonicalize_english_columns(ledger)
    if "pre_routing_quality_status" not in result:
        result["pre_routing_quality_status"] = pd.NA
    return result.reindex(columns=[
        *LEDGER_OUTPUT_COLUMNS,
        *[column for column in result.columns if column not in LEDGER_OUTPUT_COLUMNS],
    ])


def write_output_artifact(frame: pd.DataFrame, path: Path) -> dict:
    frame.to_parquet(path, index=False)
    return {
        "generated": True,
        "filename": path.name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 3),
    }
