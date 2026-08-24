"""Canonical English production-output schemas at the pipeline boundary."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


OUTPUT_MODES = frozenset({"summary", "detailed", "both"})

SUMMARY_COLUMNS = (
    "user_id", "trip_id", "physical_trip_id",
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
    if emissions.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    result = add_kepler_time(canonicalize_english_columns(emissions))
    return result.reindex(columns=SUMMARY_COLUMNS)


def _process_detailed_chunk(
    chunk: pd.DataFrame,
    emissions_lookup: pd.DataFrame | None,
    ledger_lookup: pd.DataFrame | None,
) -> pd.DataFrame:
    """Process a single bounded chunk of routes with emissions and ledger metadata."""
    base = chunk.copy()
    if emissions_lookup is not None and not emissions_lookup.empty and "_output_row_id" in base:
        base = base.merge(
            emissions_lookup,
            on="_output_row_id",
            how="left",
            suffixes=("", "_emission"),
        )
    base = base.drop(columns="_output_row_id", errors="ignore")
    result = add_kepler_time(canonicalize_english_columns(base))
    if ledger_lookup is not None and not ledger_lookup.empty and "physical_trip_id" in result:
        result = result.merge(
            ledger_lookup,
            on="physical_trip_id",
            how="left",
            suffixes=("", "_ledger"),
        )
    return result.reindex(columns=DETAILED_COLUMNS)


def _extract_emissions_lookup(emissions: pd.DataFrame | str | Path | None) -> pd.DataFrame | None:
    if emissions is None:
        return None
    em_df = pd.read_parquet(emissions) if isinstance(emissions, (str, Path)) else emissions
    if em_df.empty or "_output_row_id" not in em_df.columns:
        return None
    target_fields = set(_DIRECT_RENAMES).union(
        alias for aliases in _EMISSION_ALIASES.values() for alias in aliases
    ).union({
        "emission_lookup_status", "road_lookup_status",
        "Month", "Hour", "Day", "Road", "Source", "SpeedBin", "avg_speed_mph",
    })
    fields = ["_output_row_id", *[c for c in target_fields if c in em_df.columns]]
    return em_df[fields].drop_duplicates("_output_row_id")


def _extract_ledger_lookup(ledger: pd.DataFrame | str | Path | None) -> pd.DataFrame | None:
    if ledger is None:
        return None
    led_df = pd.read_parquet(ledger) if isinstance(ledger, (str, Path)) else ledger
    if led_df.empty or "physical_trip_id" not in led_df.columns:
        return None
    target_ledger = [
        "physical_trip_id", "hypotheses_attempted", "hypotheses_successful",
        "hypotheses_attempted_count", "hypotheses_successful_count",
        "classification_success", "processing_status", "failure_reason",
        "pre_routing_quality_status", "emissions_success",
    ]
    fields = [c for c in target_ledger if c in led_df.columns]
    return led_df[fields].drop_duplicates("physical_trip_id")


def project_detailed(
    routes: pd.DataFrame, emissions: pd.DataFrame, ledger: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Attach emission audit values to all route rows without dropping failures."""
    if routes.empty:
        return pd.DataFrame(columns=DETAILED_COLUMNS)
    base = routes.copy()
    if "_output_row_id" not in base.columns:
        base["_output_row_id"] = range(len(base))
    emissions_lookup = _extract_emissions_lookup(emissions)
    ledger_lookup = _extract_ledger_lookup(ledger)
    return _process_detailed_chunk(base, emissions_lookup, ledger_lookup)


def project_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    result = canonicalize_english_columns(ledger)
    if "pre_routing_quality_status" not in result:
        result["pre_routing_quality_status"] = pd.NA
    return result.reindex(columns=[
        *LEDGER_OUTPUT_COLUMNS,
        *[column for column in result.columns if column not in LEDGER_OUTPUT_COLUMNS],
    ])


def write_parquet_atomic(frame: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    """Write DataFrame to a temporary parquet file and atomically rename to destination."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        frame.to_parquet(tmp_path, index=False, **kwargs)
        os.replace(tmp_path, out_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return out_path


def write_output_artifact(frame: pd.DataFrame, path: Path) -> dict:
    out_path = write_parquet_atomic(frame, path)
    return {
        "generated": True,
        "filename": out_path.name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "size_mb": round(out_path.stat().st_size / (1024 * 1024), 3) if out_path.exists() else 0.0,
    }
def write_detailed_output_streaming(
    routes: pd.DataFrame | str | Path,
    emissions: pd.DataFrame | str | Path | None,
    ledger: pd.DataFrame | str | Path | None,
    output_path: str | Path,
    *,
    chunk_size: int = 50000,
    show_progress: bool = False,
) -> dict:
    """Memory-bounded streaming writer for routes_emissions_detailed.parquet using PyArrow."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(".parquet.tmp")

    emissions_lookup = _extract_emissions_lookup(emissions)
    ledger_lookup = _extract_ledger_lookup(ledger)

    writer = None
    total_rows = 0

    try:
        if isinstance(routes, (str, Path)):
            parquet_file = pq.ParquetFile(routes)
            pbar = (
                tqdm(total=parquet_file.metadata.num_rows, desc="Writing detailed output", unit="row", dynamic_ncols=True)
                if show_progress and parquet_file.metadata.num_rows > chunk_size
                else None
            )
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                chunk_df = batch.to_pandas()
                if "_output_row_id" not in chunk_df.columns:
                    chunk_df["_output_row_id"] = range(total_rows, total_rows + len(chunk_df))
                processed = _process_detailed_chunk(chunk_df, emissions_lookup, ledger_lookup)
                table = pa.Table.from_pandas(processed, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_file, table.schema, compression="snappy")
                writer.write_table(table)
                total_rows += len(processed)
                if pbar is not None:
                    pbar.update(len(processed))
                del chunk_df, processed, table
            if pbar is not None:
                pbar.close()
        else:
            if routes.empty:
                empty_df = pd.DataFrame(columns=DETAILED_COLUMNS)
                empty_df.to_parquet(tmp_file, index=False)
                total_rows = 0
            else:
                n = len(routes)
                if "_output_row_id" not in routes.columns:
                    routes = routes.copy()
                    routes["_output_row_id"] = range(n)
                pbar = (
                    tqdm(total=n, desc="Writing detailed output", unit="row", dynamic_ncols=True)
                    if show_progress and n > chunk_size
                    else None
                )
                for start in range(0, n, chunk_size):
                    chunk_df = routes.iloc[start:start + chunk_size].copy()
                    processed = _process_detailed_chunk(chunk_df, emissions_lookup, ledger_lookup)
                    table = pa.Table.from_pandas(processed, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_file, table.schema, compression="snappy")
                    writer.write_table(table)
                    total_rows += len(processed)
                    if pbar is not None:
                        pbar.update(len(processed))
                    del chunk_df, processed, table
                if pbar is not None:
                    pbar.close()

        if writer is not None:
            writer.close()
            writer = None

        if tmp_file.exists():
            os.replace(tmp_file, out_file)

    except Exception:
        if writer is not None:
            writer.close()
        if tmp_file.exists():
            tmp_file.unlink()
        raise

    return {
        "generated": True,
        "filename": out_file.name,
        "rows": total_rows,
        "columns": len(DETAILED_COLUMNS),
        "size_mb": round(out_file.stat().st_size / (1024 * 1024), 3) if out_file.exists() else 0.0,
    }

def write_summary_output_streaming(
    emissions: pd.DataFrame | str | Path | None,
    output_path: str | Path,
    *,
    chunk_size: int = 50000,
    show_progress: bool = False,
) -> dict:
    """Memory-bounded streaming writer for routes_emissions_summary.parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_suffix(".parquet.tmp")

    writer = None
    total_rows = 0

    try:
        if emissions is None:
            empty_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
            empty_df.to_parquet(tmp_file, index=False)
        elif isinstance(emissions, (str, Path)):
            parquet_file = pq.ParquetFile(emissions)
            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                chunk_df = batch.to_pandas()
                processed = project_summary(chunk_df)
                table = pa.Table.from_pandas(processed, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp_file, table.schema, compression="snappy")
                writer.write_table(table)
                total_rows += len(processed)
                del chunk_df, processed, table
        else:
            if emissions.empty:
                empty_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
                empty_df.to_parquet(tmp_file, index=False)
                total_rows = 0
            else:
                n = len(emissions)
                for start in range(0, n, chunk_size):
                    chunk_df = emissions.iloc[start:start + chunk_size]
                    processed = project_summary(chunk_df)
                    table = pa.Table.from_pandas(processed, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_file, table.schema, compression="snappy")
                    writer.write_table(table)
                    total_rows += len(processed)
                    del chunk_df, processed, table

        if writer is not None:
            writer.close()
            writer = None

        if tmp_file.exists():
            os.replace(tmp_file, out_file)

    except Exception:
        if writer is not None:
            writer.close()
        if tmp_file.exists():
            tmp_file.unlink()
        raise

    return {
        "generated": True,
        "filename": out_file.name,
        "rows": total_rows,
        "columns": len(SUMMARY_COLUMNS),
        "size_mb": round(out_file.stat().st_size / (1024 * 1024), 3) if out_file.exists() else 0.0,
    }
# End of production output writers.
