"""Callable pipeline_v4 workflow for an externally supplied GPS sample.

This module preserves the validated segmentation/routing/emissions algorithms
and makes their production handoff explicit: GPS-level classifier context,
one ledger row per routed trip, and caller-provided output directories.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
from typing import Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import geopandas as gpd
from joblib import Parallel, delayed
import pandas as pd
import shapely
from shapely import wkt
from tqdm.auto import tqdm

from pipeline_v4.preprocessing.workflow import attach_user_metadata
from pipeline_v4.src import config
from pipeline_v4.src.emissions import calculate_emissions
from pipeline_v4.src.modal_classification import (
    PriorModeClassifier,
    InfrastructureProximityCache,
    ServingContractError,
    TripServingContext,
    calcular_cercania_infraestructura,
    create_modal_evaluator,
)
from pipeline_v4.src.output_schema import (
    project_detailed,
    project_ledger,
    project_summary,
    validate_output_mode,
    write_output_artifact,
    write_parquet_atomic,
    write_detailed_output_streaming,
    write_summary_output_streaming,
)
from pipeline_v4.src import random_forest_contract
from pipeline_v4.src.random_forest_contract import MIN_EFFECTIVE_PINGS, MIN_PCT_CONSERVED
from pipeline_v4.src.route_quality import (
    QUALITY_COLUMNS,
    attach_route_quality,
    evaluate_route_quality,
    is_strict_emissions_usable,
)
from pipeline_v4.src.routing import (
    RouteHypothesisEvaluator,
    TRANSFORMER_TO_UTM,
    build_candidate_edge_index,
    get_candidates_vectorized,
)
from pipeline_v4.src.endpoint_routing import build_incident_edge_index
from pipeline_v4.src.segmentation import apply_spatial_filter, assign_trips


LEDGER_COLUMNS = (
    "user_id", "trip_id", "physical_trip_id", "processing_status", "failure_reason",
    "raw_ping_count", "effective_ping_count", "pct_pings_conserved",
    "pre_routing_quality_status",
    "hypotheses_attempted", "hypotheses_successful", "hypotheses_attempted_count",
    "hypotheses_successful_count", "route_success", "final_mode",
    "classification_success", "modal_usable", *QUALITY_COLUMNS, "emissions_usable", "emissions_success",
)


@dataclass(frozen=True)
class DayProcessingResult:
    routes: pd.DataFrame
    trip_ledger: pd.DataFrame


@dataclass(frozen=True)
class PipelineV4Result:
    routes: pd.DataFrame
    individual_emissions: pd.DataFrame
    trip_ledger: pd.DataFrame
    output_dir: Path
    output_artifacts: dict[str, dict] = field(default_factory=dict)


def load_pipeline_resources() -> dict[str, Any]:
    """Load production resources, restricting edge caches to candidate columns."""
    with open(config.FILE_GRAFO, "rb") as stream:
        graph_drive = pickle.load(stream)
    with open(config.FILE_GRAFO_WALK, "rb") as stream:
        graph_walk = pickle.load(stream)
    with open(config.FILE_CACHE_IG_DRIVE, "rb") as stream:
        ig_drive, map_drive = pickle.load(stream)
    with open(config.FILE_CACHE_IG_WALK, "rb") as stream:
        ig_walk, map_walk = pickle.load(stream)
    edge_columns = ["u", "v", "geometry"]
    edges_drive = gpd.read_parquet(config.FILE_CACHE_EDGES_DRIVE, columns=edge_columns)
    edges_walk = gpd.read_parquet(config.FILE_CACHE_EDGES_WALK, columns=edge_columns)
    candidate_edges_drive = build_candidate_edge_index(edges_drive, resource_key="drive")
    candidate_edges_walk = build_candidate_edge_index(edges_walk, resource_key="walk")
    incident_edges_drive = build_incident_edge_index(edges_drive)
    incident_edges_walk = build_incident_edge_index(edges_walk)
    bus_routes = gpd.read_file(config.FILE_BUS).to_crs("EPSG:32614")
    subway_df = pd.read_csv(config.FILE_METRO)
    if "geometry" in subway_df.columns:
        subway_df["geometry"] = subway_df["geometry"].apply(wkt.loads)
        subway_routes = gpd.GeoDataFrame(subway_df, geometry="geometry", crs="EPSG:4326")
    elif "WKT" in subway_df.columns:
        subway_df["geometry"] = subway_df["WKT"].apply(wkt.loads)
        subway_routes = gpd.GeoDataFrame(subway_df, geometry="geometry", crs="EPSG:4326")
    elif {"lat", "lon"}.issubset(subway_df.columns):
        subway_routes = gpd.GeoDataFrame(
            subway_df, geometry=gpd.points_from_xy(subway_df.lon, subway_df.lat), crs="EPSG:4326"
        )
    else:
        raise ValueError("Subway route file format is unsupported")
    subway_routes = subway_routes.to_crs("EPSG:32614")
    return {
        "graph_drive": graph_drive, "graph_walk": graph_walk,
        "ig_drive": ig_drive, "ig_walk": ig_walk,
        "map_drive": map_drive, "map_walk": map_walk,
        "edges_drive": edges_drive, "edges_walk": edges_walk,
        "candidate_edges_drive": candidate_edges_drive,
        "candidate_edges_walk": candidate_edges_walk,
        "incident_edges_drive": incident_edges_drive,
        "incident_edges_walk": incident_edges_walk,
        "bus_routes": bus_routes, "subway_routes": subway_routes,
        "infrastructure_proximity_cache": InfrastructureProximityCache(),
    }


def _raw_ping_count_for_effective_trip(raw_segmented: pd.DataFrame, effective_trip: pd.DataFrame) -> int:
    """Count supplied GPS pings inside this effective trip's temporal interval.

    The spatial filter may split one pre-filter movement sequence into multiple
    effective trips. Counting the entire pre-filter sequence for each child
    would duplicate the denominator, so the interval itself is canonical.
    """
    start = pd.to_datetime(effective_trip["local_timestamp"]).min()
    end = pd.to_datetime(effective_trip["local_timestamp"]).max()
    raw_time = pd.to_datetime(raw_segmented["local_timestamp"])
    return int(((raw_time >= start) & (raw_time <= end)).sum())


def _trip_identifier(user_id: object, trip_id: object, frame: pd.DataFrame) -> str:
    date = pd.to_datetime(frame["local_timestamp"]).min().strftime("%Y-%m-%d")
    return f"{user_id}_{date}_{trip_id}"


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def _ensure_ledger_schema(ledger: pd.DataFrame) -> pd.DataFrame:
    """Normalize a day ledger to the canonical production contract.

    ``process_user_day`` always writes the full schema.  Keeping this small
    compatibility boundary lets callers that provide a legacy day result (for
    example, a historical workflow wrapper) pass through safely while making
    the new emissions gate explicit rather than failing with a ``KeyError``.
    """
    result = ledger.copy()
    status = result.get("processing_status", pd.Series(None, index=result.index))
    route_success = result.get("route_success", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    legacy_eligible = status.isin(("emissions_pending", "success")) | route_success
    defaults = {
        "route_component_count": 0,
        "failed_row_count": 0,
        "failed_row_fraction": 0.0,
        "failure_cluster_count": 0,
        "max_continuity_gap_m": 0.0,
        "total_uncovered_distance_m": 0.0,
        "uncovered_fraction": 0.0,
        "route_gps_ratio": None,
        "endpoint_start_error_m": 0.0,
        "endpoint_end_error_m": 0.0,
        "unresolved_transition_count": 0,
        "reconstructed_distance_m": 0.0,
        "gps_distance_m": 0.0,
        "route_coverage_fraction": 0.0,
        "route_completeness_status": "complete",
        "modal_usable": False,
        "emissions_eligible": legacy_eligible,
        "emissions_usable": legacy_eligible,
        "emissions_success": False,
        "pre_routing_quality_status": "not_evaluated",
    }
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default
    return result.reindex(columns=[*LEDGER_COLUMNS, *[c for c in result.columns if c not in LEDGER_COLUMNS]])


def _ledger_row(user_id, trip_id, physical_trip_id, context, **updates):
    row = {
        "user_id": user_id,
        "trip_id": trip_id,
        "physical_trip_id": physical_trip_id,
        "processing_status": "routing_failed",
        "failure_reason": None,
        "raw_ping_count": context.raw_ping_count,
        "effective_ping_count": context.effective_ping_count,
        "pct_pings_conserved": context.pct_pings_conserved,
        "pre_routing_quality_status": "not_evaluated",
        "hypotheses_attempted": "",
        "hypotheses_successful": "",
        "hypotheses_attempted_count": 0,
        "hypotheses_successful_count": 0,
        "route_success": False,
        "final_mode": None,
        "classification_success": False,
        "modal_usable": False,
        "route_component_count": 0,
        "failed_row_count": 0,
        "failed_row_fraction": 0.0,
        "failure_cluster_count": 0,
        "max_continuity_gap_m": 0.0,
        "total_uncovered_distance_m": 0.0,
        "uncovered_fraction": 0.0,
        "route_gps_ratio": None,
        "endpoint_start_error_m": 0.0,
        "endpoint_end_error_m": 0.0,
        "unresolved_transition_count": 0,
        "reconstructed_distance_m": 0.0,
        "gps_distance_m": 0.0,
        "route_coverage_fraction": 0.0,
        "route_completeness_status": None,
        "emissions_eligible": False,
        "emissions_usable": False,
        "emissions_success": False,
    }
    row.update(updates)
    return row


def _successful_hypotheses(hypotheses):
    successful = {}
    for mode, frame in hypotheses.items():
        failed = frame.get("ruteo_fallido", pd.Series(False, index=frame.index)).fillna(True)
        if not frame.empty and bool((~failed.astype(bool)).any()):
            successful[mode] = frame
    return successful


def process_user_day(user_id: object, day_frame: pd.DataFrame, resources: dict[str, Any]) -> DayProcessingResult:
    """Process one user-day while retaining every positive trip outcome."""
    if day_frame.empty:
        return DayProcessingResult(pd.DataFrame(), _empty_ledger())
    raw_segmented = assign_trips(day_frame.copy())
    effective_day = assign_trips(apply_spatial_filter(day_frame.copy(), min_dist_m=15.0))
    effective_day = calcular_cercania_infraestructura(
        effective_day, resources["subway_routes"], resources["bus_routes"],
        proximity_cache=resources.get("infrastructure_proximity_cache"),
    )
    latitude = "lat_ruteo" if "lat_ruteo" in effective_day.columns else "latitude"
    longitude = "lon_ruteo" if "lon_ruteo" in effective_day.columns else "longitude"
    # Candidate generation only needs point geometry in network CRS (UTM 14N).
    # Using direct pyproj transform + shapely.points eliminates GeoDataFrame.to_crs overhead.
    xs, ys = TRANSFORMER_TO_UTM.transform(effective_day[longitude].to_numpy(), effective_day[latitude].to_numpy())
    points = gpd.GeoDataFrame(
        index=effective_day.index,
        geometry=shapely.points(xs, ys),
        crs=resources["edges_drive"].crs,
    )
    effective_day["drive_ids"], effective_day["drive_dists"] = get_candidates_vectorized(
        resources.get("candidate_edges_drive", resources["edges_drive"]), points, buffer_m=150
    )
    effective_day["walk_ids"], effective_day["walk_dists"] = get_candidates_vectorized(
        resources.get("candidate_edges_walk", resources["edges_walk"]), points, buffer_m=50
    )
    prior = PriorModeClassifier()
    evaluator = RouteHypothesisEvaluator(
        resources["graph_drive"], resources["graph_walk"], resources["ig_drive"],
        resources["ig_walk"], resources["map_drive"], resources["map_walk"],
        resources["subway_routes"],
        router_version=config.ROUTER_VERSION,
        edges_drive=resources["edges_drive"],
        edges_walk=resources["edges_walk"],
        candidate_edges_drive=resources.get("candidate_edges_drive"),
        candidate_edges_walk=resources.get("candidate_edges_walk"),
        incident_edges_drive=resources.get("incident_edges_drive"),
        incident_edges_walk=resources.get("incident_edges_walk"),
    )
    classifier = resources.get("modal_evaluator")
    if classifier is None:
        classifier = create_modal_evaluator(config.MODAL_CLASSIFIER)
    routed, ledger = [], []
    for trip_id, trip in effective_day.groupby("trip", sort=False):
        if trip_id <= 0:
            stop = trip.copy()
            routed.append(pd.DataFrame({
                "caid": user_id, "trip": trip_id,
                "latitude": stop.latitude, "longitude": stop.longitude,
                "Speed [km/h]": 0.0, "local_timestamp": stop.local_timestamp,
                "start_node": "N/A", "end_node": "N/A", "osmid": "N/A",
                "highway": "parada_inactiva",
                "geometry": [f"POINT ({lon} {lat})" for lon, lat in zip(stop.longitude, stop.latitude)],
                "distance_m": 0.0, "modo_transporte": "Parada",
                "ruteo_fallido": False, "corregido_espacialmente": False,
                "flag_auditoria": "Parada_Inactiva",
            }))
            continue

        raw_count = _raw_ping_count_for_effective_trip(raw_segmented, trip)
        physical_trip_id = _trip_identifier(user_id, trip_id, trip)
        context = TripServingContext.from_trip(trip, raw_ping_count=raw_count)
        base = _ledger_row(user_id, trip_id, physical_trip_id, context)
        min_pings = getattr(random_forest_contract, "MIN_EFFECTIVE_PINGS", MIN_EFFECTIVE_PINGS)
        min_pct = 100.0 * getattr(random_forest_contract, "MIN_PCT_CONSERVED", MIN_PCT_CONSERVED)
        if (context.effective_ping_count < min_pings or
                context.pct_pings_conserved < min_pct):
            base.update(
                processing_status="quality_rejected", failure_reason="quality_guardrail",
                pre_routing_quality_status="rejected",
            )
            ledger.append(base)
            continue

        candidates = prior.prune_impossible_hypotheses(
            trip, trip["near_subway_line"], trip["near_bus_route"]
        )
        base.update(
            pre_routing_quality_status="passed",
            hypotheses_attempted=";".join(candidates),
            hypotheses_attempted_count=len(candidates),
        )
        if not candidates:
            base.update(processing_status="no_hypothesis", failure_reason="no_feasible_hypothesis")
            ledger.append(base)
            continue
        try:
            hypotheses = evaluator.evaluate(user_id, trip, candidates)
        except Exception as exc:
            base.update(processing_status="routing_failed", failure_reason=f"{type(exc).__name__}: {exc}")
            ledger.append(base)
            continue
        successful = _successful_hypotheses(hypotheses)
        base.update(
            hypotheses_successful=";".join(successful),
            hypotheses_successful_count=len(successful),
            route_success=bool(successful),
        )
        if not successful:
            base.update(processing_status="routing_failed", failure_reason="all_hypotheses_failed")
            ledger.append(base)
            continue
        try:
            enriched = {
                mode: calcular_cercania_infraestructura(
                    frame.copy(), resources["subway_routes"], resources["bus_routes"],
                    proximity_cache=resources.get("infrastructure_proximity_cache"),
                )
                for mode, frame in successful.items()
            }
            final_mode, best_route, _, _ = classifier.select_final_mode(
                enriched,
                resources["subway_routes"],
                resources["bus_routes"],
                serving_context=context,
            )
        except (ServingContractError, Exception) as exc:
            base.update(processing_status="classification_failed", failure_reason=f"{type(exc).__name__}: {exc}")
            ledger.append(base)
            continue
        if best_route is None or final_mode in {None, "Calidad insuficiente"}:
            base.update(processing_status="classification_failed", failure_reason="classifier_returned_no_route")
            ledger.append(base)
            continue
        best_route = best_route.copy()
        best_route["physical_trip_id"] = physical_trip_id
        quality = evaluate_route_quality(best_route, trip)
        best_route = attach_route_quality(best_route, quality)
        routed.append(best_route)
        base.update(quality)
        
        modal_usable = bool(final_mode not in {None, "Calidad insuficiente"})
        emissions_usable = is_strict_emissions_usable(final_mode, modal_usable, quality)
        
        base.update(
            route_success=True,
            final_mode=final_mode,
            classification_success=modal_usable,
            modal_usable=modal_usable,
            emissions_eligible=emissions_usable,
            emissions_usable=emissions_usable,
        )
        if emissions_usable:
            base.update(processing_status="emissions_pending", failure_reason=None)
        elif final_mode in {"Caminar", "Metro"}:
            base.update(processing_status="success", failure_reason=None, emissions_success=False)
        else:
            base.update(
                processing_status="success",
                failure_reason="post_routing_quality_failed",
                emissions_success=False,
            )
        ledger.append(base)
    return DayProcessingResult(
        pd.concat(routed, ignore_index=True) if routed else pd.DataFrame(),
        pd.DataFrame(ledger, columns=LEDGER_COLUMNS) if ledger else _empty_ledger(),
    )


def _read_frame(value: pd.DataFrame | str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        if columns is not None:
            avail = [c for c in columns if c in value.columns]
            return value[avail].copy()
        return value.copy()
    if columns is not None:
        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(value)
            avail = [c for c in columns if c in schema.names]
            return pd.read_parquet(value, columns=avail)
        except Exception:
            pass
    return pd.read_parquet(value)


# -----------------------------------------------------------------------------
# PROCESS POOL WORKER LIFECYCLE (SAFE PROCESS TEST MODE)
# -----------------------------------------------------------------------------
_GLOBAL_PROCESS_WORKER_RESOURCES = None


def _init_process_worker():
    """Initializer for ProcessPool workers: loads routing resources once per worker process."""
    global _GLOBAL_PROCESS_WORKER_RESOURCES
    # Limit inner BLAS/OpenMP threads to 1 per worker process to prevent thread oversubscription
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    _GLOBAL_PROCESS_WORKER_RESOURCES = load_pipeline_resources()
    _GLOBAL_PROCESS_WORKER_RESOURCES["modal_evaluator"] = create_modal_evaluator(config.MODAL_CLASSIFIER)


def _process_user_day_process_worker(task_index, user_id, day_frame):
    """Executes process_user_day using pre-loaded worker process resources."""
    global _GLOBAL_PROCESS_WORKER_RESOURCES
    if _GLOBAL_PROCESS_WORKER_RESOURCES is None:
        _init_process_worker()
    return task_index, process_user_day(user_id, day_frame, _GLOBAL_PROCESS_WORKER_RESOURCES)


def _process_indexed_user_day(task_index, user_id, day, loaded_resources):
    """Preserve task order after collecting asynchronously completed work."""
    return task_index, process_user_day(user_id, day, loaded_resources)


def _run_user_day_tasks(
    tasks,
    loaded_resources,
    n_jobs: int,
    backend: str = "threading",
    progress: Any = None,
    process_executor: Any = None,
):
    """Run one bounded task window with deterministic output order."""
    if backend == "process":
        if process_executor is not None:
            futures = [
                process_executor.submit(_process_user_day_process_worker, task_index, user_id, day_frame)
                for task_index, user_id, day_frame in tasks
            ]
            completed = []
            for fut in as_completed(futures):
                completed.append(fut.result())
                if progress is not None and hasattr(progress, "update"):
                    progress.update(1)
            return [result for _, result in sorted(completed, key=lambda item: item[0])]
        else:
            with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_process_worker) as executor:
                futures = [
                    executor.submit(_process_user_day_process_worker, task_index, user_id, day_frame)
                    for task_index, user_id, day_frame in tasks
                ]
                completed = []
                for fut in as_completed(futures):
                    completed.append(fut.result())
                    if progress is not None and hasattr(progress, "update"):
                        progress.update(1)
                return [result for _, result in sorted(completed, key=lambda item: item[0])]

    # Threading backend (default safe mode)
    parallel = Parallel(n_jobs=n_jobs, backend="threading", return_as="generator_unordered")
    task_results = parallel(
        delayed(_process_indexed_user_day)(task_index, user_id, day_frame, loaded_resources)
        for task_index, user_id, day_frame in tasks
    )
    completed = []
    local_pbar = None
    if isinstance(progress, bool):
        if progress:
            local_pbar = tqdm(total=len(tasks), desc="Processing user-days", unit="task", dynamic_ncols=True)
        pbar = local_pbar
    else:
        pbar = progress

    try:
        for task_result in task_results:
            completed.append(task_result)
            if pbar is not None and hasattr(pbar, "update"):
                pbar.update(1)
    finally:
        if local_pbar is not None:
            local_pbar.close()
    return [result for _, result in sorted(completed, key=lambda item: item[0])]


def _validate_parquet_checkpoint(path: Path, min_rows: int = 0, required_columns: list[str] | None = None) -> bool:
    """Safely verify that a parquet checkpoint exists, is non-empty, and readable."""
    if not path.exists():
        return False
    try:
        if path.stat().st_size == 0:
            return False
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)
        if required_columns:
            for col in required_columns:
                if col not in schema.names:
                    return False
        meta = pq.read_metadata(path)
        if meta.num_rows < min_rows:
            return False
        return True
    except Exception:
        return False


def _data_identity(value: pd.DataFrame | str | Path | None) -> dict | None:
    """Return a stable, bounded-memory identity for checkpoint compatibility."""
    if value is None:
        return None
    if isinstance(value, (str, Path)):
        path = Path(value).resolve()
        stat = path.stat()
        return {
            "kind": "file", "path": str(path), "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    digest = hashlib.sha256()
    digest.update(json.dumps(
        [(str(column), str(dtype)) for column, dtype in value.dtypes.items()],
        separators=(",", ":"),
    ).encode("utf-8"))
    for start in range(0, len(value), 100_000):
        hashed = pd.util.hash_pandas_object(
            value.iloc[start:start + 100_000], index=False, categorize=True,
        )
        digest.update(hashed.to_numpy().tobytes())
    return {
        "kind": "dataframe", "rows": len(value), "columns": list(map(str, value.columns)),
        "sha256": digest.hexdigest(),
    }


def _path_identity(path: str | Path) -> dict:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "missing": True}
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _checkpoint_contract(
    preprocessed_gps, user_metadata, limit_users, limit_days_per_user, resources,
) -> tuple[dict, dict]:
    artifact = config.FILE_MODAL_HYBRID if config.MODAL_CLASSIFIER == "hybrid" else config.FILE_MODAL_RANDOM_FOREST
    artifact_identity = _path_identity(artifact)
    if Path(artifact).is_file():
        artifact_identity["sha256"] = hashlib.sha256(Path(artifact).read_bytes()).hexdigest()
    routing = {
        "input": _data_identity(preprocessed_gps),
        "metadata": _data_identity(user_metadata),
        "limit_users": limit_users,
        "limit_days_per_user": limit_days_per_user,
        "classifier": config.MODAL_CLASSIFIER,
        "classifier_artifact": artifact_identity,
        "min_effective_pings": random_forest_contract.MIN_EFFECTIVE_PINGS,
        "min_pct_conserved": random_forest_contract.MIN_PCT_CONSERVED,
        "metro_threshold": random_forest_contract.METRO_PROBABILITY_THRESHOLD,
        "bus_threshold": random_forest_contract.BUS_PROBABILITY_THRESHOLD,
        "router_version": config.ROUTER_VERSION,
        "max_lookahead_skipped_pings": config.MAX_LOOKAHEAD_SKIPPED_PINGS,
        # Custom in-memory graph resources are safe to resume only in the same process.
        "resources": "configured_files" if resources is None else f"in_memory:{id(resources)}",
    }
    emissions = {"routing": routing, "moves_rates": _path_identity(config.FILE_MOVES)}
    return routing, emissions


def _marker_matches(path: Path, contract: dict) -> bool:
    try:
        marker = json.loads(path.read_text(encoding="utf-8"))
        return marker.get("contract") == contract
    except Exception:
        return False


def _write_json_atomic(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def run_pipeline_v4(
    preprocessed_gps: pd.DataFrame | str | Path,
    output_dir: str | Path,
    user_metadata: pd.DataFrame | str | Path | None = None,
    *,
    n_jobs: int | None = None,
    limit_users: int | None = None,
    limit_days_per_user: int | None = None,
    resources: dict[str, Any] | None = None,
    output_mode: str = "summary",
    reuse_modal_evaluator: bool = True,
    show_progress: bool = True,
    user_day_batch_size: int | None = None,
    resume: bool = True,
    execution_profile: str | None = None,
    n_jobs_override: int | None = None,
    backend_override: str | None = None,
    batch_size_override: int | None = None,
    backend: str | None = None,
) -> PipelineV4Result:
    """Produce routes, individual emissions, and the canonical trip ledger."""
    # Resolve execution profile with strict priority: override > profile default > safe fallback
    profile_info = config.resolve_execution_profile(
        execution_profile,
        n_jobs_override=n_jobs_override if n_jobs_override is not None else n_jobs,
        backend_override=backend_override or backend,
        batch_size_override=batch_size_override if batch_size_override is not None else user_day_batch_size,
    )
    effective_n_jobs = profile_info["effective_n_jobs"]
    effective_backend = profile_info["backend"]
    effective_batch_size = profile_info["user_day_batch_size"]
    if effective_backend == "process" and resources is not None:
        raise ValueError(
            "Process execution cannot accept caller-provided in-memory resources; "
            "workers must load the configured production resources consistently."
        )

    print(f"[Pipeline] Execution profile: {profile_info['execution_profile']}", flush=True)
    print(f"[Pipeline] Parallel backend: {effective_backend}", flush=True)
    print(f"[Pipeline] Requested workers: {profile_info['requested_n_jobs']}", flush=True)
    print(f"[Pipeline] Effective workers: {effective_n_jobs}", flush=True)
    print(f"[Pipeline] User-day batch size: {effective_batch_size}", flush=True)
    if profile_info.get("guardrail_warning"):
        print(f"[Pipeline] WARNING: {profile_info['guardrail_warning']}", flush=True)

    output_mode = validate_output_mode(output_mode)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output / "checkpoints"
    stages_dir = output / "stages"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    stages_dir.mkdir(parents=True, exist_ok=True)

    stage_routing_marker = stages_dir / "stage_routing_modal.done"
    stage_emissions_marker = stages_dir / "stage_emissions.done"
    routes_ckpt_path = checkpoints_dir / "routed_trajectories.parquet"
    emissions_ckpt_path = checkpoints_dir / "emissions_results.parquet"
    ledger_ckpt_path = output / "trip_ledger.parquet"

    routing_contract, emissions_contract = _checkpoint_contract(
        preprocessed_gps, user_metadata, limit_users, limit_days_per_user, resources,
    )

    metadata = None if user_metadata is None else _read_frame(user_metadata)
    total_user_days = 0
    users_received_count = 0

    # -------------------------------------------------------------------------
    # STAGE 1: ROUTING AND MODAL INFERENCE (OR RESUME FROM CHECKPOINT)
    # -------------------------------------------------------------------------
    can_resume_routing = (
        resume
        and stage_routing_marker.exists()
        and _marker_matches(stage_routing_marker, routing_contract)
        and _validate_parquet_checkpoint(routes_ckpt_path, min_rows=0, required_columns=["modo_transporte"])
        and _validate_parquet_checkpoint(ledger_ckpt_path, min_rows=0, required_columns=["physical_trip_id", "processing_status"])
    )

    if can_resume_routing:
        print("[Pipeline] Resuming from checkpoint: Routing & modal inference already complete.", flush=True)
        routes = pd.read_parquet(routes_ckpt_path)
        ledger = _ensure_ledger_schema(pd.read_parquet(ledger_ckpt_path))
        try:
            r_meta = json.loads(stage_routing_marker.read_text(encoding="utf-8"))
            total_user_days = r_meta.get("user_days", 0)
            users_received_count = r_meta.get("users_received", 0)
        except Exception:
            pass
        batch_size = effective_batch_size
    else:
        print("[Pipeline] Loading preprocessed GPS...", flush=True)
        gps_cols = [
            "caid", "user_id", "local_timestamp", "latitude", "longitude",
            "lat_ruteo", "lon_ruteo", "Speed [km/h]", "dis lineal [m]", "trip", "travel time",
        ]
        gps = _read_frame(preprocessed_gps, columns=gps_cols)
        required = {"caid", "local_timestamp", "latitude", "longitude"}
        missing = sorted(required - set(gps.columns))
        if missing:
            raise ValueError(f"Preprocessed GPS is missing columns: {missing}")
        gps["local_timestamp"] = pd.to_datetime(gps.local_timestamp, errors="coerce")
        if gps.local_timestamp.isna().any():
            raise ValueError("Preprocessed GPS contains invalid local_timestamp values")
        gps["date"] = gps.local_timestamp.dt.date
        if metadata is not None:
            if "routing_eligible" in metadata.columns:
                eligible = metadata["routing_eligible"].fillna(False).astype(bool)
            else:
                eligible = metadata.processing_status.eq("ready_for_pipeline")
            ready = metadata.loc[eligible, "user_id"]
            gps = gps[gps.caid.isin(ready)].copy()
        users_received_count = int(gps.caid.nunique())
        print(
            f"[Pipeline] Loaded {len(gps):,} rows for {users_received_count:,} ready user(s).",
            flush=True,
        )
        users = gps.caid.drop_duplicates().tolist()
        if limit_users is not None:
            users = users[:limit_users]
        group_indices = gps.groupby(["caid", "date"], sort=False, observed=True).indices
        task_specs = []
        for user_id in users:
            days = sorted(gps.loc[gps.caid.eq(user_id), "date"].unique())
            if limit_days_per_user is not None:
                days = days[:limit_days_per_user]
            task_specs.extend((user_id, day) for day in days)
        total_user_days = len(task_specs)
        print(
            f"[Pipeline] Identified {total_user_days} user-day task(s) for {len(users)} user(s).",
            flush=True,
        )

        loaded_resources = None
        if effective_backend != "process":
            print("[Pipeline] Loading routing resources in parent process...", flush=True)
            loaded_resources = dict(resources or load_pipeline_resources())
            print("[Pipeline] Routing resources loaded.", flush=True)
            if reuse_modal_evaluator and "modal_evaluator" not in loaded_resources:
                loaded_resources["modal_evaluator"] = create_modal_evaluator(config.MODAL_CLASSIFIER)
            print("[Pipeline] Classifier resources loaded.", flush=True)

        batch_size = effective_batch_size
        if batch_size <= 0:
            raise ValueError("user_day_batch_size must be a positive integer")
        batch_count = (total_user_days + batch_size - 1) // batch_size
        print(
            f"[Pipeline] Processing {total_user_days} user-day task(s) in {batch_count} batch(es) "
            f"of at most {batch_size} with {effective_n_jobs} worker(s) ({effective_backend} backend)...",
            flush=True,
        )
        route_frames, ledger_frames = [], []
        progress_bar = (
            tqdm(
                total=total_user_days,
                desc="Processing user-days",
                unit="task",
                dynamic_ncols=True,
            )
            if show_progress and total_user_days > 0
            else None
        )
        try:
            if effective_backend == "process":
                print(f"[Pipeline] Starting persistent ProcessPoolExecutor with {effective_n_jobs} worker(s)...", flush=True)
                with ProcessPoolExecutor(max_workers=effective_n_jobs, initializer=_init_process_worker) as proc_executor:
                    for batch_number, start in enumerate(range(0, len(task_specs), batch_size), start=1):
                        window_specs = task_specs[start:start + batch_size]
                        window_tasks = [
                            (start + offset, user_id, gps.iloc[group_indices[(user_id, day)]].copy())
                            for offset, (user_id, day) in enumerate(window_specs)
                        ]
                        window_results = _run_user_day_tasks(
                            window_tasks, None, effective_n_jobs, backend="process",
                            progress=progress_bar, process_executor=proc_executor,
                        )
                        route_frames.extend(result.routes for result in window_results if not result.routes.empty)
                        ledger_frames.extend(result.trip_ledger for result in window_results if not result.trip_ledger.empty)
                        del window_results, window_tasks, window_specs
            else:
                for batch_number, start in enumerate(range(0, len(task_specs), batch_size), start=1):
                    window_specs = task_specs[start:start + batch_size]
                    window_tasks = [
                        (start + offset, user_id, gps.iloc[group_indices[(user_id, day)]].copy())
                        for offset, (user_id, day) in enumerate(window_specs)
                    ]
                    window_results = _run_user_day_tasks(
                        window_tasks, loaded_resources, effective_n_jobs, backend="threading",
                        progress=progress_bar,
                    )
                    route_frames.extend(result.routes for result in window_results if not result.routes.empty)
                    ledger_frames.extend(result.trip_ledger for result in window_results if not result.trip_ledger.empty)
                    del window_results, window_tasks, window_specs
        finally:
            if progress_bar is not None:
                progress_bar.close()
        del task_specs, group_indices, gps
        print("[Pipeline] Routing and modal inference complete.", flush=True)
        routes = pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame()
        ledger = _ensure_ledger_schema(
            pd.concat(ledger_frames, ignore_index=True) if ledger_frames else _empty_ledger()
        )
        del route_frames, ledger_frames
        if not ledger.empty and ledger["physical_trip_id"].duplicated().any():
            duplicated = ledger.loc[ledger["physical_trip_id"].duplicated(), "physical_trip_id"].tolist()
            raise RuntimeError(f"Trip ledger contains duplicate physical trips: {duplicated[:5]}")
        for column in ("start_node", "end_node", "osmid", "highway"):
            if column in routes.columns:
                routes[column] = routes[column].astype(str)
        if not routes.empty:
            routes["_output_row_id"] = range(len(routes))
        else:
            routes = pd.DataFrame(columns=["physical_trip_id", "modo_transporte", "_output_row_id"])
        if metadata is not None and not routes.empty:
            routes = attach_user_metadata(routes, metadata)

        # Checkpoint Stage 1 immediately
        write_parquet_atomic(routes, routes_ckpt_path)
        write_parquet_atomic(project_ledger(ledger), ledger_ckpt_path)
        _write_json_atomic(
            stage_routing_marker, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "route_rows": len(routes),
                "trips": len(ledger),
                "user_days": total_user_days,
                "users_received": users_received_count,
                "contract": routing_contract,
            },
        )
        print(f"[Pipeline] Checkpointed Stage 1: {len(routes):,} route rows, {len(ledger):,} trips.", flush=True)

    # -------------------------------------------------------------------------
    # STAGE 2: EMISSIONS CALCULATION (OR RESUME FROM CHECKPOINT)
    # -------------------------------------------------------------------------
    can_resume_emissions = (
        resume
        and can_resume_routing
        and stage_emissions_marker.exists()
        and _marker_matches(stage_emissions_marker, emissions_contract)
        and _validate_parquet_checkpoint(emissions_ckpt_path, min_rows=0, required_columns=["physical_trip_id"])
    )

    emission_error = None
    if can_resume_emissions:
        print("[Pipeline] Resuming from checkpoint: Emissions calculation already complete.", flush=True)
        emissions = pd.read_parquet(emissions_ckpt_path)
        ledger = _ensure_ledger_schema(pd.read_parquet(ledger_ckpt_path))
    else:
        eligible_ids = set(
            ledger.loc[ledger["emissions_usable"].fillna(False).astype(bool), "physical_trip_id"]
            .dropna().astype(str)
        ) if not ledger.empty else set()
        if not routes.empty and eligible_ids:
            valid_geometry = ~routes.get(
                "ruteo_fallido", pd.Series(False, index=routes.index)
            ).fillna(True).astype(bool)
            emission_routes = routes[
                routes.get("physical_trip_id", pd.Series(None, index=routes.index)).astype(str).isin(eligible_ids)
                & routes.get("modo_transporte", pd.Series(None, index=routes.index)).astype(str).isin(["Carro", "Bus"])
                & valid_geometry
            ].copy()
        else:
            emission_routes = pd.DataFrame(columns=routes.columns)
        try:
            if not emission_routes.empty:
                print("[Pipeline] Calculating emissions for eligible route segments...", flush=True)
            emissions = (
                calculate_emissions(emission_routes, config.FILE_MOVES)
                if not emission_routes.empty else pd.DataFrame()
            )
        except Exception as exc:
            emissions = pd.DataFrame()
            emission_error = f"{type(exc).__name__}: {exc}"
        del emission_routes

        if emissions.empty and "physical_trip_id" not in emissions.columns:
            emissions = pd.DataFrame(columns=["physical_trip_id", "_output_row_id"])

        if not ledger.empty:
            classified = ledger["classification_success"].astype(bool)
            emissions_eligible = ledger["emissions_usable"].fillna(False).astype(bool)
            if emission_error is None:
                emitted_ids = (
                    set(emissions["physical_trip_id"].dropna().astype(str))
                    if "physical_trip_id" in emissions and not emissions.empty else set()
                )
                emitted = ledger["physical_trip_id"].astype(str).isin(emitted_ids) & classified & emissions_eligible
                ledger.loc[emitted, ["processing_status", "emissions_success"]] = ["success", True]
                missing_emissions = classified & emissions_eligible & ~emitted
                ledger.loc[missing_emissions, "processing_status"] = "emissions_failed"
                ledger.loc[missing_emissions, "failure_reason"] = "no_emission_output_for_trip"
            else:
                ledger.loc[classified & emissions_eligible, "processing_status"] = "emissions_failed"
                ledger.loc[classified & emissions_eligible, "failure_reason"] = emission_error

        if metadata is not None and not emissions.empty and "home_ageb" not in emissions.columns:
            emissions = attach_user_metadata(emissions, metadata)

        # Checkpoint Stage 2 immediately
        write_parquet_atomic(emissions, emissions_ckpt_path)
        write_parquet_atomic(project_ledger(ledger), ledger_ckpt_path)
        _write_json_atomic(
            stage_emissions_marker, {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "emission_rows": len(emissions),
                "error": emission_error,
                "contract": emissions_contract,
            },
        )
        print(f"[Pipeline] Checkpointed Stage 2: {len(emissions):,} emission rows.", flush=True)

    # -------------------------------------------------------------------------
    # STAGE 3: STREAMING MEMORY-SAFE OUTPUT GENERATION & LEDGER PERSISTENCE
    # -------------------------------------------------------------------------
    print(f"[Pipeline] Writing {output_mode} output(s) and trip ledger...", flush=True)
    canonical_ledger = project_ledger(ledger)
    write_parquet_atomic(canonical_ledger, ledger_ckpt_path)

    output_artifacts = {
        "output_mode": output_mode,
        "summary_output": {"generated": False},
        "detailed_output": {"generated": False},
        "trip_ledger": {
            "generated": True,
            "filename": ledger_ckpt_path.name,
            "rows": int(len(canonical_ledger)),
            "columns": int(len(canonical_ledger.columns)),
            "size_mb": round(ledger_ckpt_path.stat().st_size / (1024 * 1024), 3) if ledger_ckpt_path.exists() else 0.0,
        },
    }

    detailed_error = None
    if output_mode in {"summary", "both"}:
        print("[Pipeline] Generating summary output...", flush=True)
        output_artifacts["summary_output"] = write_summary_output_streaming(
            emissions, output / "routes_emissions_summary.parquet", chunk_size=50000,
            show_progress=show_progress,
        )
    if output_mode in {"detailed", "both"}:
        print("[Pipeline] Streaming detailed output...", flush=True)
        try:
            output_artifacts["detailed_output"] = write_detailed_output_streaming(
                routes, emissions, canonical_ledger, output / "routes_emissions_detailed.parquet", chunk_size=50000,
                show_progress=show_progress,
            )
        except Exception as exc:
            detailed_error = f"{type(exc).__name__}: {exc}"
            output_artifacts["detailed_output"] = {
                "generated": False,
                "error": detailed_error,
            }
            print(f"[Pipeline] WARNING: Detailed output projection encountered an error: {detailed_error}", flush=True)

    manifest_status = "completed"
    if emission_error is not None:
        manifest_status = "emissions_failed"
    elif detailed_error is not None:
        manifest_status = "scientific_computation_complete_detailed_failed"
    elif int((canonical_ledger.processing_status != "success").sum()) > 0:
        manifest_status = "completed_with_trip_failures"
    output_artifacts["status"] = manifest_status

    print("[Pipeline] Finalizing manifest...", flush=True)
    manifest = {
        "stage": "pipeline_v4", "pipeline_release": config.PIPELINE_RELEASE,
        "status": manifest_status,
        "created_utc": datetime.now(timezone.utc).isoformat(), "classifier": config.MODAL_CLASSIFIER,
        "users_received": users_received_count, "user_days": total_user_days,
        "trips": int(len(canonical_ledger)), "successful_trips": int(canonical_ledger.processing_status.eq("success").sum()),
        "route_rows": len(routes), "emission_rows": len(emissions),
        "limits": {
            "users": limit_users,
            "days_per_user": limit_days_per_user,
            "user_day_batch_size": batch_size,
        },
        "output_mode": output_mode,
        **output_artifacts,
        "execution": profile_info,
        "classification": {
            "artifact": routing_contract["classifier_artifact"],
            "metro_threshold": random_forest_contract.METRO_PROBABILITY_THRESHOLD,
            "bus_threshold": random_forest_contract.BUS_PROBABILITY_THRESHOLD,
            "min_effective_pings": random_forest_contract.MIN_EFFECTIVE_PINGS,
            "min_pct_pings_conserved": random_forest_contract.MIN_PCT_CONSERVED,
            "min_pct_pings_conserved_percent": random_forest_contract.MIN_PCT_CONSERVED_PERCENT,
        },
        "moves_rate_unit_status": "confirmed",
        "moves_distance_rate_unit": config.EMISSION_RATE_DISTANCE_UNIT,
    }
    (output / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print("[Pipeline] Output writing complete.", flush=True)
    routes = routes.drop(columns="_output_row_id", errors="ignore")
    emissions = emissions.drop(columns="_output_row_id", errors="ignore")
    return PipelineV4Result(routes, emissions, canonical_ledger, output, output_artifacts)
