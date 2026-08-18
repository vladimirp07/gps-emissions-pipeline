"""Callable pipeline_v4 workflow for an externally supplied GPS sample.

This module preserves the validated segmentation/routing/emissions algorithms
and makes their production handoff explicit: GPS-level classifier context,
one ledger row per routed trip, and caller-provided output directories.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from typing import Any

import geopandas as gpd
from joblib import Parallel, delayed
import pandas as pd
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
)
from pipeline_v4.src.random_forest_contract import MIN_EFFECTIVE_PINGS, MIN_PCT_CONSERVED
from pipeline_v4.src.route_quality import QUALITY_COLUMNS, attach_route_quality, evaluate_route_quality
from pipeline_v4.src.routing import (
    RouteHypothesisEvaluator,
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
    "classification_success", *QUALITY_COLUMNS, "emissions_success",
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
        "emissions_eligible": legacy_eligible,
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
    # Candidate generation only needs point geometry; carrying all GPS columns into
    # two spatial joins caused the observed memory amplification.
    points = gpd.GeoDataFrame(
        index=effective_day.index,
        geometry=gpd.points_from_xy(effective_day[longitude], effective_day[latitude]),
        crs="EPSG:4326",
    ).to_crs(resources["edges_drive"].crs)
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
        if (context.effective_ping_count < MIN_EFFECTIVE_PINGS or
                context.pct_pings_conserved < MIN_PCT_CONSERVED):
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
        base.update(route_success=True, final_mode=final_mode, classification_success=True)
        if quality["emissions_eligible"]:
            base.update(processing_status="emissions_pending")
        else:
            base.update(
                processing_status="routing_failed",
                failure_reason="post_routing_quality_failed",
            )
        ledger.append(base)
    return DayProcessingResult(
        pd.concat(routed, ignore_index=True) if routed else pd.DataFrame(),
        pd.DataFrame(ledger, columns=LEDGER_COLUMNS) if ledger else _empty_ledger(),
    )


def _read_frame(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.read_parquet(value)


def _process_indexed_user_day(task_index, user_id, day, loaded_resources):
    """Preserve task order after collecting asynchronously completed work."""
    return task_index, process_user_day(user_id, day, loaded_resources)


def _run_user_day_tasks(tasks, loaded_resources, n_jobs: int, show_progress: bool):
    """Run user-day tasks with completion-based progress and deterministic output order."""
    parallel = Parallel(n_jobs=n_jobs, backend="threading", return_as="generator_unordered")
    task_results = parallel(
        delayed(_process_indexed_user_day)(task_index, user_id, day, loaded_resources)
        for task_index, (user_id, day) in enumerate(tasks)
    )
    completed = []
    progress = (
        tqdm(total=len(tasks), desc="Routing user-days", unit="task", dynamic_ncols=True)
        if show_progress
        else None
    )
    try:
        for task_result in task_results:
            completed.append(task_result)
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return [result for _, result in sorted(completed, key=lambda item: item[0])]


def run_pipeline_v4(
    preprocessed_gps: pd.DataFrame | str | Path,
    output_dir: str | Path,
    user_metadata: pd.DataFrame | str | Path | None = None,
    *,
    n_jobs: int = 2,
    limit_users: int | None = None,
    limit_days_per_user: int | None = None,
    resources: dict[str, Any] | None = None,
    output_mode: str = "summary",
    reuse_modal_evaluator: bool = True,
    show_progress: bool = True,
) -> PipelineV4Result:
    """Produce routes, individual emissions, and the canonical trip ledger."""
    output_mode = validate_output_mode(output_mode)
    print("[Pipeline] Reading preprocessed GPS input...", flush=True)
    gps = _read_frame(preprocessed_gps)
    required = {"caid", "local_timestamp", "latitude", "longitude"}
    missing = sorted(required - set(gps.columns))
    if missing:
        raise ValueError(f"Preprocessed GPS is missing columns: {missing}")
    gps["local_timestamp"] = pd.to_datetime(gps.local_timestamp, errors="coerce")
    if gps.local_timestamp.isna().any():
        raise ValueError("Preprocessed GPS contains invalid local_timestamp values")
    gps["date"] = gps.local_timestamp.dt.date
    metadata = None if user_metadata is None else _read_frame(user_metadata)
    if metadata is not None:
        if "routing_eligible" in metadata.columns:
            eligible = metadata["routing_eligible"].fillna(False).astype(bool)
        else:
            # Backward-compatible handoff for metadata produced before routing
            # and residence quality were explicitly decoupled.
            eligible = metadata.processing_status.eq("ready_for_pipeline")
        ready = metadata.loc[eligible, "user_id"]
        gps = gps[gps.caid.isin(ready)].copy()
    users = gps.caid.drop_duplicates().tolist()
    if limit_users is not None:
        users = users[:limit_users]
    tasks = []
    for user_id in users:
        user = gps[gps.caid.eq(user_id)]
        days = sorted(user.date.unique())
        if limit_days_per_user is not None:
            days = days[:limit_days_per_user]
        tasks.extend((user_id, user[user.date.eq(day)].copy()) for day in days)
    print(
        f"[Pipeline] Prepared {len(tasks)} user-day task(s) for {len(users)} user(s).",
        flush=True,
    )
    print("[Pipeline] Loading routing networks and modal inference resources...", flush=True)
    loaded_resources = dict(resources or load_pipeline_resources())
    if reuse_modal_evaluator and "modal_evaluator" not in loaded_resources:
        # The serving evaluator is read-only during inference. Loading it once
        # per run removes repeated model deserialization without changing calls.
        loaded_resources["modal_evaluator"] = create_modal_evaluator(config.MODAL_CLASSIFIER)
    print(f"[Pipeline] Processing routing and modal inference with {n_jobs} worker(s)...", flush=True)
    results = _run_user_day_tasks(tasks, loaded_resources, n_jobs, show_progress)
    print("[Pipeline] Routing and modal inference complete.", flush=True)
    route_frames = [result.routes for result in results if not result.routes.empty]
    ledger_frames = [result.trip_ledger for result in results if not result.trip_ledger.empty]
    routes = pd.concat(route_frames, ignore_index=True) if route_frames else pd.DataFrame()
    ledger = _ensure_ledger_schema(
        pd.concat(ledger_frames, ignore_index=True) if ledger_frames else _empty_ledger()
    )
    if not ledger.empty and ledger["physical_trip_id"].duplicated().any():
        duplicated = ledger.loc[ledger["physical_trip_id"].duplicated(), "physical_trip_id"].tolist()
        raise RuntimeError(f"Trip ledger contains duplicate physical trips: {duplicated[:5]}")
    for column in ("start_node", "end_node", "osmid", "highway"):
        if column in routes.columns:
            routes[column] = routes[column].astype(str)
    if not routes.empty:
        routes["_output_row_id"] = range(len(routes))

    emission_error = None
    eligible_ids = set(
        ledger.loc[ledger["emissions_eligible"].fillna(False).astype(bool), "physical_trip_id"]
        .dropna().astype(str)
    ) if not ledger.empty else set()
    if not routes.empty and eligible_ids:
        valid_geometry = ~routes.get(
            "ruteo_fallido", pd.Series(False, index=routes.index)
        ).fillna(True).astype(bool)
        emission_routes = routes[
            routes.get("physical_trip_id", pd.Series(None, index=routes.index)).astype(str).isin(eligible_ids)
            & valid_geometry
        ].copy()
    else:
        emission_routes = pd.DataFrame(columns=routes.columns)
    try:
        if not emission_routes.empty:
            print("[Pipeline] Calculating emissions for eligible route segments...", flush=True)
        emissions = (
            calculate_emissions(emission_routes, config.FILE_MOVES)
            if not emission_routes.empty else emission_routes.copy()
        )
    except Exception as exc:
        emissions = pd.DataFrame()
        emission_error = f"{type(exc).__name__}: {exc}"
    if not ledger.empty:
        classified = ledger["classification_success"].astype(bool)
        eligible = ledger["emissions_eligible"].fillna(False).astype(bool)
        if emission_error is None:
            emitted_ids = (
                set(emissions["physical_trip_id"].dropna().astype(str))
                if "physical_trip_id" in emissions else set(routes.get("physical_trip_id", []))
            )
            emitted = ledger["physical_trip_id"].astype(str).isin(emitted_ids) & classified & eligible
            ledger.loc[emitted, ["processing_status", "emissions_success"]] = ["success", True]
            missing_emissions = classified & eligible & ~emitted
            ledger.loc[missing_emissions, "processing_status"] = "emissions_failed"
            ledger.loc[missing_emissions, "failure_reason"] = "no_emission_output_for_trip"
        else:
            ledger.loc[classified & eligible, "processing_status"] = "emissions_failed"
            ledger.loc[classified & eligible, "failure_reason"] = emission_error

    if metadata is not None and not routes.empty:
        routes = attach_user_metadata(routes, metadata)
        if not emissions.empty:
            emissions = attach_user_metadata(emissions, metadata)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    print(f"[Pipeline] Writing {output_mode} output(s) and trip ledger...", flush=True)
    canonical_ledger = project_ledger(ledger)
    output_artifacts = {
        "output_mode": output_mode,
        "summary_output": {"generated": False},
        "detailed_output": {"generated": False},
    }
    if output_mode in {"summary", "both"}:
        output_artifacts["summary_output"] = write_output_artifact(
            project_summary(emissions), output / "routes_emissions_summary.parquet"
        )
    if output_mode in {"detailed", "both"}:
        output_artifacts["detailed_output"] = write_output_artifact(
            project_detailed(routes, emissions, canonical_ledger),
            output / "routes_emissions_detailed.parquet"
        )
    output_artifacts["trip_ledger"] = write_output_artifact(
        canonical_ledger, output / "trip_ledger.parquet"
    )
    manifest = {
        "stage": "pipeline_v4", "pipeline_release": config.PIPELINE_RELEASE,
        "created_utc": datetime.now(timezone.utc).isoformat(), "classifier": config.MODAL_CLASSIFIER,
        "users_received": int(gps.caid.nunique()), "user_days": len(tasks),
        "trips": int(len(ledger)), "successful_trips": int(ledger.processing_status.eq("success").sum()),
        "route_rows": len(routes), "emission_rows": len(emissions),
        "limits": {"users": limit_users, "days_per_user": limit_days_per_user},
        "output_mode": output_mode,
        **output_artifacts,
        "moves_rate_unit_status": "pending_external_confirmation",
    }
    (output / "pipeline_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print("[Pipeline] Output writing complete.", flush=True)
    routes = routes.drop(columns="_output_row_id", errors="ignore")
    emissions = emissions.drop(columns="_output_row_id", errors="ignore")
    return PipelineV4Result(routes, emissions, canonical_ledger, output, output_artifacts)
