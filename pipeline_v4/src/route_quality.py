"""Production post-routing quality contract for pipeline_v4.

The contract evaluates reconstructed geometry after modal selection.  It does
not alter routing, classification, or MOVES rates.  Failed geometry remains in
the route output for audit but is never emissions-eligible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from shapely import wkt

from .segmentation import haversine_vectorized


FAILED_MAX_CONTINUITY_GAP_M = 1000.0
FAILED_MIN_ROUTE_GPS_RATIO = 0.25
FAILED_MAX_ROUTE_GPS_RATIO = 4.0
FAILED_MAX_UNCOVERED_FRACTION = 0.50

PARTIAL_MAX_CONTINUITY_GAP_M = 100.0
PARTIAL_MIN_ROUTE_GPS_RATIO = 0.50
PARTIAL_MAX_ROUTE_GPS_RATIO = 2.0
PARTIAL_MAX_UNCOVERED_FRACTION = 0.20
PARTIAL_MAX_FAILED_ROW_FRACTION = 0.20


QUALITY_COLUMNS = (
    "route_component_count", "failed_row_count", "failed_row_fraction",
    "failure_cluster_count", "max_continuity_gap_m",
    "total_uncovered_distance_m", "uncovered_fraction",
    "route_gps_ratio", "endpoint_start_error_m", "endpoint_end_error_m",
    "unresolved_transition_count", "reconstructed_distance_m",
    "gps_distance_m", "route_coverage_fraction",
    "route_completeness_status", "emissions_eligible",
)


def _geometry(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return wkt.loads(value) if isinstance(value, str) else value
    except Exception:
        return None


def _endpoints(geometry):
    if geometry is None or geometry.is_empty:
        return None, None
    if hasattr(geometry, "geoms"):
        parts = [part for part in geometry.geoms if hasattr(part, "coords") and len(part.coords)]
        if not parts:
            return None, None
        return parts[0].coords[0], parts[-1].coords[-1]
    if hasattr(geometry, "coords") and len(geometry.coords):
        return geometry.coords[0], geometry.coords[-1]
    return None, None


def _distance_m(left, right):
    return float(haversine_vectorized(left[1], left[0], right[1], right[0]) * 1000.0)


def observed_gps_distance_m(trip: pd.DataFrame) -> float:
    if trip is None or len(trip) < 2:
        return 0.0
    latitude = "lat_ruteo" if "lat_ruteo" in trip else "latitude"
    longitude = "lon_ruteo" if "lon_ruteo" in trip else "longitude"
    lat = pd.to_numeric(trip[latitude], errors="coerce").to_numpy()
    lon = pd.to_numeric(trip[longitude], errors="coerce").to_numpy()
    return float(sum(
        haversine_vectorized(lat[idx], lon[idx], lat[idx + 1], lon[idx + 1]) * 1000.0
        for idx in range(len(trip) - 1)
        if np.isfinite([lat[idx], lon[idx], lat[idx + 1], lon[idx + 1]]).all()
    ))


def classify_route_completeness(metrics: dict | pd.Series) -> str:
    """Classify using the reviewed diagnostic thresholds plus explicit splits."""
    get = metrics.get
    components = int(get("route_component_count", 0) or 0)
    reconstructed = float(get("reconstructed_distance_m", 0.0) or 0.0)
    ratio = float(get("route_gps_ratio", np.nan))
    max_gap = float(get("max_continuity_gap_m", 0.0) or 0.0)
    uncovered = float(get("uncovered_fraction", 0.0) or 0.0)
    failed_fraction = float(get("failed_row_fraction", 0.0) or 0.0)
    unresolved = int(get("unresolved_transition_count", 0) or 0)

    if components == 0 or reconstructed <= 0:
        return "failed"
    if (
        max_gap > FAILED_MAX_CONTINUITY_GAP_M
        or (np.isfinite(ratio) and (ratio < FAILED_MIN_ROUTE_GPS_RATIO or ratio > FAILED_MAX_ROUTE_GPS_RATIO))
        or uncovered > FAILED_MAX_UNCOVERED_FRACTION
    ):
        return "failed"
    if (
        unresolved > 0
        or max_gap > PARTIAL_MAX_CONTINUITY_GAP_M
        or (np.isfinite(ratio) and (ratio < PARTIAL_MIN_ROUTE_GPS_RATIO or ratio > PARTIAL_MAX_ROUTE_GPS_RATIO))
        or uncovered > PARTIAL_MAX_UNCOVERED_FRACTION
        or failed_fraction > PARTIAL_MAX_FAILED_ROW_FRACTION
    ):
        return "partial"
    return "complete"


def evaluate_route_quality(route: pd.DataFrame, trip: pd.DataFrame) -> dict:
    """Return one canonical quality record for one physical GPS trip."""
    if route is None or route.empty:
        result = {column: 0 for column in QUALITY_COLUMNS}
        result.update({
            "route_gps_ratio": np.nan,
            "uncovered_fraction": 1.0,
            "route_coverage_fraction": 0.0,
            "route_completeness_status": "failed",
            "emissions_eligible": False,
        })
        return result

    frame = route.reset_index(drop=True).copy()
    failed = frame.get("ruteo_fallido", pd.Series(False, index=frame.index)).fillna(True).astype(bool)
    geometries = [_geometry(value) for value in frame.get("geometry", pd.Series(None, index=frame.index))]
    components = pd.to_numeric(
        frame.get("route_component_id", pd.Series(1, index=frame.index)), errors="coerce"
    ).fillna(1).astype(int)
    valid = [
        (idx, geometries[idx], int(components.iloc[idx]))
        for idx in range(len(frame))
        if not failed.iloc[idx] and geometries[idx] is not None and not geometries[idx].is_empty
    ]

    geometry_gaps = []
    legacy_uncovered = []
    for (left_idx, left, left_component), (right_idx, right, right_component) in zip(valid[:-1], valid[1:]):
        _, left_end = _endpoints(left)
        right_start, _ = _endpoints(right)
        if left_end is None or right_start is None:
            continue
        gap = _distance_m(left_end, right_start)
        separated = right_idx > left_idx + 1 or left_component != right_component or gap > 30.0
        if separated:
            geometry_gaps.append(gap)
            between = frame.iloc[left_idx + 1:right_idx]
            has_explicit = between.get(
                "routing_event", pd.Series(None, index=between.index)
            ).eq("transition_unresolved").any()
            if not has_explicit:
                legacy_uncovered.append(gap)

    routing_event = frame.get("routing_event", pd.Series(None, index=frame.index))
    unresolved_mask = routing_event.eq("transition_unresolved")
    explicit_uncovered = pd.to_numeric(
        frame.get("lookahead_observed_distance_m", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).where(unresolved_mask, 0.0).fillna(0.0)
    explicit_distances = explicit_uncovered[explicit_uncovered.gt(0)].tolist()

    failure_clusters = int((failed & ~failed.shift(fill_value=False)).sum())
    valid_component_ids = {component for _, _, component in valid}
    route_m = float(pd.to_numeric(frame.get("distance_m", 0.0), errors="coerce").where(~failed, 0.0).fillna(0.0).sum())
    gps_m = observed_gps_distance_m(trip)

    def endpoint_error(column):
        values = pd.to_numeric(frame.get(column, pd.Series(0.0, index=frame.index)), errors="coerce").dropna()
        return float(values.max()) if len(values) else 0.0

    start_error = endpoint_error("endpoint_start_uncovered_m")
    end_error = endpoint_error("endpoint_end_uncovered_m")
    total_uncovered = float(start_error + end_error + sum(explicit_distances) + sum(legacy_uncovered))
    max_gap = max(geometry_gaps + explicit_distances, default=0.0)
    uncovered_fraction = total_uncovered / gps_m if gps_m > 0 else 1.0
    ratio = route_m / gps_m if gps_m > 0 else np.nan
    result = {
        "route_component_count": len(valid_component_ids),
        "failed_row_count": int(failed.sum()),
        "failed_row_fraction": float(failed.mean()),
        "failure_cluster_count": failure_clusters,
        "max_continuity_gap_m": float(max_gap),
        "total_uncovered_distance_m": total_uncovered,
        "uncovered_fraction": float(uncovered_fraction),
        "route_gps_ratio": float(ratio),
        "endpoint_start_error_m": start_error,
        "endpoint_end_error_m": end_error,
        "unresolved_transition_count": int(unresolved_mask.sum()),
        "reconstructed_distance_m": route_m,
        "gps_distance_m": gps_m,
        "route_coverage_fraction": float(np.clip(1.0 - uncovered_fraction, 0.0, 1.0)),
    }
    status = classify_route_completeness(result)
    result["route_completeness_status"] = status
    result["emissions_eligible"] = bool(status in {"complete", "partial"} and route_m > 0)
    return result


def attach_route_quality(route: pd.DataFrame, metrics: dict) -> pd.DataFrame:
    result = route.copy()
    for column in QUALITY_COLUMNS:
        result[column] = metrics.get(column)
    status = metrics["route_completeness_status"]
    result["emissions_scope"] = np.where(
        status == "complete", "complete_trip",
        "reconstructed_components_only" if status == "partial" else "not_eligible",
    )
    return result
