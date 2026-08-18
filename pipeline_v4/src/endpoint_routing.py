"""Conservative endpoint preservation used by production router V2.

The GPS-to-network gap is always reported as uncovered.  Only a substring of
an actual network edge may be appended to a route; no straight connector from
the GPS observation to the road/walk network is ever synthesized.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import substring


ENDPOINT_PATCH_VERSION = "endpoint_real_edge_v2_2026-08-18"
SUPPORTED_MODES = {"carro", "bus", "road", "caminar", "walking", "walk"}


@dataclass(frozen=True)
class IncidentEdgeIndex:
    """Compact native-node lookup preserving source edge ordering."""

    nodes: np.ndarray
    offsets: np.ndarray
    positions: np.ndarray
    edge_count: int

    def positions_for(self, node) -> np.ndarray:
        location = int(np.searchsorted(self.nodes, node))
        if location >= len(self.nodes) or self.nodes[location] != node:
            return np.empty(0, dtype=self.positions.dtype)
        return self.positions[self.offsets[location]:self.offsets[location + 1]]


def build_incident_edge_index(edges: gpd.GeoDataFrame) -> IncidentEdgeIndex:
    """Index both endpoints once without changing legacy source-row order."""
    edge_count = len(edges)
    position_dtype = np.int32 if edge_count <= np.iinfo(np.int32).max else np.int64
    source_positions = np.arange(edge_count, dtype=position_dtype)
    nodes = np.concatenate((edges["u"].to_numpy(copy=False), edges["v"].to_numpy(copy=False)))
    positions = np.concatenate((source_positions, source_positions))
    order = np.lexsort((positions, nodes))
    nodes = nodes[order]
    positions = positions[order]
    # A self-loop satisfies both sides of the legacy OR mask but appears once.
    if len(nodes):
        keep = np.ones(len(nodes), dtype=bool)
        keep[1:] = (nodes[1:] != nodes[:-1]) | (positions[1:] != positions[:-1])
        nodes = nodes[keep]
        positions = positions[keep]
    unique_nodes, starts = np.unique(nodes, return_index=True)
    offsets = np.append(starts, len(nodes)).astype(np.int64, copy=False)
    return IncidentEdgeIndex(unique_nodes, offsets, positions, edge_count)


def normalized_mode(frame: pd.DataFrame) -> str:
    if frame.empty or "modo_transporte" not in frame:
        return ""
    return str(frame["modo_transporte"].iloc[0]).strip().lower()


def expand_endpoint_candidates(
    frame: pd.DataFrame,
    edges: gpd.GeoDataFrame,
    get_candidates,
    *,
    walking: bool,
    candidate_edges=None,
) -> tuple[pd.DataFrame, dict]:
    """Expand candidates at the first and last ping only."""
    if frame.empty:
        return frame.copy(), {"applied": False, "reason": "empty_trip"}
    result = frame.copy(deep=True)
    id_column = "walk_ids" if walking else "drive_ids"
    distance_column = "walk_dists" if walking else "drive_dists"
    radius_m = 100 if walking else 250
    positions = [0] if len(result) == 1 else [0, len(result) - 1]
    endpoint_rows = result.iloc[positions]
    points = gpd.GeoDataFrame(
        index=endpoint_rows.index,
        geometry=gpd.points_from_xy(endpoint_rows["longitude"], endpoint_rows["latitude"]),
        crs="EPSG:4326",
    ).to_crs(edges.crs)
    ids, distances = get_candidates(
        candidate_edges if candidate_edges is not None else edges,
        points,
        buffer_m=radius_m,
        max_cands=24,
    )
    for position in positions:
        row = result.iloc[position]
        result.at[row.name, id_column] = ids.loc[row.name]
        result.at[row.name, distance_column] = distances.loc[row.name]
    return result, {
        "applied": True,
        "endpoint_radius_m": radius_m,
        "endpoint_max_candidates": 24,
        "interior_candidates_unchanged": True,
        "_projected_endpoint_points": points,
    }


def _point_utm(frame: pd.DataFrame, position: int, crs):
    return gpd.GeoSeries(
        [Point(float(frame.longitude.iloc[position]), float(frame.latitude.iloc[position]))],
        crs="EPSG:4326",
    ).to_crs(crs).iloc[0]


def _endpoint_segment(edge, node, point, side: str):
    line = edge.geometry
    projected = float(line.project(point))
    node_is_u = str(edge.u) == str(node)
    segment = substring(line, 0, projected) if node_is_u else substring(line, projected, line.length)
    if segment.is_empty or segment.length <= 0 or segment.geom_type != "LineString":
        return None
    # Orient start segment toward the existing route and end segment away from it.
    if (side == "start" and node_is_u) or (side == "end" and not node_is_u):
        segment = LineString(list(segment.coords)[::-1])
    return segment


def attach_real_edge_endpoint_segments(
    route: pd.DataFrame,
    input_frame: pd.DataFrame,
    edges: gpd.GeoDataFrame,
    *,
    max_snap_m: float = 100.0,
    incident_edge_index: IncidentEdgeIndex | None = None,
    projected_endpoint_points: gpd.GeoDataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Append real-edge substrings and explicitly record uncovered endpoints."""
    result = route.copy()
    meta = {"segments_added": 0, "max_snap_m": float(max_snap_m)}
    if result.empty:
        return result, meta
    failed = result.get("ruteo_fallido", pd.Series(False, index=result.index)).fillna(True).astype(bool)
    valid = result[
        ~failed
        & result.start_node.astype(str).ne("N/A")
        & result.end_node.astype(str).ne("N/A")
    ]
    if valid.empty:
        result["endpoint_start_status"] = "uncovered_no_valid_route"
        result["endpoint_end_status"] = "uncovered_no_valid_route"
        return result, meta

    additions = []
    statuses = {}
    for side, node, position in (
        ("start", valid.iloc[0].start_node, 0),
        ("end", valid.iloc[-1].end_node, len(input_frame) - 1),
    ):
        if incident_edge_index is None:
            # Compatibility path for external callers without loaded resources.
            incident = edges[(edges.u.astype(str) == str(node)) | (edges.v.astype(str) == str(node))]
        else:
            if incident_edge_index.edge_count != len(edges):
                raise ValueError("Incident-edge index does not match the supplied edge table")
            incident = edges.iloc[incident_edge_index.positions_for(node)]
        if incident.empty:
            statuses[side] = "uncovered_no_incident_edge"
            meta[f"{side}_uncovered_m"] = None
            continue
        if projected_endpoint_points is None:
            point = _point_utm(input_frame, position, edges.crs)
        else:
            point = projected_endpoint_points.geometry.iloc[0 if side == "start" else -1]
        edge_index = incident.geometry.distance(point).idxmin()
        edge = incident.loc[edge_index]
        snap_m = float(edge.geometry.distance(point))
        meta[f"{side}_uncovered_m"] = snap_m
        if snap_m > max_snap_m:
            statuses[side] = "uncovered_rejected_distance"
            continue
        segment = _endpoint_segment(edge, node, point, side)
        if segment is None:
            statuses[side] = "uncovered_rejected_geometry"
            continue
        geometry_wgs = gpd.GeoSeries([segment], crs=edges.crs).to_crs("EPSG:4326").iloc[0]
        template = (valid.iloc[0] if side == "start" else valid.iloc[-1]).copy()
        template["geometry"] = geometry_wgs.wkt
        template["distance_m"] = float(segment.length)
        template["Speed [km/h]"] = 0.0
        template["ruteo_fallido"] = False
        template["flag_auditoria"] = f"Endpoint_Partial_Snap_{side}"
        template["local_timestamp"] = input_frame.local_timestamp.iloc[position]
        template["latitude"] = input_frame.latitude.iloc[position]
        template["longitude"] = input_frame.longitude.iloc[position]
        template["start_node"] = "PartialSnap_START" if side == "start" else node
        template["end_node"] = node if side == "start" else "PartialSnap_END"
        template["endpoint_role"] = side
        template["endpoint_uncovered_m"] = snap_m
        template["endpoint_edge_u"] = edge.u
        template["endpoint_edge_v"] = edge.v
        template["snap_distance_m"] = snap_m
        template["snapping_quality_status"] = "real_edge_partial_snap"
        template["router_version"] = "v2"
        additions.append((side, template))
        statuses[side] = "represented_real_edge_uncovered_gap_recorded"
        meta["segments_added"] += 1

    for side, row in additions:
        addition = pd.DataFrame([row])
        result = pd.concat([addition, result], ignore_index=True) if side == "start" else pd.concat(
            [result, addition], ignore_index=True
        )
    for side in ("start", "end"):
        result[f"endpoint_{side}_status"] = statuses.get(side, "uncovered_not_evaluated")
        result[f"endpoint_{side}_uncovered_m"] = meta.get(f"{side}_uncovered_m")
    return result, meta


def explicitly_reject_failed_geometry(route: pd.DataFrame) -> pd.DataFrame:
    """Prevent failed fallback geometry from looking like a valid route."""
    result = route.copy()
    if result.empty:
        return result
    failed = result.get("ruteo_fallido", pd.Series(False, index=result.index)).fillna(False).astype(bool)
    result.loc[failed, "geometry"] = None
    result.loc[failed, "distance_m"] = 0.0
    result.loc[failed, "flag_auditoria"] = "Explicit_Rejection_Unrecoverable"
    return result
