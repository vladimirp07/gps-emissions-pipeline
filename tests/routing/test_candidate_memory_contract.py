import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from pipeline_v4.src.routing import build_candidate_edge_index, get_candidates_vectorized


def _legacy_candidates(edges, points, buffer_m, max_cands=12):
    """Reference implementation before the memory-only column restriction."""
    joined = gpd.sjoin_nearest(
        points, edges, how="left", max_distance=buffer_m, distance_col="dist_exacta"
    ).dropna(subset=["index_right"])
    joined = joined.sort_values("dist_exacta").groupby(level=0).head(max_cands)

    def format_group(group):
        ids, distances = [], []
        for _, row in group.iterrows():
            ids.extend([row["u"], row["v"]])
            distances.extend([row["dist_exacta"], row["dist_exacta"]])
        return pd.Series({"ids": ids, "dists": distances})

    result = joined.groupby(level=0).apply(format_group).reindex(points.index)
    result["ids"] = result["ids"].apply(lambda value: value if isinstance(value, list) else [])
    result["dists"] = result["dists"].apply(lambda value: value if isinstance(value, list) else [])
    return result["ids"], result["dists"]


def test_candidate_ids_and_distances_are_unchanged_by_column_restriction():
    edges = gpd.GeoDataFrame(
        {
            "u": [1, 2], "v": [2, 3],
            "unused_text": ["x" * 1000, "y" * 1000],
            "unused_number": [10, 20],
        },
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(10, 0), (20, 0)])],
        crs="EPSG:32614",
    )
    points = gpd.GeoDataFrame(
        {"unused_gps_payload": ["z" * 1000, "q" * 1000]},
        geometry=[Point(2, 1), Point(18, 1)], crs=edges.crs,
    )
    legacy_ids, legacy_distances = _legacy_candidates(edges, points, buffer_m=5)
    optimized_ids, optimized_distances = get_candidates_vectorized(edges, points, buffer_m=5)
    assert legacy_ids.tolist() == optimized_ids.tolist()
    assert legacy_distances.tolist() == optimized_distances.tolist()


def test_persistent_candidate_index_is_exact_and_preserves_ties_and_native_ids():
    edges = gpd.GeoDataFrame(
        {
            "u": pd.Series([101, 201, 301], dtype="int64"),
            "v": pd.Series([102, 202, 302], dtype="int64"),
            "unused": ["a", "b", "c"],
        },
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(0, 2), (10, 2)]),
            LineString([(20, 0), (30, 0)]),
        ],
        crs="EPSG:32614",
    )
    points = gpd.GeoDataFrame(
        geometry=[Point(5, 1), Point(25, 1)], crs=edges.crs,
        index=pd.Index([7, 3]),
    )
    current_ids, current_distances = get_candidates_vectorized(edges, points, buffer_m=5)
    persistent = build_candidate_edge_index(edges, resource_key="test-network-v1")
    indexed_ids, indexed_distances = get_candidates_vectorized(persistent, points, buffer_m=5)

    assert persistent.resource_key == "test-network-v1"
    assert persistent.source_positions.tolist() == [0, 1, 2]
    assert current_ids.tolist() == indexed_ids.tolist()
    assert current_distances.tolist() == indexed_distances.tolist()
    assert indexed_ids.loc[7] == [101, 102, 201, 202]
    assert all(type(value) is int for values in indexed_ids for value in values)


def test_persistent_candidate_index_reuses_the_same_spatial_index_object():
    edges = gpd.GeoDataFrame(
        {"u": [1], "v": [2]},
        geometry=[LineString([(0, 0), (10, 0)])], crs="EPSG:32614",
    )
    points = gpd.GeoDataFrame(geometry=[Point(2, 1)], crs=edges.crs)
    persistent = build_candidate_edge_index(edges, resource_key="drive")
    original_index = persistent.spatial_index

    get_candidates_vectorized(persistent, points, buffer_m=5)
    get_candidates_vectorized(persistent, points, buffer_m=5)

    assert persistent.edges.sindex is original_index
