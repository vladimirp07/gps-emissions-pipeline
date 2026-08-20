import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely

from pipeline_v4.src.routing import get_candidates_vectorized


def _legacy_candidates(edges_gdf, gdf_points, buffer_m=150, max_cands=12):
    candidate_edges = edges_gdf[["u", "v", edges_gdf.geometry.name]]
    candidate_points = gdf_points[[gdf_points.geometry.name]]
    joined = gpd.sjoin_nearest(
        candidate_points, candidate_edges, how="left", max_distance=buffer_m, distance_col="dist_exacta"
    )
    if "index_right" not in joined.columns or joined["index_right"].isna().all():
        empty_series = pd.Series([[] for _ in range(len(gdf_points))], index=gdf_points.index)
        return empty_series, empty_series
    joined = joined.dropna(subset=["index_right"])
    if joined.empty:
        empty_series = pd.Series([[] for _ in range(len(gdf_points))], index=gdf_points.index)
        return empty_series, empty_series
    joined = joined.sort_values("dist_exacta")
    joined = joined.groupby(level=0).head(max_cands)
    point_index = joined.index.to_numpy().repeat(2)
    candidate_ids = joined[["u", "v"]].to_numpy(copy=False).reshape(-1).tolist()
    candidate_distances = joined["dist_exacta"].to_numpy(copy=False).repeat(2).tolist()
    ids_grouped = pd.Series(candidate_ids, index=point_index, dtype=object).groupby(level=0, sort=False).agg(list)
    distances_grouped = pd.Series(candidate_distances, index=point_index, dtype=object).groupby(level=0, sort=False).agg(list)
    ids = ids_grouped.reindex(gdf_points.index)
    distances = distances_grouped.reindex(gdf_points.index)
    missing_ids = ids.isna()
    missing_distances = distances.isna()
    if missing_ids.any():
        ids.loc[missing_ids] = pd.Series([[] for _ in range(int(missing_ids.sum()))], index=ids.index[missing_ids], dtype=object)
    if missing_distances.any():
        distances.loc[missing_distances] = pd.Series([[] for _ in range(int(missing_distances.sum()))], index=distances.index[missing_distances], dtype=object)
    return ids, distances


def test_candidate_vectorized_matches_legacy_across_edge_cases():
    line1 = shapely.linestrings([(0, 0), (10, 0)])
    line2 = shapely.linestrings([(0, 1), (10, 1)])
    edges = gpd.GeoDataFrame(
        {"u": [1, 2, 1], "v": [2, 3, 2], "geometry": [line1, line2, line1]},
        crs="EPSG:32614",
    )

    test_cases = [
        # 1. Unique index
        gpd.GeoDataFrame({"geometry": [shapely.Point(1, 0.1), shapely.Point(2, 0.2)]}, index=[10, 20], crs="EPSG:32614"),
        # 2. Duplicate index
        gpd.GeoDataFrame({"geometry": [shapely.Point(1, 0.1), shapely.Point(2, 0.2)]}, index=[11, 11], crs="EPSG:32614"),
        # 3. Non-contiguous index
        gpd.GeoDataFrame({"geometry": [shapely.Point(1, 0.1), shapely.Point(2, 0.2), shapely.Point(3, 0.3)]}, index=[100, 5, 50], crs="EPSG:32614"),
        # 4. Repeated coordinates
        gpd.GeoDataFrame({"geometry": [shapely.Point(1, 0.1), shapely.Point(1, 0.1)]}, index=[1, 2], crs="EPSG:32614"),
        # 5. Empty candidates (far away)
        gpd.GeoDataFrame({"geometry": [shapely.Point(1000, 1000), shapely.Point(2000, 2000)]}, index=[1, 2], crs="EPSG:32614"),
        # 6. Mixed candidates (some match, some far away)
        gpd.GeoDataFrame({"geometry": [shapely.Point(1, 0.1), shapely.Point(5000, 5000), shapely.Point(2, 0.2)]}, index=[7, 8, 7], crs="EPSG:32614"),
        # 7. Equal-distance ties
        gpd.GeoDataFrame({"geometry": [shapely.Point(5, 0.5)]}, index=[0], crs="EPSG:32614"),
    ]

    for pts in test_cases:
        leg_ids, leg_dists = _legacy_candidates(edges, pts)
        opt_ids, opt_dists = get_candidates_vectorized(edges, pts)

        assert leg_ids.equals(opt_ids)
        assert leg_dists.equals(opt_dists)
