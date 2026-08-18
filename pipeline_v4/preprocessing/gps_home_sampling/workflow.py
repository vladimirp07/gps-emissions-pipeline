"""Fast, transparent preprocessing for residential inference and sampling.

The implementation deliberately uses a spatial grid instead of a global pairwise
clustering algorithm. It is deterministic, processes one user at a time, and
keeps the distinction between original GPS observations and route-completed rows
explicit in the input contract.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from zoneinfo import ZoneInfo
import math
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


@dataclass(frozen=True)
class HomeConfig:
    night_start: int = 22
    night_end: int = 5
    cell_m: float = 250.0
    min_nights: int | None = 3
    min_points_per_night: int = 1
    max_gap_minutes: float = 30.0
    timezone: str = "America/Monterrey"
    confidence_reliable: float = 0.70
    confidence_probable: float = 0.45


def _first(df, names, required=True):
    for name in names:
        if name in df.columns:
            return name
    if required:
        raise KeyError(f"No se encontró ninguna columna equivalente a {names}")
    return None


def standardize_gps(df: pd.DataFrame, source_kind: str = "raw", timezone: str = "America/Monterrey") -> pd.DataFrame:
    """Return a canonical frame without modifying the source frame.

    ``source_kind`` is retained in the output so processed data cannot silently
    be mixed with raw observations. For processed route data, only rows with an
    explicit original marker are retained when one is available.
    """
    out = df.copy()
    user = _first(out, ["user_id", "caid", "device_id", "id"])
    lat = _first(out, ["latitude", "lat", "home_lat"])
    lon = _first(out, ["longitude", "lon", "lng", "home_lon"])
    ts = _first(out, ["local_timestamp", "timestamp", "utc_timestamp", "datetime"])
    out = out.rename(columns={user: "user_id", lat: "latitude", lon: "longitude", ts: "timestamp"})
    if pd.api.types.is_numeric_dtype(out["timestamp"]):
        out["timestamp"] = pd.to_datetime(out["timestamp"], unit="s", utc=True, errors="coerce")
    else:
        out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    if timezone:
        out["timestamp_local"] = out["timestamp"].dt.tz_convert(ZoneInfo(timezone))
    else:
        out["timestamp_local"] = out["timestamp"]
    for c in ["latitude", "longitude"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["user_id", "latitude", "longitude", "timestamp_local"])
    out = out.loc[out["latitude"].between(-90, 90) & out["longitude"].between(-180, 180)].copy()
    out["source_kind"] = source_kind
    out["is_original_observation"] = True
    if source_kind == "processed":
        # These are the strongest available indicators in the repository. The
        # absence of a marker is intentionally treated as an unresolved case.
        if "orden_original" in out.columns:
            out["is_original_observation"] = out["orden_original"].notna()
        elif "flag_auditoria" in out.columns:
            marker = out["flag_auditoria"].astype(str).str.lower()
            out["is_original_observation"] = ~marker.str.contains("complet|sintet|generat", na=False)
    return out.sort_values(["user_id", "timestamp_local"], kind="stable").reset_index(drop=True)


def _night_id(ts: pd.Series, start: int) -> pd.Series:
    # A night is labelled by the date on which its evening begins.
    return (ts - pd.to_timedelta((ts.dt.hour < start).astype(int), unit="D")).dt.date.astype(str)


def _grid_key(lat: float, lon: float, cell_m: float) -> tuple[int, int]:
    lat_m = lat * 111_320.0
    lon_m = lon * 111_320.0 * max(math.cos(math.radians(lat)), 0.1)
    return int(math.floor(lat_m / cell_m)), int(math.floor(lon_m / cell_m))


def _confidence(nights, home_nights, activity_fraction, separation, stay_hours):
    recurrence = min(home_nights / max(nights, 1), 1.0)
    duration = min(stay_hours / 8.0, 1.0)
    score = 0.35 * recurrence + 0.35 * activity_fraction + 0.20 * separation + 0.10 * duration
    return round(float(score), 4)


def infer_home_locations(gps: pd.DataFrame, config: HomeConfig | None = None) -> pd.DataFrame:
    """Infer one candidate home per user using recurring night cells.

    A point contributes to a cell; intervals are capped to avoid making sparse
    pings look like long stays. The winning cell must recur across nights.
    """
    cfg = config or HomeConfig()
    d = gps.loc[gps["is_original_observation"].fillna(True)].copy()
    local = d["timestamp_local"]
    hour = local.dt.hour + local.dt.minute / 60
    night = (hour >= cfg.night_start) | (hour < cfg.night_end)
    d = d.loc[night].copy()
    d["night_id"] = _night_id(d["timestamp_local"], cfg.night_start)
    d["cell"] = [_grid_key(a, o, cfg.cell_m) for a, o in zip(d.latitude, d.longitude)]
    rows = []
    for uid, u in d.groupby("user_id", sort=False):
        nights = u.groupby("night_id", sort=False)
        n_nights = len(nights)
        cell_stats = []
        for cell, c in u.groupby("cell", sort=False):
            night_ids = set(c.night_id)
            # Sum only within-night gaps and cap every gap.
            c = c.sort_values("timestamp_local")
            gaps = c["timestamp_local"].diff().dt.total_seconds().div(3600).clip(lower=0, upper=cfg.max_gap_minutes / 60)
            cell_stats.append((cell, len(night_ids), int(len(c)), float(gaps.fillna(0).sum()), night_ids, c))
        cell_stats.sort(key=lambda x: (x[1], x[3], x[2]), reverse=True)
        if not cell_stats:
            continue
        win = cell_stats[0]
        second = cell_stats[1][1] if len(cell_stats) > 1 else 0
        home_nights = win[1]
        frac = win[2] / max(len(u), 1)
        separation = (home_nights - second) / max(home_nights, 1)
        coords = win[5]
        # weighted centroid is robust and stays in geographic coordinates.
        lat, lon = float(coords.latitude.median()), float(coords.longitude.median())
        score = _confidence(n_nights, home_nights, frac, separation, win[3])
        # ``None`` permits a candidate to be assessed without imposing the
        # production recurrence threshold.  One observed night remains
        # insufficient evidence: a dominant single night must never be
        # promoted to probable/reliable merely because the threshold is lax.
        minimum_quality_nights = 2 if cfg.min_nights is None else cfg.min_nights
        if n_nights < minimum_quality_nights or home_nights < minimum_quality_nights:
            quality = "insufficient_information"
        elif score >= cfg.confidence_reliable and separation >= 0.25:
            quality = "reliable"
        elif score >= cfg.confidence_probable:
            quality = "probable"
        else:
            quality = "ambiguous"
        rows.append({"user_id": uid, "home_lat": lat, "home_lon": lon,
                     "n_nights_observed": n_nights, "n_nights_home_detected": home_nights,
                     "night_activity_fraction_home": round(frac, 4),
                     "home_confidence": score, "home_quality_flag": quality,
                     "home_cluster_n_points": win[2], "home_cluster_stay_hours": round(win[3], 3),
                     "second_cluster_nights": second, "home_cluster_night_ids": ";".join(sorted(win[4]))})
    return pd.DataFrame(rows)


def read_ageb(path: str | Path, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    p = Path(path)
    if p.suffix.lower() == ".json":
        return gpd.read_file(p).set_crs(crs, allow_override=True)
    return gpd.read_file(p)


def assign_ageb(home: pd.DataFrame, ageb: gpd.GeoDataFrame) -> tuple[pd.DataFrame, gpd.GeoDataFrame]:
    a = ageb.copy()
    a = a.to_crs("EPSG:4326") if a.crs is not None else a.set_crs("EPSG:4326")
    key = _first(a, ["CVEGEO", "CVEGEO_1", "AGEB", "CVE_AGEB"])
    pop = _first(a, ["POBTOT", "population", "POPULATION"], required=False)
    keep = [key, "geometry"] + ([pop] if pop else [])
    for c in ["NOM_MUN", "MUN", "CVE_MUN"]:
        if c in a.columns: keep.append(c)
    aa = a[keep].rename(columns={key: "home_ageb", **({pop: "ageb_population"} if pop else {})})
    pts = gpd.GeoDataFrame(home.copy(), geometry=gpd.points_from_xy(home.home_lon, home.home_lat), crs="EPSG:4326")
    joined = gpd.sjoin(pts, aa, how="left", predicate="within").drop(columns=["geometry", "index_right"], errors="ignore")
    return pd.DataFrame(joined), a


def ageb_coverage(home: pd.DataFrame, ageb: gpd.GeoDataFrame) -> pd.DataFrame:
    counts = home.dropna(subset=["home_ageb"]).groupby("home_ageb").size().rename("gps_users").reset_index()
    key = _first(ageb, ["CVEGEO", "CVEGEO_1", "AGEB", "CVE_AGEB"])
    pop = _first(ageb, ["POBTOT", "population", "POPULATION"], required=False)
    base = ageb[[key] + ([pop] if pop else [])].drop_duplicates().rename(columns={key: "home_ageb", **({pop: "ageb_population"} if pop else {})})
    out = base.merge(counts, on="home_ageb", how="left").fillna({"gps_users": 0})
    if pop: out["gps_users_per_population"] = out["gps_users"] / out["ageb_population"].replace(0, np.nan)
    out["coverage_flag"] = np.select([out.gps_users.eq(0), out.gps_users.lt(3)], ["no_users", "few_users"], default="covered")
    return out


def build_ageb_groups(ageb: gpd.GeoDataFrame, target_size: int = 4, seed: int = 42) -> gpd.GeoDataFrame:
    """Build contiguous AGEB groups with a deterministic greedy frontier."""
    a = ageb.copy().to_crs("EPSG:4326")
    key = _first(a, ["CVEGEO", "CVEGEO_1", "AGEB", "CVE_AGEB"])
    pop = _first(a, ["POBTOT", "population", "POPULATION"], required=False)
    a["_key"] = a[key].astype(str)
    a["_pop"] = pd.to_numeric(a[pop], errors="coerce").fillna(0) if pop else 0
    projected = a.to_crs("EPSG:3857")
    neighbors = {k: set() for k in a._key}
    sindex = projected.sindex
    for i, geom in enumerate(projected.geometry):
        cand = list(sindex.query(geom.buffer(2), predicate="intersects"))
        for j in cand:
            if i != j and geom.touches(projected.geometry.iloc[j]):
                neighbors[a._key.iloc[i]].add(a._key.iloc[j])
    remaining = set(a._key)
    groups = []
    while remaining:
        start = sorted(remaining, key=lambda k: (-len(neighbors[k] & remaining), k))[0]
        members = [start]; remaining.remove(start); frontier = set(neighbors[start]) & remaining
        while frontier and len(members) < target_size:
            nxt = sorted(frontier, key=lambda k: (float(a.loc[a._key.eq(k), "_pop"].iloc[0]), k))[0]
            members.append(nxt); remaining.remove(nxt); frontier.discard(nxt); frontier |= neighbors[nxt] & remaining
        groups.append(members)
    mapping = {k: f"AGEBG_{i+1:04d}" for i, members in enumerate(groups) for k in members}
    a["ageb_group_id"] = a._key.map(mapping)
    a = a.dissolve("ageb_group_id", aggfunc={"_pop": "sum", "_key": lambda x: ";".join(sorted(x))}).reset_index()
    a = a.rename(columns={"_pop": "group_population", "_key": "ageb_list"})
    return a


def _sample(df, group_col, n_target, mode, seed, min_users=0, max_users=None):
    rng = np.random.default_rng(seed)
    available = df.groupby(group_col).size().rename("available_users").reset_index()
    if mode == "fixed":
        allocation = available.assign(requested_users=int(n_target))
    else:
        pop = df.groupby(group_col)["ageb_population"].first().fillna(0)
        weights = pop / pop.sum() if pop.sum() else pd.Series(1 / len(pop), index=pop.index)
        allocation = available.copy(); allocation["requested_users"] = np.floor(weights.reindex(allocation[group_col]).fillna(0).to_numpy() * n_target).astype(int)
        allocation["requested_users"] = allocation["requested_users"].clip(lower=min_users)
        if max_users is not None: allocation["requested_users"] = allocation["requested_users"].clip(upper=max_users)
    out=[]
    for _, row in allocation.iterrows():
        g=row[group_col]; candidates=df[df[group_col].eq(g)].sort_values("user_id")
        k=min(int(row.requested_users), len(candidates)); chosen=candidates.iloc[rng.choice(len(candidates),k,replace=False)] if k else candidates.head(0)
        if k: chosen=chosen.copy(); chosen["sampling_mode"]=mode; chosen["requested_users_in_group"]=int(row.requested_users); chosen["selection_weight"] = row.available_users / k
        out.append(chosen)
    selected=pd.concat(out,ignore_index=True) if out else df.head(0).copy()
    allocation["deficit"]=(allocation.requested_users-allocation.available_users).clip(lower=0)
    return selected, allocation


def sample_by_ageb(home: pd.DataFrame, n_target=10, mode="fixed", seed=42, min_users_per_ageb=0, max_users_per_ageb=None):
    return _sample(home.dropna(subset=["home_ageb"]), "home_ageb", n_target, mode, seed, min_users_per_ageb, max_users_per_ageb)


def sample_by_group(home: pd.DataFrame, n_target=10, mode="fixed", seed=42, min_users_per_group=0, max_users_per_group=None):
    return _sample(home.dropna(subset=["ageb_group_id"]), "ageb_group_id", n_target, mode, seed, min_users_per_group, max_users_per_group)


def user_profile_features(gps: pd.DataFrame) -> pd.DataFrame:
    d=gps.copy(); d["hour"]=d.timestamp_local.dt.hour; d["is_weekend"]=d.timestamp_local.dt.dayofweek>=5; d["date"]=d.timestamp_local.dt.date
    return d.groupby("user_id").agg(records=("user_id","size"), days_observed=("date","nunique"), unique_locations=("latitude",lambda x: x.round(3).astype(str).nunique()), spatial_lat_sd=("latitude","std"), spatial_lon_sd=("longitude","std"), weekend_fraction=("is_weekend","mean")).reset_index()


def sample_size_sensitivity(gps: pd.DataFrame, sizes=(5,10,20,30), repeats=20, seed=42) -> pd.DataFrame:
    """Compare sample profiles with the full available-user profile."""
    profiles=user_profile_features(gps).set_index("user_id"); numeric=profiles.select_dtypes("number"); ref=numeric.mean()
    rng=np.random.default_rng(seed); rows=[]
    users=profiles.index.to_numpy()
    for n in sizes:
        for rep in range(repeats):
            chosen=rng.choice(users,min(n,len(users)),replace=False); mean=profiles.loc[chosen,numeric.columns].mean()
            denom=ref.abs().replace(0,np.nan); rel=((mean-ref).abs()/denom).mean(); corr=mean.corr(ref) if len(chosen)>1 else np.nan
            rows.append({"sample_size":n,"repeat":rep,"mean_relative_difference":float(rel),"profile_correlation":float(corr) if pd.notna(corr) else np.nan})
    return pd.DataFrame(rows)


def config_manifest(cfg: HomeConfig, source_path: str, source_kind: str) -> dict:
    return {"source_path":source_path,"source_kind":source_kind,"home_config":asdict(cfg),"method":"nightly recurring spatial grid; original observations only"}
