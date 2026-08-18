from __future__ import annotations

import networkx as nx
import pandas as pd

from pipeline_v4.src.routing import complete_route_v1_optimized


def _unroutable_trip(count=18):
    return pd.DataFrame({
        "caid": ["u"] * count,
        "trip": [1] * count,
        "modo_transporte": ["Carro"] * count,
        "latitude": [25.67] * count,
        "longitude": [-100.31 + idx * 0.0002 for idx in range(count)],
        "local_timestamp": pd.date_range("2019-12-16 08:00", periods=count, freq="30s"),
        "drive_ids": [[] for _ in range(count)],
        "drive_dists": [[] for _ in range(count)],
        "walk_ids": [[] for _ in range(count)],
        "walk_dists": [[] for _ in range(count)],
    })


def _run(max_skipped=None):
    graph_drive = nx.MultiDiGraph()
    graph_walk = nx.MultiDiGraph()
    return complete_route_v1_optimized(
        "u", _unroutable_trip(), graph_drive, graph_walk,
        None, None, {}, {}, None,
        max_lookahead_skipped_pings=max_skipped,
    )


def test_v2_recovery_contract_splits_and_records_uncovered_interval():
    routed = _run(max_skipped=10)
    unresolved = routed[routed.routing_event.eq("transition_unresolved")]

    assert not unresolved.empty
    assert unresolved.lookahead_skipped_pings.max() <= 10
    assert (unresolved.lookahead_elapsed_seconds > 0).all()
    assert (unresolved.lookahead_observed_distance_m > 0).all()
    assert unresolved.lookahead_failure_reason.str.contains("Topologia").all()
    assert routed.route_component_id.nunique() >= 2


def test_v1_default_retains_legacy_recovery_contract():
    routed = _run()

    assert "routing_event" not in routed.columns
    assert "route_component_id" not in routed.columns
    assert routed.flag_auditoria.str.contains("Amnesia_Definitiva|Lookahead_Skip").any()

