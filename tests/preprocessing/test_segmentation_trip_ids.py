"""Regression tests for unique positive IDs assigned to contiguous mobile runs."""
from __future__ import annotations

import pandas as pd

from pipeline_v4.src.segmentation import assign_trips


def _frame(rows):
    """Build deterministic segmentation inputs as (speed, distance, dt_seconds)."""
    timestamps = pd.date_range("2026-01-01 08:00:00", periods=len(rows), freq="min")
    return pd.DataFrame({
        "caid": "user-a",
        "local_timestamp": timestamps,
        "date": timestamps.date,
        "latitude": 25.68,
        "longitude": -100.31,
        "Speed [km/h]": [row[0] for row in rows],
        "dis lineal [m]": [row[1] for row in rows],
        "travel time": pd.to_timedelta([row[2] for row in rows], unit="s"),
    })


def _positive_run_ids(values):
    runs = []
    previous = None
    for value in values:
        if value > 0 and (previous is None or previous <= 0):
            runs.append(value)
        previous = value
    return runs


def _assert_unique_positive_runs(frame, expected):
    trips = assign_trips(frame)["trip"].tolist()
    run_ids = _positive_run_ids(trips)
    assert run_ids == expected
    assert len(run_ids) == len(set(run_ids)), "a positive trip ID was reused"
    return trips


def test_day_starting_with_short_quiet_spell_consumes_trip_one():
    trips = _assert_unique_positive_runs(_frame([
        (0.0, 0.0, 60),
        (20.0, 300.0, 60),
        (25.0, 400.0, 60),
    ]), [1])
    assert trips == [1, 1, 1]


def test_movement_stop_movement_gets_distinct_positive_ids():
    trips = _assert_unique_positive_runs(_frame([
        (20.0, 300.0, 60),
        (0.0, 0.0, 360),
        (25.0, 400.0, 60),
    ]), [1, 2])
    assert trips == [1, -1, 2]


def test_gap_over_thirty_minutes_starts_a_new_positive_trip():
    trips = _assert_unique_positive_runs(_frame([
        (20.0, 300.0, 60),
        (20.0, 300.0, 1801),
        (20.0, 300.0, 60),
    ]), [1, 2])
    assert trips == [1, -1, 2]


def test_initial_quiet_then_gap_never_reuses_trip_one():
    trips = _assert_unique_positive_runs(_frame([
        (0.0, 0.0, 60),
        (20.0, 300.0, 1801),
        (20.0, 300.0, 60),
    ]), [1, 2])
    assert trips == [1, -1, 2]


def test_multiple_stops_and_trips_are_monotonic_and_unique():
    trips = _assert_unique_positive_runs(_frame([
        (20.0, 300.0, 60),
        (0.0, 0.0, 301),
        (20.0, 300.0, 60),
        (0.0, 0.0, 301),
        (20.0, 300.0, 60),
    ]), [1, 2, 3])
    assert trips == [1, -1, 2, -2, 3]


def test_no_positive_id_is_used_by_noncontiguous_runs():
    frame = _frame([
        (0.0, 0.0, 20),
        (10.0, 150.0, 20),
        (0.0, 0.0, 400),
        (0.0, 0.0, 30),
        (15.0, 200.0, 30),
        (15.0, 200.0, 1900),
        (12.0, 180.0, 30),
    ])
    trips = _assert_unique_positive_runs(frame, [1, 2, 3])
    for positive_id in {value for value in trips if value > 0}:
        positions = [index for index, value in enumerate(trips) if value == positive_id]
        assert positions == list(range(min(positions), max(positions) + 1))


def test_multi_user_results_match_each_user_processed_in_isolation():
    user_a = _frame([
        (20.0, 300.0, 60),
        (0.0, 0.0, 360),
        (25.0, 400.0, 60),
    ])
    user_b = _frame([
        (0.0, 0.0, 60),
        (12.0, 180.0, 60),
        (0.0, 0.0, 360),
        (18.0, 250.0, 60),
    ]).assign(caid="user-b")
    together = assign_trips(pd.concat([user_a, user_b], ignore_index=True))

    for user, isolated_input in (("user-a", user_a), ("user-b", user_b)):
        isolated = assign_trips(isolated_input.copy())
        combined_user = together.loc[together.caid.eq(user)].reset_index(drop=True)
        pd.testing.assert_series_equal(
            combined_user["trip"], isolated.reset_index(drop=True)["trip"], check_exact=True,
        )


def test_single_user_trip_ids_match_frozen_production_behavior():
    frame = _frame([
        (0.0, 0.0, 20),
        (10.0, 150.0, 20),
        (0.0, 0.0, 400),
        (0.0, 0.0, 30),
        (15.0, 200.0, 30),
        (15.0, 200.0, 1900),
        (12.0, 180.0, 30),
    ])
    assert assign_trips(frame)["trip"].tolist() == [1, 1, -1, -1, 2, -2, 3]
