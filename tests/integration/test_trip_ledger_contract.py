import pandas as pd

from pipeline_v4.src.production_workflow import _raw_ping_count_for_effective_trip


def test_raw_ping_denominators_are_not_duplicated_when_effective_trips_split():
    timestamps = pd.date_range("2026-08-18 08:00", periods=20, freq="10s")
    raw = pd.DataFrame({"local_timestamp": timestamps, "trip": [1] * 20})
    first = pd.DataFrame({"local_timestamp": timestamps[1:8]})
    second = pd.DataFrame({"local_timestamp": timestamps[12:19]})
    first_count = _raw_ping_count_for_effective_trip(raw, first)
    second_count = _raw_ping_count_for_effective_trip(raw, second)
    assert first_count == 7
    assert second_count == 7
    assert first_count + second_count <= len(raw)
