from __future__ import annotations

import pandas as pd
import pytest

from pipeline_v4.src.segmentation import preprocess_gps_frame


LOCAL_TZ = "America/Monterrey"


def _raw_local_days(days):
    rows = []
    for day in pd.to_datetime(days):
        for minute, offset in ((0, 0.0), (5, 0.0001), (10, 0.0002)):
            local = pd.Timestamp(day) + pd.Timedelta(hours=12, minutes=minute)
            utc = local.tz_localize(LOCAL_TZ).tz_convert("UTC")
            rows.append({
                "caid": "user-a",
                "utc_timestamp": utc,
                "latitude": 25.6866 + offset,
                "longitude": -100.3161 + offset,
            })
    return pd.DataFrame(rows)


def _run(days, start=None, end=None):
    prepared, _ = preprocess_gps_frame(
        _raw_local_days(days), coverage_start=start, coverage_end=end,
    )
    return prepared


@pytest.mark.parametrize("day_count", [1, 2, 3, 4, 5])
def test_all_declared_complete_local_days_are_retained(day_count):
    days = pd.date_range("2019-12-01", periods=day_count, freq="D")
    start = pd.Timestamp(days[0]).tz_localize(LOCAL_TZ)
    end = (pd.Timestamp(days[-1]) + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
    prepared = _run(days, start, end)

    assert sorted(prepared["date"].unique()) == [value.date() for value in days]
    assert set(prepared["day_completeness_status"]) == {"complete"}


def test_three_complete_local_days_retains_all_three():
    days = pd.date_range("2019-12-01", periods=3, freq="D")
    prepared = _run(
        days,
        pd.Timestamp("2019-12-01 00:00", tz=LOCAL_TZ),
        pd.Timestamp("2019-12-04 00:00", tz=LOCAL_TZ),
    )
    assert prepared["date"].nunique() == 3


def test_partial_first_day_is_excluded_and_traced():
    days = pd.date_range("2019-12-01", periods=3, freq="D")
    prepared = _run(
        days,
        pd.Timestamp("2019-12-01 06:00", tz=LOCAL_TZ),
        pd.Timestamp("2019-12-04 00:00", tz=LOCAL_TZ),
    )
    assert sorted(prepared.date.unique()) == [days[1].date(), days[2].date()]
    assert prepared.attrs["day_completeness"]["status_by_date"]["2019-12-01"] == "partial_start"


def test_partial_last_day_is_excluded_and_traced():
    days = pd.date_range("2019-12-01", periods=3, freq="D")
    prepared = _run(
        days,
        pd.Timestamp("2019-12-01 00:00", tz=LOCAL_TZ),
        pd.Timestamp("2019-12-03 18:00", tz=LOCAL_TZ),
    )
    assert sorted(prepared.date.unique()) == [days[0].date(), days[1].date()]
    assert prepared.attrs["day_completeness"]["status_by_date"]["2019-12-03"] == "partial_end"


def test_both_partial_edges_are_excluded():
    days = pd.date_range("2019-12-01", periods=3, freq="D")
    prepared = _run(
        days,
        pd.Timestamp("2019-12-01 06:00", tz=LOCAL_TZ),
        pd.Timestamp("2019-12-03 18:00", tz=LOCAL_TZ),
    )
    assert prepared.date.unique().tolist() == [days[1].date()]


def test_utc_slice_can_create_two_partial_local_edge_days():
    local_days = pd.date_range("2019-11-30", periods=4, freq="D")
    prepared = _run(
        local_days,
        pd.Timestamp("2019-12-01 00:00:00Z"),
        pd.Timestamp("2019-12-04 00:00:00Z"),
    )
    assert sorted(prepared.date.unique()) == [
        pd.Timestamp("2019-12-01").date(),
        pd.Timestamp("2019-12-02").date(),
    ]


def test_unknown_coverage_keeps_days_and_marks_them_unknown():
    days = pd.date_range("2019-12-01", periods=3, freq="D")
    prepared = _run(days)
    assert prepared.date.nunique() == 3
    assert set(prepared.day_completeness_status) == {"unknown"}
