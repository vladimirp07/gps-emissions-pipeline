import numpy as np
import pandas as pd
import pytest

from pipeline_v4.src.modal_classification import RandomForestRouteEvaluator


def _legacy_window_values(frame, col_name, agg="mean"):
    if frame is None or frame.empty or "local_timestamp" not in frame:
        return []
    timestamps = pd.to_datetime(frame["local_timestamp"])
    if timestamps.isna().all():
        return []
    start, maximum = timestamps.min(), timestamps.max()
    values = []
    if agg == "mean":
        fn = lambda part: np.mean(part[col_name])
    else:
        fn = lambda part: np.sum(part[col_name])
    while start + pd.Timedelta(minutes=3) <= maximum:
        mask = (timestamps >= start) & (timestamps <= start + pd.Timedelta(minutes=3))
        if int(mask.sum()) >= 3:
            values.append(float(fn(frame.loc[mask])))
        start += pd.Timedelta(seconds=30)
    return values


def test_sliding_window_semantics_randomized_differential():
    np.random.seed(42)
    total_tests = 300

    for test_idx in range(total_tests):
        n_points = np.random.randint(0, 80)
        if n_points == 0:
            df = pd.DataFrame({"local_timestamp": [], "val": []})
        else:
            regime = test_idx % 6
            base_t = pd.Timestamp("2026-05-01 12:00:00")
            if regime == 0:  # sorted whole seconds
                offsets = np.sort(np.random.randint(0, 600, size=n_points))
                ts = [base_t + pd.Timedelta(seconds=int(o)) for o in offsets]
            elif regime == 1:  # sorted fractional nanoseconds
                offsets = np.sort(np.random.randint(0, 600_000, size=n_points)) * 1_000_000 + np.random.randint(0, 1_000_000, size=n_points)
                ts = [base_t + pd.Timedelta(nanoseconds=int(o)) for o in offsets]
            elif regime == 2:  # duplicate timestamps
                offsets = np.sort(np.random.choice([0, 10, 30, 60, 90, 120, 150, 180, 210, 240], size=n_points))
                ts = [base_t + pd.Timedelta(seconds=int(o)) for o in offsets]
            elif regime == 3:  # unsorted timestamps
                offsets = np.random.randint(0, 600, size=n_points)
                ts = [base_t + pd.Timedelta(seconds=int(o)) for o in offsets]
            elif regime == 4:  # unsorted fractional timestamps
                offsets = np.random.randint(0, 600_000, size=n_points) * 1_000_000 + np.random.randint(0, 1_000_000, size=n_points)
                ts = [base_t + pd.Timedelta(nanoseconds=int(o)) for o in offsets]
            elif regime == 5:  # exact window boundary timestamps
                offsets = np.random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300], size=n_points)
                ts = [base_t + pd.Timedelta(seconds=int(o)) for o in offsets]

            vals = np.random.uniform(0, 10, size=n_points)
            df = pd.DataFrame({"local_timestamp": ts, "val": vals})

        for agg in ["mean", "sum"]:
            leg = _legacy_window_values(df, "val", agg=agg)
            fast = RandomForestRouteEvaluator._window_values_fast(
                df["local_timestamp"] if not df.empty else None,
                df["val"].to_numpy() if not df.empty else [],
                agg=agg,
            )

            assert len(leg) == len(fast)
            if leg:
                assert np.allclose(leg, fast, rtol=1e-12, atol=1e-12)
