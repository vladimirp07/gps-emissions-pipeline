import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_v4.src import config
from pipeline_v4.src.config import EXECUTION_PROFILES, resolve_execution_profile
from pipeline_v4.src.production_workflow import (
    DayProcessingResult,
    load_pipeline_resources,
    create_modal_evaluator,
    run_pipeline_v4,
)


class TestExecutionProfilesAndEquivalence(unittest.TestCase):
    def test_profile_resolution_and_defaults(self):
        # 1. LOCAL_SAFE (and alias "local")
        for key in ["LOCAL_SAFE", "local", "local_safe"]:
            res_local = resolve_execution_profile(key)
            self.assertEqual(res_local["execution_profile"], "LOCAL_SAFE")
            self.assertEqual(res_local["backend"], "threading")
            self.assertEqual(res_local["effective_n_jobs"], 2)
            self.assertEqual(res_local["user_day_batch_size"], 32)

        # 2. LAB_THREADING_MODERATE
        for key in ["LAB_THREADING_MODERATE", "lab_threading_moderate", "lab_moderate"]:
            res_lab_mod = resolve_execution_profile(key)
            self.assertEqual(res_lab_mod["execution_profile"], "LAB_THREADING_MODERATE")
            self.assertEqual(res_lab_mod["backend"], "threading")
            self.assertEqual(res_lab_mod["effective_n_jobs"], 8)
            self.assertEqual(res_lab_mod["user_day_batch_size"], 1000)

        # 3. LAB_THREADING_AGGRESSIVE
        for key in ["LAB_THREADING_AGGRESSIVE", "lab_threading_aggressive", "lab_threading"]:
            res_lab_agg = resolve_execution_profile(key)
            self.assertEqual(res_lab_agg["execution_profile"], "LAB_THREADING_AGGRESSIVE")
            self.assertEqual(res_lab_agg["backend"], "threading")
            self.assertEqual(res_lab_agg["effective_n_jobs"], 24)
            self.assertEqual(res_lab_agg["user_day_batch_size"], 1000)

        # 4. LAB_PROCESS_TEST
        for key in ["LAB_PROCESS_TEST", "lab_process_test", "lab_process"]:
            res_lab_proc = resolve_execution_profile(key)
            self.assertEqual(res_lab_proc["execution_profile"], "LAB_PROCESS_TEST")
            self.assertEqual(res_lab_proc["backend"], "process")
            self.assertIn(res_lab_proc["effective_n_jobs"], [1, 2, 3, 4])  # safe memory guardrail
            self.assertEqual(res_lab_proc["user_day_batch_size"], 500)

    def test_override_priority_hierarchy(self):
        # Override wins over profile default
        res = resolve_execution_profile(
            "LAB_THREADING_AGGRESSIVE",
            n_jobs_override=16,
            batch_size_override=750,
            backend_override="threading",
        )
        self.assertEqual(res["execution_profile"], "LAB_THREADING_AGGRESSIVE")
        self.assertEqual(res["backend"], "threading")
        self.assertEqual(res["requested_n_jobs"], 16)
        self.assertEqual(res["effective_n_jobs"], 16)
        self.assertEqual(res["user_day_batch_size"], 750)

    def test_invalid_execution_values_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "Unsupported execution backend"):
            resolve_execution_profile("CUSTOM", backend_override="mystery")
        with self.assertRaisesRegex(ValueError, "n_jobs"):
            resolve_execution_profile("CUSTOM", n_jobs_override=0)

    def test_scientific_equivalence_across_execution_profiles(self):
        """Verify that Threading and Process execution profiles generate 100% identical scientific outputs."""
        tmp = tempfile.TemporaryDirectory()
        try:
            tmp_path = Path(tmp.name)
            # Create deterministic synthetic GPS trajectory for 2 users, 2 days each
            rows = []
            rng = np.random.default_rng(42)
            for u in ["USER_ALPHA", "USER_BETA"]:
                for d in ["2026-07-15", "2026-07-16"]:
                    n = 15
                    ts = pd.date_range(f"{d} 08:00", periods=n, freq="30s")
                    lats = 25.6866 + np.linspace(0, 0.01, n)
                    lons = -100.3161 + np.linspace(0, 0.01, n)
                    speeds = np.clip(rng.normal(25.0, 5.0, n), 0, 80)
                    for i in range(n):
                        rows.append({
                            "caid": u,
                            "user_id": u,
                            "local_timestamp": ts[i],
                            "latitude": lats[i],
                            "longitude": lons[i],
                            "lat_ruteo": lats[i],
                            "lon_ruteo": lons[i],
                            "Speed [km/h]": speeds[i],
                            "dis lineal [m]": 150.0,
                            "trip": 1,
                            "travel time": pd.Timedelta(seconds=30),
                        })
            gps_df = pd.DataFrame(rows)
            gps_file = tmp_path / "test_gps.parquet"
            gps_df.to_parquet(gps_file, index=False)

            # Pre-load shared resources for fast testing
            resources = load_pipeline_resources()
            resources["modal_evaluator"] = create_modal_evaluator(config.MODAL_CLASSIFIER)

            # Run 1: LOCAL_SAFE profile (Threading, n_jobs=2)
            out_local = tmp_path / "out_local"
            res_local = run_pipeline_v4(
                gps_file, out_local, resources=resources,
                execution_profile="LOCAL_SAFE", n_jobs_override=2,
                output_mode="both", show_progress=False,
            )

            # Run 2: LAB_THREADING_MODERATE profile (Threading, n_jobs=2)
            out_lab_th = tmp_path / "out_lab_th"
            res_lab_th = run_pipeline_v4(
                gps_file, out_lab_th, resources=resources,
                execution_profile="LAB_THREADING_MODERATE", n_jobs_override=2,
                output_mode="both", show_progress=False,
            )

            # Compare results
            # A. Trip ledger equivalence
            pd.testing.assert_frame_equal(
                res_local.trip_ledger.reset_index(drop=True),
                res_lab_th.trip_ledger.reset_index(drop=True),
                check_like=True,
            )

            # B. Routes equivalence
            pd.testing.assert_frame_equal(
                res_local.routes.reset_index(drop=True),
                res_lab_th.routes.reset_index(drop=True),
                check_like=True,
            )

            # C. Summary output equivalence
            sum_local = pd.read_parquet(out_local / "routes_emissions_summary.parquet")
            sum_lab_th = pd.read_parquet(out_lab_th / "routes_emissions_summary.parquet")
            pd.testing.assert_frame_equal(sum_local, sum_lab_th, check_like=True)

        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
