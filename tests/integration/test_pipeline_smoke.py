import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_v4.src.emissions import calculate_emissions
from pipeline_v4.src.modal_classification import create_modal_evaluator
from pipeline_v4.src.pipeline_contracts import validate_routing_output
from pipeline_v4.src.routing import _finalize_routing_contract
from pipeline_v4.src.segmentation import assign_trips

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "outputs" / "production_smoke_tests"


class TestPipelineSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SMOKE.mkdir(parents=True, exist_ok=True)
        cls.evaluator = create_modal_evaluator("hybrid")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.lookup = Path(cls.tmp.name) / "moves.parquet"
        pollutants = ["CO", "CO2", "CO2_Equiv", "HC", "NOx", "PM10", "PM25"]
        rows = []
        for source in (21, 42):
            for road in (4, 5):
                for speed_bin in range(1, 17):
                    row = {"Day": 5, "Hour": 9, "Road": road, "Source": source, "SpeedBin": speed_bin}
                    row.update({p: float(1 + source / 10 + speed_bin / 100) for p in pollutants})
                    rows.append(row)
        pd.DataFrame(rows).to_parquet(cls.lookup, index=False)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    @staticmethod
    def _frame(rng, case, speed, highway, near_bus, near_metro, snap_drive, snap_walk):
        n = 20; timestamps = pd.date_range("2026-07-15 08:00", periods=n, freq="30s")
        frame = pd.DataFrame({
            "caid": [f"SMOKE{case}"] * n, "trip": [1] * n,
            "Speed [km/h]": np.clip(rng.normal(speed, max(1, speed * .3), n), 0, 140),
            "local_timestamp": timestamps, "highway": [highway] * n,
            "distance_m": np.maximum(1, rng.normal(max(10, speed * 8), 10, n)),
            "near_bus_route": rng.binomial(1, near_bus, n), "near_subway_line": rng.binomial(1, near_metro, n),
            "snap_dist_drive": [snap_drive] * n, "snap_dist_walk": [snap_walk] * n,
            "osmid": np.arange(n), "start_node": np.arange(n), "end_node": np.arange(1, n + 1),
            "modo_transporte": ["Carro"] * n, "ruteo_fallido": [False] * n,
            "corregido_espacialmente": [False] * n, "flag_auditoria": ["Nivel1_Lazy"] * n,
        })
        return _finalize_routing_contract(frame)

    def _representative_hypotheses(self):
        rng = np.random.default_rng(3); found = {}
        for case in range(300):
            def frame(speed, highway, nb, nm, sd, sw): return self._frame(rng, case, speed, highway, nb, nm, sd, sw)
            hypotheses = {
                "Carro": frame(rng.uniform(2, 80), rng.choice(["primary", "residential", "secondary"]), rng.random(), rng.random() * .2, rng.uniform(1, 100), rng.uniform(1, 50)),
                "Caminar": frame(rng.uniform(1, 25), rng.choice(["footway", "path", "residential"]), rng.random() * .2, rng.random() * .2, rng.uniform(1, 100), rng.uniform(1, 30)),
                "Metro": frame(rng.uniform(10, 80), "railway", 0, rng.random(), rng.uniform(1, 100), rng.uniform(1, 30)),
            }
            key = f"SMOKE{case}_1"; self.evaluator.raw_counts[key] = 20
            result = self.evaluator.evaluate_with_contract(hypotheses)
            mode = result["final_class"]
            if mode in {"Carro", "Bus", "Metro", "Caminar"} and mode not in found:
                found[mode] = (result, hypotheses)
            if len(found) == 4: break
        return found

    def test_small_end_to_end_four_modes(self):
        # Datos limpios -> segmentación mínima.
        clean = pd.DataFrame({
            "caid": ["SEG"] * 4, "date": [pd.Timestamp("2026-07-15").date()] * 4,
            "local_timestamp": pd.date_range("2026-07-15 08:00", periods=4, freq="30s"),
            "latitude": [25.67, 25.671, 25.672, 25.673], "longitude": [-100.31, -100.309, -100.308, -100.307],
            "Speed [km/h]": [20.] * 4, "dis lineal [m]": [100.] * 4,
            "travel time": pd.to_timedelta([30, 30, 30, 0], unit="s"),
        })
        segmented = assign_trips(clean)
        self.assertEqual(len(segmented), len(clean))
        self.assertTrue(segmented.local_timestamp.is_monotonic_increasing)

        found = self._representative_hypotheses()
        self.assertEqual(set(found), {"Carro", "Bus", "Metro", "Caminar"})
        output_rows = []
        for expected, (classification, _) in found.items():
            routed = classification["selected_route"]
            self.assertEqual(validate_routing_output(routed), [])
            self.assertEqual(classification["quality_status"], "accepted")
            self.assertAlmostEqual(sum(classification["probabilities"].values()), 1.0, places=6)
            emitted = calculate_emissions(routed, self.lookup)
            self.assertEqual(set(emitted.physical_trip_id), set(routed.physical_trip_id))
            self.assertTrue(emitted.local_timestamp.is_monotonic_increasing)
            self.assertTrue(np.array_equal(emitted.distance_m.to_numpy(), routed.distance_m.to_numpy()))
            self.assertFalse(emitted.duplicated(["physical_trip_id", "local_timestamp", "osmid"]).any())
            self.assertTrue((emitted.modo_transporte == expected).all())
            output_rows.append({"physical_trip_id": emitted.physical_trip_id.iloc[0], "mode": expected,
                                "segments": len(emitted), "distance_m": float(emitted.distance_m.sum()),
                                "total_CO2_g": float(emitted.Total_CO2_g.sum()), "status": "PASS"})
        pd.DataFrame(output_rows).to_csv(SMOKE / "end_to_end_results.csv", index=False)
        (SMOKE / "smoke_test.log").write_text("pipeline_v4_production end-to-end: PASS\n", encoding="utf-8")

    def test_degraded_guardrail_and_explicit_failure(self):
        hypotheses = next(iter(self._representative_hypotheses().values()))[1]
        short = {name: frame.iloc[:10].copy() for name, frame in hypotheses.items()}
        result = self.evaluator.evaluate_with_contract(short)
        self.assertEqual(result["final_class"], "Calidad insuficiente")
        self.assertEqual(result["rejection_reason"], "quality_guardrail")
        failure = {"case": "guardrail", "status": "rejected", "cause": result["rejection_reason"]}
        (SMOKE / "failures.json").write_text(json.dumps([failure], indent=2), encoding="utf-8")


