import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_v4.src.emissions import calculate_emissions
from pipeline_v4.src.modal_classification import TripServingContext, create_modal_evaluator
from pipeline_v4.src.pipeline_contracts import validate_routing_output
from pipeline_v4.src.routing import _finalize_routing_contract
from pipeline_v4.src.segmentation import assign_trips

class TestPipelineSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.smoke = Path(cls.tmp.name) / "production_smoke_tests"
        cls.smoke.mkdir(parents=True, exist_ok=True)
        cls.evaluator = create_modal_evaluator("hybrid")
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
    def _frame(rng, case, speed, highway, near_bus, near_metro, snap_drive, snap_walk, stops=False):
        n = 20; timestamps = pd.date_range("2026-07-15 08:00", periods=n, freq="30s")
        speeds = np.clip(rng.normal(speed, max(1, speed * .3), n), 0, 140)
        if stops:
            for i in range(3, n, 4):
                speeds[i] = 0.5
        frame = pd.DataFrame({
            "caid": [f"SMOKE{case}"] * n, "trip": [1] * n,
            "Speed [km/h]": speeds,
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
        # 1. Caminar
        h_walk = {
            "Caminar": self._frame(rng, 1, 4.0, "footway", 0.0, 0.0, 50.0, 2.0),
            "Carro": self._frame(rng, 1, 2.0, "residential", 0.0, 0.0, 50.0, 2.0),
        }
        ctx_walk = TripServingContext(20, 20, (50.0,)*20, (2.0,)*20)
        res_walk = self.evaluator.evaluate_with_contract(h_walk, serving_context=ctx_walk)
        found[res_walk["final_class"]] = (res_walk, h_walk)

        # 2. Carro
        h_car = {
            "Carro": self._frame(rng, 2, 65.0, "primary", 0.0, 0.0, 3.0, 25.0),
        }
        ctx_car = TripServingContext(20, 20, (3.0,)*20, (25.0,)*20)
        res_car = self.evaluator.evaluate_with_contract(h_car, serving_context=ctx_car)
        found[res_car["final_class"]] = (res_car, h_car)

        # 3. Metro
        h_metro = {
            "Carro": self._frame(rng, 3, 40.0, "primary", 0.0, 0.9, 10.0, 25.0),
            "Metro": self._frame(rng, 3, 45.0, "railway", 0.0, 1.0, 10.0, 25.0),
        }
        ctx_metro = TripServingContext(20, 20, (10.0,)*20, (25.0,)*20)
        res_metro = self.evaluator.evaluate_with_contract(h_metro, serving_context=ctx_metro)
        found[res_metro["final_class"]] = (res_metro, h_metro)

        # 4. Bus
        h_bus = {
            "Carro": self._frame(rng, 4, 25.0, "primary", 1.0, 0.0, 3.0, 25.0, stops=True),
        }
        ctx_bus = TripServingContext(20, 20, (3.0,)*20, (25.0,)*20)
        res_bus = self.evaluator.evaluate_with_contract(h_bus, serving_context=ctx_bus)
        found[res_bus["final_class"]] = (res_bus, h_bus)

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
        pd.DataFrame(output_rows).to_csv(self.smoke / "end_to_end_results.csv", index=False)
        (self.smoke / "smoke_test.log").write_text("pipeline_v4_production end-to-end: PASS\n", encoding="utf-8")

    def test_degraded_guardrail_and_explicit_failure(self):
        hypotheses = next(iter(self._representative_hypotheses().values()))[1]
        short = {name: frame.iloc[:10].copy() for name, frame in hypotheses.items()}
        context = TripServingContext(20, 10, (10.0,) * 10, (5.0,) * 10)
        result = self.evaluator.evaluate_with_contract(short, serving_context=context)
        self.assertEqual(result["final_class"], "Calidad insuficiente")
        self.assertEqual(result["rejection_reason"], "quality_guardrail")
        failure = {"case": "guardrail", "status": "rejected", "cause": result["rejection_reason"]}
        (self.smoke / "failures.json").write_text(json.dumps([failure], indent=2), encoding="utf-8")


