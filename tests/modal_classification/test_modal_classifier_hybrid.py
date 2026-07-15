import json
import pickle
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

try:
    import numpy._core.numeric  # NumPy 2.x
except ImportError:  # Cache serialized with NumPy 2.x, runtime on NumPy 1.x
    import numpy.core as numpy_core
    import numpy.core.numeric as numpy_core_numeric
    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.numeric", numpy_core_numeric)

from pipeline_v4.src import config
from pipeline_v4.src.modal_classification import (
    GuardrailedBayesianRouteEvaluator, HybridRouteEvaluator,
    RandomForestRouteEvaluator, create_modal_evaluator,
)
from pipeline_v4.src.random_forest_contract import (
    BUS_PROBABILITY_THRESHOLD, N1_FEATURES, N2_FEATURES, N3_FEATURES,
)

ROOT = Path(__file__).resolve().parents[2]
GPS = ROOT / "Inputs" / "GPS User Data"
CACHE = GPS / "cache_rutas_completas_expanded"


class TestHybridModalProduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hybrid = create_modal_evaluator("hybrid")
        cls.rf = create_modal_evaluator("random_forest")
        cls.bayes = create_modal_evaluator("bayes")

    def test_hybrid_is_default_and_factory_selects_all_backends(self):
        self.assertEqual(config.MODAL_CLASSIFIER, "hybrid")
        self.assertIsInstance(create_modal_evaluator(), HybridRouteEvaluator)
        self.assertIsInstance(self.hybrid, HybridRouteEvaluator)
        self.assertIsInstance(self.rf, RandomForestRouteEvaluator)
        self.assertIsInstance(self.bayes, GuardrailedBayesianRouteEvaluator)

    def test_three_persisted_resources_load(self):
        self.assertTrue((GPS / "modal_classifier_hybrid_v1.pkl").is_file())
        self.assertTrue((GPS / "random_forest_modal.pkl").is_file())
        bayes_json = ROOT / "pipeline_v4/calibration_and_diagnostics/modal_classification/calibration/bayes/matrices_optimas.json"
        self.assertIsInstance(json.loads(bayes_json.read_text(encoding="utf-8")), dict)

    def test_per_level_feature_contract_and_threshold(self):
        self.assertEqual(len(N1_FEATURES), 16)
        self.assertEqual(len(N2_FEATURES), 52)
        self.assertEqual(len(N3_FEATURES), 25)
        self.assertEqual(self.hybrid.n1_features, list(N1_FEATURES))
        self.assertEqual(self.hybrid.n2_features, list(N2_FEATURES))
        self.assertEqual(self.hybrid.n3_features, list(N3_FEATURES))
        self.assertEqual(self.hybrid.bus_threshold, BUS_PROBABILITY_THRESHOLD)

    @staticmethod
    def _hypotheses(trip="AAF_1", deg="Raw"):
        multiplier = {"Raw": 1.0, "L1": 0.95, "L2": 0.85, "L3": 0.75}[deg]
        frame = pd.DataFrame({
            "caid": ["AAF"] * 20, "trip": [1] * 20,
            "Speed [km/h]": [20.0 * multiplier] * 20,
            "local_timestamp": pd.date_range("2026-07-15", periods=20, freq="30s"),
            "highway": ["primary"] * 20, "distance_m": [100.0] * 20,
            "near_bus_route": [0] * 20, "near_subway_line": [0] * 20,
        })
        return {"Carro": frame}

    def test_hybrid_end_to_end_for_all_degradations(self):
        for deg in ("Raw", "L1", "L2", "L3"):
            hypotheses = self._hypotheses(deg=deg)
            self.hybrid.raw_counts["AAF_1"] = len(next(iter(hypotheses.values())))
            mode, frame, probability, diagnostics = self.hybrid.select_final_mode(hypotheses)
            self.assertIn(mode, self.hybrid.modos)
            self.assertIsNotNone(frame)
            self.assertGreaterEqual(probability, 0.0)
            self.assertEqual(set(diagnostics), set(self.hybrid.modos))

    def test_guardrail_is_shared(self):
        short = {key: frame.iloc[:10].copy() for key, frame in self._hypotheses().items()}
        for evaluator in (self.hybrid, self.rf, self.bayes):
            self.assertEqual(evaluator.select_final_mode(short, None, None)[0], "Calidad insuficiente")

    def test_grouped_degradations_and_mixed_exclusion(self):
        with (GPS / "datos_entrenamiento_ml_expanded.pkl").open("rb") as handle:
            cache = pickle.load(handle)
        scenarios = {}
        for item in cache:
            physical = item["trip_id"].split("-", 1)[0]
            scenarios.setdefault(physical, set()).add(item["degradacion"])
        self.assertEqual(len(scenarios), 114)
        clean = pd.read_csv(GPS / "Datos de MATLAB GPS Limpios.csv")
        mixed = set()
        for (caid, trip), part in clean.groupby(["caid", "num_trip"]):
            if part.mode_of_transport.dropna().astype(str).str.strip().str.lower().nunique() > 1:
                mixed.add(f"{caid}_{int(float(trip))}")
        self.assertTrue(set(scenarios).isdisjoint(mixed))

    def test_orchestrator_only_uses_factory_and_configuration(self):
        notebook = json.loads((ROOT / "pipeline_v4/orchestrator.ipynb").read_text(encoding="utf-8"))
        source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("create_modal_evaluator(config.MODAL_CLASSIFIER)", source)
        self.assertNotIn("RandomForestRouteEvaluator(", source)
        self.assertNotIn("HybridRouteEvaluator(", source)
        self.assertNotIn("BayesianRouteEvaluator(", source)

    def test_rollback_and_bayes_operate(self):
        self.assertTrue(self.rf.loaded_from_disk)
        empty = self.bayes.evaluate_completed_route_with_matrices(pd.DataFrame(), "Carro", None, None)
        self.assertEqual(float(empty.sum()), 0.0)

    def test_missing_model_fails_clearly(self):
        missing = ROOT / "missing-hybrid-model.pkl"
        with self.assertRaisesRegex(FileNotFoundError, "No se encontró el modelo híbrido"):
            HybridRouteEvaluator(missing)

    def test_missing_feature_fails_clearly(self):
        hypotheses = self._hypotheses()
        self.hybrid.raw_counts["AAF_1"] = len(next(iter(hypotheses.values())))
        incomplete = self.hybrid.extract_features(hypotheses)
        incomplete.pop(N1_FEATURES[0])
        with patch.object(self.hybrid, "extract_features", return_value=incomplete):
            with self.assertRaisesRegex(ValueError, "Faltan variables requeridas"):
                self.hybrid.select_final_mode(hypotheses)


if __name__ == "__main__":
    unittest.main()

