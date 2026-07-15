import pickle
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    import numpy._core.numeric
except ModuleNotFoundError:
    import numpy.core as numpy_core
    import numpy.core.numeric as numpy_core_numeric
    sys.modules["numpy._core"] = numpy_core
    sys.modules["numpy._core.numeric"] = numpy_core_numeric

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from pipeline_v4.src import config
from pipeline_v4.src.modal_classification import (
    HybridRouteEvaluator,
    RandomForestRouteEvaluator,
    create_modal_evaluator,
)
from pipeline_v4.src.random_forest_contract import (
    EXPERIMENTAL_BUS_FEATURES,
    RF_FEATURES,
    TRAINING_SCENARIOS,
    TRAINING_TRIPS,
)


def trip_key(caid, num_trip):
    try:
        num_trip = str(int(float(num_trip)))
    except (TypeError, ValueError):
        num_trip = str(num_trip).strip()
    return f"{str(caid).strip()}_{num_trip}"


class TestRandomForestOfficial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = RandomForestRouteEvaluator()
        with (config.GPS_DIR / "random_forest_modal.pkl").open("rb") as handle:
            cls.model = pickle.load(handle)
        with (config.GPS_DIR / "datos_entrenamiento_ml.pkl").open("rb") as handle:
            cls.cache = pickle.load(handle)

    def test_feature_contract_matches_training_inference_notebook_and_pkl(self):
        self.assertEqual(len(RF_FEATURES), 52)
        self.assertEqual(list(RF_FEATURES), self.evaluator.feature_cols_v4)
        self.assertEqual(list(RF_FEATURES), self.evaluator.feature_cols_new)
        self.assertEqual(list(RF_FEATURES), self.model["feature_cols_v4"])
        self.assertEqual(list(RF_FEATURES), self.model["feature_cols_new"])
        self.assertFalse(set(RF_FEATURES).intersection(EXPERIMENTAL_BUS_FEATURES))

        training = (PROJECT_ROOT / "pipeline_v4/calibration_and_diagnostics/modal_classification/calibration/random_forest/entrenar_random_forest.py").read_text(encoding="utf-8")
        notebook = (PROJECT_ROOT / "pipeline_v4/calibration_and_diagnostics/modal_classification/notebooks/playground_modal_classifier.ipynb").read_text(encoding="utf-8")
        self.assertIn("feature_cols_v4 = list(RF_FEATURES)", training)
        self.assertIn("N2_FEATURES", notebook)
        self.assertIn("N1_FEATURES", notebook)
        self.assertIn("N3_FEATURES", notebook)
        self.assertIn("El flujo principal actual es el clasificador modal jerárquico híbrido", notebook)

    def test_model_is_loadable_and_has_compatible_estimators(self):
        self.assertTrue(self.evaluator.loaded_from_disk)
        for key in ("clf_n1", "clf_n2", "clf_n3"):
            classifier = self.model[key]
            self.assertEqual(classifier.n_features_in_, 52)
            self.assertEqual(list(classifier.feature_names_in_), list(RF_FEATURES))
            self.assertEqual(classifier.n_estimators, 100)

    def test_missing_model_fails_without_training(self):
        missing = PROJECT_ROOT / "Inputs" / "GPS User Data" / "modelo_inexistente_para_prueba.pkl"
        with self.assertRaises(FileNotFoundError):
            RandomForestRouteEvaluator(model_path=missing)

    def test_canonical_cache_has_66_trips_and_260_scenarios(self):
        clean = pd.read_csv(config.GPS_DIR / "Datos de MATLAB GPS Limpios.csv")
        canonical = {}
        for (caid, num_trip), subset in clean.groupby(["caid", "num_trip"]):
            modes = {str(mode).strip().lower() for mode in subset["mode_of_transport"].dropna() if str(mode).strip()}
            canonical[trip_key(caid, num_trip)] = next(iter(modes)) if len(modes) == 1 else None

        scenarios = set()
        for item in self.cache:
            physical, rest = item["trip_id"].split("-", 1)
            parts = rest.split("_")
            if canonical.get(physical) is not None:
                scenarios.add((physical, parts[1]))

        self.assertEqual(len({physical for physical, _ in scenarios}), TRAINING_TRIPS)
        self.assertEqual(len(scenarios), TRAINING_SCENARIOS)
        self.assertTrue(all(canonical[physical] is not None for physical, _ in scenarios))

    def test_groupkfold_never_splits_degradations_of_same_trip(self):
        clean = pd.read_csv(config.GPS_DIR / "Datos de MATLAB GPS Limpios.csv")
        canonical = {}
        for (caid, num_trip), subset in clean.groupby(["caid", "num_trip"]):
            modes = {str(mode).strip().lower() for mode in subset["mode_of_transport"].dropna() if str(mode).strip()}
            canonical[trip_key(caid, num_trip)] = len(modes) == 1
        scenarios = set()
        for item in self.cache:
            physical, rest = item["trip_id"].split("-", 1)
            if canonical.get(physical, False):
                scenarios.add((physical, rest.split("_")[1]))
        scenarios = sorted(scenarios)
        groups = np.array([physical for physical, _ in scenarios])
        for train_idx, validation_idx in GroupKFold(n_splits=5).split(scenarios, groups=groups):
            self.assertTrue(set(groups[train_idx]).isdisjoint(set(groups[validation_idx])))

    def test_end_to_end_inference(self):
        frame = pd.DataFrame({
            "caid": ["TEST"] * 20,
            "trip": [1] * 20,
            "latitude": [25.67] * 20,
            "longitude": [-100.31] * 20,
            "Speed [km/h]": np.linspace(15.0, 35.0, 20),
            "local_timestamp": pd.date_range("2026-07-15 08:00:00", periods=20, freq="30s"),
            "highway": ["primary"] * 20,
            "distance_m": [100.0] * 20,
            "near_bus_route": [0] * 20,
            "near_subway_line": [0] * 20,
        })
        self.evaluator.raw_counts["TEST_1"] = 20
        mode, selected, probability, probabilities = self.evaluator.select_final_mode({"Carro": frame})
        self.assertIn(mode, {"Carro", "Bus", "Metro", "Caminar"})
        self.assertIsNotNone(selected)
        self.assertGreaterEqual(probability, 0.0)
        self.assertEqual(set(probabilities), {"Carro", "Bus", "Metro", "Caminar"})

    def test_quality_guardrail(self):
        short = pd.DataFrame({"Speed [km/h]": [0.0] * 5})
        mode, selected, _, _ = self.evaluator.select_final_mode({"Caminar": short})
        self.assertEqual(mode, "Calidad insuficiente")
        self.assertIsNone(selected)

        low_retention = pd.DataFrame({
            "caid": ["TEST"] * 20,
            "trip": [2] * 20,
            "Speed [km/h]": [20.0] * 20,
        })
        self.evaluator.raw_counts["TEST_2"] = 100
        mode, selected, _, _ = self.evaluator.select_final_mode({"Carro": low_retention})
        self.assertEqual(mode, "Calidad insuficiente")
        self.assertIsNone(selected)

    def test_orchestrator_factory_defaults_to_hybrid_with_rf_rollback(self):
        scorer = create_modal_evaluator(config.MODAL_CLASSIFIER)
        self.assertIsInstance(scorer, HybridRouteEvaluator)
        self.assertIsInstance(create_modal_evaluator("random_forest"), RandomForestRouteEvaluator)
        self.assertEqual(config.MODAL_CLASSIFIER, "hybrid")
        self.assertFalse(config.ENABLE_BAYES_FALLBACK)


if __name__ == "__main__":
    unittest.main()

