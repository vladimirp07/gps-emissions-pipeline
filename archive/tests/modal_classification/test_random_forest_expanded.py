import hashlib
import json
import pickle
import unittest
from pathlib import Path

from pipeline_v4.src.modal_classification import RandomForestRouteEvaluator
from pipeline_v4.src.random_forest_contract import EXPERIMENTAL_BUS_FEATURES, RF_FEATURES


ROOT = Path(__file__).resolve().parent
GPS = ROOT / "Inputs" / "GPS User Data"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestExpandedCandidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads((GPS / "auditoria_dataset_ml_expanded.json").read_text(encoding="utf-8"))
        cls.baseline_metrics = json.loads((GPS / "metricas_rf_baseline_sklearn180.json").read_text(encoding="utf-8"))
        cls.expanded_metrics = json.loads((GPS / "metricas_rf_expanded_candidate.json").read_text(encoding="utf-8"))
        with (GPS / "datos_entrenamiento_ml_expanded.pkl").open("rb") as handle:
            cls.cache = pickle.load(handle)
        with (GPS / "random_forest_modal_expanded_candidate.pkl").open("rb") as handle:
            cls.model = pickle.load(handle)

    def test_baseline_artifacts_were_not_overwritten(self):
        self.assertEqual(sha256(GPS / "datos_entrenamiento_ml.pkl"), "0c42a999b20e0a0bb0caf55cb6584bdf655352ca4613fe4d986dc535d0ebb22b")
        self.assertEqual(sha256(GPS / "random_forest_modal.pkl"), "c485216fb04205612474e4e9d42076a5bcc0ead525737d5a959adb53b3961154")

    def test_expanded_coverage_and_exclusions(self):
        physical = {item["trip_id"].split("-", 1)[0] for item in self.cache}
        scenarios = {(item["trip_id"].split("-", 1)[0], item["degradacion"]) for item in self.cache}
        self.assertEqual(len(physical), 114)
        self.assertEqual(len(scenarios), 445)
        self.assertEqual(len(self.audit["selected"]), 114)
        reasons = list(self.audit["excluded"].values())
        self.assertEqual(sum(reason == "etiqueta_mixta" for reason in reasons), 15)
        self.assertEqual(sum(reason.startswith("calidad_insuficiente") for reason in reasons), 10)

    def test_candidate_keeps_exact_feature_contract(self):
        self.assertEqual(self.model["feature_cols_v4"], list(RF_FEATURES))
        self.assertEqual(self.model["feature_cols_new"], list(RF_FEATURES))
        self.assertFalse(set(RF_FEATURES).intersection(EXPERIMENTAL_BUS_FEATURES))
        for name in ("clf_n1", "clf_n2", "clf_n3"):
            self.assertEqual(self.model[name].n_features_in_, 52)

    def test_candidate_loads_and_inference_class_is_available(self):
        evaluator = RandomForestRouteEvaluator(GPS / "random_forest_modal_expanded_candidate.pkl")
        self.assertTrue(evaluator.loaded_from_disk)

    def test_rejection_rule_is_supported_by_metrics(self):
        self.assertGreater(self.expanded_metrics["physical_trips"], self.baseline_metrics["physical_trips"])
        self.assertLess(self.expanded_metrics["balanced_accuracy_mean"], self.baseline_metrics["balanced_accuracy_mean"] - 0.05)
        self.assertGreater(self.expanded_metrics["balanced_accuracy_std"], self.baseline_metrics["balanced_accuracy_std"] + 0.05)


if __name__ == "__main__":
    unittest.main()

