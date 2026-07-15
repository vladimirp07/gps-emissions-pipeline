import unittest
import numpy as np
import pandas as pd

from pipeline_v3.src.pipeline_contracts import validate_routing_output
from pipeline_v3.src.routing import _finalize_routing_contract


class TestRoutingContract(unittest.TestCase):
    def frame(self, failed=False):
        return pd.DataFrame({
            "caid": ["CAR"] * 3, "trip": [1] * 3,
            "local_timestamp": pd.date_range("2026-07-15", periods=3, freq="30s"),
            "modo_transporte": ["Carro"] * 3,
            "start_node": [1, 2, 3], "end_node": [2, 3, 4],
            "osmid": [10, 11, 12], "distance_m": [100.0, 120.0, 80.0],
            "Speed [km/h]": [12.0, 14.4, 9.6], "ruteo_fallido": [failed] * 3,
            "corregido_espacialmente": [False] * 3,
            "flag_auditoria": ["Rollback" if failed else "Nivel1_Lazy"] * 3,
        })

    def test_success_contract_and_physics(self):
        routed = _finalize_routing_contract(self.frame())
        self.assertEqual(validate_routing_output(routed), [])
        self.assertTrue(routed.local_timestamp.is_monotonic_increasing)
        self.assertTrue((routed.distance_m >= 0).all())
        self.assertTrue(np.isfinite(routed["Speed [km/h]"]).all())
        self.assertEqual(routed.distance_m.sum(), 300.0)
        self.assertFalse(routed.duplicated(["physical_trip_id", "local_timestamp", "start_node", "end_node", "osmid"]).any())

    def test_difficult_route_failure_is_explicit(self):
        routed = _finalize_routing_contract(self.frame(failed=True))
        self.assertTrue((routed.routing_status == "fallback_or_failed").all())
        self.assertTrue(routed.flag_auditoria.str.contains("Rollback").all())

    def test_invalid_distance_and_speed_are_detected(self):
        routed = _finalize_routing_contract(self.frame())
        routed.loc[0, "distance_m"] = -1
        routed.loc[1, "Speed [km/h]"] = np.inf
        errors = validate_routing_output(routed)
        self.assertIn("invalid_distance", errors)
        self.assertIn("invalid_speed", errors)

