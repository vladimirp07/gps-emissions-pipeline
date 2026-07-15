import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_v4.src.emissions import calculate_emissions


POLLUTANTS = ["CO", "CO2", "CO2_Equiv", "HC", "NOx", "PM10", "PM25"]


class TestEmissionsContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.lookup = Path(self.tmp.name) / "moves.parquet"
        rows = []
        for source in (21, 42):
            for speed_bin in range(1, 17):
                row = {"Day": 5, "Hour": 9, "Road": 4, "Source": source, "SpeedBin": speed_bin}
                row.update({p: float(source + speed_bin) for p in POLLUTANTS})
                rows.append(row)
        pd.DataFrame(rows).to_parquet(self.lookup, index=False)

    def tearDown(self): self.tmp.cleanup()

    def routes(self, mode="Carro", highway="primary"):
        return pd.DataFrame({
            "physical_trip_id": ["T_1", "T_1"], "caid": ["T", "T"], "trip": [1, 1],
            "local_timestamp": pd.to_datetime(["2026-07-15 08:00", "2026-07-15 08:01"]),
            "modo_transporte": [mode, mode], "osmid": [1, 2], "distance_m": [1000.0, 500.0],
            "Speed [km/h]": [20.0, 40.0], "highway": [highway, highway],
        })

    def test_g_per_km_times_km_equals_grams(self):
        result = calculate_emissions(self.routes(), self.lookup)
        self.assertTrue((result.emission_rate_unit == "g/km").all())
        self.assertTrue((result.distance_unit == "m").all())
        self.assertTrue((result.emission_total_unit == "g").all())
        np.testing.assert_allclose(result.Total_CO_g, result.Densidad_CO_g_km * result.distance_m / 1000)
        self.assertAlmostEqual(result.Total_CO_g.sum(), sum(result.Densidad_CO_g_km * result.distance_m / 1000))
        self.assertTrue((result.filter(regex="^Total_").select_dtypes("number") >= 0).all().all())
        for name in ("CO2", "CO2e", "CO", "HC", "NOx", "PM10", "PM2.5"):
            self.assertIn(f"Total_{name}_g", result)

    def test_speed_bins_road_mapping_and_missing_lookup_are_explicit(self):
        result = calculate_emissions(self.routes(highway="unknown_road"), self.lookup)
        self.assertTrue((result.road_lookup_status == "default_road_5").all())
        self.assertTrue(result.emission_lookup_status.isin(["imputed_speed_curve", "imputed_source", "missing_zero_fallback"]).all())

    def test_negative_distance_fails(self):
        routes = self.routes(); routes.loc[0, "distance_m"] = -1
        with self.assertRaisesRegex(ValueError, "negative_distance"):
            calculate_emissions(routes, self.lookup)


