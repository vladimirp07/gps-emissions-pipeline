import os
from pathlib import Path

# Project root (two levels above src/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_RELEASE = "pipeline_v4_production"

# Input directories.
INPUTS_DIR = PROJECT_ROOT / "Inputs"
INFRA_DIR = INPUTS_DIR / "Infrastructure"
CACHE_INFRA_DIR = INFRA_DIR / "Cache_Optimizado"
GPS_DIR = INPUTS_DIR / "GPS User Data"
MODAL_ARTIFACTS_DIR = PROJECT_ROOT / "pipeline_v4" / "calibration_and_diagnostics" / "modal_classification" / "artifacts"
RATES_DIR = INPUTS_DIR / "Emission rates"

# Output directories.
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
FINAL_DIR = OUTPUTS_DIR / "Final Outputs"
CALIBRATION_CACHE_DIR = OUTPUTS_DIR / "calibration_cache"
MODAL_ROUTE_CACHE_DIR = CALIBRATION_CACHE_DIR / "modal_classification" / "route_hypotheses"
MODAL_ROUTE_CACHE_BASELINE = MODAL_ROUTE_CACHE_DIR / "baseline"
MODAL_ROUTE_CACHE_EXPANDED = MODAL_ROUTE_CACHE_DIR / "expanded"

# Read-only compatibility paths for workspaces created before route caches were
# correctly classified as derived outputs. New code must never write here.
LEGACY_MODAL_ROUTE_CACHE_BASELINE = GPS_DIR / "cache_rutas_completas"
LEGACY_MODAL_ROUTE_CACHE_EXPANDED = GPS_DIR / "cache_rutas_completas_expanded"

# Input file paths.
FILE_GPS_ORIGINAL = GPS_DIR / "top_20users_1_month.parquet"
FILE_GRAFO = INFRA_DIR / "monterrey_drive_network_V1.pkl"
FILE_GRAFO_WALK = INFRA_DIR / "monterrey_walk_network_EydanV1.pkl"
FILE_METRO = INFRA_DIR / "lineas_metrorrey.csv"
FILE_BUS = INFRA_DIR / "rutas_buses_ZMM_oficial.geojson"

# Network-cache paths.
FILE_CACHE_EDGES_DRIVE = CACHE_INFRA_DIR / "edges_drive.parquet"
FILE_CACHE_EDGES_WALK = CACHE_INFRA_DIR / "edges_walk.parquet"
FILE_CACHE_IG_DRIVE = CACHE_INFRA_DIR / "ig_drive_y_map.pkl"
FILE_CACHE_IG_WALK = CACHE_INFRA_DIR / "ig_walk_y_map.pkl"

# MOVES emission-rate path.
FILE_MOVES = RATES_DIR / "cleaned_emission_rates_formatted_SB.parquet"
# Units used by the production emissions schema.
EMISSION_RATE_DISTANCE_UNIT = "g/km"
EMISSION_DISTANCE_UNIT = "km"
EMISSION_TOTAL_UNIT = "g"

# Modal backend selection without modifying the orchestrator.
# Valid values: hybrid (official), random_forest (rollback), bayes.
MODAL_CLASSIFIER = os.getenv("MODAL_CLASSIFIER", "hybrid").strip().lower()
ROUTER_VERSION = os.getenv("ROUTER_VERSION", "v2").strip().lower()
MAX_LOOKAHEAD_SKIPPED_PINGS = int(os.getenv("MAX_LOOKAHEAD_SKIPPED_PINGS", "10"))
ENABLE_BAYES_FALLBACK = False


def _optional_int_env(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"", "none", "null"}:
        return None
    return int(normalized)


# ``None`` relaxes the recurrence requirement used to assess a candidate but
# never turns a one-night candidate into probable/reliable evidence.
HOME_MIN_NIGHTS = _optional_int_env("HOME_MIN_NIGHTS", 3)

FILE_MODAL_HYBRID = MODAL_ARTIFACTS_DIR / "modal_classifier_hybrid_v1.pkl"
FILE_MODAL_RANDOM_FOREST = MODAL_ARTIFACTS_DIR / "random_forest_modal.pkl"
