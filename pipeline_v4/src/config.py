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
_PRIMARY_GPS = GPS_DIR / "No_asistentes_100000_aleatorios_3_meses.parquet"
FILE_GPS_ORIGINAL = _PRIMARY_GPS if _PRIMARY_GPS.exists() else GPS_DIR / "supplied_gps.parquet"
FILE_GRAFO = INFRA_DIR / "monterrey_drive_network_V1.pkl"
FILE_GRAFO_WALK = INFRA_DIR / "monterrey_walk_network_EydanV1.pkl"
FILE_METRO = INFRA_DIR / "lineas_metrorrey.csv"
FILE_BUS = INFRA_DIR / "rutas_buses_ZMM_oficial.geojson"
FILE_AGEB = INFRA_DIR / "AGEB" / "AGEB_ZMM.json"

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
# Number of user-day task frames retained by one routing window. This bounds
# task/result lifetime without changing task contents or routing behavior.
USER_DAY_BATCH_SIZE = int(os.getenv("USER_DAY_BATCH_SIZE", "32"))
ENABLE_BAYES_FALLBACK = False


# Execution Profiles for parallelism and batching control
EXECUTION_PROFILES = {
    "LOCAL_SAFE": {
        "backend": "threading",
        "n_jobs": 2,
        "user_day_batch_size": USER_DAY_BATCH_SIZE,
        "description": "Safe default for personal PC (Windows, 8 GB RAM).",
    },
    "LAB_THREADING_MODERATE": {
        "backend": "threading",
        "n_jobs": 8,
        "user_day_batch_size": 1000,
        "description": "Conservative multithreading benchmark for 128 GB lab workstation.",
    },
    "LAB_THREADING_AGGRESSIVE": {
        "backend": "threading",
        "n_jobs": 24,
        "user_day_batch_size": 1000,
        "description": "Aggressive multithreading benchmark for 128 GB lab workstation.",
    },
    "LAB_PROCESS_TEST": {
        "backend": "process",
        "n_jobs": 4,
        "user_day_batch_size": 500,
        "description": "Experimental process-pool benchmark mode for 128 GB lab workstation.",
    },
    "CUSTOM": {
        "backend": "threading",
        "n_jobs": 2,
        "user_day_batch_size": USER_DAY_BATCH_SIZE,
        "description": "User-configured custom profile.",
    },
}

# Profile Aliases for user convenience
_PROFILE_ALIASES = {
    "local": "LOCAL_SAFE",
    "local_safe": "LOCAL_SAFE",
    "lab_threading_moderate": "LAB_THREADING_MODERATE",
    "lab_moderate": "LAB_THREADING_MODERATE",
    "lab_threading_aggressive": "LAB_THREADING_AGGRESSIVE",
    "lab_aggressive": "LAB_THREADING_AGGRESSIVE",
    "lab_threading": "LAB_THREADING_AGGRESSIVE",
    "lab_process_test": "LAB_PROCESS_TEST",
    "lab_process": "LAB_PROCESS_TEST",
    "custom": "CUSTOM",
}

DEFAULT_EXECUTION_PROFILE = os.getenv("EXECUTION_PROFILE", "LOCAL_SAFE").strip().upper()


def resolve_execution_profile(
    profile: str | None = None,
    *,
    n_jobs_override: int | None = None,
    backend_override: str | None = None,
    batch_size_override: int | None = None,
) -> dict:
    """Resolve execution configuration based on explicit overrides > profile defaults > fallback defaults."""
    raw_profile = (profile or DEFAULT_EXECUTION_PROFILE or "LOCAL_SAFE").strip()
    canonical_key = _PROFILE_ALIASES.get(raw_profile.lower(), raw_profile.upper())
    base = dict(EXECUTION_PROFILES.get(canonical_key, EXECUTION_PROFILES["CUSTOM"]))

    requested_backend = backend_override if backend_override is not None else base.get("backend", "threading")
    requested_n_jobs = n_jobs_override if n_jobs_override is not None else base.get("n_jobs", 2)
    requested_batch_size = (
        batch_size_override if batch_size_override is not None else base.get("user_day_batch_size", USER_DAY_BATCH_SIZE)
    )

    # Backend normalization
    normalized_backend = str(requested_backend).strip().lower()
    if normalized_backend in {"processes", "process_pool", "processpool", "multiprocessing"}:
        normalized_backend = "process"
    elif normalized_backend in {"threads", "thread", "threadpool"}:
        normalized_backend = "threading"
    elif normalized_backend in {"serial", "sequential", "none"}:
        normalized_backend = "threading"
        requested_n_jobs = 1
    if normalized_backend not in {"threading", "process"}:
        raise ValueError(f"Unsupported execution backend: {requested_backend!r}")

    effective_n_jobs = int(requested_n_jobs)
    effective_batch_size = int(requested_batch_size)
    if effective_n_jobs <= 0:
        raise ValueError("n_jobs must be a positive integer")
    if effective_batch_size <= 0:
        raise ValueError("user_day_batch_size must be a positive integer")

    # Hardware memory inspection & guardrails
    guardrail_warning = None
    total_ram_gb = 8.0
    avail_ram_gb = 4.0
    try:
        import psutil
        vm = psutil.virtual_memory()
        total_ram_gb = vm.total / (1024 ** 3)
        avail_ram_gb = vm.available / (1024 ** 3)
    except Exception:
        pass

    if normalized_backend == "process":
        # Each process worker requires ~2.7 GB for graph network structures
        safe_max_workers = max(1, int(avail_ram_gb // 2.7))
        if total_ram_gb < 16.0 and effective_n_jobs > safe_max_workers:
            guardrail_warning = (
                f"Process backend requested with {effective_n_jobs} workers on a {total_ram_gb:.1f} GB RAM system "
                f"({avail_ram_gb:.1f} GB available). Capping effective workers to {safe_max_workers} to prevent OOM/swapping."
            )
            effective_n_jobs = safe_max_workers

    return {
        "execution_profile": canonical_key,
        "backend": normalized_backend,
        "requested_n_jobs": int(requested_n_jobs),
        "effective_n_jobs": int(effective_n_jobs),
        "user_day_batch_size": int(effective_batch_size),
        "guardrail_warning": guardrail_warning,
        "total_ram_gb": round(total_ram_gb, 2),
        "available_ram_gb": round(avail_ram_gb, 2),
    }


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
