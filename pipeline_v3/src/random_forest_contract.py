"""Contrato único del clasificador modal Random Forest ML v4 de producción."""

MODEL_VERSION = "ML_v4_52"
TRAINING_TRIPS = 66
TRAINING_SCENARIOS = 260
MIN_EFFECTIVE_PINGS = 15
MIN_PCT_CONSERVED = 30.0

RF_FEATURES = (
    "drive_mean_speed", "drive_max_speed", "drive_std_speed", "drive_stop_frac",
    "drive_p25_speed", "drive_p50_speed", "drive_p75_speed",
    "drive_max_speed_diff", "drive_mean_speed_diff",
    "drive_highway_motorway_frac", "drive_highway_residential_frac",
    "drive_near_bus_frac", "drive_near_metro_frac",
    "drive_near_bus_drift_decay", "drive_near_bus_high_drift",
    "drive_num_stops", "drive_mean_stop_duration", "drive_mean_stop_interval",
    "drive_std_stop_interval",
    "walk_mean_speed", "walk_max_speed", "walk_std_speed", "walk_highway_footway_frac",
    "walk_p25_speed", "walk_p50_speed", "walk_p75_speed",
    "walk_max_speed_diff", "walk_mean_speed_diff",
    "metro_mean_speed", "metro_max_speed", "metro_near_metro_frac",
    "metro_p25_speed", "metro_p50_speed", "metro_p75_speed",
    "metro_max_speed_diff", "metro_mean_speed_diff",
    "mean_snap_dist_drive", "max_snap_dist_drive", "std_snap_dist_drive",
    "mean_snap_dist_walk", "max_snap_dist_walk", "std_snap_dist_walk",
    "metro_win_near_metro_max", "metro_win_near_metro_p90",
    "metro_win_near_metro_consec_run",
    "drive_win_near_bus_max", "drive_win_near_bus_p90",
    "drive_win_near_bus_consec_run",
    "drive_win_stops_max", "drive_win_stops_consec_run",
    "walk_win_walk_regime_max", "walk_win_walk_regime_consec_run",
)

EXPERIMENTAL_BUS_FEATURES = (
    "stop_cycles_per_km", "median_stop_spacing_m", "cv_stop_spacing",
    "median_restart_time_s", "p90_restart_time_s", "stop_pattern_persistence",
)

RF_HYPERPARAMETERS = {
    "n1": {"n_estimators": 100, "max_depth": 7, "min_samples_leaf": 4,
           "random_state": 42, "class_weight": "balanced"},
    "n2": {"n_estimators": 100, "max_depth": 7, "min_samples_leaf": 4,
           "random_state": 42, "class_weight": "balanced"},
    "n3": {"n_estimators": 100, "max_depth": 7, "min_samples_leaf": 2,
           "random_state": 42, "class_weight": "balanced"},
}

assert len(RF_FEATURES) == 52
assert not set(RF_FEATURES).intersection(EXPERIMENTAL_BUS_FEATURES)
