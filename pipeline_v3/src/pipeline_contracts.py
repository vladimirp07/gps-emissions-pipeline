"""Contratos mínimos y validadores entre los módulos del pipeline de producción."""
from __future__ import annotations

import numpy as np
import pandas as pd

PIPELINE_VERSION = "pipeline_v4_production"
EXPECTED_POLLUTANTS = ("CO2", "CO2e", "CO", "HC", "NOx", "PM10", "PM2.5")

ROUTING_REQUIRED_COLUMNS = (
    "physical_trip_id", "caid", "trip", "local_timestamp", "network_hypothesis",
    "start_node", "end_node", "osmid", "distance_m", "Speed [km/h]",
    "duration_s", "snap_distance_m", "ruteo_fallido", "corregido_espacialmente",
    "routing_status", "flag_auditoria",
)
EMISSIONS_INPUT_COLUMNS = (
    "physical_trip_id", "local_timestamp", "modo_transporte", "osmid",
    "distance_m", "Speed [km/h]", "highway",
)


def missing_columns(frame: pd.DataFrame, required) -> list[str]:
    return [column for column in required if column not in frame.columns]


def validate_routing_output(frame: pd.DataFrame) -> list[str]:
    errors = [f"missing:{name}" for name in missing_columns(frame, ROUTING_REQUIRED_COLUMNS)]
    if errors or frame.empty:
        return errors
    timestamps = pd.to_datetime(frame["local_timestamp"], errors="coerce")
    if timestamps.isna().any(): errors.append("invalid_timestamp")
    if not timestamps.is_monotonic_increasing: errors.append("timestamps_not_sorted")
    distance = pd.to_numeric(frame["distance_m"], errors="coerce")
    speed = pd.to_numeric(frame["Speed [km/h]"], errors="coerce")
    if distance.isna().any() or (distance < 0).any(): errors.append("invalid_distance")
    if speed.isna().any() or not np.isfinite(speed).all() or (speed < 0).any() or (speed > 160).any(): errors.append("invalid_speed")
    if frame.duplicated(["physical_trip_id", "local_timestamp", "start_node", "end_node", "osmid"]).any():
        errors.append("duplicate_subsegment")
    return errors


def validate_emissions_input(frame: pd.DataFrame) -> list[str]:
    errors = [f"missing:{name}" for name in missing_columns(frame, EMISSIONS_INPUT_COLUMNS)]
    if errors: return errors
    if (pd.to_numeric(frame["distance_m"], errors="coerce") < 0).any(): errors.append("negative_distance")
    speed = pd.to_numeric(frame["Speed [km/h]"], errors="coerce")
    if speed.isna().any() or not np.isfinite(speed).all() or (speed < 0).any(): errors.append("invalid_speed")
    return errors
