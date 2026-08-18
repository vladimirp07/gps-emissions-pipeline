import json
from pathlib import Path

import pandas as pd

from pipeline_v4.preprocessing import (
    PreprocessingConfig,
    attach_user_metadata,
    run_preprocessing,
    supplied_user_ids,
)
from pipeline_v4.preprocessing.gps_home_sampling.workflow import HomeConfig


def _raw_sample(user_ids=(101, 202)):
    rows = []
    for user_id in user_ids:
        for day in range(1, 7):
            for minute in (0, 1):
                timestamp = pd.Timestamp(f"2019-12-{day:02d} 04:{minute:02d}:00", tz="UTC")
                rows.append({
                    "caid": user_id,
                    "latitude": 25.6800 + user_id / 10_000_000,
                    "longitude": -100.3100,
                    "utc_timestamp": int(timestamp.timestamp()),
                    "horizontal_accuracy": 5.0,
                })
    return pd.DataFrame(rows)


def _ageb_geojson(path: Path):
    payload = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"CVEGEO": "190390001001", "POBTOT": 1000},
            "geometry": {"type": "Polygon", "coordinates": [[
                [-100.32, 25.67], [-100.30, 25.67], [-100.30, 25.69],
                [-100.32, 25.69], [-100.32, 25.67],
            ]]},
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validated_home_window_is_the_default():
    config = HomeConfig()
    assert (config.night_start, config.night_end) == (22, 5)


def test_lax_home_configuration_keeps_one_night_low_confidence():
    gps = _raw_sample((101,)).iloc[:2]
    from pipeline_v4.preprocessing.gps_home_sampling.workflow import infer_home_locations, standardize_gps

    home = infer_home_locations(standardize_gps(gps), HomeConfig(min_nights=None)).iloc[0]
    assert pd.notna(home.home_lat)
    assert home.home_quality_flag == "insufficient_information"


def test_supplied_dataset_uses_every_user_without_sampling(tmp_path):
    source = tmp_path / "sample.parquet"
    _raw_sample().to_parquet(source, index=False)
    users = supplied_user_ids(source, PreprocessingConfig())
    assert users.user_id.tolist() == [101, 202]
    assert users.input_source.unique().tolist() == ["all_users_in_supplied_dataset"]
    explicit = supplied_user_ids(source, PreprocessingConfig(), [202, 101, 999])
    assert explicit.user_id.tolist() == [202, 101, 999]
    assert explicit.input_source.unique().tolist() == ["explicit_user_list"]


def test_preprocessing_builds_master_and_preserves_supplied_ids(tmp_path):
    source = tmp_path / "sample.parquet"
    ageb = tmp_path / "ageb.geojson"
    output = tmp_path / "processing"
    _raw_sample().to_parquet(source, index=False)
    _ageb_geojson(ageb)
    result = run_preprocessing(source, ageb, output)
    assert result.supplied_users.user_id.tolist() == [101, 202]
    assert result.user_metadata.user_id.tolist() == [101, 202]
    assert set(result.user_metadata.processing_status) == {"ready_for_pipeline"}
    assert result.user_metadata.routing_eligible.all()
    assert result.user_metadata.home_eligible_for_inventory.all()
    assert set(result.user_metadata.home_ageb) == {"190390001001"}
    assert set(result.preprocessed_gps.user_id) == {101, 202}
    assert set(result.preprocessed_gps.caid) == {101, 202}
    assert (output / "supplied_users.parquet").exists()
    assert (output / "user_home_metadata.parquet").exists()
    assert (output / "preprocessed_gps.parquet").exists()


def test_explicit_absent_user_is_reported_not_replaced(tmp_path):
    source = tmp_path / "sample.parquet"
    ageb = tmp_path / "ageb.geojson"
    _raw_sample((101,)).to_parquet(source, index=False)
    _ageb_geojson(ageb)
    result = run_preprocessing(source, ageb, tmp_path / "out", user_ids=[101, 999])
    absent = result.user_metadata.set_index("user_id").loc[999]
    assert absent.processing_status == "no_valid_gps"
    assert not bool(absent.routing_eligible)
    assert absent.raw_records == 0
    assert 999 not in set(result.preprocessed_gps.user_id)


def test_metadata_attachment_keeps_canonical_user_id():
    routes = pd.DataFrame({"caid": [101, 101], "distance_m": [10.0, 20.0]})
    metadata = pd.DataFrame({
        "user_id": [101], "home_lat": [25.68], "home_lon": [-100.31],
        "home_ageb": ["A"], "processing_status": ["ready_for_pipeline"],
    })
    attached = attach_user_metadata(routes, metadata)
    assert attached.user_id.tolist() == [101, 101]
    assert attached.caid.tolist() == [101, 101]


def test_low_confidence_home_does_not_block_routing(tmp_path):
    source = tmp_path / "sample.parquet"
    ageb = tmp_path / "ageb.geojson"
    output = tmp_path / "processing"
    one_night = pd.DataFrame([
        {
            "caid": 101,
            "latitude": 25.6800 + step * 0.0003,
            "longitude": -100.3100,
            "utc_timestamp": int(pd.Timestamp(
                f"2019-12-01 04:{step:02d}:00", tz="UTC"
            ).timestamp()),
            "horizontal_accuracy": 5.0,
        }
        for step in range(4)
    ])
    one_night.to_parquet(source, index=False)
    _ageb_geojson(ageb)

    result = run_preprocessing(source, ageb, output, home_config=HomeConfig(min_nights=3))
    user = result.user_metadata.iloc[0]
    assert user.home_quality_flag == "insufficient_information"
    assert not bool(user.home_eligible_for_inventory)
    assert bool(user.routing_eligible)
    assert user.processing_status == "ready_for_pipeline"
    assert set(result.preprocessed_gps.caid) == {101}
