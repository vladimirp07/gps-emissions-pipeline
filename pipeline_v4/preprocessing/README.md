# Production preprocessing

Canonical flow:

`supplied GPS sample → QC/cleaning → home inference → AGEB assignment → user metadata → eligible GPS`

`workflow.py` processes every user contained in the supplied Parquet, or exactly
the optional external `user_ids` list. It does not sample, replace, or search for
additional users. Users remain in `user_home_metadata.parquet` with explicit
processing metadata. Routing eligibility is determined by GPS quality; home
quality is retained as independent residence metadata.

The implementation reuses home inference and AGEB assignment from
`gps_home_sampling/workflow.py` and the production cleaning transformations from
`pipeline_v4/src/segmentation.py`. The spatial-sampling helpers in
`gps_home_sampling` are exploratory/research utilities and are not called by the
production workflow.

Validated production home parameters: `night_start=22`, `night_end=5`,
`cell_m=250`, and `min_nights=3`.

Input timestamps must represent UTC. Unix values are UTC and naive datetime
values are interpreted as UTC before conversion to `America/Monterrey`.
`PreprocessingConfig.coverage_start` and `coverage_end` describe the supplied
half-open acquisition interval. A local edge day is excluded only when that
interval proves it is truncated; without reliable boundaries it is retained
with `day_completeness_status="unknown"`.

Outputs under the configured run directory:

- `supplied_users.parquet`
- `user_home_metadata.parquet`
- `preprocessed_gps.parquet`
- `preprocessing_manifest.json`
