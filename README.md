# GPS Trajectory Processing and Emissions Estimation Pipeline

## Project purpose

This repository contains the frozen V4 production pipeline for preprocessing
GPS trajectories, reconstructing routes, classifying transport modes, and
estimating criteria-pollutant and greenhouse-gas emissions. The canonical entry
point is `notebooks/GPS_preprocessing_and_pipeline_v4.ipynb`.

## Short pipeline overview

```text
GPS observations
  -> quality control, home inference, and AGEB assignment
  -> trip segmentation and route reconstruction
  -> transport-mode classification
  -> MOVES emission-rate lookup
  -> summary, detailed, and trip-ledger outputs
```

The frozen contract uses optimized routing V2, nearest-edge candidates plus
exact-distance ties, V2 endpoint preservation, component splitting, a bounded
lookahead of 10 skipped pings, and `N_JOBS = 2`. Route outcomes are `complete`,
`partial`, or `failed`; failed routes are not eligible for emissions.

Routing processes user-day tasks in bounded windows of 32 by default
(`USER_DAY_BATCH_SIZE` can be overridden for memory-constrained deployments).
This changes task lifetime and progress reporting only; routing inputs and
scientific outputs remain unchanged.

## Repository structure

```text
README.md
requirements-production.txt
docs/                              # Production contract and release metadata
Inputs/                            # Public lookup plus private-input instructions
notebooks/                         # Canonical production notebook
pipeline_v4/
  preprocessing/                   # GPS preparation and residence metadata
  src/                             # Routing, classification, emissions, and outputs
  calibration_and_diagnostics/     # Scientific validation assets and serving models
tests/                             # Public smoke and contract tests
```

Generated outputs, private inputs, caches, and local review reports are excluded
from version control.

## Environment setup

Use Python 3.12. Production package versions, including scikit-learn 1.5.2, are
pinned in `requirements-production.txt`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-production.txt
py -3.12 -m pytest -q
```

Open the notebook with Jupyter or an IDE configured to use this environment.
The workflow validates Python and scikit-learn before loading model artifacts.

## Required inputs

The public repository includes:

- `Inputs/Emission rates/cleaned_emission_rates_formatted_SB.parquet`;
- the classifier artifacts and Bayesian matrices under
  `pipeline_v4/calibration_and_diagnostics/modal_classification/`.

Copy the ten private resources into these exact relative paths:

```text
Inputs/GPS User Data/supplied_gps.parquet
Inputs/Infrastructure/AGEB/AGEB_ZMM.json
Inputs/Infrastructure/monterrey_drive_network_V1.pkl
Inputs/Infrastructure/monterrey_walk_network_EydanV1.pkl
Inputs/Infrastructure/lineas_metrorrey.csv
Inputs/Infrastructure/rutas_buses_ZMM_oficial.geojson
Inputs/Infrastructure/Cache_Optimizado/edges_drive.parquet
Inputs/Infrastructure/Cache_Optimizado/edges_walk.parquet
Inputs/Infrastructure/Cache_Optimizado/ig_drive_y_map.pkl
Inputs/Infrastructure/Cache_Optimizado/ig_walk_y_map.pkl
```

These paths are verified against the production configuration and resource
loader. See `Inputs/README_INPUTS.md` for schemas, spatial-reference contracts,
and private-bundle verification guidance. Pickle resources must come from the
trusted project handoff and the graph, edge, and igraph resources must belong to
the same network build.

## How to run production

1. Extract or clone the repository and merge the private bundle into `Inputs/`.
2. Open `notebooks/GPS_preprocessing_and_pipeline_v4.ipynb` from the repository
   root.
3. Review `SUPPLIED_GPS_PATH`, `AGEB_PATH`, acquisition coverage, optional
   user/date limits, persistence, and `OUTPUT_MODE`.
4. Keep `N_JOBS = 2`, set `EXECUTE_PRODUCTION_RUN = True`, and run all cells.

All paths are repository-relative or configurable. Each execution writes only
to `Outputs/runs/<run_id>/`.

Optional post-run quality diagnostics can be enabled from the canonical
notebook and can also be generated for an existing run directory.

## Main outputs

```text
Outputs/runs/<run_id>/
  manifest.json
  figures/
  preprocessing/
    supplied_users.parquet
    user_home_metadata.parquet
    preprocessed_gps.parquet        # optional
    preprocessing_manifest.json
  pipeline/
    routes_emissions_summary.parquet   # when requested
    routes_emissions_detailed.parquet  # when requested
    trip_ledger.parquet
    pipeline_manifest.json
```

Residence metadata is preserved independently of routing eligibility. The trip
ledger records processing, route-quality, classification, and emissions status
for every trip.

## Calibration and diagnostics note

`pipeline_v4/calibration_and_diagnostics/` contains the public scientific
validation sources and versioned serving artifacts. Production inference loads
validated artifacts; it does not train or overwrite them. The detailed frozen
contract and static release record are in `docs/production-contract.md` and
`docs/production-v4-manifest.json`.
