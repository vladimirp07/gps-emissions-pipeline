# GPS Trajectory Processing and Emissions Estimation Pipeline

This repository contains the frozen V4 production pipeline used to preprocess
GPS trajectories, reconstruct routes, classify transport modes, and estimate
criteria-pollutant and greenhouse-gas emissions. The workflow is intended for
reproducible research runs on externally supplied GPS and geographic resources.

The canonical entry point is
`notebooks/GPS_preprocessing_and_pipeline_v4.ipynb`.

## Pipeline overview

```text
GPS input
  -> quality control, home inference, and AGEB assignment
  -> trip segmentation
  -> map matching and route reconstruction
  -> hierarchical modal classification
  -> MOVES emission-rate lookup
  -> run-scoped outputs and trip ledger
```

The frozen production contract uses optimized routing V2, nearest-edge
candidates with exact-distance ties, V2 endpoint preservation, component
splitting, a maximum lookahead of 10 skipped pings, and `N_JOBS = 2`. Route
outcomes are recorded as `complete`, `partial`, or `failed`; failed routes are
not eligible for emissions estimation.

Detailed contracts and release metadata are available in
[docs/production-contract.md](docs/production-contract.md) and
[docs/production-v4-manifest.json](docs/production-v4-manifest.json).

## Repository structure

```text
README.md
requirements-production.txt
docs/                              # Detailed production contract and frozen manifest
notebooks/                         # Canonical production notebook
pipeline_v4/
  preprocessing/                   # GPS QC, home inference, and AGEB metadata
  src/                             # Production routing, classification, and emissions
  calibration_and_diagnostics/     # Scientific reproducibility sources and model artifacts
Inputs/                            # Versioned runtime assets only
tests/                             # Small public smoke and contract suite
```

Generated outputs, raw GPS data, network files, geographic layers, caches, and
local reports are excluded from version control.

## Environment

Production requires Python 3.12 and scikit-learn 1.5.2. All production package
versions are pinned in `requirements-production.txt`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-production.txt
```

Use a Jupyter installation or IDE whose kernel points to this environment. The
notebook validates Python and scikit-learn versions before loading the
classifier.

## Required inputs

### Included in the repository

- `Inputs/Emission rates/cleaned_emission_rates_formatted_SB.parquet`: the
  588 KB production emission-rate lookup required by `config.FILE_MOVES`.
- Persisted hybrid and Random Forest serving models, with manifests, under
  `pipeline_v4/calibration_and_diagnostics/modal_classification/artifacts/`.
- Bayesian probability matrices used by the supported alternative backend.

The emission-rate table is a processed lookup based on the U.S. EPA Motor
Vehicle Emission Simulator (MOVES). EPA explains that MOVES emission-rate mode
produces rates by average speed bin; EPA-produced data are generally public
domain unless otherwise specified. See the
[EPA MOVES overview](https://www.epa.gov/moves) and
[EPA data licensing information](https://pasteur.epa.gov/license/sciencehub-license-non-epa-generated.html).

### Must be provided locally

Place these resources under `Inputs/` using the paths configured in
`pipeline_v4/src/config.py`:

- a supplied GPS Parquet, configured in the canonical notebook;
- a matching AGEB boundary file, configured in the canonical notebook;
- `Infrastructure/monterrey_drive_network_V1.pkl`;
- `Infrastructure/monterrey_walk_network_EydanV1.pkl`;
- `Infrastructure/lineas_metrorrey.csv`;
- `Infrastructure/rutas_buses_ZMM_oficial.geojson`;
- `Infrastructure/Cache_Optimizado/edges_drive.parquet`;
- `Infrastructure/Cache_Optimizado/edges_walk.parquet`;
- `Infrastructure/Cache_Optimizado/ig_drive_y_map.pkl`;
- `Infrastructure/Cache_Optimizado/ig_walk_y_map.pkl`.

These resources may be large, location-specific, or subject to separate data
agreements and are intentionally not distributed here.

## Run the pipeline

1. Start Jupyter from the repository root:

   ```powershell
   jupyter notebook notebooks/GPS_preprocessing_and_pipeline_v4.ipynb
   ```

2. Set `SUPPLIED_GPS_PATH` and `AGEB_PATH` to the local input files.
3. Set the half-open UTC acquisition interval when known. If it is unknown,
   leave the coverage values as `None`; edge-day completeness will be marked
   unknown rather than inferred.
4. Review optional user/date limits, `SAVE_PREPROCESSED_GPS`, and
   `OUTPUT_MODE` (`summary`, `detailed`, or `both`).
5. Keep the production worker setting at `N_JOBS = 2`.
6. Set `EXECUTE_PRODUCTION_RUN = True` and run the notebook.

Each execution receives a unique run ID and writes only under
`Outputs/runs/<run_id>/`.

## Outputs

```text
Outputs/runs/<run_id>/
  manifest.json
  preprocessing/
    supplied_users.parquet
    user_home_metadata.parquet
    preprocessed_gps.parquet        # optional
    preprocessing_manifest.json
  pipeline/
    routes_emissions_summary.parquet   # when requested
    routes_emissions_detailed.parquet  # when requested
    trip_ledger.parquet
    figures/
```

`trip_ledger.parquet` records one processing and quality outcome per trip. The
run manifest records configuration, software, artifact, and Git metadata.
Summary outputs provide the compact analysis schema; detailed outputs retain
routing, quality, component, and lookup audit fields. Residence metadata is
preserved independently of routing eligibility.

## Quality and validation

The public test suite verifies the production environment, module handoff,
serving contract, emissions contract, output schemas, run isolation, trip
ledger, and a lightweight synthetic pipeline smoke flow.

```powershell
py -3.12 -m pytest -q
```

Scientific calibration and reproducibility material is documented under
[`pipeline_v4/calibration_and_diagnostics/`](pipeline_v4/calibration_and_diagnostics/README.md).
Runtime inference loads validated artifacts and does not train or overwrite
models.

## Research context

The pipeline combines GPS-based mobility reconstruction with EPA MOVES
emission-rate outputs for reproducible transport-emissions research. When using
the repository in research, identify the Git commit and production manifest
used for the analysis, and cite the relevant EPA MOVES documentation alongside
the project-specific publication or dataset citation.
