# V4 Production Input Bundle

This directory layout is the input contract for the frozen V4 production
pipeline. Copy private resources into the paths below without renaming them.
Run the canonical notebook from the repository root:
`notebooks/GPS_preprocessing_and_pipeline_v4.ipynb`.

## Required directory structure

```text
Inputs/
├── README_INPUTS.md
├── INPUT_BUNDLE_MANIFEST.csv
├── GPS User Data/
│   └── supplied_gps.parquet
├── Infrastructure/
│   ├── AGEB/
│   │   └── AGEB_ZMM.json
│   ├── monterrey_drive_network_V1.pkl
│   ├── monterrey_walk_network_EydanV1.pkl
│   ├── lineas_metrorrey.csv
│   ├── rutas_buses_ZMM_oficial.geojson
│   └── Cache_Optimizado/
│       ├── edges_drive.parquet
│       ├── edges_walk.parquet
│       ├── ig_drive_y_map.pkl
│       └── ig_walk_y_map.pkl
└── Emission rates/
    └── cleaned_emission_rates_formatted_SB.parquet
```

## Required files

| Resource | Purpose | Distribution |
| --- | --- | --- |
| `GPS User Data/supplied_gps.parquet` | Raw GPS observations supplied to preprocessing | Private/local |
| `Infrastructure/AGEB/AGEB_ZMM.json` | Home-to-AGEB spatial assignment | Private/local |
| `Infrastructure/monterrey_drive_network_V1.pkl` | Drive routing graph | Private/local |
| `Infrastructure/monterrey_walk_network_EydanV1.pkl` | Walk routing graph | Private/local |
| `Infrastructure/lineas_metrorrey.csv` | Metro geometry used by routing and modal classification | Private/local |
| `Infrastructure/rutas_buses_ZMM_oficial.geojson` | Bus-route geometry used by routing and modal classification | Private/local |
| `Infrastructure/Cache_Optimizado/edges_drive.parquet` | Optimized drive candidate-edge table | Private/local, generated cache |
| `Infrastructure/Cache_Optimizado/edges_walk.parquet` | Optimized walk candidate-edge table | Private/local, generated cache |
| `Infrastructure/Cache_Optimizado/ig_drive_y_map.pkl` | Drive igraph object and node map | Private/local, generated cache |
| `Infrastructure/Cache_Optimizado/ig_walk_y_map.pkl` | Walk igraph object and node map | Private/local, generated cache |
| `Emission rates/cleaned_emission_rates_formatted_SB.parquet` | MOVES-derived emission-rate lookup | Included through GitHub |

The two optimized edge tables and two igraph bundles are reproducible caches,
but V4 loads them directly and therefore requires them for a production run.
They must come from the same network build as the corresponding graph pickle.

Classifier serving models and probability matrices are already distributed by
GitHub under `pipeline_v4/calibration_and_diagnostics/`; they are not part of
the private `Inputs/` bundle.

## GPS input schema

The canonical notebook uses the default user column `caid`. The supplied
Parquet must contain:

| Field | Contract |
| --- | --- |
| `caid` | Non-null user/device identifier. To use `user_id`, `device_id`, or `id`, explicitly change `PreprocessingConfig.user_column`. |
| `utc_timestamp`, `local_timestamp`, `timestamp`, `datetime`, or `date` | Observation time. The first matching name in this order is used. |
| `latitude` or `lat` | Numeric WGS84 latitude in `[-90, 90]`. |
| `longitude`, `lon`, or `lng` | Numeric WGS84 longitude in `[-180, 180]`. |

Timestamps must represent UTC. Numeric timestamps are interpreted as Unix
seconds. Naive datetime values are interpreted as UTC; timezone-aware values
are converted to UTC. The workflow then converts observations to
`America/Monterrey` for local-time processing. Rows with missing identifiers,
invalid coordinates, or invalid timestamps are excluded during standardization.

Do not place processed routes or interpolated GPS observations in this raw-input
location.

## Spatial reference systems and format contracts

- GPS coordinates: WGS84 (`EPSG:4326`).
- `AGEB_ZMM.json`: V4 treats JSON geometry as WGS84 (`EPSG:4326`). It must
  contain polygon geometry and one of `CVEGEO`, `CVEGEO_1`, `AGEB`, or
  `CVE_AGEB` as the geographic identifier.
- Drive/walk graph pickles: production-compatible NetworkX graph objects from
  the validated Monterrey network build. Their nodes must match the optimized
  edge and igraph mapping resources.
- `edges_drive.parquet` and `edges_walk.parquet`: GeoParquet in UTM zone 14N
  (`EPSG:32614`) with `u`, `v`, and `geometry` columns.
- Bus routes: a valid GeoJSON with a declared CRS readable by GeoPandas; V4
  projects it to `EPSG:32614` when loading.
- Metro CSV: either a `geometry` or `WKT` column containing WGS84 geometry, or
  `lat` and `lon` columns. V4 interprets these values as `EPSG:4326` and then
  projects them to `EPSG:32614`.
- `ig_drive_y_map.pkl` and `ig_walk_y_map.pkl`: each pickle must deserialize
  to `(igraph_graph, node_mapping)` and correspond to the matching network.

Pickle files can execute code while loading. Accept them only from the trusted
project handoff and do not substitute files from an unknown source.

## Copy or extraction procedure

1. Clone the production repository and check out the agreed production commit.
2. Extract or copy the private bundle into the repository root so its `Inputs/`
   contents merge with the existing `Inputs/` directory.
3. Do not rename resources unless every corresponding path in
   `pipeline_v4/src/config.py` and the notebook configuration is updated.
4. Confirm that `SUPPLIED_GPS_PATH` and `AGEB_PATH` in the canonical notebook
   point to the supplied files.
5. Compare every private file against `INPUT_BUNDLE_MANIFEST.csv` before the
   first run.

## Verification

The manifest records file sizes and SHA-256 hashes. `MISSING` means the file was
not present when the manifest was generated and the bundle is not yet complete.

PowerShell 7:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath "Inputs/Infrastructure/AGEB/AGEB_ZMM.json"
```

Cross-platform Python:

```bash
python -c "from pathlib import Path; import hashlib; p=Path('Inputs/Infrastructure/AGEB/AGEB_ZMM.json'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

The calculated digest must exactly match the manifest. A mismatch indicates a
different, incomplete, or corrupted resource; do not run production until it is
resolved.
