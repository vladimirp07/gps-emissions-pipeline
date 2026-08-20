# Pipeline V4 Results Report

The results report is optional, read-only post-processing for a completed
`pipeline_v4` run. It summarizes routing outcomes, route quality, modes,
distances, routed speeds, emissions, and lookup coverage. It never reruns or
changes routing, classification, emissions, MOVES, or production outputs.

## Notebook controls

| Variable | Meaning |
| --- | --- |
| `ENABLE_EXTENDED_QUALITY_REPORT` | `False` skips the report. `True` runs it after the pipeline or against an existing run. |
| `QUALITY_REPORT_RUN_PATH` | `None` uses the run created in the current notebook session. Set a run directory to analyze existing outputs without rerunning production. |
| `QUALITY_REPORT_OUTPUT_DIR` | `None` writes to `<run>/quality_report/`. Set a `Path` to use another destination. |
| `QUALITY_REPORT_OVERWRITE` | `False` protects a non-empty report directory. Set `True` only to replace managed report files intentionally. |

## Use the current notebook run

```python
ENABLE_EXTENDED_QUALITY_REPORT = True
QUALITY_REPORT_RUN_PATH = None
QUALITY_REPORT_OUTPUT_DIR = None
QUALITY_REPORT_OVERWRITE = False
```

For all five figures and complete lookup diagnostics, use:

```python
OUTPUT_MODE = "both"
```

## Analyze an existing run

No preprocessing or routing is required:

```python
EXECUTE_PRODUCTION_RUN = False
ENABLE_EXTENDED_QUALITY_REPORT = True
QUALITY_REPORT_RUN_PATH = ROOT / "Outputs" / "runs" / "<run_id>"
```

The Python API also accepts explicit `summary_path`, `detailed_path`,
`ledger_path`, and `manifest_path` arguments.

## Input artifacts

- `trip_ledger.parquet` is required and provides one row per physical trip.
- `routes_emissions_summary.parquet` supports modal, distance, speed, and
  emissions summaries.
- `routes_emissions_detailed.parquet` adds component-safe segment sinuosity,
  detailed lookup coverage, and summary/detailed consistency checks.
- The run manifest supplies run metadata when available.

A summary-only run remains supported. Diagnostics that require detailed rows
are reported as unavailable rather than inferred.

## Report populations

**Pipeline-valid routes** are production trips whose
`route_completeness_status` is `complete` or `partial`. Failed,
`quality_rejected`, and `routing_failed` trips do not enter valid-route plots.

**Plot-quality routes** are pipeline-valid trips with usable quality fields,
after an optional dataset-relative severe-tail filter. Every exclusion and the
retained percentage are recorded. This analytical filter does not change
completeness, `emissions_eligible`, or any production artifact.

## Generated content

```text
<run>/quality_report/
    figures/
    tables/
    quality_report.md
    quality_metrics.json
```

Figures:

1. Modal Share of Valid Routed Trips
2. Trip Distance by Transportation Mode
3. Segment-Level Routing Sinuosity
4. Routed Speed Distribution by Transportation Mode
5. Routed Speed Density by Transportation Mode

The Markdown report explains any visual axis clipping and lists the exact
plot-quality retention. Tables are written as CSV and Parquet.

## Verdicts

- `PASS`: no operational error or warning was detected.
- `PASS WITH WARNINGS`: the run is usable, but an anomaly should be reviewed.
- `REVIEW REQUIRED`: an operational inconsistency was detected, such as a
  distance mismatch or malformed emission value.

The verdict is an operational diagnostic, not a scientific validation of the
routing or emissions methodology.

## Direct Python use

```python
from pathlib import Path
from pipeline_v4.diagnostics import (
    display_quality_report_summary,
    generate_quality_report,
)

report = generate_quality_report(
    run_path=Path("Outputs/runs/<run_id>"),
    output_dir=None,
    show_figures=True,
    save_figures=True,
    overwrite=False,
)
display_quality_report_summary(report)
```
