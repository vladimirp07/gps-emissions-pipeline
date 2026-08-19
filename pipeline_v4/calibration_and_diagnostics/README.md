# Production reproducibility assets

This directory contains only files required by the validated production and
rollback contracts.

- `legacy_routing/` preserves the V1-compatible implementation imported by the
  routing module for rollback behavior.
- `modal_classification/artifacts/` contains the versioned classifier artifacts
  and manifests loaded by production serving.
- `modal_classification/calibration/` retains the minimal model-contract sources
  checked by the production test suite.
- `modal_classification/notebooks/` contains the reproducible classifier
  validation notebook; it is not the production pipeline entry point.

Generated calibration caches, figures, local datasets, exploratory notebooks,
and diagnostic reports are intentionally excluded from version control.
