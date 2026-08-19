# Modal-classification production assets

This directory contains the versioned artifacts and reproducibility sources
required by the production modal-classification serving contract.

## Serving artifacts

- `artifacts/modal_classifier_hybrid_v1.pkl`: default hierarchical hybrid model.
- `artifacts/random_forest_modal.pkl`: supported Random Forest rollback model.
- The adjacent manifest files record artifact metadata and checksums.

Runtime inference validates and loads these artifacts; it does not train or
overwrite them.

## Reproducibility sources

- `calibration/bayes/matrices_optimas.json` contains the calibrated probability
  matrices used by the supported Bayesian alternative.
- `calibration/random_forest/entrenar_random_forest.py` preserves the official
  feature and training contract validated by the test suite.
- `notebooks/playground_modal_classifier.ipynb` reproduces grouped classifier
  validation. It is not the production pipeline entry point.

The default cascade is N1 Gradient Boosting (16 features), N2 Random Forest
(52 features), and N3 Extra Trees (25 features) with a 0.50 Bus threshold.
Select `hybrid`, `random_forest`, or `bayes` through `MODAL_CLASSIFIER`.
