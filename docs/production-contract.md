# pipeline_v4 production release

**Status: READY FOR CONTROLLED PRODUCTION**

The production implementation lives under `pipeline_v4/`. Its logical release
identifier is defined by `config.PIPELINE_RELEASE`.

The official routing baseline is optimized V2. It preserves the validated V2
decisions and outputs with `n_jobs=2`, a bounded lookahead of 10 skipped pings,
component splitting, V2 endpoint preservation, and V1 rollback support.

## Workflow

`GPS -> segmentation -> map matching -> hierarchical hybrid modal classifier -> MOVES lookup -> subsegment emissions`

## Module contracts

- Routing emits the physical trip identifier, timestamps, network hypotheses,
  nodes and edges, `osmid`, distance, speed, duration, and explicit routing and
  snapping states.
- Classification consumes routed hypotheses and emits the class,
  probabilities, backend, version, quality status, and rejection reason through
  `evaluate_with_contract`.
- Emissions consumes mode, `osmid`, distance, speed, road type, and time. It
  emits `g/km` rates, calculated distance in km, totals in grams, and lookup
  status.

The shared validators are defined in
`pipeline_v4/src/pipeline_contracts.py`.

## Optimized V2 routing baseline

The production implementation avoids redundant work through a compressed
incident-edge index, batched endpoint snapping, a deterministic bounded
shortest-path cache, vectorized candidate assembly, persistent spatial indexes,
a run-scoped infrastructure-proximity cache, and a lazy thread-safe
edge-attribute cache.

The frozen production comparison dated August 18, 2026 confirmed exact
equivalence for candidates, routes, WKT, distances, components, statuses,
modes, features, emissions, ledger, summary, and detailed outputs.

Candidate generation uses `sjoin_nearest`: it returns the nearest edge and any
exact-distance ties within the configured radius, capped by `max_cands`. It is
not a K-nearest query. Production therefore remains nearest-only/ties.

## Modal classifier

- Default: `hybrid` -- N1 Gradient Boosting (16 features), N2 Random Forest
  (52 features), and N3 Extra Trees (25 features) with a 0.50 Bus threshold.
- Rollback: `random_forest`.
- Alternative: `bayes`.

Select the backend in `pipeline_v4/src/config.py` or with
`MODAL_CLASSIFIER=hybrid|random_forest|bayes`.

## Quality guardrail

Classification requires at least 15 effective pings and 30 percent of the
original trajectory to remain after cleaning. Trips that do not meet this
contract are rejected with a `quality_guardrail` reason.

## Environment and execution

Production requires Python 3.12 and scikit-learn 1.5.2. Exact dependency
versions are pinned in `requirements-production.txt`, and the canonical
workflow validates the environment before loading persisted artifacts.

Input timestamps must represent UTC. Naive timestamps are interpreted as UTC.
Acquisition coverage is configured as a half-open interval so that only local
edge days proven to be truncated are excluded.

Run the reusable validation suite with:

```powershell
py -3.12 -m pytest -q
```

Use `notebooks/GPS_preprocessing_and_pipeline_v4.ipynb` as the production entry
point. Review its input paths and acquisition coverage, then enable the explicit
execution switch. Each execution writes only to its own
`Outputs/runs/<run_id>/` directory.

## Known limitations

- The V1 rollback router does not persist numeric snap distance per subsegment.
- `max_cands` caps nearest/tied results; it does not guarantee K nearby edges.
- Persisted classifier artifacts must be loaded with scikit-learn 1.5.2.

## Release status

- Status: READY FOR CONTROLLED PRODUCTION
- Production blocker: none in the validated release
- Required operating baseline: optimized V2, nearest-only/ties, `N_JOBS = 2`,
  and normal monitoring of route quality, classification, and emissions
