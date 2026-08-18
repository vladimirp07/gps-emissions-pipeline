# Legacy routing variants

This directory contains routing implementations retained for calibration
reproducibility only. They are not production options.

- `complete_route_v2_progressive_legacy.py`: the former progressive V2 tested
  in the MATLAB V1/V2 audit. It produced route outputs equivalent to V1 in
  40/40 paired cases and was slower. Historical calibration scripts import it
  explicitly so that `pipeline_v4.src.routing.complete_route_v2_optimized`
  can unambiguously mean the endpoint-preserving production V2.

Production exposes only:

- `v1`: stable historical baseline;
- `v2`: V1 plus independently validated real-edge endpoint preservation.
