"""Lab Workstation Benchmark Harness for Block 3 Parallelism Scaling.

Runs the exact same supplied user-day workload across execution profiles and worker
configurations to empirically identify the optimal parallelism architecture on the
128 GB lab workstation.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import pandas as pd
import psutil

# Ensure repo root is in python path
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_v4.src import config
from pipeline_v4.src.production_workflow import (
    load_pipeline_resources,
    create_modal_evaluator,
    run_pipeline_v4,
)


def get_system_ram_snapshot():
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / (1024 ** 3), 2),
        "available_gb": round(vm.available / (1024 ** 3), 2),
        "used_pct": vm.percent,
    }


def get_peak_process_rss_mb():
    try:
        p = psutil.Process(os.getpid())
        # Include children processes if any
        total_rss = p.memory_info().rss
        for child in p.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return round(total_rss / (1024 * 1024), 2)
    except Exception:
        return 0.0


def _frame_signature(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    digest = hashlib.sha256()
    digest.update("|".join(available).encode("utf-8"))
    if available:
        hashed = pd.util.hash_pandas_object(
            frame[available].reset_index(drop=True), index=False, categorize=True,
        )
        digest.update(hashed.to_numpy().tobytes())
    return digest.hexdigest()


def _scientific_signature(result) -> tuple:
    return (
        _frame_signature(result.trip_ledger, [
            "physical_trip_id", "effective_ping_count", "pct_pings_conserved",
            "final_mode", "classification_success", "modal_usable",
            "emissions_usable", "emissions_success", "processing_status",
            "route_completeness_status", "failure_reason",
        ]),
        _frame_signature(result.routes, [
            "physical_trip_id", "local_timestamp", "osmid", "distance_m",
            "modo_transporte", "ruteo_fallido", "route_component_id",
        ]),
        _frame_signature(result.individual_emissions, [
            "physical_trip_id", "local_timestamp", "osmid", "distance_m",
            "modo_transporte", "Total_CO2_g", "Total_CO2e_g",
        ]),
    )


def run_lab_benchmark_suite(
    preprocessed_gps_path: str | Path,
    output_dir: str | Path,
    *,
    user_metadata_path: str | Path | None = None,
    limit_users: int = 100,
    limit_days_per_user: int | None = None,
    custom_trials: list[dict[str, Any]] | None = None,
):
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results_csv = out_root / "lab_benchmark_results.csv"
    results_json = out_root / "lab_benchmark_results.json"

    print("=" * 85, flush=True)
    print("BLOCK 3: LAB WORKSTATION PARALLELISM BENCHMARK SUITE", flush=True)
    print("=" * 85, flush=True)
    ram = get_system_ram_snapshot()
    print(f"Host: {platform.node()} | OS: {platform.system()} {platform.release()} | Python: {platform.python_version()}")
    print(f"System Hardware: {psutil.cpu_count(logical=True)} CPU cores | {ram['total_gb']} GB Total RAM ({ram['available_gb']} GB available)", flush=True)
    print(f"Fixed Workload: {limit_users} users from {Path(preprocessed_gps_path).name}", flush=True)

    # Pre-load shared resources for threading trials
    print("\n[Harness] Pre-loading pipeline resources...", flush=True)
    t0 = time.perf_counter()
    shared_resources = load_pipeline_resources()
    shared_resources["modal_evaluator"] = create_modal_evaluator(config.MODAL_CLASSIFIER)
    t_load = time.perf_counter() - t0
    print(f"[Harness] Resources pre-loaded in {t_load:.2f} s.", flush=True)

    # Standard trial matrix
    trials = custom_trials or [
        # Baseline serial
        {"name": "SERIAL_BASELINE", "profile": "LOCAL_SAFE", "n_jobs": 1, "backend": "threading", "batch_size": 500},
        # Threading scaling
        {"name": "THREADING_MODERATE_8T", "profile": "LAB_THREADING_MODERATE", "n_jobs": 8, "backend": "threading", "batch_size": 1000},
        {"name": "THREADING_AGGRESSIVE_16T", "profile": "LAB_THREADING_AGGRESSIVE", "n_jobs": 16, "backend": "threading", "batch_size": 1000},
        {"name": "THREADING_AGGRESSIVE_24T", "profile": "LAB_THREADING_AGGRESSIVE", "n_jobs": 24, "backend": "threading", "batch_size": 1000},
        {"name": "THREADING_AGGRESSIVE_26T", "profile": "LAB_THREADING_AGGRESSIVE", "n_jobs": 26, "backend": "threading", "batch_size": 1000},
        # Process scaling (starts low and controlled)
        {"name": "PROCESS_TEST_2P", "profile": "LAB_PROCESS_TEST", "n_jobs": 2, "backend": "process", "batch_size": 500},
        {"name": "PROCESS_TEST_4P", "profile": "LAB_PROCESS_TEST", "n_jobs": 4, "backend": "process", "batch_size": 500},
        {"name": "PROCESS_TEST_6P", "profile": "LAB_PROCESS_TEST", "n_jobs": 6, "backend": "process", "batch_size": 500},
        {"name": "PROCESS_TEST_8P", "profile": "LAB_PROCESS_TEST", "n_jobs": 8, "backend": "process", "batch_size": 500},
    ]

    records = []
    baseline_tps = None
    baseline_ledger_signature = None

    for idx, trial in enumerate(trials, start=1):
        name = trial["name"]
        profile = trial["profile"]
        n_jobs = trial["n_jobs"]
        backend = trial["backend"]
        batch_size = trial["batch_size"]

        trial_dir = out_root / f"trial_{idx:02d}_{name.lower()}"
        print("\n" + "-" * 85, flush=True)
        print(f"TRIAL {idx}/{len(trials)}: {name} (Profile={profile}, Backend={backend}, n_jobs={n_jobs}, batch={batch_size})", flush=True)
        print("-" * 85, flush=True)

        gc.collect()
        ram_before = get_system_ram_snapshot()
        t_start = time.perf_counter()

        try:
            res = run_pipeline_v4(
                preprocessed_gps=preprocessed_gps_path,
                output_dir=trial_dir,
                user_metadata=user_metadata_path,
                limit_users=limit_users,
                limit_days_per_user=limit_days_per_user,
                resources=shared_resources if backend != "process" else None,
                output_mode="summary",
                show_progress=True,
                execution_profile=profile,
                n_jobs_override=n_jobs,
                backend_override=backend,
                batch_size_override=batch_size,
                resume=False,
            )
            wall_time = time.perf_counter() - t_start
            peak_rss = get_peak_process_rss_mb()
            ledger = res.trip_ledger
            pipeline_manifest = json.loads(
                (trial_dir / "pipeline_manifest.json").read_text(encoding="utf-8")
            )
            user_days = int(pipeline_manifest["user_days"])
            tps = round(user_days / wall_time, 2) if wall_time > 0 else 0.0

            # Scientific regression parity check
            ledger_sig = _scientific_signature(res)
            if baseline_ledger_signature is None:
                baseline_ledger_signature = ledger_sig
                regression_status = "BASELINE"
                baseline_tps = tps
            else:
                regression_status = "PASS" if ledger_sig == baseline_ledger_signature else "FAIL_MISMATCH"

            speedup = round(tps / baseline_tps, 2) if baseline_tps and baseline_tps > 0 else 1.00

            record = {
                "trial_index": idx,
                "trial_name": name,
                "execution_profile": profile,
                "backend": backend,
                "requested_n_jobs": n_jobs,
                "effective_n_jobs": n_jobs,
                "user_day_batch_size": batch_size,
                "total_user_days": user_days,
                "trips": len(ledger),
                "successful_trips": int(ledger["processing_status"].eq("success").sum()),
                "route_rows": len(res.routes),
                "wall_time_seconds": round(wall_time, 2),
                "user_days_per_second": tps,
                "speedup_vs_baseline": speedup,
                "peak_python_rss_mb": peak_rss,
                "scientific_regression": regression_status,
                "status": "COMPLETED",
            }
            records.append(record)
            print(f"[Trial {idx} Complete] Time: {wall_time:.2f} s | Throughput: {tps:.2f} user-days/s | Speedup: {speedup:.2f}x | Parity: {regression_status}", flush=True)

            # Persist intermediate results
            pd.DataFrame(records).to_csv(results_csv, index=False)
            results_json.write_text(json.dumps(records, indent=2), encoding="utf-8")

            # Check stop condition for process scaling
            if backend == "process" and len(records) >= 2:
                prev_proc = [r for r in records[:-1] if r["backend"] == "process"]
                if prev_proc:
                    previous_tps = prev_proc[-1]["user_days_per_second"]
                    gain = (tps - previous_tps) / previous_tps if previous_tps > 0 else float("inf")
                    if gain < 0.05:
                        print(f"\n[Harness Notice] Process scaling throughput improvement was {gain*100:.1f}% (<5%). Stop condition satisfied.", flush=True)
                        break

        except Exception as exc:
            wall_time = time.perf_counter() - t_start
            print(f"[Trial {idx} FAILED] {type(exc).__name__}: {exc}", flush=True)
            record = {
                "trial_index": idx,
                "trial_name": name,
                "execution_profile": profile,
                "backend": backend,
                "requested_n_jobs": n_jobs,
                "effective_n_jobs": n_jobs,
                "user_day_batch_size": batch_size,
                "total_user_days": None,
                "wall_time_seconds": round(wall_time, 2),
                "user_days_per_second": 0.0,
                "speedup_vs_baseline": 0.0,
                "peak_python_rss_mb": get_peak_process_rss_mb(),
                "scientific_regression": f"ERROR: {type(exc).__name__}",
                "status": "FAILED",
            }
            records.append(record)
            pd.DataFrame(records).to_csv(results_csv, index=False)

    print("\n" + "=" * 85, flush=True)
    print("LAB BENCHMARK SUITE COMPLETE — SUMMARY TABLE", flush=True)
    print("=" * 85, flush=True)
    df_summary = pd.DataFrame(records)
    print(df_summary[["trial_name", "backend", "requested_n_jobs", "total_user_days", "wall_time_seconds", "user_days_per_second", "speedup_vs_baseline", "scientific_regression"]].to_string(index=False), flush=True)
    print(f"\nSaved full results to:\n  - CSV:  {results_csv}\n  - JSON: {results_json}\n", flush=True)
    return df_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lab Workstation Parallelism Benchmark Suite")
    parser.add_argument("--gps", type=str, default=None, help="Path to preprocessed GPS parquet (or raw GPS)")
    parser.add_argument("--metadata", type=str, default=None, help="Path to user_home_metadata parquet")
    parser.add_argument("--out", type=str, default=None, help="Output directory for benchmark results")
    parser.add_argument("--users", type=int, default=100, help="Number of users to evaluate (default: 100)")
    parser.add_argument("--days", type=int, default=None, help="Max days per user (default: None)")
    args = parser.parse_args()

    gps_path = Path(args.gps) if args.gps else config.FILE_GPS_ORIGINAL
    out_bench = Path(args.out) if args.out else config.OUTPUTS_DIR / "diagnostics" / "performance_parallelism"

    if gps_path.exists():
        run_lab_benchmark_suite(
            gps_path,
            out_bench,
            user_metadata_path=args.metadata,
            limit_users=args.users,
            limit_days_per_user=args.days,
        )
    else:
        print(f"[Error] Target GPS file not found at: {gps_path}")

