#!/usr/bin/env python3
# pylint: disable=broad-except
"""
Run baseline comparison multiple times and aggregate results.

This script addresses non-determinism in LightFM-based models by running
multiple iterations and reporting mean ± std metrics.

Usage:
    python run_baseline_comparison_multiple_runs.py --n-runs 5 --features-file ... --tracks-json ...
"""

import argparse
import json
import sys
import subprocess
import time
import psutil
import os
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from src.utils.logger_config import setup_logger, configure_logging
from src.utils.timing import TimingReport

logger = setup_logger("baseline_comparison_multiple_runs")


def aggregate_results(all_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across multiple runs, computing mean and std.

    Args:
        all_runs: List of result dictionaries from individual runs

    Returns:
        Aggregated results dictionary with mean and std for each metric
    """
    if not all_runs:
        raise ValueError("No runs to aggregate")

    models = list(all_runs[0].keys())
    aggregated = {}

    for model_name in models:
        # Collect all results for this model across runs
        model_results = []
        for run in all_runs:
            if model_name in run and run[model_name]:
                model_results.append(run[model_name])

        if not model_results:
            logger.warning("No results found for model %s", model_name)
            aggregated[model_name] = {}
            continue

        aggregated[model_name] = {}

        # Aggregate scalar metrics (map, prediction_time, avg_prediction_time)
        for metric in ["map", "prediction_time", "avg_prediction_time"]:
            values = [r.get(metric, 0) for r in model_results if metric in r]
            if values:
                aggregated[model_name][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_runs": len(values),
                    "values": values,
                }
            else:
                aggregated[model_name][metric] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "n_runs": 0,
                    "values": [],
                }

        # Aggregate per-K metrics (precision, recall, ndcg, etc.)
        for metric in [
            "precision",
            "recall",
            "ndcg",
            "diversity",
            "novelty",
            "coverage",
        ]:
            aggregated[model_name][metric] = {}
            k_values = ["1", "5", "10"]
            for k in k_values:
                values = [
                    r.get(metric, {}).get(k, 0)
                    for r in model_results
                    if metric in r and k in r.get(metric, {})
                ]
                if values:
                    aggregated[model_name][metric][k] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "n_runs": len(values),
                        "values": values,
                    }
                else:
                    aggregated[model_name][metric][k] = {
                        "mean": 0.0,
                        "std": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "n_runs": 0,
                        "values": [],
                    }

        # Aggregate per-user metrics
        # For each run, compute mean per-user metric, then aggregate across runs
        for metric in ["per_user_precision", "per_user_recall"]:
            aggregated[model_name][metric] = {}
            k_values = ["1", "5", "10"]
            for k in k_values:
                per_run_means = []
                for run_result in model_results:
                    per_user_values = run_result.get(metric, {}).get(k, [])
                    if per_user_values and len(per_user_values) > 0:
                        # Compute mean across users for this run
                        per_run_means.append(float(np.mean(per_user_values)))
                    else:
                        per_run_means.append(0.0)

                if per_run_means:
                    aggregated[model_name][metric][k] = {
                        "mean": float(np.mean(per_run_means)),
                        "std": float(np.std(per_run_means)),
                        "min": float(np.min(per_run_means)),
                        "max": float(np.max(per_run_means)),
                        "n_runs": len(per_run_means),
                        "values": per_run_means,
                    }
                else:
                    aggregated[model_name][metric][k] = {
                        "mean": 0.0,
                        "std": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                        "n_runs": 0,
                        "values": [],
                    }

    return aggregated


def print_summary(aggregated: Dict[str, Any]):
    """Print a summary of aggregated results."""
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS SUMMARY (Mean ± Std across runs)")
    print("=" * 80)

    models = [
        "User-based CF",
        "Item-based CF",
        "Content-based Filter",
        "SVD",
        "NMF",
        "Hybrid",
        "Path Signature",
    ]

    for model_name in models:
        if model_name not in aggregated:
            continue

        metrics = aggregated[model_name]
        print(f"\n{model_name}:")

        # Print key metrics
        if "precision" in metrics and "5" in metrics["precision"]:
            prec = metrics["precision"]["5"]
            print(
                f"  Precision@5: {prec['mean']:.4f} ± {prec['std']:.4f} "
                f"(min={prec['min']:.4f}, max={prec['max']:.4f}, n={prec['n_runs']})"
            )

        if "recall" in metrics and "5" in metrics["recall"]:
            recall = metrics["recall"]["5"]
            print(f"  Recall@5: {recall['mean']:.6f} ± {recall['std']:.6f}")

        if "ndcg" in metrics and "5" in metrics["ndcg"]:
            ndcg = metrics["ndcg"]["5"]
            print(f"  NDCG@5: {ndcg['mean']:.4f} ± {ndcg['std']:.4f}")

        if "map" in metrics:
            map_val = metrics["map"]
            print(f"  MAP: {map_val['mean']:.4f} ± {map_val['std']:.4f}")

        if "avg_prediction_time" in metrics:
            time_val = metrics["avg_prediction_time"]
            print(
                f"  Avg Prediction Time: {time_val['mean']:.4f}s ± {time_val['std']:.4f}s"
            )

    print("\n" + "=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run baseline comparison multiple times and aggregate results"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Number of runs to perform (default: 5)",
    )
    parser.add_argument(
        "--features-file",
        default="./data/processed_tracks/features.json",
        help="Path to features JSON file",
    )
    parser.add_argument(
        "--tracks-json",
        default="./data/processed_tracks/selected_tracks.json",
        help="Path to tracks metadata JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/baseline_comparison",
        help="Output directory for results",
    )
    parser.add_argument(
        "--n-users",
        type=int,
        default=200,
        help="Number of synthetic users to create (default: 200)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Ratio of users to use for testing (default: 0.15)",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
        help="Ratio of users to use for validation (default: 0.15)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--quality-report",
        action="store_true",
        help="Generate detailed data quality reports for each model",
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set up logging
    log_file = output_path / "baseline_comparison_multiple_runs.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger.info("Logging to file: %s", log_file)
    logger.info("Starting %d runs of baseline comparison...", args.n_runs)

    # Initialize timing for overall process
    overall_timing = TimingReport("baseline_comparison_multiple_runs")
    overall_timing.start()

    all_runs = []
    successful_runs = 0

    # Results file path (will be overwritten each run, so we load immediately)
    results_file = output_path / "baseline_comparison_results.json"

    # Get the path to the baseline comparison script
    script_dir = Path(__file__).parent
    baseline_script = script_dir / "run_baseline_comparison.py"

    if not baseline_script.exists():
        logger.error("Baseline comparison script not found: %s", baseline_script)
        sys.exit(1)

    # Get initial memory usage
    process = psutil.Process(os.getpid())
    initial_memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info("Initial memory usage: %.2f MB", initial_memory_mb)

    for run_num in range(1, args.n_runs + 1):
        logger.info("=" * 80)
        logger.info("Starting run %d/%d (in separate process)", run_num, args.n_runs)
        logger.info("=" * 80)

        # Check memory before run
        memory_before_mb = process.memory_info().rss / 1024 / 1024
        logger.info(
            "Main process memory before run %d: %.2f MB", run_num, memory_before_mb
        )

        try:
            # Build command to run baseline comparison in a separate process
            # This isolates memory completely - each subprocess gets its own memory space
            cmd = [
                sys.executable,  # Use the same Python interpreter
                str(baseline_script),
                "--features-file",
                args.features_file,
                "--tracks-json",
                args.tracks_json,
                "--output-dir",
                args.output_dir,
                "--n-users",
                str(args.n_users),
                "--test-ratio",
                str(args.test_ratio),
                "--validation-ratio",
                str(args.validation_ratio),
                "--log-level",
                args.log_level,
            ]

            if args.quality_report:
                cmd.append("--quality-report")

            logger.debug("Running command: %s", " ".join(cmd))

            # Run in subprocess - this isolates memory completely
            # When subprocess exits, all its memory is freed automatically
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout per run
                check=False,  # Don't raise on non-zero exit, we'll check manually
                cwd=str(script_dir.parent.parent),  # Run from code directory
            )
            elapsed_time = time.time() - start_time

            # Check if subprocess succeeded
            if result.returncode == 0:
                logger.info(
                    "Run %d/%d completed successfully in %.2f seconds",
                    run_num,
                    args.n_runs,
                    elapsed_time,
                )

                # Load results immediately after subprocess completes
                if results_file.exists():
                    with open(results_file, "r", encoding="utf-8") as f:
                        run_results = json.load(f)
                        all_runs.append(run_results)
                        successful_runs += 1
                        logger.info(
                            "Results loaded for run %d/%d", run_num, args.n_runs
                        )
                else:
                    logger.warning("Results file not found after run %d", run_num)

                # Log subprocess output if there are warnings
                if result.stderr:
                    logger.debug(
                        "Subprocess stderr (run %d):\n%s", run_num, result.stderr
                    )
            else:
                logger.error(
                    "Run %d/%d failed with return code %d",
                    run_num,
                    args.n_runs,
                    result.returncode,
                )
                if result.stdout:
                    logger.error("Subprocess stdout:\n%s", result.stdout)
                if result.stderr:
                    logger.error("Subprocess stderr:\n%s", result.stderr)
                # Continue with next run
                continue

        except subprocess.TimeoutExpired:
            logger.error("Run %d/%d timed out after 2 hours", run_num, args.n_runs)
            # Continue with next run
            continue
        except Exception as e:
            logger.error(
                "Error running subprocess for run %d/%d: %s",
                run_num,
                args.n_runs,
                str(e),
            )
            logger.exception("Full traceback:")
            # Continue with next run
            continue

        # Check memory after run (should be similar since subprocess isolated memory)
        memory_after_mb = process.memory_info().rss / 1024 / 1024
        logger.info(
            "Main process memory after run %d: %.2f MB (delta: %.2f MB)",
            run_num,
            memory_after_mb,
            memory_after_mb - memory_before_mb,
        )

        # Small delay between runs
        if run_num < args.n_runs:
            logger.debug("Waiting 2 seconds before next run...")
            time.sleep(2)

    overall_timing.stop()

    if not all_runs:
        logger.error("No successful runs completed! Cannot aggregate results.")
        overall_timing.save_report(output_path)
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("Aggregating results across %d successful runs...", successful_runs)
    logger.info("=" * 80)

    try:
        # Aggregate results
        aggregated = aggregate_results(all_runs)

        # Save aggregated results
        aggregated_file = output_path / "baseline_comparison_results_aggregated.json"
        with open(aggregated_file, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2, default=str)
        logger.info("Aggregated results saved to %s", aggregated_file)

        # Also save individual runs for reference (backup)
        individual_runs_file = (
            output_path / "baseline_comparison_results_individual_runs.json"
        )
        with open(individual_runs_file, "w", encoding="utf-8") as f:
            json.dump(all_runs, f, indent=2, default=str)
        logger.info("Individual run results saved to %s", individual_runs_file)

        # Print summary
        print_summary(aggregated)

        # Save timing report
        overall_timing.save_report(output_path)
        overall_timing.print_summary()

        logger.info("=" * 80)
        logger.info("Multiple runs completed successfully!")
        logger.info("  Successful runs: %d/%d", successful_runs, args.n_runs)
        logger.info("  Aggregated results: %s", aggregated_file)
        logger.info("  Individual runs backup: %s", individual_runs_file)
        logger.info("=" * 80)

    except Exception as e:
        logger.error("Error aggregating results: %s", str(e))
        logger.exception("Full traceback:")
        overall_timing.save_report(output_path)
        raise


if __name__ == "__main__":
    main()
