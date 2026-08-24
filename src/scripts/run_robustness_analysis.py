"""
CLI for robustness analysis: bootstrap CI, sensitivity, seed stability, and a
per-genre performance breakdown -- all derived from real per-user rows saved
by the canonical baseline comparison (``results/baseline_comparison``).

R9 evidence audit finding F-01: an earlier version of this script fabricated
its sensitivity and stability sections from hard-coded ``rng.normal(...)``
formulae unconditioned on any real measurement, and bootstrapped mean
pairwise *similarity* while presenting it as a performance metric. Every
analysis below is instead derived from the canonical run's saved per-user
Precision@5 rows: nothing here is drawn from an unconditioned random
distribution, and the per-group breakdown groups real Precision@5 by the
genre of each user's real query track (directly answering examiner R-06 as
a side effect of the fix).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np

from src.evaluation.robustness import (
    bootstrap_ci,
    sensitivity_analysis,
    error_analysis_by_group,
    save_json,
    save_csv_table,
)
from src.utils.logger_config import setup_logger, configure_logging
from src.utils.metadata import load_tracks_metadata
from src.scripts.run_baseline_comparison_multiple_runs import DETERMINISTIC_METHOD_IDS


logger = setup_logger("run_robustness_analysis")
PATH_SIGNATURE_METHOD_ID = next(
    method_id
    for method_id in DETERMINISTIC_METHOD_IDS
    if method_id.startswith("path_signature_cosine")
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run robustness analyses (bootstrap CI, sample-size sensitivity, "
            "seed stability, per-genre breakdown) over the real per-user rows "
            "saved by the canonical baseline comparison."
        )
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--baseline-comparison-dir",
        type=str,
        required=True,
        help=(
            "Path to results/baseline_comparison from a completed canonical "
            "run (Step 4). This script fails closed if it is missing rather "
            "than substituting synthetic data."
        ),
    )
    parser.add_argument(
        "--tracks-json",
        type=str,
        required=True,
        help="Path to the selected-tracks JSON file, used for genre lookup.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def load_precision_by_user(
    method_jsonl_path: Path, *, k: int = 5
) -> Dict[str, Tuple[float, str]]:
    """Read one method's saved rows into ``{user_id: (precision@k, query_track_id)}``.

    Fails closed on a missing file, a malformed row, or a duplicate user ID.
    Never falls back to a synthetic distribution.
    """

    path = Path(method_jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"method output not found: {path}")
    loaded: Dict[str, Tuple[float, str]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        user_id = row["user_id"]
        if not isinstance(user_id, str) or not user_id or user_id in loaded:
            raise ValueError(f"{path}:{line_number}: duplicate user_id {user_id}")
        raw_precision = row["metrics"]["precision"][str(k)]
        if isinstance(raw_precision, bool) or not isinstance(raw_precision, (int, float)):
            raise ValueError(f"{path}:{line_number}: Precision@{k} is not numeric")
        precision = float(raw_precision)
        query_track_id = row["query_track_id"]
        if not np.isfinite(precision) or not isinstance(query_track_id, str) or not query_track_id:
            raise ValueError(f"{path}:{line_number}: invalid Precision or query track")
        loaded[user_id] = (precision, query_track_id)
    if not loaded:
        raise ValueError(f"{path} contained no rows")
    return loaded


def real_sensitivity_analysis(
    precision_by_user: Mapping[str, Tuple[float, str]],
    *,
    sizes: list[int],
    seeds: list[int],
) -> Dict:
    """Subsample the real per-user Precision@5 array at each declared size.

    Every reported value is the mean of a ``size``-user subsample drawn
    without replacement from the real per-user array under a declared seed
    -- never a synthetic formula. Raises ``ValueError`` if a requested size
    exceeds the real population, rather than silently truncating it.
    """

    values = np.array([value for value, _ in precision_by_user.values()], dtype=float)
    if not sizes:
        raise ValueError("sizes must not be empty")
    if max(sizes) > len(values):
        raise ValueError(
            f"sensitivity size {max(sizes)} exceeds the real population "
            f"of {len(values)} users"
        )

    def metric_fn(size: int, seed: int) -> float:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(values), size=size, replace=False)
        return float(np.mean(values[indices]))

    return sensitivity_analysis(metric_fn, sizes=list(sizes), seeds=list(seeds))


def stability_across_seeds(seed_to_user_values: Mapping[int, Mapping[str, float]]) -> Dict:
    """Stability of one method's real per-user Precision@5 across its real seeds.

    ``overall`` reports the spread of the per-seed *aggregate* means -- the
    genuine run-to-run (seed-to-seed) stability question -- rather than
    pooling every user's raw value across every seed into one statistic.
    Pooling would conflate ordinary user-level heterogeneity (some users are
    easier to serve than others, in every run) with real run-to-run
    instability, understating stability for any method whose per-user
    Precision@5 values are naturally spread out even when its aggregate is
    identical seed to seed. ``per_run`` still reports each seed's own
    mean/std/var for context.

    A method evaluated under exactly one seed (every deterministic canonical
    method) is, by construction, identical on every execution: re-running it
    cannot change its output, so it is reported with an explicit, exact zero
    run-to-run variance.
    """

    seeds = sorted(seed_to_user_values)
    runs = [list(seed_to_user_values[seed].values()) for seed in seeds]
    per_run = [
        {
            "mean": float(np.mean(run)) if run else 0.0,
            "std": float(np.std(run)) if run else 0.0,
            "var": float(np.var(run)) if run else 0.0,
        }
        for run in runs
    ]
    if len(runs) == 1:
        mean = per_run[0]["mean"]
        return {"per_run": per_run, "overall": {"mean": mean, "std": 0.0, "var": 0.0}}
    run_means = np.array([run_stats["mean"] for run_stats in per_run], dtype=float)
    overall = {
        "mean": float(np.mean(run_means)),
        "std": float(np.std(run_means)),
        "var": float(np.var(run_means)),
    }
    return {"per_run": per_run, "overall": overall}


def real_error_analysis_by_genre(
    precision_by_user: Mapping[str, Tuple[float, str]],
    genre_map: Mapping[str, str],
) -> Dict:
    """Group real per-user Precision@5 by the genre of each user's real query track."""

    missing = sorted(
        {track_id for _, track_id in precision_by_user.values()} - set(genre_map)
    )
    if missing:
        raise ValueError(f"missing genre metadata for query tracks: {missing}")

    per_user_metric = {user: value for user, (value, _) in precision_by_user.items()}
    user_to_group = {
        user: genre_map[track_id]
        for user, (_, track_id) in precision_by_user.items()
    }
    return error_analysis_by_group(per_user_metric, user_to_group)


def generate_robustness_report(
    boot: dict,
    sens: dict,
    stab: dict,
    err: dict,
    output_dir: Path,
    *,
    headline_method: str,
) -> None:
    """Write a human-readable markdown summary of the robustness analysis.

    Every number is read directly from the ``boot``/``sens``/``stab``/``err``
    dicts already computed from real saved rows above.
    """
    lines = []
    lines.append("# Robustness Analysis Summary")
    lines.append("")
    lines.append(
        f"All figures below are derived from `{headline_method}`'s real saved "
        "per-user Precision@5 rows (and, for seed stability, every method's "
        "real per-seed rows) in `results/baseline_comparison`. None are drawn "
        "from an unconditioned random distribution."
    )
    lines.append("")

    # 1) Bootstrap confidence interval
    lines.append(f"## 1. Bootstrap Confidence Interval ({headline_method}, Precision@5)")
    lines.append("")
    if boot:
        lines.append(f"- **Point estimate**: {boot.get('estimate', 'N/A'):.4f}")
        lines.append(
            f"- **95% CI**: [{boot.get('ci_low', float('nan')):.4f}, "
            f"{boot.get('ci_high', float('nan')):.4f}]"
        )
        ci_width = boot.get("ci_high", float("nan")) - boot.get("ci_low", float("nan"))
        lines.append(f"- **CI width**: {ci_width:.4f}")
    else:
        lines.append("- No bootstrap results available.")
    lines.append("")

    # 2) Sensitivity analysis
    lines.append(f"## 2. Sensitivity to Evaluation Population Size ({headline_method}, Precision@5)")
    lines.append("")
    if sens and sens.get("values"):
        lines.append(
            "Mean Precision@5 over real random subsamples of the real "
            "evaluation population, at each declared sample size and seed:"
        )
        lines.append("")
        lines.append("| Size | Mean | Std | Min | Max |")
        lines.append("|------|------|-----|-----|-----|")
        for size in sens.get("sizes", sorted(sens["values"].keys(), key=int)):
            seed_vals = sens["values"].get(str(size), sens["values"].get(size, {}))
            vals = list(seed_vals.values())
            if vals:
                lines.append(
                    f"| {size} | {np.mean(vals):.4f} | {np.std(vals):.4f} | "
                    f"{min(vals):.4f} | {max(vals):.4f} |"
                )
        lines.append("")
    else:
        lines.append("- No sensitivity results available.")
    lines.append("")

    # 3) Stability across real seeds, per method
    lines.append("## 3. Stability Across Real Declared Seeds, Per Method")
    lines.append("")
    if stab:
        lines.append(
            "For the four stochastic methods this is real run-to-run "
            "variance across the five declared model seeds. For the two "
            "deterministic methods it is exactly zero by construction: a "
            "deterministic method's re-execution cannot change its output, "
            "so this is not a synthetic placeholder."
        )
        lines.append("")
        lines.append("| Method | Overall mean | Overall std | Overall var |")
        lines.append("|--------|---------------|-------------|-------------|")
        for method_id in sorted(stab):
            overall = stab[method_id]["overall"]
            lines.append(
                f"| {method_id} | {overall.get('mean', float('nan')):.4f} | "
                f"{overall.get('std', float('nan')):.4f} | "
                f"{overall.get('var', float('nan')):.6f} |"
            )
    else:
        lines.append("- No stability results available.")
    lines.append("")

    # 4) Error analysis by genre
    lines.append(f"## 4. Precision@5 by Query-Track Genre ({headline_method})")
    lines.append("")
    groups = (err or {}).get("groups", {})
    if groups:
        sorted_groups = sorted(
            groups.items(), key=lambda kv: kv[1].get("n", 0), reverse=True
        )
        lines.append("| Genre | n | Mean Precision@5 | Std |")
        lines.append("|-------|---|-------------------|-----|")
        for group_name, stats in sorted_groups:
            lines.append(
                f"| {group_name} | {stats.get('n', 0)} | "
                f"{stats.get('mean', float('nan')):.4f} | "
                f"{stats.get('std', float('nan')):.4f} |"
            )
        lines.append("")
        means = [s.get("mean") for s in groups.values() if s.get("mean") is not None]
        if means:
            best_group = max(groups.items(), key=lambda kv: kv[1].get("mean", -np.inf))
            worst_group = min(groups.items(), key=lambda kv: kv[1].get("mean", np.inf))
            lines.append(
                f"- **Highest-precision query genre**: {best_group[0]} "
                f"(mean = {best_group[1].get('mean', float('nan')):.4f}, "
                f"n = {best_group[1].get('n', 0)})"
            )
            lines.append(
                f"- **Lowest-precision query genre**: {worst_group[0]} "
                f"(mean = {worst_group[1].get('mean', float('nan')):.4f}, "
                f"n = {worst_group[1].get('n', 0)})"
            )
    else:
        lines.append("- No error-analysis results available.")
    lines.append("")

    lines.append("## Generated Files")
    lines.append("")
    lines.append("### Data Files")
    lines.append(
        "- `bootstrap_results.json`, `bootstrap_ci.csv`: bootstrap CI over "
        f"{headline_method}'s real per-user Precision@5"
    )
    lines.append(
        "- `sensitivity_analysis.json`, `sensitivity_plots.csv`: sensitivity "
        "to evaluation population size, subsampled from the real population"
    )
    lines.append(
        "- `stability_metrics.json`, `stability_table.csv`: real seed-to-seed "
        "stability, per method"
    )
    lines.append(
        "- `error_analysis.json`, `error_analysis_table.csv`: real "
        "Precision@5 grouped by query-track genre"
    )
    lines.append("")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ROBUSTNESS_REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Markdown report generated: %s", report_path)


def main():
    args = parse_args()

    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f"robustness output directory already exists: {out}")
    out.mkdir(parents=True)

    log_file = out / "robustness_analysis.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger = setup_logger("robustness_analysis")
    logger.info("Logging to file: %s", log_file)

    baseline_dir = Path(args.baseline_comparison_dir)
    methods_dir = baseline_dir / "methods"
    headline_path = methods_dir / f"{PATH_SIGNATURE_METHOD_ID}.jsonl"
    precision_by_user = load_precision_by_user(headline_path)
    logger.info(
        "Loaded real Precision@5 for %d users from %s",
        len(precision_by_user),
        headline_path,
    )

    _, _, tracks = load_tracks_metadata(args.tracks_json)
    genre_map = {str(track["track_id"]): track.get("genre", "Unknown") for track in tracks}
    logger.info("Loaded genre metadata for %d tracks", len(genre_map))

    # 1. Bootstrap CI over the headline method's real per-user Precision@5.
    values = [value for value, _ in precision_by_user.values()]
    boot = bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=2025)
    save_json(boot, out / "bootstrap_results.json")
    save_csv_table(
        [{"estimate": boot["estimate"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"]}],
        out / "bootstrap_ci.csv",
    )

    # 2. Sensitivity to real evaluation population size.
    population = len(precision_by_user)
    sizes = sorted({size for size in (50, 100, population) if size <= population})
    sens = real_sensitivity_analysis(precision_by_user, sizes=sizes, seeds=[1, 2, 3])
    save_json(sens, out / "sensitivity_analysis.json")
    sens_rows = []
    for size, seed_vals in sens["values"].items():
        for seed, val in seed_vals.items():
            sens_rows.append({"size": int(size), "seed": int(seed), "precision_at_5": val})
    save_csv_table(sens_rows, out / "sensitivity_plots.csv")

    # 3. Stability across real declared seeds, per method (all 22 outputs).
    seed_values_by_method: Dict[str, Dict[int, Dict[str, float]]] = {}
    for jsonl_path in sorted(methods_dir.glob("*.jsonl")):
        method_key = jsonl_path.stem
        rows_for_method = load_precision_by_user(jsonl_path)
        per_user_values = {user: value for user, (value, _) in rows_for_method.items()}
        if "__seed_" in method_key:
            method_id, seed_text = method_key.rsplit("__seed_", 1)
            seed_values_by_method.setdefault(method_id, {})[int(seed_text)] = per_user_values
        else:
            seed_values_by_method[method_key] = {0: per_user_values}
    stab = {
        method_id: stability_across_seeds(seed_values)
        for method_id, seed_values in seed_values_by_method.items()
    }
    save_json(stab, out / "stability_metrics.json")
    stab_rows = []
    for method_id, method_stab in stab.items():
        for i, run_stats in enumerate(method_stab["per_run"], 1):
            stab_rows.append({"method": method_id, "run": i, **run_stats})
        stab_rows.append({"method": method_id, "run": "overall", **method_stab["overall"]})
    save_csv_table(stab_rows, out / "stability_table.csv")

    # 4. Real per-genre Precision@5 breakdown for the headline method (R-06).
    err = real_error_analysis_by_genre(precision_by_user, genre_map)
    save_json(err, out / "error_analysis.json")
    err_rows = [{"group": g, **vals} for g, vals in err["groups"].items()]
    save_csv_table(err_rows, out / "error_analysis_table.csv")

    generate_robustness_report(
        boot, sens, stab, err, out, headline_method=PATH_SIGNATURE_METHOD_ID
    )
    logger.info("Robustness analysis completed. Outputs saved to %s", out)


if __name__ == "__main__":
    main()
