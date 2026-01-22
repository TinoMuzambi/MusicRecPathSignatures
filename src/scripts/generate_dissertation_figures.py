"""
Generate all dissertation figures with consistent styling.
"""

import argparse
from pathlib import Path
import json
import numpy as np

from src.visualisation.publication_plots import (
    bar_with_cis,
    heatmap_matrix,
)
from src.scripts.generate_system_architecture import create_system_architecture_diagram
from src.utils.logger_config import setup_logger, configure_logging
from src.utils.timing import TimingReport


logger = setup_logger("generate_dissertation_figures")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dissertation figures")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--baseline-results",
        type=str,
        default="results/baseline_comparison/baseline_comparison_results.json",
    )
    parser.add_argument(
        "--stats-json",
        type=str,
        default="results/baseline_comparison/statistical_comparison.json",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine output directory for log file
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Set up logging with file handler
    log_file = out / "generate_dissertation_figures.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger.info("Logging to file: %s", log_file)

    # Initialize timing
    timing = TimingReport("generate_dissertation_figures")
    timing.start()

    # Figure 05: method comparison with CIs (if available); fallback to bars without CIs
    # Figure 05: method comparison with CIs
    try:
        with timing.section("Figure 05: Method Comparison"):
            with open(args.baseline_results, "r", encoding="utf-8") as f:
                res = json.load(f)
            labels = sorted(res.keys())
            k = str(5)  # K=5 for the plot
            means = []
            ci_l = []
            ci_h = []
            
            for m in labels:
                # Use per-user data if available for accurate CIs
                per_user = res[m].get("per_user_precision", {}).get(k, [])
                if per_user and len(per_user) > 0 and sum(per_user) > 0:
                    data = np.array(per_user)
                    mean = np.mean(data)
                    # Calculate 95% CI using standard error
                    se = np.std(data, ddof=1) / np.sqrt(len(data))
                    ci = 1.96 * se
                    means.append(mean)
                    ci_l.append(max(0.0, mean - ci))
                    ci_h.append(min(1.0, mean + ci))
                    logger.info("Method %s: Mean=%.4f, CI=%.4f", m, mean, ci)
                else:
                    # Fallback to pre-computed mean if per-user data missing or empty (e.g. all zeros)
                    prec = res[m].get("precision", {})
                    mean = prec.get(k, prec.get(int(k), 0.0)) if isinstance(prec, dict) else 0.0
                    means.append(mean)
                    # If mean is 0, CI is 0
                    if mean == 0:
                         ci_l.append(0.0)
                         ci_h.append(0.0)
                    else:
                        # Warning if no per-user data but non-zero mean (shouldn't happen with current data)
                        logger.warning("Method %s: No per-user data for CI calculation, using 0", m)
                        ci_l.append(mean)
                        ci_h.append(mean)

            bar_with_cis(
                labels,
                means,
                ci_l,
                ci_h,
                ylabel="Precision@5",
                title="Baseline Comparison",
                out_path=out / "fig_05_method_comparison.png",
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not generate fig_05: %s", exc)

    # System Architecture diagram
    try:
        with timing.section("System Architecture Diagram"):
            create_system_architecture_diagram(
                out / "system_architecture.png",
                dpi=300
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not generate system architecture diagram: %s", exc)

    # Figure 06: significance heatmap
    try:
        with timing.section("Figure 06: Significance Heatmap"):
            with open(args.stats_json, "r", encoding="utf-8") as f:
                stats = json.load(f)
            models = sorted(
                set(
                    [name.split(" vs ")[0] for name in stats.keys()]
                    + [name.split(" vs ")[1] for name in stats.keys()]
                )
            )
            n = len(models)
            mat = np.ones((n, n))
            for i, m1 in enumerate(models):
                for j, m2 in enumerate(models):
                    if i == j:
                        mat[i, j] = 1.0
                    else:
                        key = f"{m1} vs {m2}"
                        reverse_key = f"{m2} vs {m1}"
                        
                        target_key = None
                        if key in stats:
                            target_key = key
                        elif reverse_key in stats:
                            target_key = reverse_key
                        
                        if target_key:
                            p = stats[target_key].get(
                                "p_value_adjusted", stats[target_key].get("p_value", 1.0)
                            )
                        else:
                            mat[i, j] = 1.0
                            continue
                        mat[i, j] = p
            heatmap_matrix(
                mat,
                models,
                models,
                title="Significance (p-values)",
                out_path=out / "fig_06_significance_heatmap.png",
                cmap="RdBu_r",
                vmin=0.0,
                vmax=1.0,
            )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not generate fig_06: %s", exc)

    # Stop timing and save report
    timing.stop()
    timing.save_report(out)
    timing.print_summary()

    logger.info("Dissertation figures generated in %s", out)


if __name__ == "__main__":
    main()
