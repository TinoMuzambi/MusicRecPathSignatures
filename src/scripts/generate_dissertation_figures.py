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
from src.scripts.run_baseline_comparison_multiple_runs import (
    CANONICAL_BASELINE_IDS,
    DETERMINISTIC_METHOD_IDS,
    STOCHASTIC_METHOD_IDS,
)
from src.utils.logger_config import setup_logger, configure_logging
from src.utils.timing import TimingReport


logger = setup_logger("generate_dissertation_figures")

# MR-06 canonical method IDs (see run_baseline_comparison_multiple_runs.py) and
# their human-readable display names for figure labels/legends.
PATH_SIGNATURE_METHOD_ID = next(
    method_id
    for method_id in DETERMINISTIC_METHOD_IDS
    if method_id.startswith("path_signature_cosine")
)
PATH_SIGNATURE_DISPLAY_NAME = "Path Signature"
METHOD_DISPLAY_NAMES = {
    PATH_SIGNATURE_METHOD_ID: PATH_SIGNATURE_DISPLAY_NAME,
    "traditional_audio_cosine": "Traditional Audio",
    "lightfm_warp": "LightFM (WARP)",
    "lightfm_warp_kos": "LightFM (WARP k-OS)",
    "lightfm_latent_blend": "LightFM (Latent Blend)",
    "implicit_als": "Implicit ALS",
}


def build_significance_annotations(
    comparisons: dict,
) -> tuple[list[float], list[str]]:
    """Build fig_06's per-baseline p-values and "p=...\\n(direction)" cell text.

    ``direction`` in a comparison record is
    sign(path_signature_precision - baseline_precision) (see
    ``wilcoxon_aligned``'s call order in
    ``run_baseline_comparison_multiple_runs.build_precision5_inference``, and
    ``wilcoxon_test``'s ``differences = first - second``): negative means
    the proposed path-signature method scored lower than that baseline, so
    it is labelled "worse", not "better" (R9 evidence audit F-09 -- colour
    alone cannot convey this direction).
    """

    p_values: list[float] = []
    annotations: list[str] = []
    for baseline_id in CANONICAL_BASELINE_IDS:
        key = f"{PATH_SIGNATURE_METHOD_ID}_vs_{baseline_id}"
        comparison = comparisons.get(key, {})
        if comparison.get("status") == "available":
            p = comparison.get("p_value_adjusted", comparison["p_value"])
            direction = comparison.get("direction", 0.0)
            direction_label = "worse" if direction < 0 else (
                "better" if direction > 0 else "tied"
            )
            annotations.append(f"p={p:.3g}\n({direction_label})")
        else:
            p = float("nan")
            annotations.append("n/a")
            logger.warning(
                "Comparison %s unavailable (%s); plotting as NaN",
                key,
                comparison.get("reason_code", "missing"),
            )
        p_values.append(p)
    return p_values, annotations


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dissertation figures")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--run-dir",
        type=str,
        default="results/baseline_comparison",
        help=(
            "MR-06 canonical baseline-comparison run directory (as written by "
            "run_baseline_comparison_cli.py / run_canonical_comparison), "
            "containing aggregate_metrics.json, uncertainty.json and "
            "precision5_inference.json."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--release-mode",
        action="store_true",
        help="Write only the two deterministic canonical PNG outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Determine output directory for log file
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Set up logging with file handler
    log_file = out / "generate_dissertation_figures.log"
    configure_logging(
        args.log_level,
        log_file=None if args.release_mode else str(log_file),
    )
    if not args.release_mode:
        logger.info("Logging to file: %s", log_file)

    # Initialize timing
    timing = TimingReport("generate_dissertation_figures")
    timing.start()

    run_dir = Path(args.run_dir)

    # Figure 05: method comparison with bootstrap CIs on Precision@5, read
    # from the MR-06 canonical run directory's uncertainty.json (the
    # aligned-percentile-bootstrap estimate/ci_low/ci_high per method,
    # computed once by run_canonical_comparison -- see
    # src/scripts/run_baseline_comparison.py::_aggregate_saved_rows).
    with timing.section("Figure 05: Method Comparison"):
        with open(run_dir / "uncertainty.json", "r", encoding="utf-8") as f:
            uncertainty = json.load(f)
        expected_methods = set(DETERMINISTIC_METHOD_IDS) | set(STOCHASTIC_METHOD_IDS)
        if set(uncertainty.get("methods", {})) != expected_methods:
            raise ValueError("uncertainty method family does not match the release contract")
        method_ids = sorted(uncertainty["methods"])
        labels = [METHOD_DISPLAY_NAMES.get(m, m) for m in method_ids]
        means = [uncertainty["methods"][m]["estimate"] for m in method_ids]
        ci_l = [uncertainty["methods"][m]["ci_low"] for m in method_ids]
        ci_h = [uncertainty["methods"][m]["ci_high"] for m in method_ids]
        numeric = np.asarray([means, ci_l, ci_h], dtype=np.float64)
        if not np.isfinite(numeric).all() or np.any(numeric < 0.0) or np.any(numeric > 1.0):
            raise ValueError("uncertainty values must be finite probabilities")
        if np.any(numeric[1] > numeric[0]) or np.any(numeric[0] > numeric[2]):
            raise ValueError("uncertainty intervals do not contain their estimates")
        for m, mean, lo, hi in zip(method_ids, means, ci_l, ci_h):
            logger.info("Method %s: estimate=%.4f, CI=[%.4f, %.4f]", m, mean, lo, hi)

        bar_with_cis(
            labels,
            means,
            ci_l,
            ci_h,
            ylabel="Precision@5",
            out_path=out / "fig_05_method_comparison.png",
            highlight_label=PATH_SIGNATURE_DISPLAY_NAME,
            sort_descending=True,
        )

    # Figure 06: significance heatmap.
    #
    # The MR-06 canonical engine does not compute a full pairwise
    # significance matrix across every method pair (there is no
    # baseline-vs-baseline Wilcoxon test); by design (build_precision5_inference
    # in run_baseline_comparison_multiple_runs.py) it only ever tests the five
    # planned "Path Signature vs each baseline" comparisons with one shared
    # Benjamini-Hochberg correction. So fig_06 is a single-row heatmap of
    # Path Signature vs each canonical baseline, using the real
    # BH-adjusted p-values from precision5_inference.json -- not a fabricated
    # full pairwise matrix.
    with timing.section("Figure 06: Significance Heatmap"):
        with open(run_dir / "precision5_inference.json", "r", encoding="utf-8") as f:
            inference = json.load(f)
        if inference.get("run_id") != uncertainty.get("run_id"):
            raise ValueError("figure inputs have mixed run IDs")
        comparisons = inference["comparisons"]
        expected_comparisons = {
            f"{PATH_SIGNATURE_METHOD_ID}_vs_{baseline}"
            for baseline in CANONICAL_BASELINE_IDS
        }
        if set(comparisons) != expected_comparisons:
            raise ValueError("inference comparison family does not match the release contract")
        p_values, annotation_row = build_significance_annotations(comparisons)
        if not np.isfinite(np.asarray(p_values, dtype=np.float64)).all():
            raise ValueError("all planned release comparisons must be available")
        mat = np.array([p_values])
        annotations = np.array([annotation_row])
        baseline_labels = [
            METHOD_DISPLAY_NAMES.get(m, m) for m in CANONICAL_BASELINE_IDS
        ]
        heatmap_matrix(
            mat,
            baseline_labels,
            [PATH_SIGNATURE_DISPLAY_NAME],
            out_path=out / "fig_06_significance_heatmap.png",
            cmap="RdBu_r",
            vmin=0.0,
            vmax=1.0,
            annotations=annotations,
        )

    # Stop timing and save report
    timing.stop()
    if not args.release_mode:
        timing.save_report(out)
    timing.print_summary()

    logger.info("Dissertation figures generated in %s", out)


if __name__ == "__main__":
    main()
