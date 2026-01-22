"""
Create a consolidated dissertation package with tables, figures, data, and indexes.

British English is used throughout (visualisations, behaviour, etc.).
"""

import argparse
import shutil
from pathlib import Path
from typing import List

from src.utils.logger_config import setup_logger, configure_logging


logger = setup_logger("create_dissertation_package")


def parse_args():
    parser = argparse.ArgumentParser(description="Create dissertation package")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument(
        "--output-dir", type=str, default="results/dissertation_package"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        return True
    return False


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def collect_tables(results_root: Path, pkg_root: Path) -> List[Path]:
    tables_dir = pkg_root / "tables"
    ensure_dir(tables_dir)
    copied: List[Path] = []

    candidates = [
        (
            results_root / "eda" / "dataset_statistics_table.csv",
            "table_01_dataset_statistics.csv",
        ),
        (
            results_root / "eda" / "genre_statistics.csv",
            "table_02_genre_statistics.csv",
        ),
        (
            results_root / "baseline_comparison" / "baseline_comparison_table.csv",
            "table_03_baseline_comparison.csv",
        ),
        (
            results_root / "baseline_comparison" / "significance_matrix.csv",
            "table_04_statistical_tests.csv",
        ),
        (
            results_root / "ablation_studies" / "ablation_summary.csv",
            "table_05_ablation_results.csv",
        ),
        (
            results_root / "robustness" / "stability_table.csv",
            "table_06_robustness_metrics.csv",
        ),
        (
            results_root / "evaluation" / "cv_results_table.csv",
            "table_07_cv_results.csv",
        ),
    ]
    for src, name in candidates:
        dst = tables_dir / name
        if copy_if_exists(src, dst):
            copied.append(dst)

    index_lines = ["# Table Index", ""]
    for p in copied:
        index_lines.append(f"- {p.name}: auto-collected from results")
    write_text(tables_dir / "TABLE_INDEX.md", "\n".join(index_lines))
    return copied


def collect_figures(results_root: Path, pkg_root: Path) -> List[Path]:
    figures_dir = pkg_root / "figures"
    ensure_dir(figures_dir)
    copied: List[Path] = []

    # Copy dissertation figures if present
    fig_root = results_root / "dissertation_figures"
    if fig_root.exists():
        for p in sorted(fig_root.glob("fig_*.png")):
            dst = figures_dir / p.name
            shutil.copy2(p, dst)
            copied.append(dst)

    index_lines = ["# Figure Index", ""]
    for p in copied:
        index_lines.append(f"- {p.name}: publication-quality figure")
    write_text(figures_dir / "FIGURE_INDEX.md", "\n".join(index_lines))
    return copied


def collect_data(results_root: Path, pkg_root: Path) -> List[Path]:
    data_dir = pkg_root / "data"
    ensure_dir(data_dir)
    copied: List[Path] = []

    sources = [
        (results_root / "evaluation" / "evaluation_summary.json", "all_metrics.json"),
        (
            results_root / "baseline_comparison" / "statistical_comparison.json",
            "statistical_tests.json",
        ),
    ]
    for src, name in sources:
        dst = data_dir / name
        if copy_if_exists(src, dst):
            copied.append(dst)

    dd = [
        "# Data Dictionary",
        "",
        "This file indexes the JSON data files included in the dissertation package.",
        "- all_metrics.json: Combined summary from the evaluation suite",
        "- statistical_tests.json: Pairwise statistical comparisons",
    ]
    write_text(data_dir / "DATA_DICTIONARY.md", "\n".join(dd))
    return copied


def write_readmes(pkg_root: Path) -> None:
    write_text(
        pkg_root / "DISSERTATION_PACKAGE_README.md",
        "\n".join(
            [
                "# Dissertation Package",
                "",
                "This package consolidates tables, figures, and data referenced in the dissertation.",
                "All files are publication-ready (CSV/PNG/MD).",
            ]
        ),
    )


def write_results_summary(results_root: Path) -> None:
    summary_path = results_root / "RESULTS_SUMMARY.md"
    lines = [
        "# Results Summary",
        "",
        "Executive overview of generated outputs with links to key reports.",
        "",
        "## Key Reports",
        "- EDA: results/eda/EDA_SUMMARY.md",
        "- Synthetic Users: results/synthetic_users/SYNTHETIC_USERS_REPORT.md",
        "- Baseline Comparison: results/baseline_comparison/BASELINE_COMPARISON_REPORT.md",
        "- Robustness: results/robustness/ROBUSTNESS_REPORT.md",
    ]
    write_text(summary_path, "\n".join(lines))


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    pkg_root = Path(args.output_dir)

    ensure_dir(pkg_root)
    
    # Set up logging with file handler
    log_file = pkg_root / "create_dissertation_package.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger.info("Logging to file: %s", log_file)
    logger.info("Creating dissertation package at %s", pkg_root)

    tables = collect_tables(results_root, pkg_root)
    figures = collect_figures(results_root, pkg_root)
    data = collect_data(results_root, pkg_root)
    write_readmes(pkg_root)
    write_results_summary(results_root)

    logger.info(
        "Package created with %d tables, %d figures, %d data files",
        len(tables),
        len(figures),
        len(data),
    )


if __name__ == "__main__":
    main()
