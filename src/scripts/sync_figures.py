"""
Script to sync figures from various locations to the workspace figures directory.

This script copies figures from their generation locations to /workspace/figures/
for use in the LaTeX dissertation.
"""

import argparse
import shutil
from pathlib import Path
from src.utils.logger_config import configure_logging, setup_logger

# Module-level logger (will be configured in main())
logger = setup_logger("sync_figures")


def sync_figures(
    code_root: Path,
    figures_dir: Path,
    overwrite: bool = True,
) -> None:
    """
    Sync figures from code directories to workspace figures directory.
    
    Args:
        code_root: Root directory of the code (e.g., /workspace/code)
        figures_dir: Target figures directory (e.g., /workspace/figures)
        overwrite: Whether to overwrite existing files
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Mapping of source files to destination names
    figure_mappings = [
        # From code/results/eda/
        (code_root / "results" / "eda" / "genre_analysis.png", "genre_analysis.png"),

        # From code/results/synthetic_users/
        (code_root / "results" / "synthetic_users" / "user_archetypes.png", "user_archetypes.png"),
        (code_root / "results" / "synthetic_users" / "interaction_heatmap.png", "interaction_heatmap.png"),

        # From code/results/evaluation/
        (code_root / "results" / "evaluation" / "confusion_matrix.png", "confusion_matrix.png"),

        # From code/results/visualisations/
        (code_root / "results" / "visualisations" / "feature_embedding_pca.png", "feature_embedding_pca.png"),
        (code_root / "results" / "visualisations" / "feature_embedding_tsne.png", "feature_embedding_tsne.png"),
        (code_root / "results" / "visualisations" / "cross_genre_similarity.png", "cross_genre_similarity.png"),

        # From code/results/dissertation_figures/
        (code_root / "results" / "dissertation_figures" / "fig_05_method_comparison.png", "method_comparison.png"),
        (code_root / "results" / "dissertation_figures" / "fig_06_significance_heatmap.png", "significance_heatmap.png"),
        (code_root / "results" / "dissertation_figures" / "system_architecture.png", "system_architecture.png"),

        # From code/results/baseline_comparison/
        (code_root / "results" / "baseline_comparison" / "performance_comparison.png", "baseline_performance.png"),

        # From code/results/ablation_studies/
        (code_root / "results" / "ablation_studies" / "ablation_overview.png", "ablation_overview.png"),
    ]
    
    copied = 0
    skipped = 0
    missing = 0
    
    for src_path, dst_name in figure_mappings:
        dst_path = figures_dir / dst_name
        
        if not src_path.exists():
            logger.warning("Source file does not exist: %s", src_path)
            missing += 1
            continue
        
        if dst_path.exists() and not overwrite:
            logger.info("Skipping (already exists): %s", dst_name)
            skipped += 1
            continue
        
        try:
            shutil.copy2(src_path, dst_path)
            logger.info("Copied: %s -> %s", src_path.name, dst_name)
            copied += 1
        except Exception as e:
            logger.error("Failed to copy %s: %s", src_path, e)
    
    logger.info("Sync complete: %d copied, %d skipped, %d missing", copied, skipped, missing)


def main():
    """Entry point for syncing figures."""
    parser = argparse.ArgumentParser(
        description="Sync figures from code directories to workspace figures directory"
    )
    parser.add_argument(
        "--code-root",
        type=str,
        default="./",
        help="Root directory of the code (default: ./)",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default="../figures",
        help="Target figures directory (default: ../figures)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Don't overwrite existing files",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    code_root = Path(args.code_root).resolve()
    figures_dir = Path(args.figures_dir).resolve()
    
    # Set up logging with file handler
    figures_dir.mkdir(parents=True, exist_ok=True)
    log_file = figures_dir / "sync_figures.log"
    configure_logging(args.log_level, log_file=str(log_file))
    logger = setup_logger("sync_figures")
    logger.info("Logging to file: %s", log_file)
    logger.info("Syncing figures from %s to %s", code_root, figures_dir)
    
    sync_figures(
        code_root=code_root,
        figures_dir=figures_dir,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()

