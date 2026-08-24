"""Copy exactly the seven cited figures into a fresh release directory."""

import argparse
import hashlib
import shutil
from pathlib import Path
from src.utils.logger_config import configure_logging, setup_logger

# Module-level logger (will be configured in main())
logger = setup_logger("sync_figures")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_figures(
    code_root: Path,
    figures_dir: Path,
    ablation_dir: Path | None,
    overwrite: bool = True,
) -> None:
    """
    Sync figures from code directories to workspace figures directory.
    
    Args:
        code_root: Root directory of the code (e.g., /workspace/code)
        figures_dir: Target figures directory (e.g., /workspace/figures)
        ablation_dir: Exact validated ablation run directory
        overwrite: Whether to overwrite existing files
    """
    if ablation_dir is None:
        raise ValueError("an exact validated ablation directory is required")
    ablation_source = ablation_dir / "ablation_overview.png"
    
    # Mapping of source files to destination names
    figure_mappings = [
        (code_root / "results" / "synthetic_user_figures" / "user_archetypes.png", "user_archetypes.png"),
        (code_root / "results" / "synthetic_user_figures" / "interaction_heatmap.png", "interaction_heatmap.png"),
        (code_root / "results" / "evaluation" / "confusion_matrix.png", "confusion_matrix.png"),
        (code_root / "results" / "eda" / "missing_value_outlier_summary.png", "missing_value_outlier_summary.png"),
        (code_root / "results" / "dissertation_figures" / "fig_05_method_comparison.png", "method_comparison.png"),
        (code_root / "results" / "dissertation_figures" / "fig_06_significance_heatmap.png", "significance_heatmap.png"),
        (ablation_source, "ablation_overview.png"),
    ]

    for src_path, _ in figure_mappings:
        if src_path.is_symlink() or not src_path.is_file():
            raise FileNotFoundError(
                f"required figure must be a regular, non-symlink file: {src_path}"
            )
    if figures_dir.exists():
        existing = tuple(figures_dir.iterdir())
        if existing and not overwrite:
            raise FileExistsError(f"figure release directory is not empty: {figures_dir}")
        expected_names = {name for _, name in figure_mappings}
        unrelated = sorted(path.name for path in existing if path.name not in expected_names)
        if unrelated:
            raise FileExistsError(
                "figure release directory contains unrelated files: " + ", ".join(unrelated)
            )
    else:
        figures_dir.mkdir(parents=True)

    for src_path, dst_name in figure_mappings:
        dst_path = figures_dir / dst_name
        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite cited figure: {dst_path}")
        shutil.copy2(src_path, dst_path)
        if _sha256_file(src_path) != _sha256_file(dst_path):
            raise OSError(f"copied figure checksum mismatch: {dst_name}")
        logger.info("Copied: %s -> %s", src_path, dst_path)


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
        "--ablation-dir",
        type=str,
        required=True,
        help="Exact validated ablation run directory",
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
    ablation_dir = Path(args.ablation_dir).resolve()
    
    configure_logging(args.log_level)
    logger.info("Syncing figures from %s to %s", code_root, figures_dir)
    
    sync_figures(
        code_root=code_root,
        figures_dir=figures_dir,
        ablation_dir=ablation_dir,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
