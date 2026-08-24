#!/usr/bin/env python3
"""Generate a fail-closed index of every results subdirectory.

This script classifies every results subdirectory present after a run,
purely from directory existence (it runs no experiment logic and computes
no statistics), into:

- CANONICAL: the dissertation's numerical source of truth.
- EXPLORATORY: a real result, but not comparable to the canonical task
  (different sample, task, or relevance definition).
- DIAGNOSTIC: internal/consistency evidence, not a recommendation-quality
  result.
- DESCRIPTIVE: describes the dataset or synthetic population, not a
  performance result.
- PROCESS: an intermediate pipeline artefact with no standalone claim.
Any directory without an explicit classification is rejected.  This makes an
unreviewed result incapable of entering a release tree merely by existing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from src.utils.logger_config import configure_logging, setup_logger

logger = setup_logger("evidence_index")


# (directory name, status, one-line reason).  This is the exact output roster
# of the final release pipeline, in pipeline order.  Historical result-tree
# names are intentionally absent and therefore fail closed.
_CLASSIFICATION: List[tuple[str, str, str]] = [
    (
        "track_processing",
        "PROCESS",
        "Strict track selection, extraction records and the checksummed compact "
        "feature bundle. No standalone recommendation-quality claim.",
    ),
    (
        "eda",
        "DESCRIPTIVE",
        "Strict 72-feature summaries of the selected 4,000-track catalogue. "
        "No recommendation-quality claim.",
    ),
    (
        "synthetic_users",
        "DESCRIPTIVE",
        "The one immutable 200-user population and its disjoint train, "
        "validation and test partition.",
    ),
    (
        "configuration_selection",
        "CANONICAL",
        "Complete validation-only selection over the declared path and "
        "baseline grids. Test records are outside this stage boundary.",
    ),
    (
        "baseline_comparison",
        "CANONICAL",
        "The one-use warm test comparison and sole numerical source for the "
        "headline recommendation-quality claims.",
    ),
    (
        "cold_start_comparison",
        "CANONICAL",
        "The additive withheld-item comparison. It is a different estimand "
        "from the warm comparison and is interpreted with its decomposition.",
    ),
    (
        "evaluation",
        "DIAGNOSTIC",
        "The deterministic held-out five-fold genre diagnostic and its "
        "confusion matrix. It is not a recommendation-quality result.",
    ),
    (
        "robustness",
        "CANONICAL",
        "Result-backed bootstrap, inference, stability, cross-genre and "
        "withheld cold-hit/warm-hit analyses bound to the validated runs.",
    ),
    (
        "dissertation_figures",
        "CANONICAL",
        "Headline method and significance figures generated from the "
        "validated warm run.",
    ),
    (
        "synthetic_user_figures",
        "DESCRIPTIVE",
        "Figures generated strictly from the immutable synthetic population.",
    ),
    (
        "dissertation_package",
        "CANONICAL",
        "Checksummed tables and vector figure derived only from the validated "
        "warm comparison.",
    ),
    (
        "dissertation_package_cold_start",
        "CANONICAL",
        "Checksummed tables and vector figure derived only from the validated "
        "additive withheld-item comparison.",
    ),
    (
        "figures",
        "RELEASE",
        "The exact seven PNG files cited by the corrected dissertation.",
    ),
    (
        "timing",
        "DIAGNOSTIC",
        "Hardware-specific wall-clock measurements, separately classified "
        "because they are intentionally non-deterministic.",
    ),
    (
        "release",
        "PROCESS",
        "The evidence index, complete checksum inventory and fail-closed final "
        "release validation seal.",
    ),
]


def build_evidence_index(results_dir: str | Path) -> List[Dict[str, str]]:
    """Classify every results subdirectory that is actually present.

    Only directories found under ``results_dir`` are included, in the order
    declared above. Any present directory without an explicit classification
    aborts the release.
    """

    root = Path(results_dir)
    present = {entry.name for entry in root.iterdir() if entry.is_dir()} if root.is_dir() else set()
    known = {name for name, _, _ in _CLASSIFICATION}

    unknown = sorted(present - known)
    if unknown:
        raise ValueError(
            "unclassified results directories are forbidden: " + ", ".join(unknown)
        )

    index: List[Dict[str, str]] = []
    for name, status, reason in _CLASSIFICATION:
        if name in present:
            index.append({"directory": name, "status": status, "reason": reason})
    return index


def render_markdown(index: List[Dict[str, str]]) -> str:
    lines = [
        "# Evidence Index",
        "",
        "What each `results/` subdirectory actually is, generated "
        "automatically from which directories exist after this run. This "
        "does not replace reading the underlying data. It identifies where "
        "the dissertation's numbers "
        "should come from.",
        "",
        "| Directory | Status | What it is |",
        "|---|---|---|",
    ]
    for entry in index:
        lines.append(f"| `{entry['directory']}/` | **{entry['status']}** | {entry['reason']} |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a written index of what each results subdirectory is."
    )
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    index = build_evidence_index(args.results_dir)
    markdown = render_markdown(index)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    logger.info("Evidence index written to %s (%d directories classified)", output_path, len(index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
