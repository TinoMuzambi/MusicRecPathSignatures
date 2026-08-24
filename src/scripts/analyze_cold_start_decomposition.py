#!/usr/bin/env python3
"""Decompose the cold-start comparison's aggregate Precision@5 by hit type.

R9 evidence audit F-07: the cold-start run's test sets contain both cold
(never-observed) items and ordinary warm items, so its aggregate Precision@5
mixes the two. Read naively, a method's aggregate can look strong purely
because it recommends warm items well, while contributing nothing to actual
cold-item retrieval -- exactly what the audit found for every collaborative
baseline (0% of their top-5 recommendations are ever cold items). No saved
artefact made this decomposition available; it had to be reconstructed by
hand from raw per-user rows.

This script computes, for every method the canonical cold-start comparison
saved, each user's real fraction of cold items in their top-k
recommendations and the real precision contributed separately by cold-item
hits and warm-item hits -- entirely from already-saved canonical rows
(``results/cold_start_comparison/methods/*.jsonl`` and
``dataset_manifest.json``'s ``cold_start_configuration``). No rerun is
required to produce this decomposition once a cold-start run exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, Set

from src.utils.logger_config import configure_logging, setup_logger

logger = setup_logger("cold_start_decomposition")


def load_cold_track_ids(dataset_manifest_path: str | Path) -> Set[str]:
    """Return the real set of cold track IDs from a cold-start dataset manifest."""

    with open(dataset_manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    cold_ids = manifest["dataset"]["cold_start_configuration"]["cold_track_ids"]
    return {str(track_id) for track_id in cold_ids}


def decompose_method(
    method_jsonl_path: str | Path,
    *,
    cold_track_ids: Mapping[str, object] | Set[str],
    k: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Per-user cold/warm precision decomposition for one method's real rows.

    For each user: what fraction of their real top-``k`` recommendations are
    cold items, and how much of their real Precision@``k`` is contributed by
    cold-item hits versus warm-item hits (the two must sum to the ordinary
    Precision@k).
    """

    path = Path(method_jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"method output not found: {path}")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    cold_ids = set(cold_track_ids)
    if any(not isinstance(track_id, str) or not track_id for track_id in cold_ids):
        raise ValueError("cold track IDs must be strings")
    decomposition: Dict[str, Dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        user_id = row["user_id"]
        if not isinstance(user_id, str) or not user_id or user_id in decomposition:
            raise ValueError(f"{path}: duplicate or invalid user ID")
        relevance = set(row["relevance_ids"])
        recommendations = row["recommendations"][:k]
        if len(recommendations) != k or len(set(recommendations)) != k:
            raise ValueError(f"{path}: {user_id} does not have an exact unique top-{k}")
        observed = set(row.get("observed_ids", ()))
        leaked = observed & cold_ids
        if leaked:
            raise ValueError(f"{path}: cold tracks occur in observed interactions")
        n_cold_in_topk = sum(1 for track_id in recommendations if track_id in cold_ids)
        cold_hits = sum(
            1 for track_id in recommendations if track_id in cold_ids and track_id in relevance
        )
        warm_hits = sum(
            1
            for track_id in recommendations
            if track_id not in cold_ids and track_id in relevance
        )
        denominator = len(recommendations)
        decomposition[user_id] = {
            "cold_fraction_in_topk": n_cold_in_topk / denominator,
            "precision_from_cold_hits": cold_hits / denominator,
            "precision_from_warm_hits": warm_hits / denominator,
            "precision_at_k": (cold_hits + warm_hits) / denominator,
        }
        saved_precision = row.get("metrics", {}).get("precision", {}).get(str(k))
        if saved_precision is not None and abs(float(saved_precision) - decomposition[user_id]["precision_at_k"]) > 1e-15:
            raise ValueError(f"{path}: decomposition does not recombine to saved Precision@{k}")
    if not decomposition:
        raise ValueError(f"{path} contained no rows")
    return decomposition


def compare_cold_start_decomposition(
    methods_dir: str | Path,
    *,
    cold_track_ids: Mapping[str, object] | Set[str],
    k: int = 5,
) -> Dict[str, Dict[str, object]]:
    """Real per-method cold/warm decomposition, averaged over each method's real seeds.

    Follows the canonical runner's own seed-then-user aggregation order: each
    user's value is first averaged across that method's declared seeds, then
    the population mean is taken.
    """

    directory = Path(methods_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"methods directory not found: {directory}")

    seed_decompositions: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = {}
    method_files = sorted(directory.glob("*.jsonl"))
    if not method_files:
        # A present-but-empty methods directory indicates a real upstream
        # problem, not "nothing to decompose" -- fail closed rather than
        # silently writing an empty summary (independent-review follow-up).
        raise ValueError(f"no method output files found in {directory}")
    for jsonl_path in method_files:
        method_key = jsonl_path.stem
        per_user = decompose_method(jsonl_path, cold_track_ids=cold_track_ids, k=k)
        if "__seed_" in method_key:
            method_id, seed_text = method_key.rsplit("__seed_", 1)
            seed_decompositions.setdefault(method_id, {})[int(seed_text)] = per_user
        else:
            seed_decompositions[method_key] = {0: per_user}

    fields = (
        "cold_fraction_in_topk",
        "precision_from_cold_hits",
        "precision_from_warm_hits",
        "precision_at_k",
    )
    summary: Dict[str, Dict[str, object]] = {}
    for method_id, seed_rows in seed_decompositions.items():
        seeds = sorted(seed_rows)
        user_ids = sorted(seed_rows[seeds[0]])
        per_user_mean = {
            field: {
                user_id: sum(seed_rows[seed][user_id][field] for seed in seeds) / len(seeds)
                for user_id in user_ids
            }
            for field in fields
        }
        method_summary: Dict[str, object] = {"n_users": len(user_ids), "n_seeds": len(seeds)}
        for field in fields:
            values = list(per_user_mean[field].values())
            method_summary[f"mean_{field}"] = sum(values) / len(values)
        summary[method_id] = method_summary
    return summary


def _write_csv(summary: Mapping[str, Mapping[str, object]], path: Path) -> None:
    columns = (
        "method",
        "mean_cold_fraction_in_topk",
        "mean_precision_from_cold_hits",
        "mean_precision_from_warm_hits",
        "mean_precision_at_k",
        "n_users",
        "n_seeds",
    )
    lines = [",".join(columns)]
    for method_id in sorted(summary):
        row = summary[method_id]
        values = [
            f"{row[col]:.6f}" if isinstance(row[col], float) else str(row[col])
            for col in columns[1:]
        ]
        lines.append(",".join([method_id, *values]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose the cold-start comparison's aggregate Precision@5 into "
            "cold-item and warm-item contributions, per method, from already-"
            "saved cold-start comparison rows."
        )
    )
    parser.add_argument("--cold-start-comparison-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    cold_start_dir = Path(args.cold_start_comparison_dir)
    cold_track_ids = load_cold_track_ids(cold_start_dir / "dataset_manifest.json")
    logger.info("Loaded %d real cold track IDs", len(cold_track_ids))

    summary = compare_cold_start_decomposition(
        cold_start_dir / "methods", cold_track_ids=cold_track_ids, k=args.k
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cold_start_decomposition.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(summary, out / "cold_start_decomposition.csv")
    logger.info("Cold-start decomposition written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
