#!/usr/bin/env python3
"""Compare real cross-genre recommendation behaviour across all six methods.

R9 evidence audit R-07/F-13: the examiner asked the dissertation to
"[c]ompare cross-genre behaviour against baselines before attributing it
uniquely to path signatures" (R-07). No such comparison existed. The
dissertation's "high similarity scores (mean 0.995)" cross-genre claim is
drawn only from a hand-picked top-N pair list produced by
``find_cross_genre_similar_songs.py`` -- not a representative measurement,
and not compared against any baseline (F-13).

This script instead computes, for every method the canonical comparison
saved, the real fraction of each user's real top-k recommendations whose
genre differs from the genre of that user's real query track -- directly
comparable across all six methods, entirely from already-saved canonical
rows (``results/baseline_comparison/methods/*.jsonl``). No rerun is
required to produce this comparison once a canonical run exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping

from src.utils.logger_config import configure_logging, setup_logger

logger = setup_logger("cross_genre_behaviour")


def load_genre_map(tracks_json_path: str | Path) -> Dict[str, str]:
    """Return ``{track_id: genre}`` from a selected-tracks JSON file."""

    with open(tracks_json_path, "r", encoding="utf-8") as handle:
        tracks = json.load(handle)
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("selected-tracks file must contain a non-empty list")
    result: Dict[str, str] = {}
    for track in tracks:
        if not isinstance(track, Mapping):
            raise ValueError("selected-track metadata row must be an object")
        track_id = track.get("track_id")
        genre = track.get("genre")
        if (
            not isinstance(track_id, (str, int))
            or isinstance(track_id, bool)
            or not str(track_id)
            or not isinstance(genre, str)
            or not genre.strip()
        ):
            raise ValueError("track ID and genre metadata must be non-empty")
        canonical_id = str(track_id)
        if canonical_id in result:
            raise ValueError(f"duplicate genre metadata for track {canonical_id}")
        result[canonical_id] = genre.strip()
    return result


def cross_genre_rate_by_user(
    method_jsonl_path: str | Path,
    genre_map: Mapping[str, str],
    *,
    k: int = 5,
) -> Dict[str, float]:
    """Fraction of each user's real top-``k`` recommendations that cross a
    genre boundary relative to the genre of that user's real query track.

    Every query and recommendation must have an explicit genre. Missing
    metadata aborts the analysis instead of being converted into a synthetic
    ``Unknown`` category.
    """

    path = Path(method_jsonl_path)
    if not path.is_file():
        raise FileNotFoundError(f"method output not found: {path}")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    rates: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        user_id = row["user_id"]
        if not isinstance(user_id, str) or not user_id or user_id in rates:
            raise ValueError(f"{path}: duplicate or invalid user ID")
        query_id = row["query_track_id"]
        if query_id not in genre_map:
            raise ValueError(f"{path}: missing genre metadata for query {query_id}")
        query_genre = genre_map[query_id]
        recommendations = row["recommendations"][:k]
        if len(recommendations) != k or len(set(recommendations)) != k:
            raise ValueError(f"{path}: {user_id} does not have an exact unique top-{k}")
        missing_genres = sorted(set(recommendations) - set(genre_map))
        if missing_genres:
            raise ValueError(
                f"{path}: missing genre metadata for recommendations {missing_genres}"
            )
        cross = sum(
            1
            for track_id in recommendations
            if genre_map[track_id] != query_genre
        )
        rates[user_id] = cross / len(recommendations)
    if not rates:
        raise ValueError(f"{path} contained no rows")
    return rates


def compare_cross_genre_behaviour(
    methods_dir: str | Path,
    genre_map: Mapping[str, str],
    *,
    k: int = 5,
) -> Dict[str, Dict[str, object]]:
    """Real per-method cross-genre rate, averaged over each method's real seeds.

    For a stochastic method, each user's rate is first averaged across that
    method's declared seeds, then the population mean is taken -- the same
    seed-then-user aggregation order the canonical runner itself uses for
    every other metric, so this figure is directly comparable to the
    headline Precision@5 table.
    """

    directory = Path(methods_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"methods directory not found: {directory}")

    seed_rates_by_method: Dict[str, Dict[int, Dict[str, float]]] = {}
    for jsonl_path in sorted(directory.glob("*.jsonl")):
        method_key = jsonl_path.stem
        rates = cross_genre_rate_by_user(jsonl_path, genre_map, k=k)
        if "__seed_" in method_key:
            method_id, seed_text = method_key.rsplit("__seed_", 1)
            seed_rates_by_method.setdefault(method_id, {})[int(seed_text)] = rates
        else:
            seed_rates_by_method[method_key] = {0: rates}

    summary: Dict[str, Dict[str, object]] = {}
    for method_id, seed_rates in seed_rates_by_method.items():
        seeds = sorted(seed_rates)
        user_ids = sorted(seed_rates[seeds[0]])
        per_user_mean = {
            user_id: sum(seed_rates[seed][user_id] for seed in seeds) / len(seeds)
            for user_id in user_ids
        }
        values = list(per_user_mean.values())
        summary[method_id] = {
            "mean_cross_genre_rate": sum(values) / len(values),
            "n_users": len(values),
            "n_seeds": len(seeds),
        }
    return summary


def _write_csv(summary: Mapping[str, Mapping[str, object]], path: Path) -> None:
    lines = ["method,mean_cross_genre_rate,n_users,n_seeds"]
    for method_id in sorted(summary):
        row = summary[method_id]
        lines.append(
            f"{method_id},{row['mean_cross_genre_rate']:.6f},{row['n_users']},{row['n_seeds']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare real cross-genre recommendation behaviour across all "
            "six canonical methods, from already-saved baseline-comparison rows."
        )
    )
    parser.add_argument("--baseline-comparison-dir", type=str, required=True)
    parser.add_argument("--tracks-json", type=str, required=True)
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

    genre_map = load_genre_map(args.tracks_json)
    methods_dir = Path(args.baseline_comparison_dir) / "methods"
    summary = compare_cross_genre_behaviour(methods_dir, genre_map, k=args.k)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cross_genre_behaviour.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_csv(summary, out / "cross_genre_behaviour.csv")
    logger.info("Cross-genre behaviour comparison written to %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
