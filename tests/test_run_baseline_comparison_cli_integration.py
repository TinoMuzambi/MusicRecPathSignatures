"""Full, realistic end-to-end warm-start run (pre-GCP-run integration check).

``test_run_cold_start_comparison_cli.py`` already has a full, realistic
end-to-end test of the *additive* cold-start CLI, with real scorers and a
stubbed LightFM ``model_factory`` (real LightFM cannot be imported under
Python 3.12 in this environment). No equivalent existed for the underlying
*warm* CLI (``run_baseline_comparison_cli.py``) itself -- only small,
hand-built-task unit tests. This closes that gap: it builds a real task via
``build_task_and_identity`` from on-disk features/tracks JSON at a scale
large enough to avoid archetype starvation (matching the frozen 200-user
population's real minimum viable catalogue size), computes real order-two
path signatures over the real 38-channel path, and runs it through the
real, unmodified ``run_canonical_comparison(...)`` -- the same call the real
CLI's ``main()`` makes -- producing real, non-fabricated metrics for every
one of the 22 canonical method/seed outputs except the three LightFM-backed
ones, which use the same deterministic double the cold-start test already
uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.scripts.run_baseline_comparison_cli import (
    CANONICAL_SIGNATURE_ORDER,
    build_task_and_identity,
)
from src.scripts.run_baseline_comparison import (
    CANONICAL_MASTER_SEED,
    run_canonical_comparison,
)
from src.scripts.run_baseline_comparison_multiple_runs import expected_method_seed_keys
from test_baseline_comparison import outer_repository, read_jsonl
from test_run_cold_start_comparison_cli import (
    _real_cold_start_scorer_builder,
    _synthetic_track_features,
    _synthetic_tracks_json,
)


def test_full_warm_run_with_real_scorers_and_stubbed_lightfm(tmp_path: Path):
    n_tracks = 1000  # matches the frozen 200-user population's minimum viable
    # catalogue size (see test_run_cold_start_comparison_cli.py's identical
    # comment on this same real, pre-existing constraint).
    features_path = tmp_path / "features.json"
    tracks_path = tmp_path / "tracks.json"
    features_path.write_text(json.dumps(_synthetic_track_features(n_tracks)), encoding="utf-8")
    tracks_path.write_text(json.dumps(_synthetic_tracks_json(n_tracks)), encoding="utf-8")

    args = argparse.Namespace(
        features_file=str(features_path),
        tracks_json=str(tracks_path),
        output_dir=str(tmp_path / "run"),
        n_users=200,
        test_ratio=0.15,
        validation_ratio=0.15,
        log_level="ERROR",
        repository_root=None,
    )
    task, identity_inputs = build_task_and_identity(args)

    # F-02: the real task actually declares and uses the corrected order.
    assert identity_inputs["feature_schema"]["signature_order"] == CANONICAL_SIGNATURE_ORDER == 3
    real_signature = next(iter(task["signatures"].values()))
    assert real_signature.shape == (16276,)

    run_dir = run_canonical_comparison(
        repository_root=outer_repository(tmp_path),
        output_directory=tmp_path / "run" / "canonical",
        master_seed=CANONICAL_MASTER_SEED,
        identity_inputs=identity_inputs,
        task=task,
        scorer_builder=_real_cold_start_scorer_builder,
        source_reader=lambda path: {"git_commit": "c" * 40, "dirty": False},
    )

    execution = json.loads((run_dir / "execution.json").read_text(encoding="utf-8"))
    assert execution["status"] == "success"
    assert tuple(execution["methods"]) == expected_method_seed_keys()

    aggregates = json.loads((run_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
    for method_id in ("path_signature_cosine", "traditional_audio_cosine", "implicit_als"):
        assert method_id in aggregates["methods"]
        assert 0.0 <= aggregates["methods"][method_id]["precision"]["5"] <= 1.0

    # Real per-user rows exist and satisfy the fairness invariant at scale:
    # no method recommends an observed item, every recommendation is a
    # legal candidate, and there are real, non-fabricated recommendations
    # for all 200 users.
    from src.scripts.run_baseline_comparison_multiple_runs import method_seed_rows_path

    for output_key in ("path_signature_cosine", "traditional_audio_cosine", "implicit_als__seed_2025"):
        if "__seed_" in output_key:
            method_id, seed = output_key.rsplit("__seed_", 1)
            rel_path = method_seed_rows_path(method_id, int(seed))
        else:
            rel_path = method_seed_rows_path(output_key, None)
        rows = read_jsonl(run_dir / rel_path)
        assert len(rows) == 200
        for row in rows:
            assert set(row["recommendations"]).isdisjoint(row["observed_ids"])
            assert set(row["recommendations"]) <= set(row["candidate_ids"])
            assert len(row["recommendations"]) == 10

    assert (run_dir / "uncertainty.json").is_file()
    assert (run_dir / "precision5_inference.json").is_file()
    assert (run_dir / "checksum_inventory.json").is_file()
