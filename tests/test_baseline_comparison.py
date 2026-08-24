"""MR-06 unit contract for the canonical runner and row writer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.scripts.run_baseline_comparison import (
    CanonicalRunError,
    run_canonical_comparison,
)
from src.scripts.run_baseline_comparison_multiple_runs import (
    expected_method_seed_keys,
)


def identity_inputs() -> dict:
    return {
        "source": {"git_commit": "placeholder", "dirty": False},
        "runtime": {"python": "test"},
        "dataset": {"selection_sha256": "1" * 64},
        "feature_schema": {"path_channels": ["time", "pitch"]},
        "task": {"name": "tiny-controlled-task"},
        "splits": {"sha256": "2" * 64},
        "index_maps": {"users_sha256": "3" * 64, "items_sha256": "4" * 64},
        "models": {"method_ids": list(expected_method_seed_keys())},
        "evaluation": {"cutoffs": [1, 5, 10], "primary": "precision@5"},
        "diagnostic_declarations": ["signature_norm_frequency_spearman"],
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    }


def tiny_task() -> dict:
    accepted = [f"t{index:02d}" for index in range(14)]
    return {
        "schema_version": 1,
        "source_track_ids": accepted + ["failed-track"],
        "accepted_track_ids": accepted,
        "catalogue_ids": accepted,
        "track_failures": [
            {"track_id": "failed-track", "stage": "signature", "reason_code": "invalid_path"}
        ],
        "users": {
            "u1": {
                "train": {"t00": {"interaction_score": 1.0}},
                "validation": {"t01": {"interaction_score": 2.0}},
                "test": {"t02": {"interaction_score": 5.0}},
            },
            "u2": {
                "train": {"t12": {"interaction_score": 1.0}},
                "validation": {"t13": {"interaction_score": 2.0}},
                "test": {"t00": {"interaction_score": 5.0}},
            },
        },
        "dataset_manifest": {"schema_version": 1, "selection": "tiny"},
    }


def outer_repository(tmp_path: Path) -> Path:
    root = tmp_path / "dissertation"
    root.mkdir(parents=True, exist_ok=True)
    (root / "AGENTS.md").write_text("test\n", encoding="utf-8")
    (root / "code").mkdir(exist_ok=True)
    (root / "latex").mkdir(exist_ok=True)
    return root


class RecordingScorerBuilder:
    def __init__(self, *, short_key: str | None = None, malformed_key: str | None = None):
        self.context = None
        self.calls: list[tuple[str, str, str, tuple[str, ...]]] = []
        self.short_key = short_key
        self.malformed_key = malformed_key

    def __call__(self, context):
        self.context = context
        assert "test" not in context
        assert all("test" not in record for record in context["observed_interactions"].values())
        scorers = {}
        for position, output_key in enumerate(expected_method_seed_keys()):
            reverse = position != 0

            def score(*, user_id, query_track_id, candidate_ids, observed_ids,
                      _key=output_key, _reverse=reverse):
                candidates = tuple(sorted(candidate_ids, reverse=_reverse))
                observed = tuple(sorted(observed_ids))
                self.calls.append((_key, user_id, query_track_id, observed))
                limit = 9 if _key == self.short_key else 10
                ranked = candidates[:limit]
                pairs = tuple(
                    (track_id, float(limit - index))
                    for index, track_id in enumerate(ranked)
                )
                if _key == self.malformed_key:
                    return pairs + ((observed[0], -1.0),)
                return pairs

            scorers[output_key] = score
        return scorers


def run_tiny(tmp_path: Path, *, builder=None, task=None, name="run") -> Path:
    root = outer_repository(tmp_path)
    return run_canonical_comparison(
        repository_root=root,
        output_directory=tmp_path / name,
        master_seed=2025,
        identity_inputs=identity_inputs(),
        task=tiny_task() if task is None else task,
        scorer_builder=builder or RecordingScorerBuilder(),
        source_reader=lambda path: {"git_commit": "a" * 40, "dirty": False},
    )


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_tiny_task_writes_all_22_aligned_outputs_and_fixed_metrics(tmp_path):
    builder = RecordingScorerBuilder()
    run_dir = run_tiny(tmp_path, builder=builder)

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    execution = json.loads((run_dir / "execution.json").read_text(encoding="utf-8"))
    assert execution["status"] == "success"
    assert tuple(execution["methods"]) == expected_method_seed_keys()
    assert manifest["run_id"] == execution["run_id"]
    assert set(builder.context["observed_interactions"]) == {"u1", "u2"}
    assert builder.context["observed_interactions"]["u1"] == {"t00": 1.0, "t01": 2.0}

    path_rows = read_jsonl(run_dir / "methods/path_signature_cosine.jsonl")
    baseline_rows = read_jsonl(run_dir / "methods/traditional_audio_cosine.jsonl")
    assert [row["user_id"] for row in path_rows] == ["u1", "u2"]
    assert path_rows[0]["query_track_id"] == "t01"
    assert path_rows[1]["query_track_id"] == "t13"
    assert path_rows[0]["metrics"]["precision"]["5"] == pytest.approx(0.2)
    assert baseline_rows[0]["metrics"]["precision"]["5"] == 0.0
    assert len({row["candidate_ids_sha256"] for row in path_rows + baseline_rows}) == 2
    assert all(len(row["recommendations"]) == 10 for row in path_rows)
    assert len(builder.calls) == 44
    assert all(call[3] in (("t00", "t01"), ("t12", "t13")) for call in builder.calls)

    aggregates = json.loads((run_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert aggregates["methods"]["path_signature_cosine"]["precision"]["5"] == pytest.approx(0.2)
    assert "training_variability" in aggregates["methods"]["implicit_als"]
    for method_id in (
        "lightfm_warp",
        "lightfm_warp_kos",
        "lightfm_latent_blend",
        "implicit_als",
    ):
        for metric_name in ("diversity", "novelty"):
            assert set(aggregates["methods"][method_id][metric_name]) == {"1", "5", "10"}
            assert all(
                record["status"] in {"available", "unavailable"}
                for record in aggregates["methods"][method_id][metric_name].values()
            )
    assert (run_dir / "uncertainty.json").is_file()
    assert (run_dir / "precision5_inference.json").is_file()


@pytest.mark.parametrize("seed", [True, 2025.0, "2025", None])
def test_master_seed_must_be_exact_builtin_integer(tmp_path, seed):
    with pytest.raises(CanonicalRunError, match="exact built-in integer"):
        run_canonical_comparison(
            repository_root=outer_repository(tmp_path),
            output_directory=tmp_path / "run",
            master_seed=seed,
            identity_inputs=identity_inputs(),
            task=tiny_task(),
            scorer_builder=RecordingScorerBuilder(),
            source_reader=lambda path: {"git_commit": "a" * 40, "dirty": False},
        )


def test_outer_repository_root_and_clean_source_are_mandatory(tmp_path):
    root = outer_repository(tmp_path)
    with pytest.raises(CanonicalRunError, match="outer dissertation repository"):
        run_canonical_comparison(
            repository_root=root / "code",
            output_directory=tmp_path / "bad-root",
            master_seed=2025,
            identity_inputs=identity_inputs(),
            task=tiny_task(),
            scorer_builder=RecordingScorerBuilder(),
            source_reader=lambda path: {"git_commit": "a" * 40, "dirty": False},
        )
    with pytest.raises(CanonicalRunError, match="dirty source"):
        run_canonical_comparison(
            repository_root=root,
            output_directory=tmp_path / "dirty",
            master_seed=2025,
            identity_inputs=identity_inputs(),
            task=tiny_task(),
            scorer_builder=RecordingScorerBuilder(),
            source_reader=lambda path: {"git_commit": "a" * 40, "dirty": True},
        )


def test_upstream_failure_partition_is_consumed_fail_closed(tmp_path):
    task = tiny_task()
    task["catalogue_ids"].append("failed-track")
    with pytest.raises(CanonicalRunError, match="failed track"):
        run_tiny(tmp_path, task=task)

    task = tiny_task()
    task["source_track_ids"].append("silently-dropped")
    with pytest.raises(CanonicalRunError, match="partition"):
        run_tiny(tmp_path, task=task)


@pytest.mark.parametrize("defect", ["short", "observed"])
def test_invalid_ranking_fails_run_and_writes_no_headline_aggregate(tmp_path, defect):
    key = "traditional_audio_cosine"
    builder = RecordingScorerBuilder(
        short_key=key if defect == "short" else None,
        malformed_key=key if defect == "observed" else None,
    )
    output = tmp_path / "failed-run"
    with pytest.raises(CanonicalRunError, match="insufficient_output|observed|candidate"):
        run_canonical_comparison(
            repository_root=outer_repository(tmp_path),
            output_directory=output,
            master_seed=2025,
            identity_inputs=identity_inputs(),
            task=tiny_task(),
            scorer_builder=builder,
            source_reader=lambda path: {"git_commit": "a" * 40, "dirty": False},
        )
    assert not output.exists()
    failed_dirs = tuple(tmp_path.glob(".failed-run.staging-*"))
    assert len(failed_dirs) == 1
    assert not (failed_dirs[0] / "aggregate_metrics.json").exists()
    failure_rows = read_jsonl(failed_dirs[0] / "method_failures.jsonl")
    assert len(failure_rows) == 1
    assert failure_rows[0]["output_key"] == key


def test_timing_sink_is_none_by_default_and_output_is_still_byte_identical(tmp_path):
    """Omitting timing_sink must not change any existing byte-for-byte behaviour."""

    first = run_tiny(tmp_path / "first_notiming", name="run")
    second = run_tiny(tmp_path / "second_notiming", name="run")
    first_files = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*") if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*") if path.is_file()
    }
    assert first_files == second_files


def test_timing_sink_records_one_real_elapsed_duration_per_call(tmp_path):
    """F-06: real per-method scoring latency, captured outside the checksummed run dir."""

    timing_sink: dict[str, list[float]] = {}
    root = outer_repository(tmp_path)
    run_dir = run_canonical_comparison(
        repository_root=root,
        output_directory=tmp_path / "timed-run",
        master_seed=2025,
        identity_inputs=identity_inputs(),
        task=tiny_task(),
        scorer_builder=RecordingScorerBuilder(),
        source_reader=lambda path: {"git_commit": "a" * 40, "dirty": False},
        timing_sink=timing_sink,
    )
    assert set(timing_sink) == set(expected_method_seed_keys())
    for output_key in expected_method_seed_keys():
        durations = timing_sink[output_key]
        assert len(durations) == 2  # one call per user (u1, u2)
        assert all(isinstance(value, float) and value >= 0.0 for value in durations)
    # The timing sink must never be written inside the checksummed run
    # directory: it would break run-to-run byte identity, since wall-clock
    # durations are inherently non-deterministic.
    checksum_inventory = json.loads((run_dir / "checksum_inventory.json").read_text(encoding="utf-8"))
    assert "timing.json" not in checksum_inventory["files"]
    assert not (run_dir / "timing.json").exists()


def test_fresh_runs_have_byte_identical_canonical_files(tmp_path):
    first = run_tiny(tmp_path / "first", name="run")
    second = run_tiny(tmp_path / "second", name="run")
    first_files = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*") if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*") if path.is_file()
    }
    assert first_files == second_files
