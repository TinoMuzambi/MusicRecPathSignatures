"""Tests-first wiring contract for the one complete release pipeline."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from src.scripts.run_release_pipeline import (
    RELEASE_STAGE_NAMES,
    ReleasePipelineError,
    _execute_release_pipeline_for_test,
    _require_executed_code_matches_repository,
    _write_run_validation_marker,
    execute_release_pipeline,
    main,
)


RELEASE_N_JOBS = 4


EXPECTED_STAGES = (
    "preflight_and_raw_custody",
    "track_processing",
    "strict_eda",
    "immutable_population",
    "validation_selection",
    "warm_and_withheld_evaluation",
    "diagnostics_and_robustness",
    "figures_and_packages",
    "release_seal",
)


def _inputs(tmp_path: Path):
    tracks = tmp_path / "tracks.csv"
    tracks.write_text("metadata\n", encoding="utf-8")
    audio = tmp_path / "audio"
    audio.mkdir()
    repository = tmp_path / "repository"
    repository.mkdir()
    return tracks, audio, repository


def test_pipeline_executes_every_stage_once_in_frozen_order(tmp_path):
    from src.scripts import run_release_pipeline

    assert getattr(run_release_pipeline, "RELEASE_N_JOBS", None) == RELEASE_N_JOBS
    tracks, audio, repository = _inputs(tmp_path)
    calls = []

    def stage(name):
        def run(state):
            calls.append(name)
            if name == EXPECTED_STAGES[0]:
                state.run_root.mkdir()
        return run

    stages = {name: stage(name) for name in EXPECTED_STAGES}
    result = _execute_release_pipeline_for_test(
        run_root=tmp_path / "run",
        tracks_csv=tracks,
        audio_root=audio,
        repository_root=repository,
        n_jobs=RELEASE_N_JOBS,
        stage_functions=stages,
    )
    assert RELEASE_STAGE_NAMES == EXPECTED_STAGES
    assert calls == list(EXPECTED_STAGES)
    assert result == (tmp_path / "run").resolve()


def test_pipeline_rejects_partial_stage_roster_and_stops_on_failure(tmp_path):
    tracks, audio, repository = _inputs(tmp_path)
    with pytest.raises(ReleasePipelineError, match="stage roster"):
        _execute_release_pipeline_for_test(
            run_root=tmp_path / "partial",
            tracks_csv=tracks,
            audio_root=audio,
            repository_root=repository,
            n_jobs=RELEASE_N_JOBS,
            stage_functions={EXPECTED_STAGES[0]: lambda _state: None},
        )

    calls = []
    stages = {}
    for name in EXPECTED_STAGES:
        if name == "strict_eda":
            def fail(_state, _name=name):
                calls.append(_name)
                raise RuntimeError("bounded failure")
            stages[name] = fail
        else:
            def run(state, _name=name):
                calls.append(_name)
                if _name == EXPECTED_STAGES[0]:
                    state.run_root.mkdir()
            stages[name] = run
    with pytest.raises(RuntimeError, match="bounded failure"):
        _execute_release_pipeline_for_test(
            run_root=tmp_path / "failed",
            tracks_csv=tracks,
            audio_root=audio,
            repository_root=repository,
            n_jobs=RELEASE_N_JOBS,
            stage_functions=stages,
        )
    assert calls == list(EXPECTED_STAGES[:3])


def test_pipeline_refuses_existing_root_and_invalid_parallelism(tmp_path):
    tracks, audio, repository = _inputs(tmp_path)
    existing = tmp_path / "existing"
    existing.mkdir()
    stages = {name: lambda _state: None for name in EXPECTED_STAGES}
    with pytest.raises(ReleasePipelineError, match="already exists"):
        _execute_release_pipeline_for_test(
            run_root=existing,
            tracks_csv=tracks,
            audio_root=audio,
            repository_root=repository,
            n_jobs=RELEASE_N_JOBS,
            stage_functions=stages,
        )
    for invalid in (None, 0, 2):
        with pytest.raises(ReleasePipelineError, match="n_jobs"):
            _execute_release_pipeline_for_test(
                run_root=tmp_path / f"invalid-{invalid}",
                tracks_csv=tracks,
                audio_root=audio,
                repository_root=repository,
                n_jobs=invalid,
                stage_functions=stages,
            )

    with pytest.raises(ReleasePipelineError, match="external|overlap"):
        _execute_release_pipeline_for_test(
            run_root=repository / "forbidden-run",
            tracks_csv=tracks,
            audio_root=audio,
            repository_root=repository,
            n_jobs=RELEASE_N_JOBS,
            stage_functions=stages,
        )
    with pytest.raises(ReleasePipelineError, match="external|overlap"):
        _execute_release_pipeline_for_test(
            run_root=audio / "forbidden-run",
            tracks_csv=tracks,
            audio_root=audio,
            repository_root=repository,
            n_jobs=RELEASE_N_JOBS,
            stage_functions=stages,
        )


def test_cli_forwards_only_explicit_non_overridable_inputs(tmp_path, monkeypatch):
    tracks, audio, repository = _inputs(tmp_path)
    captured = {}

    def fake_execute(**kwargs):
        captured.update(kwargs)
        return Path(kwargs["run_root"])

    monkeypatch.setattr(
        "src.scripts.run_release_pipeline.execute_release_pipeline", fake_execute
    )
    result = main(
        [
            "--run-root", str(tmp_path / "run"),
            "--tracks-csv", str(tracks),
            "--audio-root", str(audio),
            "--repository-root", str(repository),
            "--n-jobs", str(RELEASE_N_JOBS),
        ]
    )
    assert result == 0
    assert captured == {
        "run_root": str(tmp_path / "run"),
        "tracks_csv": str(tracks),
        "audio_root": str(audio),
        "repository_root": str(repository),
        "n_jobs": RELEASE_N_JOBS,
    }

    with pytest.raises(SystemExit) as error:
        main(
            [
                "--run-root", str(tmp_path / "missing-jobs"),
                "--tracks-csv", str(tracks),
                "--audio-root", str(audio),
                "--repository-root", str(repository),
            ]
        )
    assert error.value.code == 2


def test_run_validation_marker_is_deterministic_and_inventoried(
    tmp_path, monkeypatch
):
    import hashlib
    import json

    from src.utils.provenance import canonical_json_bytes

    run = tmp_path / "canonical-run"
    run.mkdir()
    inventory_path = run / "checksum_inventory.json"
    inventory_path.write_bytes(
        canonical_json_bytes(
            {"schema_version": 1, "run_id": "a" * 64, "files": {}}
        )
        + b"\n"
    )
    summary = {
        "schema_version": 1,
        "run_id": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "validated_methods": ["path_signature_cosine"],
        "validated_user_rows": 200,
    }
    calls = []

    def fake_validate(path):
        calls.append(Path(path))
        return summary

    monkeypatch.setattr(
        "src.scripts.create_dissertation_package.validate_run_directory",
        fake_validate,
    )
    _write_run_validation_marker(run)

    marker = run / "validated.marker"
    assert marker.read_bytes() == canonical_json_bytes(summary) + b"\n"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory["files"]["validated.marker"] == hashlib.sha256(
        marker.read_bytes()
    ).hexdigest()
    assert calls == [run, run]


def test_executed_code_must_be_the_declared_outer_repository_code(tmp_path):
    repository = tmp_path / "outer"
    declared_code = repository / "code"
    declared_code.mkdir(parents=True)
    other_code = tmp_path / "other-code"
    other_code.mkdir()

    assert _require_executed_code_matches_repository(
        repository, executed_code_root=declared_code
    ) == declared_code.resolve()
    with pytest.raises(ReleasePipelineError, match="executed code"):
        _require_executed_code_matches_repository(
            repository, executed_code_root=other_code
        )


def test_release_figure_stage_disables_timestamped_side_outputs():
    source = Path("src/scripts/run_release_pipeline.py").read_text(encoding="utf-8")
    assert '"--release-mode"' in source


def test_shell_runner_requires_frozen_parallelism_lock_and_twelve_hour_timeout():
    source = Path("run_complete_pipeline.sh").read_text(encoding="utf-8")

    assert "--n-jobs N" in source
    assert "--foreground" not in source
    assert "43140s" in source
    assert "--kill-after=60s" in source
    assert "PYTHONHASHSEED=0" in source
    assert 'ARGS+=(--n-jobs "$N_JOBS")' not in source


def test_public_pipeline_cannot_inject_replacement_stage_functions():
    assert "stage_functions" not in inspect.signature(
        execute_release_pipeline
    ).parameters


def test_python_runner_holds_a_nonblocking_global_lock_before_creating_root(
    tmp_path, monkeypatch
):
    from src.scripts import run_release_pipeline as pipeline_module

    lock = getattr(pipeline_module, "_exclusive_release_lock", None)
    assert callable(lock)
    lock_path = tmp_path / "release.lock"
    monkeypatch.setattr(pipeline_module, "RELEASE_LOCK_PATH", lock_path)
    tracks, audio, repository = _inputs(tmp_path)
    run_root = tmp_path / "locked-run"

    with lock(lock_path):
        with pytest.raises(ReleasePipelineError, match="already running|lock"):
            execute_release_pipeline(
                run_root=run_root,
                tracks_csv=tracks,
                audio_root=audio,
                repository_root=repository,
                n_jobs=RELEASE_N_JOBS,
            )
    assert not run_root.exists()
