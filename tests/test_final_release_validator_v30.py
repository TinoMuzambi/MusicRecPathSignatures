"""Fail-closed structural contracts for the final scientific release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from src.scripts.validate_release import (
    CITED_FIGURE_NAMES,
    FINAL_RELEASE_DIRECTORIES,
    ReleaseValidationError,
    _validate_timing_stage,
    seal_release,
    validate_release,
)
from src.utils.provenance import canonical_json_bytes


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


@pytest.fixture
def structural_only(monkeypatch):
    monkeypatch.setattr(
        "src.scripts.validate_release._validate_scientific_contents",
        lambda *_args, **_kwargs: None,
    )


def _make_release(root: Path) -> dict[str, str]:
    from src.scripts.generate_evidence_index import (
        build_evidence_index,
        render_markdown,
    )
    from src.scripts.run_baseline_comparison_multiple_runs import (
        expected_method_seed_keys,
    )

    for name in FINAL_RELEASE_DIRECTORIES:
        (root / name).mkdir(parents=True)

    selection = {
        "schema_version": 1,
        "provenance": {
            "source": {
                "git_commit": "a" * 40,
                "scientific_source_sha256": "b" * 64,
            },
            "dataset": {"population_sha256": "c" * 64},
        },
        "selected": {"path": {"config_id": "order_2__all_channels"}},
    }
    selection_path = root / "configuration_selection/selection_manifest.json"
    _write_json(selection_path, selection)
    selection_sha = _sha(selection_path)

    track_ids = [str(index) for index in range(1, 4001)]
    user_ids = [f"user_{index:03d}" for index in range(200)]
    for directory, task in (
        ("baseline_comparison", "warm"),
        ("cold_start_comparison", "additive_withheld_item"),
    ):
        _write_json(
            root / directory / "release_binding.json",
            {
                "schema_version": 1,
                "task": task,
                "selection_manifest_sha256": selection_sha,
                "track_count": 4000,
                "user_count": 200,
                "track_ids_sha256": hashlib.sha256(
                    canonical_json_bytes(track_ids)
                ).hexdigest(),
                "user_ids_sha256": hashlib.sha256(
                    canonical_json_bytes(user_ids)
                ).hexdigest(),
                "method_ids": [
                    "path_signature_cosine",
                    "traditional_audio_cosine",
                    "lightfm_warp",
                    "lightfm_warp_kos",
                    "lightfm_latent_blend",
                    "implicit_als",
                ],
                "model_seeds": [2025, 2026, 2027, 2028, 2029],
            },
        )
        for name in (
            "accepted_tracks.json",
            "aggregate_metrics.json",
            "checksum_inventory.json",
            "configuration_selection.json",
            "dataset_manifest.json",
            "diagnostics.json",
            "execution.json",
            "interaction_splits.json",
            "model_diagnostics.json",
            "precision5_inference.json",
            "run_manifest.json",
            "uncertainty.json",
            "users.json",
            "validated.marker",
        ):
            _write_json(root / directory / name, {})
        for name in ("method_failures.jsonl", "track_failures.jsonl"):
            (root / directory / name).write_bytes(b"")
        methods = root / directory / "methods"
        methods.mkdir()
        for key in expected_method_seed_keys():
            (methods / f"{key}.jsonl").write_bytes(b"")

    (root / "track_processing/selected_tracks.json").write_text(
        "{}\n", encoding="ascii"
    )
    (root / "track_processing/feature_bundle").mkdir()
    (root / "track_processing/feature_bundle/bundle_manifest.json").write_text(
        "{}\n", encoding="ascii"
    )
    (root / "track_processing/feature_bundle/track_ids.json").write_text(
        "{}\n", encoding="ascii"
    )
    (root / "synthetic_users/population.json").write_text(
        "{}\n", encoding="ascii"
    )
    (root / "synthetic_users/population.json.sha256").write_text(
        "0" * 64 + "  population.json\n", encoding="ascii"
    )
    (root / "configuration_selection/selection_manifest.json.sha256").write_text(
        f"{selection_sha}  selection_manifest.json\n", encoding="ascii"
    )
    (root / "configuration_selection/candidate_results.csv").write_text(
        "candidate_id\n", encoding="ascii"
    )
    (root / "configuration_selection/ablation_overview.csv").write_text(
        "config_id\n", encoding="ascii"
    )
    (root / "evaluation/genre_diagnostic.json").write_text(
        "{}\n", encoding="ascii"
    )
    (root / "evaluation/selected_signature_similarity.f64le").write_bytes(b"")
    (root / "evaluation/selected_signature_similarity.json").write_text(
        "{}\n", encoding="ascii"
    )
    (root / "robustness/robustness_summary.json").write_text(
        "{}\n", encoding="ascii"
    )
    for package in ("dissertation_package", "dissertation_package_cold_start"):
        (root / package / "PACKAGE_VALIDATION.json").write_text(
            "{}\n", encoding="ascii"
        )
        (root / package / "method_metrics.csv").write_text(
            "method_id\n", encoding="ascii"
        )
        (root / package / "precision5_comparisons.csv").write_text(
            "comparison\n", encoding="ascii"
        )
        (root / package / "precision5_intervals.svg").write_text(
            "<svg/>\n", encoding="ascii"
        )
    for name in CITED_FIGURE_NAMES:
        (root / "figures" / name).write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode())
    (root / "timing/timing.json").write_text("{}\n", encoding="ascii")
    (root / "eda/eda_manifest.json").write_text("{}\n", encoding="ascii")
    for name in (
        "scientific_source_manifest.json",
        "raw_input_manifest.json",
        "environment_manifest.json",
    ):
        (root / "release" / name).write_text("{}\n", encoding="ascii")
    (root / "release/EVIDENCE_INDEX.md").write_text(
        render_markdown(build_evidence_index(root)), encoding="utf-8"
    )
    return {
        "git_commit": "a" * 40,
        "scientific_source_sha256": "b" * 64,
        "selection_manifest_sha256": selection_sha,
    }


def test_seal_and_validate_release_are_deterministic_and_complete(
    tmp_path, structural_only
):
    expected = _make_release(tmp_path)
    manifest_path = seal_release(tmp_path, **expected)
    first = manifest_path.read_bytes()
    validated = validate_release(tmp_path, **expected)
    assert validated["status"] == "validated"
    assert validated["track_count"] == 4000
    assert validated["user_count"] == 200
    assert set(validated["directories"]) == set(FINAL_RELEASE_DIRECTORIES)
    assert set(validated["figures"]) == set(CITED_FIGURE_NAMES)
    manifest_path.unlink()
    (tmp_path / "release/release_manifest.json.sha256").unlink()
    assert seal_release(tmp_path, **expected).read_bytes() == first


@pytest.mark.parametrize("missing", ["evaluation", "robustness", "timing"])
def test_release_rejects_missing_or_unclassified_directories(
    tmp_path, missing, structural_only
):
    expected = _make_release(tmp_path)
    shutil.rmtree(tmp_path / missing)
    with pytest.raises(ReleaseValidationError, match="director"):
        seal_release(tmp_path, **expected)

    second = tmp_path / "second"
    expected = _make_release(second)
    (second / "stale_results").mkdir()
    with pytest.raises(ReleaseValidationError, match="unclassified|director"):
        seal_release(second, **expected)


def test_release_rejects_figure_substitution_and_symlink(tmp_path, structural_only):
    expected = _make_release(tmp_path)
    (tmp_path / "figures/confusion_matrix.png").unlink()
    (tmp_path / "figures/retired_plot.png").write_bytes(b"old")
    with pytest.raises(ReleaseValidationError, match="figure"):
        seal_release(tmp_path, **expected)

    (tmp_path / "figures/retired_plot.png").unlink()
    target = tmp_path / "figures/method_comparison.png"
    (tmp_path / "figures/confusion_matrix.png").symlink_to(target)
    with pytest.raises(ReleaseValidationError, match="regular|symlink"):
        seal_release(tmp_path, **expected)


@pytest.mark.parametrize(
    "relative_path",
    [
        "stale-top-level.json",
        "synthetic_users/stale-population.json",
        "configuration_selection/stale-selection.csv",
        "timing/stale-timing.json",
    ],
)
def test_release_rejects_unclassified_regular_files(
    tmp_path, structural_only, relative_path
):
    expected = _make_release(tmp_path)
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("stale\n", encoding="ascii")

    with pytest.raises(ReleaseValidationError, match="file roster|top-level"):
        seal_release(tmp_path, **expected)


def test_release_rejects_extra_canonical_run_member(tmp_path, structural_only):
    expected = _make_release(tmp_path)
    (tmp_path / "baseline_comparison/stale.json").write_text(
        "{}\n", encoding="ascii"
    )
    with pytest.raises(ReleaseValidationError, match="roster|unclassified"):
        seal_release(tmp_path, **expected)


def test_release_rejects_nonregular_canonical_run_member(
    tmp_path, structural_only
):
    expected = _make_release(tmp_path)
    member = tmp_path / "baseline_comparison/diagnostics.json"
    member.unlink()
    member.mkdir()
    with pytest.raises(ReleaseValidationError, match="regular file"):
        seal_release(tmp_path, **expected)


def test_release_rejects_extra_run_binding_field(tmp_path, structural_only):
    expected = _make_release(tmp_path)
    binding = tmp_path / "baseline_comparison/release_binding.json"
    record = json.loads(binding.read_text(encoding="utf-8"))
    record["unexpected"] = "re-signed"
    _write_json(binding, record)
    with pytest.raises(ReleaseValidationError, match="binding.*schema|schema.*binding"):
        seal_release(tmp_path, **expected)


def test_release_rejects_extra_release_manifest_field(tmp_path, structural_only):
    expected = _make_release(tmp_path)
    manifest_path = seal_release(tmp_path, **expected)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["unexpected"] = "re-signed"
    _write_json(manifest_path, manifest)
    sidecar = tmp_path / "release/release_manifest.json.sha256"
    sidecar.write_text(
        f"{_sha(manifest_path)}  release_manifest.json\n", encoding="ascii"
    )
    with pytest.raises(
        ReleaseValidationError, match="manifest.*schema|schema.*manifest"
    ):
        validate_release(tmp_path, **expected)


@pytest.mark.parametrize("mode", ["missing", "tampered"])
def test_release_requires_exact_evidence_index(tmp_path, structural_only, mode):
    expected = _make_release(tmp_path)
    index = tmp_path / "release/EVIDENCE_INDEX.md"
    if mode == "missing":
        index.unlink()
    else:
        index.write_text("# stale evidence index\n", encoding="utf-8")
    with pytest.raises(ReleaseValidationError, match="evidence index|required"):
        seal_release(tmp_path, **expected)


@pytest.mark.parametrize(
    "relative_path",
    [
        "synthetic_users/population.json",
        "track_processing/feature_bundle/bundle_manifest.json",
        "track_processing/feature_bundle/track_ids.json",
    ],
)
def test_release_rejects_noncanonical_bound_json(
    tmp_path, structural_only, relative_path
):
    expected = _make_release(tmp_path)
    (tmp_path / relative_path).write_text(
        "{\n  \"pretty\": true\n}\n", encoding="utf-8"
    )
    with pytest.raises(ReleaseValidationError, match="canonically encoded"):
        seal_release(tmp_path, **expected)


def test_release_rejects_mixed_selection_or_cardinality(tmp_path, structural_only):
    expected = _make_release(tmp_path)
    binding = tmp_path / "cold_start_comparison/release_binding.json"
    record = json.loads(binding.read_text(encoding="utf-8"))
    record["selection_manifest_sha256"] = "0" * 64
    _write_json(binding, record)
    with pytest.raises(ReleaseValidationError, match="selection"):
        seal_release(tmp_path, **expected)

    second = tmp_path / "second"
    expected = _make_release(second)
    binding = second / "baseline_comparison/release_binding.json"
    record = json.loads(binding.read_text(encoding="utf-8"))
    record["track_count"] = 3999
    _write_json(binding, record)
    with pytest.raises(ReleaseValidationError, match="4,000|track"):
        seal_release(second, **expected)


def test_release_inventory_detects_post_seal_tampering(tmp_path, structural_only):
    expected = _make_release(tmp_path)
    seal_release(tmp_path, **expected)
    target = tmp_path / "configuration_selection/candidate_results.csv"
    target.write_text("tampered\n", encoding="ascii")
    with pytest.raises(ReleaseValidationError, match="checksum|inventory"):
        validate_release(tmp_path, **expected)


def test_release_rejects_placeholder_scientific_contents(tmp_path):
    expected = _make_release(tmp_path)
    with pytest.raises(ReleaseValidationError, match="scientific|manifest|content"):
        seal_release(tmp_path, **expected)


def test_post_retrieval_cli_requires_external_identity_and_revalidates(
    tmp_path, monkeypatch, capsys
):
    from src.scripts import validate_release as validator_module

    cli_main = getattr(validator_module, "main", None)
    assert callable(cli_main)
    captured = {}

    def fake_validate(run_root, **identity):
        captured["run_root"] = Path(run_root)
        captured.update(identity)
        return {
            "status": "validated",
            "git_commit": identity["git_commit"],
            "scientific_source_sha256": identity["scientific_source_sha256"],
            "selection_manifest_sha256": identity["selection_manifest_sha256"],
        }

    monkeypatch.setattr(validator_module, "validate_release", fake_validate)
    identity = {
        "git_commit": "a" * 40,
        "scientific_source_sha256": "b" * 64,
        "selection_manifest_sha256": "c" * 64,
    }
    assert cli_main(
        [
            "--run-root", str(tmp_path),
            "--git-commit", identity["git_commit"],
            "--scientific-source-sha256", identity["scientific_source_sha256"],
            "--selection-manifest-sha256", identity["selection_manifest_sha256"],
        ]
    ) == 0
    assert captured == {"run_root": tmp_path.resolve(), **identity}
    assert json.loads(capsys.readouterr().out) == {
        "git_commit": identity["git_commit"],
        "run_root": str(tmp_path.resolve()),
        "scientific_source_sha256": identity["scientific_source_sha256"],
        "selection_manifest_sha256": identity["selection_manifest_sha256"],
        "status": "validated",
    }

    with pytest.raises(SystemExit) as error:
        cli_main(["--run-root", str(tmp_path)])
    assert error.value.code == 2


def test_release_population_must_regenerate_exactly_from_metadata_and_seed():
    from src.data.synthetic_users import build_synthetic_population
    from src.scripts import validate_release as validator_module

    verify = getattr(validator_module, "_validate_population_reproducibility", None)
    assert callable(verify)
    genres = ("Rock", "Pop", "Electronic", "Jazz", "Classical", "Hip-Hop")
    tracks = [
        {
            "id": f"track_{index:04d}",
            "title": f"Track {index}",
            "artist": f"Artist {index % 37}",
            "genre": genres[index % len(genres)],
            "duration": 180 + (index % 120),
        }
        for index in range(1000)
    ]
    population = build_synthetic_population(tracks)
    verify(population)

    user_id = sorted(population["interactions"])[0]
    track_id = sorted(population["interactions"][user_id])[0]
    rating = population["interactions"][user_id][track_id]["rating"]
    population["interactions"][user_id][track_id]["rating"] = 1 + (rating % 5)
    with pytest.raises(ReleaseValidationError, match="regenerate"):
        verify(population)


def test_timing_stage_has_exact_bound_rosters_and_recomputes_totals(tmp_path):
    from src.scripts.run_baseline_comparison_multiple_runs import (
        expected_method_seed_keys,
    )
    from src.scripts.run_release_pipeline import RELEASE_STAGE_NAMES

    root = tmp_path / "release-root"
    (root / "timing").mkdir(parents=True)
    seconds = [float(index) / 1000.0 for index in range(200)]
    row = {
        "call_count": 200,
        "total_seconds": sum(seconds),
        "mean_seconds": sum(seconds) / len(seconds),
        "seconds": seconds,
    }
    scoring = {
        task: {key: dict(row) for key in expected_method_seed_keys()}
        for task in ("warm", "additive_withheld_item")
    }
    record = {
        "schema_version": 1,
        "classification": "hardware_specific_non_deterministic_diagnostic",
        "git_commit": "a" * 40,
        "scientific_source_sha256": "b" * 64,
        "selection_manifest_sha256": "c" * 64,
        "expected_users_per_output": 200,
        "completed_pipeline_stages": list(RELEASE_STAGE_NAMES[:-1]),
        "excluded_self_timed_stage": RELEASE_STAGE_NAMES[-1],
        "pipeline_stage_seconds": {
            name: float(index)
            for index, name in enumerate(RELEASE_STAGE_NAMES[:-1])
        },
        "scoring": scoring,
    }
    _write_json(root / "timing/timing.json", record)
    _validate_timing_stage(
        root,
        git_commit="a" * 40,
        scientific_source_sha256="b" * 64,
        selection_manifest_sha256="c" * 64,
    )

    record["scoring"]["warm"][expected_method_seed_keys()[0]][
        "total_seconds"
    ] += 1.0
    _write_json(root / "timing/timing.json", record)
    with pytest.raises(ReleaseValidationError, match="timing row"):
        _validate_timing_stage(
            root,
            git_commit="a" * 40,
            scientific_source_sha256="b" * 64,
            selection_manifest_sha256="c" * 64,
        )
