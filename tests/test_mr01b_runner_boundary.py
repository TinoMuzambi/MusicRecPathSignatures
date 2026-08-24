"""Tests for the lightweight, pre-scoring MR-01B runner boundary."""

import importlib
import inspect
import json
import subprocess
import sys

import pytest


def _identity_inputs():
    return {
        "source": {"git_commit": "d29f1d9", "dirty": False},
        "runtime": {"python": "3.12.3", "architecture": "x86_64"},
        "dataset": {"accepted_tracks_checksum": "tracks-sha"},
        "feature_schema": {"signature_channels": ["time", "pitch", "loudness"]},
        "task": {"population_size": 200},
        "splits": {"manifest_checksum": "splits-sha"},
        "index_maps": {"users_checksum": "users-sha", "items_checksum": "items-sha"},
        "models": {"path_signature_cosine": {"order": 2}},
        "evaluation": {"primary_metric": "precision_at_5"},
        "diagnostic_declarations": [
            "signature_norm_recommendation_frequency_spearman"
        ],
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    }


def test_pre_scoring_boundary_requires_explicit_seed_and_builds_id_first():
    run_contract = importlib.import_module("src.scripts.run_contract")
    parameter = inspect.signature(run_contract.execute_pre_scoring).parameters[
        "master_seed"
    ]
    assert parameter.default is inspect.Parameter.empty

    events = []

    def scoring_stage(identity):
        events.append(("score", identity.run_id, identity.payload["task"]))
        return "scored"

    execution = run_contract.execute_pre_scoring(
        master_seed=2025,
        identity_inputs=_identity_inputs(),
        scoring_stage=scoring_stage,
    )

    assert execution.result == "scored"
    assert execution.identity.payload["task"]["master_seed"] == 2025
    assert events == [
        ("score", execution.identity.run_id, {"master_seed": 2025, "population_size": 200})
    ]


@pytest.mark.parametrize("declared_seed", [2026, None])
def test_pre_scoring_boundary_rejects_seed_mismatch(declared_seed):
    run_contract = importlib.import_module("src.scripts.run_contract")
    inputs = _identity_inputs()
    inputs["task"]["master_seed"] = declared_seed

    with pytest.raises(ValueError, match="master seed"):
        run_contract.execute_pre_scoring(
            master_seed=2025,
            identity_inputs=inputs,
            scoring_stage=lambda _identity: None,
        )


def test_contract_imports_do_not_eagerly_load_legacy_scientific_stack():
    code = """
import json, sys
import src.evaluation.experiment_protocol
print(json.dumps(sorted(name for name in sys.modules if name.split('.')[0] in {
    'matplotlib', 'seaborn', 'sklearn', 'scipy'
})))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_legacy_runner_module_is_importable_without_esig_or_model_loading():
    runner = importlib.import_module("src.scripts.run_baseline_comparison")

    assert runner.CANONICAL_MASTER_SEED == 2025
    execution = runner.execute_canonical_pre_scoring(
        identity_inputs=_identity_inputs(),
        scoring_stage=lambda identity: identity.run_id,
    )
    assert execution.identity.payload["task"]["master_seed"] == 2025
    assert execution.result == execution.identity.run_id

    code = """
import json, sys
import src.scripts.run_baseline_comparison
print(json.dumps(sorted(name for name in ('esig', 'lightfm', 'implicit')
                        if name in sys.modules)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_legacy_entry_points_load_dependencies_before_using_lazy_globals(
    monkeypatch, tmp_path
):
    runner = importlib.import_module("src.scripts.run_baseline_comparison")

    class LoaderReached(RuntimeError):
        pass

    def stop_at_loader():
        raise LoaderReached("loader reached")

    monkeypatch.setattr(runner, "_load_legacy_dependencies", stop_at_loader)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_baseline_comparison", "--output-dir", str(tmp_path / "output")],
    )
    with pytest.raises(LoaderReached, match="loader reached"):
        runner.main()
    with pytest.raises(LoaderReached, match="loader reached"):
        runner.create_comparison_plots({}, tmp_path, [5])
    with pytest.raises(LoaderReached, match="loader reached"):
        runner.create_statistical_comparisons({}, tmp_path, k_eval=5)
