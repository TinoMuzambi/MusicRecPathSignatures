"""MR-06 integration and legacy-exclusion tests."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from src.scripts.run_baseline_comparison import CanonicalRunError, run_canonical_comparison
from src.scripts.run_baseline_comparison_multiple_runs import (
    MODEL_SEEDS,
    aggregate_stochastic_rows,
    expected_method_seed_keys,
)

from test_baseline_comparison import (
    RecordingScorerBuilder,
    identity_inputs,
    outer_repository,
    read_jsonl,
    tiny_task,
)


class TestAccessRecord(Mapping):
    __test__ = False

    def __init__(self, value, events):
        self.value = value
        self.events = events

    def __getitem__(self, key):
        if key == "test":
            self.events.append("test-read")
        return self.value[key]

    def __iter__(self):
        return iter(self.value)

    def __len__(self):
        return len(self.value)


def test_manifest_and_model_fit_context_precede_any_test_read(tmp_path):
    events = []
    task = tiny_task()
    task["users"] = {
        user_id: TestAccessRecord(record, events)
        for user_id, record in task["users"].items()
    }

    class OrderedBuilder(RecordingScorerBuilder):
        def __call__(self, context):
            assert events == []
            assert Path(context["run_manifest_path"]).is_file()
            events.append("builder")
            return super().__call__(context)

    run_canonical_comparison(
        repository_root=outer_repository(tmp_path),
        output_directory=tmp_path / "run",
        master_seed=2025,
        identity_inputs=identity_inputs(),
        task=task,
        scorer_builder=OrderedBuilder(),
        source_reader=lambda path: {"git_commit": "a" * 40, "dirty": False},
    )
    assert events[0] == "builder"
    assert events.count("test-read") == 2


def test_aggregates_are_rebuilt_from_jsonl_not_unsaved_rows(tmp_path):
    def mutate_serialised_rows(staging: Path):
        path = staging / "methods/path_signature_cosine.jsonl"
        rows = read_jsonl(path)
        for row in rows:
            row["metrics"]["precision"]["5"] = 0.123
        path.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )

    run_dir = run_canonical_comparison(
        repository_root=outer_repository(tmp_path),
        output_directory=tmp_path / "run",
        master_seed=2025,
        identity_inputs=identity_inputs(),
        task=tiny_task(),
        scorer_builder=RecordingScorerBuilder(),
        source_reader=lambda path: {"git_commit": "a" * 40, "dirty": False},
        rows_written_hook=mutate_serialised_rows,
    )
    aggregate = json.loads((run_dir / "aggregate_metrics.json").read_text(encoding="utf-8"))
    assert aggregate["methods"]["path_signature_cosine"]["precision"]["5"] == pytest.approx(0.123)


def test_mixed_seed_optional_availability_is_structurally_unavailable(tmp_path):
    run_dir = run_canonical_comparison(
        repository_root=outer_repository(tmp_path),
        output_directory=tmp_path / "run",
        master_seed=2025,
        identity_inputs=identity_inputs(),
        task=tiny_task(),
        scorer_builder=RecordingScorerBuilder(),
        source_reader=lambda path: {"git_commit": "a" * 40, "dirty": False},
    )
    rows_by_seed = {}
    for seed in MODEL_SEEDS:
        rows = read_jsonl(run_dir / f"methods/lightfm_warp__seed_{seed}.jsonl")
        rows_by_seed[seed] = {row["user_id"]: row for row in rows}
    rows_by_seed[2025]["u1"]["metrics"]["novelty"]["5"] = {
        "status": "available", "value": 0.75
    }
    aggregate = aggregate_stochastic_rows(
        rows_by_seed, catalogue_ids=tiny_task()["catalogue_ids"]
    )
    record = aggregate["per_user"]["u1"]["novelty"][5]
    assert record["status"] == "unavailable"
    assert record["reason_code"] == "seed_metric_unavailable"
    assert "value" not in record


def _imports_and_calls(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = set()
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    return imports, calls


def test_canonical_runner_and_packager_have_no_legacy_call_graph():
    scripts = Path(__file__).parents[1] / "src" / "scripts"
    runner = scripts / "run_baseline_comparison.py"
    repeated = scripts / "run_baseline_comparison_multiple_runs.py"
    packager = scripts / "create_dissertation_package.py"

    runner_imports, runner_calls = _imports_and_calls(runner)
    package_imports, package_calls = _imports_and_calls(packager)
    repeated_tree = ast.parse(repeated.read_text(encoding="utf-8"))
    repeated_defs = {
        node.name for node in repeated_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    forbidden_import_fragments = {
        "recommendation_cv", "generate_predictions", "ablation",
        "synthetic_users", "softmax_regression",
    }
    assert not any(
        fragment in imported
        for imported in runner_imports | package_imports
        for fragment in forbidden_import_fragments
    )
    assert not ({
        "create_synthetic_ratings", "create_legacy_feature_fallback_ratings",
        "split_train_test", "export_users", "recommendation_cv",
    } & (runner_calls | package_calls))
    assert "subprocess" not in runner_imports | package_imports
    assert "subprocess" not in {
        alias.name
        for node in ast.walk(repeated_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "aggregate_results" not in repeated_defs
    assert "main" not in repeated_defs


def test_exact_output_keys_are_declared_once_in_execution(tmp_path):
    run_dir = run_canonical_comparison(
        repository_root=outer_repository(tmp_path),
        output_directory=tmp_path / "run",
        master_seed=2025,
        identity_inputs=identity_inputs(),
        task=tiny_task(),
        scorer_builder=RecordingScorerBuilder(),
        source_reader=lambda path: {"git_commit": "a" * 40, "dirty": False},
    )
    execution = json.loads((run_dir / "execution.json").read_text(encoding="utf-8"))
    assert tuple(execution["methods"]) == expected_method_seed_keys()
    assert len(execution["methods"]) == len(set(execution["methods"])) == 22


def test_incomplete_method_seed_scorer_family_fails_before_scoring(tmp_path):
    class MissingSeedBuilder(RecordingScorerBuilder):
        def __call__(self, context):
            scorers = dict(super().__call__(context))
            scorers.pop("implicit_als__seed_2029")
            return scorers

    with pytest.raises(CanonicalRunError, match="method/seed output keys mismatch"):
        run_canonical_comparison(
            repository_root=outer_repository(tmp_path),
            output_directory=tmp_path / "run",
            master_seed=2025,
            identity_inputs=identity_inputs(),
            task=tiny_task(),
            scorer_builder=MissingSeedBuilder(),
            source_reader=lambda path: {"git_commit": "a" * 40, "dirty": False},
        )
