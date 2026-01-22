# Imports not needed - tests use fixtures

import numpy as np

from src.evaluation.robustness import (
    bootstrap_ci,
    sensitivity_analysis,
    stability_metrics,
    error_analysis_by_group,
    save_json,
    save_csv_table,
)


def test_bootstrap_ci_basic():
    data = [0.6, 0.62, 0.58, 0.61, 0.59]
    res = bootstrap_ci(data, n_bootstrap=200, ci=0.90, seed=1)
    assert 0.55 < res["estimate"] < 0.65
    assert res["ci_low"] < res["estimate"] < res["ci_high"]


def test_sensitivity_analysis_grid():
    def metric_fn(size, seed):
        rng = np.random.default_rng(seed)
        return float(rng.normal(0.6 + 0.001 * size, 0.01))

    res = sensitivity_analysis(metric_fn, sizes=[10, 20], seeds=[1, 2, 3])
    assert res["sizes"] == [10, 20]
    assert list(res["values"].keys()) == ["10", "20"]


def test_stability_metrics():
    scores = [[0.6, 0.61, 0.62], [0.58, 0.6]]
    res = stability_metrics(scores_per_run=scores)
    assert "per_run" in res and "overall" in res
    assert len(res["per_run"]) == 2


def test_error_analysis_by_group_and_exports(tmp_path):
    per_user = {"u1": 0.6, "u2": 0.62, "u3": 0.58}
    user_group = {"u1": "Rock", "u2": "Pop", "u3": "Rock"}
    res = error_analysis_by_group(per_user, user_group)
    assert "groups" in res and "Rock" in res["groups"]

    # Export files
    json_path = tmp_path / "error_analysis.json"
    csv_path = tmp_path / "error_analysis.csv"
    save_json(res, json_path)
    rows = [{"group": g, **vals} for g, vals in res["groups"].items()]
    save_csv_table(rows, csv_path)
    assert Path(json_path).exists()
    assert Path(csv_path).exists()
