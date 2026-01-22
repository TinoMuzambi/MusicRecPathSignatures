from pathlib import Path

from src.scripts.run_evaluation_suite import export_evaluation_tables


def test_export_evaluation_tables(tmp_path):
    rec = {
        "precision": {1: 0.6, 5: 0.5},
        "recall": {1: 0.1, 5: 0.2},
        "ndcg": {1: 0.6, 5: 0.55},
        "diversity": {1: 0.9, 5: 0.8},
        "novelty": {1: 0.7, 5: 0.65},
        "coverage": {1: 0.3, 5: 0.4},
        "map": 0.52,
    }
    cls = {
        "accuracy": 0.8,
        "precision_macro": 0.75,
        "recall_macro": 0.74,
        "f1_macro": 0.745,
    }
    cv = [{"metrics": {"precision@10": 0.5, "recall@10": 0.25}}]

    export_evaluation_tables(str(tmp_path), rec, cls, cv)

    expected = [
        "recommendation_metrics_table.csv",
        "classification_metrics_table.csv",
        "cv_results_table.csv",
    ]
    for fname in expected:
        assert Path(tmp_path, fname).exists(), f"Missing {fname}"
