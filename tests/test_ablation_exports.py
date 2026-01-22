from pathlib import Path

from src.scripts.run_ablation_studies import AblationStudyRunner


def test_export_tables_writes_expected_files(tmp_path):
    runner = AblationStudyRunner(
        output_dir=str(tmp_path), logger=None
    )  # logger not used in export

    # Minimal fake results with required keys
    results = {
        "signature_orders": {
            1: {
                "metrics": {"precision@5": 0.6, "recall@5": 0.4, "diversity@5": 0.7},
                "signature_count": 10,
                "signature_dimensions": 64,
            }
        },
        "temperatures": {
            1.0: {
                "metrics": {"precision@5": 0.61, "recall@5": 0.41, "diversity@5": 0.71},
                "similarity_stats": {"mean": 0.5, "std": 0.1, "min": 0.0, "max": 1.0},
            }
        },
        "feature_combinations": {
            "mfccs+chroma": {
                "metrics": {"precision@5": 0.62, "recall@5": 0.42, "diversity@5": 0.72},
                "signature_count": 12,
            }
        },
        "component_contributions": {
            "path_signatures_only": {
                "metrics": {"precision@5": 0.59, "recall@5": 0.39, "diversity@5": 0.69}
            }
        },
    }

    runner.export_tables(results)

    expected = [
        "signature_order_table.csv",
        "temperature_scaling_table.csv",
        "feature_combinations_table.csv",
        "component_contributions_table.csv",
        "ablation_summary.csv",
    ]
    for fname in expected:
        assert Path(tmp_path, fname).exists(), f"Missing {fname}"
