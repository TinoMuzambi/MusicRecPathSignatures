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
        "similarity_metrics": {
            "cosine": {
                "metrics": {"precision@5": 0.58, "recall@5": 0.38, "diversity@5": 0.68}
            }
        },
        "scoring_variants": {
            "direct_cosine": {
                "metrics": {"precision@5": 0.59, "recall@5": 0.39, "diversity@5": 0.69}
            }
        },
    }

    runner.export_tables(results)

    expected = [
        "signature_order_table.csv",
        "temperature_scaling_table.csv",
        "feature_combinations_table.csv",
        "similarity_metrics_table.csv",
        "scoring_variants_table.csv",
        "ablation_summary.csv",
    ]
    for fname in expected:
        assert Path(tmp_path, fname).exists(), f"Missing {fname}"


def test_signature_order_table_marks_top_level_error_as_unavailable_not_zero(tmp_path):
    """A top-level errored arm must never look like a measured zero."""

    runner = AblationStudyRunner(output_dir=str(tmp_path), logger=None)
    results = {
        "signature_orders": {
            1: {
                "metrics": {"precision@5": 0.6, "recall@5": 0.4, "diversity@5": 0.7},
                "signature_count": 10,
                "signature_dimensions": 64,
            },
            3: {"error": "Signature computation failed"},
        }
    }
    runner.export_tables(results)

    rows = (Path(tmp_path) / "signature_order_table.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    header = rows[0].split(",")
    order_4_row = next(row.split(",") for row in rows[1:] if row.split(",")[0] == "3")
    precision_col = header.index("precision@5")
    assert order_4_row[precision_col] in ("N/A", "")
    assert order_4_row[precision_col] != "0.0"


def test_temperature_table_marks_evaluation_error_as_unavailable_not_zero(tmp_path):
    """Same bug class as the signature-order fix, in a sibling table
    (independent-review follow-up: confirmed still present)."""

    runner = AblationStudyRunner(output_dir=str(tmp_path), logger=None)
    results = {
        "temperatures": {
            1.0: {"metrics": {"error": "No valid ground truth available"}},
        }
    }
    runner.export_tables(results)

    rows = (Path(tmp_path) / "temperature_scaling_table.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    header = rows[0].split(",")
    row = rows[1].split(",")
    precision_col = header.index("precision@5")
    assert row[precision_col] in ("N/A", "")
    assert row[precision_col] != "0.0"


def test_scoring_variants_table_marks_evaluation_error_as_unavailable_not_zero(
    tmp_path,
):
    """Same bug class as the signature-order fix, in a sibling table
    (independent-review follow-up: confirmed still present)."""

    runner = AblationStudyRunner(output_dir=str(tmp_path), logger=None)
    results = {
        "scoring_variants": {
            "composite_temperature_2": {"metrics": {"error": "No valid ground truth available"}},
        }
    }
    runner.export_tables(results)

    rows = (Path(tmp_path) / "scoring_variants_table.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    header = rows[0].split(",")
    row = rows[1].split(",")
    precision_col = header.index("precision@5")
    assert row[precision_col] in ("N/A", "")
    assert row[precision_col] != "0.0"


def test_feature_and_summary_tables_mark_error_metrics_unavailable(tmp_path):
    runner = AblationStudyRunner(output_dir=str(tmp_path), logger=None)
    results = {
        "feature_combinations": {
            "all_channels": {"metrics": {"error": "failed"}, "signature_count": 100}
        }
    }
    runner.export_tables(results)
    feature = (Path(tmp_path) / "feature_combinations_table.csv").read_text(encoding="utf-8")
    summary = (Path(tmp_path) / "ablation_summary.csv").read_text(encoding="utf-8")
    assert ",N/A," in feature
    assert ",N/A," in summary


def test_signature_order_table_marks_evaluation_error_as_unavailable_not_zero(tmp_path):
    """Independent-review follow-up: a *different* failure path than the
    "skipped" one above -- ``run_path_signature_order_analysis`` still
    writes a ``"metrics"`` key when its inner ``_evaluate_performance`` call
    raises, just shaped as ``{"error": "..."}`` rather than absent entirely.
    ``export_tables`` must treat this the same way: as unavailable, not as a
    real 0.0 precision.
    """

    runner = AblationStudyRunner(output_dir=str(tmp_path), logger=None)
    results = {
        "signature_orders": {
            2: {
                "metrics": {"error": "No valid ground truth available"},
                "signature_count": 100,
                "signature_dimensions": 1483,
            },
        }
    }
    runner.export_tables(results)

    rows = (Path(tmp_path) / "signature_order_table.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    header = rows[0].split(",")
    order_2_row = next(row.split(",") for row in rows[1:] if row.split(",")[0] == "2")
    precision_col = header.index("precision@5")
    assert order_2_row[precision_col] in ("N/A", "")
    assert order_2_row[precision_col] != "0.0"
