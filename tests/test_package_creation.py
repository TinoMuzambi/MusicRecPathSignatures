from src.scripts.create_dissertation_package import main as create_pkg_main


def test_create_dissertation_package(tmp_path):
    # Create a fake results tree with some files to copy
    results_root = tmp_path / "results"
    (results_root / "eda").mkdir(parents=True, exist_ok=True)
    (results_root / "baseline_comparison").mkdir(parents=True, exist_ok=True)
    (results_root / "ablation_studies").mkdir(parents=True, exist_ok=True)
    (results_root / "robustness").mkdir(parents=True, exist_ok=True)
    (results_root / "evaluation").mkdir(parents=True, exist_ok=True)
    (results_root / "dissertation_figures").mkdir(parents=True, exist_ok=True)

    # Touch some files
    (results_root / "eda" / "dataset_statistics_table.csv").write_text("a,b\n1,2\n")
    (results_root / "eda" / "genre_statistics.csv").write_text("g,c\nX,1\n")
    (results_root / "baseline_comparison" / "baseline_comparison_table.csv").write_text(
        "m,@5\nA,0.5\n"
    )
    (results_root / "baseline_comparison" / "significance_matrix.csv").write_text(
        "A,B\n0.1,0.2\n"
    )
    (results_root / "ablation_studies" / "ablation_summary.csv").write_text(
        "c,p\nX,0.5\n"
    )
    (results_root / "robustness" / "stability_table.csv").write_text("run,std\n1,0.1\n")
    (results_root / "evaluation" / "cv_results_table.csv").write_text(
        "fold,p,r\n1,0.5,0.2\n"
    )
    (results_root / "evaluation" / "evaluation_summary.json").write_text("{}")
    (results_root / "baseline_comparison" / "statistical_comparison.json").write_text(
        "{}"
    )
    (
        results_root / "dissertation_figures" / "fig_05_method_comparison.png"
    ).write_bytes(b"PNG")

    out_dir = tmp_path / "pkg"

    # Monkeypatch argv to call the script main()
    import sys

    old_argv = sys.argv
    sys.argv = [
        "create_dissertation_package",
        "--results-root",
        str(results_root),
        "--output-dir",
        str(out_dir),
    ]
    try:
        create_pkg_main()
    finally:
        sys.argv = old_argv

    # Verify expected structure
    assert (out_dir / "tables" / "table_01_dataset_statistics.csv").exists()
    assert (out_dir / "tables" / "TABLE_INDEX.md").exists()
    assert (out_dir / "figures" / "FIGURE_INDEX.md").exists()
    assert (out_dir / "data" / "all_metrics.json").exists()
    assert (out_dir / "data" / "DATA_DICTIONARY.md").exists()
    # Summary
    assert (results_root / "RESULTS_SUMMARY.md").exists()
