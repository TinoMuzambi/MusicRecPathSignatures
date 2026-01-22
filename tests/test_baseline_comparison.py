import os
import json
import shutil
import tempfile
from pathlib import Path

from src.scripts.run_baseline_comparison import run_baseline_comparison


def test_run_baseline_comparison_generates_outputs(monkeypatch):  # pylint: disable=unused-argument
    # Create temporary directory for output
    tmpdir = tempfile.mkdtemp()
    outdir = os.path.join(tmpdir, "baseline")

    # Use small synthetic feature file
    features = {
        "track_1": {"mfccs": [0.1]},
        "track_2": {"mfccs": [0.2]},
        "track_3": {"mfccs": [0.3]},
    }
    features_path = os.path.join(tmpdir, "features.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(features, f)

    # Minimal tracks file path passed through to PathSignature component
    tracks_path = os.path.join(tmpdir, "tracks.json")
    with open(tracks_path, "w", encoding="utf-8") as f:
        json.dump([], f)

    # Run comparison with very few users for speed
    run_baseline_comparison(
        features_file=features_path,
        tracks_json=tracks_path,
        output_dir=outdir,
        n_users=6,
        test_ratio=0.5,
    )

    # Verify outputs
    expected = [
        "baseline_comparison_results.json",
        "baseline_comparison_plots.png",
        "performance_comparison.png",
        "significance_matrix.csv",
        "effect_sizes.csv",
        "baseline_comparison_table.csv",
        "BASELINE_COMPARISON_REPORT.md",
    ]
    for fname in expected:
        assert Path(outdir, fname).exists(), f"Missing output {fname}"

    shutil.rmtree(tmpdir)
