"""Regression test for the F-13 mislabelling (R9 evidence audit).

``find_cross_genre_similar_pairs`` returns at most ``top_k`` (default 10)
pairs, each already filtered to ``similarity >= min_similarity`` (default
0.7). The similarity-distribution panel's title nonetheless claimed it
plotted "(all cross-genre pairs)" -- a materially misleading label that
directly explains why the dissertation's "mean 0.995" cross-genre claim (a
mean of exactly this pre-filtered, threshold-selected top-10 subset) does not
match the real corpus-wide mean pairwise similarity (0.0229, see
``analyze_cross_genre_behaviour.py``).
"""

from __future__ import annotations

import inspect
from pathlib import Path

from src.scripts import find_cross_genre_similar_songs as cross_genre_module


def test_similarity_distribution_panel_does_not_claim_to_show_all_pairs():
    source = inspect.getsource(cross_genre_module)
    assert "(all cross-genre pairs)" not in source


def test_relative_paths_resolve_against_cwd_not_the_scripts_own_location(tmp_path):
    """Pre-run integration check: every other script in this pipeline
    resolves its relative --similarity-matrix/--tracks-json/--output paths
    against the current working directory (matching how
    run_complete_pipeline.sh invokes everything, always from the repository
    root as CWD). This script alone instead resolved them against
    Path(__file__).parent.parent.parent -- the script's own on-disk
    location -- which happens to coincide with CWD in the exact real-VM
    deployment layout (both are "code/"), but silently breaks the moment
    the script is invoked from any other working directory (as this test
    does, and as any local rerun/review/CI invocation could).
    """

    import subprocess
    import sys as _sys
    import json as _json
    import numpy as _np

    # A working directory genuinely different from the script's own
    # location under the real source tree.
    work_dir = tmp_path / "not_the_repo_root"
    (work_dir / "data" / "processed_tracks").mkdir(parents=True)
    (work_dir / "results" / "visualisations").mkdir(parents=True)

    tracks = [
        {"track_id": 1, "title": "A", "artist": "x", "genre": "Rock"},
        {"track_id": 2, "title": "B", "artist": "x", "genre": "Jazz"},
    ]
    (work_dir / "data" / "processed_tracks" / "selected_tracks.json").write_text(
        _json.dumps(tracks), encoding="utf-8"
    )
    sim = _np.array([[1.0, 0.9], [0.9, 1.0]])
    _np.savez(
        work_dir / "data" / "similarity_matrix.npz",
        similarity_matrix=sim,
        song_names=_np.array(["A", "B"]),
    )

    script_path = Path(cross_genre_module.__file__)
    result = subprocess.run(
        [
            _sys.executable, str(script_path),
            "--similarity-matrix", "data/similarity_matrix.npz",
            "--tracks-json", "data/processed_tracks/selected_tracks.json",
            "--output", "results/visualisations/cross_genre_similarity.png",
            "--min-similarity", "0.5",
        ],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "not found" not in result.stderr.lower(), result.stderr
    assert (work_dir / "results" / "visualisations" / "cross_genre_similarity.png").exists()
