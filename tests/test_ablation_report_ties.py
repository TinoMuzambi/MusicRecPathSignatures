"""Regression test for the F-12 finding (R9 evidence audit).

Temperature scaling and L2-normalised cosine-versus-Euclidean ranking are, by
construction, monotone transforms of the same underlying score ordering. The
Markdown report generator declared a single
"Best performing temperature"/"Best similarity metric" using a strict
``score > best_score`` comparison, so a genuine tie was silently broken by
iteration order and reported as if one value had won on merit. The report
must instead say the arms are tied.
"""

from __future__ import annotations

from pathlib import Path

from src.scripts import run_ablation_studies as ablation_module


def test_tied_temperatures_are_reported_as_a_tie_not_a_false_winner(tmp_path: Path):
    results = {
        "temperatures": {
            0.1: {"metrics": {"precision@5": 0.2286}},
            0.5: {"metrics": {"precision@5": 0.2286}},
            1.0: {"metrics": {"precision@5": 0.2286}},
        }
    }
    ablation_module._generate_summary_report(results, tmp_path, ablation_module.setup_logger("t"))
    report = (tmp_path / "ablation_study_report.md").read_text(encoding="utf-8")
    assert "Best performing temperature" not in report
    assert "tied" in report.lower()


def test_genuinely_best_temperature_still_reports_a_winner(tmp_path: Path):
    results = {
        "temperatures": {
            0.1: {"metrics": {"precision@5": 0.30}},
            0.5: {"metrics": {"precision@5": 0.22}},
        }
    }
    ablation_module._generate_summary_report(results, tmp_path, ablation_module.setup_logger("t"))
    report = (tmp_path / "ablation_study_report.md").read_text(encoding="utf-8")
    assert "Best performing temperature: 0.1" in report


def test_tied_similarity_metrics_are_reported_as_a_tie_not_a_false_winner(tmp_path: Path):
    results = {
        "similarity_metrics": {
            "cosine": {"metrics": {"precision@5": 0.2762}},
            "euclidean": {"metrics": {"precision@5": 0.2762}},
        }
    }
    ablation_module._generate_summary_report(results, tmp_path, ablation_module.setup_logger("t"))
    report = (tmp_path / "ablation_study_report.md").read_text(encoding="utf-8")
    assert "Best similarity metric" not in report
    assert "tied" in report.lower()


def test_report_discloses_exploratory_genre_retrieval_task_up_front(tmp_path: Path):
    """R9 evidence audit F-05: the ablation task is genre-retrieval on a
    small track subset (relevance = same-genre tracks), not the canonical
    200-user held-out-interaction task the headline table uses. This must be
    stated up front, not left implicit, so the two are never read as
    comparable.
    """

    ablation_module._generate_summary_report({}, tmp_path, ablation_module.setup_logger("t"))
    report = (tmp_path / "ablation_study_report.md").read_text(encoding="utf-8")
    assert "exploratory" in report.lower()
    assert "not" in report.lower() and "comparable" in report.lower()


def test_key_findings_does_not_synthesise_a_best_overall_configuration_across_studies():
    """R9 evidence audit F-12: separate ablation studies vary different axes
    while holding the others at unrelated defaults, so their Precision@5
    values are not directly comparable -- "Best overall configuration:
    temperatures: 0.1" versus "feature_combinations: pitch_loudness_mfccs_chroma"
    is not a coherent, actionable recommendation, just the arm with the
    highest number across incomparable studies.
    """

    results = {
        "temperatures": {0.1: {"metrics": {"precision@5": 0.30}}},
        "feature_combinations": {"core": {"metrics": {"precision@5": 0.20}}},
    }

    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as td:
        ablation_module._generate_summary_report(
            results, _Path(td), ablation_module.setup_logger("t")
        )
        report = _Path(td, "ablation_study_report.md").read_text(encoding="utf-8")
    assert "Best overall configuration" not in report


def test_report_is_deterministic_and_uses_scoring_variant_language(tmp_path: Path):
    results = {
        "scoring_variants": {
            "direct_cosine": {"metrics": {"precision@5": 0.2}},
        }
    }
    ablation_module._generate_summary_report(
        results, tmp_path, ablation_module.setup_logger("t")
    )
    one = (tmp_path / "ablation_study_report.md").read_bytes()
    ablation_module._generate_summary_report(
        results, tmp_path, ablation_module.setup_logger("t")
    )
    two = (tmp_path / "ablation_study_report.md").read_bytes()
    assert one == two
    text = one.decode("utf-8")
    assert "Scoring Variant Analysis" in text
    assert "Component Contribution" not in text
    assert "Generated on" not in text
