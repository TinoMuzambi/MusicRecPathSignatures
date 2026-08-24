"""Regression test for the F-18 finding (R9 evidence audit).

``create_visualisations`` saved its combined figure under
``ablation_study_visualisations.png``, then copied it a second time to
``ablation_overview.png`` "for LaTeX" -- leaving two byte-identical files in
the results tree with no purpose served by the extra copy, since only
``ablation_overview.png`` is ever cited. Only the LaTeX-cited name should be
written.
"""

from __future__ import annotations

import logging

from src.scripts import run_ablation_studies as ablation_module


def test_create_visualisations_writes_only_the_cited_filename(tmp_path):
    runner = ablation_module.AblationStudyRunner(
        output_dir=str(tmp_path), logger=logging.getLogger("test-ablation")
    )
    results = {
        "signature_orders": {1: {"metrics": {"precision@5": 0.2}}},
        "temperatures": {1.0: {"metrics": {"precision@5": 0.2}}},
        "feature_combinations": {"core": {"metrics": {"precision@5": 0.2}}},
        "similarity_metrics": {"cosine": {"metrics": {"precision@5": 0.2}}},
        "scoring_variants": {"direct_cosine": {"metrics": {"precision@5": 0.2}}},
    }
    runner.create_visualisations(results)
    assert (tmp_path / "ablation_overview.png").exists()
    assert not (tmp_path / "ablation_study_visualisations.png").exists()
