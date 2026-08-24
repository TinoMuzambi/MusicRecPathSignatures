"""Boundary tests retiring invalid synthetic-user entry points in MR-02."""

import inspect

import src.analysis.collaborative_filtering as collaborative
import src.data.synthetic_users as synthetic_users
import src.scripts.create_dissertation_package as package
import src.scripts.generate_synthetic_users as generation_script
import src.scripts.run_baseline_comparison as runner


def test_between_user_splitters_are_absent_from_canonical_api():
    assert not hasattr(synthetic_users.SyntheticUserGenerator, "split_train_test")
    assert not hasattr(generation_script, "split_train_test")


def test_feature_fallback_ratings_helper_is_explicitly_legacy_only():
    assert not hasattr(collaborative, "create_synthetic_ratings")
    assert callable(collaborative.create_legacy_feature_fallback_ratings)
    assert "create_synthetic_ratings" not in runner._LEGACY_EXPORTS
    assert "create_legacy_feature_fallback_ratings" in runner._LEGACY_EXPORTS


def test_canonical_seam_and_packager_do_not_reference_retired_generators():
    forbidden = ("split_train_test", "create_synthetic_ratings")
    canonical_seam = inspect.getsource(runner.execute_canonical_pre_scoring)
    packager_source = inspect.getsource(package)
    generation_source = inspect.getsource(generation_script)

    for name in forbidden:
        assert name not in canonical_seam
        assert name not in packager_source
        assert name not in generation_source


def test_generation_script_has_no_stale_custom_or_dead_cli_surface():
    source = inspect.getsource(generation_script)
    retired = (
        "--test-ratio",
        "--validation-ratio",
        "--enthusiast-ratio",
        "--specialist-ratio",
        "--casual-ratio",
        "--explorer-ratio",
        "--mainstream-ratio",
        "--dpi",
        "--no-visualisations",
        "--no-report",
        "Will be randomly assigned",
    )

    for token in retired:
        assert token not in source

    for name in (
        "create_archetype_distribution",
        "generate_synthetic_users",
        "create_visualisations",
        "generate_report",
    ):
        assert not hasattr(generation_script, name)


def test_main_refuses_legacy_output_before_configuring_a_log_file():
    assert callable(generation_script._assert_export_directory_available)
    source = inspect.getsource(generation_script.main)
    assert source.index("_assert_export_directory_available") < source.index(
        "configure_logging"
    )
