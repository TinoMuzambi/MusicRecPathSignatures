# V30 compact-feature, population and EDA implementation evidence

## Boundary

This unit was implemented only in the disposable code clone at base commit
`3e682eab21528894b75a62120f2decfe08f23ed5`. It did not access FMA data, use
the network, run an experiment or mutate GCP. The durable nested `code/`
worktree was not read from for execution and was not modified.

The implementation provides:

- a compact bundle containing concatenated `<f4` 38-channel paths, `<i8`
  element offsets and a dense `<f8` matrix of the exact 72 named traditional
  aggregates;
- a strict bundle manifest, exact inventory, streaming SHA-256 sidecars and a
  read-only ID-indexed loader;
- exact FMA Medium metadata validation, deterministic up-to-100 then Hamilton
  genre quotas, same-genre replacement before deterministic global-capacity
  replacement, and an exact 4,000-track fail-closed output;
- admission of every declared path arm through a mathematically complete set
  of non-dominated signatures. In the declared grid the order-3 all-channel
  signature contains every lower-order ordered channel projection. All 18
  configurations are nevertheless schema- and dimension-validated and
  recorded in provenance;
- one hash-bound canonical population JSON with exact configuration, metadata,
  users, interactions and train/validation/test partitions;
- overlap-free preferred/random interaction selection whose declared and
  realised counts agree;
- strict ID-aligned EDA over only the dense 72-vector rows, including a true
  72 by 72 correlation matrix and true per-genre feature summaries; and
- canonical population plots in which empty archetype/genre cells are masked
  grey and explicitly labelled `N/A (no interactions)` rather than being
  rendered as low ratings.

## Interfaces

- `src.utils.feature_bundle.write_feature_bundle(features, output_dir, ...)`
- `src.utils.feature_bundle.load_feature_bundle(input_dir, ...)`
- `src.scripts.robust_track_processing.load_selected_tracks(path, ...)`
- `src.scripts.generate_synthetic_users.export_population_file(population, path)`
- `src.scripts.generate_synthetic_users.load_population_file(path, ...)`
- `src.analysis.strict_eda.run_strict_eda(tracks, bundle_dir, output_dir, ...)`
- `src.scripts.plot_canonical_synthetic_users.plot_population_file(path, output_dir)`

The official robust-processing CLI lazily imports
`src.experiment_config.PATH_SELECTION_CONFIGS`. That module is supplied by the
configuration-selection integration unit, not duplicated here.

## Tests-first red evidence

All red runs were collection-clean and failed for the intended absent or weak
contract. Exact logs were retained outside the repository under `/tmp` during
the unit. Their SHA-256 values and exact terminal summaries are:

1. Command:

   ```text
   /tmp/figvenv/bin/python -m pytest tests/test_v30_feature_population.py::test_selected_track_loader_validates_exact_schema_and_provenance tests/test_v30_feature_population.py::test_all_signature_arm_admission_hook_checks_every_configuration tests/test_v30_feature_population.py::test_official_track_processing_cli_injects_all_signature_arms tests/test_v30_feature_population.py::test_strict_eda_cli_passes_the_validated_track_list -q
   ```

   Output: `4 failed in 1.88s`, `EXIT_CODE=1`. The failures were the accepted
   unexpected provenance field, missing all-arm hook, missing official hook
   injection and list/dictionary EDA mismatch. Log
   `/tmp/feature-population-red-additional.log`, SHA-256
   `976d083aa9f3478e5a7a3125a30accb160e78bc6c2503bf7157fdcc07a77ba86`.

2. Command:

   ```text
   /tmp/figvenv/bin/python -m pytest tests/test_v30_feature_population.py::test_interaction_heatmap_masks_empty_cells_and_labels_them_na -q
   ```

   Output: `1 failed in 1.94s`, `EXIT_CODE=1`, because
   `_build_interaction_matrix` did not exist. Log
   `/tmp/feature-population-red-heatmap.log`, SHA-256
   `beaca5d8fb0bc3e97072766494f2f740178104b981e8154081ee0c5d46dd6375`.

3. Command:

   ```text
   /tmp/figvenv/bin/python -m pytest tests/test_v30_feature_population.py::test_population_schema_and_json_loader_are_exact_and_fail_closed tests/test_v30_feature_population.py::test_synthetic_generator_consumes_the_strict_selected_wrapper tests/test_v30_feature_population.py::test_synthetic_generator_cli_passes_the_official_selected_count -q
   ```

   Output: `3 failed in 2.50s`, `EXIT_CODE=1`. The old code accepted an extra
   population field, had no strict-wrapper loader argument, and omitted the
   official count at the CLI boundary. Log
   `/tmp/feature-population-red-integration.log`, SHA-256
   `636b1d807223aff06e24a5a3c7f71b982c0bd2826b1d4a343424eedc100b60b2`.

4. Command:

   ```text
   /tmp/figvenv/bin/python -m pytest tests/test_v30_feature_population.py::test_feature_bundle_rejects_resigned_schema_mutations tests/test_v30_feature_population.py::test_feature_bundle_expected_ids_must_be_unique_sorted_and_exact -q
   ```

   Output: `4 failed in 2.65s`, `EXIT_CODE=1`. A re-signed manifest accepted an
   extra top-level field, extra member field and wrong endian dtype, while the
   expected-ID error was not specific. Log
   `/tmp/feature-population-red-bundle-schema.log`, SHA-256
   `82af693ac4c43f5d1c5d30a89927c8104ceb676468b47cb98b3c0dcc6e357b1f`.

5. Command:

   ```text
   /tmp/figvenv/bin/python -m pytest tests/test_v30_feature_population.py::test_compact_extraction_admits_the_exact_stored_dtypes -q
   ```

   Output: `1 failed in 2.04s`, `EXIT_CODE=1`, with the extracted path still
   `<f8` instead of the stored `<f4`. Log
   `/tmp/feature-population-red-compact-dtype.log`, SHA-256
   `13f93e5e55c3ef45573fbe7d7f5a8f1cc05bfb0cb1b6df0e37acbe97e7971a52`.

6. Command:

   ```text
   /tmp/figvenv/bin/python -m pytest tests/test_v30_feature_population.py::test_all_signature_arm_admission_hook_checks_every_configuration -q
   ```

   Output: `1 failed in 2.13s`, `EXIT_CODE=1`, because the hook redundantly
   instantiated both a signature and its mathematical superset. Log
   `/tmp/feature-population-red-admission-superset.log`, SHA-256
   `f0c528fe210b00b72fff942c5c8374518c469941758b8d051b7ae04bb6624613`.

## Final green evidence

Focused command:

```text
/tmp/figvenv/bin/python -m pytest tests/test_v30_feature_population.py tests/test_canonical_synthetic_user_figures.py -q
```

Exact output:

```text
................................                                         [100%]
32 passed in 7.59s
EXIT_CODE=0
```

Log `/tmp/feature-population-final-focused-green.log`, SHA-256
`dd8c66d71062cbfd2970ef0262adb7097537dc4b5eeff202a4ba52991af5b724`.

Repository-wide command, excluding only the three already declared disposable
environment artefacts (newer scikit-learn mock-tag behaviour and the two tests
requiring unavailable `implicit==0.7.0`):

```text
/tmp/figvenv/bin/python -m pytest tests/ -q \
  --deselect tests/test_evaluation.py::TestClassificationMetrics::test_cross_validate_model \
  --deselect tests/test_run_baseline_comparison_cli_integration.py::test_full_warm_run_with_real_scorers_and_stubbed_lightfm \
  --deselect tests/test_run_cold_start_comparison_cli.py::test_full_cold_start_run_with_real_scorers_and_stubbed_lightfm
```

Exact summary:

```text
632 passed, 3 deselected, 8 warnings in 52.25s
EXIT_CODE=0
```

Log `/tmp/feature-population-final-full-green.log`, SHA-256
`f9ed66aee55a1effba63d371190009cbdc14efb90389b850ff4067c89fbe7990`.
The warnings are from unchanged legacy EDA and statistical-test code.
