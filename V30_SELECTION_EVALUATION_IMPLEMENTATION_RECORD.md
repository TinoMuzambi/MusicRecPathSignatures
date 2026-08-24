# V30 selection and evaluation implementation evidence

This unit was implemented in the disposable code checkout
`/tmp/msc-selection-evaluation.AcaoTZ/code`, based on exact source commit
`3e682eab21528894b75a62120f2decfe08f23ed5`. The durable nested worktree was
neither modified nor executed. No data, network, GCP or model run was used.

## Tests-first red evidence

The first clean run used:

```text
/tmp/figvenv/bin/python -m pytest -q \
  tests/test_experiment_selection_contract.py \
  tests/test_prepared_cosine_and_standardised_audio.py \
  tests/test_strict_genre_diagnostic.py \
  tests/test_statistical_hardening_v30.py
```

It exited 1 with 17 intended behavioural failures and no collection,
dependency or import failure. The captured output is
`/tmp/selection-red.log`, SHA-256
`421070f750d496be9044c052392364ec1e533c3a968372ba0d71459b1b879db6`.
The failures covered the absent immutable experiment contract and selection
engine, unprepared path cosine, unstandardised traditional audio, the absent
held-out diagnostic, incorrect Cliff's delta, permissive paired Cohen's d and
an incomplete inference family. Smaller tests-first red runs then confirmed
the parameterised adapters (2 failures), fresh selection-stage writer (1),
compact-bundle signature provider (1), selected final-runner binding (1) and
warm/withheld task constructor (1) before those implementations were added.

## Green evidence

The final focused command was:

```text
/tmp/figvenv/bin/python -m pytest -q \
  tests/test_experiment_selection_contract.py \
  tests/test_prepared_cosine_and_standardised_audio.py \
  tests/test_parameterised_baseline_adapters_v30.py \
  tests/test_selected_final_runner_v30.py \
  tests/test_final_tasks_v30.py \
  tests/test_strict_genre_diagnostic.py \
  tests/test_statistical_hardening_v30.py \
  tests/test_run_validation_selection_cli_v30.py
```

It exited 0: `27 passed in 2.23s`. Captured output
`/tmp/selection-green-final.log` has SHA-256
`4e1368e10b03b25ae0e1e31b3652201365fe9cf893e727f5eed58340c3f12d46`.
`python -m compileall -q src tests` and `git diff --check` also exited 0.

Immediately before the final strict manifest-roster and decomposition
cross-check assertions, the full disposable-environment suite exited 1 with
`628 passed, 3 failed` in
75.43 seconds. Captured output `/tmp/selection-full.log` has SHA-256
`54f1ff1df5ef1d496ee068aa30a71d5ce526159a1ebd40b25e66aa948253df36`.
The three failures are the already declared environment artefacts:

- `test_evaluation.py::TestClassificationMetrics::test_cross_validate_model`
  uses a mock lacking the tags required by this venv's newer scikit-learn;
- the warm and cold real-scorer CLI integration tests require the absent
  frozen `implicit` dependency.

No additional full-suite failure remained.

## Implemented boundary

The implementation adds the immutable 4,000-track, 200-user contract, all 18
joint path arms and exact comparator grids, validation-only selection, compact
bundle path slicing, catalogue-column-standardised traditional audio, prepared
cosine retrieval, selected baseline adapters, stable method identifiers, exact
22-output final execution, immutable-population warm and additive withheld-item
tasks, strict five-fold genre diagnostics, complete five-comparison inference
and corrected effect sizes. Selection artefacts bind the selected-tracks file,
population file, compact-bundle manifest and every compact member. Final run
identity stores both the canonical selection content SHA-256 and the distinct
newline-bearing on-disk selection file SHA-256 under the existing `models`
identity section.
