# Provenance wording notes

## Wording used from now on

- "Frozen before test scoring": the model, features, thresholds and secondary-subset ids are hashed into
  `frozen_config.json` before any test row is scored. The data file itself is read earlier, for validation,
  partitioning and the overlap audits; test rows are **excluded from fitting, validation scoring and threshold
  selection**, not from process memory.
- New runs write `training_provenance.test_rows_excluded_from_fitting_and_threshold_selection: true` (plus a
  `test_rows_provenance_note`) and `frozen_config.json` key `frozen_before_test_scoring: true`.

## Historical wording in the already-frozen real-run artifacts (left unchanged on purpose)

`artifacts/real/model_config.json` (SHA-256 `8427cbb162aef116206e11679d649ec599ab1a41ff8db3078bfb4e3f7c616e2c`) contains
`training_provenance.test_rows_present_in_memory_during_fit: false`, and `artifacts/real/frozen_config.json`
(SHA-256 `8518296e640086a4902751a98a27f94cc5a7f6da9b9161abfc0faa0fd0b256e2`) contains
`frozen_before_test_evaluation: true`. The first is misleading: the loaded dataset, which includes test rows, *is* in
memory in the `train` process. What it was meant to say, and what the code does, is that test rows are removed from
the frame before fitting and threshold selection (`cli.cmd_train`: `keep = plan.assignment != "test"`). These frozen
files were not edited, because their hashes are pinned by `evaluate` and recorded in the review packets; rewriting them
would silently change verified evidence. Read the old field as "test rows excluded from fitting and threshold
selection".
