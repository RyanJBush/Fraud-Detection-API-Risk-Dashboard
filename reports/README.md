# Aggregate real-data results

`heldout_results.json` copies the verified counts, metrics, split/threshold rules, and provenance hashes from the
preserved frozen real run. It contains no transaction features, row identifiers, membership lists, or per-row scores.
It is a static report, not a model bundle and not the input to the synthetic demo.

Evaluation uses whole-partition batches. The API uses single-row scoring; a documented floating-point boundary case
changes one validation flag. The reported full test scan found no flag differences. No calibration, savings, or
deployment performance is claimed. Hosted CI passed on the first public commit (see the README for the run link).

Contains information from [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
made available under the [Open Database License](https://opendatacommons.org/licenses/odbl/1-0/);
contents under [DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/).
Creators: Worldline and ULB's Machine Learning Group.
Citation: Dal Pozzolo, Caelen, Johnson, Bontempi, *Calibrating Probability with Undersampling for Unbalanced
Classification*, IEEE CIDM, 2015. See [NOTICE_DATA.md](../NOTICE_DATA.md).
The repository's original code is MIT-licensed (see [LICENSE](../LICENSE)); that licence does not apply to the dataset, and
these aggregate figures derive from it, so the dataset notice above applies to them.
