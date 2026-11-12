# Data notice: attribution, licence declarations, and what this repository does with the data

This notice covers the **dataset**, not the code. The project's original code is licensed under the MIT licence (see
[LICENSE](LICENSE)); that licence does **not** apply to the dataset or to anything derived from its contents, and must not be
read as relicensing it. The owner approved this notice and the attribution below for public publication on 2026-11-12. This is
a record of declarations, not legal advice.

## Dataset and attribution (kept as published)

- Name: *Credit Card Fraud Detection* (OpenML name `creditcard`).
- Creators / holders named by the publisher: Machine Learning Group of the Universite Libre de Bruxelles (ULB) and
  Worldline. OpenML lists the creators as Andrea Dal Pozzolo, Olivier Caelen and Gianluca Bontempi.
- Original publication: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (Kaggle owner "Machine Learning Group - ULB").
- Copy used here: OpenML dataset 1597, version 1, file 1673544,
  https://openml.org/data/v1/download/1673544/creditcard.arff (SHA-256 of the file is in `data/manifest.json`).
- Citation requested by the publisher (Kaggle page and OpenML "Please cite"): Andrea Dal Pozzolo, Olivier Caelen,
  Reid A. Johnson and Gianluca Bontempi. *Calibrating Probability with Undersampling for Unbalanced Classification.*
  Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015.
  The publisher's page lists further works to cite; see `data/meta/kaggle_license_capture.md`.

## Licence declarations captured (2026-11-11)

| Where | Declaration | Linked terms |
|---|---|---|
| Kaggle (original publisher) | "Database: Open Database, Contents: Database Contents" | http://opendatacommons.org/licenses/dbcl/1.0/ |
| OpenML metadata | `"licence": "Public"` | none linked; not treated as a grant on its own |
| Amazon Science Fraud Dataset Benchmark README | "Source license: https://opendatacommons.org/licenses/dbcl/1-0/" | same DbCL 1.0 |

Texts kept in `licenses/` (plain text, fetched from opendatacommons.org on 2026-11-11):

- `licenses/DbCL-1.0.txt`, SHA-256 `3cbfa990f26df856c6a7d3fff3d1f236e3b0165b1e8257ca0d00f15758754f81`
- `licenses/ODbL-1.0.txt`, SHA-256 `607680718977f6f6c9607972afd98f208573f19251315ed1362a8589b51beaf5`

Inference, not a published statement: Kaggle names "Open Database" without a version link. DbCL 1.0 incorporates the
ODbL 1.0 definitions and says in section 2.2 "You must comply with the ODbL", so ODbL 1.0 is taken as the database
licence. The DbCL's own section 2.3 says it does not cover database rights or copyright in the database as a whole;
the ODbL covers those.

## Conditions that matter here (a plain reading of the quoted sections, confirmed by the owner for this publication; not legal advice)

- **DbCL 2.1** grants a worldwide, royalty-free, perpetual copyright licence over the Contents, including commercial use.
  **DbCL 2.2** adds that the ODbL must be complied with.
- **ODbL 4.2 (public conveyance of the Database or a Derivative Database):** give the licence text or its URI, and keep
  copyright / database-right notices intact.
- **ODbL 4.3 (publicly using a Produced Work):** include a notice that content came from the database and is available
  under the ODbL. Example wording from the licence:
  "Contains information from Credit Card Fraud Detection (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
  which is made available here under the Open Database License (ODbL) (https://opendatacommons.org/licenses/odbl/1-0/)."
- **ODbL 4.4 / 4.5 (share-alike):** a *Derivative Database* that is publicly used must stay under the ODbL (or a
  compatible licence). Extracting a Substantial part of the contents into a new database is a Derivative Database
  (4.4 b). Creating a Produced Work from the database does not by itself make a Derivative Database (4.5 b), and
  purely internal use is not public (4.5 c).

## What this repository does with the data

- Raw data is **not** committed or redistributed (`data/raw/` is git-ignored; no raw file is in this repository).
- Intended repository material is code, tests, a synthetic fixture (generated, no real records), aggregate results in
  `reports/heldout_results.json`, hashes, and source / licence evidence. Captured HTTP headers with cookies, private review
  packets, and one-off cross-project handoff scripts are excluded from the publication tree.
- **Excluded, not cleared for publication:** row-level derived files such as `split_membership.csv.gz` (row id, row number,
  Time, Class for every row), `test_scores.csv.gz` and `cross_partition_overlap_rows.csv.gz`. Whether publishing them could be a
  Substantial extraction (a Derivative Database under ODbL 4.4) would need a separate owner decision. They are git-ignored and
  never part of this repository. Aggregate metrics and counts are the public output; each public figure or table built from this
  data carries the ODbL 4.3 notice above (see the README results section and `reports/README.md`).
- Third-party page captures in `data/meta/*.html` and `data/meta/fdb_readme.md` are local evidence only (git-ignored).

## Status

- Owner approval (2026-11-12): this notice and attribution, public publication of the aggregate results, and MIT for the original code.
  Publishing any row-level derivative is not approved.
- `data/manifest.json` is historical acquisition provenance and is intentionally unchanged: its `license_evidence` text predates the
  owner's approval, and it lists the hash of a captured response-header file that is not published.

## Still open

- Confirming the OpenML ARFF equals the Kaggle CSV v3 (needs a Kaggle login); only published counts were compared.
