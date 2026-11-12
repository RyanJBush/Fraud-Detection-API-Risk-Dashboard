# Dataset license and reuse evidence

Status: **declarations and linked terms captured and written up in `NOTICE_DATA.md`; the owner's sign-off is PENDING.** This file records what was observed. It is not
legal advice and it does not claim that every use is permitted.

## Declarations observed

| Source | Observed declaration | How obtained | Evidence file (hashes in `data/manifest.json`) |
|---|---|---|---|
| Kaggle (original publisher page) | "Database: Open Database, Contents: Database Contents", linking `http://opendatacommons.org/licenses/dbcl/1.0/` | Firecrawl scrape and an independent direct `curl` (HTML contains the JSON-LD license object) on 2026-11-11 | `data/meta/kaggle_dataset_page.html`, `data/meta/kaggle_license_capture.md` |
| OpenML dataset 1597 v1 metadata | `"licence":"Public"` (no terms linked) | `GET https://www.openml.org/api/v1/json/data/1597` | `data/meta/openml_data_1597.json` |
| Amazon Science Fraud Dataset Benchmark README | Lists this dataset (key `ccfraud`) with "Source license: https://opendatacommons.org/licenses/dbcl/1-0/" | `curl` of the raw README on 2026-11-11 | `data/meta/fdb_readme.md` |
| Open Data Commons DbCL 1.0 text | Section 2.1 grants a worldwide, royalty-free, perpetual copyright licence over the Contents, explicitly including commercial use; section 2.2: "You must comply with the ODbL"; section 2.3: the DbCL does not cover Database Rights, which are covered by the ODbL | `curl` of the licence page | `data/meta/dbcl_1.0_page.html` |
| Open Data Commons ODbL 1.0 text | Saved (HTML here, plain text in `licenses/`); attribution / share-alike conditions summarised in plain words in `NOTICE_DATA.md`, not legally analysed | `curl` | `data/meta/odbl_1.0_page.html` |

## Consistency and gaps

- Kaggle (database under ODbL, contents under DbCL) and the Amazon Science listing (DbCL) agree on DbCL for the contents.
  OpenML's single word "Public" is less specific; it is not treated as a licence grant on its own.
- Kaggle's page asks for citation of Dal Pozzolo et al. (CIDM 2015); this repository's README cites it.
- `NOTICE_DATA.md` records attribution, the ODbL 4.2 / 4.3 / 4.4 / 4.5 conditions in plain words, and that the code licence does not cover the dataset. Plain-text licence copies are in `licenses/`.
- **Owner decision (2026-11-12):** the existing `NOTICE_DATA.md` and its attribution were approved for public publication, and the project's
  original code is MIT-licensed (`LICENSE`). Row-level derivatives (membership, scores, overlap rows) remain excluded; publishing any of them
  would need a separate decision because it could be a Derivative Database under ODbL 4.4.
- **Still open:** confirming the OpenML ARFF matches the Kaggle CSV v3 beyond published counts. The CSV requires a Kaggle login and was not
  downloaded.
- No raw data or row-level derivative is redistributed by this repository.
