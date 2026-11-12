# Publication record and remaining decisions

The local MVP passed independent v4 review (by ChatGPT): 309 tests passed with private real data present, and overflow handling,
strict dashboard JSON, loopback binding, artifact integrity, and slider isolation were checked.
The repository was published from a verified 67-file source package plus the licensing documentation. Hosted CI status is
recorded in the README only after an actual GitHub Actions run has been inspected.

## Repository contents

- Application/pipeline source, meaningful tests, and the generated synthetic fixture.
- Setup and synthetic demo instructions, the workflow, and reproducible real-data commands.
- Aggregate real-data metrics with split/threshold rules and provenance hashes in `reports/`.
- Dataset source metadata, captured license declaration, attribution, and the supplied ODbL/DbCL texts.
- `LICENSE` (MIT, original code only).

Excluded: raw data, real row-level derivatives, model pickles, environments, caches, private review ZIPs/logs,
captured HTTP cookies, and one-off cross-project transfer/inspection scripts. Generated artifacts remain ignored.
Reproduction creates model artifacts locally from a declared fixture or the pinned dataset.

## Licensing decisions (recorded 2026-11-12, by the owner)

1. **Code license:** MIT, for the original project code, tests, scripts and documentation. Copyright line: "RyanJBush".
2. **Dataset attribution/publication:** the existing [NOTICE_DATA.md](../NOTICE_DATA.md), source declaration, and linked terms were
   approved for public publication. The dataset licences (ODbL for the database, DbCL for the contents) do not license this
   repository's code, and the MIT licence does not cover the dataset.
3. **Row-level derivatives:** excluded. Publishing any later export would require a separate decision; private review packets
   are not release assets.
4. **Visibility:** public, `RyanJBush/fraud-risk-dashboard`.

The metadata in `data/manifest.json` records hashes of the original acquisition evidence and is intentionally unchanged. Some
private captures named there are absent, including the response headers containing cookies. Their absence does not change
the downloaded dataset checksum or the preserved frozen evaluation artifacts.

## Remaining limitations

- Hosted CI: the first public commit (`4ca5a0e`) passed on GitHub Actions (run 43294204336: 308 passed, 1 skipped, synthetic pipeline, tamper
  refusal). Later commits are covered only by their own runs in the Actions tab.
- Confirming that the OpenML ARFF equals the Kaggle CSV v3 beyond the published counts still needs a Kaggle login.
- The model is a demonstration: 75 test positives, a two-day window, undocumented PCA and label process, no entity identifiers,
  uncalibrated scores, and a documented batch-versus-single-row boundary effect. No business savings are claimed.

No further model selection, threshold tuning, or real-data retraining is needed for this MVP.
