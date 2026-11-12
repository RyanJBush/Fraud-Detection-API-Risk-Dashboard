# Kaggle license capture (manual transcription of a tool result)

Page: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Captured: 2026-11-11, about 05:52 UTC (exact second not recorded), by the Firecrawl scrape tool (a third-party fetch service) with maxAge=0.
Independent cross-check: a direct `curl` of the same URL at 2026-11-11T05:59:24Z returned HTTP 200; its embedded
JSON-LD contains the license object quoted below. That raw HTML is saved as `kaggle_dataset_page.html`
(SHA-256 in `data/manifest.json`).

## What the page declares (License section)

    Database: Open Database, Contents: Database Contents
    link: http://opendatacommons.org/licenses/dbcl/1.0/

## JSON-LD in the raw HTML (verbatim)

    "license":{"@type":"CreativeWork","name":"Database: Open Database, Contents: Database Contents","url":"http://opendatacommons.org/licenses/dbcl/1.0/"}

## Other facts shown on the page that matter here

- Dataset owner: Machine Learning Group - ULB. File: creditcard.csv (150.83 MB), "Version 3".
- The page asks users to cite (among others): Dal Pozzolo, Caelen, Johnson, Bontempi,
  "Calibrating Probability with Undersampling for Unbalanced Classification", IEEE CIDM 2015.
- Page text on the data: 492 frauds out of 284,807 transactions over two days; V1-V28 are PCA components;
  "due to confidentiality issues, we cannot provide the original features and more background information".

## Not established by this capture

- That the OpenML ARFF is byte- or row-identical to the Kaggle CSV v3 (the CSV was not downloaded; Kaggle
  requires authentication). Only the published counts were compared (they match).
- Any legal conclusion. This file records declarations only.
