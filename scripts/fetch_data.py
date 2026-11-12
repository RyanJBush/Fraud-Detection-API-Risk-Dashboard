"""Download the pinned OpenML ARFF and write data/manifest.json.

    python scripts/fetch_data.py            # downloads data/raw/creditcard.arff (git-ignored), writes data/manifest.json

Pinned source: OpenML dataset 1597, version 1, file id 1673544. Nothing is substituted: if the OpenML
MD5 in the metadata does not match the downloaded bytes the script exits non-zero.
The license evidence files under data/meta are captured separately (see docs/license_evidence.md) and
are only hashed and referenced here.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "creditcard.arff"
META = ROOT / "data" / "meta"
ADVERTISED_URL = "https://openml.org/data/v1/download/1673544/creditcard.arff"
METADATA_URL = "https://www.openml.org/api/v1/json/data/1597"
QUALITIES_URL = "https://www.openml.org/api/v1/json/data/qualities/1597"
KAGGLE_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    sha, md5 = hashlib.sha256(), hashlib.md5()
    req = urllib.request.Request(url, headers={"User-Agent": "fraudrisk-data-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        info = {"final_url": r.geturl(), "http_status": r.status,
                "content_disposition": r.headers.get("Content-Disposition"),
                "content_length_header": r.headers.get("Content-Length")}
        for block in iter(lambda: r.read(1 << 20), b""):
            f.write(block)
            sha.update(block)
            md5.update(block)
    info.update(sha256=sha.hexdigest(), md5=md5.hexdigest(), bytes=dest.stat().st_size)
    return info


def main() -> int:
    META.mkdir(parents=True, exist_ok=True)
    started = now()
    meta_info = fetch(METADATA_URL, META / "openml_data_1597.json")
    fetch(QUALITIES_URL, META / "openml_data_1597_qualities.json")
    desc = json.loads((META / "openml_data_1597.json").read_text(encoding="utf-8"))["data_set_description"]
    assert desc["id"] == "1597" and desc["version"] == "1" and desc["file_id"] == "1673544", "pinned ids changed"
    info = fetch(ADVERTISED_URL, RAW)
    finished = now()
    md5_ok = info["md5"] == desc["md5_checksum"]
    manifest = {
        "dataset": {
            "name": desc["name"], "openml_id": 1597, "version": 1, "file_id": 1673544,
            "version_label": desc["version_label"], "format": desc["format"],
            "default_target_attribute": desc["default_target_attribute"],
            "row_id_attribute_per_openml": desc["row_id_attribute"],
            "row_id_attribute_note": "OpenML marks Time as the row id; this project keeps Time as an elapsed-seconds "
                                     "column for chronological partitioning (asserted by data-check).",
            "original_source": KAGGLE_URL,
            "creators": desc["creator"], "upload_date": desc["upload_date"],
            "openml_license_field": desc["licence"],
        },
        "source_urls": {"advertised_download": ADVERTISED_URL, "resolved_download": info["final_url"],
                        "openml_metadata": METADATA_URL, "openml_qualities": QUALITIES_URL,
                        "original_kaggle_page": KAGGLE_URL},
        "retrieval": {"started_utc": started, "finished_utc": finished, "http_status": info["http_status"],
                      "content_disposition": info["content_disposition"],
                      "content_length_header": info["content_length_header"], "tool": "python urllib (scripts/fetch_data.py)"},
        "file": {"local_path": "data/raw/creditcard.arff", "original_filename_in_url": "creditcard.arff",
                 "server_supplied_filename": info["content_disposition"], "bytes": info["bytes"],
                 "sha256": info["sha256"], "md5": info["md5"],
                 "openml_published_md5": desc["md5_checksum"], "md5_matches_openml": md5_ok},
        "metadata_files": {p.name: {"sha256": sha256_of(p), "bytes": p.stat().st_size}
                           for p in sorted(META.glob("*")) if p.is_file() and p.name != "manifest.json"},
        "license_evidence": "docs/license_evidence.md (declaration captured; reuse sign-off pending reviewer)",
        "git_policy": "data/raw is git-ignored; raw data is never committed",
    }
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"bytes={info['bytes']} sha256={info['sha256']} md5={info['md5']} md5_matches_openml={md5_ok}")
    return 0 if md5_ok else 1


if __name__ == "__main__":
    sys.exit(main())
