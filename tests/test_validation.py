import numpy as np
import pandas as pd
import pytest

from fraudrisk import constants as C
from fraudrisk.dataio import load_dataset, make_row_ids, parse_fidelity_check
from fraudrisk.errors import ConflictingLabelsError, DataValidationError
from fraudrisk.validation import inspect_frame, validate_frame

from conftest import make_frame


def test_clean_frame_passes(frame):
    r = validate_frame(frame)
    assert r["status"] == "ok" and r["n_rows"] == len(frame)
    assert r["conflicting_label_groups"] == 0


def test_missing_and_unexpected_columns_reported(frame):
    bad = frame.drop(columns=["V7"]).assign(extra_col=1.0)
    r = inspect_frame(bad)
    assert r["missing_columns"] == ["V7"] and r["unexpected_columns"] == ["extra_col"]
    with pytest.raises(DataValidationError):
        validate_frame(bad)


@pytest.mark.parametrize(
    "mutate, needle",
    [
        (lambda d: d.__setitem__("V3", d["V3"].where(d.index != 2, np.nan)), "missing values"),
        (lambda d: d.__setitem__("V3", d["V3"].where(d.index != 2, np.inf)), "non-finite"),
        (lambda d: d.__setitem__("Amount", d["Amount"].where(d.index != 2, -0.01)), "negative Amount"),
        (lambda d: d.__setitem__("Time", d["Time"].where(d.index != 2, -1.0)), "negative Time"),
        (lambda d: d.__setitem__("Class", d["Class"].where(d.index != 2, 2)), "labels outside"),
    ],
)
def test_malformed_values_rejected(frame, mutate, needle):
    bad = frame.copy()
    mutate(bad)
    r = inspect_frame(bad)
    assert r["status"] == "failed"
    assert any(needle in e for e in r["errors"]), r["errors"]
    with pytest.raises(DataValidationError):
        validate_frame(bad)


def test_class_must_contain_both_labels(frame):
    bad = frame.assign(Class=0)
    assert any("both 0 and 1" in e for e in inspect_frame(bad)["errors"])


def test_time_equal_to_row_index_is_rejected(frame):
    bad = frame.assign(Time=np.arange(len(frame), dtype="float64"))
    assert any("row index" in e for e in inspect_frame(bad)["errors"])


def test_duplicate_counts_hand_calculated():
    df = make_frame(10)
    # row 5: exact copy of row 1 (same Time too) -> identical full record
    df.loc[5] = df.loc[1]
    # row 7: same features as row 2 but different Time (and same label) -> feature repeat only
    df.loc[7, list(C.FEATURE_COLUMNS)] = df.loc[2, list(C.FEATURE_COLUMNS)].to_numpy()
    df.loc[7, "Class"] = df.loc[2, "Class"]
    d = inspect_frame(df)["duplicates"]
    assert d["identical_records_all_columns_incl_time_and_class"]["extra_copies"] == 1
    assert d["identical_time_plus_features"]["extra_copies"] == 1
    assert d["identical_scoring_features_excluding_time"]["extra_copies"] == 2
    assert d["identical_scoring_features_excluding_time"]["groups_with_repeats"] == 2
    assert d["identical_scoring_features_excluding_time"]["rows_in_repeated_groups"] == 4


def test_conflicting_labels_detected_and_stop_code():
    df = make_frame(10)
    df.loc[6, list(C.FEATURE_COLUMNS)] = df.loc[1, list(C.FEATURE_COLUMNS)].to_numpy()
    df.loc[6, "Class"] = 1 - int(df.loc[1, "Class"])
    r = inspect_frame(df)
    assert r["conflicting_label_groups"] == 1 and r["conflicting_label_rows"] == 2
    with pytest.raises(ConflictingLabelsError) as ei:
        validate_frame(df)
    assert ei.value.exit_code == C.EXIT_CONFLICTING_LABELS


def test_negative_zero_fingerprints_equal():
    df = make_frame(8)  # fraud rows are 3 and 7; rows 2 and 6 are both Class 0
    df.loc[6, list(C.FEATURE_COLUMNS)] = df.loc[2, list(C.FEATURE_COLUMNS)].to_numpy()
    df.loc[2, "V1"] = 0.0
    df.loc[6, "V1"] = -0.0
    assert inspect_frame(df)["duplicates"]["identical_scoring_features_excluding_time"]["extra_copies"] == 1


def test_published_expectation_mismatch_fails(frame):
    r = inspect_frame(frame, expect_published=True)
    assert r["status"] == "failed" and "published expectations" in r["errors"][-1]


def test_row_ids_are_stable_and_derived_from_checksum_and_row_number():
    a = make_row_ids("ab" * 32, 3)
    assert a == ["abababababab-0000000", "abababababab-0000001", "abababababab-0000002"]
    assert make_row_ids("cd" * 32, 1)[0] != a[0]


def _write_arff(path, df, class_decl="{'0','1'}", names=None):
    names = names or list(C.SOURCE_COLUMNS)
    lines = ["@relation test"]
    for n in names:
        lines.append(f"@attribute {n} " + (class_decl if n == "Class" else "numeric"))
    lines.append("@data")
    for _, r in df.iterrows():
        vals = [repr(float(r[c])) for c in C.SOURCE_COLUMNS[:-1]] + [f"'{int(r['Class'])}'"]
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_arff_load_keeps_time_and_exact_values(tmp_path, frame):
    p = tmp_path / "t.arff"
    _write_arff(p, frame)
    loaded = load_dataset(p)
    assert loaded.frame["Time"].tolist() == frame["Time"].tolist()  # Time survives loading
    assert loaded.frame["Time"].nunique() > 1
    assert np.array_equal(loaded.frame["V1"].to_numpy(), frame["V1"].to_numpy())
    assert validate_frame(loaded.frame)["status"] == "ok"
    pf = parse_fidelity_check(p, loaded.frame, k=5)
    assert pf["performed"] and pf["mismatching_rows"] == 0
    assert loaded.row_ids[0].startswith(loaded.source_sha256[:12])


def test_arff_wrong_schema_rejected(tmp_path, frame):
    p = tmp_path / "bad.arff"
    names = list(C.SOURCE_COLUMNS)
    names[0], names[1] = names[1], names[0]  # swapped attribute order
    _write_arff(p, frame, names=names)
    with pytest.raises(DataValidationError, match="do not match"):
        load_dataset(p)


def test_arff_class_declaration_checked(tmp_path, frame):
    p = tmp_path / "bad2.arff"
    _write_arff(p, frame, class_decl="{'0','1','2'}")
    with pytest.raises(DataValidationError, match="Class declaration"):
        load_dataset(p)
