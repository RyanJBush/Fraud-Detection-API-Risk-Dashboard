import numpy as np
import pytest

from fraudrisk import constants as C
from fraudrisk.errors import PartitionUnusableError
from fraudrisk.split import assert_usable, cross_partition_audit, plan_chronological_split, summarize_split

from conftest import make_frame


def test_boundary_moves_to_start_of_timestamp_group():
    # N=10 -> nominal boundaries 6 and 8. sorted[5]==sorted[6]==5 -> first boundary moves to 5.
    time = np.array([0, 1, 2, 3, 4, 5, 5, 6, 7, 8], dtype=float)
    plan = plan_chronological_split(time)
    assert plan.nominal_boundaries == (6, 8)
    assert plan.boundaries == (5, 8)
    assert list(plan.assignment) == ["train"] * 5 + ["validation"] * 3 + ["test"] * 2


def test_boundary_already_at_group_start_does_not_move():
    time = np.arange(10, dtype=float)
    assert plan_chronological_split(time).boundaries == (6, 8)


def test_sort_by_time_then_row_number_and_original_positions_preserved():
    time = np.array([9, 1, 5, 5, 3, 7, 5, 2, 8, 0], dtype=float)  # unsorted with a 3-way tie at 5
    plan = plan_chronological_split(time)
    # sorted order: 0@9, 1@1, 2... compute expected by hand: times ascending, ties by row number
    expected = [9, 1, 7, 4, 2, 3, 6, 5, 8, 0]
    assert list(plan.order) == expected
    assert plan.assignment[9] == "train" and plan.assignment[0] == "test"


@pytest.mark.parametrize("seed", range(8))
def test_equal_timestamps_never_cross_partitions(seed):
    rng = np.random.default_rng(seed)
    time = rng.integers(0, 15, size=200).astype(float)  # heavy ties
    plan = plan_chronological_split(time)
    sets = {n: set(time[plan.assignment == n].tolist()) for n in C.SPLIT_NAMES}
    assert not (sets["train"] & sets["validation"])
    assert not (sets["validation"] & sets["test"])
    assert not (sets["train"] & sets["test"])
    # every row exactly once
    assert sum((plan.assignment == n).sum() for n in C.SPLIT_NAMES) == 200


def test_sizes_are_close_to_60_20_20_without_ties():
    plan = plan_chronological_split(np.arange(1000, dtype=float))
    counts = [int((plan.assignment == n).sum()) for n in C.SPLIT_NAMES]
    assert counts == [600, 200, 200]


def test_giant_tie_group_makes_validation_empty_and_fails_without_random_fallback():
    # N=20: nominal boundaries 12 and 16 both fall inside the Time==1 group (sorted positions 5..14)
    # and 16 is in the Time==2 group; hand check: boundary 12 -> 5, boundary 16 -> 15.
    time = np.array([0.0] * 5 + [1.0] * 10 + [2.0] * 5)
    plan = plan_chronological_split(time)
    assert plan.boundaries == (5, 15)
    # now put both boundaries inside ONE group: Time==1 spans sorted positions 2..17
    time = np.array([0.0] * 2 + [1.0] * 16 + [2.0] * 2)
    plan = plan_chronological_split(time)
    assert plan.boundaries == (2, 2)  # validation becomes empty
    summary = summarize_split(plan, time, np.array([0, 1] * 10))
    assert summary["partitions"]["validation"]["n_rows"] == 0
    assert summary["usable"] is False
    with pytest.raises(PartitionUnusableError):
        assert_usable(summary)
    assert summary["randomization"].startswith("none")


def test_partition_without_positives_is_unusable():
    time = np.arange(20, dtype=float)
    y = np.zeros(20, dtype=int)
    y[0] = 1  # only train has a positive
    summary = summarize_split(plan_chronological_split(time), time, y)
    assert not summary["usable"]
    assert any("validation partition has no positive labels" in r for r in summary["unusable_reasons"])
    assert any("test partition has no positive labels" in r for r in summary["unusable_reasons"])


def test_summary_counts_and_prevalence_are_observed_not_rebalanced():
    time = np.arange(10, dtype=float)
    y = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    s = summarize_split(plan_chronological_split(time), time, y)
    p = s["partitions"]
    assert (p["train"]["n_rows"], p["train"]["n_fraud"]) == (6, 2)
    assert (p["validation"]["n_rows"], p["validation"]["n_fraud"]) == (2, 1)
    assert (p["test"]["n_rows"], p["test"]["n_fraud"]) == (2, 1)
    assert p["train"]["fraud_prevalence"] == pytest.approx(2 / 6)
    assert s["usable"] and s["rows_preserved"] and s["time_ranges_strictly_ordered"]


def _audit(df, plan):
    y = df[C.LABEL_COLUMN].to_numpy()
    ids = [f"r{i}" for i in range(len(df))]
    return cross_partition_audit(df.reset_index(drop=True), plan.assignment, y, ids)


def test_cross_partition_overlap_detected_ignoring_time():
    df = make_frame(20)
    plan = plan_chronological_split(df["Time"].to_numpy())
    audit, _ = _audit(df, plan)
    assert audit["cross_partition_repeated_fingerprints"] == 0 and not audit["review_required"]

    # copy row 1 (train) into row 18 (test) with identical features; Time differs naturally
    df.loc[18, list(C.FEATURE_COLUMNS)] = df.loc[1, list(C.FEATURE_COLUMNS)].to_numpy()
    audit, rows = _audit(df, plan)
    assert audit["cross_partition_repeated_fingerprints"] == 1
    assert audit["crossing_fingerprints_by_partition_pair"]["train_and_test"] == 1
    assert audit["test_rows_with_fingerprint_in_train"]["rows"] == 1
    assert audit["review_required"]
    assert set(rows["split"]) == {"train", "test"}
    assert audit["time_plus_features_fingerprints_crossing_partitions"] == 0  # Time differs


def test_amount_difference_means_different_fingerprint():
    df = make_frame(20)
    plan = plan_chronological_split(df["Time"].to_numpy())
    df.loc[18, list(C.FEATURE_COLUMNS)] = df.loc[1, list(C.FEATURE_COLUMNS)].to_numpy()
    df.loc[18, "Amount"] = df.loc[1, "Amount"] + 0.01
    audit, _ = _audit(df, plan)
    assert audit["cross_partition_repeated_fingerprints"] == 0


def test_within_partition_repeats_are_not_cross_partition():
    df = make_frame(20)
    plan = plan_chronological_split(df["Time"].to_numpy())
    df.loc[2, list(C.FEATURE_COLUMNS)] = df.loc[1, list(C.FEATURE_COLUMNS)].to_numpy()  # both train
    audit, _ = _audit(df, plan)
    assert audit["cross_partition_repeated_fingerprints"] == 0
    assert audit["per_partition"]["train"]["repeated_fingerprint_groups_within_partition"] == 1
