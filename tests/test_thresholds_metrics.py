import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from fraudrisk.errors import EvaluationPrerequisiteError
from fraudrisk.thresholds import (
    confusion_counts,
    evaluate_at_threshold,
    review_flags,
    select_max_f1_threshold,
)


# ---- decision rule: review_flag = model_score >= threshold --------------------------------------
def test_threshold_equality_and_neighbours():
    assert review_flags(np.array([0.79, 0.80, 0.92]), 0.80).tolist() == [False, True, True]
    below, above = np.nextafter(0.8, 0.0), np.nextafter(0.8, 1.0)
    assert review_flags(np.array([below, 0.8, above]), 0.8).tolist() == [False, True, True]


def test_full_precision_comparison_not_rounded():
    # displays as 0.8 at 1 decimal but is strictly below 0.8
    s = np.array([0.7999999999999999])
    assert review_flags(s, 0.8).tolist() == [False]


# ---- selection ------------------------------------------------------------------------------------
Y = np.array([1, 0, 1, 0, 0, 1])
S = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])


def test_selection_hand_calculated_table_and_exact_tie_goes_to_highest_threshold():
    # F1 by threshold (hand): 0.9->2/4, 0.8->2/5, 0.7->4/6, 0.6->4/7, 0.5->4/8, 0.4->6/9
    res, table = select_max_f1_threshold(Y, S)
    assert table["threshold"].tolist() == [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    assert table["f1_numerator"].tolist() == [2, 2, 4, 4, 4, 6]
    assert table["f1_denominator"].tolist() == [4, 5, 6, 7, 8, 9]
    assert res["threshold"] == 0.7  # 4/6 == 6/9 exactly; highest threshold wins
    assert (res["validation_tp"], res["validation_fp"], res["validation_fn"], res["validation_tn"]) == (2, 1, 1, 2)
    assert res["n_candidates_tied_at_best_f1"] == 2
    assert res["n_candidate_thresholds"] == 6


def test_selection_independent_of_row_order():
    rng = np.random.default_rng(0)
    base, _ = select_max_f1_threshold(Y, S)
    for _ in range(10):
        p = rng.permutation(len(Y))
        res, _ = select_max_f1_threshold(Y[p], S[p])
        assert res["threshold"] == base["threshold"]


def test_candidates_are_distinct_scores_with_inclusive_ties():
    y = np.array([1, 0, 1, 0])
    s = np.array([0.5, 0.5, 0.2, 0.2])  # two distinct scores
    res, table = select_max_f1_threshold(y, s)
    assert len(table) == 2
    # threshold 0.5 flags both 0.5-rows: TP1 FP1 FN1 -> 2/4 ; threshold 0.2 flags all: TP2 FP2 FN0 -> 4/6
    assert table["f1_numerator"].tolist() == [2, 4] and table["f1_denominator"].tolist() == [4, 6]
    assert res["threshold"] == 0.2


def test_constant_scores_give_single_candidate_that_flags_everything():
    y = np.array([1, 0, 0, 0])
    res, table = select_max_f1_threshold(y, np.full(4, 0.25))
    assert len(table) == 1 and res["threshold"] == 0.25
    assert review_flags(np.full(4, 0.25), res["threshold"]).all()
    assert res["validation_tp"] == 1 and res["validation_fp"] == 3 and res["validation_fn"] == 0


def test_selection_requires_positives():
    with pytest.raises(EvaluationPrerequisiteError):
        select_max_f1_threshold(np.zeros(4, dtype=int), np.arange(4) / 4)


# ---- metrics --------------------------------------------------------------------------------------
def test_confusion_and_metrics_hand_calculated():
    y = np.array([1, 1, 0, 0, 0, 1, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.3, 0.7, 0.05])
    m = evaluate_at_threshold(y, s, 0.7)  # flags rows 0,2,5 -> TP=2 (0,5) FP=1 (2) FN=1 (1) TN=3
    assert confusion_counts(y, review_flags(s, 0.7)) == {"tp": 2, "fp": 1, "tn": 3, "fn": 1}
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 3, 1)
    assert m["precision"] == pytest.approx(2 / 3) and m["precision_defined"] is True
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(4 / 6)
    assert m["fraction_flagged"] == pytest.approx(3 / 7)
    assert (m["n_rows"], m["n_fraud"], m["n_non_fraud"]) == (7, 3, 4)
    assert m["fraud_prevalence"] == pytest.approx(3 / 7)
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == m["n_rows"]


def test_precision_undefined_when_nothing_flagged_has_machine_readable_form():
    y = np.array([1, 0, 0, 1])
    m = evaluate_at_threshold(y, np.array([0.1, 0.2, 0.3, 0.4]), 0.99)
    assert m["precision"] is None
    assert m["precision_defined"] is False and m["precision_status"] == "undefined_no_rows_flagged"
    assert m["tp"] == 0 and m["fp"] == 0 and m["recall"] == 0.0 and m["f1"] == 0.0
    assert m["fraction_flagged"] == 0.0


def test_all_flagged_baseline_precision_equals_prevalence_and_ap_equals_prevalence():
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0])
    m = evaluate_at_threshold(y, np.full(8, 0.25), 0.25)
    assert m["precision"] == pytest.approx(0.25) and m["recall"] == 1.0
    assert m["fraction_flagged"] == 1.0
    assert m["average_precision"] == pytest.approx(0.25)


def test_average_precision_hand_value_and_uses_sklearn():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    m = evaluate_at_threshold(y, s, 0.5)
    assert m["average_precision"] == pytest.approx(5 / 6)  # (1*0.5 + 2/3*0.5)
    assert m["average_precision"] == average_precision_score(y, s)
    assert "average_precision_score" in m["average_precision_method"]


def test_partition_without_positives_fails_evaluation_prerequisite():
    with pytest.raises(EvaluationPrerequisiteError):
        evaluate_at_threshold(np.zeros(5, dtype=int), np.linspace(0, 1, 5), 0.5)


def test_bad_inputs_rejected():
    with pytest.raises(EvaluationPrerequisiteError):
        evaluate_at_threshold(np.array([1, 0]), np.array([0.5]), 0.5)
    with pytest.raises(EvaluationPrerequisiteError):
        evaluate_at_threshold(np.array([1, 0]), np.array([0.5, np.nan]), 0.5)
    with pytest.raises(EvaluationPrerequisiteError):
        evaluate_at_threshold(np.array([1, 2]), np.array([0.5, 0.1]), 0.5)
