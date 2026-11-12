"""Command line entry points: data-check, split, train, evaluate.

    python -m fraudrisk data-check --data F --data-kind real|synthetic --out DIR [--data-manifest M]
    python -m fraudrisk split      (same args)
    python -m fraudrisk train      (same args)  # fits on train, selects thresholds on validation, freezes
    python -m fraudrisk evaluate   (same args)  # scores the test partition at the frozen thresholds
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import constants as C
from .analysis import earliest_errors
from .dataio import LoadedData, load_dataset, parse_fidelity_check
from .errors import (
    CrossPartitionOverlapError,
    DataValidationError,
    EvaluationPrerequisiteError,
    OverlapPolicyMismatchError,
    PipelineError,
)
from .modeling import ConstantScoreBaseline, fit_on_training_rows, model_scores, prepare_scoring_frame
from .policy import compare, enforce, load_policy, observed_findings
from .provenance import (
    canonical_json,
    code_hashes,
    dependency_versions,
    read_json,
    sha256_file,
    utc_now,
    write_csv_gz,
    write_json,
)
from .split import (
    assert_usable,
    cross_partition_audit,
    plan_chronological_split,
    summarize_split,
    unseen_test_mask,
)
from .thresholds import confusion_summary, evaluate_at_threshold, select_max_f1_threshold
from .validation import inspect_frame, labels_as_int, validate_frame

F_QUALITY = "data_quality_report.json"
F_SPLIT = "split_manifest.json"
F_MEMBERSHIP = "split_membership.csv.gz"
F_AUDIT = "leakage_audit.json"
F_OVERLAP_ROWS = "cross_partition_overlap_rows.csv.gz"
F_MODEL = "model_pipeline.joblib"
F_MODEL_CONFIG = "model_config.json"
F_VAL_SCORES = "validation_scores.csv.gz"
F_VAL_SELECTION = "threshold_selection_validation.json"
F_CAND_MODEL = "threshold_candidates_model.csv.gz"
F_CAND_BASELINE = "threshold_candidates_baseline.csv.gz"
F_FROZEN = "frozen_config.json"
F_TEST_METRICS = "test_metrics.json"
F_TEST_SCORES = "test_scores.csv.gz"
F_ERRORS = "error_analysis.json"
F_SEC_IDS = "secondary_subset_row_ids.csv.gz"
F_SEC_MANIFEST = "secondary_subset_manifest.json"


def _header(args, loaded: LoadedData) -> dict:
    h = {
        "data_kind": args.data_kind,
        "source_file": Path(loaded.source_path).name,
        "source_sha256": loaded.source_sha256,
        "source_bytes": loaded.source_bytes,
    }
    if args.data_kind == "synthetic":
        h["notice"] = C.SYNTHETIC_NOTICE
    return h


def _load(args) -> tuple[LoadedData, dict]:
    loaded = load_dataset(args.data)
    manifest_info = None
    if args.data_kind == "real":
        if not args.data_manifest:
            raise DataValidationError("--data-manifest is required for --data-kind real")
        m = read_json(args.data_manifest)
        if m["file"]["sha256"] != loaded.source_sha256:
            raise DataValidationError(
                f"data file SHA-256 {loaded.source_sha256} does not match manifest {m['file']['sha256']}"
            )
        manifest_info = {"manifest_path": str(args.data_manifest), "manifest_sha256": sha256_file(args.data_manifest)}
    return loaded, {"data_manifest": manifest_info}


def _validated(args, loaded: LoadedData) -> dict:
    return validate_frame(loaded.frame, expect_published=(args.data_kind == "real"))


def _split_from_loaded(loaded: LoadedData):
    y = labels_as_int(loaded.frame)
    time = loaded.frame[C.TIME_COLUMN].to_numpy(dtype="float64")
    plan = plan_chronological_split(time, np.arange(len(time)))
    return y, time, plan


# ---------------------------------------------------------------- data-check
def cmd_data_check(args) -> int:
    out = Path(args.out)
    loaded, extra = _load(args)
    report = inspect_frame(loaded.frame, expect_published=(args.data_kind == "real"))
    report["parse_fidelity_check"] = parse_fidelity_check(args.data, loaded.frame)
    pf = report["parse_fidelity_check"]
    if pf.get("performed") and pf["mismatching_rows"]:
        report["errors"].append(f"parse fidelity mismatches: {pf['mismatching_rows']}")
        report["status"] = "failed"
    doc = {**_header(args, loaded), **extra, "generated_at_utc": utc_now(), "report": report}
    write_json(out / F_QUALITY, doc)
    print(f"data-check status: {report['status']}  rows={report['n_rows']}")
    for k in ("class_counts", "missing_values", "nonfinite_values", "negative_amount_rows", "negative_time_rows"):
        print(f"  {k}: {report.get(k)}")
    if "duplicates" in report:
        for k, v in report["duplicates"].items():
            print(f"  duplicates[{k}]: {v}")
    print(f"  conflicting_label_groups: {report.get('conflicting_label_groups')}")
    print(f"  parse_fidelity_check: {pf}")
    if report["status"] == "ok":
        return C.EXIT_OK
    for e in report["errors"]:
        print(f"ERROR: {e}", file=sys.stderr)
    if report.get("conflicting_label_groups"):
        print("STOP FOR REVIEW: conflicting labels found.", file=sys.stderr)
        return C.EXIT_CONFLICTING_LABELS
    return C.EXIT_VALIDATION_FAILED


# --------------------------------------------------------------------- split
def _policy_arg(args):
    path = getattr(args, "overlap_policy", None)
    return load_policy(path) if path else (None, None)


def cmd_split(args) -> int:
    out = Path(args.out)
    loaded, extra = _load(args)
    _validated(args, loaded)
    y, time, plan = _split_from_loaded(loaded)
    summary = summarize_split(plan, time, y)
    doc = {**_header(args, loaded), **extra, "generated_at_utc": utc_now(), **summary,
           "entity_separation": "NOT ESTABLISHED: the schema has no customer/card identifiers."}
    write_json(out / F_SPLIT, doc)
    print(f"split: n={summary['n_rows']} usable={summary['usable']}")
    for name, p in summary["partitions"].items():
        print(f"  {name:10s} rows={p['n_rows']:7d} fraud={p['n_fraud']:5d} prevalence={p['fraud_prevalence']} "
              f"time=[{p['time_min']}, {p['time_max']}]")
    print(f"  boundaries nominal={plan.nominal_boundaries} adjusted={plan.boundaries}")
    assert_usable(summary)  # raises PartitionUnusableError -> exit 5 (manifest already written)

    membership = pd.DataFrame(
        {
            "row_id": loaded.row_ids,
            "row_number": np.arange(len(y)),
            "split": plan.assignment,
            "Time": time,
            "Class": y,
        }
    )
    mem_sha = write_csv_gz(membership, out / F_MEMBERSHIP)
    audit, overlap_rows = cross_partition_audit(loaded.frame, plan.assignment, y, loaded.row_ids)
    write_csv_gz(overlap_rows, out / F_OVERLAP_ROWS)

    # Secondary subset: frozen and saved NOW, before any fitting or scoring. Features only.
    unseen = unseen_test_mask(loaded.frame, plan.assignment)
    sec_ids = [r for r, u in zip(loaded.row_ids, unseen) if u]
    ids_sha = write_csv_gz(pd.DataFrame({"row_id": sec_ids}), out / F_SEC_IDS)
    n_test = int((plan.assignment == "test").sum())
    write_json(out / F_SEC_MANIFEST, {
        **_header(args, loaded), "generated_at_utc": utc_now(),
        "label": C.SECONDARY_LABEL,
        "selection_rule": ("test-partition rows whose exact V1..V28 + Amount float64 vector (Time excluded, "
                           "-0.0 normalised) equals no row of the training or validation partition"),
        "selection_uses": "input features only: no labels, no scores, no rounding",
        "n_test_rows": n_test, "n_rows": len(sec_ids), "n_excluded_test_rows": n_test - len(sec_ids),
        "n_fraud_reported_only": int(y[unseen].sum()),
        "ids_file": F_SEC_IDS, "ids_sha256": ids_sha,
        "membership_sha256": mem_sha, "data_sha256": loaded.source_sha256,
        "claims": ("Makes no claim of entity independence, duplicate removal or leakage removal; "
                   "it only excludes test rows whose exact feature vector was seen in training or validation."),
    })

    observed = observed_findings(data_sha256=loaded.source_sha256, membership_sha256=mem_sha, summary=summary,
                                 audit=audit, unseen_mask=unseen, y=y, assignment=plan.assignment)
    policy, policy_sha = _policy_arg(args)
    policy_block = None
    code = C.EXIT_OK
    if policy is not None:
        ok, rows = compare(policy, observed)
        policy_block = {"policy_id": policy["policy_id"], "policy_sha256": policy_sha, "matched": ok,
                        "comparisons": rows}
        if not ok:
            code = C.EXIT_POLICY_MISMATCH
    elif audit["review_required"]:
        code = C.EXIT_CROSS_PARTITION_OVERLAP

    write_json(out / F_AUDIT, {
        **_header(args, loaded), "generated_at_utc": utc_now(), "audit": audit,
        "overlap_policy": policy_block,
        "split_manifest_sha256": sha256_file(out / F_SPLIT), "membership_sha256": mem_sha,
        "secondary_subset_ids_sha256": ids_sha,
    })
    print(f"leakage audit: cross-partition repeated fingerprints = {audit['cross_partition_repeated_fingerprints']}")
    print(f"  test rows with fingerprint in train: {audit['test_rows_with_fingerprint_in_train']}")
    print(f"  validation rows with fingerprint in train: {audit['validation_rows_with_fingerprint_in_train']}")
    print(f"  {C.SECONDARY_LABEL}: {len(sec_ids)} of {n_test} test rows "
          f"({int(y[unseen].sum())} of {int(y[plan.assignment == 'test'].sum())} test positives)")
    if policy_block:
        print(f"  overlap policy {policy['policy_id']}: matched={policy_block['matched']}")
        for r in rows:
            if not r["match"]:
                print(f"    MISMATCH {r['key']}: expected {r['expected']!r} observed {r['observed']!r}", file=sys.stderr)
    if code == C.EXIT_CROSS_PARTITION_OVERLAP:
        print("STOP FOR REVIEW: repeated feature fingerprints cross partitions and no --overlap-policy was given. "
              "Nothing was dropped, re-split or fitted.", file=sys.stderr)
    elif code == C.EXIT_POLICY_MISMATCH:
        print("STOP: findings differ from the reviewed report; the overlap policy does not apply.", file=sys.stderr)
    return code


# --------------------------------------------------------------------- train
def _load_split_artifacts(out: Path, loaded: LoadedData, y, plan) -> tuple[dict, dict]:
    manifest = read_json(out / F_SPLIT)
    audit = read_json(out / F_AUDIT)
    if manifest["source_sha256"] != loaded.source_sha256:
        raise DataValidationError("split manifest was produced from a different data file")
    mem = pd.read_csv(out / F_MEMBERSHIP)
    if list(mem["row_id"]) != loaded.row_ids or not np.array_equal(mem["split"].to_numpy(dtype=object), plan.assignment):
        raise DataValidationError("saved split membership does not match the deterministic re-derivation")
    return manifest, audit


def _overlap_gate(args, out: Path, loaded: LoadedData, y, time, plan):
    """Recompute the audit and the secondary subset; stop unless they equal the saved ones and the policy."""
    saved = read_json(out / F_AUDIT)
    audit_now, _ = cross_partition_audit(loaded.frame, plan.assignment, y, loaded.row_ids)
    if canonical_json(audit_now) != canonical_json(saved["audit"]):
        raise OverlapPolicyMismatchError("recomputed leakage audit differs from the saved leakage_audit.json")
    mem_sha = sha256_file(out / F_MEMBERSHIP)
    if mem_sha != saved["membership_sha256"]:
        raise OverlapPolicyMismatchError("split_membership.csv.gz changed since the audit")
    unseen = unseen_test_mask(loaded.frame, plan.assignment)
    ids_now = [r for r, u in zip(loaded.row_ids, unseen) if u]
    sec = read_json(out / F_SEC_MANIFEST)
    ids_sha = sha256_file(out / F_SEC_IDS)
    if (pd.read_csv(out / F_SEC_IDS)["row_id"].tolist() != ids_now or ids_sha != sec["ids_sha256"]
            or sec["membership_sha256"] != mem_sha or sec["data_sha256"] != loaded.source_sha256):
        raise OverlapPolicyMismatchError("saved secondary subset differs from the recomputed one")
    summary = summarize_split(plan, time, y)
    observed = observed_findings(data_sha256=loaded.source_sha256, membership_sha256=mem_sha, summary=summary,
                                 audit=audit_now, unseen_mask=unseen, y=y, assignment=plan.assignment)
    policy, policy_sha = _policy_arg(args)
    if policy is None:
        if audit_now["review_required"]:
            raise CrossPartitionOverlapError(
                "cross-partition repeated fingerprints exist (see leakage_audit.json). Pass --overlap-policy "
                "pointing at the reviewed policy file; overlap detection stays active."
            )
        info = {"policy_id": None, "policy_sha256": None, "overlap_present": False, "matched": None}
    else:
        enforce(policy, observed)
        info = {"policy_id": policy["policy_id"], "policy_sha256": policy_sha,
                "overlap_present": bool(audit_now["review_required"]), "matched": True}
    sec_info = {"label": C.SECONDARY_LABEL, "ids_file": F_SEC_IDS, "ids_sha256": ids_sha,
                "manifest_sha256": sha256_file(out / F_SEC_MANIFEST), "n_rows": len(ids_now)}
    return info, sec_info, unseen, observed


def cmd_train(args) -> int:
    out = Path(args.out)
    loaded, extra = _load(args)
    _validated(args, loaded)
    y, time, plan = _split_from_loaded(loaded)
    manifest, audit = _load_split_artifacts(out, loaded, y, plan)
    if not manifest["usable"]:
        raise DataValidationError("split manifest says partitions are unusable")
    policy_info, sec_info, _, _ = _overlap_gate(args, out, loaded, y, time, plan)

    # Physically exclude test rows: nothing downstream can see them.
    keep = plan.assignment != "test"
    frame_tv = loaded.frame.loc[keep].reset_index(drop=True)
    y_tv = y[keep]
    assign_tv = plan.assignment[keep]
    ids_tv = [r for r, k in zip(loaded.row_ids, keep) if k]
    X_tv = prepare_scoring_frame(frame_tv[list(C.FEATURE_COLUMNS)])

    pipe, baseline, train_info = fit_on_training_rows(X_tv, y_tv, assign_tv)
    vmask = assign_tv == "validation"
    X_val, y_val = X_tv.loc[vmask], y_tv[vmask]
    ids_val = [r for r, m in zip(ids_tv, vmask) if m]
    s_model = model_scores(pipe, X_val)
    s_base = baseline.score(len(X_val))

    sel_model, cand_model = select_max_f1_threshold(y_val, s_model)
    sel_base, cand_base = select_max_f1_threshold(y_val, s_base)
    val_metrics_model = evaluate_at_threshold(y_val, s_model, sel_model["threshold"])
    val_metrics_base = evaluate_at_threshold(y_val, s_base, sel_base["threshold"])

    model_path = out / F_MODEL
    joblib.dump(pipe, model_path)
    # our own just-written artifact: reload only to prove the round trip is bitwise identical
    reloaded = joblib.load(model_path)
    if not np.array_equal(model_scores(reloaded, X_val), s_model):
        raise PipelineError("serialisation round trip changed validation scores")

    hdr = _header(args, loaded)
    write_csv_gz(
        pd.DataFrame({"row_id": ids_val, "Class": y_val, "model_score": s_model, "baseline_score": s_base}),
        out / F_VAL_SCORES,
    )
    write_csv_gz(cand_model, out / F_CAND_MODEL)
    write_csv_gz(cand_base, out / F_CAND_BASELINE)
    sel_doc = {
        **hdr, "generated_at_utc": utc_now(),
        "selection_partition": "validation (the full chronological validation partition; no test population used)",
        "model": sel_model, "baseline": sel_base,
        "validation_metrics_at_selected_thresholds": {"model": val_metrics_model, "baseline": val_metrics_base},
        "caveat": ("Validation metrics are computed on the partition used to pick the thresholds and are "
                   "optimistic for the model. F1 is a demonstration objective, not a business cost model "
                   "or a review-capacity constraint. Scores are model scores; calibration was not evaluated."),
        "baseline_note": ("The constant baseline has a single distinct score, so its only candidate threshold "
                          "is that score and it flags every row; 'no rows flagged' is not a candidate."),
        "candidate_tables": {"model": F_CAND_MODEL, "baseline": F_CAND_BASELINE},
    }
    write_json(out / F_VAL_SELECTION, sel_doc)

    clf = pipe.named_steps["classifier"]
    mc = {
        **hdr, "generated_at_utc": utc_now(),
        "feature_order": list(C.FEATURE_COLUMNS),
        "excluded_from_scoring": ["Class", "Time (partitioning only)", "row identifiers"],
        "model_config": C.MODEL_CONFIG,
        "sklearn_params": {k: str(v) for k, v in pipe.get_params(deep=True).items() if "__" in k},
        "dependency_versions": dependency_versions(),
        "code_sha256": code_hashes(),
        "training_provenance": {
            **train_info,
            "fit_partition": "train only (scaler and classifier)",
            "n_coef": int(clf.coef_.shape[1]),
            "split_manifest_sha256": sha256_file(out / F_SPLIT),
            "test_rows_excluded_from_fitting_and_threshold_selection": True,
            "test_rows_provenance_note": ("The full dataset, including test rows, is loaded in process memory for validation, partitioning and overlap audits. Test rows are removed from the frame before fitting, validation scoring and threshold selection; they are first scored in the evaluate command."),
            "overlap_policy": policy_info,
        },
        "unresolved_source_limitations": [
            "Upstream PCA fitting procedure is undocumented; training-only scaling does not resolve it.",
            "Label-confirmation process unknown.",
            "Label-availability delays unknown.",
            "No customer/card entity identifiers; entity separation cannot be established.",
            "Very short (two-day), historical observation window.",
        ],
    }
    write_json(out / F_MODEL_CONFIG, mc)

    frozen = {
        **hdr,
        "frozen_at_utc": utc_now(),
        "frozen_before_test_scoring": True,
        "data_sha256": loaded.source_sha256,
        "split_manifest_sha256": sha256_file(out / F_SPLIT),
        "split_membership_sha256": sha256_file(out / F_MEMBERSHIP),
        "leakage_audit_sha256": sha256_file(out / F_AUDIT),
        "overlap_policy": policy_info,
        "secondary_subset": sec_info,
        "model_file": F_MODEL,
        "model_sha256": sha256_file(model_path),
        "model_config_sha256": sha256_file(out / F_MODEL_CONFIG),
        "feature_order": list(C.FEATURE_COLUMNS),
        "decision_rule": C.THRESHOLD_RULE,
        "threshold_objective": C.THRESHOLD_OBJECTIVE,
        "threshold_selected_on": "full chronological validation partition only",
        "same_model_and_thresholds_for_all_test_populations": True,
        "refit_on_train_plus_validation": False,
        "thresholds": {
            "model": {"threshold": sel_model["threshold"], "threshold_hex": sel_model["threshold_hex"]},
            "baseline": {"threshold": sel_base["threshold"], "threshold_hex": sel_base["threshold_hex"],
                         "baseline_prevalence": baseline.prevalence},
        },
    }
    h = write_json(out / F_FROZEN, frozen)
    print(f"train: converged in {train_info['n_iter']} iterations (max_iter={C.MODEL_CONFIG['max_iter']})")
    print(f"  overlap policy: {policy_info}")
    print(f"  train rows={train_info['n_train_rows']} fraud={train_info['n_train_fraud']} "
          f"baseline prevalence score={baseline.prevalence!r}")
    print(f"  model    threshold={sel_model['threshold']!r} validation F1={sel_model['validation_f1']:.6f} "
          f"(candidates={sel_model['n_candidate_thresholds']})")
    print(f"  baseline threshold={sel_base['threshold']!r} validation F1={sel_base['validation_f1']:.6f}")
    print(f"frozen_config.json sha256={h}")
    return C.EXIT_OK


# ------------------------------------------------------------------ evaluate
INTERPRETATION = [
    "Primary result: all chronological test rows, unchanged partition, observed class distribution.",
    ("Secondary result (" + C.SECONDARY_LABEL + "): test rows whose exact V1..V28 + Amount vector appears in neither "
     "training nor validation. Same fitted model, same baseline score, same frozen thresholds; nothing was "
     "re-tuned or re-trained for it."),
    ("If the excluded rows are all non-fraud, every positive stays in the secondary subset and predictions are "
     "unchanged, so recall is identical by construction (verified below), and the secondary subset has a higher "
     "fraud prevalence."),
    ("Because only non-fraud rows are excluded, precision, F1 and average precision change partly through the change "
     "in prevalence. Do not read those differences as a pure measure of leakage."),
    ("The secondary subset makes no claim of entity independence, duplicate removal or leakage removal: the data has "
     "no entity identifiers, and near-identical vectors that are not byte-equal remain."),
    "Model scores are not calibrated (calibration was not evaluated). F1 thresholds are a demonstration objective.",
    "No business savings are claimed.",
]


def cmd_evaluate(args) -> int:
    out = Path(args.out)
    frozen = read_json(out / F_FROZEN)
    loaded, extra = _load(args)
    _validated(args, loaded)
    y, time, plan = _split_from_loaded(loaded)
    _load_split_artifacts(out, loaded, y, plan)
    policy_info, sec_info, unseen, observed = _overlap_gate(args, out, loaded, y, time, plan)

    checks = {
        "data_sha256": (frozen["data_sha256"], loaded.source_sha256),
        "split_manifest_sha256": (frozen["split_manifest_sha256"], sha256_file(out / F_SPLIT)),
        "split_membership_sha256": (frozen["split_membership_sha256"], sha256_file(out / F_MEMBERSHIP)),
        "leakage_audit_sha256": (frozen["leakage_audit_sha256"], sha256_file(out / F_AUDIT)),
        "model_sha256": (frozen["model_sha256"], sha256_file(out / frozen["model_file"])),
        "model_config_sha256": (frozen["model_config_sha256"], sha256_file(out / F_MODEL_CONFIG)),
        "secondary_ids_sha256": (frozen["secondary_subset"]["ids_sha256"], sec_info["ids_sha256"]),
        "secondary_manifest_sha256": (frozen["secondary_subset"]["manifest_sha256"], sec_info["manifest_sha256"]),
        "overlap_policy_sha256": (frozen["overlap_policy"]["policy_sha256"], policy_info["policy_sha256"]),
    }
    bad = {k: v for k, v in checks.items() if v[0] != v[1]}
    if bad:
        raise EvaluationPrerequisiteError(f"frozen artifacts changed since training: {sorted(bad)}")
    if list(frozen["feature_order"]) != list(C.FEATURE_COLUMNS):
        raise EvaluationPrerequisiteError("frozen feature order differs from code")

    tmask = plan.assignment == "test"
    if y[tmask].sum() == 0:
        raise EvaluationPrerequisiteError("test partition has no positive labels")
    frame_t = loaded.frame.loc[tmask].reset_index(drop=True)
    y_t = y[tmask]
    ids_t = [r for r, m in zip(loaded.row_ids, tmask) if m]
    rn_t = np.arange(len(y))[tmask]
    sec_t = unseen[tmask]

    # joblib/pickle executes code on load: only our own artifact is loaded, and only after its
    # SHA-256 was verified against frozen_config.json above.
    pipe = joblib.load(out / frozen["model_file"])
    s_model = model_scores(pipe, frame_t[list(C.FEATURE_COLUMNS)])
    base = ConstantScoreBaseline(prevalence=frozen["thresholds"]["baseline"]["baseline_prevalence"])
    s_base = base.score(len(frame_t))
    thr_m = float.fromhex(frozen["thresholds"]["model"]["threshold_hex"])
    thr_b = float.fromhex(frozen["thresholds"]["baseline"]["threshold_hex"])
    if thr_m != frozen["thresholds"]["model"]["threshold"] or thr_b != frozen["thresholds"]["baseline"]["threshold"]:
        raise EvaluationPrerequisiteError("frozen threshold decimal and hex forms disagree")

    scorers = {"model": (s_model, thr_m), "constant_score_baseline": (s_base, thr_b)}
    all_rows = np.ones(len(y_t), dtype=bool)

    def population(label: str, mask: np.ndarray) -> dict:
        d = {"label": label, "n_rows": int(mask.sum()), "row_id_first": None, "row_id_last": None}
        ids = [r for r, m in zip(ids_t, mask) if m]
        if ids:
            d["row_id_first"], d["row_id_last"] = ids[0], ids[-1]
        for name, (s, thr) in scorers.items():
            d[name] = evaluate_at_threshold(y_t[mask], s[mask], thr)
        return d

    primary = population(C.PRIMARY_LABEL, all_rows)
    secondary = population(C.SECONDARY_LABEL, sec_t)
    exc_mask = ~sec_t
    excluded = {"label": C.EXCLUDED_LABEL, "n_rows": int(exc_mask.sum())}
    for name, (s, thr) in scorers.items():
        excluded[name] = confusion_summary(y_t[exc_mask], s[exc_mask], thr)

    recon = {"checks": {}, "all_reconcile": True}
    for name in scorers:
        for k in ("n_rows", "n_fraud", "n_non_fraud", "tp", "fp", "tn", "fn", "n_flagged"):
            lhs = primary[name][k]
            rhs = secondary[name][k] + excluded[name][k]
            recon["checks"][f"{name}.{k}"] = {"primary": lhs, "secondary_plus_excluded": rhs, "equal": lhs == rhs}
            recon["all_reconcile"] &= lhs == rhs
    if not recon["all_reconcile"]:
        raise EvaluationPrerequisiteError("primary does not equal secondary + excluded; refusing to report")
    recon["recall_identical_primary_vs_secondary"] = {
        name: primary[name]["recall"] == secondary[name]["recall"] for name in scorers}
    recon["recall_identity_expected"] = excluded["model"]["n_fraud"] == 0
    recon["all_positive_rows_retained_in_secondary"] = secondary["model"]["n_fraud"] == primary["model"]["n_fraud"]
    recon["same_rows_for_both_scorers_within_each_population"] = True

    doc = {
        **_header(args, loaded), "evaluated_at_utc": utc_now(),
        "frozen_config_sha256": sha256_file(out / F_FROZEN),
        "frozen_at_utc": frozen["frozen_at_utc"],
        "overlap_policy": policy_info,
        "selection_rule": {"decision_rule": C.THRESHOLD_RULE, "objective": C.THRESHOLD_OBJECTIVE,
                           "selected_on": "full chronological validation partition only; no test population used"},
        "same_fitted_model_and_thresholds_in_every_population": True,
        "populations": {"primary": primary, "secondary_previously_unseen_feature_vector_subset": secondary,
                        "excluded_test_rows": excluded},
        "reconciliation": recon,
        "interpretation": INTERPRETATION,
        "caveats": [
            "Entity separation is not established (no entity identifiers).",
            "Unresolved: upstream PCA fitting, label confirmation, label delays, short historical window.",
        ],
    }
    write_json(out / F_TEST_METRICS, doc)
    write_csv_gz(
        pd.DataFrame({"row_id": ids_t, "Class": y_t, "model_score": s_model, "baseline_score": s_base,
                      "in_secondary_subset": sec_t}),
        out / F_TEST_SCORES,
    )
    err = {**_header(args, loaded)}
    for key, mask in (("primary", all_rows), ("secondary_previously_unseen_feature_vector_subset", sec_t)):
        sub = frame_t.loc[mask].reset_index(drop=True)
        ids = [r for r, m in zip(ids_t, mask) if m]
        err[key] = {name: earliest_errors(sub, ids, rn_t[mask], y_t[mask], s[mask], thr)
                    for name, (s, thr) in scorers.items()}
    write_json(out / F_ERRORS, err)
    for pname, pop in (("primary", primary), ("secondary", secondary)):
        for name in scorers:
            m = pop[name]
            print(f"{pname}/{name}: rows={m['n_rows']} fraud={m['n_fraud']} AP={m['average_precision']:.6f} "
                  f"precision={m['precision']} recall={m['recall']:.6f} F1={m['f1']:.6f} "
                  f"TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']} flagged={m['fraction_flagged']:.6f}")
    for name in scorers:
        e = excluded[name]
        print(f"excluded/{name}: rows={e['n_rows']} fraud={e['n_fraud']} TP={e['tp']} FP={e['fp']} "
              f"FN={e['fn']} TN={e['tn']} flagged={e['fraction_flagged']}")
    print(f"reconciliation: all_reconcile={recon['all_reconcile']} "
          f"recall_identical={recon['recall_identical_primary_vs_secondary']}")
    return C.EXIT_OK


# --------------------------------------------------------------------- serve
def cmd_serve(args) -> int:
    """Run the local scoring API on 127.0.0.1 only. Refuses to start if any artifact checksum mismatches."""
    import uvicorn

    from .api import create_app, load_bundle

    bundle = load_bundle(args.artifact_dir, args.expected_frozen_sha256)  # fail fast with exit code 9
    print(f"serving model {bundle.model_version} / {bundle.threshold_version} "
          f"(threshold {bundle.threshold!r}) on http://127.0.0.1:{args.port}  [local only]")
    uvicorn.run(create_app(args.artifact_dir, args.expected_frozen_sha256), host="127.0.0.1", port=args.port,
                log_level="info")
    return C.EXIT_OK


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fraudrisk", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in (("data-check", cmd_data_check), ("split", cmd_split), ("train", cmd_train),
                     ("evaluate", cmd_evaluate)):
        sp = sub.add_parser(name)
        sp.add_argument("--data", required=True, help="path to creditcard.arff (real) or a CSV (synthetic)")
        sp.add_argument("--data-kind", required=True, choices=["real", "synthetic"])
        sp.add_argument("--data-manifest", help="data/manifest.json (required for real data)")
        sp.add_argument("--out", required=True, help="output directory for artifacts")
        if name in ("split", "train", "evaluate"):
            sp.add_argument("--overlap-policy",
                            help="reviewed overlap policy JSON (policy/overlap_policy_real_v1.json). Required to "
                                 "proceed when repeated fingerprints cross partitions; findings must match it exactly")
        sp.set_defaults(func=fn)
    sv = sub.add_parser("serve", help="run the local scoring API (127.0.0.1 only)")
    sv.add_argument("--artifact-dir", required=True, help="directory holding frozen_config.json and the model")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--expected-frozen-sha256", help="optional pin of frozen_config.json's SHA-256")
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except PipelineError as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return e.exit_code


if __name__ == "__main__":
    sys.exit(main())
