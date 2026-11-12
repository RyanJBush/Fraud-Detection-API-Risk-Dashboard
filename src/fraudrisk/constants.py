"""Schema, feature allowlist and fixed rules. Everything here is explicit on purpose."""

TIME_COLUMN = "Time"
LABEL_COLUMN = "Class"
AMOUNT_COLUMN = "Amount"
PCA_COLUMNS = tuple(f"V{i}" for i in range(1, 29))

# Column order of the source file (ARFF attribute order).
SOURCE_COLUMNS = (TIME_COLUMN, *PCA_COLUMNS, AMOUNT_COLUMN, LABEL_COLUMN)

# Scoring feature allowlist, in the saved explicit order. Time is used only for
# partitioning. Class and row identifiers are never scoring inputs.
FEATURE_COLUMNS = (*PCA_COLUMNS, AMOUNT_COLUMN)

# Names that get a specific error message if they show up as scoring inputs.
# (Any column outside FEATURE_COLUMNS is rejected regardless of this list.)
KNOWN_FORBIDDEN_SCORING_FIELDS = frozenset(
    {"Class", "Time", "row_id", "row_number", "label", "is_fraud", "fraud", "split", "index"}
)

LABELS = (0, 1)

# Chronological split: boundaries at floor(0.60 N) and floor(0.80 N), computed
# in integer arithmetic ((60 * N) // 100) to avoid float rounding.
TRAIN_PERCENT = 60
VALIDATION_END_PERCENT = 80

SPLIT_NAMES = ("train", "validation", "test")

# Classifier configuration (fixed by the milestone spec).
MODEL_CONFIG = {
    "scaler": "StandardScaler()",
    "classifier": "LogisticRegression",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "class_weight": None,
}

THRESHOLD_RULE = "review_flag = model_score >= threshold"
THRESHOLD_OBJECTIVE = (
    "maximize validation F1 = 2TP / (2TP + FP + FN); candidates are every distinct "
    "validation score under the inclusive >= rule; F1 ties are broken by choosing the "
    "highest threshold"
)

# Published expectations for OpenML 1597 v1 (to be checked, not assumed).
PUBLISHED_EXPECTATIONS = {
    "n_rows": 284807,
    "n_fraud": 492,
    "n_non_fraud": 284315,
    "fraud_prevalence_percent_approx": 0.17275,
}

SYNTHETIC_NOTICE = (
    "SYNTHETIC FIXTURE: this output exercises the pipeline code only. It is not "
    "evidence of real-world fraud-detection performance."
)

# Process exit codes (documented in README).
EXIT_OK = 0
EXIT_VALIDATION_FAILED = 2
EXIT_CONFLICTING_LABELS = 3
EXIT_CROSS_PARTITION_OVERLAP = 4
EXIT_PARTITIONS_UNUSABLE = 5
EXIT_NOT_CONVERGED = 6
EXIT_EVALUATION_PREREQUISITE = 7
EXIT_POLICY_MISMATCH = 8
EXIT_ARTIFACT_MISMATCH = 9

# Population labels used in every evaluation artifact (wording fixed by the reviewer).
PRIMARY_LABEL = "all chronological test rows"
SECONDARY_LABEL = "previously unseen feature-vector subset"
EXCLUDED_LABEL = "excluded test rows (feature vector also appears in training or validation)"
