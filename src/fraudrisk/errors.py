from . import constants as C


class PipelineError(Exception):
    exit_code = 1


class DataValidationError(PipelineError):
    exit_code = C.EXIT_VALIDATION_FAILED


class ScoringNumericalError(DataValidationError):
    """Inputs were individually finite but standardisation or the linear score overflowed to inf/NaN."""


class ConflictingLabelsError(PipelineError):
    exit_code = C.EXIT_CONFLICTING_LABELS


class CrossPartitionOverlapError(PipelineError):
    exit_code = C.EXIT_CROSS_PARTITION_OVERLAP


class OverlapPolicyMismatchError(PipelineError):
    exit_code = C.EXIT_POLICY_MISMATCH


class ArtifactIntegrityError(PipelineError):
    exit_code = C.EXIT_ARTIFACT_MISMATCH


class PartitionUnusableError(PipelineError):
    exit_code = C.EXIT_PARTITIONS_UNUSABLE


class TrainingNotConvergedError(PipelineError):
    exit_code = C.EXIT_NOT_CONVERGED


class EvaluationPrerequisiteError(PipelineError):
    exit_code = C.EXIT_EVALUATION_PREREQUISITE
