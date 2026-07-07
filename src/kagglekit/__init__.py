"""kagglekit: battle-tested machinery for ML competitions.

Companion package to the kaggle-competitor AI skill (see docs/SKILL.md).
The skill supplies strategy; this package supplies the recurring mechanics:
fold management, fold-safe encoding, metric-optimal decision rules, bagged
ensembling, leakage checks, and experiment tracking.
"""
from .encoding import (TargetEncoder, add_categorical_crosses,
                       add_missingness_features, frequency_encode)
from .ensemble import (bagged_hill_climb, blend, diversity_matrix,
                       equal_weight_baseline, rank_average)
from .leakage import (duplicate_report, exact_match_lookup,
                      shuffle_sanity_check, train_test_overlap)
from .metrics import (OptimizedRounder, balanced_accuracy_from_probs,
                      noise_floor, optimize_binary_threshold,
                      optimize_multiclass_thresholds, prior_corrected_argmax)
from .tracking import ExperimentLog, SubmissionLog
from .validation import FoldManager, adversarial_validation, shared_category_dtypes

__version__ = "0.1.0"
__all__ = [
    "TargetEncoder", "add_categorical_crosses", "add_missingness_features",
    "frequency_encode", "bagged_hill_climb", "blend", "diversity_matrix",
    "equal_weight_baseline", "rank_average", "duplicate_report",
    "exact_match_lookup", "shuffle_sanity_check", "train_test_overlap",
    "OptimizedRounder", "balanced_accuracy_from_probs", "noise_floor",
    "optimize_binary_threshold", "optimize_multiclass_thresholds",
    "prior_corrected_argmax", "ExperimentLog", "SubmissionLog",
    "FoldManager", "adversarial_validation", "shared_category_dtypes",
]
