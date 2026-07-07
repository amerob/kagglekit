"""Ensembling: bagged greedy weight search and blending utilities.

The most common silent endgame mistake is fitting ensemble weights on a single
OOF evaluation, which optimizes metric noise. Everything here scores candidate
blends as the mean metric over random subsamples of the OOF.
"""
from __future__ import annotations

import numpy as np


def bagged_hill_climb(oof_preds: np.ndarray, y: np.ndarray, metric,
                      greater_is_better: bool = True, n_rounds: int = 60,
                      n_bags: int = 15, bag_frac: float = 0.5,
                      patience: int = 10, seed: int = 42
                      ) -> tuple[np.ndarray, list[float]]:
    """Caruana greedy forward ensemble with bootstrap-bagged scoring.

    Parameters
    ----------
    oof_preds : array (n_models, n_samples, ...) of OOF predictions in the
        space the metric consumes (probabilities, values, or ranks).
    metric : callable(y_true, y_pred_blend) -> float, applied per bag.
    Returns (weights summing to 1, greedy score history). The returned weights
    are the composition of the BEST blend, not the last iterate.
    """
    oof_preds = np.asarray(oof_preds, dtype=np.float64)
    y = np.asarray(y)
    n_models, n_samples = oof_preds.shape[0], oof_preds.shape[1]
    rng = np.random.default_rng(seed)
    sign = 1.0 if greater_is_better else -1.0
    bags = [rng.choice(n_samples, max(1, int(bag_frac * n_samples)), replace=False)
            for _ in range(n_bags)]

    def score(blend):
        return sign * float(np.mean([metric(y[b], blend[b]) for b in bags]))

    counts = np.zeros(n_models, dtype=int)
    blend_sum = np.zeros_like(oof_preds[0])
    history: list[float] = []
    best_counts, best_score = None, -np.inf
    for r in range(n_rounds):
        round_best_j, round_best_s = -1, -np.inf
        for j in range(n_models):
            cand = (blend_sum + oof_preds[j]) / (counts.sum() + 1)
            s = score(cand)
            if s > round_best_s:
                round_best_s, round_best_j = s, j
        counts[round_best_j] += 1
        blend_sum += oof_preds[round_best_j]
        history.append(sign * round_best_s)
        if round_best_s > best_score:
            best_score = round_best_s
            best_counts = counts.copy()
        if r + 1 >= patience and len(history) > patience:
            recent = history[-patience:] if greater_is_better else [-h for h in history[-patience:]]
            earlier = history[:-patience] if greater_is_better else [-h for h in history[:-patience]]
            if max(recent) <= max(earlier) + 1e-9:
                break
    counts = best_counts if best_counts is not None else counts
    return counts / counts.sum(), history


def blend(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted blend along the model axis: (n_models, n_samples, ...) -> (n_samples, ...)."""
    preds = np.asarray(preds, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return np.tensordot(weights, preds, axes=(0, 0))


def rank_average(preds: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Rank-transform each model's predictions then average.

    The correct blending space for AUC and other pure-ranking metrics, and a
    robust choice when models have very different calibration profiles.
    Accepts (n_models, n_samples).
    """
    from scipy.stats import rankdata

    preds = np.asarray(preds, dtype=np.float64)
    ranks = np.stack([rankdata(p) / len(p) for p in preds])
    if weights is None:
        weights = np.full(len(preds), 1.0 / len(preds))
    return blend(ranks, np.asarray(weights))


def equal_weight_baseline(oof_preds: np.ndarray, y: np.ndarray, metric) -> float:
    """Score of the uniform blend. Reject learned weights that do not beat this
    by more than the metric noise floor: equal weights are a strong prior."""
    uniform = np.full(len(oof_preds), 1.0 / len(oof_preds))
    return float(metric(np.asarray(y), blend(oof_preds, uniform)))


def diversity_matrix(oof_preds: np.ndarray) -> np.ndarray:
    """Pairwise Pearson correlation between flattened model predictions.

    Pairs above ~0.98 add little; prefer adding a different architecture
    family over another seed of the same one.
    """
    flat = np.asarray(oof_preds, dtype=np.float64).reshape(len(oof_preds), -1)
    return np.corrcoef(flat)
