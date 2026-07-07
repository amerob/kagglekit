"""Metric-optimal decision rules and threshold optimization.

Principle: train with a smooth, well-calibrated proxy loss (log-loss, MSE),
then apply the evaluation metric's Bayes-optimal decision rule at inference.
Decision-rule errors routinely dwarf model-quality differences.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def prior_corrected_argmax(probs: np.ndarray, priors: np.ndarray | None = None,
                           y: np.ndarray | None = None) -> np.ndarray:
    """Bayes-optimal decision rule for balanced accuracy / macro recall.

    argmax_k P(k|x) / P(k). Pass either class priors directly or labels `y`
    from which priors are computed. Worth several points over plain argmax
    under class imbalance; per-class weight tuning beyond this is ~ +0.0001.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if priors is None:
        if y is None:
            raise ValueError("provide priors or y")
        priors = np.bincount(np.asarray(y), minlength=probs.shape[1]) / len(y)
    priors = np.asarray(priors, dtype=np.float64)
    if (priors <= 0).any():
        raise ValueError("all class priors must be positive")
    return (probs / priors[None, :]).argmax(axis=1)


def optimize_binary_threshold(y_true: np.ndarray, probs: np.ndarray,
                              metric=f1_score, n_grid: int = 199,
                              n_bags: int = 15, bag_frac: float = 0.5,
                              seed: int = 42) -> tuple[float, float]:
    """Tune a single decision threshold on OOF predictions with bagging.

    Scores each candidate threshold as the mean metric over random
    half-subsamples so the choice does not overfit OOF noise.
    Returns (best_threshold, bagged_score_at_best).
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bags = [rng.choice(len(y_true), int(bag_frac * len(y_true)), replace=False)
            for _ in range(n_bags)]
    grid = np.linspace(0.005, 0.995, n_grid)
    best_t, best_s = 0.5, -np.inf
    for t in grid:
        pred = (probs >= t).astype(int)
        s = float(np.mean([metric(y_true[b], pred[b]) for b in bags]))
        if s > best_s:
            best_t, best_s = float(t), s
    return best_t, best_s


def optimize_multiclass_thresholds(y_true: np.ndarray, probs: np.ndarray,
                                   metric=None, n_rounds: int = 4,
                                   n_grid: int = 25, seed: int = 42
                                   ) -> tuple[np.ndarray, float]:
    """Per-class multiplicative weights via coordinate ascent on an OOF metric.

    Default metric: macro F1 via argmax of weighted probabilities. Use when the
    metric is threshold-sensitive (macro-F1, MCC). For balanced accuracy use
    ``prior_corrected_argmax`` instead; it is already optimal.
    Returns (weights, score).
    """
    from sklearn.metrics import f1_score as _f1

    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=np.float64)
    k = probs.shape[1]
    if metric is None:
        def metric(yt, yp):
            return _f1(yt, yp, average="macro")
    w = np.ones(k)
    best = metric(y_true, (probs * w).argmax(1))
    grid = np.geomspace(0.2, 5.0, n_grid)
    for _ in range(n_rounds):
        improved = False
        for j in range(k):
            for g in grid:
                w2 = w.copy()
                w2[j] = g
                s = metric(y_true, (probs * w2).argmax(1))
                if s > best + 1e-12:
                    best, w, improved = s, w2, True
        if not improved:
            break
    return w, float(best)


class OptimizedRounder:
    """Optimize integer bin edges for ordinal targets scored by QWK-like metrics.

    Fit on OOF regression predictions; apply to test predictions.
    """

    def __init__(self, n_classes: int, metric=None):
        self.n_classes = n_classes
        self.metric = metric
        self.edges_: np.ndarray | None = None

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray, n_grid: int = 60):
        from sklearn.metrics import cohen_kappa_score
        metric = self.metric or (lambda a, b: cohen_kappa_score(a, b, weights="quadratic"))
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        edges = np.quantile(y_pred, np.linspace(0, 1, self.n_classes + 1)[1:-1])
        best = metric(y_true, self._digitize(y_pred, edges))
        span = y_pred.std()
        for _ in range(3):
            for i in range(len(edges)):
                lo = edges[i - 1] if i > 0 else y_pred.min()
                hi = edges[i + 1] if i < len(edges) - 1 else y_pred.max()
                for c in np.linspace(lo + 1e-9, hi - 1e-9, n_grid):
                    cand = edges.copy()
                    cand[i] = c
                    s = metric(y_true, self._digitize(y_pred, cand))
                    if s > best:
                        best, edges = s, cand
            span *= 0.5
        self.edges_ = edges
        self.score_ = float(best)
        return self

    @staticmethod
    def _digitize(y_pred: np.ndarray, edges: np.ndarray) -> np.ndarray:
        return np.digitize(y_pred, edges)

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        if self.edges_ is None:
            raise RuntimeError("call fit first")
        return self._digitize(np.asarray(y_pred, dtype=np.float64), self.edges_)


def noise_floor(y_true: np.ndarray, y_score, metric, n_boot: int = 1000,
                seed: int = 42, is_proba: bool = True) -> float:
    """Bootstrap standard error of an OOF metric.

    Report every experiment as delta / noise_floor; treat deltas below
    1 to 2 floors as noise unless reproduced across seeds.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = metric(y_true[idx], y_score[idx])
    return float(vals.std(ddof=1))


def balanced_accuracy_from_probs(y_true: np.ndarray, probs: np.ndarray,
                                 priors: np.ndarray | None = None) -> float:
    """Convenience: balanced accuracy under the optimal decision rule."""
    pred = prior_corrected_argmax(probs, priors=priors, y=y_true if priors is None else None)
    return float(balanced_accuracy_score(np.asarray(y_true), pred))
