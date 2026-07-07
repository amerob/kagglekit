"""Fold management and validation diagnostics.

The CV scheme is the competition strategy: it must mirror the test split,
be frozen on day one, and be shared by every model so OOF matrices are
directly comparable and blendable.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd


class FoldManager:
    """Create, persist, and reload fold indices so an entire campaign shares them.

    Usage::

        fm = FoldManager.create(y=y, scheme="stratified", n_splits=5, seed=42)
        fm.save("folds.json")
        fm = FoldManager.load("folds.json")
        for tr_idx, vl_idx in fm.splits():
            ...
    """

    def __init__(self, fold_of: np.ndarray, n_splits: int, meta: dict):
        self.fold_of = np.asarray(fold_of, dtype=int)
        self.n_splits = int(n_splits)
        self.meta = meta

    # ---------------- construction ----------------
    @classmethod
    def create(cls, y=None, groups=None, time_values=None,
               scheme: str = "stratified", n_splits: int = 5, seed: int = 42,
               n_rows: int | None = None) -> "FoldManager":
        from sklearn.model_selection import (GroupKFold, KFold, StratifiedKFold,
                                             StratifiedGroupKFold)

        if n_rows is None:
            for arr in (y, groups, time_values):
                if arr is not None:
                    n_rows = len(arr)
                    break
        if n_rows is None:
            raise ValueError("cannot infer number of rows")
        fold_of = np.full(n_rows, -1, dtype=int)

        if scheme == "stratified":
            splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
            it = splitter.split(np.zeros(n_rows), y)
        elif scheme == "kfold":
            splitter = KFold(n_splits, shuffle=True, random_state=seed)
            it = splitter.split(np.zeros(n_rows))
        elif scheme == "group":
            it = GroupKFold(n_splits).split(np.zeros(n_rows), y, groups)
        elif scheme == "stratified_group":
            splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
            it = splitter.split(np.zeros(n_rows), y, groups)
        elif scheme == "time":
            if time_values is None:
                raise ValueError("time scheme requires time_values")
            order = np.argsort(np.asarray(time_values), kind="stable")
            chunks = np.array_split(order, n_splits)
            for f, chunk in enumerate(chunks):
                fold_of[chunk] = f
            return cls(fold_of, n_splits,
                       {"scheme": scheme, "n_splits": n_splits, "seed": seed,
                        "note": "fold f validates on time-chunk f; train on chunks < f"})
        else:
            raise ValueError(f"unknown scheme {scheme!r}")

        for f, (_, vl) in enumerate(it):
            fold_of[vl] = f
        return cls(fold_of, n_splits, {"scheme": scheme, "n_splits": n_splits, "seed": seed})

    # ---------------- iteration ----------------
    def splits(self):
        """Yield (train_idx, val_idx). Time scheme yields expanding windows."""
        idx = np.arange(len(self.fold_of))
        if self.meta.get("scheme") == "time":
            for f in range(1, self.n_splits):
                yield idx[self.fold_of < f], idx[self.fold_of == f]
        else:
            for f in range(self.n_splits):
                yield idx[self.fold_of != f], idx[self.fold_of == f]

    # ---------------- persistence ----------------
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"fold_of": self.fold_of.tolist(),
                       "n_splits": self.n_splits, "meta": self.meta}, f)

    @classmethod
    def load(cls, path: str) -> "FoldManager":
        with open(path) as f:
            d = json.load(f)
        return cls(np.array(d["fold_of"]), d["n_splits"], d["meta"])


def adversarial_validation(train_df: pd.DataFrame, test_df: pd.DataFrame,
                           features: list[str], n_estimators: int = 200,
                           seed: int = 42) -> dict:
    """Train-vs-test classifier to detect distribution shift.

    AUC ~ 0.5: no shift, random CV is trustworthy.
    AUC >> 0.6: inspect ``importances`` and drop/transform the top features,
    or reweight validation toward test-like rows.
    Returns {"auc": float, "importances": pd.Series (descending)}.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    X = pd.concat([train_df[features], test_df[features]], ignore_index=True)
    y = np.r_[np.zeros(len(train_df)), np.ones(len(test_df))]
    # ordinal-encode object/category columns for portability
    Xn = X.copy()
    for c in Xn.columns:
        if Xn[c].dtype == object or str(Xn[c].dtype) == "category":
            Xn[c] = Xn[c].astype("category").cat.codes
    clf = HistGradientBoostingClassifier(max_iter=n_estimators, random_state=seed)
    probs = cross_val_predict(clf, Xn, y, cv=3, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(y, probs))
    clf.fit(Xn, y)
    try:
        from sklearn.inspection import permutation_importance
        sub = np.random.default_rng(seed).choice(len(Xn), min(20000, len(Xn)), replace=False)
        imp = permutation_importance(clf, Xn.iloc[sub], y[sub], n_repeats=3,
                                     random_state=seed)
        importances = pd.Series(imp.importances_mean, index=features).sort_values(ascending=False)
    except Exception:  # pragma: no cover - fallback path
        importances = pd.Series(dtype=float)
    return {"auc": auc, "importances": importances}


def shared_category_dtypes(frames: list[pd.DataFrame], columns: list[str]
                           ) -> dict[str, pd.CategoricalDtype]:
    """One CategoricalDtype per column built from the union of all frames.

    Prevents XGBoost 'category not in training set' errors when a rare level
    appears in validation/test but not a given train fold. Feature values only,
    hence no target leakage.
    """
    out = {}
    for c in columns:
        cats = sorted(set().union(*[set(f[c].dropna().unique()) for f in frames]))
        out[c] = pd.CategoricalDtype(categories=cats)
    return out
