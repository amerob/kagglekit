"""Leakage detection and exploitation utilities.

Two sides of the same coin: leaks into your CV destroy validity; leaks from
auxiliary data into test are free score. Both start with exact-match detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _row_keys(df: pd.DataFrame, cols: list[str], sep: str = "|") -> pd.Series:
    return df[cols].astype(str).fillna("NA").agg(sep.join, axis=1)


def duplicate_report(train: pd.DataFrame, feature_cols: list[str],
                     target_col: str | None = None) -> dict:
    """Exact-duplicate structure of the training set.

    ``conflicting`` counts duplicate feature rows with differing targets, a
    signal of label noise. Duplicates should be deduped or forced into the
    same fold to avoid CV inflation.
    """
    keys = _row_keys(train, feature_cols)
    dup_mask = keys.duplicated(keep=False)
    out = {"n_rows": len(train),
           "n_duplicate_rows": int(dup_mask.sum()),
           "n_duplicate_groups": int(keys[dup_mask].nunique())}
    if target_col is not None and dup_mask.any():
        conf = (train.loc[dup_mask].groupby(keys[dup_mask])[target_col]
                .nunique() > 1)
        out["conflicting_label_groups"] = int(conf.sum())
    return out


def train_test_overlap(train: pd.DataFrame, test: pd.DataFrame,
                       feature_cols: list[str]) -> dict:
    """How many test rows exactly match a train row on all features.

    High overlap on real data suggests contamination; on synthetic data it is
    an exploitable lookup (see ``exact_match_lookup``).
    """
    train_keys = set(_row_keys(train, feature_cols))
    test_keys = _row_keys(test, feature_cols)
    hits = test_keys.isin(train_keys)
    return {"n_test": len(test), "n_matches": int(hits.sum()),
            "match_rate": float(hits.mean()), "mask": hits.values}


def exact_match_lookup(source: pd.DataFrame, test: pd.DataFrame,
                       feature_cols: list[str], target_col: str,
                       id_col: str | None = None) -> dict:
    """Label lookup for test rows exactly matching a source dataset.

    Use with the ORIGINAL dataset behind a synthetic competition: if the
    generator memorized rows, the source label is near-certain truth. Only
    unambiguous source rows (a single label per key) are used. Returns a
    mapping from test id (or positional index) to the looked-up label;
    overwrite model predictions with it at submission time.
    """
    missing = [c for c in feature_cols + [target_col] if c not in source.columns]
    if missing:
        return {"usable": False, "reason": f"source missing columns {missing}", "map": {}}
    src_keys = _row_keys(source, feature_cols)
    counts = source.groupby(src_keys.values)[target_col].nunique()
    unambiguous = set(counts[counts == 1].index)
    first = source.groupby(src_keys.values)[target_col].first()
    lookup = {k: v for k, v in first.items() if k in unambiguous}
    test_keys = _row_keys(test, feature_cols)
    matched = test_keys.map(lookup).dropna()
    ids = (test.loc[matched.index, id_col] if id_col is not None
           else pd.Series(matched.index, index=matched.index))
    return {"usable": True, "n_matches": int(len(matched)),
            "match_rate": float(len(matched) / len(test)),
            "map": dict(zip(ids.values, matched.values))}


def shuffle_sanity_check(cv_score_fn, X: pd.DataFrame, y: np.ndarray,
                         seed: int = 42) -> dict:
    """Row-order leak probe.

    Runs the caller-provided ``cv_score_fn(X, y)`` on data as-is and on a
    consistently shuffled copy. A material score drop after shuffling means the
    pipeline exploits row order (an ID/ordering leak that will not exist in a
    shuffled or future test set). Expensive: run once at Phase 0 if row order
    is suspicious.
    """
    base = float(cv_score_fn(X, y))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    shuffled = float(cv_score_fn(X.iloc[perm].reset_index(drop=True), np.asarray(y)[perm]))
    return {"score_original_order": base, "score_shuffled": shuffled,
            "gap": base - shuffled}
