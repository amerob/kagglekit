"""Fold-safe encoders.

The classic silent leak in competition pipelines is fitting an encoder on the
full training set. Everything here is designed to be fitted on the train side
of a fold and applied to validation/test frames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class TargetEncoder:
    """Smoothed multiclass/binary/regression target encoder, fold-safe by usage.

    Fit on the TRAIN FOLD ONLY, then transform any frame. For K-class problems
    encodes P(class = k | value) for the requested classes (skip the majority
    class: rows sum to ~1, it is redundant). For regression encodes the
    smoothed conditional mean.

    Parameters
    ----------
    cols : columns to encode
    smooth : additive smoothing strength toward the global prior/mean
    classes : class ids to encode (classification); None = regression mode
    """

    def __init__(self, cols: list[str], smooth: float = 20.0,
                 classes: tuple[int, ...] | None = None):
        self.cols = list(cols)
        self.smooth = float(smooth)
        self.classes = tuple(classes) if classes is not None else None
        self.maps_: dict = {}
        self.priors_: dict = {}

    def fit(self, X: pd.DataFrame, y) -> "TargetEncoder":
        y = np.asarray(y)
        self.maps_.clear()
        self.priors_.clear()
        for col in self.cols:
            vals = X[col].astype(object).values
            if self.classes is None:
                prior = float(y.mean())
                df = pd.DataFrame({"v": vals, "y": y.astype(float)})
                agg = df.groupby("v", dropna=False)["y"].agg(["sum", "size"])
                enc = (agg["sum"] + self.smooth * prior) / (agg["size"] + self.smooth)
                self.maps_[col] = {None: enc.to_dict()}
                self.priors_[col] = {None: prior}
            else:
                self.maps_[col] = {}
                self.priors_[col] = {}
                df = pd.DataFrame({"v": vals})
                for k in self.classes:
                    df[f"i{k}"] = (y == k).astype(float)
                agg = df.groupby("v", dropna=False).agg(
                    size=("v", "size"), **{f"s{k}": (f"i{k}", "sum") for k in self.classes})
                for k in self.classes:
                    prior = float((y == k).mean())
                    enc = (agg[f"s{k}"] + self.smooth * prior) / (agg["size"] + self.smooth)
                    self.maps_[col][k] = enc.to_dict()
                    self.priors_[col][k] = prior
        return self

    def transform(self, X: pd.DataFrame, prefix: str = "te") -> pd.DataFrame:
        if not self.maps_:
            raise RuntimeError("call fit first")
        out = X.copy()
        for col in self.cols:
            vals = out[col].astype(object)
            for k, mapping in self.maps_[col].items():
                name = f"{prefix}_{col}" if k is None else f"{prefix}{k}_{col}"
                out[name] = vals.map(mapping).fillna(self.priors_[col][k]).astype("float32")
        return out

    def fit_transform_fold(self, X_tr: pd.DataFrame, y_tr,
                           frames: list[pd.DataFrame], prefix: str = "te"
                           ) -> list[pd.DataFrame]:
        """One-call fold pattern: fit on (X_tr, y_tr), transform every frame."""
        self.fit(X_tr, y_tr)
        return [self.transform(f, prefix=prefix) for f in frames]


def frequency_encode(train: pd.DataFrame, frames: list[pd.DataFrame],
                     cols: list[str], normalize: bool = True,
                     prefix: str = "freq") -> list[pd.DataFrame]:
    """Count/frequency encoding fitted on train, applied to all frames.

    Frequency is a feature-side statistic; fitting on the train fold is
    conservative and always safe.
    """
    out = [f.copy() for f in frames]
    n = len(train)
    for c in cols:
        vc = train[c].value_counts(dropna=False)
        if normalize:
            vc = vc / n
        mapping = vc.to_dict()
        default = 0.0 if normalize else 0
        for f in out:
            f[f"{prefix}_{c}"] = f[c].map(mapping).fillna(default).astype("float32")
    return out


def add_missingness_features(df: pd.DataFrame, cols: list[str],
                             add_indicators: bool = True) -> pd.DataFrame:
    """Per-row missing count and optional per-column indicators.

    Synthetic-data generators often leak signal through missingness patterns;
    these features are nearly free and frequently pay.
    """
    out = df.copy()
    out["missing_count"] = out[cols].isna().sum(axis=1).astype("int16")
    if add_indicators:
        for c in cols:
            out[f"na_{c}"] = out[c].isna().astype("int8")
    return out


def add_categorical_crosses(df: pd.DataFrame, crosses: list[tuple[str, ...]],
                            sep: str = "|", na_token: str = "NA",
                            prefix: str = "x") -> tuple[pd.DataFrame, list[str]]:
    """String-concatenation crosses of categorical columns.

    Consume the result BOTH as native categoricals (GBDTs) and through
    ``TargetEncoder``. Returns (frame, new_column_names).
    """
    out = df.copy()
    names = []
    for cols in crosses:
        name = prefix + "_" + "_".join(c.split("_")[0] for c in cols)
        out[name] = (out[list(cols)].astype("object").fillna(na_token)
                     .astype(str).agg(sep.join, axis=1))
        names.append(name)
    return out, names
