"""kagglekit quickstart: the fold-safe competition loop in ~60 lines.

Synthetic 3-class imbalanced problem standing in for a real competition.
Swap in your train/test frames and model of choice; the machinery is identical.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

from kagglekit import (ExperimentLog, FoldManager, TargetEncoder,
                       add_missingness_features, bagged_hill_climb, blend,
                       noise_floor, prior_corrected_argmax)

rng = np.random.default_rng(42)
n = 20_000
train = pd.DataFrame({
    "num1": rng.normal(0, 1, n),
    "num2": rng.normal(0, 1, n),
    "cat1": rng.choice(list("abcd"), n),
})
logit = 1.5 * train["num1"] + (train["cat1"] == "a") * 2.0
y = pd.cut(logit + rng.normal(0, 1, n), [-np.inf, 0.5, 2.5, np.inf],
           labels=[0, 1, 2]).astype(int).values
train.loc[rng.random(n) < 0.1, "num2"] = np.nan

# 1. Freeze folds for the whole campaign
fm = FoldManager.create(y=y, scheme="stratified", n_splits=5, seed=42)
fm.save("folds.json")

# 2. Fold-safe features + two models -> OOF probabilities
train = add_missingness_features(train, ["num1", "num2", "cat1"])
models = {"hgb_shallow": dict(max_depth=3), "hgb_deep": dict(max_depth=8)}
oof = {m: np.zeros((n, 3)) for m in models}
for tr_idx, vl_idx in fm.splits():
    X_tr, X_vl = train.iloc[tr_idx], train.iloc[vl_idx]
    enc = TargetEncoder(["cat1"], smooth=20, classes=(0, 2))
    X_tr, X_vl = enc.fit_transform_fold(X_tr, y[tr_idx], [X_tr, X_vl])
    feats = [c for c in X_tr.columns if c != "cat1"]
    for name, kw in models.items():
        mdl = HistGradientBoostingClassifier(random_state=0, **kw)
        mdl.fit(X_tr[feats], y[tr_idx])
        oof[name][vl_idx] = mdl.predict_proba(X_vl[feats])

# 3. Metric-optimal decision rule + noise floor
def metric(yt, probs):
    return balanced_accuracy_score(yt, prior_corrected_argmax(probs, y=yt))

floor = noise_floor(y, oof["hgb_deep"],
                    lambda a, b: metric(a, b), n_boot=300)
print(f"noise floor: +/-{floor:.4f}")
for name in models:
    print(f"{name}: bacc(argmax)={balanced_accuracy_score(y, oof[name].argmax(1)):.4f} "
          f"bacc(prior-corrected)={metric(y, oof[name]):.4f}")

# 4. Bagged ensemble + experiment log
stack = np.stack(list(oof.values()))
w, _ = bagged_hill_climb(stack, y, metric, n_rounds=30, n_bags=10)
ens = metric(y, blend(stack, w))
print(f"ensemble weights={w.round(3)} bacc={ens:.4f}")

log = ExperimentLog("experiments.csv", noise_floor=floor)
base_id = log.record("baseline", "hgb_deep alone", oof_score=metric(y, oof["hgb_deep"]))
log.record("ensemble", "bagged hill-climb 2 models", oof_score=ens, parent_id=base_id)
print(log.summary())
