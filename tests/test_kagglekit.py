"""Tests for kagglekit. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from kagglekit import (ExperimentLog, FoldManager, OptimizedRounder,
                       SubmissionLog, TargetEncoder, add_categorical_crosses,
                       add_missingness_features, adversarial_validation,
                       bagged_hill_climb, blend, diversity_matrix,
                       duplicate_report, equal_weight_baseline,
                       exact_match_lookup, frequency_encode, noise_floor,
                       optimize_binary_threshold, prior_corrected_argmax,
                       shared_category_dtypes, train_test_overlap)

RNG = np.random.default_rng(0)


def make_imbalanced_probs(n=6000, priors=(0.85, 0.09, 0.06), sharp=3.0):
    y = RNG.choice(3, size=n, p=priors)
    probs = RNG.dirichlet(np.ones(3), size=n)
    probs[np.arange(n), y] += sharp
    probs /= probs.sum(1, keepdims=True)
    # bias probabilities toward the majority prior, as calibrated models do
    probs = probs * np.array(priors)[None, :]
    probs /= probs.sum(1, keepdims=True)
    return y, probs


# ---------------- metrics ----------------
class TestMetrics:
    def test_prior_correction_beats_argmax_under_imbalance(self):
        y, probs = make_imbalanced_probs()
        bacc_argmax = balanced_accuracy_score(y, probs.argmax(1))
        bacc_prior = balanced_accuracy_score(y, prior_corrected_argmax(probs, y=y))
        assert bacc_prior > bacc_argmax + 0.02

    def test_prior_correction_validates_inputs(self):
        with pytest.raises(ValueError):
            prior_corrected_argmax(np.ones((4, 3)) / 3)
        with pytest.raises(ValueError):
            prior_corrected_argmax(np.ones((4, 3)) / 3, priors=np.array([0.5, 0.5, 0.0]))

    def test_binary_threshold_recovers_shifted_optimum(self):
        n = 5000
        y = (RNG.random(n) < 0.15).astype(int)
        probs = np.clip(0.25 * y + 0.12 + RNG.normal(0, 0.05, n), 0, 1)
        t, s = optimize_binary_threshold(y, probs, seed=1)
        assert 0.05 < t < 0.5
        from sklearn.metrics import f1_score
        assert f1_score(y, probs >= t) >= f1_score(y, probs >= 0.5)

    def test_optimized_rounder_improves_over_naive(self):
        n = 3000
        y = RNG.integers(0, 4, n)
        pred = y + RNG.normal(0.4, 0.5, n)  # biased regression output
        from sklearn.metrics import cohen_kappa_score
        naive = cohen_kappa_score(y, np.clip(np.round(pred), 0, 3).astype(int),
                                  weights="quadratic")
        r = OptimizedRounder(4).fit(y, pred)
        assert r.score_ >= naive

    def test_noise_floor_positive_and_shrinks_with_n(self):
        def acc(a, b):
            return float((a == (b > 0.5)).mean())
        y = (RNG.random(2000) < 0.5).astype(int)
        p = np.clip(0.3 * y + 0.35 + RNG.normal(0, 0.25, 2000), 0, 1)  # imperfect
        small = noise_floor(y[:200], p[:200], acc, n_boot=300)
        large = noise_floor(y, p, acc, n_boot=300)
        assert small > large > 0


# ---------------- validation ----------------
class TestValidation:
    def test_foldmanager_stratified_roundtrip(self, tmp_path):
        y = RNG.choice(3, 1000, p=[0.8, 0.12, 0.08])
        fm = FoldManager.create(y=y, scheme="stratified", n_splits=5)
        assert set(fm.fold_of) == set(range(5))
        p = tmp_path / "folds.json"
        fm.save(str(p))
        fm2 = FoldManager.load(str(p))
        assert np.array_equal(fm.fold_of, fm2.fold_of)
        seen = np.zeros(1000, dtype=int)
        for tr, vl in fm2.splits():
            assert len(np.intersect1d(tr, vl)) == 0
            seen[vl] += 1
        assert (seen == 1).all()

    def test_group_folds_never_split_a_group(self):
        groups = RNG.integers(0, 50, 500)
        fm = FoldManager.create(y=np.zeros(500), groups=groups, scheme="group",
                                n_splits=5)
        for g in np.unique(groups):
            assert len(np.unique(fm.fold_of[groups == g])) == 1

    def test_time_folds_validate_strictly_forward(self):
        t = np.arange(300)
        fm = FoldManager.create(time_values=t, scheme="time", n_splits=5)
        for tr, vl in fm.splits():
            assert t[tr].max() < t[vl].min()

    def test_adversarial_validation_detects_planted_shift(self):
        n = 1500
        tr = pd.DataFrame({"a": RNG.normal(0, 1, n), "b": RNG.normal(0, 1, n)})
        te = pd.DataFrame({"a": RNG.normal(2.0, 1, n), "b": RNG.normal(0, 1, n)})
        res = adversarial_validation(tr, te, ["a", "b"], n_estimators=60)
        assert res["auc"] > 0.85
        same = adversarial_validation(tr, tr.copy(), ["a", "b"], n_estimators=60)
        assert same["auc"] < 0.6

    def test_shared_category_dtypes_union(self):
        f1 = pd.DataFrame({"c": ["a", "b"]})
        f2 = pd.DataFrame({"c": ["b", "z"]})
        d = shared_category_dtypes([f1, f2], ["c"])
        assert list(d["c"].categories) == ["a", "b", "z"]
        assert not f2["c"].astype(d["c"]).isna().any()


# ---------------- encoding ----------------
class TestEncoding:
    def test_target_encoder_fold_safety_and_smoothing(self):
        n = 2000
        X = pd.DataFrame({"c": RNG.choice(list("abc"), n)})
        y = (X["c"] == "a").astype(int).values
        enc = TargetEncoder(["c"], smooth=10, classes=(1,))
        Xt, = enc.fit_transform_fold(X, y, [X])
        m = Xt.groupby("c")["te1_c"].first()
        assert m["a"] > 0.9 and m["b"] < 0.1
        unseen = enc.transform(pd.DataFrame({"c": ["zzz"]}))
        prior = y.mean()
        assert abs(unseen["te1_c"].iloc[0] - prior) < 1e-6

    def test_target_encoder_regression_mode(self):
        X = pd.DataFrame({"c": ["a"] * 50 + ["b"] * 50})
        y = np.r_[np.full(50, 10.0), np.full(50, 20.0)]
        Xt = TargetEncoder(["c"], smooth=0.0).fit(X, y).transform(X)
        assert abs(Xt.loc[0, "te_c"] - 10) < 1e-6
        assert abs(Xt.loc[99, "te_c"] - 20) < 1e-6

    def test_frequency_and_missingness_and_crosses(self):
        df = pd.DataFrame({"c": ["a", "a", "b", None], "d": ["x", None, "x", "x"],
                           "n": [1.0, np.nan, 3.0, 4.0]})
        out, = frequency_encode(df, [df], ["c"])
        assert out["freq_c"].iloc[0] == 0.5
        out2 = add_missingness_features(df, ["c", "d", "n"])
        assert out2["missing_count"].tolist() == [0, 2, 0, 1]
        out3, names = add_categorical_crosses(df, [("c", "d")])
        assert names == ["x_c_d"] and out3["x_c_d"].iloc[3] == "NA|x"


# ---------------- ensemble ----------------
class TestEnsemble:
    def test_hill_climb_prefers_better_model_and_returns_best(self):
        y, probs = make_imbalanced_probs(4000)
        noise = RNG.dirichlet(np.ones(3), size=4000)
        good, bad = probs, 0.5 * probs + 0.5 * noise
        stack = np.stack([bad, good])

        def metric(yt, blend_probs):
            return balanced_accuracy_score(yt, prior_corrected_argmax(blend_probs, y=yt))
        w, hist = bagged_hill_climb(stack, y, metric, n_rounds=25, n_bags=6, seed=1)
        assert w[1] > w[0]
        assert abs(w.sum() - 1) < 1e-9
        blended = blend(stack, w)
        assert metric(y, blended) >= metric(y, bad) - 1e-9

    def test_equal_weight_baseline_and_diversity(self):
        y, probs = make_imbalanced_probs(2000)
        stack = np.stack([probs, probs])

        def metric(yt, bp):
            return balanced_accuracy_score(yt, prior_corrected_argmax(bp, y=yt))
        base = equal_weight_baseline(stack, y, metric)
        assert base == pytest.approx(metric(y, probs))
        corr = diversity_matrix(stack)
        assert corr[0, 1] == pytest.approx(1.0)


# ---------------- leakage ----------------
class TestLeakage:
    def test_duplicate_report_flags_conflicts(self):
        df = pd.DataFrame({"a": [1, 1, 2, 3], "b": ["x", "x", "y", "z"],
                           "t": [0, 1, 0, 0]})
        rep = duplicate_report(df, ["a", "b"], target_col="t")
        assert rep["n_duplicate_rows"] == 2
        assert rep["conflicting_label_groups"] == 1

    def test_overlap_and_exact_match_lookup(self):
        src = pd.DataFrame({"a": [1, 2, 3, 3], "b": ["x", "y", "z", "z"],
                            "t": ["A", "B", "C", "D"], })
        test = pd.DataFrame({"id": [10, 11, 12], "a": [1, 3, 9],
                             "b": ["x", "z", "q"]})
        ov = train_test_overlap(src, test, ["a", "b"])
        assert ov["n_matches"] == 2
        res = exact_match_lookup(src, test, ["a", "b"], "t", id_col="id")
        # (3, z) is ambiguous in source (labels C and D) so only id 10 matches
        assert res["usable"] and res["map"] == {10: "A"}


# ---------------- tracking ----------------
class TestTracking:
    def test_experiment_log_verdicts_and_persistence(self, tmp_path):
        p = str(tmp_path / "exp.csv")
        log = ExperimentLog(p, noise_floor=0.003)
        e1 = log.record("baseline", "lgbm raw", oof_score=0.950,
                        fold_scores=[0.949, 0.951, 0.950])
        e2 = log.record("crosses", "add TE crosses", oof_score=0.9545, parent_id=e1)
        e3 = log.record("tweak", "lr 0.03->0.028", oof_score=0.9505, parent_id=e1)
        df = ExperimentLog(p).df
        assert df.loc[df["id"] == e2, "verdict"].iloc[0] == "keep"
        assert df.loc[df["id"] == e3, "verdict"].iloc[0] == "noise"
        assert ExperimentLog(p).best()["id"] == e2

    def test_submission_log_correlation(self, tmp_path):
        p = str(tmp_path / "subs.csv")
        sl = SubmissionLog(p)
        for oof, lb in [(0.94, 0.941), (0.95, 0.951), (0.955, 0.954), (0.96, 0.961)]:
            sl.record(1, "sub.csv", "test", oof, public_lb=lb)
        assert SubmissionLog(p).cv_lb_correlation() > 0.95
