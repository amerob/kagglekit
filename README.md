# kaggle-competitor

**An AI skill plus a Python toolkit for competing in any ML competition, built to maximize final private leaderboard score.**

Two layers in one repository:

1. **`kagglekit`** (Python package): the recurring competition mechanics as tested, importable code. Fold management, fold-safe encoding, metric-optimal decision rules, bagged ensembling, leakage detection, experiment tracking.
2. **The skill** (`docs/`): a domain-agnostic standard operating procedure that turns any capable AI model into a disciplined competitor. Strategy, decision trees, and the operating loop that decides what to build next.

The skill supplies judgment; the package supplies machinery. Together they remove the unforced errors that decide most leaderboards: misread metrics, leaky encoders, CV schemes that do not mirror the test split, and ensemble weights fitted to noise.

## Install

```bash
pip install git+https://github.com/amerob/kaggle-competitor.git
# or for development
git clone https://github.com/amerob/kaggle-competitor.git
cd kaggle-competitor
pip install -e ".[dev]" && pytest -q
```

Requires Python >= 3.9, numpy, pandas, scikit-learn. GBDT libraries are your choice; kagglekit is model-agnostic.

## Quickstart

```python
from kagglekit import (FoldManager, TargetEncoder, prior_corrected_argmax,
                       bagged_hill_climb, blend, noise_floor, ExperimentLog)

# 1. Freeze folds once; every model in the campaign shares them
fm = FoldManager.create(y=y, scheme="stratified", n_splits=5, seed=42)
fm.save("folds.json")

# 2. Fold-safe target encoding inside the loop
for tr_idx, vl_idx in fm.splits():
    enc = TargetEncoder(cols=["cat_cross"], smooth=20, classes=(0, 2))
    X_tr, X_vl, X_te = enc.fit_transform_fold(X.iloc[tr_idx], y[tr_idx],
                                              [X.iloc[tr_idx], X.iloc[vl_idx], X_test])
    ...  # train any model, collect OOF and test probabilities

# 3. Metric-optimal decision rule (balanced accuracy example: +5 to +8 points
#    over plain argmax under heavy class imbalance)
pred = prior_corrected_argmax(oof_probs, y=y)

# 4. Ensemble with bagged hill-climbing so weights do not overfit OOF noise
weights, history = bagged_hill_climb(stacked_oof, y, metric_fn)
test_blend = blend(stacked_test, weights)

# 5. Pre-registered acceptance: keep only what clears the noise floor
floor = noise_floor(y, oof_probs, metric_fn)
log = ExperimentLog("experiments.csv", noise_floor=floor)
log.record(hypothesis="TE crosses", change="+9 encoded crosses",
           oof_score=0.9531, parent_id=1)   # verdict: keep / noise / revert
```

Runnable end-to-end version: [`examples/quickstart.py`](examples/quickstart.py).

## What is inside

### `src/kagglekit/`

| Module | Contents |
|---|---|
| `validation.py` | `FoldManager` (stratified / group / stratified-group / kfold / expanding time splits, persisted to JSON), `adversarial_validation` (train-vs-test shift detector with feature attribution), `shared_category_dtypes` (kills cross-fold category mismatch errors) |
| `encoding.py` | `TargetEncoder` (smoothed, multiclass or regression, fold-safe by design), `frequency_encode`, `add_missingness_features`, `add_categorical_crosses` |
| `metrics.py` | `prior_corrected_argmax` (Bayes-optimal rule for balanced accuracy), `optimize_binary_threshold` and `optimize_multiclass_thresholds` (bagged, noise-resistant), `OptimizedRounder` (QWK-style ordinal binning), `noise_floor` (bootstrap OOF standard error) |
| `ensemble.py` | `bagged_hill_climb` (Caruana greedy selection scored over OOF subsamples, returns best-not-last weights), `blend`, `rank_average` (AUC-correct blending), `equal_weight_baseline`, `diversity_matrix` |
| `leakage.py` | `duplicate_report` (conflicting-label detection), `train_test_overlap`, `exact_match_lookup` (original-dataset label lookup for synthetic competitions), `shuffle_sanity_check` (row-order leak probe) |
| `tracking.py` | `ExperimentLog` (CSV-backed, auto verdicts keep/noise/revert against the noise floor), `SubmissionLog` (OOF vs LB correlation, the health monitor of your CV) |

### `docs/` (the AI skill)

`SKILL.md` holds the operating loop, prioritization scorecard, validation doctrine, threat checklist, and endgame protocol. `docs/references/` holds ten domain files (tabular, NLP, CV, time series, recsys/geospatial/multimodal/RL/code comps, metric playbook, ensembling, tracking) loaded on demand. Install into Claude via Settings -> Capabilities -> Skills, drop into `~/.claude/skills/` for Claude Code, or load `SKILL.md` as a system instruction in any agent framework.

## Design principles

**Metric first.** The evaluation metric dictates loss, decision rule, and post-processing. A correct decision rule is routinely worth more than a better model.

**The CV scheme is the strategy.** Folds mirror the test split, are frozen on day one, and are shared by every model. Everything that learns from data is fitted inside the fold.

**Noise-floor discipline.** Every experiment is judged as delta over the bootstrapped OOF standard error, with the acceptance rule registered before the run.

**Best, not last.** Weight searches and threshold sweeps are bagged and return the best iterate. Both requirements exist because their absence has cost real leaderboard positions.

## Tests

19 tests cover fold integrity (group folds never split a group, time folds validate strictly forward), encoder fold-safety and unseen-category handling, decision-rule superiority under imbalance, hill-climb weight recovery, leak lookup ambiguity handling, and log verdict logic.

```bash
pytest -q
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New metric rules, failure modes, and domain deepening are the most valuable contributions.

## Author

**Amer Hussein** - AI/ML Engineer, Double Kaggle Master

- GitHub: [github.com/amerob](https://github.com/amerob)
- LinkedIn: [linkedin.com/in/aamero](https://linkedin.com/in/aamero)

## License

MIT. See [LICENSE](LICENSE).
