# kagglekit

**A Python toolkit for reliable machine learning experiments with reproducible validation, leakage detection, encoding, threshold optimization, ensembling, and experiment tracking.**

The repository has two complementary components:

1. **`kagglekit`** (Python package): reusable building blocks for machine learning workflows, including fold management, fold-safe feature encoding, threshold optimization, ensembling, leakage detection, and experiment tracking.
2. **`docs/`**: a structured methodology for planning experiments, selecting validation strategies, prioritizing improvements, and maintaining reproducible model development across projects.

Together they help eliminate common sources of performance loss, including inconsistent validation schemes, data leakage, improperly fitted encoders, unstable ensemble weights, and untracked experimentation.

## Install

```bash
pip install git+https://github.com/amerob/kagglekit.git
# or for development
git clone https://github.com/amerob/kagglekit.git
cd kaggle-competitor
pip install -e ".[dev]" && pytest -q
```

Requires Python >= 3.9, NumPy, pandas, and scikit-learn. `kagglekit` is model-agnostic and works with any machine learning framework.

## Quickstart

```python
from kagglekit import (
    FoldManager,
    TargetEncoder,
    prior_corrected_argmax,
    bagged_hill_climb,
    blend,
    noise_floor,
    ExperimentLog,
)

# 1. Create reproducible validation folds
fm = FoldManager.create(y=y, scheme="stratified", n_splits=5, seed=42)
fm.save("folds.json")

# 2. Apply fold-safe target encoding
for tr_idx, vl_idx in fm.splits():
    enc = TargetEncoder(cols=["cat_cross"], smooth=20, classes=(0, 2))
    X_tr, X_vl, X_te = enc.fit_transform_fold(
        X.iloc[tr_idx],
        y[tr_idx],
        [X.iloc[tr_idx], X.iloc[vl_idx], X_test],
    )

    # Train any model and collect validation predictions

# 3. Optimize predictions for the evaluation metric
pred = prior_corrected_argmax(oof_probs, y=y)

# 4. Build a robust ensemble
weights, history = bagged_hill_climb(stacked_oof, y, metric_fn)
test_blend = blend(stacked_test, weights)

# 5. Track experiments against the estimated noise floor
floor = noise_floor(y, oof_probs, metric_fn)
log = ExperimentLog("experiments.csv", noise_floor=floor)

log.record(
    hypothesis="Target encoding on interaction features",
    change="+9 engineered categorical crosses",
    oof_score=0.9531,
    parent_id=1,
)
```

Runnable example: [`examples/quickstart.py`](examples/quickstart.py).

---

## What is inside

### `src/kagglekit/`

| Module | Contents |
|--------|----------|
| `validation.py` | `FoldManager` (stratified, group, stratified-group, k-fold, expanding time splits), `adversarial_validation`, `shared_category_dtypes` |
| `encoding.py` | `TargetEncoder`, `frequency_encode`, `add_missingness_features`, `add_categorical_crosses` |
| `metrics.py` | `prior_corrected_argmax`, threshold optimization utilities, `OptimizedRounder`, `noise_floor` |
| `ensemble.py` | `bagged_hill_climb`, `blend`, `rank_average`, `equal_weight_baseline`, `diversity_matrix` |
| `leakage.py` | Duplicate detection, train/test overlap analysis, exact-match lookup, shuffle sanity checks |
| `tracking.py` | `ExperimentLog` and `SubmissionLog` for experiment and validation tracking |

### `docs/`

The documentation describes a repeatable workflow for machine learning experimentation, including validation strategy, experiment prioritization, evaluation methodology, ensemble construction, and project organization. The reference guides cover tabular data, NLP, computer vision, time series, recommender systems, multimodal learning, evaluation metrics, ensembling, and experiment tracking.

---

## Design principles

### Validation first

Reliable validation is the foundation of trustworthy model evaluation. Validation splits should mirror deployment conditions, remain fixed throughout development, and ensure every learned transformation is fitted only on training data.

### Metric-driven optimization

The evaluation metric should determine the optimization objective, prediction strategy, threshold selection, and post-processing pipeline.

### Reproducible experimentation

Every experiment should have a clearly stated hypothesis, a measurable outcome, and an objective acceptance criterion based on expected statistical variation.

### Robust over lucky

Threshold optimization and ensemble weighting should prioritize stability across resamples rather than improvements driven by validation noise.

---

## Tests

The project includes tests covering:

- validation split integrity
- fold-safe encoding
- unseen category handling
- threshold optimization
- ensemble weight recovery
- leakage detection
- experiment logging

```bash
pytest -q
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

Contributions are welcome, particularly new validation strategies, evaluation metrics, feature engineering utilities, leakage detection techniques, and experiment management improvements.

---

## Author

**Amer Hussein** 

- GitHub: https://github.com/amerob
- LinkedIn: https://linkedin.com/in/aamero

---

## License

MIT. See [LICENSE](LICENSE).
