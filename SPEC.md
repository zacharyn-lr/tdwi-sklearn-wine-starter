# Wine classifier — spec

Fill this in with your agent during the lab. Do not implement until you and the agent agree on the spec.

**Not** [`requirements.txt`](requirements.txt) — that file lists Python packages. This file is your **project spec** (goal, metrics, deliverables).

## Goal

Classify wine cultivar (three classes) from the chemical features in the scikit-learn Wine dataset. Use the fixed holdout split in `wine_data.py` (`test_size=0.2`, `random_state=42`).

## Success metrics

Holdout accuracy ≥ 0.95 on that fixed split. No other metrics.

## Modeling approach

`StandardScaler` followed by `LogisticRegression` with fixed hyperparameters. No cross-validation and no hyperparameter search. Fit only on the training split from `wine_data.get_train_test_split`.

## Deliverables

- `test_model.py` — acceptance tests derived from the success metric (holdout accuracy ≥ 0.95).
- `train_model.py` — training implementation that loads data through `wine_data.py`.
- `models/wine_classifier.joblib` — saved classifier written by `train_model.py`.

## Out of scope

- Deep learning / neural nets
- REST API, model serving, or a prediction CLI beyond training
- Plots, dashboards, or EDA notebooks
- Changing the split in `wine_data.py` (`test_size=0.2`, `random_state=42` stays)
- Feature engineering beyond scaling
- Multi-model comparison or a model-selection report

## Open questions

