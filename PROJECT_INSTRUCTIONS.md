# Project Instructions — HeatRisk SHAP Analysis

Apply SHAP to a modest MLP for next-day heat-day classification in HeatRisk Deutschland. Prioritise methodological correctness and interpretable results, not model complexity.

## Read-only source audit
Inspect https://github.com/amk0098/heatrisk-deutschland. Verify raw schema, city/station identifiers, dates, target construction, features, lag/rolling logic, split boundaries and existing metrics. Do not assume prior descriptions are current.

## Leakage controls
- Features at time t must use only information available at prediction time t.
- Verify target at t means next-day maximum temperature meets the defined threshold.
- Compute lags and rolling windows separately within each city/station.
- Use past-only rolling windows and chronological splits.
- Fit imputation/scaling on training data only.
- Tune hyperparameters and thresholds using train/validation only; hold test period untouched.
- Select SHAP background rows from training period only.

## Model and evaluation
Start with a small MLP. Record architecture, seed, package versions and preprocessing. Accuracy alone is insufficient for imbalanced labels; report precision, recall, F1, PR-AUC and confusion matrix. Consider calibration.

## SHAP
Verify package/model compatibility before selecting an explainer. State whether output is positive-class probability or raw logit. Include global and local explanations, additivity checks where supported, background sensitivity and feature-correlation discussion.

## Interpretation
SHAP explains the fitted model relative to a background/reference distribution. It does not prove causality or explain the physical climate system. Discuss correlated predictors, temporal dependence, city generalisation, class imbalance and operational limitations.

## Workflow
First produce a read-only audit and minimal implementation plan. Do not claim results until code has run and outputs have been checked. Keep data provenance and experiment log.
