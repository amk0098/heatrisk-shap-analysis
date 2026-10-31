# SHAP Analysis of HeatRisk Deutschland

A focused research project applying SHAP explainability to a Multi-Layer Perceptron (MLP) for next-day heat-day classification using the HeatRisk Deutschland project.

**Source project:** https://github.com/amk0098/heatrisk-deutschland

## Research question
Which meteorological inputs contribute to an MLP's next-day heat-day predictions, how stable are those attributions, and how do correlated weather variables limit interpretation?

## Workflow
1. Audit source data, schema, target alignment and leakage.
2. Verify chronological train/validation/test boundaries.
3. Train a modest MLP with preprocessing fitted on training data only.
4. Evaluate using precision, recall, F1, PR-AUC and confusion matrix.
5. Apply a compatible SHAP explainer with background data from training only.
6. Analyse global and local explanations, feature correlation and stability.

## Starting assumptions to verify
Candidate features: TMK, TXK, TNK, VPM, UPM, TXK_lag_1, TXK_rolling_3. Previously documented splits: train through 2020-12-31; validation 2021-01-01 to 2023-12-31; test from 2024-01-01. Verify against source code before use.

## Critical rules
No random row split for time-series data. Check per-city lag/rolling calculations. Tune only on train/validation. Do not use test data for tuning. State whether SHAP explains probability or raw logits. SHAP is not causal inference.

## Status
Starter project only. No MLP or SHAP results are claimed yet.
