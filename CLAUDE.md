# CLAUDE.md — HeatRisk SHAP Analysis

Project context. Full rules: `PROJECT_INSTRUCTIONS.md`. Theory and slides: the seminar repository
[shapley-mlp-seminar](https://github.com/amk0098/shapley-mlp-seminar).

## What this is
Shapley analysis of a small MLP for next-day heat-day classification, based on
[amk0098/heatrisk-deutschland](https://github.com/amk0098/heatrisk-deutschland) (pinned version: see README.md).
Practical part of the seminar talk „Shapley-Analyse eines MLP“ (Bergische Universität Wuppertal, 1 December 2026).
Methodological correctness and interpretable results matter more than model complexity.

## Scope
- Small PyTorch MLP (7 → 16 → 8 → 1) on the HeatRisk data; dataset analysis (5 stations, 7 features, rare positive
  class, chronological split).
- Shapley values on the **logit** and on the **embeddings** (last hidden layer) with **exact Shapley values**
  (all 2^7 = 128 coalitions, own implementation), **KernelSHAP** and **DeepSHAP**.
- Runtime versus number of explained days, background size and number of features; background sensitivity.

Results: `README.md`; method: `docs/methodology.md`; all runs: `docs/experiment_log.md`. The state used in the talk
is tagged `v1.0-talk`.

## Key rules (from PROJECT_INSTRUCTIONS.md)
- No leakage: features at time t use only information available at t; per-city lags/rolling windows; chronological
  splits; preprocessing fitted on training data only; test period untouched.
- Evaluate with precision, recall, F1, PR-AUC and confusion matrix, not accuracy alone.
- SHAP: verify explainer compatibility; state probability vs. logit; background data from the training period only.
- SHAP is not causal inference. Do not report results before the code has run and the outputs are checked; record
  every run in the experiment log.

## Conventions
- Raw data is not committed (`data/README.md`); outputs are committed so that the slides can be traced to them.
- One script per analysis step in `scripts/`, reusable code in `src/`, tests in `tests/` (`python -m pytest`).
- Fixed seed 42; pinned package versions in `requirements.txt`.
