# CLAUDE.md — HeatRisk SHAP Analysis

Project context for Claude. Full rules: `PROJECT_INSTRUCTIONS.md`. Theory background: the seminar repo
[shapley-mlp-seminar](https://github.com/amk0098/shapley-mlp-seminar).

## What this is
Case study: SHAP on a modest MLP for next-day heat-day classification, based on
[amk0098/heatrisk-deutschland](https://github.com/amk0098/heatrisk-deutschland) (pinned version: see README.md).
Methodological correctness and interpretable results matter more than model complexity.

## Key rules (from PROJECT_INSTRUCTIONS.md)
- Start with a read-only audit of the source repo; don't assume earlier descriptions are current.
- No leakage: features at time t use only information available at t; per-city lags/rolling windows; chronological splits; preprocessing fitted on training data only; test period untouched.
- Evaluate with precision, recall, F1, PR-AUC and confusion matrix, not accuracy alone.
- SHAP: verify explainer compatibility; state probability vs. logit; background data from the training period only.
- SHAP is not causal inference. Don't claim results until code has run and outputs are checked. Keep the experiment log.

## Status
Starter project. No MLP or SHAP results yet. The MLP will be added later.

## Session log
| Date | What was done | Next step |
|---|---|---|
| 03.11.2026 | Added CLAUDE.md, .gitignore, data/README.md, pinned requirements.txt, source version (heatrisk-deutschland 9974c0e), MIT LICENSE | Read-only audit of the source repo |
