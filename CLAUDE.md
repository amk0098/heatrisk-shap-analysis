# CLAUDE.md — HeatRisk SHAP Analysis

Project context for Claude. Full rules: `PROJECT_INSTRUCTIONS.md`. Theory background: the seminar repo
[shapley-mlp-seminar](https://github.com/amk0098/shapley-mlp-seminar).

## What this is
Case study: SHAP on a modest MLP for next-day heat-day classification, based on
[amk0098/heatrisk-deutschland](https://github.com/amk0098/heatrisk-deutschland) (pinned version: see README.md).
Methodological correctness and interpretable results matter more than model complexity.

## Role in the seminar (decided 05.11.2026)
This repo is the practical part of the seminar talk. Supervisor requirements (verbatim in the seminar repo,
`notes/meetings/2026-11-05-kickoff.md`):
- Train an MLP (**PyTorch**, small) on the HeatRisk data; explain its architecture for this task.
- Analyse and explain the dataset (5 cities, 7 features, rare positive class, chronological split).
- Shapley values on the **logit** (single score, binary task) and on the **embeddings** (last hidden layer),
  with **DeepSHAP and KernelSHAP**; check against exact Shapley (2^7 = 128 coalitions).
- Measure runtime vs. batch size / background size.
- Open: what exactly "Shapley on the embeddings" means — ask the supervisor before building it.

Known risks: strongly correlated temperature features (discuss, don't hide); ~3.7 % positive rate (class weights,
PR metrics); raw files mark missing values as -999 (verify the loader removes them); 26 days until the talk —
reuse the source repo's loader/split logic, keep the model small.

## Key rules (from PROJECT_INSTRUCTIONS.md)
- Start with a read-only audit of the source repo; don't assume earlier descriptions are current.
- No leakage: features at time t use only information available at t; per-city lags/rolling windows; chronological splits; preprocessing fitted on training data only; test period untouched.
- Evaluate with precision, recall, F1, PR-AUC and confusion matrix, not accuracy alone.
- SHAP: verify explainer compatibility; state probability vs. logit; background data from the training period only.
- SHAP is not causal inference. Don't claim results until code has run and outputs are checked. Keep the experiment log.

## Status
Starter project. No MLP or SHAP results yet.

## Session log
| Date | What was done | Next step |
|---|---|---|
| 03.11.2026 | Added CLAUDE.md, .gitignore, data/README.md, pinned requirements.txt, source version (heatrisk-deutschland 9974c0e), MIT LICENSE | Read-only audit of the source repo |
| 05.11.2026 | Chosen as dataset/MLP for the seminar; seminar requirements, risks and repo role recorded in CLAUDE.md, README, config (pytorch, logit + embedding, DeepSHAP + KernelSHAP) | Clarify "embeddings" with supervisor; audit source loader |
