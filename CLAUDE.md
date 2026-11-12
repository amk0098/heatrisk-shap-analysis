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
- Open: what exactly "Shapley on the embeddings" means — asked 11.11.2026; both readings built meanwhile.

Known risks (all handled, see docs/methodology.md): strongly correlated temperature features; ~3 % positive rate
(class weights, PR metrics); -999 missing values (removed by the loader, tested).

## Key rules (from PROJECT_INSTRUCTIONS.md)
- Start with a read-only audit of the source repo; don't assume earlier descriptions are current.
- No leakage: features at time t use only information available at t; per-city lags/rolling windows; chronological splits; preprocessing fitted on training data only; test period untouched.
- Evaluate with precision, recall, F1, PR-AUC and confusion matrix, not accuracy alone.
- SHAP: verify explainer compatibility; state probability vs. logit; background data from the training period only.
- SHAP is not causal inference. Don't claim results until code has run and outputs are checked. Keep the experiment log.

## Status
Complete for the talk on 01.12.2026 (release tag `v1.0-talk`). All supervisor requirements done: dataset analysis,
PyTorch MLP (Run 1), Shapley values on the logit with exact / KernelSHAP / DeepSHAP (Runs 2-3), runtime (Run 4),
background sensitivity (Run 5), embeddings in both readings (Run 6), slide figures. Results: `README.md`,
`docs/experiment_log.md`. Open: supervisor's answer on which embedding reading he means.

## Session log
| Date | What was done | Next step |
|---|---|---|
| 03.11.2026 | Added CLAUDE.md, .gitignore, data/README.md, pinned requirements.txt, source version (heatrisk-deutschland 9974c0e), MIT LICENSE | Read-only audit of the source repo |
| 05.11.2026 | Chosen as dataset/MLP for the seminar; seminar requirements, risks and repo role recorded in CLAUDE.md, README, config (pytorch, logit + embedding, DeepSHAP + KernelSHAP) | Clarify "embeddings" with supervisor; audit source loader |
| 11.11.2026 | Read-only audit of source repo done (`docs/data_audit.md`): loader/target/lags/split correct, no leakage, no date gaps; positive rate 3.15 % (test 3.7 %); TXK_rolling_3 is a 3-day mean; München from 1992. Config hidden layers set to [16, 8] to match slides | Step 2: pin torch, check shap explainers |
| 11.11.2026 | Steps 2-6: torch pinned + environment check; `src/data.py` + dataset analysis + tests; MLP 7-16-8-1 trained (test F1 0.516, PR-AUC 0.431); exact Shapley (own code, 128 coalitions); KernelSHAP (exact here) and DeepSHAP (max dev 1.12) compared; figures for slides 6, 21-23, 36 | Step 7 embeddings (after supervisor answer) or Step 8 runtime |
| 12.11.2026 | Steps 7-12: runtime (Run 4), background sensitivity (Run 5), embeddings both readings (Run 6), slide-size figures; deck filled (Shapley_MLP_Vortrag_3); README, methodology, references, config finalised; full pipeline reproduced from a fresh clone; tag v1.0-talk | Supervisor answer on embeddings; final clean-up of comments |
