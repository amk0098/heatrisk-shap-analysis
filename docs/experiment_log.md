# Experiment Log

For each run record date, code revision, configuration, random seed, data version, model metrics, SHAP method, background sample and observations. Distinguish planned experiments from completed experiments.

## Run 1 — MLP training (Step 4)

- **Date:** 11.11.2026
- **Code revision:** `0bdff9a` (commit that added the model and its outputs)
- **Command:** `python scripts/train_mlp.py`
- **Data:** heatrisk-deutschland 9974c0e raw files, checked by `tests/test_data.py`
- **Configuration:** 7-16-8-1 ReLU MLP (273 parameters), BCEWithLogitsLoss with pos_weight = n_neg/n_pos on train,
  Adam lr 1e-3, batch 256, max 300 epochs, early stopping on validation loss (patience 20), seed 42.
  Preprocessing: mean imputation + standardisation fitted on train. Threshold = best F1 on validation (grid 0.05–0.95).
- **Machine:** MacBook Air, arm64, Python 3.13.3, torch 2.15.0 (training time 12.5 s)

| Result | Value |
|---|---|
| pos_weight | 32.0 |
| best epoch / epochs run | 40 / 60 |
| threshold | 0.9 |
| test precision / recall / F1 | 0.421 / 0.667 / 0.516 |
| test PR-AUC | 0.431 |
| confusion matrix (tn, fp, fn, tp) | 3391, 124, 45, 90 |

**Observations:** The MLP is slightly better than the Random Forest of the source project (F1 0.516 vs 0.503,
PR-AUC 0.431 vs 0.428) and much more precise than the Logistic Regression (0.421 vs 0.252). It finds 90 of 135
heat days with 124 false alarms. Identical results on two machines (seed 42).

### Run 1b — same, but pos_weight = 1 (not kept)

Threshold 0.3, precision 0.457, recall 0.556 (75 of 135), F1 0.502, PR-AUC 0.448.
Without weighting the model finds 15 fewer heat days; the ranking quality (PR-AUC) is similar.
The weighting mainly moves the decision threshold. Kept pos_weight = 32 (higher recall, as on the slides).

## Run 2 — exact Shapley values on the logit (Step 5)

- **Date:** 11.11.2026
- **Code revision:** `d72b733`
- **Command:** `python scripts/exact_shapley.py`
- **Model:** `outputs/models/mlp.pt` from Run 1
- **Background:** 100 random training days (seed 42), saved in `outputs/shap/background.npz`
- **Method:** own implementation (`src/shapley.py`), all 2^7 = 128 coalitions, interventional value function
  (unknown features replaced by each background day, outputs averaged). Tested in `tests/test_shapley.py`
  (slide example, four axioms, linear closed form).

| Result | Value |
|---|---|
| test days explained | 3650 (46.7 million model evaluations) |
| runtime | 3.1 s (MacBook Air, arm64) |
| base value (mean logit over background) | −13.25 |
| efficiency check (max gap) | 7.6e-06 (float32 rounding of the model) |
| ranking by mean abs phi | Max.-Temp. 2.41, Dampfdruck 2.10, Mittel-Temp. 1.96, 3-Tage-Mittel Max. 1.72, rel. Feuchte 1.48, Vortags-Max. 0.43, Min.-Temp. 0.26 |

**Observations:** FILL IN (one or two sentences in your own words)

## Run 3 — KernelSHAP and DeepSHAP vs exact (Step 6)

- **Date:** FILL IN
- **Code revision:** FILL IN
- **Command:** `python scripts/shap_logit.py` (shap 0.54.1)
- **Same model and background as Run 2.** KernelSHAP: default nsamples (2·7 + 2048, i.e. all coalitions).
  DeepSHAP: `shap.DeepExplainer` on the PyTorch model. Convergence: KernelSHAP with 16/32/64/126 coalitions,
  `l1_reg=False`, 300 random test days, seed 42.

| Result | KernelSHAP | DeepSHAP |
|---|---|---|
| runtime, all 3650 test days | FILL IN | FILL IN |
| max / mean abs deviation from exact | FILL IN | FILL IN |
| same most important feature as exact | FILL IN | FILL IN |

Waterfall days: FILL IN (detected heat day, false alarm). Temperature group mean |sum phi|: FILL IN.

**Observations:** FILL IN
