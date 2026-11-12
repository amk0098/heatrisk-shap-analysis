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
The weighting mainly moves the decision threshold. Kept pos_weight = 32 (higher recall).

## Run 2 — exact Shapley values on the logit (Step 5)

- **Date:** 11.11.2026
- **Code revision:** `d72b733`
- **Command:** `python scripts/exact_shapley.py`
- **Model:** `outputs/models/mlp.pt` from Run 1
- **Background:** 100 random training days (seed 42), saved in `outputs/shap/background.npz`
- **Method:** own implementation (`src/shapley.py`), all 2^7 = 128 coalitions, interventional value function
  (unknown features replaced by each background day, outputs averaged). Tested in `tests/test_shapley.py`
  (worked example, four axioms, linear closed form).

| Result | Value |
|---|---|
| test days explained | 3650 (46.7 million model evaluations) |
| runtime | 3.1 s (MacBook Air, arm64) |
| base value (mean logit over background) | −13.25 |
| efficiency check (max gap) | 7.6e-06 (float32 rounding of the model) |
| ranking by mean abs phi | Max.-Temp. 2.41, Dampfdruck 2.10, Mittel-Temp. 1.96, 3-Tage-Mittel Max. 1.72, rel. Feuchte 1.48, Vortags-Max. 0.43, Min.-Temp. 0.26 |

**Observations:** Exact computation is cheap for 7 features (3 s for all test days). The base value is very low
(−13.25) because most background days are winter days with logits around −25; it must not be read as a probability.
The correlated temperature features share their effect: on single days some of them get opposite signs
(e.g. Frankfurt 2025-07-01: Max.-Temp. +8.07, Vortags-Max. −1.07).

## Run 3 — KernelSHAP and DeepSHAP vs exact (Step 6)

- **Date:** 11.11.2026
- **Code revision:** `b80b4b1`
- **Command:** `python scripts/shap_logit.py` (shap 0.54.1)
- **Same model and background as Run 2.** KernelSHAP: default nsamples (2·7 + 2048, i.e. all coalitions).
  DeepSHAP: `shap.DeepExplainer` on the PyTorch model. Convergence: KernelSHAP with 16/32/64/126 coalitions,
  `l1_reg=False`, 300 random test days, seed 42.

| Result | KernelSHAP | DeepSHAP |
|---|---|---|
| runtime, all 3650 test days | 6.8 s | 4.2 s |
| max / mean abs deviation from exact | 1.1e-06 / 7.0e-08 | 1.12 / 0.096 (6.5 % of mean abs phi) |
| same most important feature as exact | 100 % of days | 93.6 % of days |

KernelSHAP convergence (mean abs deviation): 16 coalitions 0.134, 32: 0.040, 64: 0.021, 126: 7e-08.

Waterfall days: detected heat day München 2025-06-29 (logit 3.29, p 0.96); most confident false alarm
Frankfurt 2025-07-02 (38.4 °C, logit 5.79; next day only 27.5 °C). Temperature group mean |sum phi| 6.19
vs. strongest single feature (Max.-Temp.) 2.41.

**Observations:** KernelSHAP is exact here because its default sample budget (2062) exceeds the 128 possible
coalitions; the approximation error only appears with fewer coalitions. DeepSHAP keeps efficiency but distributes
the contributions differently (DeepLIFT rescale rule); its largest deviations occur where ReLU neurons switch.
On this small model KernelSHAP is only about 1.6x slower than DeepSHAP, not "slow". The false alarm is a
sudden cool-down that today's features cannot show (no wind, pressure or front information).

## Run 4 — runtime analysis (Step 8)

- **Date:** 12.11.2026
- **Code revision:** `ecfc18d`
- **Command:** `python scripts/runtime.py`
- **Machine:** MacBook Air, Apple M1, 8 GB RAM, macOS 15.8, 8 cores (torch uses 4 threads); Python 3.13.3,
  torch 2.15.0, shap 0.54.1 (`outputs/runtime/hardware.txt`)
- **Setup:** wall-clock time of the explanation only (explainer creation measured separately, < 1 ms);
  one warm-up call per method; median of 5 / 3 / 1 repeats for ≤ 10 / ≤ 100 / 1000 days.
  A and B on the trained model; C on untrained networks M-16-8-1 with random inputs (10 days, background 20).

**A) Number of explained days** (background 100), seconds:

| Days | exact | KernelSHAP | DeepSHAP |
|---:|---:|---:|---:|
| 1 | 0.001 | 0.003 | 0.001 |
| 10 | 0.010 | 0.020 | 0.012 |
| 100 | 0.075 | 0.202 | 0.111 |
| 1000 | 0.723 | 1.885 | 1.132 |

**B) Background size** (100 days), seconds:

| Background | exact | KernelSHAP | DeepSHAP |
|---:|---:|---:|---:|
| 10 | 0.010 | 0.129 | 0.045 |
| 50 | 0.040 | 0.166 | 0.075 |
| 100 | 0.072 | 0.190 | 0.114 |
| 500 | 0.355 | 0.441 | 0.467 |

**C) Number of features** (random networks), seconds:

| M | coalitions | exact | KernelSHAP | DeepSHAP |
|---:|---:|---:|---:|---:|
| 4 | 16 | 0.001 | 0.007 | 0.006 |
| 6 | 64 | 0.002 | 0.010 | 0.006 |
| 8 | 256 | 0.007 | 0.023 | 0.006 |
| 10 | 1 024 | 0.026 | 0.062 | 0.006 |
| 12 | 4 096 | 0.117 | 0.161 | 0.006 |
| 14 | 16 384 | 0.489 | 0.161 | 0.006 |
| 16 | 65 536 | 2.064 | 0.157 | 0.006 |

**Observations:** All three methods scale linearly with the number of explained days. Exact and DeepSHAP also scale
linearly with the background size; KernelSHAP is dominated by its per-day overhead (sampling, weighted regression)
up to ~100 background days. With M = 7 features the exact computation is the fastest method. Its cost doubles with
every additional feature and overtakes KernelSHAP between 12 and 14 features; KernelSHAP stays flat from M = 12 on
because its default budget (2M + 2048 coalitions) then no longer covers all coalitions (it becomes an approximation).
DeepSHAP is practically independent of M. The simple claim "KernelSHAP slow, DeepSHAP fast" does not hold for this
small model; what matters is the number of features.

## Run 5 — background sensitivity (Step 9)

- **Date:** 12.11.2026
- **Code revision:** `3050975`
- **Command:** `python scripts/background_sensitivity.py`
- **Method:** exact Shapley values (own code) for all 3650 test days with three backgrounds from the training period:
  100 random days (whole year, as in Runs 2-3), 100 random June-August days, and the training mean as a single
  reference. Sampling noise: 10 random 100-day backgrounds (seeds 0-9).

| Background | base value | Max.-Temp. | Dampfdruck | Mittel-Temp. | 3-Tage-Mittel | rel. Feuchte | Vortags-Max. | Min.-Temp. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| whole year (100) | −13.25 | 2.41 | 2.10 | 1.96 | 1.72 | 1.48 | 0.43 | 0.26 |
| summer only (100) | −1.23 | 3.32 | 2.42 | 2.44 | 1.74 | 1.36 | 0.48 | 0.36 |
| training mean (1) | −8.82 | 2.57 | 2.09 | 1.93 | 1.45 | 1.45 | 0.62 | 0.33 |

(mean |phi| over the test days, logit units)

Same day, two backgrounds — München 2025-06-29, logit 3.29: whole year base −13.25, Max.-Temp. +5.90,
Mittel-Temp. +4.14, Dampfdruck +3.32; summer only base −1.23, Max.-Temp. +2.47, Mittel-Temp. +1.33, Dampfdruck +0.67.
Sampling noise of a random 100-day background: std of phi ≈ 11.5 % of the mean |phi|.

**Observations:** The background defines the question a Shapley value answers ("compared to which days?"). For the
same day and the same logit, the contributions with a summer background are less than half as large as with a
whole-year background. The global ranking stays similar (Max.-Temp. first, Min.-Temp. and Vortags-Max. last).
Every reported explanation must therefore name its background.

## Run 6 — Shapley values of the embeddings (Step 7)

- **Date:** 12.11.2026
- **Code revision:** `3d0c083`
- **Command:** `python scripts/shap_embeddings.py`
- **Setup:** two readings of "Shapley values on the embeddings" are analysed. Same model and 100-day background
  as Runs 2-3.
- **(a) features → embedding:** the 7 features are the players, each of the 8 neurons of the last hidden layer is
  explained separately (`exact_shapley_multi`, 128 coalitions, all 8 outputs at once).
- **(b) embedding → logit:** the 8 neuron values are the players for the logit (256 coalitions); background =
  embeddings of the 100 background days. The last layer is linear, so phi_j = w_j (h_j − mean h_j).

| Check | Value |
|---|---|
| (a) efficiency gap | 3.6e-15 |
| (b) base value | −13.25 (same as for the input features) |
| (b) exact vs closed form w_j (h_j − mean h_j) | 1.3e-06 |
| (a) KernelSHAP / DeepSHAP max deviation | 2.8e-05 / 0.41 |
| (b) KernelSHAP / DeepSHAP max deviation | 3.0e-05 / 2.9e-06 |

| Neuron | weight to logit | active (all test days) | active before heat days | mean abs phi (b) | strongest input |
|---|---:|---:|---:|---:|---|
| N7 | −1.07 | 92 % | 25 % | 2.86 | Max.-Temp. |
| N5 | −0.97 | 91 % | 27 % | 2.69 | Dampfdruck |
| N2 | −0.56 | 97 % | 60 % | 1.57 | Max.-Temp. |
| N4 | −0.48 | 96 % | 50 % | 1.43 | Max.-Temp. |
| N1 | +0.58 | 36 % | 100 % | 0.36 | Max.-Temp. |
| N3 | +0.27 | 35 % | 100 % | 0.20 | Max.-Temp. |
| N0 | +0.32 | 31 % | 99 % | 0.16 | Max.-Temp. |
| N6 | +0.09 | 19 % | 0 % | 0.01 | (practically unused) |

**Observations:** The decision is dominated by four neurons with negative weights that are active on more than 90 %
of all days ("normal weather") and switch off before most heat days; the network recognises heat mainly by these
neurons turning off. Three neurons with positive weights are active before almost every heat day but contribute
little because of their small weights. One neuron (N6) is practically unused. The important neurons react to the
temperatures, vapour pressure and humidity, hardly to Min.-Temp. and Vortags-Max. DeepSHAP is exact for (b) because
the last layer is linear, and approximate for (a) because of the ReLU layers.

## Run 7 — full reproduction from a fresh clone (Step 12)

- **Date:** 12.11.2026
- **Code revision:** `f5b588a` (fresh `git clone`, raw data copied as in `data/README.md`)
- **Machine:** Linux x86_64, 2 cores, Python 3.13, same pinned packages (the committed results come from an Apple M1)
- **Commands:** `python -m pytest`, then every script in `scripts/` in the order of the README; no errors.

| Result | Fresh run vs. committed |
|---|---|
| tests | 21 passed |
| dataset summary | identical |
| MLP (best epoch, confusion matrix, precision, recall, F1, PR-AUC) | identical |
| exact Shapley values, all 3650 days × 7 features | max. difference 3.5e-06 |
| KernelSHAP / DeepSHAP deviations, background sensitivity, embeddings | max. difference < 1e-06 |
| KernelSHAP convergence, 32 / 64 / 126 coalitions | identical |
| KernelSHAP convergence, 16 coalitions | mean deviation 0.142 vs. 0.134, max. 7.9 vs. 12.8 |

**Observations:** All results are reproducible across machines up to float32 rounding (≤ 3.5e-06). The only
exception is KernelSHAP with 16 coalitions: with only 16 samples for 7 unknowns the weighted regression is
numerically unstable, so rounding differences change the result (three runs: mean deviation 0.134 / 0.142 / 0.149,
maximum 8–14). This is itself a finding: far too few coalitions make KernelSHAP not only inaccurate but unstable.
Runtimes differ by machine and are only comparable within one run.
