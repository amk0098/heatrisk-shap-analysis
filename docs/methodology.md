# Methodology

All decisions behind the results in `docs/experiment_log.md`. Code revision and run details are recorded there.

## 1. Data

- **Source:** DWD Climate Data Center, daily station observations ("kl"), five stations: Berlin (400),
  Frankfurt (1420), Hamburg (1975), Köln (2667), München (1262). Files as shipped with heatrisk-deutschland,
  commit `9974c0e`; data until 31.12.2025. Used from 01.01.1990 (München: from 17.05.1992, start of the file).
- **Missing values:** `-999` in the raw files → NaN. Checked: none survive loading (`tests/test_data.py`).
- **Audit:** the source pipeline was audited read-only before use (`docs/data_audit.md`): no date gaps, no
  duplicates, target and lags verified on real days, no leakage found.

## 2. Prediction task and features

- **Prediction time:** end of day t. **Target:** `heat_day_next` = 1 if the maximum temperature TXK of day t+1 at
  the same station is ≥ 30.0 °C. Rows without a next-day value are dropped.
- **Features (7), all known at the end of day t:** TMK, TXK, TNK (mean, max., min. temperature of t), VPM (vapour
  pressure), UPM (relative humidity), `TXK_lag_1` (TXK of t−1), `TXK_rolling_3` (mean TXK of t−1, t−2, t−3; today
  excluded). Lags and rolling windows are computed per station.
- **Class balance:** 3.15 % heat days overall (train 3.03 %, validation 4.00 %, test 3.70 %); 93 % of them in
  June–August. The five temperature features correlate at 0.87–0.98 (training period).

## 3. Split

Chronological, never shuffled across time: training 1990-01-04 – 2020-12-31 (55 543 rows), validation
2021-01-01 – 2023-12-31 (5 475), test 2024-01-01 – 2025-12-30 (3 650). The test period was used once, for the
final evaluation. All choices (epochs, threshold, background) use training or validation data only.

## 4. Preprocessing

Mean imputation and standardisation with mean and standard deviation of the **training** period
(`src/model.py: Preprocessor`). Remaining missing values: at most 26 rows per split (mostly VPM/UPM). All Shapley
values are computed on the standardised inputs; plots are labelled with the raw values (°C, hPa, %).

## 5. Model

- PyTorch MLP 7 → 16 → 8 → 1, ReLU, 273 parameters. `body` = both hidden layers, its output is the 8-dimensional
  **embedding**; `head` = last linear layer, its output is the **logit**. `forward` returns the logit (no sigmoid).
- Training: `BCEWithLogitsLoss` with `pos_weight` = n_neg / n_pos on training data (32.0), Adam (lr 1e-3),
  batch size 256, up to 300 epochs, early stopping on validation loss (patience 20; best epoch 40), seed 42.
- Decision threshold: best F1 on validation over 0.05–0.95 (step 0.05) → 0.9. The weighting shifts all
  probabilities upwards, which explains the high threshold; the probabilities are not calibrated.
- Evaluation on the test period: precision, recall, F1, PR-AUC (average precision) and confusion matrix, compared
  with the Logistic Regression and Random Forest of the source project on the same rows.
- Side experiment without weighting (pos_weight 1): F1 0.502, PR-AUC 0.448, threshold 0.3 (not kept).

## 6. Shapley values

- **Explained output:** the **logit** (additive; the sigmoid would distort contributions). For the embeddings:
  (a) each of the 8 neurons with the 7 features as players; (b) the logit with the 8 neurons as players.
- **Value function (interventional / marginal):** v(S) = mean over the background days of f(x_S, b_S̄): features
  in S keep today's value, the others take the value of a background day. This is the game approximated by
  KernelSHAP and DeepSHAP in the `shap` library, so all three methods answer the same question.
- **Background:** 100 random training days (seed 42), saved in `outputs/shap/background.npz` and reused by all
  methods. Base value = mean logit over these days = −13.25 (most background days are cold; the mean of logits is
  not the logit of the mean probability, so it is not converted to a percentage).
- **Exact computation** (`src/shapley.py`): all 2^M coalitions as bit masks, Shapley weights
  |S|!(M−|S|−1)!/M!. Tested against the slide example (φT = 1.5, φH = 2.5), the four axioms (efficiency,
  symmetry, dummy, additivity) and the closed form for linear models.
- **KernelSHAP:** `shap.KernelExplainer` on a numpy wrapper of the logit, default `nsamples` (2·M + 2048 ≥ 128,
  i.e. all coalitions). Convergence study with 16/32/64/126 coalitions on 300 test days with `l1_reg=False`
  (otherwise `shap` adds a Lasso feature selection for small sample sizes).
- **DeepSHAP:** `shap.DeepExplainer` on the PyTorch model (DeepLIFT rescale rule, averaged over the background).
- **Checks:** efficiency (base value + Σφ = logit) for every method and day; deviation of KernelSHAP and DeepSHAP
  from the exact values.

## 7. Further analyses

- **Runtime** (`scripts/runtime.py`): wall-clock time of the explanation, warm-up call, median of repeats;
  (A) 1–1000 explained days, (B) 10–500 background days, (C) 4–16 features on untrained random networks of the same
  shape (to show the 2^M growth the 7-feature model cannot show).
- **Background sensitivity** (`scripts/background_sensitivity.py`): whole-year vs. summer-only (June–August)
  training days vs. the training mean as single reference; 10 random 100-day draws for the sampling noise.

## 8. Limitations

- Correlated features: the interventional value function evaluates the model on combinations that do not occur
  (e.g. high maximum with low mean temperature); this affects the exact values as well as both approximations.
  Conditional Shapley values (Aas et al. 2021) would address this.
- Results depend on the background; every explanation must state it.
- Shapley values explain the fitted model, not the physical climate system; no causal interpretation.
- One model, one seed, five stations; generalisation to other stations or climates was not tested.
- The features describe only the current day; weather changes (fronts, pressure, wind) are invisible to the model.
