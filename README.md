# Shapley Analysis of a Heat-Day MLP (HeatRisk Deutschland)

Practical part of the seminar talk **„Shapley-Analyse eines MLP“** (Bergische Universität Wuppertal,
1 December 2026; theory and slides: [shapley-mlp-seminar](https://github.com/amk0098/shapley-mlp-seminar)).

A small PyTorch MLP predicts whether **tomorrow** will be a heat day (maximum temperature ≥ 30 °C) at five German
weather stations. Its predictions are explained with **exact Shapley values** (own implementation, all 2⁷ = 128
coalitions), **KernelSHAP** and **DeepSHAP**, on the **logit** and on the **embedding** (last hidden layer).
Because the model has only 7 features, the exact values serve as ground truth for both approximations.

**Data:** DWD daily station data as shipped with
[heatrisk-deutschland](https://github.com/amk0098/heatrisk-deutschland), commit
`9974c0eb22b027d3a9ed66f89a7ecf3925d6b102`. Loading, target, features and split are reused unchanged, so the MLP is
compared with that project's baselines on exactly the same test days.

## Results (test period 2024–2025, 3650 city-days, 135 heat days)

**Model** — 7 → 16 → 8 → 1, ReLU, 273 parameters, class-weighted loss (pos_weight 32), threshold 0.9 chosen on
validation:

| Model | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression (source project) | 0.252 | 0.956 | 0.399 | 0.411 |
| Random Forest (source project) | 0.429 | 0.607 | 0.503 | 0.428 |
| **MLP (this project)** | **0.421** | **0.667** | **0.516** | **0.431** |

**Shapley values on the logit** — background: 100 random training days, base value −13.25:

| | max. abs. deviation from exact | mean abs. deviation | runtime, all test days* |
|---|---:|---:|---:|
| exact (own code) | — | — | 3.1 s |
| KernelSHAP | 0.000001 | 0.0000001 | 6.8 s |
| DeepSHAP | 1.12 | 0.096 (6.5 % of mean abs. φ) | 4.2 s |

\* MacBook Air, Apple M1, 8 GB RAM.

Main findings:

- **Temperature dominates**, but spread over five correlated features (r ≥ 0.87): together mean |φ| 6.19, the
  strongest single feature (Max.-Temp.) only 2.41. Correlated features sometimes receive opposite signs.
- **KernelSHAP is exact here** because its default budget (2·M + 2048 samples) exceeds the 128 coalitions; with
  16 / 32 / 64 coalitions the mean deviation is 0.13 / 0.04 / 0.02. **DeepSHAP** keeps efficiency but distributes
  differently (DeepLIFT rescale rule at ReLUs).
- **Runtime depends on the number of features, not on "Kernel vs. Deep"**: with 7 features the exact computation is
  the fastest; it doubles with every feature and overtakes KernelSHAP between 12 and 14 features (random networks).
  DeepSHAP is practically independent of the number of features.
- **The background decides the question**: for the same day and logit, a summer-only background makes the
  contributions less than half as large (Max.-Temp. +5.90 → +2.47). Ten random 100-day backgrounds differ by
  ≈ 11.5 % of the mean |φ|.
- **Embeddings**: four neurons with negative weights are active on > 90 % of days ("normal weather") and dominate
  the logit; before most heat days they switch off. The network recognises heat mainly by these neurons turning off.
- **A false alarm explained**: Frankfurt, 2 July 2025 (38.4 °C, model ≈ 99.7 %), next day only 27.5 °C. The
  explanation shows the model reacts to today's heat only; it has no information about a coming weather change.

Shapley values describe how the **model** reacts relative to a background; they are not causal statements about the
weather. Full numbers and observations: [`docs/experiment_log.md`](docs/experiment_log.md); method details:
[`docs/methodology.md`](docs/methodology.md).

## Reproduce

Tested with Python 3.13 on macOS (Apple M1) and Linux. Total runtime about 5–10 minutes.

```bash
git clone https://github.com/amk0098/heatrisk-shap-analysis
cd heatrisk-shap-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# raw data (not committed, see data/README.md)
git clone https://github.com/amk0098/heatrisk-deutschland ../heatrisk-deutschland
mkdir -p data/raw
cp ../heatrisk-deutschland/data/raw/*.txt data/raw/

python -m pytest                              # 21 tests: data pipeline, model, Shapley axioms
python scripts/check_environment.py           # torch + shap compatibility
python scripts/analyze_dataset.py             # Step 3: dataset analysis
python scripts/train_mlp.py                   # Step 4: train and evaluate the MLP
python scripts/exact_shapley.py               # Step 5: exact Shapley values (128 coalitions)
python scripts/shap_logit.py                  # Step 6: KernelSHAP and DeepSHAP vs. exact
python scripts/runtime.py                     # Step 8: runtime analysis
python scripts/background_sensitivity.py      # Step 9: background sensitivity
python scripts/shap_embeddings.py             # Step 7: Shapley values of the embeddings
python scripts/slide_figures.py               # figures at slide size for the talk
```

Steps 5–9 and the slide figures read the model from `outputs/models/mlp.pt`; run them after `train_mlp.py`.
Runtimes depend on the machine; all other numbers are reproducible (seed 42).

## Repository structure

| Path | Content |
|---|---|
| `src/data.py` | loading, target, lag features and chronological split (from the source project) |
| `src/model.py` | preprocessing (fitted on training data) and the MLP, returning logit and embedding |
| `src/shapley.py` | exact Shapley values for one or several outputs, background sampling |
| `src/plots.py`, `src/plot_style.py` | waterfall and comparison plots in the slide colours |
| `scripts/` | one script per step (see above) |
| `tests/` | data pipeline, model shapes/parameter count, Shapley axioms and worked example |
| `outputs/models/mlp.pt` | trained model with preprocessing statistics and threshold |
| `outputs/shap/` | all Shapley values and summaries (`.npz`, `.csv`) |
| `outputs/figures/` | figures; `outputs/figures/slides/` = figures as used in the talk |
| `outputs/runtime/` | runtime measurements and hardware description |
| `docs/data_audit.md` | read-only audit of the source data pipeline |
| `docs/methodology.md` | all methodological decisions |
| `docs/experiment_log.md` | Runs 1–6 with configuration, numbers and observations |
| `docs/references.md` | literature |
| `configs/experiment.yaml` | record of the settings used |

## Licence

Code: MIT (see `LICENSE`). Data: Deutscher Wetterdienst, open data, use with attribution („Quelle: Deutscher
Wetterdienst“).
