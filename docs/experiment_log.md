# Experiment Log

For each run record date, code revision, configuration, random seed, data version, model metrics, SHAP method, background sample and observations. Distinguish planned experiments from completed experiments.

## Run 1 — MLP training (Step 4)

- **Date:** FILL IN
- **Code revision:** FILL IN (`git rev-parse --short HEAD` before the commit that adds the outputs)
- **Command:** `python scripts/train_mlp.py`
- **Data:** heatrisk-deutschland 9974c0e raw files, checked by `tests/test_data.py`
- **Configuration:** 7-16-8-1 ReLU MLP (273 parameters), BCEWithLogitsLoss with pos_weight = n_neg/n_pos on train,
  Adam lr 1e-3, batch 256, max 300 epochs, early stopping on validation loss (patience 20), seed 42.
  Preprocessing: mean imputation + standardisation fitted on train. Threshold = best F1 on validation (grid 0.05–0.95).
- **Machine:** FILL IN (from `outputs/mlp_metrics.json` → training.machine, training.torch)

| Result | Value |
|---|---|
| pos_weight | FILL IN |
| best epoch / epochs run | FILL IN |
| threshold | FILL IN |
| test precision / recall / F1 | FILL IN |
| test PR-AUC | FILL IN |
| confusion matrix (tn, fp, fn, tp) | FILL IN |

**Observations:** FILL IN (compare with LR F1 0.399 / RF F1 0.503 from the source project)
