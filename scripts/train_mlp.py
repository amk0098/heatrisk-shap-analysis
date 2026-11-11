"""Step 4: train the heat-day MLP and evaluate it on the test period.

Run from the repository root:
    python scripts/train_mlp.py

Writes:
    outputs/models/mlp.pt              trained model + preprocessor + threshold
    outputs/mlp_metrics.json           all numbers (training info, validation, test)
    outputs/figures/training_curve.png loss per epoch (training vs validation)
"""

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import FEATURE_COLUMNS, TARGET, build_dataset, chronological_split  # noqa: E402
from src.model import HeatMLP, Preprocessor, count_parameters, save_checkpoint  # noqa: E402
from src.plot_style import DARK, GREEN, apply_style  # noqa: E402

# ---- settings (also recorded in outputs/mlp_metrics.json) -------------------
SEED = 42
HIDDEN = (16, 8)
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
MAX_EPOCHS = 300
PATIENCE = 20          # stop if validation loss has not improved for 20 epochs
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)

OUTPUT_DIR = REPO_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIGURE_DIR = OUTPUT_DIR / "figures"


def set_seed(seed: int) -> None:
    """Same seed -> same initial weights and same batch order -> same result."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics(y_true, probabilities, threshold) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "pr_auc": average_precision_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def evaluate_loss(model, loss_fn, X, y) -> float:
    model.eval()                      # evaluation mode (no training behaviour)
    with torch.no_grad():             # no gradients needed -> faster, less memory
        return loss_fn(model(X), y).item()


def main() -> None:
    set_seed(SEED)
    apply_style()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. data -------------------------------------------------------------
    df = build_dataset()
    train, validation, test = chronological_split(df)

    preprocessor = Preprocessor(FEATURE_COLUMNS).fit(train)   # fitted on TRAIN only
    X_train = torch.from_numpy(preprocessor.transform(train))
    X_val = torch.from_numpy(preprocessor.transform(validation))
    X_test = torch.from_numpy(preprocessor.transform(test))
    y_train = torch.tensor(train[TARGET].to_numpy(), dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(validation[TARGET].to_numpy(), dtype=torch.float32).unsqueeze(1)
    y_test = test[TARGET].to_numpy()

    # ---- 2. model, loss, optimiser --------------------------------------------
    model = HeatMLP(n_features=len(FEATURE_COLUMNS), hidden=HIDDEN)
    print(f"Model: {model}\nParameters: {count_parameters(model)}")

    # Heat days are rare (3 %). pos_weight makes a missed heat day count as much
    # as ~32 normal days in the loss, so the model cannot just say "never".
    n_pos = y_train.sum().item()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)    # sigmoid + log-loss in one, numerically stable
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"pos_weight = {pos_weight.item():.1f}")

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,                                         # new order every epoch
        generator=torch.Generator().manual_seed(SEED),
    )

    # ---- 3. training loop with early stopping ---------------------------------
    history = {"train_loss": [], "val_loss": []}
    best_val_loss, best_epoch, best_state = float("inf"), 0, None
    start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()                 # 1. forget gradients of the last batch
            logits = model(X_batch)               # 2. forward pass
            loss = loss_fn(logits, y_batch)       # 3. how wrong are we?
            loss.backward()                       # 4. backpropagation: d loss / d weights
            optimizer.step()                      # 5. move weights a small step downhill
            running += loss.item() * len(X_batch)

        train_loss = running / len(X_train)
        val_loss = evaluate_loss(model, loss_fn, X_val, y_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  train loss {train_loss:.4f}  val loss {val_loss:.4f}")
        if epoch - best_epoch >= PATIENCE:
            print(f"Early stopping: no improvement since epoch {best_epoch}")
            break

    train_seconds = time.perf_counter() - start
    model.load_state_dict(best_state)          # go back to the best epoch
    model.eval()

    # ---- 4. threshold on VALIDATION, then test once ----------------------------
    val_proba = model.predict_proba(X_val).numpy()
    val_results = [metrics(y_val.squeeze(1).numpy(), val_proba, t) for t in THRESHOLDS]
    best_val = max(val_results, key=lambda r: r["f1"])
    threshold = best_val["threshold"]

    test_proba = model.predict_proba(X_test).numpy()
    test_metrics = metrics(y_test, test_proba, threshold)

    # ---- 5. report and save ---------------------------------------------------
    print(f"\nBest epoch {best_epoch}, training time {train_seconds:.0f} s")
    print(f"Threshold (best F1 on validation): {threshold}")
    print("\nTest period 2024-2025:")
    print("  Model                Precision  Recall   F1      PR-AUC")
    print("  Logistic Regression  0.252      0.956    0.399   0.411   (source project)")
    print("  Random Forest        0.429      0.607    0.503   0.428   (source project)")
    print(f"  MLP                  {test_metrics['precision']:.3f}      {test_metrics['recall']:.3f}    "
          f"{test_metrics['f1']:.3f}   {test_metrics['pr_auc']:.3f}")
    print(f"  Confusion matrix: {test_metrics['confusion_matrix']}")

    info = {
        "seed": SEED, "hidden": list(HIDDEN), "parameters": count_parameters(model),
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE, "pos_weight": round(pos_weight.item(), 3),
        "best_epoch": best_epoch, "epochs_run": len(history["train_loss"]),
        "train_seconds": round(train_seconds, 1),
        "torch": torch.__version__, "python": platform.python_version(), "machine": platform.machine(),
    }
    save_checkpoint(MODEL_DIR / "mlp.pt", model, preprocessor, threshold, info)
    with open(OUTPUT_DIR / "mlp_metrics.json", "w") as f:
        json.dump({"training": info, "validation": best_val, "test": test_metrics,
                   "history": history}, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color=DARK, label="Training")
    ax.plot(epochs, history["val_loss"], color=GREEN, label="Validierung")
    ax.axvline(best_epoch, color=DARK, linestyle=":", label=f"bestes Modell (Epoche {best_epoch})")
    ax.set_xlabel("Epoche")
    ax.set_ylabel("gewichteter Verlust")
    legend = ax.legend(frameon=True, framealpha=1, edgecolor="none")
    legend.set_zorder(5)  # draw the legend on top of the dotted line
    ax.set_title("Trainingsverlauf")
    fig.savefig(FIGURE_DIR / "training_curve.png")
    plt.close(fig)
    print("\nSaved outputs/models/mlp.pt, outputs/mlp_metrics.json, outputs/figures/training_curve.png")


if __name__ == "__main__":
    main()
