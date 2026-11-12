"""Step 5: exact Shapley values of the trained MLP's logit for every test day.

Run from the repository root:
    python scripts/exact_shapley.py

Writes:
    outputs/shap/background.npz     the 100 training days used as reference (reused in Steps 6-9)
    outputs/shap/exact_logit.npz    phi (3650 x 7), base value, logits, dates, cities, raw feature values
    outputs/shap/exact_summary.csv  mean |phi| and mean phi per feature
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import FEATURE_COLUMNS, FEATURE_LABELS_DE, TARGET, build_dataset, chronological_split  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.shapley import exact_shapley, sample_background  # noqa: E402

BACKGROUND_SIZE = 100
SEED = 42

MODEL_PATH = REPO_ROOT / "outputs" / "models" / "mlp.pt"
SHAP_DIR = REPO_ROOT / "outputs" / "shap"


def main() -> None:
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    # ---- model and data (same preprocessing as in training) ----------------
    model, preprocessor, threshold, _ = load_checkpoint(MODEL_PATH)
    df = build_dataset()
    train, _, test = chronological_split(df)
    X_train = preprocessor.transform(train)
    X_test = preprocessor.transform(test)

    def predict_logit(z: np.ndarray) -> np.ndarray:
        """numpy (n, 7) -> numpy (n,): explained output (logit)."""
        with torch.no_grad():
            return model(torch.from_numpy(z.astype(np.float32))).numpy().astype(np.float64).ravel()

    # ---- background: 100 random training days -------------------------------
    background, background_idx = sample_background(X_train, n=BACKGROUND_SIZE, seed=SEED)
    np.savez(SHAP_DIR / "background.npz", X=background, train_row=background_idx,
             dates=train["MESS_DATUM"].to_numpy()[background_idx].astype("datetime64[D]").astype(str),
             cities=train["city"].to_numpy()[background_idx].astype(str))
    print(f"Background: {BACKGROUND_SIZE} training days, "
          f"{int(train[TARGET].to_numpy()[background_idx].sum())} of them before a heat day")

    # ---- exact Shapley values for all test days ------------------------------
    n_evals = len(X_test) * 2 ** X_test.shape[1] * BACKGROUND_SIZE
    print(f"Computing exact Shapley values: {len(X_test)} days x 128 coalitions x "
          f"{BACKGROUND_SIZE} background days = {n_evals:,} model evaluations ...")
    start = time.perf_counter()
    phi, base_value = exact_shapley(predict_logit, X_test, background)
    seconds = time.perf_counter() - start
    print(f"Done in {seconds:.1f} s")

    # ---- checks ------------------------------------------------------------------
    logits = predict_logit(X_test)
    gap = np.abs(base_value + phi.sum(axis=1) - logits).max()
    print(f"\nBase value (mean logit over background): {base_value:.3f}")
    print(f"Efficiency check, max |base + sum(phi) - logit| over all test days: {gap:.2e}")

    # ---- save ---------------------------------------------------------------------
    np.savez(
        SHAP_DIR / "exact_logit.npz",
        phi=phi, base_value=base_value, logit=logits, threshold=threshold,
        X_scaled=X_test, X_raw=test[FEATURE_COLUMNS].to_numpy(dtype=float),
        y=test[TARGET].to_numpy(), dates=test["MESS_DATUM"].to_numpy().astype("datetime64[D]").astype(str),
        cities=test["city"].to_numpy().astype(str), features=np.array(FEATURE_COLUMNS),
        seconds=seconds,
    )
    summary = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "label_de": [FEATURE_LABELS_DE[c] for c in FEATURE_COLUMNS],
        "mean_abs_phi": np.abs(phi).mean(axis=0),
        "mean_phi": phi.mean(axis=0),
    }).sort_values("mean_abs_phi", ascending=False)
    summary.to_csv(SHAP_DIR / "exact_summary.csv", index=False)

    print("\nMean |phi| per feature on the test days (logit units):")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    temperature = ["TMK", "TXK", "TNK", "TXK_lag_1", "TXK_rolling_3"]
    temp_idx = [FEATURE_COLUMNS.index(c) for c in temperature]
    print(f"\nSum of the 5 temperature features, mean |sum phi|: "
          f"{np.abs(phi[:, temp_idx].sum(axis=1)).mean():.3f}")
    print(f"\nSaved to {SHAP_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
