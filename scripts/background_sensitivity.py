"""Step 9: how much do the Shapley values depend on the background data? (backup slide 38)

All backgrounds come from the TRAINING period; values are exact (own code, 128 coalitions).
    Ganzes Jahr   100 random training days (the default, same as Steps 5-6)
    Nur Sommer    100 random training days from June-August
    Mittelwert    one single reference day = training mean of every feature
plus: 10 different random draws of "Ganzes Jahr" (seeds 0-9) to see the sampling noise.

Run from the repository root:
    python scripts/background_sensitivity.py

Writes:
    outputs/shap/background_sensitivity.csv      base value and mean |phi| per feature and background
    outputs/figures/background_waterfalls.png    slide 38: the same day with two backgrounds
    outputs/figures/background_ranking.png       mean |phi| per feature for the three backgrounds
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import FEATURE_COLUMNS, FEATURE_LABELS_DE, build_dataset, chronological_split  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.plot_style import DARK, GREEN, GREY, apply_style  # noqa: E402
from src.plots import CITY_NAMES_DE, waterfall  # noqa: E402
from src.shapley import exact_shapley, sample_background  # noqa: E402

SEED = 42
N_BACKGROUND = 100
SEEDS_FOR_NOISE = range(10)
SHAP_DIR = REPO_ROOT / "outputs" / "shap"
FIGURE_DIR = REPO_ROOT / "outputs" / "figures"


def main() -> None:
    apply_style()
    model, preprocessor, threshold, _ = load_checkpoint(REPO_ROOT / "outputs" / "models" / "mlp.pt")
    train, _, test = chronological_split(build_dataset())
    X_train = preprocessor.transform(train)
    exact = np.load(SHAP_DIR / "exact_logit.npz")
    X_test = exact["X_scaled"]

    def predict_logit(z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(z.astype(np.float32))).numpy().astype(np.float64).ravel()

    # ---- the three backgrounds -------------------------------------------------
    summer = train["MESS_DATUM"].dt.month.isin([6, 7, 8]).to_numpy()
    summer_bg, _ = sample_background(X_train[summer], n=N_BACKGROUND, seed=SEED)
    backgrounds = {
        "Ganzes Jahr": np.load(SHAP_DIR / "background.npz")["X"],
        "Nur Sommer": summer_bg,
        "Mittelwert": np.zeros((1, X_train.shape[1])),   # standardised data: training mean = 0
    }

    results, rows = {}, []
    for name, background in backgrounds.items():
        phi, base = exact_shapley(predict_logit, X_test, background)
        results[name] = (phi, base)
        row = {"background": name, "size": len(background), "base_value": base}
        row.update({FEATURE_LABELS_DE[c]: np.abs(phi[:, i]).mean() for i, c in enumerate(FEATURE_COLUMNS)})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(SHAP_DIR / "background_sensitivity.csv", index=False)
    print("Base value and mean |phi| per feature (test days, logit units):")
    print(table.set_index("background").T.to_string(float_format=lambda v: f"{v:.2f}"))

    # ---- sampling noise: 10 different random draws of 100 training days -------------
    draws = np.stack([exact_shapley(predict_logit, X_test, sample_background(X_train, N_BACKGROUND, seed=s)[0])[0]
                      for s in SEEDS_FOR_NOISE])                      # (10, days, 7)
    noise = draws.std(axis=0).mean() / np.abs(results["Ganzes Jahr"][0]).mean()
    print(f"\nSampling noise of a random 100-day background (10 draws): "
          f"std of phi = {100 * noise:.1f} % of the mean |phi|")

    # ---- the day for slide 38: the detected heat day from slide 22 --------------------
    # same rule as in scripts/shap_logit.py: true positive with the median logit
    logits = exact["logit"]
    predicted = logits >= np.log(threshold / (1 - threshold))
    tp = np.flatnonzero(predicted & exact["y"].astype(bool))
    day = tp[np.argsort(logits[tp])[len(tp) // 2]]
    city = CITY_NAMES_DE[str(exact["cities"][day])]
    print(f"\nDay for slide 38: {city}, {exact['dates'][day]} (logit {logits[day]:.2f})")
    for name in ["Ganzes Jahr", "Nur Sommer"]:
        phi, base = results[name]
        print(f"  {name:12s} base {base:7.2f}  " +
              ", ".join(f"{FEATURE_LABELS_DE[c]} {v:+.2f}" for c, v in zip(FEATURE_COLUMNS, phi[day])))

    # ---- figure: two waterfalls of the same day (slide 38) ------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, name in zip(axes, ["Ganzes Jahr", "Nur Sommer"]):
        phi, base = results[name]
        waterfall(ax, phi[day], base, exact["X_raw"][day], title=f"Hintergrund: {name}")
    fig.suptitle(f"{city}, {exact['dates'][day]}: gleicher Tag, gleiches Logit", y=1.02, fontsize=16, color=DARK)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "background_waterfalls.png")
    plt.close(fig)

    # ---- figure: global ranking per background ---------------------------------------------
    labels = [FEATURE_LABELS_DE[c] for c in FEATURE_COLUMNS]
    order = np.argsort(np.abs(results["Ganzes Jahr"][0]).mean(0))[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.27
    for k, (name, color) in enumerate([("Ganzes Jahr", DARK), ("Nur Sommer", GREEN), ("Mittelwert", GREY)]):
        values = np.abs(results[name][0]).mean(0)[order]
        ax.bar(np.arange(7) + (k - 1) * width, values, width, label=name, color=color)
    ax.set_xticks(range(7), [labels[i] for i in order], rotation=30, ha="right")
    ax.set_ylabel("mittleres |φ| (Logit)")
    ax.legend(frameon=False, title="Hintergrund")
    fig.savefig(FIGURE_DIR / "background_ranking.png")
    plt.close(fig)
    print("\nSaved outputs/shap/background_sensitivity.csv and two figures in outputs/figures/")


if __name__ == "__main__":
    main()
