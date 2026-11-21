"""Figures of the trained MLP itself for the final slides (network with activations, first-layer weights).

Uses only the trained model and two test days; no Shapley computation.

Run from the repository root (after Steps 3 and 5):
    python scripts/mlp_figures.py

Writes outputs/figures/slides/mlp_network_two_days.png and mlp_layer1_weights.png,
and prints the numbers shown on the slides.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import FEATURE_COLUMNS  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.plot_style import DARK, GREEN, GREY, apply_style  # noqa: E402
from src.plots import UNITS  # noqa: E402

OUT = REPO_ROOT / "outputs" / "figures" / "slides"
SHORT = {"TMK": "Mittel-T.", "TXK": "Max.-T.", "TNK": "Min.-T.", "VPM": "Dampfdr.", "UPM": "Feuchte",
         "TXK_lag_1": "Vortag", "TXK_rolling_3": "3-Tage"}
DAYS = [("2025-01-15", "Muenchen", "Wintertag: München, 15.01.2025"),
        ("2025-06-29", "Muenchen", "Sommertag: München, 29.06.2025")]
ACTIVE = LinearSegmentedColormap.from_list("active", ["#FFFFFF", GREEN, "#4F7A0B"])


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, facecolor="white", dpi=300)
    plt.close(fig)
    Image.open(OUT / name).convert("RGB").save(OUT / name)
    print(f"  {name}")


def forward(model, pre, raw):
    x = torch.tensor(pre.transform(pd.DataFrame([raw], columns=FEATURE_COLUMNS)), dtype=torch.float32)
    with torch.no_grad():
        h1 = torch.relu(model.body[0](x))[0].numpy()
        h2 = model.body(x)[0].numpy()
        z = float(model(x)[0, 0])
    return h1, h2, z


def draw_network(ax, model, raw, h1, h2, z, title):
    layers = [7, 16, 8, 1]
    xs = [0.0, 1.0, 2.0, 2.85]
    ys = [np.linspace(0.9, 0.1, n) if n > 1 else np.array([0.5]) for n in layers]
    weights = [model.body[0].weight.detach().numpy(), model.body[2].weight.detach().numpy(),
               model.head.weight.detach().numpy()]
    acts = [None, h1, h2]
    # edges: only those that carry signal on this day (source active), width ~ |w|
    for li, W in enumerate(weights):
        src_on = np.ones(layers[li], bool) if li == 0 else acts[li] > 0
        for j in range(W.shape[0]):
            for i in range(W.shape[1]):
                on = src_on[i]
                ax.plot([xs[li], xs[li + 1]], [ys[li][i], ys[li + 1][j]], lw=0.25 + 0.9 * min(abs(W[j, i]), 1.2),
                        color=GREY if on else "#E4E4E4", alpha=0.55 if on else 0.35, zorder=1)
    # nodes
    ax.scatter(np.full(7, xs[0]), ys[0], s=90, c="#DCEBF0", edgecolors=DARK, linewidths=0.8, zorder=3)
    for li, h in ((1, h1), (2, h2)):
        ax.scatter(np.full(len(h), xs[li]), ys[li], s=80 if li == 1 else 130, c=h, cmap=ACTIVE,
                   vmin=0, vmax=max(h.max(), 1e-6), edgecolors=DARK, linewidths=0.8, zorder=3)
    ax.scatter([xs[3]], [0.5], s=260, c=GREEN if z > 0 else "#FFFFFF", edgecolors=DARK, linewidths=1.0, zorder=3)
    for c, y, v in zip(FEATURE_COLUMNS, ys[0], raw):
        ax.text(xs[0] - 0.17, y, f"{SHORT[c]} {v:.1f}".replace(" -", " −") + f" {UNITS[c]}", ha="right",
                va="center", fontsize=11)
    for k, y in enumerate(ys[2]):
        ax.text(xs[2] + 0.15, y, f"N{k}", fontsize=9, color=GREY, ha="left", va="center")
    p = 1 / (1 + np.exp(-z))
    ax.text(xs[3] + 0.2, 0.5, (f"z = {z:.1f}\np = {p:.2f}" if p > 0.01 else f"z = {z:.1f}\np ≈ 0").replace("-", "−"),
            ha="left", va="center", fontsize=13, fontweight="bold")
    ax.text(xs[1], -0.02, f"{int((h1 > 0).sum())}/16 aktiv", ha="center", va="top", fontsize=11)
    ax.text(xs[2], -0.02, f"{int((h2 > 0).sum())}/8 aktiv", ha="center", va="top", fontsize=11)
    ax.text(-1.0, 1.08, title, fontsize=13, fontweight="bold", ha="left", va="bottom")
    for x, t in zip(xs, ["Eingabe", "Schicht 1", "Embedding", "Logit"]):
        ax.text(x, 0.97, t, ha="center", va="bottom", fontsize=10, color=GREY)
    ax.set_xlim(-1.0, 3.75)
    ax.set_ylim(-0.1, 1.12)
    ax.axis("off")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    model, pre, _, info = load_checkpoint(REPO_ROOT / "outputs" / "models" / "mlp.pt")
    exact = np.load(REPO_ROOT / "outputs" / "shap" / "exact_logit.npz", allow_pickle=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 2.65))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.07, wspace=0.42)
    for ax, (date, city, title) in zip(axes, DAYS):
        i = int(np.flatnonzero((exact["dates"] == date) & (exact["cities"] == city))[0])
        raw = exact["X_raw"][i]
        h1, h2, z = forward(model, pre, raw)
        draw_network(ax, model, raw, h1, h2, z, title)
        print(f"{title}: layer 1 active {int((h1 > 0).sum())}/16, embedding {np.round(h2, 2)}, "
              f"logit {z:.2f}, p {1 / (1 + np.exp(-z)):.3g}")
    save(fig, "mlp_network_two_days.png")

    W1 = model.body[0].weight.detach().numpy()
    fig, ax = plt.subplots(figsize=(3.65, 2.95), layout="constrained")
    lim = np.abs(W1).max()
    im = ax.imshow(W1, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(7), [SHORT[c] for c in FEATURE_COLUMNS], rotation=90, fontsize=11)
    ax.set_yticks(range(0, 16, 3), [f"n{k}" for k in range(0, 16, 3)], fontsize=10)
    ax.set_ylabel("Neuron in Schicht 1", fontsize=11)
    for j in range(7):
        pos, neg = int((W1[:, j] > 0).sum()), int((W1[:, j] < 0).sum())
        ax.text(j, -0.75, f"+{pos}\n−{neg}", ha="center", va="bottom", fontsize=9, linespacing=1.0)
    ax.set_ylim(15.5, -2.6)
    ax.spines[:].set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.ax.tick_params(labelsize=10)
    cb.set_label("Gewicht", fontsize=11)
    save(fig, "mlp_layer1_weights.png")
    print("positive/negative first-layer weights per feature:",
          {c: (int((W1[:, j] > 0).sum()), int((W1[:, j] < 0).sum())) for j, c in enumerate(FEATURE_COLUMNS)})
    print("parameters:", info["parameters"])


if __name__ == "__main__":
    main()
