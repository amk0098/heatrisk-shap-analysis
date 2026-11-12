"""Step 7: Shapley values for the embeddings (8 neurons of the last hidden layer), slide 24.

Two views, both with the same 100 background training days as Steps 5-6:
    (a) features -> embedding:  the 7 input features are the players, each of the 8 neurons is explained
                                ("What has each neuron learned to react to?")
    (b) embedding -> logit:     the 8 neuron values are the players, the logit is explained
                                ("Which neurons drive the decision?")
Each view: exact values (own code) + KernelSHAP + DeepSHAP, compared with each other.

Run from the repository root (after scripts/exact_shapley.py):
    python scripts/shap_embeddings.py

Writes:
    outputs/shap/embedding_features.npz      (a) phi: days x 7 features x 8 neurons
    outputs/shap/embedding_logit.npz         (b) phi: days x 8 neurons
    outputs/shap/embedding_neurons.csv       one row per neuron: weight, activity, importance, top feature
    outputs/shap/embedding_comparison.csv    deviation of KernelSHAP / DeepSHAP from exact, both views
    outputs/figures/embeddings.png           slide 24: heatmap (a) + bars (b)
"""

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import FEATURE_COLUMNS, FEATURE_LABELS_DE  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.plot_style import DARK, GREY, TEAL, apply_style  # noqa: E402
from src.plots import UP  # noqa: E402
from src.shapley import exact_shapley, exact_shapley_multi  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning, module="shap")
logging.getLogger("shap").setLevel(logging.ERROR)

SHAP_DIR = REPO_ROOT / "outputs" / "shap"
FIGURE_DIR = REPO_ROOT / "outputs" / "figures"


def deviation(name, view, phi, phi_exact):
    dev = np.abs(phi - phi_exact)
    return {"view": view, "method": name, "max_abs_deviation": dev.max(), "mean_abs_deviation": dev.mean()}


def main() -> None:
    apply_style()
    model, _, threshold, _ = load_checkpoint(REPO_ROOT / "outputs" / "models" / "mlp.pt")
    exact = np.load(SHAP_DIR / "exact_logit.npz")
    X = exact["X_scaled"]
    background = np.load(SHAP_DIR / "background.npz")["X"]
    y = exact["y"].astype(bool)

    def embed(z: np.ndarray) -> np.ndarray:
        """input features (n, 7) -> embedding (n, 8)"""
        with torch.no_grad():
            return model.embed(torch.from_numpy(z.astype(np.float32))).numpy().astype(np.float64)

    def head(h: np.ndarray) -> np.ndarray:
        """embedding (n, 8) -> logit (n,)"""
        with torch.no_grad():
            return model.head(torch.from_numpy(h.astype(np.float32))).numpy().astype(np.float64).ravel()

    H = embed(X)                       # embeddings of the test days
    H_background = embed(background)   # embeddings of the 100 background days
    comparison = []

    # =========================================================================
    # (a) features -> each of the 8 embedding neurons
    # =========================================================================
    phi_a, base_a = exact_shapley_multi(embed, X, background)          # (days, 7, 8), (8,)
    gap_a = np.abs(base_a + phi_a.sum(axis=1) - H).max()
    print(f"(a) features -> neurons: efficiency gap {gap_a:.1e}")

    kernel_a = shap.KernelExplainer(embed, background)
    phi_kernel_a = np.asarray(kernel_a.shap_values(X, silent=True))     # (days, 7, 8)
    deep_a = shap.DeepExplainer(model.body, torch.from_numpy(background.astype(np.float32)))
    phi_deep_a = np.asarray(deep_a.shap_values(torch.from_numpy(X.astype(np.float32))))
    comparison += [deviation("KernelSHAP", "a", phi_kernel_a, phi_a), deviation("DeepSHAP", "a", phi_deep_a, phi_a)]
    np.savez(SHAP_DIR / "embedding_features.npz", phi=phi_a, base_values=base_a, embedding=H,
             phi_kernel=phi_kernel_a, phi_deep=phi_deep_a)

    # =========================================================================
    # (b) 8 embedding neurons -> logit
    # =========================================================================
    phi_b, base_b = exact_shapley(head, H, H_background)                # 2^8 = 256 coalitions
    weights = model.head.weight.detach().numpy().ravel()
    closed_form = weights * (H - H_background.mean(axis=0))             # last layer is linear
    print(f"(b) neurons -> logit: base value {base_b:.2f} (same as for the input features: "
          f"{float(exact['base_value']):.2f})")
    print(f"    exact vs closed form w_j (h_j - mean h_j): max difference {np.abs(phi_b - closed_form).max():.1e}")

    kernel_b = shap.KernelExplainer(head, H_background)
    phi_kernel_b = np.asarray(kernel_b.shap_values(H, silent=True))
    deep_b = shap.DeepExplainer(model.head, torch.from_numpy(H_background.astype(np.float32)))
    phi_deep_b = np.asarray(deep_b.shap_values(torch.from_numpy(H.astype(np.float32)))).reshape(len(H), -1)
    comparison += [deviation("KernelSHAP", "b", phi_kernel_b, phi_b), deviation("DeepSHAP", "b", phi_deep_b, phi_b)]
    np.savez(SHAP_DIR / "embedding_logit.npz", phi=phi_b, base_value=base_b, weights=weights,
             phi_kernel=phi_kernel_b, phi_deep=phi_deep_b)

    comparison = pd.DataFrame(comparison)
    comparison.to_csv(SHAP_DIR / "embedding_comparison.csv", index=False)
    print("\nDeviation from the exact values:")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.2e}"))

    # =========================================================================
    # one row per neuron: what does it do?
    # =========================================================================
    labels = [FEATURE_LABELS_DE[c] for c in FEATURE_COLUMNS]
    importance_a = np.abs(phi_a).mean(axis=0)                           # (7, 8)
    neurons = pd.DataFrame({
        "neuron": np.arange(8),
        "weight_to_logit": weights,
        "active_all_days": (H > 0).mean(axis=0),
        "active_before_heat_days": (H[y] > 0).mean(axis=0),
        "mean_abs_phi_logit": np.abs(phi_b).mean(axis=0),
        "top_feature": [labels[i] for i in importance_a.argmax(axis=0)],
        "top_feature_share": importance_a.max(axis=0) / np.maximum(importance_a.sum(axis=0), 1e-12),
    })
    neurons["role"] = np.where(neurons.weight_to_logit > 0, "pro Hitzetag", "gegen Hitzetag")
    neurons.loc[neurons.mean_abs_phi_logit < 0.05 * neurons.mean_abs_phi_logit.max(), "role"] = "kaum genutzt"
    neurons.to_csv(SHAP_DIR / "embedding_neurons.csv", index=False)
    print("\nNeurons (test days):")
    print(neurons.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    # =========================================================================
    # figure for slide 24: (a) heatmap, (b) bars, neurons in the same order
    # =========================================================================
    order = np.argsort(neurons.mean_abs_phi_logit.to_numpy())[::-1]     # most important neuron first
    neuron_names = [f"N{j}" for j in order]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})

    matrix = importance_a[:, order]
    image = ax_a.imshow(matrix, cmap="Greens", aspect="auto")
    ax_a.set_xticks(range(8), neuron_names)
    ax_a.set_yticks(range(7), labels)
    for i in range(7):
        for k in range(8):
            ax_a.text(k, i, f"{matrix[i, k]:.1f}", ha="center", va="center", fontsize=11,
                      color="white" if matrix[i, k] > 0.6 * matrix.max() else DARK)
    ax_a.spines[:].set_visible(False)
    ax_a.set_title("(a) Merkmale → Embedding-Neuronen\nmittleres |φ|")
    fig.colorbar(image, ax=ax_a, shrink=0.8)

    values = neurons.mean_abs_phi_logit.to_numpy()[order]
    colors = [GREY if neurons.role[j] == "kaum genutzt" else (UP if weights[j] > 0 else TEAL) for j in order]
    ax_b.bar(neuron_names, values, color=colors)
    for k, j in enumerate(order):
        ax_b.text(k, values[k] + 0.05, f"w={weights[j]:+.2f}", ha="center", va="bottom", fontsize=10, color=DARK)
    ax_b.set_ylabel("mittleres |φ| (Logit)")
    ax_b.set_title("(b) Embedding-Neuronen → Logit")
    ax_b.set_ylim(0, values.max() * 1.18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (UP, TEAL, GREY)]
    ax_b.legend(handles, ["pro Hitzetag (w > 0)", "gegen Hitzetag (w < 0)", "kaum genutzt"], frameon=False,
                loc="upper right", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "embeddings.png")
    plt.close(fig)
    print("\nSaved outputs/shap/embedding_*.npz/.csv and outputs/figures/embeddings.png")


if __name__ == "__main__":
    main()
