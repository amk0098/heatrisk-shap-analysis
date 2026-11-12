"""Step 6: KernelSHAP and DeepSHAP on the logit, compared with the exact values from Step 5.

Run from the repository root (after scripts/exact_shapley.py):
    python scripts/shap_logit.py

Writes to outputs/shap/:
    kernel_logit.npz, deep_logit.npz       phi for all 3650 test days
    comparison_logit.csv                   deviation from exact per method
    kernel_convergence.csv                 KernelSHAP error vs number of coalitions
and to outputs/figures/:
    comparison_scatter.png      KernelSHAP / DeepSHAP vs exact
    comparison_bars.png         mean |phi| per feature and method
    waterfall_two_days.png      a detected heat day and a false alarm
    beeswarm_test.png           all test days
    group_importance.png        temperature features summed
    kernel_convergence.png      KernelSHAP error vs number of coalitions
"""

import sys
import time
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
from src.plot_style import DARK, GREEN, GREY, TEAL, apply_style  # noqa: E402
from src.plots import CITY_NAMES_DE, deviation_scatter, waterfall  # noqa: E402

# DeprecationWarnings from shap internals
warnings.filterwarnings("ignore", category=DeprecationWarning, module="shap")

SHAP_DIR = REPO_ROOT / "outputs" / "shap"
FIGURE_DIR = REPO_ROOT / "outputs" / "figures"
CONVERGENCE_SAMPLES = [16, 32, 64, 126]     # 126 = all coalitions except empty and full set
CONVERGENCE_DAYS = 300
SEED = 42
TEMPERATURE = ["TMK", "TXK", "TNK", "TXK_lag_1", "TXK_rolling_3"]


def main() -> None:
    apply_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- inputs from Steps 4 and 5 ------------------------------------------
    model, _, threshold, _ = load_checkpoint(REPO_ROOT / "outputs" / "models" / "mlp.pt")
    exact = np.load(SHAP_DIR / "exact_logit.npz")
    background = np.load(SHAP_DIR / "background.npz")["X"].astype(np.float32)
    X = exact["X_scaled"].astype(np.float32)
    phi_exact, base_value, logits = exact["phi"], float(exact["base_value"]), exact["logit"]

    def predict_logit(z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(z.astype(np.float32))).numpy().astype(np.float64).ravel()

    # ---- KernelSHAP -----------------------------------------------------------------
    kernel = shap.KernelExplainer(predict_logit, background)
    start = time.perf_counter()
    phi_kernel = np.asarray(kernel.shap_values(X, silent=True))     # default nsamples = 2*7 + 2048
    kernel_seconds = time.perf_counter() - start

    # ---- DeepSHAP -------------------------------------------------------------------
    deep = shap.DeepExplainer(model, torch.from_numpy(background))
    start = time.perf_counter()
    phi_deep = np.asarray(deep.shap_values(torch.from_numpy(X))).reshape(len(X), -1)
    deep_seconds = time.perf_counter() - start

    np.savez(SHAP_DIR / "kernel_logit.npz", phi=phi_kernel, base_value=kernel.expected_value, seconds=kernel_seconds)
    np.savez(SHAP_DIR / "deep_logit.npz", phi=phi_deep, base_value=float(np.ravel(deep.expected_value)[0]),
             seconds=deep_seconds)

    # ---- comparison with the exact values -----------------------------------------
    rows = []
    for name, phi, seconds in [("KernelSHAP", phi_kernel, kernel_seconds), ("DeepSHAP", phi_deep, deep_seconds)]:
        dev = np.abs(phi - phi_exact)
        worst_day = int(dev.max(axis=1).argmax())
        rows.append({
            "method": name,
            "seconds_all_test_days": round(seconds, 2),
            "max_abs_deviation": dev.max(),
            "mean_abs_deviation": dev.mean(),
            "relative_mean_deviation": dev.mean() / np.abs(phi_exact).mean(),
            "efficiency_gap": np.abs(base_value + phi.sum(axis=1) - logits).max(),
            "same_top_feature_share": (np.abs(phi).argmax(1) == np.abs(phi_exact).argmax(1)).mean(),
            "worst_day": f"{exact['dates'][worst_day]} {exact['cities'][worst_day]}",
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(SHAP_DIR / "comparison_logit.csv", index=False)
    print("Comparison with exact Shapley values (3650 test days, logit units):")
    print(comparison.T.to_string(header=False))

    # ---- KernelSHAP convergence ------------------------------------------------------
    rng = np.random.default_rng(SEED)
    subset = np.sort(rng.choice(len(X), size=CONVERGENCE_DAYS, replace=False))
    conv = []
    for n in CONVERGENCE_SAMPLES:
        np.random.seed(SEED)                       # KernelExplainer draws coalitions with numpy's global RNG
        # l1_reg=False: without it shap adds Lasso feature selection when < 20 % of coalitions are sampled
        phi_n = np.asarray(kernel.shap_values(X[subset], nsamples=n, l1_reg=False, silent=True))
        dev = np.abs(phi_n - phi_exact[subset])
        conv.append({"nsamples": n, "mean_abs_deviation": dev.mean(), "max_abs_deviation": dev.max()})
    conv = pd.DataFrame(conv)
    conv.to_csv(SHAP_DIR / "kernel_convergence.csv", index=False)
    print("\nKernelSHAP error vs number of coalitions (300 test days):")
    print(conv.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    labels = [FEATURE_LABELS_DE[c] for c in FEATURE_COLUMNS]

    # ---- figure: scatter vs exact ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    deviation_scatter(axes[0], phi_exact, phi_kernel, "KernelSHAP", GREEN)
    deviation_scatter(axes[1], phi_exact, phi_deep, "DeepSHAP", TEAL)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "comparison_scatter.png")
    plt.close(fig)

    # ---- figure: mean |phi| per feature and method -----------------------------------------
    order = np.argsort(np.abs(phi_exact).mean(0))[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.27
    for k, (name, phi, color) in enumerate([("exakt", phi_exact, DARK), ("KernelSHAP", phi_kernel, GREEN),
                                            ("DeepSHAP", phi_deep, TEAL)]):
        ax.bar(np.arange(7) + (k - 1) * width, np.abs(phi).mean(0)[order], width, label=name, color=color)
    ax.set_xticks(range(7), [labels[i] for i in order], rotation=30, ha="right")
    ax.set_ylabel("mittleres |φ| (Logit)")
    ax.legend(frameon=False)
    fig.savefig(FIGURE_DIR / "comparison_bars.png")
    plt.close(fig)

    # ---- figure: two waterfalls --------------------------------------------------------------
    logit_threshold = np.log(threshold / (1 - threshold))      # probability 0.9 -> logit 2.20
    predicted = logits >= logit_threshold
    y = exact["y"].astype(bool)
    tp = np.flatnonzero(predicted & y)
    fp = np.flatnonzero(predicted & ~y)
    tp_day = tp[np.argsort(logits[tp])[len(tp) // 2]]     # a typical detected heat day (median logit)
    fp_day = fp[np.argmax(logits[fp])]                    # the most confident false alarm

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, day, title in [(axes[0], tp_day, "Hitzetag, richtig erkannt"), (axes[1], fp_day, "Fehlalarm")]:
        waterfall(ax, phi_exact[day], base_value, exact["X_raw"][day],
                  title=f"{title}: {CITY_NAMES_DE[str(exact['cities'][day])]}, {exact['dates'][day]}")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "waterfall_two_days.png")
    plt.close(fig)
    for name, day in [("detected heat day", tp_day), ("false alarm", fp_day)]:
        p = 1 / (1 + np.exp(-logits[day]))
        print(f"\n{name}: {exact['dates'][day]} {exact['cities'][day]}, logit {logits[day]:.2f} (p = {p:.2f}), "
              f"heat day next = {bool(y[day])}")
        print("   " + ", ".join(f"{FEATURE_LABELS_DE[c]} {v:+.2f}" for c, v in zip(FEATURE_COLUMNS, phi_exact[day])))

    # ---- figure: beeswarm of all test days -----------------------------------------------------
    explanation = shap.Explanation(values=phi_exact, base_values=np.full(len(X), base_value),
                                   data=exact["X_raw"], feature_names=labels)
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=7, show=False, plot_size=(9, 5), color_bar_label="Merkmalswert")
    fig = plt.gcf()
    fig.axes[0].set_xlabel("Shapley-Wert φ (Logit)")
    colorbar_ax = fig.axes[-1]                               # shap draws its colour bar as the last axes
    colorbar_ax.set_yticks([0, 1], ["niedrig", "hoch"])
    plt.savefig(FIGURE_DIR / "beeswarm_test.png")
    plt.close("all")

    # ---- figure: temperature features as one group ---------------------------------------------
    temp_idx = [FEATURE_COLUMNS.index(c) for c in TEMPERATURE]
    other = [i for i in range(7) if i not in temp_idx]
    strongest = temp_idx[int(np.abs(phi_exact[:, temp_idx]).mean(0).argmax())]
    group_names = ["Temperatur (5 Merkmale, Summe)", f"{labels[strongest]} (einzeln)"] + [labels[i] for i in other]
    group_values = ([np.abs(phi_exact[:, temp_idx].sum(1)).mean(), np.abs(phi_exact[:, strongest]).mean()]
                    + [np.abs(phi_exact[:, i]).mean() for i in other])
    colors = [GREEN, "#B8D67A"] + [GREY] * len(other)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    rows = np.arange(len(group_names))[::-1]                   # first entry at the top
    ax.barh(rows, group_values, color=colors)
    ax.set_yticks(rows, group_names)
    for row, value in zip(rows, group_values):
        ax.text(value + 0.08, row, f"{value:.2f}", va="center", fontsize=12)
    single_max = group_values[1]
    ax.set_xlabel("mittleres |φ| (Logit)")
    ax.set_xlim(0, max(group_values) * 1.2)
    fig.savefig(FIGURE_DIR / "group_importance.png")
    plt.close(fig)
    print(f"\nTemperature group: mean |sum phi| {group_values[0]:.2f}  vs. strongest single feature {single_max:.2f}")

    # ---- figure: convergence -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(conv["nsamples"], conv["mean_abs_deviation"], marker="o", color=GREEN, label="mittlere Abw.")
    ax.plot(conv["nsamples"], conv["max_abs_deviation"], marker="o", color=DARK, label="max. Abw.")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(conv["nsamples"], [str(n) for n in conv["nsamples"]])
    ax.set_xlabel("Anzahl Koalitionen (KernelSHAP)")
    ax.set_ylabel("Abweichung zu exakt (Logit)")
    ax.legend(frameon=False)
    fig.savefig(FIGURE_DIR / "kernel_convergence.png")
    plt.close(fig)

    print(f"\nSaved results to {SHAP_DIR.relative_to(REPO_ROOT)}/ and figures to {FIGURE_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
