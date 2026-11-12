"""Step 10: result figures at the size they have on the seminar slides (inches as on the slide, 9-11 pt).

Reads the saved results of Steps 3-9; only the slide 38 explanation is recomputed.

Run from the repository root (after Steps 3-9):
    python scripts/slide_figures.py

Writes outputs/figures/slides/sNN_*.png  (NN = slide number in Shapley_MLP_Vortrag)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import FEATURE_COLUMNS, FEATURE_LABELS_DE, TARGET, build_dataset, chronological_split  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.plot_style import DARK, GREEN, GREY, TEAL, apply_style  # noqa: E402
from src.plots import CITY_NAMES_DE, UNITS, UP, waterfall  # noqa: E402
from src.shapley import exact_shapley, sample_background  # noqa: E402

SHAP_DIR = REPO_ROOT / "outputs" / "shap"
OUT = REPO_ROOT / "outputs" / "figures" / "slides"
LABELS = [FEATURE_LABELS_DE[c] for c in FEATURE_COLUMNS]
SHORT = {"TMK": "Mittel-T.", "TXK": "Max.-T.", "TNK": "Min.-T.", "VPM": "Dampfdr.", "UPM": "Feuchte",
         "TXK_lag_1": "Vortag", "TXK_rolling_3": "3-Tage"}
MONTHS = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]


def slide_style() -> None:
    apply_style()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "savefig.dpi": 300,
                         "savefig.bbox": "standard", "savefig.pad_inches": 0})


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, facecolor="white")
    plt.close(fig)
    # RGB without alpha channel
    Image.open(OUT / name).convert("RGB").save(OUT / name)
    print(f"  {name}")


def raw_labels(raw):
    return [f"{FEATURE_LABELS_DE[c]} = {v:.1f} {UNITS[c]}" for c, v in zip(FEATURE_COLUMNS, raw)]


def find_day(exact, date, city):
    return int(np.flatnonzero((exact["dates"] == date) & (exact["cities"] == city))[0])


def main() -> None:
    slide_style()
    OUT.mkdir(parents=True, exist_ok=True)
    exact = np.load(SHAP_DIR / "exact_logit.npz")
    phi, base = exact["phi"], float(exact["base_value"])
    tp_day = find_day(exact, "2025-06-29", "Muenchen")      # detected heat day (slide 22)
    fp_day = find_day(exact, "2025-07-02", "Frankfurt")     # most confident false alarm (slide 22)
    df = build_dataset()
    train, _, _ = chronological_split(df)

    # ---- slide 6: heat days per month, correlation ---------------------------------
    month = (df["MESS_DATUM"] + pd.Timedelta(days=1)).dt.month
    rate = df.groupby(month)[TARGET].mean().reindex(range(1, 13), fill_value=0) * 100
    fig, ax = plt.subplots(figsize=(2.9, 3.85), layout="constrained")
    rows = np.arange(12)[::-1]
    ax.barh(rows, rate.values, color=[GREEN if m in (6, 7, 8) else GREY for m in range(1, 13)], height=0.7)
    for r, v in zip(rows, rate.values):
        if v >= 0.5:
            ax.text(v + 0.3, r, f"{v:.0f} %", va="center", fontsize=9)
    ax.set_yticks(rows, MONTHS)
    ax.set_xlim(0, rate.max() * 1.35)
    ax.set_xlabel("Hitzetage in %")
    ax.set_title("Hitzetage pro Monat")
    save(fig, "s06_heat_days_per_month.png")

    corr = train[FEATURE_COLUMNS].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(2.9, 3.85), layout="constrained")
    ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    short = [SHORT[c] for c in FEATURE_COLUMNS]
    ax.set_xticks(range(7), short, rotation=90)
    ax.set_yticks(range(7), short)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{corr[i, j]:.1f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center",
                    fontsize=7.5, color="white" if abs(corr[i, j]) > 0.6 else DARK)
    ax.spines[:].set_visible(False)
    ax.set_title("Korrelation (Training)")
    save(fig, "s06_correlation.png")

    # ---- slide 15: the real example, logit units ------------------------------------
    top = ["TXK", "TMK", "VPM"]
    idx = [FEATURE_COLUMNS.index(c) for c in top]
    rest = [i for i in range(7) if i not in idx]
    values = np.array([phi[tp_day, i] for i in idx] + [phi[tp_day, rest].sum()])
    names = [FEATURE_LABELS_DE[c] for c in top] + ["Rest (4 Merkmale)"]
    fig, ax = plt.subplots(figsize=(5.6, 2.65), layout="constrained")
    waterfall(ax, values, base, np.zeros(4), fontsize=10, labels=names)
    ax.set_ylim(-0.6, 4.6)
    save(fig, "s15_example_waterfall.png")

    # ---- slide 21: approximations vs exact --------------------------------------------
    kernel = np.load(SHAP_DIR / "kernel_logit.npz")["phi"]
    deep = np.load(SHAP_DIR / "deep_logit.npz")["phi"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.85), layout="constrained", sharey=True)
    lim = [min(phi.min(), deep.min()) - 0.5, max(phi.max(), deep.max()) + 0.5]
    for ax, values, name, color in [(axes[0], kernel, "KernelSHAP", GREEN), (axes[1], deep, "DeepSHAP", TEAL)]:
        ax.plot(lim, lim, color=GREY, linewidth=1, zorder=0)
        ax.scatter(phi.ravel(), values.ravel(), s=2, alpha=0.35, color=color, linewidths=0)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_xlabel("φ exakt")
    axes[0].set_ylabel("φ Näherung")
    save(fig, "s21_comparison_scatter.png")

    # ---- slide 22: two days -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.05, 3.3), layout="constrained")
    for ax, day, title in [(axes[0], tp_day, "Hitzetag, richtig erkannt"), (axes[1], fp_day, "Fehlalarm")]:
        waterfall(ax, phi[day], base, exact["X_raw"][day], fontsize=9,
                  title=f"{title}: {CITY_NAMES_DE[str(exact['cities'][day])]}, {exact['dates'][day]}")
        ax.tick_params(axis="y", labelsize=8.5)
    save(fig, "s22_waterfalls.png")

    # ---- slide 23: beeswarm + temperature group ----------------------------------------
    explanation = shap.Explanation(values=phi, base_values=np.full(len(phi), base), data=exact["X_raw"],
                                   feature_names=LABELS)
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=7, show=False, plot_size=(5.0, 3.85),
                        color_bar_label="Merkmalswert", s=6)
    fig = plt.gcf()
    for ax in fig.axes:
        ax.tick_params(labelsize=9)
    fig.axes[0].set_xlabel("Shapley-Wert φ (Logit)", fontsize=10)
    fig.axes[-1].set_yticks([0, 1], ["niedrig", "hoch"])
    fig.axes[-1].set_ylabel("Merkmalswert", fontsize=9)
    fig.set_size_inches(5.0, 3.85)
    fig.subplots_adjust(left=0.30, right=0.90, top=0.97, bottom=0.14)
    save(fig, "s23_beeswarm.png")

    temperature = [FEATURE_COLUMNS.index(c) for c in ["TMK", "TXK", "TNK", "TXK_lag_1", "TXK_rolling_3"]]
    groups = ["Temperatur (Summe)", "Max.-Temp. (einzeln)", "Dampfdruck", "rel. Feuchte"]
    values = [np.abs(phi[:, temperature].sum(1)).mean(), np.abs(phi[:, 1]).mean(),
              np.abs(phi[:, 3]).mean(), np.abs(phi[:, 4]).mean()]
    fig, ax = plt.subplots(figsize=(3.8, 1.95), layout="constrained")
    rows = np.arange(4)[::-1]
    ax.barh(rows, values, color=[GREEN, "#B8D67A", GREY, GREY], height=0.65)
    for r, v in zip(rows, values):
        ax.text(v + 0.1, r, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(rows, groups)
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_xlabel("mittleres |φ| (Logit)", fontsize=9)
    ax.set_title("Temperatur als Gruppe", fontsize=10)
    save(fig, "s23_temperature_group.png")

    # ---- slide 24: embeddings --------------------------------------------------------------
    emb = np.load(SHAP_DIR / "embedding_features.npz")
    neurons = pd.read_csv(SHAP_DIR / "embedding_neurons.csv")
    order = np.argsort(neurons.mean_abs_phi_logit.to_numpy())[::-1]
    names = [f"N{j}" for j in order]
    matrix = np.abs(emb["phi"]).mean(axis=0)[:, order]
    fig, ax = plt.subplots(figsize=(4.45, 2.6), layout="constrained")
    ax.imshow(matrix, cmap="Greens", aspect="auto")
    ax.set_xticks(range(8), names)
    ax.set_yticks(range(7), [SHORT[c] for c in FEATURE_COLUMNS])
    for i in range(7):
        for k in range(8):
            ax.text(k, i, f"{matrix[i, k]:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if matrix[i, k] > 0.6 * matrix.max() else DARK)
    ax.spines[:].set_visible(False)
    ax.set_title("Merkmale → Neuronen (mittl. |φ|)")
    save(fig, "s24_features_to_neurons.png")

    weights = neurons.weight_to_logit.to_numpy()
    values = neurons.mean_abs_phi_logit.to_numpy()[order]
    unused = neurons.mean_abs_phi_logit.to_numpy() < 0.05 * neurons.mean_abs_phi_logit.max()
    colors = [GREY if unused[j] else (UP if weights[j] > 0 else TEAL) for j in order]
    fig, ax = plt.subplots(figsize=(4.45, 2.6), layout="constrained")
    ax.bar(names, values, color=colors)
    for k, j in enumerate(order):
        ax.text(k, values[k] + 0.06, f"{weights[j]:+.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, values.max() * 1.22)
    ax.set_ylabel("mittl. |φ| (Logit)")
    ax.set_title("Neuronen → Logit (Zahl: Gewicht w)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (UP, TEAL)]
    ax.legend(handles, ["w > 0: pro Hitzetag", "w < 0: gegen Hitzetag"], frameon=False, fontsize=8,
              loc="upper right")
    save(fig, "s24_neurons_to_logit.png")

    # ---- slide 25: runtime -----------------------------------------------------------------------
    runtime = pd.read_csv(REPO_ROOT / "outputs" / "runtime" / "runtime.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.05, 2.65), layout="constrained")
    colors = {"exakt": DARK, "KernelSHAP": GREEN, "DeepSHAP": TEAL}
    for ax, experiment, column, xlabel, log_x in [
        (axes[0], "A_days", "days", "erklärte Tage (Hintergrund 100)", True),
        (axes[1], "B_background", "background", "Hintergrundtage (100 Tage)", True),
        (axes[2], "C_features", "features", "Merkmale M (Zufallsnetze)", False),
    ]:
        part = runtime[runtime.experiment == experiment]
        for name, color in colors.items():
            line = part[part.method == name]
            ax.plot(line[column], line["explain_seconds"], marker="o", markersize=4, color=color, label=name)
        if log_x:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = sorted(int(v) for v in part[column].unique())
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel(xlabel)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Laufzeit in s")
    axes[2].axvline(7, color=DARK, linestyle=":", linewidth=1)
    axes[2].text(7.3, 1.5, "unser\nModell", fontsize=8, va="top")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "s25_runtime.png")

    # ---- slide 36: KernelSHAP convergence -----------------------------------------------------------
    conv = pd.read_csv(SHAP_DIR / "kernel_convergence.csv")
    fig, ax = plt.subplots(figsize=(5.0, 3.85), layout="constrained")
    ax.plot(conv.nsamples, conv.mean_abs_deviation, marker="o", color=GREEN, label="mittlere Abweichung")
    ax.plot(conv.nsamples, conv.max_abs_deviation, marker="o", color=DARK, label="maximale Abweichung")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(conv.nsamples, [str(n) for n in conv.nsamples])
    ax.minorticks_off()
    ax.set_xlabel("ausgewertete Koalitionen (von 128)")
    ax.set_ylabel("Abweichung zu exakt (Logit)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    save(fig, "s36_kernel_convergence.png")

    # ---- slide 38: same day, two backgrounds (one exact explanation recomputed) ----------------------
    model, preprocessor, _, _ = load_checkpoint(REPO_ROOT / "outputs" / "models" / "mlp.pt")

    def predict_logit(z):
        with torch.no_grad():
            return model(torch.from_numpy(z.astype(np.float32))).numpy().astype(np.float64).ravel()

    X_train = preprocessor.transform(train)
    summer = train["MESS_DATUM"].dt.month.isin([6, 7, 8]).to_numpy()
    backgrounds = {"Hintergrund: ganzes Jahr": np.load(SHAP_DIR / "background.npz")["X"],
                   "Hintergrund: nur Sommer": sample_background(X_train[summer], n=100, seed=42)[0]}
    fig, axes = plt.subplots(1, 2, figsize=(9.05, 3.3), layout="constrained")
    for ax, (title, background) in zip(axes, backgrounds.items()):
        values, b = exact_shapley(predict_logit, exact["X_scaled"][[tp_day]], background)
        waterfall(ax, values[0], b, exact["X_raw"][tp_day], fontsize=9, title=title)
        ax.tick_params(axis="y", labelsize=8.5)
    save(fig, "s38_background_waterfalls.png")
    print(f"Saved to {OUT.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
