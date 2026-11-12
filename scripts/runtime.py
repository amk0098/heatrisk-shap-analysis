"""Step 8: runtime of exact Shapley, KernelSHAP and DeepSHAP (slide 25).

Two experiments on the trained MLP (logit output):
    A) number of explained days     1, 10, 100, 1000   (background fixed at 100 days)
    B) background size              10, 50, 100, 500   (explained days fixed at 100)
and one on untrained networks of the same shape with more inputs:
    C) number of features M         4, 6, ..., 16      (10 days, background 20), to show the 2^M growth
       of the exact computation, which the 7-feature model cannot show.

Run from the repository root:
    python scripts/runtime.py

Writes:
    outputs/runtime/runtime.csv        one row per method x setting (median of repeats)
    outputs/runtime/hardware.txt       machine the numbers were measured on
    outputs/figures/runtime.png        slide 25
"""

import logging
import os
import platform
import subprocess
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

from src.data import build_dataset, chronological_split  # noqa: E402
from src.model import HeatMLP, load_checkpoint  # noqa: E402
from src.plot_style import DARK, GREEN, TEAL, apply_style  # noqa: E402
from src.shapley import exact_shapley, sample_background  # noqa: E402

warnings.filterwarnings("ignore", category=DeprecationWarning, module="shap")
# shap logs a hint for backgrounds > 100 rows; large backgrounds are exactly what experiment B tests.
logging.getLogger("shap").setLevel(logging.ERROR)

DAYS_GRID = [1, 10, 100, 1000]
BACKGROUND_GRID = [10, 50, 100, 500]
FIXED_BACKGROUND = 100
FIXED_DAYS = 100
FEATURE_GRID = [4, 6, 8, 10, 12, 14, 16]
C_DAYS = 10
C_BACKGROUND = 20
SEED = 42

OUTPUT_DIR = REPO_ROOT / "outputs" / "runtime"
FIGURE_DIR = REPO_ROOT / "outputs" / "figures"
COLORS = {"exakt": DARK, "KernelSHAP": GREEN, "DeepSHAP": TEAL}


def repeats_for(n_days: int) -> int:
    """Short runs are noisy, so repeat them more often and take the median."""
    return 5 if n_days <= 10 else 3 if n_days <= 100 else 1


def hardware_description() -> str:
    lines = [f"platform: {platform.platform()}", f"machine: {platform.machine()}",
             f"python: {platform.python_version()}", f"torch: {torch.__version__}", f"shap: {shap.__version__}",
             f"cpu cores: {os.cpu_count()}", f"torch threads: {torch.get_num_threads()}"]
    if platform.system() == "Darwin":                       # macOS: chip name and memory
        for key, name in [("machdep.cpu.brand_string", "chip"), ("hw.memsize", "memory bytes")]:
            try:
                value = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
                lines.append(f"{name}: {value}")
            except OSError:
                pass
    elif Path("/proc/meminfo").exists():
        lines.append(Path("/proc/meminfo").read_text().splitlines()[0])
    return "\n".join(lines)


def main() -> None:
    apply_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    model, preprocessor, _, _ = load_checkpoint(REPO_ROOT / "outputs" / "models" / "mlp.pt")
    train, _, test = chronological_split(build_dataset())
    X_train = preprocessor.transform(train)
    X_test = preprocessor.transform(test)
    rng = np.random.default_rng(SEED)
    days_all = X_test[np.sort(rng.choice(len(X_test), size=max(DAYS_GRID), replace=False))]

    def predict_logit(z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return model(torch.from_numpy(z.astype(np.float32))).numpy().astype(np.float64).ravel()

    # One function per method: returns (setup seconds, explain seconds)
    def run_exact(X, background):
        start = time.perf_counter()
        exact_shapley(predict_logit, X, background)
        return 0.0, time.perf_counter() - start

    def run_kernel(X, background):
        start = time.perf_counter()
        explainer = shap.KernelExplainer(predict_logit, background)
        setup = time.perf_counter() - start
        start = time.perf_counter()
        explainer.shap_values(X, silent=True)
        return setup, time.perf_counter() - start

    def run_deep(X, background):
        start = time.perf_counter()
        explainer = shap.DeepExplainer(model, torch.from_numpy(background.astype(np.float32)))
        setup = time.perf_counter() - start
        start = time.perf_counter()
        explainer.shap_values(torch.from_numpy(X.astype(np.float32)))
        return setup, time.perf_counter() - start

    methods = {"exakt": run_exact, "KernelSHAP": run_kernel, "DeepSHAP": run_deep}

    # warm-up: the first call of each method loads code and allocates memory
    warm_bg, _ = sample_background(X_train, n=10, seed=SEED)
    for run in methods.values():
        run(days_all[:2], warm_bg)

    rows = []

    def measure(experiment, n_days, n_background):
        background, _ = sample_background(X_train, n=n_background, seed=SEED)
        X = days_all[:n_days]
        for name, run in methods.items():
            times = [run(X, background) for _ in range(repeats_for(n_days))]
            setup = float(np.median([t[0] for t in times]))
            explain = float(np.median([t[1] for t in times]))
            rows.append({"experiment": experiment, "method": name, "days": n_days, "background": n_background,
                         "setup_seconds": setup, "explain_seconds": explain,
                         "ms_per_day": 1000 * explain / n_days, "repeats": len(times)})
            print(f"  {experiment}  {name:10s} days={n_days:5d} background={n_background:4d}  "
                  f"explain {explain:8.3f} s  ({1000 * explain / n_days:7.2f} ms/day)")

    print("A) number of explained days (background 100)")
    for n in DAYS_GRID:
        measure("A_days", n, FIXED_BACKGROUND)
    print("B) background size (100 days)")
    for b in BACKGROUND_GRID:
        measure("B_background", FIXED_DAYS, b)

    # ---- C) number of features: random (untrained) networks M -> 16 -> 8 -> 1 ----
    print("C) number of features (untrained networks, 10 days, background 20)")
    for M in FEATURE_GRID:
        torch.manual_seed(SEED)
        net = HeatMLP(n_features=M).eval()
        data_rng = np.random.default_rng(SEED)
        X = data_rng.normal(size=(C_DAYS, M)).astype(np.float32)
        background = data_rng.normal(size=(C_BACKGROUND, M)).astype(np.float32)

        def predict_net(z: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return net(torch.from_numpy(z.astype(np.float32))).numpy().astype(np.float64).ravel()

        timings = {}
        start = time.perf_counter()
        exact_shapley(predict_net, X, background, batch_size=1)   # one day at a time: 2^M x 20 rows each
        timings["exakt"] = time.perf_counter() - start
        kernel = shap.KernelExplainer(predict_net, background)
        start = time.perf_counter()
        kernel.shap_values(X, silent=True)                         # default: 2M + 2048 coalitions
        timings["KernelSHAP"] = time.perf_counter() - start
        deep = shap.DeepExplainer(net, torch.from_numpy(background))
        start = time.perf_counter()
        deep.shap_values(torch.from_numpy(X))
        timings["DeepSHAP"] = time.perf_counter() - start
        for name, seconds in timings.items():
            rows.append({"experiment": "C_features", "method": name, "days": C_DAYS, "background": C_BACKGROUND,
                         "features": M, "setup_seconds": np.nan, "explain_seconds": seconds,
                         "ms_per_day": 1000 * seconds / C_DAYS, "repeats": 1})
        print(f"  M={M:2d}  coalitions={2 ** M:6d}  " +
              "  ".join(f"{name} {seconds:7.3f} s" for name, seconds in timings.items()))

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "runtime.csv", index=False)
    hardware = hardware_description()
    (OUTPUT_DIR / "hardware.txt").write_text(hardware + "\n")
    print("\n" + hardware)

    # ---- figure (slide 25) -----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, experiment, x_column, x_label, title, log_x in [
        (axes[0], "A_days", "days", "Anzahl erklärter Tage", f"Hintergrund: {FIXED_BACKGROUND} Tage", True),
        (axes[1], "B_background", "background", "Größe der Hintergrunddaten", f"{FIXED_DAYS} erklärte Tage", True),
        (axes[2], "C_features", "features", "Anzahl Merkmale M",
         f"Zufallsnetze, {C_DAYS} Tage, Hintergrund {C_BACKGROUND}", False),
    ]:
        part = results[results.experiment == experiment]
        for name, color in COLORS.items():
            line = part[part.method == name]
            ax.plot(line[x_column], line["explain_seconds"], marker="o", color=color, label=name, linewidth=2)
        if log_x:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = sorted(int(v) for v in part[x_column].unique())
        ax.set_xticks(ticks, [str(v) for v in ticks])
        ax.set_xlabel(x_label)
        ax.set_ylabel("Laufzeit in s")
        ax.set_title(title)
        ax.grid(True, which="major", alpha=0.3)
    axes[2].axvline(7, color=DARK, linestyle=":", linewidth=1.5)
    axes[2].text(7.2, axes[2].get_ylim()[1], "unser Modell", va="top", fontsize=12)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "runtime.png")
    plt.close(fig)
    print(f"\nSaved {OUTPUT_DIR.relative_to(REPO_ROOT)}/runtime.csv, hardware.txt and outputs/figures/runtime.png")


if __name__ == "__main__":
    main()
