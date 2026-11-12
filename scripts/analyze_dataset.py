"""Step 3: dataset analysis.

Run from the repository root:
    python scripts/analyze_dataset.py

Writes:
    outputs/dataset_summary.csv          rows, heat days and rate per split
    outputs/figures/heat_days_per_month.png
    outputs/figures/feature_correlation.png  (training period only)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_LABELS_DE,
    TARGET,
    build_dataset,
    chronological_split,
)
from src.plot_style import DARK, GREEN, GREY, apply_style  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

MONTHS_DE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]


def summarise_splits(train, validation, test) -> pd.DataFrame:
    rows = []
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        rows.append({
            "split": name,
            "first_day": part["MESS_DATUM"].min().date(),
            "last_day": part["MESS_DATUM"].max().date(),
            "rows": len(part),
            "heat_days": int(part[TARGET].sum()),
            "heat_rate_percent": round(100 * part[TARGET].mean(), 2),
        })
    everything = pd.concat([train, validation, test])
    rows.append({
        "split": "all",
        "first_day": everything["MESS_DATUM"].min().date(),
        "last_day": everything["MESS_DATUM"].max().date(),
        "rows": len(everything),
        "heat_days": int(everything[TARGET].sum()),
        "heat_rate_percent": round(100 * everything[TARGET].mean(), 2),
    })
    return pd.DataFrame(rows)


def plot_heat_days_per_month(df: pd.DataFrame, path: Path) -> pd.Series:
    """Share of days in each month that are followed by a heat day, in percent."""
    # month of the predicted day
    target_month = (df["MESS_DATUM"] + pd.Timedelta(days=1)).dt.month
    rate = df.groupby(target_month)[TARGET].mean().reindex(range(1, 13), fill_value=0) * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [GREEN if m in (6, 7, 8) else GREY for m in rate.index]
    bars = ax.bar(MONTHS_DE, rate.values, color=colors)
    for bar, value in zip(bars, rate.values):
        if value >= 0.5:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:.0f} %",
                    ha="center", va="bottom", fontsize=12, color=DARK)
    ax.set_ylabel("Hitzetage in %")
    ax.set_ylim(0, rate.max() * 1.2)
    ax.set_title("Anteil Hitzetage pro Monat (1990–2025)")
    fig.savefig(path)
    plt.close(fig)
    return rate


def plot_correlation(train: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Pearson correlation of the 7 features, training period only."""
    corr = train[FEATURE_COLUMNS].corr()
    labels = [FEATURE_LABELS_DE[c] for c in FEATURE_COLUMNS]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if abs(value) > 0.6 else DARK)
    ax.spines[:].set_visible(False)
    fig.colorbar(image, ax=ax, shrink=0.8, label="Korrelation")
    ax.set_title("Korrelation der Merkmale (Training)")
    fig.savefig(path)
    plt.close(fig)
    return corr


def main() -> None:
    apply_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataset()
    train, validation, test = chronological_split(df)

    summary = summarise_splits(train, validation, test)
    summary.to_csv(OUTPUT_DIR / "dataset_summary.csv", index=False)
    print("Splits:")
    print(summary.to_string(index=False))

    rate = plot_heat_days_per_month(df, FIGURE_DIR / "heat_days_per_month.png")
    summer_share = df.loc[(df["MESS_DATUM"] + pd.Timedelta(days=1)).dt.month.isin([6, 7, 8]), TARGET].sum() / df[TARGET].sum()
    print(f"\nHeat-day rate per month (%): {rate.round(1).to_dict()}")
    print(f"Share of all heat days in June-August: {100 * summer_share:.0f} %")

    corr = plot_correlation(train, FIGURE_DIR / "feature_correlation.png")
    temperature = ["TMK", "TXK", "TNK", "TXK_lag_1", "TXK_rolling_3"]
    pairs = corr.loc[temperature, temperature].values[np.triu_indices(5, k=1)]
    print(f"\nCorrelation among the 5 temperature features: {pairs.min():.2f} to {pairs.max():.2f}")
    print(f"\nFigures written to {FIGURE_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
