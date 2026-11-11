"""Plots for the Shapley results, in the style of the slides."""

import matplotlib.pyplot as plt
import numpy as np

from src.data import FEATURE_COLUMNS, FEATURE_LABELS_DE
from src.plot_style import DARK, GREY, GREEN, TEAL

UNITS = {"TMK": "°C", "TXK": "°C", "TNK": "°C", "VPM": "hPa", "UPM": "%",
         "TXK_lag_1": "°C", "TXK_rolling_3": "°C"}

CITY_NAMES_DE = {"Berlin": "Berlin", "Frankfurt": "Frankfurt", "Hamburg": "Hamburg",
                 "Koeln": "Köln", "Muenchen": "München"}

UP = "#C0392B"    # pushes the logit up (towards heat day)
DOWN = TEAL       # pushes the logit down


def waterfall(ax, phi, base_value, raw_values, title=""):
    """Horizontal waterfall: from the base value, each feature adds its phi, ending at the logit.

    phi, raw_values: arrays of length 7 in FEATURE_COLUMNS order (raw_values in °C / hPa / %).
    """
    order = np.argsort(np.abs(phi))            # smallest at the bottom, largest at the top
    labels = [f"{FEATURE_LABELS_DE[FEATURE_COLUMNS[i]]} = {raw_values[i]:.1f} {UNITS[FEATURE_COLUMNS[i]]}"
              for i in order]
    # running total: start of each bar
    starts = base_value + np.concatenate([[0], np.cumsum(phi[order])[:-1]])
    logit = base_value + phi.sum()

    for row, (i, start) in enumerate(zip(order, starts)):
        value = phi[i]
        color = UP if value > 0 else DOWN
        ax.barh(row, value, left=start, color=color, height=0.6)
        text_x = start + value + (0.15 if value > 0 else -0.15)
        ax.text(text_x, row, f"{value:+.2f}", va="center", ha="left" if value > 0 else "right",
                fontsize=12, color=DARK)

    ax.set_yticks(range(len(order)), labels)
    ax.axvline(base_value, color=GREY, linestyle=":", linewidth=1.5)
    ax.axvline(logit, color=DARK, linestyle="-", linewidth=1.5)
    ymax = len(order) - 0.4
    label_box = dict(facecolor="white", edgecolor="none", pad=1)
    ax.text(base_value, ymax, f"Basiswert\n{base_value:.2f}", ha="center", va="bottom", fontsize=12,
            color=GREY, bbox=label_box)
    ax.text(logit, ymax, f"Logit\n{logit:.2f}", ha="center", va="bottom", fontsize=12, color=DARK, bbox=label_box)
    lo = min(base_value, logit, *(starts + phi[order]), *starts)
    hi = max(base_value, logit, *(starts + phi[order]), *starts)
    pad = 0.15 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(-0.6, len(order) + 0.6)
    ax.set_xlabel("Logit")
    if title:
        ax.set_title(title, pad=40)


def deviation_scatter(ax, exact, approx, name, color):
    """Each point = one (day, feature). On the diagonal = identical to the exact value."""
    lim = [min(exact.min(), approx.min()), max(exact.max(), approx.max())]
    ax.plot(lim, lim, color=GREY, linewidth=1, zorder=0)
    ax.scatter(exact.ravel(), approx.ravel(), s=4, alpha=0.4, color=color, linewidths=0)
    dev = np.abs(approx - exact)
    ax.set_title(f"{name}\nmax. Abw. {dev.max():.2f} · mittl. Abw. {dev.mean():.3f}", fontsize=14)
    ax.set_xlabel("φ exakt")
    ax.set_ylabel(f"φ {name}")
    ax.set_aspect("equal")


__all__ = ["waterfall", "deviation_scatter", "UNITS", "UP", "DOWN", "GREEN"]
