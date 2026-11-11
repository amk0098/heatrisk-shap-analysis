"""One plot style for all figures, using the colours of the BUW slide template."""

import matplotlib.pyplot as plt

# Colours from the theme of the seminar slides
DARK = "#143C46"    # dark blue-green (text, main bars)
GREEN = "#89BA17"   # uni green
TEAL = "#008DA3"
DARK_GREEN = "#006932"
GREY = "#87888A"

FIGURE_DIR_NAME = "figures"


def apply_style() -> None:
    """Large fonts (slides need >= 18 pt when the figure is shown at full width)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.edgecolor": GREY,
        "axes.labelcolor": DARK,
        "text.color": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
