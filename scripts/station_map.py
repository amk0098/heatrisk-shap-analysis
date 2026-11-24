"""Outline map of Germany for the dataset slide; the station points are placed on the slide itself.

Station coordinates: DWD station list `KL_Tageswerte_Beschreibung_Stationen.txt` (daily kl).
Outline: Natural Earth 1:50m Admin 0 Countries (public domain), `data/external/germany_outline.geojson`.

Run from the repository root:
    python scripts/station_map.py

Writes outputs/figures/slides/germany_outline.png (transparent background) whose edges are exactly the
bounding box BOUNDS, so a point (lon, lat) is at the same relative position on the slide picture.
"""

import json
import sys
from math import cos, radians
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.plot_style import DARK  # noqa: E402

OUT = REPO_ROOT / "outputs" / "figures" / "slides" / "germany_outline.png"
BOUNDS = (5.8, 15.1, 47.2, 55.1)  # lon_min, lon_max, lat_min, lat_max
# DWD station ID: (name, latitude, longitude)
STATIONS = {400: ("Berlin-Buch", 52.6309, 13.5022), 1420: ("Frankfurt/Main", 50.0260, 8.5215),
            1975: ("Hamburg-Fuhlsbüttel", 53.6332, 9.9881), 2667: ("Köln-Bonn", 50.8646, 7.1575),
            1262: ("München-Flughafen", 48.3477, 11.8133)}


def aspect() -> float:
    """Width / height of the map in an equirectangular projection centred on the bounding box."""
    lon0, lon1, lat0, lat1 = BOUNDS
    return (lon1 - lon0) * cos(radians((lat0 + lat1) / 2)) / (lat1 - lat0)


def main() -> None:
    geo = json.loads((REPO_ROOT / "data" / "external" / "germany_outline.geojson").read_text())
    height = 4.0
    fig = plt.figure(figsize=(height * aspect(), height))
    ax = fig.add_axes((0, 0, 1, 1))
    for polygon in geo["features"][0]["geometry"]["coordinates"]:
        ax.add_patch(Polygon(polygon[0], closed=True, facecolor="#FFFFFF", edgecolor=DARK, linewidth=1.2))
    ax.set_xlim(BOUNDS[0], BOUNDS[1])
    ax.set_ylim(BOUNDS[2], BOUNDS[3])
    ax.axis("off")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, transparent=True)
    plt.close(fig)
    print(f"  {OUT.name}  aspect {aspect():.3f}")
    for sid, (name, lat, lon) in STATIONS.items():
        print(f"  {sid:5} {name:22} x {(lon - BOUNDS[0]) / (BOUNDS[1] - BOUNDS[0]):.3f}  "
              f"y {(BOUNDS[3] - lat) / (BOUNDS[3] - BOUNDS[2]):.3f}")


if __name__ == "__main__":
    main()
