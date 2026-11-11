"""Load the DWD data and build the heat-day dataset.

The loading, target, lag features and split are taken from the source project
heatrisk-deutschland at commit 9974c0e (src/data_loader.py, src/features.py,
src/split.py). The logic is unchanged, so the MLP is evaluated on exactly the
same rows as the Logistic Regression and Random Forest baselines there.
See docs/data_audit.md for the checks of this pipeline.
"""

from pathlib import Path

import pandas as pd

# Repository root = two levels above this file (src/data.py -> repo/)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "raw"

CITY_MAP = {
    "produkt_klima_tag_Berlin.txt": "Berlin",
    "produkt_klima_tag_Frankfurt.txt": "Frankfurt",
    "produkt_klima_tag_Hamburg.txt": "Hamburg",
    "produkt_klima_tag_Muenchen.txt": "Muenchen",
    "produkt_klima_tag_koeln.txt": "Koeln",
}

RAW_COLUMNS = ["TMK", "TXK", "TNK", "VPM", "UPM"]

FEATURE_COLUMNS = [
    "TMK",            # mean temperature of day t
    "TXK",            # maximum temperature of day t
    "TNK",            # minimum temperature of day t
    "VPM",            # mean vapour pressure of day t
    "UPM",            # mean relative humidity of day t
    "TXK_lag_1",      # maximum temperature of day t-1
    "TXK_rolling_3",  # MEAN of the maximum temperatures of days t-1, t-2, t-3
]

# Short German labels for the slides, in the same order as FEATURE_COLUMNS
FEATURE_LABELS_DE = {
    "TMK": "Mittel-Temp.",
    "TXK": "Max.-Temp.",
    "TNK": "Min.-Temp.",
    "VPM": "Dampfdruck",
    "UPM": "rel. Feuchte",
    "TXK_lag_1": "Vortags-Max.",
    "TXK_rolling_3": "3-Tage-Mittel Max.",
}

TARGET = "heat_day_next"
HEAT_THRESHOLD = 30.0  # °C, a heat day has TXK >= 30

START_DATE = "1990-01-01"
TRAIN_END = "2020-12-31"
VALIDATION_START = "2021-01-01"
VALIDATION_END = "2023-12-31"
TEST_START = "2024-01-01"


def load_station_data(file_path: Path) -> pd.DataFrame:
    """Read one DWD station file. Missing values are marked -999 in the file."""
    df = pd.read_csv(file_path, sep=";", na_values=["-999", "-999.0"])
    df.columns = df.columns.str.strip()
    # Dates are written as numbers like 19900101; read them as text "YYYYMMDD".
    df["MESS_DATUM"] = pd.to_datetime(df["MESS_DATUM"].astype(str), format="%Y%m%d")
    df = df[df["MESS_DATUM"] >= START_DATE].copy()
    df["city"] = CITY_MAP[file_path.name]
    return df[["MESS_DATUM", *RAW_COLUMNS, "city"]]


def load_all_stations(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read all five station files and sort by city and date."""
    files = sorted(Path(data_dir).glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No DWD files in {data_dir}. Copy them from heatrisk-deutschland/data/raw "
            "(see data/README.md)."
        )
    df = pd.concat([load_station_data(f) for f in files], ignore_index=True)
    return df.sort_values(["city", "MESS_DATUM"]).reset_index(drop=True)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """heat_day_next = 1 if TOMORROW's maximum temperature in the same city is >= 30 °C."""
    df = df.copy()
    next_txk = df.groupby("city")["TXK"].shift(-1)
    df[TARGET] = (next_txk >= HEAT_THRESHOLD).astype("Int64")
    df.loc[next_txk.isna(), TARGET] = pd.NA
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Past-only features, computed separately for each city."""
    df = df.copy()
    df["TXK_lag_1"] = df.groupby("city")["TXK"].shift(1)
    df["TXK_rolling_3"] = df.groupby("city")["TXK"].transform(
        lambda s: s.shift(1).rolling(3).mean()
    )
    return df.dropna(subset=["TXK_lag_1", "TXK_rolling_3"]).copy()


def build_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Full pipeline: load -> target -> lag features. One row per city and day."""
    df = load_all_stations(data_dir)
    df = add_target(df)
    df = add_lag_features(df)
    return df.reset_index(drop=True)


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train 1990-2020, validation 2021-2023, test 2024-2025. Never shuffled."""
    train = df[df["MESS_DATUM"] <= TRAIN_END].copy()
    validation = df[
        (df["MESS_DATUM"] >= VALIDATION_START) & (df["MESS_DATUM"] <= VALIDATION_END)
    ].copy()
    test = df[df["MESS_DATUM"] >= TEST_START].copy()
    return train, validation, test


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS].copy(), df[TARGET].copy()
