"""Checks of the data pipeline: missing values, target, lag features, split."""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.data import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET,
    build_dataset,
    chronological_split,
    load_all_stations,
)


@pytest.fixture(scope="module")
def raw():
    return load_all_stations()


@pytest.fixture(scope="module")
def df():
    return build_dataset()


def test_missing_marker_removed(raw):
    assert (raw[["TMK", "TXK", "TNK", "VPM", "UPM"]] <= -998).sum().sum() == 0


def test_no_date_gaps(raw):
    # lags use the previous row, so it must be the previous day
    gaps = raw.groupby("city")["MESS_DATUM"].diff().dt.days.dropna()
    assert (gaps == 1).all()


def test_target_is_tomorrows_heat(raw, df):
    berlin_raw = raw[raw.city == "Berlin"].set_index("MESS_DATUM")["TXK"]
    berlin = df[df.city == "Berlin"].set_index("MESS_DATUM")
    for day in berlin.index[:500]:
        tomorrow = day + pd.Timedelta(days=1)
        assert berlin.loc[day, TARGET] == int(berlin_raw.loc[tomorrow] >= 30.0)


def test_lag_features_use_only_the_past(raw, df):
    berlin_raw = raw[raw.city == "Berlin"].set_index("MESS_DATUM")["TXK"]
    row = df[(df.city == "Berlin") & (df.MESS_DATUM == "2019-06-26")].iloc[0]
    past = berlin_raw.loc["2019-06-23":"2019-06-25"]  # t-3 .. t-1, today excluded
    assert row["TXK_lag_1"] == berlin_raw.loc["2019-06-25"]
    assert row["TXK_rolling_3"] == pytest.approx(past.mean())


def test_split_is_chronological_and_complete(df):
    train, validation, test = chronological_split(df)
    assert train.MESS_DATUM.max() < validation.MESS_DATUM.min()
    assert validation.MESS_DATUM.max() < test.MESS_DATUM.min()
    assert len(train) + len(validation) + len(test) == len(df)


def test_split_sizes_match_audit(df):
    # numbers from docs/data_audit.md
    train, validation, test = chronological_split(df)
    assert (len(train), len(validation), len(test)) == (55543, 5475, 3650)
    assert (train[TARGET].sum(), validation[TARGET].sum(), test[TARGET].sum()) == (1682, 219, 135)


def test_features_present(df):
    assert list(df[FEATURE_COLUMNS].columns) == FEATURE_COLUMNS
