"""Checks of the model code. Run from the repository root:  python -m pytest"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model import HeatMLP, Preprocessor, count_parameters  # noqa: E402


def test_parameter_count_matches_slide():
    assert count_parameters(HeatMLP(7, (16, 8))) == 273


def test_output_shapes():
    model = HeatMLP()
    x = torch.zeros(5, 7)
    assert model(x).shape == (5, 1)        # logit
    assert model.embed(x).shape == (5, 8)  # embedding
    assert model.predict_proba(x).shape == (5,)


def test_logit_is_head_of_embedding():
    # logit = last layer applied to the embedding; this is what the embedding analysis relies on
    torch.manual_seed(0)
    model = HeatMLP()
    x = torch.randn(10, 7)
    assert torch.allclose(model(x), model.head(model.embed(x)))


def test_preprocessor_uses_only_fit_data():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, np.nan, 30.0]})
    other = pd.DataFrame({"a": [100.0], "b": [np.nan]})
    pre = Preprocessor(["a", "b"]).fit(train)
    scaled_train = pre.transform(train)
    assert np.allclose(scaled_train.mean(axis=0), 0, atol=1e-6)  # centred on train
    # missing value in new data -> training mean -> 0 after scaling
    assert pre.transform(other)[0, 1] == 0
    # new data does not change the fitted statistics
    assert np.allclose(pre.mean, [2.0, 20.0])
