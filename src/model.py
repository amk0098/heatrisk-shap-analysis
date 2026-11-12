"""The MLP for next-day heat-day prediction and its input preprocessing.

Architecture:  7 inputs -> 16 (ReLU) -> 8 (ReLU) = embedding -> 1 = logit
Parameters: 7*16+16 + 16*8+8 + 8*1+1 = 128 + 136 + 9 = 273
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Preprocessing: fill missing values, then standardise
# ---------------------------------------------------------------------------

@dataclass
class Preprocessor:
    """Mean imputation and standardisation with statistics of the training data.

    transform():  x_scaled = (x - mean) / std,  missing values -> mean (0 after scaling)
    """
    columns: list
    mean: np.ndarray = None
    std: np.ndarray = None

    def fit(self, X: pd.DataFrame) -> "Preprocessor":
        values = X[self.columns].to_numpy(dtype=np.float64)
        self.mean = np.nanmean(values, axis=0)
        self.std = np.nanstd(values, axis=0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self.columns].to_numpy(dtype=np.float64)
        values = np.where(np.isnan(values), self.mean, values)  # mean imputation
        return ((values - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        """Back to °C / hPa / %."""
        return scaled * self.std + self.mean

    def to_dict(self) -> dict:
        return {"columns": list(self.columns), "mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Preprocessor":
        return cls(columns=d["columns"], mean=np.array(d["mean"]), std=np.array(d["std"]))


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------

class HeatMLP(nn.Module):
    def __init__(self, n_features: int = 7, hidden: tuple = (16, 8)):
        super().__init__()
        # body: hidden layers, output = embedding
        self.body = nn.Sequential(
            nn.Linear(n_features, hidden[0]),  # h1 = ReLU(W1 x + b1)
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),   # h2 = ReLU(W2 h1 + b2)
            nn.ReLU(),
        )
        # head: last layer, output = logit
        self.head = nn.Linear(hidden[1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logit, shape (n, 1)."""
        return self.head(self.body(x))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embedding (output of the second hidden layer), shape (n, 8)."""
        return self.body(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Probability of a heat day tomorrow, shape (n,)."""
        return torch.sigmoid(self.forward(x)).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(path, model: HeatMLP, preprocessor: Preprocessor, threshold: float, info: dict) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "hidden": [model.body[0].out_features, model.body[2].out_features],
        "preprocessor": preprocessor.to_dict(),
        "threshold": threshold,
        "info": info,
    }, path)


def load_checkpoint(path) -> tuple[HeatMLP, Preprocessor, float, dict]:
    ckpt = torch.load(path, weights_only=False)
    pre = Preprocessor.from_dict(ckpt["preprocessor"])
    model = HeatMLP(n_features=len(pre.columns), hidden=tuple(ckpt["hidden"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, pre, ckpt["threshold"], ckpt["info"]
