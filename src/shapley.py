"""Exact Shapley values over all coalitions.

Setup ("interventional" Shapley values, the same game KernelSHAP and DeepSHAP approximate):
    - players  = the M input features
    - v(S)     = average model output when the features in S have TODAY's values
                 and the other features take the values of each background day
    - phi_i    = sum over all S without i of  |S|! (M-|S|-1)! / M!  *  [ v(S + i) - v(S) ]

With M = 7 there are 2^7 = 128 coalitions, so this is cheap enough to compute exactly
and serves as the ground truth for KernelSHAP and DeepSHAP.
"""

from math import factorial

import numpy as np


def coalition_masks(n_features: int) -> np.ndarray:
    """All 2^M coalitions as a boolean matrix of shape (2^M, M).

    Row k is the binary number k: bit j set <=> feature j is "known" in coalition k.
    For M = 2:  k=0 -> [F, F] (empty),  k=1 -> [T, F] ({0}),  k=2 -> [F, T] ({1}),  k=3 -> [T, T]
    """
    k = np.arange(2 ** n_features)
    return ((k[:, None] >> np.arange(n_features)) & 1).astype(bool)


def shapley_weight(size: int, n_features: int) -> float:
    """|S|! (M - |S| - 1)! / M!  = share of orderings in which exactly S comes before i."""
    return factorial(size) * factorial(n_features - size - 1) / factorial(n_features)


def coalition_values(predict, X: np.ndarray, background: np.ndarray) -> np.ndarray:
    """v(S) for every day in X and every coalition S.

    predict:     function, array (n, M) -> array (n,)   (e.g. the MLP's logit)
    X:           days to explain, shape (n_days, M)
    background:  reference days, shape (B, M)
    returns:     shape (n_days, 2^M)
    """
    n_days, M = X.shape
    masks = coalition_masks(M)                       # (C, M) with C = 2^M
    # day x coalition x background row: features in S from the day, the others from the background row
    z = np.where(
        masks[None, :, None, :],                      # (1, C, 1, M)
        X[:, None, None, :],                          # (n, 1, 1, M)
        background[None, None, :, :],                 # (1, 1, B, M)
    )                                                 # -> (n, C, B, M)
    out = predict(z.reshape(-1, M)).reshape(n_days, len(masks), len(background))
    return out.mean(axis=2)                           # average over the background


def shapley_from_values(v: np.ndarray, n_features: int) -> np.ndarray:
    """Apply the Shapley formula to the coalition values v (n_days, 2^M) -> phi (n_days, M)."""
    masks = coalition_masks(n_features)
    sizes = masks.sum(axis=1)
    phi = np.zeros((v.shape[0], n_features))
    for i in range(n_features):
        without_i = np.flatnonzero(~masks[:, i])              # all coalitions S that do not contain i
        with_i = without_i | (1 << i)                          # the same coalitions plus i
        weights = np.array([shapley_weight(s, n_features) for s in sizes[without_i]])
        phi[:, i] = (v[:, with_i] - v[:, without_i]) @ weights
    return phi


def exact_shapley(predict, X: np.ndarray, background: np.ndarray, batch_size: int = 25):
    """Exact Shapley values for many days, computed in small batches to save memory.

    returns (phi, base_value):
        phi         (n_days, M)  contribution of each feature to each day's output
        base_value  float        v(empty set) = mean output over the background
    Efficiency holds:  base_value + phi.sum(axis=1) == predict(X)
    """
    X = np.asarray(X, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    M = X.shape[1]
    phis, base_value = [], None
    for start in range(0, len(X), batch_size):
        v = coalition_values(predict, X[start:start + batch_size], background)
        phis.append(shapley_from_values(v, M))
        base_value = float(v[0, 0])                   # coalition 0 = empty set, same for every day
    return np.vstack(phis), base_value


def exact_shapley_multi(predict, X: np.ndarray, background: np.ndarray, batch_size: int = 25):
    """Like exact_shapley, but for a model with K outputs (e.g. the 8 embedding neurons).

    predict:  function, array (n, M) -> array (n, K)
    returns (phi, base_values):
        phi          (n_days, M, K)   contribution of feature i to output k
        base_values  (K,)             mean of each output over the background
    Every output is its own game; the coalitions are evaluated only once for all K outputs.
    """
    X = np.asarray(X, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    n_all, M = X.shape
    masks = coalition_masks(M)
    sizes = masks.sum(axis=1)
    phis, base_values = [], None
    for start in range(0, n_all, batch_size):
        Xb = X[start:start + batch_size]
        z = np.where(masks[None, :, None, :], Xb[:, None, None, :], background[None, None, :, :])
        out = predict(z.reshape(-1, M))                               # (rows, K)
        v = out.reshape(len(Xb), len(masks), len(background), -1).mean(axis=2)   # (n, C, K)
        phi = np.zeros((len(Xb), M, v.shape[2]))
        for i in range(M):
            without_i = np.flatnonzero(~masks[:, i])
            with_i = without_i | (1 << i)
            weights = np.array([shapley_weight(s, M) for s in sizes[without_i]])
            phi[:, i, :] = np.einsum("ncK,c->nK", v[:, with_i, :] - v[:, without_i, :], weights)
        phis.append(phi)
        base_values = v[0, 0, :]
    return np.concatenate(phis), base_values


def sample_background(X_train: np.ndarray, n: int = 100, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Random sample of training days used as background.

    Returns the sampled rows and their row indices (to save and reuse the same background
    for exact Shapley, KernelSHAP and DeepSHAP, so the three are comparable).
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(X_train), size=n, replace=False))
    return X_train[idx], idx
