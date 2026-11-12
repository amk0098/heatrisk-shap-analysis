"""Exact Shapley values: worked example, the four Shapley axioms, linear closed form.

Run from the repository root:  python -m pytest tests/test_shapley.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.shapley import coalition_masks, exact_shapley, shapley_weight  # noqa: E402


def example_neuron(z):
    """ReLU(T + 0.1*H - 30), worked example with known values."""
    return np.maximum(0.0, z[:, 0] + 0.1 * z[:, 1] - 30.0)


def test_masks_cover_all_coalitions():
    masks = coalition_masks(3)
    assert masks.shape == (8, 3)
    assert len({tuple(m) for m in masks}) == 8           # all different
    assert not masks[0].any() and masks[-1].all()        # first = empty set, last = all features


def test_weight_of_empty_coalition():
    assert shapley_weight(0, 7) == pytest.approx(1 / 7)  # 0! 6! / 7!


def test_weights_sum_to_one():
    M = 7
    masks = coalition_masks(M)
    total = sum(shapley_weight(int(m.sum()), M) for m in masks if not m[0])
    assert total == pytest.approx(1.0)


def test_worked_example():
    today = np.array([[28.0, 60.0]])
    reference = np.array([[25.0, 20.0]])
    phi, base = exact_shapley(example_neuron, today, reference)
    assert phi[0] == pytest.approx([1.5, 2.5])
    assert base == 0.0


def random_model(seed=0, M=5):
    """Small random ReLU network in numpy."""
    rng = np.random.default_rng(seed)
    W1, b1 = rng.normal(size=(M, 6)), rng.normal(size=6)
    w2 = rng.normal(size=6)
    return lambda z: np.maximum(0, z @ W1 + b1) @ w2


def test_efficiency():
    rng = np.random.default_rng(1)
    f = random_model()
    X, background = rng.normal(size=(4, 5)), rng.normal(size=(20, 5))
    phi, base = exact_shapley(f, X, background)
    assert base + phi.sum(axis=1) == pytest.approx(f(X))


def test_symmetry():
    # f only depends on x0 + x1, and both have the same values -> equal contributions
    f = lambda z: np.maximum(0, z[:, 0] + z[:, 1] - 1) + z[:, 2]
    X = np.array([[2.0, 2.0, 5.0]])
    background = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    phi, _ = exact_shapley(f, X, background)
    assert phi[0, 0] == pytest.approx(phi[0, 1])


def test_dummy():
    # feature 2 is never used by f -> phi = 0
    f = lambda z: z[:, 0] * z[:, 1]
    rng = np.random.default_rng(2)
    phi, _ = exact_shapley(f, rng.normal(size=(3, 3)), rng.normal(size=(10, 3)))
    assert phi[:, 2] == pytest.approx(0.0)


def test_additivity():
    rng = np.random.default_rng(3)
    f, g = random_model(4), random_model(5)
    X, background = rng.normal(size=(3, 5)), rng.normal(size=(15, 5))
    phi_f, _ = exact_shapley(f, X, background)
    phi_g, _ = exact_shapley(g, X, background)
    phi_sum, _ = exact_shapley(lambda z: f(z) + g(z), X, background)
    assert phi_sum == pytest.approx(phi_f + phi_g)


def test_linear_model_has_closed_form():
    # For f(x) = w . x the Shapley value is w_i * (x_i - mean of background_i)
    w = np.array([2.0, -1.0, 0.5])
    f = lambda z: z @ w
    rng = np.random.default_rng(4)
    X, background = rng.normal(size=(5, 3)), rng.normal(size=(30, 3))
    phi, _ = exact_shapley(f, X, background)
    assert phi == pytest.approx(w * (X - background.mean(axis=0)))


def test_multi_output_equals_one_output_at_a_time():
    from src.shapley import exact_shapley_multi
    rng = np.random.default_rng(5)
    f, g = random_model(6), random_model(7)
    both = lambda z: np.stack([f(z), g(z)], axis=1)        # model with K = 2 outputs
    X, background = rng.normal(size=(4, 5)), rng.normal(size=(12, 5))
    phi, base = exact_shapley_multi(both, X, background)
    phi_f, base_f = exact_shapley(f, X, background)
    phi_g, base_g = exact_shapley(g, X, background)
    assert phi.shape == (4, 5, 2)
    assert phi[:, :, 0] == pytest.approx(phi_f)
    assert phi[:, :, 1] == pytest.approx(phi_g)
    assert base == pytest.approx([base_f, base_g])
