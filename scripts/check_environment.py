import numpy as np
import shap
import torch
import torch.nn as nn


def print_versions():
    print("torch:", torch.__version__)
    print("shap: ", shap.__version__)
    print("numpy:", np.__version__)
    print()


# ---------------------------------------------------------------------------
# Test 1: the slide neuron, answers known by hand
# ---------------------------------------------------------------------------

class SlideNeuron(nn.Module):
    """One neuron: ReLU(1.0 * T + 0.1 * H - 30)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        with torch.no_grad():
            self.linear.weight[:] = torch.tensor([[1.0, 0.1]])
            self.linear.bias[:] = torch.tensor([-30.0])
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))


def test_slide_neuron():
    print("Test 1: slide neuron ReLU(T + 0.1*H - 30)")
    model = SlideNeuron().eval()

    today = np.array([[28.0, 60.0]], dtype=np.float32)      # T = 28, H = 60
    reference = np.array([[25.0, 20.0]], dtype=np.float32)  # T = 25, H = 20

    # DeepSHAP: works directly on the PyTorch model.
    deep = shap.DeepExplainer(model, torch.from_numpy(reference))
    deep_values = deep.shap_values(torch.from_numpy(today))
    deep_values = np.asarray(deep_values).reshape(-1)

    # KernelSHAP: only needs a function numpy -> numpy, it never looks inside.
    def predict(x):
        with torch.no_grad():
            return model(torch.from_numpy(x.astype(np.float32))).numpy().reshape(-1)

    kernel = shap.KernelExplainer(predict, reference)
    kernel_values = np.asarray(kernel.shap_values(today, silent=True)).reshape(-1)

    print(f"  KernelSHAP: T = {kernel_values[0]:.3f}, H = {kernel_values[1]:.3f}   (expected 1.500, 2.500)")
    print(f"  DeepSHAP:   T = {deep_values[0]:.3f}, H = {deep_values[1]:.3f}   (expected 1.714, 2.286)")

    assert np.allclose(kernel_values, [1.5, 2.5], atol=1e-3), "KernelSHAP does not match the exact values"
    assert np.allclose(deep_values, [12 / 7, 16 / 7], atol=1e-3), "DeepSHAP does not match the rescale rule"
    print("  OK\n")


# ---------------------------------------------------------------------------
# Test 2: a network with the real shape 7 -> 16 -> 8 -> 1
# ---------------------------------------------------------------------------

class HeatMLP(nn.Module):
    """Same shape as the seminar model. Untrained here, only for the check."""

    def __init__(self, n_features=7, hidden=(16, 8)):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden[1], 1)

    def forward(self, x):
        # Returns the logit (no sigmoid), as in the talk.
        return self.head(self.body(x))


def test_real_shape():
    print("Test 2: untrained 7-16-8-1 network")
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    model = HeatMLP().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params}   (expected 273)")
    assert n_params == 273

    background = rng.normal(size=(50, 7)).astype(np.float32)
    days = rng.normal(size=(5, 7)).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.from_numpy(days)).numpy().reshape(-1)
        base_value = model(torch.from_numpy(background)).numpy().mean()

    # DeepSHAP on the logit
    deep = shap.DeepExplainer(model, torch.from_numpy(background))
    deep_values = np.asarray(deep.shap_values(torch.from_numpy(days))).reshape(5, 7)

    # KernelSHAP on the logit
    def predict(x):
        with torch.no_grad():
            return model(torch.from_numpy(x.astype(np.float32))).numpy().reshape(-1)

    kernel = shap.KernelExplainer(predict, background)
    kernel_values = np.asarray(kernel.shap_values(days, silent=True)).reshape(5, 7)

    # Efficiency: base value + sum of Shapley values = logit
    deep_gap = np.abs(base_value + deep_values.sum(axis=1) - logits).max()
    kernel_gap = np.abs(base_value + kernel_values.sum(axis=1) - logits).max()
    print(f"  efficiency gap DeepSHAP:   {deep_gap:.2e}")
    print(f"  efficiency gap KernelSHAP: {kernel_gap:.2e}")
    assert deep_gap < 1e-3 and kernel_gap < 1e-3, "values do not add up to the logit"

    # DeepSHAP on the 8 embedding neurons (multi-output model)
    deep_emb = shap.DeepExplainer(model.body, torch.from_numpy(background))
    emb_values = np.asarray(deep_emb.shap_values(torch.from_numpy(days)))
    print(f"  embedding SHAP array shape: {emb_values.shape}   (5 days, 7 features, 8 neurons)")
    print("  OK\n")


if __name__ == "__main__":
    print_versions()
    test_slide_neuron()
    test_real_shape()
    print("Environment works: torch and shap are compatible.")