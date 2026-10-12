"""Shape and training-step tests for ANNc (CPU, small embeddings)."""

import torch

from analogy_kg.model import ANNc
from analogy_kg.train import parallelogram_scores


def test_output_shape():
    model = ANNc(emb_size=16)
    x = torch.randn(5, 16, 4)
    assert model(x).shape == (5,)
    p = model.predict_proba(x)
    assert torch.all((p >= 0) & (p <= 1))


def test_one_training_step_reduces_loss():
    torch.manual_seed(0)
    model = ANNc(emb_size=16)
    x, y = torch.randn(64, 16, 4), torch.randint(0, 2, (64,)).float()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    first = loss_fn(model(x), y).item()
    for _ in range(20):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_parallelogram_perfect_analogy():
    a, b = torch.randn(1, 8), torch.randn(1, 8)
    c = torch.randn(1, 8)
    d = c + (b - a)                       # exact parallelogram
    x = torch.stack([a, b, c, d], dim=-1)
    assert abs(parallelogram_scores(x)[0] - 1.0) < 1e-5
