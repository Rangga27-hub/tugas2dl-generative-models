import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.common import load_checkpoint, load_config, seed_all
from src.evaluate import reconstruction_scores
from src.models import build_model
from src.train import NAMES, gan_step, objective, train_model


@pytest.fixture
def cfg():
    seed_all(42)
    config = load_config()
    config.update(device="cpu", ddpm_steps=4, ddpm_channels=8)
    return config


@pytest.mark.parametrize("name", NAMES)
def test_forward_backward_updates(name, cfg):
    model = build_model(name, cfg)
    x, y = torch.rand(4, 1, 28, 28), torch.tensor([0, 1, 2, 3])
    before = [p.detach().clone() for p in model.parameters()]
    if name in ("gan", "dcgan"):
        opts = [torch.optim.Adam(model.generator.parameters()), torch.optim.Adam(model.discriminator.parameters())]
        loss, _ = gan_step(model, x, opts, cfg, True)
        model.eval()
        assert model(torch.randn(4, cfg["noise_dim"])).shape == x.shape
    else:
        classifier = build_model("mlp", cfg).eval().requires_grad_(False) if name == "decoder" else None
        opt = torch.optim.Adam(model.parameters())
        loss, _ = objective(name, model, x, y, cfg, classifier)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_ddpm_schedule_and_sampling(cfg):
    cfg = load_config()
    model = build_model("ddpm", {**cfg, "ddpm_channels": 8})
    assert model.abar[-1] < 1e-3, "terminal forward distribution must approach Gaussian"
    tiny = build_model("ddpm", {**cfg, "ddpm_channels": 8, "ddpm_steps": 4}).eval()
    x = torch.randn(2, 1, 28, 28)
    out, chain = tiny.sample(x, torch.Generator().manual_seed(123), snapshots=True)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert out.min() >= -1.0001 and out.max() <= 1.0001
    assert 4 in chain and 0 in chain


def test_transpose_matches_column_formula(cfg):
    mlp = build_model("mlp", cfg)
    one_hot = torch.eye(10)
    w1, w2, w3 = mlp.fc1.weight, mlp.fc2.weight, mlp.fc3.weight
    expected = torch.sigmoid(w1.T @ F.relu(w2.T @ F.relu(w3.T @ one_hot.T))).T.reshape(10, 1, 28, 28)
    assert torch.allclose(mlp.transpose_class(one_hot), expected, atol=1e-6)


def test_metrics_identity_and_mean(cfg):
    x = torch.rand(4, 1, 28, 28)
    mse, psnr, ssim = reconstruction_scores(x, x)
    assert torch.equal(mse, torch.zeros(4))
    assert torch.isfinite(psnr).all()
    assert torch.allclose(ssim, torch.ones(4), atol=1e-5)


class TinyData:
    split_sha256 = "test-only"

    def loader(self, split, epoch=0, batch_size=None):
        gen = torch.Generator().manual_seed(991)
        x = torch.randint(0, 256, (8, 1, 28, 28), dtype=torch.uint8, generator=gen)
        y = torch.arange(8) % 10
        return DataLoader(TensorDataset(x, y, torch.arange(8)), batch_size=4,
                          shuffle=split == "train", generator=torch.Generator().manual_seed(42+epoch))

    def summary(self):
        return {"synthetic_test_fixture": True}


def test_resume_reproduces_uninterrupted(cfg, tmp_path):
    cfg["output_dir"] = str(tmp_path / "uninterrupted")
    data = TinyData()
    full = train_model("ae", cfg, data, epochs=2)
    cfg2 = copy.deepcopy(cfg)
    cfg2["output_dir"] = str(tmp_path / "resumed")
    train_model("ae", cfg2, data, epochs=1)
    resumed = train_model("ae", cfg2, data, resume=True, epochs=2)
    for a, b in zip(full.parameters(), resumed.parameters()):
        assert torch.equal(a, b)
    saved = load_checkpoint(Path(cfg2["output_dir"]) / "ae" / "last.pt")
    assert saved["epoch"] == 2 and len(saved["history"]) == 2
    assert saved["optimizers"] and saved["rng"] and saved["config"]
    with pytest.raises(FileExistsError):
        train_model("ae", cfg2, data, epochs=2)


def test_split_persistence_and_disjoint(monkeypatch, tmp_path, cfg):
    import src.data as module

    class FakeFashion:
        def __init__(self, root, train, download):
            n = 60000 if train else 10000
            Path(root).mkdir(parents=True, exist_ok=True)
            self.data = torch.zeros(n, 28, 28, dtype=torch.uint8)
            self.targets = torch.arange(n) % 10

    monkeypatch.setattr(module, "FashionMNIST", FakeFashion)
    cfg["data_dir"] = str(tmp_path / "data")
    a, b = module.DataModule(cfg), module.DataModule(cfg)
    assert a.split_sha256 == b.split_sha256
    assert [len(a.indices[k]) for k in ("train", "validation", "test")] == [56000, 7000, 7000]
    combined = np.concatenate(list(a.indices.values()))
    assert len(np.unique(combined)) == 70000
    for key in a.indices:
        np.testing.assert_array_equal(a.indices[key], b.indices[key])
