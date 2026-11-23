from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST, MNIST

from .common import load_config, save_json


class DataModule:
    """Pool official 60k + 10k, then persist one random 80/10/10 split."""

    def __init__(self, config, download=True):
        self.config = config
        dataset_cls = {"FashionMNIST": FashionMNIST, "MNIST": MNIST}[config["dataset"]]
        root = Path(config["data_dir"])
        official_train = dataset_cls(root, train=True, download=download)
        official_test = dataset_cls(root, train=False, download=download)
        self.images = torch.cat([official_train.data, official_test.data]).unsqueeze(1)
        self.labels = torch.cat([official_train.targets, official_test.targets])
        assert self.images.shape == (70000, 1, 28, 28)
        self.split_path = root / f'{config["dataset"]}_split_seed{config["seed"]}.npz'
        if not self.split_path.exists():
            order = np.random.default_rng(config["seed"]).permutation(70000)
            np.savez(self.split_path, train=order[:56000], validation=order[56000:63000], test=order[63000:])
        with np.load(self.split_path) as stored:
            self.indices = {k: stored[k].copy() for k in ("train", "validation", "test")}
        expected = (56000, 7000, 7000)
        assert tuple(len(self.indices[k]) for k in self.indices) == expected
        all_indices = np.concatenate(list(self.indices.values()))
        assert np.array_equal(np.sort(all_indices), np.arange(70000)), "Split overlaps or is incomplete"
        self.split_sha256 = hashlib.sha256(self.split_path.read_bytes()).hexdigest()

    def loader(self, split, epoch=0, batch_size=None):
        idx = torch.from_numpy(self.indices[split])
        dataset = TensorDataset(self.images[idx], self.labels[idx], idx)
        generator = torch.Generator().manual_seed(self.config["seed"] + epoch)
        return DataLoader(dataset, batch_size=batch_size or self.config["batch_size"],
                          shuffle=split == "train", num_workers=self.config["num_workers"],
                          pin_memory=torch.cuda.is_available(), generator=generator, drop_last=False)

    def summary(self):
        return {"dataset": self.config["dataset"], "pool_order": "torchvision train then test",
                "split_algorithm": "numpy default_rng(seed).permutation(70000)",
                "seed": self.config["seed"], "sizes": {k: len(v) for k, v in self.indices.items()},
                "split_file": str(self.split_path), "split_sha256": self.split_sha256,
                "pixel_storage": "uint8 [0,255]; converted per batch to float32 [0,1]",
                "disjoint_and_complete": True}


def to_device(batch, device):
    images, labels, indices = batch
    return images.to(device, non_blocking=True).float().div_(255), labels.to(device), indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data = DataModule(cfg)
    save_json(Path(cfg["output_dir"]) / "data.json", data.summary())
    print(data.summary())
