"""Build the concise presentation notebook using this project's own definitions."""
from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def add(kind, text, tag=None):
    entry = {"cell_type": kind, "id": f"simple-{len(cells):02d}", "metadata": {},
             "source": textwrap.dedent(text).strip() + "\n"}
    if tag:
        entry["metadata"]["tags"] = [tag]
    if kind == "code":
        entry.update(execution_count=None, outputs=[])
    cells.append(entry)


def code(text, tag=None):
    add("code", text, tag)


def own_definitions(module, names):
    source = (ROOT / f"src/{module}.py").read_text(encoding="utf-8")
    nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    lines = source.splitlines()
    result = []
    for name in names:
        node = nodes[name]
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        result.append("\n".join(lines[start-1:node.end_lineno]))
    return "\n\n".join(result)


code("""
from __future__ import annotations
import hashlib
import json
import math
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST, MNIST
from IPython.display import Image, display
""", "imports")
config = (ROOT / "configs/default.json").read_text(encoding="utf-8").strip()
code('cfg = json.loads(r\'\'\'\n' + config + '\n\'\'\')\n\nRUN_MLP = False\n', "embedded-config")
code(own_definitions("common", ["seed_all", "get_device"]) + '\n\nseed_all(cfg["seed"])\ndevice = get_device(cfg)\nprint("Device:", device, "| Dataset:", cfg["dataset"], "| Seed:", cfg["seed"])\n')
code(own_definitions("data", ["DataModule", "to_device"]) + '''


data = None
if RUN_MLP:
    split_name = f'{cfg["dataset"]}_split_seed{cfg["seed"]}.npz'
    split_file = Path(cfg["data_dir"]) / split_name
    saved_split = Path("results/protocol") / split_name
    if not split_file.exists() and saved_split.exists():
        split_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved_split, split_file)
    data = DataModule(cfg)
    print(data.summary())
''', "data-definitions")
code(own_definitions("models", ["Classifier", "Decoder"]), "model-definitions")
code(own_definitions("models", ["Autoencoder", "VAE"]), "model-definitions")
code(own_definitions("models", ["VanillaGAN", "DCGAN"]), "model-definitions")
code(own_definitions("models", ["TimeEmbedding", "TimeBlock", "Denoiser", "DDPM", "build_model"]), "model-definitions")
code("""
def load_trained(name):
    path = Path(cfg["output_dir"]) / name / "best.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["config"] != cfg or saved["split_sha256"] != data.split_sha256:
        raise ValueError(f"Config/split checkpoint {name} tidak cocok")
    if saved.get("max_batches") is not None:
        raise ValueError("Checkpoint smoke tidak boleh menjadi hasil final")
    if name == "decoder":
        classifier_path = Path(cfg["output_dir"]) / "mlp/best.pt"
        if saved["classifier_sha256"] != hashlib.sha256(classifier_path.read_bytes()).hexdigest():
            raise ValueError("Decoder tidak cocok dengan classifier")
    model = build_model(name, cfg).to(device)
    model.load_state_dict(saved["model"])
    return model.eval().requires_grad_(False)
""", "checkpoint-definition")
code("""
mlp_demo = None
if RUN_MLP:
    classifier = load_trained("mlp")
    decoder = load_trained("decoder")
    ae = load_trained("ae")
    features, correct, total = [], 0, 0
    with torch.inference_mode():
        for batch in data.loader("test"):
            x, y, _ = to_device(batch, device)
            z = classifier.encode(x)
            features.append(z.cpu())
            correct += (classifier.fc3(z).argmax(1) == y).sum().item()
            total += len(x)
        features = torch.cat(features)
        x, _, _ = to_device(next(iter(data.loader("test"))), device)
        x = x[:10]
        mu = features.mean(0).to(device)
        sigma = features.std(0, correction=0).to(device)
        generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 14000)
        z_random = mu + sigma * torch.randn(16, cfg["latent_dim"], device=device, generator=generator)
        mlp_demo = {
            "accuracy": correct / total, "n_test": total,
            "W1_shape": list(classifier.fc1.weight.shape),
            "W2_shape": list(classifier.fc2.weight.shape),
            "W3_shape": list(classifier.fc3.weight.shape),
            "original": x.cpu(),
            "transpose_class": classifier.transpose_class(torch.eye(10, device=device)).cpu(),
            "transpose_latent": classifier.transpose_latent(features[:10].to(device)).cpu(),
            "frozen_decoder": decoder(features[:10].to(device)).cpu(),
            "vanilla_ae": ae(x)[0].cpu(),
            "gaussian_samples": decoder(z_random).cpu()
        }
    print(f"Akurasi test: {100 * mlp_demo['accuracy']:.2f}% dari {total} citra")
    print("W1/W2/W3:", mlp_demo["W1_shape"], mlp_demo["W2_shape"], mlp_demo["W3_shape"])
else:
    print("MLP dilewati. Aktifkan RUN_MLP setelah checkpoint tepercaya dan split asli tersedia.")
""", "mlp-action")
code("""
metrics_file = Path("results/evaluation/comparison.csv")
if metrics_file.exists():
    metrics = pd.read_csv(metrics_file, keep_default_na=False)
    display(metrics)
else:
    print("Tabel metrik belum tersedia.")
""", "saved-metrics")
code("""
figure_names = [
    "figure_1_reconstruction", "figure_2_sampling", "figure_3_mlp",
    "figure_4_interpolation", "figure_5_curves_trajectory"
]
for name in figure_names:
    path = Path("results/figures") / f"{name}.png"
    if path.exists():
        print(name)
        display(Image(filename=str(path)))
    else:
        print("Figur belum tersedia:", name)
""", "figure-gallery")
notebook = {"cells": cells,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"},
                         "language_info": {"name": "python", "version": "3.10.9"},
                         "description": "Code-only model notebook; optional MLP inference and display of saved experiment results"},
            "nbformat": 4, "nbformat_minor": 5}
(ROOT / "main.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("Wrote main.ipynb:", len(cells), "cells;", sum(len(c["source"].splitlines()) for c in cells if c["cell_type"] == "code"), "code lines")
