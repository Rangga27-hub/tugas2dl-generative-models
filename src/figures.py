from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from .common import atomic_torch_save, get_device, load_checkpoint, load_config, save_json, seed_all
from .data import DataModule
from .evaluate import load_model, make_noise, reconstruct, sample

LABELS = {"ae": "Vanilla AE", "sparse_ae": "Sparse AE", "vae": "VAE", "gan": "Vanilla GAN", "dcgan": "DCGAN", "ddpm": "DDPM"}


def image(ax, tensor, title=None):
    ax.imshow(tensor.detach().cpu().squeeze(), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def grid(ax, tensor, nrow=4, title=None):
    data = make_grid(tensor.detach().cpu(), nrow=nrow, padding=2, pad_value=1)
    ax.imshow(data.permute(1, 2, 0).numpy(), vmin=0, vmax=1)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)


def export(fig, root, name):
    fig.savefig(root / f"{name}.png", dpi=250, bbox_inches="tight")
    fig.savefig(root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def main(cfg):
    seed_all(cfg["seed"])
    device = get_device(cfg)
    data = DataModule(cfg)
    root = Path(cfg["output_dir"])
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    names = ("ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm")
    models, metadata = {}, {}
    for name in names:
        models[name], metadata[name] = load_model(name, cfg, device, data)
    classifier, metadata["mlp"] = load_model("mlp", cfg, device, data)
    mlp = load_checkpoint(root / "mlp_experiments" / "artifacts.pt")
    selected = data.indices["test"][:8]
    x = data.images[selected].to(device).float().div(255)
    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for row, name in enumerate(("original", "ae", "sparse_ae", "vae")):
        values = x if name == "original" else reconstruct(name, models[name], x)
        for col in range(8):
            image(axes[row, col], values[col])
        axes[row, 0].set_ylabel("Original" if row == 0 else LABELS[name], fontsize=9)
    fig.suptitle("Figur 1 — Rekonstruksi citra test yang sama")
    fig.tight_layout()
    export(fig, out, "figure_1_reconstruction")

    samples = {}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    for i, name in enumerate(("vae", "gan", "dcgan", "ddpm")):
        gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 16000)
        noise = make_noise(name, 16, cfg, device, gen)
        samples[name] = sample(name, models[name], noise, gen).cpu()
        grid(axes[i], samples[name], title=LABELS[name])
    fig.suptitle("Figur 2 — Sampling dari prior (16 citra per model)")
    export(fig, out, "figure_2_sampling")
    atomic_torch_save(samples, out / "samples.pt")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    grid(axes[0, 0], mlp["originals"], 10, "A. Original test (10 citra)")
    grid(axes[0, 1], torch.cat([mlp["class_transpose"], mlp["latent_transpose"]]), 10,
         "B. Transpos kelas 0–9 (atas); transpos laten test (bawah)")
    grid(axes[1, 0], torch.cat([mlp["frozen_reconstruction"], mlp["ae_reconstruction"]]), 10,
         "C. Frozen decoder (atas); Vanilla AE laten 64 (bawah)")
    grid(axes[1, 1], mlp["diagonal_gaussian_samples"], 4,
         "D. Gaussian diagonal dari statistik fitur test (analisis akhir)")
    fig.suptitle("Figur 3 — Lima tahap eksperimen MLP")
    export(fig, out, "figure_3_mlp")

    # Same true test endpoints for AE/VAE; contrasting Fashion-MNIST classes 0 and 9.
    target_classes = (0, 9)
    test_ids = torch.from_numpy(data.indices["test"])
    test_labels = data.labels[test_ids]
    endpoint_ids = [int(test_ids[(test_labels == c).nonzero()[0]]) for c in target_classes]
    endpoints = data.images[endpoint_ids].to(device).float().div(255)
    alpha = torch.linspace(0, 1, 6, device=device)[:, None]
    interpolation_meta = {"classes": target_classes, "test_endpoint_indices": endpoint_ids,
                          "alphas": alpha.flatten().tolist(), "gan_selection_seed": cfg["seed"] + 17000,
                          "gan_selection_method": "2048 fixed prior candidates, select maximum MLP confidence for predicted classes 0 and 9; requires visual verification, classifier is only proxy"}
    fig, axes = plt.subplots(5, 6, figsize=(9, 7))
    interpolation_tensors = {}
    for row, name in enumerate(("ae", "sparse_ae", "vae", "gan", "dcgan")):
        model = models[name]
        if name in ("ae", "sparse_ae"):
            z = model.encoder(endpoints)
        elif name == "vae":
            z = model.encode(endpoints)[0]
        else:
            gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 17000)
            candidate_z = make_noise(name, 2048, cfg, device, gen)
            probabilities = []
            for batch in candidate_z.split(128):
                probabilities.append(classifier(sample(name, model, batch)).softmax(1))
            probs = torch.cat(probabilities)
            predicted = probs.argmax(1)
            selected_candidates, confidence = [], []
            for c in target_classes:
                valid = (predicted == c).nonzero().flatten()
                if len(valid) == 0:
                    raise RuntimeError(f"{name} has no candidate predicted class {c}; inspect coverage, do not fabricate endpoints")
                best = valid[probs[valid, c].argmax()]
                selected_candidates.append(int(best))
                confidence.append(float(probs[best, c]))
            z = candidate_z[selected_candidates]
            interpolation_meta[name] = {"candidate_indices": selected_candidates, "mlp_confidence": confidence,
                                         "visual_verification": "pending human inspection"}
        intermediate = (1 - alpha) * z[:1] + alpha * z[1:]
        values = model.decoder(intermediate) if name in ("ae", "sparse_ae", "vae") else sample(name, model, intermediate)
        interpolation_tensors[name] = values.cpu()
        for col in range(6):
            image(axes[row, col], values[col], f"α={col/5:.1f}" if row == 0 else None)
        axes[row, 0].set_ylabel(LABELS[name], fontsize=9)
    fig.suptitle("Figur 4 — Interpolasi linear; endpoint kelas 0 vs 9")
    fig.tight_layout()
    export(fig, out, "figure_4_interpolation")
    save_json(out / "interpolation.json", interpolation_meta)
    atomic_torch_save(interpolation_tensors, out / "interpolation.pt")

    fig = plt.figure(figsize=(12, 10))
    layout = fig.add_gridspec(4, 6)
    for i, name in enumerate(names):
        ax = fig.add_subplot(layout[i // 2, (i % 2)*3:(i % 2)*3+3])
        history = json.loads((root / name / "history.json").read_text(encoding="utf-8"))
        epoch = [h["epoch"] for h in history]
        if name in ("gan", "dcgan"):
            for key, label in (("train_g_loss", "G"), ("train_d_loss", "D")):
                ax.plot(epoch, [h[key] for h in history], label=label)
        else:
            ax.plot(epoch, [h["train_loss"] for h in history], label="train objective")
            ax.plot(epoch, [h["val_loss"] for h in history], label="validation objective")
            if name == "vae":
                ax.plot(epoch, [h["train_recon_bce_sum"] for h in history], label="BCE")
                ax.plot(epoch, [h["train_kl_sum"] for h in history], label="KL")
        ax.set_title(LABELS[name], fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.set_ylabel("loss (model-specific)", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
    gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 18000)
    noise = make_noise("ddpm", 1, cfg, device, gen)
    _, chain = models["ddpm"].sample(noise, gen, snapshots=True)
    for col, (t, state) in enumerate(chain.items()):
        ax = fig.add_subplot(layout[3, col])
        image(ax, state.add(1).div(2).clamp(0, 1), f"x_{t}")
    fig.suptitle("Figur 5 — Kurva loss dan satu reverse chain DDPM")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    export(fig, out, "figure_5_curves_trajectory")
    atomic_torch_save(chain, out / "ddpm_trajectory.pt")
    save_json(out / "provenance.json", {"checkpoints": metadata, "split_sha256": data.split_sha256,
                                        "figure_1_test_indices": selected.tolist(),
                                        "sampling_seed": cfg["seed"] + 16000, "sampling_indices": list(range(16)),
                                        "trajectory_seed": cfg["seed"] + 18000})
    print(f"Figures exported to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    main(load_config(parser.parse_args().config))
