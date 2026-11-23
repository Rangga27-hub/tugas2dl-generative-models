from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.utils import save_image

from .common import atomic_torch_save, get_device, load_config, save_json, seed_all
from .data import DataModule, to_device
from .evaluate import load_model, reconstruction_metrics


@torch.inference_mode()
def run(cfg):
    seed_all(cfg["seed"])
    device = get_device(cfg)
    data = DataModule(cfg)
    mlp, mlp_info = load_model("mlp", cfg, device, data)
    decoder, decoder_info = load_model("decoder", cfg, device, data)
    ae, ae_info = load_model("ae", cfg, device, data)
    output = Path(cfg["output_dir"]) / "mlp_experiments"
    output.mkdir(parents=True, exist_ok=True)
    codes, labels, ids = [], [], []
    correct, total = 0, 0
    for batch in data.loader("test"):
        x, y, idx = to_device(batch, device)
        z = mlp.encode(x)
        correct += (mlp.fc3(z).argmax(1) == y).sum().item()
        total += len(x)
        codes.append(z.cpu())
        labels.append(y.cpu())
        ids.append(idx)
    codes, labels, ids = torch.cat(codes), torch.cat(labels), torch.cat(ids)
    # Population std, not the unbiased sample estimator; diagonal approximation.
    mu, sigma = codes.mean(0), codes.std(0, correction=0)
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 14000)
    random_z = mu.to(device) + sigma.to(device) * torch.randn(16, cfg["latent_dim"], device=device, generator=generator)
    first = data.images[ids[:10]].to(device).float().div(255)
    one_hot = torch.eye(10, device=device)
    artifacts = {
        "indices": ids, "test_labels": labels, "test_latent": codes,
        "latent_mu": mu, "latent_sigma_population": sigma,
        "W1": mlp.fc1.weight.cpu(), "W2": mlp.fc2.weight.cpu(), "W3": mlp.fc3.weight.cpu(),
        "class_transpose": mlp.transpose_class(one_hot).cpu(),
        "originals": first.cpu(), "latent_transpose": mlp.transpose_latent(codes[:10].to(device)).cpu(),
        "frozen_reconstruction": decoder(codes[:10].to(device)).cpu(),
        "ae_reconstruction": ae(first)[0].cpu(), "diagonal_gaussian_samples": decoder(random_z).cpu(),
        "sampling_seed": cfg["seed"] + 14000,
        "test_statistics_warning": "Final analysis only; test latent statistics not used for model/checkpoint selection or generator FID"}
    atomic_torch_save(artifacts, output / "artifacts.pt")
    result = {"test_accuracy": correct / total, "target_accuracy": 0.88, "target_met": correct / total >= 0.88,
              "n_test": total, "weight_shapes": {k: list(artifacts[k].shape) for k in ("W1", "W2", "W3")},
              "classifier": mlp_info, "decoder": decoder_info, "vanilla_ae": ae_info,
              "frozen_decoder_metrics": reconstruction_metrics("decoder", decoder, data, device, cfg, mlp),
              "vanilla_ae_metrics": reconstruction_metrics("ae", ae, data, device, cfg),
              "sampling_distribution": "per-dimension Gaussian from test activations; population std; decoder locked",
              "sampling_seed": cfg["seed"] + 14000, "split_sha256": data.split_sha256,
              "figure_test_indices": ids[:10].tolist(), "transpose_bias": "not used, as specified by experimental formula"}
    save_json(output / "results.json", result)
    for name in ("class_transpose", "latent_transpose", "frozen_reconstruction", "ae_reconstruction", "diagonal_gaussian_samples"):
        save_image(artifacts[name], output / f"{name}.png", nrow=4 if name == "diagonal_gaussian_samples" else 10)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    run(load_config(parser.parse_args().config))
