from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchmetrics.functional.image import structural_similarity_index_measure

from .common import (atomic_torch_save, environment, get_device, load_checkpoint,
                     load_config, save_json, seed_all, sync)
from .data import DataModule, to_device
from .models import build_model

GENERATORS = ("vae", "gan", "dcgan", "ddpm")
RECONSTRUCTORS = ("ae", "sparse_ae", "vae", "decoder")


def load_model(name, cfg, device, data=None):
    path = Path(cfg["output_dir"]) / name / "best.pt"
    saved = load_checkpoint(path)
    if saved["name"] != name or saved["config"] != cfg:
        raise ValueError(f"Configuration mismatch in {path}")
    if saved["max_batches"] is not None:
        raise ValueError("Smoke checkpoints cannot be used for final evaluation")
    if data is not None and saved["split_sha256"] != data.split_sha256:
        raise ValueError("Split mismatch")
    model = build_model(name, cfg).to(device)
    model.load_state_dict(saved["model"])
    model.eval().requires_grad_(False)
    last = load_checkpoint(path.parent / "last.pt")
    if last["config"] != cfg or last["split_sha256"] != saved["split_sha256"]:
        raise ValueError("Final/selected checkpoint provenance mismatch")
    metadata = {"checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "selected_epoch": saved["epoch"], "completed_epochs": last["epoch"],
                "mean_seconds_epoch": statistics.mean(x["seconds_train_validation"] for x in last["history"]),
                "timing_scope": saved["timing_scope"], "selection": cfg["checkpoint_selection"][name]}
    if name == "decoder":
        mlp_path = Path(cfg["output_dir"]) / "mlp" / "best.pt"
        if saved["classifier_sha256"] != hashlib.sha256(mlp_path.read_bytes()).hexdigest():
            raise ValueError("Decoder belongs to another classifier checkpoint")
    return model, metadata


@torch.inference_mode()
def reconstruct(name, model, x, classifier=None):
    if name in ("ae", "sparse_ae"):
        return model(x)[0]
    if name == "vae":
        return model(x, deterministic=True)[0]
    if name == "decoder":
        return model(classifier.encode(x))
    raise ValueError(name)


def reconstruction_scores(pred, target):
    mse = (pred - target).square().flatten(1).mean(1)
    # Exact zero uses float32 tiny to produce a finite, explicitly capped PSNR.
    psnr = -10 * torch.log10(mse.clamp_min(torch.finfo(torch.float32).tiny))
    ssim = structural_similarity_index_measure(pred, target, data_range=1.0, reduction="none")
    return mse, psnr, ssim


@torch.inference_mode()
def reconstruction_metrics(name, model, data, device, cfg, classifier=None):
    sums = torch.zeros(3, dtype=torch.float64, device=device)
    n, zero_count, sparse_count, sparse_total = 0, 0, 0, 0
    for batch in data.loader("test"):
        x, _, _ = to_device(batch, device)
        pred = reconstruct(name, model, x, classifier).clamp(0, 1)
        values = reconstruction_scores(pred, x)
        sums += torch.stack([v.double().sum() for v in values])
        zero_count += (values[0] == 0).sum().item()
        n += len(x)
        if name in ("ae", "sparse_ae"):
            z = model.encoder(x)
            sparse_count += (z.abs() < cfg["sparsity_threshold"]).sum().item()
            sparse_total += z.numel()
    result = dict(zip(("MSE", "PSNR", "SSIM"), (sums / n).cpu().tolist()))
    result.update({"n_test": n, "zero_mse_images": zero_count, "psnr_zero_policy": "clamp MSE to float32 tiny",
                   "aggregation": "mean of per-image metrics; SSIM Gaussian 11x11 sigma1.5, data_range=1"})
    if sparse_total:
        result.update({"latent_near_zero_fraction": sparse_count / sparse_total,
                       "latent_near_zero_threshold": cfg["sparsity_threshold"]})
    return result


@torch.inference_mode()
def sample(name, model, noise, generator=None):
    if name == "vae":
        return model.decoder(noise).clamp(0, 1)
    if name in ("gan", "dcgan"):
        return model(noise).add(1).div(2).clamp(0, 1)
    if name == "ddpm":
        return model.sample(noise, generator=generator).add(1).div(2).clamp(0, 1)
    raise ValueError(name)


def make_noise(name, n, cfg, device, generator):
    shape = (n, 1, 28, 28) if name == "ddpm" else (n, cfg["latent_dim"] if name == "vae" else cfg["noise_dim"])
    return torch.randn(shape, device=device, generator=generator)


@torch.inference_mode()
def latency(name, model, cfg, device, real_batch=None):
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 12000)
    value = real_batch[:64] if name in ("ae", "sparse_ae") else make_noise(name, 64, cfg, device, generator)
    if len(value) != 64:
        raise ValueError("Timing needs exactly 64 images")
    def operation():
        return model(value)[0] if name in ("ae", "sparse_ae") else sample(name, model, value, generator)
    for _ in range(cfg["timing_warmup"]):
        operation()
    times = []
    for _ in range(cfg["timing_repeats"]):
        sync(device)
        start = time.perf_counter()
        operation()
        sync(device)
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms_batch64": statistics.median(times), "all_ms": times,
            "warmup": cfg["timing_warmup"], "repeats": cfg["timing_repeats"],
            "scope": "AE reconstruction / generators from prepared initial noise; DDPM includes full reverse chain and per-step RNG; no H2D or disk I/O",
            "ddpm_steps": cfg["ddpm_steps"] if name == "ddpm" else None}


def build_fid(device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    # float32 RGB in [0,1], internally converted to uint8 by normalize=True.
    # torch-fidelity compatible Inception-v3 features, no manual resize.
    return FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False).to(device)


@torch.inference_mode()
def fid_metrics(name, model, data, cfg, device, n):
    if not 2 <= n <= 7000:
        raise ValueError("FID needs 2 <= N <= 7000")
    metric = build_fid(device)
    classifier, classifier_info = load_model("mlp", cfg, device, data)
    predicted_counts = torch.zeros(10, dtype=torch.long, device=device)
    confidence_sum = 0.0
    indices = np.random.default_rng(cfg["seed"] + 333).permutation(data.indices["test"])[:n]
    output = Path(cfg["output_dir"]) / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    index_path = output / f"fid_real_indices_n{n}.npy"
    if index_path.exists() and not np.array_equal(np.load(index_path), indices):
        raise ValueError("Existing FID subset differs")
    np.save(index_path, indices)
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 13000)
    batch_size = cfg["fid_batch_size"]
    sync(device)
    start = time.perf_counter()
    for i in range(0, n, batch_size):
        real = data.images[indices[i:i+batch_size]].to(device).float().div(255)
        metric.update(real.repeat(1, 3, 1, 1), real=True)
        noise = make_noise(name, len(real), cfg, device, generator)
        fake = sample(name, model, noise, generator)
        metric.update(fake.repeat(1, 3, 1, 1), real=False)
        probabilities = classifier(fake).softmax(1)
        confidence, prediction = probabilities.max(1)
        predicted_counts += torch.bincount(prediction, minlength=10)
        confidence_sum += confidence.sum().item()
    score = metric.compute().item()
    sync(device)
    return {"FID": score, "n_real": n, "n_synthetic": n, "seconds": time.perf_counter() - start,
            "subset_file": str(index_path), "subset_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            "feature_dim": 2048, "extractor": "torch-fidelity Inception-v3 via TorchMetrics",
            "preprocessing": "grayscale repeated RGB; float32 [0,1]; normalize=True; extractor handles resize",
            "seed": cfg["seed"] + 13000, "batch_size": batch_size,
            "predicted_label_proxy": {"counts": predicted_counts.cpu().tolist(),
                                      "mean_confidence": confidence_sum / n,
                                      "classifier_sha256": classifier_info["sha256"],
                                      "warning": "Classifier predictions on synthetic/OOD images are not true labels, nor formal precision/recall or proof of mode coverage"},
            "limitation": "ImageNet features on 28x28 grayscale; finite-sample bias, especially N<=2048; not directly comparable to other pipelines"}


def write_table(cfg):
    root = Path(cfg["output_dir"]) / "evaluation"
    notes_path = Path(cfg["output_dir"]) / "visual_notes.json"
    if notes_path.exists():
        import json
        notes = json.loads(notes_path.read_text(encoding="utf-8"))
    else:
        notes = {}
    rows = []
    fid_n = set()
    for name in ("ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm"):
        path = root / f"{name}.json"
        if not path.exists():
            continue
        import json
        result = json.loads(path.read_text(encoding="utf-8"))
        fid = result.get("fid", {})
        if fid:
            fid_n.add((fid["n_real"], fid["subset_sha256"]))
        row = {"model": name, "latent_dim": "1x28x28" if name == "ddpm" else cfg["latent_dim"] if name in RECONSTRUCTORS else cfg["noise_dim"],
               "seconds_epoch": result["checkpoint"]["mean_seconds_epoch"],
               "ms_batch64": result["latency"]["median_ms_batch64"],
               **{key: result.get("reconstruction", {}).get(key, "N/A") for key in ("MSE", "PSNR", "SSIM")},
               "FID": fid.get("FID", "not measured" if name in GENERATORS else "N/A"),
               "FID_N": fid.get("n_real", "N/A"), "stability": notes.get(name) or "requires curve and sample inspection"}
        rows.append(row)
    if len(fid_n) > 1:
        raise ValueError("Do not compare FID with inconsistent N/subsets; reevaluate all generators")
    if rows:
        with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--models", nargs="+", default=["ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm"])
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--fid-n", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    device = get_device(cfg)
    data = DataModule(cfg)
    root = Path(cfg["output_dir"]) / "evaluation"
    x, _, _ = to_device(next(iter(data.loader("test"))), device)
    for name in args.models:
        model, info = load_model(name, cfg, device, data)
        result = {"checkpoint": info, "environment": environment(device), "split_sha256": data.split_sha256}
        classifier = load_model("mlp", cfg, device, data)[0] if name == "decoder" else None
        if name in RECONSTRUCTORS:
            result["reconstruction"] = reconstruction_metrics(name, model, data, device, cfg, classifier)
        if name != "decoder":
            result["latency"] = latency(name, model, cfg, device, x)
        if args.fid and name in GENERATORS:
            result["fid"] = fid_metrics(name, model, data, cfg, device, args.fid_n or cfg["fid_samples"])
        save_json(root / f"{name}.json", result)
        print(name, result, flush=True)
        del model, classifier
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_table(cfg)


if __name__ == "__main__":
    main()
