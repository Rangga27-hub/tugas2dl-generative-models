from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from .common import environment, get_device, load_config, save_json, seed_all, sync
from .data import DataModule, to_device
from .evaluate import build_fid, make_noise, sample
from .models import build_model
from .train import run_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--max-batches", type=int, help="Omit to measure one complete epoch")
    parser.add_argument("--skip-fid", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    device = get_device(cfg)
    data = DataModule(cfg)
    model = build_model("ddpm", cfg).to(device)
    opt = [torch.optim.Adam(model.parameters(), lr=cfg["ddpm_lr"])]
    # Warm-up is separate from the timed epoch and does not produce a final checkpoint.
    run_epoch("ddpm", model, data.loader("train"), device, cfg, opt, train=True, max_batches=2)
    sync(device)
    start = time.perf_counter()
    metrics = run_epoch("ddpm", model, data.loader("train"), device, cfg, opt, train=True, max_batches=args.max_batches)
    sync(device)
    epoch_seconds = time.perf_counter() - start
    model.eval()
    gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 15000)
    noise = make_noise("ddpm", 64, cfg, device, gen)
    sync(device)
    start = time.perf_counter()
    images = sample("ddpm", model, noise, gen)
    sync(device)
    sampling_seconds = time.perf_counter() - start
    batches = len(data.loader("train"))
    measured_batches = min(args.max_batches, batches) if args.max_batches else batches
    estimated_epoch_seconds = epoch_seconds * batches / measured_batches
    result = {"environment": environment(device), "config": cfg, "split_sha256": data.split_sha256,
              "measured_train_batches": measured_batches, "full_train_batches": batches,
              "full_epoch_measured": measured_batches == batches, "training_loss": metrics,
              "measured_training_seconds": epoch_seconds, "estimated_full_train_epoch_seconds": estimated_epoch_seconds,
              "train_timing_scope": "train only; excludes validation/I/O; two warm-up batches already executed",
              "reverse_sampling_seconds_batch64": sampling_seconds,
              "estimated_ddpm_training_seconds": estimated_epoch_seconds * cfg["epochs"]["ddpm"],
              "estimated_ddpm_fid_sampling_seconds": sampling_seconds * math.ceil(cfg["fid_samples"] / 64),
              "ddpm_steps": model.steps, "terminal_alpha_bar": model.abar[-1].item(),
              "estimation_limit": "single timing; validation, extractor time, and thermal variation add overhead"}
    if not args.skip_fid:
        metric = build_fid(device)
        x, _, _ = to_device(next(iter(data.loader("validation", batch_size=64))), device)
        sync(device)
        start = time.perf_counter()
        with torch.inference_mode():
            metric.update(x.repeat(1, 3, 1, 1), real=True)
            metric.update(images.repeat(1, 3, 1, 1), real=False)
        sync(device)
        result["fid_extractor_smoke"] = {"passed": True, "n_each": 64,
                                         "update_seconds_real_and_fake": time.perf_counter() - start,
                                         "note": "feature path test, not a final FID estimate"}
    save_json(Path(cfg["output_dir"]) / "benchmark_ddpm.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
