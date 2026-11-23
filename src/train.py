from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.utils import save_image

from .common import (atomic_torch_save, environment, get_device, load_checkpoint,
                     load_config, restore_rng, rng_state, save_json, seed_all, sync)
from .data import DataModule, to_device
from .models import build_model

NAMES = ("mlp", "ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm", "decoder")


def objective(name, model, x, y, cfg, classifier=None):
    if name in ("ae", "sparse_ae"):
        pred, z = model(x)
        mse = F.mse_loss(pred, x)
        penalty = z.abs().sum(1).mean()
        loss = mse + (cfg["sparse_lambda"] * penalty if name == "sparse_ae" else 0)
        return loss, {"mse": mse, "latent_l1_sum": penalty}
    if name == "vae":
        pred, mu, logvar = model(x)
        recon = F.binary_cross_entropy(pred, x, reduction="none").flatten(1).sum(1).mean()
        kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp()).sum(1)).mean()
        return recon + kl, {"recon_bce_sum": recon, "kl_sum": kl}
    if name == "mlp":
        logits = model(x)
        return F.cross_entropy(logits, y), {"accuracy": (logits.argmax(1) == y).float().mean()}
    if name == "decoder":
        with torch.no_grad():
            z = classifier.encode(x)
        return F.mse_loss(model(z), x), {}
    if name == "ddpm":
        t = torch.randint(model.steps, (len(x),), device=x.device)
        noise = torch.randn_like(x)
        xt = model.q_sample(x * 2 - 1, t, noise)
        return F.mse_loss(model(xt, t), noise), {}
    raise ValueError(name)


def gan_step(model, x, optimizers, cfg, train):
    n = len(x)
    opt_g, opt_d = optimizers
    real = x * 2 - 1
    z = torch.randn(n, cfg["noise_dim"], device=x.device)
    if train:
        opt_d.zero_grad(set_to_none=True)
    fake = model.generator(z)
    real_logits, fake_logits = model.discriminator(real), model.discriminator(fake.detach())
    d_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    d_loss += F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    if train:
        d_loss.backward()
        opt_d.step()
        opt_g.zero_grad(set_to_none=True)
        model.discriminator.requires_grad_(False)
    logits = model.discriminator(fake)
    g_loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    if train:
        g_loss.backward()
        opt_g.step()
        model.discriminator.requires_grad_(True)
    return g_loss, {"g_loss": g_loss, "d_loss": d_loss}


def run_epoch(name, model, loader, device, cfg, optimizers, classifier=None, train=True, max_batches=None):
    model.train(train)
    totals, count = {}, 0
    with torch.set_grad_enabled(train):
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x, y, _ = to_device(batch, device)
            if name in ("gan", "dcgan"):
                loss, parts = gan_step(model, x, optimizers, cfg, train)
            else:
                if train:
                    optimizers[0].zero_grad(set_to_none=True)
                loss, parts = objective(name, model, x, y, cfg, classifier)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite {name} loss")
                if train:
                    loss.backward()
                    optimizers[0].step()
            values = {"loss": loss, **parts}
            for key, value in values.items():
                number = value.detach().item()
                if not torch.isfinite(torch.tensor(number)):
                    raise FloatingPointError(f"Non-finite {name}/{key}")
                totals[key] = totals.get(key, 0.0) + number * len(x)
            count += len(x)
    if count == 0:
        raise ValueError("Empty epoch")
    return {k: v / count for k, v in totals.items()}


def train_model(name, cfg, data=None, resume=False, epochs=None, max_batches=None):
    seed_all(cfg["seed"])
    device = get_device(cfg)
    root = Path(cfg["output_dir"])
    run = root / name
    run.mkdir(parents=True, exist_ok=True)
    data = data or DataModule(cfg)
    model = build_model(name, cfg).to(device)
    classifier, classifier_hash = None, None
    if name == "decoder":
        classifier_path = root / "mlp" / "best.pt"
        saved = load_checkpoint(classifier_path)
        if saved["split_sha256"] != data.split_sha256:
            raise ValueError("Classifier split mismatch")
        classifier_hash = hashlib.sha256(classifier_path.read_bytes()).hexdigest()
        classifier = build_model("mlp", cfg).to(device)
        classifier.load_state_dict(saved["model"])
        classifier.eval().requires_grad_(False)
    if name in ("gan", "dcgan"):
        optimizers = [torch.optim.Adam(model.generator.parameters(), lr=cfg["gan_lr"], betas=(0.5, 0.999)),
                      torch.optim.Adam(model.discriminator.parameters(), lr=cfg["gan_lr"], betas=(0.5, 0.999))]
    else:
        lr = cfg["ddpm_lr"] if name == "ddpm" else cfg["lr"]
        optimizers = [torch.optim.Adam(model.parameters(), lr=lr)]
    target = epochs if epochs is not None else cfg["epochs"][name]
    if target < 1:
        raise ValueError("Epochs must be positive")
    start, best, history = 0, float("inf"), []
    last = run / "last.pt"
    if last.exists() and not resume:
        raise FileExistsError(f"{last} exists; use --resume or another output_dir")
    if resume:
        saved = load_checkpoint(last)
        if saved["name"] != name or saved["split_sha256"] != data.split_sha256 or saved["config"] != cfg:
            raise ValueError("Resume requires identical model, config and split; use --epochs to extend budget")
        if saved.get("classifier_sha256") != classifier_hash:
            raise ValueError("Frozen classifier changed; cannot resume decoder")
        if saved["max_batches"] != max_batches:
            raise ValueError("Cannot mix smoke and full epochs")
        model.load_state_dict(saved["model"])
        for opt, state in zip(optimizers, saved["optimizers"]):
            opt.load_state_dict(state)
        start, best, history = saved["epoch"], saved["best_score"], saved["history"]
        restore_rng(saved["rng"])
    env = environment(device)
    save_json(run / "environment.json", env)
    save_json(run / "config.json", cfg)
    save_json(root / "data.json", data.summary())
    fixed_gen = torch.Generator(device=device).manual_seed(cfg["seed"] + 9000)
    fixed_z = torch.randn(16, cfg["noise_dim"], device=device, generator=fixed_gen)
    for epoch in range(start, target):
        sync(device)
        begin = time.perf_counter()
        train_metrics = run_epoch(name, model, data.loader("train", epoch), device, cfg, optimizers,
                                  classifier, True, max_batches)
        # Fixed validation random draws across epochs, without changing training RNG.
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg["seed"] + 555)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(cfg["seed"] + 555)
            val_metrics = run_epoch(name, model, data.loader("validation"), device, cfg, optimizers,
                                    classifier, False, max_batches)
        sync(device)
        seconds = time.perf_counter() - begin
        score = -val_metrics["accuracy"] if name == "mlp" else val_metrics["loss"]
        improved = score < best or name in ("gan", "dcgan")
        if improved:
            best = score
        record = {"epoch": epoch + 1, "seconds_train_validation": seconds,
                  **{"train_" + k: v for k, v in train_metrics.items()},
                  **{"val_" + k: v for k, v in val_metrics.items()}}
        history.append(record)
        state = {"name": name, "model": model.state_dict(), "optimizers": [o.state_dict() for o in optimizers],
                 "epoch": epoch + 1, "best_score": best, "config": cfg, "history": history,
                 "rng": rng_state(), "environment": env, "split_sha256": data.split_sha256,
                 "max_batches": max_batches, "classifier_sha256": classifier_hash,
                 "timing_scope": "train+validation; excludes checkpoint/log/sample I/O"}
        atomic_torch_save(state, last)
        if improved:
            atomic_torch_save(state, run / "best.pt")
        save_json(run / "history.json", history)
        with (run / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        if name in ("gan", "dcgan"):
            model.eval()
            with torch.no_grad():
                samples = model(fixed_z).add(1).div(2).clamp(0, 1)
            save_image(samples, run / f"samples_epoch_{epoch+1:03d}.png", nrow=4)
        print(f"{name} epoch {epoch+1}/{target} {seconds:.2f}s train={train_metrics} val={val_metrics}", flush=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--model", choices=NAMES, required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-batches", type=int, help="Smoke only; output_dir must be separate from final runs")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.max_batches is not None and (args.max_batches < 1 or "smoke" not in cfg["output_dir"].lower()):
        parser.error("--max-batches needs positive count and a smoke output_dir")
    train_model(args.model, cfg, resume=args.resume, epochs=args.epochs, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
