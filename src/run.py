"""Sequential GPU queue, no concurrent model training on the same device."""
from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path

from .common import load_checkpoint, load_config, save_json
from .data import DataModule
from .train import train_model


def command(*args):
    print("RUN", sys.executable, *args, flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["smoke", "pilot", "train", "evaluate", "figures", "all-after-pilot"])
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--fid-n", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stage == "smoke":
        smoke = copy.deepcopy(cfg)
        smoke["output_dir"] += "_smoke"
        smoke["ddpm_steps"] = 8
        path = Path(smoke["output_dir"]) / "smoke_config.json"
        save_json(path, smoke)
        data = DataModule(smoke)
        for name in ("mlp", "ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm", "decoder"):
            last = Path(smoke["output_dir"]) / name / "last.pt"
            train_model(name, smoke, data, resume=last.exists(), epochs=1, max_batches=2)
            saved = load_checkpoint(last)
            assert saved["epoch"] == 1 and saved["max_batches"] == 2
        print("Smoke completed; these checkpoints are NOT final experiment evidence", flush=True)
    elif args.stage == "pilot":
        command("-m", "src.benchmark", "--config", args.config)
    else:
        if args.stage in ("train", "all-after-pilot"):
            benchmark_path = Path(cfg["output_dir"]) / "benchmark_ddpm.json"
            if not benchmark_path.exists():
                parser.error("Run pilot first and inspect runtime before full training")
            import json
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
            if benchmark["config"] != cfg or not benchmark.get("full_epoch_measured") or not benchmark.get("fid_extractor_smoke", {}).get("passed"):
                parser.error("Pilot must match config, measure full epoch and test FID extractor")
            data = DataModule(cfg)
            if benchmark["split_sha256"] != data.split_sha256:
                parser.error("Pilot split mismatch")
            # MLP establishes an early baseline; DDPM then gets the first long GPU slot.
            for name in ("mlp", "ddpm", "ae", "sparse_ae", "vae", "gan", "dcgan", "decoder"):
                last = Path(cfg["output_dir"]) / name / "last.pt"
                train_model(name, cfg, data, resume=last.exists())
        if args.stage in ("evaluate", "all-after-pilot"):
            extra = ["--fid-n", str(args.fid_n)] if args.fid_n is not None else []
            command("-m", "src.evaluate", "--config", args.config, "--fid", *extra)
            command("-m", "src.evaluate", "--config", args.config, "--models", "decoder")
            command("-m", "src.mlp_experiments", "--config", args.config)
        if args.stage in ("figures", "all-after-pilot"):
            command("-m", "src.figures", "--config", args.config)


if __name__ == "__main__":
    main()
