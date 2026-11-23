from __future__ import annotations

import json
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import torchvision


def load_config(path="configs/default.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(temp, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(min(8, os.cpu_count() or 1))


def get_device(config):
    name = config.get("device", "auto")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment(device):
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "precision": "float32", "cudnn_deterministic": torch.backends.cudnn.deterministic}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def load_checkpoint(path, device="cpu"):
    # Only load trusted, locally generated checkpoints. RNG state requires pickle.
    return torch.load(path, map_location=device, weights_only=False)
