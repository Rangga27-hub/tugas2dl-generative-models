"""Synthetic fixtures ONLY: verify layout, never use these as experiment evidence."""
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from pypdf import PdfReader

from src.common import load_config, save_json


def test_draft_report_layout_with_synthetic_fixture(tmp_path):
    cfg = load_config()
    root = tmp_path / "synthetic_layout_fixture"
    cfg["output_dir"] = str(root)
    config_path = tmp_path / "config.json"
    save_json(config_path, cfg)
    checkpoint = {"sha256": "synthetic", "selected_epoch": 1, "completed_epochs": 1,
                  "mean_seconds_epoch": 1.0}
    metrics = {"MSE": 0.1, "PSNR": 10.0, "SSIM": 0.5, "n_test": 7000,
               "latent_near_zero_fraction": 0.5}
    env = {"gpu": "SYNTHETIC FIXTURE", "device": "cpu", "python": "test", "torch": "test",
           "torchvision": "test", "cuda_runtime": None}
    provenance = {"checkpoints": {}}
    for name in ("ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm"):
        result = {"split_sha256": "synthetic", "checkpoint": checkpoint, "environment": env,
                  "latency": {"median_ms_batch64": 1.0}}
        if name in ("ae", "sparse_ae", "vae"):
            result["reconstruction"] = metrics
        if name in ("vae", "gan", "dcgan", "ddpm"):
            result["fid"] = {"FID": 100.0, "n_real": 1000, "n_synthetic": 1000, "subset_sha256": "synthetic"}
        save_json(root / "evaluation" / f"{name}.json", result)
        provenance["checkpoints"][name] = checkpoint
    save_json(root / "data.json", {"split_sha256": "synthetic"})
    save_json(root / "mlp_experiments" / "results.json", {"test_accuracy": 0.5, "n_test": 7000, "target_met": False,
               "classifier": checkpoint, "vanilla_ae_metrics": metrics, "frozen_decoder_metrics": metrics})
    save_json(root / "figures" / "provenance.json", provenance)
    save_json(root / "figures" / "interpolation.json", {"fixture": True})
    for name, shape in (("figure_1_reconstruction", (500,1000)), ("figure_2_sampling", (350,1200)),
                        ("figure_3_mlp", (800,1200)), ("figure_4_interpolation", (700,900)),
                        ("figure_5_curves_trajectory", (1000,1200))):
        Image.fromarray(np.full((*shape, 3), 230, dtype=np.uint8)).save(root / "figures" / f"{name}.png")
    proc = subprocess.run([sys.executable, "-m", "src.report", "--config", str(config_path), "--draft"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    pdf = PdfReader(root / "report" / "DRAFT_Laporan.pdf")
    assert 10 <= len(pdf.pages) <= 12
    text = "\n".join(page.extract_text() for page in pdf.pages)
    assert "DRAFT" in text and "Q6" in text and "SYNTHETIC FIXTURE" in text
    # Missing identity/verification is intentionally rejected for a final report.
    rejected = subprocess.run([sys.executable, "-m", "src.report", "--config", str(config_path)],
                              capture_output=True, text=True)
    assert rejected.returncode != 0
