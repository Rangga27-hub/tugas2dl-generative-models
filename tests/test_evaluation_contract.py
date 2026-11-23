"""Contract tests use synthetic temporary records, not experiment results."""
import csv

import pytest

from src.common import load_config, save_json
from src.evaluate import write_table


def result(n, subset):
    return {"checkpoint": {"mean_seconds_epoch": 1.0}, "latency": {"median_ms_batch64": 1.0},
            "fid": {"FID": 100.0, "n_real": n, "subset_sha256": subset}}


def test_comparison_rejects_mixed_fid_subsets(tmp_path):
    cfg = load_config()
    cfg["output_dir"] = str(tmp_path)
    save_json(tmp_path / "evaluation/vae.json", result(1000, "subset-a"))
    save_json(tmp_path / "evaluation/gan.json", result(1000, "subset-b"))
    with pytest.raises(ValueError, match="inconsistent"):
        write_table(cfg)
    save_json(tmp_path / "evaluation/gan.json", result(7000, "subset-a"))
    with pytest.raises(ValueError, match="inconsistent"):
        write_table(cfg)


def test_comparison_marks_non_applicable_reconstruction(tmp_path):
    cfg = load_config()
    cfg["output_dir"] = str(tmp_path)
    save_json(tmp_path / "evaluation/gan.json", result(1000, "subset-a"))
    save_json(tmp_path / "visual_notes.json", {"gan": "synthetic test note"})
    write_table(cfg)
    with (tmp_path / "evaluation/comparison.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["MSE"] == rows[0]["PSNR"] == rows[0]["SSIM"] == "N/A"
    assert rows[0]["stability"] == "synthetic test note"
