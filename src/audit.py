"""Check actual artifact consistency and optionally package small, reviewable evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from pypdf import PdfReader

from .common import load_checkpoint, load_config, save_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--destination", default="results")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["output_dir"])
    data = read_json(root / "data.json")
    split_file = Path(data["split_file"])
    if sha256(split_file) != data["split_sha256"]:
        raise ValueError("Persisted split changed")
    artifacts = [root / "data.json", root / "benchmark_ddpm.json"]
    summary = {"dataset": cfg["dataset"], "seed": cfg["seed"], "split_sha256": data["split_sha256"],
               "config_sha256": sha256(args.config), "model_runs": {}, "warnings": []}
    for name in ("mlp", "ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm", "decoder"):
        run = root / name
        best, last = load_checkpoint(run / "best.pt"), load_checkpoint(run / "last.pt")
        history = read_json(run / "history.json")
        for state in (best, last):
            if state["config"] != cfg or state["split_sha256"] != data["split_sha256"] or state["max_batches"] is not None:
                raise ValueError(f"{name}: invalid provenance or smoke checkpoint")
        if last["epoch"] < cfg["epochs"][name] or len(history) != last["epoch"]:
            raise ValueError(f"{name}: incomplete budget/history")
        if history != last["history"]:
            raise ValueError(f"{name}: history differs from checkpoint")
        summary["model_runs"][name] = {"completed_epochs": last["epoch"], "selected_epoch": best["epoch"],
                                       "best_checkpoint_sha256": sha256(run / "best.pt"),
                                       "checkpoint_local_path": str(run / "best.pt")}
        artifacts.extend(run / file for file in ("history.csv", "history.json", "config.json", "environment.json"))
        if name in ("gan", "dcgan"):
            for epoch in sorted({1, max(1, last["epoch"]//2), last["epoch"]}):
                artifacts.append(run / f"samples_epoch_{epoch:03d}.png")
    n_values = set()
    for name in ("ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm"):
        path = root / "evaluation" / f"{name}.json"
        result = read_json(path)
        if result["checkpoint"]["sha256"] != summary["model_runs"][name]["best_checkpoint_sha256"]:
            raise ValueError(f"{name}: metrics from stale checkpoint")
        if result["split_sha256"] != data["split_sha256"]:
            raise ValueError("Evaluation split mismatch")
        if name in ("ae", "sparse_ae", "vae") and result["reconstruction"]["n_test"] != 7000:
            raise ValueError("Incomplete reconstruction test set")
        if name in ("vae", "gan", "dcgan", "ddpm"):
            fid = result["fid"]
            n_values.add((fid["n_real"], fid["n_synthetic"], fid["subset_sha256"]))
            if sha256(fid["subset_file"]) != fid["subset_sha256"]:
                raise ValueError("FID subset changed")
            artifacts.append(Path(fid["subset_file"]))
        artifacts.append(path)
    if len(n_values) != 1:
        raise ValueError("Inconsistent FID protocols")
    summary["fid_protocol"] = dict(zip(("n_real", "n_synthetic", "subset_sha256"), next(iter(n_values))))
    mlp_path = root / "mlp_experiments" / "results.json"
    mlp = read_json(mlp_path)
    summary["mlp_test_accuracy"] = mlp["test_accuracy"]
    summary["mlp_target_met"] = mlp["target_met"]
    for key, name in (("classifier", "mlp"), ("decoder", "decoder"), ("vanilla_ae", "ae")):
        if mlp[key]["sha256"] != summary["model_runs"][name]["best_checkpoint_sha256"]:
            raise ValueError("MLP experiment uses stale checkpoints")
    artifacts.append(mlp_path)
    figures = root / "figures"
    provenance = read_json(figures / "provenance.json")
    for name in ("mlp", "ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm"):
        if provenance["checkpoints"][name]["sha256"] != summary["model_runs"][name]["best_checkpoint_sha256"]:
            raise ValueError("Figures use stale checkpoints")
    for pattern in ("figure_*.png", "figure_*.pdf", "*.json"):
        artifacts.extend(figures.glob(pattern))
    if len(list(figures.glob("figure_*.png"))) != 5:
        raise ValueError("Expected exactly five required figures")
    artifacts.append(root / "evaluation" / "comparison.csv")
    report_audit = read_json(root / "report" / "audit.json")
    report_path = Path(report_audit["pdf"])
    pages = len(PdfReader(report_path).pages)
    if not 10 <= pages <= 12:
        raise ValueError("PDF must have 10–12 pages")
    summary["report"] = {**report_audit, "verified_pages": pages, "sha256": sha256(report_path)}
    if report_audit["draft"]:
        summary["warnings"].append("Report is DRAFT: not submission-ready; verify notes, identity, public GitHub URL and PDF render")
    if not mlp["target_met"]:
        summary["warnings"].append("MLP test accuracy below required threshold")
    artifacts.extend([report_path, root / "report" / "audit.json"])
    tests_xml = Path("artifacts/setup/tests.xml")
    if tests_xml.exists():
        tree = ET.parse(tests_xml)
        suites = list(tree.getroot().iter("testsuite"))
        summary["tests"] = {k: sum(int(s.attrib.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
        if summary["tests"]["failures"] or summary["tests"]["errors"]:
            raise ValueError("Test suite failed")
    else:
        summary["warnings"].append("JUnit test evidence not found")
    summary["code_commit_at_audit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    save_json(root / "audit.json", summary)
    if args.export:
        destination = Path(args.destination)
        destination.mkdir(parents=True, exist_ok=True)
        exports = {}
        for source in sorted(set(artifacts)):
            relative = source.relative_to(root) if source.is_relative_to(root) else Path("protocol") / source.name
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            exports[str(relative)] = sha256(target)
        for source, relative in [(split_file, Path("protocol") / split_file.name),
                                 (Path(args.config), Path("protocol/config.json")),
                                 (Path("requirements-lock.txt"), Path("protocol/requirements-lock.txt")),
                                 (tests_xml, Path("protocol/tests.xml"))]:
            if source.exists():
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                exports[str(relative)] = sha256(target)
        save_json(destination / "audit.json", summary)
        save_json(destination / "manifest.json", exports)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
