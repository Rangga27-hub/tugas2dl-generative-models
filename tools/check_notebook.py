"""Execute notebook-only and bounded integration paths; save evidence outside Git."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def fingerprint_originals():
    paths = list((ROOT / "src").glob("*.py")) + list((ROOT / "configs").glob("*.json"))
    paths += [p for p in (ROOT / "results").rglob("*") if p.is_file()]
    paths += [p for p in (ROOT / "artifacts/fashion").rglob("*") if p.is_file()]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(nb, cwd, evidence, filename):
    # Guard imports even when integration executes with the repo as cwd.
    guard = nbformat.v4.new_code_cell('''
import sys
import importlib.abc
class BlockProjectImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "src" or fullname.startswith("src."):
            raise AssertionError("Notebook must not import project modules")
sys.meta_path.insert(0, BlockProjectImport())
''')
    nb.cells.insert(0, guard)
    client = NotebookClient(nb, timeout=240, kernel_name="python3", resources={"metadata": {"path": str(cwd)}})
    client.execute()
    nbformat.write(nb, evidence / filename)
    return {"code_cells_executed": sum(c.cell_type == "code" for c in nb.cells),
            "errors": sum(o.output_type == "error" for c in nb.cells if c.cell_type == "code" for o in c.outputs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true", help="Requires original local data and all checkpoints")
    args = parser.parse_args()
    evidence = ROOT / "artifacts/notebook_verification"
    evidence.mkdir(parents=True, exist_ok=True)
    original = fingerprint_originals()
    summary = {}
    with tempfile.TemporaryDirectory(prefix="standalone_notebook_") as temporary:
        cwd = Path(temporary)
        nb = nbformat.read(ROOT / "main.ipynb", as_version=4)
        nbformat.write(nb, cwd / "main.ipynb")
        summary["standalone_default"] = run(nb, cwd, evidence, "standalone.executed.ipynb")
        summary["standalone_default"].update({"repo_files_available": False, "full_training": False})
    if args.integration:
        nb = nbformat.read(ROOT / "main.ipynb", as_version=4)
        for c in nb.cells:
            if "controls" in c.metadata.get("tags", []):
                c.source = c.source.replace("RUN_SMOKE = False", "RUN_SMOKE = True")
                c.source = c.source.replace("RECOMPUTE_METRICS = False", "RECOMPUTE_METRICS = True")
                c.source = c.source.replace("REGENERATE_FIGURES = False", "REGENERATE_FIGURES = True")
        nb.cells.append(nbformat.v4.new_code_cell('''
assert set(models) == set(ALL_NAMES)
assert smoke_report["resume_ae_epoch2"]
assert all(v["parameter_update"] for v in smoke_report["models"].values())
assert all(v["reconstruction"]["n_test"] == 7000 for n, v in evaluation.items() if n in RECONSTRUCTORS)
assert mlp_results["n_test"] == 7000
assert len(list(out.glob("figure_*.png"))) == 5
identity_x = torch.rand(4, 1, 28, 28, device=device)
identity_mse, identity_psnr, identity_ssim = reconstruction_scores(identity_x, identity_x)
assert (identity_mse == 0).all() and torch.isfinite(identity_psnr).all()
assert torch.allclose(identity_ssim, torch.ones_like(identity_ssim), atol=1e-5)
print("Integration assertions passed: models, smoke/resume, MLP7000, reconstruction7000, five figures, zero-MSE policy")
'''))
        summary["checkpoint_integration"] = run(nb, ROOT, evidence, "integration.executed.ipynb")
        summary["checkpoint_integration"].update({"train_batches_each": 2, "validation_batches_each": 2,
            "smoke_models": 8, "ddpm_steps": 400, "ae_resume": True, "fid_extractor_update_only": True,
            "full_training": False, "full_fid_recomputed": False, "mlp_test_n": 7000,
            "reconstruction_n_each": 7000, "figures_regenerated": 5})
    assert original == fingerprint_originals(), "Original source/config/results/experiment artifacts changed"
    summary["original_files_unchanged"] = True
    summary["project_imports_blocked"] = True
    (evidence / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
