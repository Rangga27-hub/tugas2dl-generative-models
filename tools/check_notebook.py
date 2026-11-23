"""Execute the concise notebook standalone, then optional local MLP/cache checks."""
import hashlib
import json
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    files = list((ROOT / "src").glob("*.py")) + list((ROOT / "configs").glob("*.json"))
    files += [p for folder in ("results", "artifacts/fashion") for p in (ROOT/folder).rglob("*") if p.is_file()]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def execute(nb, cwd, target):
    nb.cells.insert(0, nbformat.v4.new_code_cell('''
import sys
import importlib.abc
class NoProjectImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "src" or fullname.startswith("src."):
            raise AssertionError("Project imports are not allowed")
sys.meta_path.insert(0, NoProjectImports())
'''))
    NotebookClient(nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(cwd)}}).execute()
    nbformat.write(nb, target)
    errors = sum(o.output_type == "error" for c in nb.cells if c.cell_type == "code" for o in c.outputs)
    return {"code_cells": sum(c.cell_type == "code" for c in nb.cells), "errors": errors}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--integration", action="store_true")
    args = parser.parse_args()
    evidence = ROOT / "artifacts/simple_notebook_verification"
    evidence.mkdir(parents=True, exist_ok=True)
    before = snapshot()
    report = {}
    with tempfile.TemporaryDirectory(prefix="simple_notebook_") as folder:
        nb = nbformat.read(ROOT / "main.ipynb", as_version=4)
        nbformat.write(nb, Path(folder)/"main.ipynb")
        report["standalone_default"] = execute(nb, folder, evidence/"standalone.ipynb")
    if args.integration:
        nb = nbformat.read(ROOT / "main.ipynb", as_version=4)
        for cell in nb.cells:
            if "embedded-config" in cell.metadata.get("tags", []):
                cell.source = cell.source.replace("RUN_MLP = False", "RUN_MLP = True")
        nb.cells.append(nbformat.v4.new_code_cell('''
assert mlp_demo["n_test"] == 7000
expected = json.loads(Path("results/mlp_experiments/results.json").read_text(encoding="utf-8"))
assert abs(mlp_demo["accuracy"] - expected["test_accuracy"]) < 1e-12
for key in ("transpose_class", "transpose_latent", "frozen_decoder", "vanilla_ae"):
    assert mlp_demo[key].shape == (10, 1, 28, 28)
assert mlp_demo["gaussian_samples"].shape == (16, 1, 28, 28)
assert torch.isfinite(mlp_demo["gaussian_samples"]).all()
assert len(metrics) == 6
assert all((Path("results/figures")/f"{n}.png").exists() for n in figure_names)
print("MLP7000, transpose/decoder/sampling shapes, six saved metric rows and five saved figures passed")
'''))
        report["mlp_and_saved_outputs"] = execute(nb, ROOT, evidence/"mlp_and_outputs.ipynb")
    assert before == snapshot(), "Original experiment files changed"
    report["original_files_unchanged"] = True
    report["full_training"] = False
    report["fresh_fid"] = False
    (evidence/"summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
