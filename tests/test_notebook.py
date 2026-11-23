"""Contracts for concise main and the preserved, standalone full pipeline."""
import ast
import json
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat", reason="Install requirements.txt for notebook tests")
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("main.ipynb", "notebooks/full_pipeline.ipynb")


def test_notebooks_valid_and_no_project_runtime_imports():
    for path in NOTEBOOKS:
        nb = nbformat.read(ROOT / path, as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            for node in ast.walk(ast.parse(cell.source)):
                if isinstance(node, ast.ImportFrom):
                    assert node.level == 0 and not (node.module or "").startswith("src")
                if isinstance(node, ast.Import):
                    assert not any(a.name == "src" or a.name.startswith("src.") for a in node.names)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id not in {"exec", "eval", "__import__"}
            assert cell.execution_count is None and not cell.outputs


def test_embedded_config_and_default_safety():
    expected = json.loads((ROOT / "configs/default.json").read_text())
    short = nbformat.read(ROOT / NOTEBOOKS[0], as_version=4)
    config = next(c for c in short.cells if "embedded-config" in c.metadata.get("tags", []))
    values = {"json": json}
    exec(compile(config.source, "<inline-config>", "exec"), values)
    assert values["cfg"] == expected and values["RUN_MLP"] is False
    full = nbformat.read(ROOT / NOTEBOOKS[1], as_version=4)
    config = next(c for c in full.cells if "embedded-config" in c.metadata.get("tags", []))
    values = {}
    exec(compile(config.source, "<full-config>", "exec"), values)
    assert values["DEFAULT_CONFIG"] == expected
    controls = next(c for c in full.cells if "controls" in c.metadata.get("tags", []))
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(controls.source).body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and isinstance(n.value, ast.Constant)}
    assert constants["MODE"] == "load"
    for key in ("CONFIRM_TRAINING", "RUN_SMOKE", "RUN_PILOT", "RECOMPUTE_MLP", "RECOMPUTE_METRICS", "RECOMPUTE_FID", "REGENERATE_FIGURES"):
        assert constants[key] is False


def test_inline_models_and_functions_match_own_project():
    for path in NOTEBOOKS:
        inline = {}
        for cell in nbformat.read(ROOT / path, as_version=4).cells:
            if cell.cell_type == "code":
                for node in ast.parse(cell.source).body:
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                        inline[node.name] = node
        for module in ("models", "train", "data", "common", "evaluate"):
            for node in ast.parse((ROOT / f"src/{module}.py").read_text()).body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in inline:
                    if node.name == "fid_metrics":
                        continue  # Full pipeline only parameterizes the output destination.
                    assert ast.dump(node) == ast.dump(inline[node.name]), (path, node.name)
        assert all(name in inline for name in ("Autoencoder", "VAE", "VanillaGAN", "DCGAN", "DDPM", "Classifier", "Decoder"))


def test_main_is_concise_and_labels_saved_outputs():
    nb = nbformat.read(ROOT / "main.ipynb", as_version=4)
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    positions = [text.index(f"## {i}.") for i in range(1, 8)]
    assert positions == sorted(positions)
    assert "dibantu AI" in text and "bukan dihitung ulang" in text
    assert "notebooks/full_pipeline.ipynb" in text
    assert sum(len(c.source.splitlines()) for c in nb.cells if c.cell_type == "code") <= 450
    gallery = next(c for c in nb.cells if "figure-gallery" in c.metadata.get("tags", []))
    tree = ast.parse(gallery.source)
    names = ast.literal_eval(tree.body[0].value)
    assert len(names) == 5 and names[0] == "figure_1_reconstruction" and names[-1] == "figure_5_curves_trajectory"
