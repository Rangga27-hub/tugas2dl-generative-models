"""Contracts for the standalone code-only main notebook."""
import ast
import json
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat", reason="Install requirements.txt for notebook tests")
ROOT = Path(__file__).resolve().parents[1]


def notebook():
    return nbformat.read(ROOT / "main.ipynb", as_version=4)


def test_notebook_valid_and_no_project_runtime_imports():
    nb = notebook()
    nbformat.validate(nb)
    assert len(nb.cells) == 12 and all(c.cell_type == "code" for c in nb.cells)
    for cell in nb.cells:
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
    config = next(c for c in notebook().cells if "embedded-config" in c.metadata.get("tags", []))
    values = {"json": json}
    exec(compile(config.source, "<inline-config>", "exec"), values)
    assert values["cfg"] == expected and values["RUN_MLP"] is False


def test_inline_models_and_functions_match_own_project():
    inline = {}
    for cell in notebook().cells:
        for node in ast.parse(cell.source).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                inline[node.name] = node
    for module in ("models", "data", "common"):
        for node in ast.parse((ROOT / f"src/{module}.py").read_text()).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in inline:
                assert ast.dump(node) == ast.dump(inline[node.name]), node.name
    assert all(name in inline for name in ("Autoencoder", "VAE", "VanillaGAN", "DCGAN", "DDPM", "Classifier", "Decoder"))


def test_main_is_concise_and_uses_saved_outputs():
    nb = notebook()
    assert sum(len(c.source.splitlines()) for c in nb.cells) <= 450
    gallery = next(c for c in nb.cells if "figure-gallery" in c.metadata.get("tags", []))
    names = ast.literal_eval(ast.parse(gallery.source).body[0].value)
    assert len(names) == 5 and names[0] == "figure_1_reconstruction" and names[-1] == "figure_5_curves_trajectory"
    assert "pd.read_csv" in next(c.source for c in nb.cells if "saved-metrics" in c.metadata.get("tags", []))
    assert not (ROOT / "notebooks/full_pipeline.ipynb").exists()
