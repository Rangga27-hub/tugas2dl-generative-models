"""Static/in-process contracts for the self-contained notebook; no full training."""
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
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0
                assert not (node.module or "").startswith("src")
            if isinstance(node, ast.Import):
                assert not any(a.name == "src" or a.name.startswith("src.") for a in node.names)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"exec", "eval", "__import__"}
    assert all(c.execution_count is None and not c.outputs for c in nb.cells if c.cell_type == "code")


def test_embedded_config_and_default_safety():
    nb = notebook()
    config_cell = next(c for c in nb.cells if "embedded-config" in c.metadata.get("tags", []))
    values = {}
    exec(compile(config_cell.source, "<embedded-config>", "exec"), values)
    assert values["DEFAULT_CONFIG"] == json.loads((ROOT / "configs/default.json").read_text())
    controls = next(c for c in nb.cells if "controls" in c.metadata.get("tags", []))
    literals = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(controls.source).body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and isinstance(n.value, ast.Constant)}
    assert literals["MODE"] == "load"
    for key in ("CONFIRM_TRAINING", "RUN_SMOKE", "RUN_PILOT", "RECOMPUTE_MLP", "RECOMPUTE_METRICS", "RECOMPUTE_FID", "REGENERATE_FIGURES"):
        assert literals[key] is False


def test_inline_models_and_training_match_project():
    nb = notebook()
    inline = {}
    for cell in nb.cells:
        if cell.cell_type == "code":
            for node in ast.parse(cell.source).body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    inline[node.name] = node
    for module in ("models", "train", "data", "common", "evaluate"):
        original = ast.parse((ROOT / f"src/{module}.py").read_text())
        for node in original.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in inline:
                if node.name == "fid_metrics":  # Only destination path is parameterized.
                    continue
                assert ast.dump(node, include_attributes=False) == ast.dump(inline[node.name], include_attributes=False), node.name


def test_notebook_sections_and_five_figures():
    nb = notebook()
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    positions = [text.index(heading) for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7.")]
    assert positions == sorted(positions)
    for i in range(1, 6):
        assert sum(f"figure-{i}" in c.metadata.get("tags", []) for c in nb.cells) == 1
    assert "dibantu AI" in text
