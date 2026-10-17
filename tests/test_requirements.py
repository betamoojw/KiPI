"""The dependency lists agree: requirements.txt (what KiCad installs into the plugin's venv)
is pyproject.toml's dependencies (standalone) plus its "kicad" extra (kicad-python)."""

from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")      # Python 3.11+; the lists are the same anywhere

ROOT = Path(__file__).resolve().parents[1]


def _requirements(path: Path):
    lines = (line.split("#", 1)[0].strip() for line in path.read_text().splitlines())
    return {line.replace(" ", "") for line in lines if line}


def _pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_the_plugin_gets_standalone_plus_kicad():
    project = _pyproject()
    standalone = {d.replace(" ", "") for d in project["dependencies"]}
    kicad = {d.replace(" ", "") for d in project["optional-dependencies"]["kicad"]}
    assert _requirements(ROOT / "requirements.txt") == standalone | kicad


def test_standalone_needs_no_kicad():
    project = _pyproject()
    assert not any(d.startswith("kicad-python") for d in project["dependencies"])
    assert any(d.startswith("kicad-python") for d in project["optional-dependencies"]["kicad"])


def test_dev_tools_match():
    dev = {d.replace(" ", "") for d in _pyproject()["optional-dependencies"]["dev"]}
    assert _requirements(ROOT / "requirements-dev.txt") == dev
