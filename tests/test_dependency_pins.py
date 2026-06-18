from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _requirement_lines(filename: str) -> list[str]:
    return [
        line.strip()
        for line in (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_cli_model_specs_imports_without_tensorflow():
    restored_modules = {
        name: module
        for name, module in sys.modules.items()
        if (
            name == "src.cli.model_specs"
            or name == "src.models.classifiers"
            or name.startswith("src.models.classifiers.neuralnets")
            or name == "tensorflow"
            or name.startswith("tensorflow.")
        )
    }
    for name in restored_modules:
        sys.modules.pop(name, None)
    sys.modules["tensorflow"] = None

    try:
        model_specs = importlib.import_module("src.cli.model_specs")

        assert set(model_specs.MODEL_SPECS) == {"ngboost", "catboost", "lightgbm", "xgboost"}
        assert "src.models.classifiers.neuralnets.nn" not in sys.modules
    finally:
        for name in list(sys.modules):
            if (
                name == "src.cli.model_specs"
                or name == "src.models.classifiers"
                or name.startswith("src.models.classifiers.neuralnets")
                or name == "tensorflow"
                or name.startswith("tensorflow.")
            ):
                sys.modules.pop(name, None)
        sys.modules.update(restored_modules)


def test_base_requirements_pin_numpy_126():
    assert "numpy==1.26.4" in _requirement_lines("requirements.txt")


def test_cuda13_gpu_requirements_pin_native_cuda13_cupy_and_numpy_126():
    lines = _requirement_lines("requirements-gpu-cuda13.txt")

    assert "numpy==1.26.4" in lines
    assert "cupy-cuda13x==13.6.0" in lines
    assert "cuda-toolkit[cudart,nvrtc]==13.3.0" in lines
    assert not any(line.lower() == "cupy-cuda13x" for line in lines)
