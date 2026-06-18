from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache
from typing import Any, Dict, List


@lru_cache(maxsize=None)
def accelerator_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(str(module_name)) is not None
    except Exception:
        return False


@lru_cache(maxsize=None)
def import_optional_accelerator(module_name: str) -> Any | None:
    if not accelerator_available(module_name):
        return None
    try:
        return importlib.import_module(str(module_name))
    except Exception:
        return None


@lru_cache(maxsize=None)
def cupy_runtime_status() -> Dict[str, Any]:
    cp = import_optional_accelerator("cupy")
    if cp is None:
        return {
            "installed": False,
            "cuda_available": False,
            "device_count": 0,
            "device_names": [],
            "warning": "CuPy no instalado",
        }
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count <= 0:
            return {
                "installed": True,
                "cuda_available": False,
                "device_count": 0,
                "device_names": [],
                "warning": "CuPy sin dispositivos CUDA",
            }
        probe = cp.arange(8, dtype=cp.float32)
        float(cp.sum(probe).get())
        names: List[str] = []
        for index in range(device_count):
            try:
                props = cp.cuda.runtime.getDeviceProperties(index)
                raw_name = props.get("name", "") if isinstance(props, dict) else ""
                if isinstance(raw_name, bytes):
                    raw_name = raw_name.decode("utf-8", errors="ignore")
                names.append(str(raw_name or f"CUDA device {index}").strip())
            except Exception:
                names.append(f"CUDA device {index}")
        return {
            "installed": True,
            "cuda_available": True,
            "device_count": device_count,
            "device_names": names,
            "warning": "",
        }
    except Exception as exc:
        return {
            "installed": True,
            "cuda_available": False,
            "device_count": 0,
            "device_names": [],
            "warning": f"CuPy CUDA no usable ({exc.__class__.__name__}: {exc})",
        }


def acceleration_status() -> Dict[str, Any]:
    modules = {
        "polars": "polars",
        "cupy": "cupy",
        "numba": "numba",
        "cudf": "cudf",
        "cuml": "cuml",
    }
    available = {key: accelerator_available(module) for key, module in modules.items()}
    cupy_status = cupy_runtime_status()
    return {
        **available,
        "cupy_cuda": bool(cupy_status.get("cuda_available")),
        "cupy_cuda_device_count": int(cupy_status.get("device_count", 0) or 0),
        "cupy_cuda_device_names": list(cupy_status.get("device_names", [])),
        "cupy_cuda_warning": str(cupy_status.get("warning") or ""),
        "gpu_ready": bool(cupy_status.get("cuda_available")),
        "dataframe_engine": "polars" if available["polars"] else "pandas",
        "score_array_engine": "cupy" if cupy_status.get("cuda_available") else "numpy",
        "rapids_dataframe": "cudf" if available["cudf"] else "",
        "rapids_ml": "cuml" if available["cuml"] else "",
    }
