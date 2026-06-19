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
            "reason": "not_installed",
            "fallback_backend": "numpy",
            "remediation": "Instala un paquete CuPy compatible con tu CUDA local para activar scoring GPU.",
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
                "reason": "no_cuda_devices",
                "fallback_backend": "numpy",
                "remediation": "La app continua en CPU/NumPy hasta detectar un dispositivo CUDA usable.",
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
            "reason": "",
            "fallback_backend": "",
            "remediation": "",
        }
    except Exception as exc:
        failure = classify_cupy_runtime_failure(exc)
        return {
            "installed": True,
            "cuda_available": False,
            "device_count": 0,
            "device_names": [],
            **failure,
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
        "cupy_cuda_reason": str(cupy_status.get("reason") or ""),
        "cupy_cuda_remediation": str(cupy_status.get("remediation") or ""),
        "gpu_ready": bool(cupy_status.get("cuda_available")),
        "dataframe_engine": "polars" if available["polars"] else "pandas",
        "score_array_engine": "cupy" if cupy_status.get("cuda_available") else "numpy",
        "score_array_fallback": "numpy" if not cupy_status.get("cuda_available") else "",
        "rapids_dataframe": "cudf" if available["cudf"] else "",
        "rapids_ml": "cuml" if available["cuml"] else "",
    }


def classify_cupy_runtime_failure(exc: Exception) -> Dict[str, str]:
    raw_detail = f"{exc.__class__.__name__}: {exc}"
    lower = raw_detail.lower()
    if "nvrtc" in lower or "nvrtc64" in lower:
        reason = "missing_nvrtc_runtime"
        warning = "CuPy CUDA no usable: falta NVRTC/DLL compatible; fallback CPU/NumPy activo."
        remediation = (
            "Usa requirements-gpu-cuda13.txt en este proyecto: mantiene NumPy 1.26 y fija "
            "cupy-cuda13x==13.6.0 con CUDA Toolkit/runtime 13.0.2 y NVRTC. "
            "Evita cupy-cuda13x sin version."
        )
    elif "could not find module" in lower or "dll" in lower or "shared object" in lower:
        reason = "missing_cuda_runtime_library"
        warning = "CuPy CUDA no usable: falta una libreria runtime de CUDA; fallback CPU/NumPy activo."
        remediation = "Instala el runtime CUDA compatible con el paquete CuPy del entorno o usa el paquete CuPy con extras de runtime."
    elif "insufficient driver" in lower or "driver version" in lower:
        reason = "cuda_driver_incompatible"
        warning = "CuPy CUDA no usable: driver CUDA incompatible; fallback CPU/NumPy activo."
        remediation = "Actualiza el driver NVIDIA o instala un paquete CuPy compatible con el driver disponible."
    else:
        reason = "cupy_runtime_error"
        warning = f"CuPy CUDA no usable; fallback CPU/NumPy activo ({exc.__class__.__name__})."
        remediation = "Ejecuta cupy.show_config() en la maquina GPU para revisar runtime, toolkit y librerias CUDA."
    return {
        "warning": warning,
        "reason": reason,
        "fallback_backend": "numpy",
        "remediation": remediation,
        "error_detail": raw_detail,
    }
