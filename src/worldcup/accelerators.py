from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import numpy as np

try:
    from numba import cuda, njit, prange
except Exception:
    cuda = None
    njit = None
    prange = range


if njit is not None:
    @njit(parallel=True, fastmath=True, cache=True)
    def _sanitize_float32_cpu(values):
        flat = values.ravel()
        for index in prange(flat.size):
            value = flat[index]
            if not math.isfinite(value):
                flat[index] = 0.0
else:
    def _sanitize_float32_cpu(values):
        values[~np.isfinite(values)] = 0.0


if cuda is not None:
    @cuda.jit
    def _sanitize_float32_cuda(values):
        index = cuda.grid(1)
        if index < values.size:
            value = values[index]
            if not math.isfinite(value):
                values[index] = 0.0
else:
    _sanitize_float32_cuda = None


def numba_cuda_available() -> bool:
    if cuda is None:
        return False
    try:
        return bool(cuda.is_available())
    except Exception:
        return False


def numba_jit_available() -> bool:
    return bool(njit is not None)


def training_acceleration_backend(prefer_cuda: bool = False, cuda_available: bool | None = None) -> Dict[str, Any]:
    numba_cuda = numba_cuda_available()
    model_cuda = numba_cuda if cuda_available is None else bool(cuda_available)
    backend = "cuda" if prefer_cuda and model_cuda else "cpu"
    numba_backend = "cuda" if prefer_cuda and numba_cuda and _sanitize_float32_cuda is not None else "cpu"
    return {
        "backend": backend,
        "model_backend": backend,
        "numba_backend": numba_backend,
        "numba": numba_jit_available(),
        "numba_cuda_available": numba_cuda,
        "xgboost_cuda_ready": backend == "cuda",
        "cpu_jit_available": numba_jit_available(),
    }


def prepare_numeric_matrix(values: Any, use_cuda: bool = False) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        array = values.to_numpy(dtype=np.float32, copy=True)
    else:
        array = np.asarray(values, dtype=np.float32)
    array = np.ascontiguousarray(array, dtype=np.float32)
    np.nan_to_num(array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if use_cuda and numba_cuda_available() and _sanitize_float32_cuda is not None:
        flat = array.reshape(array.size)
        threads = 256
        blocks = (flat.size + threads - 1) // threads
        _sanitize_float32_cuda[blocks, threads](flat)
        cuda.synchronize()
    else:
        _sanitize_float32_cpu(array)
    return array


def prepare_xgboost_training_arrays(x: Any, y: Any, use_cuda: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    return prepare_numeric_matrix(x, use_cuda=use_cuda), np.ascontiguousarray(np.asarray(y, dtype=np.int32))


def prepare_xgboost_prediction_array(x: Any, use_cuda: bool = False) -> np.ndarray:
    return prepare_numeric_matrix(x, use_cuda=use_cuda)
