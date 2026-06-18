from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache
from typing import Any, Dict


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


def acceleration_status() -> Dict[str, Any]:
    modules = {
        "polars": "polars",
        "cupy": "cupy",
        "numba": "numba",
        "cudf": "cudf",
        "cuml": "cuml",
    }
    available = {key: accelerator_available(module) for key, module in modules.items()}
    return {
        **available,
        "dataframe_engine": "polars" if available["polars"] else "pandas",
        "score_array_engine": "cupy" if available["cupy"] else "numpy",
        "rapids_dataframe": "cudf" if available["cudf"] else "",
        "rapids_ml": "cuml" if available["cuml"] else "",
    }
