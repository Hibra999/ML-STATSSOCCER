from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


ALTERNATIVES_BENCHMARK_PIPELINE_MODE = "alternatives_benchmark"
ALTERNATIVES_BENCHMARK_LABEL = "Benchmark alternativas"
ALTERNATIVES_EVIDENCE_POLICY = "local_backtest_vs_poisson"

ALTERNATIVE_SCORE_MODEL_KEYS = [
    "dixon_coles_mle",
    "bivariate_poisson_mle",
]


ALTERNATIVE_SCORE_MODELS: List[Dict[str, Any]] = [
    {
        "rank": 1,
        "key": "dixon_coles_mle",
        "model_name": "Dixon-Coles MLE",
        "family": "low_score_correlation",
        "description": "Corrige resultados 0-0, 1-0, 0-1 y 1-1 con rho estimado por maxima verosimilitud.",
    },
    {
        "rank": 2,
        "key": "bivariate_poisson_mle",
        "model_name": "Poisson bivariado MLE",
        "family": "correlated_counts",
        "description": "Agrega un componente comun para correlacion positiva entre goles locales y visitantes.",
    },
]


def sota_alternatives_catalog() -> List[Dict[str, Any]]:
    return deepcopy(ALTERNATIVE_SCORE_MODELS)


def sota_baseline_context() -> List[Dict[str, Any]]:
    return [
        {
            "model_name": "Poisson independiente",
            "key": "independent_poisson",
            "note": "Baseline separado: no participa como alternativa ni se promedia con otros modelos.",
        }
    ]


def alternatives_table_rows(alternatives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in alternatives:
        rows.append({
            "Rank": item.get("rank", ""),
            "Modelo": item.get("model_name", ""),
            "Key": item.get("key", ""),
            "Familia": item.get("family", ""),
            "Descripcion": item.get("description", ""),
        })
    return rows
