from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.worldcup.model import (
    TOTAL_GOAL_LINES,
    dixon_coles_score_grid,
    poisson_score_grid,
    total_line_suffix,
)


DEFAULT_SCORE_MODEL = "independent_poisson"
STATSMODELS_POISSON_GLM_MODEL = "statsmodels_poisson_glm"
NEGATIVE_BINOMIAL_GLM_MODEL = "negative_binomial_glm"
STAT_MODEL_ROOT = Path("storage") / "worldcup" / "stat_models"
LOCAL_XG_FILE = Path("storage") / "worldcup" / "xg" / "manual_xg.csv"
_CUPY_BACKEND_STATUS: Dict[str, Any] | None = None

SCORE_MODEL_OPTIONS = [
    {
        "key": "independent_poisson",
        "label": "Poisson independiente",
        "description": "Modelo actual: dos Poisson independientes con lambdas Elo/ataque/defensa.",
        "heavy": False,
    },
    {
        "key": "dixon_coles_mle",
        "label": "Dixon-Coles MLE",
        "description": "Corrige 0-0, 1-0, 0-1 y 1-1 con rho estimado por maxima verosimilitud.",
        "heavy": False,
    },
    {
        "key": STATSMODELS_POISSON_GLM_MODEL,
        "label": "Poisson GLM statsmodels",
        "description": "Calibra lambdas con GLM Poisson regularizado, offset log(lambda base) y pesos temporales.",
        "heavy": False,
    },
    {
        "key": NEGATIVE_BINOMIAL_GLM_MODEL,
        "label": "Negative Binomial GLM",
        "description": "Variante de conteo sobredispersa: lambdas GLM y margenes Negative Binomial NB2.",
        "heavy": False,
    },
    {
        "key": "bivariate_poisson_mle",
        "label": "Poisson bivariado MLE",
        "description": "Introduce un componente comun de goles para correlacion positiva entre equipos.",
        "heavy": False,
    },
    {
        "key": "bayesian_hierarchical_poisson",
        "label": "Bayes jerarquico Poisson",
        "description": "Ataque/defensa/localia con priors jerarquicos en PyMC.",
        "heavy": True,
    },
    {
        "key": "bayesian_dynamic_poisson",
        "label": "Bayes dinamico Poisson",
        "description": "Variante dinamica por periodos; requiere backend bayesiano disponible.",
        "heavy": True,
    },
    {
        "key": "xg_poisson_local",
        "label": "xG local + Poisson",
        "description": "Usa xG de storage/worldcup/xg/manual_xg.csv si esta disponible.",
        "heavy": False,
    },
]

SCORE_MODEL_KEYS = {option["key"] for option in SCORE_MODEL_OPTIONS}


@dataclass(frozen=True)
class ScoreModelState:
    key: str
    label: str
    available: bool
    params: Dict[str, Any]
    warnings: Tuple[str, ...] = ()
    fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "available": self.available,
            "params": self.params,
            "warnings": list(self.warnings),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ScoreModelState":
        key = normalize_score_model_key(payload.get("key"))
        return cls(
            key=key,
            label=score_model_label(key),
            available=bool(payload.get("available", False)),
            params=dict(payload.get("params") or {}),
            warnings=tuple(str(item) for item in payload.get("warnings", []) if str(item)),
            fingerprint=str(payload.get("fingerprint") or ""),
        )


class AdvancedScoreWorldCupModel:
    def __init__(self, base_model: Any, state: ScoreModelState):
        self.base_model = base_model
        self.state = state
        self.max_goals = int(getattr(base_model, "max_goals", 10))

    @property
    def score_model_key(self) -> str:
        return self.state.key

    def score_model_metadata(self) -> Dict[str, Any]:
        return self.state.to_dict()

    def profile(self, team: str):
        return self.base_model.profile(team)

    def adjusted(self, rating_adjustments: Dict[str, float]):
        adjusted = self.base_model.adjusted(rating_adjustments)
        return AdvancedScoreWorldCupModel(adjusted, self.state)

    def expected_goals(self, team1: str, team2: str) -> Tuple[float, float]:
        return self.expected_goals_for_match(team1, team2, match=None)

    def expected_goals_for_match(self, team1: str, team2: str, match: Dict[str, Any] | None = None) -> Tuple[float, float]:
        if self.state.key == "xg_poisson_local":
            xg_lambdas = _xg_lambdas_for_match(self.state, team1, team2, match)
            if xg_lambdas is not None:
                return xg_lambdas
        if self.state.key in {"bayesian_hierarchical_poisson", "bayesian_dynamic_poisson"}:
            bayesian_lambdas = _bayesian_lambdas_for_match(self.state, team1, team2, match=match)
            if bayesian_lambdas is not None:
                return bayesian_lambdas
        method = getattr(self.base_model, "expected_goals_for_match", None)
        if callable(method):
            base_lambdas = method(team1, team2, match=match)
        else:
            base_lambdas = self.base_model.expected_goals(team1, team2)
        calibrated = _statsmodels_lambdas_for_match(self.state, team1, team2, base_lambdas)
        return calibrated if calibrated is not None else base_lambdas

    def score_grid(self, team1: str, team2: str, match: Dict[str, Any] | None = None, max_goals: int | None = None) -> np.ndarray:
        lambda1, lambda2 = self.expected_goals_for_match(team1, team2, match=match)
        predictive_overdispersion: float | None = None
        if self.state.key in {"bayesian_hierarchical_poisson", "bayesian_dynamic_poisson"}:
            predictive_overdispersion = _bayesian_predictive_overdispersion(self.state, team1, team2, match=match)
        return score_grid_from_lambdas(
            self.state,
            lambda1=lambda1,
            lambda2=lambda2,
            predictive_overdispersion=predictive_overdispersion,
            max_goals=int(max_goals if max_goals is not None else self.max_goals),
        )

    def score_grid_from_lambdas(self, lambda1: float, lambda2: float, max_goals: int | None = None) -> np.ndarray:
        return score_grid_from_lambdas(
            self.state,
            lambda1=lambda1,
            lambda2=lambda2,
            max_goals=int(max_goals if max_goals is not None else self.max_goals),
        )

    def score_grids_from_lambdas(
            self,
            lambda1_values: Sequence[float] | np.ndarray,
            lambda2_values: Sequence[float] | np.ndarray,
            max_goals: int | None = None,
            backend: Any = "auto",
    ) -> Tuple[np.ndarray, str, List[str]]:
        return score_grids_from_lambdas_with_backend(
            self.state,
            lambda1_values=lambda1_values,
            lambda2_values=lambda2_values,
            max_goals=int(max_goals if max_goals is not None else self.max_goals),
            backend=backend,
        )

    def match_probabilities(self, team1: str, team2: str, max_goals: int | None = None) -> Dict[str, float]:
        return self.match_probabilities_for_match(team1, team2, match=None, max_goals=max_goals)

    def match_probabilities_for_match(
            self,
            team1: str,
            team2: str,
            match: Dict[str, Any] | None = None,
            max_goals: int | None = None,
    ) -> Dict[str, float]:
        limit_goals = int(max_goals if max_goals is not None else self.max_goals)
        lambda1, lambda2 = self.expected_goals_for_match(team1, team2, match=match)
        predictive_overdispersion: float | None = None
        if self.state.key in {"bayesian_hierarchical_poisson", "bayesian_dynamic_poisson"}:
            predictive_overdispersion = _bayesian_predictive_overdispersion(self.state, team1, team2, match=match)
        grid = score_grid_from_lambdas(
            self.state,
            lambda1=lambda1,
            lambda2=lambda2,
            predictive_overdispersion=predictive_overdispersion,
            max_goals=limit_goals,
        )
        output = probabilities_from_score_grid(grid, lambda1=lambda1, lambda2=lambda2)
        output["score_model"] = self.state.key
        output["score_model_label"] = self.state.label
        output["score_model_available"] = bool(self.state.available)
        if self.state.warnings:
            output["score_model_warning"] = "; ".join(self.state.warnings)
        return output

    def sample_score(self, team1: str, team2: str, rng: np.random.Generator) -> Tuple[int, int]:
        grid = self.score_grid(team1, team2)
        return sample_score_from_grid(grid, rng)

    def sample_scores(
            self,
            team1: str,
            team2: str,
            rng: np.random.Generator,
            size: int,
            match: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        grid = self.score_grid(team1, team2, match=match)
        return sample_scores_from_grid(grid, rng, size=size)

    def sample_knockout_winner(self, team1: str, team2: str, rng: np.random.Generator):
        goals1, goals2 = self.sample_score(team1, team2, rng)
        if goals1 > goals2:
            return team1, team2, goals1, goals2
        if goals2 > goals1:
            return team2, team1, goals1, goals2
        probabilities = self.match_probabilities(team1, team2)
        win_share = probabilities["home"] / max(probabilities["home"] + probabilities["away"], 1e-9)
        if rng.random() <= win_share:
            return team1, team2, goals1, goals2
        return team2, team1, goals1, goals2


def score_model_options() -> List[Dict[str, Any]]:
    return [dict(option) for option in SCORE_MODEL_OPTIONS]


def normalize_score_model_key(value: Any) -> str:
    key = str(value or DEFAULT_SCORE_MODEL).strip().lower().replace("-", "_")
    return key if key in SCORE_MODEL_KEYS else DEFAULT_SCORE_MODEL


def score_model_label(key: Any) -> str:
    normalized = normalize_score_model_key(key)
    for option in SCORE_MODEL_OPTIONS:
        if option["key"] == normalized:
            return str(option["label"])
    return "Poisson independiente"


def build_score_model(
        base_model: Any,
        history_df: pd.DataFrame | None,
        teams: Iterable[str],
        config: Dict[str, Any] | None = None,
) -> Any:
    config = config or {}
    key = normalize_score_model_key(config.get("score_model"))
    if key == DEFAULT_SCORE_MODEL:
        return base_model
    state = fit_score_model_state(key, base_model=base_model, history_df=history_df, teams=teams, config=config)
    return AdvancedScoreWorldCupModel(base_model, state)


def fit_score_model_state(
        key: str,
        base_model: Any,
        history_df: pd.DataFrame | None,
        teams: Iterable[str],
        config: Dict[str, Any] | None = None,
) -> ScoreModelState:
    config = config or {}
    key = normalize_score_model_key(key)
    label = score_model_label(key)
    fingerprint = _fit_fingerprint(key, history_df, teams, config)
    use_cache = bool(config.get("stat_model_cache", True))
    force_refit = bool(config.get("stat_model_refit", False))
    cache_path = _state_cache_path(key, fingerprint)
    if use_cache and not force_refit:
        cached = _read_state(cache_path)
        if cached is not None:
            return cached

    rows = _history_model_rows(history_df, base_model, recency_weight=config.get("score_mle_recency_weight", config.get("recency_weight", 0.0)))
    if key == "xg_poisson_local":
        state = _fit_xg_local_state(label=label, fingerprint=fingerprint)
    elif key in {"bayesian_hierarchical_poisson", "bayesian_dynamic_poisson"}:
        state = _fit_bayesian_state(key, label, fingerprint, rows, teams, config)
    elif not rows:
        state = ScoreModelState(
            key=key,
            label=label,
            available=False,
            params={},
            warnings=("Sin historico suficiente; se usa Poisson independiente.",),
            fingerprint=fingerprint,
        )
    elif key == STATSMODELS_POISSON_GLM_MODEL:
        lambda_model, fit_warnings = _fit_statsmodels_lambda_model(rows, teams=teams, config=config)
        state = ScoreModelState(
            key,
            label,
            bool(lambda_model),
            {"lambda_model": lambda_model} if lambda_model else {},
            warnings=tuple(fit_warnings),
            fingerprint=fingerprint,
        )
    elif key == NEGATIVE_BINOMIAL_GLM_MODEL:
        lambda_model, fit_warnings = _fit_statsmodels_lambda_model(rows, teams=teams, config=config)
        params = {"lambda_model": lambda_model, "alpha": _negative_binomial_alpha(lambda_model)} if lambda_model else {}
        state = ScoreModelState(
            key,
            label,
            bool(lambda_model),
            params,
            warnings=tuple(fit_warnings),
            fingerprint=fingerprint,
        )
    elif key == "dixon_coles_mle":
        lambda_model, lambda_warnings = _fit_statsmodels_lambda_model(rows, teams=teams, config=config)
        mle_rows = _rows_with_lambda_model(rows, lambda_model) if lambda_model else rows
        rho, fit_backend, fit_warnings = _estimate_dixon_coles_rho(
            mle_rows,
            backend=config.get("score_backend") or config.get("sota_device", "auto"),
        )
        params = {"rho": rho, "fit_backend": fit_backend}
        if lambda_model:
            params["lambda_model"] = lambda_model
        state = ScoreModelState(
            key,
            label,
            True,
            params,
            warnings=tuple(_merge_warnings(lambda_warnings, fit_warnings)),
            fingerprint=fingerprint,
        )
    elif key == "bivariate_poisson_mle":
        lambda_model, lambda_warnings = _fit_statsmodels_lambda_model(rows, teams=teams, config=config)
        mle_rows = _rows_with_lambda_model(rows, lambda_model) if lambda_model else rows
        corr_share, fit_backend, fit_warnings = _estimate_bivariate_corr_share(
            mle_rows,
            backend=config.get("score_backend") or config.get("sota_device", "auto"),
        )
        params = {"corr_share": corr_share, "fit_backend": fit_backend}
        if lambda_model:
            params["lambda_model"] = lambda_model
        state = ScoreModelState(
            key,
            label,
            True,
            params,
            warnings=tuple(_merge_warnings(lambda_warnings, fit_warnings)),
            fingerprint=fingerprint,
        )
    else:
        state = ScoreModelState(
            key=DEFAULT_SCORE_MODEL,
            label=score_model_label(DEFAULT_SCORE_MODEL),
            available=True,
            params={},
            warnings=("Modelo no reconocido; se usa Poisson independiente.",),
            fingerprint=fingerprint,
        )
    if use_cache:
        _write_state(cache_path, state)
    return state


def score_grid_from_lambdas(
        state: ScoreModelState | Dict[str, Any],
        lambda1: float,
        lambda2: float,
        max_goals: int = 10,
        predictive_overdispersion: float | None = None,
) -> np.ndarray:
    return score_grid_from_lambdas_with_overdispersion(
        state=state,
        lambda1=lambda1,
        lambda2=lambda2,
        predictive_overdispersion=predictive_overdispersion,
        max_goals=max_goals,
    )


def score_grid_from_lambdas_with_overdispersion(
        state: ScoreModelState | Dict[str, Any],
        lambda1: float,
        lambda2: float,
        max_goals: int = 10,
        predictive_overdispersion: float | None = None,
) -> np.ndarray:
    if not isinstance(state, ScoreModelState):
        state = ScoreModelState.from_dict(dict(state or {}))
    key = state.key if state.available else DEFAULT_SCORE_MODEL
    params = state.params or {}
    lambda1 = _clamp_rate(lambda1)
    lambda2 = _clamp_rate(lambda2)
    max_goals = int(min(max(int(max_goals or 10), 4), 14))
    if key == "dixon_coles_mle":
        return dixon_coles_score_grid(lambda1, lambda2, rho=float(params.get("rho", 0.0)), max_goals=max_goals)
    if key == "bivariate_poisson_mle":
        return bivariate_poisson_score_grid(lambda1, lambda2, float(params.get("corr_share", 0.0)), max_goals=max_goals)
    if key == NEGATIVE_BINOMIAL_GLM_MODEL:
        return negative_binomial_score_grid(lambda1, lambda2, alpha=float(params.get("alpha", 0.0)), max_goals=max_goals)
    if key in {"bayesian_hierarchical_poisson", "bayesian_dynamic_poisson"}:
        alpha = float(params.get("predictive_overdispersion", 0.0))
        if predictive_overdispersion is not None:
            alpha = float(np.clip(float(predictive_overdispersion), 0.0, 2.0))
        if alpha > 1e-8:
            return negative_binomial_score_grid(lambda1, lambda2, alpha=alpha, max_goals=max_goals)
    return poisson_score_grid(lambda1=lambda1, lambda2=lambda2, max_goals=max_goals)


def score_grids_from_lambdas_with_backend(
        state: ScoreModelState | Dict[str, Any],
        lambda1_values: Sequence[float] | np.ndarray,
        lambda2_values: Sequence[float] | np.ndarray,
        max_goals: int = 10,
        backend: Any = "auto",
) -> Tuple[np.ndarray, str, List[str]]:
    if not isinstance(state, ScoreModelState):
        state = ScoreModelState.from_dict(dict(state or {}))
    max_goals = int(min(max(int(max_goals or 10), 4), 14))
    lambda1_array = np.asarray(lambda1_values, dtype=float).reshape(-1)
    lambda2_array = np.asarray(lambda2_values, dtype=float).reshape(-1)
    if lambda1_array.shape != lambda2_array.shape:
        raise ValueError("lambda1_values y lambda2_values deben tener la misma longitud.")
    if lambda1_array.size == 0:
        return np.zeros((0, max_goals + 1, max_goals + 1), dtype=float), "numpy", []
    key = state.key if state.available else DEFAULT_SCORE_MODEL
    params = state.params or {}
    requested = _normalize_backend_request(backend)
    warnings: List[str] = []
    if requested in {"auto", "cuda", "cupy"}:
        status = score_backend_status("cuda" if requested in {"cuda", "cupy"} else "auto")
        if status.get("score_backend") == "cupy":
            try:
                cp = _import_cupy()
                grids = _score_grids_from_lambdas_xp(
                    cp,
                    key,
                    params,
                    lambda1_array,
                    lambda2_array,
                    predictive_overdispersion=float(params.get("predictive_overdispersion", 0.0)),
                    max_goals=max_goals,
                )
                return cp.asnumpy(grids).astype(float), "cupy", []
            except Exception as exc:
                warnings.append(f"CuPy scoring fallo ({exc.__class__.__name__}: {exc}); se usa NumPy.")
        elif requested in {"cuda", "cupy"} and status.get("warning"):
            warnings.append(str(status.get("warning")))
    grids = _score_grids_from_lambdas_xp(
        np,
        key,
        params,
        lambda1_array,
        lambda2_array,
        predictive_overdispersion=float(params.get("predictive_overdispersion", 0.0)),
        max_goals=max_goals,
    )
    return np.asarray(grids, dtype=float), "numpy", warnings


def score_grids_from_lambdas(
        state: ScoreModelState | Dict[str, Any],
        lambda1_values: Sequence[float] | np.ndarray,
        lambda2_values: Sequence[float] | np.ndarray,
        max_goals: int = 10,
        backend: Any = "auto",
) -> np.ndarray:
    grids, _, _ = score_grids_from_lambdas_with_backend(
        state,
        lambda1_values=lambda1_values,
        lambda2_values=lambda2_values,
        max_goals=max_goals,
        backend=backend,
    )
    return grids


def probabilities_from_score_grids(
        grids: np.ndarray,
        lambda1_values: Sequence[float] | np.ndarray,
        lambda2_values: Sequence[float] | np.ndarray,
) -> List[Dict[str, float]]:
    lambda1_array = np.asarray(lambda1_values, dtype=float).reshape(-1)
    lambda2_array = np.asarray(lambda2_values, dtype=float).reshape(-1)
    grid_array = np.asarray(grids, dtype=float)
    if grid_array.ndim != 3:
        raise ValueError("grids debe ser un arreglo con forma (n, goles, goles).")
    if grid_array.shape[0] != lambda1_array.size or lambda1_array.size != lambda2_array.size:
        raise ValueError("grids y lambdas deben tener la misma longitud.")
    return [
        probabilities_from_score_grid(grid_array[index], lambda1=float(lambda1_array[index]), lambda2=float(lambda2_array[index]))
        for index in range(grid_array.shape[0])
    ]


def probabilities_from_score_grid(grid: np.ndarray, lambda1: float, lambda2: float) -> Dict[str, float]:
    grid = _normalize_grid(grid)
    goals = np.arange(grid.shape[0], dtype=int)
    home_goals, away_goals = np.meshgrid(goals, goals, indexing="ij")
    margin = home_goals - away_goals
    total_goals = home_goals + away_goals
    modal_index = int(np.argmax(grid))
    modal_score = np.unravel_index(modal_index, grid.shape)
    output = {
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "home": float(grid[margin > 0].sum()),
        "draw": float(grid[margin == 0].sum()),
        "away": float(grid[margin < 0].sum()),
        "modal_g1": int(modal_score[0]),
        "modal_g2": int(modal_score[1]),
    }
    for line in TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        over_prob = float(grid[total_goals > line].sum())
        output[f"over{suffix}"] = over_prob
        output[f"under{suffix}"] = 1.0 - over_prob
    return output


def sample_score_from_grid(grid: np.ndarray, rng: np.random.Generator) -> Tuple[int, int]:
    home, away = sample_scores_from_grid(grid, rng, size=1)
    return int(home[0]), int(away[0])


def sample_scores_from_grid(grid: np.ndarray, rng: np.random.Generator, size: int) -> Tuple[np.ndarray, np.ndarray]:
    grid = _normalize_grid(grid)
    flat = grid.ravel()
    cdf = np.cumsum(flat)
    cdf[-1] = 1.0
    draws = rng.random(int(max(size, 0)))
    indices = np.searchsorted(cdf, draws, side="right")
    home, away = np.unravel_index(indices, grid.shape)
    return home.astype(int), away.astype(int)


def bivariate_poisson_score_grid(lambda1: float, lambda2: float, corr_share: float, max_goals: int = 10) -> np.ndarray:
    corr_share = float(np.clip(corr_share, 0.0, 0.65))
    lambda_common = min(corr_share * math.sqrt(max(lambda1 * lambda2, 1e-9)), 0.95 * min(lambda1, lambda2))
    mu1 = max(lambda1 - lambda_common, 1e-9)
    mu2 = max(lambda2 - lambda_common, 1e-9)
    max_goals = int(min(max(int(max_goals or 10), 4), 14))
    grid = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    log_norm = -(mu1 + mu2 + lambda_common)
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            terms = []
            for shared in range(min(home_goals, away_goals) + 1):
                terms.append(
                    log_norm
                    + (home_goals - shared) * math.log(mu1)
                    - math.lgamma(home_goals - shared + 1)
                    + (away_goals - shared) * math.log(mu2)
                    - math.lgamma(away_goals - shared + 1)
                    + shared * math.log(max(lambda_common, 1e-12))
                    - math.lgamma(shared + 1)
                )
            grid[home_goals, away_goals] = _logsumexp(terms)
    return _normalize_grid(grid)


def negative_binomial_score_grid(lambda1: float, lambda2: float, alpha: float, max_goals: int = 10) -> np.ndarray:
    max_goals = int(min(max(int(max_goals or 10), 4), 14))
    home = _negative_binomial_pmf_vector(lambda1, alpha=alpha, max_goals=max_goals)
    away = _negative_binomial_pmf_vector(lambda2, alpha=alpha, max_goals=max_goals)
    return _normalize_grid(np.outer(home, away))


def _negative_binomial_pmf_vector(rate: float, alpha: float, max_goals: int) -> np.ndarray:
    rate = _clamp_rate(rate)
    alpha = float(np.clip(alpha, 0.0, 2.0))
    if alpha <= 1e-8:
        goals = np.arange(max_goals + 1, dtype=float)
        log_p = -rate + goals * math.log(max(rate, 1e-12)) - np.asarray([math.lgamma(int(goal) + 1) for goal in goals])
        return np.exp(log_p)
    size = 1.0 / alpha
    success = size / (size + rate)
    values = np.zeros(max_goals + 1, dtype=float)
    values[0] = math.exp(size * math.log(max(success, 1e-12)))
    ratio = rate / (size + rate)
    for goal in range(max_goals):
        values[goal + 1] = values[goal] * ((goal + size) / float(goal + 1)) * ratio
    return values


def score_backend_status(requested_device: Any = "auto") -> Dict[str, Any]:
    requested = _normalize_backend_request(requested_device)
    if requested in {"cpu", "numpy"}:
        return {
            "score_backend": "numpy",
            "actual_device": "cpu",
            "backend_supports_cuda": False,
            "cuda_available": False,
            "cuda_device_names": [],
            "warning": "",
        }
    status = _cupy_backend_status()
    if status.get("available"):
        return {
            "score_backend": "cupy",
            "actual_device": "cuda",
            "backend_supports_cuda": True,
            "cuda_available": True,
            "cuda_device_names": list(status.get("device_names") or []),
            "warning": "",
        }
    return {
        "score_backend": "numpy",
        "actual_device": "cpu",
        "backend_supports_cuda": False,
        "cuda_available": False,
        "cuda_device_names": [],
        "warning": str(status.get("warning") or "CuPy/CUDA no disponible"),
    }


def _score_grids_from_lambdas_xp(
        xp: Any,
        key: str,
        params: Dict[str, Any],
        lambda1_values: np.ndarray,
        lambda2_values: np.ndarray,
        max_goals: int,
        predictive_overdispersion: float = 0.0,
) -> Any:
    lambda1 = xp.clip(xp.asarray(lambda1_values, dtype=xp.float64), 0.05, 6.5)
    lambda2 = xp.clip(xp.asarray(lambda2_values, dtype=xp.float64), 0.05, 6.5)
    if key == "dixon_coles_mle":
        return _dixon_coles_score_grids_xp(
            xp,
            lambda1,
            lambda2,
            rho=float(params.get("rho", 0.0)),
            max_goals=max_goals,
        )
    if key == "bivariate_poisson_mle":
        return _bivariate_poisson_score_grids_xp(
            xp,
            lambda1,
            lambda2,
            corr_share=float(params.get("corr_share", 0.0)),
            max_goals=max_goals,
        )
    if key == NEGATIVE_BINOMIAL_GLM_MODEL:
        return _negative_binomial_score_grids_xp(
            xp,
            lambda1,
            lambda2,
            alpha=float(params.get("alpha", 0.0)),
            max_goals=max_goals,
        )
    if key in {"bayesian_hierarchical_poisson", "bayesian_dynamic_poisson"}:
        alpha = float(np.clip(predictive_overdispersion, 0.0, 2.0))
        if alpha > 1e-8:
            return _negative_binomial_score_grids_xp(
                xp,
                lambda1,
                lambda2,
                alpha=alpha,
                max_goals=max_goals,
            )
    return _poisson_score_grids_xp(xp, lambda1, lambda2, max_goals=max_goals)


def _poisson_score_grids_xp(xp: Any, lambda1: Any, lambda2: Any, max_goals: int) -> Any:
    goals_np = np.arange(max_goals + 1, dtype=float)
    log_factorials_np = np.asarray([math.lgamma(int(goal) + 1) for goal in goals_np], dtype=float)
    goals = xp.asarray(goals_np, dtype=xp.float64)
    log_factorials = xp.asarray(log_factorials_np, dtype=xp.float64)
    lambda1 = xp.asarray(lambda1, dtype=xp.float64).reshape(-1)
    lambda2 = xp.asarray(lambda2, dtype=xp.float64).reshape(-1)
    log_p1 = -lambda1[:, None] + goals[None, :] * xp.log(xp.maximum(lambda1[:, None], xp.float64(1e-12))) - log_factorials[None, :]
    log_p2 = -lambda2[:, None] + goals[None, :] * xp.log(xp.maximum(lambda2[:, None], xp.float64(1e-12))) - log_factorials[None, :]
    probs1 = xp.exp(log_p1)
    probs2 = xp.exp(log_p2)
    grids = probs1[:, :, None] * probs2[:, None, :]
    return _normalize_batched_grids_xp(xp, grids)


def _dixon_coles_score_grids_xp(xp: Any, lambda1: Any, lambda2: Any, rho: float, max_goals: int) -> Any:
    grids = _poisson_score_grids_xp(xp, lambda1, lambda2, max_goals=max_goals)
    adjusted = grids.copy()
    rho = float(np.clip(rho, -0.25, 0.25))
    low_factors = xp.empty((int(adjusted.shape[0]), 2, 2), dtype=xp.float64)
    low_factors[:, 0, 0] = xp.maximum(1.0 - lambda1 * lambda2 * rho, 1e-6)
    low_factors[:, 0, 1] = xp.maximum(1.0 + lambda1 * rho, 1e-6)
    low_factors[:, 1, 0] = xp.maximum(1.0 + lambda2 * rho, 1e-6)
    low_factors[:, 1, 1] = xp.maximum(1.0 - rho, 1e-6)
    adjusted[:, :2, :2] *= low_factors
    return _normalize_batched_grids_xp(xp, adjusted)


def _bivariate_poisson_score_grids_xp(xp: Any, lambda1: Any, lambda2: Any, corr_share: Any, max_goals: int) -> Any:
    lambda1 = xp.asarray(lambda1, dtype=xp.float64).reshape(-1)
    lambda2 = xp.asarray(lambda2, dtype=xp.float64).reshape(-1)
    corr = xp.clip(xp.asarray(corr_share, dtype=xp.float64), 0.0, 0.65)
    if int(corr.size) == 1:
        corr = xp.ones_like(lambda1, dtype=xp.float64) * corr.reshape(-1)[0]
    else:
        corr = corr.reshape(-1)
    lambda_common = xp.minimum(corr * xp.sqrt(xp.maximum(lambda1 * lambda2, 1e-9)), 0.95 * xp.minimum(lambda1, lambda2))
    mu1 = xp.maximum(lambda1 - lambda_common, 1e-9)
    mu2 = xp.maximum(lambda2 - lambda_common, 1e-9)
    goals_np = np.arange(max_goals + 1, dtype=int)
    log_factorials_np = np.asarray([math.lgamma(int(goal) + 1) for goal in goals_np], dtype=float)
    goals = xp.asarray(goals_np, dtype=xp.int64)
    home_goals, away_goals = xp.meshgrid(goals, goals, indexing="ij")
    log_factorials = xp.asarray(log_factorials_np, dtype=xp.float64)
    grids = xp.zeros((int(lambda1.size), max_goals + 1, max_goals + 1), dtype=xp.float64)
    log_norm = -(mu1 + mu2 + lambda_common)
    log_mu1 = xp.log(xp.maximum(mu1, 1e-12))
    log_mu2 = xp.log(xp.maximum(mu2, 1e-12))
    log_common = xp.log(xp.maximum(lambda_common, 1e-12))
    for shared in range(max_goals + 1):
        home_private = home_goals - shared
        away_private = away_goals - shared
        valid = (home_private >= 0) & (away_private >= 0)
        safe_home = xp.maximum(home_private, 0)
        safe_away = xp.maximum(away_private, 0)
        term = (
            log_norm[:, None, None]
            + safe_home[None, :, :] * log_mu1[:, None, None]
            - log_factorials[safe_home][None, :, :]
            + safe_away[None, :, :] * log_mu2[:, None, None]
            - log_factorials[safe_away][None, :, :]
            + shared * log_common[:, None, None]
            - float(math.lgamma(shared + 1))
        )
        grids += xp.where(valid[None, :, :], xp.exp(term), 0.0)
    return _normalize_batched_grids_xp(xp, grids)


def _negative_binomial_score_grids_xp(xp: Any, lambda1: Any, lambda2: Any, alpha: float, max_goals: int) -> Any:
    home = _negative_binomial_pmf_matrix_xp(xp, lambda1, alpha=alpha, max_goals=max_goals)
    away = _negative_binomial_pmf_matrix_xp(xp, lambda2, alpha=alpha, max_goals=max_goals)
    grids = home[:, :, None] * away[:, None, :]
    return _normalize_batched_grids_xp(xp, grids)


def _negative_binomial_pmf_matrix_xp(xp: Any, rate: Any, alpha: float, max_goals: int) -> Any:
    rate = xp.asarray(rate, dtype=xp.float64).reshape(-1)
    alpha = float(np.clip(alpha, 0.0, 2.0))
    if alpha <= 1e-8:
        goals_np = np.arange(max_goals + 1, dtype=float)
        log_factorials_np = np.asarray([math.lgamma(int(goal) + 1) for goal in goals_np], dtype=float)
        goals = xp.asarray(goals_np, dtype=xp.float64)
        log_factorials = xp.asarray(log_factorials_np, dtype=xp.float64)
        log_p = -rate[:, None] + goals[None, :] * xp.log(xp.maximum(rate[:, None], xp.float64(1e-12))) - log_factorials[None, :]
        return xp.exp(log_p)
    size = xp.float64(1.0 / alpha)
    success = size / (size + rate)
    ratio = rate / (size + rate)
    values = xp.zeros((int(rate.size), max_goals + 1), dtype=xp.float64)
    values[:, 0] = xp.exp(size * xp.log(xp.maximum(success, xp.float64(1e-12))))
    for goal in range(max_goals):
        values[:, goal + 1] = values[:, goal] * ((goal + size) / float(goal + 1)) * ratio
    return values


def _normalize_batched_grids_xp(xp: Any, grids: Any) -> Any:
    grids = xp.maximum(xp.nan_to_num(grids, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    totals = xp.sum(grids, axis=(1, 2), keepdims=True)
    uniform = xp.full_like(grids, 1.0 / max(int(grids.shape[1]) * int(grids.shape[2]), 1))
    return xp.where(totals > 0.0, grids / xp.maximum(totals, 1e-12), uniform)


def _fit_bayesian_state(
        key: str,
        label: str,
        fingerprint: str,
        rows: List[Dict[str, Any]],
        teams: Iterable[str],
        config: Dict[str, Any],
) -> ScoreModelState:
    try:
        _ensure_pytensor_cxx_flag()
        import pymc as pm  # type: ignore
    except Exception:
        return ScoreModelState(
            key=key,
            label=label,
            available=False,
            params={},
            warnings=("PyMC no esta instalado; instala dependencias bayesianas para activar este modelo.",),
            fingerprint=fingerprint,
        )
    if not rows:
        return ScoreModelState(
            key=key,
            label=label,
            available=False,
            params={},
            warnings=("Sin historico suficiente para inferencia bayesiana.",),
            fingerprint=fingerprint,
        )
    team_list = sorted(set(str(team) for team in teams) | {row["home"] for row in rows} | {row["away"] for row in rows})
    team_index = {team: index for index, team in enumerate(team_list)}
    home_idx = np.asarray([team_index[row["home"]] for row in rows], dtype=int)
    away_idx = np.asarray([team_index[row["away"]] for row in rows], dtype=int)
    home_goals = np.asarray([row["g1"] for row in rows], dtype=int)
    away_goals = np.asarray([row["g2"] for row in rows], dtype=int)
    weights = np.asarray([max(float(row.get("weight", 1.0) or 1.0), 1e-6) for row in rows], dtype=float)
    is_neutral = np.asarray([1.0 if bool(row.get("neutral", False)) else 0.0 for row in rows], dtype=float)
    is_home_not_neutral = 1.0 - is_neutral
    draws = int(config.get("bayes_draws") or 500)
    tune = int(config.get("bayes_tune") or 500)
    chains = int(config.get("bayes_chains") or 2)
    seed = int(config.get("seed") or 2026)
    target_accept = float(np.clip(config.get("bayes_target_accept", 0.92), 0.8, 0.995))
    max_treedepth = int(config.get("bayes_max_treedepth", 12))
    sample_kwargs = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "random_seed": seed,
        "target_accept": target_accept,
        "nuts": {"max_treedepth": max(6, max_treedepth)},
        "progressbar": False,
    }
    fit_warnings: List[str] = []
    if key == "bayesian_dynamic_poisson":
        unique_periods = sorted(set(_row_periods(rows)))
        period_idx = np.asarray(_coalesce_model_period_indices(_row_periods(rows)), dtype=int)
        if len(unique_periods) >= 2:
            with pm.Model(coords={"period": unique_periods, "team": team_list, "obs_id": np.arange(len(rows))}):
                attack_rw_scale = pm.HalfNormal("attack_rw_scale", 0.25)
                defense_rw_scale = pm.HalfNormal("defense_rw_scale", 0.25)
                attack_raw = pm.GaussianRandomWalk(
                    "attack_raw",
                    sigma=attack_rw_scale,
                    init_dist=pm.Normal.dist(0.0, 0.35),
                    dims=("period", "team"),
                )
                defense_raw = pm.GaussianRandomWalk(
                    "defense_raw",
                    sigma=defense_rw_scale,
                    init_dist=pm.Normal.dist(0.0, 0.35),
                    dims=("period", "team"),
                )
                attack = pm.Deterministic("attack", attack_raw - pm.math.mean(attack_raw, axis=1, keepdims=True), dims=("period", "team"))
                defense = pm.Deterministic("defense", defense_raw - pm.math.mean(defense_raw, axis=1, keepdims=True), dims=("period", "team"))
                intercept = pm.Normal("intercept", math.log(1.2), 0.45)
                home_adv = pm.Normal("home_adv", 0.0, 0.35)
                neutral_adv = pm.Normal("neutral_adv", 0.0, 0.25)
                obs_overdispersion = pm.HalfNormal("obs_overdispersion", 0.8)
                home_mu = pm.math.exp(
                    intercept
                    + home_adv * is_home_not_neutral
                    + neutral_adv * is_neutral
                    + attack[period_idx, home_idx]
                    - defense[period_idx, away_idx]
                )
                away_mu = pm.math.exp(
                    intercept
                    + neutral_adv * is_neutral
                    + attack[period_idx, away_idx]
                    - defense[period_idx, home_idx]
                )
                home_likelihood = pm.logp(pm.NegativeBinomial.dist(mu=home_mu, alpha=obs_overdispersion), home_goals)
                away_likelihood = pm.logp(pm.NegativeBinomial.dist(mu=away_mu, alpha=obs_overdispersion), away_goals)
                pm.Potential("weighted_likelihood", pm.math.sum(weights * (home_likelihood + away_likelihood)))
                trace = pm.sample(**sample_kwargs)
            posterior = trace.posterior
            latest_period = -1
            attack_mean = np.asarray(
                posterior["attack"].isel(period=latest_period).mean(dim=("chain", "draw"))
            ).astype(float).tolist()
            attack_sd = np.asarray(
                posterior["attack"].isel(period=latest_period).std(dim=("chain", "draw"))
            ).astype(float).tolist()
            defense_mean = np.asarray(
                posterior["defense"].isel(period=latest_period).mean(dim=("chain", "draw"))
            ).astype(float).tolist()
            defense_sd = np.asarray(
                posterior["defense"].isel(period=latest_period).std(dim=("chain", "draw"))
            ).astype(float).tolist()
            intercept_mean = float(np.asarray(posterior["intercept"].mean(dim=("chain", "draw"))))
            intercept_sd = float(np.asarray(posterior["intercept"].std(dim=("chain", "draw"))))
            home_adv_mean = float(np.asarray(posterior["home_adv"].mean(dim=("chain", "draw"))))
            home_adv_sd = float(np.asarray(posterior["home_adv"].std(dim=("chain", "draw"))))
            neutral_adv_mean = float(np.asarray(posterior["neutral_adv"].mean(dim=("chain", "draw"))))
            neutral_adv_sd = float(np.asarray(posterior["neutral_adv"].std(dim=("chain", "draw"))))
            obs_alpha = float(np.asarray(posterior["obs_overdispersion"].mean(dim=("chain", "draw"))))
            obs_alpha_sd = float(np.asarray(posterior["obs_overdispersion"].std(dim=("chain", "draw"))))
            predictive_overdispersion = float(
                np.clip(
                    obs_alpha + 0.5 * float(np.mean(np.square(np.asarray(attack_sd)) + np.square(np.asarray(defense_sd)))),
                    0.0,
                    2.0,
                )
            )
            params = {
                "teams": team_list,
                "periods": unique_periods,
                "attack": attack_mean,
                "attack_sd": attack_sd,
                "defense": defense_mean,
                "defense_sd": defense_sd,
                "intercept": intercept_mean,
                "intercept_sd": intercept_sd,
                "home_adv": home_adv_mean,
                "home_adv_sd": home_adv_sd,
                "neutral_adv": neutral_adv_mean,
                "neutral_adv_sd": neutral_adv_sd,
                "obs_overdispersion": obs_alpha,
                "obs_overdispersion_sd": obs_alpha_sd,
                "predictive_overdispersion": predictive_overdispersion,
                "draws": draws,
                "tune": tune,
                "chains": chains,
                "dynamic_period": unique_periods[latest_period],
            }
            fit_warnings.append("Bayes dinámico habilitado con random walk temporal por periodos.")
            return ScoreModelState(key=key, label=label, available=True, params=params, warnings=tuple(fit_warnings), fingerprint=fingerprint)
        fit_warnings.append("Bayes dinámico sin suficientes periodos; se usa modelo jerárquico estático.")
    with pm.Model(coords={"team": team_list, "obs_id": np.arange(len(rows))}):
        attack_scale = pm.HalfNormal("attack_scale", 0.35)
        defense_scale = pm.HalfNormal("defense_scale", 0.35)
        attack_offset = pm.Normal("attack_offset", 0.0, 1.0, dims="team")
        defense_offset = pm.Normal("defense_offset", 0.0, 1.0, dims="team")
        attack_raw = attack_scale * attack_offset
        defense_raw = defense_scale * defense_offset
        attack = pm.Deterministic("attack", attack_raw - pm.math.mean(attack_raw), dims="team")
        defense = pm.Deterministic("defense", defense_raw - pm.math.mean(defense_raw), dims="team")
        intercept = pm.Normal("intercept", math.log(1.2), 0.45)
        home_adv = pm.Normal("home_adv", 0.0, 0.35)
        neutral_adv = pm.Normal("neutral_adv", 0.0, 0.25)
        obs_overdispersion = pm.HalfNormal("obs_overdispersion", 0.8)
        home_mu = pm.math.exp(
            intercept
            + home_adv * is_home_not_neutral
            + neutral_adv * is_neutral
            + attack[home_idx]
            - defense[away_idx]
        )
        away_mu = pm.math.exp(
            intercept
            + neutral_adv * is_neutral
            + attack[away_idx]
            - defense[home_idx]
        )
        home_likelihood = pm.logp(pm.NegativeBinomial.dist(mu=home_mu, alpha=obs_overdispersion), home_goals)
        away_likelihood = pm.logp(pm.NegativeBinomial.dist(mu=away_mu, alpha=obs_overdispersion), away_goals)
        pm.Potential("weighted_likelihood", pm.math.sum(weights * (home_likelihood + away_likelihood)))
        trace = pm.sample(**sample_kwargs)
    posterior = trace.posterior
    params = {
        "teams": team_list,
        "attack": np.asarray(posterior["attack"].mean(dim=("chain", "draw"))).astype(float).tolist(),
        "attack_sd": np.asarray(posterior["attack"].std(dim=("chain", "draw"))).astype(float).tolist(),
        "defense": np.asarray(posterior["defense"].mean(dim=("chain", "draw"))).astype(float).tolist(),
        "defense_sd": np.asarray(posterior["defense"].std(dim=("chain", "draw"))).astype(float).tolist(),
        "intercept": float(np.asarray(posterior["intercept"].mean(dim=("chain", "draw")))),
        "intercept_sd": float(np.asarray(posterior["intercept"].std(dim=("chain", "draw")))),
        "home_adv": float(np.asarray(posterior["home_adv"].mean(dim=("chain", "draw")))),
        "home_adv_sd": float(np.asarray(posterior["home_adv"].std(dim=("chain", "draw")))),
        "neutral_adv": float(np.asarray(posterior["neutral_adv"].mean(dim=("chain", "draw")))),
        "neutral_adv_sd": float(np.asarray(posterior["neutral_adv"].std(dim=("chain", "draw")))),
        "obs_overdispersion": float(np.asarray(posterior["obs_overdispersion"].mean(dim=("chain", "draw")))),
        "obs_overdispersion_sd": float(np.asarray(posterior["obs_overdispersion"].std(dim=("chain", "draw")))),
        "draws": draws,
        "tune": tune,
        "chains": chains,
    }
    attack_sd_array = np.asarray(params.get("attack_sd", []), dtype=float)
    defense_sd_array = np.asarray(params.get("defense_sd", []), dtype=float)
    params["predictive_overdispersion"] = float(
        np.clip(
            params["obs_overdispersion"] + 0.5 * float(np.mean(np.square(attack_sd_array) + np.square(defense_sd_array))),
            0.0,
            2.0,
        )
    )
    if not fit_warnings:
        fit_warnings.append("Bayes jerárquico Poisson con priors de ataque/defensa y ventaja local/neutral.")
    return ScoreModelState(key=key, label=label, available=True, params=params, warnings=tuple(fit_warnings), fingerprint=fingerprint)


def _ensure_pytensor_cxx_flag() -> bool:
    if _has_cxx_compiler() or _pytensor_flags_include_cxx(os.environ.get("PYTENSOR_FLAGS", "")):
        return False
    existing = os.environ.get("PYTENSOR_FLAGS", "").strip()
    os.environ["PYTENSOR_FLAGS"] = f"{existing},cxx=" if existing else "cxx="
    return True


def _has_cxx_compiler() -> bool:
    if str(os.environ.get("CXX") or "").strip():
        return True
    return any(shutil.which(candidate) for candidate in ("g++", "clang++", "cl.exe"))


def _pytensor_flags_include_cxx(flags: str) -> bool:
    for part in str(flags or "").split(","):
        key = part.strip().split("=", 1)[0].strip().lower()
        if key == "cxx":
            return True
    return False


def _history_model_rows(history_df: pd.DataFrame | None, base_model: Any, recency_weight: Any = 0.0) -> List[Dict[str, Any]]:
    if history_df is None or history_df.empty:
        return []
    required = {"Team 1", "Team 2", "G1", "G2"}
    if not required.issubset(history_df.columns):
        return []
    rows: List[Dict[str, Any]] = []
    working = history_df.copy()
    if "Date" in working:
        working = working.sort_values("Date", kind="stable")
    working["_score_model_weight"] = _history_row_weights(working, recency_weight)
    for _, row in working.iterrows():
        home = str(row.get("Team 1", "")).strip()
        away = str(row.get("Team 2", "")).strip()
        if not home or not away:
            continue
        try:
            g1 = int(float(row.get("G1")))
            g2 = int(float(row.get("G2")))
            lambda1, lambda2 = base_model.expected_goals(home, away)
        except Exception:
            continue
        rows.append({
            "date": str(row.get("Date", "")),
            "home": home,
            "away": away,
            "g1": max(g1, 0),
            "g2": max(g2, 0),
            "lambda1": _clamp_rate(lambda1),
            "lambda2": _clamp_rate(lambda2),
            "weight": max(float(row.get("_score_model_weight", 1.0) or 1.0), 1e-6),
            "neutral": _fixture_neutral({
                "neutral": row.get("neutral", False),
                "Venue": row.get("Venue", ""),
                "venue": row.get("venue", ""),
            }),
        })
    return rows


def _history_row_weights(history: pd.DataFrame, recency_weight: Any) -> pd.Series:
    try:
        weight = float(recency_weight)
    except (TypeError, ValueError):
        weight = 0.0
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 0.0 or "Date" not in history:
        return pd.Series([1.0] * len(history), index=history.index, dtype=float)
    dates = pd.to_datetime(history["Date"], errors="coerce")
    if dates.notna().sum() < 2:
        return pd.Series([1.0] * len(history), index=history.index, dtype=float)
    latest = dates.max()
    years_ago = (latest - dates).dt.days.fillna(0).clip(lower=0) / 365.25
    recent_curve = np.exp(-years_ago / 12.0)
    weights = (1.0 - weight) + weight * recent_curve
    return pd.Series(weights, index=history.index, dtype=float).clip(lower=0.1)


def _fit_statsmodels_lambda_model(
        rows: List[Dict[str, Any]],
        teams: Iterable[str],
        config: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    mode = str(config.get("stat_lambda_model", STATSMODELS_POISSON_GLM_MODEL) or "").strip().lower()
    if mode in {"0", "false", "off", "none", "raw", "disabled"}:
        return {}, []
    min_matches = int(max(int(config.get("stat_glm_min_matches") or 12), 4))
    if len(rows) < min_matches:
        return {}, [f"GLM statsmodels omitido: {len(rows)} partidos < minimo {min_matches}."]
    try:
        import statsmodels.api as sm  # type: ignore
    except Exception as exc:
        return {}, [f"statsmodels no disponible ({exc.__class__.__name__}: {exc}); lambdas raw."]

    frame = _lambda_model_training_frame(rows)
    if frame.empty:
        return {}, ["GLM statsmodels omitido: sin filas de conteo validas."]
    columns = _lambda_design_columns(frame, teams)
    alphas = _glm_alpha_candidates(config.get("stat_glm_alphas"))
    train_frame, validation_frame = _lambda_temporal_split(frame, config.get("stat_glm_validation_fraction", 0.2))
    best_alpha = alphas[0]
    validation_scores: List[Dict[str, Any]] = []
    if not validation_frame.empty and train_frame["match_idx"].nunique() >= 4:
        for alpha in alphas:
            try:
                params = _fit_poisson_glm_params(sm, train_frame, columns, alpha)
                score = _lambda_model_validation_score(validation_frame, columns, params)
            except Exception as exc:
                validation_scores.append({"alpha": float(alpha), "available": False, "warning": f"{exc.__class__.__name__}: {exc}"})
                continue
            validation_scores.append({"alpha": float(alpha), "available": True, **score})
        available_scores = [item for item in validation_scores if item.get("available")]
        if available_scores:
            best = min(available_scores, key=lambda item: float(item.get("objective", float("inf"))))
            best_alpha = float(best["alpha"])

    try:
        params = _fit_poisson_glm_params(sm, frame, columns, best_alpha)
    except Exception as exc:
        return {}, [f"GLM statsmodels fallo ({exc.__class__.__name__}: {exc}); lambdas raw."]

    diagnostics = _lambda_model_diagnostics(frame, columns, params)
    warnings: List[str] = []
    dispersion = float(diagnostics.get("pearson_dispersion") or 0.0)
    if dispersion > 1.5:
        warnings.append(f"Sobredispersion Poisson detectada ({dispersion:.2f}); considerar Negative Binomial si mejora walk-forward.")
    zero_gap = float(diagnostics.get("zero_rate_gap") or 0.0)
    if zero_gap > 0.08:
        warnings.append(f"Exceso de ceros detectado ({zero_gap:.3f}); considerar ZIP/ZINB solo si gana en backtest temporal.")

    params_dict = {column: float(value) for column, value in zip(columns, params)}
    model = {
        "type": STATSMODELS_POISSON_GLM_MODEL,
        "offset": "log_base_lambda",
        "regularization": "ridge_l2",
        "alpha": float(best_alpha),
        "columns": list(columns),
        "params": params_dict,
        "n_matches": int(frame["match_idx"].nunique()),
        "n_count_rows": int(frame.shape[0]),
        "validation": {
            "source": "temporal_holdout_pretest",
            "objective": "score_log_loss + 0.75*result_log_loss + 0.25*ou25_log_loss",
            "trials": validation_scores,
        },
        "diagnostics": diagnostics,
    }
    return model, warnings


def _lambda_model_training_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for match_idx, row in enumerate(rows):
        weight = max(float(row.get("weight", 1.0) or 1.0), 1e-6)
        records.append({
            "match_idx": match_idx,
            "date": row.get("date", ""),
            "side": "home",
            "team": str(row.get("home", "")),
            "opponent": str(row.get("away", "")),
            "is_home": 1.0,
            "goals": int(row.get("g1", 0)),
            "base_lambda": _clamp_rate(row.get("lambda1", 1.0)),
            "weight": weight,
            "actual_home_goals": int(row.get("g1", 0)),
            "actual_away_goals": int(row.get("g2", 0)),
        })
        records.append({
            "match_idx": match_idx,
            "date": row.get("date", ""),
            "side": "away",
            "team": str(row.get("away", "")),
            "opponent": str(row.get("home", "")),
            "is_home": 0.0,
            "goals": int(row.get("g2", 0)),
            "base_lambda": _clamp_rate(row.get("lambda2", 1.0)),
            "weight": weight,
            "actual_home_goals": int(row.get("g1", 0)),
            "actual_away_goals": int(row.get("g2", 0)),
        })
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.sort_values(["match_idx", "side"], kind="stable").reset_index(drop=True)


def _lambda_design_columns(frame: pd.DataFrame, teams: Iterable[str]) -> List[str]:
    team_names = sorted(
        set(str(team) for team in teams if str(team).strip())
        | set(str(team) for team in frame.get("team", pd.Series(dtype=str)).dropna().tolist() if str(team).strip())
        | set(str(team) for team in frame.get("opponent", pd.Series(dtype=str)).dropna().tolist() if str(team).strip())
    )
    return ["const", "is_home", *[f"attack::{team}" for team in team_names], *[f"defense::{team}" for team in team_names]]


def _lambda_design_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    matrix = np.zeros((int(frame.shape[0]), len(columns)), dtype=float)
    index = {column: idx for idx, column in enumerate(columns)}
    if "const" in index:
        matrix[:, index["const"]] = 1.0
    if "is_home" in index and "is_home" in frame:
        matrix[:, index["is_home"]] = pd.to_numeric(frame["is_home"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for row_index, (_, row) in enumerate(frame.iterrows()):
        attack_key = f"attack::{str(row.get('team', ''))}"
        defense_key = f"defense::{str(row.get('opponent', ''))}"
        if attack_key in index:
            matrix[row_index, index[attack_key]] = 1.0
        if defense_key in index:
            matrix[row_index, index[defense_key]] = 1.0
    return matrix


def _fit_poisson_glm_params(sm: Any, frame: pd.DataFrame, columns: Sequence[str], alpha: float) -> np.ndarray:
    x = _lambda_design_matrix(frame, columns)
    y = pd.to_numeric(frame["goals"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    offset = np.log(np.clip(pd.to_numeric(frame["base_lambda"], errors="coerce").fillna(1.0).to_numpy(dtype=float), 1e-9, None))
    weights = np.clip(pd.to_numeric(frame["weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float), 1e-6, None)
    model = sm.GLM(y, x, family=sm.families.Poisson(), offset=offset, freq_weights=weights)
    result = model.fit_regularized(alpha=max(float(alpha), 1e-8), L1_wt=0.0, maxiter=300, cnvrg_tol=1e-8)
    return np.asarray(result.params, dtype=float)


def _glm_alpha_candidates(value: Any) -> List[float]:
    if isinstance(value, str) and value.strip():
        candidates = []
        for part in value.split(","):
            try:
                candidates.append(float(part.strip()))
            except ValueError:
                continue
        if candidates:
            return sorted({max(candidate, 1e-8) for candidate in candidates})
    if isinstance(value, (list, tuple, set)):
        candidates = []
        for item in value:
            try:
                candidates.append(float(item))
            except (TypeError, ValueError):
                continue
        if candidates:
            return sorted({max(candidate, 1e-8) for candidate in candidates})
    return [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]


def _lambda_temporal_split(frame: pd.DataFrame, validation_fraction: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        fraction = float(validation_fraction)
    except (TypeError, ValueError):
        fraction = 0.2
    fraction = float(np.clip(fraction, 0.05, 0.4))
    match_ids = list(dict.fromkeys(frame["match_idx"].astype(int).tolist()))
    if len(match_ids) < 8:
        return frame.copy(), frame.iloc[0:0].copy()
    validation_count = int(np.clip(round(len(match_ids) * fraction), 2, max(len(match_ids) - 4, 1)))
    validation_ids = set(match_ids[-validation_count:])
    train = frame[~frame["match_idx"].isin(validation_ids)].copy()
    validation = frame[frame["match_idx"].isin(validation_ids)].copy()
    return train, validation


def _lambda_model_validation_score(frame: pd.DataFrame, columns: Sequence[str], params: np.ndarray) -> Dict[str, float]:
    predicted = _predict_lambda_frame(frame, columns, params)
    working = frame.copy()
    working["_predicted_lambda"] = predicted
    score_log_loss = 0.0
    result_log_loss = 0.0
    ou25_log_loss = 0.0
    evaluated = 0
    for _, group in working.groupby("match_idx", sort=True):
        home_rows = group[group["side"] == "home"]
        away_rows = group[group["side"] == "away"]
        if home_rows.empty or away_rows.empty:
            continue
        home_row = home_rows.iloc[0]
        away_row = away_rows.iloc[0]
        actual_home = int(home_row.get("actual_home_goals", home_row.get("goals", 0)))
        actual_away = int(home_row.get("actual_away_goals", away_row.get("goals", 0)))
        lambda_home = _clamp_rate(home_row.get("_predicted_lambda", home_row.get("base_lambda", 1.0)))
        lambda_away = _clamp_rate(away_row.get("_predicted_lambda", away_row.get("base_lambda", 1.0)))
        grid = poisson_score_grid(lambda_home, lambda_away, max_goals=10)
        capped_home = min(max(actual_home, 0), grid.shape[0] - 1)
        capped_away = min(max(actual_away, 0), grid.shape[1] - 1)
        probabilities = probabilities_from_score_grid(grid, lambda_home, lambda_away)
        actual_outcome = "home" if actual_home > actual_away else "away" if actual_away > actual_home else "draw"
        score_log_loss += -math.log(max(float(grid[capped_home, capped_away]), 1e-12))
        result_log_loss += -math.log(max(float(probabilities.get(actual_outcome, 0.0)), 1e-12))
        actual_ou = "over25" if (actual_home + actual_away) > 2.5 else "under25"
        ou25_log_loss += -math.log(max(float(probabilities.get(actual_ou, 0.0)), 1e-12))
        evaluated += 1
    denominator = max(evaluated, 1)
    score_ll = score_log_loss / denominator
    result_ll = result_log_loss / denominator
    ou25_ll = ou25_log_loss / denominator
    return {
        "evaluated_matches": int(evaluated),
        "score_log_loss": round(float(score_ll), 6),
        "result_log_loss": round(float(result_ll), 6),
        "ou25_log_loss": round(float(ou25_ll), 6),
        "objective": round(float(score_ll + (0.75 * result_ll) + (0.25 * ou25_ll)), 6),
    }


def _lambda_model_diagnostics(frame: pd.DataFrame, columns: Sequence[str], params: np.ndarray) -> Dict[str, Any]:
    mu = np.clip(_predict_lambda_frame(frame, columns, params), 1e-9, None)
    y = pd.to_numeric(frame["goals"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    weights = np.clip(pd.to_numeric(frame["weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float), 1e-6, None)
    pearson = float(np.sum(weights * np.square(y - mu) / mu))
    df_resid = max(float(np.sum(weights) - len(params)), 1.0)
    observed_zero_rate = float(np.average((y <= 0).astype(float), weights=weights))
    expected_zero_rate = float(np.average(np.exp(-mu), weights=weights))
    return {
        "pearson_chi2": round(pearson, 6),
        "df_resid": round(df_resid, 6),
        "pearson_dispersion": round(pearson / df_resid, 6),
        "observed_zero_rate": round(observed_zero_rate, 6),
        "expected_zero_rate": round(expected_zero_rate, 6),
        "zero_rate_gap": round(observed_zero_rate - expected_zero_rate, 6),
        "mean_observed_goals": round(float(np.average(y, weights=weights)), 6),
        "mean_predicted_goals": round(float(np.average(mu, weights=weights)), 6),
    }


def _negative_binomial_alpha(lambda_model: Dict[str, Any]) -> float:
    diagnostics = (lambda_model or {}).get("diagnostics") or {}
    dispersion = float(diagnostics.get("pearson_dispersion") or 1.0)
    mean_mu = max(float(diagnostics.get("mean_predicted_goals") or 1.0), 1e-6)
    alpha = max((dispersion - 1.0) / mean_mu, 0.0)
    return float(np.clip(alpha, 0.0, 2.0))


def _predict_lambda_frame(frame: pd.DataFrame, columns: Sequence[str], params: np.ndarray) -> np.ndarray:
    x = _lambda_design_matrix(frame, columns)
    offset = np.log(np.clip(pd.to_numeric(frame["base_lambda"], errors="coerce").fillna(1.0).to_numpy(dtype=float), 1e-9, None))
    eta = offset + np.asarray(x @ np.asarray(params, dtype=float), dtype=float)
    return np.clip(np.exp(np.clip(eta, -4.0, 2.2)), 0.05, 6.5)


def _statsmodels_lambdas_for_match(
        state: ScoreModelState,
        home: str,
        away: str,
        base_lambdas: Tuple[float, float],
) -> Tuple[float, float] | None:
    lambda_model = (state.params or {}).get("lambda_model") or {}
    if not lambda_model:
        return None
    try:
        home_lambda = _predict_lambda_from_model(lambda_model, team=home, opponent=away, is_home=True, base_lambda=base_lambdas[0])
        away_lambda = _predict_lambda_from_model(lambda_model, team=away, opponent=home, is_home=False, base_lambda=base_lambdas[1])
    except Exception:
        return None
    return _clamp_rate(home_lambda), _clamp_rate(away_lambda)


def _predict_lambda_from_model(lambda_model: Dict[str, Any], team: str, opponent: str, is_home: bool, base_lambda: Any) -> float:
    params = lambda_model.get("params") or {}
    eta = math.log(max(_clamp_rate(base_lambda), 1e-9))
    eta += float(params.get("const", 0.0))
    eta += float(params.get("is_home", 0.0)) * (1.0 if is_home else 0.0)
    eta += float(params.get(f"attack::{str(team)}", 0.0))
    eta += float(params.get(f"defense::{str(opponent)}", 0.0))
    return float(np.clip(math.exp(float(np.clip(eta, -4.0, 2.2))), 0.05, 6.5))


def _rows_with_lambda_model(rows: List[Dict[str, Any]], lambda_model: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not lambda_model:
        return rows
    output: List[Dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        home = str(row.get("home", ""))
        away = str(row.get("away", ""))
        updated["lambda1"] = _predict_lambda_from_model(lambda_model, team=home, opponent=away, is_home=True, base_lambda=row.get("lambda1", 1.0))
        updated["lambda2"] = _predict_lambda_from_model(lambda_model, team=away, opponent=home, is_home=False, base_lambda=row.get("lambda2", 1.0))
        output.append(updated)
    return output


def _fixture_neutral(match: Dict[str, Any] | None) -> bool:
    if not isinstance(match, dict):
        return False
    if bool(match.get("neutral", False)):
        return True
    if bool(match.get("is_neutral", False)):
        return True
    for key in ("venue", "Venue", "venue_name", "VenueName", "venue_name_raw", "venue_name_clean"):
        value = str(match.get(key, "")).strip().lower()
        if value in {"neutral", "home neutral", "neutro", "sede neutral", "venue: neutral"}:
            return True
    return False


def _fixture_period_from_match(match: Dict[str, Any] | None) -> int | None:
    if not isinstance(match, dict):
        return None
    for key in ("date", "Date", "match_date", "event_date", "kickoff", "fecha"):
        parsed = pd.to_datetime(match.get(key), errors="coerce")
        if pd.isna(parsed):
            continue
        return int(parsed.year * 100 + parsed.month)
    return None


def _bayesian_match_period_index(
        periods: Sequence[int] | None,
        match: Dict[str, Any] | None,
) -> int:
    if not periods:
        return -1
    available = sorted(set(int(item) for item in periods))
    if not available:
        return -1
    fixture_period = _fixture_period_from_match(match)
    if fixture_period is None:
        return len(available) - 1
    candidates = [period for period in available if period <= fixture_period]
    if not candidates:
        return 0
    selected = max(candidates)
    try:
        return available.index(selected)
    except ValueError:
        return len(available) - 1


def _bayesian_predictive_overdispersion(
        state: ScoreModelState,
        home: str,
        away: str,
        match: Dict[str, Any] | None = None,
) -> float:
    if not state.available:
        return 0.0
    params = state.params or {}
    obs_alpha = float(params.get("obs_overdispersion", 0.0) or 0.0)
    teams = [str(team) for team in params.get("teams", [])]
    if not teams:
        return float(np.clip(obs_alpha, 0.0, 2.0))
    try:
        home_idx = teams.index(str(home))
        away_idx = teams.index(str(away))
    except ValueError:
        return float(np.clip(obs_alpha, 0.0, 2.0))
    attack_sd = np.asarray(params.get("attack_sd") or [], dtype=float)
    defense_sd = np.asarray(params.get("defense_sd") or [], dtype=float)
    if attack_sd.size == 0 or defense_sd.size == 0:
        return float(np.clip(obs_alpha, 0.0, 2.0))
    if attack_sd.ndim == 2:
        period_index = _bayesian_match_period_index(params.get("periods"), match)
        if period_index < 0 or period_index >= attack_sd.shape[0]:
            return float(np.clip(obs_alpha, 0.0, 2.0))
        if home_idx >= attack_sd.shape[1] or away_idx >= attack_sd.shape[1]:
            return float(np.clip(obs_alpha, 0.0, 2.0))
        home_attack_var = attack_sd[period_index, home_idx] ** 2
        home_defense_var = defense_sd[period_index, away_idx] ** 2
        away_attack_var = attack_sd[period_index, away_idx] ** 2
        away_defense_var = defense_sd[period_index, home_idx] ** 2
    elif attack_sd.ndim == 1 and defense_sd.ndim == 1:
        if home_idx >= attack_sd.size or away_idx >= attack_sd.size or home_idx >= defense_sd.size or away_idx >= defense_sd.size:
            return float(np.clip(obs_alpha, 0.0, 2.0))
        home_attack_var = attack_sd[home_idx] ** 2
        home_defense_var = defense_sd[away_idx] ** 2
        away_attack_var = attack_sd[away_idx] ** 2
        away_defense_var = defense_sd[home_idx] ** 2
    else:
        return float(np.clip(obs_alpha, 0.0, 2.0))
    neutral = _fixture_neutral(match)
    home_adv_sd = float(params.get("home_adv_sd", 0.0) or 0.0) ** 2
    neutral_adv_sd = float(params.get("neutral_adv_sd", 0.0) or 0.0) ** 2
    if neutral:
        home_margin_var = home_attack_var + home_defense_var + neutral_adv_sd
        away_margin_var = away_attack_var + away_defense_var + neutral_adv_sd
    else:
        home_margin_var = home_attack_var + home_defense_var + home_adv_sd
        away_margin_var = away_attack_var + away_defense_var
    alpha = float(obs_alpha + 0.5 * float(home_margin_var + away_margin_var))
    return float(np.clip(alpha, 0.0, 2.0))


def _bayesian_lambdas_for_match(
        state: ScoreModelState,
        home: str,
        away: str,
        match: Dict[str, Any] | None = None,
) -> Tuple[float, float] | None:
    if not state.available:
        return None
    params = state.params or {}
    teams = [str(team) for team in params.get("teams", [])]
    if not teams:
        return None
    try:
        home_idx = teams.index(str(home))
        away_idx = teams.index(str(away))
    except ValueError:
        return None
    attack = np.asarray(params.get("attack") or [], dtype=float)
    defense = np.asarray(params.get("defense") or [], dtype=float)
    try:
        intercept = float(params.get("intercept", math.log(1.2)))
        home_adv = float(params.get("home_adv", 0.0))
        neutral_adv = float(params.get("neutral_adv", 0.0))
        neutral = _fixture_neutral(match)
        home_not_neutral = 1.0 - float(neutral)
        if attack.ndim == 2 and defense.ndim == 2:
            period_index = _bayesian_match_period_index(params.get("periods"), match)
            if period_index < 0 or period_index >= attack.shape[0]:
                return None
            if home_idx >= attack.shape[1] or away_idx >= attack.shape[1]:
                return None
            attack_home = float(attack[period_index, home_idx])
            defense_away = float(defense[period_index, away_idx])
            attack_away = float(attack[period_index, away_idx])
            defense_home = float(defense[period_index, home_idx])
        else:
            if attack.ndim != 1 or defense.ndim != 1:
                return None
            if home_idx >= attack.size or away_idx >= attack.size or home_idx >= defense.size or away_idx >= defense.size:
                return None
            attack_home = float(attack[home_idx])
            defense_away = float(defense[away_idx])
            attack_away = float(attack[away_idx])
            defense_home = float(defense[home_idx])
        lambda_home = math.exp(
            intercept
            + home_adv * home_not_neutral
            + neutral_adv * float(neutral)
            + attack_home
            - defense_away
        )
        lambda_away = math.exp(
            intercept
            + neutral_adv * float(neutral)
            + attack_away
            - defense_home
        )
    except Exception:
        return None
    return _clamp_rate(lambda_home), _clamp_rate(lambda_away)


def _row_periods(rows: List[Dict[str, Any]]) -> List[int]:
    dates = pd.to_datetime([row.get("date", "") for row in rows], errors="coerce")
    periods: List[int] = []
    for index, value in enumerate(dates):
        if pd.isna(value):
            periods.append(int(index))
        else:
            periods.append(int(value.year * 100 + value.month))
    return periods


def _coalesce_model_period_indices(periods: List[int]) -> List[int]:
    unique_periods = sorted(set(int(period) for period in periods))
    if not unique_periods:
        return []
    lookup = {period: index for index, period in enumerate(unique_periods)}
    return [lookup[int(period)] for period in periods]


def _estimate_dixon_coles_rho(rows: List[Dict[str, Any]], backend: Any = "auto") -> Tuple[float, str, List[str]]:
    candidates = np.linspace(-0.24, 0.24, 121, dtype=float)
    try:
        value, backend_name, warnings = _estimate_dixon_coles_rho_batched(rows, candidates, backend=backend)
        step = float(candidates[1] - candidates[0])
        fine_candidates = np.linspace(max(-0.24, value - step), min(0.24, value + step), 81, dtype=float)
        value, backend_name, fine_warnings = _estimate_dixon_coles_rho_batched(rows, fine_candidates, backend=backend_name)
        return float(np.clip(value, -0.24, 0.24)), backend_name, _merge_warnings(warnings, fine_warnings)
    except Exception as exc:
        warnings = [f"MLE Dixon-Coles batched fallo ({exc.__class__.__name__}: {exc}); se usa optimizacion CPU."]

    def objective(rho: float) -> float:
        return -sum(
            float(row.get("weight", 1.0) or 1.0)
            * _grid_log_probability(dixon_coles_score_grid(row["lambda1"], row["lambda2"], rho=rho, max_goals=10), row)
            for row in rows
        )

    result = _minimize_scalar(objective, -0.24, 0.24)
    return float(np.clip(result, -0.24, 0.24)), "numpy", warnings


def _estimate_bivariate_corr_share(rows: List[Dict[str, Any]], backend: Any = "auto") -> Tuple[float, str, List[str]]:
    candidates = np.linspace(0.0, 0.55, 111, dtype=float)
    try:
        value, backend_name, warnings = _estimate_bivariate_corr_share_batched(rows, candidates, backend=backend)
        step = float(candidates[1] - candidates[0])
        fine_candidates = np.linspace(max(0.0, value - step), min(0.55, value + step), 81, dtype=float)
        value, backend_name, fine_warnings = _estimate_bivariate_corr_share_batched(rows, fine_candidates, backend=backend_name)
        return float(np.clip(value, 0.0, 0.55)), backend_name, _merge_warnings(warnings, fine_warnings)
    except Exception as exc:
        warnings = [f"MLE bivariado batched fallo ({exc.__class__.__name__}: {exc}); se usa optimizacion CPU."]

    def objective(share: float) -> float:
        return -sum(
            float(row.get("weight", 1.0) or 1.0)
            * _grid_log_probability(bivariate_poisson_score_grid(row["lambda1"], row["lambda2"], share, max_goals=10), row)
            for row in rows
        )

    return float(np.clip(_minimize_scalar(objective, 0.0, 0.55), 0.0, 0.55)), "numpy", warnings


def _estimate_dixon_coles_rho_batched(
        rows: List[Dict[str, Any]],
        candidates: np.ndarray,
        backend: Any,
) -> Tuple[float, str, List[str]]:
    row_arrays = _row_arrays_for_mle(rows, max_goals=10)
    xp, backend_name, warnings = _array_module_for_backend(backend)
    lambda1 = xp.asarray(row_arrays["lambda1"], dtype=xp.float64)[None, :]
    lambda2 = xp.asarray(row_arrays["lambda2"], dtype=xp.float64)[None, :]
    goals_home = xp.asarray(row_arrays["g1"], dtype=xp.int64)
    goals_away = xp.asarray(row_arrays["g2"], dtype=xp.int64)
    weights = xp.asarray(row_arrays["weight"], dtype=xp.float64)
    candidate_array = xp.asarray(candidates, dtype=xp.float64)[:, None]
    base_grids = _poisson_score_grids_xp(
        xp,
        xp.asarray(row_arrays["lambda1"], dtype=xp.float64),
        xp.asarray(row_arrays["lambda2"], dtype=xp.float64),
        max_goals=10,
    )
    row_index = xp.arange(int(goals_home.size), dtype=xp.int64)
    base_observed = base_grids[row_index, goals_home, goals_away][None, :]
    base_low = base_grids[:, :2, :2]
    factors = xp.ones((int(candidate_array.shape[0]), int(goals_home.size)), dtype=xp.float64)
    factors = xp.where((goals_home[None, :] == 0) & (goals_away[None, :] == 0), xp.maximum(1.0 - lambda1 * lambda2 * candidate_array, 1e-6), factors)
    factors = xp.where((goals_home[None, :] == 0) & (goals_away[None, :] == 1), xp.maximum(1.0 + lambda1 * candidate_array, 1e-6), factors)
    factors = xp.where((goals_home[None, :] == 1) & (goals_away[None, :] == 0), xp.maximum(1.0 + lambda2 * candidate_array, 1e-6), factors)
    factors = xp.where((goals_home[None, :] == 1) & (goals_away[None, :] == 1), xp.maximum(1.0 - candidate_array, 1e-6), factors)
    low_factors = xp.empty((int(candidate_array.shape[0]), int(goals_home.size), 2, 2), dtype=xp.float64)
    low_factors[:, :, 0, 0] = xp.maximum(1.0 - lambda1 * lambda2 * candidate_array, 1e-6)
    low_factors[:, :, 0, 1] = xp.maximum(1.0 + lambda1 * candidate_array, 1e-6)
    low_factors[:, :, 1, 0] = xp.maximum(1.0 + lambda2 * candidate_array, 1e-6)
    low_factors[:, :, 1, 1] = xp.maximum(1.0 - candidate_array, 1e-6)
    adjusted_totals = 1.0 + xp.sum(base_low[None, :, :, :] * (low_factors - 1.0), axis=(2, 3))
    probabilities = base_observed * factors / xp.maximum(adjusted_totals, 1e-12)
    likelihoods = xp.sum(weights[None, :] * xp.log(xp.maximum(probabilities, 1e-12)), axis=1)
    index = int(_scalar_from_xp(xp, xp.argmax(likelihoods)))
    return float(candidates[index]), backend_name, warnings


def _estimate_bivariate_corr_share_batched(
        rows: List[Dict[str, Any]],
        candidates: np.ndarray,
        backend: Any,
) -> Tuple[float, str, List[str]]:
    row_arrays = _row_arrays_for_mle(rows, max_goals=10)
    xp, backend_name, warnings = _array_module_for_backend(backend)
    lambda1 = xp.asarray(row_arrays["lambda1"], dtype=xp.float64)
    lambda2 = xp.asarray(row_arrays["lambda2"], dtype=xp.float64)
    goals_home = xp.asarray(row_arrays["g1"], dtype=xp.int64)
    goals_away = xp.asarray(row_arrays["g2"], dtype=xp.int64)
    weights = xp.asarray(row_arrays["weight"], dtype=xp.float64)
    row_index = xp.arange(int(goals_home.size), dtype=xp.int64)
    likelihoods = []
    chunk_size = 32
    for start in range(0, len(candidates), chunk_size):
        chunk = np.asarray(candidates[start:start + chunk_size], dtype=float)
        repeated_lambda1 = xp.tile(lambda1, int(chunk.size))
        repeated_lambda2 = xp.tile(lambda2, int(chunk.size))
        repeated_share = xp.repeat(xp.asarray(chunk, dtype=xp.float64), int(lambda1.size))
        grids = _bivariate_poisson_score_grids_xp(
            xp,
            repeated_lambda1,
            repeated_lambda2,
            corr_share=repeated_share,
            max_goals=10,
        ).reshape(int(chunk.size), int(lambda1.size), 11, 11)
        observed = grids[:, row_index, goals_home, goals_away]
        likelihoods.append(xp.sum(weights[None, :] * xp.log(xp.maximum(observed, 1e-12)), axis=1))
    all_likelihoods = xp.concatenate(likelihoods)
    index = int(_scalar_from_xp(xp, xp.argmax(all_likelihoods)))
    return float(candidates[index]), backend_name, warnings


def _row_arrays_for_mle(rows: List[Dict[str, Any]], max_goals: int) -> Dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Sin filas para ajuste MLE.")
    return {
        "lambda1": np.asarray([_clamp_rate(row["lambda1"]) for row in rows], dtype=float),
        "lambda2": np.asarray([_clamp_rate(row["lambda2"]) for row in rows], dtype=float),
        "g1": np.asarray([min(max(int(row["g1"]), 0), max_goals) for row in rows], dtype=int),
        "g2": np.asarray([min(max(int(row["g2"]), 0), max_goals) for row in rows], dtype=int),
        "weight": np.asarray([max(float(row.get("weight", 1.0) or 1.0), 1e-6) for row in rows], dtype=float),
    }


def _merge_warnings(*groups: Iterable[Any]) -> List[str]:
    seen = set()
    output: List[str] = []
    for group in groups:
        for item in group:
            text = str(item or "").strip()
            if text and text not in seen:
                seen.add(text)
                output.append(text)
    return output


def _array_module_for_backend(backend: Any) -> Tuple[Any, str, List[str]]:
    requested = _normalize_backend_request(backend)
    warnings: List[str] = []
    if requested in {"auto", "cuda", "cupy"}:
        status = score_backend_status("cuda" if requested in {"cuda", "cupy"} else "auto")
        if status.get("score_backend") == "cupy":
            try:
                return _import_cupy(), "cupy", warnings
            except Exception as exc:
                warnings.append(f"CuPy no disponible para MLE ({exc.__class__.__name__}: {exc}); se usa NumPy.")
        elif requested in {"cuda", "cupy"}:
            warnings.append(f"CuPy no disponible para MLE ({status.get('warning')}); se usa NumPy.")
    return np, "numpy", warnings


def _normalize_backend_request(value: Any) -> str:
    requested = str(value or "auto").strip().lower()
    if requested in {"gpu"}:
        return "cuda"
    if requested in {"cpu"}:
        return "numpy"
    if requested not in {"auto", "cuda", "cupy", "numpy"}:
        return "auto"
    return requested


def _cupy_backend_status() -> Dict[str, Any]:
    global _CUPY_BACKEND_STATUS
    if _CUPY_BACKEND_STATUS is not None:
        return dict(_CUPY_BACKEND_STATUS)
    try:
        cp = _import_cupy()
        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count <= 0:
            _CUPY_BACKEND_STATUS = {"available": False, "warning": "CuPy sin dispositivos CUDA", "device_names": []}
            return dict(_CUPY_BACKEND_STATUS)
        device_names: List[str] = []
        for index in range(device_count):
            try:
                props = cp.cuda.runtime.getDeviceProperties(index)
                raw_name = props.get("name", "") if isinstance(props, dict) else ""
                if isinstance(raw_name, bytes):
                    raw_name = raw_name.decode("utf-8", errors="ignore")
                device_names.append(str(raw_name or f"CUDA device {index}").strip())
            except Exception:
                device_names.append(f"CUDA device {index}")
        _CUPY_BACKEND_STATUS = {"available": True, "warning": "", "device_names": device_names}
        return dict(_CUPY_BACKEND_STATUS)
    except Exception as exc:
        _CUPY_BACKEND_STATUS = {"available": False, "warning": f"CuPy no disponible: {exc.__class__.__name__}: {exc}", "device_names": []}
        return dict(_CUPY_BACKEND_STATUS)


def _import_cupy() -> Any:
    import cupy as cp  # type: ignore

    return cp


def _scalar_from_xp(xp: Any, value: Any) -> float:
    if xp is np:
        return float(value)
    return float(value.get())


def _fit_xg_local_state(label: str, fingerprint: str) -> ScoreModelState:
    if not LOCAL_XG_FILE.exists():
        return ScoreModelState(
            key="xg_poisson_local",
            label=label,
            available=False,
            params={"path": str(LOCAL_XG_FILE)},
            warnings=("No existe storage/worldcup/xg/manual_xg.csv; se usa Poisson independiente.",),
            fingerprint=fingerprint,
        )
    try:
        frame = pd.read_csv(LOCAL_XG_FILE)
    except Exception as exc:
        return ScoreModelState(
            key="xg_poisson_local",
            label=label,
            available=False,
            params={"path": str(LOCAL_XG_FILE)},
            warnings=(f"No se pudo leer xG local: {exc}",),
            fingerprint=fingerprint,
        )
    mapping: Dict[str, Dict[str, float]] = {}
    for _, row in frame.iterrows():
        home = str(row.get("home") or row.get("Home") or row.get("Equipo 1") or "").strip()
        away = str(row.get("away") or row.get("Away") or row.get("Equipo 2") or "").strip()
        date = str(row.get("date") or row.get("Date") or row.get("Fecha") or "").strip()
        try:
            home_xg = float(row.get("home_xg", row.get("xg_home", row.get("xG Local"))))
            away_xg = float(row.get("away_xg", row.get("xg_away", row.get("xG Visita"))))
        except (TypeError, ValueError):
            continue
        if home and away and home_xg > 0 and away_xg > 0:
            mapping[_xg_key(home, away, date)] = {"home": float(home_xg), "away": float(away_xg)}
            mapping.setdefault(_xg_key(home, away, ""), {"home": float(home_xg), "away": float(away_xg)})
    return ScoreModelState(
        key="xg_poisson_local",
        label=label,
        available=bool(mapping),
        params={"path": str(LOCAL_XG_FILE), "matches": mapping},
        warnings=() if mapping else ("El CSV xG no contiene filas validas; se usa Poisson independiente.",),
        fingerprint=fingerprint,
    )


def _xg_lambdas_for_match(state: ScoreModelState, home: str, away: str, match: Dict[str, Any] | None) -> Tuple[float, float] | None:
    matches = (state.params or {}).get("matches") or {}
    date = ""
    if isinstance(match, dict):
        date = str(match.get("date") or match.get("Fecha") or "").strip()
    payload = matches.get(_xg_key(home, away, date)) or matches.get(_xg_key(home, away, ""))
    if not payload:
        return None
    try:
        return _clamp_rate(payload["home"]), _clamp_rate(payload["away"])
    except Exception:
        return None


def _xg_key(home: str, away: str, date: str) -> str:
    return f"{str(date).strip()}|{str(home).strip().lower()}|{str(away).strip().lower()}"


def _grid_log_probability(grid: np.ndarray, row: Dict[str, Any]) -> float:
    home = min(int(row["g1"]), grid.shape[0] - 1)
    away = min(int(row["g2"]), grid.shape[1] - 1)
    return math.log(max(float(grid[home, away]), 1e-12))


def _minimize_scalar(fn, lower: float, upper: float) -> float:
    try:
        from scipy import optimize

        result = optimize.minimize_scalar(fn, bounds=(lower, upper), method="bounded", options={"xatol": 1e-4})
        if result.success:
            return float(result.x)
    except Exception:
        pass
    candidates = np.linspace(lower, upper, 41)
    values = [float(fn(float(candidate))) for candidate in candidates]
    return float(candidates[int(np.argmin(values))])


def _fit_fingerprint(key: str, history_df: pd.DataFrame | None, teams: Iterable[str], config: Dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(normalize_score_model_key(key).encode("utf-8"))
    digest.update("|".join(sorted(str(team) for team in teams)).encode("utf-8"))
    relevant = {
        "history_weight": config.get("history_weight"),
        "recency_weight": config.get("recency_weight"),
        "host_advantage": config.get("host_advantage"),
        "max_goals": config.get("max_goals"),
        "bayes_draws": config.get("bayes_draws"),
        "bayes_tune": config.get("bayes_tune"),
        "bayes_chains": config.get("bayes_chains"),
        "bayes_target_accept": config.get("bayes_target_accept"),
        "bayes_max_treedepth": config.get("bayes_max_treedepth"),
        "score_backend_generation": "cupy-batched-v1",
        "sota_device": config.get("sota_device"),
        "score_mle_recency_weight": config.get("score_mle_recency_weight", config.get("recency_weight")),
        "stat_lambda_model": config.get("stat_lambda_model", STATSMODELS_POISSON_GLM_MODEL),
        "stat_glm_alphas": config.get("stat_glm_alphas"),
        "stat_glm_min_matches": config.get("stat_glm_min_matches"),
        "stat_glm_validation_fraction": config.get("stat_glm_validation_fraction"),
    }
    digest.update(json.dumps(relevant, sort_keys=True, default=str).encode("utf-8"))
    if history_df is not None and not history_df.empty:
        columns = [column for column in ("Date", "Team 1", "Team 2", "G1", "G2") if column in history_df.columns]
        digest.update(pd.util.hash_pandas_object(history_df[columns].astype(str), index=False).values.tobytes())
    if normalize_score_model_key(key) == "xg_poisson_local" and LOCAL_XG_FILE.exists():
        stat = LOCAL_XG_FILE.stat()
        digest.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8"))
    return digest.hexdigest()[:16]


def _state_cache_path(key: str, fingerprint: str) -> Path:
    return STAT_MODEL_ROOT / normalize_score_model_key(key) / f"{fingerprint}.json"


def _read_state(path: Path) -> ScoreModelState | None:
    try:
        if not path.exists():
            return None
        return ScoreModelState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def _write_state(path: Path, state: ScoreModelState) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        return


def _clamp_rate(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 1.0
    if not math.isfinite(number):
        number = 1.0
    return float(np.clip(number, 0.05, 6.5))


def _normalize_grid(grid: np.ndarray) -> np.ndarray:
    output = np.asarray(grid, dtype=float)
    output = np.maximum(np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total = float(output.sum())
    if total <= 0:
        return np.full_like(output, 1.0 / max(output.size, 1), dtype=float)
    return output / total


def _logsumexp(values: List[float]) -> float:
    if not values:
        return 0.0
    max_value = max(values)
    return math.exp(max_value) * sum(math.exp(value - max_value) for value in values)
