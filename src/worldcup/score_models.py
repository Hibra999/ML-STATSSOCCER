from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.worldcup.model import (
    TOTAL_GOAL_LINES,
    dixon_coles_score_grid,
    poisson_score_grid,
    total_line_suffix,
)


DEFAULT_SCORE_MODEL = "independent_poisson"
STAT_MODEL_ROOT = Path("storage") / "worldcup" / "stat_models"
LOCAL_XG_FILE = Path("storage") / "worldcup" / "xg" / "manual_xg.csv"

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
        "key": "bivariate_poisson_mle",
        "label": "Poisson bivariado MLE",
        "description": "Introduce un componente comun de goles para correlacion positiva entre equipos.",
        "heavy": False,
    },
    {
        "key": "diagonal_inflated_bivariate_poisson",
        "label": "Bivariado diagonal-inflado",
        "description": "Poisson bivariado con masa extra en empates para calibrar draws.",
        "heavy": False,
    },
    {
        "key": "zero_inflated_generalized_poisson",
        "label": "Poisson generalizado ZI",
        "description": "Marginales Poisson generalizadas con inflacion de ceros para sobredispersion y 0-0.",
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
        "key": "skellam_margin",
        "label": "Skellam margen",
        "description": "Repondera la matriz por distribucion de diferencia de goles.",
        "heavy": False,
    },
    {
        "key": "copula_weibull_count",
        "label": "Weibull + copula Frank",
        "description": "Marginales discrete-Weibull unidas por copula Frank experimental.",
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
            bayesian_lambdas = _bayesian_lambdas_for_match(self.state, team1, team2)
            if bayesian_lambdas is not None:
                return bayesian_lambdas
        method = getattr(self.base_model, "expected_goals_for_match", None)
        if callable(method):
            return method(team1, team2, match=match)
        return self.base_model.expected_goals(team1, team2)

    def score_grid(self, team1: str, team2: str, match: Dict[str, Any] | None = None, max_goals: int | None = None) -> np.ndarray:
        lambda1, lambda2 = self.expected_goals_for_match(team1, team2, match=match)
        return score_grid_from_lambdas(
            self.state,
            lambda1=lambda1,
            lambda2=lambda2,
            max_goals=int(max_goals if max_goals is not None else self.max_goals),
        )

    def score_grid_from_lambdas(self, lambda1: float, lambda2: float, max_goals: int | None = None) -> np.ndarray:
        return score_grid_from_lambdas(
            self.state,
            lambda1=lambda1,
            lambda2=lambda2,
            max_goals=int(max_goals if max_goals is not None else self.max_goals),
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
        grid = score_grid_from_lambdas(self.state, lambda1=lambda1, lambda2=lambda2, max_goals=limit_goals)
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

    rows = _history_model_rows(history_df, base_model)
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
    elif key == "dixon_coles_mle":
        state = ScoreModelState(key, label, True, {"rho": _estimate_dixon_coles_rho(rows)}, fingerprint=fingerprint)
    elif key == "bivariate_poisson_mle":
        state = ScoreModelState(key, label, True, {"corr_share": _estimate_bivariate_corr_share(rows)}, fingerprint=fingerprint)
    elif key == "diagonal_inflated_bivariate_poisson":
        corr_share, diagonal_boost = _estimate_diagonal_inflated_params(rows)
        state = ScoreModelState(
            key,
            label,
            True,
            {"corr_share": corr_share, "diagonal_boost": diagonal_boost},
            fingerprint=fingerprint,
        )
    elif key == "zero_inflated_generalized_poisson":
        state = ScoreModelState(key, label, True, _estimate_zigp_params(rows), fingerprint=fingerprint)
    elif key == "skellam_margin":
        state = ScoreModelState(key, label, True, _estimate_skellam_params(rows), fingerprint=fingerprint)
    elif key == "copula_weibull_count":
        state = ScoreModelState(key, label, True, _estimate_copula_weibull_params(rows), fingerprint=fingerprint)
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


def score_grid_from_lambdas(state: ScoreModelState | Dict[str, Any], lambda1: float, lambda2: float, max_goals: int = 10) -> np.ndarray:
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
    if key == "diagonal_inflated_bivariate_poisson":
        grid = bivariate_poisson_score_grid(lambda1, lambda2, float(params.get("corr_share", 0.0)), max_goals=max_goals)
        return diagonal_inflate_grid(grid, float(params.get("diagonal_boost", 1.0)))
    if key == "zero_inflated_generalized_poisson":
        return zero_inflated_generalized_poisson_grid(lambda1, lambda2, params, max_goals=max_goals)
    if key == "skellam_margin":
        return skellam_reweighted_grid(lambda1, lambda2, float(params.get("margin_scale", 1.0)), max_goals=max_goals)
    if key == "copula_weibull_count":
        return copula_weibull_score_grid(lambda1, lambda2, params, max_goals=max_goals)
    return poisson_score_grid(lambda1=lambda1, lambda2=lambda2, max_goals=max_goals)


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


def diagonal_inflate_grid(grid: np.ndarray, diagonal_boost: float) -> np.ndarray:
    adjusted = np.asarray(grid, dtype=float).copy()
    boost = float(np.clip(diagonal_boost, 0.05, 20.0))
    diagonal = np.diag_indices(min(adjusted.shape))
    adjusted[diagonal] *= boost
    return _normalize_grid(adjusted)


def zero_inflated_generalized_poisson_grid(lambda1: float, lambda2: float, params: Dict[str, Any], max_goals: int = 10) -> np.ndarray:
    home = generalized_poisson_pmf_vector(
        lambda1,
        alpha=float(params.get("alpha_home", params.get("alpha", 0.0))),
        zero_inflation=float(params.get("zero_home", params.get("zero_inflation", 0.0))),
        max_goals=max_goals,
    )
    away = generalized_poisson_pmf_vector(
        lambda2,
        alpha=float(params.get("alpha_away", params.get("alpha", 0.0))),
        zero_inflation=float(params.get("zero_away", params.get("zero_inflation", 0.0))),
        max_goals=max_goals,
    )
    return _normalize_grid(np.outer(home, away))


def generalized_poisson_pmf_vector(rate: float, alpha: float, zero_inflation: float, max_goals: int = 10) -> np.ndarray:
    rate = _clamp_rate(rate)
    alpha = float(np.clip(alpha, -0.45, 0.95))
    zero_inflation = float(np.clip(zero_inflation, 0.0, 0.75))
    probs = np.zeros(int(max_goals) + 1, dtype=float)
    for goals in range(int(max_goals) + 1):
        term = rate + alpha * goals
        if goals == 0:
            gp = math.exp(-rate)
        elif term <= 0:
            gp = 0.0
        else:
            gp = rate * (term ** (goals - 1)) * math.exp(-term) / math.factorial(goals)
        probs[goals] = (1.0 - zero_inflation) * gp
    probs[0] += zero_inflation
    return _normalize_vector(probs)


def skellam_reweighted_grid(lambda1: float, lambda2: float, margin_scale: float, max_goals: int = 10) -> np.ndarray:
    grid = poisson_score_grid(lambda1, lambda2, max_goals=max_goals)
    margin_scale = float(np.clip(margin_scale, 0.45, 2.5))
    if abs(margin_scale - 1.0) < 1e-6:
        return grid
    goals = np.arange(grid.shape[0], dtype=int)
    home_goals, away_goals = np.meshgrid(goals, goals, indexing="ij")
    margins = np.abs(home_goals - away_goals)
    weights = np.power(margin_scale, margins.astype(float))
    return _normalize_grid(grid * weights)


def copula_weibull_score_grid(lambda1: float, lambda2: float, params: Dict[str, Any], max_goals: int = 10) -> np.ndarray:
    beta_home = float(params.get("beta_home", params.get("beta", 1.15)))
    beta_away = float(params.get("beta_away", params.get("beta", 1.15)))
    theta = float(np.clip(params.get("theta", 0.0), -12.0, 12.0))
    home_pmf = discrete_weibull_pmf(lambda1, beta_home, max_goals=max_goals)
    away_pmf = discrete_weibull_pmf(lambda2, beta_away, max_goals=max_goals)
    home_cdf = np.cumsum(home_pmf)
    away_cdf = np.cumsum(away_pmf)
    grid = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
    for home_goals in range(max_goals + 1):
        u1 = float(home_cdf[home_goals])
        u0 = float(home_cdf[home_goals - 1]) if home_goals else 0.0
        for away_goals in range(max_goals + 1):
            v1 = float(away_cdf[away_goals])
            v0 = float(away_cdf[away_goals - 1]) if away_goals else 0.0
            grid[home_goals, away_goals] = (
                frank_copula_cdf(u1, v1, theta)
                - frank_copula_cdf(u0, v1, theta)
                - frank_copula_cdf(u1, v0, theta)
                + frank_copula_cdf(u0, v0, theta)
            )
    return _normalize_grid(np.maximum(grid, 0.0))


def discrete_weibull_pmf(mean: float, beta: float, max_goals: int = 10) -> np.ndarray:
    beta = float(np.clip(beta, 0.45, 3.5))
    mean = _clamp_rate(mean)
    q = _discrete_weibull_q_for_mean(mean, beta, max_goals=max(40, max_goals * 8))
    goals = np.arange(max_goals + 1, dtype=float)
    probs = np.power(q, np.power(goals, beta)) - np.power(q, np.power(goals + 1.0, beta))
    probs[-1] += max(0.0, 1.0 - float(probs.sum()))
    return _normalize_vector(probs)


def frank_copula_cdf(u: float, v: float, theta: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    theta = float(theta)
    if abs(theta) < 1e-7:
        return u * v
    denominator = math.expm1(-theta)
    if abs(denominator) < 1e-12:
        return u * v
    inner = 1.0 + (math.expm1(-theta * u) * math.expm1(-theta * v)) / denominator
    return float(np.clip(-math.log(max(inner, 1e-12)) / theta, 0.0, 1.0))


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
    draws = int(config.get("bayes_draws") or 500)
    tune = int(config.get("bayes_tune") or 500)
    chains = int(config.get("bayes_chains") or 2)
    seed = int(config.get("seed") or 2026)
    if key == "bayesian_dynamic_poisson":
        periods = _row_periods(rows)
        period_list = sorted(set(periods))
        period_index = {period: index for index, period in enumerate(period_list)}
        period_idx = np.asarray([period_index[period] for period in periods], dtype=int)
        with pm.Model(coords={"period": period_list, "team": team_list}):
            attack_raw = pm.GaussianRandomWalk(
                "attack_raw",
                sigma=0.18,
                init_dist=pm.Normal.dist(0.0, 0.35),
                dims=("period", "team"),
            )
            defense_raw = pm.GaussianRandomWalk(
                "defense_raw",
                sigma=0.18,
                init_dist=pm.Normal.dist(0.0, 0.35),
                dims=("period", "team"),
            )
            attack = pm.Deterministic("attack", attack_raw - pm.math.mean(attack_raw, axis=1, keepdims=True), dims=("period", "team"))
            defense = pm.Deterministic("defense", defense_raw - pm.math.mean(defense_raw, axis=1, keepdims=True), dims=("period", "team"))
            intercept = pm.Normal("intercept", math.log(1.2), 0.45)
            home_adv = pm.Normal("home_adv", 0.0, 0.25)
            home_rate = pm.math.exp(intercept + home_adv + attack[period_idx, home_idx] - defense[period_idx, away_idx])
            away_rate = pm.math.exp(intercept + attack[period_idx, away_idx] - defense[period_idx, home_idx])
            pm.Poisson("home_goals", mu=home_rate, observed=home_goals)
            pm.Poisson("away_goals", mu=away_rate, observed=away_goals)
            trace = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=False)
        posterior = trace.posterior
        latest_period = -1
        params = {
            "teams": team_list,
            "periods": period_list,
            "attack": np.asarray(posterior["attack"].isel(period=latest_period).mean(dim=("chain", "draw"))).astype(float).tolist(),
            "defense": np.asarray(posterior["defense"].isel(period=latest_period).mean(dim=("chain", "draw"))).astype(float).tolist(),
            "intercept": float(np.asarray(posterior["intercept"].mean(dim=("chain", "draw")))),
            "home_adv": float(np.asarray(posterior["home_adv"].mean(dim=("chain", "draw")))),
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "dynamic_period": period_list[latest_period],
        }
        return ScoreModelState(key=key, label=label, available=True, params=params, fingerprint=fingerprint)
    with pm.Model(coords={"team": team_list}):
        attack_raw = pm.Normal("attack_raw", 0.0, 0.45, dims="team")
        defense_raw = pm.Normal("defense_raw", 0.0, 0.45, dims="team")
        attack = pm.Deterministic("attack", attack_raw - pm.math.mean(attack_raw), dims="team")
        defense = pm.Deterministic("defense", defense_raw - pm.math.mean(defense_raw), dims="team")
        intercept = pm.Normal("intercept", math.log(1.2), 0.45)
        home_adv = pm.Normal("home_adv", 0.0, 0.25)
        home_rate = pm.math.exp(intercept + home_adv + attack[home_idx] - defense[away_idx])
        away_rate = pm.math.exp(intercept + attack[away_idx] - defense[home_idx])
        pm.Poisson("home_goals", mu=home_rate, observed=home_goals)
        pm.Poisson("away_goals", mu=away_rate, observed=away_goals)
        trace = pm.sample(draws=draws, tune=tune, chains=chains, random_seed=seed, progressbar=False)
    posterior = trace.posterior
    params = {
        "teams": team_list,
        "attack": np.asarray(posterior["attack"].mean(dim=("chain", "draw"))).astype(float).tolist(),
        "defense": np.asarray(posterior["defense"].mean(dim=("chain", "draw"))).astype(float).tolist(),
        "intercept": float(np.asarray(posterior["intercept"].mean(dim=("chain", "draw")))),
        "home_adv": float(np.asarray(posterior["home_adv"].mean(dim=("chain", "draw")))),
        "draws": draws,
        "tune": tune,
        "chains": chains,
    }
    return ScoreModelState(key=key, label=label, available=True, params=params, fingerprint=fingerprint)


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


def _history_model_rows(history_df: pd.DataFrame | None, base_model: Any) -> List[Dict[str, Any]]:
    if history_df is None or history_df.empty:
        return []
    required = {"Team 1", "Team 2", "G1", "G2"}
    if not required.issubset(history_df.columns):
        return []
    rows: List[Dict[str, Any]] = []
    working = history_df.copy()
    if "Date" in working:
        working = working.sort_values("Date", kind="stable")
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
        })
    return rows


def _bayesian_lambdas_for_match(state: ScoreModelState, home: str, away: str) -> Tuple[float, float] | None:
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
    attack = params.get("attack") or []
    defense = params.get("defense") or []
    try:
        intercept = float(params.get("intercept", math.log(1.2)))
        home_adv = float(params.get("home_adv", 0.0))
        lambda_home = math.exp(intercept + home_adv + float(attack[home_idx]) - float(defense[away_idx]))
        lambda_away = math.exp(intercept + float(attack[away_idx]) - float(defense[home_idx]))
    except Exception:
        return None
    return _clamp_rate(lambda_home), _clamp_rate(lambda_away)


def _row_periods(rows: List[Dict[str, Any]]) -> List[int]:
    dates = pd.to_datetime([row.get("date", "") for row in rows], errors="coerce")
    periods: List[int] = []
    for index, value in enumerate(dates):
        if pd.isna(value):
            periods.append(index)
        else:
            periods.append(int(value.year))
    return periods


def _estimate_dixon_coles_rho(rows: List[Dict[str, Any]]) -> float:
    def objective(rho: float) -> float:
        return -sum(_grid_log_probability(dixon_coles_score_grid(row["lambda1"], row["lambda2"], rho=rho, max_goals=10), row) for row in rows)

    result = _minimize_scalar(objective, -0.24, 0.24)
    return float(np.clip(result, -0.24, 0.24))


def _estimate_bivariate_corr_share(rows: List[Dict[str, Any]]) -> float:
    def objective(share: float) -> float:
        return -sum(_grid_log_probability(bivariate_poisson_score_grid(row["lambda1"], row["lambda2"], share, max_goals=10), row) for row in rows)

    return float(np.clip(_minimize_scalar(objective, 0.0, 0.55), 0.0, 0.55))


def _estimate_diagonal_inflated_params(rows: List[Dict[str, Any]]) -> Tuple[float, float]:
    try:
        from scipy import optimize

        def objective(values: np.ndarray) -> float:
            share = float(np.clip(values[0], 0.0, 0.55))
            boost = float(math.exp(np.clip(values[1], -2.5, 3.0)))
            total = 0.0
            for row in rows:
                grid = bivariate_poisson_score_grid(row["lambda1"], row["lambda2"], share, max_goals=10)
                total -= _grid_log_probability(diagonal_inflate_grid(grid, boost), row)
            return total

        result = optimize.minimize(objective, np.asarray([0.05, math.log(1.35)]), bounds=[(0.0, 0.55), (-2.5, 3.0)], method="L-BFGS-B")
        if result.success:
            return float(result.x[0]), float(math.exp(result.x[1]))
    except Exception:
        pass
    share = _estimate_bivariate_corr_share(rows)
    draws = sum(1 for row in rows if row["g1"] == row["g2"])
    base_draw = np.mean([
        bivariate_poisson_score_grid(row["lambda1"], row["lambda2"], share, max_goals=10).diagonal().sum()
        for row in rows
    ])
    observed_draw = draws / max(len(rows), 1)
    boost = float(np.clip(observed_draw / max(base_draw, 1e-9), 0.4, 8.0))
    return share, boost


def _estimate_zigp_params(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    try:
        from scipy import optimize

        def objective(values: np.ndarray) -> float:
            params = {
                "alpha_home": float(values[0]),
                "alpha_away": float(values[1]),
                "zero_home": float(values[2]),
                "zero_away": float(values[3]),
            }
            total = 0.0
            for row in rows:
                grid = zero_inflated_generalized_poisson_grid(row["lambda1"], row["lambda2"], params, max_goals=10)
                total -= _grid_log_probability(grid, row)
            return total

        result = optimize.minimize(
            objective,
            np.asarray([0.05, 0.05, 0.04, 0.04]),
            bounds=[(-0.35, 0.75), (-0.35, 0.75), (0.0, 0.45), (0.0, 0.45)],
            method="L-BFGS-B",
        )
        if result.success:
            return {
                "alpha_home": float(result.x[0]),
                "alpha_away": float(result.x[1]),
                "zero_home": float(result.x[2]),
                "zero_away": float(result.x[3]),
            }
    except Exception:
        pass
    zero_home = sum(1 for row in rows if row["g1"] == 0) / max(len(rows), 1)
    zero_away = sum(1 for row in rows if row["g2"] == 0) / max(len(rows), 1)
    return {
        "alpha_home": 0.05,
        "alpha_away": 0.05,
        "zero_home": float(np.clip(zero_home * 0.2, 0.0, 0.25)),
        "zero_away": float(np.clip(zero_away * 0.2, 0.0, 0.25)),
    }


def _estimate_skellam_params(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    margins = np.asarray([abs(row["g1"] - row["g2"]) for row in rows], dtype=float)
    expected = np.asarray([
        abs(row["lambda1"] - row["lambda2"]) + math.sqrt(max(row["lambda1"] + row["lambda2"], 1e-9)) * 0.55
        for row in rows
    ], dtype=float)
    scale = float(np.clip((margins.mean() + 1e-9) / max(float(expected.mean()), 1e-9), 0.55, 1.9))
    return {"margin_scale": scale}


def _estimate_copula_weibull_params(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    beta = 1.12

    def objective(theta: float) -> float:
        return -sum(
            _grid_log_probability(
                copula_weibull_score_grid(row["lambda1"], row["lambda2"], {"beta": beta, "theta": theta}, max_goals=10),
                row,
            )
            for row in rows
        )

    theta = float(np.clip(_minimize_scalar(objective, -8.0, 8.0), -8.0, 8.0))
    return {"beta": beta, "theta": theta}


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


def _discrete_weibull_q_for_mean(mean: float, beta: float, max_goals: int) -> float:
    def expected(q_value: float) -> float:
        goals = np.arange(max_goals + 1, dtype=float)
        probs = np.power(q_value, np.power(goals, beta)) - np.power(q_value, np.power(goals + 1.0, beta))
        probs[-1] += max(0.0, 1.0 - float(probs.sum()))
        return float(np.sum(goals * _normalize_vector(probs)))

    low = 1e-6
    high = 0.999999
    for _ in range(48):
        mid = (low + high) / 2.0
        if expected(mid) < mean:
            low = mid
        else:
            high = mid
    return float((low + high) / 2.0)


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


def _normalize_vector(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float)
    output = np.maximum(np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    total = float(output.sum())
    if total <= 0:
        fallback = np.zeros_like(output)
        fallback[0] = 1.0
        return fallback
    return output / total


def _logsumexp(values: List[float]) -> float:
    if not values:
        return 0.0
    max_value = max(values)
    return math.exp(max_value) * sum(math.exp(value - max_value) for value in values)
