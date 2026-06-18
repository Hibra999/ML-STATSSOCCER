from __future__ import annotations

import csv
import hashlib
import html as html_lib
import io
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.worldcup import (
    WorldCupModel,
    groups_dataframe,
    lineups_table,
    load_historical_matches,
    load_players,
    load_tournament_2026,
    simulate_worldcup,
    teams_dataframe,
    tournament_fixtures_dataframe,
)
from src.worldcup.model import TOTAL_GOAL_LINES, poisson_score_grid, total_line_suffix
from src.worldcup.score_models import (
    DEFAULT_SCORE_MODEL,
    DYNAMIC_STRENGTH_KALMAN_MODEL,
    NEGATIVE_BINOMIAL_DIXON_COLES_MODEL,
    NEGATIVE_BINOMIAL_GLM_MODEL,
    STACKED_META_MNLOGIT_MODEL,
    STATSMODELS_POISSON_GLM_MODEL,
    XG_DIXON_COLES_MODEL,
    build_score_model,
    normalize_score_model_key,
    probabilities_from_score_grid,
    probabilities_from_score_grids,
    sample_scores_from_grid,
    score_backend_status,
    score_grids_from_lambdas_with_backend,
    score_model_options,
)
from src.worldcup.advanced_data import (
    advanced_data_status as worldcup_advanced_data_status,
    prepare_advanced_data as prepare_worldcup_advanced_data,
)
from src.worldcup.sota_alternatives import (
    ALTERNATIVE_SCORE_MODEL_KEYS,
    ALTERNATIVES_BENCHMARK_LABEL,
    ALTERNATIVES_BENCHMARK_PIPELINE_MODE,
    ALTERNATIVES_EVIDENCE_POLICY,
    sota_alternatives_catalog,
    sota_baseline_context,
)
from src.worldcup.statistical_audit import build_prediction_statistical_audit
from src.worldcup.data import (
    CACHE_ROOT,
    fixture_results_status,
    group_letter,
    groups_from_tournament,
    refresh_worldcup_2026_results,
    team_name_key,
)
from src.worldcup.api_football_provider import api_football_feature_table, load_api_football_data
from src.worldcup.international_provider import (
    INTERNATIONAL_ROOT,
    INTERNATIONAL_RECENT_START_DATE,
    contextual_poisson_for_match,
    international_cutoff_timestamp,
    international_results_status,
    is_friendly_tournament,
    load_international_matches,
    recent15_feature_table,
    recent_match_importance_label,
    recent_match_recency_weight,
    tournament_weight,
)
from src.worldcup.market_provider import load_market_data, normalize_market_frame, qualifier_feature_table
from src.worldcup.training import (
    HISTORY_REFERENCE_DATE,
    TARGET_WORLDCUP_YEAR,
    TRAINING_OBJECTIVES,
    XG_LIGHTGBM_PROFILE,
    FEATURE_SELECTION_FAMILY_BALANCED,
    FEATURE_SELECTION_SUPERVISED_MODEL,
    build_history_feature_table,
    build_matchup_feature_table,
    dataset_status as worldcup_training_dataset_status,
    default_model_id,
    filter_training_scope_sources,
    international_history_rows,
    json_safe,
    list_worldcup_models,
    match_feature_row,
    prepare_training_dataset,
    predict_match_payload,
    read_model_metadata,
    train_hybrid_model,
    training_options,
)
from src.worldcup.lanus_provider import (
    LINEUPS_ROOT,
    PLAYER_STATS_ROOT,
    SOFASCORE_ROOT,
    auto_refresh_lineups,
    autodetect_fixture_event,
    link_fixture_lineup,
    lineup_payload_for_fixture,
    lineup_payload_from_detected_event,
    lineup_rating_adjustments,
    lineups_summary,
    player_feature_rating_adjustments,
    player_features_dataframe,
    player_stats_payload_for_fixture,
    sofa_player_photo_url,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COUNTRY_FLAGS_ROOT = PROJECT_ROOT / "storage" / "graphics" / "countries"
REPORTS_ROOT = Path("storage") / "worldcup" / "reports"
FEATURE_STORE_ROOT = Path("storage") / "worldcup" / "features"
WALK_FORWARD_ROOT = Path("storage") / "worldcup" / "walk_forward"
POISSON_SOTA_PIPELINE_MODE = "poisson_sota"
XG_LIGHTGBM_PIPELINE_MODE = "xg_lightgbm"
XG_LIGHTGBM_PIPELINE_LABEL = "xG-LightGBM"
ADVANCED_MODELS_PIPELINE_MODE = "advanced_models"
ADVANCED_MODELS_PIPELINE_LABEL = "Modelos avanzados"
SOTA_SCORE_MODEL_SEQUENCE = [
    "independent_poisson",
    STATSMODELS_POISSON_GLM_MODEL,
    NEGATIVE_BINOMIAL_GLM_MODEL,
    "dixon_coles_mle",
    "bivariate_poisson_mle",
]
ADVANCED_SCORE_MODEL_SEQUENCE = [
    XG_DIXON_COLES_MODEL,
    NEGATIVE_BINOMIAL_DIXON_COLES_MODEL,
    DYNAMIC_STRENGTH_KALMAN_MODEL,
    STACKED_META_MNLOGIT_MODEL,
]
ADVANCED_HEAVY_SCORE_MODEL_SEQUENCE = [
    "bayesian_dynamic_poisson",
]
ALTERNATIVE_SCORE_MODEL_SEQUENCE = list(ALTERNATIVE_SCORE_MODEL_KEYS)
BENCHMARK_SCORE_MODEL_SEQUENCE = list(dict.fromkeys([*SOTA_SCORE_MODEL_SEQUENCE, *ALTERNATIVE_SCORE_MODEL_KEYS]))
REPORT_TOTAL_GOAL_LINES = (0.5, 1.5, 2.5, 3.5)
REPORT_SCORE_MATRIX_GOALS = 6
REPORT_MAX_ITERATIONS = 100_000
REPORT_DOWNLOAD_KINDS = {"predictions", "backtest"}
REPORT_DOWNLOAD_FORMATS = {"html", "csv"}
OUTCOME_KEYS = ("home", "draw", "away")
CALIBRATION_BIN_COUNT = 10
DEFAULT_CONFIG = {
    "iterations": 5000,
    "seed": 2026,
    "history_weight": 1.0,
    "recency_weight": 0.35,
    "host_advantage": 45.0,
    "max_goals": 10,
    "poisson_recent_matches": 15,
    "score_model": DEFAULT_SCORE_MODEL,
    "stat_model_cache": True,
    "stat_model_refit": False,
    "stat_lambda_model": STATSMODELS_POISSON_GLM_MODEL,
    "stat_glm_min_matches": 12,
    "stat_glm_validation_fraction": 0.2,
    "score_mle_recency_weight": None,
    "bayes_draws": 500,
    "bayes_tune": 500,
    "bayes_chains": 2,
    "bayes_target_accept": 0.92,
    "bayes_max_treedepth": 12,
    "advanced_include_bayesian": False,
    "refresh": False,
}
LAST_SIMULATION_RESULT: Dict[str, Any] = {}
_MONTE_CARLO_CUDA_BACKEND: Tuple[str, str] | None = None


class RecentPoissonWorldCupModel:
    def __init__(self, base_model: WorldCupModel, recent_match_limit: int = 15):
        self.base_model = base_model
        self.recent_match_limit = int(min(50, max(3, int(recent_match_limit or 15))))
        self.max_goals = int(getattr(base_model, "max_goals", DEFAULT_CONFIG["max_goals"]))
        self.matches = load_international_matches(required=False)

    def profile(self, team: str):
        return self.base_model.profile(team)

    def adjusted(self, rating_adjustments: Dict[str, float]):
        return RecentPoissonWorldCupModel(
            base_model=self.base_model.adjusted(rating_adjustments),
            recent_match_limit=self.recent_match_limit,
        )

    def expected_goals(self, team1: str, team2: str):
        return self.expected_goals_for_match(team1, team2, match=None)

    def expected_goals_for_match(self, team1: str, team2: str, match: Dict[str, Any] | None = None):
        probabilities = self.match_probabilities_for_match(team1, team2, match=match)
        return float(probabilities.get("lambda1", 1.0)), float(probabilities.get("lambda2", 1.0))

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
        context = contextual_poisson_for_match(
            team1,
            team2,
            base_model=self.base_model,
            before_date=_match_before_date(match),
            max_goals=limit_goals,
            matches=self.matches,
            limit=self.recent_match_limit,
        )
        if context.get("matrix_available"):
            lambda1 = float(context.get("context_lambda_home") or (context.get("lambdas") or {}).get("home") or 0.0)
            lambda2 = float(context.get("context_lambda_away") or (context.get("lambdas") or {}).get("away") or 0.0)
            if lambda1 > 0 and lambda2 > 0:
                probabilities = poisson_probabilities_from_lambdas(lambda1, lambda2, max_goals=limit_goals)
                probabilities["recent_match_limit"] = self.recent_match_limit
                probabilities["recent_matrix_available"] = True
                return probabilities
        probabilities = self.base_model.match_probabilities(team1, team2, max_goals=limit_goals)
        probabilities["recent_match_limit"] = self.recent_match_limit
        probabilities["recent_matrix_available"] = False
        return probabilities

    def sample_score(self, team1: str, team2: str, rng: np.random.Generator):
        lambda1, lambda2 = self.expected_goals(team1, team2)
        return int(rng.poisson(lambda1)), int(rng.poisson(lambda2))

    def sample_knockout_winner(self, team1: str, team2: str, rng: np.random.Generator):
        probabilities = self.match_probabilities(team1, team2)
        goals1 = int(rng.poisson(float(probabilities.get("lambda1", 1.0))))
        goals2 = int(rng.poisson(float(probabilities.get("lambda2", 1.0))))
        if goals1 > goals2:
            return team1, team2, goals1, goals2
        if goals2 > goals1:
            return team2, team1, goals1, goals2
        win_share = float(probabilities.get("home", 0.0)) / max(
            float(probabilities.get("home", 0.0)) + float(probabilities.get("away", 0.0)),
            1e-9,
        )
        if rng.random() <= win_share:
            return team1, team2, goals1, goals2
        return team2, team1, goals1, goals2


def apply_recent_context_model(model: Any, config: Dict[str, Any]) -> Any:
    if isinstance(model, RecentPoissonWorldCupModel):
        return model
    return RecentPoissonWorldCupModel(
        base_model=model,
        recent_match_limit=int(config.get("poisson_recent_matches") or DEFAULT_CONFIG["poisson_recent_matches"]),
    )


def _match_before_date(match: Dict[str, Any] | None) -> Any:
    if not isinstance(match, dict):
        return None
    return match.get("date") or match.get("Fecha")


def poisson_probabilities_from_lambdas(lambda1: float, lambda2: float, max_goals: int) -> Dict[str, float]:
    probs1 = [_poisson_pmf(goals, lambda1) for goals in range(max_goals + 1)]
    probs2 = [_poisson_pmf(goals, lambda2) for goals in range(max_goals + 1)]
    total = 0.0
    home = 0.0
    draw = 0.0
    away = 0.0
    totals = {line: 0.0 for line in TOTAL_GOAL_LINES}
    modal_score = (0, 0)
    modal_prob = -1.0
    for goals1, prob1 in enumerate(probs1):
        for goals2, prob2 in enumerate(probs2):
            prob = prob1 * prob2
            total += prob
            if goals1 > goals2:
                home += prob
            elif goals1 == goals2:
                draw += prob
            else:
                away += prob
            total_goals = goals1 + goals2
            for line in totals:
                if total_goals > line:
                    totals[line] += prob
            if prob > modal_prob:
                modal_prob = prob
                modal_score = (goals1, goals2)
    total = max(total, 1e-9)
    output = {
        "lambda1": float(lambda1),
        "lambda2": float(lambda2),
        "home": home / total,
        "draw": draw / total,
        "away": away / total,
        "modal_g1": modal_score[0],
        "modal_g2": modal_score[1],
    }
    for line, over_prob in totals.items():
        suffix = total_line_suffix(line)
        output[f"over{suffix}"] = over_prob / total
        output[f"under{suffix}"] = 1.0 - output[f"over{suffix}"]
    return output


def _poisson_pmf(goals: int, rate: float) -> float:
    safe_rate = max(float(rate), 1e-9)
    return math.exp(-safe_rate) * (safe_rate ** int(goals)) / math.factorial(int(goals))


class BenchmarkFeatureSource:
    def __init__(self, tournament: Dict[str, Any], history_df: pd.DataFrame | None, config: Dict[str, Any]):
        self.tournament = tournament or {}
        self.history_df = normalize_feature_history(history_df)
        self.config = dict(config or {})
        self.warnings: List[str] = []
        self.market_rows = pd.DataFrame()
        self.qualifier_rows = pd.DataFrame()
        self.api_football: Dict[str, pd.DataFrame] = {}
        self.international_matches = pd.DataFrame()
        self.fixture_feature_rows = pd.DataFrame()
        self.context_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self._load_cached_sources()

    def _load_cached_sources(self) -> None:
        try:
            market = load_market_data(allow_download=False, force_download=False, use_scraper=False)
            self.market_rows = market.get("matches", pd.DataFrame()).copy()
            self.qualifier_rows = market.get("qualifiers", pd.DataFrame()).copy()
            self.warnings.extend(str(item) for item in market.get("warnings", []) if str(item))
        except Exception as exc:
            self.warnings.append(f"Odds cache no disponible para features ({exc.__class__.__name__}).")
        try:
            api_bundle = load_api_football_data(allow_download=False, force_download=False)
            self.api_football = {
                key: api_bundle.get(key, pd.DataFrame()).copy()
                for key in ("team_stats", "lineups", "injuries", "market_rows")
            }
            api_market = self.api_football.get("market_rows", pd.DataFrame())
            if api_market is not None and not api_market.empty:
                self.market_rows = normalize_market_frame(pd.concat([self.market_rows, api_market], ignore_index=True))
            self.warnings.extend(str(item) for item in api_bundle.get("warnings", []) if str(item))
        except Exception as exc:
            self.warnings.append(f"API-Football cache no disponible para features ({exc.__class__.__name__}).")
        try:
            self.international_matches = load_international_matches(required=False)
            scope_teams = [team for group_teams in groups_from_tournament(self.tournament).values() for team in group_teams]
            self.international_matches = filter_training_scope_sources(
                self.international_matches,
                scope_teams,
                keep_unknown_date=False,
                keep_without_team_columns=False,
            )
        except Exception as exc:
            self.warnings.append(f"all_matches.csv no disponible para features recientes ({exc.__class__.__name__}).")
        try:
            self.fixture_feature_rows = player_features_dataframe(self.tournament)
        except Exception as exc:
            self.warnings.append(f"Cache XI/jugadores no disponible para features ({exc.__class__.__name__}).")

    def context_for_match(
            self,
            model: Any,
            fixture: pd.Series | Dict[str, Any],
            model_key: str,
            history_df: pd.DataFrame | None = None,
    ) -> Dict[str, Any]:
        record = fixture.to_dict() if hasattr(fixture, "to_dict") else dict(fixture or {})
        home = str(record.get("Equipo 1", record.get("Team 1", record.get("home", ""))) or "")
        away = str(record.get("Equipo 2", record.get("Team 2", record.get("away", ""))) or "")
        fixture_id = record.get("No.", record.get("FixtureId", record.get("id", "")))
        match_date = record.get("Fecha", record.get("Date", record.get("date", "")))
        date_ts = pd.to_datetime(match_date, errors="coerce")
        reference_date = str(date_ts.date()) if pd.notna(date_ts) else HISTORY_REFERENCE_DATE
        match_year = int(date_ts.year) if pd.notna(date_ts) else int(record.get("Year", 0) or TARGET_WORLDCUP_YEAR)
        history_source = normalize_feature_history(history_df if history_df is not None else self.history_df)
        cache_key = (
            str(model_key),
            str(fixture_id),
            str(reference_date),
            str(home),
            str(away),
            dataframe_fingerprint_for_report(history_source),
            id(model),
        )
        cached = self.context_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            history_cutoff = history_before_feature_date(history_source, date_ts)
            teams = sorted({team for team in [home, away, *teams_from_feature_history(history_cutoff)] if str(team).strip()})
            history_features = build_history_feature_table(history_cutoff, reference_date=reference_date)
            matchup_features = build_matchup_feature_table(history_cutoff, reference_date=reference_date)
            qualifier_features = qualifier_feature_table(self.qualifier_rows, reference_date=reference_date, teams=teams)
            api_features = api_football_feature_table(
                self.api_football.get("team_stats", pd.DataFrame()),
                reference_date=reference_date,
                teams=teams,
                lineups=self.api_football.get("lineups", pd.DataFrame()),
                injuries=self.api_football.get("injuries", pd.DataFrame()),
            )
            recent_features = recent15_feature_table(
                self.international_matches,
                teams=teams,
                before_date=reference_date,
                base_model=model,
            )
            date_for_current_fixture = pd.Timestamp(date_ts).tz_localize(None) if pd.notna(date_ts) else pd.NaT
            xi_features = safe_fixture_feature_rows_asof(
                self.fixture_feature_rows,
                fixture_id=fixture_id,
                reference_date=reference_date,
                allow_current_fixture=pd.notna(date_for_current_fixture)
                and date_for_current_fixture.normalize() >= pd.Timestamp(_now_utc().date()),
            )
            feature_row = match_feature_row(
                model,
                pd.DataFrame(),
                home,
                away,
                history_team_features=history_features,
                matchup_features=matchup_features,
                market_rows=self.market_rows,
                qualifier_features=qualifier_features,
                api_football_features=api_features,
                recent15_features=recent_features,
                fixture_feature_rows=xi_features,
                fixture_id=fixture_id,
                match_date=reference_date,
                match_year=match_year,
                fixture_context=record,
                dc_rho=score_model_rho(model),
                feature_profile="balanced",
            )
            context = public_feature_context(
                feature_row=feature_row,
                model_key=model_key,
                reference_date=reference_date,
                history_rows=int(history_cutoff.shape[0]),
                source_warnings=self.warnings,
            )
        except Exception as exc:
            context = {
                "available": False,
                "model_key": str(model_key),
                "reference_date": reference_date,
                "cutoff": "strictly_before_match",
                "usage_counts": {},
                "available_families": [],
                "warnings": unique_strings([*self.warnings, f"Feature context no disponible ({exc.__class__.__name__}: {exc})."]),
                "_feature_row": {},
            }
        self.context_cache[cache_key] = context
        return context


class BenchmarkFeatureEnhancedScoreModel:
    def __init__(
            self,
            base_model: Any,
            model_key: str,
            feature_source: BenchmarkFeatureSource,
            history_df: pd.DataFrame | None = None,
    ):
        self.base_model = base_model
        self.model_key = str(model_key or DEFAULT_SCORE_MODEL)
        self.feature_source = feature_source
        self.history_df = history_df
        self.max_goals = int(getattr(base_model, "max_goals", DEFAULT_CONFIG["max_goals"]))
        self.last_feature_context: Dict[str, Any] = {}

    def profile(self, team: str):
        return self.base_model.profile(team)

    def adjusted(self, rating_adjustments: Dict[str, float]):
        adjusted = self.base_model.adjusted(rating_adjustments)
        return BenchmarkFeatureEnhancedScoreModel(adjusted, self.model_key, self.feature_source, self.history_df)

    def score_model_metadata(self) -> Dict[str, Any]:
        metadata = dict(score_model_metadata(self.base_model))
        metadata["feature_enhanced"] = True
        return metadata

    def expected_goals(self, team1: str, team2: str) -> Tuple[float, float]:
        return self.expected_goals_for_match(team1, team2, match=None)

    def expected_goals_for_match(self, team1: str, team2: str, match: Dict[str, Any] | None = None) -> Tuple[float, float]:
        lambda_home, lambda_away, _ = self.adjusted_lambdas_and_context(team1, team2, match=match)
        return lambda_home, lambda_away

    def match_probabilities(self, team1: str, team2: str, max_goals: int | None = None) -> Dict[str, Any]:
        return self.match_probabilities_for_match(team1, team2, match=None, max_goals=max_goals)

    def match_probabilities_for_match(
            self,
            team1: str,
            team2: str,
            match: Dict[str, Any] | None = None,
            max_goals: int | None = None,
    ) -> Dict[str, Any]:
        limit_goals = int(max_goals if max_goals is not None else self.max_goals)
        base_probabilities = score_model_probabilities_for_match(self.base_model, team1, team2, match, limit_goals)
        lambda_home, lambda_away, context = self.adjusted_lambdas_and_context(
            team1,
            team2,
            match=match,
            base_probabilities=base_probabilities,
        )
        if not (context.get("lambda_adjustment") or {}).get("active"):
            output = dict(base_probabilities)
        else:
            grid = self.score_grid_from_lambdas(lambda_home, lambda_away, max_goals=limit_goals)
            output = probabilities_from_score_grid(grid, lambda1=lambda_home, lambda2=lambda_away)
        output["feature_context"] = strip_internal_feature_context(context)
        return output

    def score_grid(self, team1: str, team2: str, match: Dict[str, Any] | None = None, max_goals: int | None = None) -> np.ndarray:
        limit_goals = int(max_goals if max_goals is not None else self.max_goals)
        lambda_home, lambda_away, _ = self.adjusted_lambdas_and_context(team1, team2, match=match)
        return self.score_grid_from_lambdas(lambda_home, lambda_away, max_goals=limit_goals)

    def score_grid_from_lambdas(self, lambda1: float, lambda2: float, max_goals: int | None = None) -> np.ndarray:
        grid = match_score_grid_for_lambdas(self.base_model, lambda1, lambda2, max_goals=int(max_goals or self.max_goals))
        if grid is not None:
            return normalize_score_grid_array(grid)
        return normalize_score_grid_array(poisson_score_grid(lambda1, lambda2, max_goals=int(max_goals or self.max_goals)))

    def adjusted_lambdas_and_context(
            self,
            team1: str,
            team2: str,
            match: Dict[str, Any] | None = None,
            base_probabilities: Dict[str, Any] | None = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        base_probabilities = base_probabilities or score_model_probabilities_for_match(
            self.base_model,
            team1,
            team2,
            match,
            self.max_goals,
        )
        lambda_home = float_or_zero(base_probabilities.get("lambda1")) or 1.25
        lambda_away = float_or_zero(base_probabilities.get("lambda2")) or 1.05
        context = self.feature_source.context_for_match(
            self.base_model,
            match or {"Equipo 1": team1, "Equipo 2": team2},
            model_key=self.model_key,
            history_df=self.history_df,
        )
        adjustment = lambda_adjustment_from_feature_row(context.get("_feature_row") or {})
        lambda_home = float(np.clip(lambda_home * adjustment["home_factor"], 0.2, 4.8))
        lambda_away = float(np.clip(lambda_away * adjustment["away_factor"], 0.2, 4.8))
        public_context = dict(context)
        public_context["lambda_adjustment"] = {
            **adjustment,
            "base_lambda_home": round(float_or_zero(base_probabilities.get("lambda1")), 4),
            "base_lambda_away": round(float_or_zero(base_probabilities.get("lambda2")), 4),
            "lambda_home": round(lambda_home, 4),
            "lambda_away": round(lambda_away, 4),
        }
        self.last_feature_context = public_context
        return lambda_home, lambda_away, public_context


def benchmark_feature_source(tournament: Dict[str, Any], history_df: pd.DataFrame | None, config: Dict[str, Any]) -> BenchmarkFeatureSource:
    return BenchmarkFeatureSource(tournament=tournament, history_df=history_df, config=config)


def apply_benchmark_feature_model(
        model: Any,
        model_key: str,
        feature_source: BenchmarkFeatureSource | None,
        history_df: pd.DataFrame | None,
) -> Any:
    if feature_source is None:
        return model
    if isinstance(model, BenchmarkFeatureEnhancedScoreModel):
        return model
    return BenchmarkFeatureEnhancedScoreModel(model, model_key, feature_source, history_df=history_df)


def score_model_probabilities_for_match(
        model: Any,
        home: str,
        away: str,
        match: Dict[str, Any] | None,
        max_goals: int,
) -> Dict[str, Any]:
    method = getattr(model, "match_probabilities_for_match", None)
    if callable(method):
        return method(home, away, match=match, max_goals=max_goals)
    return model.match_probabilities(home, away, max_goals=max_goals)


def normalize_feature_history(history_df: pd.DataFrame | None) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=["Date", "Year", "Team 1", "Team 2", "G1", "G2", "Round", "Group"])
    working = history_df.copy()
    for column in ("Date", "Team 1", "Team 2", "G1", "G2"):
        if column not in working.columns:
            working[column] = np.nan
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    return working[working["Date"].notna()].sort_values("Date", kind="stable").reset_index(drop=True)


def history_before_feature_date(history_df: pd.DataFrame, date_ts: pd.Timestamp) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return normalize_feature_history(history_df)
    working = normalize_feature_history(history_df)
    if pd.notna(date_ts):
        cutoff = pd.Timestamp(date_ts).tz_localize(None)
        return working[working["Date"] < cutoff].copy()
    return working.copy()


def teams_from_feature_history(history_df: pd.DataFrame) -> List[str]:
    if history_df is None or history_df.empty:
        return []
    teams: List[str] = []
    for column in ("Team 1", "Team 2"):
        if column in history_df.columns:
            teams.extend(str(value) for value in history_df[column].dropna().tolist() if str(value).strip())
    return sorted(set(teams))


def dataframe_fingerprint_for_report(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "empty"
    columns = [column for column in ("Date", "Team 1", "Team 2", "G1", "G2") if column in frame.columns]
    payload = frame[columns].tail(20).astype(str).to_json(orient="records", force_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def safe_fixture_feature_rows_asof(
        fixture_feature_rows: pd.DataFrame,
        fixture_id: Any,
        reference_date: str,
        allow_current_fixture: bool,
) -> pd.DataFrame:
    if fixture_feature_rows is None or fixture_feature_rows.empty or fixture_id in {"", None}:
        return pd.DataFrame()
    if not allow_current_fixture:
        return pd.DataFrame(columns=fixture_feature_rows.columns)
    working = fixture_feature_rows.copy()
    if "fixture_id" not in working.columns and "Fixture" in working.columns:
        working["fixture_id"] = working["Fixture"].astype(str)
    if "Equipo" not in working.columns:
        return pd.DataFrame(columns=working.columns)
    scoped = working[working["fixture_id"].astype(str) == str(fixture_id)].copy() if "fixture_id" in working.columns else pd.DataFrame()
    if scoped.empty:
        return pd.DataFrame(columns=working.columns)
    if "Prediction safe" in scoped.columns:
        scoped = scoped[scoped["Prediction safe"].astype(str).str.lower().isin({"si", "sí", "yes", "true", "1"})].copy()
    return scoped


def score_model_rho(model: Any) -> float:
    try:
        metadata = score_model_metadata(model)
        params = metadata.get("params") if isinstance(metadata.get("params"), dict) else {}
        return float(params.get("rho", 0.0) or 0.0)
    except Exception:
        return 0.0


FEATURE_FAMILY_PREFIXES = {
    "rating": ("rating_", "attack_", "defense_", "matches_", "host_"),
    "form": ("history_", "recent15_", "trend_", "form_"),
    "odds": ("market_", "model_vs_market_", "market_vs_model_", "model_market_"),
    "xg_shots": ("qualifier_xg", "qualifier_shots", "api_football_xg", "api_football_total_shots", "api_football_shots"),
    "api_football": ("api_football_",),
    "xi_players": ("xi_", "lineup_", "injury_"),
    "h2h": ("h2h_",),
    "fixture": ("fixture_", "stage_", "venue_"),
    "score_grid": ("prob_score_", "prob_home_", "prob_away_", "dc_", "model_entropy_", "poisson_"),
}


def public_feature_context(
        feature_row: Dict[str, Any],
        model_key: str,
        reference_date: str,
        history_rows: int,
        source_warnings: Iterable[Any],
) -> Dict[str, Any]:
    counts = feature_usage_counts(feature_row)
    available_families = [key for key, count in counts.items() if int(count or 0) > 0]
    sample = feature_context_sample(feature_row)
    feature_list = feature_context_list(feature_row)
    return {
        "available": bool(available_families),
        "model_key": str(model_key),
        "reference_date": reference_date,
        "cutoff": "strictly_before_match",
        "history_rows": int(history_rows),
        "usage_counts": counts,
        "available_families": available_families,
        "feature_count": len(feature_list),
        "feature_list": feature_list,
        "sample": sample,
        "warnings": unique_strings(source_warnings),
        "_feature_row": feature_row,
    }


def strip_internal_feature_context(context: Dict[str, Any]) -> Dict[str, Any]:
    return {key: json_safe(value) for key, value in (context or {}).items() if not str(key).startswith("_")}


def feature_usage_counts(feature_row: Dict[str, Any]) -> Dict[str, int]:
    counts = {family: 0 for family in FEATURE_FAMILY_PREFIXES}
    for key, value in (feature_row or {}).items():
        if not feature_value_is_present(value):
            continue
        key_text = str(key)
        for family, prefixes in FEATURE_FAMILY_PREFIXES.items():
            if key_text.startswith(prefixes):
                counts[family] += 1
                break
    return counts


def feature_family_for_key(key: Any) -> str:
    key_text = str(key or "")
    for family, prefixes in FEATURE_FAMILY_PREFIXES.items():
        if key_text.startswith(prefixes):
            return family
    return "other"


def feature_context_list(feature_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key, value in (feature_row or {}).items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(number):
            continue
        rows.append({
            "name": str(key),
            "family": feature_family_for_key(key),
            "value": round(float(number), 6),
            "present": feature_value_is_present(number),
        })
    return sorted(rows, key=lambda item: (item["family"], item["name"]))


def combined_feature_usage_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    totals = {family: 0 for family in FEATURE_FAMILY_PREFIXES}
    for row in rows or []:
        context = row.get("feature_context") if isinstance(row, dict) else {}
        counts = (context or {}).get("usage_counts") if isinstance(context, dict) else {}
        for family in totals:
            totals[family] += int((counts or {}).get(family) or 0)
    return totals


def feature_value_is_present(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and abs(number) > 1e-9)


def feature_context_sample(feature_row: Dict[str, Any], limit: int = 12) -> Dict[str, float]:
    preferred_tokens = ("market_prob", "recent15", "history_last_5", "history_last_10", "api_football", "qualifier_xg", "xi_")
    ranked = []
    for index, (key, value) in enumerate((feature_row or {}).items()):
        if not feature_value_is_present(value):
            continue
        key_text = str(key)
        preferred = 0 if any(token in key_text for token in preferred_tokens) else 1
        ranked.append((preferred, index, key_text, round(float(value), 6)))
    return {key: value for _, _, key, value in sorted(ranked)[:limit]}


def lambda_adjustment_from_feature_row(feature_row: Dict[str, Any]) -> Dict[str, Any]:
    terms: List[Dict[str, Any]] = []

    def add(name: str, value: Any, scale: float, weight: float) -> None:
        number = float_or_zero(value)
        if abs(number) <= 1e-9:
            return
        normalized = float(np.clip(number / max(scale, 1e-9), -1.0, 1.0))
        contribution = normalized * float(weight)
        if abs(contribution) <= 1e-9:
            return
        terms.append({
            "name": name,
            "value": round(number, 6),
            "weight": round(float(weight), 4),
            "contribution": round(contribution, 6),
        })

    add("rating_diff", feature_row.get("rating_diff"), 240.0, 0.035)
    add("recent15_goal_diff", feature_row.get("recent15_adjusted_goal_diff_avg_diff"), 2.0, 0.065)
    add("history_form_5", feature_row.get("history_last_5_goal_diff_avg_diff"), 2.0, 0.045)
    add("history_form_10", feature_row.get("history_last_10_goal_diff_avg_diff"), 2.0, 0.035)
    add("history_form_15", feature_row.get("history_last_15_goal_diff_avg_diff"), 2.0, 0.03)
    add("qualifier_xg", feature_row.get("qualifier_xg_avg_diff"), 1.5, 0.045)
    add("api_xg", first_feature_value(feature_row, ("api_football_xg_for_avg_diff", "api_football_expected_goals_for_avg_diff")), 1.5, 0.045)
    add("api_shots", first_feature_value(feature_row, ("api_football_total_shots_for_avg_diff", "api_football_shots_for_avg_diff")), 10.0, 0.025)
    add("xi_rating", first_feature_value(feature_row, ("xi_xi_rating_prom_diff", "xi_rating_prom_diff")), 1.0, 0.04)
    if float_or_zero(feature_row.get("market_has_1x2")) > 0:
        market_edge = float_or_zero(feature_row.get("market_prob_home")) - float_or_zero(feature_row.get("market_prob_away"))
        add("market_no_vig_edge", market_edge, 0.35, 0.07)

    edge = float(np.clip(sum(float(item["contribution"]) for item in terms), -0.18, 0.18))
    return {
        "active": bool(terms),
        "edge": round(edge, 6),
        "home_factor": round(math.exp(edge), 6),
        "away_factor": round(math.exp(-0.85 * edge), 6),
        "terms": terms[:8],
    }


def first_feature_value(feature_row: Dict[str, Any], keys: Iterable[str]) -> float:
    for key in keys:
        if key in feature_row and feature_value_is_present(feature_row.get(key)):
            return float(feature_row.get(key))
    return 0.0
COUNTRY_CODES = {
    "Algeria": "dz",
    "Argentina": "ar",
    "Australia": "au",
    "Austria": "at",
    "Belgium": "be",
    "Bosnia & Herzegovina": "ba",
    "Brazil": "br",
    "Canada": "ca",
    "Cape Verde": "cv",
    "Colombia": "co",
    "Croatia": "hr",
    "Czech Republic": "cz",
    "Curaçao": "cw",
    "Curacao": "cw",
    "DR Congo": "cd",
    "Ecuador": "ec",
    "Egypt": "eg",
    "England": "gb-eng",
    "France": "fr",
    "Germany": "de",
    "Ghana": "gh",
    "Haiti": "ht",
    "Iran": "ir",
    "Iraq": "iq",
    "Ivory Coast": "ci",
    "Japan": "jp",
    "Jordan": "jo",
    "Mexico": "mx",
    "Morocco": "ma",
    "Netherlands": "nl",
    "New Zealand": "nz",
    "Norway": "no",
    "Panama": "pa",
    "Paraguay": "py",
    "Portugal": "pt",
    "Qatar": "qa",
    "Saudi Arabia": "sa",
    "Scotland": "gb-sct",
    "Senegal": "sn",
    "South Africa": "za",
    "South Korea": "kr",
    "Spain": "es",
    "Sweden": "se",
    "Switzerland": "ch",
    "Tunisia": "tn",
    "Turkey": "tr",
    "USA": "us",
    "Uruguay": "uy",
    "Uzbekistan": "uz",
}
LOCAL_FLAG_ALIASES = {
    "United States": "USA",
    "United States of America": "USA",
    "Curacao": "Curaçao",
}
_WORLD_CUP_RESULTS_AUTO_REFRESH_LOCK = threading.Lock()
_WORLD_CUP_RESULTS_AUTO_REFRESHED = False
_WORLD_CUP_RESULTS_AUTO_REFRESH_SUMMARY: Dict[str, Any] = {}
_WORLD_CUP_RESULTS_AUTO_REFRESH_EXPIRES_AT: datetime | None = None
_WORLD_CUP_FIXTURES_AUTO_REFRESH_EXPIRES_AT: datetime | None = None


def _utcify_datetime(value: Any | None = None) -> datetime:
    current = value if value is not None else _now_utc()
    if current.tzinfo is None:
        return current.replace(tzinfo=timezone.utc)
    return current.astimezone(timezone.utc)


def _world_cup_results_autorefresh_stale(now: Any | None = None) -> bool:
    current = _utcify_datetime(now)
    if _WORLD_CUP_RESULTS_AUTO_REFRESH_EXPIRES_AT is None:
        return True
    last = _utcify_datetime(_WORLD_CUP_RESULTS_AUTO_REFRESH_EXPIRES_AT)
    return current.date() != last.date()


def _world_cup_fixture_autorefresh_stale(now: Any | None = None) -> bool:
    current = _utcify_datetime(now)
    if _WORLD_CUP_FIXTURES_AUTO_REFRESH_EXPIRES_AT is None:
        return True
    last = _utcify_datetime(_WORLD_CUP_FIXTURES_AUTO_REFRESH_EXPIRES_AT)
    return current.date() != last.date()


def ensure_worldcup_results_autorefreshed_once(tournament: Dict[str, Any] | None = None) -> Dict[str, Any]:
    global _WORLD_CUP_RESULTS_AUTO_REFRESHED, _WORLD_CUP_RESULTS_AUTO_REFRESH_SUMMARY, _WORLD_CUP_RESULTS_AUTO_REFRESH_EXPIRES_AT
    now = _now_utc()
    if _WORLD_CUP_RESULTS_AUTO_REFRESHED and not _world_cup_results_autorefresh_stale(now):
        return dict(_WORLD_CUP_RESULTS_AUTO_REFRESH_SUMMARY)
    with _WORLD_CUP_RESULTS_AUTO_REFRESH_LOCK:
        if _WORLD_CUP_RESULTS_AUTO_REFRESHED and not _world_cup_results_autorefresh_stale(now):
            return dict(_WORLD_CUP_RESULTS_AUTO_REFRESH_SUMMARY)
        now = _utcify_datetime(now)
        try:
            target = tournament if tournament is not None else load_tournament_2026(refresh=False)[0]
            summary = refresh_worldcup_2026_results(target, refresh=True)
        except Exception as exc:
            summary = {
                "source": "unavailable:auto-refresh",
                "provider": "auto-refresh",
                "refresh_attempted": True,
                "refresh_added": 0,
                "refresh_updated": 0,
                "fotmob_final_rows": 0,
                "sofascore_final_rows": 0,
                "verified_final_rows": 0,
                "conflicts": [],
                "warnings": [f"Auto-refresh resultados Mundial no disponible: {exc.__class__.__name__}: {exc}"],
                "provider_warnings": [],
                "missing_result_fixtures": [],
                "auto_refresh_error": str(exc),
                "auto_refreshed_at": _utcify_datetime(now).isoformat(),
            }
        _WORLD_CUP_RESULTS_AUTO_REFRESH_SUMMARY = dict(summary or {})
        _WORLD_CUP_RESULTS_AUTO_REFRESHED = True
        _WORLD_CUP_RESULTS_AUTO_REFRESH_EXPIRES_AT = _utcify_datetime(now)
        return dict(_WORLD_CUP_RESULTS_AUTO_REFRESH_SUMMARY)


def ensure_worldcup_results_autorefresh_once(tournament: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Compatibility shim for legacy call sites/tests."""
    return ensure_worldcup_results_autorefreshed_once(tournament)


def overview(refresh: bool = False) -> Dict[str, Any]:
    tournament, fixture_source = load_tournament_2026(refresh=bool(refresh))
    results_autorefresh = ensure_worldcup_results_autorefreshed_once(tournament)
    groups = groups_from_tournament(tournament)
    fixture_df = tournament_fixtures_dataframe(tournament)
    players_df, players_source = load_players(refresh=False)
    fixture_summary = fixture_overview_payload(fixture_df)
    standings = group_standings_payload(groups, fixture_df)
    results_status = fixture_results_status(fixture_df)
    international_status = international_results_status()
    advanced_status = advanced_data_status()
    return {
        "name": tournament.get("name", "World Cup 2026"),
        "teams": sum(len(teams) for teams in groups.values()),
        "groups": len(groups),
        "fixtures": int(fixture_df.shape[0]),
        "group_fixtures": int((fixture_df["Grupo"] != "").sum()) if not fixture_df.empty else 0,
        "players": int(players_df.shape[0]),
        "fixture_source": fixture_source,
        "result_source": results_status["source"],
        "result_override_rows": results_status["override_rows"],
        "result_override_applied": results_status["override_applied"],
        "confirmed_results": results_status["confirmed_results"],
        "results_updated_at": results_status["updated_at"],
        "results_autorefresh": results_autorefresh,
        "players_source": players_source,
        "opener": fixture_summary["opener"],
        "featured_matches": fixture_summary["featured_matches"],
        "highlight": fixture_summary["highlight"],
        "next_matches": fixture_summary["next_matches"],
        "countdown_target": fixture_summary["countdown_target"],
        "countdown_state": fixture_summary["countdown_state"],
        "group_standings": standings,
        "default_config": DEFAULT_CONFIG,
        "score_models": score_model_options(),
        "advanced_data": advanced_status,
        "advanced_model_catalog": advanced_models_catalog(advanced_status),
        "international_recent": international_status,
        "hardware": detect_hardware(),
        "model": "Elo + modelos de marcador Monte Carlo",
        "last_simulation": LAST_SIMULATION_RESULT,
        "assets_policy": "Banderas locales/publicas y fotos publicas de SofaScore con fallback visual.",
    }


def groups(refresh: bool = False) -> Dict[str, Any]:
    tournament, source = load_tournament_2026(refresh=bool(refresh))
    ensure_worldcup_results_autorefreshed_once(tournament)
    group_map = groups_from_tournament(tournament)
    fixture_df = tournament_fixtures_dataframe(tournament)
    results_status = fixture_results_status(fixture_df)
    items = []
    for group_name, team_names in group_map.items():
        items.append({
            "name": group_name,
            "letter": group_letter(group_name),
            "played_matches": group_finished_fixture_count(group_name, fixture_df),
            "standings": group_standing_rows(group_name, team_names, fixture_df),
            "teams": [
                {
                    **team_asset(team),
                    "seed": seed,
                    "group": group_name,
                }
                for seed, team in enumerate(team_names, start=1)
            ],
        })
    return {
        "groups": items,
        "table": table_payload(groups_dataframe(tournament), page=1, page_size=80),
        "source": source,
        "result_source": results_status["source"],
        "confirmed_results": results_status["confirmed_results"],
        "results_updated_at": results_status["updated_at"],
    }


def fixtures(refresh: bool = False) -> Dict[str, Any]:
    tournament, source = load_tournament_2026(refresh=bool(refresh))
    ensure_worldcup_results_autorefreshed_once(tournament)
    df = tournament_fixtures_dataframe(tournament)
    results_status = fixture_results_status(df)
    rows = []
    for _, row in df.iterrows():
        home = str(row.get("Equipo 1", ""))
        away = str(row.get("Equipo 2", ""))
        rows.append({
            "id": str(row.get("No.", "")),
            "date": row.get("Fecha", ""),
            "time": row.get("Hora", ""),
            "round": row.get("Ronda", ""),
            "group": row.get("Grupo", ""),
            "venue": row.get("Sede", ""),
            "score_home": row.get("Goles 1", ""),
            "score_away": row.get("Goles 2", ""),
            "finished": str(row.get("Finalizado", "")).lower() == "si",
            "result_source": row.get("Fuente Resultado", ""),
            "result_override": str(row.get("Resultado Override", "")).lower() == "si",
            "home": team_asset(home),
            "away": team_asset(away),
            "label": f"{home} vs {away}",
        })
    return {
        "fixtures": rows,
        "table": table_payload(df, page=1, page_size=150),
        "source": source,
        "result_source": results_status["source"],
        "confirmed_results": results_status["confirmed_results"],
    }


def teams(refresh: bool = False, config_payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    config = simulation_config(config_payload or {})
    tournament, _ = load_tournament_2026(refresh=bool(refresh))
    model, history_source = build_model(tournament, config)
    df = teams_dataframe(tournament, model=model)
    records = []
    for row in table_payload(df, page=1, page_size=80)["rows"]:
        team = str(row.get("Equipo", ""))
        records.append({
            **row,
            "asset": team_asset(team),
            "is_host": team in {"Mexico", "USA", "Canada"},
        })
    return {
        "teams": records,
        "table": table_payload(df, page=1, page_size=80),
        "history_source": history_source,
        "config": public_report_config(config),
    }


def players(refresh: bool = False) -> Dict[str, Any]:
    df, source = load_players(refresh=bool(refresh))
    records = []
    for _, row in df.iterrows():
        team = str(row.get("Equipo", ""))
        player_id = row.get("SofaScore ID") or row.get("id") or ""
        name = str(row.get("Jugador", ""))
        records.append({
            "team": team_asset(team),
            "name": name,
            "position": row.get("Posicion", ""),
            "club": row.get("Club", ""),
            "age": row.get("Edad", ""),
            "photo_url": sofa_player_photo_url(player_id),
            "initials": initials(name),
            "source": row.get("Fuente", source),
        })
    return {
        "players": records,
        "table": table_payload(df, page=1, page_size=250),
        "source": source,
    }


def lineups(refresh: bool = False) -> Dict[str, Any]:
    tournament, source = load_tournament_2026(refresh=bool(refresh))
    df = lineups_summary(tournament)
    rows = []
    for _, row in df.iterrows():
        home = str(row.get("Equipo 1", ""))
        away = str(row.get("Equipo 2", ""))
        rows.append({
            "fixture_id": str(row.get("Fixture", "")),
            "date": row.get("Fecha", ""),
            "group": row.get("Grupo", ""),
            "home": team_asset(home),
            "away": team_asset(away),
            "status": row.get("Estado", ""),
            "starters_home": int(row.get("Local 11", 0) or 0),
            "starters_away": int(row.get("Visitante 11", 0) or 0),
            "match_url": row.get("URL SofaScore", ""),
            "event_id": row.get("SofaScore ID", ""),
            "auto_match": row.get("Auto match", ""),
        })
    return {
        "lineups": rows,
        "table": table_payload(df, page=1, page_size=100),
        "fixture_source": source,
    }


def fixture_lineup(fixture_id: str, refresh: bool = False) -> Dict[str, Any]:
    tournament, _ = load_tournament_2026(refresh=False)
    payload = lineup_payload_for_fixture(tournament=tournament, fixture_id=fixture_id, refresh=bool(refresh))
    return lineup_response(payload)


def autodetect_fixture(fixture_id: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    tournament, _ = load_tournament_2026(refresh=bool(payload.get("refresh_fixtures", False)))
    event = autodetect_fixture_event(tournament=tournament, fixture_id=fixture_id, refresh=bool(payload.get("refresh", False)))
    lineup = {}
    if event.get("match_url") and bool(payload.get("fetch_lineup", True)):
        lineup = lineup_response(lineup_payload_from_detected_event(tournament=tournament, fixture_id=fixture_id, event=event, refresh=True))
    return {
        "event": event,
        "lineup": lineup,
    }


def auto_refresh(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    tournament, _ = load_tournament_2026(refresh=bool(payload.get("refresh_fixtures", False)))
    return auto_refresh_lineups(
        tournament=tournament,
        refresh_events=bool(payload.get("refresh_events", False)),
        limit=int(payload.get("limit") or 0),
    )


def refresh_fixture_lineup(fixture_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    tournament, _ = load_tournament_2026(refresh=bool(payload.get("refresh_fixtures", False)))
    lineup = lineup_payload_for_fixture(
        tournament=tournament,
        fixture_id=fixture_id,
        refresh=True,
        match_url=payload.get("match_url"),
    )
    return lineup_response(lineup)


def link_lineup(fixture_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    tournament, _ = load_tournament_2026(refresh=False)
    lineup = link_fixture_lineup(
        tournament=tournament,
        fixture_id=fixture_id,
        match_url=str(payload.get("match_url") or ""),
        refresh=bool(payload.get("refresh", True)),
    )
    return lineup_response(lineup)


def fixture_player_stats(fixture_id: str, refresh: bool = False) -> Dict[str, Any]:
    tournament, _ = load_tournament_2026(refresh=False)
    payload = player_stats_payload_for_fixture(tournament=tournament, fixture_id=fixture_id, refresh=bool(refresh))
    return {
        "stats": enrich_lineup_payload(payload),
        "features": table_payload(pd.DataFrame(payload.get("features", [])), page=1, page_size=10),
        "players": table_payload(lineups_table(payload), page=1, page_size=40),
    }


def player_features(refresh: bool = False) -> Dict[str, Any]:
    tournament, _ = load_tournament_2026(refresh=bool(refresh))
    df = player_features_dataframe(tournament)
    return {
        "features": table_payload(df, page=1, page_size=120),
        "rows": jsonable(df.to_dict(orient="records")) if not df.empty else [],
    }


def maintenance_clear(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    clear_cache = bool(payload.get("clear_cache", True))
    removed: List[str] = []
    recreated: List[str] = []
    for root in (FEATURE_STORE_ROOT, LINEUPS_ROOT, PLAYER_STATS_ROOT, SOFASCORE_ROOT, WALK_FORWARD_ROOT):
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
            removed.append(str(root))
        root.mkdir(parents=True, exist_ok=True)
        recreated.append(str(root))
    if clear_cache and CACHE_ROOT.exists():
        for path in sorted(CACHE_ROOT.iterdir()):
            if path.is_file() and re.fullmatch(r"worldcup_\d{4}\.json", path.name):
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()
            removed.append(str(path))
    return {
        "removed": removed,
        "recreated": recreated,
        "cache_preserved": [
            str(INTERNATIONAL_ROOT),
            "storage/worldcup/cache/worldcup_*.json",
            "storage/worldcup/models",
        ],
    }


def predict_match(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    config = simulation_config(payload)
    tournament, fixture_source = load_tournament_2026(refresh=bool(config["refresh"]))
    results_autorefresh = ensure_worldcup_results_autorefreshed_once(tournament)
    model, history_source = build_model(tournament, config)
    model = apply_configured_score_model(model, tournament, config)
    result = poisson_match_payload(
        tournament=tournament,
        base_model=model,
        fixture_id=payload.get("fixture_id"),
        home=payload.get("home"),
        away=payload.get("away"),
        config=config,
        poisson_recent_matches=int(config["poisson_recent_matches"]),
    )
    result["fixture_source"] = fixture_source
    result["history_source"] = history_source
    result["score_model"] = score_model_metadata(model)
    result["results_autorefresh"] = results_autorefresh
    return result


def predict_upcoming(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    config = simulation_config(payload)
    tournament, fixture_source = load_tournament_2026(refresh=bool(config["refresh"]))
    results_autorefresh = ensure_worldcup_results_autorefreshed_once(tournament)
    model, history_source = build_model(tournament, config)
    model = apply_configured_score_model(model, tournament, config)

    limit = int(_clamp_int(payload.get("limit", 8), 1, 72))
    group_filter = str(payload.get("group") or "").strip()
    fixture_df = upcoming_fixture_rows(tournament, group_filter=group_filter)
    predictions = []
    rows = []
    for _, fixture in fixture_df.head(limit).iterrows():
        result = poisson_match_payload(
            tournament=tournament,
            base_model=model,
            fixture_id=fixture.get("No."),
            config=config,
            poisson_recent_matches=int(config["poisson_recent_matches"]),
        )
        predictions.append(result)
        rows.append(upcoming_prediction_row(result))
    return {
        "predictions": predictions,
        "table": table_payload(pd.DataFrame(rows), page=1, page_size=limit),
        "summary": {
            "requested": limit,
            "returned": len(predictions),
            "group": group_filter or "Todos",
            "fixture_source": fixture_source,
            "history_source": history_source,
            "poisson_recent_matches": config["poisson_recent_matches"],
            "score_model": config["score_model"],
            "score_model_label": score_model_metadata(model).get("label", ""),
            "results_autorefresh": results_autorefresh,
        },
    }


def predict_upcoming_monte_carlo(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    config = simulation_config({**payload, "mode": "poisson_live"})
    config["iterations"] = monte_carlo_match_iterations(payload.get("iterations", DEFAULT_CONFIG["iterations"]))
    tournament, fixture_source = load_tournament_2026(refresh=bool(config["refresh"]))
    results_autorefresh = ensure_worldcup_results_autorefreshed_once(tournament)
    model, history_source = build_model(tournament, config)
    model = apply_configured_score_model(model, tournament, config)
    international_matches = load_international_matches(required=False)
    limit = int(_clamp_int(payload.get("limit", 8), 1, 72))
    group_filter = str(payload.get("group") or "").strip()
    fixture_df = upcoming_fixture_rows(tournament, group_filter=group_filter)
    rng = np.random.default_rng(int(config["seed"]))
    predictions = []
    rows = []
    for _, fixture in fixture_df.head(limit).iterrows():
        result = monte_carlo_match_prediction(
            fixture=fixture,
            base_model=model,
            config=config,
            rng=rng,
            international_matches=international_matches,
        )
        predictions.append(result)
        rows.append(monte_carlo_match_row(result))
    return {
        "predictions": predictions,
        "table": table_payload(pd.DataFrame(rows), page=1, page_size=limit),
        "summary": {
            "method": "Monte Carlo Poisson por partido",
            "requested": limit,
            "returned": len(predictions),
            "group": group_filter or "Todos",
            "iterations": config["iterations"],
            "seed": config["seed"],
            "poisson_recent_matches": config["poisson_recent_matches"],
            "score_model": config["score_model"],
            "score_model_label": score_model_metadata(model).get("label", ""),
            "fixture_source": fixture_source,
            "history_source": history_source,
            "results_autorefresh": results_autorefresh,
            "source": "Modelo de marcador contextual + simulacion por fixture",
        },
    }


def poisson_match_payload(
        tournament: Dict[str, Any],
        base_model: Any,
        fixture_id: Any = None,
        home: Any = None,
        away: Any = None,
        config: Dict[str, Any] | None = None,
        poisson_recent_matches: int = 15,
) -> Dict[str, Any]:
    config = config or DEFAULT_CONFIG
    fixture = resolve_prediction_fixture(tournament, fixture_id=fixture_id, home=home, away=away)
    home_team = str(fixture.get("Equipo 1", fixture.get("home", home or "")))
    away_team = str(fixture.get("Equipo 2", fixture.get("away", away or "")))
    probabilities = model_probabilities_for_fixture(base_model, fixture, config)
    probs_pct = {
        "home": probability_percent(probabilities.get("home", 0.0)),
        "draw": probability_percent(probabilities.get("draw", 0.0)),
        "away": probability_percent(probabilities.get("away", 0.0)),
    }
    for line in REPORT_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        probs_pct[f"over{suffix}"] = probability_percent(probabilities.get(f"over{suffix}", 0.0))
        probs_pct[f"under{suffix}"] = probability_percent(probabilities.get(f"under{suffix}", 0.0))
    lambda_home = float_or_zero(probabilities.get("lambda1", 0.0))
    lambda_away = float_or_zero(probabilities.get("lambda2", 0.0))
    score_distribution = score_distribution_for_fixture(base_model, fixture, probabilities, config)
    top_scores = score_distribution.get("top_scores", [])
    modal_score = top_scores[0]["score"] if top_scores else f"{round_half_up_int(lambda_home)}-{round_half_up_int(lambda_away)}"
    outcome = outcome_decision(probs_pct)
    score_metadata = score_model_metadata(base_model)
    source_name = str(score_metadata.get("label") or "Poisson/SOTA")
    score_warnings = score_metadata.get("warnings", [])
    if not isinstance(score_warnings, list):
        score_warnings = [score_warnings]
    context = contextual_poisson_for_match(
        home_team,
        away_team,
        base_model=base_model,
        before_date=fixture.get("Fecha", ""),
        max_goals=int(config.get("max_goals") or DEFAULT_CONFIG["max_goals"]),
        limit=int(poisson_recent_matches),
    )
    context_warnings = context.get("warnings", [])
    if not isinstance(context_warnings, list):
        context_warnings = [context_warnings] if context_warnings else []
    market_sources = {
        "result": {"label": "1X2", "source": "Poisson/SOTA", "model_name": source_name},
    }
    for line in REPORT_TOTAL_GOAL_LINES:
        key = f"over_under_{total_line_suffix(line)}"
        market_sources[key] = {"label": f"U/O {line:.1f}", "source": "Poisson/SOTA", "model_name": source_name}
    return {
        "fixture": {
            "id": str(fixture.get("No.", "")),
            "date": fixture.get("Fecha", ""),
            "time": fixture.get("Hora", ""),
            "group": fixture.get("Grupo", ""),
            "home": home_team,
            "away": away_team,
            "venue": fixture.get("Sede", ""),
        },
        "probabilities": probs_pct,
        "expected_goals": {
            "home": round(lambda_home, 3),
            "away": round(lambda_away, 3),
        },
        "modal_score": modal_score,
        "top_score": modal_score,
        "top_scores": top_scores,
        "prediction": f"{outcome_label(outcome)} {outcome_team(outcome, fixture)}".strip(),
        "score_model": score_metadata,
        "score_distribution": score_distribution,
        "contextual_poisson": context,
        "market_sources": market_sources,
        "market_readout": {},
        "notes": unique_strings([*score_warnings, *context_warnings]),
        "source": source_name,
    }


def resolve_prediction_fixture(tournament: Dict[str, Any], fixture_id: Any = None, home: Any = None, away: Any = None) -> pd.Series:
    fixture_df = tournament_fixtures_dataframe(tournament)
    if not fixture_df.empty:
        if fixture_id not in (None, ""):
            target = str(fixture_id)
            matched = fixture_df[fixture_df["No."].astype(str) == target]
            if not matched.empty:
                return matched.iloc[0]
        home_text = str(home or "").strip()
        away_text = str(away or "").strip()
        if home_text and away_text:
            matched = fixture_df[
                fixture_df["Equipo 1"].astype(str).eq(home_text)
                & fixture_df["Equipo 2"].astype(str).eq(away_text)
            ]
            if not matched.empty:
                return matched.iloc[0]
    return pd.Series({
        "No.": fixture_id or "",
        "Fecha": "",
        "Hora": "",
        "Grupo": "",
        "Equipo 1": str(home or ""),
        "Equipo 2": str(away or ""),
        "Sede": "",
    })


OPTIONAL_WARNING_PATTERNS = (
    "no disponible en cache",
    "sin cache avanzado local",
    "sin cache local",
    "no se encontraron filas locales",
    "socceraction no instalado",
    "no existe storage/worldcup/xg",
    "api-football omitido",
    "api-football no disponible",
    "football-data xlsx no disponible",
    "bundle xg-lightgbm",
    "fallback poisson",
    "respaldo poisson",
)


def normalize_report_message(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def normalize_report_messages(values: Iterable[Any]) -> List[str]:
    return unique_strings(normalize_report_message(value) for value in values)


def warning_is_optional_limitation(text: str) -> bool:
    normalized = normalize_report_message(text).lower()
    return any(pattern in normalized for pattern in OPTIONAL_WARNING_PATTERNS)


def public_warning_payload(warnings: Iterable[Any], pipeline_mode: str = "") -> Dict[str, List[str]]:
    technical = normalize_report_messages(warnings)
    visible: List[str] = []
    if any(warning_is_optional_limitation(item) for item in technical):
        if pipeline_mode == XG_LIGHTGBM_PIPELINE_MODE:
            visible.append("Fuentes o modelo xG pendientes; el reporte usó fallback Poisson donde hizo falta.")
        elif pipeline_mode == ADVANCED_MODELS_PIPELINE_MODE:
            visible.append("Fuentes avanzadas opcionales pendientes; el reporte usó fallback estadístico donde hizo falta.")
        else:
            visible.append("Fuentes opcionales pendientes; el reporte usó fallback estadístico donde hizo falta.")
    for item in technical:
        lower = item.lower()
        if warning_is_optional_limitation(item):
            continue
        if "cuda fue solicitada explicitamente" in lower or "error" in lower or "fallo" in lower:
            visible.append(item)
        elif "potencia limitada" in lower:
            visible.append("Backtest con muestra limitada; interpreta p-values como exploratorios.")
    return {
        "visible_warnings": unique_strings(visible),
        "technical_warnings": technical,
    }


def merge_warning_payloads(*payloads: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        "visible_warnings": unique_strings(
            item
            for payload in payloads
            for item in payload.get("visible_warnings", [])
        ),
        "technical_warnings": normalize_report_messages(
            item
            for payload in payloads
            for item in payload.get("technical_warnings", [])
        ),
    }


def public_report_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in dict(config or {}).items()
        if not str(key).startswith("_")
    }


def source_rows_count(frame: Any) -> int:
    return int(frame.shape[0]) if hasattr(frame, "shape") and len(frame.shape) >= 1 else 0


def resolve_worldcup_sources_for_pipeline(
        payload: Dict[str, Any],
        config: Dict[str, Any],
        pipeline_mode: str,
        progress_callback=None,
) -> Dict[str, Any]:
    if pipeline_mode not in {ADVANCED_MODELS_PIPELINE_MODE, XG_LIGHTGBM_PIPELINE_MODE}:
        return {"status": "skipped", "status_label": "No requerido", "sources": {}, "visible_warnings": [], "technical_warnings": []}
    emit_job_progress(progress_callback, "source_preflight", 0, 4, "Resolviendo fuentes cacheables")
    result: Dict[str, Any] = {
        "status": "ok",
        "status_label": "Fuentes revisadas",
        "sources": {},
        "actions": [],
        "visible_warnings": [],
        "technical_warnings": [],
    }
    warnings: List[str] = []
    if pipeline_mode == XG_LIGHTGBM_PIPELINE_MODE:
        model_id, model_meta, model_warnings = resolve_xg_lightgbm_model(payload)
        warnings.extend(model_warnings)
        model_ready = bool(model_meta.get("trained") and model_meta.get("model_profile") == XG_LIGHTGBM_PROFILE)
        result["sources"]["xg_lightgbm_model"] = {
            "status": "trained" if model_ready else "missing",
            "available": model_ready,
            "model_id": model_id,
        }
        if model_ready:
            emit_job_progress(progress_callback, "source_preflight", 4, 4, "Bundle xG-LightGBM listo")
            result.update(public_warning_payload(warnings, pipeline_mode=pipeline_mode))
            return json_safe(result)
    try:
        market = load_market_data(allow_download=True, force_download=False, use_scraper=False)
        result["sources"]["football_data"] = {
            "status": market.get("status", "missing"),
            "rows": int(market.get("market_rows", 0) or 0),
            "qualifier_rows": int(market.get("qualifier_rows", 0) or 0),
            "sources": [normalize_report_message(item) for item in market.get("sources", [])],
        }
        warnings.extend(market.get("warnings", []))
    except Exception as exc:
        message = f"Football-Data XLSX no pudo resolverse ({exc.__class__.__name__}: {exc})."
        result["sources"]["football_data"] = {"status": "error", "error": normalize_report_message(message)}
        warnings.append(message)
    emit_job_progress(progress_callback, "source_preflight", 1, 4, "Fuentes de mercado revisadas")

    if pipeline_mode == ADVANCED_MODELS_PIPELINE_MODE:
        try:
            api_bundle = load_api_football_data(allow_download=True, force_download=False)
            api_rows = sum(source_rows_count(api_bundle.get(key)) for key in ("fixtures", "team_stats", "lineups", "injuries", "odds", "market_rows"))
            result["sources"]["api_football"] = {
                "status": api_bundle.get("status", "missing"),
                "rows": api_rows,
                "sources": [normalize_report_message(item) for item in api_bundle.get("sources", [])],
                "downloaded": [normalize_report_message(item) for item in api_bundle.get("downloaded", [])],
            }
            warnings.extend(api_bundle.get("warnings", []))
        except Exception as exc:
            message = f"API-Football no pudo resolverse ({exc.__class__.__name__}: {exc})."
            result["sources"]["api_football"] = {"status": "error", "error": normalize_report_message(message)}
            warnings.append(message)
        emit_job_progress(progress_callback, "source_preflight", 2, 4, "API-Football revisado")
        try:
            data_status = advanced_data_prepare({"force": True, "snapshot_statsbomb": False}, progress_callback=None)
            config["_advanced_data_status"] = data_status
            result["sources"]["advanced_data"] = {
                "status": "prepared" if int(data_status.get("prepared_rows", 0) or 0) > 0 else "fallback",
                "rows": int(data_status.get("prepared_rows", 0) or 0),
                "active_sources": len(data_status.get("active_sources", []) or []),
            }
            warnings.extend(data_status.get("warnings", []))
            result["actions"].append("advanced_data_prepare")
        except Exception as exc:
            message = f"Preparación avanzada falló ({exc.__class__.__name__}: {exc})."
            result["sources"]["advanced_data"] = {"status": "error", "error": normalize_report_message(message)}
            warnings.append(message)
    elif pipeline_mode == XG_LIGHTGBM_PIPELINE_MODE:
        try:
            status = xg_lightgbm_training_status()
            dataset = status.get("dataset", {})
            if not dataset.get("etl_ready") or dataset.get("etl_stale"):
                emit_job_progress(progress_callback, "source_preflight", 2, 4, "Preparando ETL xG-LightGBM")
                status = xg_lightgbm_prepare_training({**payload, "force": True, "refresh_history": False}, progress_callback=None)
                result["actions"].append("xg_prepare_etl")
            if status.get("can_train"):
                emit_job_progress(progress_callback, "source_preflight", 3, 4, "Entrenando xG-LightGBM")
                train_result = xg_lightgbm_train_model(payload, progress_callback=None)
                status = train_result.get("status", status)
                result["actions"].append("xg_train_model")
            config["_xg_lightgbm_status"] = status
            model = status.get("model", {})
            result["sources"]["xg_lightgbm_model"] = {
                "status": "trained" if model.get("trained") else "fallback",
                "available": bool(model.get("trained")),
                "model_id": model.get("model_id", model_id),
                "rows": int(model.get("train_rows", 0) or 0),
            }
            warnings.extend((model or {}).get("warnings", []))
            dataset_warnings = (status.get("dataset", {}) or {}).get("prepared_warnings", [])
            warnings.extend(dataset_warnings if isinstance(dataset_warnings, list) else [])
        except Exception as exc:
            message = f"xG-LightGBM no pudo prepararse automaticamente ({exc.__class__.__name__}: {exc})."
            result["sources"]["xg_lightgbm_model"]["status"] = "fallback"
            result["sources"]["xg_lightgbm_model"]["error"] = normalize_report_message(message)
            warnings.append(message)
    emit_job_progress(progress_callback, "source_preflight", 4, 4, "Fuentes resueltas")
    warning_payload = public_warning_payload(warnings, pipeline_mode=pipeline_mode)
    result.update(warning_payload)
    if any((source or {}).get("error") for source in result["sources"].values() if isinstance(source, dict)):
        result["status"] = "partial"
        result["status_label"] = "Fuentes parciales"
    elif result["visible_warnings"]:
        result["status"] = "fallback"
        result["status_label"] = "Fallback disponible"
    return json_safe(result)


def predict_upcoming_report(payload: Dict[str, Any] | None = None, progress_callback=None) -> Dict[str, Any]:
    payload = payload or {}
    pipeline_mode = normalize_report_pipeline_mode(payload.get("pipeline_mode"))
    config = report_pipeline_config(payload, pipeline_mode)
    start_time = time.monotonic()
    hardware = stat_report_hardware(
        config.get("sota_device", "auto"),
        pipeline_mode,
        config.get("sota_calculation_mode", "exact"),
    )
    config["score_backend"] = str(hardware.get("score_backend") or "numpy")
    emit_report_progress(
        progress_callback,
        stage="preparing",
        start_time=start_time,
        model_index=0,
        model_total=1,
        model_key="",
        fixture_index=0,
        fixture_total=1,
        hardware=hardware,
        message="Preparando fixtures y modelo base",
    )
    source_preflight = resolve_worldcup_sources_for_pipeline(
        payload=payload,
        config=config,
        pipeline_mode=pipeline_mode,
        progress_callback=progress_callback,
    )
    config["_source_preflight"] = source_preflight
    if pipeline_mode == ALTERNATIVES_BENCHMARK_PIPELINE_MODE:
        return alternatives_benchmark_report(
            payload=payload,
            config=config,
            start_time=start_time,
            hardware=hardware,
            progress_callback=progress_callback,
        )
    if pipeline_mode == ADVANCED_MODELS_PIPELINE_MODE:
        return advanced_models_report(
            payload=payload,
            config=config,
            start_time=start_time,
            hardware=hardware,
            progress_callback=progress_callback,
        )
    if pipeline_mode == XG_LIGHTGBM_PIPELINE_MODE:
        return xg_lightgbm_report(
            payload=payload,
            config=config,
            start_time=start_time,
            hardware=hardware,
            progress_callback=progress_callback,
        )
    tournament, fixture_source = load_tournament_2026(refresh=bool(config["refresh"]))
    results_autorefresh = ensure_worldcup_results_autorefreshed_once(tournament)
    base_model, history_source = build_model(tournament, config)
    feature_history_df, _ = score_history_for_tournament(tournament, config)
    feature_source = benchmark_feature_source(tournament, feature_history_df, config)
    limit = int(_clamp_int(payload.get("limit", 8), 1, 72))
    group_filter = str(payload.get("group") or "").strip()
    fixture_df = upcoming_fixture_rows(tournament, group_filter=group_filter).head(limit).copy()
    fixture_records = [fixture for _, fixture in fixture_df.iterrows()]
    model_sequence = SOTA_SCORE_MODEL_SEQUENCE
    fixture_reports = upcoming_sota_fixture_reports(
        tournament=tournament,
        base_model=base_model,
        fixtures=fixture_records,
        config=config,
        start_time=start_time,
        hardware=hardware,
        model_sequence=model_sequence,
        history_df=feature_history_df,
        feature_source=feature_source,
        progress_callback=progress_callback,
    )
    monte_carlo_seed_rng = (
        np.random.default_rng(int(config["seed"]))
        if pipeline_mode == "poisson_sota" and config.get("sota_calculation_mode") == "monte_carlo"
        else None
    )
    for report in fixture_reports:
        report["consensus"] = fixture_consensus(report.get("models", []))
        report.update(fixture_model_analysis(report.get("models", [])))
        report["sota_calculation_mode"] = config.get("sota_calculation_mode", "exact")
        report["sota_calculation_label"] = sota_calculation_summary(config)
        if monte_carlo_seed_rng is not None:
            report["monte_carlo_consensus"] = monte_carlo_consensus_from_distribution(
                distribution=report.get("consensus_score_distribution", {}),
                fixture=report.get("fixture", {}),
                config=config,
                hardware=hardware,
                seed=int(monte_carlo_seed_rng.integers(1, np.iinfo(np.int32).max)),
            )
        report["warnings"] = fixture_report_warnings(report)
        if report.get("monte_carlo_consensus"):
            report["warnings"] = unique_strings([
                *report["warnings"],
                *[str(item) for item in (report.get("monte_carlo_consensus") or {}).get("warnings", []) if str(item)],
            ])
    table = table_payload(pd.DataFrame(upcoming_report_table_rows(fixture_reports)), page=1, page_size=max(limit * 12, 1))
    backtest = alternatives_backtest_report(
        history_df=feature_history_df,
        tournament=tournament,
        config={**config, "benchmark_tuning_enabled": False},
        model_sequence=model_sequence,
        start_time=start_time,
        hardware=hardware,
        feature_source=feature_source,
        progress_callback=progress_callback,
    )
    backtests = backtest.get("models", [])
    statistical_audit = build_prediction_statistical_audit(backtests, baseline_key=DEFAULT_SCORE_MODEL)
    best_model = best_alternative_from_backtests(backtests)
    backtest_summary = backtest.get("summary", {})
    generated_at = str(backtest_summary.get("generated_at") or _now_utc().isoformat())
    backtest_range = backtest_summary.get("backtest_range") or empty_backtest_range(generated_at)
    raw_warnings = unique_strings([*hardware.get("warnings", []), *feature_source.warnings, *backtest.get("warnings", [])])
    warning_payload = public_warning_payload(raw_warnings, pipeline_mode=pipeline_mode)
    summary = {
        "pipeline_mode": pipeline_mode,
        "pipeline_label": "Poisson + SOTA",
        "generated_at": generated_at,
        "requested": limit,
        "returned": len(fixture_reports),
        "group": group_filter or "Todos",
        "fixture_source": fixture_source,
        "history_source": history_source,
        "poisson_recent_matches": config["poisson_recent_matches"],
        "backtest_last_n": int(config["backtest_last_n"]),
        "backtest_auto_n": int(backtest_summary.get("evaluated_matches") or backtest_summary.get("confirmed_matches") or 0),
        "backtest_scope": backtest_summary.get("scope", config.get("backtest_scope", "")),
        "backtest_source": backtest_summary.get("source", ""),
        "backtest_confirmed_matches": backtest_summary.get("confirmed_matches_detail", []),
        "backtest_range": backtest_range,
        "anti_leakage": backtest_summary.get("anti_leakage", ""),
        "iterations": config["iterations"],
        "seed": config["seed"],
        "bayes_profile": config.get("bayes_profile", ""),
        "sota_device": config.get("sota_device", "auto"),
        "sota_calculation_mode": config.get("sota_calculation_mode", "exact"),
        "sota_calculation_label": sota_calculation_summary(config),
        "monte_carlo_iterations": config["iterations"] if config.get("sota_calculation_mode") == "monte_carlo" else 0,
        "score_models": model_sequence,
        "pipeline_steps": sota_pipeline_steps(config),
        "feature_source_warnings": unique_strings(feature_source.warnings),
        "hardware": hardware,
        "results_autorefresh": results_autorefresh,
        "best_model": best_model,
        "statistical_audit": {
            "available": statistical_audit.get("available", False),
            "evaluated_models": statistical_audit.get("evaluated_models", 0),
            "evaluated_matches": statistical_audit.get("evaluated_matches", 0),
            "baseline_model_key": statistical_audit.get("baseline_model_key", DEFAULT_SCORE_MODEL),
            "recommendations": statistical_audit.get("recommendations", []),
            "warnings": statistical_audit.get("warnings", []),
        },
        "backtest": backtest_summary,
        "warnings": warning_payload["visible_warnings"],
        "visible_warnings": warning_payload["visible_warnings"],
        "technical_warnings": warning_payload["technical_warnings"],
        "config": public_report_config(config),
    }
    report = persist_upcoming_report({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "fixture_reports": fixture_reports,
        "ranked_models": benchmark_models_with_backtests(backtests),
        "model_backtests": backtests,
        "statistical_audit": statistical_audit,
        "best_model": best_model,
        "backtest": backtest,
        "table": table,
    })
    emit_report_progress(
        progress_callback,
        stage="complete",
        start_time=start_time,
        model_index=len(summary["score_models"]),
        model_total=max(len(summary["score_models"]), 1),
        model_key="",
        fixture_index=len(fixture_reports),
        fixture_total=max(len(fixture_reports), 1),
        hardware=hardware,
        message="Reporte guardado",
        force_complete=True,
    )
    return report


def xg_lightgbm_report(
        payload: Dict[str, Any],
        config: Dict[str, Any],
        start_time: float,
        hardware: Dict[str, Any],
        progress_callback=None,
) -> Dict[str, Any]:
    model_id, model_meta, model_warnings = resolve_xg_lightgbm_model(payload)
    tournament, fixture_source = load_tournament_2026(refresh=bool(config["refresh"]))
    results_autorefresh = ensure_worldcup_results_autorefreshed_once(tournament)
    base_model, history_source = build_model(tournament, config)
    base_model = apply_recent_context_model(base_model, config)
    history_df, _ = score_history_for_tournament(tournament, config)
    feature_source = benchmark_feature_source(tournament, history_df, config)
    limit = int(_clamp_int(payload.get("limit", 8), 1, 72))
    group_filter = str(payload.get("group") or "").strip()
    fixture_df = upcoming_fixture_rows(tournament, group_filter=group_filter).head(limit).copy()
    fixture_records = [fixture for _, fixture in fixture_df.iterrows()]
    fixture_total = max(len(fixture_records), 1)
    fixture_reports: List[Dict[str, Any]] = []
    for index, fixture in enumerate(fixture_records, start=1):
        emit_report_progress(
            progress_callback,
            stage="predicting",
            start_time=start_time,
            model_index=1,
            model_total=1,
            model_key=XG_LIGHTGBM_PIPELINE_MODE,
            fixture_index=index - 1,
            fixture_total=fixture_total,
            hardware=hardware,
            message=f"xG-LightGBM {index}/{fixture_total}",
        )
        prediction = predict_match_payload(
            tournament,
            base_model,
            fixture_id=fixture.get("No.", ""),
            home=fixture.get("Equipo 1", ""),
            away=fixture.get("Equipo 2", ""),
            use_ml_model=True,
            ml_weight=1.0,
            model_id=model_id,
            poisson_recent_matches=int(config.get("poisson_recent_matches") or DEFAULT_CONFIG["poisson_recent_matches"]),
        )
        fixture_reports.append(xg_lightgbm_fixture_report(prediction))
    table_rows = xg_lightgbm_report_table_rows(fixture_reports)
    table = table_payload(pd.DataFrame(table_rows), page=1, page_size=max(len(table_rows), 1))
    xg_backtest = xg_lightgbm_backtest_report(
        history_df=history_df,
        tournament=tournament,
        model_id=model_id,
        model_meta=model_meta,
        config=config,
        start_time=start_time,
        hardware=hardware,
        progress_callback=progress_callback,
    )
    sota_backtest = alternatives_backtest_report(
        history_df=history_df,
        tournament=tournament,
        config={**config, "benchmark_tuning_enabled": False, "sota_calculation_mode": "exact"},
        model_sequence=SOTA_SCORE_MODEL_SEQUENCE,
        start_time=start_time,
        hardware=hardware,
        feature_source=feature_source,
        progress_callback=progress_callback,
    )
    xg_models = xg_backtest.get("models", [])
    sota_models = sota_backtest.get("models", [])
    combined_backtests = rank_backtest_models([*xg_models, *sota_models], xg_backtest.get("summary", {}) or sota_backtest.get("summary", {}))
    statistical_audit = build_prediction_statistical_audit(combined_backtests, baseline_key=DEFAULT_SCORE_MODEL)
    best_model = best_alternative_from_backtests(combined_backtests)
    backtest_summary = xg_backtest.get("summary", {}) if xg_models else sota_backtest.get("summary", {})
    generated_at = str(backtest_summary.get("generated_at") or _now_utc().isoformat())
    backtest_range = backtest_summary.get("backtest_range") or empty_backtest_range(generated_at)
    fixture_warnings = [
        warning
        for report in fixture_reports
        for warning in report.get("warnings", [])
    ]
    model_summary = xg_lightgbm_model_summary(model_meta)
    source_preflight = config.get("_source_preflight", {}) if isinstance(config.get("_source_preflight", {}), dict) else {}
    raw_warnings = unique_strings([
        *model_warnings,
        *hardware.get("warnings", []),
        *fixture_warnings,
        *feature_source.warnings,
        *xg_backtest.get("warnings", []),
        *sota_backtest.get("warnings", []),
    ])
    warning_payload = merge_warning_payloads(
        public_warning_payload(raw_warnings, pipeline_mode=XG_LIGHTGBM_PIPELINE_MODE),
        source_preflight,
    )
    summary = {
        "pipeline_mode": XG_LIGHTGBM_PIPELINE_MODE,
        "pipeline_label": XG_LIGHTGBM_PIPELINE_LABEL,
        "requested": limit,
        "returned": len(fixture_reports),
        "group": group_filter or "Todos",
        "fixture_source": fixture_source,
        "history_source": history_source,
        "poisson_recent_matches": config["poisson_recent_matches"],
        "generated_at": generated_at,
        "backtest_last_n": int(config["backtest_last_n"]),
        "backtest_auto_n": int(backtest_summary.get("evaluated_matches") or backtest_summary.get("confirmed_matches") or 0),
        "backtest_scope": backtest_summary.get("scope", config.get("backtest_scope", "")),
        "backtest_source": backtest_summary.get("source", ""),
        "backtest_confirmed_matches": backtest_summary.get("confirmed_matches_detail", []),
        "backtest_range": backtest_range,
        "anti_leakage": backtest_summary.get("anti_leakage", ""),
        "iterations": 0,
        "seed": config["seed"],
        "sota_device": "not_applicable",
        "sota_calculation_mode": "not_applicable",
        "sota_calculation_label": "Bundle ML xG-LightGBM",
        "monte_carlo_iterations": 0,
        "score_models": [XG_LIGHTGBM_PIPELINE_MODE],
        "model_id": model_id,
        "model": model_summary,
        "model_device": model_summary.get("hardware", {}),
        "hardware": hardware,
        "results_autorefresh": results_autorefresh,
        "source_preflight": source_preflight,
        "best_model": best_model,
        "backtest": backtest_summary,
        "xg_backtest": xg_backtest.get("summary", {}),
        "sota_backtest": sota_backtest.get("summary", {}),
        "statistical_audit": {
            "available": statistical_audit.get("available", False),
            "evaluated_models": statistical_audit.get("evaluated_models", 0),
            "evaluated_matches": statistical_audit.get("evaluated_matches", 0),
            "baseline_model_key": statistical_audit.get("baseline_model_key", DEFAULT_SCORE_MODEL),
            "recommendations": statistical_audit.get("recommendations", []),
            "warnings": statistical_audit.get("warnings", []),
        },
        "warnings": warning_payload["visible_warnings"],
        "visible_warnings": warning_payload["visible_warnings"],
        "technical_warnings": warning_payload["technical_warnings"],
        "config": public_report_config(config),
    }
    report = persist_upcoming_report({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "fixture_reports": fixture_reports,
        "model_backtests": combined_backtests,
        "xg_backtest": xg_backtest,
        "sota_backtest": sota_backtest,
        "statistical_audit": statistical_audit,
        "best_model": best_model,
        "ranked_models": benchmark_models_with_backtests(combined_backtests),
        "table": table,
    })
    emit_report_progress(
        progress_callback,
        stage="complete",
        start_time=start_time,
        model_index=1,
        model_total=1,
        model_key=XG_LIGHTGBM_PIPELINE_MODE,
        fixture_index=len(fixture_reports),
        fixture_total=fixture_total,
        hardware=hardware,
        message="Reporte xG-LightGBM guardado",
        force_complete=True,
    )
    return report


def active_advanced_score_model_sequence(config: Dict[str, Any]) -> List[str]:
    sequence = list(ADVANCED_SCORE_MODEL_SEQUENCE)
    include_bayes = bool(config.get("advanced_include_bayesian")) or str(config.get("bayes_profile") or "").lower() == "deep"
    if include_bayes:
        sequence.extend(ADVANCED_HEAVY_SCORE_MODEL_SEQUENCE)
    return list(dict.fromkeys(sequence))


def advanced_models_report(
        payload: Dict[str, Any],
        config: Dict[str, Any],
        start_time: float,
        hardware: Dict[str, Any],
        progress_callback=None,
) -> Dict[str, Any]:
    tournament, fixture_source = load_tournament_2026(refresh=bool(config["refresh"]))
    results_autorefresh = ensure_worldcup_results_autorefreshed_once(tournament)
    history_df, history_source = score_history_for_tournament(tournament, config)
    feature_source = benchmark_feature_source(tournament, history_df, config)
    data_status = config.get("_advanced_data_status") if isinstance(config.get("_advanced_data_status"), dict) else advanced_data_status()
    model_sequence = active_advanced_score_model_sequence(config)
    group_map = groups_from_tournament(tournament)
    team_names = [team for group_teams in group_map.values() for team in group_teams]
    base_model = WorldCupModel.from_history(
        history_df,
        teams=team_names,
        history_weight=float(config["history_weight"]),
        recency_weight=float(config["recency_weight"]),
        host_advantage=float(config["host_advantage"]),
        max_goals=int(config["max_goals"]),
    )
    limit = int(_clamp_int(payload.get("limit", 8), 1, 72))
    group_filter = str(payload.get("group") or "").strip()
    fixture_df = upcoming_fixture_rows(tournament, group_filter=group_filter).head(limit).copy()
    fixture_records = [fixture for _, fixture in fixture_df.iterrows()]
    fixture_reports = upcoming_sota_fixture_reports(
        tournament=tournament,
        base_model=base_model,
        fixtures=fixture_records,
        config=config,
        start_time=start_time,
        hardware=hardware,
        model_sequence=model_sequence,
        history_df=history_df,
        feature_source=feature_source,
        progress_callback=progress_callback,
    )
    backtest = alternatives_backtest_report(
        history_df=history_df,
        tournament=tournament,
        config={**config, "benchmark_tuning_enabled": False},
        model_sequence=model_sequence,
        start_time=start_time,
        hardware=hardware,
        feature_source=feature_source,
        progress_callback=progress_callback,
    )
    backtests = backtest.get("models", [])
    statistical_audit = build_prediction_statistical_audit(backtests, baseline_key=DEFAULT_SCORE_MODEL)
    best_model = best_alternative_from_backtests(backtests)
    backtest_by_key = {str(item.get("model_key") or ""): item for item in backtests}
    ranked_model_keys = [str(item.get("model_key") or "") for item in backtests if str(item.get("model_key") or "")]
    fixture_reports = rank_fixture_report_models(fixture_reports, ranked_model_keys or model_sequence)
    for fixture_report, fixture in zip(fixture_reports, fixture_records):
        fixture_report["baseline_poisson"] = poisson_baseline_report_for_fixture(base_model, fixture, config)
        fixture_report["primary_model"] = primary_model_for_fixture(fixture_report, best_model, backtest_by_key)
        if not fixture_report["primary_model"].get("available") and fixture_report.get("models"):
            fixture_report["primary_model"] = dict(fixture_report["models"][0])
            fixture_report["primary_model"]["selection_policy"] = "Fallback visual: primer modelo avanzado disponible; auditoria aun no eligio ganador."
        strip_consensus_fields_from_alternative_report(fixture_report)
        fixture_report["warnings"] = fixture_report_warnings(fixture_report)
    ranked_models = advanced_models_with_backtests(backtests, data_status)
    table_rows = alternatives_benchmark_table_rows(fixture_reports, backtest_by_key)
    table = table_payload(pd.DataFrame(table_rows), page=1, page_size=max(len(table_rows), 1))
    backtest_summary = backtest.get("summary", {})
    generated_at = str(backtest_summary.get("generated_at") or _now_utc().isoformat())
    backtest_range = backtest_summary.get("backtest_range") or empty_backtest_range(generated_at)
    raw_warnings = unique_strings([
        *hardware.get("warnings", []),
        *data_status.get("warnings", []),
        *feature_source.warnings,
        *backtest.get("warnings", []),
        *[
            warning
            for report in fixture_reports
            for warning in report.get("warnings", [])
        ],
    ])
    source_preflight = config.get("_source_preflight", {}) if isinstance(config.get("_source_preflight", {}), dict) else {}
    warning_payload = merge_warning_payloads(
        public_warning_payload(raw_warnings, pipeline_mode=ADVANCED_MODELS_PIPELINE_MODE),
        source_preflight,
    )
    summary = {
        "pipeline_mode": ADVANCED_MODELS_PIPELINE_MODE,
        "pipeline_label": ADVANCED_MODELS_PIPELINE_LABEL,
        "evidence_policy": "advanced_models_local_backtest_vs_poisson",
        "generated_at": generated_at,
        "requested": limit,
        "returned": len(fixture_reports),
        "group": group_filter or "Todos",
        "fixture_source": fixture_source,
        "history_source": history_source,
        "poisson_recent_matches": config["poisson_recent_matches"],
        "backtest_last_n": int(config["backtest_last_n"]),
        "backtest_auto_n": int(backtest_summary.get("evaluated_matches") or backtest_summary.get("confirmed_matches") or 0),
        "backtest_scope": backtest_summary.get("scope", config.get("backtest_scope", "")),
        "backtest_range": backtest_range,
        "anti_leakage": backtest_summary.get("anti_leakage", data_status.get("anti_leakage", "")),
        "iterations": 0,
        "seed": config["seed"],
        "bayes_profile": config.get("bayes_profile", ""),
        "advanced_include_bayesian": bool(config.get("advanced_include_bayesian")),
        "sota_device": config.get("sota_device", "auto"),
        "sota_calculation_mode": "not_applicable",
        "sota_calculation_label": "Familias avanzadas exactas",
        "monte_carlo_iterations": 0,
        "score_models": model_sequence,
        "advanced_data_status": data_status,
        "advanced_models_catalog": advanced_models_catalog(data_status),
        "feature_research": worldcup_feature_research_summary(feature_source),
        "statistical_audit": {
            "available": statistical_audit.get("available", False),
            "evaluated_models": statistical_audit.get("evaluated_models", 0),
            "evaluated_matches": statistical_audit.get("evaluated_matches", 0),
            "baseline_model_key": statistical_audit.get("baseline_model_key", DEFAULT_SCORE_MODEL),
            "recommendations": statistical_audit.get("recommendations", []),
            "warnings": statistical_audit.get("warnings", []),
        },
        "hardware": hardware,
        "source_preflight": source_preflight,
        "warnings": warning_payload["visible_warnings"],
        "visible_warnings": warning_payload["visible_warnings"],
        "technical_warnings": warning_payload["technical_warnings"],
        "config": public_report_config(config),
        "best_model": best_model,
        "backtest": backtest_summary,
    }
    report = persist_upcoming_report({
        "created_at": generated_at,
        "summary": summary,
        "advanced_data_status": data_status,
        "advanced_models_catalog": summary["advanced_models_catalog"],
        "feature_research": summary["feature_research"],
        "fixture_reports": fixture_reports,
        "ranked_models": ranked_models,
        "model_backtests": backtests,
        "statistical_audit": statistical_audit,
        "best_model": best_model,
        "backtest": backtest,
        "table": table,
    })
    emit_report_progress(
        progress_callback,
        stage="complete",
        start_time=start_time,
        model_index=len(model_sequence),
        model_total=max(len(model_sequence), 1),
        model_key="",
        fixture_index=len(fixture_reports),
        fixture_total=max(len(fixture_reports), 1),
        hardware=hardware,
        message="Reporte avanzado guardado",
        force_complete=True,
    )
    return report


def advanced_models_with_backtests(backtests: List[Dict[str, Any]], data_status: Dict[str, Any]) -> List[Dict[str, Any]]:
    catalog_by_key = {str(item.get("key") or ""): item for item in advanced_models_catalog(data_status)}
    output = []
    for index, backtest in enumerate(backtests or [], start=1):
        key = str(backtest.get("model_key") or "")
        catalog = dict(catalog_by_key.get(key, {"key": key, "label": backtest.get("model_label", key)}))
        output.append({
            "rank": backtest.get("rank", index),
            "key": key,
            "model_name": catalog.get("label") or catalog.get("model_name") or backtest.get("model_label", key),
            "family": catalog.get("family", "advanced"),
            "description": catalog.get("detail", ""),
            "status": catalog.get("status", ""),
            "backtest": backtest,
        })
    if output:
        return output
    return [
        {
            "rank": index,
            "key": item.get("key", ""),
            "model_name": item.get("label", ""),
            "family": item.get("family", ""),
            "description": item.get("detail", ""),
            "status": item.get("status", ""),
            "backtest": {},
        }
        for index, item in enumerate(advanced_models_catalog(data_status), start=1)
    ]


def resolve_xg_lightgbm_model(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any], List[str]]:
    explicit_model_id = str(payload.get("model_id") or payload.get("xg_model_id") or "").strip()
    if explicit_model_id:
        meta = read_model_metadata(model_id=explicit_model_id)
        warnings = xg_lightgbm_model_warnings(meta, explicit_model_id, explicit=True)
        return explicit_model_id, meta, warnings
    active_meta = read_model_metadata()
    if active_meta.get("trained") and active_meta.get("model_profile") == XG_LIGHTGBM_PROFILE:
        return str(active_meta.get("model_id") or ""), active_meta, []
    default_id = default_model_id(XG_LIGHTGBM_PROFILE, "dual_markets")
    default_meta = read_model_metadata(model_id=default_id)
    if default_meta.get("trained") and default_meta.get("model_profile") == XG_LIGHTGBM_PROFILE:
        return default_id, default_meta, []
    warnings = xg_lightgbm_model_warnings(default_meta, default_id, explicit=False)
    if active_meta.get("trained") and active_meta.get("model_id") and active_meta.get("model_profile") != XG_LIGHTGBM_PROFILE:
        warnings.append(
            f"Modelo activo {active_meta.get('model_id')} ignorado: no tiene perfil xG-LightGBM."
        )
    return default_id, default_meta, unique_strings(warnings)


def xg_lightgbm_model_warnings(meta: Dict[str, Any], model_id: str, explicit: bool) -> List[str]:
    warnings = []
    if not meta.get("trained"):
        warnings.append(
            f"Bundle xG-LightGBM {model_id or 'default'} no entrenado; el pipeline usa respaldo Poisson."
        )
    elif meta.get("model_profile") != XG_LIGHTGBM_PROFILE:
        scope = "solicitado" if explicit else "detectado"
        warnings.append(
            f"Modelo {scope} {model_id or meta.get('model_id') or ''} no tiene perfil xG-LightGBM."
        )
    elif meta.get("market_mode") != "dual_markets":
        warnings.append(
            f"Modelo {model_id or meta.get('model_id') or ''} no es dual_markets; algunos U/O pueden venir de Poisson."
        )
    return unique_strings([*warnings, *list(meta.get("warnings") or [])])


def xg_lightgbm_model_summary(meta: Dict[str, Any]) -> Dict[str, Any]:
    hardware = meta.get("hardware") or {}
    return json_safe({
        "trained": bool(meta.get("trained")),
        "bundle": bool(meta.get("bundle")),
        "model_id": meta.get("model_id", ""),
        "model_name": meta.get("model_name", ""),
        "model_type": meta.get("model_type", ""),
        "model_profile": meta.get("model_profile", ""),
        "model_label": meta.get("model_label") or XG_LIGHTGBM_PIPELINE_LABEL,
        "market_mode": meta.get("market_mode", ""),
        "trained_at": meta.get("trained_at", ""),
        "train_rows": int(meta.get("train_rows", 0) or 0),
        "validation_rows": int(meta.get("validation_rows", 0) or 0),
        "test_rows": int(meta.get("test_rows", meta.get("prediction_rows", 0)) or 0),
        "metrics": meta.get("metrics", {}),
        "confusion_matrix": meta.get("confusion_matrix", {}),
        "tuning": meta.get("tuning", {}),
        "top_features": meta.get("top_features", [])[:20],
        "markets": meta.get("markets", {}),
        "market_models": meta.get("market_models", {}),
        "hardware": {
            "requested_device": hardware.get("requested_device") or hardware.get("device", ""),
            "actual_device": hardware.get("actual_device") or hardware.get("device_default", ""),
            "cuda_available": bool(hardware.get("cuda_available")),
            "cuda_devices": hardware.get("cuda_devices", []),
            "cuda_warning": hardware.get("cuda_warning", ""),
            "warnings": hardware.get("warnings", []),
        },
    })


def xg_lightgbm_fixture_report(prediction: Dict[str, Any]) -> Dict[str, Any]:
    fixture = report_fixture_payload(prediction.get("fixture") or {})
    probabilities = prediction.get("probabilities") or {}
    outcome = outcome_decision(probabilities)
    decision = {
        "outcome": outcome,
        "label": outcome_label(outcome),
        "team": outcome_team(outcome, fixture),
    }
    model_probs = prediction.get("model_probs") or {}
    warnings = list(prediction.get("notes") or [])
    if not model_probs.get("ml"):
        warnings.append("1X2 ML no disponible; probabilidades 1X2 salen de respaldo Poisson.")
    if not model_probs.get("over_under_ml"):
        warnings.append("U/O ML no disponible; totales salen de respaldo Poisson.")
    return {
        "fixture": fixture,
        "prediction": prediction.get("prediction", ""),
        "probabilities": probabilities,
        "decision": decision,
        "totals": total_decisions(probabilities),
        "expected_goals": prediction.get("expected_goals", {}),
        "modal_score": prediction.get("modal_score", ""),
        "model_probs": model_probs,
        "market_readout": prediction.get("market_readout", {}),
        "data_quality": prediction.get("data_quality", {}),
        "contextual_poisson": prediction.get("contextual_poisson", {}),
        "warnings": unique_strings(warnings),
    }


def xg_lightgbm_report_table_rows(fixture_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in fixture_reports:
        fixture = report.get("fixture", {})
        probabilities = report.get("probabilities", {})
        expected = report.get("expected_goals", {})
        model_probs = report.get("model_probs", {})
        decision = report.get("decision", {})
        rows.append({
            "No.": fixture.get("id", ""),
            "Fecha": fixture.get("date", ""),
            "Grupo": fixture.get("group", ""),
            "Partido": fixture.get("label", ""),
            "Pipeline": XG_LIGHTGBM_PIPELINE_LABEL,
            "Modelo": model_probs.get("model_name", "") or model_probs.get("model_id", ""),
            "Pick": f"{decision.get('label', '')} {decision.get('team', '')}".strip(),
            "Top score": report.get("modal_score", ""),
            "Lambda Local": expected.get("home", ""),
            "Lambda Visita": expected.get("away", ""),
            "Peso ML 1X2": model_probs.get("result_weight", ""),
            "Peso ML U/O": model_probs.get("over_under_weight", ""),
            "1 %": probabilities.get("home", ""),
            "X %": probabilities.get("draw", ""),
            "2 %": probabilities.get("away", ""),
            "O0.5": probabilities.get("over05", ""),
            "U0.5": probabilities.get("under05", ""),
            "O1.5": probabilities.get("over15", ""),
            "U1.5": probabilities.get("under15", ""),
            "O2.5": probabilities.get("over25", ""),
            "U2.5": probabilities.get("under25", ""),
            "O3.5": probabilities.get("over35", ""),
            "U3.5": probabilities.get("under35", ""),
            "Warnings": " | ".join(report.get("warnings", [])),
        })
    return rows


def xg_lightgbm_training_status() -> Dict[str, Any]:
    dataset = worldcup_training_dataset_status()
    options = training_options()
    catalog = list_worldcup_models()
    default_id = default_model_id(XG_LIGHTGBM_PROFILE, "dual_markets")
    active_model = read_model_metadata()
    default_model = read_model_metadata(model_id=default_id)
    xg_models = [
        model for model in catalog.get("models", [])
        if model.get("model_profile") == XG_LIGHTGBM_PROFILE
    ]
    selected_model = active_model if active_model.get("model_profile") == XG_LIGHTGBM_PROFILE else default_model
    if not selected_model.get("trained") and xg_models:
        selected_model = xg_models[0]
    required_markets = [
        {"key": "result", "label": "1X2"},
        {"key": "over_under_05", "label": "U/O 0.5"},
        {"key": "over_under_15", "label": "U/O 1.5"},
        {"key": "over_under_25", "label": "U/O 2.5"},
        {"key": "over_under_35", "label": "U/O 3.5"},
    ]
    optional_markets = [{"key": "goals_distribution", "label": "Distribucion goles"}] if dataset.get("prepared_goals_distribution_ready") else []
    defaults = xg_lightgbm_training_payload({})
    split = {
        "policy": dataset.get("split_policy") or "temporal_80_10_10",
        "train_rows": int(dataset.get("train_rows", 0) or 0),
        "validation_rows": int(dataset.get("validation_rows", 0) or 0),
        "test_rows": int(dataset.get("test_rows", 0) or 0),
        "eval_strategy": dataset.get("eval_strategy", ""),
        "training_start_year": dataset.get("training_start_year", ""),
        "team_scope_policy": dataset.get("training_team_scope_policy", ""),
        "team_scope_count": int(dataset.get("training_scope_team_count", 0) or 0),
        "raw_international_source_rows": int(dataset.get("raw_international_source_rows", 0) or 0),
        "date_scoped_international_source_rows": int(dataset.get("date_scoped_international_source_rows", 0) or 0),
        "team_scoped_international_source_rows": int(dataset.get("team_scoped_international_source_rows", 0) or 0),
        "removed_outside_team_scope_rows": int(dataset.get("removed_outside_team_scope_rows", 0) or 0),
        "removed_outside_team_scope_label_rows": int(dataset.get("removed_outside_team_scope_label_rows", 0) or 0),
        "max_label_date": dataset.get("max_label_date", ""),
        "max_label_cutoff": dataset.get("max_label_cutoff", ""),
        "label_source": dataset.get("prepared_label_source", ""),
    }
    planned_market_count = len(required_markets) + len(optional_markets)
    trials_per_market = int(defaults.get("n_trials", 12) or 12) if defaults.get("tuning_enabled") else 0
    can_train = bool(dataset.get("etl_ready") and not dataset.get("etl_stale") and dataset.get("trainable") and split["train_rows"] and split["validation_rows"] and split["test_rows"])
    return json_safe({
        "title": XG_LIGHTGBM_PIPELINE_LABEL,
        "pipeline_mode": XG_LIGHTGBM_PIPELINE_MODE,
        "procedure": xg_lightgbm_training_procedure(),
        "dataset": dataset,
        "split": split,
        "can_train": can_train,
        "defaults": defaults,
        "options": {
            "profile": next((profile for profile in options.get("model_profiles", []) if profile.get("key") == XG_LIGHTGBM_PROFILE), {}),
            "feature_profiles": options.get("feature_profiles", []),
            "hardware": options.get("hardware", {}),
            "samplers": ["tpe", "random", "cmaes"],
            "pruners": ["none", "median", "successive-halving"],
            "objectives": options.get("objectives", list(TRAINING_OBJECTIVES)),
            "calibration_methods": options.get("calibration_methods", ["sigmoid", "isotonic"]),
            "feature_selection_modes": options.get("feature_selection_modes", [FEATURE_SELECTION_FAMILY_BALANCED, FEATURE_SELECTION_SUPERVISED_MODEL]),
            "required_markets": required_markets,
            "optional_markets": optional_markets,
            "planned_market_count": planned_market_count,
            "default_trials_per_market": trials_per_market,
            "default_total_trial_budget": planned_market_count * trials_per_market,
        },
        "model": xg_lightgbm_model_summary(selected_model),
        "models": xg_models,
        "active_model_is_xg": bool(active_model.get("model_profile") == XG_LIGHTGBM_PROFILE),
        "default_model_id": default_id,
        "anti_leakage": "Features historicas, recent15, API/xG y mercados se calculan as-of antes de la fecha del partido; Optuna usa validation temporal y test queda bloqueado.",
    })


def advanced_data_status() -> Dict[str, Any]:
    status = worldcup_advanced_data_status()
    status["models"] = advanced_models_catalog(status)
    return json_safe(status)


def advanced_data_prepare(payload: Dict[str, Any] | None = None, progress_callback=None) -> Dict[str, Any]:
    status = prepare_worldcup_advanced_data(payload or {}, progress_callback=progress_callback)
    status["models"] = advanced_models_catalog(status)
    return json_safe(status)


def advanced_models_catalog(status: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    status = status or worldcup_advanced_data_status()
    family_status = {str(item.get("key") or ""): item for item in status.get("families", [])}
    xg_active = (family_status.get("xg_shot_quality") or {}).get("status") in {"active", "cached"}
    return [
        {
            "key": XG_DIXON_COLES_MODEL,
            "label": score_model_display_label(XG_DIXON_COLES_MODEL),
            "family": "xg_dixon_coles",
            "status": "active" if xg_active else "missing_data",
            "detail": "xG agregado por partido como lambda y correccion de marcadores bajos.",
        },
        {
            "key": NEGATIVE_BINOMIAL_DIXON_COLES_MODEL,
            "label": score_model_display_label(NEGATIVE_BINOMIAL_DIXON_COLES_MODEL),
            "family": "overdispersed_counts",
            "status": "active",
            "detail": "NB2 para sobredispersion + rho Dixon-Coles.",
        },
        {
            "key": DYNAMIC_STRENGTH_KALMAN_MODEL,
            "label": score_model_display_label(DYNAMIC_STRENGTH_KALMAN_MODEL),
            "family": "dynamic_strength",
            "status": "active",
            "detail": "Estado latente ligero de ataque/defensa con recencia.",
        },
        {
            "key": STACKED_META_MNLOGIT_MODEL,
            "label": score_model_display_label(STACKED_META_MNLOGIT_MODEL),
            "family": "stacking",
            "status": "active",
            "detail": "Meta-modelo MNLogit sobre probabilidades base y lambdas.",
        },
        {
            "key": "bayesian_dynamic_poisson",
            "label": score_model_display_label("bayesian_dynamic_poisson"),
            "family": "bayesian_dynamic",
            "status": "optional_heavy",
            "detail": "PyMC dinamico disponible solo con perfil profundo o bandera explicita.",
        },
    ]


def xg_lightgbm_training_procedure() -> Dict[str, Any]:
    return {
        "title": "Procedimiento xG-LightGBM Mundial 2026",
        "steps": [
            {"name": "Preparar ETL", "detail": "Construye artifact internacional con labels 1X2, U/O 0.5-3.5 y split temporal 80/10/10."},
            {"name": "Features sin leakage", "detail": "Calcula Elo, forma recent15, xG, tiros, mercado y API-Football solo con informacion anterior al partido."},
            {"name": "Fine-tuning Optuna", "detail": "Optimiza LightGBM por mercado usando exclusivamente validation temporal; el test no participa en seleccion."},
            {"name": "Fit final", "detail": "Reentrena cada mercado con train + validation, calibra solo con datos pre-test y conserva test para reporte final."},
            {"name": "Reporte profesional", "detail": "Guarda metricas raw/calibradas, matrices de confusion, top features, device usado y warnings por mercado."},
            {"name": "Activacion", "detail": "Guarda el bundle dual_markets como modelo activo para el pipeline de predicciones xG-LightGBM."},
        ],
    }


def sota_pipeline_steps(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "name": "Fixtures futuros",
            "detail": "Carga calendario Mundial 2026, resultados confirmados y filtro de grupo/limite solicitado.",
        },
        {
            "name": "Poisson base",
            "detail": (
                f"Construye rating ofensivo/defensivo con historico mundialista y max_goals={int(config.get('max_goals') or DEFAULT_CONFIG['max_goals'])}."
            ),
        },
        {
            "name": "Contexto recent15",
            "detail": (
                f"Ajusta lambdas con ultimos {int(config.get('poisson_recent_matches') or DEFAULT_CONFIG['poisson_recent_matches'])} partidos disponibles antes del fixture."
            ),
        },
        {
            "name": "Features as-of",
            "detail": "Genera historial, H2H, mercado, xG/API-Football, recent15, XI/jugadores y stage sin usar datos posteriores al partido.",
        },
        {
            "name": "Modelos SOTA",
            "detail": "Evalua Poisson independiente, Dixon-Coles MLE y bivariado Poisson MLE sobre el mismo fixture.",
        },
        {
            "name": "Matriz de marcador",
            "detail": "Calcula P(goles_local, goles_visita), 1X2, U/O 0.5-3.5 y top marcadores por modelo.",
        },
        {
            "name": "Consenso",
            "detail": "Promedia matrices/probabilidades elegibles; si se elige Monte Carlo, simula sobre la matriz consenso.",
        },
        {
            "name": "Reporte",
            "detail": "Entrega tarjetas cliente, detalle tecnico, matrices P, lista de features y trazabilidad de fuentes.",
        },
    ]


def xg_lightgbm_training_payload(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = payload or {}
    sampler = str(payload.get("optuna_sampler") or "tpe").strip().lower()
    if sampler not in {"tpe", "random", "cmaes"}:
        sampler = "tpe"
    pruner = str(payload.get("optuna_pruner") or "none").strip().lower()
    if pruner not in {"none", "median", "successive-halving"}:
        pruner = "none"
    objective = str(payload.get("objective") or "PredictiveScore").strip() or "PredictiveScore"
    if objective not in set(TRAINING_OBJECTIVES):
        objective = "PredictiveScore"
    calibration_method = str(payload.get("calibration_method") or "sigmoid").strip().lower()
    if calibration_method not in {"sigmoid", "isotonic"}:
        calibration_method = "sigmoid"
    feature_selection_mode = str(payload.get("feature_selection_mode") or FEATURE_SELECTION_FAMILY_BALANCED).strip().lower().replace("-", "_")
    if feature_selection_mode not in {FEATURE_SELECTION_FAMILY_BALANCED, FEATURE_SELECTION_SUPERVISED_MODEL}:
        feature_selection_mode = FEATURE_SELECTION_FAMILY_BALANCED
    device = str(payload.get("device") or "auto").strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        device = "auto"
    default_id = default_model_id(XG_LIGHTGBM_PROFILE, "dual_markets")
    return {
        "model_profile": XG_LIGHTGBM_PROFILE,
        "model_type": "lightgbm",
        "model_id": str(payload.get("model_id") or default_id).strip() or default_id,
        "model_name": str(payload.get("model_name") or "xG-LightGBM Mundial 2026").strip() or "xG-LightGBM Mundial 2026",
        "training_target": "result",
        "target": "result",
        "market_mode": "dual_markets",
        "feature_profile": str(payload.get("feature_profile") or "balanced").strip() or "balanced",
        "max_features": int(_clamp_int(payload.get("max_features", 450), 120, 1200)),
        "device": device,
        "n_jobs": int(_clamp_int(payload.get("n_jobs", -1), -1, 128)),
        "tuning_enabled": bool(payload.get("tuning_enabled", True)),
        "n_trials": int(_clamp_int(payload.get("n_trials", 12), 1, 100)),
        "optuna_sampler": sampler,
        "optuna_pruner": pruner,
        "objective": objective,
        "tune_params": str(payload.get("tune_params") or "all").strip() or "all",
        "calibration_enabled": bool(payload.get("calibration_enabled", True)),
        "calibration_method": calibration_method,
        "feature_selection_mode": feature_selection_mode,
        "seed": int(_clamp_int(payload.get("seed", 2026), 1, 999999)),
        "refresh_history": bool(payload.get("refresh_history", False)),
        "feature_progress_every": int(_clamp_int(payload.get("feature_progress_every", 1000), 100, 5000)),
    }


def xg_lightgbm_prepare_training(payload: Dict[str, Any] | None = None, progress_callback=None) -> Dict[str, Any]:
    payload = payload or {}
    start_time = time.monotonic()
    emit_job_progress(progress_callback, "prepare_etl", 0, 2, "Preparando ETL xG-LightGBM")
    prepare_training_dataset(
        force=bool(payload.get("force", True)),
        refresh_history=bool(payload.get("refresh_history", False)),
    )
    emit_job_progress(
        progress_callback,
        "prepare_etl",
        1,
        2,
        "ETL preparado; recalculando estado",
        elapsed_seconds=round(time.monotonic() - start_time, 1),
    )
    status = xg_lightgbm_training_status()
    emit_job_progress(
        progress_callback,
        "complete",
        2,
        2,
        "ETL xG-LightGBM listo",
        elapsed_seconds=round(time.monotonic() - start_time, 1),
    )
    return status


def xg_lightgbm_train_model(payload: Dict[str, Any] | None = None, progress_callback=None) -> Dict[str, Any]:
    payload = xg_lightgbm_training_payload(payload)
    tournament, fixture_source = load_tournament_2026(refresh=False)
    result = train_hybrid_model(tournament, payload=payload, progress_callback=progress_callback)
    status = xg_lightgbm_training_status()
    return {
        "fixture_source": fixture_source,
        "payload": payload,
        "training": result,
        "status": status,
        "model": status.get("model", {}),
    }


def alternatives_benchmark_report(
        payload: Dict[str, Any],
        config: Dict[str, Any],
        start_time: float,
        hardware: Dict[str, Any],
        progress_callback=None,
) -> Dict[str, Any]:
    global _WORLD_CUP_FIXTURES_AUTO_REFRESH_EXPIRES_AT
    now = _now_utc()
    refresh_fixtures = bool(config["refresh"]) or _world_cup_fixture_autorefresh_stale(now)
    if refresh_fixtures:
        _WORLD_CUP_FIXTURES_AUTO_REFRESH_EXPIRES_AT = _utcify_datetime(now)
    tournament, fixture_source = load_tournament_2026(refresh=refresh_fixtures)
    ensure_worldcup_results_autorefreshed_once(tournament)
    results_refresh = refresh_worldcup_2026_results(tournament, refresh=True)
    history_df, history_source = score_history_for_tournament(tournament, config)
    feature_source = benchmark_feature_source(tournament, history_df, config)
    model_sequence = list(BENCHMARK_SCORE_MODEL_SEQUENCE)
    tuning_summary = tune_benchmark_poisson_recent_matches(
        history_df=history_df,
        tournament=tournament,
        config=config,
        model_sequence=model_sequence,
        start_time=start_time,
        hardware=hardware,
        progress_callback=progress_callback,
    )
    if tuning_summary.get("best_poisson_recent_matches"):
        config = {
            **config,
            "poisson_recent_matches": int(tuning_summary["best_poisson_recent_matches"]),
            "benchmark_tuning": tuning_summary,
        }
    group_map = groups_from_tournament(tournament)
    team_names = [team for group_teams in group_map.values() for team in group_teams]
    base_model = WorldCupModel.from_history(
        history_df,
        teams=team_names,
        history_weight=float(config["history_weight"]),
        recency_weight=float(config["recency_weight"]),
        host_advantage=float(config["host_advantage"]),
        max_goals=int(config["max_goals"]),
    )
    limit = int(_clamp_int(payload.get("limit", 8), 1, 72))
    group_filter = str(payload.get("group") or "").strip()
    fixture_df = upcoming_fixture_rows(tournament, group_filter=group_filter).head(limit).copy()
    fixture_records = [fixture for _, fixture in fixture_df.iterrows()]
    fixture_reports = upcoming_sota_fixture_reports(
        tournament=tournament,
        base_model=base_model,
        fixtures=fixture_records,
        config=config,
        start_time=start_time,
        hardware=hardware,
        model_sequence=model_sequence,
        history_df=history_df,
        feature_source=feature_source,
        progress_callback=progress_callback,
    )

    backtest = alternatives_backtest_report(
        history_df=history_df,
        tournament=tournament,
        config=config,
        model_sequence=model_sequence,
        start_time=start_time,
        hardware=hardware,
        feature_source=feature_source,
        progress_callback=progress_callback,
    )
    backtests = backtest.get("models", [])
    statistical_audit = build_prediction_statistical_audit(backtests, baseline_key=DEFAULT_SCORE_MODEL)
    best_model = best_alternative_from_backtests(backtests)
    backtest_by_key = {str(item.get("model_key") or ""): item for item in backtests}
    ranked_model_keys = [str(item.get("model_key") or "") for item in backtests if str(item.get("model_key") or "")]
    fixture_reports = rank_fixture_report_models(fixture_reports, ranked_model_keys)
    for fixture_report, fixture in zip(fixture_reports, fixture_records):
        fixture_report["baseline_poisson"] = poisson_baseline_report_for_fixture(base_model, fixture, config)
        fixture_report["primary_model"] = primary_model_for_fixture(fixture_report, best_model, backtest_by_key)
        strip_consensus_fields_from_alternative_report(fixture_report)
        fixture_report["warnings"] = fixture_report_warnings(fixture_report)
    ranked_models = benchmark_models_with_backtests(backtests)
    alternatives = alternatives_with_backtests(sota_alternatives_catalog(), backtest_by_key)
    table_rows = alternatives_benchmark_table_rows(fixture_reports, backtest_by_key)
    table = table_payload(pd.DataFrame(table_rows), page=1, page_size=max(len(table_rows), 1))
    raw_warnings = unique_strings([
        *hardware.get("warnings", []),
        *results_refresh.get("warnings", []),
        *tuning_summary.get("warnings", []),
        *backtest.get("warnings", []),
        *[
            f"Conflicto resultado {item.get('date', '')} {item.get('home', '')} vs {item.get('away', '')}: "
            f"{item.get('existing_score', '')} -> {item.get('incoming_score', '')} ({item.get('resolved_source', '')})"
            for item in results_refresh.get("conflicts", [])
        ],
        *[
            warning
            for report in fixture_reports
            for warning in report.get("warnings", [])
        ],
    ])
    warning_payload = public_warning_payload(raw_warnings, pipeline_mode=ALTERNATIVES_BENCHMARK_PIPELINE_MODE)
    backtest_summary = backtest.get("summary", {})
    generated_at = str(backtest_summary.get("generated_at") or _now_utc().isoformat())
    backtest_range = backtest_summary.get("backtest_range") or empty_backtest_range(generated_at)
    summary = {
        "pipeline_mode": ALTERNATIVES_BENCHMARK_PIPELINE_MODE,
        "pipeline_label": ALTERNATIVES_BENCHMARK_LABEL,
        "evidence_policy": ALTERNATIVES_EVIDENCE_POLICY,
        "generated_at": generated_at,
        "requested": limit,
        "returned": len(fixture_reports),
        "group": group_filter or "Todos",
        "fixture_source": fixture_source,
        "result_source": results_refresh.get("source", ""),
        "results_refresh": results_refresh,
        "history_source": history_source,
        "poisson_recent_matches": config["poisson_recent_matches"],
        "benchmark_tuning": tuning_summary,
        "backtest_last_n": int(config["backtest_last_n"]),
        "backtest_auto_n": int(backtest_summary.get("evaluated_matches") or backtest_summary.get("confirmed_matches") or 0),
        "backtest_scope": backtest_summary.get("scope", config.get("backtest_scope", "")),
        "backtest_source": backtest_summary.get("source", results_refresh.get("source", "")),
        "backtest_confirmed_matches": backtest_summary.get("confirmed_matches_detail", []),
        "backtest_range": backtest_range,
        "anti_leakage": backtest_summary.get("anti_leakage", ""),
        "iterations": 0,
        "seed": config["seed"],
        "bayes_profile": config.get("bayes_profile", ""),
        "sota_device": config.get("sota_device", "auto"),
        "sota_calculation_mode": "not_applicable",
        "sota_calculation_label": "Modelos estadisticos individuales",
        "monte_carlo_iterations": 0,
        "score_models": ranked_model_keys or model_sequence,
        "baseline_model": {
            "key": DEFAULT_SCORE_MODEL,
            "label": score_model_display_label(DEFAULT_SCORE_MODEL),
            "role": "ranked_reference",
        },
        "hardware": hardware,
        "warnings": warning_payload["visible_warnings"],
        "visible_warnings": warning_payload["visible_warnings"],
        "technical_warnings": warning_payload["technical_warnings"],
        "config": public_report_config(config),
        "best_model": best_model,
        "backtest": backtest_summary,
        "statistical_audit": {
            "available": statistical_audit.get("available", False),
            "evaluated_models": statistical_audit.get("evaluated_models", 0),
            "evaluated_matches": statistical_audit.get("evaluated_matches", 0),
            "baseline_model_key": statistical_audit.get("baseline_model_key", DEFAULT_SCORE_MODEL),
            "recommendations": statistical_audit.get("recommendations", []),
            "warnings": statistical_audit.get("warnings", []),
        },
        "feature_research": worldcup_feature_research_summary(feature_source),
    }
    report = persist_upcoming_report({
        "created_at": generated_at,
        "summary": summary,
        "alternatives": alternatives,
        "ranked_models": ranked_models,
        "baseline_context": sota_baseline_context(),
        "feature_research": summary["feature_research"],
        "baseline": summary["baseline_model"],
        "fixture_reports": fixture_reports,
        "model_backtests": backtests,
        "statistical_audit": statistical_audit,
        "best_model": best_model,
        "backtest": backtest,
        "table": table,
    })
    emit_report_progress(
        progress_callback,
        stage="complete",
        start_time=start_time,
        model_index=len(model_sequence),
        model_total=max(len(model_sequence), 1),
        model_key="",
        fixture_index=len(fixture_reports),
        fixture_total=max(len(fixture_reports), 1),
        hardware=hardware,
        message="Benchmark estadistico guardado",
        force_complete=True,
    )
    return report


def poisson_baseline_report_for_fixture(
        base_model: WorldCupModel,
        fixture: pd.Series,
        config: Dict[str, Any],
        feature_source: BenchmarkFeatureSource | None = None,
        history_df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    prediction_model = apply_recent_context_model(base_model, config)
    prediction_model = apply_benchmark_feature_model(prediction_model, DEFAULT_SCORE_MODEL, feature_source, history_df)
    metadata = score_model_metadata(prediction_model)
    probabilities = model_probabilities_for_fixture(prediction_model, fixture, config)
    report = score_prediction_model_report(
        model_key=DEFAULT_SCORE_MODEL,
        metadata=metadata,
        probabilities=probabilities,
        fixture=fixture,
        config=config,
        already_percent=False,
    )
    score_distribution = score_distribution_for_fixture(prediction_model, fixture, probabilities, config)
    report["score_distribution"] = score_distribution
    report["top_scores"] = score_distribution.get("top_scores", [])
    if report["top_scores"]:
        report["top_score"] = report["top_scores"][0].get("score", report.get("top_score", ""))
        report["top_score_probability"] = report["top_scores"][0].get("probability", 0.0)
    report["heatmap"] = score_distribution.get("heatmap", {})
    report["source"] = "Poisson baseline"
    return report


def strip_consensus_fields_from_alternative_report(fixture_report: Dict[str, Any]) -> None:
    for model in [fixture_report.get("baseline_poisson", {}), *fixture_report.get("models", [])]:
        if isinstance(model, dict):
            model.pop("consensus_eligible", None)
            model.pop("signature", None)


def tune_benchmark_poisson_recent_matches(
        history_df: pd.DataFrame,
        tournament: Dict[str, Any],
        config: Dict[str, Any],
        model_sequence: List[str],
        start_time: float,
        hardware: Dict[str, Any],
        progress_callback=None,
) -> Dict[str, Any]:
    enabled = bool(config.get("benchmark_tuning_enabled", False))
    current_n = int(config.get("poisson_recent_matches") or DEFAULT_CONFIG["poisson_recent_matches"])
    summary: Dict[str, Any] = {
        "enabled": enabled,
        "available": False,
        "best_poisson_recent_matches": current_n,
        "best_value": None,
        "n_trials": int(config.get("benchmark_tuning_trials") or 0),
        "sampler": str(config.get("benchmark_tuning_sampler") or "tpe"),
        "objective": "mean_score_resultados",
        "scope": "all_active_models",
        "model_sequence": list(model_sequence),
        "trials": [],
        "warnings": [],
    }
    if not enabled:
        return summary
    if history_df is None or history_df.empty:
        summary["warnings"] = ["Optuna benchmark no disponible: historico vacio."]
        return summary
    required = {"Date", "Team 1", "Team 2", "G1", "G2"}
    if not required.issubset(history_df.columns):
        summary["warnings"] = ["Optuna benchmark no disponible: columnas historicas incompletas."]
        return summary
    confirmed_df = confirmed_worldcup_2026_backtest_rows(tournament)
    if confirmed_df.empty:
        summary["warnings"] = ["Optuna benchmark no disponible: no hay partidos 2026 finalizados."]
        return summary
    try:
        import optuna  # type: ignore  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        summary["warnings"] = [f"Optuna benchmark no disponible: {exc.__class__.__name__}."]
        return summary

    working = history_df.copy()
    working["_date"] = pd.to_datetime(working["Date"], errors="coerce")
    working = working[working["_date"].notna()].sort_values("_date", kind="stable").reset_index(drop=True)
    historical_train_df = working.drop(columns=["_date"]).copy()
    n_trials = int(_clamp_int(config.get("benchmark_tuning_trials", 20), 1, 100))
    summary["n_trials"] = n_trials
    sampler_name = str(config.get("benchmark_tuning_sampler") or "tpe")
    study = optuna.create_study(
        direction="maximize",
        sampler=benchmark_optuna_sampler(optuna, sampler_name, int(config.get("seed") or DEFAULT_CONFIG["seed"])),
    )

    def objective(trial) -> float:
        recent_n = int(trial.suggest_int("poisson_recent_matches", 3, 50))
        trial_config = {**config, "poisson_recent_matches": recent_n, "benchmark_tuning_enabled": False}
        scores: List[float] = []
        for model_index, model_key in enumerate(model_sequence, start=1):
            metrics = evaluate_score_model_walk_forward_2026(
                model_key=model_key,
                history_df=historical_train_df,
                confirmed_df=confirmed_df,
                pre_eval_confirmed_df=None,
                tournament=tournament,
                config=trial_config,
                start_time=start_time,
                hardware=hardware,
                model_index=model_index,
                model_total=max(len(model_sequence), 1),
                progress_callback=None,
            )
            if metrics.get("available"):
                scores.append(backtest_result_score_value(metrics))
        trial.set_user_attr("available_models", len(scores))
        if not scores:
            return -1.0
        return float(np.mean(scores))

    def on_trial_complete(study, trial) -> None:
        value = float(trial.value) if trial.value is not None else None
        row = {
            "number": int(trial.number),
            "poisson_recent_matches": int(trial.params.get("poisson_recent_matches", current_n)),
            "value": round(value, 6) if value is not None else None,
            "state": str(getattr(trial.state, "name", trial.state)),
            "available_models": int(trial.user_attrs.get("available_models") or 0),
        }
        summary["trials"].append(row)
        emit_report_progress(
            progress_callback,
            stage="benchmark_tuning",
            start_time=start_time,
            model_index=min(int(trial.number) + 1, n_trials),
            model_total=n_trials,
            model_key="optuna",
            fixture_index=1,
            fixture_total=1,
            hardware=hardware,
            message=(
                f"Optuna Poisson ultimos: trial {int(trial.number) + 1}/{n_trials}, "
                f"N={row['poisson_recent_matches']}, score={row['value']}"
            ),
        )

    try:
        study.optimize(objective, n_trials=n_trials, callbacks=[on_trial_complete], show_progress_bar=False)
    except Exception as exc:
        summary["warnings"] = [f"Optuna benchmark interrumpido: {exc.__class__.__name__}: {exc}"]
        return summary

    if study.best_trial is not None:
        summary["available"] = True
        summary["best_poisson_recent_matches"] = int(study.best_trial.params.get("poisson_recent_matches", current_n))
        summary["best_value"] = round(float(study.best_value), 6)
    return summary


def benchmark_optuna_sampler(optuna, name: str, seed: int):
    key = str(name or "tpe").strip().lower().replace("_", "-")
    if key == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)


def alternatives_backtest_report(
        history_df: pd.DataFrame,
        tournament: Dict[str, Any],
        config: Dict[str, Any],
        model_sequence: List[str],
        start_time: float,
        hardware: Dict[str, Any],
        feature_source: BenchmarkFeatureSource | None = None,
        progress_callback=None,
) -> Dict[str, Any]:
    scope = "worldcup_2026_confirmed_auto"
    generated_at = _now_utc().isoformat()
    if history_df is None or history_df.empty:
        return {
            "summary": {
                "available": False,
                "scope": scope,
                "generated_at": generated_at,
                "backtest_range": empty_backtest_range(generated_at),
                "confirmed_matches": 0,
                "evaluated_matches": 0,
                "train_matches": 0,
            },
            "models": [],
            "warnings": ["Backtest no disponible: historico vacio."],
        }
    required = {"Date", "Team 1", "Team 2", "G1", "G2"}
    if not required.issubset(history_df.columns):
        return {
            "summary": {
                "available": False,
                "scope": scope,
                "generated_at": generated_at,
                "backtest_range": empty_backtest_range(generated_at),
                "confirmed_matches": 0,
                "evaluated_matches": 0,
                "train_matches": 0,
            },
            "models": [],
            "warnings": ["Backtest no disponible: columnas historicas incompletas."],
        }
    working = history_df.copy()
    working["_date"] = pd.to_datetime(working["Date"], errors="coerce")
    working = working[working["_date"].notna()].sort_values("_date", kind="stable").reset_index(drop=True)
    historical_train_df = working.drop(columns=["_date"]).copy()
    confirmed_all_df = confirmed_worldcup_2026_backtest_rows(tournament)
    confirmed_total = int(confirmed_all_df.shape[0])
    results_status = fixture_results_status(tournament_fixtures_dataframe(tournament))
    if confirmed_total <= 0:
        return {
            "summary": {
                "available": False,
                "scope": scope,
                "source": results_status.get("source", ""),
                "generated_at": generated_at,
                "backtest_range": empty_backtest_range(generated_at),
                "confirmed_matches": 0,
                "confirmed_matches_detail": [],
                "evaluated_matches": 0,
                "train_matches": len(historical_train_df),
                "anti_leakage": "Sin partidos Mundial 2026 finalizados con marcador completo; no se ejecuta backtest.",
            },
            "models": [],
            "warnings": ["Backtest no disponible: no hay partidos confirmados del Mundial 2026 con marcador final."],
        }
    requested_n = int(config.get("backtest_last_n") or confirmed_total)
    eval_count = min(max(requested_n, 1), confirmed_total)
    prefix_count = max(confirmed_total - eval_count, 0)
    pre_eval_confirmed_df = confirmed_all_df.iloc[:prefix_count].copy()
    confirmed_df = confirmed_all_df.iloc[prefix_count:].reset_index(drop=True).copy()
    baseline_metrics = evaluate_score_model_walk_forward_2026(
        model_key=DEFAULT_SCORE_MODEL,
        history_df=historical_train_df,
        confirmed_df=confirmed_df,
        pre_eval_confirmed_df=pre_eval_confirmed_df,
        tournament=tournament,
        config=config,
        start_time=start_time,
        hardware=hardware,
        model_index=1,
        model_total=max(len(model_sequence), 1),
        feature_source=feature_source,
        progress_callback=progress_callback,
    )
    models: List[Dict[str, Any]] = []
    model_total = len(model_sequence)
    for model_index, model_key in enumerate(model_sequence, start=1):
        if model_key == DEFAULT_SCORE_MODEL:
            metrics = dict(baseline_metrics)
        else:
            metrics = evaluate_score_model_walk_forward_2026(
                model_key=model_key,
                history_df=historical_train_df,
                confirmed_df=confirmed_df,
                pre_eval_confirmed_df=pre_eval_confirmed_df,
                tournament=tournament,
                config=config,
                start_time=start_time,
                hardware=hardware,
                model_index=model_index,
                model_total=max(model_total, 1),
                feature_source=feature_source,
                progress_callback=progress_callback,
            )
        models.append(compare_backtest_to_baseline(metrics, baseline_metrics))
    summary = {
        "available": True,
        "scope": scope,
        "source": results_status.get("source", ""),
        "generated_at": generated_at,
        "requested_matches": requested_n,
        "confirmed_matches": confirmed_total,
        "confirmed_matches_detail": confirmed_backtest_match_payloads(confirmed_df),
        "evaluated_matches": eval_count,
        "train_matches": len(historical_train_df),
        "holdout_start": str(confirmed_df.iloc[0].get("Date", "")) if not confirmed_df.empty else "",
        "holdout_end": str(confirmed_df.iloc[-1].get("Date", "")) if not confirmed_df.empty else "",
        "backtest_range": backtest_range_summary(confirmed_df, generated_at),
        "baseline": baseline_metrics,
        "anti_leakage": (
            "Walk-forward Mundial 2026: cada partido confirmado se evalua con all_matches.csv desde 2014 "
            "y solo resultados 2026 estrictamente anteriores en orden cronologico; nunca se entrena con el "
            "partido evaluado ni con resultados 2026 posteriores."
        ),
    }
    models = rank_backtest_models(models, summary)
    return {"summary": summary, "models": models, "warnings": []}


def xg_lightgbm_backtest_report(
        history_df: pd.DataFrame,
        tournament: Dict[str, Any],
        model_id: str,
        model_meta: Dict[str, Any],
        config: Dict[str, Any],
        start_time: float,
        hardware: Dict[str, Any],
        progress_callback=None,
) -> Dict[str, Any]:
    scope = "worldcup_2026_confirmed_auto"
    generated_at = _now_utc().isoformat()
    if history_df is None or history_df.empty:
        return {
            "summary": {
                "available": False,
                "scope": scope,
                "generated_at": generated_at,
                "backtest_range": empty_backtest_range(generated_at),
                "confirmed_matches": 0,
                "evaluated_matches": 0,
                "train_matches": 0,
            },
            "models": [],
            "warnings": ["Backtest xG no disponible: historico vacio."],
        }
    required = {"Date", "Team 1", "Team 2", "G1", "G2"}
    if not required.issubset(history_df.columns):
        return {
            "summary": {
                "available": False,
                "scope": scope,
                "generated_at": generated_at,
                "backtest_range": empty_backtest_range(generated_at),
                "confirmed_matches": 0,
                "evaluated_matches": 0,
                "train_matches": 0,
            },
            "models": [],
            "warnings": ["Backtest xG no disponible: columnas historicas incompletas."],
        }
    working = history_df.copy()
    working["_date"] = pd.to_datetime(working["Date"], errors="coerce")
    working = working[working["_date"].notna()].sort_values("_date", kind="stable").reset_index(drop=True)
    historical_train_df = working.drop(columns=["_date"]).copy()
    confirmed_all_df = confirmed_worldcup_2026_backtest_rows(tournament)
    confirmed_total = int(confirmed_all_df.shape[0])
    results_status = fixture_results_status(tournament_fixtures_dataframe(tournament))
    if confirmed_total <= 0:
        return {
            "summary": {
                "available": False,
                "scope": scope,
                "source": results_status.get("source", ""),
                "generated_at": generated_at,
                "backtest_range": empty_backtest_range(generated_at),
                "confirmed_matches": 0,
                "confirmed_matches_detail": [],
                "evaluated_matches": 0,
                "train_matches": len(historical_train_df),
                "anti_leakage": "Sin partidos Mundial 2026 finalizados con marcador completo; no se ejecuta backtest xG.",
            },
            "models": [],
            "warnings": ["Backtest xG no disponible: no hay partidos confirmados del Mundial 2026 con marcador final."],
        }
    requested_n = int(config.get("backtest_last_n") or confirmed_total)
    eval_count = min(max(requested_n, 1), confirmed_total)
    prefix_count = max(confirmed_total - eval_count, 0)
    prefix_df = confirmed_all_df.iloc[:prefix_count].copy()
    confirmed_df = confirmed_all_df.iloc[prefix_count:].reset_index(drop=True).copy()
    metrics = evaluate_xg_lightgbm_walk_forward_2026(
        history_df=historical_train_df,
        confirmed_df=confirmed_df,
        pre_eval_confirmed_df=prefix_df,
        tournament=tournament,
        model_id=model_id,
        model_meta=model_meta,
        config=config,
        start_time=start_time,
        hardware=hardware,
        progress_callback=progress_callback,
    )
    summary = {
        "available": bool(metrics.get("available")),
        "scope": scope,
        "source": results_status.get("source", ""),
        "generated_at": generated_at,
        "requested_matches": requested_n,
        "confirmed_matches": confirmed_total,
        "confirmed_matches_detail": confirmed_backtest_match_payloads(confirmed_df),
        "evaluated_matches": int(metrics.get("evaluated_matches") or 0),
        "train_matches": len(historical_train_df),
        "holdout_start": str(confirmed_df.iloc[0].get("Date", "")) if not confirmed_df.empty else "",
        "holdout_end": str(confirmed_df.iloc[-1].get("Date", "")) if not confirmed_df.empty else "",
        "backtest_range": backtest_range_summary(confirmed_df, generated_at),
        "anti_leakage": (
            "Walk-forward xG Mundial 2026: cada partido confirmado se evalua con historico desde 2014 "
            "y solo resultados 2026 anteriores como contexto Poisson; el partido evaluado no entra al entrenamiento base."
        ),
    }
    models = rank_backtest_models([metrics], summary)
    return {"summary": summary, "models": models, "warnings": unique_strings(metrics.get("warnings", []))}


def evaluate_xg_lightgbm_walk_forward_2026(
        history_df: pd.DataFrame,
        confirmed_df: pd.DataFrame,
        pre_eval_confirmed_df: pd.DataFrame,
        tournament: Dict[str, Any],
        model_id: str,
        model_meta: Dict[str, Any],
        config: Dict[str, Any],
        start_time: float,
        hardware: Dict[str, Any],
        progress_callback=None,
) -> Dict[str, Any]:
    totals = empty_backtest_totals()
    sample_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    meta_warnings = model_meta.get("warnings") or []
    if not isinstance(meta_warnings, list):
        meta_warnings = [meta_warnings]
    warnings: List[str] = [str(item) for item in meta_warnings if str(item)]
    model_available = bool(model_meta.get("trained") or model_meta.get("bundle") or model_meta.get("model_id"))
    prefix_df = pre_eval_confirmed_df if pre_eval_confirmed_df is not None else pd.DataFrame()
    for fixture_index, (_, eval_row) in enumerate(confirmed_df.iterrows(), start=1):
        emit_report_progress(
            progress_callback,
            stage="backtesting",
            start_time=start_time,
            model_index=1,
            model_total=1,
            model_key=XG_LIGHTGBM_PIPELINE_MODE,
            fixture_index=fixture_index,
            fixture_total=max(int(confirmed_df.shape[0]), 1),
            hardware=hardware,
            message=f"Backtest {XG_LIGHTGBM_PIPELINE_LABEL}: {eval_row.get('Team 1', '')} vs {eval_row.get('Team 2', '')}",
        )
        previous_frames = []
        if not prefix_df.empty:
            previous_frames.append(prefix_df[["Date", "Year", "Team 1", "Team 2", "G1", "G2", "Round", "Group"]])
        current_previous = confirmed_df.iloc[:fixture_index - 1][["Date", "Year", "Team 1", "Team 2", "G1", "G2", "Round", "Group"]]
        if not current_previous.empty:
            previous_frames.append(current_previous)
        previous_2026 = pd.concat(previous_frames, ignore_index=True) if previous_frames else pd.DataFrame()
        train_frames = [history_df]
        if not previous_2026.empty:
            train_frames.append(previous_2026)
        train_df = pd.concat(train_frames, ignore_index=True)
        teams = alternatives_backtest_teams(tournament, train_df, pd.DataFrame([eval_row]))
        baseline_model = WorldCupModel.from_history(
            train_df,
            teams=teams,
            history_weight=float(config["history_weight"]),
            recency_weight=float(config["recency_weight"]),
            host_advantage=float(config["host_advantage"]),
            max_goals=int(config["max_goals"]),
        )
        try:
            prediction = predict_match_payload(
                tournament,
                apply_recent_context_model(baseline_model, config),
                fixture_id=eval_row.get("No.", ""),
                home=str(eval_row.get("Team 1", "")),
                away=str(eval_row.get("Team 2", "")),
                use_ml_model=True,
                ml_weight=1.0,
                model_id=model_id,
                poisson_recent_matches=int(config.get("poisson_recent_matches") or DEFAULT_CONFIG["poisson_recent_matches"]),
            )
        except Exception as exc:
            model_available = False
            warnings.append(f"Backtest xG omitio {eval_row.get('Team 1', '')} vs {eval_row.get('Team 2', '')}: {exc.__class__.__name__}.")
            continue
        row_metrics = xg_lightgbm_backtest_prediction(prediction, eval_row, config)
        if not row_metrics:
            continue
        row_metrics["sample"]["recent_matches_15"] = recent_matches_for_fixture(train_df, eval_row, limit=15)
        accumulate_backtest_totals(totals, row_metrics)
        match_rows.append(row_metrics["sample"])
        if len(sample_rows) < 8:
            sample_rows.append(row_metrics["sample"])
    return {
        "model_key": XG_LIGHTGBM_PIPELINE_MODE,
        "model_label": XG_LIGHTGBM_PIPELINE_LABEL,
        "available": bool(model_available) and totals["evaluated"] > 0,
        "warnings": unique_strings(warnings),
        "evaluated_matches": int(totals["evaluated"]),
        "feature_usage_counts": combined_feature_usage_counts(match_rows),
        **backtest_metric_summary(totals),
        "matches": match_rows,
        "sample": sample_rows,
    }


def xg_lightgbm_backtest_prediction(prediction: Dict[str, Any], row: pd.Series | Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any] | None:
    probabilities_pct = prediction.get("probabilities") or {}
    expected = prediction.get("expected_goals") or {}
    probabilities = {
        "home": float_or_zero(probabilities_pct.get("home")) / 100.0,
        "draw": float_or_zero(probabilities_pct.get("draw")) / 100.0,
        "away": float_or_zero(probabilities_pct.get("away")) / 100.0,
        "lambda1": float_or_zero(expected.get("home")),
        "lambda2": float_or_zero(expected.get("away")),
    }
    for line in REPORT_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        probabilities[f"over{suffix}"] = float_or_zero(probabilities_pct.get(f"over{suffix}")) / 100.0
        probabilities[f"under{suffix}"] = float_or_zero(probabilities_pct.get(f"under{suffix}")) / 100.0
    actual_home = int(float(row.get("G1")))
    actual_away = int(float(row.get("G2")))
    actual_outcome = score_outcome(actual_home, actual_away)
    outcome_probabilities = {
        "home": float_or_zero(probabilities.get("home")),
        "draw": float_or_zero(probabilities.get("draw")),
        "away": float_or_zero(probabilities.get("away")),
    }
    total_prob = sum(outcome_probabilities.values())
    if total_prob > 0:
        outcome_probabilities = {key: value / total_prob for key, value in outcome_probabilities.items()}
    pick = outcome_decision(outcome_probabilities)
    total_goals = actual_home + actual_away
    max_goals = int(config.get("max_goals") or DEFAULT_CONFIG["max_goals"])
    grid = normalize_score_grid_array(poisson_score_grid(probabilities["lambda1"], probabilities["lambda2"], max_goals=max_goals))
    actual_score_key = f"{actual_home}-{actual_away}"
    top_score_predictions = []
    for rank, score_item in enumerate(
            sorted(score_distribution_cells(grid), key=lambda item: item["probability_raw"], reverse=True)[:5],
            start=1,
    ):
        top_score_predictions.append({
            "rank": rank,
            "score": score_item.get("score", ""),
            "probability": score_item.get("probability", 0.0),
            "hit": score_item.get("score") == actual_score_key,
        })
    modal_index = int(np.argmax(grid))
    modal_home, modal_away = np.unravel_index(modal_index, grid.shape)
    modal_probability = float(grid[int(modal_home), int(modal_away)])
    score_probability = score_grid_actual_probability(grid, actual_home, actual_away)
    actual_score_rank = score_rank_from_grid(grid, actual_home, actual_away)
    expected_home, expected_away, expected_total, expected_margin = expected_score_from_grid(grid)
    actual_total = float(total_goals)
    actual_margin = float(actual_home - actual_away)
    over_under_rows = backtest_over_under_rows(probabilities, total_goals)
    confidence = max(outcome_probabilities.values()) if outcome_probabilities else 0.0
    modal_score = f"{int(modal_home)}-{int(modal_away)}"
    modal_hit = int(modal_home) == actual_home and int(modal_away) == actual_away
    home_team = str(row.get("Team 1", ""))
    away_team = str(row.get("Team 2", ""))
    return {
        "log_loss": -math.log(max(outcome_probabilities.get(actual_outcome, 0.0), 1e-12)),
        "brier": multiclass_brier_score(outcome_probabilities, actual_outcome),
        "score_log_loss": -math.log(max(score_probability, 1e-12)),
        "rps": ranked_probability_score(outcome_probabilities, actual_outcome),
        "entropy": probability_entropy(outcome_probabilities),
        "sharpness": confidence,
        "pick_hit": 1 if pick == actual_outcome else 0,
        "score_hit": 1 if modal_hit else 0,
        "top3_score_hit": 1 if actual_score_rank <= 3 else 0,
        "top5_score_hit": 1 if actual_score_rank <= 5 else 0,
        "expected_home_goals": expected_home,
        "expected_away_goals": expected_away,
        "expected_total_goals": expected_total,
        "expected_margin": expected_margin,
        "home_goals_abs_error": abs(expected_home - float(actual_home)),
        "away_goals_abs_error": abs(expected_away - float(actual_away)),
        "total_goals_abs_error": abs(expected_total - actual_total),
        "margin_abs_error": abs(expected_margin - actual_margin),
        "home_goals_squared_error": (expected_home - float(actual_home)) ** 2,
        "away_goals_squared_error": (expected_away - float(actual_away)) ** 2,
        "total_goals_squared_error": (expected_total - actual_total) ** 2,
        "margin_squared_error": (expected_margin - actual_margin) ** 2,
        "actual_outcome": actual_outcome,
        "predicted_outcome": pick,
        "confidence": confidence,
        "over_under": over_under_rows,
        "sample": {
            "fixture_id": str(row.get("No.", "")),
            "date": str(row.get("Date", "")),
            "home": home_team,
            "away": away_team,
            "home_asset": team_asset(home_team),
            "away_asset": team_asset(away_team),
            "match": f"{home_team} vs {away_team}",
            "actual_score": f"{actual_home}-{actual_away}",
            "actual_home_goals": actual_home,
            "actual_away_goals": actual_away,
            "total_goals": int(total_goals),
            "pick": outcome_label(pick),
            "pick_key": pick,
            "actual_pick": outcome_label(actual_outcome),
            "actual_pick_key": actual_outcome,
            "pick_hit": bool(pick == actual_outcome),
            "modal_score": modal_score,
            "most_probable_score": modal_score,
            "most_probable_score_probability": round(modal_probability * 100.0, 3),
            "most_probable_score_hit": bool(modal_hit),
            "top_scores": top_score_predictions,
            "expected_score": f"{expected_home:.2f}-{expected_away:.2f}",
            "actual_score_rank": actual_score_rank,
            "score_hit": bool(modal_hit),
            "top3_score_hit": bool(actual_score_rank <= 3),
            "top5_score_hit": bool(actual_score_rank <= 5),
            "rps": round(ranked_probability_score(outcome_probabilities, actual_outcome), 6),
            "confidence": round(confidence * 100.0, 3),
            "actual_probability": round(outcome_probabilities.get(actual_outcome, 0.0) * 100.0, 3),
            "score_probability": round(score_probability * 100.0, 3),
            "feature_context": prediction.get("data_quality", {}),
            "probabilities": {
                "home": round(outcome_probabilities["home"] * 100.0, 3),
                "draw": round(outcome_probabilities["draw"] * 100.0, 3),
                "away": round(outcome_probabilities["away"] * 100.0, 3),
            },
            "over_under": over_under_rows,
        },
    }


def empty_backtest_range(generated_at: str = "") -> Dict[str, Any]:
    return {
        "evaluated_matches": 0,
        "first_match": {},
        "last_match": {},
        "first_date": "",
        "last_date": "",
        "generated_at": generated_at,
    }


def backtest_range_summary(confirmed_df: pd.DataFrame, generated_at: str) -> Dict[str, Any]:
    if confirmed_df is None or confirmed_df.empty:
        return empty_backtest_range(generated_at)
    matches = confirmed_backtest_match_payloads(confirmed_df)
    first = matches[0] if matches else {}
    last = matches[-1] if matches else {}
    return {
        "evaluated_matches": int(confirmed_df.shape[0]),
        "first_match": first,
        "last_match": last,
        "first_date": str(first.get("date", "")),
        "last_date": str(last.get("date", "")),
        "generated_at": generated_at,
    }


def confirmed_worldcup_2026_backtest_rows(tournament: Dict[str, Any]) -> pd.DataFrame:
    fixture_df = tournament_fixtures_dataframe(tournament)
    if fixture_df.empty:
        return pd.DataFrame(columns=["No.", "Date", "Year", "Team 1", "Team 2", "G1", "G2", "Round", "Group", "Source"])
    working = fixture_df.copy()
    working["HG"] = pd.to_numeric(working["Goles 1"], errors="coerce")
    working["AG"] = pd.to_numeric(working["Goles 2"], errors="coerce")
    working = working[
        working["Fecha"].astype(str).str.len().gt(0)
        & working["Grupo"].astype(str).str.len().gt(0)
        & working["Equipo 1"].astype(str).str.len().gt(1)
        & working["Equipo 2"].astype(str).str.len().gt(1)
        & ~working["Equipo 1"].astype(str).str.match(r"^[123W][A-Z0-9/]+$")
        & ~working["Equipo 2"].astype(str).str.match(r"^[123W][A-Z0-9/]+$")
        & working["HG"].notna()
        & working["AG"].notna()
    ].copy()
    if working.empty:
        return pd.DataFrame(columns=["No.", "Date", "Year", "Team 1", "Team 2", "G1", "G2", "Round", "Group", "Source"])
    working = attach_fixture_schedule(working)
    now = pd.Timestamp(_utcify_datetime(_now_utc())).tz_convert(timezone.utc)
    today = now.normalize()
    working["_date"] = pd.to_datetime(working["_date"], utc=True, errors="coerce")
    working["_kickoff"] = pd.to_datetime(working["_kickoff"], utc=True, errors="coerce")
    has_time = working["_kickoff"].notna()
    available_by_time = has_time & (working["_kickoff"] <= now)
    available_by_date = ~has_time & working["_date"].notna() & (working["_date"] <= today)
    verified_source = working.get("Fuente Resultado", pd.Series("", index=working.index)).astype(str).str.lower().str.startswith("verified:")
    available_by_verified = verified_source & working["_date"].notna() & (working["_date"] <= today)
    working = working[working["_date"].notna() & (available_by_time | available_by_date | available_by_verified)].copy()
    if working.empty:
        return pd.DataFrame(columns=["No.", "Date", "Year", "Team 1", "Team 2", "G1", "G2", "Round", "Group", "Source"])
    working = working[working["_date"].notna()].sort_values(["_sort_time", "No."], kind="stable").reset_index(drop=True)
    rows = []
    for _, fixture in working.iterrows():
        rows.append({
            "No.": fixture.get("No.", ""),
            "Date": str(fixture.get("Fecha", ""))[:10],
            "Year": 2026,
            "Team 1": fixture.get("Equipo 1", ""),
            "Team 2": fixture.get("Equipo 2", ""),
            "G1": int(fixture.get("HG")),
            "G2": int(fixture.get("AG")),
            "Round": fixture.get("Ronda", ""),
            "Group": fixture.get("Grupo", ""),
            "Source": fixture.get("Fuente Resultado", ""),
        })
    return pd.DataFrame(rows)


def confirmed_backtest_match_payloads(confirmed_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in confirmed_df.iterrows():
        rows.append({
            "id": str(row.get("No.", "")),
            "date": str(row.get("Date", "")),
            "home": str(row.get("Team 1", "")),
            "away": str(row.get("Team 2", "")),
            "home_asset": team_asset(str(row.get("Team 1", ""))),
            "away_asset": team_asset(str(row.get("Team 2", ""))),
            "score": f"{int(row.get('G1'))}-{int(row.get('G2'))}",
            "source": str(row.get("Source", "")),
        })
    return rows


def history_with_confirmed_worldcup_results(history_df: pd.DataFrame, tournament: Dict[str, Any]) -> pd.DataFrame:
    frames = []
    if history_df is not None and not history_df.empty:
        frames.append(history_df.copy())
    confirmed = confirmed_worldcup_2026_backtest_rows(tournament)
    if not confirmed.empty:
        frames.append(confirmed[["Date", "Year", "Team 1", "Team 2", "G1", "G2", "Round", "Group"]].copy())
    if not frames:
        return pd.DataFrame(columns=["Date", "Year", "Team 1", "Team 2", "G1", "G2", "Round", "Group"])
    return pd.concat(frames, ignore_index=True)


def recent_matches_for_fixture(
        history_df: pd.DataFrame,
        fixture: pd.Series | Dict[str, Any],
        limit: int = 15,
        international_matches: pd.DataFrame | None = None,
        international_status: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    getter = fixture.get if hasattr(fixture, "get") else (lambda key, default=None: default)
    home = str(getter("Equipo 1") or getter("Team 1") or getter("home") or "")
    away = str(getter("Equipo 2") or getter("Team 2") or getter("away") or "")
    before_date = str(getter("Fecha") or getter("Date") or getter("date") or "")[:10]
    if international_matches is not None:
        status = international_status or international_results_status()
        warnings = list(status.get("warnings") or [])
        if status.get("warning") and not warnings:
            warnings.append(str(status.get("warning")))
        return {
            "limit": int(limit),
            "home_team": home,
            "away_team": away,
            "source": "all_matches.csv",
            "source_path": str(getattr(international_matches, "attrs", {}).get("source_path") or status.get("source_path") or ""),
            "max_scored_date": status.get("max_scored_date", ""),
            "warnings": unique_strings(warnings),
            "home": recent_international_team_matches(international_matches, home, before_date=before_date, limit=limit),
            "away": recent_international_team_matches(international_matches, away, before_date=before_date, limit=limit),
        }
    return {
        "limit": int(limit),
        "home_team": home,
        "away_team": away,
        "home": recent_team_matches(history_df, home, before_date=before_date, limit=limit),
        "away": recent_team_matches(history_df, away, before_date=before_date, limit=limit),
    }


def recent_international_team_matches(matches: pd.DataFrame, team: str, before_date: str = "", limit: int = 15) -> List[Dict[str, Any]]:
    if matches is None or matches.empty or not str(team or "").strip():
        return []
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(matches.columns):
        return []
    working = matches.copy()
    working["_date"] = pd.to_datetime(working["date"], errors="coerce", utc=True).dt.tz_convert(None)
    working["_home_score"] = pd.to_numeric(working["home_score"], errors="coerce")
    working["_away_score"] = pd.to_numeric(working["away_score"], errors="coerce")
    working = working[
        working["_date"].notna()
        & working["_home_score"].notna()
        & working["_away_score"].notna()
    ].copy()
    working = working[working["_date"] >= pd.Timestamp(INTERNATIONAL_RECENT_START_DATE)].copy()
    cutoff = international_cutoff_timestamp(before_date)
    working = working[working["_date"] < cutoff].copy()
    team_key = team_name_key(team)
    home_keys = working["home_team"].map(team_name_key)
    away_keys = working["away_team"].map(team_name_key)
    working = working[(home_keys == team_key) | (away_keys == team_key)].copy()
    working = working.sort_values("_date", ascending=False, kind="stable").head(int(limit)).copy()
    rows: List[Dict[str, Any]] = []
    total = max(int(working.shape[0]), 1)
    for index, (_, row) in enumerate(working.iterrows()):
        is_home = team_name_key(row.get("home_team")) == team_key
        goals_for = int(row.get("_home_score") if is_home else row.get("_away_score"))
        goals_against = int(row.get("_away_score") if is_home else row.get("_home_score"))
        opponent = str(row.get("away_team" if is_home else "home_team", ""))
        tournament = str(row.get("tournament") or "")
        is_friendly = is_friendly_tournament(tournament)
        tournament_score = tournament_weight(tournament)
        recency_score = recent_match_recency_weight(total - index, total)
        weight = tournament_score * recency_score
        rows.append({
            "date": str(row.get("date", ""))[:10],
            "opponent": opponent,
            "venue": "Neutral" if bool(row.get("neutral", False)) else "Local" if is_home else "Visitante",
            "score": f"{goals_for}-{goals_against}",
            "result": "G" if goals_for > goals_against else "E" if goals_for == goals_against else "P",
            "match_type": "Friendly" if is_friendly else "Official",
            "tournament": tournament,
            "weight": round(float(weight), 3),
            "tournament_weight": round(float(tournament_score), 3),
            "recency_weight": round(float(recency_score), 3),
            "importance_label": recent_match_importance_label(weight, is_friendly),
        })
    return rows


def recent_team_matches(history_df: pd.DataFrame, team: str, before_date: str = "", limit: int = 15) -> List[Dict[str, Any]]:
    if history_df is None or history_df.empty or not str(team or "").strip():
        return []
    required = {"Date", "Team 1", "Team 2", "G1", "G2"}
    if not required.issubset(history_df.columns):
        return []
    working = history_df.copy()
    working["_date"] = pd.to_datetime(working["Date"], errors="coerce")
    working = working[working["_date"].notna()].copy()
    cutoff = pd.to_datetime(before_date, errors="coerce")
    if pd.notna(cutoff):
        working = working[working["_date"] < cutoff].copy()
    team_key = team_name_key(team)
    home_keys = working["Team 1"].map(team_name_key)
    away_keys = working["Team 2"].map(team_name_key)
    working = working[(home_keys == team_key) | (away_keys == team_key)].copy()
    working["_g1"] = pd.to_numeric(working["G1"], errors="coerce")
    working["_g2"] = pd.to_numeric(working["G2"], errors="coerce")
    working = working[working["_g1"].notna() & working["_g2"].notna()]
    working = working.sort_values("_date", ascending=False, kind="stable").head(int(limit)).copy()
    rows: List[Dict[str, Any]] = []
    total = max(int(working.shape[0]), 1)
    for index, (_, row) in enumerate(working.iterrows()):
        is_home = team_name_key(row.get("Team 1")) == team_key
        goals_for = int(row.get("_g1") if is_home else row.get("_g2"))
        goals_against = int(row.get("_g2") if is_home else row.get("_g1"))
        opponent = str(row.get("Team 2" if is_home else "Team 1", ""))
        tournament = str(row.get("Round") or row.get("tournament") or row.get("Group") or "")
        is_friendly = is_friendly_tournament(tournament)
        tournament_score = tournament_weight(tournament)
        recency_score = recent_match_recency_weight(total - index, total)
        weight = tournament_score * recency_score
        rows.append({
            "date": str(row.get("Date", ""))[:10],
            "opponent": opponent,
            "venue": "Local" if is_home else "Visitante",
            "score": f"{goals_for}-{goals_against}",
            "result": "G" if goals_for > goals_against else "E" if goals_for == goals_against else "P",
            "match_type": "Friendly" if is_friendly else "Official",
            "tournament": tournament,
            "weight": round(float(weight), 3),
            "tournament_weight": round(float(tournament_score), 3),
            "recency_weight": round(float(recency_score), 3),
            "importance_label": recent_match_importance_label(weight, is_friendly),
        })
    return rows


def evaluate_score_model_walk_forward_2026(
        model_key: str,
        history_df: pd.DataFrame,
        confirmed_df: pd.DataFrame,
        pre_eval_confirmed_df: pd.DataFrame | None,
        tournament: Dict[str, Any],
        config: Dict[str, Any],
        start_time: float,
        hardware: Dict[str, Any],
        model_index: int,
        model_total: int,
        feature_source: BenchmarkFeatureSource | None = None,
        progress_callback=None,
) -> Dict[str, Any]:
    totals = empty_backtest_totals()
    sample_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    model_available = True
    international_status = international_results_status() if feature_source is not None else None
    prefix_df = pre_eval_confirmed_df if pre_eval_confirmed_df is not None else pd.DataFrame()
    for fixture_index, (_, eval_row) in enumerate(confirmed_df.iterrows(), start=1):
        emit_report_progress(
            progress_callback,
            stage="backtesting",
            start_time=start_time,
            model_index=model_index,
            model_total=max(model_total, 1),
            model_key=model_key,
            fixture_index=fixture_index,
            fixture_total=max(int(confirmed_df.shape[0]), 1),
            hardware=hardware,
            message=f"Backtest {score_model_display_label(model_key)}: {eval_row.get('Team 1', '')} vs {eval_row.get('Team 2', '')}",
        )
        previous_frames = []
        if not prefix_df.empty:
            previous_frames.append(prefix_df[["Date", "Year", "Team 1", "Team 2", "G1", "G2", "Round", "Group"]])
        current_previous = confirmed_df.iloc[:fixture_index - 1][["Date", "Year", "Team 1", "Team 2", "G1", "G2", "Round", "Group"]]
        if not current_previous.empty:
            previous_frames.append(current_previous)
        previous_2026 = pd.concat(previous_frames, ignore_index=True) if previous_frames else pd.DataFrame()
        train_df = pd.concat(
            [history_df, previous_2026] if not previous_2026.empty else [history_df],
            ignore_index=True,
        )
        teams = alternatives_backtest_teams(tournament, train_df, pd.DataFrame([eval_row]))
        baseline_model = WorldCupModel.from_history(
            train_df,
            teams=teams,
            history_weight=float(config["history_weight"]),
            recency_weight=float(config["recency_weight"]),
            host_advantage=float(config["host_advantage"]),
            max_goals=int(config["max_goals"]),
        )
        model = baseline_model
        metadata = score_model_metadata(model) if model_key == DEFAULT_SCORE_MODEL else {
            "key": model_key,
            "label": score_model_display_label(model_key),
            "available": True,
            "params": {},
            "warnings": [],
        }
        if model_key != DEFAULT_SCORE_MODEL:
            try:
                model = build_score_model(
                    baseline_model,
                    history_df=train_df,
                    teams=teams,
                    config={**config, "score_model": model_key},
                )
                metadata = score_model_metadata(model)
            except Exception as exc:
                model_available = False
                warning = f"{exc.__class__.__name__}: {exc}; se usa Poisson independiente para este punto walk-forward."
                warnings.append(warning)
                metadata = {
                    "key": model_key,
                    "label": score_model_display_label(model_key),
                    "available": False,
                    "params": {},
                    "warnings": [warning],
                }
        prediction_model = apply_recent_context_model(model, config)
        prediction_model = apply_benchmark_feature_model(prediction_model, model_key, feature_source, history_df=train_df)
        row_metrics = score_model_backtest_prediction(prediction_model, eval_row, config)
        if not row_metrics:
            continue
        row_metrics["sample"]["recent_matches_15"] = recent_matches_for_fixture(
            train_df,
            eval_row,
            limit=15,
            international_matches=feature_source.international_matches if feature_source is not None else None,
            international_status=international_status,
        )
        accumulate_backtest_totals(totals, row_metrics)
        match_rows.append(row_metrics["sample"])
        if len(sample_rows) < 8:
            sample_rows.append(row_metrics["sample"])
        if not bool(metadata.get("available", True)):
            model_available = False
            warnings.extend(str(item) for item in metadata.get("warnings", []) if str(item))
    feature_usage = combined_feature_usage_counts(match_rows)
    return {
        "model_key": str(model_key),
        "model_label": score_model_display_label(model_key),
        "available": bool(model_available) and totals["evaluated"] > 0,
        "warnings": unique_strings(warnings),
        "evaluated_matches": int(totals["evaluated"]),
        "feature_usage_counts": feature_usage,
        **backtest_metric_summary(totals),
        "matches": match_rows,
        "sample": sample_rows,
    }


def rank_backtest_models(models: List[Dict[str, Any]], summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    indexed = list(enumerate(models or []))
    ranked_pairs = sorted(
        indexed,
        key=lambda pair: (
            0 if pair[1].get("available") else 1,
            -backtest_result_score_value(pair[1]) if pair[1].get("available") else float("inf"),
            float_or_zero(pair[1].get("log_loss")) if pair[1].get("available") else float("inf"),
            float_or_zero(pair[1].get("rps")) if pair[1].get("available") else float("inf"),
            float_or_zero(pair[1].get("expected_calibration_error")) if pair[1].get("available") else float("inf"),
            float_or_zero(pair[1].get("brier")) if pair[1].get("available") else float("inf"),
            float_or_zero(pair[1].get("score_log_loss")) if pair[1].get("available") else float("inf"),
            float_or_zero(pair[1].get("ou25_log_loss")) if pair[1].get("available") else float("inf"),
            pair[0],
        ),
    )
    ranked: List[Dict[str, Any]] = []
    for rank, (_, item) in enumerate(ranked_pairs, start=1):
        payload = dict(item)
        payload["score_resultados"] = backtest_result_score_value(payload)
        payload["rank"] = rank
        payload["reliability_score"] = payload["score_resultados"] if payload.get("available") else 0.0
        payload["ranking_metric"] = "score_resultados"
        payload["ranking_reason"] = (
            "Ordenado por Score de resultados: pick 1X2, top-3 marcador, over/under, marcador #1 "
            "y calibracion; log-loss queda solo como desempate tecnico."
        )
        payload["holdout_start"] = summary.get("holdout_start", "")
        payload["holdout_end"] = summary.get("holdout_end", "")
        ranked.append(payload)
    return ranked


def rank_fixture_report_models(fixture_reports: List[Dict[str, Any]], ranked_model_keys: List[str]) -> List[Dict[str, Any]]:
    rank_by_key = {str(key): index for index, key in enumerate(ranked_model_keys)}
    output = []
    for report in fixture_reports:
        payload = dict(report)
        payload["models"] = sorted(
            list(payload.get("models", [])),
            key=lambda model: rank_by_key.get(str(model.get("model_key") or ""), len(rank_by_key)),
        )
        output.append(payload)
    return output


def primary_model_for_fixture(
        fixture_report: Dict[str, Any],
        best_model: Dict[str, Any],
        backtest_by_key: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    best_key = str(best_model.get("model_key") or "")
    if not best_key or not best_model.get("available"):
        return {
            "available": False,
            "model_key": best_key,
            "model_label": best_model.get("model_label", ""),
            "reason": best_model.get("reason", "Sin modelo #1 disponible para este fixture."),
            "selection_policy": "Solo se publica el modelo #1 cuando existe backtest valido.",
        }
    for model in fixture_report.get("models", []) or []:
        if str(model.get("model_key") or "") != best_key:
            continue
        backtest = backtest_by_key.get(best_key, {})
        primary = dict(model)
        primary["primary"] = True
        primary["rank"] = backtest.get("rank", best_model.get("rank", 1))
        primary["backtest"] = backtest
        primary["selection_policy"] = "Prediccion principal tomada solo del modelo #1 del backtesting; no se pondera con otros modelos."
        return primary
    return {
        "available": False,
        "model_key": best_key,
        "model_label": best_model.get("model_label", best_key),
        "reason": "El modelo #1 del backtesting no genero prediccion para este fixture.",
        "selection_policy": "Sin fallback silencioso: si el modelo #1 no esta disponible, la prediccion principal queda no disponible.",
        "backtest": backtest_by_key.get(best_key, best_model),
    }


def benchmark_models_with_backtests(backtests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    catalog_by_key = {str(item.get("key") or ""): item for item in benchmark_score_model_catalog()}
    output = []
    for backtest in backtests or []:
        key = str(backtest.get("model_key") or "")
        catalog = dict(catalog_by_key.get(key, {"key": key, "model_name": backtest.get("model_label", key)}))
        output.append({
            **catalog,
            "rank": backtest.get("rank", catalog.get("rank", "")),
            "backtest": backtest,
        })
    return output


def benchmark_score_model_catalog() -> List[Dict[str, Any]]:
    alternatives = {str(item.get("key") or ""): item for item in sota_alternatives_catalog()}
    catalog: List[Dict[str, Any]] = []
    for index, key in enumerate(BENCHMARK_SCORE_MODEL_SEQUENCE, start=1):
        if key == DEFAULT_SCORE_MODEL:
            catalog.append({
                "rank": index,
                "key": DEFAULT_SCORE_MODEL,
                "model_name": score_model_display_label(DEFAULT_SCORE_MODEL),
                "family": "baseline_poisson",
                "description": "Poisson independiente usado tambien como modelo rankeado de referencia.",
            })
            continue
        item = dict(alternatives.get(key, {}))
        if not item:
            item = {
                "rank": index,
                "key": key,
                "model_name": score_model_display_label(key),
                "family": "score_model",
                "description": "",
            }
        item["rank"] = index
        catalog.append(item)
    return catalog


def alternatives_backtest_teams(tournament: Dict[str, Any], train_df: pd.DataFrame, holdout_df: pd.DataFrame) -> List[str]:
    group_map = groups_from_tournament(tournament)
    teams = {team for group_teams in group_map.values() for team in group_teams}
    for frame in (train_df, holdout_df):
        if frame is None or frame.empty:
            continue
        for column in ("Team 1", "Team 2"):
            if column in frame:
                teams.update(str(value) for value in frame[column].dropna().tolist() if str(value).strip())
    return sorted(teams)


def evaluate_score_model_backtest(
        model: Any,
        model_key: str,
        metadata: Dict[str, Any],
        holdout_df: pd.DataFrame,
        config: Dict[str, Any],
) -> Dict[str, Any]:
    totals = empty_backtest_totals()
    sample_rows: List[Dict[str, Any]] = []
    match_rows: List[Dict[str, Any]] = []
    for _, row in holdout_df.iterrows():
        row_metrics = score_model_backtest_prediction(model, row, config)
        if not row_metrics:
            continue
        accumulate_backtest_totals(totals, row_metrics)
        match_rows.append(row_metrics["sample"])
        if len(sample_rows) < 8:
            sample_rows.append(row_metrics["sample"])
    return {
        "model_key": str(model_key),
        "model_label": str(metadata.get("label") or score_model_display_label(model_key)),
        "available": bool(metadata.get("available", True)) and totals["evaluated"] > 0,
        "warnings": [str(item) for item in metadata.get("warnings", []) if str(item)],
        "evaluated_matches": int(totals["evaluated"]),
        "feature_usage_counts": combined_feature_usage_counts(match_rows),
        **backtest_metric_summary(totals),
        "matches": match_rows,
        "sample": sample_rows,
    }


def empty_backtest_totals() -> Dict[str, Any]:
    return {
        "log_loss": 0.0,
        "brier": 0.0,
        "score_log_loss": 0.0,
        "rps": 0.0,
        "entropy": 0.0,
        "sharpness": 0.0,
        "pick_hits": 0,
        "score_hits": 0,
        "top3_score_hits": 0,
        "top5_score_hits": 0,
        "home_goals_abs_error": 0.0,
        "away_goals_abs_error": 0.0,
        "total_goals_abs_error": 0.0,
        "margin_abs_error": 0.0,
        "home_goals_squared_error": 0.0,
        "away_goals_squared_error": 0.0,
        "total_goals_squared_error": 0.0,
        "margin_squared_error": 0.0,
        "over_under_hits": 0,
        "over_under_total": 0,
        "over_under_by_line": {
            f"{line:.1f}": {"hits": 0, "total": 0, "log_loss": 0.0, "brier": 0.0}
            for line in REPORT_TOTAL_GOAL_LINES
        },
        "calibration_bins": [
            {"count": 0, "confidence_sum": 0.0, "correct_sum": 0.0}
            for _ in range(CALIBRATION_BIN_COUNT)
        ],
        "confusion_matrix": {
            actual: {predicted: 0 for predicted in OUTCOME_KEYS}
            for actual in OUTCOME_KEYS
        },
        "evaluated": 0,
    }


def backtest_metric_summary(totals: Dict[str, Any]) -> Dict[str, Any]:
    evaluated = max(int(totals.get("evaluated") or 0), 1)
    calibration = calibration_metrics(totals)
    class_metrics = classification_metrics(totals)
    over_under_metrics = over_under_metrics_by_line(totals)
    ou25 = over_under_metrics.get("2.5", {})
    summary = {
        "log_loss": round(float(totals["log_loss"]) / evaluated, 6),
        "brier": round(float(totals["brier"]) / evaluated, 6),
        "score_log_loss": round(float(totals["score_log_loss"]) / evaluated, 6),
        "rps": round(float(totals["rps"]) / evaluated, 6),
        "entropy": round(float(totals["entropy"]) / evaluated, 6),
        "sharpness": round(float(totals["sharpness"]) / evaluated, 6),
        "expected_calibration_error": calibration["expected_calibration_error"],
        "max_calibration_error": calibration["max_calibration_error"],
        "calibration_bins": calibration["bins"],
        "pick_accuracy": round(float(totals["pick_hits"]) / evaluated, 6),
        "score_accuracy": round(float(totals["score_hits"]) / evaluated, 6),
        "top3_score_accuracy": round(float(totals["top3_score_hits"]) / evaluated, 6),
        "top5_score_accuracy": round(float(totals["top5_score_hits"]) / evaluated, 6),
        "home_goals_mae": round(float(totals["home_goals_abs_error"]) / evaluated, 6),
        "away_goals_mae": round(float(totals["away_goals_abs_error"]) / evaluated, 6),
        "total_goals_mae": round(float(totals["total_goals_abs_error"]) / evaluated, 6),
        "margin_mae": round(float(totals["margin_abs_error"]) / evaluated, 6),
        "home_goals_rmse": round(math.sqrt(float(totals["home_goals_squared_error"]) / evaluated), 6),
        "away_goals_rmse": round(math.sqrt(float(totals["away_goals_squared_error"]) / evaluated), 6),
        "total_goals_rmse": round(math.sqrt(float(totals["total_goals_squared_error"]) / evaluated), 6),
        "margin_rmse": round(math.sqrt(float(totals["margin_squared_error"]) / evaluated), 6),
        "over_under_accuracy": round(float(totals["over_under_hits"]) / max(int(totals["over_under_total"]), 1), 6),
        "over_under_accuracy_by_line": over_under_accuracy_by_line(totals),
        "over_under_metrics_by_line": over_under_metrics,
        "ou25_log_loss": ou25.get("log_loss", 0.0),
        "ou25_brier": ou25.get("brier", 0.0),
        "confusion_matrix": totals.get("confusion_matrix", {}),
        "class_metrics": class_metrics["by_class"],
        "macro_f1": class_metrics["macro_f1"],
        "balanced_accuracy": class_metrics["balanced_accuracy"],
    }
    summary["score_resultados"] = backtest_result_score(summary)
    return summary


def backtest_result_score(metrics: Dict[str, Any]) -> float:
    pick = float_or_zero(metrics.get("pick_accuracy"))
    top3 = float_or_zero(metrics.get("top3_score_accuracy"))
    over_under = float_or_zero(metrics.get("over_under_accuracy"))
    exact_score = float_or_zero(metrics.get("score_accuracy"))
    calibration = max(0.0, 1.0 - min(float_or_zero(metrics.get("expected_calibration_error")), 1.0))
    score = (
        (0.35 * pick)
        + (0.25 * top3)
        + (0.20 * over_under)
        + (0.10 * exact_score)
        + (0.10 * calibration)
    ) * 100.0
    return round(score, 3)


def backtest_result_score_value(metrics: Dict[str, Any]) -> float:
    if metrics.get("score_resultados") not in (None, ""):
        return round(float_or_zero(metrics.get("score_resultados")), 3)
    return backtest_result_score(metrics)


def calibration_metrics(totals: Dict[str, Any]) -> Dict[str, Any]:
    bins = []
    total = max(int(totals.get("evaluated") or 0), 1)
    ece = 0.0
    mce = 0.0
    for index, item in enumerate(totals.get("calibration_bins") or []):
        count = int(item.get("count") or 0)
        confidence = float(item.get("confidence_sum") or 0.0) / count if count else 0.0
        accuracy = float(item.get("correct_sum") or 0.0) / count if count else 0.0
        gap = abs(accuracy - confidence) if count else 0.0
        ece += (count / total) * gap
        mce = max(mce, gap)
        bins.append({
            "bin": index + 1,
            "min_confidence": round(index / CALIBRATION_BIN_COUNT, 3),
            "max_confidence": round((index + 1) / CALIBRATION_BIN_COUNT, 3),
            "count": count,
            "confidence": round(confidence, 6),
            "accuracy": round(accuracy, 6),
            "gap": round(gap, 6),
        })
    return {
        "expected_calibration_error": round(ece, 6),
        "max_calibration_error": round(mce, 6),
        "bins": bins,
    }


def classification_metrics(totals: Dict[str, Any]) -> Dict[str, Any]:
    matrix = totals.get("confusion_matrix") or {}
    by_class: Dict[str, Dict[str, float]] = {}
    f1_values: List[float] = []
    recall_values: List[float] = []
    for key in OUTCOME_KEYS:
        true_positive = float((matrix.get(key) or {}).get(key) or 0)
        predicted_total = float(sum((matrix.get(actual) or {}).get(key) or 0 for actual in OUTCOME_KEYS))
        actual_total = float(sum((matrix.get(key) or {}).get(predicted) or 0 for predicted in OUTCOME_KEYS))
        precision = true_positive / predicted_total if predicted_total > 0 else 0.0
        recall = true_positive / actual_total if actual_total > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        by_class[key] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": int(actual_total),
        }
        if actual_total > 0:
            recall_values.append(recall)
        f1_values.append(f1)
    return {
        "by_class": by_class,
        "macro_f1": round(float(np.mean(f1_values)) if f1_values else 0.0, 6),
        "balanced_accuracy": round(float(np.mean(recall_values)) if recall_values else 0.0, 6),
    }


def ranked_probability_score(probabilities: Dict[str, float], actual: str) -> float:
    predicted = np.asarray([float_or_zero(probabilities.get(key)) for key in OUTCOME_KEYS], dtype=float)
    total = float(predicted.sum())
    predicted = predicted / total if total > 0 else np.ones(len(OUTCOME_KEYS), dtype=float) / len(OUTCOME_KEYS)
    observed = np.asarray([1.0 if key == actual else 0.0 for key in OUTCOME_KEYS], dtype=float)
    return float(np.sum(np.square(np.cumsum(predicted) - np.cumsum(observed))) / max(len(OUTCOME_KEYS) - 1, 1))


def probability_entropy(probabilities: Dict[str, float]) -> float:
    values = np.asarray([max(float_or_zero(probabilities.get(key)), 1e-12) for key in OUTCOME_KEYS], dtype=float)
    values = values / max(float(values.sum()), 1e-12)
    return float(-np.sum(values * np.log(values)))


def expected_score_from_grid(grid: np.ndarray) -> Tuple[float, float, float, float]:
    grid = normalize_score_grid_array(grid)
    goals = np.arange(grid.shape[0], dtype=float)
    home_goals, away_goals = np.meshgrid(goals, goals, indexing="ij")
    expected_home = float(np.sum(grid * home_goals))
    expected_away = float(np.sum(grid * away_goals))
    return expected_home, expected_away, expected_home + expected_away, expected_home - expected_away


def score_rank_from_grid(grid: np.ndarray, home_goals: int, away_goals: int) -> int:
    grid = normalize_score_grid_array(grid)
    home = min(max(int(home_goals), 0), grid.shape[0] - 1)
    away = min(max(int(away_goals), 0), grid.shape[1] - 1)
    flat_index = int(np.ravel_multi_index((home, away), grid.shape))
    ranking = np.argsort(grid.ravel())[::-1]
    matches = np.where(ranking == flat_index)[0]
    return int(matches[0] + 1) if matches.size else int(grid.size)


def score_model_backtest_prediction(model: Any, row: pd.Series | Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any] | None:
    fixture = pd.Series({
        "No.": row.get("No.", ""),
        "Fecha": row.get("Date", ""),
        "Grupo": row.get("Group", ""),
        "Equipo 1": row.get("Team 1", ""),
        "Equipo 2": row.get("Team 2", ""),
        "Sede": "",
    })
    try:
        probabilities = model_probabilities_for_fixture(model, fixture, config)
        actual_home = int(float(row.get("G1")))
        actual_away = int(float(row.get("G2")))
    except Exception:
        return None
    actual_outcome = score_outcome(actual_home, actual_away)
    outcome_probabilities = {
        "home": float_or_zero(probabilities.get("home")),
        "draw": float_or_zero(probabilities.get("draw")),
        "away": float_or_zero(probabilities.get("away")),
    }
    pick = outcome_decision(outcome_probabilities)
    total_goals = actual_home + actual_away
    grid = model_score_grid_for_fixture(
        model=model,
        home=str(fixture.get("Equipo 1", "")),
        away=str(fixture.get("Equipo 2", "")),
        fixture=fixture,
        lambda_home=float_or_zero(probabilities.get("lambda1")),
        lambda_away=float_or_zero(probabilities.get("lambda2")),
        max_goals=int(config.get("max_goals") or DEFAULT_CONFIG["max_goals"]),
    )
    top_score_predictions = []
    actual_score_key = f"{actual_home}-{actual_away}"
    for rank, score_item in enumerate(
            sorted(score_distribution_cells(grid), key=lambda item: item["probability_raw"], reverse=True)[:5],
            start=1,
    ):
        top_score_predictions.append({
            "rank": rank,
            "score": score_item.get("score", ""),
            "probability": score_item.get("probability", 0.0),
            "hit": score_item.get("score") == actual_score_key,
        })
    modal_index = int(np.argmax(grid))
    modal_home, modal_away = np.unravel_index(modal_index, grid.shape)
    modal_probability = float(grid[int(modal_home), int(modal_away)])
    score_probability = score_grid_actual_probability(grid, actual_home, actual_away)
    actual_score_rank = score_rank_from_grid(grid, actual_home, actual_away)
    expected_home, expected_away, expected_total, expected_margin = expected_score_from_grid(grid)
    actual_total = float(actual_home + actual_away)
    actual_margin = float(actual_home - actual_away)
    over_under_rows = backtest_over_under_rows(probabilities, total_goals)
    confidence = max(outcome_probabilities.values()) if outcome_probabilities else 0.0
    modal_score = f"{int(modal_home)}-{int(modal_away)}"
    modal_hit = int(modal_home) == actual_home and int(modal_away) == actual_away
    home_team = str(fixture.get("Equipo 1", ""))
    away_team = str(fixture.get("Equipo 2", ""))
    return {
        "log_loss": -math.log(max(outcome_probabilities.get(actual_outcome, 0.0), 1e-12)),
        "brier": multiclass_brier_score(outcome_probabilities, actual_outcome),
        "score_log_loss": -math.log(max(score_probability, 1e-12)),
        "rps": ranked_probability_score(outcome_probabilities, actual_outcome),
        "entropy": probability_entropy(outcome_probabilities),
        "sharpness": confidence,
        "pick_hit": 1 if pick == actual_outcome else 0,
        "score_hit": 1 if modal_hit else 0,
        "top3_score_hit": 1 if actual_score_rank <= 3 else 0,
        "top5_score_hit": 1 if actual_score_rank <= 5 else 0,
        "expected_home_goals": expected_home,
        "expected_away_goals": expected_away,
        "expected_total_goals": expected_total,
        "expected_margin": expected_margin,
        "home_goals_abs_error": abs(expected_home - float(actual_home)),
        "away_goals_abs_error": abs(expected_away - float(actual_away)),
        "total_goals_abs_error": abs(expected_total - actual_total),
        "margin_abs_error": abs(expected_margin - actual_margin),
        "home_goals_squared_error": (expected_home - float(actual_home)) ** 2,
        "away_goals_squared_error": (expected_away - float(actual_away)) ** 2,
        "total_goals_squared_error": (expected_total - actual_total) ** 2,
        "margin_squared_error": (expected_margin - actual_margin) ** 2,
        "actual_outcome": actual_outcome,
        "predicted_outcome": pick,
        "confidence": confidence,
        "over_under": over_under_rows,
        "sample": {
            "fixture_id": str(row.get("No.", "")),
            "date": str(row.get("Date", "")),
            "home": home_team,
            "away": away_team,
            "home_asset": team_asset(home_team),
            "away_asset": team_asset(away_team),
            "match": f"{home_team} vs {away_team}",
            "actual_score": f"{actual_home}-{actual_away}",
            "actual_home_goals": actual_home,
            "actual_away_goals": actual_away,
            "total_goals": int(total_goals),
            "pick": outcome_label(pick),
            "pick_key": pick,
            "actual_pick": outcome_label(actual_outcome),
            "actual_pick_key": actual_outcome,
            "pick_hit": bool(pick == actual_outcome),
            "modal_score": modal_score,
            "most_probable_score": modal_score,
            "most_probable_score_probability": round(modal_probability * 100.0, 3),
            "most_probable_score_hit": bool(modal_hit),
            "top_scores": top_score_predictions,
            "expected_score": f"{expected_home:.2f}-{expected_away:.2f}",
            "actual_score_rank": actual_score_rank,
            "score_hit": bool(modal_hit),
            "top3_score_hit": bool(actual_score_rank <= 3),
            "top5_score_hit": bool(actual_score_rank <= 5),
            "rps": round(ranked_probability_score(outcome_probabilities, actual_outcome), 6),
            "confidence": round(confidence * 100.0, 3),
            "actual_probability": round(outcome_probabilities.get(actual_outcome, 0.0) * 100.0, 3),
            "score_probability": round(score_probability * 100.0, 3),
            "feature_context": probabilities.get("feature_context", {}),
            "probabilities": {
                "home": round(outcome_probabilities["home"] * 100.0, 3),
                "draw": round(outcome_probabilities["draw"] * 100.0, 3),
                "away": round(outcome_probabilities["away"] * 100.0, 3),
            },
            "over_under": over_under_rows,
        },
    }


def accumulate_backtest_totals(totals: Dict[str, Any], row_metrics: Dict[str, Any]) -> None:
    totals["log_loss"] += float_or_zero(row_metrics.get("log_loss"))
    totals["brier"] += float_or_zero(row_metrics.get("brier"))
    totals["score_log_loss"] += float_or_zero(row_metrics.get("score_log_loss"))
    totals["rps"] += float_or_zero(row_metrics.get("rps"))
    totals["entropy"] += float_or_zero(row_metrics.get("entropy"))
    totals["sharpness"] += float_or_zero(row_metrics.get("sharpness"))
    totals["pick_hits"] += int(row_metrics.get("pick_hit") or 0)
    totals["score_hits"] += int(row_metrics.get("score_hit") or 0)
    totals["top3_score_hits"] += int(row_metrics.get("top3_score_hit") or 0)
    totals["top5_score_hits"] += int(row_metrics.get("top5_score_hit") or 0)
    for key in (
            "home_goals_abs_error",
            "away_goals_abs_error",
            "total_goals_abs_error",
            "margin_abs_error",
            "home_goals_squared_error",
            "away_goals_squared_error",
            "total_goals_squared_error",
            "margin_squared_error",
    ):
        totals[key] += float_or_zero(row_metrics.get(key))
    actual = str(row_metrics.get("actual_outcome") or "")
    predicted = str(row_metrics.get("predicted_outcome") or "")
    if actual in OUTCOME_KEYS and predicted in OUTCOME_KEYS:
        totals["confusion_matrix"].setdefault(actual, {key: 0 for key in OUTCOME_KEYS})
        totals["confusion_matrix"][actual][predicted] = int(totals["confusion_matrix"][actual].get(predicted) or 0) + 1
    confidence = float(np.clip(float_or_zero(row_metrics.get("confidence")), 0.0, 1.0))
    bin_index = min(int(confidence * CALIBRATION_BIN_COUNT), CALIBRATION_BIN_COUNT - 1)
    calibration_bins = totals.get("calibration_bins") or []
    if 0 <= bin_index < len(calibration_bins):
        calibration_bins[bin_index]["count"] += 1
        calibration_bins[bin_index]["confidence_sum"] += confidence
        calibration_bins[bin_index]["correct_sum"] += 1.0 if row_metrics.get("pick_hit") else 0.0
    for item in row_metrics.get("over_under", []) or []:
        line_key = str(item.get("line") or "")
        totals["over_under_total"] += 1
        totals["over_under_hits"] += 1 if item.get("hit") else 0
        line_totals = totals["over_under_by_line"].setdefault(line_key, {"hits": 0, "total": 0, "log_loss": 0.0, "brier": 0.0})
        line_totals["total"] += 1
        line_totals["hits"] += 1 if item.get("hit") else 0
        line_totals["log_loss"] += float_or_zero(item.get("log_loss"))
        line_totals["brier"] += float_or_zero(item.get("brier"))
    totals["evaluated"] += 1


def backtest_over_under_rows(probabilities: Dict[str, Any], total_goals: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in REPORT_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        over_probability = float_or_zero(probabilities.get(f"over{suffix}"))
        under_probability = float_or_zero(probabilities.get(f"under{suffix}"))
        predicted = "over" if over_probability >= under_probability else "under"
        actual = "over" if int(total_goals) > float(line) else "under"
        actual_probability = over_probability if actual == "over" else under_probability
        actual_over = 1.0 if actual == "over" else 0.0
        rows.append({
            "line": f"{line:.1f}",
            "prediction": predicted,
            "prediction_label": "Over" if predicted == "over" else "Under",
            "actual": actual,
            "actual_label": "Over" if actual == "over" else "Under",
            "hit": predicted == actual,
            "over_probability": round(over_probability * 100.0, 3),
            "under_probability": round(under_probability * 100.0, 3),
            "actual_probability": round(actual_probability * 100.0, 3),
            "log_loss": round(-math.log(max(actual_probability, 1e-12)), 6),
            "brier": round((over_probability - actual_over) ** 2, 6),
            "confidence": round(max(over_probability, under_probability) * 100.0, 3),
        })
    return rows


def over_under_accuracy_by_line(totals: Dict[str, Any]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for line, values in (totals.get("over_under_by_line") or {}).items():
        total = max(int(values.get("total") or 0), 1)
        output[str(line)] = round(float(values.get("hits") or 0) / total, 6)
    return output


def over_under_metrics_by_line(totals: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    output: Dict[str, Dict[str, float]] = {}
    for line, values in (totals.get("over_under_by_line") or {}).items():
        total = max(int(values.get("total") or 0), 1)
        output[str(line)] = {
            "accuracy": round(float(values.get("hits") or 0) / total, 6),
            "log_loss": round(float(values.get("log_loss") or 0.0) / total, 6),
            "brier": round(float(values.get("brier") or 0.0) / total, 6),
            "total": int(values.get("total") or 0),
        }
    return output


def compare_backtest_to_baseline(metrics: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    log_loss_delta = round(float_or_zero(metrics.get("log_loss")) - float_or_zero(baseline.get("log_loss")), 6)
    brier_delta = round(float_or_zero(metrics.get("brier")) - float_or_zero(baseline.get("brier")), 6)
    rps_delta = round(float_or_zero(metrics.get("rps")) - float_or_zero(baseline.get("rps")), 6)
    pick_delta = round(float_or_zero(metrics.get("pick_accuracy")) - float_or_zero(baseline.get("pick_accuracy")), 6)
    score_delta = round(float_or_zero(metrics.get("score_accuracy")) - float_or_zero(baseline.get("score_accuracy")), 6)
    top3_delta = round(float_or_zero(metrics.get("top3_score_accuracy")) - float_or_zero(baseline.get("top3_score_accuracy")), 6)
    ou25_delta = round(float_or_zero(metrics.get("ou25_log_loss")) - float_or_zero(baseline.get("ou25_log_loss")), 6)
    wins = (
        int(log_loss_delta < 0.0)
        + int(brier_delta < 0.0)
        + int(rps_delta < 0.0)
        + int(pick_delta > 0.0)
        + int(score_delta > 0.0)
        + int(top3_delta > 0.0)
        + int(ou25_delta < 0.0)
    )
    beats = wins >= 4 or (log_loss_delta < 0.0 and rps_delta < 0.0 and (brier_delta < 0.0 or pick_delta > 0.0 or score_delta > 0.0 or top3_delta > 0.0))
    return {
        **metrics,
        "baseline_model_key": DEFAULT_SCORE_MODEL,
        "baseline_model_label": score_model_display_label(DEFAULT_SCORE_MODEL),
        "vs_poisson": {
            "log_loss_delta": log_loss_delta,
            "brier_delta": brier_delta,
            "rps_delta": rps_delta,
            "pick_accuracy_delta": pick_delta,
            "score_accuracy_delta": score_delta,
            "top3_score_accuracy_delta": top3_delta,
            "ou25_log_loss_delta": ou25_delta,
            "metric_wins": wins,
            "metric_total": 7,
            "beats_poisson": bool(beats),
            "summary": backtest_delta_summary(log_loss_delta, brier_delta, rps_delta, pick_delta, score_delta, top3_delta, ou25_delta, wins),
        },
    }


def backtest_delta_summary(
        log_loss_delta: float,
        brier_delta: float,
        rps_delta: float,
        pick_delta: float,
        score_delta: float,
        top3_delta: float,
        ou25_delta: float,
        wins: int,
) -> str:
    return (
        f"{wins}/7 metricas; "
        f"LL {format_signed_metric(log_loss_delta)}; "
        f"RPS {format_signed_metric(rps_delta)}; "
        f"Brier {format_signed_metric(brier_delta)}; "
        f"pick {format_signed_pp(pick_delta)}; "
        f"marcador#1 {format_signed_pp(score_delta)}; "
        f"top3 {format_signed_pp(top3_delta)}; "
        f"U/O2.5 {format_signed_metric(ou25_delta)}"
    )


def best_alternative_from_backtests(backtests: List[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = [
        item for item in backtests
        if item.get("available") and int(item.get("evaluated_matches") or 0) > 0
    ]
    if not candidates:
        return {"available": False, "reason": "Sin modelos evaluables en backtest."}
    winner = dict(sorted(candidates, key=lambda item: int(item.get("rank") or 9999))[0])
    winner["available"] = True
    winner["selection_policy"] = "Modelo #1 del backtesting walk-forward por Score de resultados; log-loss solo desempata tecnicamente. Sin ponderar otros modelos."
    return winner


def alternatives_with_backtests(alternatives: List[Dict[str, Any]], backtest_by_key: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for item in alternatives:
        key = str(item.get("key") or "")
        output.append({
            **item,
            "backtest": backtest_by_key.get(key, {}),
        })
    return output


def worldcup_feature_research_summary(feature_source: BenchmarkFeatureSource | None = None) -> Dict[str, Any]:
    feature_store_files = sorted(FEATURE_STORE_ROOT.glob("*.json")) if FEATURE_STORE_ROOT.exists() else []
    market_root = PROJECT_ROOT / "storage" / "worldcup" / "market"
    xg_root = PROJECT_ROOT / "storage" / "worldcup" / "xg"
    api_root = PROJECT_ROOT / "storage" / "worldcup" / "api_football"
    lineup_root = PROJECT_ROOT / "storage" / "worldcup" / "lineups"
    source_status = benchmark_feature_source_status(feature_source)
    families = [
        {
            "key": "dynamic_team_strength",
            "label": "Ratings dinamicos y forma reciente",
            "status": "active",
            "features": ["Elo/ataque/defensa", "ventanas 3/5/10/15", "oponente ajustado", "recencia"],
            "impact": "Mejora lambdas base y reduce dependencia de promedios historicos largos.",
        },
        {
            "key": "market_odds",
            "label": "Cuotas de mercado",
            "status": "cached" if source_status.get("odds_rows", 0) > 0 or (market_root.exists() and any(market_root.glob("*.csv"))) else "optional",
            "features": ["1X2 no-vig", "U/O 2.5 no-vig", "movimiento de linea", "benchmark vs mercado"],
            "impact": "Las cuotas suelen ser un baseline muy fuerte; se usan como feature solo si existen antes del partido.",
        },
        {
            "key": "xg_xga",
            "label": "xG/xGA y tiros",
            "status": "cached" if xg_root.exists() and any(xg_root.glob("*.csv")) else "optional",
            "features": ["xG rolling", "xGA rolling", "diferencial xG", "tiros y tiros al arco"],
            "impact": "Ajusta la calidad de ocasiones, no solo goles observados.",
        },
        {
            "key": "lineups_players",
            "label": "Alineaciones, lesiones y jugadores",
            "status": "cached" if source_status.get("xi_rows", 0) > 0 or (lineup_root.exists() and any(lineup_root.glob("*.json"))) else "optional",
            "features": ["XI confirmado/probable", "ratings por jugador", "minutos", "bajas"],
            "impact": "Permite multiplicadores de lambda cuando hay XI confiable antes del kickoff.",
        },
        {
            "key": "api_football_context",
            "label": "API-Football contextual",
            "status": "cached" if source_status.get("api_team_stats_rows", 0) > 0 or (api_root.exists() and any(api_root.glob("*.json"))) else "optional",
            "features": ["estadisticas recientes", "odds", "lineups", "injuries"],
            "impact": "Fuente opcional; el reporte no debe depender de red ni credenciales para funcionar.",
        },
    ]
    active = [item["key"] for item in families if item["status"] in {"active", "cached"}]
    return {
        "anti_leakage": "Toda feature debe calcularse con corte temporal anterior al partido evaluado.",
        "recommendation": "Primero usar ratings/form + cuotas/xG/XI cacheados; despues comparar contra mercado y Poisson por walk-forward.",
        "feature_store_files": [str(path) for path in feature_store_files[:8]],
        "active_or_cached_families": active,
        "source_status": source_status,
        "families": families,
        "research_basis": [
            {
                "title": "Dixon-Coles 1997",
                "url": "https://doi.org/10.1111/1467-9876.00065",
                "finding": "Corrige marcadores bajos y usa decaimiento temporal.",
                "apply": "Mantener Dixon-Coles como candidato estadistico activo.",
            },
            {
                "title": "Pi-ratings / Elo dinamico",
                "url": "https://doi.org/10.1007/s10994-012-5285-8",
                "finding": "Ratings locales/visitantes capturan fuerza cambiante por equipo.",
                "apply": "Seguir agregando forma, ratings y ajuste por rival con corte temporal.",
            },
            {
                "title": "Market odds as benchmark",
                "url": "https://arxiv.org/abs/2604.17194",
                "finding": "Cuotas no-vig son un baseline fuerte y revelan sesgo favorito-longshot.",
                "apply": "Usar odds cacheadas como feature y como comparador, no como dato en vivo obligatorio.",
            },
            {
                "title": "xG+ team signal",
                "url": "https://arxiv.org/abs/2512.00203",
                "finding": "Modelar probabilidad de tiro + calidad mejora senal de equipo.",
                "apply": "Priorizar xG/xGA rolling y volumen/calidad de tiros cuando exista fuente historica.",
            },
            {
                "title": "VAEP / socceraction",
                "url": "https://github.com/ML-KULeuven/socceraction",
                "finding": "Valora acciones con contexto mediante SPADL, VAEP y xT.",
                "apply": "Agregar features de jugadores/acciones solo si vienen cacheadas antes del partido.",
            },
            {
                "title": "footBayes dinamico",
                "url": "https://github.com/LeoEgidi/footBayes",
                "finding": "Modelos bayesianos dinamicos mejoran fuerza ataque/defensa cuando hay historia suficiente.",
                "apply": "Dejar Bayes como linea futura evaluada por benchmark antes de activarla.",
            },
        ],
    }


def benchmark_feature_source_status(feature_source: BenchmarkFeatureSource | None) -> Dict[str, Any]:
    if feature_source is None:
        return {
            "history_rows": 0,
            "odds_rows": 0,
            "qualifier_rows": 0,
            "api_team_stats_rows": 0,
            "api_lineup_rows": 0,
            "api_injury_rows": 0,
            "international_rows": 0,
            "xi_rows": 0,
            "warnings": [],
        }
    return {
        "history_rows": int(feature_source.history_df.shape[0]),
        "odds_rows": int(feature_source.market_rows.shape[0]) if feature_source.market_rows is not None else 0,
        "qualifier_rows": int(feature_source.qualifier_rows.shape[0]) if feature_source.qualifier_rows is not None else 0,
        "api_team_stats_rows": int((feature_source.api_football.get("team_stats", pd.DataFrame())).shape[0]),
        "api_lineup_rows": int((feature_source.api_football.get("lineups", pd.DataFrame())).shape[0]),
        "api_injury_rows": int((feature_source.api_football.get("injuries", pd.DataFrame())).shape[0]),
        "international_rows": int(feature_source.international_matches.shape[0]) if feature_source.international_matches is not None else 0,
        "xi_rows": int(feature_source.fixture_feature_rows.shape[0]) if feature_source.fixture_feature_rows is not None else 0,
        "warnings": unique_strings(feature_source.warnings),
    }


def alternatives_benchmark_table_rows(
        fixture_reports: List[Dict[str, Any]],
        backtest_by_key: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in fixture_reports:
        fixture = report.get("fixture", {})
        model = report.get("primary_model") or {}
        probs = model.get("probabilities") or {}
        top_scores = ", ".join(
            f"{score.get('score', '')} {format_metric(score.get('probability', ''))}%"
            for score in (model.get("top_scores") or [])[:3]
        )
        rows.append({
            "No.": fixture.get("id", ""),
            "Fecha": fixture.get("date", ""),
            "Grupo": fixture.get("group", ""),
            "Partido": fixture.get("label", ""),
            "Marcador #1": model.get("top_score", ""),
            "Marcador #1 %": model.get("top_score_probability", ""),
            "Top scores": top_scores,
            "1 %": probs.get("home", ""),
            "X %": probs.get("draw", ""),
            "2 %": probs.get("away", ""),
            "Over 0.5 %": probs.get("over05", ""),
            "Under 0.5 %": probs.get("under05", ""),
            "Over 1.5 %": probs.get("over15", ""),
            "Under 1.5 %": probs.get("under15", ""),
            "Over 2.5 %": probs.get("over25", ""),
            "Under 2.5 %": probs.get("under25", ""),
            "Over 3.5 %": probs.get("over35", ""),
            "Under 3.5 %": probs.get("under35", ""),
        })
    return rows


def score_outcome(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if away_goals > home_goals:
        return "away"
    return "draw"


def multiclass_brier_score(probabilities: Dict[str, float], actual: str) -> float:
    return float(sum((float_or_zero(probabilities.get(key)) - (1.0 if key == actual else 0.0)) ** 2 for key in ("home", "draw", "away")))


def score_grid_actual_probability(grid: np.ndarray, home_goals: int, away_goals: int) -> float:
    grid = normalize_score_grid_array(grid)
    home = min(max(int(home_goals), 0), grid.shape[0] - 1)
    away = min(max(int(away_goals), 0), grid.shape[1] - 1)
    return float(grid[home, away])


def format_signed_metric(value: Any) -> str:
    number = float_or_zero(value)
    return f"{number:+.4f}"


def format_signed_pp(value: Any) -> str:
    number = float_or_zero(value) * 100.0
    return f"{number:+.1f}pp"


def format_metric(value: Any) -> str:
    number = float_or_zero(value)
    return f"{number:.2f}".rstrip("0").rstrip(".")


def upcoming_sota_fixture_reports(
        tournament: Dict[str, Any],
        base_model: WorldCupModel,
        fixtures: List[pd.Series],
        config: Dict[str, Any],
        start_time: float,
        hardware: Dict[str, Any],
        model_sequence: List[str] | Tuple[str, ...] | None = None,
        history_df: pd.DataFrame | None = None,
        feature_source: BenchmarkFeatureSource | None = None,
        progress_callback=None,
) -> List[Dict[str, Any]]:
    fixture_reports = [
        {
            "fixture": report_fixture_payload({
                "id": str(fixture.get("No.", "")),
                "date": fixture.get("Fecha", ""),
                "time": fixture.get("Hora", ""),
                "group": fixture.get("Grupo", ""),
                "home": str(fixture.get("Equipo 1", "")),
                "away": str(fixture.get("Equipo 2", "")),
                "venue": fixture.get("Sede", ""),
                "finished": str(fixture.get("Finalizado", "")).strip().lower() in {"si", "sí", "yes", "true", "1"},
            }),
            "contextual_poisson": contextual_poisson_for_match(
                str(fixture.get("Equipo 1", "")),
                str(fixture.get("Equipo 2", "")),
                base_model=base_model,
                before_date=fixture.get("Fecha", ""),
                max_goals=int(config["max_goals"]),
                limit=int(config["poisson_recent_matches"]),
            ),
            "models": [],
        }
        for fixture in fixtures
    ]
    group_map = groups_from_tournament(tournament)
    team_names = [team for group_teams in group_map.values() for team in group_teams]
    if history_df is None or history_df.empty:
        history_df, _ = score_history_for_tournament(tournament, config)
    international_matches = load_international_matches(required=False)
    international_status = international_results_status()
    for report, fixture in zip(fixture_reports, fixtures):
        report["recent_matches_15"] = recent_matches_for_fixture(
            history_df,
            fixture,
            limit=15,
            international_matches=international_matches,
            international_status=international_status,
        )
    sequence = list(model_sequence or SOTA_SCORE_MODEL_SEQUENCE)
    model_total = len(sequence)
    fixture_total = max(len(fixtures), 1)
    for model_index, model_key in enumerate(sequence, start=1):
        emit_report_progress(
            progress_callback,
            stage="fitting",
            start_time=start_time,
            model_index=model_index,
            model_total=model_total,
            model_key=model_key,
            fixture_index=0,
            fixture_total=fixture_total,
            hardware=hardware,
            message=f"Ajustando {score_model_display_label(model_key)}",
        )
        score_model = base_model
        metadata = {"key": model_key, "label": score_model_display_label(model_key), "available": True, "params": {}, "warnings": []}
        try:
            if model_key == DEFAULT_SCORE_MODEL:
                score_model = base_model
                metadata = score_model_metadata(score_model)
            else:
                score_model = build_score_model(
                    base_model,
                    history_df=history_df,
                    teams=team_names,
                    config={**config, "score_model": model_key},
                )
                metadata = score_model_metadata(score_model)
        except Exception as exc:
            score_model = base_model
            metadata = {
                "key": model_key,
                "label": score_model_display_label(model_key),
                "available": False,
                "params": {},
                "warnings": [f"{exc.__class__.__name__}: {exc}; se usa Poisson independiente."],
            }
        score_model = apply_recent_context_model(score_model, config)
        score_model = apply_benchmark_feature_model(score_model, model_key, feature_source, history_df=history_df)
        batched_reports = batched_score_model_reports_for_fixtures(
            score_model=score_model,
            model_key=model_key,
            metadata=metadata,
            fixtures=fixtures,
            config=config,
            hardware=hardware,
            feature_source=feature_source,
        )
        if batched_reports is not None:
            for fixture_index, (fixture, model_report) in enumerate(zip(fixtures, batched_reports), start=1):
                emit_report_progress(
                    progress_callback,
                    stage="predicting",
                    start_time=start_time,
                    model_index=model_index,
                    model_total=model_total,
                    model_key=model_key,
                    fixture_index=fixture_index,
                    fixture_total=fixture_total,
                    hardware=hardware,
                    message=f"{score_model_display_label(model_key)}: {fixture.get('Equipo 1', '')} vs {fixture.get('Equipo 2', '')}",
                )
                fixture_reports[fixture_index - 1]["models"].append(model_report)
            continue
        for fixture_index, fixture in enumerate(fixtures, start=1):
            emit_report_progress(
                progress_callback,
                stage="predicting",
                start_time=start_time,
                model_index=model_index,
                model_total=model_total,
                model_key=model_key,
                fixture_index=fixture_index,
                fixture_total=fixture_total,
                hardware=hardware,
                message=f"{score_model_display_label(model_key)}: {fixture.get('Equipo 1', '')} vs {fixture.get('Equipo 2', '')}",
            )
            probabilities = model_probabilities_for_fixture(score_model, fixture, config)
            model_report = score_prediction_model_report(
                model_key=model_key,
                metadata=metadata,
                probabilities=probabilities,
                fixture=fixture,
                config=config,
                already_percent=False,
            )
            score_distribution = score_distribution_for_fixture(score_model, fixture, probabilities, config)
            model_report["score_distribution"] = score_distribution
            model_report["top_scores"] = score_distribution.get("top_scores", [])
            if model_report["top_scores"]:
                model_report["top_score"] = model_report["top_scores"][0].get("score", model_report.get("top_score", ""))
                model_report["top_score_probability"] = model_report["top_scores"][0].get("probability", 0.0)
            model_report["heatmap"] = score_distribution.get("heatmap", {})
            fixture_reports[fixture_index - 1]["models"].append(model_report)
    return fixture_reports


def batched_score_model_reports_for_fixtures(
        score_model: Any,
        model_key: str,
        metadata: Dict[str, Any],
        fixtures: List[pd.Series],
        config: Dict[str, Any],
        hardware: Dict[str, Any],
        feature_source: BenchmarkFeatureSource | None = None,
) -> List[Dict[str, Any]] | None:
    requested_backend = str((hardware or {}).get("score_backend") or config.get("score_backend") or "numpy").strip().lower()
    if requested_backend != "cupy" or feature_source is not None or not fixtures:
        return None
    key = normalize_score_model_key(model_key)
    if key not in SOTA_SCORE_MODEL_SEQUENCE:
        return None
    lambda_home: List[float] = []
    lambda_away: List[float] = []
    try:
        for fixture in fixtures:
            home_lambda, away_lambda = expected_lambdas_for_batched_report(score_model, fixture, config)
            lambda_home.append(float(home_lambda))
            lambda_away.append(float(away_lambda))
        grids, backend, backend_warnings = score_grids_from_lambdas_with_backend(
            metadata,
            lambda1_values=lambda_home,
            lambda2_values=lambda_away,
            max_goals=int(config.get("max_goals") or DEFAULT_CONFIG["max_goals"]),
            backend=requested_backend,
        )
        probability_rows = probabilities_from_score_grids(grids, lambda_home, lambda_away)
    except Exception:
        return None
    reports: List[Dict[str, Any]] = []
    metadata_with_backend = dict(metadata)
    metadata_warnings = [str(item) for item in metadata_with_backend.get("warnings", []) if str(item)]
    metadata_with_backend["warnings"] = unique_strings([
        *metadata_warnings,
        *[str(item) for item in backend_warnings if str(item)],
    ])
    params = dict(metadata_with_backend.get("params") or {})
    params["score_backend"] = backend
    metadata_with_backend["params"] = params
    for index, (fixture, probabilities) in enumerate(zip(fixtures, probability_rows)):
        probabilities = dict(probabilities)
        probabilities["score_model"] = key
        probabilities["score_model_label"] = metadata_with_backend.get("label", score_model_display_label(key))
        probabilities["score_model_available"] = bool(metadata_with_backend.get("available", True))
        model_report = score_prediction_model_report(
            model_key=key,
            metadata=metadata_with_backend,
            probabilities=probabilities,
            fixture=fixture,
            config=config,
            already_percent=False,
        )
        score_distribution = score_distribution_payload(
            grids[index],
            lambda_home=float(lambda_home[index]),
            lambda_away=float(lambda_away[index]),
        )
        model_report["score_distribution"] = score_distribution
        model_report["top_scores"] = score_distribution.get("top_scores", [])
        if model_report["top_scores"]:
            model_report["top_score"] = model_report["top_scores"][0].get("score", model_report.get("top_score", ""))
            model_report["top_score_probability"] = model_report["top_scores"][0].get("probability", 0.0)
        model_report["heatmap"] = score_distribution.get("heatmap", {})
        model_report["score_backend"] = backend
        reports.append(model_report)
    return reports


def expected_lambdas_for_batched_report(model: Any, fixture: pd.Series, config: Dict[str, Any]) -> Tuple[float, float]:
    home = str(fixture.get("Equipo 1", ""))
    away = str(fixture.get("Equipo 2", ""))
    if isinstance(model, RecentPoissonWorldCupModel):
        context = contextual_poisson_for_match(
            home,
            away,
            base_model=model.base_model,
            before_date=fixture.get("Fecha", ""),
            max_goals=int(config.get("max_goals") or DEFAULT_CONFIG["max_goals"]),
            matches=getattr(model, "matches", pd.DataFrame()),
            limit=int(getattr(model, "recent_match_limit", config.get("poisson_recent_matches") or DEFAULT_CONFIG["poisson_recent_matches"])),
        )
        try:
            lambda_home = float(context.get("context_lambda_home") or (context.get("lambdas") or {}).get("home") or 0.0)
            lambda_away = float(context.get("context_lambda_away") or (context.get("lambdas") or {}).get("away") or 0.0)
        except (TypeError, ValueError):
            lambda_home = 0.0
            lambda_away = 0.0
        if context.get("matrix_available") and lambda_home > 0.0 and lambda_away > 0.0:
            return lambda_home, lambda_away
        return expected_lambdas_for_batched_report(model.base_model, fixture, config)
    record = fixture.to_dict() if hasattr(fixture, "to_dict") else dict(fixture)
    method = getattr(model, "expected_goals_for_match", None)
    if callable(method):
        return method(home, away, match=record)
    return model.expected_goals(home, away)


def normalize_report_pipeline_mode(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or POISSON_SOTA_PIPELINE_MODE).strip().lower()).strip("_")
    if normalized in {
        XG_LIGHTGBM_PIPELINE_MODE,
        "xg_light_gbm",
        "xg_lgbm",
        "xglightgbm",
        "xg_lightgbm_cuda",
        "lightgbm_xg",
        "lightgbm_xg_cuda",
        "xg",
    }:
        return XG_LIGHTGBM_PIPELINE_MODE
    if normalized in {
        ADVANCED_MODELS_PIPELINE_MODE,
        "advanced",
        "advanced_modelos",
        "modelos_avanzados",
        "todo_documento",
        "all_advanced_models",
        "document_models",
    }:
        return ADVANCED_MODELS_PIPELINE_MODE
    if normalized in {
        ALTERNATIVES_BENCHMARK_PIPELINE_MODE,
        "alternatives",
        "alternative_benchmark",
        "benchmark_alternativas",
        "benchmark_alternatives",
        "alternativas",
        "sota_alternatives",
        "sota_alternativas",
        "modelos_mejores",
        "mejores_modelos",
        "research_benchmark",
    }:
        return ALTERNATIVES_BENCHMARK_PIPELINE_MODE
    return POISSON_SOTA_PIPELINE_MODE


def normalize_sota_calculation_mode(value: Any) -> str:
    mode = str(value or "exact").strip().lower().replace("-", "_")
    return "monte_carlo" if mode in {"monte_carlo", "montecarlo", "mc", "simulation"} else "exact"


def sota_calculation_summary(config: Dict[str, Any]) -> str:
    if config.get("sota_calculation_mode") == "monte_carlo":
        return f"SOTA Monte Carlo sobre matriz consenso: N={int(config.get('iterations') or DEFAULT_CONFIG['iterations']):,}"
    return "Consenso exacto: matriz promedio, sin simulacion"


def report_pipeline_config(payload: Dict[str, Any], pipeline_mode: str) -> Dict[str, Any]:
    config = simulation_config(payload)
    config["pipeline_mode"] = pipeline_mode
    default_backtest_last_n = 20
    config["backtest_last_n"] = int(_clamp_int(payload.get("backtest_last_n", default_backtest_last_n), 5, 100))
    config["backtest_scope"] = "worldcup_2026_confirmed_auto"
    default_bayes_profile = "light" if pipeline_mode == ADVANCED_MODELS_PIPELINE_MODE else "deep"
    config["bayes_profile"] = str(payload.get("bayes_profile") or default_bayes_profile).strip().lower()
    config["sota_device"] = str(payload.get("sota_device") or "auto").strip().lower()
    config["sota_calculation_mode"] = normalize_sota_calculation_mode(payload.get("sota_calculation_mode"))
    config["advanced_include_bayesian"] = bool(payload.get("advanced_include_bayesian", DEFAULT_CONFIG["advanced_include_bayesian"]))
    if config["sota_device"] not in {"auto", "cpu", "cuda"}:
        config["sota_device"] = "auto"
    config["score_model"] = DEFAULT_SCORE_MODEL
    if pipeline_mode == XG_LIGHTGBM_PIPELINE_MODE:
        config["sota_calculation_mode"] = "not_applicable"
        config["bayes_profile"] = "not_applicable"
        config["stat_model_cache"] = True
        config["stat_model_refit"] = False
    if pipeline_mode == ADVANCED_MODELS_PIPELINE_MODE and config["bayes_profile"] != "deep":
        config["bayes_draws"] = 100
        config["bayes_tune"] = 100
        config["bayes_chains"] = 1
        config["stat_model_cache"] = True
        config["stat_model_refit"] = False
    if config["bayes_profile"] == "deep":
        config["bayes_draws"] = 2000
        config["bayes_tune"] = 2000
        config["bayes_chains"] = 4
        config["stat_model_cache"] = True
        config["stat_model_refit"] = False
    if pipeline_mode == ALTERNATIVES_BENCHMARK_PIPELINE_MODE:
        config["sota_calculation_mode"] = "exact"
        config["benchmark_tuning_enabled"] = bool(payload.get("benchmark_tuning_enabled", payload.get("tuning_enabled", False)))
        config["benchmark_tuning_trials"] = int(_clamp_int(payload.get("benchmark_tuning_trials", payload.get("n_trials", 20)), 1, 100))
        sampler = str(payload.get("benchmark_tuning_sampler", payload.get("optuna_sampler", "tpe")) or "tpe").strip().lower()
        config["benchmark_tuning_sampler"] = sampler if sampler in {"tpe", "random"} else "tpe"
    else:
        config["benchmark_tuning_enabled"] = False
        config["benchmark_tuning_trials"] = 0
        config["benchmark_tuning_sampler"] = "tpe"
    return config


def detect_hardware() -> Dict[str, Any]:
    cpu_count = int(os.cpu_count() or 1)
    devices: List[str] = []
    warning = ""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            completed = subprocess.run(
                [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            if completed.returncode == 0:
                devices = [
                    f"GPU {index}: {line.strip()}"
                    for index, line in enumerate(completed.stdout.splitlines())
                    if line.strip()
                ]
            else:
                warning = completed.stderr.strip() or "nvidia-smi no devolvio dispositivos"
        except Exception as exc:
            warning = f"nvidia-smi no disponible: {exc.__class__.__name__}"
    else:
        warning = "nvidia-smi no disponible"
    cuda_available = bool(devices)
    cuda_error = "" if cuda_available else (warning or "sin dispositivos CUDA detectados")
    return {
        "cpu_count": cpu_count,
        "default_n_jobs": -1,
        "effective_n_jobs": cpu_count,
        "cuda_available": cuda_available,
        "cuda_devices": devices,
        "cuda_device_names": cuda_device_names(devices),
        "cuda_detection_source": f"nvidia-smi:{nvidia_smi}" if nvidia_smi and cuda_available else "none",
        "cuda_detection_sources": [f"nvidia-smi:{nvidia_smi}"] if nvidia_smi and cuda_available else [],
        "cuda_error": cuda_error,
        "cuda_warning": warning,
        "device_default": "cuda" if cuda_available else "cpu",
    }


def cuda_device_names(devices: Iterable[Any]) -> List[str]:
    names = []
    for item in devices:
        text = str(item or "").strip()
        if not text:
            continue
        names.append(re.sub(r"^GPU\s+\d+\s*:\s*", "", text))
    return names


def stat_report_hardware(requested_device: Any, pipeline_mode: str, sota_calculation_mode: str = "exact") -> Dict[str, Any]:
    requested = str(requested_device or "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"}:
        requested = "auto"
    calculation_mode = normalize_sota_calculation_mode(sota_calculation_mode)
    detected = detect_hardware()
    warnings: List[str] = []
    backend_supports_cuda = False
    actual_device = "cpu"
    device_error = ""
    monte_carlo_backend = "numpy"
    score_status = score_backend_status(requested)
    score_backend = str(score_status.get("score_backend") or "numpy")
    score_backend_warning = str(score_status.get("warning") or "")
    if score_backend == "cupy":
        actual_device = "cuda"
        backend_supports_cuda = True
        detected = dict(detected)
        detected["cuda_available"] = True
        detected["device_default"] = "cuda"
        score_device_names = [str(item) for item in score_status.get("cuda_device_names", []) if str(item)]
        if score_device_names:
            detected["cuda_device_names"] = score_device_names
            if not detected.get("cuda_devices"):
                detected["cuda_devices"] = [f"GPU {index}: {name}" for index, name in enumerate(score_device_names)]
        sources = list(detected.get("cuda_detection_sources") or [])
        if "cupy" not in sources:
            sources.append("cupy")
        detected["cuda_detection_sources"] = sources
        if detected.get("cuda_detection_source") in {"", "none"}:
            detected["cuda_detection_source"] = "cupy"
        detected["cuda_error"] = ""
    if pipeline_mode == POISSON_SOTA_PIPELINE_MODE:
        cuda_reason = detected.get("cuda_error") or detected.get("cuda_warning") or "sin dispositivos"
        if calculation_mode == "monte_carlo":
            if requested == "cpu":
                warnings.append("Monte Carlo SOTA configurado en CPU por solicitud explicita.")
            elif score_backend == "cupy":
                monte_carlo_backend = "cupy"
            elif detected.get("cuda_available"):
                backend_name, backend_warning = monte_carlo_cuda_backend()
                if backend_name:
                    actual_device = "cuda"
                    backend_supports_cuda = True
                    monte_carlo_backend = backend_name
                    if score_backend != "cupy":
                        warnings.append(
                            f"CUDA activa para Monte Carlo SOTA via {backend_name}; scoring exacto/MLE usa NumPy porque CuPy no esta usable ({score_backend_warning or cuda_reason})."
                        )
                elif requested == "cuda":
                    device_error = f"CUDA fue solicitada explicitamente, pero no hay backend CuPy/Torch usable ({backend_warning or cuda_reason}); Monte Carlo SOTA corre en CPU."
                    warnings.append(device_error)
                else:
                    warnings.append(f"CUDA detectada, pero no hay backend CuPy/Torch usable ({backend_warning or cuda_reason}); Monte Carlo SOTA corre en CPU.")
            elif requested == "cuda":
                device_error = f"CUDA fue solicitada explicitamente, pero no se detecto GPU ({cuda_reason}); Monte Carlo SOTA corre en CPU."
                warnings.append(device_error)
            else:
                warnings.append(f"CUDA no disponible ({cuda_reason}); Monte Carlo SOTA corre en CPU.")
        elif requested == "cpu":
            actual_device = "cpu"
            backend_supports_cuda = False
            score_backend = "numpy"
        elif score_backend == "cupy":
            actual_device = "cuda"
            backend_supports_cuda = True
        elif requested == "cuda":
            device_error = f"CUDA fue solicitada explicitamente, pero CuPy no esta usable ({score_backend_warning or cuda_reason}); SOTA exacto corre en CPU/NumPy."
            warnings.append(device_error)
        elif requested == "auto" and detected.get("cuda_available"):
            warnings.append(f"CUDA detectada, pero CuPy no esta usable ({score_backend_warning or cuda_reason}); SOTA exacto corre en CPU/NumPy.")
        elif requested == "auto" and not detected.get("cuda_available"):
            warnings.append(f"CUDA no disponible ({cuda_reason}); SOTA corre en CPU.")
    return {
        **detected,
        "requested_device": requested,
        "actual_device": actual_device,
        "backend_supports_cuda": backend_supports_cuda,
        "monte_carlo_backend": monte_carlo_backend,
        "score_backend": score_backend,
        "score_backend_warning": score_backend_warning,
        "device_error": device_error,
        "warnings": warnings,
    }


def monte_carlo_cuda_backend() -> Tuple[str, str]:
    global _MONTE_CARLO_CUDA_BACKEND
    if _MONTE_CARLO_CUDA_BACKEND is not None:
        return _MONTE_CARLO_CUDA_BACKEND
    errors: List[str] = []
    try:
        import cupy as cp  # type: ignore

        device_count = int(cp.cuda.runtime.getDeviceCount())
        if device_count > 0:
            _MONTE_CARLO_CUDA_BACKEND = ("cupy", "")
            return _MONTE_CARLO_CUDA_BACKEND
        errors.append("CuPy sin dispositivos CUDA")
    except Exception as exc:
        errors.append(f"CuPy no disponible: {exc.__class__.__name__}")
    try:
        import torch  # type: ignore

        if bool(torch.cuda.is_available()):
            _MONTE_CARLO_CUDA_BACKEND = ("torch", "")
            return _MONTE_CARLO_CUDA_BACKEND
        errors.append("Torch sin CUDA disponible")
    except Exception as exc:
        errors.append(f"Torch no disponible: {exc.__class__.__name__}")
    _MONTE_CARLO_CUDA_BACKEND = ("", "; ".join(errors))
    return _MONTE_CARLO_CUDA_BACKEND


def emit_report_progress(
        callback,
        stage: str,
        start_time: float,
        model_index: int,
        model_total: int,
        model_key: str,
        fixture_index: int,
        fixture_total: int,
        hardware: Dict[str, Any],
        message: str,
        force_complete: bool = False,
):
    elapsed = max(time.monotonic() - float(start_time), 0.0)
    model_total = max(int(model_total or 1), 1)
    fixture_total = max(int(fixture_total or 1), 1)
    completed = max((max(int(model_index or 1), 1) - 1) * fixture_total + max(int(fixture_index or 0), 0), 0)
    total = max(model_total * fixture_total, 1)
    if force_complete:
        completed = total
    eta = 0
    if completed > 0 and completed < total:
        eta = int(round((elapsed / completed) * (total - completed)))
    emit_job_progress(
        callback,
        stage,
        completed,
        total,
        message,
        model_index=int(model_index or 0),
        model_total=model_total,
        model_key=model_key,
        fixture_index=int(fixture_index or 0),
        fixture_total=fixture_total,
        elapsed_seconds=round(elapsed, 1),
        eta_seconds=eta,
        hardware=hardware,
    )


def report_fixture_payload(fixture: Dict[str, Any]) -> Dict[str, Any]:
    home = str(fixture.get("home", ""))
    away = str(fixture.get("away", ""))
    kickoff_iso = fixture_kickoff_iso(fixture.get("date", ""), fixture.get("time", ""))
    return {
        "id": str(fixture.get("id", "")),
        "date": fixture.get("date", ""),
        "time": fixture.get("time", ""),
        "group": fixture.get("group", ""),
        "home": home,
        "away": away,
        "home_asset": team_asset(home),
        "away_asset": team_asset(away),
        "venue": fixture.get("venue", ""),
        "label": f"{home} vs {away}",
        "kickoff_iso": kickoff_iso,
        "countdown_state": fixture_countdown_state(kickoff_iso, fixture.get("finished", False)),
    }


def model_probabilities_for_fixture(model: Any, fixture: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    home = str(fixture.get("Equipo 1", ""))
    away = str(fixture.get("Equipo 2", ""))
    max_goals = int(config.get("max_goals") or DEFAULT_CONFIG["max_goals"])
    record = fixture.to_dict() if hasattr(fixture, "to_dict") else dict(fixture)
    method = getattr(model, "match_probabilities_for_match", None)
    if callable(method):
        return method(home, away, match=record, max_goals=max_goals)
    return model.match_probabilities(home, away, max_goals=max_goals)


def score_prediction_model_report(
        model_key: str,
        metadata: Dict[str, Any],
        probabilities: Dict[str, Any],
        fixture: pd.Series,
        config: Dict[str, Any],
        already_percent: bool,
) -> Dict[str, Any]:
    key = str(model_key or metadata.get("key") or DEFAULT_SCORE_MODEL)
    available = bool(metadata.get("available", True))
    probs_pct = {
        "home": probability_percent(probabilities.get("home", 0.0), already_percent=already_percent),
        "draw": probability_percent(probabilities.get("draw", 0.0), already_percent=already_percent),
        "away": probability_percent(probabilities.get("away", 0.0), already_percent=already_percent),
    }
    for line in REPORT_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        probs_pct[f"over{suffix}"] = probability_percent(probabilities.get(f"over{suffix}", 0.0), already_percent=already_percent)
        probs_pct[f"under{suffix}"] = probability_percent(probabilities.get(f"under{suffix}", 0.0), already_percent=already_percent)
    outcome = outcome_decision(probs_pct)
    totals = total_decisions(probs_pct)
    signature = consensus_signature(outcome, totals)
    lambda_home = float_or_zero(probabilities.get("lambda1", 0.0))
    lambda_away = float_or_zero(probabilities.get("lambda2", 0.0))
    modal_home = round_half_up_int(probabilities.get("modal_g1", lambda_home))
    modal_away = round_half_up_int(probabilities.get("modal_g2", lambda_away))
    return {
        "model_key": key,
        "model_label": str(metadata.get("label") or score_model_display_label(key)),
        "available": available,
        "consensus_eligible": bool(available or key == DEFAULT_SCORE_MODEL),
        "fallback": not available,
        "warnings": [str(item) for item in metadata.get("warnings", []) if str(item)],
        "decision": {
            "outcome": outcome,
            "label": outcome_label(outcome),
            "team": outcome_team(outcome, fixture),
        },
        "totals": totals,
        "signature": signature,
        "probabilities": probs_pct,
        "expected_goals": {
            "home": round(lambda_home, 3),
            "away": round(lambda_away, 3),
            "rounded_home": round_half_up_int(lambda_home),
            "rounded_away": round_half_up_int(lambda_away),
        },
        "top_score": f"{modal_home}-{modal_away}",
        "modal_score": f"{modal_home}-{modal_away}",
        "feature_context": probabilities.get("feature_context", {}),
        "params": metadata.get("params", {}),
        "source": "Poisson/SOTA",
    }


def score_distribution_for_fixture(model: Any, fixture: pd.Series, probabilities: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    home = str(fixture.get("Equipo 1", fixture.get("home", "")))
    away = str(fixture.get("Equipo 2", fixture.get("away", "")))
    lambda_home = float_or_zero(probabilities.get("lambda1", 0.0))
    lambda_away = float_or_zero(probabilities.get("lambda2", 0.0))
    max_goals = int(config.get("max_goals") or DEFAULT_CONFIG["max_goals"])
    grid = model_score_grid_for_fixture(model, home, away, fixture, lambda_home, lambda_away, max_goals=max_goals)
    return score_distribution_payload(grid, lambda_home=lambda_home, lambda_away=lambda_away)


def model_score_grid_for_fixture(
        model: Any,
        home: str,
        away: str,
        fixture: pd.Series,
        lambda_home: float,
        lambda_away: float,
        max_goals: int,
) -> np.ndarray:
    record = fixture.to_dict() if hasattr(fixture, "to_dict") else dict(fixture)
    method = getattr(model, "score_grid", None)
    if callable(method):
        try:
            return normalize_score_grid_array(method(home, away, match=record, max_goals=max_goals))
        except Exception:
            pass
    grid = match_score_grid_for_lambdas(model, lambda_home, lambda_away, max_goals=max_goals)
    if grid is not None:
        return normalize_score_grid_array(grid)
    return normalize_score_grid_array(poisson_score_grid(lambda_home, lambda_away, max_goals=max_goals))


def normalize_score_grid_array(grid: Any) -> np.ndarray:
    array = np.asarray(grid, dtype=float)
    if array.ndim != 2 or array.size == 0:
        return poisson_score_grid(1.2, 1.0, max_goals=REPORT_SCORE_MATRIX_GOALS)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.maximum(array, 0.0)
    total = float(array.sum())
    if total <= 0.0:
        return poisson_score_grid(1.2, 1.0, max_goals=max(array.shape[0] - 1, REPORT_SCORE_MATRIX_GOALS))
    return array / total


def score_distribution_payload(grid: np.ndarray, lambda_home: float, lambda_away: float) -> Dict[str, Any]:
    grid = normalize_score_grid_array(grid)
    cells = score_distribution_cells(grid)
    top_scores = sorted(cells, key=lambda item: item["probability_raw"], reverse=True)[:5]
    visible_goals = min(REPORT_SCORE_MATRIX_GOALS, grid.shape[0] - 1, grid.shape[1] - 1)
    visible_cells = [
        cell for cell in cells
        if cell["home_goals"] <= visible_goals and cell["away_goals"] <= visible_goals
    ]
    max_visible_probability = max((cell["probability"] for cell in visible_cells), default=0.0)
    probabilities = score_grid_probabilities(grid)
    return {
        "available": True,
        "lambdas": {"home": round(float(lambda_home), 3), "away": round(float(lambda_away), 3)},
        "probabilities": probabilities,
        "over_under": {
            f"{line:.1f}": {
                "over": probabilities.get(f"over{total_line_suffix(line)}", 0.0),
                "under": probabilities.get(f"under{total_line_suffix(line)}", 0.0),
            }
            for line in REPORT_TOTAL_GOAL_LINES
        },
        "score_matrix": [
            [round(float(grid[home_goals, away_goals]) * 100.0, 3) for away_goals in range(grid.shape[1])]
            for home_goals in range(grid.shape[0])
        ],
        "score_matrix_home_goals": list(range(grid.shape[0])),
        "score_matrix_away_goals": list(range(grid.shape[1])),
        "heatmap": {
            "home_goals": list(range(visible_goals + 1)),
            "away_goals": list(range(visible_goals + 1)),
            "max_probability": round(max_visible_probability, 3),
            "cells": visible_cells,
        },
        "top_scores": top_scores,
    }


def score_distribution_cells(grid: np.ndarray) -> List[Dict[str, Any]]:
    return [
        {
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
            "score": f"{home_goals}-{away_goals}",
            "probability": round(float(grid[home_goals, away_goals]) * 100.0, 3),
            "probability_raw": float(grid[home_goals, away_goals]),
        }
        for home_goals in range(grid.shape[0])
        for away_goals in range(grid.shape[1])
    ]


def score_grid_probabilities(grid: np.ndarray) -> Dict[str, float]:
    grid = normalize_score_grid_array(grid)
    goals = np.arange(grid.shape[0], dtype=int)
    home_goals, away_goals = np.meshgrid(goals, goals, indexing="ij")
    margin = home_goals - away_goals
    total_goals = home_goals + away_goals
    output = {
        "home": round(float(grid[margin > 0].sum()) * 100.0, 2),
        "draw": round(float(grid[margin == 0].sum()) * 100.0, 2),
        "away": round(float(grid[margin < 0].sum()) * 100.0, 2),
    }
    for line in REPORT_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        over = float(grid[total_goals > line].sum())
        output[f"over{suffix}"] = round(over * 100.0, 2)
        output[f"under{suffix}"] = round((1.0 - over) * 100.0, 2)
    return output


def fixture_model_analysis(model_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = [
        model for model in model_reports
        if model.get("consensus_eligible") and not model.get("fallback")
    ]
    return {
        "top_models_1x2": top_models_1x2(eligible),
        "consensus_score_distribution": consensus_score_distribution(eligible),
        "model_statistics": model_statistics_payload(eligible),
    }


def top_models_1x2(model_reports: List[Dict[str, Any]], limit: int = 4) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for model in model_reports:
        decision = model.get("decision") or {}
        outcome = str(decision.get("outcome") or "")
        probabilities = model.get("probabilities") or {}
        confidence = float_or_zero(probabilities.get(outcome, 0.0))
        ranked.append({
            "model_key": model.get("model_key", ""),
            "model_label": model.get("model_label", ""),
            "pick": outcome,
            "pick_label": decision.get("label", ""),
            "team": decision.get("team", ""),
            "confidence": round(confidence, 2),
            "top_score": model.get("top_score", ""),
            "expected_goals": model.get("expected_goals", {}),
            "consensus_eligible": bool(model.get("consensus_eligible")),
        })
    ranked.sort(key=lambda item: float_or_zero(item.get("confidence")), reverse=True)
    for index, item in enumerate(ranked[:limit], start=1):
        item["rank"] = index
    return ranked[:limit]


def consensus_score_distribution(model_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    matrices: List[np.ndarray] = []
    lambda_home: List[float] = []
    lambda_away: List[float] = []
    model_keys: List[str] = []
    model_labels: List[str] = []
    for model in model_reports:
        distribution = model.get("score_distribution") or {}
        matrix = distribution.get("score_matrix")
        if not matrix:
            continue
        array = normalize_score_grid_array(np.asarray(matrix, dtype=float) / 100.0)
        matrices.append(array)
        model_keys.append(str(model.get("model_key") or ""))
        model_labels.append(str(model.get("model_label") or model.get("model_key") or "Modelo"))
        lambdas = distribution.get("lambdas") or {}
        lambda_home.append(float_or_zero(lambdas.get("home")))
        lambda_away.append(float_or_zero(lambdas.get("away")))
    if not matrices:
        return {"available": False, "model_count": 0, "reason": "Sin matrices validas para consenso."}
    rows = min(matrix.shape[0] for matrix in matrices)
    cols = min(matrix.shape[1] for matrix in matrices)
    stacked = np.stack([matrix[:rows, :cols] for matrix in matrices], axis=0)
    consensus_grid = normalize_score_grid_array(stacked.mean(axis=0))
    return {
        "available": True,
        "model_count": len(matrices),
        "calculation_mode": "exact",
        "source": "Consenso exacto",
        "matrix_source": "consensus_exact_average",
        "model_keys": model_keys,
        "model_labels": model_labels,
        **score_distribution_payload(
            consensus_grid,
            lambda_home=float(np.mean(lambda_home)) if lambda_home else 0.0,
            lambda_away=float(np.mean(lambda_away)) if lambda_away else 0.0,
        ),
    }


def monte_carlo_consensus_from_distribution(
        distribution: Dict[str, Any],
        fixture: Dict[str, Any],
        config: Dict[str, Any],
        hardware: Dict[str, Any],
        seed: int,
) -> Dict[str, Any]:
    iterations = monte_carlo_match_iterations(config.get("iterations", DEFAULT_CONFIG["iterations"]))
    matrix = (distribution or {}).get("score_matrix")
    if not matrix:
        return {
            "available": False,
            "calculation_mode": "monte_carlo",
            "iterations": iterations,
            "seed": int(seed),
            "reason": "Sin matriz consenso exacta para simular.",
            "warnings": ["Monte Carlo SOTA no se ejecuto porque falta la matriz consenso."],
        }
    grid = normalize_score_grid_array(np.asarray(matrix, dtype=float) / 100.0)
    requested_backend = str((hardware or {}).get("monte_carlo_backend") or "numpy").strip().lower()
    warnings: List[str] = []
    try:
        count_matrix, backend = monte_carlo_count_matrix_from_grid(
            grid=grid,
            iterations=iterations,
            seed=seed,
            backend=requested_backend,
        )
    except Exception as exc:
        if requested_backend != "numpy":
            warnings.append(f"Monte Carlo CUDA fallo ({exc.__class__.__name__}); se recalculo en CPU/NumPy.")
            count_matrix, backend = monte_carlo_count_matrix_from_grid(
                grid=grid,
                iterations=iterations,
                seed=seed,
                backend="numpy",
            )
        else:
            return {
                "available": False,
                "calculation_mode": "monte_carlo",
                "iterations": iterations,
                "seed": int(seed),
                "reason": f"Monte Carlo no disponible: {exc.__class__.__name__}",
                "warnings": [f"Monte Carlo SOTA fallo: {exc}"],
            }
    payload = monte_carlo_consensus_payload_from_counts(
        count_matrix=count_matrix,
        iterations=iterations,
        source_distribution=distribution,
        fixture=fixture,
    )
    payload.update({
        "available": True,
        "calculation_mode": "monte_carlo",
        "source": "SOTA Monte Carlo sobre matriz consenso",
        "matrix_source": "monte_carlo_consensus",
        "iterations": iterations,
        "seed": int(seed),
        "backend": backend,
        "requested_backend": requested_backend,
        "requested_device": (hardware or {}).get("requested_device", "auto"),
        "actual_device": "cuda" if backend in {"cupy", "torch"} else "cpu",
        "cuda": backend in {"cupy", "torch"},
        "warnings": warnings,
    })
    return payload


def monte_carlo_count_matrix_from_grid(
        grid: np.ndarray,
        iterations: int,
        seed: int,
        backend: str = "numpy",
) -> Tuple[np.ndarray, str]:
    backend = str(backend or "numpy").strip().lower()
    if backend == "cupy":
        return monte_carlo_count_matrix_cupy(grid, iterations, seed), "cupy"
    if backend == "torch":
        return monte_carlo_count_matrix_torch(grid, iterations, seed), "torch"
    return monte_carlo_count_matrix_numpy(grid, iterations, seed), "numpy"


def monte_carlo_count_matrix_numpy(grid: np.ndarray, iterations: int, seed: int) -> np.ndarray:
    normalized = normalize_score_grid_array(grid)
    rng = np.random.default_rng(int(seed))
    sampled_home, sampled_away = sample_scores_from_grid(normalized, rng, size=int(iterations))
    cols = int(normalized.shape[1])
    indices = sampled_home.astype(int) * cols + sampled_away.astype(int)
    counts = np.bincount(indices, minlength=int(normalized.size))
    return counts.reshape(normalized.shape).astype(int)


def monte_carlo_count_matrix_cupy(grid: np.ndarray, iterations: int, seed: int) -> np.ndarray:
    import cupy as cp  # type: ignore

    normalized = normalize_score_grid_array(grid)
    flat = cp.asarray(normalized.ravel(), dtype=cp.float64)
    flat = flat / cp.maximum(cp.sum(flat), cp.float64(1e-12))
    cdf = cp.cumsum(flat)
    cdf[-1] = 1.0
    rng = cp.random.default_rng(int(seed))
    draws = rng.random(int(iterations)).astype(cp.float64)
    indices = cp.searchsorted(cdf, draws, side="right").astype(cp.int64)
    counts = cp.bincount(indices, minlength=int(flat.size))
    return cp.asnumpy(counts.reshape(normalized.shape)).astype(int)


def monte_carlo_count_matrix_torch(grid: np.ndarray, iterations: int, seed: int) -> np.ndarray:
    import torch  # type: ignore

    normalized = normalize_score_grid_array(grid)
    device = torch.device("cuda")
    flat = torch.as_tensor(normalized.ravel(), dtype=torch.float64, device=device)
    flat = flat / torch.clamp(torch.sum(flat), min=1e-12)
    cdf = torch.cumsum(flat, dim=0)
    cdf[-1] = 1.0
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    draws = torch.rand(int(iterations), generator=generator, device=device, dtype=torch.float64)
    indices = torch.searchsorted(cdf, draws, right=True).to(torch.int64)
    counts = torch.bincount(indices, minlength=int(flat.numel()))
    return counts.reshape(normalized.shape).detach().cpu().numpy().astype(int)


def monte_carlo_consensus_payload_from_counts(
        count_matrix: np.ndarray,
        iterations: int,
        source_distribution: Dict[str, Any],
        fixture: Dict[str, Any],
) -> Dict[str, Any]:
    counts = np.asarray(count_matrix, dtype=float)
    total = max(float(counts.sum()), 1.0)
    simulated_grid = normalize_score_grid_array(counts / total)
    rows, cols = simulated_grid.shape
    home_axis = np.arange(rows, dtype=float)
    away_axis = np.arange(cols, dtype=float)
    home_goals, away_goals = np.meshgrid(home_axis, away_axis, indexing="ij")
    simulated_home = float((counts * home_goals).sum() / total)
    simulated_away = float((counts * away_goals).sum() / total)
    payload = score_distribution_payload(
        simulated_grid,
        lambda_home=simulated_home,
        lambda_away=simulated_away,
    )
    payload["top_scores"] = monte_carlo_top_scores_from_matrix(counts, int(iterations))
    probabilities = payload.get("probabilities", {})
    outcome = outcome_decision(probabilities)
    totals = total_decisions(probabilities)
    over_under = payload.get("over_under", {})
    total_payload: Dict[str, Dict[str, Any]] = {}
    for line in REPORT_TOTAL_GOAL_LINES:
        line_key = f"{line:.1f}"
        item = over_under.get(line_key, {})
        pick = totals.get(line_key, "")
        total_payload[line_key] = {
            "pick": pick,
            "label": "Over" if pick == "over" else "Under" if pick == "under" else "",
            "over": float_or_zero(item.get("over")),
            "under": float_or_zero(item.get("under")),
        }
    return {
        **payload,
        "model_count": int((source_distribution or {}).get("model_count") or 0),
        "source_lambdas": (source_distribution or {}).get("lambdas", {}),
        "simulated_goals": {
            "home": round(simulated_home, 3),
            "away": round(simulated_away, 3),
        },
        "probabilities": probabilities,
        "outcome": outcome,
        "outcome_label": outcome_label(outcome),
        "outcome_team": outcome_team(outcome, fixture or {}),
        "outcome_probability": float_or_zero(probabilities.get(outcome)),
        "totals": total_payload,
    }


def monte_carlo_top_scores_from_matrix(count_matrix: np.ndarray, iterations: int, limit: int = 5) -> List[Dict[str, Any]]:
    counts = np.asarray(count_matrix, dtype=int)
    flat = counts.ravel()
    if flat.size == 0:
        return []
    ranked_indices = np.argsort(flat)[::-1]
    cols = counts.shape[1]
    output = []
    for index in ranked_indices[:limit]:
        count = int(flat[int(index)])
        if count <= 0:
            continue
        home_goals = int(index // cols)
        away_goals = int(index % cols)
        output.append({
            "score": f"{home_goals}-{away_goals}",
            "home_goals": home_goals,
            "away_goals": away_goals,
            "probability": round(float(count) * 100.0 / max(int(iterations), 1), 2),
            "count": count,
        })
    return output


def model_statistics_payload(model_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    probabilities = {key: [] for key in ("home", "draw", "away")}
    lambdas = {key: [] for key in ("home", "away", "total")}
    total_stats: Dict[str, Dict[str, Any]] = {}
    top_score_counts: Counter[str] = Counter()
    signature_counts: Counter[str] = Counter()
    for model in model_reports:
        probs = model.get("probabilities") or {}
        expected = model.get("expected_goals") or {}
        for key in probabilities:
            probabilities[key].append(float_or_zero(probs.get(key)))
        home_lambda = float_or_zero(expected.get("home"))
        away_lambda = float_or_zero(expected.get("away"))
        lambdas["home"].append(home_lambda)
        lambdas["away"].append(away_lambda)
        lambdas["total"].append(home_lambda + away_lambda)
        top_score_counts.update([str(model.get("top_score") or "")])
        signature_counts.update([str(model.get("signature") or "")])
    for line in REPORT_TOTAL_GOAL_LINES:
        line_key = f"{line:.1f}"
        suffix = total_line_suffix(line)
        over_values = [float_or_zero((model.get("probabilities") or {}).get(f"over{suffix}")) for model in model_reports]
        under_values = [float_or_zero((model.get("probabilities") or {}).get(f"under{suffix}")) for model in model_reports]
        picks = [str((model.get("totals") or {}).get(line_key) or "") for model in model_reports]
        pick_counts = Counter(picks)
        pick, count = most_common_item(pick_counts)
        total_stats[line_key] = {
            "over": numeric_summary(over_values),
            "under": numeric_summary(under_values),
            "pick": pick,
            "label": "Over" if pick == "over" else "Under" if pick == "under" else "",
            "count": count,
            "share": round(count / len(model_reports), 3) if model_reports else 0.0,
            "pick_counts": dict(pick_counts),
        }
    outcome_summaries = {key: numeric_summary(values) for key, values in probabilities.items()}
    return {
        "model_count": len(model_reports),
        "outcomes": outcome_summaries,
        "probability_ranges": {
            key: {
                "min": summary.get("min", 0.0),
                "max": summary.get("max", 0.0),
                "spread": summary.get("spread", 0.0),
            }
            for key, summary in outcome_summaries.items()
        },
        "lambdas": {key: numeric_summary(values) for key, values in lambdas.items()},
        "totals": total_stats,
        "top_scores": counter_rank_payload(top_score_counts, len(model_reports)),
        "signatures": counter_rank_payload(signature_counts, len(model_reports)),
    }


def numeric_summary(values: Iterable[Any]) -> Dict[str, Any]:
    numbers = [float_or_zero(value) for value in values if value is not None]
    if not numbers:
        return {"avg": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "spread": 0.0}
    array = np.asarray(numbers, dtype=float)
    return {
        "avg": round(float(array.mean()), 2),
        "std": round(float(array.std()), 2),
        "min": round(float(array.min()), 2),
        "max": round(float(array.max()), 2),
        "spread": round(float(array.max() - array.min()), 2),
    }


def counter_rank_payload(counter: Counter[str], total: int, limit: int = 5) -> List[Dict[str, Any]]:
    return [
        {
            "value": key,
            "count": int(count),
            "share": round(int(count) / total, 3) if total else 0.0,
        }
        for key, count in counter.most_common(limit)
        if key
    ]


def round_half_up_int(value: Any) -> int:
    return int(math.floor(float_or_zero(value) + 0.5))


def consensus_signature(outcome: str, totals: Dict[str, str]) -> str:
    parts = [str(outcome or "")]
    for line in REPORT_TOTAL_GOAL_LINES:
        parts.append(str(totals.get(f"{line:.1f}", "")))
    return "|".join(parts)


def fixture_consensus(model_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = [
        report for report in model_reports
        if report.get("consensus_eligible") and report.get("signature")
    ]
    valid_total = len(eligible)
    signature_counts = Counter(str(report.get("signature", "")) for report in eligible)
    outcome_counts = Counter(str((report.get("decision") or {}).get("outcome", "")) for report in eligible)
    total_counts: Dict[str, Dict[str, int]] = {}
    for line in REPORT_TOTAL_GOAL_LINES:
        line_key = f"{line:.1f}"
        total_counts[line_key] = dict(Counter(str((report.get("totals") or {}).get(line_key, "")) for report in eligible))
    leader_signature, leader_signature_count = most_common_item(signature_counts)
    leader_outcome, leader_outcome_count = most_common_item(outcome_counts)
    signature_share = leader_signature_count / valid_total if valid_total else 0.0
    outcome_share = leader_outcome_count / valid_total if valid_total else 0.0
    if valid_total and signature_share >= 1.0:
        strength = "Muy fuerte"
    elif valid_total and signature_share >= 0.70:
        strength = "Fuerte"
    elif valid_total and outcome_share >= 0.60:
        strength = "Media"
    else:
        strength = "Baja"
    leader_totals = {}
    for line in REPORT_TOTAL_GOAL_LINES:
        line_key = f"{line:.1f}"
        pick, count = most_common_item(Counter(total_counts.get(line_key, {})))
        leader_totals[line_key] = {
            "pick": pick,
            "label": "Over" if pick == "over" else "Under" if pick == "under" else "",
            "count": count,
            "share": round(count / valid_total, 3) if valid_total else 0.0,
        }
    return {
        "strength": strength,
        "eligible_models": valid_total,
        "excluded_models": max(len(model_reports) - valid_total, 0),
        "signature": leader_signature,
        "signature_count": leader_signature_count,
        "signature_share": round(signature_share, 3),
        "outcome": leader_outcome,
        "outcome_label": outcome_label(leader_outcome),
        "outcome_count": leader_outcome_count,
        "outcome_share": round(outcome_share, 3),
        "agreement": {
            "market": "1X2",
            "pick": leader_outcome,
            "pick_label": outcome_label(leader_outcome),
            "count": leader_outcome_count,
            "total": valid_total,
            "share": round(outcome_share, 3),
        },
        "outcome_counts": dict(outcome_counts),
        "signature_counts": dict(signature_counts),
        "totals": leader_totals,
        "total_counts": total_counts,
    }


def most_common_item(counter: Counter) -> Tuple[str, int]:
    if not counter:
        return "", 0
    key, count = counter.most_common(1)[0]
    return str(key), int(count)


def outcome_decision(probabilities: Dict[str, Any]) -> str:
    candidates = {
        "home": float_or_zero(probabilities.get("home", 0.0)),
        "draw": float_or_zero(probabilities.get("draw", 0.0)),
        "away": float_or_zero(probabilities.get("away", 0.0)),
    }
    return max(candidates, key=candidates.get)


def total_decisions(probabilities: Dict[str, Any]) -> Dict[str, str]:
    decisions = {}
    for line in REPORT_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        over = float_or_zero(probabilities.get(f"over{suffix}", 0.0))
        under = float_or_zero(probabilities.get(f"under{suffix}", 0.0))
        decisions[f"{line:.1f}"] = "over" if over >= under else "under"
    return decisions


def outcome_label(outcome: Any) -> str:
    return {"home": "1", "draw": "X", "away": "2"}.get(str(outcome or ""), "")


def outcome_team(outcome: Any, fixture: pd.Series | Dict[str, Any]) -> str:
    if str(outcome) == "draw":
        return "Empate"
    if str(outcome) == "home":
        return str(fixture.get("Equipo 1", fixture.get("home", "Local")))
    if str(outcome) == "away":
        return str(fixture.get("Equipo 2", fixture.get("away", "Visitante")))
    return ""


def probability_percent(value: Any, already_percent: bool = False) -> float:
    number = float_or_zero(value)
    if not already_percent:
        number *= 100.0
    return round(float(np.clip(number, 0.0, 100.0)), 2)


def parse_score_part(score: Any, index: int) -> int:
    parts = re.findall(r"\d+", str(score or ""))
    if index < len(parts):
        return int(parts[index])
    return 0


def float_or_zero(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def fixture_report_warnings(report: Dict[str, Any]) -> List[str]:
    warning_labels: Dict[str, List[str]] = {}
    for model in report.get("models", []):
        for warning in model.get("warnings", []):
            warning_text = str(warning or "").strip()
            if not warning_text:
                continue
            label = model.get("model_label") or model.get("model_key") or "Modelo"
            warning_labels.setdefault(warning_text, []).append(str(label))
    warnings: List[str] = []
    for warning, labels in warning_labels.items():
        unique_labels = unique_strings(labels)
        if len(unique_labels) > 1:
            warnings.append(f"{warning} ({len(unique_labels)} modelos)")
        elif unique_labels:
            warnings.append(f"{unique_labels[0]}: {warning}")
    return warnings


def upcoming_report_table_rows(fixture_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for report in fixture_reports:
        fixture = report.get("fixture", {})
        consensus = report.get("consensus", {})
        for model in report.get("models", []):
            probs = model.get("probabilities", {})
            expected = model.get("expected_goals", {})
            rows.append({
                "No.": fixture.get("id", ""),
                "Fecha": fixture.get("date", ""),
                "Grupo": fixture.get("group", ""),
                "Partido": fixture.get("label", ""),
                "Consenso": consensus.get("outcome_label", ""),
                "Fuerza": consensus.get("strength", ""),
                "Modelo": model.get("model_label", ""),
                "Disponible": "Si" if model.get("available") else "No",
                "Cuenta consenso": "Si" if model.get("consensus_eligible") else "No",
                "Pick": (model.get("decision") or {}).get("label", ""),
                "Top score": model.get("top_score", ""),
                "Lambda Local": expected.get("home", ""),
                "Lambda Visita": expected.get("away", ""),
                "1 %": probs.get("home", ""),
                "X %": probs.get("draw", ""),
                "2 %": probs.get("away", ""),
                "O0.5": probs.get("over05", ""),
                "U0.5": probs.get("under05", ""),
                "O1.5": probs.get("over15", ""),
                "U1.5": probs.get("under15", ""),
                "O2.5": probs.get("over25", ""),
                "U2.5": probs.get("under25", ""),
                "O3.5": probs.get("over35", ""),
                "U3.5": probs.get("under35", ""),
                "Warnings": " | ".join(model.get("warnings", [])),
            })
    return rows


def persist_upcoming_report(report: Dict[str, Any]) -> Dict[str, Any]:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    safe_report = jsonable(report)
    created_at = str(safe_report.get("created_at") or datetime.now(timezone.utc).isoformat())
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha256(json.dumps(safe_report, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:10]
    report_id = f"report_{timestamp}_{digest}"
    report_path = REPORTS_ROOT / f"{report_id}.json"
    output = {
        "report_id": report_id,
        "report_path": str(report_path),
        "created_at": created_at,
        **safe_report,
    }
    output["downloads"] = report_download_links(report_id, output)
    ensure_report_download_files(output)
    report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REPORTS_ROOT / "latest.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def report_download_links(report_id: str, report: Dict[str, Any]) -> Dict[str, str]:
    base = f"/api/mundial/reports/{report_id}/download"
    has_backtest = report_has_backtest(report)
    return {
        "predictions_html": f"{base}?kind=predictions&format=html",
        "predictions_csv": f"{base}?kind=predictions&format=csv",
        "backtest_html": f"{base}?kind=backtest&format=html" if has_backtest else "",
        "backtest_csv": f"{base}?kind=backtest&format=csv" if has_backtest else "",
    }


def resolve_report_download(report_id: Any, kind: Any, format_value: Any) -> Dict[str, Any]:
    requested_kind = validate_report_download_kind(kind)
    requested_format = validate_report_download_format(format_value)
    report = load_persisted_report(report_id)
    resolved_report_id = validate_report_id(report.get("report_id"))
    if requested_kind == "backtest" and not report_has_backtest(report):
        raise ValueError("Descarga de backtesting no disponible para este reporte.")
    ensure_report_download_files(report)
    path = report_download_file_path(resolved_report_id, requested_kind, requested_format)
    if not path.exists() or not path.is_file():
        raise ValueError("Archivo de reporte no disponible.")
    return {
        "path": path,
        "filename": path.name,
        "media_type": "text/html; charset=utf-8" if requested_format == "html" else "text/csv; charset=utf-8",
    }


def ensure_report_download_files(report: Dict[str, Any]) -> Dict[str, str]:
    report_id = validate_report_id(report.get("report_id"))
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, str] = {}
    for kind in ("predictions", "backtest"):
        if kind == "backtest" and not report_has_backtest(report):
            continue
        for format_value in ("html", "csv"):
            path = report_download_file_path(report_id, kind, format_value)
            if format_value == "html":
                content = report_download_html(report, kind)
            else:
                content = report_download_csv(report, kind)
            path.write_text(content, encoding="utf-8")
            outputs[f"{kind}_{format_value}"] = str(path)
    return outputs


def load_persisted_report(report_id: Any) -> Dict[str, Any]:
    path = report_json_path(report_id)
    if not path.exists() or not path.is_file():
        raise ValueError("Reporte no encontrado.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Reporte invalido.")
    resolved_report_id = validate_report_id(payload.get("report_id"))
    payload["downloads"] = report_download_links(resolved_report_id, payload)
    return payload


def report_json_path(report_id: Any) -> Path:
    value = str(report_id or "").strip()
    if value in {"latest", "latest.json"}:
        return REPORTS_ROOT / "latest.json"
    safe_id = validate_report_id(value)
    return REPORTS_ROOT / f"{safe_id}.json"


def validate_report_id(report_id: Any) -> str:
    value = str(report_id or "").strip()
    if not re.fullmatch(r"report_\d{8}_\d{6}_[0-9a-f]{10}", value):
        raise ValueError("report_id invalido.")
    return value


def validate_report_download_kind(kind: Any) -> str:
    value = str(kind or "").strip().lower()
    if value not in REPORT_DOWNLOAD_KINDS:
        raise ValueError("kind invalido; usa predictions o backtest.")
    return value


def validate_report_download_format(format_value: Any) -> str:
    value = str(format_value or "").strip().lower()
    if value not in REPORT_DOWNLOAD_FORMATS:
        raise ValueError("format invalido; usa html o csv.")
    return value


def report_download_file_path(report_id: str, kind: str, format_value: str) -> Path:
    safe_report_id = validate_report_id(report_id)
    safe_kind = validate_report_download_kind(kind)
    safe_format = validate_report_download_format(format_value)
    return REPORTS_ROOT / f"{safe_report_id}_{safe_kind}.{safe_format}"


def report_has_backtest(report: Dict[str, Any]) -> bool:
    if report.get("model_backtests"):
        return True
    summary = ((report.get("backtest") or {}).get("summary") or {})
    return bool(summary.get("available") or summary.get("evaluated_matches"))


def report_download_csv(report: Dict[str, Any], kind: str) -> str:
    if kind == "backtest":
        return backtest_csv_text(report)
    table = report.get("table") or {}
    return table_csv_text(table)


def report_download_html(report: Dict[str, Any], kind: str) -> str:
    summary = report.get("summary") or {}
    title = "Predicciones Mundial 2026" if kind == "predictions" else "Backtesting Mundial 2026"
    subtitle = "Generado " + str(report.get("created_at") or "")
    body = predictions_report_html_body(report) if kind == "predictions" else backtest_report_html_body(report)
    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape_report_html(title)}</title>
  <style>{standalone_report_css()}</style>
</head>
<body>
  <main>
    <header class="report-title">
      <div>
        <p>{escape_report_html(summary.get("pipeline_label") or summary.get("pipeline_mode") or "Reporte")}</p>
        <h1>{escape_report_html(title)}</h1>
        <small>{escape_report_html(subtitle)} · {escape_report_html(report.get("report_id") or "")}</small>
      </div>
      <strong>Mundial 2026</strong>
    </header>
    {body}
  </main>
</body>
</html>
"""


def predictions_report_html_body(report: Dict[str, Any]) -> str:
    summary = report.get("summary") or {}
    fixtures = report.get("fixture_reports") or []
    cards = "\n".join(prediction_fixture_report_card_html(item) for item in fixtures)
    table = table_html_fragment(report.get("table") or {})
    return f"""
    <section class="summary-grid">
      {summary_card_html("Partidos", f"{summary.get('returned', 0)}/{summary.get('requested', 0)}")}
      {summary_card_html("Grupo", summary.get("group", ""))}
      {summary_card_html("Modelos", len(summary.get("score_models") or []))}
      {summary_card_html("Poisson ultimos", summary.get("poisson_recent_matches", ""))}
    </section>
    <section class="fixture-grid">{cards or '<p>Sin predicciones disponibles.</p>'}</section>
    <section class="table-section">
      <h2>Tabla de predicciones</h2>
      {table}
    </section>
"""


def prediction_fixture_report_card_html(report: Dict[str, Any]) -> str:
    if report.get("probabilities") and report.get("decision"):
        return xg_lightgbm_prediction_fixture_report_card_html(report)
    fixture = report.get("fixture") or {}
    primary = report.get("primary_model") or {}
    if not primary.get("available"):
        monte_carlo = report.get("monte_carlo_consensus") or {}
        distribution = monte_carlo if monte_carlo.get("available") else report.get("consensus_score_distribution") or {}
        consensus = report.get("consensus") or {}
        if distribution or consensus:
            primary = {
                "available": True,
                "model_label": "Consenso",
                "decision": {
                    "label": consensus.get("outcome_label", ""),
                    "team": outcome_team(consensus.get("outcome"), fixture),
                },
                "probabilities": distribution.get("probabilities") or {},
                "top_scores": distribution.get("top_scores") or [],
            }
    decision = primary.get("decision") or {}
    probabilities = primary.get("probabilities") or {}
    top_scores = primary.get("top_scores") or []
    return f"""
      <article class="fixture-card">
        <header><span>{escape_report_html(fixture.get("date", ""))}</span><strong>{escape_report_html(fixture.get("group", ""))}</strong></header>
        <h2>{escape_report_html(fixture.get("label", ""))}</h2>
        <div class="pick"><span>{escape_report_html(primary.get("model_label", ""))}</span><strong>{escape_report_html(decision.get("label", ""))} · {escape_report_html(decision.get("team", ""))}</strong></div>
        {outcome_bars_html(probabilities)}
        {total_25_html(probabilities)}
        <div class="top-scores">{''.join(f'<span>{escape_report_html(score.get("score", ""))} <b>{escape_report_html(format_metric(score.get("probability", "")))}%</b></span>' for score in top_scores[:3])}</div>
        {recent_matches_report_html(report.get("recent_matches_15") or {}, fixture)}
      </article>
"""


def xg_lightgbm_prediction_fixture_report_card_html(report: Dict[str, Any]) -> str:
    fixture = report.get("fixture") or {}
    decision = report.get("decision") or {}
    probabilities = report.get("probabilities") or {}
    expected = report.get("expected_goals") or {}
    model_probs = report.get("model_probs") or {}
    warnings = report.get("warnings") or []
    warning_html = "".join(f"<p>{escape_report_html(item)}</p>" for item in warnings)
    return f"""
      <article class="fixture-card">
        <header><span>{escape_report_html(fixture.get("date", ""))}</span><strong>{escape_report_html(fixture.get("group", ""))}</strong></header>
        <h2>{escape_report_html(fixture.get("label", ""))}</h2>
        <div class="pick"><span>{escape_report_html(model_probs.get("model_name") or "xG-LightGBM")}</span><strong>{escape_report_html(decision.get("label", ""))} · {escape_report_html(decision.get("team", ""))}</strong></div>
        {outcome_bars_html(probabilities)}
        {total_25_html(probabilities)}
        <div class="top-scores">
          <span>Top {escape_report_html(report.get("modal_score", ""))}</span>
          <span>xG local <b>{escape_report_html(format_metric(expected.get("home", "")))}</b></span>
          <span>xG visita <b>{escape_report_html(format_metric(expected.get("away", "")))}</b></span>
        </div>
        {f'<div class="warnings">{warning_html}</div>' if warning_html else ''}
      </article>
"""


def recent_matches_report_html(recent: Dict[str, Any], fixture: Dict[str, Any]) -> str:
    home_rows = recent.get("home") or []
    away_rows = recent.get("away") or []
    if not home_rows and not away_rows:
        return ""
    limit = recent.get("limit", 15)
    home_team = recent.get("home_team") or fixture.get("home") or ""
    away_team = recent.get("away_team") or fixture.get("away") or ""
    warnings = recent.get("warnings") or []
    source_note = recent_matches_source_note(recent)
    warning_html = "".join(f"<p>{escape_report_html(item)}</p>" for item in warnings)
    return f"""
        <details class="recent15-report">
          <summary>Ultimos {escape_report_html(limit)} partidos por equipo</summary>
          {f'<p>{escape_report_html(source_note)}</p>' if source_note else ''}
          {warning_html}
          <div class="recent15-columns">
            {recent_matches_report_panel(home_rows, home_team)}
            {recent_matches_report_panel(away_rows, away_team)}
          </div>
        </details>
"""


def recent_matches_source_note(recent: Dict[str, Any]) -> str:
    source = str(recent.get("source") or "").strip()
    max_scored = str(recent.get("max_scored_date") or "").strip()
    if source and max_scored:
        return f"Fuente {source}; ultimo marcador disponible {max_scored}."
    if source:
        return f"Fuente {source}."
    return ""


def recent_matches_report_panel(rows: List[Dict[str, Any]], team: str) -> str:
    if not rows:
        return f'<div class="recent15-report-team"><h3>{escape_report_html(team)}</h3><p>Sin partidos recientes.</p></div>'
    stats = recent_matches_summary(rows)
    body = "".join(recent_match_report_card(row) for row in rows[:15])
    return f"""
          <div class="recent15-report-team">
            <header>
              <div><h3>{escape_report_html(team)}</h3><small>{escape_report_html(stats["latest"])} ultimo partido</small></div>
              <strong>{escape_report_html(stats["official"])}/{escape_report_html(stats["total"])} oficiales</strong>
            </header>
            <div class="recent15-report-summary">
              <span><b>{escape_report_html(stats["record"])}</b><small>G-E-P</small></span>
              <span><b>{escape_report_html(stats["avg_weight"])}</b><small>Peso medio</small></span>
            </div>
            <div class="recent15-report-list">{body}</div>
          </div>
"""


def recent_matches_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    wins = sum(1 for row in rows if str(row.get("result") or "").upper() in {"G", "W"})
    draws = sum(1 for row in rows if str(row.get("result") or "").upper() in {"E", "D"})
    losses = sum(1 for row in rows if str(row.get("result") or "").upper() in {"P", "L"})
    official = sum(1 for row in rows if str(row.get("match_type") or "").lower() == "official")
    weights = [float_or_zero(row.get("weight")) for row in rows if float_or_zero(row.get("weight")) > 0.0]
    avg_weight = format_metric(float(np.mean(weights))) if weights else "-"
    return {
        "total": total,
        "official": official,
        "record": f"{wins}-{draws}-{losses}",
        "latest": str((rows[0] if rows else {}).get("date") or "-"),
        "avg_weight": avg_weight,
    }


def recent_match_report_card(row: Dict[str, Any]) -> str:
    match_type = str(row.get("match_type") or "")
    type_label = "Oficial" if match_type.lower() == "official" else "Amistoso" if match_type.lower() == "friendly" else match_type
    type_class = "official" if match_type.lower() == "official" else "friendly" if match_type.lower() == "friendly" else "neutral"
    tournament = row.get("tournament") or row.get("match_type") or ""
    weight = row.get("weight", "")
    importance = row.get("importance_label") or ""
    return (
        f'<article class="recent15-report-match {escape_report_html(type_class)}">'
        '<div class="recent15-report-main">'
        f'<span>{escape_report_html(row.get("date", ""))}</span>'
        f'<strong>vs {escape_report_html(row.get("opponent", ""))}</strong>'
        f'<small>Torneo: {escape_report_html(tournament)}</small>'
        '</div>'
        '<div class="recent15-report-score">'
        f'<b>{escape_report_html(row.get("score", ""))}</b>'
        f'<span>{escape_report_html(row.get("result", ""))}</span>'
        '</div>'
        '<div class="recent15-report-tags">'
        f'<span>{escape_report_html(type_label)}</span>'
        f'<span>{escape_report_html(row.get("venue", ""))}</span>'
        f'<span>Peso {escape_report_html(format_metric(weight))}</span>'
        f'{f"<span>{escape_report_html(importance)}</span>" if importance else ""}'
        '</div>'
        '</article>'
    )


def backtest_report_html_body(report: Dict[str, Any]) -> str:
    summary = ((report.get("backtest") or {}).get("summary") or (report.get("summary") or {}).get("backtest") or {})
    models = report.get("model_backtests") or []
    model_cards = "\n".join(backtest_model_report_card_html(item) for item in models)
    backtest_range = summary.get("backtest_range") or {}
    first_match = backtest_range.get("first_match") or {}
    last_match = backtest_range.get("last_match") or {}
    return f"""
    <section class="summary-grid">
      {summary_card_html("Evaluados", summary.get("evaluated_matches", 0))}
      {summary_card_html("Primer partido", f"{first_match.get('home', '')} vs {first_match.get('away', '')}")}
      {summary_card_html("Ultimo partido", f"{last_match.get('home', '')} vs {last_match.get('away', '')}")}
      {summary_card_html("Generado", summary.get("generated_at", backtest_range.get("generated_at", "")))}
      {summary_card_html("Fuente", summary.get("source", ""))}
      {summary_card_html("Historicos base", summary.get("train_matches", 0))}
    </section>
    <section class="fixture-grid">{model_cards or '<p>Sin backtesting disponible.</p>'}</section>
    <section class="table-section">
      <h2>Detalle backtest</h2>
      {table_html_fragment(backtest_metrics_table(report))}
    </section>
"""


def backtest_model_report_card_html(model: Dict[str, Any]) -> str:
    vs = model.get("vs_poisson") or {}
    return f"""
      <article class="fixture-card">
        <header><span>#{escape_report_html(model.get("rank", ""))}</span><strong>{escape_report_html(model.get("model_label", ""))}</strong></header>
        <div class="metric-bars">
          {metric_bar_html("Score resultados", format_metric(model.get("score_resultados", "")), float_or_zero(model.get("score_resultados")))}
          {metric_bar_html("Log-loss", format_metric(model.get("log_loss", "")), inverse_metric_percent(model.get("log_loss"), 1.6))}
          {metric_bar_html("RPS", format_metric(model.get("rps", "")), inverse_metric_percent(model.get("rps"), 0.75))}
          {metric_bar_html("ECE", format_metric(model.get("expected_calibration_error", "")), inverse_metric_percent(model.get("expected_calibration_error"), 0.35))}
          {metric_bar_html("Pick %", f"{format_metric(float_or_zero(model.get('pick_accuracy')) * 100)}%", float_or_zero(model.get("pick_accuracy")) * 100)}
          {metric_bar_html("Marcador #1", f"{format_metric(float_or_zero(model.get('score_accuracy')) * 100)}%", float_or_zero(model.get("score_accuracy")) * 100)}
          {metric_bar_html("Top-3 marcador", f"{format_metric(float_or_zero(model.get('top3_score_accuracy')) * 100)}%", float_or_zero(model.get("top3_score_accuracy")) * 100)}
          {metric_bar_html("Brier", format_metric(model.get("brier", "")), inverse_metric_percent(model.get("brier"), 0.75))}
          {metric_bar_html("U/O 2.5 LL", format_metric(model.get("ou25_log_loss", "")), inverse_metric_percent(model.get("ou25_log_loss"), 1.2))}
          {metric_bar_html("Vs Poisson", f"{escape_report_html(vs.get('metric_wins', 0))}/{escape_report_html(vs.get('metric_total', 7))}", float_or_zero(vs.get("metric_wins")) * (100.0 / max(float_or_zero(vs.get("metric_total")), 1.0)))}
        </div>
        <p>{escape_report_html(vs.get("summary", ""))}</p>
      </article>
"""


def backtest_csv_text(report: Dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    metrics = backtest_metrics_table(report)
    writer.writerow(["Resumen modelos"])
    writer.writerow(metrics.get("columns", []))
    for row in metrics.get("rows", []):
        writer.writerow([row.get(column, "") for column in metrics.get("columns", [])])
    writer.writerow([])
    writer.writerow(["Detalle partidos"])
    match_columns = [
        "Rank", "Modelo", "Fecha", "Partido", "Resultado", "Pick", "Pick real", "Acierto",
        "Marcador #1", "Marcador #1 %", "Esperado", "Top-3 marcador", "Top-5 marcador", "RPS",
        "Confianza", "Prob. real", "Prob. marcador",
    ]
    writer.writerow(match_columns)
    for model in report.get("model_backtests") or []:
        for row in model.get("matches") or []:
            writer.writerow([
                model.get("rank", ""),
                model.get("model_label", ""),
                row.get("date", ""),
                row.get("match", ""),
                row.get("actual_score", ""),
                row.get("pick", ""),
                row.get("actual_pick", ""),
                "Si" if row.get("pick_hit") else "No",
                row.get("most_probable_score", row.get("modal_score", "")),
                row.get("most_probable_score_probability", row.get("score_probability", "")),
                row.get("expected_score", ""),
                "Si" if row.get("top3_score_hit") else "No",
                "Si" if row.get("top5_score_hit") else "No",
                row.get("rps", ""),
                row.get("confidence", ""),
                row.get("actual_probability", ""),
                row.get("score_probability", ""),
            ])
    return output.getvalue()


def backtest_metrics_table(report: Dict[str, Any]) -> Dict[str, Any]:
    columns = [
        "Rank", "Modelo", "Disponible", "Evaluados", "Score resultados", "Log-loss", "RPS", "ECE", "MCE", "Brier",
        "Pick %", "Marcador #1 %", "Top-3 marcador %", "Top-5 marcador %", "U/O 2.5 LL", "U/O 2.5 Brier",
        "U/O %", "Score-log", "Goles MAE", "Total goles MAE", "Margen MAE", "Macro F1",
        "Balanced acc", "Vs Poisson", "Warnings",
    ]
    rows = []
    for item in report.get("model_backtests") or []:
        vs = item.get("vs_poisson") or {}
        rows.append({
            "Rank": item.get("rank", ""),
            "Modelo": item.get("model_label", item.get("model_key", "")),
            "Disponible": "Si" if item.get("available") else "No",
            "Evaluados": item.get("evaluated_matches", ""),
            "Score resultados": item.get("score_resultados", ""),
            "Log-loss": item.get("log_loss", ""),
            "RPS": item.get("rps", ""),
            "ECE": item.get("expected_calibration_error", ""),
            "MCE": item.get("max_calibration_error", ""),
            "Brier": item.get("brier", ""),
            "Pick %": round(float_or_zero(item.get("pick_accuracy")) * 100.0, 3),
            "Marcador #1 %": round(float_or_zero(item.get("score_accuracy")) * 100.0, 3),
            "Top-3 marcador %": round(float_or_zero(item.get("top3_score_accuracy")) * 100.0, 3),
            "Top-5 marcador %": round(float_or_zero(item.get("top5_score_accuracy")) * 100.0, 3),
            "U/O 2.5 LL": item.get("ou25_log_loss", ""),
            "U/O 2.5 Brier": item.get("ou25_brier", ""),
            "U/O %": round(float_or_zero(item.get("over_under_accuracy")) * 100.0, 3),
            "Score-log": item.get("score_log_loss", ""),
            "Goles MAE": f"{format_metric(item.get('home_goals_mae', ''))}/{format_metric(item.get('away_goals_mae', ''))}",
            "Total goles MAE": item.get("total_goals_mae", ""),
            "Margen MAE": item.get("margin_mae", ""),
            "Macro F1": item.get("macro_f1", ""),
            "Balanced acc": item.get("balanced_accuracy", ""),
            "Vs Poisson": vs.get("summary", ""),
            "Warnings": " | ".join(item.get("warnings", [])),
        })
    return {"columns": columns, "rows": rows, "total": len(rows)}


def table_csv_text(table: Dict[str, Any]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    columns = list(table.get("columns") or [])
    writer.writerow(columns)
    for row in table.get("rows") or []:
        writer.writerow([row.get(column, "") for column in columns])
    return output.getvalue()


def table_html_fragment(table: Dict[str, Any]) -> str:
    columns = list(table.get("columns") or [])
    if not columns:
        return "<p>Sin filas.</p>"
    head = "".join(f"<th>{escape_report_html(column)}</th>" for column in columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{escape_report_html(row.get(column, ''))}</td>" for column in columns) + "</tr>"
        for row in table.get("rows") or []
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def summary_card_html(label: Any, value: Any) -> str:
    return f'<article><span>{escape_report_html(label)}</span><strong>{escape_report_html(value)}</strong></article>'


def outcome_bars_html(probabilities: Dict[str, Any]) -> str:
    items = (("1", probabilities.get("home", 0)), ("X", probabilities.get("draw", 0)), ("2", probabilities.get("away", 0)))
    return '<div class="metric-bars">' + "".join(metric_bar_html(label, f"{format_metric(value)}%", float_or_zero(value)) for label, value in items) + "</div>"


def total_25_html(probabilities: Dict[str, Any]) -> str:
    over = float_or_zero(probabilities.get("over25"))
    under = float_or_zero(probabilities.get("under25"))
    pick = "Over" if over >= under else "Under"
    value = max(over, under)
    return f'<div class="pick secondary"><span>U/O 2.5</span><strong>{escape_report_html(pick)} · {escape_report_html(format_metric(value))}%</strong></div>'


def metric_bar_html(label: Any, value: Any, percent: Any) -> str:
    return (
        f'<div class="metric-bar"><span>{escape_report_html(label)}</span>'
        f'<i><b style="width:{escape_report_html(clamp_report_percent(percent))}%"></b></i>'
        f'<strong>{escape_report_html(value)}</strong></div>'
    )


def inverse_metric_percent(value: Any, ceiling: float) -> float:
    number = float_or_zero(value)
    if number <= 0.0:
        return 0.0
    return float(np.clip((1.0 - min(number / max(float(ceiling), 1e-9), 1.0)) * 100.0, 0.0, 100.0))


def clamp_report_percent(value: Any) -> str:
    return format_metric(float(np.clip(float_or_zero(value), 0.0, 100.0)))


def escape_report_html(value: Any) -> str:
    return html_lib.escape(str(value if value is not None else ""), quote=True)


def standalone_report_css() -> str:
    return """
body{margin:0;background:#f5f7f8;color:#16202a;font-family:Inter,Arial,sans-serif;font-size:14px}
main{width:min(1120px,100%);margin:0 auto;padding:24px}
.report-title{display:flex;justify-content:space-between;gap:16px;align-items:end;margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid #d9e2e8}
.report-title p,.report-title small,article span{color:#65717d}.report-title h1{margin:3px 0;font-size:30px;line-height:1.15}
.report-title strong{color:#0f7a5f}.summary-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin-bottom:16px}
.summary-grid article,.fixture-card{border:1px solid #d9e2e8;border-radius:8px;background:#fff}
.summary-grid article{padding:12px}.summary-grid strong{display:block;margin-top:4px;color:#0f7a5f;font-size:18px}
.fixture-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.fixture-card{display:grid;gap:10px;padding:14px;break-inside:avoid}
.fixture-card header{display:flex;justify-content:space-between;gap:10px;color:#65717d;font-size:12px}.fixture-card h2{margin:0;font-size:18px}
.pick{display:grid;gap:3px;padding:10px;border:1px solid #c8e5dc;border-radius:8px;background:#eaf6f2}.pick strong{color:#0f7a5f;font-size:20px}
.pick.secondary{background:#f8fafb;border-color:#d9e2e8}.top-scores{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}
.top-scores span{padding:8px;border-radius:8px;background:#f8fafb;text-align:center}.top-scores b{color:#16202a}
.recent15-report{border:1px solid #d9e2e8;border-radius:8px;background:#f8fafb;padding:8px}.recent15-report summary{cursor:pointer;font-weight:700}
.recent15-columns{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin-top:8px}.recent15-report-team{display:grid;gap:8px;padding:10px;border:1px solid #d9e2e8;border-radius:8px;background:#fff}
.recent15-report-team header{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;align-items:start}.recent15-report-team h3{margin:0;font-size:14px}.recent15-report-team small{color:#65717d}
.recent15-report-team header strong{padding:4px 7px;border-radius:999px;background:#eaf6f2;color:#0f7a5f;font-size:11px;white-space:nowrap}.recent15-report-summary{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}
.recent15-report-summary span{padding:7px;border:1px solid #d9e2e8;border-radius:8px;background:#f8fafb}.recent15-report-summary b,.recent15-report-summary small{display:block}
.recent15-report-list{display:grid;gap:6px}.recent15-report-match{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:5px 8px;padding:8px;border:1px solid #d9e2e8;border-left:4px solid #96a3ad;border-radius:8px;background:#fff;break-inside:avoid}
.recent15-report-match.official{border-left-color:#0f7a5f}.recent15-report-match.friendly{border-left-color:#d9822b}.recent15-report-main{min-width:0}.recent15-report-main span,.recent15-report-main small{display:block;color:#65717d;font-size:10px;font-weight:700}
.recent15-report-main strong{display:block;margin:2px 0;color:#16202a;font-size:12px;overflow-wrap:anywhere}.recent15-report-score{display:grid;justify-items:end;align-content:start;gap:3px}.recent15-report-score b{font-size:14px}
.recent15-report-score span{padding:3px 6px;border-radius:999px;background:#edf2f4;color:#40505d;font-size:10px;font-weight:800}.recent15-report-tags{display:flex;flex-wrap:wrap;gap:4px;grid-column:1/-1}.recent15-report-tags span{padding:3px 6px;border-radius:999px;background:#eef3f5;color:#40505d;font-size:10px;font-weight:800}
.metric-bars{display:grid;gap:7px}.metric-bar{display:grid;grid-template-columns:84px minmax(0,1fr) 72px;gap:8px;align-items:center}
.metric-bar i{height:9px;border-radius:999px;background:#edf2f4;overflow:hidden}.metric-bar b{display:block;height:100%;border-radius:inherit;background:#0f7a5f}
.metric-bar strong{text-align:right}.table-section{margin-top:18px}.table-wrap{max-width:100%;overflow:auto;border:1px solid #d9e2e8;border-radius:8px;background:#fff}
table{width:100%;min-width:760px;border-collapse:collapse}th,td{padding:8px 9px;border-bottom:1px solid #d9e2e8;text-align:left;vertical-align:top;font-size:12px}
th{background:#f0f4f6;color:#40505d}@media print{body{background:#fff}main{padding:0}.fixture-grid{grid-template-columns:1fr}.table-wrap{overflow:visible}table{min-width:0}}
@media(max-width:760px){main{padding:14px}.report-title,.summary-grid,.fixture-grid,.recent15-columns{grid-template-columns:1fr;display:grid}.metric-bar{grid-template-columns:64px minmax(0,1fr) 58px}}
"""


def score_model_display_label(key: Any) -> str:
    normalized = normalize_score_model_key(key)
    for option in score_model_options():
        if option.get("key") == normalized:
            return str(option.get("label") or normalized)
    return str(key or "Poisson independiente")


def unique_strings(values: Iterable[Any]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def monte_carlo_match_prediction(
        fixture: pd.Series,
        base_model: WorldCupModel,
        config: Dict[str, Any],
        rng: np.random.Generator,
        international_matches: pd.DataFrame,
) -> Dict[str, Any]:
    home_team = str(fixture.get("Equipo 1", ""))
    away_team = str(fixture.get("Equipo 2", ""))
    context = contextual_poisson_for_match(
        home_team,
        away_team,
        base_model=base_model,
        before_date=fixture.get("Fecha", ""),
        max_goals=int(config["max_goals"]),
        matches=international_matches,
        limit=int(config["poisson_recent_matches"]),
    )
    lambda_home, lambda_away = contextual_lambdas(context, base_model, home_team, away_team)
    iterations = int(config["iterations"])
    score_grid = match_score_grid_for_lambdas(base_model, lambda_home, lambda_away, max_goals=int(config["max_goals"]))
    score_metadata = score_model_metadata(base_model)
    counts = {
        "home": 0,
        "draw": 0,
        "away": 0,
        "sum_home_goals": 0,
        "sum_away_goals": 0,
    }
    total_line_counts = {line: 0 for line in TOTAL_GOAL_LINES}
    score_counts: Counter[Tuple[int, int]] = Counter()
    chunk_size = 200_000
    remaining = iterations
    while remaining > 0:
        size = min(chunk_size, remaining)
        if score_grid is None:
            sampled_home = rng.poisson(lambda_home, size=size)
            sampled_away = rng.poisson(lambda_away, size=size)
        else:
            sampled_home, sampled_away = sample_scores_from_grid(score_grid, rng, size=size)
        total_goals = sampled_home + sampled_away
        counts["home"] += int(np.sum(sampled_home > sampled_away))
        counts["draw"] += int(np.sum(sampled_home == sampled_away))
        counts["away"] += int(np.sum(sampled_away > sampled_home))
        counts["sum_home_goals"] += int(np.sum(sampled_home))
        counts["sum_away_goals"] += int(np.sum(sampled_away))
        for line in TOTAL_GOAL_LINES:
            total_line_counts[line] += int(np.sum(total_goals > float(line)))
        score_counts.update(zip(sampled_home.astype(int).tolist(), sampled_away.astype(int).tolist()))
        remaining -= size
    probabilities = {
        "home": _pct_count(counts["home"], iterations),
        "draw": _pct_count(counts["draw"], iterations),
        "away": _pct_count(counts["away"], iterations),
    }
    for line in TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        probabilities[f"over{suffix}"] = _pct_count(total_line_counts[line], iterations)
        probabilities[f"under{suffix}"] = _pct_count(iterations - total_line_counts[line], iterations)
    top_scores = monte_carlo_top_scores(score_counts, iterations)
    return {
        "fixture": {
            "id": str(fixture.get("No.", "")),
            "date": fixture.get("Fecha", ""),
            "time": fixture.get("Hora", ""),
            "group": fixture.get("Grupo", ""),
            "home": home_team,
            "away": away_team,
            "venue": fixture.get("Sede", ""),
        },
        "probabilities": probabilities,
        "expected_goals": {
            "home": round(float(lambda_home), 3),
            "away": round(float(lambda_away), 3),
        },
        "simulated_goals": {
            "home": round(float(counts["sum_home_goals"]) / max(iterations, 1), 3),
            "away": round(float(counts["sum_away_goals"]) / max(iterations, 1), 3),
        },
        "top_scores": top_scores,
        "modal_score": (top_scores[0] or {}).get("score", "") if top_scores else "",
        "iterations": iterations,
        "seed": int(config["seed"]),
        "poisson_recent_matches": int(config["poisson_recent_matches"]),
        "score_model": score_metadata,
        "contextual_poisson": context,
        "source": score_metadata.get("label") or ("Poisson contextual" if context.get("available") else "Poisson base"),
    }


def contextual_lambdas(context: Dict[str, Any], base_model: WorldCupModel, home: str, away: str) -> Tuple[float, float]:
    try:
        lambda_home = float(context.get("context_lambda_home") or (context.get("lambdas") or {}).get("home") or 0.0)
        lambda_away = float(context.get("context_lambda_away") or (context.get("lambdas") or {}).get("away") or 0.0)
    except (TypeError, ValueError):
        lambda_home = 0.0
        lambda_away = 0.0
    if lambda_home > 0 and lambda_away > 0:
        return lambda_home, lambda_away
    probabilities = base_model.match_probabilities(home, away)
    return float(probabilities.get("lambda1", 1.0)), float(probabilities.get("lambda2", 1.0))


def monte_carlo_match_iterations(value: Any) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = int(DEFAULT_CONFIG["iterations"])
    return min(max(number, 100), REPORT_MAX_ITERATIONS)


def monte_carlo_top_scores(score_counts: Counter[Tuple[int, int]], iterations: int) -> List[Dict[str, Any]]:
    return [
        {
            "score": f"{home_goals}-{away_goals}",
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
            "probability": round(float(count) * 100.0 / max(iterations, 1), 2),
            "count": int(count),
        }
        for (home_goals, away_goals), count in score_counts.most_common(5)
    ]


def monte_carlo_match_row(result: Dict[str, Any]) -> Dict[str, Any]:
    fixture = result.get("fixture", {})
    probs = result.get("probabilities", {})
    expected = result.get("expected_goals", {})
    top_score = (result.get("top_scores") or [{}])[0]
    return {
        "No.": fixture.get("id", ""),
        "Fecha": fixture.get("date", ""),
        "Grupo": fixture.get("group", ""),
        "Partido": f"{fixture.get('home', '')} vs {fixture.get('away', '')}",
        "MC 1 %": probs.get("home", ""),
        "MC X %": probs.get("draw", ""),
        "MC 2 %": probs.get("away", ""),
        "Over 0.5 %": probs.get("over05", ""),
        "Under 0.5 %": probs.get("under05", ""),
        "Over 1.5 %": probs.get("over15", ""),
        "Under 1.5 %": probs.get("under15", ""),
        "Over 2.5 %": probs.get("over25", ""),
        "Under 2.5 %": probs.get("under25", ""),
        "Over 3.5 %": probs.get("over35", ""),
        "Under 3.5 %": probs.get("under35", ""),
        "Lambda Local": expected.get("home", ""),
        "Lambda Visita": expected.get("away", ""),
        "Top score": top_score.get("score", ""),
        "Top score %": top_score.get("probability", ""),
        "Modelo marcador": (result.get("score_model") or {}).get("label", ""),
        "Iteraciones": result.get("iterations", ""),
        "Fuente": result.get("source", ""),
    }


def _pct_count(count: int, total: int) -> float:
    return round(float(count) * 100.0 / max(int(total or 1), 1), 2)


def upcoming_fixture_rows(tournament: Dict[str, Any], group_filter: str = "") -> pd.DataFrame:
    df = tournament_fixtures_dataframe(tournament)
    df = df[df["Grupo"].astype(str) != ""].copy()
    if group_filter:
        df = df[df["Grupo"].astype(str) == group_filter]
    df = df[
        df["Equipo 1"].astype(str).str.len().gt(1) &
        df["Equipo 2"].astype(str).str.len().gt(1) &
        ~df["Equipo 1"].astype(str).str.match(r"^[123W][A-Z0-9/]+$") &
        ~df["Equipo 2"].astype(str).str.match(r"^[123W][A-Z0-9/]+$")
    ].copy()
    df = attach_fixture_schedule(df)
    upcoming = future_fixture_rows(df)
    if upcoming.empty:
        upcoming = df[df["_date"].notna()]
    if upcoming.empty:
        upcoming = df
    return drop_internal_fixture_columns(upcoming.sort_values(["_sort_time", "No."], kind="stable"))


def upcoming_prediction_row(result: Dict[str, Any]) -> Dict[str, Any]:
    fixture = result.get("fixture", {})
    probs = result.get("probabilities", {})
    expected = result.get("expected_goals", {})
    sources = result.get("market_sources", {})
    contextual = result.get("contextual_poisson", {}) or {}
    context_top = (contextual.get("top_scores") or [{}])[0] if contextual.get("top_scores") else {}
    recent_limit = int(contextual.get("match_limit") or 15)
    return {
        "No.": fixture.get("id", ""),
        "Fecha": fixture.get("date", ""),
        "Grupo": fixture.get("group", ""),
        "Partido": f"{fixture.get('home', '')} vs {fixture.get('away', '')}",
        "1 %": probs.get("home", ""),
        "X %": probs.get("draw", ""),
        "2 %": probs.get("away", ""),
        "Over 0.5 %": probs.get("over05", ""),
        "Under 0.5 %": probs.get("under05", ""),
        "Over 1.5 %": probs.get("over15", ""),
        "Under 1.5 %": probs.get("under15", ""),
        "Over 2.5 %": probs.get("over25", ""),
        "Under 2.5 %": probs.get("under25", ""),
        "Over 3.5 %": probs.get("over35", ""),
        "Under 3.5 %": probs.get("under35", ""),
        "Prediccion": result.get("prediction", ""),
        "xG": f"{expected.get('home', '')}-{expected.get('away', '')}",
        f"Poisson {recent_limit}": "Si" if contextual.get("available") else "Base" if contextual.get("matrix_available") else "No",
        f"Lambda {recent_limit} Local": contextual.get("context_lambda_home", ""),
        f"Lambda {recent_limit} Visita": contextual.get("context_lambda_away", ""),
        f"Top score {recent_limit}": context_top.get("score", ""),
        "Fuente 1X2": (sources.get("result") or {}).get("source", ""),
        "Fuente O/U": (sources.get("over_under_25") or {}).get("source", ""),
        "Fuente U/O 0.5": (sources.get("over_under_05") or {}).get("source", ""),
        "Fuente U/O 1.5": (sources.get("over_under_15") or {}).get("source", ""),
        "Fuente U/O 2.5": (sources.get("over_under_25") or {}).get("source", ""),
        "Fuente U/O 3.5": (sources.get("over_under_35") or {}).get("source", ""),
    }


def lineup_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    enriched = enrich_lineup_payload(payload)
    return {
        "lineup": enriched,
        "players": table_payload(lineups_table(enriched), page=1, page_size=40),
    }


def simulate(payload: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
    global LAST_SIMULATION_RESULT
    config = simulation_config(payload)
    emit_job_progress(progress_callback, "preparing", 0, 100, "Preparando Monte Carlo")
    tournament, fixture_source = load_tournament_2026(refresh=bool(config["refresh"]))
    results_autorefresh = ensure_worldcup_results_autorefreshed_once(tournament)
    model, history_source = build_model(tournament, config)
    poisson_layers = ["Poisson base"]
    if config["include_confirmed_results"]:
        poisson_layers.append("resultados confirmados")
    if config["mode"] == "poisson_live":
        model = RecentPoissonWorldCupModel(model, recent_match_limit=int(config["poisson_recent_matches"]))
        poisson_layers.append(f"Poisson ultimos {config['poisson_recent_matches']}")
    model = apply_configured_score_model(model, tournament, config)
    score_metadata = score_model_metadata(model)
    if score_metadata.get("key") != DEFAULT_SCORE_MODEL:
        poisson_layers.append(str(score_metadata.get("label") or score_metadata.get("key")))
    confirmed_results = confirmed_group_results(tournament) if config["include_confirmed_results"] else []
    result = simulate_worldcup(
        tournament=tournament,
        model=model,
        iterations=int(config["iterations"]),
        seed=int(config["seed"]),
        include_confirmed_results=bool(config["include_confirmed_results"]),
        confirmed_results=confirmed_results,
        progress_callback=progress_callback,
    )
    emit_job_progress(progress_callback, "rendering", 100, 100, "Preparando resultados")
    output = {
        "summary": {
            "model": "Elo + modelos de marcador Monte Carlo",
            "config": public_report_config(config),
            "mode": config["mode"],
            "fixture_source": fixture_source,
            "history_source": history_source,
            "results_autorefresh": results_autorefresh,
            "score_model": score_metadata,
            "poisson_layers": poisson_layers,
            "confirmed_results": len(confirmed_results),
            "result_policy": (
                f"Poisson live ultimos {config['poisson_recent_matches']} + resultados confirmados"
                if config["mode"] == "poisson_live" and config["include_confirmed_results"]
                else f"Poisson live ultimos {config['poisson_recent_matches']} sin resultados 2026"
                if config["mode"] == "poisson_live"
                else "Poisson base + resultados confirmados"
                if config["include_confirmed_results"]
                else "Poisson prospectivo base sin resultados 2026"
            ),
            "anti_leakage": [
                "Historico filtrado antes del 2026-06-11.",
                "Predicciones calculadas con historico internacional y modelos Poisson/SOTA sin entrenamiento supervisado en Mundial.",
                "Los resultados 2026 confirmados solo se incluyen cuando la simulacion se ejecuta en modo Poisson live.",
            ],
        },
        "advancement": table_payload(result["advancement"], page=1, page_size=80),
        "matches": table_payload(result["matches"], page=1, page_size=120),
        "procedure": procedure()["steps"],
    }
    LAST_SIMULATION_RESULT = output
    emit_job_progress(progress_callback, "complete", 100, 100, "Monte Carlo completado")
    return output


def emit_job_progress(callback, stage: str, current: int, total: int, message: str, **extra):
    if callback is None:
        return
    total = max(int(total or 1), 1)
    current = min(max(int(current or 0), 0), total)
    callback({
        "stage": stage,
        "current": current,
        "total": total,
        "current_trial": "",
        "total_trials": "",
        "percent": int(round(current * 100 / total)),
        "message": message,
        **extra,
    })

def procedure() -> Dict[str, Any]:
    return {
        "title": "Procedimiento de prediccion Mundial 2026",
        "steps": [
            {
                "name": "Datos del torneo",
                "detail": "Carga grupos y fixtures 2026 desde cache/openfootball, con fallback local si la fuente publica no responde.",
            },
            {
                "name": "Historico sin leakage",
                "detail": "Carga partidos 1930-2022 y filtra cualquier registro igual o posterior al 2026-06-11.",
            },
            {
                "name": "Modelo base",
                "detail": "Actualiza ratings Elo, ataque y defensa por seleccion; despues transforma esos perfiles a goles esperados Poisson.",
            },
            {
                "name": "Parámetros estadísticos",
                "detail": "Ajusta peso histórico, recencia, ventaja local, límite de goles y modelo de marcador sin entrenamiento supervisado.",
            },
            {
                "name": "Walk-forward",
                "detail": "Cuando hay partidos ya jugados, guarda snapshots de resultado para un reentreno incremental sin mezclar datos incompletos.",
            },
            {
                "name": "Monte Carlo",
                "detail": "Simula fase de grupos, mejores terceros y bracket completo para estimar avance, final y campeon.",
            },
            {
                "name": "Predicciones futuras",
                "detail": "Genera consenso Poisson/SOTA y reporta 1X2 junto con U/O 0.5, 1.5, 2.5 y 3.5 para los proximos N partidos.",
            },
        ],
        "sources": [
            "openfootball/worldcup.json",
            "Football-Data WorldCup2026.xlsx para odds 1X2 historicas y clasificatorios",
            "storage/worldcup/market/manual_odds.csv opcional para odds actuales/O-U 2.5",
            "GitHub: martj42/international_results/results.csv para resultados internacionales recientes; Kaggle queda como fallback",
            "storage/worldcup/cache/*.json",
            "Wikipedia squads opcional para jugadores",
        ],
    }


def score_history_for_tournament(tournament: Dict[str, Any], config: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    group_map = groups_from_tournament(tournament)
    team_names = [team for group_teams in group_map.values() for team in group_teams]
    international_matches = load_international_matches(required=False)
    history_df = international_history_rows(international_matches, teams=team_names)
    history_source = str(history_df.attrs.get("source") or "")
    if not history_df.empty:
        return history_df, history_source
    history_df, history_source = load_historical_matches(refresh=bool(config.get("refresh", False)))
    history_df = filter_training_scope_sources(history_df, team_names, keep_unknown_date=False)
    return history_df, history_source


def build_model(tournament: Dict[str, Any], config: Dict[str, Any]) -> Tuple[WorldCupModel, str]:
    group_map = groups_from_tournament(tournament)
    team_names = [team for group_teams in group_map.values() for team in group_teams]
    history_df, history_source = score_history_for_tournament(tournament, config)
    model = WorldCupModel.from_history(
        history_df,
        teams=team_names,
        history_weight=float(config["history_weight"]),
        recency_weight=float(config["recency_weight"]),
        host_advantage=float(config["host_advantage"]),
        max_goals=int(config["max_goals"]),
    )
    return model, history_source


def apply_configured_score_model(model: Any, tournament: Dict[str, Any], config: Dict[str, Any]) -> Any:
    if normalize_score_model_key(config.get("score_model")) == DEFAULT_SCORE_MODEL:
        return model
    group_map = groups_from_tournament(tournament)
    team_names = [team for group_teams in group_map.values() for team in group_teams]
    history_df, _ = score_history_for_tournament(tournament, config)
    return build_score_model(model, history_df=history_df, teams=team_names, config=config)


def score_model_metadata(model: Any) -> Dict[str, Any]:
    method = getattr(model, "score_model_metadata", None)
    if callable(method):
        return method()
    base_model = getattr(model, "base_model", None)
    if base_model is not None and base_model is not model:
        return score_model_metadata(base_model)
    return {
        "key": DEFAULT_SCORE_MODEL,
        "label": "Poisson independiente",
        "available": True,
        "params": {},
        "warnings": [],
    }


def match_score_grid_for_lambdas(model: Any, lambda_home: float, lambda_away: float, max_goals: int) -> np.ndarray | None:
    method = getattr(model, "score_grid_from_lambdas", None)
    if callable(method):
        try:
            return method(lambda_home, lambda_away, max_goals=max_goals)
        except Exception:
            return None
    base_model = getattr(model, "base_model", None)
    if base_model is not None and base_model is not model:
        return match_score_grid_for_lambdas(base_model, lambda_home, lambda_away, max_goals=max_goals)
    return None


def simulation_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = payload or {}
    mode = str(payload.get("mode") or "poisson").strip().lower()
    if mode in {"hybrid", "default"}:
        mode = "poisson"
    if mode not in {"poisson", "poisson_live"}:
        mode = "poisson"
    return {
        "mode": mode,
        "iterations": int(_clamp_int(payload.get("iterations", DEFAULT_CONFIG["iterations"]), 100, REPORT_MAX_ITERATIONS)),
        "seed": int(payload.get("seed") if payload.get("seed") is not None else DEFAULT_CONFIG["seed"]),
        "history_weight": _clamp_float(payload.get("history_weight", DEFAULT_CONFIG["history_weight"]), 0.2, 2.0),
        "recency_weight": _clamp_float(payload.get("recency_weight", DEFAULT_CONFIG["recency_weight"]), 0.0, 1.0),
        "host_advantage": _clamp_float(payload.get("host_advantage", DEFAULT_CONFIG["host_advantage"]), 0.0, 120.0),
        "max_goals": int(_clamp_int(payload.get("max_goals", DEFAULT_CONFIG["max_goals"]), 6, 14)),
        "poisson_recent_matches": int(_clamp_int(payload.get("poisson_recent_matches", DEFAULT_CONFIG["poisson_recent_matches"]), 3, 50)),
        "score_model": normalize_score_model_key(payload.get("score_model", DEFAULT_CONFIG["score_model"])),
        "stat_model_cache": bool(payload.get("stat_model_cache", DEFAULT_CONFIG["stat_model_cache"])),
        "stat_model_refit": bool(payload.get("stat_model_refit", DEFAULT_CONFIG["stat_model_refit"])),
        "stat_lambda_model": str(payload.get("stat_lambda_model", DEFAULT_CONFIG["stat_lambda_model"]) or STATSMODELS_POISSON_GLM_MODEL).strip().lower(),
        "stat_glm_min_matches": int(_clamp_int(payload.get("stat_glm_min_matches", DEFAULT_CONFIG["stat_glm_min_matches"]), 4, 500)),
        "stat_glm_validation_fraction": _clamp_float(payload.get("stat_glm_validation_fraction", DEFAULT_CONFIG["stat_glm_validation_fraction"]), 0.05, 0.4),
        "score_mle_recency_weight": _clamp_float(
            payload.get("score_mle_recency_weight", payload.get("recency_weight", DEFAULT_CONFIG["recency_weight"])),
            0.0,
            1.0,
        ),
        "bayes_draws": int(_clamp_int(payload.get("bayes_draws", DEFAULT_CONFIG["bayes_draws"]), 100, 10000)),
        "bayes_tune": int(_clamp_int(payload.get("bayes_tune", DEFAULT_CONFIG["bayes_tune"]), 100, 10000)),
        "bayes_chains": int(_clamp_int(payload.get("bayes_chains", DEFAULT_CONFIG["bayes_chains"]), 1, 8)),
        "bayes_target_accept": float(np.clip(float(payload.get("bayes_target_accept", DEFAULT_CONFIG["bayes_target_accept"])), 0.8, 0.995)),
        "bayes_max_treedepth": int(_clamp_int(payload.get("bayes_max_treedepth", DEFAULT_CONFIG["bayes_max_treedepth"]), 6, 15)),
        "advanced_include_bayesian": bool(payload.get("advanced_include_bayesian", DEFAULT_CONFIG["advanced_include_bayesian"])),
        "refresh": bool(payload.get("refresh", DEFAULT_CONFIG["refresh"])),
        "include_confirmed_results": bool(payload.get("include_confirmed_results", mode == "poisson_live")),
    }


def confirmed_group_results(tournament: Dict[str, Any]) -> List[Dict[str, Any]]:
    fixture_df = tournament_fixtures_dataframe(tournament)
    if fixture_df.empty or not {"Goles 1", "Goles 2"}.issubset(fixture_df.columns):
        return []
    working = fixture_df.copy()
    working["HG"] = pd.to_numeric(working["Goles 1"], errors="coerce")
    working["AG"] = pd.to_numeric(working["Goles 2"], errors="coerce")
    working = working[
        working["Grupo"].astype(str).str.len().gt(0) &
        working["Equipo 1"].astype(str).str.len().gt(1) &
        working["Equipo 2"].astype(str).str.len().gt(1) &
        working["HG"].notna() &
        working["AG"].notna()
    ].copy()
    rows: List[Dict[str, Any]] = []
    for _, fixture in working.iterrows():
        rows.append({
            "fixture_id": str(fixture.get("No.", "")),
            "date": fixture.get("Fecha", ""),
            "group": fixture.get("Grupo", ""),
            "team1": fixture.get("Equipo 1", ""),
            "team2": fixture.get("Equipo 2", ""),
            "goals1": int(fixture.get("HG")),
            "goals2": int(fixture.get("AG")),
            "source": fixture.get("Fuente Resultado", ""),
        })
    return rows


def enrich_lineup_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    lineup = dict(payload or {})
    lineup["home_asset"] = team_asset(str(lineup.get("home", "")))
    lineup["away_asset"] = team_asset(str(lineup.get("away", "")))
    players = [dict(player) for player in lineup.get("players", [])]
    for player in players:
        player["initials"] = initials(player.get("name", ""))
        if not player.get("photo_url"):
            player["photo_url"] = sofa_player_photo_url(player.get("id", ""))
        player["asset"] = team_asset(str(player.get("team", "")))
        player["x"] = ""
        player["y"] = ""
    for team_key, formation_key in (("home", "formation_home"), ("away", "formation_away")):
        team = str(lineup.get(team_key, ""))
        starters = [player for player in players if player.get("team") == team and player.get("starter")]
        positions = lineup_positions(len(starters), str(lineup.get(formation_key, "")))
        for player, position in zip(starters, positions):
            player.update(position)
    lineup["players"] = players
    return lineup


def lineup_positions(total: int, formation: str) -> List[Dict[str, float]]:
    if total <= 0:
        return []
    lines = [int(part) for part in re.findall(r"\d+", formation or "") if int(part) > 0]
    if sum(lines) != max(total - 1, 0):
        if total >= 11:
            lines = [4, 3, 3]
        else:
            lines = [max(total - 1, 0)]
    line_counts = [1] + lines
    if sum(line_counts) < total:
        line_counts[-1] += total - sum(line_counts)
    positions = []
    y_values = _line_y_values(len(line_counts))
    for line_index, count in enumerate(line_counts):
        y = y_values[line_index]
        for slot in range(count):
            x = ((slot + 1) / (count + 1)) * 100.0
            positions.append({"x": round(x, 1), "y": round(y, 1), "line": line_index})
            if len(positions) >= total:
                return positions
    return positions[:total]


def _line_y_values(lines: int) -> List[float]:
    if lines <= 1:
        return [50.0]
    return [88.0 - (index * (74.0 / max(lines - 1, 1))) for index in range(lines)]


def team_asset(team: str) -> Dict[str, str]:
    team = str(team or "").strip()
    local = local_flag_url(team)
    code = COUNTRY_CODES.get(team, "")
    return {
        "name": team,
        "slug": safe_slug(team),
        "flag_url": local or (f"https://flagcdn.com/w80/{code}.png" if code else ""),
        "flag_fallback": initials(team),
        "country_code": code,
    }


def local_flag_url(team: str) -> str:
    names = [team, LOCAL_FLAG_ALIASES.get(team, "")]
    ascii_name = team.replace("Curaçao", "Curacao")
    names.append(ascii_name)
    for name in names:
        if not name:
            continue
        path = COUNTRY_FLAGS_ROOT / f"{name}.png"
        if path.exists():
            return f"/assets/graphics/countries/{path.name}"
    return ""


def fixture_overview_payload(fixture_df: pd.DataFrame) -> Dict[str, Any]:
    playable = fixture_df[
        fixture_df["Grupo"].astype(str).str.len().gt(0) &
        fixture_df["Equipo 1"].astype(str).str.len().gt(1) &
        fixture_df["Equipo 2"].astype(str).str.len().gt(1)
    ].copy()
    if playable.empty:
        fallback = _opener_payload(fixture_df)
        return {
            "opener": fallback,
            "featured_matches": [],
            "highlight": {},
            "next_matches": [],
            "countdown_target": "",
            "countdown_state": "pending",
        }
    playable = attach_fixture_schedule(playable).sort_values(["_sort_time", "No."], kind="stable").reset_index(drop=True)
    opener = fixture_card_payload(playable.iloc[0])
    upcoming = future_fixture_rows(playable)
    if upcoming.empty:
        return {
            "opener": opener,
            "featured_matches": [],
            "highlight": {},
            "next_matches": [],
            "countdown_target": "",
            "countdown_state": "finished",
        }
    first_sort = upcoming.iloc[0].get("_sort_time")
    if pd.isna(first_sort):
        featured = upcoming.iloc[[0]].copy()
    else:
        featured = upcoming[upcoming["_sort_time"].eq(first_sort)].copy()
    featured_matches = [fixture_card_payload(row) for _, row in featured.iterrows()]
    next_rows = upcoming.loc[~upcoming.index.isin(featured.index)].copy()
    next_matches = [
        fixture_card_payload(row)
        for _, row in next_rows.head(4).iterrows()
    ]
    highlight = featured_matches[0] if featured_matches else {}
    return {
        "opener": opener,
        "featured_matches": featured_matches,
        "highlight": highlight,
        "next_matches": next_matches,
        "countdown_target": highlight.get("kickoff_iso", ""),
        "countdown_state": "ready" if highlight.get("kickoff_iso") else "pending",
    }


def group_standings_payload(group_map: Dict[str, List[str]], fixture_df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "name": group_name,
            "letter": group_letter(group_name),
            "played_matches": group_finished_fixture_count(group_name, fixture_df),
            "rows": group_standing_rows(group_name, team_names, fixture_df),
        }
        for group_name, team_names in group_map.items()
    ]


def group_finished_fixture_count(group_name: str, fixture_df: pd.DataFrame) -> int:
    if fixture_df is None or fixture_df.empty or "Grupo" not in fixture_df.columns:
        return 0
    scoped = fixture_df[fixture_df["Grupo"].astype(str) == str(group_name)].copy()
    if scoped.empty or not {"Goles 1", "Goles 2"}.issubset(scoped.columns):
        return 0
    goals_1 = pd.to_numeric(scoped["Goles 1"], errors="coerce")
    goals_2 = pd.to_numeric(scoped["Goles 2"], errors="coerce")
    return int((goals_1.notna() & goals_2.notna()).sum())


def group_standing_rows(group_name: str, team_names: List[str], fixture_df: pd.DataFrame) -> List[Dict[str, Any]]:
    seed_map = {team: index for index, team in enumerate(team_names, start=1)}
    rows: Dict[str, Dict[str, Any]] = {
        team: {
            **team_asset(team),
            "team": team,
            "seed": seed,
            "PJ": 0,
            "Pts": 0,
            "GF": 0,
            "GC": 0,
            "DG": 0,
        }
        for team, seed in seed_map.items()
    }
    if fixture_df is not None and not fixture_df.empty and "Grupo" in fixture_df.columns:
        group_rows = fixture_df[fixture_df["Grupo"].astype(str) == str(group_name)].copy()
        for _, fixture in group_rows.iterrows():
            home = str(fixture.get("Equipo 1", "") or "").strip()
            away = str(fixture.get("Equipo 2", "") or "").strip()
            if home not in rows or away not in rows:
                continue
            goals_home = _score_value(fixture.get("Goles 1", ""))
            goals_away = _score_value(fixture.get("Goles 2", ""))
            if goals_home is None or goals_away is None:
                continue
            rows[home]["PJ"] += 1
            rows[away]["PJ"] += 1
            rows[home]["GF"] += goals_home
            rows[home]["GC"] += goals_away
            rows[away]["GF"] += goals_away
            rows[away]["GC"] += goals_home
            if goals_home > goals_away:
                rows[home]["Pts"] += 3
            elif goals_home < goals_away:
                rows[away]["Pts"] += 3
            else:
                rows[home]["Pts"] += 1
                rows[away]["Pts"] += 1
    for row in rows.values():
        row["DG"] = int(row["GF"] - row["GC"])
    ordered = sorted(
        rows.values(),
        key=lambda row: (-int(row["Pts"]), -int(row["DG"]), -int(row["GF"]), int(row["seed"])),
    )
    return jsonable(ordered)


def _score_value(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number)


def fixture_card_payload(row: pd.Series | Dict[str, Any]) -> Dict[str, Any]:
    record = row if isinstance(row, dict) else row.to_dict()
    home = str(record.get("Equipo 1", "Mexico"))
    away = str(record.get("Equipo 2", "South Africa"))
    kickoff = str(record.get("_kickoff_iso") or fixture_kickoff_iso(record.get("Fecha", ""), record.get("Hora", "")))
    return {
        "id": str(record.get("No.", "")),
        "date": record.get("Fecha", "2026-06-11"),
        "time": record.get("Hora", ""),
        "round": record.get("Ronda", ""),
        "group": record.get("Grupo", ""),
        "match": f"{home} vs {away}",
        "home": team_asset(home),
        "away": team_asset(away),
        "venue": record.get("Sede", ""),
        "kickoff_iso": kickoff,
    }


def _opener_payload(fixture_df: pd.DataFrame) -> Dict[str, Any]:
    if not fixture_df.empty:
        return fixture_card_payload(fixture_df.iloc[0])
    return {
        "id": "1",
        "date": "2026-06-11",
        "time": "13:00 UTC-6",
        "round": "Matchday 1",
        "group": "Group A",
        "match": "Mexico vs South Africa",
        "home": team_asset("Mexico"),
        "away": team_asset("South Africa"),
        "venue": "Mexico City",
        "kickoff_iso": fixture_kickoff_iso("2026-06-11", "13:00 UTC-6"),
    }


def fixture_kickoff_iso(date_value: Any, time_value: Any) -> str:
    kickoff = fixture_kickoff_datetime(date_value, time_value, require_time=True)
    return kickoff.isoformat() if kickoff else ""


def fixture_countdown_state(kickoff_iso: Any, finished: Any = False) -> str:
    if str(finished).strip().lower() in {"si", "sí", "yes", "true", "1"} or finished is True:
        return "finished"
    target = pd.to_datetime(kickoff_iso, utc=True, errors="coerce")
    if pd.isna(target):
        return "pending"
    now = pd.Timestamp(_now_utc())
    if now.tzinfo is None:
        now = now.tz_localize(timezone.utc)
    else:
        now = now.tz_convert(timezone.utc)
    if target > now:
        return "ready"
    if now <= target + pd.Timedelta(hours=3):
        return "live"
    return "finished"


def fixture_kickoff_datetime(date_value: Any, time_value: Any, require_time: bool = False) -> datetime | None:
    date_text = str(date_value or "").strip()[:10]
    if not date_text:
        return None
    time_text = str(time_value or "").strip()
    offset_hours = 0
    hour = 0
    minute = 0
    match = re.search(r"(\d{1,2}):(\d{2})(?:\s*UTC([+-]\d{1,2}))?", time_text)
    if require_time and not match:
        return None
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        if match.group(3):
            offset_hours = int(match.group(3))
    try:
        kickoff = datetime.strptime(date_text, "%Y-%m-%d").replace(
            hour=hour,
            minute=minute,
            tzinfo=timezone(timedelta(hours=offset_hours)),
        )
    except ValueError:
        return None
    return kickoff.astimezone(timezone.utc)


def attach_fixture_schedule(df: pd.DataFrame) -> pd.DataFrame:
    scheduled = df.copy()
    kickoff_values = []
    kickoff_iso_values = []
    has_kickoff_time = []
    for _, row in scheduled.iterrows():
        kickoff = fixture_kickoff_datetime(row.get("Fecha", ""), row.get("Hora", ""), require_time=True)
        kickoff_values.append(kickoff)
        kickoff_iso_values.append(kickoff.isoformat() if kickoff else "")
        has_kickoff_time.append(kickoff is not None)
    scheduled["_date"] = pd.to_datetime(scheduled["Fecha"], utc=True, errors="coerce")
    scheduled["_kickoff_iso"] = kickoff_iso_values
    scheduled["_kickoff"] = pd.to_datetime(kickoff_values, utc=True, errors="coerce")
    scheduled["_has_kickoff_time"] = has_kickoff_time
    date_sort = pd.to_datetime(scheduled["Fecha"], utc=True, errors="coerce")
    scheduled["_sort_time"] = scheduled["_kickoff"].where(scheduled["_kickoff"].notna(), date_sort)
    return scheduled


def future_fixture_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    now = _utcify_datetime(_now_utc())
    today = pd.Timestamp(now).tz_convert(timezone.utc).normalize()
    finished = pd.Series(False, index=df.index)
    if "Finalizado" in df.columns:
        finished = df["Finalizado"].astype(str).str.strip().str.lower().isin({"si", "sí", "yes", "true", "1"})
    has_time = df["_has_kickoff_time"].astype(bool)
    future_by_time = has_time & df["_kickoff"].notna() & (df["_kickoff"] > now)
    future_by_date = ~has_time & df["_date"].notna() & (df["_date"] >= today)
    effective_finished = finished & ~future_by_time
    return df[~effective_finished & (future_by_time | future_by_date)].sort_values(["_sort_time", "No."], kind="stable").copy()


def drop_internal_fixture_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[column for column in df.columns if str(column).startswith("_")], errors="ignore")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def table_payload(df: pd.DataFrame, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
    page = max(int(page or 1), 1)
    page_size = min(max(int(page_size or 50), 1), 500)
    total = int(df.shape[0])
    start = (page - 1) * page_size
    page_df = df.iloc[start:start + page_size].copy()
    page_df = page_df.astype(object).where(pd.notna(page_df), "")
    return {
        "columns": [str(column) for column in page_df.columns],
        "rows": jsonable(page_df.to_dict(orient="records")),
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": int(math.ceil(total / page_size)) if page_size else 0,
    }


def metrics_dataframe(metrics: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split, values in (metrics or {}).items():
        row = {"Split": split}
        row.update(values or {})
        rows.append(row)
    return pd.DataFrame(rows)


def jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return table_payload(value)
    if isinstance(value, pd.Series):
        return jsonable(value.to_dict())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (datetime, date, Path)):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return value


def initials(value: Any) -> str:
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]+", str(value or ""))
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:2].upper()
    return "".join(word[0] for word in words[:2]).upper()


def safe_slug(value: Any) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return slug or "unknown"


def _clamp_float(value: Any, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = lower
    if not math.isfinite(number):
        number = lower
    return min(max(number, lower), upper)


def _clamp_int(value: Any, lower: int, upper: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = lower
    return min(max(number, lower), upper)
