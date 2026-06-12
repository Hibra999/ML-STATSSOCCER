from __future__ import annotations

import math
import hashlib
import json
import re
import shutil
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
    build_score_model,
    normalize_score_model_key,
    sample_scores_from_grid,
    score_model_options,
)
from src.worldcup.data import CACHE_ROOT, fixture_results_status, group_letter, groups_from_tournament
from src.worldcup.international_provider import (
    contextual_poisson_for_match,
    download_international_results,
    international_results_status,
    load_international_matches,
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
from src.worldcup.training import (
    KAGGLE_ROOT,
    FEATURE_STORE_ROOT,
    WALK_FORWARD_ROOT,
    WORLD_CUP_MODELS_ROOT,
    capture_walk_forward_snapshot,
    clear_active_worldcup_model,
    dataset_status,
    delete_worldcup_model,
    detect_hardware,
    download_kaggle_dataset,
    list_worldcup_models,
    prepare_training_dataset,
    predict_match_payload,
    predict_ml_outputs,
    read_model_metadata,
    set_active_worldcup_model,
    training_options as worldcup_training_options,
    train_hybrid_model,
    walk_forward_refresh_state,
    walk_forward_status,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COUNTRY_FLAGS_ROOT = PROJECT_ROOT / "storage" / "graphics" / "countries"
REPORTS_ROOT = Path("storage") / "worldcup" / "reports"
SOTA_SCORE_MODEL_SEQUENCE = [
    "independent_poisson",
    "dixon_coles_mle",
    "bivariate_poisson_mle",
    "diagonal_inflated_bivariate_poisson",
    "zero_inflated_generalized_poisson",
    "skellam_margin",
    "copula_weibull_count",
    "xg_poisson_local",
]
REPORT_TOTAL_GOAL_LINES = (0.5, 1.5, 2.5, 3.5)
REPORT_SCORE_MATRIX_GOALS = 6
REPORT_MAX_ITERATIONS = 100_000
DEFAULT_CONFIG = {
    "iterations": 5000,
    "seed": 2026,
    "use_ml_model": False,
    "ml_weight": 0.5,
    "history_weight": 1.0,
    "recency_weight": 0.35,
    "host_advantage": 45.0,
    "max_goals": 10,
    "poisson_recent_matches": 15,
    "score_model": DEFAULT_SCORE_MODEL,
    "stat_model_cache": True,
    "stat_model_refit": False,
    "bayes_draws": 500,
    "bayes_tune": 500,
    "bayes_chains": 2,
    "refresh": False,
}
LAST_SIMULATION_RESULT: Dict[str, Any] = {}
_MONTE_CARLO_CUDA_BACKEND: Tuple[str, str] | None = None


class BlendedWorldCupModel:
    def __init__(self, base_model: WorldCupModel, model_id: str = "", ml_weight: float = 0.5):
        self.base_model = base_model
        self.model_id = str(model_id or "")
        self.ml_weight = _clamp_float(ml_weight, 0.0, 1.0)

    def profile(self, team: str):
        return self.base_model.profile(team)

    def adjusted(self, rating_adjustments: Dict[str, float]):
        return BlendedWorldCupModel(
            base_model=self.base_model.adjusted(rating_adjustments),
            model_id=self.model_id,
            ml_weight=self.ml_weight,
        )

    def expected_goals(self, team1: str, team2: str):
        lambda1, lambda2, _ = self._adjusted_lambdas(team1, team2)
        return lambda1, lambda2

    def match_probabilities(self, team1: str, team2: str, max_goals: int | None = None) -> Dict[str, float]:
        ml = predict_ml_outputs(self.base_model, team1, team2, model_id=self.model_id)
        lambda1, lambda2, adjusted = self._adjusted_lambdas(team1, team2, ml=ml)
        base = poisson_probabilities_from_lambdas(
            lambda1=lambda1,
            lambda2=lambda2,
            max_goals=int(max_goals if max_goals is not None else self.base_model.max_goals),
        )
        result_ml = ml.get("result", {})
        output = dict(base)
        weight = min(max(self.ml_weight * 0.75, 0.0), 1.0)
        if result_ml:
            output["home"] = base["home"] * (1.0 - weight) + result_ml.get("H", base["home"]) * weight
            output["draw"] = base["draw"] * (1.0 - weight) + result_ml.get("D", base["draw"]) * weight
            output["away"] = base["away"] * (1.0 - weight) + result_ml.get("A", base["away"]) * weight
            total = max(output["home"] + output["draw"] + output["away"], 1e-9)
            output["home"] /= total
            output["draw"] /= total
            output["away"] /= total
        totals_ml = ml.get("over_under_ml") or ml.get("over_under_25", {})
        if totals_ml:
            for line in (0.5, 1.5, 2.5, 3.5):
                suffix = total_line_suffix(line)
                over_key = f"over{suffix}"
                under_key = f"under{suffix}"
                output[over_key] = base.get(over_key, 0.0) * (1.0 - weight) + totals_ml.get(over_key, base.get(over_key, 0.0)) * weight
                output[under_key] = base.get(under_key, 0.0) * (1.0 - weight) + totals_ml.get(under_key, base.get(under_key, 0.0)) * weight
                total_goals = max(output[over_key] + output[under_key], 1e-9)
                output[over_key] /= total_goals
                output[under_key] /= total_goals
        output["lambda1"] = adjusted["lambda1"]
        output["lambda2"] = adjusted["lambda2"]
        return output

    def sample_score(self, team1: str, team2: str, rng: np.random.Generator):
        probabilities = self.match_probabilities(team1, team2)
        outcome = rng.choice(["home", "draw", "away"], p=[probabilities["home"], probabilities["draw"], probabilities["away"]])
        lambda1, lambda2, _ = self._adjusted_lambdas(team1, team2)
        goals1 = int(rng.poisson(lambda1))
        goals2 = int(rng.poisson(lambda2))
        if outcome == "home" and goals1 <= goals2:
            goals1 = goals2 + 1
        elif outcome == "away" and goals2 <= goals1:
            goals2 = goals1 + 1
        elif outcome == "draw":
            draw_goals = int(round((goals1 + goals2) / 2.0))
            goals1 = draw_goals
            goals2 = draw_goals
        return goals1, goals2

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

    def _adjusted_lambdas(self, team1: str, team2: str, ml: Dict[str, Any] | None = None) -> Tuple[float, float, Dict[str, float]]:
        base = self.base_model.match_probabilities(team1, team2)
        ml = ml or predict_ml_outputs(self.base_model, team1, team2, model_id=self.model_id)
        result_ml = ml.get("result", {}) or {}
        totals_ml = ml.get("over_under_ml") or ml.get("over_under_25", {}) or {}
        lambda1 = float(base.get("lambda1", 1.0))
        lambda2 = float(base.get("lambda2", 1.0))
        total_goals = max(lambda1 + lambda2, 0.4)
        home_share = lambda1 / total_goals
        if totals_ml:
            delta_total = float(totals_ml.get("over25", base.get("over25", 0.0))) - float(base.get("over25", 0.0))
            total_scale = float(np.clip(math.exp(delta_total * 1.15), 0.78, 1.34))
            total_goals *= 1.0 + (total_scale - 1.0) * self.ml_weight
        if result_ml:
            base_edge = float(base.get("home", 0.0)) - float(base.get("away", 0.0))
            ml_edge = float(result_ml.get("H", base.get("home", 0.0))) - float(result_ml.get("A", base.get("away", 0.0)))
            draw_delta = float(result_ml.get("D", base.get("draw", 0.0))) - float(base.get("draw", 0.0))
            home_share = float(np.clip(home_share + (ml_edge - base_edge) * 0.38 * self.ml_weight, 0.18, 0.82))
            total_goals *= float(np.clip(1.0 - draw_delta * 0.55 * self.ml_weight, 0.82, 1.22))
        lambda1 = float(np.clip(total_goals * home_share, 0.2, 4.8))
        lambda2 = float(np.clip(total_goals * (1.0 - home_share), 0.2, 4.8))
        return lambda1, lambda2, {"lambda1": lambda1, "lambda2": lambda2}


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


def overview(refresh: bool = False) -> Dict[str, Any]:
    tournament, fixture_source = load_tournament_2026(refresh=bool(refresh))
    groups = groups_from_tournament(tournament)
    fixture_df = tournament_fixtures_dataframe(tournament)
    players_df, players_source = load_players(refresh=False)
    fixture_summary = fixture_overview_payload(fixture_df)
    standings = group_standings_payload(groups, fixture_df)
    results_status = fixture_results_status(fixture_df)
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
        "model": "Elo + modelos de marcador Monte Carlo",
        "last_simulation": LAST_SIMULATION_RESULT,
        "assets_policy": "Banderas locales/publicas y fotos publicas de SofaScore con fallback visual.",
    }


def groups(refresh: bool = False) -> Dict[str, Any]:
    tournament, source = load_tournament_2026(refresh=bool(refresh))
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
        "config": config,
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
    walk_forward = capture_walk_forward_snapshot(payload)
    return {
        "stats": enrich_lineup_payload(payload),
        "features": table_payload(pd.DataFrame(payload.get("features", [])), page=1, page_size=10),
        "players": table_payload(lineups_table(payload), page=1, page_size=40),
        "walk_forward": walk_forward,
    }


def player_features(refresh: bool = False) -> Dict[str, Any]:
    tournament, _ = load_tournament_2026(refresh=bool(refresh))
    df = player_features_dataframe(tournament)
    return {
        "features": table_payload(df, page=1, page_size=120),
        "rows": jsonable(df.to_dict(orient="records")) if not df.empty else [],
    }


def training_download(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    status = download_kaggle_dataset(force=bool(payload.get("force", False)))
    try:
        status["international_recent"] = download_international_results(force=bool(payload.get("force", False)))
    except Exception as exc:
        status["international_recent"] = {
            **international_results_status(),
            "warning": f"{exc.__class__.__name__}: {exc}",
        }
    return status


def training_prepare(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    status = prepare_training_dataset(
        force=bool(payload.get("force", False)),
        refresh_history=bool(payload.get("refresh_history", False)),
    )
    status["options"] = worldcup_training_options()
    return status


def refresh_player_snapshots(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    tournament, _ = load_tournament_2026(refresh=bool(payload.get("refresh_fixtures", False)))
    limit = int(_clamp_int(payload.get("limit", 8), 1, 48))
    lineups_result = auto_refresh_lineups(
        tournament=tournament,
        refresh_events=bool(payload.get("refresh", True)),
        limit=limit,
    )
    return {
        "lineups": lineups_result,
        "walk_forward": walk_forward_status(),
        "walk_forward_refresh": walk_forward_refresh_state(),
    }


def training_dataset() -> Dict[str, Any]:
    return dataset_status()


def training_status() -> Dict[str, Any]:
    status = dataset_status()
    status["options"] = worldcup_training_options()
    return status


def training_options() -> Dict[str, Any]:
    return worldcup_training_options()


def models_catalog() -> Dict[str, Any]:
    return list_worldcup_models()


def select_model(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    model = set_active_worldcup_model(payload.get("model_id") or payload.get("id"))
    catalog = list_worldcup_models()
    return {
        "selected": model,
        **catalog,
    }


def delete_model(model_id: str) -> Dict[str, Any]:
    return delete_worldcup_model(model_id)


def maintenance_clear(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    clear_cache = bool(payload.get("clear_cache", True))
    removed: List[str] = []
    recreated: List[str] = []
    for root in (WORLD_CUP_MODELS_ROOT, FEATURE_STORE_ROOT, LINEUPS_ROOT, PLAYER_STATS_ROOT, SOFASCORE_ROOT, WALK_FORWARD_ROOT):
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
            removed.append(str(root))
        root.mkdir(parents=True, exist_ok=True)
        recreated.append(str(root))
    clear_active_worldcup_model()
    if clear_cache and CACHE_ROOT.exists():
        for path in sorted(CACHE_ROOT.iterdir()):
            if path.resolve() == KAGGLE_ROOT.resolve():
                continue
            if path.is_file() and re.fullmatch(r"worldcup_\d{4}\.json", path.name):
                continue
            if path.is_dir() and path.resolve() == KAGGLE_ROOT.resolve():
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
            str(KAGGLE_ROOT),
            "storage/worldcup/cache/worldcup_*.json",
        ],
        "models": list_worldcup_models(),
        "training": dataset_status(),
    }


def training_train(payload: Dict[str, Any] | None = None, progress_callback=None) -> Dict[str, Any]:
    tournament, _ = load_tournament_2026(refresh=bool((payload or {}).get("refresh_fixtures", False)))
    result = train_hybrid_model(tournament=tournament, payload=payload or {}, progress_callback=progress_callback)
    result["metrics_table"] = table_payload(metrics_dataframe(result.get("metrics", {})), page=1, page_size=10)
    result["models"] = list_worldcup_models()
    return result


def predict_match(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    config = simulation_config(payload)
    tournament, fixture_source = load_tournament_2026(refresh=bool(config["refresh"]))
    model, history_source = build_model(tournament, config)
    model = apply_configured_score_model(model, tournament, config)
    result = predict_match_payload(
        tournament=tournament,
        base_model=model,
        fixture_id=payload.get("fixture_id"),
        home=payload.get("home"),
        away=payload.get("away"),
        use_ml_model=bool(config["use_ml_model"]),
        ml_weight=float(config["ml_weight"]),
        model_id=config["model_id"],
        poisson_recent_matches=int(config["poisson_recent_matches"]),
    )
    result["fixture_source"] = fixture_source
    result["history_source"] = history_source
    result["active_model"] = read_model_metadata(model_id=config["model_id"] or None)
    result["score_model"] = score_model_metadata(model)
    return result


def predict_upcoming(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    config = simulation_config(payload)
    tournament, fixture_source = load_tournament_2026(refresh=bool(config["refresh"]))
    model, history_source = build_model(tournament, config)
    model = apply_configured_score_model(model, tournament, config)

    limit = int(_clamp_int(payload.get("limit", 8), 1, 72))
    group_filter = str(payload.get("group") or "").strip()
    fixture_df = upcoming_fixture_rows(tournament, group_filter=group_filter)
    predictions = []
    rows = []
    for _, fixture in fixture_df.head(limit).iterrows():
        result = predict_match_payload(
            tournament=tournament,
            base_model=model,
            fixture_id=fixture.get("No."),
            use_ml_model=bool(config["use_ml_model"]),
            ml_weight=float(config["ml_weight"]),
            model_id=config["model_id"],
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
            "use_ml_model": config["use_ml_model"],
            "model_id": config["model_id"],
            "poisson_recent_matches": config["poisson_recent_matches"],
            "score_model": config["score_model"],
            "score_model_label": score_model_metadata(model).get("label", ""),
            "active_model": read_model_metadata(model_id=config["model_id"] or None),
        },
    }


def predict_upcoming_monte_carlo(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    config = simulation_config({**payload, "mode": "poisson_live", "use_ml_model": False})
    config["iterations"] = monte_carlo_match_iterations(payload.get("iterations", DEFAULT_CONFIG["iterations"]))
    tournament, fixture_source = load_tournament_2026(refresh=bool(config["refresh"]))
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
            "source": "Modelo de marcador contextual + simulacion por fixture",
        },
    }


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
    tournament, fixture_source = load_tournament_2026(refresh=bool(config["refresh"]))
    base_model, history_source = build_model(tournament, config)
    limit = int(_clamp_int(payload.get("limit", 8), 1, 72))
    group_filter = str(payload.get("group") or "").strip()
    fixture_df = upcoming_fixture_rows(tournament, group_filter=group_filter).head(limit).copy()
    fixture_records = [fixture for _, fixture in fixture_df.iterrows()]
    if pipeline_mode == "poisson_sota":
        fixture_reports = upcoming_sota_fixture_reports(
            tournament=tournament,
            base_model=base_model,
            fixtures=fixture_records,
            config=config,
            start_time=start_time,
            hardware=hardware,
            progress_callback=progress_callback,
        )
    else:
        fixture_reports = upcoming_default_fixture_reports(
            tournament=tournament,
            base_model=base_model,
            fixtures=fixture_records,
            config=config,
            start_time=start_time,
            hardware=hardware,
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
        if pipeline_mode == "poisson_sota":
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
    summary = {
        "pipeline_mode": pipeline_mode,
        "pipeline_label": "Poisson + SOTA" if pipeline_mode == "poisson_sota" else "Default: IA + Poisson simple",
        "requested": limit,
        "returned": len(fixture_reports),
        "group": group_filter or "Todos",
        "fixture_source": fixture_source,
        "history_source": history_source,
        "use_ml_model": bool(config["use_ml_model"]),
        "model_id": config["model_id"],
        "poisson_recent_matches": config["poisson_recent_matches"],
        "iterations": config["iterations"],
        "seed": config["seed"],
        "bayes_profile": config.get("bayes_profile", ""),
        "sota_device": config.get("sota_device", "auto"),
        "sota_calculation_mode": config.get("sota_calculation_mode", "exact") if pipeline_mode == "poisson_sota" else "",
        "sota_calculation_label": sota_calculation_summary(config) if pipeline_mode == "poisson_sota" else "",
        "monte_carlo_iterations": config["iterations"] if pipeline_mode == "poisson_sota" and config.get("sota_calculation_mode") == "monte_carlo" else 0,
        "score_models": SOTA_SCORE_MODEL_SEQUENCE if pipeline_mode == "poisson_sota" else ["default_ai_poisson"],
        "hardware": hardware,
        "warnings": list(hardware.get("warnings", [])),
        "config": config,
    }
    report = persist_upcoming_report({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "fixture_reports": fixture_reports,
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


def upcoming_default_fixture_reports(
        tournament: Dict[str, Any],
        base_model: WorldCupModel,
        fixtures: List[pd.Series],
        config: Dict[str, Any],
        start_time: float,
        hardware: Dict[str, Any],
        progress_callback=None,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    total = max(len(fixtures), 1)
    for fixture_index, fixture in enumerate(fixtures, start=1):
        emit_report_progress(
            progress_callback,
            stage="predicting",
            start_time=start_time,
            model_index=1,
            model_total=1,
            model_key="default_ai_poisson",
            fixture_index=fixture_index,
            fixture_total=total,
            hardware=hardware,
            message=f"Default IA + Poisson: {fixture.get('Equipo 1', '')} vs {fixture.get('Equipo 2', '')}",
        )
        prediction = predict_match_payload(
            tournament=tournament,
            base_model=base_model,
            fixture_id=fixture.get("No."),
            use_ml_model=bool(config["use_ml_model"]),
            ml_weight=float(config["ml_weight"]),
            model_id=config["model_id"],
            poisson_recent_matches=int(config["poisson_recent_matches"]),
        )
        reports.append({
            "fixture": report_fixture_payload(prediction.get("fixture", {})),
            "contextual_poisson": prediction.get("contextual_poisson", {}),
            "models": [default_prediction_model_report(prediction, config)],
        })
    return reports


def upcoming_sota_fixture_reports(
        tournament: Dict[str, Any],
        base_model: WorldCupModel,
        fixtures: List[pd.Series],
        config: Dict[str, Any],
        start_time: float,
        hardware: Dict[str, Any],
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
    history_df, _ = load_historical_matches(refresh=bool(config.get("refresh", False)))
    model_total = len(SOTA_SCORE_MODEL_SEQUENCE)
    fixture_total = max(len(fixtures), 1)
    for model_index, model_key in enumerate(SOTA_SCORE_MODEL_SEQUENCE, start=1):
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
        try:
            if model_key == DEFAULT_SCORE_MODEL:
                score_model = base_model
                metadata = score_model_metadata(score_model)
            else:
                score_model = build_score_model(
                    base_model,
                    history_df=history_df,
                    teams=team_names,
                    config={**config, "score_model": model_key, "use_ml_model": False},
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
            model_report["heatmap"] = score_distribution.get("heatmap", {})
            fixture_reports[fixture_index - 1]["models"].append(model_report)
    return fixture_reports


def normalize_report_pipeline_mode(value: Any) -> str:
    mode = str(value or "default_ai_poisson").strip().lower().replace("-", "_")
    return "poisson_sota" if mode == "poisson_sota" else "default_ai_poisson"


def normalize_sota_calculation_mode(value: Any) -> str:
    mode = str(value or "exact").strip().lower().replace("-", "_")
    return "monte_carlo" if mode in {"monte_carlo", "montecarlo", "mc", "simulation"} else "exact"


def sota_calculation_summary(config: Dict[str, Any]) -> str:
    if config.get("sota_calculation_mode") == "monte_carlo":
        return f"SOTA Monte Carlo: N={int(config.get('iterations') or DEFAULT_CONFIG['iterations']):,}"
    return "SOTA rápido: matriz exacta, sin simulación"


def report_pipeline_config(payload: Dict[str, Any], pipeline_mode: str) -> Dict[str, Any]:
    config = simulation_config(payload)
    config["pipeline_mode"] = pipeline_mode
    config["bayes_profile"] = str(payload.get("bayes_profile") or "deep").strip().lower()
    config["sota_device"] = str(payload.get("sota_device") or "auto").strip().lower()
    config["sota_calculation_mode"] = normalize_sota_calculation_mode(payload.get("sota_calculation_mode"))
    if config["sota_device"] not in {"auto", "cpu", "cuda"}:
        config["sota_device"] = "auto"
    if pipeline_mode == "poisson_sota":
        config["use_ml_model"] = False
        config["score_model"] = DEFAULT_SCORE_MODEL
        if config["bayes_profile"] == "deep":
            config["bayes_draws"] = 2000
            config["bayes_tune"] = 2000
            config["bayes_chains"] = 4
            config["stat_model_cache"] = True
            config["stat_model_refit"] = False
    return config


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
    if pipeline_mode == "poisson_sota":
        cuda_reason = detected.get("cuda_error") or detected.get("cuda_warning") or "sin dispositivos"
        if calculation_mode == "monte_carlo":
            if requested == "cpu":
                warnings.append("Monte Carlo SOTA configurado en CPU por solicitud explicita.")
            elif detected.get("cuda_available"):
                backend_name, backend_warning = monte_carlo_cuda_backend()
                if backend_name:
                    actual_device = "cuda"
                    backend_supports_cuda = True
                    monte_carlo_backend = backend_name
                    warnings.append(f"CUDA activa para Monte Carlo SOTA via {backend_name}; el ajuste estadistico previo sigue en CPU.")
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
        elif requested == "cuda" and detected.get("cuda_available"):
            actual_device = "cuda"
            warnings.append("CUDA fue solicitada y detectada; SOTA rapido usa matriz exacta CPU-bound. Cambia Calculo a Monte Carlo para muestreo GPU.")
        elif requested == "cuda":
            device_error = f"CUDA fue solicitada explicitamente, pero no se detecto GPU ({cuda_reason}); SOTA corre en CPU."
            warnings.append(device_error)
        elif requested == "auto" and detected.get("cuda_available"):
            warnings.append("CUDA detectada; SOTA rapido usa matriz exacta CPU-bound. Cambia Calculo a Monte Carlo para muestreo GPU.")
        elif requested == "auto" and not detected.get("cuda_available"):
            warnings.append(f"CUDA no disponible ({cuda_reason}); SOTA corre en CPU.")
    return {
        **detected,
        "requested_device": requested,
        "actual_device": actual_device,
        "backend_supports_cuda": backend_supports_cuda,
        "monte_carlo_backend": monte_carlo_backend,
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
    }


def default_prediction_model_report(prediction: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    probabilities = prediction.get("probabilities", {})
    metadata = {
        "key": "default_ai_poisson",
        "label": "IA + Poisson simple" if config.get("use_ml_model") else "Poisson independiente",
        "available": True,
        "warnings": prediction.get("notes", []),
    }
    return score_prediction_model_report(
        model_key="default_ai_poisson",
        metadata=metadata,
        probabilities={
            **probabilities,
            "lambda1": (prediction.get("expected_goals") or {}).get("home", 0.0),
            "lambda2": (prediction.get("expected_goals") or {}).get("away", 0.0),
            "modal_g1": parse_score_part(prediction.get("modal_score", ""), 0),
            "modal_g2": parse_score_part(prediction.get("modal_score", ""), 1),
        },
        fixture=pd.Series(prediction.get("fixture", {})),
        config=config,
        already_percent=True,
    )


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
        "consensus_eligible": bool(available or key in {DEFAULT_SCORE_MODEL, "default_ai_poisson"}),
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
        "params": metadata.get("params", {}),
        "source": "ML + Poisson" if key == "default_ai_poisson" and config.get("use_ml_model") else "Poisson/SOTA",
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
    for model in model_reports:
        distribution = model.get("score_distribution") or {}
        matrix = distribution.get("score_matrix")
        if not matrix:
            continue
        array = normalize_score_grid_array(np.asarray(matrix, dtype=float) / 100.0)
        matrices.append(array)
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
        "matrix_source": "consensus_average",
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
    return {
        "model_count": len(model_reports),
        "outcomes": {key: numeric_summary(values) for key, values in probabilities.items()},
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
    report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (REPORTS_ROOT / "latest.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def score_model_display_label(key: Any) -> str:
    normalized = normalize_score_model_key(key)
    for option in score_model_options():
        if option.get("key") == normalized:
            return str(option.get("label") or normalized)
    if str(key) == "default_ai_poisson":
        return "IA + Poisson simple"
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
    model, history_source = build_model(tournament, config)
    active_model = read_model_metadata(model_id=config["model_id"] or None)
    hybrid_layers = ["Poisson base"]
    if config["include_confirmed_results"]:
        hybrid_layers.append("resultados confirmados")
    if config["mode"] == "poisson_live":
        model = RecentPoissonWorldCupModel(model, recent_match_limit=int(config["poisson_recent_matches"]))
        hybrid_layers.append(f"Poisson ultimos {config['poisson_recent_matches']}")
    model = apply_configured_score_model(model, tournament, config)
    score_metadata = score_model_metadata(model)
    if score_metadata.get("key") != DEFAULT_SCORE_MODEL:
        hybrid_layers.append(str(score_metadata.get("label") or score_metadata.get("key")))
    if config["use_ml_model"] and active_model.get("trained"):
        model = BlendedWorldCupModel(model, model_id=config["model_id"], ml_weight=float(config["ml_weight"]))
        hybrid_layers.extend(["ML blend 1X2", "ML ajuste lambdas"])
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
            "config": config,
            "mode": config["mode"],
            "fixture_source": fixture_source,
            "history_source": history_source,
            "use_ml_model": config["use_ml_model"],
            "model_id": config["model_id"],
            "active_model": active_model,
            "score_model": score_metadata,
            "hybrid_layers": hybrid_layers,
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
                "Modelo ML entrenado/evaluado con partidos internacionales no Mundial y sin partidos 2026.",
                "Los resultados 2026 confirmados solo fijan standings de simulacion live; no entrenan ni calibran el modelo.",
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
        "current_trial": current if stage == "tuning" else "",
        "total_trials": total if stage == "tuning" else "",
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
                "name": "Fine-tuning",
                "detail": "Ajusta peso historico, recencia, ventaja local, limite de goles y mezcla opcional con ML.",
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
                "detail": "Combina Elo/Poisson con el modelo Kaggle si esta entrenado y reporta 1X2 junto con U/O 0.5, 1.5, 2.5 y 3.5 para los proximos N partidos.",
            },
        ],
        "sources": [
            "openfootball/worldcup.json",
            "Football-Data WorldCup2026.xlsx para odds 1X2 historicas y clasificatorios",
            "storage/worldcup/market/manual_odds.csv opcional para odds actuales/O-U 2.5",
            "Kaggle: harrachimustapha/fifa-world-cup-team-dataset",
            "storage/worldcup/cache/*.json",
            "Wikipedia squads opcional para jugadores",
        ],
    }


def build_model(tournament: Dict[str, Any], config: Dict[str, Any]) -> Tuple[WorldCupModel, str]:
    group_map = groups_from_tournament(tournament)
    team_names = [team for group_teams in group_map.values() for team in group_teams]
    history_df, history_source = load_historical_matches(refresh=bool(config.get("refresh", False)))
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
    history_df, _ = load_historical_matches(refresh=bool(config.get("refresh", False)))
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
    mode = str(payload.get("mode") or "hybrid").strip().lower()
    if mode not in {"hybrid", "poisson_live"}:
        mode = "hybrid"
    use_ml_model = bool(payload.get("use_ml_model", DEFAULT_CONFIG["use_ml_model"]))
    if mode == "poisson_live":
        use_ml_model = False
    return {
        "mode": mode,
        "iterations": int(_clamp_int(payload.get("iterations", DEFAULT_CONFIG["iterations"]), 100, REPORT_MAX_ITERATIONS)),
        "seed": int(payload.get("seed") if payload.get("seed") is not None else DEFAULT_CONFIG["seed"]),
        "use_ml_model": use_ml_model,
        "ml_weight": _clamp_float(payload.get("ml_weight", DEFAULT_CONFIG["ml_weight"]), 0.0, 1.0),
        "history_weight": _clamp_float(payload.get("history_weight", DEFAULT_CONFIG["history_weight"]), 0.2, 2.0),
        "recency_weight": _clamp_float(payload.get("recency_weight", DEFAULT_CONFIG["recency_weight"]), 0.0, 1.0),
        "host_advantage": _clamp_float(payload.get("host_advantage", DEFAULT_CONFIG["host_advantage"]), 0.0, 120.0),
        "max_goals": int(_clamp_int(payload.get("max_goals", DEFAULT_CONFIG["max_goals"]), 6, 14)),
        "poisson_recent_matches": int(_clamp_int(payload.get("poisson_recent_matches", DEFAULT_CONFIG["poisson_recent_matches"]), 3, 50)),
        "score_model": normalize_score_model_key(payload.get("score_model", DEFAULT_CONFIG["score_model"])),
        "stat_model_cache": bool(payload.get("stat_model_cache", DEFAULT_CONFIG["stat_model_cache"])),
        "stat_model_refit": bool(payload.get("stat_model_refit", DEFAULT_CONFIG["stat_model_refit"])),
        "bayes_draws": int(_clamp_int(payload.get("bayes_draws", DEFAULT_CONFIG["bayes_draws"]), 100, 10000)),
        "bayes_tune": int(_clamp_int(payload.get("bayes_tune", DEFAULT_CONFIG["bayes_tune"]), 100, 10000)),
        "bayes_chains": int(_clamp_int(payload.get("bayes_chains", DEFAULT_CONFIG["bayes_chains"]), 1, 8)),
        "refresh": bool(payload.get("refresh", DEFAULT_CONFIG["refresh"])),
        "model_id": str(payload.get("model_id") or "").strip(),
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


def fixture_kickoff_datetime(date_value: Any, time_value: Any, require_time: bool = False) -> datetime | None:
    date_text = str(date_value or "").strip()
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
    scheduled["_date"] = pd.to_datetime(scheduled["Fecha"], errors="coerce")
    scheduled["_kickoff_iso"] = kickoff_iso_values
    scheduled["_kickoff"] = pd.to_datetime(kickoff_values, utc=True, errors="coerce")
    scheduled["_has_kickoff_time"] = has_kickoff_time
    date_sort = pd.to_datetime(scheduled["Fecha"], utc=True, errors="coerce")
    scheduled["_sort_time"] = scheduled["_kickoff"].where(scheduled["_kickoff"].notna(), date_sort)
    return scheduled


def future_fixture_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    now = pd.Timestamp(_now_utc())
    if now.tzinfo is None:
        now = now.tz_localize(timezone.utc)
    else:
        now = now.tz_convert(timezone.utc)
    today = pd.Timestamp(now.date())
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
