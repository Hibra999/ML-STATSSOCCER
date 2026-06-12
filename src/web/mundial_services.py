from __future__ import annotations

import math
import re
import shutil
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
from src.worldcup.model import TOTAL_GOAL_LINES, total_line_suffix
from src.worldcup.data import CACHE_ROOT, group_letter, groups_from_tournament
from src.worldcup.international_provider import download_international_results, international_results_status
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
    WALK_FORWARD_ROOT,
    WORLD_CUP_MODELS_ROOT,
    capture_walk_forward_snapshot,
    clear_active_worldcup_model,
    dataset_status,
    delete_worldcup_model,
    download_kaggle_dataset,
    list_worldcup_models,
    prepare_training_dataset,
    predict_match_payload,
    predict_ml_outputs,
    read_model_metadata,
    set_active_worldcup_model,
    training_options as worldcup_training_options,
    train_hybrid_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COUNTRY_FLAGS_ROOT = PROJECT_ROOT / "storage" / "graphics" / "countries"
DEFAULT_CONFIG = {
    "iterations": 5000,
    "seed": 2026,
    "use_ml_model": False,
    "ml_weight": 0.5,
    "history_weight": 1.0,
    "recency_weight": 0.35,
    "host_advantage": 45.0,
    "max_goals": 10,
    "refresh": False,
}


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
    return {
        "name": tournament.get("name", "World Cup 2026"),
        "teams": sum(len(teams) for teams in groups.values()),
        "groups": len(groups),
        "fixtures": int(fixture_df.shape[0]),
        "group_fixtures": int((fixture_df["Grupo"] != "").sum()) if not fixture_df.empty else 0,
        "players": int(players_df.shape[0]),
        "fixture_source": fixture_source,
        "players_source": players_source,
        "opener": fixture_summary["opener"],
        "featured_matches": fixture_summary["featured_matches"],
        "highlight": fixture_summary["highlight"],
        "next_matches": fixture_summary["next_matches"],
        "countdown_target": fixture_summary["countdown_target"],
        "countdown_state": fixture_summary["countdown_state"],
        "group_standings": standings,
        "default_config": DEFAULT_CONFIG,
        "model": "Elo + Poisson Monte Carlo",
        "assets_policy": "Banderas locales/publicas y fotos publicas de SofaScore con fallback visual.",
    }


def groups(refresh: bool = False) -> Dict[str, Any]:
    tournament, source = load_tournament_2026(refresh=bool(refresh))
    group_map = groups_from_tournament(tournament)
    items = []
    for group_name, team_names in group_map.items():
        items.append({
            "name": group_name,
            "letter": group_letter(group_name),
            "standings": group_standing_rows(group_name, team_names, tournament_fixtures_dataframe(tournament)),
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
    }


def fixtures(refresh: bool = False) -> Dict[str, Any]:
    tournament, source = load_tournament_2026(refresh=bool(refresh))
    df = tournament_fixtures_dataframe(tournament)
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
            "home": team_asset(home),
            "away": team_asset(away),
            "label": f"{home} vs {away}",
        })
    return {
        "fixtures": rows,
        "table": table_payload(df, page=1, page_size=150),
        "source": source,
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
    for root in (WORLD_CUP_MODELS_ROOT, LINEUPS_ROOT, PLAYER_STATS_ROOT, SOFASCORE_ROOT, WALK_FORWARD_ROOT):
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
    result = predict_match_payload(
        tournament=tournament,
        base_model=model,
        fixture_id=payload.get("fixture_id"),
        home=payload.get("home"),
        away=payload.get("away"),
        use_ml_model=bool(config["use_ml_model"]),
        ml_weight=float(config["ml_weight"]),
        model_id=config["model_id"],
    )
    result["fixture_source"] = fixture_source
    result["history_source"] = history_source
    result["active_model"] = read_model_metadata(model_id=config["model_id"] or None)
    return result


def predict_upcoming(payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    payload = payload or {}
    config = simulation_config(payload)
    tournament, fixture_source = load_tournament_2026(refresh=bool(config["refresh"]))
    model, history_source = build_model(tournament, config)

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
            "active_model": read_model_metadata(model_id=config["model_id"] or None),
        },
    }


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
        "Poisson 15": "Si" if contextual.get("available") else "Base" if contextual.get("matrix_available") else "No",
        "Lambda 15 Local": contextual.get("context_lambda_home", ""),
        "Lambda 15 Visita": contextual.get("context_lambda_away", ""),
        "Top score 15": context_top.get("score", ""),
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
    config = simulation_config(payload)
    emit_job_progress(progress_callback, "preparing", 0, 100, "Preparando Monte Carlo")
    tournament, fixture_source = load_tournament_2026(refresh=bool(config["refresh"]))
    model, history_source = build_model(tournament, config)
    active_model = read_model_metadata(model_id=config["model_id"] or None)
    hybrid_layers = ["Poisson base"]
    if config["use_ml_model"] and active_model.get("trained"):
        model = BlendedWorldCupModel(model, model_id=config["model_id"], ml_weight=float(config["ml_weight"]))
        hybrid_layers.extend(["ML blend 1X2", "ML ajuste lambdas"])
    result = simulate_worldcup(
        tournament=tournament,
        model=model,
        iterations=int(config["iterations"]),
        seed=int(config["seed"]),
        progress_callback=progress_callback,
    )
    emit_job_progress(progress_callback, "rendering", 100, 100, "Preparando resultados")
    output = {
        "summary": {
            "model": "Elo + Poisson Monte Carlo",
            "config": config,
            "fixture_source": fixture_source,
            "history_source": history_source,
            "use_ml_model": config["use_ml_model"],
            "model_id": config["model_id"],
            "active_model": active_model,
            "hybrid_layers": hybrid_layers,
            "anti_leakage": [
                "Historico filtrado antes del 2026-06-11.",
                "Modelo ML entrenado/evaluado con partidos internacionales no Mundial y sin partidos 2026.",
                "No se usan resultados del Mundial 2026 para entrenar ni calibrar.",
            ],
        },
        "advancement": table_payload(result["advancement"], page=1, page_size=80),
        "matches": table_payload(result["matches"], page=1, page_size=120),
        "procedure": procedure()["steps"],
    }
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


def simulation_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = payload or {}
    return {
        "iterations": int(_clamp_int(payload.get("iterations", DEFAULT_CONFIG["iterations"]), 100, 20000)),
        "seed": int(payload.get("seed") if payload.get("seed") is not None else DEFAULT_CONFIG["seed"]),
        "use_ml_model": bool(payload.get("use_ml_model", DEFAULT_CONFIG["use_ml_model"])),
        "ml_weight": _clamp_float(payload.get("ml_weight", DEFAULT_CONFIG["ml_weight"]), 0.0, 1.0),
        "history_weight": _clamp_float(payload.get("history_weight", DEFAULT_CONFIG["history_weight"]), 0.2, 2.0),
        "recency_weight": _clamp_float(payload.get("recency_weight", DEFAULT_CONFIG["recency_weight"]), 0.0, 1.0),
        "host_advantage": _clamp_float(payload.get("host_advantage", DEFAULT_CONFIG["host_advantage"]), 0.0, 120.0),
        "max_goals": int(_clamp_int(payload.get("max_goals", DEFAULT_CONFIG["max_goals"]), 6, 14)),
        "refresh": bool(payload.get("refresh", DEFAULT_CONFIG["refresh"])),
        "model_id": str(payload.get("model_id") or "").strip(),
    }


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
            "rows": group_standing_rows(group_name, team_names, fixture_df),
        }
        for group_name, team_names in group_map.items()
    ]


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
    return df[~finished & (future_by_time | future_by_date)].sort_values(["_sort_time", "No."], kind="stable").copy()


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
