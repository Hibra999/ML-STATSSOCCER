from __future__ import annotations

import math
import re
from datetime import date, datetime
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
from src.worldcup.data import group_letter, groups_from_tournament
from src.worldcup.lanus_provider import (
    auto_refresh_lineups,
    autodetect_fixture_event,
    link_fixture_lineup,
    lineup_payload_for_fixture,
    lineup_rating_adjustments,
    lineups_summary,
    player_feature_rating_adjustments,
    player_features_dataframe,
    player_stats_payload_for_fixture,
    sofa_player_photo_url,
)


COUNTRY_FLAGS_ROOT = Path("storage") / "graphics" / "countries"
DEFAULT_CONFIG = {
    "iterations": 5000,
    "seed": 2026,
    "use_lineups": False,
    "use_player_features": False,
    "lineup_weight": 1.0,
    "player_feature_weight": 1.0,
    "history_weight": 1.0,
    "recency_weight": 0.35,
    "host_advantage": 45.0,
    "max_goals": 10,
    "refresh": False,
}
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
    opener = _opener_payload(fixture_df)
    return {
        "name": tournament.get("name", "World Cup 2026"),
        "teams": sum(len(teams) for teams in groups.values()),
        "groups": len(groups),
        "fixtures": int(fixture_df.shape[0]),
        "group_fixtures": int((fixture_df["Grupo"] != "").sum()) if not fixture_df.empty else 0,
        "players": int(players_df.shape[0]),
        "fixture_source": fixture_source,
        "players_source": players_source,
        "opener": opener,
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
        lineup = lineup_response(lineup_payload_for_fixture(tournament=tournament, fixture_id=fixture_id, refresh=True, match_url=event["match_url"]))
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


def lineup_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    enriched = enrich_lineup_payload(payload)
    return {
        "lineup": enriched,
        "players": table_payload(lineups_table(enriched), page=1, page_size=40),
    }


def simulate(payload: Dict[str, Any]) -> Dict[str, Any]:
    config = simulation_config(payload)
    tournament, fixture_source = load_tournament_2026(refresh=bool(config["refresh"]))
    model, history_source = build_model(tournament, config)
    lineup_notes: List[str] = []
    if config["use_lineups"]:
        adjustments, lineup_notes = lineup_rating_adjustments(tournament, weight=config["lineup_weight"])
        if adjustments:
            model = model.adjusted(adjustments)
    feature_notes: List[str] = []
    if config["use_player_features"]:
        adjustments, feature_notes = player_feature_rating_adjustments(tournament, weight=config["player_feature_weight"])
        if adjustments:
            model = model.adjusted(adjustments)
    result = simulate_worldcup(
        tournament=tournament,
        model=model,
        iterations=int(config["iterations"]),
        seed=int(config["seed"]),
    )
    return {
        "summary": {
            "model": "Elo + Poisson Monte Carlo",
            "config": config,
            "fixture_source": fixture_source,
            "history_source": history_source,
            "use_lineups": config["use_lineups"],
            "use_player_features": config["use_player_features"],
            "lineup_notes": lineup_notes,
            "player_feature_notes": feature_notes,
            "anti_leakage": [
                "Historico filtrado antes del 2026-06-11.",
                "Alineaciones ignoradas si fueron obtenidas despues de la fecha del partido.",
                "Features del XI ignoradas si fueron obtenidas despues de la fecha del partido.",
                "No se usan resultados del Mundial 2026 para entrenar ni calibrar.",
            ],
        },
        "advancement": table_payload(result["advancement"], page=1, page_size=80),
        "matches": table_payload(result["matches"], page=1, page_size=120),
        "procedure": procedure()["steps"],
    }


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
                "detail": "Ajusta peso historico, recencia, ventaja local, limite de goles y peso opcional de alineaciones.",
            },
            {
                "name": "11 iniciales",
                "detail": "Detecta automaticamente eventos SofaScore por fecha/equipos, extrae titulares, formacion, ratings y stats disponibles.",
            },
            {
                "name": "Features del XI",
                "detail": "Calcula rating promedio, dispersion, min/max y promedios por linea; solo impactan la prediccion si son pre-partido.",
            },
            {
                "name": "Monte Carlo",
                "detail": "Simula fase de grupos, mejores terceros y bracket completo para estimar avance, final y campeon.",
            },
        ],
        "sources": [
            "openfootball/worldcup.json",
            "storage/worldcup/cache/*.json",
            "LanusStats/SofaScore opcional para alineaciones",
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
        "use_lineups": bool(payload.get("use_lineups", DEFAULT_CONFIG["use_lineups"])),
        "use_player_features": bool(payload.get("use_player_features", DEFAULT_CONFIG["use_player_features"])),
        "lineup_weight": _clamp_float(payload.get("lineup_weight", DEFAULT_CONFIG["lineup_weight"]), 0.0, 2.0),
        "player_feature_weight": _clamp_float(payload.get("player_feature_weight", DEFAULT_CONFIG["player_feature_weight"]), 0.0, 2.0),
        "history_weight": _clamp_float(payload.get("history_weight", DEFAULT_CONFIG["history_weight"]), 0.2, 2.0),
        "recency_weight": _clamp_float(payload.get("recency_weight", DEFAULT_CONFIG["recency_weight"]), 0.0, 1.0),
        "host_advantage": _clamp_float(payload.get("host_advantage", DEFAULT_CONFIG["host_advantage"]), 0.0, 120.0),
        "max_goals": int(_clamp_int(payload.get("max_goals", DEFAULT_CONFIG["max_goals"]), 6, 14)),
        "refresh": bool(payload.get("refresh", DEFAULT_CONFIG["refresh"])),
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


def _opener_payload(fixture_df: pd.DataFrame) -> Dict[str, Any]:
    if not fixture_df.empty:
        opener = fixture_df.iloc[0].to_dict()
        home = str(opener.get("Equipo 1", "Mexico"))
        away = str(opener.get("Equipo 2", "South Africa"))
        return {
            "date": opener.get("Fecha", "2026-06-11"),
            "time": opener.get("Hora", ""),
            "match": f"{home} vs {away}",
            "home": team_asset(home),
            "away": team_asset(away),
            "venue": opener.get("Sede", ""),
        }
    return {
        "date": "2026-06-11",
        "time": "",
        "match": "Mexico vs South Africa",
        "home": team_asset("Mexico"),
        "away": team_asset("South Africa"),
        "venue": "",
    }


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
