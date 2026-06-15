from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from unicodedata import normalize as unicode_normalize

import pandas as pd

from src.worldcup.data import clean_team_name, team_name_similarity, tournament_fixtures_dataframe
from src.worldcup.fotmob_provider import fetch_best_fotmob_event, fetch_fotmob_lineup


LINEUPS_ROOT = Path("storage") / "worldcup" / "lineups"
LINEUP_LINKS_FILE = LINEUPS_ROOT / "links.json"
SOFASCORE_ROOT = Path("storage") / "worldcup" / "sofascore"
SOFASCORE_EVENTS_FILE = SOFASCORE_ROOT / "events.json"
PLAYER_STATS_ROOT = Path("storage") / "worldcup" / "player_stats"
LINEUP_STATUSES = {
    "official": "Oficial",
    "probable": "Probable",
    "last_xi": "Ultimo XI",
    "pending": "Pendiente",
}
PLAYER_FEATURE_COLUMNS = [
    "Fixture",
    "Fecha",
    "Grupo",
    "Equipo",
    "Rival",
    "Prediction safe",
    "Titulares",
    "Stats conocidos",
    "XI rating prom",
    "XI rating std",
    "XI rating min",
    "XI rating max",
    "POR rating",
    "DEF rating",
    "MED rating",
    "ATA rating",
    "Formacion",
    "Fuente",
]


class LineupProviderError(RuntimeError):
    pass


def lineup_payload_for_fixture(
        tournament: Dict[str, Any],
        fixture_id: Any,
        refresh: bool = False,
        match_url: Optional[str] = None,
) -> Dict[str, Any]:
    fixture = find_fixture(tournament=tournament, fixture_id=fixture_id)
    fixture_key = str(fixture["No."])
    cache_path = lineup_cache_path(fixture_key)
    links = read_lineup_links()
    match_url = normalize_match_url(match_url or links.get(fixture_key, ""))

    if not refresh and cache_path.exists():
        return read_lineup_cache(cache_path)

    if match_url:
        try:
            payload = fetch_lanus_lineup(fixture=fixture, fixture_key=fixture_key, match_url=match_url)
            write_lineup_cache(cache_path, payload)
            links[fixture_key] = match_url
            write_lineup_links(links)
            return payload
        except Exception as exc:
            cached = read_lineup_cache(cache_path) if cache_path.exists() else None
            if cached is not None:
                cached["error"] = clean_error(exc)
                return cached
            return pending_lineup_payload(fixture, fixture_key, match_url=match_url, error=clean_error(exc))

    return pending_lineup_payload(fixture, fixture_key, match_url="", error="")


def autodetect_fixture_event(tournament: Dict[str, Any], fixture_id: Any, refresh: bool = False) -> Dict[str, Any]:
    fixture = find_fixture(tournament=tournament, fixture_id=fixture_id)
    fixture_key = str(fixture["No."])
    event_cache = read_event_cache()
    if fixture_key in event_cache and not refresh:
        return event_cache[fixture_key]

    attempts: List[Dict[str, str]] = []
    event = fetch_best_event_with_fallbacks(fixture, attempts)
    event["source_attempts"] = attempts
    if not event.get("event_id"):
        event_cache[fixture_key] = event
        write_event_cache(event_cache)
        return event

    event_cache[fixture_key] = event
    write_event_cache(event_cache)
    if "sofascore.com" in str(event.get("match_url", "")):
        links = read_lineup_links()
        links[fixture_key] = event["match_url"]
        write_lineup_links(links)
    return event


def fetch_best_event_with_fallbacks(fixture: pd.Series, attempts: List[Dict[str, str]]) -> Dict[str, Any]:
    fixture_key = str(fixture.get("No.", ""))
    date = str(fixture.get("Fecha", ""))[:10]
    home = clean_team_name(fixture.get("Equipo 1"))
    away = clean_team_name(fixture.get("Equipo 2"))

    for provider, fetcher in (
            ("FotMob", lambda: fetch_best_fotmob_event(
                fixture=fixture,
                similarity_fn=team_similarity,
                event_builder=fotmob_event_payload,
                pending_builder=lambda fixture_key, date, home, away, error: pending_event_payload(
                    fixture_key, date, home, away, error, source="FotMob", provider="FotMob",
                ),
            )),
            ("SofaScore", lambda: fetch_best_sofascore_event(fixture)),
    ):
        try:
            event = fetcher()
            attempts.append({"provider": provider, "status": event.get("status", ""), "error": event.get("error", "")})
            if event.get("event_id"):
                return event
        except Exception as exc:
            attempts.append({"provider": provider, "status": "Error", "error": clean_error(exc)})

    error = "; ".join(f"{attempt['provider']}: {attempt['error'] or attempt['status']}" for attempt in attempts if attempt.get("error") or attempt.get("status"))
    return pending_event_payload(fixture_key, date, home, away, error or "No se detecto evento en FotMob ni SofaScore.", source="multi-provider", provider="none")


def lineup_payload_from_detected_event(
        tournament: Dict[str, Any],
        fixture_id: Any,
        event: Dict[str, Any],
        refresh: bool = True,
) -> Dict[str, Any]:
    fixture = find_fixture(tournament=tournament, fixture_id=fixture_id)
    fixture_key = str(fixture["No."])
    provider = str(event.get("provider") or event.get("source") or "").lower()
    try:
        if provider.startswith("fotmob"):
            payload = fetch_fotmob_lineup(fixture=fixture, fixture_key=fixture_key, event=event)
            write_lineup_cache(lineup_cache_path(fixture_key), payload)
            write_player_stats_cache(player_stats_cache_path(fixture_key), player_stats_payload(payload))
            return payload
        if event.get("match_url"):
            payload = lineup_payload_for_fixture(tournament=tournament, fixture_id=fixture_key, refresh=refresh, match_url=event.get("match_url"))
            write_player_stats_cache(player_stats_cache_path(fixture_key), player_stats_payload(payload))
            return payload
    except Exception as exc:
        cached = read_lineup_cache(lineup_cache_path(fixture_key)) if lineup_cache_path(fixture_key).exists() else None
        if cached is not None:
            cached["error"] = clean_error(exc)
            return cached
        return pending_lineup_payload(fixture, fixture_key, match_url=event.get("match_url", ""), error=clean_error(exc))
    return pending_lineup_payload(fixture, fixture_key, match_url=event.get("match_url", ""), error=event.get("error", ""))


def auto_refresh_lineups(tournament: Dict[str, Any], refresh_events: bool = False, limit: int = 0) -> Dict[str, Any]:
    rows = []
    refreshed = 0
    failures = 0
    for _, fixture in tournament_fixtures_dataframe(tournament).iterrows():
        if not str(fixture.get("Grupo", "")):
            continue
        if limit and len(rows) >= int(limit):
            break
        fixture_key = str(fixture["No."])
        event = autodetect_fixture_event(tournament, fixture_key, refresh=refresh_events)
        if not event.get("event_id"):
            failures += 1
            rows.append({**event, "fixture_id": fixture_key, "status": "No detectado"})
            continue
        lineup = lineup_payload_from_detected_event(tournament, fixture_key, event, refresh=True)
        refreshed += int(lineup.get("starters_home", 0) == 11 and lineup.get("starters_away", 0) == 11)
        rows.append({
            **event,
            "fixture_id": fixture_key,
            "status": lineup.get("status", ""),
            "starters_home": lineup.get("starters_home", 0),
            "starters_away": lineup.get("starters_away", 0),
        })
    return {
        "attempted": len(rows),
        "refreshed": refreshed,
        "failures": failures,
        "rows": rows,
    }


def lineups_summary(tournament: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    links = read_lineup_links()
    event_cache = read_event_cache()
    for _, fixture in tournament_fixtures_dataframe(tournament).iterrows():
        fixture_key = str(fixture["No."])
        if not str(fixture.get("Grupo", "")):
            continue
        cache_path = lineup_cache_path(fixture_key)
        payload = read_lineup_cache(cache_path) if cache_path.exists() else None
        status = payload.get("status") if payload else LINEUP_STATUSES["pending"]
        rows.append({
            "Fixture": fixture_key,
            "Fecha": fixture.get("Fecha", ""),
            "Grupo": fixture.get("Grupo", ""),
            "Equipo 1": fixture.get("Equipo 1", ""),
            "Equipo 2": fixture.get("Equipo 2", ""),
            "Estado": status,
            "Local 11": payload.get("starters_home", 0) if payload else 0,
            "Visitante 11": payload.get("starters_away", 0) if payload else 0,
            "URL SofaScore": links.get(fixture_key, payload.get("match_url", "") if payload else ""),
            "SofaScore ID": event_cache.get(fixture_key, {}).get("event_id", ""),
            "Auto match": event_cache.get(fixture_key, {}).get("confidence", ""),
        })
    return pd.DataFrame(rows)


def link_fixture_lineup(tournament: Dict[str, Any], fixture_id: Any, match_url: str, refresh: bool = True) -> Dict[str, Any]:
    fixture = find_fixture(tournament=tournament, fixture_id=fixture_id)
    fixture_key = str(fixture["No."])
    url = normalize_match_url(match_url)
    if not url:
        raise LineupProviderError("La URL de SofaScore es obligatoria.")
    links = read_lineup_links()
    links[fixture_key] = url
    write_lineup_links(links)
    return lineup_payload_for_fixture(tournament=tournament, fixture_id=fixture_key, refresh=refresh, match_url=url)


def lineups_table(payload: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for player in payload.get("players", []):
        stats = player.get("stats", {}) if isinstance(player.get("stats"), dict) else {}
        rows.append({
            "Equipo": player.get("team", ""),
            "Lado": player.get("side", ""),
            "Jugador": player.get("name", ""),
            "Posicion": player.get("position", ""),
            "Dorsal": player.get("shirt_number", ""),
            "Titular": "Si" if player.get("starter") else "No",
            "Capitan": "Si" if player.get("captain") else "",
            "SofaScore ID": player.get("id", ""),
            "Rating": player.get("rating", ""),
            "Minutos": stats.get("minutesPlayed", stats.get("minutes", "")),
            "Goles": stats.get("goals", ""),
            "Asistencias": stats.get("goalAssist", stats.get("assists", "")),
            "Tiros": stats.get("totalShots", ""),
            "Pases %": stats.get("accuratePassesPercentage", ""),
            "Estado": payload.get("status", ""),
            "Fuente": payload.get("source", ""),
        })
    return pd.DataFrame(rows, columns=[
        "Equipo", "Lado", "Jugador", "Posicion", "Dorsal", "Titular", "Capitan", "SofaScore ID", "Rating",
        "Minutos", "Goles", "Asistencias", "Tiros", "Pases %", "Estado", "Fuente",
    ])


def lineup_rating_adjustments(tournament: Dict[str, Any], weight: float = 1.0) -> Tuple[Dict[str, float], List[str]]:
    adjustments: Dict[str, List[float]] = {}
    notes: List[str] = []
    weight = max(min(float(weight or 1.0), 2.0), 0.0)
    for _, fixture in tournament_fixtures_dataframe(tournament).iterrows():
        cache_path = lineup_cache_path(str(fixture["No."]))
        if not cache_path.exists():
            continue
        payload = read_lineup_cache(cache_path)
        if payload.get("status") not in {LINEUP_STATUSES["official"], LINEUP_STATUSES["probable"], LINEUP_STATUSES["last_xi"]}:
            continue
        if not lineup_is_prediction_safe(payload):
            continue
        for team in {payload.get("home", ""), payload.get("away", "")}:
            starters = [
                player for player in payload.get("players", [])
                if player.get("team") == team and player.get("starter")
            ]
            if len(starters) != 11:
                continue
            ratings = [_to_float(player.get("rating")) for player in starters]
            ratings = [rating for rating in ratings if rating is not None]
            if not ratings:
                continue
            adjustment = max(min((sum(ratings) / len(ratings) - 6.75) * 35.0 * weight, 45.0), -45.0)
            adjustments.setdefault(team, []).append(adjustment)
    output = {team: sum(values) / len(values) for team, values in adjustments.items() if values}
    if output:
        notes.append(f"Lineups aplicados a {len(output)} equipos con rating de SofaScore y peso {weight:g}.")
    return output, notes


def player_feature_rating_adjustments(tournament: Dict[str, Any], weight: float = 1.0) -> Tuple[Dict[str, float], List[str]]:
    df = player_features_dataframe(tournament)
    if df.empty:
        return {}, []
    weight = max(min(float(weight or 1.0), 2.0), 0.0)
    usable = df[df["Prediction safe"] == "Si"].copy()
    adjustments_by_team: Dict[str, List[float]] = {}
    for _, row in usable.iterrows():
        rating = _to_float(row.get("XI rating prom"))
        known = _to_float(row.get("Stats conocidos")) or 0.0
        starters = _to_float(row.get("Titulares")) or 0.0
        if rating is None or starters < 11:
            continue
        confidence = min(max(known / 11.0, 0.25), 1.0)
        adjustment = max(min((rating - 6.75) * 42.0 * confidence * weight, 45.0), -45.0)
        adjustments_by_team.setdefault(str(row["Equipo"]), []).append(adjustment)
    adjustments = {team: sum(values) / len(values) for team, values in adjustments_by_team.items() if values}
    notes = []
    if adjustments:
        notes.append(f"Features pre-partido de XI aplicadas a {len(adjustments)} equipos con peso {weight:g}.")
    return adjustments, notes


def player_features_dataframe(tournament: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    fixtures = tournament_fixtures_dataframe(tournament)
    for _, fixture in fixtures.iterrows():
        if not str(fixture.get("Grupo", "")):
            continue
        fixture_key = str(fixture["No."])
        payload = read_prediction_payload(fixture_key)
        if not payload:
            continue
        rows.extend(team_feature_rows(payload))
    return pd.DataFrame(rows, columns=PLAYER_FEATURE_COLUMNS)


def player_stats_payload_for_fixture(tournament: Dict[str, Any], fixture_id: Any, refresh: bool = False) -> Dict[str, Any]:
    fixture = find_fixture(tournament=tournament, fixture_id=fixture_id)
    fixture_key = str(fixture["No."])
    stats_path = player_stats_cache_path(fixture_key)
    lineup_path = lineup_cache_path(fixture_key)
    if stats_path.exists() and not refresh:
        return read_player_stats_cache(stats_path)
    if lineup_path.exists() and not refresh:
        payload = player_stats_payload(read_lineup_cache(lineup_path))
        write_player_stats_cache(stats_path, payload)
        return payload
    lineup = lineup_payload_for_fixture(tournament=tournament, fixture_id=fixture_key, refresh=refresh)
    payload = player_stats_payload(lineup)
    write_player_stats_cache(stats_path, payload)
    return payload


def player_stats_payload(lineup: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "fixture_id": lineup.get("fixture_id", ""),
        "date": lineup.get("date", ""),
        "group": lineup.get("group", ""),
        "home": lineup.get("home", ""),
        "away": lineup.get("away", ""),
        "status": lineup.get("status", ""),
        "source": lineup.get("source", ""),
        "match_url": lineup.get("match_url", ""),
        "fetched_at": lineup.get("fetched_at", ""),
        "formation_home": lineup.get("formation_home", ""),
        "formation_away": lineup.get("formation_away", ""),
        "prediction_safe": lineup_is_prediction_safe(lineup),
        "players": lineup.get("players", []),
        "features": team_feature_rows(lineup),
    }
    return payload


def team_feature_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for team, rival, formation in (
            (payload.get("home", ""), payload.get("away", ""), payload.get("formation_home", "")),
            (payload.get("away", ""), payload.get("home", ""), payload.get("formation_away", "")),
    ):
        if not team:
            continue
        starters = [player for player in payload.get("players", []) if player.get("team") == team and player.get("starter")]
        ratings = [_to_float(player.get("rating")) for player in starters]
        ratings = [rating for rating in ratings if rating is not None]
        row = {
            "Fixture": payload.get("fixture_id", ""),
            "Fecha": payload.get("date", ""),
            "Grupo": payload.get("group", ""),
            "Equipo": team,
            "Rival": rival,
            "Prediction safe": "Si" if lineup_is_prediction_safe(payload) else "No",
            "Titulares": len(starters),
            "Stats conocidos": sum(1 for player in starters if player_has_stats(player)),
            "XI rating prom": round(sum(ratings) / len(ratings), 3) if ratings else "",
            "XI rating std": round(_std(ratings), 3) if len(ratings) > 1 else "",
            "XI rating min": round(min(ratings), 3) if ratings else "",
            "XI rating max": round(max(ratings), 3) if ratings else "",
            "POR rating": _position_avg(starters, ("G",)),
            "DEF rating": _position_avg(starters, ("D",)),
            "MED rating": _position_avg(starters, ("M",)),
            "ATA rating": _position_avg(starters, ("F", "A")),
            "Formacion": formation,
            "Fuente": payload.get("source", ""),
        }
        rows.append(row)
    return rows


def fetch_lanus_lineup(fixture: pd.Series, fixture_key: str, match_url: str) -> Dict[str, Any]:
    lanus = import_lanusstats()
    sofascore = lanus.SofaScore()
    try:
        raw = sofascore.get_lineups(match_url)
        try:
            home_stats, away_stats = sofascore.get_players_match_stats(match_url)
        except Exception:
            home_stats, away_stats = pd.DataFrame(), pd.DataFrame()
    finally:
        close = getattr(sofascore, "close", None)
        if callable(close):
            close()
    return normalize_lanus_lineups(
        raw=raw,
        fixture=fixture,
        fixture_key=fixture_key,
        match_url=match_url,
        home_stats=home_stats,
        away_stats=away_stats,
    )


def fetch_best_sofascore_event(fixture: pd.Series) -> Dict[str, Any]:
    fixture_key = str(fixture.get("No.", ""))
    date = str(fixture.get("Fecha", ""))[:10]
    home = clean_team_name(fixture.get("Equipo 1"))
    away = clean_team_name(fixture.get("Equipo 2"))
    if not date or not home or not away:
        return pending_event_payload(fixture_key, date, home, away, "Fixture incompleto para autodeteccion.")
    lanus = import_lanusstats()
    sofascore = lanus.SofaScore()
    try:
        data = sofascore.sofascore_request(f"api/v1/sport/football/scheduled-events/{date}")
    finally:
        close = getattr(sofascore, "close", None)
        if callable(close):
            close()
    events = data.get("events", []) if isinstance(data, dict) else []
    best = best_event_match(events, home, away)
    if not best:
        return pending_event_payload(fixture_key, date, home, away, "No se encontro evento SofaScore para fecha/equipos.")
    event, confidence, reverse = best
    event_id = str(event.get("id", ""))
    event_home = event_team_name(event, "home")
    event_away = event_team_name(event, "away")
    match_url = sofa_event_url(event_id=event_id, home=event_home, away=event_away)
    return {
        "fixture_id": fixture_key,
        "date": date,
        "home": home,
        "away": away,
        "event_home": event_home,
        "event_away": event_away,
        "event_id": event_id,
        "match_url": match_url,
        "confidence": round(confidence, 3),
        "reverse": bool(reverse),
        "source": "SofaScore scheduled-events",
        "provider": "SofaScore",
        "status": "Detectado",
        "error": "",
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }


def fotmob_event_payload(
        fixture_key: str,
        date: str,
        home: str,
        away: str,
        event_home: str,
        event_away: str,
        event_id: str,
        match_url: str,
        confidence: float,
        reverse: bool,
) -> Dict[str, Any]:
    return {
        "fixture_id": str(fixture_key),
        "date": date,
        "home": home,
        "away": away,
        "event_home": event_home,
        "event_away": event_away,
        "event_id": str(event_id),
        "match_url": match_url,
        "confidence": round(float(confidence or 0.0), 3),
        "reverse": bool(reverse),
        "source": "FotMob matches",
        "provider": "FotMob",
        "status": "Detectado",
        "error": "",
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }


def normalize_lanus_lineups(
        raw: Dict[str, Any],
        fixture: pd.Series,
        fixture_key: str,
        match_url: str = "",
        home_stats: Optional[pd.DataFrame] = None,
        away_stats: Optional[pd.DataFrame] = None,
        fetched_at: Optional[str] = None,
) -> Dict[str, Any]:
    home = clean_team_name(fixture.get("Equipo 1"))
    away = clean_team_name(fixture.get("Equipo 2"))
    players: List[Dict[str, Any]] = []
    home_players = _normalize_side_players(raw.get("home", {}), "home", home, home_stats)
    away_players = _normalize_side_players(raw.get("away", {}), "away", away, away_stats)
    players.extend(home_players)
    players.extend(away_players)
    starters_home = sum(1 for player in home_players if player["starter"])
    starters_away = sum(1 for player in away_players if player["starter"])
    status = detect_lineup_status(raw, starters_home, starters_away)
    return {
        "fixture_id": str(fixture_key),
        "date": _clean_scalar(fixture.get("Fecha", "")),
        "group": _clean_scalar(fixture.get("Grupo", "")),
        "home": home,
        "away": away,
        "status": status,
        "source": "LanusStats/SofaScore",
        "match_url": match_url,
        "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(),
        "formation_home": _side_formation(raw.get("home", {})),
        "formation_away": _side_formation(raw.get("away", {})),
        "starters_home": starters_home,
        "starters_away": starters_away,
        "players": players,
        "error": "",
    }


def best_event_match(events: Iterable[Dict[str, Any]], home: str, away: str) -> Optional[Tuple[Dict[str, Any], float, bool]]:
    candidates: List[Tuple[Dict[str, Any], float, bool]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_home = event_team_name(event, "home")
        event_away = event_team_name(event, "away")
        if not event_home or not event_away:
            continue
        direct = (team_similarity(home, event_home) + team_similarity(away, event_away)) / 2.0
        reverse = (team_similarity(home, event_away) + team_similarity(away, event_home)) / 2.0
        if direct >= reverse:
            confidence = direct
            is_reverse = False
        else:
            confidence = reverse
            is_reverse = True
        if confidence >= 0.72:
            candidates.append((event, confidence, is_reverse))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[1], reverse=True)[0]


def event_team_name(event: Dict[str, Any], side: str) -> str:
    team = event.get(f"{side}Team", {})
    if not isinstance(team, dict):
        return ""
    return clean_team_name(team.get("name") or team.get("shortName") or team.get("slug") or "")


def team_similarity(expected: str, candidate: str) -> float:
    return team_name_similarity(expected, candidate)


def detect_lineup_status(raw: Dict[str, Any], starters_home: int, starters_away: int) -> str:
    if starters_home != 11 or starters_away != 11:
        return LINEUP_STATUSES["pending"]
    confirmed_values = [
        raw.get("confirmed"),
        raw.get("isConfirmed"),
        raw.get("lineupsConfirmed"),
        raw.get("home", {}).get("confirmed") if isinstance(raw.get("home"), dict) else None,
        raw.get("away", {}).get("confirmed") if isinstance(raw.get("away"), dict) else None,
    ]
    if any(value is True for value in confirmed_values):
        return LINEUP_STATUSES["official"]
    return LINEUP_STATUSES["probable"]


def pending_lineup_payload(fixture: pd.Series, fixture_key: str, match_url: str = "", error: str = "") -> Dict[str, Any]:
    home = clean_team_name(fixture.get("Equipo 1"))
    away = clean_team_name(fixture.get("Equipo 2"))
    players = _last_known_players(home) + _last_known_players(away)
    status = LINEUP_STATUSES["last_xi"] if players else LINEUP_STATUSES["pending"]
    return {
        "fixture_id": str(fixture_key),
        "date": _clean_scalar(fixture.get("Fecha", "")),
        "group": _clean_scalar(fixture.get("Grupo", "")),
        "home": home,
        "away": away,
        "status": status,
        "source": "cache:last-xi" if players else "unavailable:lineups",
        "match_url": match_url,
        "fetched_at": "",
        "formation_home": "",
        "formation_away": "",
        "starters_home": sum(1 for player in players if player.get("team") == home and player.get("starter")),
        "starters_away": sum(1 for player in players if player.get("team") == away and player.get("starter")),
        "players": players,
        "error": error,
    }


def find_fixture(tournament: Dict[str, Any], fixture_id: Any) -> pd.Series:
    fixtures = tournament_fixtures_dataframe(tournament)
    match = fixtures[fixtures["No."].astype(str) == str(fixture_id)]
    if match.empty:
        raise LineupProviderError(f'No existe el fixture "{fixture_id}".')
    return match.iloc[0]


def import_lanusstats():
    try:
        import LanusStats as lanusstats
    except ImportError as exc:
        raise LineupProviderError("LanusStats no esta instalado. Ejecuta pip install -r requirements.txt.") from exc
    return lanusstats


def read_lineup_links() -> Dict[str, str]:
    try:
        if LINEUP_LINKS_FILE.exists():
            return json.loads(LINEUP_LINKS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def read_event_cache() -> Dict[str, Dict[str, Any]]:
    try:
        if SOFASCORE_EVENTS_FILE.exists():
            data = json.loads(SOFASCORE_EVENTS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def write_event_cache(events: Dict[str, Dict[str, Any]]) -> None:
    SOFASCORE_ROOT.mkdir(parents=True, exist_ok=True)
    SOFASCORE_EVENTS_FILE.write_text(json.dumps(_json_safe(events), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_lineup_links(links: Dict[str, str]) -> None:
    LINEUPS_ROOT.mkdir(parents=True, exist_ok=True)
    LINEUP_LINKS_FILE.write_text(json.dumps(links, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_lineup_cache(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_lineup_cache(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_player_stats_cache(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_player_stats_cache(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def lineup_cache_path(fixture_key: str) -> Path:
    return LINEUPS_ROOT / f"fixture_{safe_key(fixture_key)}.json"


def player_stats_cache_path(fixture_key: str) -> Path:
    return PLAYER_STATS_ROOT / f"fixture_{safe_key(fixture_key)}.json"


def read_prediction_payload(fixture_key: str) -> Optional[Dict[str, Any]]:
    stats_path = player_stats_cache_path(fixture_key)
    lineup_path = lineup_cache_path(fixture_key)
    try:
        if stats_path.exists():
            stats = read_player_stats_cache(stats_path)
            return {
                "fixture_id": stats.get("fixture_id", ""),
                "date": stats.get("date", ""),
                "group": stats.get("group", ""),
                "home": stats.get("home", ""),
                "away": stats.get("away", ""),
                "status": stats.get("status", ""),
                "source": stats.get("source", ""),
                "match_url": stats.get("match_url", ""),
                "fetched_at": stats.get("fetched_at", ""),
                "formation_home": stats.get("formation_home", ""),
                "formation_away": stats.get("formation_away", ""),
                "players": stats.get("players", []),
            }
        if lineup_path.exists():
            return read_lineup_cache(lineup_path)
    except Exception:
        return None
    return None


def normalize_match_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "sofascore.com" not in text or "id:" not in text:
        raise LineupProviderError("La URL debe ser de SofaScore e incluir id:<match_id>.")
    return text


def safe_key(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_") or "unknown"


def normalize_team_key(value: Any) -> str:
    text = unicode_normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"\b(fc|cf|national team|team)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sofa_event_url(event_id: str, home: str, away: str) -> str:
    home_slug = re.sub(r"[^a-z0-9]+", "-", normalize_team_key(home)).strip("-") or "home"
    away_slug = re.sub(r"[^a-z0-9]+", "-", normalize_team_key(away)).strip("-") or "away"
    return f"https://www.sofascore.com/football/match/{home_slug}-{away_slug}#id:{event_id}"


def pending_event_payload(
        fixture_key: str,
        date: str,
        home: str,
        away: str,
        error: str,
        source: str = "SofaScore scheduled-events",
        provider: str = "SofaScore",
) -> Dict[str, Any]:
    return {
        "fixture_id": str(fixture_key),
        "date": date,
        "home": home,
        "away": away,
        "event_home": "",
        "event_away": "",
        "event_id": "",
        "match_url": "",
        "confidence": 0.0,
        "reverse": False,
        "source": source,
        "provider": provider,
        "status": "Pendiente",
        "error": error,
        "detected_at": datetime.now(timezone.utc).isoformat(),
    }


def clean_error(exc: Exception) -> str:
    return re.sub(r"\s+", " ", str(exc)).strip() or exc.__class__.__name__


def _normalize_side_players(side_data: Any, side: str, team: str, stats_df: Optional[pd.DataFrame]) -> List[Dict[str, Any]]:
    if not isinstance(side_data, dict):
        return []
    raw_players = side_data.get("players") or []
    stats_by_id, stats_by_name = _stats_lookup(stats_df)
    players = []
    for item in raw_players:
        if not isinstance(item, dict):
            continue
        player = item.get("player") if isinstance(item.get("player"), dict) else {}
        player_id = player.get("id") or item.get("playerId") or item.get("id")
        name = player.get("name") or player.get("shortName") or item.get("name") or ""
        stats = stats_by_id.get(str(player_id), {}) or stats_by_name.get(str(name).lower(), {})
        rating = item.get("rating")
        if rating in {None, ""} and isinstance(item.get("statistics"), dict):
            rating = item.get("statistics", {}).get("rating")
        rating = rating if rating not in {None, ""} else stats.get("rating", "")
        player_stats = item.get("statistics") if isinstance(item.get("statistics"), dict) else {}
        player_stats = {**stats, **player_stats}
        players.append({
            "team": team,
            "side": "Local" if side == "home" else "Visitante",
            "name": str(name),
            "position": _clean_scalar(item.get("position") or player.get("position") or ""),
            "shirt_number": _clean_scalar(item.get("shirtNumber") or item.get("jerseyNumber") or player.get("jerseyNumber") or ""),
            "starter": not bool(item.get("substitute", False)),
            "captain": bool(item.get("captain", False)),
            "id": _clean_scalar(player_id or ""),
            "photo_url": sofa_player_photo_url(player_id),
            "rating": _clean_scalar(rating),
            "stats": _json_safe(player_stats),
        })
    return players


def _stats_lookup(stats_df: Optional[pd.DataFrame]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if stats_df is None or stats_df.empty:
        return {}, {}
    by_id: Dict[str, Dict[str, Any]] = {}
    by_name: Dict[str, Dict[str, Any]] = {}
    for _, row in stats_df.iterrows():
        record = row.to_dict()
        if "id" in record:
            by_id[str(record.get("id"))] = record
        if "name" in record:
            by_name[str(record.get("name")).lower()] = record
        if "player" in record:
            by_name[str(record.get("player")).lower()] = record
    return by_id, by_name


def player_has_stats(player: Dict[str, Any]) -> bool:
    stats = player.get("stats")
    if not isinstance(stats, dict) or not stats:
        return False
    return any(value not in {"", None} for value in stats.values())


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _position_avg(players: List[Dict[str, Any]], prefixes: Tuple[str, ...]) -> Any:
    ratings = []
    for player in players:
        position = str(player.get("position") or "").upper()
        if not any(position.startswith(prefix) for prefix in prefixes):
            continue
        rating = _to_float(player.get("rating"))
        if rating is not None:
            ratings.append(rating)
    return round(sum(ratings) / len(ratings), 3) if ratings else ""


def _side_formation(side_data: Any) -> str:
    if not isinstance(side_data, dict):
        return ""
    return str(side_data.get("formation") or side_data.get("initialFormation") or "")


def _last_known_players(team: str) -> List[Dict[str, Any]]:
    candidates: List[Tuple[str, List[Dict[str, Any]]]] = []
    for path in sorted(LINEUPS_ROOT.glob("fixture_*.json")):
        try:
            payload = read_lineup_cache(path)
        except Exception:
            continue
        if payload.get("status") not in {LINEUP_STATUSES["official"], LINEUP_STATUSES["probable"]}:
            continue
        players = [
            {**player, "source_status": payload.get("status", "")}
            for player in payload.get("players", [])
            if player.get("team") == team and player.get("starter")
        ]
        if len(players) == 11:
            candidates.append((payload.get("date", ""), players))
    if not candidates:
        return []
    _, players = sorted(candidates, key=lambda item: item[0])[-1]
    return [{**player, "starter": True} for player in players]


def _to_float(value: Any) -> Optional[float]:
    try:
        if value in {"", None}:
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def sofa_player_photo_url(player_id: Any) -> str:
    player_id = _clean_scalar(player_id)
    if player_id in {"", None}:
        return ""
    return f"https://api.sofascore.app/api/v1/player/{player_id}/image"


def lineup_is_prediction_safe(payload: Dict[str, Any]) -> bool:
    fetched = str(payload.get("fetched_at") or "")
    match_date = str(payload.get("date") or "")
    if not fetched or not match_date:
        return True
    try:
        fetched_date = datetime.fromisoformat(fetched.replace("Z", "+00:00")).date()
        fixture_date = datetime.strptime(match_date[:10], "%Y-%m-%d").date()
    except ValueError:
        return True
    return fetched_date <= fixture_date


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _json_safe(value: Any) -> Any:
    value = _clean_scalar(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value
