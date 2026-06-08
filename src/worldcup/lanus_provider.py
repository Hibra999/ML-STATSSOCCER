from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.worldcup.data import clean_team_name, tournament_fixtures_dataframe


LINEUPS_ROOT = Path("storage") / "worldcup" / "lineups"
LINEUP_LINKS_FILE = LINEUPS_ROOT / "links.json"
LINEUP_STATUSES = {
    "official": "Oficial",
    "probable": "Probable",
    "last_xi": "Ultimo XI",
    "pending": "Pendiente",
}


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


def lineups_summary(tournament: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    links = read_lineup_links()
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
            "Estado": payload.get("status", ""),
            "Fuente": payload.get("source", ""),
        })
    return pd.DataFrame(rows, columns=[
        "Equipo", "Lado", "Jugador", "Posicion", "Dorsal", "Titular", "Capitan", "SofaScore ID", "Rating", "Estado", "Fuente",
    ])


def lineup_rating_adjustments(tournament: Dict[str, Any]) -> Tuple[Dict[str, float], List[str]]:
    adjustments: Dict[str, List[float]] = {}
    notes: List[str] = []
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
            adjustment = max(min((sum(ratings) / len(ratings) - 6.75) * 35.0, 30.0), -30.0)
            adjustments.setdefault(team, []).append(adjustment)
    output = {team: sum(values) / len(values) for team, values in adjustments.items() if values}
    if output:
        notes.append(f"Lineups aplicados a {len(output)} equipos con rating de SofaScore.")
    return output, notes


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


def write_lineup_links(links: Dict[str, str]) -> None:
    LINEUPS_ROOT.mkdir(parents=True, exist_ok=True)
    LINEUP_LINKS_FILE.write_text(json.dumps(links, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_lineup_cache(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_lineup_cache(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def lineup_cache_path(fixture_key: str) -> Path:
    return LINEUPS_ROOT / f"fixture_{safe_key(fixture_key)}.json"


def normalize_match_url(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if "sofascore.com" not in text or "id:" not in text:
        raise LineupProviderError("La URL debe ser de SofaScore e incluir id:<match_id>.")
    return text


def safe_key(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_") or "unknown"


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
        players.append({
            "team": team,
            "side": "Local" if side == "home" else "Visitante",
            "name": str(name),
            "position": _clean_scalar(item.get("position") or player.get("position") or ""),
            "shirt_number": _clean_scalar(item.get("shirtNumber") or item.get("jerseyNumber") or player.get("jerseyNumber") or ""),
            "starter": not bool(item.get("substitute", False)),
            "captain": bool(item.get("captain", False)),
            "id": _clean_scalar(player_id or ""),
            "rating": _clean_scalar(rating),
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
