from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests

from src.worldcup.data import clean_team_name


FOTMOB_ROOT = Path("storage") / "worldcup" / "fotmob"
FOTMOB_EVENTS_FILE = FOTMOB_ROOT / "events.json"
FOTMOB_MATCHES_URL = "https://www.fotmob.com/api/matches"
FOTMOB_MATCH_DETAILS_URL = "https://www.fotmob.com/api/matchDetails"
FOTMOB_HEADERS = {
    "Accept": "application/json,text/plain,*/*",
    "User-Agent": "ML-STATSSOCCER/1.0 (+https://www.fotmob.com)",
}
FOTMOB_TIMEOUT = 15


class FotMobProviderError(RuntimeError):
    pass


def fetch_best_fotmob_event(
        fixture: pd.Series,
        similarity_fn,
        event_builder,
        pending_builder,
) -> Dict[str, Any]:
    fixture_key = str(fixture.get("No.", ""))
    date = str(fixture.get("Fecha", ""))[:10]
    home = clean_team_name(fixture.get("Equipo 1"))
    away = clean_team_name(fixture.get("Equipo 2"))
    if not date or not home or not away:
        return pending_builder(fixture_key, date, home, away, "Fixture incompleto para FotMob.")

    payload = fotmob_get_json(FOTMOB_MATCHES_URL, params={"date": date.replace("-", "")})
    events = extract_fotmob_matches(payload)
    best = best_fotmob_match(events, home, away, similarity_fn)
    if not best:
        return pending_builder(fixture_key, date, home, away, "No se encontro evento FotMob para fecha/equipos.")

    event, confidence, reverse = best
    event_id = str(event.get("id") or event.get("matchId") or "")
    event_home = fotmob_team_name(event.get("home"))
    event_away = fotmob_team_name(event.get("away"))
    return event_builder(
        fixture_key=fixture_key,
        date=date,
        home=home,
        away=away,
        event_home=event_home,
        event_away=event_away,
        event_id=event_id,
        match_url=fotmob_event_url(event_id, event_home, event_away),
        confidence=confidence,
        reverse=reverse,
    )


def fetch_fotmob_lineup(fixture: pd.Series, fixture_key: str, event: Dict[str, Any]) -> Dict[str, Any]:
    event_id = str(event.get("event_id") or "")
    if not event_id:
        raise FotMobProviderError("Evento FotMob sin match id.")

    details = fotmob_get_json(FOTMOB_MATCH_DETAILS_URL, params={"matchId": event_id})
    home = clean_team_name(fixture.get("Equipo 1"))
    away = clean_team_name(fixture.get("Equipo 2"))
    home_players, away_players, formation_home, formation_away, confirmed = normalize_fotmob_players(details, home, away)
    starters_home = sum(1 for player in home_players if player["starter"])
    starters_away = sum(1 for player in away_players if player["starter"])
    status = "Oficial" if confirmed and starters_home == 11 and starters_away == 11 else "Probable" if starters_home == 11 and starters_away == 11 else "Pendiente"
    error = "" if starters_home == 11 and starters_away == 11 else "FotMob no publico 11 completos para este partido."
    return {
        "fixture_id": str(fixture_key),
        "date": clean_scalar(fixture.get("Fecha", "")),
        "group": clean_scalar(fixture.get("Grupo", "")),
        "home": home,
        "away": away,
        "status": status,
        "source": "FotMob",
        "provider": "FotMob",
        "event_id": event_id,
        "match_url": event.get("match_url", ""),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "formation_home": formation_home,
        "formation_away": formation_away,
        "starters_home": starters_home,
        "starters_away": starters_away,
        "players": home_players + away_players,
        "error": error,
    }


def fotmob_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response = requests.get(url, params=params or {}, headers=FOTMOB_HEADERS, timeout=FOTMOB_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise FotMobProviderError("FotMob no devolvio JSON de objeto.")
    return data


def extract_fotmob_matches(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    candidates = [
        payload.get("matches"),
        payload.get("allMatches"),
        payload.get("fixtures"),
        payload.get("matchList"),
    ]
    for candidate in candidates:
        matches.extend(list(walk_match_items(candidate)))
    if not matches:
        matches.extend(list(walk_match_items(payload)))
    deduped = []
    seen = set()
    for match in matches:
        match_id = str(match.get("id") or match.get("matchId") or id(match))
        if match_id in seen:
            continue
        seen.add(match_id)
        deduped.append(match)
    return deduped


def walk_match_items(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("home"), dict) and isinstance(value.get("away"), dict):
            yield value
            return
        for child in value.values():
            yield from walk_match_items(child)
    elif isinstance(value, list):
        for item in value:
            yield from walk_match_items(item)


def best_fotmob_match(events: Iterable[Dict[str, Any]], home: str, away: str, similarity_fn) -> Optional[Tuple[Dict[str, Any], float, bool]]:
    candidates: List[Tuple[Dict[str, Any], float, bool]] = []
    for event in events:
        event_home = fotmob_team_name(event.get("home"))
        event_away = fotmob_team_name(event.get("away"))
        if not event_home or not event_away:
            continue
        direct = (similarity_fn(home, event_home) + similarity_fn(away, event_away)) / 2.0
        reverse = (similarity_fn(home, event_away) + similarity_fn(away, event_home)) / 2.0
        confidence, reversed_match = (direct, False) if direct >= reverse else (reverse, True)
        if confidence >= 0.72:
            candidates.append((event, confidence, reversed_match))
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[1], reverse=True)[0]


def normalize_fotmob_players(details: Dict[str, Any], home: str, away: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str, bool]:
    lineup_root = first_dict_path(details, ("content", "lineup")) or first_dict_path(details, ("lineup",)) or {}
    team_blocks = extract_lineup_team_blocks(lineup_root)
    if len(team_blocks) < 2:
        team_blocks = extract_lineup_team_blocks(details)
    home_block, away_block = match_lineup_blocks(team_blocks, home, away)
    confirmed = bool(lineup_root.get("confirmed") or lineup_root.get("isConfirmed") or details.get("lineupConfirmed"))
    home_players = normalize_side_players(home_block, "home", home)
    away_players = normalize_side_players(away_block, "away", away)
    return (
        home_players,
        away_players,
        clean_scalar(home_block.get("formation") or home_block.get("formationName") or ""),
        clean_scalar(away_block.get("formation") or away_block.get("formationName") or ""),
        confirmed,
    )


def extract_lineup_team_blocks(value: Any) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        for key in ("lineup", "teams", "teamLineups", "lineups"):
            child = value.get(key)
            if isinstance(child, list):
                for item in child:
                    if isinstance(item, dict) and looks_like_team_lineup(item):
                        blocks.append(item)
            elif isinstance(child, dict):
                blocks.extend(extract_lineup_team_blocks(child))
        if looks_like_team_lineup(value):
            blocks.append(value)
        for child in value.values():
            if isinstance(child, (dict, list)):
                blocks.extend(extract_lineup_team_blocks(child))
    elif isinstance(value, list):
        for item in value:
            blocks.extend(extract_lineup_team_blocks(item))
    output = []
    seen = set()
    for block in blocks:
        key = str(block.get("teamId") or block.get("id") or block.get("teamName") or id(block))
        if key not in seen:
            seen.add(key)
            output.append(block)
    return output


def looks_like_team_lineup(value: Dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    has_team = any(key in value for key in ("teamName", "teamId", "team", "name"))
    has_players = any(isinstance(value.get(key), list) for key in ("players", "lineup", "starters", "bench", "subs", "substitutes"))
    return has_team and has_players


def match_lineup_blocks(blocks: List[Dict[str, Any]], home: str, away: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if len(blocks) < 2:
        return {}, {}
    scored = []
    for block in blocks:
        name = block_team_name(block)
        scored.append((block, name))
    home_block = max(scored, key=lambda item: name_similarity(home, item[1]))[0]
    away_candidates = [item for item in scored if item[0] is not home_block]
    away_block = max(away_candidates, key=lambda item: name_similarity(away, item[1]))[0] if away_candidates else {}
    return home_block, away_block


def normalize_side_players(block: Dict[str, Any], side: str, team: str) -> List[Dict[str, Any]]:
    players: List[Dict[str, Any]] = []
    starters = flatten_players(block.get("players") or block.get("lineup") or block.get("starters") or [])
    bench = flatten_players(block.get("bench") or block.get("subs") or block.get("substitutes") or [])
    for item in starters:
        players.append(fotmob_player_payload(item, side, team, starter=True))
    for item in bench:
        players.append(fotmob_player_payload(item, side, team, starter=False))
    if len([player for player in players if player["starter"]]) > 11:
        seen = 0
        for player in players:
            if player["starter"]:
                seen += 1
                player["starter"] = seen <= 11
    return players


def flatten_players(value: Any) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        if any(key in value for key in ("id", "playerId", "name", "fullName")):
            output.append(value)
        for child in value.values():
            if isinstance(child, (dict, list)):
                output.extend(flatten_players(child))
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                output.extend(flatten_players(item))
    return output


def fotmob_player_payload(item: Dict[str, Any], side: str, team: str, starter: bool) -> Dict[str, Any]:
    player = item.get("player") if isinstance(item.get("player"), dict) else item
    player_id = player.get("id") or player.get("playerId") or item.get("id") or item.get("playerId")
    name = player.get("name") or player.get("fullName") or player.get("shortName") or item.get("name") or ""
    stats = player.get("stats") if isinstance(player.get("stats"), dict) else item.get("stats") if isinstance(item.get("stats"), dict) else {}
    rating = player.get("rating") or item.get("rating") or stats.get("rating") or ""
    is_starter = bool(item.get("isStarter", starter))
    if str(item.get("role", "")).lower() in {"substitute", "bench"}:
        is_starter = False
    return {
        "team": team,
        "side": "Local" if side == "home" else "Visitante",
        "name": str(name),
        "position": clean_scalar(player.get("position") or player.get("positionString") or item.get("position") or item.get("positionString") or ""),
        "shirt_number": clean_scalar(player.get("shirtNumber") or player.get("shirt") or item.get("shirtNumber") or item.get("shirt") or ""),
        "starter": is_starter,
        "captain": bool(player.get("isCaptain") or item.get("isCaptain") or item.get("captain")),
        "id": clean_scalar(player_id or ""),
        "photo_url": fotmob_player_photo_url(player_id),
        "rating": clean_scalar(rating),
        "stats": json_safe(stats),
    }


def block_team_name(block: Dict[str, Any]) -> str:
    team = block.get("team") if isinstance(block.get("team"), dict) else {}
    return clean_team_name(block.get("teamName") or block.get("name") or team.get("name") or team.get("shortName") or "")


def fotmob_team_name(value: Any) -> str:
    if isinstance(value, dict):
        return clean_team_name(value.get("name") or value.get("shortName") or value.get("longName") or "")
    return clean_team_name(value)


def first_dict_path(value: Dict[str, Any], path: Tuple[str, ...]) -> Dict[str, Any]:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def fotmob_event_url(event_id: str, home: str, away: str) -> str:
    slug = f"{slugify(home)}-vs-{slugify(away)}"
    return f"https://www.fotmob.com/matches/{slug}/{event_id}"


def fotmob_player_photo_url(player_id: Any) -> str:
    player_id = clean_scalar(player_id)
    if player_id in {"", None}:
        return ""
    return f"https://images.fotmob.com/image_resources/playerimages/{player_id}.png"


def slugify(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return text or "match"


def name_similarity(left: str, right: str) -> float:
    left_key = re.sub(r"[^a-z0-9]+", "", str(left or "").lower())
    right_key = re.sub(r"[^a-z0-9]+", "", str(right or "").lower())
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0
    if left_key in right_key or right_key in left_key:
        return 0.9
    return 0.0


def clean_scalar(value: Any) -> Any:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return value


def json_safe(value: Any) -> Any:
    value = clean_scalar(value)
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value
