from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

import pandas as pd
import requests


FOTMOB_ALL_LEAGUES_URL = "https://www.fotmob.com/api/data/allLeagues"
FOTMOB_LEAGUE_URL = "https://www.fotmob.com/api/data/leagues"
FOTMOB_TIMEOUT = 12
MEXICO_CITY_TZ = ZoneInfo("America/Mexico_City")
FOTMOB_HEADERS = {
    "Accept": "application/json,text/plain,*/*",
    "User-Agent": "ML-STATSSOCCER/1.0 (+https://www.fotmob.com)",
}
FOTMOB_DASHBOARD_COLUMNS = ["Date", "Dia", "Hora MX", "Home", "Away", "Fuente"]

COUNTRY_TO_FOTMOB_CCODE = {
    "Argentina": "ARG",
    "Belgium": "BEL",
    "Brazil": "BRA",
    "China": "CHN",
    "Denmark": "DEN",
    "England": "ENG",
    "Finland": "FIN",
    "France": "FRA",
    "Germany": "GER",
    "Greece": "GRE",
    "Ireland": "IRL",
    "Italy": "ITA",
    "Japan": "JPN",
    "Mexico": "MEX",
    "Netherlands": "NED",
    "Norway": "NOR",
    "Poland": "POL",
    "Portugal": "POR",
    "Romania": "ROU",
    "Russia": "RUS",
    "Scotland": "SCO",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "SUI",
    "Turkey": "TUR",
    "USA": "USA",
}

LEAGUE_NAME_ALIASES = {
    ("Argentina", "Primera-Division"): "Liga Profesional",
    ("Belgium", "Jupiler-League"): "First Division A",
    ("China", "Super-League"): "Super League",
    ("Denmark", "Super-Liga"): "Superligaen",
    ("England", "League-1"): "League One",
    ("England", "League-2"): "League Two",
    ("Germany", "Bundesliga-1"): "Bundesliga",
    ("Germany", "Bundesliga-2"): "2. Bundesliga",
    ("Greece", "Super-League"): "Super League 1",
    ("Japan", "J-1"): "J. League",
    ("Mexico", "Liga-MX"): "Liga MX",
    ("Portugal", "Liga-1"): "Liga Portugal",
    ("Romania", "Liga-1"): "Liga I",
    ("Spain", "La-Liga"): "LaLiga",
    ("Spain", "Segunda-Division"): "LaLiga2",
    ("Turkey", "Super-Lig"): "Super Lig",
}

SPANISH_WEEKDAYS = {
    0: "Lunes",
    1: "Martes",
    2: "Miercoles",
    3: "Jueves",
    4: "Viernes",
    5: "Sabado",
    6: "Domingo",
}


class FotMobFixtureError(RuntimeError):
    pass


def scrape_fotmob_upcoming_fixtures(
        league,
        days: int = 7,
        limit: Optional[int] = None,
        now: Optional[datetime] = None,
        session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    source = resolve_fotmob_league(league=league, session=session)
    payload = fetch_fotmob_league_payload(league_id=source["id"], session=session)
    fixture_df = parse_fotmob_upcoming_fixtures(
        payload=payload,
        days=days,
        limit=limit,
        now=now,
        source_name=f"FotMob: {source['name']}",
    )
    return fixture_df


def resolve_fotmob_league(league, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    ccode = COUNTRY_TO_FOTMOB_CCODE.get(str(league.country))
    if not ccode:
        raise FotMobFixtureError(f"Sin codigo de pais FotMob para {league.country}.")

    target_name = LEAGUE_NAME_ALIASES.get((str(league.country), str(league.name)), str(league.name).replace("-", " "))
    target_key = normalize_text(target_name)
    candidates = [
        item for item in fetch_all_fotmob_leagues(session=session)
        if item.get("ccode") == ccode and not is_secondary_fotmob_competition(item)
    ]

    for item in candidates:
        if any(normalize_text(str(item.get(key, ""))) == target_key for key in ("name", "localizedName")):
            return item

    for item in candidates:
        values = [str(item.get("name", "")), str(item.get("localizedName", "")), str(item.get("pageUrl", ""))]
        if any(target_key in normalize_text(value) for value in values):
            return item

    raise FotMobFixtureError(f"Sin fuente FotMob para {league.country} / {league.name}.")


def fetch_fotmob_league_payload(league_id: Any, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    data = fotmob_get_json(
        FOTMOB_LEAGUE_URL,
        params={"id": league_id, "ccode3": "MEX"},
        session=session,
    )
    if not isinstance(data, dict):
        raise FotMobFixtureError(f"FotMob no devolvio datos para liga {league_id}.")
    return data


def parse_fotmob_upcoming_fixtures(
        payload: Dict[str, Any],
        days: int = 7,
        limit: Optional[int] = None,
        now: Optional[datetime] = None,
        source_name: str = "FotMob",
) -> pd.DataFrame:
    if not isinstance(payload, dict):
        return pd.DataFrame(columns=FOTMOB_DASHBOARD_COLUMNS)

    now_mx = (now or datetime.now(tz=MEXICO_CITY_TZ)).astimezone(MEXICO_CITY_TZ)
    start_date = now_mx.date()
    end_date = start_date + timedelta(days=max(int(days or 1), 1))

    rows = []
    for match in extract_fotmob_matches(payload):
        status = match.get("status") if isinstance(match.get("status"), dict) else {}
        if status.get("finished") or status.get("cancelled") or status.get("awarded"):
            continue
        utc_text = status.get("utcTime") or match.get("utcTime") or match.get("timeUTC")
        match_time = parse_fotmob_datetime(utc_text)
        if match_time is None:
            continue
        match_time_mx = match_time.astimezone(MEXICO_CITY_TZ)
        if not (start_date <= match_time_mx.date() < end_date):
            continue
        home = team_name(match.get("home"))
        away = team_name(match.get("away"))
        if not home or not away:
            continue
        rows.append({
            "Date": match_time_mx.date().isoformat(),
            "Dia": SPANISH_WEEKDAYS[match_time_mx.weekday()],
            "Hora MX": match_time_mx.strftime("%H:%M"),
            "Home": home,
            "Away": away,
            "Fuente": source_name,
        })

    fixture_df = pd.DataFrame(rows, columns=FOTMOB_DASHBOARD_COLUMNS)
    if fixture_df.empty:
        return fixture_df
    fixture_df = fixture_df.sort_values(["Date", "Hora MX", "Home", "Away"], kind="stable").drop_duplicates(
        subset=["Date", "Hora MX", "Home", "Away"],
    )
    if limit is not None:
        fixture_df = fixture_df.head(max(int(limit), 0))
    return fixture_df.reset_index(drop=True)


def extract_fotmob_matches(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    fixtures = payload.get("fixtures") if isinstance(payload.get("fixtures"), dict) else {}
    candidates = [
        fixtures.get("allMatches"),
        fixtures.get("matches"),
        payload.get("allMatches"),
        payload.get("matches"),
    ]
    matches: List[Dict[str, Any]] = []
    for candidate in candidates:
        matches.extend(list(walk_fotmob_match_items(candidate)))
    return matches


def walk_fotmob_match_items(value: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(value, dict):
        if isinstance(value.get("home"), dict) and isinstance(value.get("away"), dict):
            yield value
            return
        for child in value.values():
            yield from walk_fotmob_match_items(child)
    elif isinstance(value, list):
        for item in value:
            yield from walk_fotmob_match_items(item)


def team_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("shortName") or "").strip()
    return ""


def parse_fotmob_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed


def fetch_all_fotmob_leagues(session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
    if session is None:
        return list(cached_all_fotmob_leagues())
    data = fotmob_get_json(FOTMOB_ALL_LEAGUES_URL, session=session)
    return flatten_fotmob_leagues(data)


@lru_cache(maxsize=1)
def cached_all_fotmob_leagues() -> tuple:
    data = fotmob_get_json(FOTMOB_ALL_LEAGUES_URL, session=None)
    return tuple(flatten_fotmob_leagues(data))


def flatten_fotmob_leagues(data: Any) -> List[Dict[str, Any]]:
    leagues: List[Dict[str, Any]] = []

    def walk(value: Any):
        if isinstance(value, dict):
            if "id" in value and ("name" in value or "localizedName" in value) and "ccode" in value:
                leagues.append({
                    "id": value.get("id"),
                    "name": value.get("name") or value.get("localizedName") or "",
                    "localizedName": value.get("localizedName") or value.get("name") or "",
                    "pageUrl": value.get("pageUrl") or "",
                    "ccode": value.get("ccode") or "",
                })
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(data)
    return leagues


def is_secondary_fotmob_competition(item: Dict[str, Any]) -> bool:
    text = normalize_text(" ".join(str(item.get(key, "")) for key in ("name", "localizedName", "pageUrl")))
    secondary_terms = ("qualification", "qualifying", "playoff", "play off", "cup", "women", "femenil", "u18", "u20")
    return any(term in text for term in secondary_terms)


def fotmob_get_json(url: str, params: Optional[Dict[str, Any]] = None, session: Optional[requests.Session] = None) -> Any:
    client = session or requests
    request_url = url
    if params:
        request_url = f"{url}?{urlencode(params)}"
    try:
        response = client.get(request_url, headers=FOTMOB_HEADERS, timeout=FOTMOB_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise FotMobFixtureError(f"No se pudo cargar FotMob: {exc}") from exc
    except ValueError as exc:
        raise FotMobFixtureError("FotMob no devolvio JSON valido.") from exc


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
