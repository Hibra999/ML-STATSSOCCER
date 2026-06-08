from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from lxml import html


OPENFOOTBALL_RAW = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/{year}/worldcup.json"
WORLD_CUP_2026_URL = OPENFOOTBALL_RAW.format(year=2026)
HISTORY_YEARS = (
    1930, 1934, 1938, 1950, 1954, 1958, 1962, 1966, 1970, 1974, 1978,
    1982, 1986, 1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022,
)
CACHE_ROOT = Path("storage") / "worldcup" / "cache"
PLAYERS_LOCAL_FILE = Path("storage") / "worldcup" / "players_2026.csv"
PLAYERS_CACHE_FILE = CACHE_ROOT / "players_2026.csv"
WIKIPEDIA_SQUADS_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"

FALLBACK_2026_GROUPS: Dict[str, List[str]] = {
    "Group A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "Group B": ["Canada", "Bosnia & Herzegovina", "Qatar", "Switzerland"],
    "Group C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "Group D": ["USA", "Paraguay", "Australia", "Turkey"],
    "Group E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "Group F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "Group G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "Group H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "Group I": ["France", "Senegal", "Iraq", "Norway"],
    "Group J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "Group K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "Group L": ["England", "Croatia", "Ghana", "Panama"],
}

FALLBACK_GROUP_DATES = {
    "Group A": ("2026-06-11", "2026-06-18", "2026-06-24"),
    "Group B": ("2026-06-12", "2026-06-18", "2026-06-24"),
    "Group C": ("2026-06-13", "2026-06-19", "2026-06-24"),
    "Group D": ("2026-06-12", "2026-06-19", "2026-06-25"),
    "Group E": ("2026-06-14", "2026-06-20", "2026-06-25"),
    "Group F": ("2026-06-14", "2026-06-20", "2026-06-25"),
    "Group G": ("2026-06-15", "2026-06-21", "2026-06-26"),
    "Group H": ("2026-06-15", "2026-06-21", "2026-06-26"),
    "Group I": ("2026-06-16", "2026-06-22", "2026-06-26"),
    "Group J": ("2026-06-16", "2026-06-22", "2026-06-27"),
    "Group K": ("2026-06-17", "2026-06-23", "2026-06-27"),
    "Group L": ("2026-06-17", "2026-06-23", "2026-06-27"),
}

FALLBACK_KNOCKOUT_MATCHES = [
    {"round": "Round of 32", "num": 73, "date": "2026-06-28", "team1": "2A", "team2": "2B"},
    {"round": "Round of 32", "num": 74, "date": "2026-06-29", "team1": "1E", "team2": "3A/B/C/D/F"},
    {"round": "Round of 32", "num": 75, "date": "2026-06-29", "team1": "1F", "team2": "2C"},
    {"round": "Round of 32", "num": 76, "date": "2026-06-29", "team1": "1C", "team2": "2F"},
    {"round": "Round of 32", "num": 77, "date": "2026-06-30", "team1": "1I", "team2": "3C/D/F/G/H"},
    {"round": "Round of 32", "num": 78, "date": "2026-06-30", "team1": "2E", "team2": "2I"},
    {"round": "Round of 32", "num": 79, "date": "2026-06-30", "team1": "1A", "team2": "3C/E/F/H/I"},
    {"round": "Round of 32", "num": 80, "date": "2026-07-01", "team1": "1L", "team2": "3E/H/I/J/K"},
    {"round": "Round of 32", "num": 81, "date": "2026-07-01", "team1": "1D", "team2": "3B/E/F/I/J"},
    {"round": "Round of 32", "num": 82, "date": "2026-07-01", "team1": "1G", "team2": "3A/E/H/I/J"},
    {"round": "Round of 32", "num": 83, "date": "2026-07-02", "team1": "2K", "team2": "2L"},
    {"round": "Round of 32", "num": 84, "date": "2026-07-02", "team1": "1H", "team2": "2J"},
    {"round": "Round of 32", "num": 85, "date": "2026-07-02", "team1": "1B", "team2": "3E/F/G/I/J"},
    {"round": "Round of 32", "num": 86, "date": "2026-07-03", "team1": "1J", "team2": "2H"},
    {"round": "Round of 32", "num": 87, "date": "2026-07-03", "team1": "1K", "team2": "3D/E/I/J/L"},
    {"round": "Round of 32", "num": 88, "date": "2026-07-03", "team1": "2D", "team2": "2G"},
    {"round": "Round of 16", "num": 89, "date": "2026-07-04", "team1": "W74", "team2": "W77"},
    {"round": "Round of 16", "num": 90, "date": "2026-07-04", "team1": "W73", "team2": "W75"},
    {"round": "Round of 16", "num": 91, "date": "2026-07-05", "team1": "W76", "team2": "W78"},
    {"round": "Round of 16", "num": 92, "date": "2026-07-05", "team1": "W79", "team2": "W80"},
    {"round": "Round of 16", "num": 93, "date": "2026-07-06", "team1": "W83", "team2": "W84"},
    {"round": "Round of 16", "num": 94, "date": "2026-07-06", "team1": "W81", "team2": "W82"},
    {"round": "Round of 16", "num": 95, "date": "2026-07-07", "team1": "W86", "team2": "W88"},
    {"round": "Round of 16", "num": 96, "date": "2026-07-07", "team1": "W85", "team2": "W87"},
    {"round": "Quarter-final", "num": 97, "date": "2026-07-09", "team1": "W89", "team2": "W90"},
    {"round": "Quarter-final", "num": 98, "date": "2026-07-10", "team1": "W93", "team2": "W94"},
    {"round": "Quarter-final", "num": 99, "date": "2026-07-11", "team1": "W91", "team2": "W92"},
    {"round": "Quarter-final", "num": 100, "date": "2026-07-11", "team1": "W95", "team2": "W96"},
    {"round": "Semi-final", "num": 101, "date": "2026-07-14", "team1": "W97", "team2": "W98"},
    {"round": "Semi-final", "num": 102, "date": "2026-07-15", "team1": "W99", "team2": "W100"},
    {"round": "Final", "num": 103, "date": "2026-07-19", "team1": "W101", "team2": "W102"},
]


def load_tournament_2026(refresh: bool = False) -> Tuple[Dict[str, Any], str]:
    """Load World Cup 2026 fixtures from cache/openfootball with a local fallback."""

    cache_path = CACHE_ROOT / "worldcup_2026.json"
    if not refresh:
        cached = _read_json(cache_path)
        if cached:
            return cached, f"cache:{cache_path}"

    data = _download_json(WORLD_CUP_2026_URL)
    if data:
        _write_json(cache_path, data)
        return data, WORLD_CUP_2026_URL

    fallback = fallback_tournament_2026()
    return fallback, "fallback:embedded-groups"


def fallback_tournament_2026() -> Dict[str, Any]:
    matches: List[Dict[str, Any]] = []
    pairings = ((0, 1), (2, 3), (3, 1), (0, 2), (3, 0), (1, 2))
    for group, teams in FALLBACK_2026_GROUPS.items():
        dates = FALLBACK_GROUP_DATES[group]
        for idx, (home_id, away_id) in enumerate(pairings):
            date_index = min(idx // 2, len(dates) - 1)
            match = {
                "round": f"Matchday {idx + 1}",
                "date": dates[date_index],
                "time": "",
                "team1": teams[home_id],
                "team2": teams[away_id],
                "group": group,
                "ground": "",
            }
            if group == "Group A" and idx == 0:
                match.update({"round": "Matchday 1", "time": "13:00 UTC-6", "ground": "Mexico City"})
            matches.append(match)
    matches.extend({**match, "time": "", "ground": ""} for match in FALLBACK_KNOCKOUT_MATCHES)
    return {"name": "World Cup 2026", "matches": matches}


def load_historical_matches(refresh: bool = False, cutoff: str = "2026-06-11") -> Tuple[pd.DataFrame, str]:
    rows: List[Dict[str, Any]] = []
    sources: List[str] = []
    for year in HISTORY_YEARS:
        data, source = load_worldcup_year(year, refresh=refresh)
        if not data:
            continue
        sources.append(source)
        rows.extend(_historical_rows(data, year))

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["Date", "Year", "Team 1", "Team 2", "G1", "G2", "Round", "Group"]), "fallback:no-history"

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()]
    cutoff_ts = pd.Timestamp(cutoff)
    df = df[df["Date"] < cutoff_ts].sort_values("Date", kind="stable").reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df, ", ".join(sorted(set(sources)))


def load_worldcup_year(year: int, refresh: bool = False) -> Tuple[Optional[Dict[str, Any]], str]:
    cache_path = CACHE_ROOT / f"worldcup_{year}.json"
    if not refresh:
        cached = _read_json(cache_path)
        if cached:
            return cached, f"cache:{cache_path}"
    url = OPENFOOTBALL_RAW.format(year=year)
    data = _download_json(url)
    if data:
        _write_json(cache_path, data)
        return data, url
    return None, f"unavailable:{year}"


def group_stage_matches(tournament: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {**match, "num": match.get("num") or index}
        for index, match in enumerate(tournament.get("matches", []), start=1)
        if match.get("group") and match.get("team1") and match.get("team2")
    ]


def knockout_matches(tournament: Dict[str, Any]) -> List[Dict[str, Any]]:
    group_count = len(group_stage_matches(tournament))
    rows = []
    for index, match in enumerate(tournament.get("matches", []), start=1):
        if match.get("group"):
            continue
        round_name = str(match.get("round", ""))
        if round_name == "Match for third place":
            continue
        rows.append({**match, "num": match.get("num") or group_count + len(rows) + 1 or index})
    return rows


def groups_from_tournament(tournament: Dict[str, Any]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for match in group_stage_matches(tournament):
        group = str(match.get("group") or "")
        if not group:
            continue
        groups.setdefault(group, [])
        for team_key in ("team1", "team2"):
            team = clean_team_name(match.get(team_key))
            if team and team not in groups[group]:
                groups[group].append(team)
    if len(groups) < 12 or any(len(teams) < 4 for teams in groups.values()):
        return {group: teams[:] for group, teams in FALLBACK_2026_GROUPS.items()}
    return dict(sorted(groups.items(), key=lambda item: group_sort_key(item[0])))


def groups_dataframe(tournament: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for group, teams in groups_from_tournament(tournament).items():
        for seed, team in enumerate(teams, start=1):
            rows.append({"Grupo": group, "Bombo": seed, "Equipo": team})
    return pd.DataFrame(rows, columns=["Grupo", "Bombo", "Equipo"])


def tournament_fixtures_dataframe(tournament: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for index, match in enumerate(tournament.get("matches", []), start=1):
        rows.append({
            "No.": match.get("num") or index,
            "Fecha": match.get("date", ""),
            "Hora": match.get("time", ""),
            "Ronda": match.get("round", ""),
            "Grupo": match.get("group", ""),
            "Equipo 1": clean_team_name(match.get("team1")),
            "Equipo 2": clean_team_name(match.get("team2")),
            "Sede": match.get("ground", ""),
        })
    return pd.DataFrame(rows, columns=["No.", "Fecha", "Hora", "Ronda", "Grupo", "Equipo 1", "Equipo 2", "Sede"])


def teams_dataframe(tournament: Dict[str, Any], model=None) -> pd.DataFrame:
    groups = groups_from_tournament(tournament)
    rows = []
    for group, teams in groups.items():
        for team in teams:
            profile = model.profile(team) if model is not None else None
            rows.append({
                "Grupo": group,
                "Equipo": team,
                "Rating": round(profile.rating, 1) if profile else "",
                "Partidos hist.": profile.matches if profile else "",
                "GF/Partido": round(profile.gf_per_match, 2) if profile else "",
                "GA/Partido": round(profile.ga_per_match, 2) if profile else "",
                "Ataque": round(profile.attack, 2) if profile else "",
                "Defensa rival": round(profile.defense, 2) if profile else "",
            })
    return pd.DataFrame(rows)


def load_players(refresh: bool = False) -> Tuple[pd.DataFrame, str]:
    if PLAYERS_LOCAL_FILE.exists() and not refresh:
        return normalize_players_dataframe(pd.read_csv(PLAYERS_LOCAL_FILE), f"local:{PLAYERS_LOCAL_FILE}")
    if PLAYERS_CACHE_FILE.exists() and not refresh:
        return normalize_players_dataframe(pd.read_csv(PLAYERS_CACHE_FILE), f"cache:{PLAYERS_CACHE_FILE}")

    try:
        scraped = scrape_wikipedia_squads()
    except Exception:
        scraped = pd.DataFrame()
    if not scraped.empty:
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        scraped.to_csv(PLAYERS_CACHE_FILE, index=False)
        return scraped, WIKIPEDIA_SQUADS_URL

    if PLAYERS_LOCAL_FILE.exists():
        return normalize_players_dataframe(pd.read_csv(PLAYERS_LOCAL_FILE), f"local:{PLAYERS_LOCAL_FILE}")
    return pd.DataFrame(columns=["Equipo", "Jugador", "Posicion", "Club", "Edad", "Fuente"]), "unavailable:players"


def scrape_wikipedia_squads() -> pd.DataFrame:
    response = requests.get(WIKIPEDIA_SQUADS_URL, timeout=15, headers={"User-Agent": "ML-STATSSOCCER/1.0"})
    response.raise_for_status()
    doc = html.fromstring(response.text)
    rows: List[pd.DataFrame] = []
    for table in doc.xpath('//table[contains(concat(" ", normalize-space(@class), " "), " wikitable ")]'):
        team = _nearest_heading(table)
        if not team or team.lower() in {"notes", "references"}:
            continue
        parsed = pd.read_html(StringIO(html.tostring(table, encoding="unicode")))[0]
        normalized = normalize_players_dataframe(parsed, WIKIPEDIA_SQUADS_URL, default_team=team)
        if not normalized.empty:
            rows.append(normalized)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["Equipo", "Jugador"], keep="first")


def normalize_players_dataframe(df: pd.DataFrame, source: str, default_team: str = "") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Equipo", "Jugador", "Posicion", "Club", "Edad", "Fuente"])
    clean_df = df.copy()
    clean_df.columns = [_flatten_column(column) for column in clean_df.columns]
    column_map = _players_column_map(clean_df.columns)
    if "Jugador" not in column_map:
        return pd.DataFrame(columns=["Equipo", "Jugador", "Posicion", "Club", "Edad", "Fuente"])
    output = pd.DataFrame()
    output["Equipo"] = clean_df[column_map.get("Equipo")].astype(str) if "Equipo" in column_map else default_team
    output["Jugador"] = clean_df[column_map["Jugador"]].astype(str)
    output["Posicion"] = clean_df[column_map.get("Posicion")].astype(str) if "Posicion" in column_map else ""
    output["Club"] = clean_df[column_map.get("Club")].astype(str) if "Club" in column_map else ""
    output["Edad"] = clean_df[column_map.get("Edad")] if "Edad" in column_map else ""
    output["Fuente"] = source
    output = output.replace({"nan": ""}).fillna("")
    output = output[output["Jugador"].str.len() > 1].reset_index(drop=True)
    return output[["Equipo", "Jugador", "Posicion", "Club", "Edad", "Fuente"]]


def clean_team_name(value: Any) -> str:
    return str(value or "").strip()


def group_letter(group: str) -> str:
    match = re.search(r"([A-L])$", str(group))
    return match.group(1) if match else str(group).replace("Group ", "")[:1]


def group_sort_key(group: str) -> Tuple[int, str]:
    letter = group_letter(group)
    return (ord(letter) - ord("A"), str(group))


def _historical_rows(data: Dict[str, Any], year: int) -> Iterable[Dict[str, Any]]:
    for match in data.get("matches", []):
        score = match.get("score") or {}
        ft = score.get("ft") if isinstance(score, dict) else None
        if not isinstance(ft, list) or len(ft) != 2:
            continue
        team1 = clean_team_name(match.get("team1"))
        team2 = clean_team_name(match.get("team2"))
        if not team1 or not team2 or re.match(r"^[WL]\d+$", team1) or re.match(r"^[WL]\d+$", team2):
            continue
        rows = {
            "Date": match.get("date", ""),
            "Year": year,
            "Team 1": team1,
            "Team 2": team2,
            "G1": int(ft[0]),
            "G2": int(ft[1]),
            "Round": match.get("round", ""),
            "Group": match.get("group", ""),
        }
        yield rows


def _download_json(url: str) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(url, timeout=12, headers={"User-Agent": "ML-STATSSOCCER/1.0"})
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
    except Exception:
        return None
    return None


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _flatten_column(column: Any) -> str:
    if isinstance(column, tuple):
        return " ".join(str(part) for part in column if str(part) != "nan").strip()
    return str(column).strip()


def _players_column_map(columns: Iterable[str]) -> Dict[str, str]:
    aliases = {
        "Equipo": ("team", "country", "nation", "seleccion", "equipo"),
        "Jugador": ("player", "name", "jugador"),
        "Posicion": ("pos", "position", "posicion"),
        "Club": ("club", "club team"),
        "Edad": ("age", "edad"),
    }
    result: Dict[str, str] = {}
    for column in columns:
        normalized = re.sub(r"[^a-z0-9]+", " ", column.lower()).strip()
        for output, keys in aliases.items():
            if output in result:
                continue
            if any(key in normalized for key in keys):
                result[output] = column
    return result


def _nearest_heading(table) -> str:
    headings = table.xpath("preceding::h2[span[@class='mw-headline']][1]//span[@class='mw-headline']/text()")
    if not headings:
        headings = table.xpath("preceding::h3[span[@class='mw-headline']][1]//span[@class='mw-headline']/text()")
    return clean_team_name(headings[-1]) if headings else ""
