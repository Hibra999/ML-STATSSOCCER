from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.worldcup.data import clean_team_name


API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_ROOT = Path("storage") / "worldcup" / "api_football"
API_FOOTBALL_RAW_ROOT = API_FOOTBALL_ROOT / "raw"
DOTENV_FILE = Path(".env")
API_FOOTBALL_TIMEOUT = 18
API_FOOTBALL_ENV_KEYS = ("API_FOOTBALL_KEY", "APIFOOTBALL_KEY", "API_SPORTS_KEY")
DEFAULT_WORLD_CUP_LEAGUE_ID = 1
DEFAULT_WORLD_CUP_SEASONS = (2014, 2018, 2022, 2026)
API_STAT_ALIASES = {
    "shots_on_goal": "shots_on_goal",
    "shots_off_goal": "shots_off_goal",
    "total_shots": "total_shots",
    "blocked_shots": "blocked_shots",
    "shots_insidebox": "shots_inside_box",
    "shots_outsidebox": "shots_outside_box",
    "fouls": "fouls",
    "corner_kicks": "corners",
    "offsides": "offsides",
    "ball_possession": "possession",
    "yellow_cards": "yellow_cards",
    "red_cards": "red_cards",
    "goalkeeper_saves": "saves",
    "total_passes": "passes",
    "passes_accurate": "passes_accurate",
    "passes": "passes_pct",
    "expected_goals": "xg",
    "xg": "xg",
}


class ApiFootballProviderError(RuntimeError):
    pass


def load_api_football_data(
        force_download: bool = False,
        allow_download: bool = False,
        seasons: Optional[Iterable[int]] = None,
        league_ids: Optional[Iterable[int]] = None,
        session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    warnings: List[str] = []
    downloaded: List[str] = []
    if allow_download:
        key = api_football_key()
        if not key:
            warnings.append("API-Football omitido: define API_FOOTBALL_KEY para descargar datos oficiales.")
        else:
            try:
                downloaded = download_api_football_cache(
                    api_key=key,
                    force=force_download,
                    seasons=seasons,
                    league_ids=league_ids,
                    session=session,
                )
            except Exception as exc:
                warnings.append(f"API-Football no pudo descargarse ({exc.__class__.__name__}: {exc}).")
    elif not API_FOOTBALL_RAW_ROOT.exists():
        warnings.append(f"API-Football no disponible en cache: {API_FOOTBALL_RAW_ROOT}.")

    payloads = read_api_football_cache()
    normalized = normalize_api_football_payloads(payloads)
    has_rows = any(
        not normalized[name].empty
        for name in ("fixtures", "team_stats", "lineups", "injuries", "odds", "market_rows")
    )
    return {
        **normalized,
        "status": "ok" if has_rows else "missing",
        "sources": [str(payload.get("cache_file", "")) for payload in payloads if payload.get("cache_file")],
        "downloaded": downloaded,
        "loaded_at": datetime.now(timezone.utc).isoformat(),
        "warnings": unique_strings([*warnings, *normalized.get("warnings", [])]),
    }


def api_football_key() -> str:
    for key in API_FOOTBALL_ENV_KEYS:
        value = env_value(key)
        if value and str(value).strip():
            return str(value).strip()
    return ""


def env_value(name: str) -> str:
    value = os.environ.get(name)
    if value not in {None, ""}:
        return str(value)
    return read_dotenv_file().get(name, "")


def read_dotenv_file(path: Optional[Path] = None) -> Dict[str, str]:
    path = path or DOTENV_FILE
    if not Path(path).exists():
        return {}
    values: Dict[str, str] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except Exception:
        return values
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def download_api_football_cache(
        api_key: str,
        force: bool = False,
        seasons: Optional[Iterable[int]] = None,
        league_ids: Optional[Iterable[int]] = None,
        session: Optional[requests.Session] = None,
) -> List[str]:
    API_FOOTBALL_RAW_ROOT.mkdir(parents=True, exist_ok=True)
    selected_seasons = [int(season) for season in (seasons or env_int_list("API_FOOTBALL_SEASONS") or DEFAULT_WORLD_CUP_SEASONS)]
    selected_leagues = [int(league) for league in (league_ids or env_int_list("API_FOOTBALL_LEAGUE_IDS") or [DEFAULT_WORLD_CUP_LEAGUE_ID])]
    downloaded: List[str] = []
    for league_id in selected_leagues:
        for season in selected_seasons:
            payload = cached_or_fetch(
                endpoint="/fixtures",
                params={"league": league_id, "season": season},
                api_key=api_key,
                force=force,
                session=session,
            )
            downloaded.append(payload)

    if env_flag("API_FOOTBALL_FETCH_FIXTURE_DETAILS"):
        max_details = max(int(env_value("API_FOOTBALL_MAX_DETAIL_FIXTURES") or 20), 0)
        fixtures = normalize_api_football_payloads(read_api_football_cache())["fixtures"]
        fixture_ids = fixtures["FixtureId"].dropna().astype(str).drop_duplicates().head(max_details).tolist() if not fixtures.empty else []
        for fixture_id in fixture_ids:
            for endpoint in ("/fixtures/statistics", "/fixtures/lineups", "/injuries", "/odds"):
                downloaded.append(cached_or_fetch(
                    endpoint=endpoint,
                    params={"fixture": fixture_id},
                    api_key=api_key,
                    force=force,
                    session=session,
                ))
    return downloaded


def cached_or_fetch(
        endpoint: str,
        params: Dict[str, Any],
        api_key: str,
        force: bool,
        session: Optional[requests.Session],
) -> str:
    path = api_cache_path(endpoint, params)
    if path.exists() and not force:
        return str(path)
    data = api_football_get(endpoint=endpoint, params=params, api_key=api_key, session=session)
    wrapped = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "params": params,
        "payload": data,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(wrapped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(path)


def api_football_get(
        endpoint: str,
        params: Dict[str, Any],
        api_key: str,
        session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    client = session or requests.Session()
    url = f"{API_FOOTBALL_BASE_URL}{endpoint if endpoint.startswith('/') else '/' + endpoint}"
    response = client.get(
        url,
        params=params,
        headers={"x-apisports-key": api_key},
        timeout=API_FOOTBALL_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ApiFootballProviderError("API-Football no devolvio JSON de objeto.")
    return data


def api_cache_path(endpoint: str, params: Dict[str, Any]) -> Path:
    param_text = "_".join(f"{normalize_token(key)}-{normalize_token(value)}" for key, value in sorted(params.items()))
    name = f"{normalize_token(endpoint)}_{param_text or 'all'}.json"
    return API_FOOTBALL_RAW_ROOT / name


def read_api_football_cache() -> List[Dict[str, Any]]:
    if not API_FOOTBALL_RAW_ROOT.exists():
        return []
    payloads: List[Dict[str, Any]] = []
    for path in sorted(API_FOOTBALL_RAW_ROOT.rglob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        data.setdefault("cache_file", str(path))
        payloads.append(data)
    return payloads


def normalize_api_football_payloads(payloads: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    fixture_parts: List[pd.DataFrame] = []
    statistics_parts: List[pd.DataFrame] = []
    lineup_parts: List[pd.DataFrame] = []
    injury_parts: List[pd.DataFrame] = []
    odds_parts: List[pd.DataFrame] = []
    warnings: List[str] = []
    for wrapper in payloads:
        payload = wrapper.get("payload", wrapper.get("response", wrapper))
        if not isinstance(payload, dict):
            continue
        endpoint = str(wrapper.get("endpoint") or "")
        params = wrapper.get("params") if isinstance(wrapper.get("params"), dict) else {}
        fetched_at = str(wrapper.get("fetched_at") or "")
        source = str(wrapper.get("cache_file") or endpoint or "api-football")
        items = payload.get("response", [])
        if not isinstance(items, list):
            warnings.append(f"API-Football cache ignorado: response no es lista ({source}).")
            continue
        kind = infer_payload_kind(endpoint, items)
        if kind == "fixtures":
            fixture_parts.append(parse_fixture_rows(items, fetched_at=fetched_at, source=source))
        elif kind == "statistics":
            statistics_parts.append(parse_statistics_rows(items, fixture_id=params.get("fixture"), fetched_at=fetched_at, source=source))
        elif kind == "lineups":
            lineup_parts.append(parse_lineup_rows(items, fixture_id=params.get("fixture"), fetched_at=fetched_at, source=source))
        elif kind == "injuries":
            injury_parts.append(parse_injury_rows(items, fixture_id=params.get("fixture"), fetched_at=fetched_at, source=source))
        elif kind == "odds":
            odds_parts.append(parse_odds_rows(items, fixture_id=params.get("fixture"), fetched_at=fetched_at, source=source))

    fixtures = dedupe_frame(pd.concat(fixture_parts, ignore_index=True) if fixture_parts else pd.DataFrame(), ["FixtureId"])
    statistics = dedupe_frame(pd.concat(statistics_parts, ignore_index=True) if statistics_parts else pd.DataFrame(), ["FixtureId", "Team"])
    lineups = dedupe_frame(pd.concat(lineup_parts, ignore_index=True) if lineup_parts else pd.DataFrame(), ["FixtureId", "Team"])
    injuries = dedupe_frame(pd.concat(injury_parts, ignore_index=True) if injury_parts else pd.DataFrame(), ["FixtureId", "Team", "Player"])
    odds = pd.concat(odds_parts, ignore_index=True) if odds_parts else pd.DataFrame()
    team_stats = build_team_match_stats(fixtures, statistics)
    market_rows = api_football_market_rows(fixtures, odds)
    return {
        "fixtures": fixtures,
        "statistics": statistics,
        "team_stats": team_stats,
        "lineups": lineups,
        "injuries": injuries,
        "odds": odds,
        "market_rows": market_rows,
        "warnings": unique_strings(warnings),
    }


def infer_payload_kind(endpoint: str, items: List[Any]) -> str:
    endpoint_key = endpoint.strip("/").lower()
    if "statistics" in endpoint_key:
        return "statistics"
    if "lineups" in endpoint_key:
        return "lineups"
    if "injuries" in endpoint_key:
        return "injuries"
    if "odds" in endpoint_key:
        return "odds"
    sample = next((item for item in items if isinstance(item, dict)), {})
    if "fixture" in sample and "teams" in sample:
        return "fixtures"
    if "statistics" in sample:
        return "statistics"
    if "startXI" in sample or "formation" in sample:
        return "lineups"
    if "player" in sample and "team" in sample:
        return "injuries"
    if "bookmakers" in sample:
        return "odds"
    return ""


def parse_fixture_rows(items: List[Dict[str, Any]], fetched_at: str, source: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fixture = item.get("fixture") if isinstance(item.get("fixture"), dict) else {}
        league = item.get("league") if isinstance(item.get("league"), dict) else {}
        teams = item.get("teams") if isinstance(item.get("teams"), dict) else {}
        goals = item.get("goals") if isinstance(item.get("goals"), dict) else {}
        home = team_name((teams.get("home") if isinstance(teams.get("home"), dict) else {}))
        away = team_name((teams.get("away") if isinstance(teams.get("away"), dict) else {}))
        if not home or not away:
            continue
        match_date = pd.to_datetime(fixture.get("date"), errors="coerce")
        hg = numeric_or_nan(goals.get("home"))
        ag = numeric_or_nan(goals.get("away"))
        rows.append({
            "FixtureId": str(fixture.get("id") or ""),
            "Date": match_date,
            "Year": int(match_date.year) if pd.notna(match_date) else numeric_or_nan(league.get("season")),
            "Home": home,
            "Away": away,
            "HG": hg,
            "AG": ag,
            "Label": label_from_goals(hg, ag),
            "OverUnder25": int((hg + ag) >= 3.0) if np.isfinite(hg) and np.isfinite(ag) else np.nan,
            "Round": str(league.get("round") or ""),
            "Group": group_from_round(league.get("round")),
            "LeagueId": str(league.get("id") or ""),
            "LeagueName": str(league.get("name") or ""),
            "Season": str(league.get("season") or ""),
            "Venue": venue_name(fixture.get("venue")),
            "Neutral": bool((teams.get("home") or {}).get("winner") is None and (teams.get("away") or {}).get("winner") is None),
            "Status": str((fixture.get("status") or {}).get("short") or ""),
            "Source": "api-football:fixtures",
            "fetched_at": fetched_at,
            "source": source,
        })
    return pd.DataFrame(rows)


def parse_statistics_rows(items: List[Dict[str, Any]], fixture_id: Any, fetched_at: str, source: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        team = item.get("team") if isinstance(item.get("team"), dict) else {}
        name = team_name(team)
        if not name:
            continue
        record = {
            "FixtureId": str(fixture_id or item.get("fixture") or ""),
            "Team": name,
            "TeamId": str(team.get("id") or ""),
            "fetched_at": fetched_at,
            "source": source,
        }
        for stat in item.get("statistics") or []:
            if not isinstance(stat, dict):
                continue
            key = api_stat_key(stat.get("type"))
            if not key:
                continue
            record[key] = numeric_or_nan(stat.get("value"))
        rows.append(record)
    return pd.DataFrame(rows)


def parse_lineup_rows(items: List[Dict[str, Any]], fixture_id: Any, fetched_at: str, source: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        team = item.get("team") if isinstance(item.get("team"), dict) else {}
        name = team_name(team)
        if not name:
            continue
        start_xi = item.get("startXI") if isinstance(item.get("startXI"), list) else []
        substitutes = item.get("substitutes") if isinstance(item.get("substitutes"), list) else []
        rows.append({
            "FixtureId": str(fixture_id or item.get("fixture") or ""),
            "Team": name,
            "TeamId": str(team.get("id") or ""),
            "Formation": str(item.get("formation") or ""),
            "StartXI": float(len(start_xi)),
            "Substitutes": float(len(substitutes)),
            "fetched_at": fetched_at,
            "source": source,
        })
    return pd.DataFrame(rows)


def parse_injury_rows(items: List[Dict[str, Any]], fixture_id: Any, fetched_at: str, source: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        team = item.get("team") if isinstance(item.get("team"), dict) else {}
        player = item.get("player") if isinstance(item.get("player"), dict) else {}
        fixture = item.get("fixture") if isinstance(item.get("fixture"), dict) else {}
        name = team_name(team)
        player_name = str(player.get("name") or "").strip()
        if not name or not player_name:
            continue
        rows.append({
            "FixtureId": str(fixture_id or fixture.get("id") or ""),
            "Date": pd.to_datetime(fixture.get("date"), errors="coerce"),
            "Team": name,
            "Player": player_name,
            "Reason": str(player.get("reason") or ""),
            "Type": str(player.get("type") or ""),
            "fetched_at": fetched_at,
            "source": source,
        })
    return pd.DataFrame(rows)


def parse_odds_rows(items: List[Dict[str, Any]], fixture_id: Any, fetched_at: str, source: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        fixture = item.get("fixture") if isinstance(item.get("fixture"), dict) else {}
        item_fixture_id = str(fixture_id or fixture.get("id") or "")
        for bookmaker in item.get("bookmakers") or []:
            if not isinstance(bookmaker, dict):
                continue
            bookmaker_name = str(bookmaker.get("name") or "")
            for bet in bookmaker.get("bets") or []:
                if not isinstance(bet, dict):
                    continue
                market = odds_market_key(bet.get("name"))
                if not market:
                    continue
                for value in bet.get("values") or []:
                    if not isinstance(value, dict):
                        continue
                    selection = odds_selection_key(market, value.get("value"))
                    odd = numeric_or_nan(value.get("odd"))
                    if not selection or not np.isfinite(odd) or odd <= 1.0:
                        continue
                    rows.append({
                        "FixtureId": item_fixture_id,
                        "Market": market,
                        "Selection": selection,
                        "Odd": odd,
                        "Bookmaker": bookmaker_name,
                        "fetched_at": fetched_at,
                        "source": source,
                    })
    return pd.DataFrame(rows)


def build_team_match_stats(fixtures: pd.DataFrame, statistics: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty:
        return pd.DataFrame(columns=["Team", "Opponent", "Date"])
    stats_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not statistics.empty:
        for _, row in statistics.iterrows():
            stats_by_key[(str(row.get("FixtureId") or ""), normalize_team_key(row.get("Team")))] = row.to_dict()
    rows: List[Dict[str, Any]] = []
    for _, match in fixtures.iterrows():
        hg = numeric_or_nan(match.get("HG"))
        ag = numeric_or_nan(match.get("AG"))
        if not np.isfinite(hg) or not np.isfinite(ag):
            continue
        fixture_id = str(match.get("FixtureId") or "")
        home = clean_team_name(match.get("Home"))
        away = clean_team_name(match.get("Away"))
        if not home or not away:
            continue
        home_stats = stats_by_key.get((fixture_id, normalize_team_key(home)), {})
        away_stats = stats_by_key.get((fixture_id, normalize_team_key(away)), {})
        rows.append(team_stat_row(match, home, away, hg, ag, "home", home_stats, away_stats))
        rows.append(team_stat_row(match, away, home, ag, hg, "away", away_stats, home_stats))
    return pd.DataFrame(rows).fillna(0.0)


def team_stat_row(
        match: pd.Series,
        team: str,
        opponent: str,
        gf: float,
        ga: float,
        side: str,
        team_stats: Dict[str, Any],
        opponent_stats: Dict[str, Any],
) -> Dict[str, Any]:
    record = {
        "Team": team,
        "Opponent": opponent,
        "Date": pd.to_datetime(match.get("Date"), errors="coerce"),
        "FixtureId": str(match.get("FixtureId") or ""),
        "Side": side,
        "GF": float(gf),
        "GA": float(ga),
        "GoalDiff": float(gf - ga),
        "Points": float(3 if gf > ga else 1 if gf == ga else 0),
        "Win": float(gf > ga),
        "Draw": float(gf == ga),
        "Loss": float(gf < ga),
        "Over25": float((gf + ga) >= 3.0),
        "Under25": float((gf + ga) < 3.0),
        "BTTS": float(gf > 0 and ga > 0),
        "CleanSheet": float(ga == 0),
        "Scored": float(gf > 0),
        "Source": "api-football:team-match-stats",
    }
    for raw_key in set(team_stats) | set(opponent_stats):
        key = normalize_column(raw_key)
        if key in {"fixtureid", "fixture_id", "team", "teamid", "team_id", "fetched_at", "source"}:
            continue
        team_value = numeric_or_nan(team_stats.get(raw_key))
        opponent_value = numeric_or_nan(opponent_stats.get(raw_key))
        if np.isfinite(team_value):
            record[f"{key}_for"] = float(team_value)
        if np.isfinite(opponent_value):
            record[f"{key}_against"] = float(opponent_value)
    return record


def api_football_feature_table(
        team_stats: pd.DataFrame,
        reference_date: str,
        teams: Optional[Iterable[str]] = None,
        lineups: Optional[pd.DataFrame] = None,
        injuries: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if team_stats is None or team_stats.empty:
        return pd.DataFrame(columns=["Team"])
    working = team_stats.copy()
    if "Team" not in working.columns or "Date" not in working.columns:
        return pd.DataFrame(columns=["Team"])
    working["Team"] = working["Team"].map(clean_team_name)
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    reference_ts = pd.to_datetime(reference_date, errors="coerce")
    if pd.notna(reference_ts):
        working = working[working["Date"].notna() & (working["Date"] < pd.Timestamp(reference_ts))].copy()
    else:
        working = working[working["Date"].notna()].copy()
    team_filter = {normalize_team_key(team) for team in teams or [] if str(team).strip()}
    if team_filter:
        working = working[working["Team"].map(normalize_team_key).isin(team_filter)].copy()
    if working.empty:
        return pd.DataFrame(columns=["Team"])

    rows: List[Dict[str, Any]] = []
    for team, frame in working.groupby("Team", sort=True):
        frame = frame.sort_values("Date", kind="stable").reset_index(drop=True)
        last_date = frame["Date"].max()
        days_since = float(max((pd.Timestamp(reference_ts) - last_date).days, 0)) if pd.notna(reference_ts) and pd.notna(last_date) else 0.0
        rest_days = frame["Date"].diff().dt.days.dropna()
        record: Dict[str, Any] = {
            "Team": team,
            "matches": float(frame.shape[0]),
            "days_since_last_match": days_since,
            "recent_match_volume_90d": float(frame[frame["Date"] >= (pd.Timestamp(reference_ts) - pd.Timedelta(days=90))].shape[0]) if pd.notna(reference_ts) else 0.0,
            "recent_match_volume_180d": float(frame[frame["Date"] >= (pd.Timestamp(reference_ts) - pd.Timedelta(days=180))].shape[0]) if pd.notna(reference_ts) else 0.0,
            "recent_match_volume_365d": float(frame[frame["Date"] >= (pd.Timestamp(reference_ts) - pd.Timedelta(days=365))].shape[0]) if pd.notna(reference_ts) else 0.0,
            "rest_days_avg": float(rest_days.mean()) if not rest_days.empty else 0.0,
            "rest_days_std": float(rest_days.std(ddof=0)) if not rest_days.empty else 0.0,
            "rest_days_last": float(rest_days.iloc[-1]) if not rest_days.empty else 0.0,
        }
        record.update(api_window_features(frame, window=len(frame), prefix="all"))
        for window in (3, 5, 10):
            record.update(api_window_features(frame, window=window, prefix=f"last_{window}"))
        record["trend_points_ppg_3_vs_10"] = record.get("last_3_points_ppg", 0.0) - record.get("last_10_points_ppg", 0.0)
        record["trend_goal_diff_3_vs_10"] = record.get("last_3_goal_diff_avg", 0.0) - record.get("last_10_goal_diff_avg", 0.0)
        record["trend_total_shots_3_vs_10"] = record.get("last_3_total_shots_for_avg", 0.0) - record.get("last_10_total_shots_for_avg", 0.0)
        record.update(api_numeric_stat_features(frame))
        rows.append(record)
    output = pd.DataFrame(rows).fillna(0.0)
    merge_context_counts(output, lineups, reference_date, prefix="lineup")
    merge_context_counts(output, injuries, reference_date, prefix="injury")
    return output.fillna(0.0)


def api_window_features(frame: pd.DataFrame, window: int, prefix: str) -> Dict[str, float]:
    recent = frame.tail(int(window)).copy()
    if recent.empty:
        return {}
    weights = np.linspace(1.0, 1.0 + max(len(recent) - 1, 0) * 0.10, num=len(recent))
    features = {
        f"{prefix}_points_ppg": safe_mean(recent.get("Points")),
        f"{prefix}_weighted_points": float(np.average(pd.to_numeric(recent.get("Points"), errors="coerce").fillna(0.0), weights=weights)),
        f"{prefix}_win_rate": safe_mean(recent.get("Win")),
        f"{prefix}_draw_rate": safe_mean(recent.get("Draw")),
        f"{prefix}_loss_rate": safe_mean(recent.get("Loss")),
        f"{prefix}_goals_for_avg": safe_mean(recent.get("GF")),
        f"{prefix}_goals_against_avg": safe_mean(recent.get("GA")),
        f"{prefix}_goal_diff_avg": safe_mean(recent.get("GoalDiff")),
        f"{prefix}_goal_diff_std": safe_std(recent.get("GoalDiff")),
        f"{prefix}_over25_rate": safe_mean(recent.get("Over25")),
        f"{prefix}_under25_rate": safe_mean(recent.get("Under25")),
        f"{prefix}_btts_rate": safe_mean(recent.get("BTTS")),
        f"{prefix}_clean_sheet_rate": safe_mean(recent.get("CleanSheet")),
    }
    for column in numeric_stat_columns(recent):
        features[f"{prefix}_{normalize_column(column)}_avg"] = safe_mean(recent[column])
        features[f"{prefix}_{normalize_column(column)}_last"] = safe_last(recent[column])
    return features


def api_numeric_stat_features(frame: pd.DataFrame) -> Dict[str, float]:
    features: Dict[str, float] = {}
    for column in numeric_stat_columns(frame):
        key = normalize_column(column)
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        features[f"{key}_avg"] = float(series.mean()) if not series.empty else 0.0
        features[f"{key}_std"] = float(series.std(ddof=0)) if not series.empty else 0.0
        features[f"{key}_last"] = float(series.iloc[-1]) if not series.empty else 0.0
    return features


def numeric_stat_columns(frame: pd.DataFrame) -> List[str]:
    excluded = {
        "GF", "GA", "GoalDiff", "Points", "Win", "Draw", "Loss", "Over25", "Under25", "BTTS",
        "CleanSheet", "Scored",
    }
    return [
        column for column in frame.columns
        if column not in excluded
        and not str(column).lower() in {"fixtureid", "fixture_id", "team", "opponent", "date", "side", "source"}
        and pd.api.types.is_numeric_dtype(pd.to_numeric(frame[column], errors="coerce"))
    ]


def merge_context_counts(output: pd.DataFrame, rows: Optional[pd.DataFrame], reference_date: str, prefix: str) -> None:
    output[f"{prefix}_context_available"] = 0.0
    output[f"{prefix}_rows"] = 0.0
    if rows is None or rows.empty or "Team" not in rows.columns:
        return
    working = rows.copy()
    working["Team"] = working["Team"].map(clean_team_name)
    reference_ts = pd.to_datetime(reference_date, errors="coerce")
    if "fetched_at" in working.columns and pd.notna(reference_ts):
        fetched = pd.to_datetime(working["fetched_at"], errors="coerce", utc=True).dt.tz_convert(None)
        working = working[fetched.notna() & (fetched <= pd.Timestamp(reference_ts))].copy()
    else:
        working = working.iloc[0:0].copy()
    if "Date" in working.columns and pd.notna(reference_ts):
        dates = pd.to_datetime(working["Date"], errors="coerce")
        working = working[dates.isna() | (dates >= pd.Timestamp(reference_ts))].copy()
    if working.empty:
        return
    counts = working.groupby(working["Team"].map(normalize_team_key)).size().to_dict()
    for index, row in output.iterrows():
        count = float(counts.get(normalize_team_key(row.get("Team")), 0.0))
        output.at[index, f"{prefix}_context_available"] = 1.0 if count else 0.0
        output.at[index, f"{prefix}_rows"] = count


def api_football_market_rows(fixtures: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Date", "Home", "Away", "market_odds_home", "market_odds_draw", "market_odds_away",
        "market_odds_over25", "market_odds_under25", "market_source",
    ]
    if fixtures.empty or odds.empty:
        return pd.DataFrame(columns=columns)
    fixture_lookup = {str(row.get("FixtureId") or ""): row.to_dict() for _, row in fixtures.iterrows()}
    rows: List[Dict[str, Any]] = []
    for fixture_id, scoped in odds.groupby(odds["FixtureId"].astype(str), sort=True):
        fixture = fixture_lookup.get(fixture_id)
        if not fixture:
            continue
        record = {
            "Date": fixture.get("Date"),
            "Home": fixture.get("Home"),
            "Away": fixture.get("Away"),
            "market_odds_home": average_selection(scoped, "1x2", "home"),
            "market_odds_draw": average_selection(scoped, "1x2", "draw"),
            "market_odds_away": average_selection(scoped, "1x2", "away"),
            "market_odds_over25": average_selection(scoped, "ou25", "over25"),
            "market_odds_under25": average_selection(scoped, "ou25", "under25"),
            "market_source": "api-football:odds",
        }
        rows.append(record)
    return pd.DataFrame(rows, columns=columns)


def average_selection(scoped: pd.DataFrame, market: str, selection: str) -> float:
    values = scoped[(scoped["Market"] == market) & (scoped["Selection"] == selection)]["Odd"]
    values = pd.to_numeric(values, errors="coerce")
    values = values[values > 1.0]
    return float(values.mean()) if not values.empty else np.nan


def dedupe_frame(frame: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    valid_subset = [column for column in subset if column in frame.columns]
    if valid_subset:
        frame = frame.drop_duplicates(subset=valid_subset, keep="last")
    return frame.reset_index(drop=True)


def api_stat_key(value: Any) -> str:
    key = normalize_column(value)
    return API_STAT_ALIASES.get(key, key if "expected" in key and "goal" in key else "")


def odds_market_key(value: Any) -> str:
    key = normalize_column(value)
    if key in {"match_winner", "fulltime_result", "1x2", "winner"}:
        return "1x2"
    if "over_under" in key or "goals_over_under" in key:
        return "ou25"
    return ""


def odds_selection_key(market: str, value: Any) -> str:
    key = normalize_column(value)
    if market == "1x2":
        if key in {"home", "1"}:
            return "home"
        if key in {"draw", "x"}:
            return "draw"
        if key in {"away", "2"}:
            return "away"
    if market == "ou25":
        if "over" in key and "2_5" in key:
            return "over25"
        if "under" in key and "2_5" in key:
            return "under25"
    return ""


def group_from_round(value: Any) -> str:
    text = str(value or "")
    match = re.search(r"group\s+([a-z])", text, flags=re.IGNORECASE)
    return f"Group {match.group(1).upper()}" if match else ""


def team_name(value: Dict[str, Any]) -> str:
    return clean_team_name(value.get("name") if isinstance(value, dict) else value)


def venue_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("city") or "")
    return ""


def label_from_goals(home_goals: Any, away_goals: Any) -> str:
    home = numeric_or_nan(home_goals)
    away = numeric_or_nan(away_goals)
    if not np.isfinite(home) or not np.isfinite(away):
        return ""
    if home > away:
        return "H"
    if away > home:
        return "A"
    return "D"


def safe_mean(values: Any) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna() if values is not None else pd.Series(dtype=float)
    return float(series.mean()) if not series.empty else 0.0


def safe_std(values: Any) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna() if values is not None else pd.Series(dtype=float)
    return float(series.std(ddof=0)) if not series.empty else 0.0


def safe_last(values: Any) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna() if values is not None else pd.Series(dtype=float)
    return float(series.iloc[-1]) if not series.empty else 0.0


def numeric_or_nan(value: Any) -> float:
    if isinstance(value, str):
        value = value.strip().replace("%", "")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def env_int_list(name: str) -> List[int]:
    raw = env_value(name)
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(float(item)))
        except ValueError:
            continue
    return values


def env_flag(name: str) -> bool:
    return str(env_value(name)).strip().lower() in {"1", "true", "yes", "si", "on"}


def normalize_column(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_team_key(value: Any) -> str:
    text = str(value or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "blank"


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
