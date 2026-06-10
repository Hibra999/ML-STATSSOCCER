from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.worldcup.data import clean_team_name


FOOTBALL_DATA_WORLD_CUP_URL = "https://www.football-data.co.uk/WorldCup2026.xlsx"
MARKET_ROOT = Path("storage") / "worldcup" / "market"
FOOTBALL_DATA_XLSX = MARKET_ROOT / "WorldCup2026.xlsx"
MANUAL_ODDS_CSV = MARKET_ROOT / "manual_odds.csv"
SCRAPED_ODDS_CSV = MARKET_ROOT / "scraped_odds.csv"
FOOTBALL_DATA_SHEETS = ("WorldCup2014", "WorldCup2018", "WorldCup2022", "WorldCup2026Qualifiers")
TOTAL_ODDS_LINES = ("05", "15", "25", "35", "45")
MANUAL_ODDS_COLUMNS = [
    "Date",
    "Home",
    "Away",
    "market_odds_home",
    "market_odds_draw",
    "market_odds_away",
    *[f"market_odds_over{suffix}" for suffix in TOTAL_ODDS_LINES],
    *[f"market_odds_under{suffix}" for suffix in TOTAL_ODDS_LINES],
    "market_source",
]


def load_market_data(
        force_download: bool = False,
        allow_download: bool = False,
        use_scraper: bool = False,
        scraper_urls: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    warnings: List[str] = []
    sources: List[str] = []
    frames: List[pd.DataFrame] = []

    try:
        workbook_path = ensure_football_data_workbook(force=force_download, allow_download=allow_download)
    except Exception as exc:
        workbook_path = FOOTBALL_DATA_XLSX if FOOTBALL_DATA_XLSX.exists() else None
        warnings.append(f"Football-Data XLSX no pudo descargarse ({exc.__class__.__name__}: {exc}).")
    if workbook_path:
        try:
            workbook = load_football_data_workbook(workbook_path)
            if not workbook.empty:
                frames.append(workbook)
                sources.append(str(workbook_path))
        except Exception as exc:
            warnings.append(f"Football-Data XLSX no pudo leerse ({exc.__class__.__name__}: {exc}).")
    elif not allow_download:
        warnings.append(f"Football-Data XLSX no disponible en cache: {FOOTBALL_DATA_XLSX}.")

    manual = load_manual_odds(MANUAL_ODDS_CSV)
    if not manual.empty:
        frames.append(manual)
        sources.append(str(MANUAL_ODDS_CSV))

    scraped_cache = load_manual_odds(SCRAPED_ODDS_CSV, default_source="scraped-cache")
    if not scraped_cache.empty:
        frames.append(scraped_cache)
        sources.append(str(SCRAPED_ODDS_CSV))

    if use_scraper:
        scraped, scrape_warnings = scrape_current_odds(scraper_urls=scraper_urls)
        warnings.extend(scrape_warnings)
        if not scraped.empty:
            frames.append(scraped)
            sources.append("selenium-scraper")
            MARKET_ROOT.mkdir(parents=True, exist_ok=True)
            scraped.to_csv(SCRAPED_ODDS_CSV, index=False)

    rows = normalize_market_frame(pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
    qualifiers = rows[rows.get("is_qualifier", pd.Series(dtype=bool)).fillna(False)].copy() if not rows.empty else pd.DataFrame()
    matches = rows[~rows.get("is_qualifier", pd.Series(dtype=bool)).fillna(False)].copy() if not rows.empty else pd.DataFrame()
    has_1x2 = bool(not matches.empty and matches[["market_odds_home", "market_odds_draw", "market_odds_away"]].notna().all(axis=1).any())
    has_ou25 = bool(not matches.empty and matches[["market_odds_over25", "market_odds_under25"]].notna().all(axis=1).any())
    status = "ok" if has_1x2 or has_ou25 or not qualifiers.empty else "missing"
    return {
        "rows": rows,
        "matches": matches,
        "qualifiers": qualifiers,
        "market_rows": int(matches.shape[0]),
        "qualifier_rows": int(qualifiers.shape[0]),
        "status": status,
        "has_1x2": has_1x2,
        "has_ou25": has_ou25,
        "sources": sources,
        "warnings": unique_strings(warnings),
        "loaded_at": datetime.now(timezone.utc).isoformat(),
    }


def ensure_football_data_workbook(force: bool = False, allow_download: bool = True) -> Optional[Path]:
    if FOOTBALL_DATA_XLSX.exists() and not force:
        return FOOTBALL_DATA_XLSX
    if not allow_download:
        return FOOTBALL_DATA_XLSX if FOOTBALL_DATA_XLSX.exists() else None
    MARKET_ROOT.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        FOOTBALL_DATA_WORLD_CUP_URL,
        timeout=12,
        headers={"User-Agent": "ML-STATSSOCCER/1.0"},
    )
    response.raise_for_status()
    FOOTBALL_DATA_XLSX.write_bytes(response.content)
    return FOOTBALL_DATA_XLSX


def load_football_data_workbook(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    excel = pd.ExcelFile(path)
    frames: List[pd.DataFrame] = []
    for sheet_name in FOOTBALL_DATA_SHEETS:
        if sheet_name not in excel.sheet_names:
            continue
        raw = pd.read_excel(excel, sheet_name=sheet_name)
        normalized = normalize_football_data_sheet(raw, sheet_name=sheet_name, source=str(path))
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return pd.DataFrame()
    return normalize_market_frame(pd.concat(frames, ignore_index=True))


def normalize_football_data_sheet(df: pd.DataFrame, sheet_name: str, source: str = "") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    working = df.copy()
    original_columns = list(working.columns)
    normalized_columns = {column: normalize_column(column) for column in original_columns}
    working = working.rename(columns=normalized_columns)

    home_col = first_existing(working.columns, ["home", "home_team", "team_1", "team1", "hteam", "local"])
    away_col = first_existing(working.columns, ["away", "away_team", "team_2", "team2", "ateam", "visitor"])
    date_col = first_existing(working.columns, ["date", "match_date", "fecha", "kickoff"])
    year_col = first_existing(working.columns, ["year", "season", "worldcup_year", "tournament_year"])
    round_col = first_existing(working.columns, ["round", "stage", "ronda"])
    group_col = first_existing(working.columns, ["group", "grupo"])
    home_goals_col = first_existing(working.columns, ["hg", "home_goals", "goals_home", "g1", "fthg", "score1"])
    away_goals_col = first_existing(working.columns, ["ag", "away_goals", "goals_away", "g2", "ftag", "score2"])
    if not home_col or not away_col:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    is_qualifier = "qualifier" in normalize_column(sheet_name)
    for index, row in working.iterrows():
        home = clean_team_name(row.get(home_col))
        away = clean_team_name(row.get(away_col))
        if not home or not away:
            continue
        match_date = pd.to_datetime(row.get(date_col), errors="coerce") if date_col else pd.NaT
        year = row.get(year_col, np.nan) if year_col else match_date.year if pd.notna(match_date) else inferred_year_from_sheet(sheet_name)
        odds_home, odds_draw, odds_away, odds_source = extract_1x2_odds(row, working.columns)
        odds_over, odds_under, totals_source = extract_totals_odds(row, working.columns)
        record = {
            "Date": match_date,
            "Year": year,
            "Home": home,
            "Away": away,
            "HG": row.get(home_goals_col, np.nan) if home_goals_col else np.nan,
            "AG": row.get(away_goals_col, np.nan) if away_goals_col else np.nan,
            "Round": row.get(round_col, "") if round_col else "",
            "Group": row.get(group_col, "") if group_col else "",
            "market_odds_home": odds_home,
            "market_odds_draw": odds_draw,
            "market_odds_away": odds_away,
            "market_odds_over25": odds_over,
            "market_odds_under25": odds_under,
            "market_source": odds_source or totals_source or f"football-data:{sheet_name}",
            "market_sheet": sheet_name,
            "is_qualifier": bool(is_qualifier),
            "source": source,
        }
        record.update(extract_match_stats(row))
        rows.append(record)
    return normalize_market_frame(pd.DataFrame(rows))


def load_manual_odds(path: Path, default_source: str = "manual") -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if raw.empty:
        return pd.DataFrame()
    working = raw.copy()
    for column in MANUAL_ODDS_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    working = working[MANUAL_ODDS_COLUMNS].copy()
    working["Home"] = working["Home"].map(clean_team_name)
    working["Away"] = working["Away"].map(clean_team_name)
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    working["Year"] = working["Date"].dt.year
    for column in ("market_odds_home", "market_odds_draw", "market_odds_away", *[f"market_odds_over{suffix}" for suffix in TOTAL_ODDS_LINES], *[f"market_odds_under{suffix}" for suffix in TOTAL_ODDS_LINES]):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["market_source"] = working["market_source"].replace("", np.nan).fillna(default_source)
    working["market_sheet"] = default_source
    working["is_qualifier"] = False
    working["source"] = str(path)
    return normalize_market_frame(working)


def scrape_current_odds(scraper_urls: Optional[Iterable[str]] = None) -> Tuple[pd.DataFrame, List[str]]:
    urls = [str(url).strip() for url in (scraper_urls or []) if str(url).strip()]
    if not urls:
        return pd.DataFrame(), ["Scraper Selenium omitido: no se configuraron URLs de odds actuales."]
    warnings: List[str] = []
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
    except Exception as exc:
        return pd.DataFrame(), [f"Scraper Selenium no disponible ({exc.__class__.__name__}: {exc})."]

    driver = None
    try:
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        driver = webdriver.Chrome(options=options)
        for url in urls:
            try:
                driver.get(url)
                warnings.append(f"Scraper Selenium visito {url}, pero no hay parser especifico estable para esa pagina; usa manual_odds.csv si necesitas odds actuales.")
            except Exception as exc:
                warnings.append(f"Scraper Selenium fallo en {url} ({exc.__class__.__name__}: {exc}).")
    except Exception as exc:
        warnings.append(f"Scraper Selenium fallo al iniciar ({exc.__class__.__name__}: {exc}).")
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass
    return pd.DataFrame(), warnings


def normalize_market_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Date", "Year", "Home", "Away", "HG", "AG", "Round", "Group",
        "market_odds_home", "market_odds_draw", "market_odds_away",
        *[f"market_odds_over{suffix}" for suffix in TOTAL_ODDS_LINES],
        *[f"market_odds_under{suffix}" for suffix in TOTAL_ODDS_LINES],
        "market_source", "market_sheet", "is_qualifier", "source",
        "home_xg", "away_xg", "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    working = df.copy()
    for column in columns:
        if column not in working.columns:
            working[column] = np.nan if column not in {"Home", "Away", "Round", "Group", "market_source", "market_sheet", "source", "is_qualifier"} else ""
    working["Home"] = working["Home"].map(clean_team_name)
    working["Away"] = working["Away"].map(clean_team_name)
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    working["Year"] = pd.to_numeric(working["Year"], errors="coerce").fillna(working["Date"].dt.year)
    for column in ("HG", "AG", "home_xg", "away_xg", "home_shots", "away_shots", "home_shots_on_target", "away_shots_on_target"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    for column in ("market_odds_home", "market_odds_draw", "market_odds_away", *[f"market_odds_over{suffix}" for suffix in TOTAL_ODDS_LINES], *[f"market_odds_under{suffix}" for suffix in TOTAL_ODDS_LINES]):
        working[column] = pd.to_numeric(working[column], errors="coerce")
        working.loc[working[column] <= 1.0, column] = np.nan
    working["market_source"] = working["market_source"].fillna("").astype(str)
    working["market_sheet"] = working["market_sheet"].fillna("").astype(str)
    working["is_qualifier"] = working["is_qualifier"].fillna(False).astype(bool)
    working = working[working["Home"].astype(str).str.len().gt(1) & working["Away"].astype(str).str.len().gt(1)].copy()
    working["_priority"] = working["market_source"].map(market_source_priority)
    working["_date_sort"] = working["Date"].fillna(pd.Timestamp("1900-01-01"))
    working = working.sort_values(["Home", "Away", "_date_sort", "_priority"], kind="stable").drop(columns=["_priority", "_date_sort"])
    return working.reset_index(drop=True)


def market_for_match(
        market_rows: pd.DataFrame,
        home: str,
        away: str,
        match_date: Optional[Any] = None,
        year: Optional[int] = None,
) -> Dict[str, Any]:
    if market_rows is None or market_rows.empty:
        return {}
    working = normalize_market_frame(market_rows)
    home_key = normalize_team_key(home)
    away_key = normalize_team_key(away)
    scoped = working[
        (working["Home"].map(normalize_team_key) == home_key) &
        (working["Away"].map(normalize_team_key) == away_key)
    ].copy()
    if scoped.empty:
        return {}
    date_ts = pd.to_datetime(match_date, errors="coerce") if match_date is not None else pd.NaT
    if pd.notna(date_ts):
        same_day = scoped[pd.to_datetime(scoped["Date"], errors="coerce").dt.date == pd.Timestamp(date_ts).date()].copy()
        if not same_day.empty:
            scoped = same_day
        else:
            return {}
    elif year is not None:
        years = pd.to_numeric(scoped["Year"], errors="coerce")
        same_year = scoped[years == int(year)].copy()
        if not same_year.empty:
            scoped = same_year
    scoped["_priority"] = scoped["market_source"].map(market_source_priority)
    return scoped.sort_values("_priority", ascending=False, kind="stable").iloc[0].to_dict()


def market_feature_defaults() -> Dict[str, float]:
    keys = {
        "market_has_1x2": 0.0,
        "market_has_ou25": 0.0,
        "market_odds_home": 0.0,
        "market_odds_draw": 0.0,
        "market_odds_away": 0.0,
        "market_implied_home": 0.0,
        "market_implied_draw": 0.0,
        "market_implied_away": 0.0,
        "market_prob_home": 0.0,
        "market_prob_draw": 0.0,
        "market_prob_away": 0.0,
        "market_logit_home": 0.0,
        "market_logit_draw": 0.0,
        "market_logit_away": 0.0,
        "market_vig_1x2": 0.0,
        "market_entropy_1x2": 0.0,
        "market_entropy_1x2_norm": 0.0,
        "market_max_prob_1x2": 0.0,
        "market_second_prob_1x2": 0.0,
        "market_gap_1x2": 0.0,
        "market_home_draw_gap": 0.0,
        "market_home_away_gap": 0.0,
        "market_draw_away_gap": 0.0,
        "market_favorite_home": 0.0,
        "market_favorite_draw": 0.0,
        "market_favorite_away": 0.0,
        "model_vs_market_home": 0.0,
        "model_vs_market_draw": 0.0,
        "model_vs_market_away": 0.0,
        "model_vs_market_home_abs": 0.0,
        "model_vs_market_draw_abs": 0.0,
        "model_vs_market_away_abs": 0.0,
        "model_vs_market_kl_1x2": 0.0,
        "market_vs_model_kl_1x2": 0.0,
        "model_market_entropy_delta_1x2": 0.0,
    }
    for suffix in TOTAL_ODDS_LINES:
        keys.update({
            f"market_odds_over{suffix}": 0.0,
            f"market_odds_under{suffix}": 0.0,
            f"market_has_ou{suffix}": 0.0,
            f"market_prob_over{suffix}": 0.0,
            f"market_prob_under{suffix}": 0.0,
            f"market_logit_over{suffix}": 0.0,
            f"market_logit_under{suffix}": 0.0,
            f"market_vig_ou{suffix}": 0.0,
            f"market_entropy_ou{suffix}": 0.0,
            f"market_entropy_ou{suffix}_norm": 0.0,
            f"market_ou{suffix}_gap": 0.0,
            f"model_vs_market_over{suffix}": 0.0,
            f"model_vs_market_under{suffix}": 0.0,
            f"model_vs_market_over{suffix}_abs": 0.0,
            f"model_vs_market_under{suffix}_abs": 0.0,
            f"model_vs_market_kl_ou{suffix}": 0.0,
            f"market_vs_model_kl_ou{suffix}": 0.0,
            f"model_market_entropy_delta_ou{suffix}": 0.0,
        })
    return keys


def market_feature_row(
        market_row: Optional[Dict[str, Any]],
        model_probs: Dict[str, float],
        model_totals: Dict[str, float],
) -> Dict[str, float]:
    features = market_feature_defaults()
    market_row = market_row or {}
    odds_home = as_float(market_row.get("market_odds_home"))
    odds_draw = as_float(market_row.get("market_odds_draw"))
    odds_away = as_float(market_row.get("market_odds_away"))
    features.update({
        "market_odds_home": valid_decimal_odd(odds_home),
        "market_odds_draw": valid_decimal_odd(odds_draw),
        "market_odds_away": valid_decimal_odd(odds_away),
    })
    for suffix in TOTAL_ODDS_LINES:
        features[f"market_odds_over{suffix}"] = valid_decimal_odd(as_float(market_row.get(f"market_odds_over{suffix}")))
        features[f"market_odds_under{suffix}"] = valid_decimal_odd(as_float(market_row.get(f"market_odds_under{suffix}")))
    if all(valid_decimal_odd(value) > 0.0 for value in (odds_home, odds_draw, odds_away)):
        implied, no_vig, vig = no_vig_probabilities({"home": odds_home, "draw": odds_draw, "away": odds_away})
        probs = [no_vig["home"], no_vig["draw"], no_vig["away"]]
        top = sorted(probs, reverse=True)
        favorite_index = int(np.argmax(probs))
        model = normalize_distribution({
            "home": float(model_probs.get("H", model_probs.get("home", 0.0))),
            "draw": float(model_probs.get("D", model_probs.get("draw", 0.0))),
            "away": float(model_probs.get("A", model_probs.get("away", 0.0))),
        })
        market = normalize_distribution(no_vig)
        market_entropy = entropy(probs)
        model_entropy = entropy([model["home"], model["draw"], model["away"]])
        features.update({
            "market_has_1x2": 1.0,
            "market_implied_home": implied["home"],
            "market_implied_draw": implied["draw"],
            "market_implied_away": implied["away"],
            "market_prob_home": no_vig["home"],
            "market_prob_draw": no_vig["draw"],
            "market_prob_away": no_vig["away"],
            "market_logit_home": logit(no_vig["home"]),
            "market_logit_draw": logit(no_vig["draw"]),
            "market_logit_away": logit(no_vig["away"]),
            "market_vig_1x2": vig,
            "market_entropy_1x2": market_entropy,
            "market_entropy_1x2_norm": normalized_entropy(probs),
            "market_max_prob_1x2": top[0],
            "market_second_prob_1x2": top[1],
            "market_gap_1x2": top[0] - top[1],
            "market_home_draw_gap": no_vig["home"] - no_vig["draw"],
            "market_home_away_gap": no_vig["home"] - no_vig["away"],
            "market_draw_away_gap": no_vig["draw"] - no_vig["away"],
            "market_favorite_home": 1.0 if favorite_index == 0 else 0.0,
            "market_favorite_draw": 1.0 if favorite_index == 1 else 0.0,
            "market_favorite_away": 1.0 if favorite_index == 2 else 0.0,
            "model_vs_market_home": model["home"] - market["home"],
            "model_vs_market_draw": model["draw"] - market["draw"],
            "model_vs_market_away": model["away"] - market["away"],
            "model_vs_market_home_abs": abs(model["home"] - market["home"]),
            "model_vs_market_draw_abs": abs(model["draw"] - market["draw"]),
            "model_vs_market_away_abs": abs(model["away"] - market["away"]),
            "model_vs_market_kl_1x2": kl_divergence(model, market),
            "market_vs_model_kl_1x2": kl_divergence(market, model),
            "model_market_entropy_delta_1x2": model_entropy - market_entropy,
        })
    for suffix in TOTAL_ODDS_LINES:
        odds_over = as_float(market_row.get(f"market_odds_over{suffix}"))
        odds_under = as_float(market_row.get(f"market_odds_under{suffix}"))
        if not all(valid_decimal_odd(value) > 0.0 for value in (odds_over, odds_under)):
            continue
        over_key = f"over{suffix}"
        under_key = f"under{suffix}"
        implied, no_vig, vig = no_vig_probabilities({over_key: odds_over, under_key: odds_under})
        probs = [no_vig[over_key], no_vig[under_key]]
        model = normalize_distribution({
            over_key: float(model_totals.get(over_key, 0.0)),
            under_key: float(model_totals.get(under_key, 0.0)),
        })
        market = normalize_distribution(no_vig)
        market_entropy = entropy(probs)
        model_entropy = entropy([model[over_key], model[under_key]])
        features.update({
            f"market_has_ou{suffix}": 1.0,
            f"market_prob_over{suffix}": no_vig[over_key],
            f"market_prob_under{suffix}": no_vig[under_key],
            f"market_logit_over{suffix}": logit(no_vig[over_key]),
            f"market_logit_under{suffix}": logit(no_vig[under_key]),
            f"market_vig_ou{suffix}": vig,
            f"market_entropy_ou{suffix}": market_entropy,
            f"market_entropy_ou{suffix}_norm": normalized_entropy(probs),
            f"market_ou{suffix}_gap": no_vig[over_key] - no_vig[under_key],
            f"model_vs_market_over{suffix}": model[over_key] - market[over_key],
            f"model_vs_market_under{suffix}": model[under_key] - market[under_key],
            f"model_vs_market_over{suffix}_abs": abs(model[over_key] - market[over_key]),
            f"model_vs_market_under{suffix}_abs": abs(model[under_key] - market[under_key]),
            f"model_vs_market_kl_ou{suffix}": kl_divergence(model, market),
            f"market_vs_model_kl_ou{suffix}": kl_divergence(market, model),
            f"model_market_entropy_delta_ou{suffix}": model_entropy - market_entropy,
        })
    return features


def qualifier_feature_table(
        qualifier_rows: pd.DataFrame,
        reference_date: str,
        teams: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    if qualifier_rows is None or qualifier_rows.empty:
        return pd.DataFrame(columns=["Team"])
    working = normalize_market_frame(qualifier_rows)
    working = working[working["is_qualifier"]].copy()
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    reference_ts = pd.to_datetime(reference_date, errors="coerce")
    if pd.notna(reference_ts):
        working = working[working["Date"].notna() & (working["Date"] < pd.Timestamp(reference_ts))].copy()
    if working.empty:
        return pd.DataFrame(columns=["Team"])
    team_filter = {normalize_team_key(team) for team in teams or [] if str(team).strip()}
    rows: List[Dict[str, Any]] = []
    for _, row in working.iterrows():
        rows.extend(qualifier_team_rows(row))
    team_df = pd.DataFrame(rows)
    if team_df.empty:
        return pd.DataFrame(columns=["Team"])
    if team_filter:
        team_df = team_df[team_df["Team"].map(normalize_team_key).isin(team_filter)].copy()
    if team_df.empty:
        return pd.DataFrame(columns=["Team"])
    opponent_ppg = team_df.groupby("Team")["Points"].mean().to_dict()
    records: List[Dict[str, Any]] = []
    for team, frame in team_df.groupby("Team", sort=True):
        frame = frame.sort_values("Date", kind="stable").copy()
        weights = np.linspace(1.0, 1.0 + max(len(frame) - 1, 0) * 0.08, num=len(frame))
        opp_strength = [float(opponent_ppg.get(opponent, 0.0)) for opponent in frame["Opponent"]]
        record = {
            "Team": team,
            "qualifier_context_available": 1.0,
            "qualifier_matches": float(frame.shape[0]),
            "qualifier_points_ppg": float(frame["Points"].mean()),
            "qualifier_weighted_points": float(np.average(frame["Points"], weights=weights)),
            "qualifier_win_rate": float(frame["Win"].mean()),
            "qualifier_draw_rate": float(frame["Draw"].mean()),
            "qualifier_loss_rate": float(frame["Loss"].mean()),
            "qualifier_gf_avg": float(frame["GF"].mean()),
            "qualifier_ga_avg": float(frame["GA"].mean()),
            "qualifier_goal_diff_avg": float(frame["GoalDiff"].mean()),
            "qualifier_goal_diff_std": float(frame["GoalDiff"].std(ddof=0)),
            "qualifier_over25_rate": float(frame["Over25"].mean()),
            "qualifier_btts_rate": float(frame["BTTS"].mean()),
            "qualifier_clean_sheet_rate": float(frame["CleanSheet"].mean()),
            "qualifier_sos_ppg": float(np.mean(opp_strength)) if opp_strength else 0.0,
            "qualifier_days_since_last": float(max((pd.Timestamp(reference_ts) - frame["Date"].max()).days, 0)) if pd.notna(reference_ts) else 0.0,
        }
        for source_col, feature_col in (
            ("XG", "xg"),
            ("XGA", "xga"),
            ("Shots", "shots"),
            ("ShotsAgainst", "shots_against"),
            ("ShotsOnTarget", "shots_on_target"),
            ("ShotsOnTargetAgainst", "shots_on_target_against"),
        ):
            values = pd.to_numeric(frame[source_col], errors="coerce").dropna()
            record[f"qualifier_{feature_col}_avg"] = float(values.mean()) if not values.empty else 0.0
            record[f"qualifier_{feature_col}_last"] = float(values.iloc[-1]) if not values.empty else 0.0
        records.append(record)
    return pd.DataFrame(records).fillna(0.0)


def qualifier_team_rows(row: pd.Series) -> List[Dict[str, Any]]:
    try:
        hg = float(row.get("HG"))
        ag = float(row.get("AG"))
    except (TypeError, ValueError):
        return []
    if not np.isfinite(hg) or not np.isfinite(ag):
        return []
    home = clean_team_name(row.get("Home"))
    away = clean_team_name(row.get("Away"))
    if not home or not away:
        return []
    date_value = pd.to_datetime(row.get("Date"), errors="coerce")
    home_stats = {
        "XG": as_float(row.get("home_xg")),
        "XGA": as_float(row.get("away_xg")),
        "Shots": as_float(row.get("home_shots")),
        "ShotsAgainst": as_float(row.get("away_shots")),
        "ShotsOnTarget": as_float(row.get("home_shots_on_target")),
        "ShotsOnTargetAgainst": as_float(row.get("away_shots_on_target")),
    }
    away_stats = {
        "XG": as_float(row.get("away_xg")),
        "XGA": as_float(row.get("home_xg")),
        "Shots": as_float(row.get("away_shots")),
        "ShotsAgainst": as_float(row.get("home_shots")),
        "ShotsOnTarget": as_float(row.get("away_shots_on_target")),
        "ShotsOnTargetAgainst": as_float(row.get("home_shots_on_target")),
    }
    return [
        qualifier_team_row(home, away, date_value, hg, ag, **home_stats),
        qualifier_team_row(away, home, date_value, ag, hg, **away_stats),
    ]


def qualifier_team_row(
        team: str,
        opponent: str,
        date_value: Any,
        gf: float,
        ga: float,
        **stats: float,
) -> Dict[str, Any]:
    return {
        "Team": team,
        "Opponent": opponent,
        "Date": pd.to_datetime(date_value, errors="coerce"),
        "GF": float(gf),
        "GA": float(ga),
        "GoalDiff": float(gf - ga),
        "Points": float(3 if gf > ga else 1 if gf == ga else 0),
        "Win": float(gf > ga),
        "Draw": float(gf == ga),
        "Loss": float(gf < ga),
        "Over25": float((gf + ga) >= 3.0),
        "BTTS": float(gf > 0 and ga > 0),
        "CleanSheet": float(ga == 0),
        **stats,
    }


def no_vig_probabilities(odds: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float], float]:
    implied = {key: (1.0 / float(value) if valid_decimal_odd(value) > 0.0 else 0.0) for key, value in odds.items()}
    total = sum(implied.values())
    no_vig = {key: (value / total if total > 0.0 else 0.0) for key, value in implied.items()}
    return implied, no_vig, float(total - 1.0) if total > 0.0 else 0.0


def entropy(values: Iterable[float]) -> float:
    probs = [max(float(value), 1e-12) for value in values]
    return float(-sum(prob * math.log(prob) for prob in probs))


def normalized_entropy(values: Iterable[float]) -> float:
    probs = [max(float(value), 1e-12) for value in values]
    if len(probs) <= 1:
        return 0.0
    return float(entropy(probs) / math.log(len(probs)))


def kl_divergence(left: Dict[str, float], right: Dict[str, float]) -> float:
    keys = sorted(set(left) | set(right))
    left_norm = normalize_distribution({key: left.get(key, 0.0) for key in keys})
    right_norm = normalize_distribution({key: right.get(key, 0.0) for key in keys})
    total = 0.0
    for key in keys:
        p = max(float(left_norm.get(key, 0.0)), 1e-12)
        q = max(float(right_norm.get(key, 0.0)), 1e-12)
        total += p * math.log(p / q)
    return float(total)


def logit(value: float) -> float:
    p = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return float(math.log(p / (1.0 - p)))


def normalize_distribution(values: Dict[str, float]) -> Dict[str, float]:
    clean = {key: max(float(value), 0.0) for key, value in values.items()}
    total = sum(clean.values())
    if total <= 0.0:
        size = max(len(clean), 1)
        return {key: 1.0 / size for key in clean}
    return {key: value / total for key, value in clean.items()}


def extract_1x2_odds(row: pd.Series, columns: Iterable[str]) -> Tuple[float, float, float, str]:
    home_avg = first_numeric(row, ["h_avg", "avg_h", "avgh", "average_h", "average_home"])
    draw_avg = first_numeric(row, ["d_avg", "avg_d", "avgd", "average_d", "average_draw"])
    away_avg = first_numeric(row, ["a_avg", "avg_a", "avga", "average_a", "average_away"])
    if all(valid_decimal_odd(value) > 0.0 for value in (home_avg, draw_avg, away_avg)):
        return home_avg, draw_avg, away_avg, "football-data:avg"

    home_max = first_numeric(row, ["h_max", "max_h", "maxh", "maximum_h", "maximum_home"])
    draw_max = first_numeric(row, ["d_max", "max_d", "maxd", "maximum_d", "maximum_draw"])
    away_max = first_numeric(row, ["a_max", "max_a", "maxa", "maximum_a", "maximum_away"])
    if all(valid_decimal_odd(value) > 0.0 for value in (home_max, draw_max, away_max)):
        return home_max, draw_max, away_max, "football-data:max"

    home_cols, draw_cols, away_cols = bookmaker_1x2_columns(columns)
    home = average_odds(row, home_cols)
    draw = average_odds(row, draw_cols)
    away = average_odds(row, away_cols)
    if all(valid_decimal_odd(value) > 0.0 for value in (home, draw, away)):
        return home, draw, away, "football-data:bookmaker"
    return np.nan, np.nan, np.nan, ""


def extract_totals_odds(row: pd.Series, columns: Iterable[str]) -> Tuple[float, float, str]:
    over = first_numeric(row, ["over25", "over_25", "o25", "o_25", "o2_5", "over_2_5", "market_odds_over25"])
    under = first_numeric(row, ["under25", "under_25", "u25", "u_25", "u2_5", "under_2_5", "market_odds_under25"])
    if all(valid_decimal_odd(value) > 0.0 for value in (over, under)):
        return over, under, "football-data:totals"
    over_cols = [column for column in columns if normalize_column(column) in {"o25", "o_25", "o2_5", "over25", "over_25"}]
    under_cols = [column for column in columns if normalize_column(column) in {"u25", "u_25", "u2_5", "under25", "under_25"}]
    over = average_odds(row, over_cols)
    under = average_odds(row, under_cols)
    if all(valid_decimal_odd(value) > 0.0 for value in (over, under)):
        return over, under, "football-data:totals-bookmaker"
    return np.nan, np.nan, ""


def bookmaker_1x2_columns(columns: Iterable[str]) -> Tuple[List[str], List[str], List[str]]:
    home_cols: List[str] = []
    draw_cols: List[str] = []
    away_cols: List[str] = []
    excluded = {"home", "away", "date", "year", "hg", "ag", "g1", "g2", "group", "round", "hteam", "ateam"}
    for column in columns:
        normalized = normalize_column(column)
        if normalized in excluded or normalized.startswith(("market_", "home_", "away_")):
            continue
        if normalized.endswith("h"):
            home_cols.append(column)
        elif normalized.endswith("d"):
            draw_cols.append(column)
        elif normalized.endswith("a"):
            away_cols.append(column)
    return home_cols, draw_cols, away_cols


def extract_match_stats(row: pd.Series) -> Dict[str, float]:
    return {
        "home_xg": first_numeric(row, ["home_xg", "hxg", "xgh", "xg_home", "h_xg", "xg1"]),
        "away_xg": first_numeric(row, ["away_xg", "axg", "xga", "xg_away", "a_xg", "xg2"]),
        "home_shots": first_numeric(row, ["hs", "home_shots", "shots_home", "h_shots"]),
        "away_shots": first_numeric(row, ["as", "away_shots", "shots_away", "a_shots"]),
        "home_shots_on_target": first_numeric(row, ["hst", "home_shots_on_target", "shots_on_target_home", "h_sot"]),
        "away_shots_on_target": first_numeric(row, ["ast", "away_shots_on_target", "shots_on_target_away", "a_sot"]),
    }


def first_numeric(row: pd.Series, candidates: Iterable[str]) -> float:
    for candidate in candidates:
        key = normalize_column(candidate)
        if key not in row.index:
            continue
        value = as_float(row.get(key))
        if np.isfinite(value):
            return value
    return np.nan


def average_odds(row: pd.Series, columns: Iterable[str]) -> float:
    values = [valid_decimal_odd(as_float(row.get(column))) for column in columns]
    values = [value for value in values if value > 0.0]
    return float(np.mean(values)) if values else np.nan


def valid_decimal_odd(value: Any) -> float:
    number = as_float(value)
    return float(number) if np.isfinite(number) and number > 1.0 else 0.0


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str:
    column_set = set(columns)
    for candidate in candidates:
        normalized = normalize_column(candidate)
        if normalized in column_set:
            return normalized
    return ""


def inferred_year_from_sheet(sheet_name: str) -> float:
    match = re.search(r"(19|20)\d{2}", str(sheet_name))
    return float(match.group(0)) if match else np.nan


def market_source_priority(source: Any) -> int:
    text = str(source or "").lower()
    if "manual" in text:
        return 30
    if "scraped" in text or "selenium" in text:
        return 20
    if "avg" in text:
        return 12
    if "max" in text:
        return 11
    if "football-data" in text:
        return 10
    return 0


def normalize_column(value: Any) -> str:
    text = str(value or "").strip().lower().replace("%", "pct")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_team_key(value: Any) -> str:
    text = str(value or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def unique_strings(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output
