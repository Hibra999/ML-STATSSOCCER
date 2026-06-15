from __future__ import annotations

import math
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.worldcup.model import HOST_TEAMS, TEAM_RATING_PRIORS, TOTAL_GOAL_LINES, poisson_score_grid, total_line_suffix


INTERNATIONAL_DATASET_SLUG = "martj42/international_results"
INTERNATIONAL_KAGGLE_DATASET_SLUG = "patateriedata/all-international-football-results"
INTERNATIONAL_GITHUB_RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/refs/heads/master/results.csv"
INTERNATIONAL_ROOT = Path("storage") / "worldcup" / "international"
INTERNATIONAL_MATCHES_FILE = INTERNATIONAL_ROOT / "all_matches.csv"
RECENT_MATCH_LIMIT = 15
INTERNATIONAL_FRESHNESS_MAX_AGE_DAYS = 30
INTERNATIONAL_MIN_CURRENT_SCORED_DATE = "2026-01-01"
CONTEXT_TOTAL_GOAL_LINES = tuple(line for line in TOTAL_GOAL_LINES if line <= 3.5)

TEAM_ALIASES = {
    "USA": "United States",
    "United States of America": "United States",
    "Czech Republic": "Czechia",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}
RATING_ALIAS_FALLBACKS = {
    "United States": "USA",
    "Czechia": "Czech Republic",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
}
RECENT15_NUMERIC_COLUMNS = [
    "recent15_matches",
    "recent15_gf_avg",
    "recent15_ga_avg",
    "recent15_goal_diff_avg",
    "recent15_weighted_gf_avg",
    "recent15_weighted_ga_avg",
    "recent15_weighted_goal_diff_avg",
    "recent15_adjusted_gf_avg",
    "recent15_adjusted_ga_avg",
    "recent15_adjusted_goal_diff_avg",
    "recent15_points_avg",
    "recent15_weighted_points_avg",
    "recent15_win_rate",
    "recent15_draw_rate",
    "recent15_loss_rate",
    "recent15_official_matches",
    "recent15_friendly_matches",
    "recent15_official_gf_avg",
    "recent15_official_ga_avg",
    "recent15_friendly_gf_avg",
    "recent15_friendly_ga_avg",
    "recent15_opponent_rating_avg",
    "recent15_weighted_opponent_rating_avg",
    "recent15_weight_sum",
    "recent15_days_since_last_match",
    "recent15_goal_total_avg",
    "recent15_over25_rate",
    "recent15_btts_rate",
]
INTERNATIONAL_REQUIRED_COLUMNS = ("date", "home_team", "away_team", "home_score", "away_score")
INTERNATIONAL_COLUMN_ALIASES = {
    "date": ("date", "match_date", "game_date", "fixture_date"),
    "home_team": ("home_team", "home", "home_name", "home_country", "home_team_name", "team1", "team_1", "team_a", "local"),
    "away_team": ("away_team", "away", "away_name", "away_country", "away_team_name", "team2", "team_2", "team_b", "visitor"),
    "home_score": ("home_score", "home_goals", "home_goal", "home_ft", "home_score_ft", "score_home", "goals_home", "hg", "g1"),
    "away_score": ("away_score", "away_goals", "away_goal", "away_ft", "away_score_ft", "score_away", "goals_away", "ag", "g2"),
    "tournament": ("tournament", "competition", "competition_name", "tournament_name", "event"),
    "country": ("country", "host_country", "venue_country"),
    "neutral": ("neutral", "is_neutral", "neutral_site", "neutral_venue"),
}


def download_international_results(force: bool = False, source: str = "auto") -> Dict[str, Any]:
    canonical_valid = False
    if INTERNATIONAL_MATCHES_FILE.exists() and not force:
        status = international_results_status()
        canonical_valid = bool(status.get("available") and Path(str(status.get("source_path") or "")) == INTERNATIONAL_MATCHES_FILE)
        if canonical_valid:
            return status

    source_key = str(source or "auto").strip().lower()
    github_error = ""
    if source_key in {"auto", "github", "martj42"}:
        try:
            return download_international_results_from_github(force=force)
        except Exception as exc:
            github_error = f"{exc.__class__.__name__}: {exc}"
            if source_key in {"github", "martj42"}:
                raise
    status = download_international_results_from_kaggle(force=force)
    if github_error:
        status["warning"] = "GitHub martj42 no disponible; se uso fallback Kaggle. " + str(status.get("warning") or github_error)
        status["github_error"] = github_error
    return status


def download_international_results_from_github(force: bool = False) -> Dict[str, Any]:
    INTERNATIONAL_ROOT.mkdir(parents=True, exist_ok=True)
    response = requests.get(INTERNATIONAL_GITHUB_RESULTS_URL, timeout=30)
    response.raise_for_status()
    candidate_path = INTERNATIONAL_ROOT / "results.csv"
    candidate_path.write_bytes(response.content)
    matches, reason = read_normalized_international_csv(candidate_path)
    if matches.empty:
        raise RuntimeError(f"GitHub martj42 no entrego un CSV internacional valido: {reason}")
    if force or not INTERNATIONAL_MATCHES_FILE.exists() or candidate_path.absolute() != INTERNATIONAL_MATCHES_FILE.absolute():
        shutil.copy2(candidate_path, INTERNATIONAL_MATCHES_FILE)
    status = international_results_status()
    status["downloaded_path"] = INTERNATIONAL_GITHUB_RESULTS_URL
    status["copied_files"] = [str(INTERNATIONAL_MATCHES_FILE)]
    status["source_file"] = str(candidate_path)
    status["scanned_files"] = 1
    status["provider"] = "github:martj42/international_results"
    return status


def download_international_results_from_kaggle(force: bool = False) -> Dict[str, Any]:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub no esta instalado. Ejecuta pip install -r requirements.txt.") from exc

    source_path = Path(kagglehub.dataset_download(INTERNATIONAL_KAGGLE_DATASET_SLUG))
    if not source_path.exists():
        raise RuntimeError(f"Kaggle no devolvio una ruta valida para {INTERNATIONAL_KAGGLE_DATASET_SLUG}.")
    INTERNATIONAL_ROOT.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    invalid_files: List[str] = []
    candidates = discover_international_csv_files(source_path)
    selected_source = ""
    for path in candidates:
        matches, reason = read_normalized_international_csv(path)
        if matches.empty:
            invalid_files.append(f"{path}: {reason}")
            continue
        target = INTERNATIONAL_MATCHES_FILE
        if path.absolute() != target.absolute() and (force or not target.exists() or not canonical_valid):
            shutil.copy2(path, target)
        elif path.absolute() == target.absolute():
            pass
        elif not force and target.exists():
            pass
        copied.append(str(target))
        selected_source = str(path)
        break
    status = international_results_status()
    status["downloaded_path"] = str(source_path)
    status["copied_files"] = copied
    status["source_file"] = selected_source
    status["scanned_files"] = len(candidates)
    if invalid_files:
        status["invalid_files"] = invalid_files[:8]
    if selected_source and Path(selected_source).name != "all_matches.csv":
        status["warning"] = f"Kaggle no entrego all_matches.csv como nombre principal; se uso {Path(selected_source).name} y se guardo como all_matches.csv."
    elif not selected_source:
        status["warning"] = f"No se encontro un CSV internacional valido en {source_path}."
    return status


def international_results_status() -> Dict[str, Any]:
    matches = load_international_matches(required=False)
    teams = set()
    if not matches.empty:
        teams.update(matches["home_team"].dropna().astype(str).tolist())
        teams.update(matches["away_team"].dropna().astype(str).tolist())
    source_path = str(matches.attrs.get("source_path") or "")
    reason = str(matches.attrs.get("reason") or "")
    warning = str(matches.attrs.get("warning") or "")
    expected_exists = INTERNATIONAL_MATCHES_FILE.exists()
    available = bool(not matches.empty)
    worldcup_rows = int(matches["tournament"].map(is_worldcup_tournament).sum()) if available and "tournament" in matches.columns else 0
    friendly_rows = int(matches["tournament"].map(is_friendly_tournament).sum()) if available and "tournament" in matches.columns else 0
    max_scored_date = date_to_string(matches["date"].max()) if available and "date" in matches.columns else ""
    max_dataset_date = str(matches.attrs.get("max_dataset_date") or max_scored_date)
    freshness_warning = international_freshness_warning(max_scored_date)
    status = {
        "dataset_slug": INTERNATIONAL_DATASET_SLUG,
        "github_url": INTERNATIONAL_GITHUB_RESULTS_URL,
        "fallback_dataset_slug": INTERNATIONAL_KAGGLE_DATASET_SLUG,
        "local_path": str(INTERNATIONAL_ROOT),
        "file_path": str(INTERNATIONAL_MATCHES_FILE),
        "source_path": source_path,
        "exists": bool(expected_exists),
        "available": available,
        "rows": int(matches.shape[0]),
        "all_matches_rows": int(matches.shape[0]) if available else 0,
        "scored_rows": int(matches.shape[0]) if available else 0,
        "raw_rows": int(matches.attrs.get("raw_rows", matches.shape[0] if available else 0) or 0),
        "unscored_rows": int(matches.attrs.get("unscored_rows", 0) or 0),
        "future_unscored_rows": int(matches.attrs.get("future_unscored_rows", 0) or 0),
        "max_dataset_date": max_dataset_date,
        "max_scored_date": max_scored_date,
        "worldcup_rows": worldcup_rows,
        "official_rows": int(matches.shape[0] - friendly_rows) if available else 0,
        "friendly_rows": friendly_rows,
        "teams": int(len(teams)),
    }
    if not available:
        status["reason"] = reason or f"No existe {INTERNATIONAL_MATCHES_FILE}."
    warnings = [warning, freshness_warning]
    warnings = [item for item in warnings if item]
    if warnings:
        status["warning"] = " ".join(warnings)
        status["warnings"] = warnings
    elif available and source_path and Path(source_path) != INTERNATIONAL_MATCHES_FILE:
        status["warning"] = f"Usando {source_path}; se recomienda descargar/guardar el artifact canonico en {INTERNATIONAL_MATCHES_FILE}."
    return status


def load_international_matches(required: bool = False) -> pd.DataFrame:
    candidates = local_international_candidate_files()
    reasons: List[str] = []
    for path in candidates:
        matches, reason = read_normalized_international_csv(path)
        if not matches.empty:
            if Path(path) != INTERNATIONAL_MATCHES_FILE:
                matches.attrs["warning"] = f"Usando {path}; falta el artifact canonico {INTERNATIONAL_MATCHES_FILE}."
            return matches
        reasons.append(f"{path}: {reason}")
    if not candidates:
        reason = f"No existe {INTERNATIONAL_MATCHES_FILE}. Descarga primero {INTERNATIONAL_DATASET_SLUG}."
    else:
        reason = "No se encontro un CSV internacional valido. " + " | ".join(reasons[:4])
    if required:
        raise RuntimeError(reason)
    return empty_matches_frame(reason=reason, source_path=str(INTERNATIONAL_MATCHES_FILE))


def local_international_candidate_files() -> List[Path]:
    candidates: List[Path] = []
    if INTERNATIONAL_MATCHES_FILE.exists():
        candidates.append(INTERNATIONAL_MATCHES_FILE)
    for path in discover_international_csv_files(INTERNATIONAL_ROOT):
        if Path(path) != INTERNATIONAL_MATCHES_FILE:
            candidates.append(path)
    return candidates


def discover_international_csv_files(root: Path) -> List[Path]:
    root = Path(root)
    if root.is_file():
        return [root] if root.suffix.lower() == ".csv" else []
    if not root.exists():
        return []
    return sorted(
        (path for path in root.rglob("*.csv") if path.is_file()),
        key=lambda path: (0 if path.name == "all_matches.csv" else 1, str(path).lower()),
    )


def read_normalized_international_csv(path: Path) -> Tuple[pd.DataFrame, str]:
    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        return empty_matches_frame(reason=f"No se pudo leer {path}: {exc}", source_path=str(path)), str(exc)
    matches = normalize_international_matches(raw)
    if matches.empty:
        reason = str(matches.attrs.get("reason") or "CSV sin filas internacionales validas.")
        matches.attrs["source_path"] = str(path)
        return matches, reason
    matches.attrs["source_path"] = str(path)
    matches.attrs["file_path"] = str(INTERNATIONAL_MATCHES_FILE)
    return matches, ""


def normalize_international_matches(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return empty_matches_frame(reason="CSV vacio.")
    clean = raw.copy()
    clean.columns = [normalize_column(column) for column in clean.columns]
    column_map = resolve_international_columns(clean.columns)
    missing = [column for column in INTERNATIONAL_REQUIRED_COLUMNS if column not in column_map]
    if missing:
        return empty_matches_frame(reason=f"Columnas requeridas faltantes: {', '.join(missing)}.")
    output = pd.DataFrame()
    output["date"] = pd.to_datetime(clean[column_map["date"]], errors="coerce")
    output["home_team"] = clean[column_map["home_team"]].map(canonical_team_name)
    output["away_team"] = clean[column_map["away_team"]].map(canonical_team_name)
    output["home_score"] = pd.to_numeric(clean[column_map["home_score"]], errors="coerce")
    output["away_score"] = pd.to_numeric(clean[column_map["away_score"]], errors="coerce")
    output["tournament"] = clean[column_map["tournament"]].astype(str) if "tournament" in column_map else ""
    output["country"] = clean[column_map["country"]].astype(str) if "country" in column_map else ""
    output["neutral"] = clean[column_map["neutral"]].map(coerce_bool) if "neutral" in column_map else False
    metadata = international_raw_metadata(output)
    output = output[
        output["date"].notna()
        & output["home_team"].astype(str).str.len().gt(1)
        & output["away_team"].astype(str).str.len().gt(1)
        & output["home_score"].notna()
        & output["away_score"].notna()
    ].copy()
    if output.empty:
        frame = empty_matches_frame(reason="CSV sin filas con fecha, equipos y marcadores validos.")
        frame.attrs.update(metadata)
        return frame
    output["home_score"] = output["home_score"].astype(float)
    output["away_score"] = output["away_score"].astype(float)
    output = output.sort_values("date", kind="stable").reset_index(drop=True)
    output.attrs.update(metadata)
    return output


def international_raw_metadata(output: pd.DataFrame) -> Dict[str, Any]:
    if output is None or output.empty or "date" not in output.columns:
        return {
            "raw_rows": 0,
            "unscored_rows": 0,
            "future_unscored_rows": 0,
            "max_dataset_date": "",
        }
    dates = pd.to_datetime(output["date"], errors="coerce")
    scored = (
        dates.notna()
        & pd.to_numeric(output.get("home_score"), errors="coerce").notna()
        & pd.to_numeric(output.get("away_score"), errors="coerce").notna()
    )
    unscored = dates.notna() & ~scored
    max_scored = dates[scored].max() if scored.any() else pd.NaT
    future_unscored = int((unscored & dates.gt(max_scored)).sum()) if pd.notna(max_scored) else int(unscored.sum())
    return {
        "raw_rows": int(output.shape[0]),
        "unscored_rows": int(unscored.sum()),
        "future_unscored_rows": future_unscored,
        "max_dataset_date": date_to_string(dates.max()) if dates.notna().any() else "",
    }


def resolve_international_columns(columns: Iterable[str]) -> Dict[str, str]:
    available = {normalize_column(column): str(column) for column in columns}
    resolved: Dict[str, str] = {}
    for canonical, aliases in INTERNATIONAL_COLUMN_ALIASES.items():
        for alias in aliases:
            normalized = normalize_column(alias)
            if normalized in available:
                resolved[canonical] = available[normalized]
                break
    return resolved


def recent15_feature_table(
        matches: Optional[pd.DataFrame] = None,
        teams: Optional[Iterable[str]] = None,
        before_date: Optional[Any] = None,
        limit: int = RECENT_MATCH_LIMIT,
        base_model: Optional[Any] = None,
        match_index: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    if match_index is not None:
        return recent15_feature_table_from_index(
            match_index=match_index,
            teams=teams,
            before_date=before_date,
            limit=limit,
            base_model=base_model,
        )
    matches = load_international_matches(required=False) if matches is None else matches
    if matches is None or matches.empty:
        return pd.DataFrame(columns=["Team", *RECENT15_NUMERIC_COLUMNS])
    if teams is None:
        teams = sorted(set(matches["home_team"].dropna().astype(str)) | set(matches["away_team"].dropna().astype(str)))
    rows = recent15_feature_rows_vectorized(
        matches=matches,
        teams=list(teams),
        before_date=before_date,
        limit=limit,
        base_model=base_model,
    )
    return pd.DataFrame(rows).fillna(0.0) if rows else pd.DataFrame(columns=["Team", *RECENT15_NUMERIC_COLUMNS])


def build_recent15_match_index(matches: Optional[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if matches is None or matches.empty:
        return {}
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(matches.columns):
        return {}

    working = matches.copy()
    working["date"] = pd.to_datetime(working.get("date"), errors="coerce")
    working["home_score"] = pd.to_numeric(working.get("home_score"), errors="coerce")
    working["away_score"] = pd.to_numeric(working.get("away_score"), errors="coerce")
    working = working[
        working["date"].notna()
        & working["home_score"].notna()
        & working["away_score"].notna()
    ].copy()
    if working.empty:
        return {}

    working["_match_order"] = np.arange(len(working), dtype=int)
    neutral = working["neutral"].map(coerce_bool) if "neutral" in working.columns else False
    tournament = working.get("tournament", "").astype(str) if "tournament" in working.columns else ""
    home_rows = pd.DataFrame({
        "date": working["date"],
        "_match_order": working["_match_order"],
        "team": working["home_team"].astype(str),
        "opponent": working["away_team"].astype(str),
        "is_home": True,
        "neutral": neutral,
        "tournament": tournament,
        "gf": working["home_score"].astype(float),
        "ga": working["away_score"].astype(float),
    })
    away_rows = pd.DataFrame({
        "date": working["date"],
        "_match_order": working["_match_order"],
        "team": working["away_team"].astype(str),
        "opponent": working["home_team"].astype(str),
        "is_home": False,
        "neutral": neutral,
        "tournament": tournament,
        "gf": working["away_score"].astype(float),
        "ga": working["home_score"].astype(float),
    })
    long = pd.concat([home_rows, away_rows], ignore_index=True)
    long["team_key"] = long["team"].map(canonical_team_name)
    long = long[long["team_key"].astype(str).str.len().gt(0)].copy()
    if long.empty:
        return {}
    long = long.sort_values(["team_key", "date", "_match_order"], kind="stable")
    return {
        str(team_key): frame.reset_index(drop=True)
        for team_key, frame in long.groupby("team_key", sort=False)
    }


def recent15_feature_table_from_index(
        match_index: Dict[str, pd.DataFrame],
        teams: Optional[Iterable[str]] = None,
        before_date: Optional[Any] = None,
        limit: int = RECENT_MATCH_LIMIT,
        base_model: Optional[Any] = None,
) -> pd.DataFrame:
    if not match_index:
        return pd.DataFrame(columns=["Team", *RECENT15_NUMERIC_COLUMNS])
    if teams is None:
        teams = sorted(match_index.keys())
    requested = [(str(team or "").strip(), canonical_team_name(team)) for team in teams]
    cutoff = pd.to_datetime(before_date, errors="coerce")
    max_rows = max(int(limit or RECENT_MATCH_LIMIT), 1)
    rows: List[Dict[str, Any]] = []
    for display, canonical in requested:
        frame = match_index.get(canonical)
        if frame is None or frame.empty:
            rows.append(recent15_summary_features(display, [], before_date=before_date))
            continue
        scoped = frame
        if pd.notna(cutoff):
            scoped = scoped[scoped["date"] < pd.Timestamp(cutoff)]
        if scoped.empty:
            rows.append(recent15_summary_features(display, [], before_date=before_date))
            continue
        recent = scoped.tail(max_rows).copy().reset_index(drop=True)
        total = max(int(recent.shape[0]), 1)
        positions = np.arange(1, total + 1, dtype=float)
        recent["goal_diff"] = recent["gf"].astype(float) - recent["ga"].astype(float)
        recent["points"] = np.select(
            [recent["goal_diff"].gt(0.0), recent["goal_diff"].eq(0.0)],
            [3.0, 1.0],
            default=0.0,
        )
        recent["is_friendly"] = recent["tournament"].map(is_friendly_tournament)
        recent["tournament_weight"] = recent["tournament"].map(tournament_weight).astype(float)
        recent["recency_weight"] = 0.72 + (0.56 * (positions / float(total)))
        recent["weight"] = recent["tournament_weight"] * recent["recency_weight"]
        rating_cache = {
            opponent: rating_for_team(opponent, base_model=base_model)
            for opponent in recent["opponent"].dropna().astype(str).unique()
        }
        recent["opponent_rating"] = recent["opponent"].map(rating_cache).fillna(1500.0).astype(float)
        recent["difficulty_factor"] = np.clip(1.0 + ((recent["opponent_rating"] - 1500.0) / 900.0), 0.72, 1.32)
        recent["adjusted_gf"] = recent["gf"].astype(float) * recent["difficulty_factor"]
        recent["adjusted_ga"] = recent["ga"].astype(float) / recent["difficulty_factor"].clip(lower=1e-9)
        feature_row = recent15_summary_features(display, recent.to_dict(orient="records"), before_date=before_date)
        rows.append(feature_row)
    return pd.DataFrame(rows).fillna(0.0) if rows else pd.DataFrame(columns=["Team", *RECENT15_NUMERIC_COLUMNS])


def recent15_feature_rows_vectorized(
        matches: pd.DataFrame,
        teams: Iterable[str],
        before_date: Optional[Any] = None,
        limit: int = RECENT_MATCH_LIMIT,
        base_model: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    requested = [(str(team or "").strip(), canonical_team_name(team)) for team in teams]
    if not requested:
        return []
    base_rows = {display: recent15_summary_features(display, [], before_date=before_date) for display, _ in requested}
    if matches is None or matches.empty:
        return [base_rows[display] for display, _ in requested]

    working = matches.copy()
    working["date"] = pd.to_datetime(working.get("date"), errors="coerce")
    working = working[working["date"].notna()].copy()
    cutoff = pd.to_datetime(before_date, errors="coerce")
    if pd.notna(cutoff):
        working = working[working["date"] < pd.Timestamp(cutoff)].copy()
    if working.empty:
        return [base_rows[display] for display, _ in requested]

    working["_match_order"] = np.arange(len(working), dtype=int)
    home_rows = pd.DataFrame({
        "date": working["date"],
        "_match_order": working["_match_order"],
        "team": working["home_team"].astype(str),
        "opponent": working["away_team"].astype(str),
        "is_home": True,
        "neutral": working.get("neutral", False).astype(bool) if "neutral" in working.columns else False,
        "tournament": working.get("tournament", "").astype(str) if "tournament" in working.columns else "",
        "gf": pd.to_numeric(working["home_score"], errors="coerce"),
        "ga": pd.to_numeric(working["away_score"], errors="coerce"),
    })
    away_rows = pd.DataFrame({
        "date": working["date"],
        "_match_order": working["_match_order"],
        "team": working["away_team"].astype(str),
        "opponent": working["home_team"].astype(str),
        "is_home": False,
        "neutral": working.get("neutral", False).astype(bool) if "neutral" in working.columns else False,
        "tournament": working.get("tournament", "").astype(str) if "tournament" in working.columns else "",
        "gf": pd.to_numeric(working["away_score"], errors="coerce"),
        "ga": pd.to_numeric(working["home_score"], errors="coerce"),
    })
    long = pd.concat([home_rows, away_rows], ignore_index=True)
    long["team_key"] = long["team"].map(canonical_team_name)
    team_keys = {canonical for _, canonical in requested if canonical}
    if team_keys:
        long = long[long["team_key"].isin(team_keys)].copy()
    long = long[long["gf"].notna() & long["ga"].notna()].copy()
    if long.empty:
        return [base_rows[display] for display, _ in requested]

    long = long.sort_values(["team_key", "date", "_match_order"], kind="stable")
    recent = long.groupby("team_key", sort=False, group_keys=False).tail(max(int(limit or RECENT_MATCH_LIMIT), 1)).copy()
    group_sizes = recent.groupby("team_key", sort=False)["date"].transform("size").astype(float)
    group_positions = recent.groupby("team_key", sort=False).cumcount().astype(float) + 1.0
    recent["goal_diff"] = recent["gf"].astype(float) - recent["ga"].astype(float)
    recent["points"] = np.select(
        [recent["goal_diff"].gt(0.0), recent["goal_diff"].eq(0.0)],
        [3.0, 1.0],
        default=0.0,
    )
    recent["is_friendly"] = recent["tournament"].map(is_friendly_tournament)
    recent["tournament_weight"] = recent["tournament"].map(tournament_weight).astype(float)
    recent["recency_weight"] = 0.72 + (0.56 * (group_positions / group_sizes.clip(lower=1.0)))
    recent["weight"] = recent["tournament_weight"] * recent["recency_weight"]
    rating_cache = {
        opponent: rating_for_team(opponent, base_model=base_model)
        for opponent in recent["opponent"].dropna().astype(str).unique()
    }
    recent["opponent_rating"] = recent["opponent"].map(rating_cache).fillna(1500.0).astype(float)
    recent["difficulty_factor"] = np.clip(1.0 + ((recent["opponent_rating"] - 1500.0) / 900.0), 0.72, 1.32)
    recent["adjusted_gf"] = recent["gf"].astype(float) * recent["difficulty_factor"]
    recent["adjusted_ga"] = recent["ga"].astype(float) / recent["difficulty_factor"].clip(lower=1e-9)

    summaries: Dict[str, Dict[str, Any]] = {}
    for team_key, frame in recent.groupby("team_key", sort=False):
        output: Dict[str, Any] = {"Team": str(team_key or "").strip()}
        output.update({column: 0.0 for column in RECENT15_NUMERIC_COLUMNS})
        weights = frame["weight"].astype(float)
        official = frame[~frame["is_friendly"].astype(bool)]
        friendly = frame[frame["is_friendly"].astype(bool)]
        totals = frame["gf"].astype(float) + frame["ga"].astype(float)
        output.update({
            "recent15_matches": float(frame.shape[0]),
            "recent15_gf_avg": float(frame["gf"].mean()),
            "recent15_ga_avg": float(frame["ga"].mean()),
            "recent15_goal_diff_avg": float(frame["goal_diff"].mean()),
            "recent15_weighted_gf_avg": weighted_average(frame["gf"], weights),
            "recent15_weighted_ga_avg": weighted_average(frame["ga"], weights),
            "recent15_weighted_goal_diff_avg": weighted_average(frame["goal_diff"], weights),
            "recent15_adjusted_gf_avg": weighted_average(frame["adjusted_gf"], weights),
            "recent15_adjusted_ga_avg": weighted_average(frame["adjusted_ga"], weights),
            "recent15_adjusted_goal_diff_avg": weighted_average(frame["adjusted_gf"] - frame["adjusted_ga"], weights),
            "recent15_points_avg": float(frame["points"].mean()),
            "recent15_weighted_points_avg": weighted_average(frame["points"], weights),
            "recent15_win_rate": float((frame["goal_diff"] > 0).mean()),
            "recent15_draw_rate": float((frame["goal_diff"] == 0).mean()),
            "recent15_loss_rate": float((frame["goal_diff"] < 0).mean()),
            "recent15_official_matches": float(official.shape[0]),
            "recent15_friendly_matches": float(friendly.shape[0]),
            "recent15_official_gf_avg": safe_mean(official.get("gf", pd.Series(dtype=float))),
            "recent15_official_ga_avg": safe_mean(official.get("ga", pd.Series(dtype=float))),
            "recent15_friendly_gf_avg": safe_mean(friendly.get("gf", pd.Series(dtype=float))),
            "recent15_friendly_ga_avg": safe_mean(friendly.get("ga", pd.Series(dtype=float))),
            "recent15_opponent_rating_avg": float(frame["opponent_rating"].mean()),
            "recent15_weighted_opponent_rating_avg": weighted_average(frame["opponent_rating"], weights),
            "recent15_weight_sum": float(weights.sum()),
            "recent15_goal_total_avg": float(totals.mean()),
            "recent15_over25_rate": float((totals > 2.5).mean()),
            "recent15_btts_rate": float(((frame["gf"] > 0) & (frame["ga"] > 0)).mean()),
        })
        last_date = pd.Timestamp(frame["date"].max()) if not frame.empty else pd.NaT
        if pd.notna(cutoff) and pd.notna(last_date):
            output["recent15_days_since_last_match"] = float(max((pd.Timestamp(cutoff) - last_date).days, 0))
        summaries[str(team_key)] = output

    rows: List[Dict[str, Any]] = []
    for display, canonical in requested:
        row = dict(base_rows[display])
        if canonical in summaries:
            row.update(summaries[canonical])
            row["Team"] = display
        rows.append(row)
    return rows


def recent15_team_context(
        team: str,
        matches: Optional[pd.DataFrame] = None,
        before_date: Optional[Any] = None,
        limit: int = RECENT_MATCH_LIMIT,
        base_model: Optional[Any] = None,
        include_matches: bool = True,
) -> Dict[str, Any]:
    matches = load_international_matches(required=False) if matches is None else matches
    display_team = str(team or "").strip()
    canonical = canonical_team_name(display_team)
    rows = team_recent_rows(matches, canonical, before_date=before_date, limit=limit, base_model=base_model)
    features = recent15_summary_features(display_team, rows, before_date=before_date)
    return {
        "team": display_team,
        "canonical_team": canonical,
        "features": features,
        "matches": public_recent_match_rows(rows) if include_matches else [],
    }


def contextual_poisson_for_match(
        home: str,
        away: str,
        base_model: Optional[Any] = None,
        before_date: Optional[Any] = None,
        max_goals: int = 10,
        matches: Optional[pd.DataFrame] = None,
        limit: int = RECENT_MATCH_LIMIT,
) -> Dict[str, Any]:
    matches = load_international_matches(required=False) if matches is None else matches
    source_path = str(getattr(matches, "attrs", {}).get("source_path") or INTERNATIONAL_MATCHES_FILE)
    if matches is None or matches.empty:
        return unavailable_context("all_matches.csv no disponible", source_path, home, away, before_date, base_model=base_model, max_goals=max_goals, limit=limit)

    home_context = recent15_team_context(home, matches, before_date=before_date, limit=limit, base_model=base_model)
    away_context = recent15_team_context(away, matches, before_date=before_date, limit=limit, base_model=base_model)
    home_features = home_context["features"]
    away_features = away_context["features"]
    if not home_context["matches"] and not away_context["matches"]:
        return unavailable_context("Sin partidos recientes para las selecciones solicitadas", source_path, home, away, before_date, base_model=base_model, max_goals=max_goals, limit=limit)

    base_lambda_home, base_lambda_away = base_lambdas(base_model, home, away)
    home_attack = positive_or_default(home_features.get("recent15_adjusted_gf_avg"), base_lambda_home)
    home_defense = positive_or_default(home_features.get("recent15_adjusted_ga_avg"), base_lambda_away)
    away_attack = positive_or_default(away_features.get("recent15_adjusted_gf_avg"), base_lambda_away)
    away_defense = positive_or_default(away_features.get("recent15_adjusted_ga_avg"), base_lambda_home)

    recent_lambda_home = math.sqrt(max(home_attack, 0.05) * max(away_defense, 0.05))
    recent_lambda_away = math.sqrt(max(away_attack, 0.05) * max(home_defense, 0.05))
    home_rating = rating_for_team(home, base_model=base_model)
    away_rating = rating_for_team(away, base_model=base_model)
    host_bonus = 45.0 if str(home or "") in HOST_TEAMS else 0.0
    rating_factor = math.exp(np.clip((home_rating + host_bonus - away_rating) / 1200.0, -0.55, 0.55))
    recent_lambda_home *= rating_factor
    recent_lambda_away /= max(rating_factor, 1e-9)

    coverage = min(
        1.0,
        (
            float(home_features.get("recent15_matches", 0.0))
            + float(away_features.get("recent15_matches", 0.0))
        ) / float(limit * 2),
    )
    recent_weight = 0.35 + 0.45 * coverage
    lambda_home = clamp((recent_lambda_home * recent_weight) + (base_lambda_home * (1.0 - recent_weight)), 0.2, 4.8)
    lambda_away = clamp((recent_lambda_away * recent_weight) + (base_lambda_away * (1.0 - recent_weight)), 0.2, 4.8)
    matrix_payload = poisson_matrix_payload(lambda_home, lambda_away, max_goals=max_goals)
    return {
        "available": True,
        "matrix_available": True,
        "matrix_source": "recent15",
        "source": "all_matches.csv",
        "dataset_slug": INTERNATIONAL_DATASET_SLUG,
        "source_path": source_path,
        "before_date": date_to_string(before_date),
        "match_limit": int(limit),
        **matrix_payload,
        "home_recent": home_context,
        "away_recent": away_context,
        "recent_matches": {
            "home": home_context["matches"],
            "away": away_context["matches"],
        },
    }


def poisson_matrix_payload(lambda_home: float, lambda_away: float, max_goals: int = 10) -> Dict[str, Any]:
    max_goals = int(clamp(max_goals, 4, 14))
    grid = poisson_score_grid(lambda_home, lambda_away, max_goals=max_goals)
    raw_probs = grid_result_probabilities(grid)
    pct_probs = {key: round(value * 100.0, 2) for key, value in raw_probs.items()}
    matrix = [[round(float(grid[h, a]) * 100.0, 3) for a in range(grid.shape[1])] for h in range(grid.shape[0])]
    cells = score_cells(grid)
    top_scores = sorted(cells, key=lambda item: item["probability_raw"], reverse=True)[:5]
    max_probability = max((cell["probability"] for cell in cells), default=0.0)
    return {
        "context_lambda_home": round(float(lambda_home), 3),
        "context_lambda_away": round(float(lambda_away), 3),
        "lambdas": {"home": round(float(lambda_home), 3), "away": round(float(lambda_away), 3)},
        "probabilities": pct_probs,
        "probabilities_raw": raw_probs,
        "over_under": {
            f"{line:.1f}": {
                "over": pct_probs.get(f"over{total_line_suffix(line)}", 0.0),
                "under": pct_probs.get(f"under{total_line_suffix(line)}", 0.0),
            }
            for line in CONTEXT_TOTAL_GOAL_LINES
        },
        "score_matrix": matrix,
        "score_cells": cells,
        "heatmap": {
            "home_goals": list(range(grid.shape[0])),
            "away_goals": list(range(grid.shape[1])),
            "max_probability": round(max_probability, 3),
            "cells": cells,
        },
        "top_scores": top_scores,
    }


def team_recent_rows(
        matches: pd.DataFrame,
        canonical_team: str,
        before_date: Optional[Any] = None,
        limit: int = RECENT_MATCH_LIMIT,
        base_model: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    if matches is None or matches.empty:
        return []
    team_key = canonical_team_name(canonical_team)
    scoped = matches[(matches["home_team"] == team_key) | (matches["away_team"] == team_key)].copy()
    cutoff = pd.to_datetime(before_date, errors="coerce")
    if pd.notna(cutoff):
        scoped = scoped[scoped["date"] < cutoff].copy()
    if scoped.empty:
        return []
    scoped = scoped.sort_values("date", kind="stable").tail(int(limit)).reset_index(drop=True)
    total = max(int(scoped.shape[0]), 1)
    rows: List[Dict[str, Any]] = []
    for index, row in scoped.iterrows():
        is_home = row["home_team"] == team_key
        opponent = str(row["away_team"] if is_home else row["home_team"])
        gf = float(row["home_score"] if is_home else row["away_score"])
        ga = float(row["away_score"] if is_home else row["home_score"])
        tournament = str(row.get("tournament", "") or "")
        tournament_weight_value = tournament_weight(tournament)
        recency = 0.72 + (0.56 * ((index + 1) / total))
        opponent_rating = rating_for_team(opponent, base_model=base_model)
        difficulty = clamp(1.0 + ((opponent_rating - 1500.0) / 900.0), 0.72, 1.32)
        rows.append({
            "date": pd.Timestamp(row["date"]),
            "team": team_key,
            "opponent": opponent,
            "is_home": bool(is_home),
            "neutral": bool(row.get("neutral", False)),
            "venue": "neutral" if bool(row.get("neutral", False)) else "home" if is_home else "away",
            "tournament": tournament,
            "is_friendly": is_friendly_tournament(tournament),
            "gf": gf,
            "ga": ga,
            "goal_diff": gf - ga,
            "points": 3.0 if gf > ga else 1.0 if gf == ga else 0.0,
            "opponent_rating": opponent_rating,
            "difficulty_factor": difficulty,
            "tournament_weight": tournament_weight_value,
            "recency_weight": recency,
            "weight": tournament_weight_value * recency,
            "adjusted_gf": gf * difficulty,
            "adjusted_ga": ga / max(difficulty, 1e-9),
        })
    return rows


def recent15_summary_features(team: str, rows: List[Dict[str, Any]], before_date: Optional[Any] = None) -> Dict[str, Any]:
    output: Dict[str, Any] = {"Team": str(team or "").strip()}
    output.update({column: 0.0 for column in RECENT15_NUMERIC_COLUMNS})
    if not rows:
        return output
    frame = pd.DataFrame(rows)
    weights = frame["weight"].astype(float)
    official = frame[~frame["is_friendly"].astype(bool)]
    friendly = frame[frame["is_friendly"].astype(bool)]
    output.update({
        "recent15_matches": float(frame.shape[0]),
        "recent15_gf_avg": float(frame["gf"].mean()),
        "recent15_ga_avg": float(frame["ga"].mean()),
        "recent15_goal_diff_avg": float(frame["goal_diff"].mean()),
        "recent15_weighted_gf_avg": weighted_average(frame["gf"], weights),
        "recent15_weighted_ga_avg": weighted_average(frame["ga"], weights),
        "recent15_weighted_goal_diff_avg": weighted_average(frame["goal_diff"], weights),
        "recent15_adjusted_gf_avg": weighted_average(frame["adjusted_gf"], weights),
        "recent15_adjusted_ga_avg": weighted_average(frame["adjusted_ga"], weights),
        "recent15_adjusted_goal_diff_avg": weighted_average(frame["adjusted_gf"] - frame["adjusted_ga"], weights),
        "recent15_points_avg": float(frame["points"].mean()),
        "recent15_weighted_points_avg": weighted_average(frame["points"], weights),
        "recent15_win_rate": float((frame["goal_diff"] > 0).mean()),
        "recent15_draw_rate": float((frame["goal_diff"] == 0).mean()),
        "recent15_loss_rate": float((frame["goal_diff"] < 0).mean()),
        "recent15_official_matches": float(official.shape[0]),
        "recent15_friendly_matches": float(friendly.shape[0]),
        "recent15_official_gf_avg": safe_mean(official.get("gf", pd.Series(dtype=float))),
        "recent15_official_ga_avg": safe_mean(official.get("ga", pd.Series(dtype=float))),
        "recent15_friendly_gf_avg": safe_mean(friendly.get("gf", pd.Series(dtype=float))),
        "recent15_friendly_ga_avg": safe_mean(friendly.get("ga", pd.Series(dtype=float))),
        "recent15_opponent_rating_avg": float(frame["opponent_rating"].mean()),
        "recent15_weighted_opponent_rating_avg": weighted_average(frame["opponent_rating"], weights),
        "recent15_weight_sum": float(weights.sum()),
        "recent15_goal_total_avg": float((frame["gf"] + frame["ga"]).mean()),
        "recent15_over25_rate": float(((frame["gf"] + frame["ga"]) > 2.5).mean()),
        "recent15_btts_rate": float(((frame["gf"] > 0) & (frame["ga"] > 0)).mean()),
    })
    cutoff = pd.to_datetime(before_date, errors="coerce")
    last_date = pd.Timestamp(frame["date"].max()) if not frame.empty else pd.NaT
    if pd.notna(cutoff) and pd.notna(last_date):
        output["recent15_days_since_last_match"] = float(max((pd.Timestamp(cutoff) - last_date).days, 0))
    return output


def public_recent_match_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        gf = float(row.get("gf", 0.0))
        ga = float(row.get("ga", 0.0))
        result = "W" if gf > ga else "D" if gf == ga else "L"
        output.append({
            "date": date_to_string(row.get("date")),
            "opponent": row.get("opponent", ""),
            "venue": row.get("venue", ""),
            "tournament": row.get("tournament", ""),
            "match_type": "Friendly" if row.get("is_friendly") else "Official",
            "gf": int(gf) if gf.is_integer() else gf,
            "ga": int(ga) if ga.is_integer() else ga,
            "score": f"{int(gf) if gf.is_integer() else gf}-{int(ga) if ga.is_integer() else ga}",
            "result": result,
            "weight": round(float(row.get("weight", 0.0)), 3),
            "opponent_rating": round(float(row.get("opponent_rating", 0.0)), 1),
        })
    return list(reversed(output))


def grid_result_probabilities(grid: np.ndarray) -> Dict[str, float]:
    goals = np.arange(grid.shape[0], dtype=int)
    home_goals, away_goals = np.meshgrid(goals, goals, indexing="ij")
    margin = home_goals - away_goals
    total_goals = home_goals + away_goals
    output = {
        "home": float(grid[margin > 0].sum()),
        "draw": float(grid[margin == 0].sum()),
        "away": float(grid[margin < 0].sum()),
    }
    for line in CONTEXT_TOTAL_GOAL_LINES:
        over_prob = float(grid[total_goals > line].sum())
        suffix = total_line_suffix(line)
        output[f"over{suffix}"] = float(over_prob)
        output[f"under{suffix}"] = float(1.0 - over_prob)
    return output


def score_cells(grid: np.ndarray) -> List[Dict[str, Any]]:
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


def unavailable_context(
        reason: str,
        source_path: str,
        home: str,
        away: str,
        before_date: Optional[Any],
        base_model: Optional[Any] = None,
        max_goals: int = 10,
        limit: int = RECENT_MATCH_LIMIT,
) -> Dict[str, Any]:
    lambda_home, lambda_away = base_lambdas(base_model, home, away)
    matrix_payload = poisson_matrix_payload(lambda_home, lambda_away, max_goals=max_goals)
    return {
        "available": False,
        "matrix_available": True,
        "matrix_source": "base_model",
        "reason": reason,
        "source": "all_matches.csv",
        "dataset_slug": INTERNATIONAL_DATASET_SLUG,
        "source_path": source_path,
        "before_date": date_to_string(before_date),
        "match_limit": int(limit),
        **matrix_payload,
        "home_recent": {"team": home, "canonical_team": canonical_team_name(home), "features": recent15_summary_features(home, []), "matches": []},
        "away_recent": {"team": away, "canonical_team": canonical_team_name(away), "features": recent15_summary_features(away, []), "matches": []},
        "recent_matches": {"home": [], "away": []},
    }


def base_lambdas(base_model: Optional[Any], home: str, away: str) -> tuple[float, float]:
    if base_model is not None:
        try:
            lambda_home, lambda_away = base_model.expected_goals(home, away)
            return float(lambda_home), float(lambda_away)
        except Exception:
            pass
    return 1.25, 1.05


def rating_for_team(team: str, base_model: Optional[Any] = None) -> float:
    candidates = [str(team or "").strip(), canonical_team_name(team)]
    canonical = canonical_team_name(team)
    if canonical in RATING_ALIAS_FALLBACKS:
        candidates.append(RATING_ALIAS_FALLBACKS[canonical])
    for candidate in candidates:
        if not candidate:
            continue
        if base_model is not None:
            try:
                return float(base_model.profile(candidate).rating)
            except Exception:
                pass
        if candidate in TEAM_RATING_PRIORS:
            return float(TEAM_RATING_PRIORS[candidate])
    return 1500.0


def tournament_weight(tournament: str) -> float:
    text = normalize_key(tournament)
    if "friendly" in text:
        return 0.55
    if any(token in text for token in ("qualification", "qualifier", "qualifiers")):
        return 1.2
    if "nations league" in text:
        return 1.1
    if any(token in text for token in (
        "world cup",
        "copa america",
        "euro",
        "african cup",
        "asian cup",
        "gold cup",
        "nations cup",
        "championship",
        "confederations",
    )):
        return 1.3
    return 0.95


def is_worldcup_tournament(tournament: str) -> bool:
    text = normalize_key(tournament)
    if not text:
        return False
    blocked_terms = (
        "qualification",
        "qualifier",
        "qualifiers",
        "futsal",
        "women",
        "womens",
        "woman",
        "u17",
        "u 17",
        "under 17",
        "u20",
        "u 20",
        "under 20",
        "club",
        "beach soccer",
        "beach",
        "youth",
        "junior",
        "olympic",
    )
    if any(token in text for token in blocked_terms):
        return False
    parts = text.split()
    return parts in (["world", "cup"], ["fifa", "world", "cup"]) or (
        len(parts) == 3 and parts[0:2] == ["world", "cup"] and parts[2].isdigit()
    ) or (
        len(parts) == 4 and parts[0:3] == ["fifa", "world", "cup"] and parts[3].isdigit()
    )


def is_friendly_tournament(tournament: str) -> bool:
    return "friendly" in normalize_key(tournament)


def canonical_team_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return TEAM_ALIASES.get(text, TEAM_ALIASES_BY_KEY.get(normalize_key(text), text))


def normalize_key(value: Any) -> str:
    text = str(value or "").lower().replace("&", " and ")
    text = "".join(char for char in text if ord(char) < 128)
    text = "".join(char if char.isalnum() else " " for char in text)
    return " ".join(text.split())


TEAM_ALIASES_BY_KEY = {normalize_key(key): value for key, value in TEAM_ALIASES.items()}


def normalize_column(value: Any) -> str:
    text = str(value or "").strip().lower()
    cleaned = []
    for char in text:
        cleaned.append(char if char.isalnum() else "_")
    return "_".join("".join(cleaned).split("_")).strip("_")


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "si", "y"}


def weighted_average(values: Iterable[Any], weights: Iterable[Any]) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    weight_series = pd.to_numeric(pd.Series(weights), errors="coerce")
    mask = series.notna() & weight_series.notna() & weight_series.gt(0)
    if not mask.any():
        return 0.0
    return float(np.average(series[mask].astype(float), weights=weight_series[mask].astype(float)))


def safe_mean(values: Iterable[Any]) -> float:
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(series.mean()) if not series.empty else 0.0


def positive_or_default(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(number) or number <= 0.0:
        return float(default)
    return number


def clamp(value: Any, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = lower
    return float(max(lower, min(upper, number)))


def date_to_string(value: Optional[Any]) -> str:
    if value in {None, ""}:
        return ""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat() if pd.notna(value) else ""
    if isinstance(value, (datetime, date)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.date().isoformat() if pd.notna(parsed) else str(value)


def international_freshness_warning(max_scored_date: Any, today: Optional[date] = None) -> str:
    scored = pd.to_datetime(max_scored_date, errors="coerce")
    if pd.isna(scored):
        return "Dataset internacional sin fecha maxima de partidos finalizados; no se puede validar actualidad."
    min_current = pd.Timestamp(INTERNATIONAL_MIN_CURRENT_SCORED_DATE)
    if scored < min_current:
        return (
            f"Dataset internacional desactualizado: ultimo partido finalizado {date_to_string(scored)}, "
            f"por debajo del minimo requerido {INTERNATIONAL_MIN_CURRENT_SCORED_DATE}."
        )
    today_value = today or date.today()
    stale_cutoff = pd.Timestamp(today_value - timedelta(days=INTERNATIONAL_FRESHNESS_MAX_AGE_DAYS))
    if scored < stale_cutoff:
        return (
            f"Dataset internacional posiblemente viejo: ultimo partido finalizado {date_to_string(scored)}; "
            f"actualiza all_matches.csv desde martj42/international_results."
        )
    return ""


def empty_matches_frame(reason: str = "", source_path: str = "") -> pd.DataFrame:
    frame = pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament", "country", "neutral"])
    if reason:
        frame.attrs["reason"] = reason
    if source_path:
        frame.attrs["source_path"] = source_path
    return frame
