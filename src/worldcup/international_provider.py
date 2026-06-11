from __future__ import annotations

import math
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.worldcup.model import HOST_TEAMS, TEAM_RATING_PRIORS, TOTAL_GOAL_LINES, poisson_score_grid, total_line_suffix


INTERNATIONAL_DATASET_SLUG = "patateriedata/all-international-football-results"
INTERNATIONAL_ROOT = Path("storage") / "worldcup" / "international"
INTERNATIONAL_MATCHES_FILE = INTERNATIONAL_ROOT / "all_matches.csv"
RECENT_MATCH_LIMIT = 15
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


def download_international_results(force: bool = False) -> Dict[str, Any]:
    if INTERNATIONAL_MATCHES_FILE.exists() and not force:
        return international_results_status()
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("kagglehub no esta instalado. Ejecuta pip install -r requirements.txt.") from exc

    source_path = Path(kagglehub.dataset_download(INTERNATIONAL_DATASET_SLUG))
    if not source_path.exists():
        raise RuntimeError(f"Kaggle no devolvio una ruta valida para {INTERNATIONAL_DATASET_SLUG}.")
    INTERNATIONAL_ROOT.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    candidates = sorted(source_path.rglob("all_matches.csv"))
    if not candidates:
        candidates = sorted(path for path in source_path.rglob("*.csv") if path.is_file())
    for path in candidates:
        if not path.is_file():
            continue
        target = INTERNATIONAL_ROOT / ("all_matches.csv" if path.name == "all_matches.csv" else path.name)
        if force or not target.exists():
            shutil.copy2(path, target)
        copied.append(str(target))
        if target.name == "all_matches.csv":
            break
    status = international_results_status()
    status["downloaded_path"] = str(source_path)
    status["copied_files"] = copied
    return status


def international_results_status() -> Dict[str, Any]:
    matches = load_international_matches(required=False)
    teams = set()
    if not matches.empty:
        teams.update(matches["home_team"].dropna().astype(str).tolist())
        teams.update(matches["away_team"].dropna().astype(str).tolist())
    return {
        "dataset_slug": INTERNATIONAL_DATASET_SLUG,
        "local_path": str(INTERNATIONAL_ROOT),
        "file_path": str(INTERNATIONAL_MATCHES_FILE),
        "available": bool(INTERNATIONAL_MATCHES_FILE.exists() and not matches.empty),
        "rows": int(matches.shape[0]),
        "teams": int(len(teams)),
    }


def load_international_matches(required: bool = False) -> pd.DataFrame:
    if not INTERNATIONAL_MATCHES_FILE.exists():
        if required:
            raise RuntimeError(f"No existe {INTERNATIONAL_MATCHES_FILE}. Descarga primero {INTERNATIONAL_DATASET_SLUG}.")
        return empty_matches_frame()
    try:
        raw = pd.read_csv(INTERNATIONAL_MATCHES_FILE)
    except Exception as exc:
        if required:
            raise RuntimeError(f"No se pudo leer {INTERNATIONAL_MATCHES_FILE}: {exc}") from exc
        return empty_matches_frame()
    return normalize_international_matches(raw)


def normalize_international_matches(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return empty_matches_frame()
    clean = raw.copy()
    clean.columns = [normalize_column(column) for column in clean.columns]
    required = {"date", "home_team", "away_team", "home_score", "away_score"}
    if not required.issubset(clean.columns):
        return empty_matches_frame()
    output = pd.DataFrame()
    output["date"] = pd.to_datetime(clean["date"], errors="coerce")
    output["home_team"] = clean["home_team"].map(canonical_team_name)
    output["away_team"] = clean["away_team"].map(canonical_team_name)
    output["home_score"] = pd.to_numeric(clean["home_score"], errors="coerce")
    output["away_score"] = pd.to_numeric(clean["away_score"], errors="coerce")
    output["tournament"] = clean["tournament"].astype(str) if "tournament" in clean.columns else ""
    output["country"] = clean["country"].astype(str) if "country" in clean.columns else ""
    output["neutral"] = clean["neutral"].map(coerce_bool) if "neutral" in clean.columns else False
    output = output[
        output["date"].notna()
        & output["home_team"].astype(str).str.len().gt(1)
        & output["away_team"].astype(str).str.len().gt(1)
        & output["home_score"].notna()
        & output["away_score"].notna()
    ].copy()
    if output.empty:
        return empty_matches_frame()
    output["home_score"] = output["home_score"].astype(float)
    output["away_score"] = output["away_score"].astype(float)
    return output.sort_values("date", kind="stable").reset_index(drop=True)


def recent15_feature_table(
        matches: Optional[pd.DataFrame] = None,
        teams: Optional[Iterable[str]] = None,
        before_date: Optional[Any] = None,
        limit: int = RECENT_MATCH_LIMIT,
        base_model: Optional[Any] = None,
) -> pd.DataFrame:
    matches = load_international_matches(required=False) if matches is None else matches
    if matches is None or matches.empty:
        return pd.DataFrame(columns=["Team", *RECENT15_NUMERIC_COLUMNS])
    if teams is None:
        teams = sorted(set(matches["home_team"].dropna().astype(str)) | set(matches["away_team"].dropna().astype(str)))
    rows = [
        recent15_team_context(
            team=team,
            matches=matches,
            before_date=before_date,
            limit=limit,
            base_model=base_model,
            include_matches=False,
        )["features"]
        for team in teams
    ]
    return pd.DataFrame(rows).fillna(0.0) if rows else pd.DataFrame(columns=["Team", *RECENT15_NUMERIC_COLUMNS])


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
    source_path = str(INTERNATIONAL_MATCHES_FILE)
    if matches is None or matches.empty:
        return unavailable_context("all_matches.csv no disponible", source_path, home, away, before_date)

    home_context = recent15_team_context(home, matches, before_date=before_date, limit=limit, base_model=base_model)
    away_context = recent15_team_context(away, matches, before_date=before_date, limit=limit, base_model=base_model)
    home_features = home_context["features"]
    away_features = away_context["features"]
    if not home_context["matches"] and not away_context["matches"]:
        return unavailable_context("Sin partidos recientes para las selecciones solicitadas", source_path, home, away, before_date)

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
    max_goals = int(clamp(max_goals, 4, 14))
    grid = poisson_score_grid(lambda_home, lambda_away, max_goals=max_goals)
    raw_probs = grid_result_probabilities(grid)
    pct_probs = {key: round(value * 100.0, 2) for key, value in raw_probs.items()}
    matrix = [[round(float(grid[h, a]) * 100.0, 3) for a in range(grid.shape[1])] for h in range(grid.shape[0])]
    cells = score_cells(grid)
    top_scores = sorted(cells, key=lambda item: item["probability_raw"], reverse=True)[:5]
    max_probability = max((cell["probability"] for cell in cells), default=0.0)
    return {
        "available": True,
        "source": "all_matches.csv",
        "dataset_slug": INTERNATIONAL_DATASET_SLUG,
        "source_path": source_path,
        "before_date": date_to_string(before_date),
        "match_limit": int(limit),
        "context_lambda_home": round(lambda_home, 3),
        "context_lambda_away": round(lambda_away, 3),
        "lambdas": {"home": round(lambda_home, 3), "away": round(lambda_away, 3)},
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
        "home_recent": home_context,
        "away_recent": away_context,
        "recent_matches": {
            "home": home_context["matches"],
            "away": away_context["matches"],
        },
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
    home = draw = away = 0.0
    totals = {line: 0.0 for line in CONTEXT_TOTAL_GOAL_LINES}
    for home_goals in range(grid.shape[0]):
        for away_goals in range(grid.shape[1]):
            prob = float(grid[home_goals, away_goals])
            if home_goals > away_goals:
                home += prob
            elif home_goals == away_goals:
                draw += prob
            else:
                away += prob
            total_goals = home_goals + away_goals
            for line in totals:
                if total_goals > line:
                    totals[line] += prob
    output = {"home": home, "draw": draw, "away": away}
    for line, over_prob in totals.items():
        suffix = total_line_suffix(line)
        output[f"over{suffix}"] = float(over_prob)
        output[f"under{suffix}"] = float(1.0 - over_prob)
    return output


def score_cells(grid: np.ndarray) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    for home_goals in range(grid.shape[0]):
        for away_goals in range(grid.shape[1]):
            raw = float(grid[home_goals, away_goals])
            cells.append({
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
                "score": f"{home_goals}-{away_goals}",
                "probability": round(raw * 100.0, 3),
                "probability_raw": raw,
            })
    return cells


def unavailable_context(reason: str, source_path: str, home: str, away: str, before_date: Optional[Any]) -> Dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "source": "all_matches.csv",
        "dataset_slug": INTERNATIONAL_DATASET_SLUG,
        "source_path": source_path,
        "before_date": date_to_string(before_date),
        "context_lambda_home": 0.0,
        "context_lambda_away": 0.0,
        "lambdas": {"home": 0.0, "away": 0.0},
        "probabilities": {},
        "over_under": {},
        "score_matrix": [],
        "score_cells": [],
        "heatmap": {"home_goals": [], "away_goals": [], "max_probability": 0.0, "cells": []},
        "top_scores": [],
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


def empty_matches_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "home_team", "away_team", "home_score", "away_score", "tournament", "country", "neutral"])
