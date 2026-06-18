from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.worldcup.accelerators import acceleration_status, import_optional_accelerator
from src.worldcup.api_football_provider import api_football_key, load_api_football_data


ADVANCED_ROOT = Path("storage") / "worldcup" / "advanced"
XG_ROOT = Path("storage") / "worldcup" / "xg"
STATSBOMB_ROOT = Path("storage") / "worldcup" / "statsbomb"
OPEN_DATA_ROOT = Path("storage") / "worldcup" / "open-data"
MATCH_FEATURES_FILE = ADVANCED_ROOT / "match_features.csv"
STATUS_FILE = ADVANCED_ROOT / "status.json"

LOCAL_SOURCE_FILES = {
    "manual_xg": XG_ROOT / "manual_xg.csv",
    "api_football_xg": XG_ROOT / "api_football_xg.csv",
    "shots": XG_ROOT / "shots.csv",
    "events": XG_ROOT / "events.csv",
    "xthreat": XG_ROOT / "xthreat.csv",
    "psxg": XG_ROOT / "psxg.csv",
}


def advanced_data_status() -> Dict[str, Any]:
    source_files = {
        key: _file_status(path)
        for key, path in LOCAL_SOURCE_FILES.items()
    }
    match_features = _file_status(MATCH_FEATURES_FILE)
    statsbomb = _statsbomb_status()
    source_rows = sum(int(item.get("rows") or 0) for item in source_files.values())
    prepared_rows = int(match_features.get("rows") or 0)
    socceraction_available = importlib.util.find_spec("socceraction") is not None
    active_sources = [
        key
        for key, item in source_files.items()
        if bool(item.get("exists")) and int(item.get("rows") or 0) > 0
    ]
    if prepared_rows:
        active_sources.append("advanced_match_features")
    if statsbomb.get("available"):
        active_sources.append("statsbomb_open_data_cache")
    warnings: List[str] = []
    if not source_rows and not statsbomb.get("available") and not prepared_rows:
        warnings.append("Sin cache avanzado local; los modelos xG usan fallback Poisson/GLM.")
    if not socceraction_available:
        warnings.append("socceraction no instalado; xT/VAEP quedan como features opcionales.")
    metadata = _read_json(STATUS_FILE)
    families = _advanced_feature_families(
        prepared_rows=prepared_rows,
        source_rows=source_rows,
        statsbomb_available=bool(statsbomb.get("available")),
        socceraction_available=socceraction_available,
    )
    return {
        "available": bool(prepared_rows or source_rows or statsbomb.get("available")),
        "prepared": bool(prepared_rows),
        "prepared_rows": prepared_rows,
        "source_rows": source_rows,
        "active_sources": active_sources,
        "source_files": source_files,
        "match_features": match_features,
        "statsbomb": statsbomb,
        "socceraction_available": socceraction_available,
        "families": families,
        "active_feature_families": [item["key"] for item in families if item["status"] in {"active", "cached"}],
        "accelerators": acceleration_status(),
        "warnings": _unique([*warnings, *[str(item) for item in metadata.get("warnings", []) if str(item)]]),
        "last_prepared_at": metadata.get("prepared_at", ""),
        "anti_leakage": "Features avanzadas se preparan cache-first y deben usarse con corte temporal anterior al fixture.",
    }


def prepare_advanced_data(payload: Dict[str, Any] | None = None, progress_callback=None) -> Dict[str, Any]:
    payload = payload or {}
    ADVANCED_ROOT.mkdir(parents=True, exist_ok=True)
    XG_ROOT.mkdir(parents=True, exist_ok=True)
    _emit(progress_callback, "prepare", 0, 5, "Inspeccionando fuentes avanzadas")
    rows = _normalized_match_feature_rows()
    warnings: List[str] = []
    _emit(progress_callback, "prepare", 1, 5, "Normalizando xG/PSxG/xThreat locales")
    if bool(payload.get("use_api_football", False)):
        api_rows, api_warnings = _api_football_match_feature_rows(
            allow_download=bool(payload.get("allow_api_download", False)),
            force_download=bool(payload.get("force_api_football", False)),
        )
        warnings.extend(api_warnings)
        if api_rows:
            _write_xg_cache(LOCAL_SOURCE_FILES["api_football_xg"], api_rows)
            if not LOCAL_SOURCE_FILES["manual_xg"].exists():
                _write_xg_cache(LOCAL_SOURCE_FILES["manual_xg"], api_rows)
            rows.extend(api_rows)
    _emit(progress_callback, "prepare", 2, 5, "API-Football xG revisado")
    if rows:
        _write_match_features(rows)
    else:
        warnings.append("No se encontraron filas locales para preparar match_features.csv.")
    _emit(progress_callback, "prepare", 3, 5, "match_features.csv actualizado")
    if bool(payload.get("snapshot_statsbomb", False)):
        copied = _snapshot_statsbomb_cache()
        if copied:
            warnings.append(f"StatsBomb cache referenciado en {copied}.")
    _emit(progress_callback, "prepare", 4, 5, "Fuentes opcionales revisadas")
    status = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "warnings": warnings,
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    completion_message = "Preparado con filas" if rows else "Completado sin cache local"
    _emit(progress_callback, "complete", 5, 5, completion_message)
    return advanced_data_status()


def _normalized_match_feature_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.extend(_manual_xg_rows(LOCAL_SOURCE_FILES["manual_xg"], source="manual_xg"))
    rows.extend(_manual_xg_rows(LOCAL_SOURCE_FILES["api_football_xg"], source="api_football_xg"))
    rows.extend(_generic_feature_rows(LOCAL_SOURCE_FILES["psxg"], "psxg"))
    rows.extend(_generic_feature_rows(LOCAL_SOURCE_FILES["xthreat"], "xthreat"))
    return rows


def _api_football_match_feature_rows(allow_download: bool, force_download: bool) -> tuple[List[Dict[str, Any]], List[str]]:
    if allow_download and not api_football_key():
        return [], ["API-Football no descargado: API_FOOTBALL_KEY no está definido en .env."]
    try:
        bundle = load_api_football_data(allow_download=allow_download, force_download=force_download)
    except Exception as exc:
        return [], [f"API-Football no pudo preparar xG avanzado ({exc.__class__.__name__}: {exc})."]
    warnings = [str(item) for item in bundle.get("warnings", []) if str(item)]
    team_stats = bundle.get("team_stats", pd.DataFrame())
    if team_stats is None or team_stats.empty:
        return [], warnings
    rows = _api_football_team_stats_xg_rows(team_stats)
    if not rows and any(str(column).lower() == "xg_for" for column in team_stats.columns):
        warnings.append("API-Football cache no contiene pares home/away completos para xG avanzado.")
    return rows, warnings


def _api_football_team_stats_xg_rows(team_stats: pd.DataFrame) -> List[Dict[str, Any]]:
    if team_stats is None or team_stats.empty or "FixtureId" not in team_stats.columns:
        return []
    required = {"Team", "Opponent", "Date"}
    if not required.issubset(set(str(column) for column in team_stats.columns)):
        return []
    has_xg = "xg_for" in team_stats.columns
    has_shots = any(str(column).endswith("_for") and "shot" in str(column).lower() for column in team_stats.columns)
    if not has_xg and not has_shots:
        return []
    working = team_stats.copy()
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    output: List[Dict[str, Any]] = []
    for _, scoped in working.groupby(working["FixtureId"].astype(str), sort=True):
        if scoped.empty:
            continue
        side = scoped["Side"] if "Side" in scoped.columns else pd.Series([""] * len(scoped), index=scoped.index)
        home_rows = scoped[side.astype(str).str.lower().eq("home")]
        if home_rows.empty:
            home_rows = scoped.head(1)
        home_row = home_rows.iloc[0]
        away_name = _text(home_row, "Opponent")
        away_rows = scoped[scoped["Team"].astype(str).eq(away_name)] if away_name else scoped.iloc[0:0]
        away_row = away_rows.iloc[0] if not away_rows.empty else None
        home_xg = _row_xg_or_shot_proxy(home_row, "for")
        away_xg = _row_xg_or_shot_proxy(home_row, "against")
        if away_row is not None:
            away_xg = away_xg if away_xg is not None else _row_xg_or_shot_proxy(away_row, "for")
        home = _text(home_row, "Team")
        away = away_name or (_text(away_row, "Team") if away_row is not None else "")
        if not home or not away or home_xg is None or away_xg is None:
            continue
        match_date = home_row.get("Date")
        date_text = match_date.date().isoformat() if pd.notna(match_date) else ""
        output.append({
            "date": date_text,
            "home_team": home,
            "away_team": away,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "home_psxg": "",
            "away_psxg": "",
            "home_xthreat": "",
            "away_xthreat": "",
            "home_shots": _finite_number(home_row.get("total_shots_for"), default=""),
            "away_shots": _finite_number(home_row.get("total_shots_against"), default=""),
            "source": "api_football_xg" if has_xg else "api_football_shot_xg_proxy",
            "cutoff": date_text,
        })
    return output


def _row_xg_or_shot_proxy(row: Any, scope: str) -> float | None:
    direct = _finite_number(row.get(f"xg_{scope}"))
    if direct is not None:
        return float(direct)
    shots_on = _finite_number(row.get(f"shots_on_goal_{scope}"), default=0.0) or 0.0
    shots_inside = _finite_number(row.get(f"shots_inside_box_{scope}"), default=0.0) or 0.0
    shots_outside = _finite_number(row.get(f"shots_outside_box_{scope}"), default=0.0) or 0.0
    blocked = _finite_number(row.get(f"blocked_shots_{scope}"), default=0.0) or 0.0
    shots_off = _finite_number(row.get(f"shots_off_goal_{scope}"), default=0.0) or 0.0
    total = _finite_number(row.get(f"total_shots_{scope}"), default=0.0) or 0.0
    if max(shots_on, shots_inside, shots_outside, blocked, shots_off, total) <= 0:
        return None
    unknown = max(total - shots_on - shots_off - blocked, 0.0)
    proxy = (
        0.13 * shots_on
        + 0.055 * max(shots_inside - shots_on, 0.0)
        + 0.025 * shots_outside
        + 0.018 * shots_off
        + 0.012 * blocked
        + 0.035 * unknown
    )
    return float(max(min(proxy, 5.5), 0.05))


def _manual_xg_rows(path: Path, source: str = "manual_xg") -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    polars_rows = _manual_xg_rows_polars(path, source=source)
    if polars_rows is not None:
        return polars_rows
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    output: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        home = _text(row, "home", "Home", "Equipo 1", "home_team")
        away = _text(row, "away", "Away", "Equipo 2", "away_team")
        date = _text(row, "date", "Date", "Fecha")
        home_xg = _number(row, "home_xg", "xg_home", "xG Local", "home_expected_goals")
        away_xg = _number(row, "away_xg", "xg_away", "xG Visita", "away_expected_goals")
        if not home or not away or home_xg is None or away_xg is None:
            continue
        output.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_xg": home_xg,
            "away_xg": away_xg,
            "home_psxg": _number(row, "home_psxg", "psxg_home", default=""),
            "away_psxg": _number(row, "away_psxg", "psxg_away", default=""),
            "home_xthreat": _number(row, "home_xthreat", "xthreat_home", default=""),
            "away_xthreat": _number(row, "away_xthreat", "xthreat_away", default=""),
            "home_shots": _number(row, "home_shots", "shots_home", default=""),
            "away_shots": _number(row, "away_shots", "shots_away", default=""),
            "source": source,
            "cutoff": date,
        })
    return output


def _generic_feature_rows(path: Path, family: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    polars_rows = _generic_feature_rows_polars(path, family)
    if polars_rows is not None:
        return polars_rows
    try:
        frame = pd.read_csv(path)
    except Exception:
        return []
    home_key = f"home_{family}"
    away_key = f"away_{family}"
    output: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        home = _text(row, "home", "Home", "Equipo 1", "home_team")
        away = _text(row, "away", "Away", "Equipo 2", "away_team")
        date = _text(row, "date", "Date", "Fecha")
        home_value = _number(row, home_key, f"{family}_home", f"{family} Local")
        away_value = _number(row, away_key, f"{family}_away", f"{family} Visita")
        if not home or not away or home_value is None or away_value is None:
            continue
        output.append({
            "date": date,
            "home_team": home,
            "away_team": away,
            "home_xg": "",
            "away_xg": "",
            "home_psxg": home_value if family == "psxg" else "",
            "away_psxg": away_value if family == "psxg" else "",
            "home_xthreat": home_value if family == "xthreat" else "",
            "away_xthreat": away_value if family == "xthreat" else "",
            "home_shots": "",
            "away_shots": "",
            "source": family,
            "cutoff": date,
        })
    return output


def _write_match_features(rows: List[Dict[str, Any]]) -> None:
    pl = import_optional_accelerator("polars")
    if pl is not None:
        try:
            (
                pl.DataFrame(rows)
                .unique(subset=["date", "home_team", "away_team", "source"], keep="last", maintain_order=True)
                .sort(["date", "home_team", "away_team"])
                .write_csv(MATCH_FEATURES_FILE)
            )
            return
        except Exception:
            pass
    frame = pd.DataFrame(rows)
    frame = frame.drop_duplicates(subset=["date", "home_team", "away_team", "source"], keep="last")
    frame = frame.sort_values(["date", "home_team", "away_team"], kind="stable")
    frame.to_csv(MATCH_FEATURES_FILE, index=False)


def _write_xg_cache(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_rows = [
        {
            "date": row.get("date", ""),
            "home": row.get("home_team", ""),
            "away": row.get("away_team", ""),
            "home_xg": row.get("home_xg", ""),
            "away_xg": row.get("away_xg", ""),
            "home_shots": row.get("home_shots", ""),
            "away_shots": row.get("away_shots", ""),
            "source": row.get("source", "api_football_xg"),
            "cutoff": row.get("cutoff", row.get("date", "")),
        }
        for row in rows
    ]
    pl = import_optional_accelerator("polars")
    if pl is not None:
        try:
            (
                pl.DataFrame(cache_rows)
                .unique(subset=["date", "home", "away"], keep="last", maintain_order=True)
                .sort(["date", "home", "away"])
                .write_csv(path)
            )
            return
        except Exception:
            pass
    frame = pd.DataFrame(cache_rows)
    frame = frame.drop_duplicates(subset=["date", "home", "away"], keep="last")
    frame = frame.sort_values(["date", "home", "away"], kind="stable")
    frame.to_csv(path, index=False)


def _manual_xg_rows_polars(path: Path, source: str = "manual_xg") -> List[Dict[str, Any]] | None:
    pl = import_optional_accelerator("polars")
    if pl is None:
        return None
    try:
        frame = pl.read_csv(path, ignore_errors=True)
        if frame.is_empty():
            return []
        output = frame.select(
            _pl_text_expr(pl, frame, ("home", "Home", "Equipo 1", "home_team"), "home_team"),
            _pl_text_expr(pl, frame, ("away", "Away", "Equipo 2", "away_team"), "away_team"),
            _pl_text_expr(pl, frame, ("date", "Date", "Fecha"), "date"),
            _pl_number_expr(pl, frame, ("home_xg", "xg_home", "xG Local", "home_expected_goals"), "home_xg"),
            _pl_number_expr(pl, frame, ("away_xg", "xg_away", "xG Visita", "away_expected_goals"), "away_xg"),
            _pl_number_expr(pl, frame, ("home_psxg", "psxg_home"), "home_psxg"),
            _pl_number_expr(pl, frame, ("away_psxg", "psxg_away"), "away_psxg"),
            _pl_number_expr(pl, frame, ("home_xthreat", "xthreat_home"), "home_xthreat"),
            _pl_number_expr(pl, frame, ("away_xthreat", "xthreat_away"), "away_xthreat"),
            _pl_number_expr(pl, frame, ("home_shots", "shots_home"), "home_shots"),
            _pl_number_expr(pl, frame, ("away_shots", "shots_away"), "away_shots"),
        ).filter(
            pl.col("home_team").str.len_chars() > 0,
            pl.col("away_team").str.len_chars() > 0,
            pl.col("home_xg").is_not_null(),
            pl.col("away_xg").is_not_null(),
        ).with_columns(
            source=pl.lit(source),
            cutoff=pl.col("date"),
        )
        return output.to_dicts()
    except Exception:
        return None


def _generic_feature_rows_polars(path: Path, family: str) -> List[Dict[str, Any]] | None:
    pl = import_optional_accelerator("polars")
    if pl is None:
        return None
    home_key = f"home_{family}"
    away_key = f"away_{family}"
    try:
        frame = pl.read_csv(path, ignore_errors=True)
        if frame.is_empty():
            return []
        output = frame.select(
            _pl_text_expr(pl, frame, ("home", "Home", "Equipo 1", "home_team"), "home_team"),
            _pl_text_expr(pl, frame, ("away", "Away", "Equipo 2", "away_team"), "away_team"),
            _pl_text_expr(pl, frame, ("date", "Date", "Fecha"), "date"),
            _pl_number_expr(pl, frame, (home_key, f"{family}_home", f"{family} Local"), "home_value"),
            _pl_number_expr(pl, frame, (away_key, f"{family}_away", f"{family} Visita"), "away_value"),
        ).filter(
            pl.col("home_team").str.len_chars() > 0,
            pl.col("away_team").str.len_chars() > 0,
            pl.col("home_value").is_not_null(),
            pl.col("away_value").is_not_null(),
        ).with_columns(
            home_xg=pl.lit(None),
            away_xg=pl.lit(None),
            home_psxg=pl.when(pl.lit(family) == "psxg").then(pl.col("home_value")).otherwise(pl.lit(None)),
            away_psxg=pl.when(pl.lit(family) == "psxg").then(pl.col("away_value")).otherwise(pl.lit(None)),
            home_xthreat=pl.when(pl.lit(family) == "xthreat").then(pl.col("home_value")).otherwise(pl.lit(None)),
            away_xthreat=pl.when(pl.lit(family) == "xthreat").then(pl.col("away_value")).otherwise(pl.lit(None)),
            home_shots=pl.lit(None),
            away_shots=pl.lit(None),
            source=pl.lit(family),
            cutoff=pl.col("date"),
        ).select(
            "date",
            "home_team",
            "away_team",
            "home_xg",
            "away_xg",
            "home_psxg",
            "away_psxg",
            "home_xthreat",
            "away_xthreat",
            "home_shots",
            "away_shots",
            "source",
            "cutoff",
        )
        return output.to_dicts()
    except Exception:
        return None


def _pl_first_existing_column(frame: Any, keys: Iterable[str]) -> str:
    columns = {str(column): str(column) for column in frame.columns}
    lower_lookup = {str(column).lower(): str(column) for column in frame.columns}
    for key in keys:
        if str(key) in columns:
            return columns[str(key)]
        if str(key).lower() in lower_lookup:
            return lower_lookup[str(key).lower()]
    return ""


def _pl_text_expr(pl: Any, frame: Any, keys: Iterable[str], alias: str):
    column = _pl_first_existing_column(frame, keys)
    if not column:
        return pl.lit("").alias(alias)
    return pl.col(column).cast(pl.String, strict=False).fill_null("").str.strip_chars().alias(alias)


def _pl_number_expr(pl: Any, frame: Any, keys: Iterable[str], alias: str):
    column = _pl_first_existing_column(frame, keys)
    if not column:
        return pl.lit(None).cast(pl.Float64).alias(alias)
    return pl.col(column).cast(pl.Float64, strict=False).alias(alias)


def _file_status(path: Path) -> Dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "rows": _csv_row_count(path) if exists and path.suffix.lower() == ".csv" else 0,
        "size_bytes": int(stat.st_size) if stat else 0,
        "updated_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat() if stat else "",
    }


def _csv_row_count(path: Path) -> int:
    pl = import_optional_accelerator("polars")
    if pl is not None:
        try:
            return int(pl.scan_csv(path).select(pl.len()).collect().item())
        except Exception:
            pass
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            count = sum(1 for _ in reader)
        return max(count - 1, 0)
    except Exception:
        return 0


def _statsbomb_status() -> Dict[str, Any]:
    root = STATSBOMB_ROOT if STATSBOMB_ROOT.exists() else OPEN_DATA_ROOT
    json_files = list(root.rglob("*.json")) if root.exists() else []
    return {
        "path": str(root),
        "available": bool(json_files),
        "json_files": len(json_files),
        "has_events": any("/events/" in str(path).replace("\\", "/") for path in json_files),
        "has_matches": any("/matches/" in str(path).replace("\\", "/") for path in json_files),
    }


def _snapshot_statsbomb_cache() -> str:
    if not STATSBOMB_ROOT.exists() and not OPEN_DATA_ROOT.exists():
        return ""
    source = STATSBOMB_ROOT if STATSBOMB_ROOT.exists() else OPEN_DATA_ROOT
    target = ADVANCED_ROOT / "statsbomb_cache_path.txt"
    target.write_text(str(source), encoding="utf-8")
    return str(target)


def _advanced_feature_families(
        prepared_rows: int,
        source_rows: int,
        statsbomb_available: bool,
        socceraction_available: bool,
) -> List[Dict[str, Any]]:
    has_xg = prepared_rows > 0 or source_rows > 0 or statsbomb_available
    return [
        {
            "key": "xg_shot_quality",
            "label": "xG por tiros",
            "status": "active" if has_xg else "missing",
            "detail": "manual_xg.csv, shots.csv o StatsBomb Open Data cacheado.",
        },
        {
            "key": "negative_binomial_overdispersion",
            "label": "Sobredispersion NB2",
            "status": "active",
            "detail": "Se estima desde diagnosticos Poisson GLM del historico.",
        },
        {
            "key": "dynamic_strength",
            "label": "Fuerza dinamica",
            "status": "active",
            "detail": "Estados ataque/defensa suavizados con recencia temporal.",
        },
        {
            "key": "stacking",
            "label": "Stacking ML/estadistico",
            "status": "active",
            "detail": "MNLogit statsmodels sobre probabilidades base y lambdas.",
        },
        {
            "key": "xthreat_vaep",
            "label": "xThreat/VAEP",
            "status": "cached" if socceraction_available and has_xg else "optional",
            "detail": "socceraction opcional; el pipeline no depende de esta libreria.",
        },
    ]


def _text(row: Any, *keys: str) -> str:
    for key in keys:
        value = row.get(key, "")
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    return ""


def _number(row: Any, *keys: str, default: Any = None) -> float | str | None:
    for key in keys:
        value = row.get(key, None)
        if value is None or value == "" or pd.isna(value):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _finite_number(value: Any, default: Any = None) -> float | str | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if pd.notna(numeric) else default


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _emit(callback, stage: str, current: int, total: int, message: str) -> None:
    if not callable(callback):
        return
    callback({
        "stage": stage,
        "current": int(current),
        "total": int(total),
        "percent": int(round((max(current, 0) / max(total, 1)) * 100)),
        "message": message,
    })


def _unique(values: Iterable[Any]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output
