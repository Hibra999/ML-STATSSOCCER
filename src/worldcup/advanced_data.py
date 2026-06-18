from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from src.worldcup.accelerators import acceleration_status, import_optional_accelerator


ADVANCED_ROOT = Path("storage") / "worldcup" / "advanced"
XG_ROOT = Path("storage") / "worldcup" / "xg"
STATSBOMB_ROOT = Path("storage") / "worldcup" / "statsbomb"
OPEN_DATA_ROOT = Path("storage") / "worldcup" / "open-data"
MATCH_FEATURES_FILE = ADVANCED_ROOT / "match_features.csv"
STATUS_FILE = ADVANCED_ROOT / "status.json"

LOCAL_SOURCE_FILES = {
    "manual_xg": XG_ROOT / "manual_xg.csv",
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
    _emit(progress_callback, "prepare", 0, 3, "Inspeccionando fuentes avanzadas")
    rows = _normalized_match_feature_rows()
    warnings: List[str] = []
    _emit(progress_callback, "prepare", 1, 3, "Normalizando xG/PSxG/xThreat locales")
    if rows:
        _write_match_features(rows)
    else:
        warnings.append("No se encontraron filas locales para preparar match_features.csv.")
    if bool(payload.get("snapshot_statsbomb", False)):
        copied = _snapshot_statsbomb_cache()
        if copied:
            warnings.append(f"StatsBomb cache referenciado en {copied}.")
    status = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "warnings": warnings,
    }
    STATUS_FILE.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")
    completion_message = "Preparado con filas" if rows else "Completado sin cache local"
    _emit(progress_callback, "complete", 3, 3, completion_message)
    return advanced_data_status()


def _normalized_match_feature_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.extend(_manual_xg_rows(LOCAL_SOURCE_FILES["manual_xg"]))
    rows.extend(_generic_feature_rows(LOCAL_SOURCE_FILES["psxg"], "psxg"))
    rows.extend(_generic_feature_rows(LOCAL_SOURCE_FILES["xthreat"], "xthreat"))
    return rows


def _manual_xg_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    polars_rows = _manual_xg_rows_polars(path)
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
            "source": "manual_xg",
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


def _manual_xg_rows_polars(path: Path) -> List[Dict[str, Any]] | None:
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
            source=pl.lit("manual_xg"),
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
