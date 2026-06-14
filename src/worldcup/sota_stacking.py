from __future__ import annotations

import hashlib
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.worldcup.market_provider import (
    load_market_data,
    market_for_match,
    no_vig_probabilities,
    valid_decimal_odd,
)
from src.worldcup.model import TOTAL_GOAL_LINES, WorldCupModel, poisson_score_grid, total_line_suffix
from src.worldcup.score_models import (
    DEFAULT_SCORE_MODEL,
    build_score_model,
    probabilities_from_score_grid,
    score_model_label,
)


ML_SOTA_SCORE_MODEL_SEQUENCE = [
    "independent_poisson",
    "dixon_coles_mle",
    "bivariate_poisson_mle",
    "diagonal_inflated_bivariate_poisson",
    "zero_inflated_generalized_poisson",
    "skellam_margin",
    "copula_weibull_count",
    "bayesian_hierarchical_poisson",
    "bayesian_dynamic_poisson",
    "xg_poisson_local",
    "market_blended_poisson",
]
ML_SOTA_PIPELINE_MODE = "ml_sota_poisson"
POISSON_SOTA_PIPELINE_MODE = "poisson_sota"
ML_SOTA_ARTIFACT_SCHEMA = "ml_sota_poisson_v1"
ML_SOTA_STACK_ROOT = Path("storage") / "worldcup" / "sota_stack"
ML_SOTA_PICKLE = ML_SOTA_STACK_ROOT / "ml_sota_poisson.pkl"
ML_SOTA_META = ML_SOTA_STACK_ROOT / "ml_sota_poisson.json"
ML_SOTA_DEFAULT_MAX_TRAIN_ROWS = 180
ML_SOTA_MIN_PRIOR_ROWS = 16
FEATURE_PROFILE_BALANCED = "balanced"
TRAIN_TOTAL_GOAL_LINES = tuple(line for line in TOTAL_GOAL_LINES if line <= 3.5)
ML_SOTA_TARGETS = [
    "result",
    *[f"over_under_{total_line_suffix(line)}" for line in TRAIN_TOTAL_GOAL_LINES],
]
RESULT_CLASSES = ["H", "D", "A"]
RESULT_CLASS_TO_OUTCOME = {"H": "home", "D": "draw", "A": "away"}
OUTCOME_TO_RESULT_CLASS = {value: key for key, value in RESULT_CLASS_TO_OUTCOME.items()}


class IdentityCalibrator:
    def __init__(self, classes: Sequence[Any]):
        self.classes_ = np.asarray(list(classes))

    def predict_proba(self, x: Any) -> np.ndarray:
        array = np.asarray(x, dtype=float)
        if array.ndim != 2:
            return np.zeros((0, len(self.classes_)), dtype=float)
        return _normalize_rows(np.exp(array))


def normalize_pipeline_mode(value: Any) -> str:
    text = str(value or POISSON_SOTA_PIPELINE_MODE).strip().lower()
    compact = "".join(char for char in text if char.isalnum())
    if compact in {
        "mlsotapoisson",
        "mlsotapoission",
        "mlsotapoissoon",
        "mlsotapoission",
        "mlsotapredf",
    }:
        return ML_SOTA_PIPELINE_MODE
    normalized = text.replace("+", "_").replace("-", "_").replace(" ", "_")
    normalized = "_".join(part for part in normalized.split("_") if part)
    aliases = {
        "ml_sota_poisson",
        "ml_sota_poission",
        "ml_sotapoisson",
        "ml_sotapoission",
        "ml_sota_poisson_stack",
        "ml_sota_stack",
    }
    return ML_SOTA_PIPELINE_MODE if normalized in aliases else POISSON_SOTA_PIPELINE_MODE


def is_ml_sota_pipeline(value: Any) -> bool:
    return normalize_pipeline_mode(value) == ML_SOTA_PIPELINE_MODE


def ml_sota_label() -> str:
    return "ML + SOTA Poisson"


def ml_sota_artifact_paths(root: Path | None = None) -> Tuple[Path, Path]:
    base = Path(root) if root is not None else ML_SOTA_STACK_ROOT
    return base / "ml_sota_poisson.pkl", base / "ml_sota_poisson.json"


def load_or_train_ml_sota_artifact(
        history_df: pd.DataFrame,
        teams: Iterable[str],
        config: Dict[str, Any] | None = None,
        market_rows: pd.DataFrame | None = None,
        force_refit: bool = False,
        artifact_root: Path | None = None,
) -> Dict[str, Any]:
    config = config or {}
    pickle_path, meta_path = ml_sota_artifact_paths(artifact_root)
    fingerprint = ml_sota_fingerprint(history_df, teams, config, market_rows)
    if not force_refit:
        artifact = read_ml_sota_artifact(pickle_path)
        if artifact and str(artifact.get("fingerprint") or "") == fingerprint:
            artifact["cache_status"] = "hit"
            return artifact
    artifact = train_ml_sota_artifact(
        history_df=history_df,
        teams=teams,
        config=config,
        market_rows=market_rows,
        fingerprint=fingerprint,
    )
    artifact["cache_status"] = "created"
    write_ml_sota_artifact(artifact, pickle_path, meta_path)
    return artifact


def read_ml_sota_artifact(path: Path | None = None) -> Dict[str, Any] | None:
    artifact_path = Path(path) if path is not None else ML_SOTA_PICKLE
    if not artifact_path.exists():
        return None
    try:
        with artifact_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != ML_SOTA_ARTIFACT_SCHEMA:
        return None
    return payload


def write_ml_sota_artifact(artifact: Dict[str, Any], pickle_path: Path, meta_path: Path) -> None:
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with pickle_path.open("wb") as handle:
        pickle.dump(artifact, handle)
    meta_path.write_text(json.dumps(artifact_metadata(artifact), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def artifact_metadata(artifact: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": artifact.get("schema_version"),
        "pipeline_mode": ML_SOTA_PIPELINE_MODE,
        "trained_at": artifact.get("trained_at"),
        "fingerprint": artifact.get("fingerprint"),
        "model_type": artifact.get("model_type"),
        "feature_count": len(artifact.get("feature_columns") or []),
        "train_rows": artifact.get("train_rows", 0),
        "validation_rows": artifact.get("validation_rows", 0),
        "test_rows": artifact.get("test_rows", 0),
        "label_policy_notes": artifact.get("label_policy_notes", []),
        "market_blend_weight": artifact.get("market_blend_weight", 0.5),
        "metrics": artifact.get("metrics", {}),
        "warnings": artifact.get("warnings", []),
    }


def ml_sota_fingerprint(
        history_df: pd.DataFrame,
        teams: Iterable[str],
        config: Dict[str, Any],
        market_rows: pd.DataFrame | None = None,
) -> str:
    history = normalize_history_frame(history_df)
    dates = pd.to_datetime(history.get("Date"), errors="coerce") if not history.empty else pd.Series(dtype="datetime64[ns]")
    payload = {
        "schema": ML_SOTA_ARTIFACT_SCHEMA,
        "sequence": ML_SOTA_SCORE_MODEL_SEQUENCE,
        "teams": sorted(str(team) for team in teams),
        "history_rows": int(history.shape[0]),
        "history_min_date": dates.min().date().isoformat() if not dates.empty and pd.notna(dates.min()) else "",
        "history_max_date": dates.max().date().isoformat() if not dates.empty and pd.notna(dates.max()) else "",
        "max_train_rows": int(config.get("ml_sota_max_train_rows") or ML_SOTA_DEFAULT_MAX_TRAIN_ROWS),
        "poisson_recent_matches": int(config.get("poisson_recent_matches") or 15),
        "history_weight": round(float(config.get("history_weight", 1.0) or 1.0), 6),
        "recency_weight": round(float(config.get("recency_weight", 0.35) or 0.35), 6),
        "host_advantage": round(float(config.get("host_advantage", 45.0) or 45.0), 6),
        "max_goals": int(config.get("max_goals") or 10),
        "market_rows": int(market_rows.shape[0]) if isinstance(market_rows, pd.DataFrame) else 0,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def train_ml_sota_artifact(
        history_df: pd.DataFrame,
        teams: Iterable[str],
        config: Dict[str, Any] | None = None,
        market_rows: pd.DataFrame | None = None,
        fingerprint: str = "",
) -> Dict[str, Any]:
    config = config or {}
    warnings: List[str] = []
    rows = normalize_history_frame(history_df)
    rows, removed_2026 = exclude_worldcup_2026_labels(rows)
    if removed_2026:
        warnings.append(f"{removed_2026} labels Mundial 2026 excluidos del stacking.")
    max_rows = int(config.get("ml_sota_max_train_rows") or ML_SOTA_DEFAULT_MAX_TRAIN_ROWS)
    min_prior = int(config.get("ml_sota_min_prior_rows") or ML_SOTA_MIN_PRIOR_ROWS)
    teams = sorted(set(str(team) for team in teams) | set(rows.get("Team 1", pd.Series(dtype=str)).astype(str)) | set(rows.get("Team 2", pd.Series(dtype=str)).astype(str)))
    rows = rows.reset_index(drop=True)
    eligible_indices = [index for index in range(len(rows)) if index >= min_prior]
    if max_rows > 0:
        eligible_indices = eligible_indices[-max_rows:]
    market_rows = market_rows if isinstance(market_rows, pd.DataFrame) else load_cached_market_rows()
    market_blend_weight = optimize_market_blend_weight(rows, eligible_indices, teams, config, market_rows)
    config = {**config, "ml_sota_market_blend_weight": market_blend_weight}

    records: List[Dict[str, float]] = []
    labels: List[Dict[str, Any]] = []
    row_meta: List[Dict[str, Any]] = []
    model_warnings: List[str] = []
    for index in eligible_indices:
        row = rows.iloc[index]
        prior = rows.iloc[:index].copy()
        if prior.empty:
            continue
        try:
            base_model = WorldCupModel.from_history(
                prior,
                teams=teams,
                history_weight=float(config.get("history_weight", 1.0)),
                recency_weight=float(config.get("recency_weight", 0.35)),
                host_advantage=float(config.get("host_advantage", 45.0)),
                max_goals=int(config.get("max_goals") or 10),
            )
            expert_reports = historical_expert_reports(
                base_model=base_model,
                prior_df=prior,
                teams=teams,
                row=row,
                config=config,
                market_rows=market_rows,
            )
            model_warnings.extend(report_warnings(expert_reports))
            records.append(stack_feature_row(
                model_reports=expert_reports,
                base_model=base_model,
                fixture=history_row_to_fixture(row),
                market_rows=market_rows,
                config=config,
            ))
            labels.append(label_payload(row))
            row_meta.append({
                "row_index": int(index),
                "date": str(row.get("Date", "")),
                "home": str(row.get("Team 1", "")),
                "away": str(row.get("Team 2", "")),
                "prior_rows": int(prior.shape[0]),
                "max_prior_date": str(prior["Date"].max()) if "Date" in prior else "",
            })
        except Exception as exc:
            warnings.append(f"Fila historica {index} omitida ({exc.__class__.__name__}: {exc}).")

    if not records:
        warnings.append("Sin filas suficientes para entrenar ML+SOTA; se usara consenso como fallback.")
        return heuristic_artifact(
            fingerprint=fingerprint,
            warnings=warnings,
            rows=rows,
            removed_2026=removed_2026,
            market_blend_weight=market_blend_weight,
        )

    x = pd.DataFrame(records).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    feature_columns = list(x.columns)
    split = temporal_split_indices(len(x))
    models: Dict[str, Any] = {}
    calibrators: Dict[str, Any] = {}
    classes: Dict[str, List[Any]] = {}
    metrics: Dict[str, Any] = {}
    model_type = "heuristic_consensus"
    train_x = x.iloc[split["train"]]
    validation_x = x.iloc[split["validation"]]
    test_x = x.iloc[split["test"]]

    for target in ML_SOTA_TARGETS:
        target_values = pd.Series([item.get(target) for item in labels])
        train_y = target_values.iloc[split["train"]]
        validation_y = target_values.iloc[split["validation"]]
        test_y = target_values.iloc[split["test"]]
        target_classes = RESULT_CLASSES if target == "result" else [0, 1]
        model, fitted_type, fit_warning = fit_probability_model(train_x, train_y, target_classes, int(config.get("seed") or 2026), target)
        if fit_warning:
            warnings.append(fit_warning)
        if model is not None:
            models[target] = model
            model_type = fitted_type if model_type == "heuristic_consensus" else model_type
        else:
            models[target] = None
        classes[target] = list(target_classes)
        raw_validation = predict_target_raw_probabilities(models[target], validation_x, target_classes, target)
        calibrator, calibration_warning = fit_probability_calibrator(raw_validation, validation_y, target_classes, target)
        calibrators[target] = calibrator
        if calibration_warning:
            warnings.append(calibration_warning)
        raw_test = predict_target_raw_probabilities(models[target], test_x, target_classes, target)
        calibrated_test = apply_probability_calibrator(raw_test, calibrator, target_classes)
        metric_y = test_y if len(test_y) else validation_y
        metric_probs = calibrated_test if len(test_y) else apply_probability_calibrator(raw_validation, calibrator, target_classes)
        metrics[target] = probability_metrics(metric_y.tolist(), metric_probs, target_classes)

    return {
        "schema_version": ML_SOTA_ARTIFACT_SCHEMA,
        "pipeline_mode": ML_SOTA_PIPELINE_MODE,
        "fingerprint": fingerprint,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": model_type,
        "models": models,
        "calibrators": calibrators,
        "classes": classes,
        "feature_columns": feature_columns,
        "train_rows": len(split["train"]),
        "validation_rows": len(split["validation"]),
        "test_rows": len(split["test"]),
        "row_metadata": row_meta,
        "metrics": metrics,
        "market_blend_weight": market_blend_weight,
        "warnings": unique_strings([*warnings, *model_warnings]),
        "label_policy_notes": [
            "No usa partidos del Mundial 2026 como labels.",
            "Cada fila historica se featuriza con historial estrictamente anterior a la fecha del partido.",
            "El calibrador usa la particion temporal de validacion; test temporal queda para metricas finales.",
        ],
        "removed_worldcup_2026_labels": int(removed_2026),
    }


def heuristic_artifact(
        fingerprint: str,
        warnings: List[str],
        rows: pd.DataFrame,
        removed_2026: int,
        market_blend_weight: float,
) -> Dict[str, Any]:
    return {
        "schema_version": ML_SOTA_ARTIFACT_SCHEMA,
        "pipeline_mode": ML_SOTA_PIPELINE_MODE,
        "fingerprint": fingerprint,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "heuristic_consensus",
        "models": {target: None for target in ML_SOTA_TARGETS},
        "calibrators": {target: IdentityCalibrator(RESULT_CLASSES if target == "result" else [0, 1]) for target in ML_SOTA_TARGETS},
        "classes": {target: (RESULT_CLASSES if target == "result" else [0, 1]) for target in ML_SOTA_TARGETS},
        "feature_columns": [],
        "train_rows": 0,
        "validation_rows": 0,
        "test_rows": 0,
        "row_metadata": [],
        "metrics": {target: {"brier": 0.0, "log_loss": 0.0, "ece": 0.0, "samples": 0} for target in ML_SOTA_TARGETS},
        "market_blend_weight": market_blend_weight,
        "warnings": unique_strings(warnings),
        "label_policy_notes": [
            "No usa partidos del Mundial 2026 como labels.",
            "Fallback sin entrenamiento por falta de filas historicas suficientes.",
        ],
        "removed_worldcup_2026_labels": int(removed_2026),
        "history_rows": int(rows.shape[0]),
    }


def historical_expert_reports(
        base_model: WorldCupModel,
        prior_df: pd.DataFrame,
        teams: Iterable[str],
        row: pd.Series,
        config: Dict[str, Any],
        market_rows: pd.DataFrame | None,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    fixture = history_row_to_fixture(row)
    for key in ML_SOTA_SCORE_MODEL_SEQUENCE:
        if key == "market_blended_poisson":
            probabilities, metadata = market_blended_poisson_probabilities(base_model, fixture, config, market_rows)
        else:
            try:
                if key == DEFAULT_SCORE_MODEL:
                    model = base_model
                    metadata = {"key": key, "label": score_model_label(key), "available": True, "warnings": [], "params": {}}
                else:
                    model = build_score_model(
                        base_model,
                        history_df=prior_df,
                        teams=teams,
                        config={**config, "score_model": key, "stat_model_cache": True, "stat_model_refit": False},
                    )
                    metadata = score_model_metadata(model, key)
                probabilities = model_probabilities(model, fixture, config)
            except Exception as exc:
                probabilities = base_model.match_probabilities(str(row.get("Team 1", "")), str(row.get("Team 2", "")), max_goals=int(config.get("max_goals") or 10))
                metadata = {
                    "key": key,
                    "label": score_model_label(key),
                    "available": False,
                    "warnings": [f"{exc.__class__.__name__}: {exc}; fallback Poisson independiente."],
                    "params": {},
                }
        reports.append(expert_report_from_probabilities(key, metadata, probabilities, fixture, config))
    return reports


def score_model_metadata(model: Any, fallback_key: str) -> Dict[str, Any]:
    method = getattr(model, "score_model_metadata", None)
    if callable(method):
        payload = method()
        if isinstance(payload, dict):
            return payload
    return {"key": fallback_key, "label": score_model_label(fallback_key), "available": True, "warnings": [], "params": {}}


def model_probabilities(model: Any, fixture: pd.Series, config: Dict[str, Any]) -> Dict[str, Any]:
    home = str(fixture.get("Equipo 1", ""))
    away = str(fixture.get("Equipo 2", ""))
    record = fixture.to_dict() if hasattr(fixture, "to_dict") else dict(fixture)
    max_goals = int(config.get("max_goals") or 10)
    method = getattr(model, "match_probabilities_for_match", None)
    if callable(method):
        return method(home, away, match=record, max_goals=max_goals)
    return model.match_probabilities(home, away, max_goals=max_goals)


def expert_report_from_probabilities(
        model_key: str,
        metadata: Dict[str, Any],
        probabilities: Dict[str, Any],
        fixture: pd.Series,
        config: Dict[str, Any],
) -> Dict[str, Any]:
    key = str(model_key or metadata.get("key") or DEFAULT_SCORE_MODEL)
    available = bool(metadata.get("available", True))
    max_goals = int(config.get("max_goals") or 10)
    lambda_home = safe_float(probabilities.get("lambda1", 0.0))
    lambda_away = safe_float(probabilities.get("lambda2", 0.0))
    grid = poisson_score_grid(lambda_home, lambda_away, max_goals=max_goals)
    if key != "market_blended_poisson" and metadata.get("available", True):
        try:
            grid = model_score_grid_from_probabilities(probabilities, lambda_home, lambda_away, max_goals)
        except Exception:
            grid = poisson_score_grid(lambda_home, lambda_away, max_goals=max_goals)
    return {
        "model_key": key,
        "model_label": str(metadata.get("label") or score_model_label(key)),
        "available": available,
        "consensus_eligible": bool(available or key == DEFAULT_SCORE_MODEL),
        "fallback": not available,
        "warnings": [str(item) for item in metadata.get("warnings", []) if str(item)],
        "probabilities": {
            "home": safe_float(probabilities.get("home", 0.0)) * 100.0,
            "draw": safe_float(probabilities.get("draw", 0.0)) * 100.0,
            "away": safe_float(probabilities.get("away", 0.0)) * 100.0,
            **{
                f"over{total_line_suffix(line)}": safe_float(probabilities.get(f"over{total_line_suffix(line)}", 0.0)) * 100.0
                for line in TRAIN_TOTAL_GOAL_LINES
            },
            **{
                f"under{total_line_suffix(line)}": safe_float(probabilities.get(f"under{total_line_suffix(line)}", 0.0)) * 100.0
                for line in TRAIN_TOTAL_GOAL_LINES
            },
        },
        "expected_goals": {"home": lambda_home, "away": lambda_away},
        "score_distribution": score_distribution_from_grid(grid, lambda_home, lambda_away),
        "params": metadata.get("params", {}),
    }


def model_score_grid_from_probabilities(probabilities: Dict[str, Any], lambda_home: float, lambda_away: float, max_goals: int) -> np.ndarray:
    return poisson_score_grid(lambda_home, lambda_away, max_goals=max_goals)


def score_distribution_from_grid(grid: np.ndarray, lambda_home: float, lambda_away: float) -> Dict[str, Any]:
    grid = normalize_grid(grid)
    probs = score_grid_probabilities_percent(grid)
    top_scores = top_scores_from_grid(grid)
    return {
        "available": True,
        "lambdas": {"home": round(lambda_home, 3), "away": round(lambda_away, 3)},
        "probabilities": probs,
        "score_matrix": [[round(float(value) * 100.0, 3) for value in row] for row in grid.tolist()],
        "top_scores": top_scores,
    }


def score_grid_probabilities_percent(grid: np.ndarray) -> Dict[str, float]:
    grid = normalize_grid(grid)
    goals = np.arange(grid.shape[0], dtype=int)
    home_goals, away_goals = np.meshgrid(goals, goals, indexing="ij")
    margin = home_goals - away_goals
    total_goals = home_goals + away_goals
    output = {
        "home": round(float(grid[margin > 0].sum()) * 100.0, 2),
        "draw": round(float(grid[margin == 0].sum()) * 100.0, 2),
        "away": round(float(grid[margin < 0].sum()) * 100.0, 2),
    }
    for line in TRAIN_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        over = float(grid[total_goals > line].sum())
        output[f"over{suffix}"] = round(over * 100.0, 2)
        output[f"under{suffix}"] = round((1.0 - over) * 100.0, 2)
    return output


def top_scores_from_grid(grid: np.ndarray, limit: int = 5) -> List[Dict[str, Any]]:
    grid = normalize_grid(grid)
    ranked = np.argsort(grid.ravel())[::-1]
    rows, cols = grid.shape
    output = []
    for index in ranked[:limit]:
        home = int(index // cols)
        away = int(index % cols)
        output.append({
            "score": f"{home}-{away}",
            "home_goals": home,
            "away_goals": away,
            "probability": round(float(grid[home, away]) * 100.0, 3),
            "probability_raw": float(grid[home, away]),
        })
    return output


def stack_feature_row(
        model_reports: List[Dict[str, Any]],
        base_model: WorldCupModel,
        fixture: pd.Series | Dict[str, Any],
        market_rows: pd.DataFrame | None,
        config: Dict[str, Any],
) -> Dict[str, float]:
    fixture_series = fixture if isinstance(fixture, pd.Series) else pd.Series(fixture)
    home = str(fixture_series.get("Equipo 1", fixture_series.get("home", "")))
    away = str(fixture_series.get("Equipo 2", fixture_series.get("away", "")))
    match_date = fixture_series.get("Fecha", fixture_series.get("Date", ""))
    try:
        from src.worldcup.training import (
            build_history_feature_table,
            build_market_lookup,
            build_matchup_feature_table,
            match_feature_row,
        )

        row = match_feature_row(
            base_model,
            pd.DataFrame(),
            home,
            away,
            history_team_features=build_history_feature_table(pd.DataFrame()),
            matchup_features=build_matchup_feature_table(pd.DataFrame()),
            market_rows=market_rows if isinstance(market_rows, pd.DataFrame) else pd.DataFrame(),
            market_lookup=build_market_lookup(market_rows) if isinstance(market_rows, pd.DataFrame) else {},
            fixture_id=fixture_series.get("No.", fixture_series.get("FixtureId", "")),
            match_date=match_date,
            match_year=safe_int(fixture_series.get("Year", 0)),
            fixture_context=fixture_series.to_dict(),
            feature_profile=FEATURE_PROFILE_BALANCED,
        )
    except Exception:
        row = {}
    reports_by_key = {str(report.get("model_key") or ""): report for report in model_reports}
    for key in ML_SOTA_SCORE_MODEL_SEQUENCE:
        row.update(expert_feature_block(key, reports_by_key.get(key)))
    row.update(consensus_feature_block(model_reports))
    return {str(key): safe_float(value) for key, value in row.items()}


def expert_feature_block(model_key: str, report: Dict[str, Any] | None) -> Dict[str, float]:
    prefix = f"expert_{model_key}"
    output = {f"{prefix}_available": 0.0}
    for key in ("lambda_home", "lambda_away", "home", "draw", "away", "entropy_1x2", "gap_1x2", "p00", "p11", "low_score_mass", "top_score_probability"):
        output[f"{prefix}_{key}"] = 0.0
    for line in TRAIN_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        output[f"{prefix}_over{suffix}"] = 0.0
        output[f"{prefix}_under{suffix}"] = 0.0
    if not report or not report.get("available"):
        return output
    probs = probabilities_fraction(report.get("probabilities") or {})
    expected = report.get("expected_goals") or {}
    p1x2 = [probs.get("home", 0.0), probs.get("draw", 0.0), probs.get("away", 0.0)]
    sorted_probs = sorted(p1x2, reverse=True)
    output.update({
        f"{prefix}_available": 1.0,
        f"{prefix}_lambda_home": safe_float(expected.get("home")),
        f"{prefix}_lambda_away": safe_float(expected.get("away")),
        f"{prefix}_home": probs.get("home", 0.0),
        f"{prefix}_draw": probs.get("draw", 0.0),
        f"{prefix}_away": probs.get("away", 0.0),
        f"{prefix}_entropy_1x2": entropy(p1x2),
        f"{prefix}_gap_1x2": sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0.0,
    })
    for line in TRAIN_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        output[f"{prefix}_over{suffix}"] = probs.get(f"over{suffix}", 0.0)
        output[f"{prefix}_under{suffix}"] = probs.get(f"under{suffix}", 0.0)
    matrix = ((report.get("score_distribution") or {}).get("score_matrix") or [])
    if matrix:
        grid = normalize_grid(np.asarray(matrix, dtype=float) / 100.0)
        output[f"{prefix}_p00"] = float(grid[0, 0]) if grid.shape[0] > 0 and grid.shape[1] > 0 else 0.0
        output[f"{prefix}_p11"] = float(grid[1, 1]) if grid.shape[0] > 1 and grid.shape[1] > 1 else 0.0
        visible = grid[:min(3, grid.shape[0]), :min(3, grid.shape[1])]
        output[f"{prefix}_low_score_mass"] = float(visible.sum())
        output[f"{prefix}_top_score_probability"] = float(grid.max()) if grid.size else 0.0
    return output


def consensus_feature_block(model_reports: List[Dict[str, Any]]) -> Dict[str, float]:
    eligible = [report for report in model_reports if report.get("available") and not report.get("fallback")]
    output: Dict[str, float] = {"consensus_model_count": float(len(eligible))}
    for key in ("home", "draw", "away"):
        values = [probabilities_fraction(report.get("probabilities") or {}).get(key, 0.0) for report in eligible]
        output[f"consensus_{key}_mean"] = float(np.mean(values)) if values else 0.0
        output[f"consensus_{key}_std"] = float(np.std(values)) if values else 0.0
    for line in TRAIN_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        values = [probabilities_fraction(report.get("probabilities") or {}).get(f"over{suffix}", 0.0) for report in eligible]
        output[f"consensus_over{suffix}_mean"] = float(np.mean(values)) if values else 0.0
        output[f"consensus_over{suffix}_std"] = float(np.std(values)) if values else 0.0
    return output


def predict_ml_sota_for_report(
        artifact: Dict[str, Any],
        model_reports: List[Dict[str, Any]],
        base_model: WorldCupModel,
        fixture: pd.Series | Dict[str, Any],
        config: Dict[str, Any] | None = None,
        market_rows: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    config = config or {}
    feature_columns = list(artifact.get("feature_columns") or [])
    feature_row = stack_feature_row(model_reports, base_model, fixture, market_rows, config)
    x = pd.DataFrame([feature_row])
    if feature_columns:
        x = align_frame_to_columns(x, feature_columns)
    raw: Dict[str, Dict[str, float]] = {}
    calibrated: Dict[str, Dict[str, float]] = {}
    for target in ML_SOTA_TARGETS:
        target_classes = list((artifact.get("classes") or {}).get(target) or (RESULT_CLASSES if target == "result" else [0, 1]))
        raw_array = predict_target_raw_probabilities((artifact.get("models") or {}).get(target), x, target_classes, target)
        calibrated_array = apply_probability_calibrator(raw_array, (artifact.get("calibrators") or {}).get(target), target_classes)
        raw[target] = target_probability_payload(target, target_classes, raw_array[0] if raw_array.size else np.asarray([]), percent=True)
        calibrated[target] = target_probability_payload(target, target_classes, calibrated_array[0] if calibrated_array.size else np.asarray([]), percent=True)
    probabilities = merge_target_probability_payloads(calibrated)
    raw_probabilities = merge_target_probability_payloads(raw)
    outcome = max(("home", "draw", "away"), key=lambda key: probabilities.get(key, 0.0))
    return {
        "available": True,
        "pipeline_mode": ML_SOTA_PIPELINE_MODE,
        "model_label": ml_sota_label(),
        "artifact": artifact_metadata(artifact),
        "cache_status": artifact.get("cache_status", ""),
        "raw_probabilities": raw_probabilities,
        "calibrated_probabilities": probabilities,
        "target_probabilities": calibrated,
        "raw_target_probabilities": raw,
        "outcome": outcome,
        "outcome_label": {"home": "1", "draw": "X", "away": "2"}.get(outcome, ""),
        "outcome_probability": round(probabilities.get(outcome, 0.0), 2),
        "metrics": artifact.get("metrics", {}),
        "feature_count": len(feature_columns),
        "warnings": artifact.get("warnings", []),
        "calibrated_probability_units": "percent",
    }


def merge_target_probability_payloads(targets: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    result = dict(targets.get("result") or {})
    for target in ML_SOTA_TARGETS:
        if target == "result":
            continue
        result.update(targets.get(target) or {})
    return result


def target_probability_payload(target: str, classes: List[Any], probabilities: np.ndarray, percent: bool = False) -> Dict[str, float]:
    probs = align_probabilities_to_classes(probabilities, classes)
    multiplier = 100.0 if percent else 1.0
    if target == "result":
        return {
            RESULT_CLASS_TO_OUTCOME.get(str(label), str(label)): round(float(prob) * multiplier, 4)
            for label, prob in zip(classes, probs)
        }
    line_suffix = target.split("_")[-1]
    return {
        f"under{line_suffix}": round(float(probs[0]) * multiplier, 4) if len(probs) > 0 else 0.0,
        f"over{line_suffix}": round(float(probs[1]) * multiplier, 4) if len(probs) > 1 else 0.0,
    }


def fit_probability_model(x: pd.DataFrame, y: pd.Series, classes: List[Any], seed: int, target: str) -> Tuple[Any, str, str]:
    y = y.dropna()
    if x.empty or y.empty or y.nunique() < 2:
        return None, "heuristic_consensus", f"{target}: clases insuficientes; se usa consenso como fallback ML."
    x_fit = x.loc[y.index]
    try:
        from catboost import CatBoostClassifier  # type: ignore

        loss = "MultiClass" if len(classes) > 2 else "Logloss"
        model = CatBoostClassifier(
            iterations=120,
            depth=4,
            learning_rate=0.05,
            loss_function=loss,
            random_seed=int(seed),
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x_fit, y)
        return model, "catboost", ""
    except Exception as exc:
        catboost_error = f"CatBoost no disponible para {target}: {exc.__class__.__name__}"
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        model = LogisticRegression(max_iter=1200, class_weight="balanced")
        model.fit(x_fit, y)
        return model, "sklearn_logistic", catboost_error
    except Exception as exc:
        return None, "heuristic_consensus", f"{catboost_error}; sklearn fallback no disponible para {target}: {exc.__class__.__name__}."


def predict_target_raw_probabilities(model: Any, x: pd.DataFrame, classes: List[Any], target: str) -> np.ndarray:
    if x is None:
        x = pd.DataFrame()
    if model is None:
        return heuristic_target_probabilities(x, classes, target)
    try:
        probs = np.asarray(model.predict_proba(x), dtype=float)
        model_classes = [str(item) for item in getattr(model, "classes_", classes)]
        return align_probability_matrix(probs, model_classes, [str(item) for item in classes])
    except Exception:
        return heuristic_target_probabilities(x, classes, target)


def heuristic_target_probabilities(x: pd.DataFrame, classes: List[Any], target: str) -> np.ndarray:
    n = max(int(getattr(x, "shape", [0])[0]), 0)
    if n <= 0:
        return np.zeros((0, len(classes)), dtype=float)
    if target == "result":
        home = x.get("consensus_home_mean", pd.Series([0.0] * n)).to_numpy(dtype=float)
        draw = x.get("consensus_draw_mean", pd.Series([0.0] * n)).to_numpy(dtype=float)
        away = x.get("consensus_away_mean", pd.Series([0.0] * n)).to_numpy(dtype=float)
        return _normalize_rows(np.column_stack([home, draw, away]))
    suffix = target.split("_")[-1]
    over = x.get(f"consensus_over{suffix}_mean", pd.Series([0.0] * n)).to_numpy(dtype=float)
    over = np.clip(over, 0.0, 1.0)
    return _normalize_rows(np.column_stack([1.0 - over, over]))


def fit_probability_calibrator(raw_probs: np.ndarray, y: pd.Series, classes: List[Any], target: str) -> Tuple[Any, str]:
    if raw_probs.size == 0 or y.empty:
        return IdentityCalibrator(classes), f"{target}: calibracion identidad por falta de validacion."
    y = y.reset_index(drop=True)
    raw_probs = raw_probs[:len(y)]
    valid = y.notna()
    y_valid = y[valid]
    raw_valid = raw_probs[valid.to_numpy()]
    if len(y_valid) < max(8, len(classes) * 3) or y_valid.nunique() < 2:
        return IdentityCalibrator(classes), f"{target}: validacion insuficiente; calibracion identidad."
    if target == "result" and set(map(str, y_valid.unique())) != set(map(str, classes)):
        return IdentityCalibrator(classes), f"{target}: validacion sin todas las clases; calibracion identidad."
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore

        model = LogisticRegression(max_iter=1000)
        model.fit(np.log(np.clip(raw_valid, 1e-9, 1.0)), y_valid)
        return model, ""
    except Exception as exc:
        return IdentityCalibrator(classes), f"{target}: calibrador logistico no disponible ({exc.__class__.__name__}); identidad."


def apply_probability_calibrator(raw_probs: np.ndarray, calibrator: Any, classes: List[Any]) -> np.ndarray:
    raw_probs = _normalize_rows(raw_probs)
    if raw_probs.size == 0:
        return raw_probs
    if calibrator is None:
        return raw_probs
    try:
        calibrated = np.asarray(calibrator.predict_proba(np.log(np.clip(raw_probs, 1e-9, 1.0))), dtype=float)
        model_classes = [str(item) for item in getattr(calibrator, "classes_", classes)]
        return align_probability_matrix(calibrated, model_classes, [str(item) for item in classes])
    except Exception:
        return raw_probs


def probability_metrics(y_true: List[Any], probabilities: np.ndarray, classes: List[Any]) -> Dict[str, Any]:
    valid_pairs = [(label, probabilities[index]) for index, label in enumerate(y_true) if index < len(probabilities) and label is not None and str(label) != "nan"]
    if not valid_pairs:
        return {"brier": 0.0, "log_loss": 0.0, "ece": 0.0, "samples": 0}
    class_text = [str(item) for item in classes]
    y = [str(label) for label, _ in valid_pairs]
    probs = _normalize_rows(np.asarray([prob for _, prob in valid_pairs], dtype=float))
    one_hot = np.zeros_like(probs)
    for index, label in enumerate(y):
        if label in class_text:
            one_hot[index, class_text.index(label)] = 1.0
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    clipped = np.clip(probs, 1e-12, 1.0)
    log_values = []
    correct = []
    confidence = []
    for index, label in enumerate(y):
        class_index = class_text.index(label) if label in class_text else int(np.argmax(one_hot[index]))
        log_values.append(-math.log(float(clipped[index, class_index])))
        pred_index = int(np.argmax(probs[index]))
        correct.append(1.0 if pred_index == class_index else 0.0)
        confidence.append(float(probs[index, pred_index]))
    return {
        "brier": round(brier, 6),
        "log_loss": round(float(np.mean(log_values)), 6),
        "ece": round(expected_calibration_error(confidence, correct), 6),
        "samples": len(y),
    }


def expected_calibration_error(confidence: List[float], correct: List[float], bins: int = 10) -> float:
    if not confidence:
        return 0.0
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct, dtype=float)
    ece = 0.0
    for bin_index in range(int(bins)):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        mask = (conf > lower if bin_index else conf >= lower) & (conf <= upper)
        if not mask.any():
            continue
        ece += float(mask.mean()) * abs(float(conf[mask].mean()) - float(corr[mask].mean()))
    return float(ece)


def market_blended_poisson_probabilities(
        base_model: WorldCupModel,
        fixture: pd.Series | Dict[str, Any],
        config: Dict[str, Any] | None = None,
        market_rows: pd.DataFrame | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    config = config or {}
    fixture_series = fixture if isinstance(fixture, pd.Series) else pd.Series(fixture)
    home = str(fixture_series.get("Equipo 1", fixture_series.get("home", "")))
    away = str(fixture_series.get("Equipo 2", fixture_series.get("away", "")))
    max_goals = int(config.get("max_goals") or 10)
    base = base_model.match_probabilities(home, away, max_goals=max_goals)
    base_lambdas = (safe_float(base.get("lambda1", 1.2)), safe_float(base.get("lambda2", 1.0)))
    market_rows = market_rows if isinstance(market_rows, pd.DataFrame) else load_cached_market_rows()
    market_row = market_for_match(
        market_rows,
        home,
        away,
        match_date=fixture_series.get("Fecha", fixture_series.get("date", "")),
        year=safe_int(fixture_series.get("Year", 2026)) or 2026,
    )
    inferred = infer_lambdas_from_market_odds(market_row, base_lambdas, max_goals=max_goals)
    if not inferred.get("available"):
        return base, {
            "key": "market_blended_poisson",
            "label": "Market blended Poisson",
            "available": False,
            "params": {"blend_weight": safe_float(config.get("ml_sota_market_blend_weight", 0.5))},
            "warnings": [str(inferred.get("warning") or "Odds de mercado insuficientes para market_blended_poisson.")],
        }
    weight = float(np.clip(safe_float(config.get("ml_sota_market_blend_weight", 0.5)), 0.0, 1.0))
    market_lambda_home = safe_float(inferred.get("lambda_home"))
    market_lambda_away = safe_float(inferred.get("lambda_away"))
    lambda_home = ((1.0 - weight) * base_lambdas[0]) + (weight * market_lambda_home)
    lambda_away = ((1.0 - weight) * base_lambdas[1]) + (weight * market_lambda_away)
    probabilities = probabilities_from_score_grid(
        poisson_score_grid(lambda_home, lambda_away, max_goals=max_goals),
        lambda1=lambda_home,
        lambda2=lambda_away,
    )
    return probabilities, {
        "key": "market_blended_poisson",
        "label": "Market blended Poisson",
        "available": True,
        "params": {
            "blend_weight": round(weight, 4),
            "lambda_hist_home": round(base_lambdas[0], 4),
            "lambda_hist_away": round(base_lambdas[1], 4),
            "lambda_market_home": round(market_lambda_home, 4),
            "lambda_market_away": round(market_lambda_away, 4),
            "market_source": str(market_row.get("market_source", "")),
        },
        "warnings": [],
    }


def infer_lambdas_from_market_odds(
        market_row: Dict[str, Any] | None,
        base_lambdas: Tuple[float, float] = (1.2, 1.0),
        max_goals: int = 10,
) -> Dict[str, Any]:
    market_row = market_row or {}
    odds_home = market_row.get("market_odds_home")
    odds_draw = market_row.get("market_odds_draw")
    odds_away = market_row.get("market_odds_away")
    has_1x2 = all(valid_decimal_odd(value) > 0.0 for value in (odds_home, odds_draw, odds_away))
    odds_over25 = market_row.get("market_odds_over25")
    odds_under25 = market_row.get("market_odds_under25")
    has_ou25 = all(valid_decimal_odd(value) > 0.0 for value in (odds_over25, odds_under25))
    if not has_1x2 and not has_ou25:
        return {"available": False, "warning": "Sin odds 1X2 ni U/O 2.5 validas."}
    targets: Dict[str, float] = {}
    vig: Dict[str, float] = {}
    if has_1x2:
        _, no_vig, market_vig = no_vig_probabilities({"home": float(odds_home), "draw": float(odds_draw), "away": float(odds_away)})
        targets.update(no_vig)
        vig["1x2"] = market_vig
    if has_ou25:
        _, no_vig, market_vig = no_vig_probabilities({"over25": float(odds_over25), "under25": float(odds_under25)})
        targets.update(no_vig)
        vig["ou25"] = market_vig
    lambda_home, lambda_away = optimize_lambdas_to_market(targets, base_lambdas, max_goals=max_goals)
    return {
        "available": True,
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "targets": targets,
        "vig": vig,
    }


def optimize_lambdas_to_market(targets: Dict[str, float], base_lambdas: Tuple[float, float], max_goals: int = 10) -> Tuple[float, float]:
    base_home = float(np.clip(base_lambdas[0], 0.2, 4.5))
    base_away = float(np.clip(base_lambdas[1], 0.2, 4.5))

    def objective(values: Sequence[float]) -> float:
        home, away = float(values[0]), float(values[1])
        probs = probabilities_from_score_grid(poisson_score_grid(home, away, max_goals=max_goals), home, away)
        total = 0.0
        for key, target in targets.items():
            total += (float(probs.get(key, 0.0)) - float(target)) ** 2
        total += 0.01 * ((home - base_home) ** 2 + (away - base_away) ** 2)
        return total

    try:
        from scipy import optimize  # type: ignore

        result = optimize.minimize(
            objective,
            np.asarray([base_home, base_away], dtype=float),
            bounds=[(0.2, 4.8), (0.2, 4.8)],
            method="L-BFGS-B",
        )
        if result.success:
            return float(result.x[0]), float(result.x[1])
    except Exception:
        pass
    candidates = np.linspace(0.55, 1.55, 17)
    best = (base_home, base_away)
    best_score = float("inf")
    for home_scale in candidates:
        for away_scale in candidates:
            values = (float(np.clip(base_home * home_scale, 0.2, 4.8)), float(np.clip(base_away * away_scale, 0.2, 4.8)))
            score = objective(values)
            if score < best_score:
                best = values
                best_score = score
    return best


def optimize_market_blend_weight(
        rows: pd.DataFrame,
        indices: List[int],
        teams: Iterable[str],
        config: Dict[str, Any],
        market_rows: pd.DataFrame | None,
) -> float:
    scored: List[Tuple[float, str, np.ndarray]] = []
    validation_indices = indices[int(len(indices) * 0.8):] if len(indices) > 5 else indices
    for index in validation_indices[-40:]:
        row = rows.iloc[index]
        prior = rows.iloc[:index]
        if prior.empty:
            continue
        fixture = history_row_to_fixture(row)
        base_model = WorldCupModel.from_history(
            prior,
            teams=teams,
            history_weight=float(config.get("history_weight", 1.0)),
            recency_weight=float(config.get("recency_weight", 0.35)),
            host_advantage=float(config.get("host_advantage", 45.0)),
            max_goals=int(config.get("max_goals") or 10),
        )
        base = base_model.match_probabilities(str(row.get("Team 1", "")), str(row.get("Team 2", "")), max_goals=int(config.get("max_goals") or 10))
        inferred = infer_lambdas_from_market_odds(
            market_for_match(
                market_rows if isinstance(market_rows, pd.DataFrame) else pd.DataFrame(),
                str(row.get("Team 1", "")),
                str(row.get("Team 2", "")),
                match_date=row.get("Date", ""),
                year=safe_int(row.get("Year", 0)),
            ),
            (safe_float(base.get("lambda1", 1.2)), safe_float(base.get("lambda2", 1.0))),
            max_goals=int(config.get("max_goals") or 10),
        )
        if not inferred.get("available"):
            continue
        label = label_from_goals(row.get("G1"), row.get("G2"))
        if label not in RESULT_CLASSES:
            continue
        scored.append((
            safe_float(base.get("lambda1", 1.2)),
            safe_float(base.get("lambda2", 1.0)),
            label,
            np.asarray([safe_float(inferred.get("lambda_home")), safe_float(inferred.get("lambda_away"))], dtype=float),
        ))
    if not scored:
        return 0.5
    candidates = [0.0, 0.25, 0.5, 0.65, 0.8, 1.0]
    best_weight = 0.5
    best_loss = float("inf")
    class_index = {label: index for index, label in enumerate(RESULT_CLASSES)}
    for weight in candidates:
        losses = []
        for base_home, base_away, label, market_lambdas in scored:
            lambda_home = ((1.0 - weight) * base_home) + (weight * market_lambdas[0])
            lambda_away = ((1.0 - weight) * base_away) + (weight * market_lambdas[1])
            probs = probabilities_from_score_grid(poisson_score_grid(lambda_home, lambda_away, max_goals=int(config.get("max_goals") or 10)), lambda_home, lambda_away)
            vector = np.asarray([probs["home"], probs["draw"], probs["away"]], dtype=float)
            one_hot = np.zeros(3, dtype=float)
            one_hot[class_index[label]] = 1.0
            losses.append(float(np.sum((vector - one_hot) ** 2)))
        loss = float(np.mean(losses))
        if loss < best_loss:
            best_loss = loss
            best_weight = weight
    return float(best_weight)


def normalize_history_frame(history_df: pd.DataFrame | None) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return pd.DataFrame(columns=["Date", "Year", "Team 1", "Team 2", "G1", "G2", "Round", "Group"])
    working = history_df.copy()
    rename = {}
    if "Home" in working.columns and "Team 1" not in working.columns:
        rename["Home"] = "Team 1"
    if "Away" in working.columns and "Team 2" not in working.columns:
        rename["Away"] = "Team 2"
    if "HG" in working.columns and "G1" not in working.columns:
        rename["HG"] = "G1"
    if "AG" in working.columns and "G2" not in working.columns:
        rename["AG"] = "G2"
    working = working.rename(columns=rename)
    for column in ("Date", "Team 1", "Team 2", "G1", "G2"):
        if column not in working.columns:
            working[column] = np.nan
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    working["Year"] = pd.to_numeric(working.get("Year", working["Date"].dt.year), errors="coerce").fillna(working["Date"].dt.year)
    working["G1"] = pd.to_numeric(working["G1"], errors="coerce")
    working["G2"] = pd.to_numeric(working["G2"], errors="coerce")
    working = working[
        working["Date"].notna()
        & working["Team 1"].astype(str).str.len().gt(1)
        & working["Team 2"].astype(str).str.len().gt(1)
        & working["G1"].notna()
        & working["G2"].notna()
    ].copy()
    working["Date"] = working["Date"].dt.strftime("%Y-%m-%d")
    for column in ("Round", "Group"):
        if column not in working.columns:
            working[column] = ""
    return working.sort_values("Date", kind="stable").reset_index(drop=True)


def exclude_worldcup_2026_labels(rows: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    if rows.empty:
        return rows.copy(), 0
    years = pd.to_numeric(rows.get("Year"), errors="coerce")
    is_worldcup_raw = rows.get("is_worldcup_match", pd.Series(False, index=rows.index, dtype=bool))
    is_worldcup = pd.Series(is_worldcup_raw, index=rows.index).map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y", "si"} if not isinstance(value, (bool, np.bool_)) else bool(value))
    tournament = rows.get("tournament", pd.Series("", index=rows.index)).astype(str).str.lower()
    mask = years.eq(2026) & (is_worldcup | tournament.str.contains("world cup|mundial", regex=True, na=False))
    return rows[~mask].copy().reset_index(drop=True), int(mask.sum())


def history_row_to_fixture(row: pd.Series) -> pd.Series:
    return pd.Series({
        "No.": row.get("FixtureId", ""),
        "Fecha": row.get("Date", ""),
        "Year": row.get("Year", ""),
        "Hora": "",
        "Grupo": row.get("Group", ""),
        "Equipo 1": row.get("Team 1", row.get("Home", "")),
        "Equipo 2": row.get("Team 2", row.get("Away", "")),
        "Sede": row.get("Venue", ""),
        "Round": row.get("Round", row.get("stage", "")),
    })


def label_payload(row: pd.Series) -> Dict[str, Any]:
    home_goals = safe_float(row.get("G1", row.get("HG", 0)))
    away_goals = safe_float(row.get("G2", row.get("AG", 0)))
    total_goals = home_goals + away_goals
    payload: Dict[str, Any] = {"result": label_from_goals(home_goals, away_goals)}
    for line in TRAIN_TOTAL_GOAL_LINES:
        payload[f"over_under_{total_line_suffix(line)}"] = int(total_goals > float(line))
    return payload


def label_from_goals(home_goals: Any, away_goals: Any) -> str:
    home = safe_float(home_goals)
    away = safe_float(away_goals)
    if home > away:
        return "H"
    if away > home:
        return "A"
    return "D"


def align_frame_to_columns(x: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    working = x.copy()
    for column in feature_columns:
        if column not in working.columns:
            working[column] = 0.0
    return working[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)


def temporal_split_indices(size: int) -> Dict[str, List[int]]:
    if size <= 0:
        return {"train": [], "validation": [], "test": []}
    train_end = max(1, int(math.floor(size * 0.70)))
    validation_end = max(train_end + 1, int(math.floor(size * 0.85))) if size >= 3 else size
    if validation_end >= size:
        validation_end = max(train_end, size - 1)
    return {
        "train": list(range(0, train_end)),
        "validation": list(range(train_end, validation_end)),
        "test": list(range(validation_end, size)),
    }


def load_cached_market_rows() -> pd.DataFrame:
    try:
        payload = load_market_data(force_download=False, allow_download=False, use_scraper=False)
        return payload.get("matches", pd.DataFrame())
    except Exception:
        return pd.DataFrame()


def report_warnings(reports: List[Dict[str, Any]]) -> List[str]:
    output = []
    for report in reports:
        for warning in report.get("warnings", []):
            if warning:
                output.append(f"{report.get('model_key', '')}: {warning}")
    return unique_strings(output)


def probabilities_fraction(probabilities: Dict[str, Any]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for key, value in probabilities.items():
        number = safe_float(value)
        output[str(key)] = number / 100.0 if number > 1.0 else number
    return output


def align_probability_matrix(probs: np.ndarray, source_classes: List[str], target_classes: List[str]) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    output = np.zeros((probs.shape[0], len(target_classes)), dtype=float)
    for source_index, source_class in enumerate(source_classes):
        if source_class not in target_classes or source_index >= probs.shape[1]:
            continue
        output[:, target_classes.index(source_class)] = probs[:, source_index]
    missing = output.sum(axis=1) <= 0.0
    if missing.any():
        output[missing, :] = 1.0 / max(len(target_classes), 1)
    return _normalize_rows(output)


def align_probabilities_to_classes(probabilities: np.ndarray, classes: List[Any]) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float).ravel()
    if probs.size < len(classes):
        probs = np.pad(probs, (0, len(classes) - probs.size), constant_values=0.0)
    return _normalize_rows(probs[:len(classes)].reshape(1, -1))[0]


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.maximum(array, 0.0)
    totals = array.sum(axis=1, keepdims=True)
    missing = totals <= 0.0
    totals[missing] = 1.0
    output = array / totals
    if missing.any() and output.shape[1] > 0:
        output[missing[:, 0], :] = 1.0 / output.shape[1]
    return output


def normalize_grid(grid: Any) -> np.ndarray:
    array = np.asarray(grid, dtype=float)
    if array.ndim != 2 or array.size == 0:
        return poisson_score_grid(1.2, 1.0, max_goals=10)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.maximum(array, 0.0)
    total = float(array.sum())
    if total <= 0.0:
        return poisson_score_grid(1.2, 1.0, max_goals=max(array.shape[0] - 1, 4))
    return array / total


def entropy(values: Iterable[float]) -> float:
    probs = [max(float(value), 1e-12) for value in values]
    total = sum(probs)
    if total <= 0:
        return 0.0
    probs = [value / total for value in probs]
    return float(-sum(prob * math.log(max(prob, 1e-12)) for prob in probs))


def safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return float(number) if math.isfinite(number) else 0.0


def safe_int(value: Any) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return 0
    return number


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
