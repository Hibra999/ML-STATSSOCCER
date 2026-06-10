from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.cli.model_specs import MODEL_SPECS, normalize_model_key, tunable_param_names
from src.worldcup.data import CACHE_ROOT, clean_team_name, fallback_tournament_2026, load_historical_matches, load_tournament_2026, tournament_fixtures_dataframe
from src.worldcup.model import HOST_TEAMS, WorldCupModel


KAGGLE_DATASET_SLUG = "harrachimustapha/fifa-world-cup-team-dataset"
KAGGLE_ROOT = Path("storage") / "worldcup" / "kaggle"
WORLD_CUP_MODELS_ROOT = Path("storage") / "worldcup" / "models"
HYBRID_MODEL_FILE = WORLD_CUP_MODELS_ROOT / "hybrid_worldcup_model.pkl"
HYBRID_MODEL_META_FILE = WORLD_CUP_MODELS_ROOT / "hybrid_worldcup_model.json"
PREPARED_DATASET_FILE = CACHE_ROOT / "worldcup_training_prepared.pkl"
PREPARED_DATASET_META_FILE = CACHE_ROOT / "worldcup_training_prepared.json"
LEGACY_MODEL_ID = "legacy-hybrid"
ACTIVE_MODEL_STATE_FILE = "active_model.json"
BASE_FEATURE_COLUMNS = [
    "rating_home",
    "rating_away",
    "rating_diff",
    "attack_home",
    "attack_away",
    "attack_diff",
    "defense_home",
    "defense_away",
    "defense_diff",
    "matches_home",
    "matches_away",
    "home_is_host",
    "away_is_host",
]
MATCH_ROW_COLUMNS = ["FixtureId", "Date", "Year", "Home", "Away", "Label", "HG", "AG", "OverUnder25", "Source"]
TARGET_LABELS = ["H", "D", "A"]
TEAM_TARGET_COLUMNS = ["quarter_finalist", "semi_finalist", "finalist", "winner"]
MODEL_PARAM_KEYS = [
    "n_estimators",
    "max_depth",
    "min_child_weight",
    "learning_rate",
    "lambda_regularization",
    "alpha_regularization",
    "num_leaves",
    "min_child_samples",
    "minibatch_frac",
    "natural_gradient",
    "l2_leaf_reg",
    "random_strength",
]
WORLD_CUP_MODEL_LABELS = {
    "ngboost": "NGBoost",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}
HISTORY_FEATURE_WINDOWS = (1, 2, 3, 5, 7, 10, 12)
HISTORY_REFERENCE_DATE = "2026-06-11"
WALK_FORWARD_ROOT = Path("storage") / "worldcup" / "walk_forward"
WALK_FORWARD_MATCHES_FILE = WALK_FORWARD_ROOT / "matches.csv"
WALK_FORWARD_PLAYERS_FILE = WALK_FORWARD_ROOT / "player_match_stats.csv"
WALK_FORWARD_TEAM_FEATURES_FILE = WALK_FORWARD_ROOT / "team_match_features.csv"


class WorldCupTrainingError(RuntimeError):
    pass


def emit_training_progress(callback, stage: str, current: int, total: int, message: str, **extra) -> None:
    if callback is None:
        return
    total = max(int(total or 1), 1)
    current = min(max(int(current or 0), 0), total)
    callback({
        "stage": stage,
        "current": current,
        "total": total,
        "current_trial": current if stage == "tuning" else "",
        "total_trials": total if stage == "tuning" else "",
        "percent": int(round(current * 100 / total)),
        "message": message,
        **extra,
    })


def market_label_for_progress(target: str) -> str:
    return "O/U 2.5" if target == "over_under_25" else "1X2"


def training_options() -> Dict[str, Any]:
    models = []
    for key, spec in MODEL_SPECS.items():
        tunables = {}
        for param in tunable_param_names(spec):
            if param in {"normalizer", "sampler", "calibrate_probabilities"}:
                continue
            try:
                tunables[param] = spec.model_cls.get_suggest_param_values(param=param)
            except ValueError:
                continue
        models.append({
            "key": key,
            "label": WORLD_CUP_MODEL_LABELS.get(key, spec.label),
            "defaults": spec.defaults,
            "tunables": tunables,
            "supports_cuda": key in {"xgboost", "catboost", "lightgbm"},
        })
    return {
        "models": json_safe(models),
        "targets": [
            {"key": "dual_markets", "label": "Ambos: 1X2 + O/U 2.5"},
        ],
        "hardware": detect_hardware(),
        "defaults": default_training_payload(),
    }


def detect_hardware() -> Dict[str, Any]:
    cpu_count = int(os.cpu_count() or 1)
    cuda_devices: List[str] = []
    cuda_error = ""
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode == 0:
                cuda_devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            elif result.stderr:
                cuda_error = result.stderr.strip().splitlines()[0]
        except Exception as exc:
            cuda_error = f"{exc.__class__.__name__}: {exc}"
    else:
        cuda_error = "nvidia-smi no disponible"
    return {
        "cpu_count": cpu_count,
        "default_n_jobs": -1,
        "cuda_available": bool(cuda_devices),
        "cuda_devices": cuda_devices,
        "cuda_error": cuda_error,
        "device_default": "cuda" if cuda_devices else "cpu",
    }


def default_training_payload() -> Dict[str, Any]:
    return {
        "model_type": "xgboost",
        "training_target": "result",
        "market_mode": "dual_markets",
        "device": "auto",
        "n_jobs": -1,
        "tuning_enabled": False,
        "n_trials": 12,
        "optuna_sampler": "tpe",
        "optuna_pruner": "none",
        "objective": "F1",
        "tune_params": "all",
        **MODEL_SPECS["xgboost"].defaults,
    }


def download_kaggle_dataset(force: bool = False) -> Dict[str, Any]:
    if KAGGLE_ROOT.exists() and list(discover_dataset_files(KAGGLE_ROOT)) and not force:
        return dataset_status()
    try:
        import kagglehub
    except ImportError as exc:
        raise WorldCupTrainingError("kagglehub no esta instalado. Ejecuta pip install -r requirements.txt.") from exc

    source_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    if not source_path.exists():
        raise WorldCupTrainingError(f"Kaggle no devolvio una ruta valida para {KAGGLE_DATASET_SLUG}.")
    KAGGLE_ROOT.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in source_path.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        target = KAGGLE_ROOT / path.relative_to(source_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if force or not target.exists():
            shutil.copy2(path, target)
        copied.append(str(target))
    status = dataset_status()
    status["downloaded_path"] = str(source_path)
    status["copied_files"] = copied
    return status


def dataset_status() -> Dict[str, Any]:
    files = list(discover_dataset_files(KAGGLE_ROOT))
    normalized = normalize_dataset_files(files)
    prepared = prepared_dataset_status(files=files, normalized=normalized)
    model_meta = read_model_metadata()
    active_dataset = prepared["dataset"] if prepared["ready"] else normalized
    train_rows = labeled_train_row_count(active_dataset)
    test_rows = labeled_test_row_count(active_dataset)
    eval_strategy = evaluation_strategy(active_dataset)
    walk_forward = walk_forward_status()
    refresh_state = walk_forward_refresh_state()
    return {
        "dataset_slug": KAGGLE_DATASET_SLUG,
        "local_path": str(KAGGLE_ROOT),
        "files": [str(path) for path in files],
        "available": bool(files),
        "etl_ready": bool(prepared["ready"]),
        "etl_stale": bool(prepared["stale"]),
        "etl_status": prepared["status"],
        "etl_artifact_path": str(PREPARED_DATASET_FILE),
        "prepared_at": prepared.get("prepared_at", ""),
        "prepared_mode": prepared.get("mode", ""),
        "prepared_label_source": prepared.get("label_source", ""),
        "final_test_year": prepared.get("final_test_year", ""),
        "split_policy": prepared.get("split_policy", ""),
        "prepared_over_under_ready": bool(prepared.get("over_under_ready", False)),
        "prepared_warnings": prepared.get("warnings", []),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "eval_rows": test_rows if test_rows else planned_holdout_rows(train_rows),
        "prediction_rows": int(active_dataset["team_prediction"].shape[0]),
        "eval_strategy": eval_strategy,
        "team_feature_rows": int(active_dataset["team_features"].shape[0]),
        "target_column": active_dataset["target_column"],
        "team_columns": active_dataset["team_columns"],
        "training_mode": active_dataset["training_mode"],
        "raw_training_mode": normalized["training_mode"],
        "etl_steps": etl_steps(files, active_dataset, eval_strategy, prepared=prepared),
        "trainable": bool(active_dataset["trainable"]),
        "walk_forward": walk_forward,
        "walk_forward_refresh": refresh_state,
        "model": model_meta,
        "preview": active_dataset["preview"],
    }


def prepare_training_dataset(force: bool = False, refresh_history: bool = False) -> Dict[str, Any]:
    files = list(discover_dataset_files(KAGGLE_ROOT))
    if not files:
        raise WorldCupTrainingError("No hay dataset Kaggle local. Primero descarga el dataset.")
    normalized = normalize_dataset_files(files)
    if PREPARED_DATASET_FILE.exists() and not force:
        current = prepared_dataset_status(files=files, normalized=normalized)
        if current["ready"] and not current["stale"]:
            return dataset_status()
    prepared = build_prepared_dataset(files=files, normalized=normalized, refresh_history=refresh_history)
    save_prepared_dataset(prepared)
    return dataset_status()


def train_hybrid_model(tournament: Dict[str, Any], payload: Optional[Dict[str, Any]] = None, progress_callback=None) -> Dict[str, Any]:
    payload = payload or {}
    train_config = training_config(payload)
    if train_config["market_mode"] != "dual_markets":
        raise WorldCupTrainingError("Mundial 2026 entrena siempre el bundle dual 1X2 + O/U 2.5.")
    emit_training_progress(progress_callback, "preparing", 0, 6, "Preparando entrenamiento Mundial")
    return train_dual_market_model(
        tournament=tournament,
        payload=payload,
        train_config=train_config,
        progress_callback=progress_callback,
    )


def train_single_hybrid_model(
        tournament: Dict[str, Any],
        payload: Optional[Dict[str, Any]] = None,
        progress_callback=None,
        market_label: str = "",
) -> Dict[str, Any]:
    payload = payload or {}
    train_config = training_config(payload)
    model_id = train_config["model_id"]
    label = market_label or market_label_for_progress(train_config["training_target"])
    emit_training_progress(progress_callback, "preparing", 1, 5, f"Preparando datos {label}", market=label, model_id=model_id)
    files = list(discover_dataset_files(KAGGLE_ROOT))
    normalized = load_prepared_dataset(required=True)
    train_rows = normalized["train"].copy()
    test_rows = normalized["test"].copy()
    if not normalized["trainable"]:
        raise WorldCupTrainingError("El ETL no genero columnas trainables de partido con goles y resultado reales.")
    walk_forward_mode = normalize_walk_forward_mode(payload.get("walk_forward_mode", "none"))
    walk_forward_summary = supplemental_training_rows(
        tournament=tournament,
        mode=walk_forward_mode,
        dataset_mode=normalized["training_mode"],
    )
    supplemental_rows = walk_forward_summary["rows"]
    if not supplemental_rows.empty:
        train_rows = pd.concat([train_rows, supplemental_rows], ignore_index=True)

    group_teams = teams_from_tournament(tournament)
    model_teams = sorted(set(group_teams) | set(teams_from_rows(train_rows)) | set(teams_from_rows(test_rows)))
    history_df, history_source = load_historical_matches(refresh=bool(payload.get("refresh_history", False)))
    feature_store = normalized["team_features"]
    fixture_feature_rows = read_fixture_feature_rows() if walk_forward_mode == "result_plus_players" else pd.DataFrame()
    target_warning = ""
    eval_strategy = "unavailable"
    effective_target = train_config["training_target"]
    if effective_target == "over_under_25" and not has_over_under_target(train_rows):
        raise WorldCupTrainingError("El ETL preparado no contiene goles suficientes para entrenar O/U 2.5.")
    eval_size = float(payload.get("eval_size", 0.25) or 0.25)
    train_rows = sort_match_rows(train_rows)
    if test_rows.empty:
        eval_strategy = "holdout_temporal"
        split_train_rows, split_eval_rows = safe_temporal_row_split(
            train_rows,
            test_size=eval_size,
        )
        x_train, y_train, feature_columns = build_training_matrix(
            split_train_rows,
            history_df=history_df,
            teams=model_teams,
            team_features=feature_store,
            fixture_feature_rows=fixture_feature_rows,
            history_weight=float(payload.get("history_weight", 1.0) or 1.0),
            recency_weight=float(payload.get("recency_weight", 0.35) or 0.35),
            host_advantage=float(payload.get("host_advantage", 45.0) or 45.0),
            max_goals=int(payload.get("max_goals", 10) or 10),
            target=effective_target,
        )
        x_eval, y_eval, _ = build_training_matrix(
            split_eval_rows,
            history_df=history_df,
            teams=model_teams,
            team_features=feature_store,
            fixture_feature_rows=fixture_feature_rows,
            feature_columns=feature_columns,
            frozen_years=years_from_rows(split_eval_rows),
            history_weight=float(payload.get("history_weight", 1.0) or 1.0),
            recency_weight=float(payload.get("recency_weight", 0.35) or 0.35),
            host_advantage=float(payload.get("host_advantage", 45.0) or 45.0),
            max_goals=int(payload.get("max_goals", 10) or 10),
            target=effective_target,
        )
    else:
        eval_strategy = "final_worldcup_test"
        test_rows = sort_match_rows(test_rows)
        x_train, y_train, feature_columns = build_training_matrix(
            train_rows,
            history_df=history_df,
            teams=model_teams,
            team_features=feature_store,
            fixture_feature_rows=fixture_feature_rows,
            history_weight=float(payload.get("history_weight", 1.0) or 1.0),
            recency_weight=float(payload.get("recency_weight", 0.35) or 0.35),
            host_advantage=float(payload.get("host_advantage", 45.0) or 45.0),
            max_goals=int(payload.get("max_goals", 10) or 10),
            target=effective_target,
        )
        x_eval, y_eval, _ = build_training_matrix(
            test_rows,
            history_df=history_df,
            teams=model_teams,
            team_features=feature_store,
            fixture_feature_rows=fixture_feature_rows,
            feature_columns=feature_columns,
            frozen_years=years_from_rows(test_rows),
            history_weight=float(payload.get("history_weight", 1.0) or 1.0),
            recency_weight=float(payload.get("recency_weight", 0.35) or 0.35),
            host_advantage=float(payload.get("host_advantage", 45.0) or 45.0),
            max_goals=int(payload.get("max_goals", 10) or 10),
            target=effective_target,
        )

    if x_train.empty or pd.Series(y_train).dropna().empty:
        raise WorldCupTrainingError("No hay filas entrenables para el objetivo seleccionado.")
    y_train_encoded, label_classes = encode_labels(y_train)
    y_eval_encoded = encode_existing_labels(y_eval, label_classes)
    tuned = tune_model_if_requested(
        train_config,
        x_train,
        y_train_encoded,
        progress_callback=progress_callback,
        market_label=label,
    )
    if tuned.get("best_params"):
        train_config["params"].update(tuned["best_params"])
    emit_training_progress(progress_callback, "fit", 4, 5, f"Entrenando clasificador {label}", market=label, model_id=model_id)
    fit_result = fit_configured_classifier(
        x_train=x_train,
        y_train=y_train_encoded,
        model_key=train_config["model_type"],
        params=train_config["params"],
        n_jobs=train_config["n_jobs"],
        requested_device=train_config["device"],
        seed=train_config["seed"],
        num_classes=len(label_classes),
    )
    clf = fit_result["classifier"]
    y_train_pred = classifier_predict(clf, x_train)
    y_eval_pred = classifier_predict(clf, x_eval)
    emit_training_progress(progress_callback, "metrics", 5, 5, f"Calculando métricas {label}", market=label, model_id=model_id)
    metrics = classification_metrics_from_predictions(y_train_encoded, y_train_pred, y_eval_encoded, y_eval_pred)
    confusion = confusion_matrix_payload(y_eval_encoded, y_eval_pred, label_classes, target=effective_target)
    etl = etl_steps(files, normalized, eval_strategy, prepared=prepared_dataset_status(files=files, normalized=normalized))
    hardware = detect_hardware()
    hardware.update({
        "requested_device": train_config["device"],
        "actual_device": fit_result["device"],
        "n_jobs": train_config["n_jobs"],
        "effective_n_jobs": effective_n_jobs(train_config["n_jobs"], hardware["cpu_count"]),
    })
    record = {
        "classifier": clf,
        "feature_columns": feature_columns,
        "team_features": feature_store.to_dict(orient="records"),
        "history_team_features": build_history_feature_table(history_df).to_dict(orient="records"),
        "matchup_features": build_matchup_feature_table(history_df).to_dict(orient="records"),
        "kaggle_files": [str(path) for path in files],
        "history_source": normalized.get("history_source", history_source),
        "metrics": metrics,
        "confusion_matrix": confusion,
        "classes": label_classes,
        "encoded_classes": list(range(len(label_classes))),
        "mode": normalized["training_mode"],
        "eval_strategy": eval_strategy,
        "prediction_rows": int(normalized["team_prediction"].shape[0]),
        "effective_target": effective_target,
        "requested_target": train_config["training_target"],
        "target_column": normalized["target_column"],
        "model_type": train_config["model_type"],
        "model_label": WORLD_CUP_MODEL_LABELS.get(train_config["model_type"], train_config["model_type"]),
        "model_id": model_id,
        "model_name": train_config["model_name"],
        "hidden_from_catalog": bool(payload.get("hidden_from_catalog", False)),
        "model_params": train_config["params"],
        "tuning": tuned,
        "tuning_trace": tuning_trace(tuned),
        "etl_steps": etl,
        "hardware": hardware,
        "warnings": unique_strings([warning for warning in [target_warning, *normalized.get("warnings", []), *fit_result.get("warnings", [])] if warning]),
        "top_features": top_feature_importances(clf, feature_columns),
        "walk_forward_mode": walk_forward_mode,
        "walk_forward_summary": walk_forward_summary,
        "final_test_year": normalized.get("final_test_year", ""),
        "split_policy": normalized.get("split_policy", ""),
    }
    if walk_forward_summary["warnings"]:
        record["warnings"] = unique_strings([*record["warnings"], *walk_forward_summary["warnings"]])
    save_hybrid_model(record, model_id=model_id)
    if walk_forward_summary["fixture_ids"]:
        mark_walk_forward_ingested(walk_forward_summary["fixture_ids"], walk_forward_mode)
    return {
        "model": read_model_metadata(),
        "metrics": metrics,
        "confusion_matrix": confusion,
        "features": feature_columns,
        "train_rows": int(len(y_train)),
        "eval_rows": int(len(y_eval)),
        "source": KAGGLE_DATASET_SLUG,
        "mode": normalized["training_mode"],
        "eval_strategy": eval_strategy,
        "prediction_rows": int(normalized["team_prediction"].shape[0]),
        "effective_target": effective_target,
        "requested_target": train_config["training_target"],
        "model_id": model_id,
        "model_type": train_config["model_type"],
        "hardware": hardware,
        "tuning": tuned,
        "tuning_trace": tuning_trace(tuned),
        "etl_steps": etl,
        "warnings": record["warnings"],
        "walk_forward": walk_forward_summary,
        "final_test_year": normalized.get("final_test_year", ""),
        "split_policy": normalized.get("split_policy", ""),
    }


def train_dual_market_model(
        tournament: Dict[str, Any],
        payload: Optional[Dict[str, Any]] = None,
        train_config: Optional[Dict[str, Any]] = None,
        progress_callback=None,
) -> Dict[str, Any]:
    payload = dict(payload or {})
    train_config = train_config or training_config(payload)
    bundle_id = train_config["model_id"]
    bundle_name = train_config["model_name"]
    files = list(discover_dataset_files(KAGGLE_ROOT))
    normalized = load_prepared_dataset(required=True)
    if not normalized["trainable"]:
        raise WorldCupTrainingError("El ETL preparado no dejo filas entrenables para el bundle dual.")

    result_child_id = child_market_model_id(bundle_id, "result")
    over_child_id = child_market_model_id(bundle_id, "over_under_25")
    common_payload = dict(payload)
    common_payload["market_mode"] = "result"

    result_payload = {
        **common_payload,
        "training_target": "result",
        "model_id": result_child_id,
        "model_name": f"{bundle_name} - 1X2",
        "hidden_from_catalog": True,
    }
    emit_training_progress(progress_callback, "market-result", 1, 6, "Entrenando mercado 1X2", market="1X2", model_id=bundle_id)
    result_result = train_single_hybrid_model(
        tournament=tournament,
        payload=result_payload,
        progress_callback=progress_callback,
        market_label="1X2",
    )
    result_record = load_hybrid_model(result_child_id) or {}

    warnings: List[str] = []
    market_results = {"result": market_training_summary(result_record, result_result, "1X2")}
    market_models = {"result": result_child_id}

    can_train_over_under = normalized["training_mode"] == "match_result" and has_over_under_target(normalized["train"])
    over_result: Optional[Dict[str, Any]] = None
    over_record: Dict[str, Any] = {}
    if can_train_over_under:
        over_payload = {
            **common_payload,
            "training_target": "over_under_25",
            "model_id": over_child_id,
            "model_name": f"{bundle_name} - O/U 2.5",
            "hidden_from_catalog": True,
        }
        try:
            emit_training_progress(progress_callback, "market-over-under", 3, 6, "Entrenando mercado O/U 2.5", market="O/U 2.5", model_id=bundle_id)
            over_result = train_single_hybrid_model(
                tournament=tournament,
                payload=over_payload,
                progress_callback=progress_callback,
                market_label="O/U 2.5",
            )
            over_record = load_hybrid_model(over_child_id) or {}
            if over_record.get("effective_target") == "over_under_25":
                market_results["over_under_25"] = market_training_summary(over_record, over_result, "O/U 2.5")
                market_models["over_under_25"] = over_child_id
            else:
                warnings.append("No se guardo modelo O/U porque el dataset no produjo target Over/Under 2.5 valido.")
                delete_model_files(over_child_id)
        except Exception as exc:
            delete_model_files(over_child_id)
            raise WorldCupTrainingError(f"O/U 2.5 no se pudo entrenar con goles reales ({exc.__class__.__name__}: {exc}).")
    else:
        raise WorldCupTrainingError("O/U 2.5 no se puede entrenar: el ETL preparado no contiene goles reales suficientes.")

    warnings.extend(result_record.get("warnings", []))
    warnings.extend(over_record.get("warnings", []))
    warnings = unique_strings(warnings)
    trained_at = datetime.now(timezone.utc).isoformat()
    bundle_target_column = str(normalized.get("target_column") or result_record.get("target_column") or "result")
    bundle_record = {
        "bundle": True,
        "market_mode": "dual_markets",
        "classifier": None,
        "feature_columns": result_record.get("feature_columns", []),
        "team_features": result_record.get("team_features", []),
        "history_team_features": result_record.get("history_team_features", []),
        "matchup_features": result_record.get("matchup_features", []),
        "kaggle_files": [str(path) for path in files],
        "history_source": result_record.get("history_source", over_record.get("history_source", "")),
        "metrics": result_record.get("metrics", {}),
        "confusion_matrix": result_record.get("confusion_matrix", {}),
        "classes": result_record.get("classes", []),
        "mode": normalized["training_mode"],
        "eval_strategy": result_record.get("eval_strategy", ""),
        "prediction_rows": int(normalized["team_prediction"].shape[0]),
        "effective_target": "result+over_under_25",
        "requested_target": "dual_markets",
        "target_column": bundle_target_column,
        "model_type": train_config["model_type"],
        "model_label": WORLD_CUP_MODEL_LABELS.get(train_config["model_type"], train_config["model_type"]),
        "model_id": bundle_id,
        "model_name": bundle_name,
        "model_params": train_config["params"],
        "tuning": result_record.get("tuning", {}),
        "tuning_trace": bundle_tuning_trace(market_results),
        "etl_steps": bundle_etl_steps(result_record.get("etl_steps", []), market_models),
        "hardware": result_record.get("hardware", detect_hardware()),
        "warnings": warnings,
        "top_features": result_record.get("top_features", []),
        "markets": market_results,
        "market_models": market_models,
        "trained_at": trained_at,
        "walk_forward_mode": result_record.get("walk_forward_mode", "none"),
        "walk_forward_summary": result_record.get("walk_forward_summary", {}),
        "final_test_year": normalized.get("final_test_year", ""),
        "split_policy": normalized.get("split_policy", ""),
    }
    emit_training_progress(progress_callback, "saving", 5, 6, "Guardando bundle dual", model_id=bundle_id)
    save_hybrid_model(bundle_record, model_id=bundle_id)
    model_meta = read_model_metadata(model_id=bundle_id)
    emit_training_progress(progress_callback, "complete", 6, 6, "Entrenamiento completado", model_id=bundle_id)
    return {
        "model": model_meta,
        "metrics": bundle_record["metrics"],
        "confusion_matrix": bundle_record["confusion_matrix"],
        "features": bundle_record["feature_columns"],
        "train_rows": int(max(result_result.get("train_rows", 0), (over_result or {}).get("train_rows", 0))),
        "eval_rows": int(max(result_result.get("eval_rows", 0), (over_result or {}).get("eval_rows", 0))),
        "source": KAGGLE_DATASET_SLUG,
        "mode": normalized["training_mode"],
        "eval_strategy": bundle_record["eval_strategy"],
        "prediction_rows": int(normalized["team_prediction"].shape[0]),
        "effective_target": bundle_record["effective_target"],
        "requested_target": "dual_markets",
        "model_id": bundle_id,
        "model_type": train_config["model_type"],
        "hardware": bundle_record["hardware"],
        "tuning": bundle_record["tuning"],
        "tuning_trace": bundle_record["tuning_trace"],
        "etl_steps": bundle_record["etl_steps"],
        "warnings": warnings,
        "markets": market_results,
        "market_models": market_models,
        "walk_forward": bundle_record["walk_forward_summary"],
        "final_test_year": bundle_record["final_test_year"],
        "split_policy": bundle_record["split_policy"],
    }


def predict_match_payload(
        tournament: Dict[str, Any],
        base_model: WorldCupModel,
        fixture_id: Optional[Any] = None,
        home: Optional[str] = None,
        away: Optional[str] = None,
        use_ml_model: bool = True,
        ml_weight: float = 0.5,
        model_id: Optional[str] = None,
) -> Dict[str, Any]:
    fixture = select_prediction_fixture(tournament, fixture_id=fixture_id, home=home, away=away)
    home_team = str(fixture.get("Equipo 1", home or ""))
    away_team = str(fixture.get("Equipo 2", away or ""))
    poisson = base_model.match_probabilities(home_team, away_team)
    base_probs = {"H": poisson["home"], "D": poisson["draw"], "A": poisson["away"]}
    base_totals = {"over25": poisson["over25"], "under25": poisson["under25"]}
    ml_outputs = {"result": {}, "over_under_25": {}, "notes": ["Modelo Kaggle no entrenado."]}
    if use_ml_model:
        ml_outputs = predict_ml_outputs(base_model, home_team, away_team, model_id=model_id, fixture_id=fixture.get("No."))
    result_ml = ml_outputs.get("result", {})
    over_under_ml = ml_outputs.get("over_under_25", {})
    result_weight = ml_weight if result_ml else 0.0
    totals_weight = ml_weight if over_under_ml else 0.0
    blended = blend_probabilities(base_probs, result_ml, result_weight)
    blended_totals = blend_total_probabilities(base_totals, over_under_ml, totals_weight)
    market_sources = market_sources_payload(result_ml, over_under_ml, ml_outputs)
    return {
        "fixture": {
            "id": str(fixture.get("No.", "")),
            "date": fixture.get("Fecha", ""),
            "time": fixture.get("Hora", ""),
            "group": fixture.get("Grupo", ""),
            "home": home_team,
            "away": away_team,
            "venue": fixture.get("Sede", ""),
        },
        "probabilities": {
            "home": round(blended["H"] * 100.0, 2),
            "draw": round(blended["D"] * 100.0, 2),
            "away": round(blended["A"] * 100.0, 2),
            "over25": round(blended_totals["over25"] * 100.0, 2),
            "under25": round(blended_totals["under25"] * 100.0, 2),
        },
        "model_probs": {
            "poisson": {key: round(value * 100.0, 2) for key, value in base_probs.items()},
            "poisson_totals": {key: round(value * 100.0, 2) for key, value in base_totals.items()},
            "ml": {key: round(value * 100.0, 2) for key, value in result_ml.items()},
            "over_under_ml": {key: round(value * 100.0, 2) for key, value in over_under_ml.items()},
            "ml_weight": round(float(ml_weight if result_ml or over_under_ml else 0.0), 3),
            "result_weight": round(float(result_weight), 3),
            "over_under_weight": round(float(totals_weight), 3),
            "model_id": ml_outputs.get("model_id", ""),
            "model_name": ml_outputs.get("model_name", ""),
            "result_source": market_sources["result"]["source"],
            "over_under_source": market_sources["over_under_25"]["source"],
        },
        "market_sources": market_sources,
        "expected_goals": {
            "home": round(poisson["lambda1"], 3),
            "away": round(poisson["lambda2"], 3),
        },
        "modal_score": f"{poisson['modal_g1']}-{poisson['modal_g2']}",
        "prediction": label_display(max(blended, key=blended.get), home_team, away_team),
        "notes": ml_outputs.get("notes", []),
    }


def normalize_dataset_files(files: Iterable[Path]) -> Dict[str, Any]:
    train_frames = []
    test_frames = []
    all_frames = []
    team_feature_frames = []
    team_train_frames = []
    team_test_frames = []
    team_prediction_frames = []
    target_column = ""
    team_columns: List[str] = []
    for path in files:
        raw = read_table(path)
        if raw.empty:
            continue
        is_test_like = is_test_or_eval_file(path)
        standardized = standardize_match_rows(raw, source=str(path))
        team_features = extract_team_features(raw, source=str(path))
        if not team_features.empty:
            team_feature_frames.append(team_features)
        team_rows = standardize_team_target_rows(raw, source=str(path))
        if not team_rows.empty:
            if is_test_like:
                team_test_frames.append(team_rows)
            else:
                team_train_frames.append(team_rows)
        elif is_test_like and not team_features.empty:
            team_prediction_frames.append(team_features)
        if not standardized.empty:
            all_frames.append(standardized)
            if "train" in path.name.lower():
                train_frames.append(standardized)
            elif is_test_like:
                test_frames.append(standardized)
            else:
                train_frames.append(standardized)
        if standardized.attrs.get("target_column"):
            target_column = standardized.attrs["target_column"]
        if standardized.attrs.get("team_columns"):
            team_columns = standardized.attrs["team_columns"]
        if team_rows.attrs.get("target_column") and not target_column:
            target_column = team_rows.attrs["target_column"]

    train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    test_df = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    team_train_df = pd.concat(team_train_frames, ignore_index=True) if team_train_frames else pd.DataFrame()
    team_test_df = pd.concat(team_test_frames, ignore_index=True) if team_test_frames else pd.DataFrame()
    team_prediction_df = pd.concat(team_prediction_frames, ignore_index=True) if team_prediction_frames else pd.DataFrame()
    team_features_df = merge_team_features(team_feature_frames)
    training_mode = "match_result" if not train_df.empty else "team_strength" if not team_train_df.empty else ""
    preview_source = train_df if not train_df.empty else team_train_df if not team_train_df.empty else pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    preview = preview_payload(preview_source)
    return {
        "train": train_df,
        "test": test_df,
        "team_train": team_train_df,
        "team_test": team_test_df,
        "team_prediction": team_prediction_df,
        "team_features": team_features_df,
        "target_column": target_column,
        "team_columns": team_columns,
        "training_mode": training_mode,
        "trainable": (
            not train_df.empty and "Label" in train_df.columns and train_df["Label"].isin(TARGET_LABELS).any()
        ) or (
            not team_train_df.empty and "Label" in team_train_df.columns and team_train_df["Label"].notna().any()
        ),
        "preview": preview,
    }


def build_prepared_dataset(
        files: List[Path],
        normalized: Dict[str, Any],
        refresh_history: bool = False,
) -> Dict[str, Any]:
    history_df, history_source = load_historical_matches(refresh=bool(refresh_history))
    history_rows = history_match_rows(history_df, source=history_source)
    warnings: List[str] = []
    label_source = "kaggle_match_result"
    raw_train = normalized["train"].copy()
    raw_test = normalized["test"].copy()
    raw_mode = normalized["training_mode"]
    team_features = normalized["team_features"].copy()

    if raw_train.empty:
        if history_rows.empty:
            raise WorldCupTrainingError("El ETL no encontro partidos con goles reales para construir el dataset de entrenamiento.")
        labeled_rows = history_rows
        label_source = "historical_worldcup"
        warnings.append("El Kaggle actual no trae filas de partido entrenables; el ETL usa resultados historicos abiertos del Mundial para 1X2 y O/U 2.5.")
    elif has_over_under_target(raw_train):
        labeled_parts = [raw_train]
        if has_over_under_target(raw_test):
            labeled_parts.append(raw_test)
        labeled_rows = pd.concat(labeled_parts, ignore_index=True)
        label_source = "kaggle_match_result"
        if raw_test.empty or not has_over_under_target(raw_test):
            warnings.append("El test Kaggle no trae goles suficientes para O/U 2.5; el ETL separara el ultimo Mundial etiquetado como test final si hay fechas.")
    else:
        if history_rows.empty:
            raise WorldCupTrainingError("El Kaggle actual no trae goles suficientes para O/U 2.5 y no se encontraron partidos historicos con goles reales.")
        labeled_parts = [raw_train, history_rows]
        if has_over_under_target(raw_test):
            labeled_parts.append(raw_test)
        labeled_rows = pd.concat(labeled_parts, ignore_index=True)
        label_source = "kaggle_match_result + historical_worldcup"
        warnings.append("El Kaggle actual no alcanza para O/U 2.5; el ETL complemento las etiquetas de partido con el historico abierto del Mundial.")
        if raw_test.empty or not has_over_under_target(raw_test):
            warnings.append("El test Kaggle no trae goles suficientes; el ultimo Mundial etiquetado se usara como test final si hay fechas.")

    labeled_rows = sanitize_match_rows(labeled_rows)
    train_df, test_df, final_test_year, split_warning = split_latest_worldcup_test(labeled_rows)
    if split_warning:
        warnings.append(split_warning)
    if final_test_year:
        warnings.append(f"Test final bloqueado al Mundial {final_test_year}; entrenamiento/validacion usan solo años anteriores.")
    over_under_ready = has_over_under_target(train_df)
    if not over_under_ready:
        raise WorldCupTrainingError("El ETL no pudo construir un target real de O/U 2.5 con goles observados.")

    prepared_at = datetime.now(timezone.utc).isoformat()
    preview_source = train_df if not train_df.empty else test_df if not test_df.empty else team_features
    return {
        "prepared_at": prepared_at,
        "source_files": [str(path) for path in files],
        "source_mode": raw_mode,
        "training_mode": "match_result",
        "train": train_df,
        "test": test_df,
        "team_train": pd.DataFrame(),
        "team_test": pd.DataFrame(),
        "team_prediction": normalized["team_prediction"].copy(),
        "team_features": team_features,
        "target_column": "Label + OverUnder25",
        "team_columns": normalized["team_columns"],
        "trainable": bool(not train_df.empty and train_df["Label"].isin(TARGET_LABELS).any()),
        "preview": preview_payload(preview_source),
        "warnings": unique_strings(warnings),
        "label_source": label_source,
        "history_source": history_source,
        "final_test_year": final_test_year,
        "split_policy": "latest_worldcup_final_test" if final_test_year else "temporal_holdout_from_train",
        "over_under_ready": over_under_ready,
        "result_ready": bool(not train_df.empty and train_df["Label"].isin(TARGET_LABELS).any()),
    }


def split_latest_worldcup_test(rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    if rows.empty:
        return rows.copy(), pd.DataFrame(columns=MATCH_ROW_COLUMNS), "", ""
    working = sort_match_rows(rows)
    years = pd.to_numeric(working.get("Year"), errors="coerce")
    valid_years = sorted({int(year) for year in years.dropna().tolist()})
    if len(valid_years) < 2:
        return working.reset_index(drop=True), pd.DataFrame(columns=working.columns), "", "No hay al menos dos Mundiales fechados; se usara holdout temporal interno desde train."
    final_year = int(valid_years[-1])
    train = working[years < final_year].copy()
    test = working[years == final_year].copy()
    if train.empty or test.empty:
        return working.reset_index(drop=True), pd.DataFrame(columns=working.columns), "", "No se pudo aislar el ultimo Mundial como test final; se usara holdout temporal interno desde train."
    return train.reset_index(drop=True), test.reset_index(drop=True), str(final_year), ""


def sanitize_match_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=MATCH_ROW_COLUMNS)
    working = rows.copy()
    required = ["Home", "Away", "Label", "Source"]
    for column in required:
        if column not in working.columns:
            working[column] = ""
    if "FixtureId" not in working.columns:
        working["FixtureId"] = ""
    if "HG" not in working.columns:
        working["HG"] = np.nan
    if "AG" not in working.columns:
        working["AG"] = np.nan
    if "OverUnder25" not in working.columns:
        working["OverUnder25"] = np.nan
    if "Date" not in working.columns:
        working["Date"] = pd.NaT
    if "Year" not in working.columns:
        working["Year"] = np.nan
    working["Home"] = working["Home"].map(clean_team_name)
    working["Away"] = working["Away"].map(clean_team_name)
    working["Label"] = working["Label"].astype(str)
    working["HG"] = pd.to_numeric(working["HG"], errors="coerce")
    working["AG"] = pd.to_numeric(working["AG"], errors="coerce")
    needs_over = working["OverUnder25"].isna() & working["HG"].notna() & working["AG"].notna()
    if needs_over.any():
        working.loc[needs_over, "OverUnder25"] = ((working.loc[needs_over, "HG"] + working.loc[needs_over, "AG"]) >= 3.0).astype(int)
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    inferred_year = working["Date"].dt.year
    working["Year"] = pd.to_numeric(working["Year"], errors="coerce").fillna(inferred_year)
    working["FixtureId"] = working["FixtureId"].astype(str)
    working = working[
        working["Home"].astype(str).str.len().gt(1) &
        working["Away"].astype(str).str.len().gt(1) &
        working["Label"].isin(TARGET_LABELS)
    ].copy()
    for column in MATCH_ROW_COLUMNS:
        if column not in working.columns:
            working[column] = np.nan
    return sort_match_rows(working[MATCH_ROW_COLUMNS + [column for column in working.columns if column not in MATCH_ROW_COLUMNS]]).reset_index(drop=True)


def sort_match_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    working = rows.copy()
    working["_date_sort"] = pd.to_datetime(working.get("Date"), errors="coerce")
    working["_year_sort"] = pd.to_numeric(working.get("Year"), errors="coerce")
    working["_row_sort"] = np.arange(len(working))
    return working.sort_values(["_year_sort", "_date_sort", "_row_sort"], kind="stable", na_position="last").drop(columns=["_date_sort", "_year_sort", "_row_sort"])


def history_match_rows(history_df: pd.DataFrame, source: str) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(columns=MATCH_ROW_COLUMNS)
    working = history_df.copy()
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    rows: List[Dict[str, Any]] = []
    for index, row in working.iterrows():
        home = clean_team_name(row.get("Team 1"))
        away = clean_team_name(row.get("Team 2"))
        if not home or not away:
            continue
        try:
            goals_home = float(row.get("G1", np.nan))
            goals_away = float(row.get("G2", np.nan))
        except (TypeError, ValueError):
            continue
        rows.append({
            "FixtureId": str(row.get("FixtureId", row.get("No.", index))),
            "Date": row.get("Date"),
            "Year": row.get("Year", pd.Timestamp(row.get("Date")).year if pd.notna(row.get("Date")) else np.nan),
            "Home": home,
            "Away": away,
            "Label": label_from_goals(goals_home, goals_away),
            "HG": goals_home,
            "AG": goals_away,
            "OverUnder25": int((goals_home + goals_away) >= 3.0),
            "Source": source,
            "Date": row.get("Date"),
        })
    return sanitize_match_rows(pd.DataFrame(rows))


def save_prepared_dataset(dataset: Dict[str, Any]) -> None:
    PREPARED_DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PREPARED_DATASET_FILE.open("wb") as handle:
        pickle.dump(dataset, handle)
    PREPARED_DATASET_META_FILE.write_text(
        json.dumps(json_safe(prepared_dataset_metadata(dataset)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_prepared_dataset(required: bool = False) -> Optional[Dict[str, Any]]:
    if not PREPARED_DATASET_FILE.exists():
        if required:
            raise WorldCupTrainingError("Primero ejecuta Preparar ETL para construir el dataset de entrenamiento Mundial.")
        return None
    with PREPARED_DATASET_FILE.open("rb") as handle:
        dataset = pickle.load(handle)
    if not isinstance(dataset, dict):
        if required:
            raise WorldCupTrainingError("El artifact ETL Mundial esta dañado. Vuelve a ejecutar Preparar ETL.")
        return None
    return dataset


def prepared_dataset_status(files: List[Path], normalized: Dict[str, Any]) -> Dict[str, Any]:
    dataset = load_prepared_dataset(required=False)
    if not dataset:
        return {
            "ready": False,
            "stale": False,
            "status": "pending",
            "dataset": normalized,
            "prepared_at": "",
            "mode": "",
            "label_source": "",
            "final_test_year": "",
            "split_policy": "",
            "over_under_ready": False,
            "warnings": [],
        }
    source_files = {str(path) for path in files}
    artifact_sources = set(dataset.get("source_files", []))
    artifact_time = PREPARED_DATASET_FILE.stat().st_mtime if PREPARED_DATASET_FILE.exists() else 0.0
    source_times = [path.stat().st_mtime for path in files if path.exists()]
    stale = bool(source_times and max(source_times) > artifact_time) or bool(source_files != artifact_sources)
    return {
        "ready": True,
        "stale": stale,
        "status": "stale" if stale else "ready",
        "dataset": dataset,
        "prepared_at": str(dataset.get("prepared_at") or ""),
        "mode": str(dataset.get("training_mode") or ""),
        "label_source": str(dataset.get("label_source") or ""),
        "final_test_year": str(dataset.get("final_test_year") or ""),
        "split_policy": str(dataset.get("split_policy") or ""),
        "over_under_ready": bool(dataset.get("over_under_ready", False)),
        "warnings": dataset.get("warnings", []),
    }


def prepared_dataset_metadata(dataset: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prepared_at": dataset.get("prepared_at", ""),
        "training_mode": dataset.get("training_mode", ""),
        "source_mode": dataset.get("source_mode", ""),
        "source_files": dataset.get("source_files", []),
        "label_source": dataset.get("label_source", ""),
        "warnings": dataset.get("warnings", []),
        "target_column": dataset.get("target_column", ""),
        "team_columns": dataset.get("team_columns", []),
        "train_rows": labeled_train_row_count(dataset),
        "test_rows": labeled_test_row_count(dataset),
        "prediction_rows": int(dataset.get("team_prediction", pd.DataFrame()).shape[0]),
        "team_feature_rows": int(dataset.get("team_features", pd.DataFrame()).shape[0]),
        "over_under_ready": bool(dataset.get("over_under_ready", False)),
        "result_ready": bool(dataset.get("result_ready", False)),
        "preview": dataset.get("preview", {"columns": [], "rows": [], "total": 0}),
        "history_source": dataset.get("history_source", ""),
        "final_test_year": dataset.get("final_test_year", ""),
        "split_policy": dataset.get("split_policy", ""),
    }


def standardize_match_rows(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    clean = df.copy()
    clean.columns = [normalize_column(column) for column in clean.columns]
    home_col = first_existing(clean.columns, ["home_team", "home", "team1", "team_1", "team_a", "country1", "country_1"])
    away_col = first_existing(clean.columns, ["away_team", "away", "team2", "team_2", "team_b", "country2", "country_2"])
    target_col = first_existing(clean.columns, ["result", "outcome", "target", "label", "winner", "winning_team", "match_result"])
    goals_home = first_existing(clean.columns, ["home_goals", "goals_home", "team1_goals", "g1", "score1"])
    goals_away = first_existing(clean.columns, ["away_goals", "goals_away", "team2_goals", "g2", "score2"])
    date_col = first_existing(clean.columns, ["date", "match_date", "fecha", "kickoff", "datetime", "match_datetime"])
    year_col = first_existing(clean.columns, ["year", "version", "worldcup_year", "tournament_year", "season"])
    fixture_col = first_existing(clean.columns, ["fixture_id", "fixture", "match_id", "id", "no", "no_"])
    if not home_col or not away_col:
        return pd.DataFrame()
    rows = []
    for index, row in clean.iterrows():
        home = clean_team_name(row.get(home_col))
        away = clean_team_name(row.get(away_col))
        if not home or not away:
            continue
        label = label_from_goals(row.get(goals_home), row.get(goals_away)) if goals_home and goals_away else ""
        if not label and target_col:
            label = label_from_target(row.get(target_col), home, away)
        if label not in TARGET_LABELS:
            continue
        record = {
            "FixtureId": str(row.get(fixture_col, index)) if fixture_col else str(index),
            "Date": row.get(date_col, pd.NaT) if date_col else pd.NaT,
            "Year": row.get(year_col, np.nan) if year_col else np.nan,
            "Home": home,
            "Away": away,
            "Label": label,
            "Source": source,
        }
        if goals_home and goals_away:
            try:
                record["HG"] = float(row.get(goals_home))
                record["AG"] = float(row.get(goals_away))
                record["OverUnder25"] = int((record["HG"] + record["AG"]) >= 2.5)
            except (TypeError, ValueError):
                pass
        rows.append(record)
    output = pd.DataFrame(rows)
    output.attrs["target_column"] = target_col or f"{goals_home}/{goals_away}"
    output.attrs["team_columns"] = [home_col, away_col]
    return sanitize_match_rows(output)


def extract_team_features(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    clean = df.copy()
    clean.columns = [normalize_column(column) for column in clean.columns]
    team_col = first_existing(clean.columns, ["team", "squad", "country", "nation", "team_name"])
    if not team_col:
        return pd.DataFrame()
    numeric_cols = [
        column for column in clean.columns
        if column != team_col and column not in TEAM_TARGET_COLUMNS and pd.api.types.is_numeric_dtype(clean[column])
    ][:24]
    if not numeric_cols:
        return pd.DataFrame()
    output = clean[[team_col] + numeric_cols].copy()
    output = output.rename(columns={team_col: "Team"})
    output["Team"] = output["Team"].map(clean_team_name)
    output = output[output["Team"].str.len() > 1]
    output["Source"] = source
    return output


def standardize_team_target_rows(df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    clean = df.copy()
    clean.columns = [normalize_column(column) for column in clean.columns]
    team_col = first_existing(clean.columns, ["team", "squad", "country", "nation", "team_name"])
    target_col = first_existing(clean.columns, TEAM_TARGET_COLUMNS)
    if not team_col or not target_col:
        return pd.DataFrame()
    numeric_cols = [
        column for column in clean.columns
        if column not in {team_col, "continent", "source"} and column not in TEAM_TARGET_COLUMNS and pd.api.types.is_numeric_dtype(clean[column])
    ]
    rows = []
    for _, row in clean.iterrows():
        team = clean_team_name(row.get(team_col))
        if not team:
            continue
        target = row.get(target_col)
        label = numeric_label(target)
        record = {"Team": team, "Label": label, "Source": source}
        for column in numeric_cols:
            record[column] = row.get(column)
        if label in {0, 1}:
            rows.append(record)
    output = pd.DataFrame(rows)
    output.attrs["target_column"] = target_col
    return output


def merge_team_features(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["Team"])
    df = pd.concat(frames, ignore_index=True)
    numeric_cols = [column for column in df.columns if column not in {"Team", "Source"} and pd.api.types.is_numeric_dtype(df[column])]
    if not numeric_cols:
        return pd.DataFrame(columns=["Team"])
    if "version" in numeric_cols:
        latest = df.sort_values(["Team", "version"], kind="stable").groupby("Team", as_index=False).tail(1)
        return latest[["Team"] + numeric_cols].reset_index(drop=True)
    return df.groupby("Team", as_index=False)[numeric_cols].mean(numeric_only=True)


def build_team_training_matrix(
        rows: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    x = rows.drop(columns=["Team", "Label", "Source"], errors="ignore").copy()
    x = x.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if feature_columns is None:
        feature_columns = list(x.columns)
    for column in feature_columns:
        if column not in x.columns:
            x[column] = 0.0
    return x[feature_columns].astype(float), rows["Label"].astype(int), feature_columns


def match_date_from_row(row: pd.Series) -> Optional[pd.Timestamp]:
    value = row.get("Date", pd.NaT)
    timestamp = pd.to_datetime(value, errors="coerce")
    return pd.Timestamp(timestamp) if pd.notna(timestamp) else None


def match_year_from_row(row: pd.Series) -> Optional[int]:
    value = pd.to_numeric(pd.Series([row.get("Year", np.nan)]), errors="coerce").iloc[0]
    if pd.notna(value):
        return int(value)
    timestamp = match_date_from_row(row)
    return int(timestamp.year) if timestamp is not None else None


def reference_date_for_row(row_date: Optional[pd.Timestamp], row_year: Optional[int]) -> str:
    if row_date is not None:
        return str(row_date.date())
    if row_year:
        return f"{int(row_year)}-06-01"
    return HISTORY_REFERENCE_DATE


def history_before_row(history_df: pd.DataFrame, row_date: Optional[pd.Timestamp], row_year: Optional[int], freeze_year: bool = False) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    working = history_df.copy()
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    working = working[working["Date"].notna()].copy()
    if row_year:
        return working[working["Date"].dt.year < int(row_year)].copy()
    if row_date is not None:
        return working[working["Date"] < row_date].copy()
    return pd.DataFrame(columns=working.columns)


def years_from_rows(rows: pd.DataFrame) -> set[int]:
    if rows.empty or "Year" not in rows.columns:
        return set()
    return {int(year) for year in pd.to_numeric(rows["Year"], errors="coerce").dropna().tolist()}


def teams_from_rows(rows: pd.DataFrame) -> List[str]:
    teams = []
    if rows.empty:
        return teams
    for column in ("Home", "Away"):
        if column in rows.columns:
            teams.extend(rows[column].dropna().astype(str).map(clean_team_name).tolist())
    return sorted({team for team in teams if team})


def team_features_asof(features: pd.DataFrame, row_year: Optional[int]) -> pd.DataFrame:
    if features is None or features.empty or "Team" not in features.columns:
        return pd.DataFrame()
    working = features.copy()
    year_col = first_existing(working.columns, ["version", "Year", "year", "WorldCupYear", "worldcup_year"])
    if not year_col or row_year is None:
        return pd.DataFrame(columns=working.columns)
    years = pd.to_numeric(working[year_col], errors="coerce")
    scoped = working[years <= int(row_year)].copy()
    if scoped.empty:
        return pd.DataFrame(columns=working.columns)
    scoped[year_col] = pd.to_numeric(scoped[year_col], errors="coerce")
    latest = scoped.sort_values(["Team", year_col], kind="stable").groupby("Team", as_index=False).tail(1)
    return latest.reset_index(drop=True)


def build_training_matrix(
        rows: pd.DataFrame,
        base_model: Optional[WorldCupModel] = None,
        team_features: Optional[pd.DataFrame] = None,
        history_team_features: Optional[pd.DataFrame] = None,
        matchup_features: Optional[pd.DataFrame] = None,
        fixture_feature_rows: Optional[pd.DataFrame] = None,
        feature_columns: Optional[List[str]] = None,
        target: str = "result",
        history_df: Optional[pd.DataFrame] = None,
        teams: Optional[Iterable[str]] = None,
        frozen_years: Optional[set[int]] = None,
        history_weight: float = 1.0,
        recency_weight: float = 0.35,
        host_advantage: float = 45.0,
        max_goals: int = 10,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    working = sort_match_rows(rows)
    if target == "over_under_25":
        working = working[working["OverUnder25"].notna()] if "OverUnder25" in working.columns else working.iloc[0:0]
    if team_features is None:
        team_features = pd.DataFrame()
    records = []
    static_model = base_model
    if static_model is None and history_df is None:
        static_model = WorldCupModel.from_history(pd.DataFrame(), teams=teams or teams_from_rows(working))
    snapshot_cache: Dict[Tuple[str, str], Tuple[WorldCupModel, pd.DataFrame, pd.DataFrame]] = {}
    for _, row in working.iterrows():
        row_year = match_year_from_row(row)
        row_date = match_date_from_row(row)
        if history_df is not None:
            frozen = row_year in (frozen_years or set())
            cache_key = ("year", str(row_year or reference_date_for_row(row_date, row_year)))
            if cache_key not in snapshot_cache:
                history_cutoff = history_before_row(history_df, row_date=row_date, row_year=row_year, freeze_year=frozen)
                reference_date = reference_date_for_row(row_date, row_year)
                snapshot_cache[cache_key] = (
                    WorldCupModel.from_history(
                        history_cutoff,
                        teams=teams or teams_from_rows(working),
                        history_weight=history_weight,
                        recency_weight=recency_weight,
                        host_advantage=host_advantage,
                        max_goals=max_goals,
                    ),
                    build_history_feature_table(history_cutoff, reference_date=reference_date),
                    build_matchup_feature_table(history_cutoff, reference_date=reference_date),
                )
            row_model, row_history_features, row_matchup_features = snapshot_cache[cache_key]
        else:
            row_model = static_model
            row_history_features = history_team_features
            row_matchup_features = matchup_features
        records.append(
            match_feature_row(
                row_model,
                team_features_asof(team_features, row_year),
                row["Home"],
                row["Away"],
                history_team_features=row_history_features,
                matchup_features=row_matchup_features,
                fixture_feature_rows=fixture_feature_rows,
                fixture_id=row.get("FixtureId"),
            )
        )
    x = pd.DataFrame(records).fillna(0.0)
    if feature_columns is None:
        feature_columns = list(x.columns)
    for column in feature_columns:
        if column not in x.columns:
            x[column] = 0.0
    x = x[feature_columns].astype(float)
    if target == "over_under_25":
        return x, working["OverUnder25"].astype(int), feature_columns
    return x, working["Label"].astype(str), feature_columns


def match_feature_row(
        base_model: WorldCupModel,
        team_features: pd.DataFrame,
        home: str,
        away: str,
        history_team_features: Optional[pd.DataFrame] = None,
        matchup_features: Optional[pd.DataFrame] = None,
        fixture_feature_rows: Optional[pd.DataFrame] = None,
        fixture_id: Optional[Any] = None,
) -> Dict[str, float]:
    p_home = base_model.profile(home)
    p_away = base_model.profile(away)
    poisson = base_model.match_probabilities(home, away)
    row = {
        "rating_home": p_home.rating,
        "rating_away": p_away.rating,
        "rating_diff": p_home.rating - p_away.rating,
        "rating_ratio": float((p_home.rating + 1e-6) / max(p_away.rating + 1e-6, 1e-6)),
        "attack_home": p_home.attack,
        "attack_away": p_away.attack,
        "attack_diff": p_home.attack - p_away.attack,
        "attack_ratio": float((p_home.attack + 1e-6) / max(p_away.attack + 1e-6, 1e-6)),
        "defense_home": p_home.defense,
        "defense_away": p_away.defense,
        "defense_diff": p_home.defense - p_away.defense,
        "defense_ratio": float((p_home.defense + 1e-6) / max(p_away.defense + 1e-6, 1e-6)),
        "matches_home": float(p_home.matches),
        "matches_away": float(p_away.matches),
        "home_is_host": 1.0 if home in HOST_TEAMS else 0.0,
        "away_is_host": 1.0 if away in HOST_TEAMS else 0.0,
        "poisson_home_win": float(poisson.get("home", 0.0)),
        "poisson_draw": float(poisson.get("draw", 0.0)),
        "poisson_away_win": float(poisson.get("away", 0.0)),
        "poisson_over25": float(poisson.get("over25", 0.0)),
        "poisson_under25": float(poisson.get("under25", 0.0)),
        "poisson_xg_home": float(poisson.get("lambda1", 0.0)),
        "poisson_xg_away": float(poisson.get("lambda2", 0.0)),
        "poisson_xg_diff": float(poisson.get("lambda1", 0.0)) - float(poisson.get("lambda2", 0.0)),
        "poisson_xg_total": float(poisson.get("lambda1", 0.0)) + float(poisson.get("lambda2", 0.0)),
        "poisson_home_edge": float(poisson.get("home", 0.0)) - float(poisson.get("away", 0.0)),
        "poisson_draw_pressure": float(poisson.get("draw", 0.0)) - abs(float(poisson.get("home", 0.0)) - float(poisson.get("away", 0.0))),
        "host_flag_diff": float((1.0 if home in HOST_TEAMS else 0.0) - (1.0 if away in HOST_TEAMS else 0.0)),
        "rating_attack_interaction": float((p_home.rating - p_away.rating) * (p_home.attack - p_away.attack)),
        "rating_defense_interaction": float((p_home.rating - p_away.rating) * (p_home.defense - p_away.defense)),
    }
    merge_team_feature_block(row, team_features, home, away, prefix="kaggle", limit=24)
    merge_team_feature_block(
        row,
        history_team_features if history_team_features is not None else pd.DataFrame(),
        home,
        away,
        prefix="history",
    )
    merge_matchup_feature_block(row, matchup_features if matchup_features is not None else pd.DataFrame(), home, away)
    merge_fixture_feature_block(
        row,
        fixture_feature_rows if fixture_feature_rows is not None else pd.DataFrame(),
        home,
        away,
        fixture_id=fixture_id,
        prefix="xi",
    )
    return row


def merge_team_feature_block(
        row: Dict[str, float],
        features: pd.DataFrame,
        home: str,
        away: str,
        prefix: str,
        limit: Optional[int] = None,
) -> None:
    if features.empty or "Team" not in features.columns:
        return
    home_features = features[features["Team"].map(normalize_team_key) == normalize_team_key(home)]
    away_features = features[features["Team"].map(normalize_team_key) == normalize_team_key(away)]
    numeric_cols = [column for column in features.columns if column != "Team" and pd.api.types.is_numeric_dtype(features[column])]
    if limit is not None:
        numeric_cols = numeric_cols[:limit]
    for column in numeric_cols:
        home_value = float(home_features[column].iloc[0]) if not home_features.empty else 0.0
        away_value = float(away_features[column].iloc[0]) if not away_features.empty else 0.0
        safe = normalize_column(column)
        row[f"{prefix}_{safe}_home"] = home_value
        row[f"{prefix}_{safe}_away"] = away_value
        row[f"{prefix}_{safe}_diff"] = home_value - away_value


def merge_matchup_feature_block(
        row: Dict[str, float],
        matchup_features: pd.DataFrame,
        home: str,
        away: str,
) -> None:
    if matchup_features.empty or not {"HomeKey", "AwayKey"}.issubset(matchup_features.columns):
        return
    match = matchup_features[
        (matchup_features["HomeKey"] == normalize_team_key(home)) &
        (matchup_features["AwayKey"] == normalize_team_key(away))
    ]
    if match.empty:
        return
    record = match.iloc[0].to_dict()
    for column, value in record.items():
        if column in {"HomeKey", "AwayKey"}:
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
            row[f"h2h_{normalize_column(column)}"] = float(value)


def merge_fixture_feature_block(
        row: Dict[str, float],
        fixture_feature_rows: pd.DataFrame,
        home: str,
        away: str,
        fixture_id: Optional[Any],
        prefix: str,
) -> None:
    if fixture_feature_rows.empty or "fixture_id" not in fixture_feature_rows.columns:
        return
    if fixture_id in {"", None}:
        return
    fixture_key = str(fixture_id)
    scoped = fixture_feature_rows[fixture_feature_rows["fixture_id"].astype(str) == fixture_key]
    if scoped.empty or "Equipo" not in scoped.columns:
        return
    home_features = scoped[scoped["Equipo"].map(normalize_team_key) == normalize_team_key(home)]
    away_features = scoped[scoped["Equipo"].map(normalize_team_key) == normalize_team_key(away)]
    numeric_cols = [column for column in scoped.columns if column not in {"fixture_id", "Equipo", "Rival"} and pd.api.types.is_numeric_dtype(scoped[column])]
    for column in numeric_cols:
        home_value = float(home_features[column].iloc[0]) if not home_features.empty else 0.0
        away_value = float(away_features[column].iloc[0]) if not away_features.empty else 0.0
        safe = normalize_column(column)
        row[f"{prefix}_{safe}_home"] = home_value
        row[f"{prefix}_{safe}_away"] = away_value
        row[f"{prefix}_{safe}_diff"] = home_value - away_value


def build_history_feature_table(history_df: pd.DataFrame, reference_date: str = HISTORY_REFERENCE_DATE) -> pd.DataFrame:
    team_rows = team_history_rows(history_df)
    if team_rows.empty:
        return pd.DataFrame(columns=["Team"])
    reference_ts = pd.Timestamp(reference_date)
    rows: List[Dict[str, Any]] = []
    for team, team_df in team_rows.groupby("Team", sort=True):
        team_df = team_df.sort_values("Date", kind="stable").reset_index(drop=True)
        last_date = team_df["Date"].max()
        days_since = int(max((reference_ts - last_date).days, 0)) if pd.notna(last_date) else 0
        base = {
            "Team": team,
            "matches_total": float(team_df.shape[0]),
            "days_since_last_match": float(days_since),
            "recent_match_volume_365d": float(team_df[team_df["Date"] >= (reference_ts - pd.Timedelta(days=365))].shape[0]),
            "recent_match_volume_730d": float(team_df[team_df["Date"] >= (reference_ts - pd.Timedelta(days=730))].shape[0]),
            "recent_match_volume_1095d": float(team_df[team_df["Date"] >= (reference_ts - pd.Timedelta(days=1095))].shape[0]),
        }
        rest_days = team_df["Date"].diff().dt.days.dropna()
        base["rest_days_avg"] = float(rest_days.mean()) if not rest_days.empty else 0.0
        base["rest_days_std"] = float(rest_days.std(ddof=0)) if not rest_days.empty else 0.0
        base["rest_days_last"] = float(rest_days.iloc[-1]) if not rest_days.empty else 0.0
        base["last_result_points"] = float(team_df["Points"].iloc[-1]) if not team_df.empty else 0.0
        base["last_goal_diff"] = float(team_df["GoalDiff"].iloc[-1]) if not team_df.empty else 0.0
        base["last_goals_for"] = float(team_df["GF"].iloc[-1]) if not team_df.empty else 0.0
        base["last_goals_against"] = float(team_df["GA"].iloc[-1]) if not team_df.empty else 0.0
        base.update(window_summary_features(team_df, len(team_df), prefix="all"))
        for window in HISTORY_FEATURE_WINDOWS:
            base.update(window_summary_features(team_df, window, prefix=f"last_{window}"))
        base["trend_points_ppg_1_vs_5"] = base.get("last_1_points_ppg", 0.0) - base.get("last_5_points_ppg", 0.0)
        base["trend_points_ppg_3_vs_10"] = base.get("last_3_points_ppg", 0.0) - base.get("last_10_points_ppg", 0.0)
        base["trend_points_ppg_5_vs_12"] = base.get("last_5_points_ppg", 0.0) - base.get("last_12_points_ppg", 0.0)
        base["trend_goal_diff_1_vs_5"] = base.get("last_1_goal_diff_avg", 0.0) - base.get("last_5_goal_diff_avg", 0.0)
        base["trend_goal_diff_3_vs_10"] = base.get("last_3_goal_diff_avg", 0.0) - base.get("last_10_goal_diff_avg", 0.0)
        base["trend_win_rate_3_vs_10"] = base.get("last_3_win_rate", 0.0) - base.get("last_10_win_rate", 0.0)
        base["trend_clean_sheet_3_vs_10"] = base.get("last_3_clean_sheet_rate", 0.0) - base.get("last_10_clean_sheet_rate", 0.0)
        base["trend_over25_3_vs_10"] = base.get("last_3_over25_rate", 0.0) - base.get("last_10_over25_rate", 0.0)
        base["trend_attack_2_vs_7"] = base.get("last_2_goals_for_avg", 0.0) - base.get("last_7_goals_for_avg", 0.0)
        base["trend_attack_5_vs_12"] = base.get("last_5_goals_for_avg", 0.0) - base.get("last_12_goals_for_avg", 0.0)
        base["trend_defense_2_vs_7"] = base.get("last_7_goals_against_avg", 0.0) - base.get("last_2_goals_against_avg", 0.0)
        base["trend_defense_5_vs_12"] = base.get("last_12_goals_against_avg", 0.0) - base.get("last_5_goals_against_avg", 0.0)
        base["trend_btts_3_vs_10"] = base.get("last_3_btts_rate", 0.0) - base.get("last_10_btts_rate", 0.0)
        base["trend_under25_3_vs_10"] = base.get("last_3_under25_rate", 0.0) - base.get("last_10_under25_rate", 0.0)
        base["trend_points_ppg_7_vs_12"] = base.get("last_7_points_ppg", 0.0) - base.get("last_12_points_ppg", 0.0)
        base["trend_goal_diff_7_vs_12"] = base.get("last_7_goal_diff_avg", 0.0) - base.get("last_12_goal_diff_avg", 0.0)
        base["volatility_goal_diff_short_vs_long"] = base.get("last_3_goal_diff_std", 0.0) - base.get("last_10_goal_diff_std", 0.0)
        base["volatility_points_short_vs_long"] = base.get("last_3_points_std", 0.0) - base.get("last_10_points_std", 0.0)
        base["volatility_goals_for_short_vs_long"] = base.get("last_3_goals_for_std", 0.0) - base.get("last_10_goals_for_std", 0.0)
        base["volatility_goals_against_short_vs_long"] = base.get("last_3_goals_against_std", 0.0) - base.get("last_10_goals_against_std", 0.0)
        base["form_reversion_points"] = base.get("all_points_ppg", 0.0) - base.get("last_3_points_ppg", 0.0)
        base["form_reversion_goal_diff"] = base.get("all_goal_diff_avg", 0.0) - base.get("last_3_goal_diff_avg", 0.0)
        base["weighted_points_short_vs_long"] = base.get("last_3_weighted_points", 0.0) - base.get("last_10_weighted_points", 0.0)
        base["scoring_trend_3_vs_10"] = base.get("last_3_scoring_rate", 0.0) - base.get("last_10_scoring_rate", 0.0)
        rows.append(base)
    return pd.DataFrame(rows).fillna(0.0)


def team_history_rows(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(columns=["Team", "Date", "GF", "GA", "GoalDiff", "Points", "Win", "Draw", "Loss", "Over25", "Under25", "BTTS", "CleanSheet", "Scored"])
    working = history_df.copy()
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    working = working[working["Date"].notna()].copy()
    rows: List[Dict[str, Any]] = []
    for _, row in working.iterrows():
        try:
            home_goals = float(row.get("G1", 0) or 0)
            away_goals = float(row.get("G2", 0) or 0)
        except (TypeError, ValueError):
            continue
        home_team = clean_team_name(row.get("Team 1"))
        away_team = clean_team_name(row.get("Team 2"))
        if home_team:
            rows.append(team_match_row(home_team, row["Date"], home_goals, away_goals))
        if away_team:
            rows.append(team_match_row(away_team, row["Date"], away_goals, home_goals))
    return pd.DataFrame(rows)


def team_match_row(team: str, match_date: pd.Timestamp, gf: float, ga: float) -> Dict[str, Any]:
    return {
        "Team": team,
        "Date": match_date,
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
    }


def window_summary_features(team_df: pd.DataFrame, window: int, prefix: str) -> Dict[str, float]:
    if team_df.empty:
        return {
            f"{prefix}_points_sum": 0.0,
            f"{prefix}_points_ppg": 0.0,
            f"{prefix}_points_std": 0.0,
            f"{prefix}_goals_for_avg": 0.0,
            f"{prefix}_goals_for_std": 0.0,
            f"{prefix}_goals_against_avg": 0.0,
            f"{prefix}_goals_against_std": 0.0,
            f"{prefix}_goal_diff_avg": 0.0,
            f"{prefix}_goal_diff_std": 0.0,
            f"{prefix}_win_rate": 0.0,
            f"{prefix}_draw_rate": 0.0,
            f"{prefix}_loss_rate": 0.0,
            f"{prefix}_non_loss_rate": 0.0,
            f"{prefix}_over25_rate": 0.0,
            f"{prefix}_under25_rate": 0.0,
            f"{prefix}_btts_rate": 0.0,
            f"{prefix}_clean_sheet_rate": 0.0,
            f"{prefix}_scoring_rate": 0.0,
            f"{prefix}_weighted_points": 0.0,
        }
    recent = team_df.tail(int(window)).copy()
    weights = np.linspace(1.0, 1.0 + max(len(recent) - 1, 0) * 0.12, num=len(recent)) if len(recent) else np.array([1.0])
    return {
        f"{prefix}_points_sum": float(recent["Points"].sum()),
        f"{prefix}_points_ppg": float(recent["Points"].mean()),
        f"{prefix}_points_std": float(recent["Points"].std(ddof=0)),
        f"{prefix}_goals_for_avg": float(recent["GF"].mean()),
        f"{prefix}_goals_for_std": float(recent["GF"].std(ddof=0)),
        f"{prefix}_goals_against_avg": float(recent["GA"].mean()),
        f"{prefix}_goals_against_std": float(recent["GA"].std(ddof=0)),
        f"{prefix}_goal_diff_avg": float(recent["GoalDiff"].mean()),
        f"{prefix}_goal_diff_std": float(recent["GoalDiff"].std(ddof=0)),
        f"{prefix}_win_rate": float(recent["Win"].mean()),
        f"{prefix}_draw_rate": float(recent["Draw"].mean()),
        f"{prefix}_loss_rate": float(recent["Loss"].mean()),
        f"{prefix}_non_loss_rate": float((recent["Win"] + recent["Draw"]).mean()),
        f"{prefix}_over25_rate": float(recent["Over25"].mean()),
        f"{prefix}_under25_rate": float(recent["Under25"].mean()),
        f"{prefix}_btts_rate": float(recent["BTTS"].mean()),
        f"{prefix}_clean_sheet_rate": float(recent["CleanSheet"].mean()),
        f"{prefix}_scoring_rate": float(recent["Scored"].mean()),
        f"{prefix}_weighted_points": float(np.average(recent["Points"], weights=weights)),
    }


def build_matchup_feature_table(history_df: pd.DataFrame, reference_date: str = HISTORY_REFERENCE_DATE) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(columns=["HomeKey", "AwayKey"])
    working = history_df.copy()
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    working = working[working["Date"].notna()].copy()
    rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Tuple[pd.Timestamp, float, float]]] = {}
    for _, row in working.iterrows():
        home = clean_team_name(row.get("Team 1"))
        away = clean_team_name(row.get("Team 2"))
        try:
            g1 = float(row.get("G1", 0) or 0)
            g2 = float(row.get("G2", 0) or 0)
        except (TypeError, ValueError):
            continue
        if home and away:
            match_date = pd.Timestamp(row["Date"])
            grouped.setdefault((normalize_team_key(home), normalize_team_key(away)), []).append((match_date, g1, g2))
            grouped.setdefault((normalize_team_key(away), normalize_team_key(home)), []).append((match_date, g2, g1))
    reference_ts = pd.Timestamp(reference_date)
    for (home_key, away_key), matches in grouped.items():
        ordered_matches = sorted(matches, key=lambda item: item[0])
        recent_matches = ordered_matches[-3:]
        total = float(len(matches))
        home_goals = [item[1] for item in ordered_matches]
        away_goals = [item[2] for item in ordered_matches]
        last_date = ordered_matches[-1][0] if ordered_matches else None
        rows.append({
            "HomeKey": home_key,
            "AwayKey": away_key,
            "matches": total,
            "home_win_rate": float(sum(1 for _, g1, g2 in ordered_matches if g1 > g2) / max(total, 1.0)),
            "draw_rate": float(sum(1 for _, g1, g2 in ordered_matches if g1 == g2) / max(total, 1.0)),
            "away_win_rate": float(sum(1 for _, g1, g2 in ordered_matches if g2 > g1) / max(total, 1.0)),
            "goal_diff_avg": float(np.mean([g1 - g2 for _, g1, g2 in ordered_matches])),
            "goal_diff_std": float(np.std([g1 - g2 for _, g1, g2 in ordered_matches], ddof=0)),
            "goals_for_avg": float(np.mean(home_goals)),
            "goals_against_avg": float(np.mean(away_goals)),
            "over25_rate": float(np.mean([(g1 + g2) >= 3.0 for _, g1, g2 in ordered_matches])),
            "under25_rate": float(np.mean([(g1 + g2) < 3.0 for _, g1, g2 in ordered_matches])),
            "btts_rate": float(np.mean([(g1 > 0 and g2 > 0) for _, g1, g2 in ordered_matches])),
            "days_since_last_h2h": float(max((reference_ts - last_date).days, 0)) if last_date is not None else 0.0,
            "recent_3_matches": float(len(recent_matches)),
            "recent_3_home_win_rate": float(np.mean([g1 > g2 for _, g1, g2 in recent_matches])) if recent_matches else 0.0,
            "recent_3_draw_rate": float(np.mean([g1 == g2 for _, g1, g2 in recent_matches])) if recent_matches else 0.0,
            "recent_3_goal_diff_avg": float(np.mean([g1 - g2 for _, g1, g2 in recent_matches])) if recent_matches else 0.0,
            "recent_3_over25_rate": float(np.mean([(g1 + g2) >= 3.0 for _, g1, g2 in recent_matches])) if recent_matches else 0.0,
        })
    return pd.DataFrame(rows).fillna(0.0)


def labeled_train_row_count(normalized: Dict[str, Any]) -> int:
    return int(normalized["train"].shape[0] or normalized["team_train"].shape[0])


def labeled_test_row_count(normalized: Dict[str, Any]) -> int:
    return int(normalized["test"].shape[0] or normalized["team_test"].shape[0])


def evaluation_strategy(normalized: Dict[str, Any]) -> str:
    if labeled_test_row_count(normalized) > 0:
        if normalized.get("final_test_year"):
            return "final_worldcup_test"
        return "test_file"
    if labeled_train_row_count(normalized) > 0:
        return "holdout_temporal"
    return "unavailable"


def planned_holdout_rows(train_rows: int, eval_size: float = 0.25) -> int:
    if train_rows <= 0:
        return 0
    if train_rows < 4:
        return train_rows
    return max(1, int(round(train_rows * float(eval_size))))


def is_test_or_eval_file(path: Path) -> bool:
    name = path.name.lower()
    return "test" in name or "eval" in name


def training_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_key = normalize_model_key(str(payload.get("model_type") or default_training_payload()["model_type"]))
    spec = MODEL_SPECS[model_key]
    params = dict(spec.defaults)
    for key in MODEL_PARAM_KEYS:
        if key in params and payload.get(key) not in {None, ""}:
            params[key] = coerce_param(payload.get(key), params[key])
    n_jobs = normalize_n_jobs(payload.get("n_jobs", -1))
    device = str(payload.get("device") or "auto").strip().lower()
    if device not in {"auto", "cpu", "cuda"}:
        device = "auto"
    target = normalize_training_target(payload.get("training_target", payload.get("target", "result")))
    raw_market_mode = payload.get("market_mode")
    if raw_market_mode in {None, ""}:
        raw_market_mode = "dual_markets"
    market_mode = normalize_market_mode(raw_market_mode, target)
    default_target = "dual_markets" if market_mode == "dual_markets" else target
    model_id = normalize_worldcup_model_id(payload.get("model_id") or default_model_id(model_key, default_target))
    return {
        "model_id": model_id,
        "model_name": str(payload.get("model_name") or model_id).strip() or model_id,
        "model_type": model_key,
        "training_target": target,
        "market_mode": market_mode,
        "params": params,
        "seed": int(float(payload.get("seed", 2026) or 2026)),
        "n_jobs": n_jobs,
        "device": device,
        "tuning_enabled": bool(payload.get("tuning_enabled", False)),
        "n_trials": max(int(float(payload.get("n_trials", payload.get("trials", 12)) or 12)), 1),
        "optuna_sampler": str(payload.get("optuna_sampler") or "tpe"),
        "optuna_pruner": str(payload.get("optuna_pruner") or "none"),
        "objective": str(payload.get("objective") or "F1"),
        "tune_params": payload.get("tune_params", payload.get("tune", "all")),
    }


def default_model_id(model_key: str, target: str) -> str:
    short_model = {
        "xgboost": "xgb",
        "lightgbm": "lgbm",
        "catboost": "cat",
        "ngboost": "ngb",
    }.get(model_key, model_key)
    short_target = "hibrido" if target == "dual_markets" else "uo25" if target == "over_under_25" else "result"
    return f"mundial-{short_model}-{short_target}"


def normalize_worldcup_model_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "-", text).strip("-._")
    if not text:
        raise WorldCupTrainingError("El nombre/id del modelo Mundial es obligatorio.")
    if len(text) > 80:
        raise WorldCupTrainingError("El id del modelo Mundial debe tener 80 caracteres o menos.")
    return text


def normalize_training_target(value: Any) -> str:
    key = str(value or "result").strip().lower().replace("-", "_")
    if key in {"over_under", "over_under_25", "uo25", "u_o_25", "overunder25"}:
        return "over_under_25"
    return "result"


def normalize_market_mode(value: Any, target: str = "result") -> str:
    key = str(value or target or "result").strip().lower().replace("-", "_")
    if key in {"dual", "dual_markets", "both", "ambos", "all", "result_over_under_25", "result_uo25"}:
        return "dual_markets"
    if key in {"over_under", "over_under_25", "uo25", "u_o_25", "overunder25"}:
        return "over_under_25"
    return "result"


def normalize_walk_forward_mode(value: Any) -> str:
    key = str(value or "none").strip().lower().replace("-", "_")
    if key in {"result_plus_players", "players", "with_players", "player_features"}:
        return "result_plus_players"
    if key in {"result_only", "base", "match_only", "without_players"}:
        return "result_only"
    return "none"


def coerce_param(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "si", "on"}
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    return value


def normalize_n_jobs(value: Any) -> int:
    try:
        n_jobs = int(float(value))
    except (TypeError, ValueError):
        n_jobs = -1
    if n_jobs == 0 or n_jobs < -1:
        return -1
    return n_jobs


def effective_n_jobs(n_jobs: int, cpu_count: int) -> int:
    return cpu_count if n_jobs == -1 else max(1, min(int(n_jobs), int(cpu_count or 1)))


def has_over_under_target(rows: pd.DataFrame) -> bool:
    if "OverUnder25" not in rows.columns:
        return False
    values = pd.to_numeric(rows["OverUnder25"], errors="coerce").dropna()
    return values.shape[0] > 1 and values.astype(int).nunique() >= 2


def read_fixture_feature_rows() -> pd.DataFrame:
    frame = read_optional_csv(WALK_FORWARD_TEAM_FEATURES_FILE)
    if frame.empty:
        return pd.DataFrame()
    output = frame.copy()
    if "fixture_id" not in output.columns and "Fixture" in output.columns:
        output["fixture_id"] = output["Fixture"].astype(str)
    object_cols = {"fixture_id", "Fixture", "Fecha", "Grupo", "Equipo", "Rival", "Prediction safe", "Formacion", "Fuente", "fetched_at"}
    for column in output.columns:
        if column in object_cols:
            continue
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def worldcup_played_fixture_rows(tournament: Dict[str, Any]) -> pd.DataFrame:
    fixture_df = tournament_fixtures_dataframe(tournament)
    if fixture_df.empty:
        return pd.DataFrame(columns=fixture_df.columns)
    working = fixture_df.copy()
    working["_date"] = pd.to_datetime(working["Fecha"], errors="coerce")
    today = pd.Timestamp.now(tz=None).normalize()
    working = working[
        working["Grupo"].astype(str).str.len().gt(0) &
        working["Equipo 1"].astype(str).str.len().gt(1) &
        working["Equipo 2"].astype(str).str.len().gt(1) &
        working["_date"].notna() &
        (working["_date"] < today)
    ].copy()
    return working.sort_values(["_date", "No."], kind="stable")


def completed_worldcup_training_rows(tournament: Dict[str, Any]) -> pd.DataFrame:
    fixture_df = tournament_fixtures_dataframe(tournament)
    if fixture_df.empty or "Goles 1" not in fixture_df.columns or "Goles 2" not in fixture_df.columns:
        return pd.DataFrame(columns=["FixtureId", "Home", "Away", "HG", "AG", "Label", "OverUnder25", "Source"])
    working = fixture_df.copy()
    working["HG"] = pd.to_numeric(working["Goles 1"], errors="coerce")
    working["AG"] = pd.to_numeric(working["Goles 2"], errors="coerce")
    working = working[
        working["HG"].notna() &
        working["AG"].notna() &
        working["Equipo 1"].astype(str).str.len().gt(1) &
        working["Equipo 2"].astype(str).str.len().gt(1) &
        ~working["Equipo 1"].astype(str).str.match(r"^[123W][A-Z0-9/]+$") &
        ~working["Equipo 2"].astype(str).str.match(r"^[123W][A-Z0-9/]+$")
    ].copy()
    if working.empty:
        return pd.DataFrame(columns=["FixtureId", "Home", "Away", "HG", "AG", "Label", "OverUnder25", "Source"])
    working["Label"] = working.apply(lambda row: label_from_goals(row["HG"], row["AG"]), axis=1)
    working["OverUnder25"] = ((working["HG"] + working["AG"]) >= 3.0).astype(int)
    return pd.DataFrame({
        "FixtureId": working["No."].astype(str),
        "Home": working["Equipo 1"].map(clean_team_name),
        "Away": working["Equipo 2"].map(clean_team_name),
        "HG": working["HG"].astype(float),
        "AG": working["AG"].astype(float),
        "Label": working["Label"].astype(str),
        "OverUnder25": working["OverUnder25"].astype(int),
        "Source": "worldcup_2026_walk_forward",
    })


def player_ready_fixture_ids(fixture_features: pd.DataFrame) -> set[str]:
    if fixture_features.empty or "fixture_id" not in fixture_features.columns:
        return set()
    usable = fixture_features.copy()
    if "Prediction safe" in usable.columns:
        usable = usable[usable["Prediction safe"].astype(str).str.lower() == "si"]
    if "Titulares" in usable.columns:
        usable = usable[pd.to_numeric(usable["Titulares"], errors="coerce").fillna(0) >= 11]
    if "Stats conocidos" in usable.columns:
        usable = usable[pd.to_numeric(usable["Stats conocidos"], errors="coerce").fillna(0) >= 10]
    if usable.empty or "Equipo" not in usable.columns:
        return set()
    counts = usable.groupby("fixture_id")["Equipo"].nunique()
    return {str(index) for index, value in counts.items() if int(value) >= 2}


def supplemental_training_rows(
        tournament: Dict[str, Any],
        mode: str,
        dataset_mode: str,
) -> Dict[str, Any]:
    normalized_mode = normalize_walk_forward_mode(mode)
    empty = {
        "mode": normalized_mode,
        "rows": pd.DataFrame(),
        "fixture_ids": [],
        "added_rows": 0,
        "warnings": [],
    }
    if normalized_mode == "none":
        return empty
    if dataset_mode != "match_result":
        empty["warnings"] = ["El reentreno walk-forward solo aplica cuando el dataset Kaggle es de partidos (match_result)."]
        return empty
    completed = completed_worldcup_training_rows(tournament)
    if completed.empty:
        empty["warnings"] = ["No hay partidos 2026 con marcador oficial disponible para incorporar al reentreno."]
        return empty
    warnings: List[str] = []
    if normalized_mode == "result_plus_players":
        fixture_features = read_fixture_feature_rows()
        ready_ids = player_ready_fixture_ids(fixture_features)
        before = int(completed.shape[0])
        completed = completed[completed["FixtureId"].astype(str).isin(ready_ids)].copy()
        if completed.empty:
            empty["warnings"] = ["No hay partidos cerrados con XI y stats pre-partido listos para reentreno enriquecido."]
            return empty
        if int(completed.shape[0]) < before:
            warnings.append("Solo se incorporaron partidos con XI/stats pre-partido completos para el reentreno enriquecido.")
    return {
        "mode": normalized_mode,
        "rows": completed.reset_index(drop=True),
        "fixture_ids": completed["FixtureId"].astype(str).tolist(),
        "added_rows": int(completed.shape[0]),
        "warnings": warnings,
    }


def walk_forward_refresh_state() -> Dict[str, Any]:
    tournament = fallback_tournament_2026()
    cache_2026 = CACHE_ROOT / "worldcup_2026.json"
    if cache_2026.exists():
        try:
            tournament, _ = load_tournament_2026(refresh=False)
        except Exception:
            tournament = fallback_tournament_2026()
    played = worldcup_played_fixture_rows(tournament)
    completed = completed_worldcup_training_rows(tournament)
    matches = read_optional_csv(WALK_FORWARD_MATCHES_FILE)
    fixture_features = read_fixture_feature_rows()
    snapshot_ids = set(matches["fixture_id"].astype(str)) if not matches.empty and "fixture_id" in matches.columns else set()
    included_result_ids = set()
    included_player_ids = set()
    if not matches.empty and "fixture_id" in matches.columns:
        if "included_result_only_at" in matches.columns:
            included_result_ids = set(matches[matches["included_result_only_at"].fillna("").astype(str).str.len() > 0]["fixture_id"].astype(str))
        if "included_with_players_at" in matches.columns:
            included_player_ids = set(matches[matches["included_with_players_at"].fillna("").astype(str).str.len() > 0]["fixture_id"].astype(str))
    played_ids = set(played["No."].astype(str)) if not played.empty else set()
    completed_ids = set(completed["FixtureId"].astype(str)) if not completed.empty else set()
    player_ready_ids = player_ready_fixture_ids(fixture_features)
    stale_ids = sorted(played_ids - snapshot_ids)
    ready_result_ids = sorted(completed_ids - included_result_ids)
    ready_player_ids = sorted((completed_ids & player_ready_ids) - included_player_ids)
    latest_fixture = ""
    if not played.empty:
        latest = played.iloc[-1]
        latest_fixture = f"{latest.get('Equipo 1', '')} vs {latest.get('Equipo 2', '')} ({latest.get('Fecha', '')})"
    note = ""
    if stale_ids:
        note = f"Hay {len(stale_ids)} partidos ya jugados sin snapshot de XI/stats. Conviene recargar datos."
    elif ready_player_ids:
        note = f"Hay {len(ready_player_ids)} partidos listos para reentreno enriquecido."
    elif ready_result_ids:
        note = f"Hay {len(ready_result_ids)} partidos listos para reentreno base."
    return {
        "played_matches": int(len(played_ids)),
        "completed_results": int(len(completed_ids)),
        "snapshot_matches": int(len(snapshot_ids)),
        "player_ready_matches": int(len(player_ready_ids)),
        "stale_match_ids": stale_ids,
        "ready_result_ids": ready_result_ids,
        "ready_player_ids": ready_player_ids,
        "ready_result_only": int(len(ready_result_ids)),
        "ready_with_players": int(len(ready_player_ids)),
        "requires_reload": bool(stale_ids),
        "latest_played_fixture": latest_fixture,
        "note": note,
    }


def mark_walk_forward_ingested(fixture_ids: List[str], mode: str) -> None:
    if not fixture_ids:
        return
    normalized_mode = normalize_walk_forward_mode(mode)
    if normalized_mode == "none":
        return
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = read_optional_csv(WALK_FORWARD_MATCHES_FILE)
    if rows.empty:
        rows = pd.DataFrame({"fixture_id": fixture_ids})
    if "fixture_id" not in rows.columns:
        rows["fixture_id"] = ""
    rows["fixture_id"] = rows["fixture_id"].astype(str)
    for fixture_id in fixture_ids:
        key = str(fixture_id)
        if key not in set(rows["fixture_id"]):
            rows = pd.concat([rows, pd.DataFrame([{"fixture_id": key}])], ignore_index=True)
        if normalized_mode == "result_plus_players":
            rows.loc[rows["fixture_id"] == key, "included_with_players_at"] = timestamp
        else:
            rows.loc[rows["fixture_id"] == key, "included_result_only_at"] = timestamp
    WALK_FORWARD_ROOT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(WALK_FORWARD_MATCHES_FILE, index=False)


def safe_temporal_row_split(rows: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    working = sort_match_rows(rows).reset_index(drop=True)
    if working.shape[0] < 4:
        return working.copy(), working.copy()
    test_count = max(1, int(round(working.shape[0] * float(test_size))))
    split_at = max(1, min(working.shape[0] - test_count, working.shape[0] - 1))
    train_rows = working.iloc[:split_at].copy()
    eval_rows = working.iloc[split_at:].copy()
    if train_rows.empty or eval_rows.empty:
        return working.copy(), working.copy()
    return train_rows.reset_index(drop=True), eval_rows.reset_index(drop=True)


def safe_train_eval_split(x: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    y_series = pd.Series(y).reset_index(drop=True)
    if len(y_series) < 4 or y_series.nunique(dropna=True) < 2:
        return x.copy(), x.copy(), y_series.copy(), y_series.copy()
    test_count = max(1, int(round(len(y_series) * float(test_size))))
    split_at = max(1, min(len(y_series) - test_count, len(y_series) - 1))
    x_ordered = x.reset_index(drop=True)
    x_train = x_ordered.iloc[:split_at].copy()
    x_eval = x_ordered.iloc[split_at:].copy()
    y_train = y_series.iloc[:split_at].copy()
    y_eval = y_series.iloc[split_at:].copy()
    if x_train.empty or x_eval.empty or y_train.nunique(dropna=True) < 2:
        return x.copy(), x.copy(), y_series.copy(), y_series.copy()
    return x_train, x_eval, y_train, y_eval


def encode_labels(y: pd.Series) -> Tuple[pd.Series, List[Any]]:
    values = pd.Series(y).reset_index(drop=True)
    if values.astype(str).isin(TARGET_LABELS).all():
        classes: List[Any] = [label for label in TARGET_LABELS if label in set(values.astype(str))]
        mapping = {label: index for index, label in enumerate(classes)}
        return values.astype(str).map(mapping).astype(int), classes
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        classes = sorted({int(value) for value in numeric.astype(int).tolist()})
        mapping = {label: index for index, label in enumerate(classes)}
        return numeric.astype(int).map(mapping).astype(int), classes
    classes = sorted(values.astype(str).unique().tolist())
    mapping = {label: index for index, label in enumerate(classes)}
    return values.astype(str).map(mapping).astype(int), classes


def encode_existing_labels(y: pd.Series, classes: List[Any]) -> pd.Series:
    values = pd.Series(y).reset_index(drop=True)
    mapping = {str(label): index for index, label in enumerate(classes)}
    return values.map(lambda item: mapping.get(str(item), -1)).astype(int)


def fit_configured_classifier(
        x_train: pd.DataFrame,
        y_train: pd.Series,
        model_key: str,
        params: Dict[str, Any],
        n_jobs: int,
        requested_device: str,
        seed: int,
        num_classes: int,
) -> Dict[str, Any]:
    device, device_warnings = resolve_device(model_key, requested_device)
    try:
        classifier = build_worldcup_classifier(
            model_key=model_key,
            params=params,
            n_jobs=n_jobs,
            device=device,
            seed=seed,
            num_classes=num_classes,
        )
        classifier.fit(x_train, y_train)
        finalize_classifier_for_inference(classifier, model_key=model_key, trained_device=device)
        return {"classifier": classifier, "device": device, "warnings": device_warnings}
    except Exception as exc:
        if device == "cuda":
            fallback = build_worldcup_classifier(
                model_key=model_key,
                params=params,
                n_jobs=n_jobs,
                device="cpu",
                seed=seed,
                num_classes=num_classes,
            )
            fallback.fit(x_train, y_train)
            finalize_classifier_for_inference(fallback, model_key=model_key, trained_device="cpu")
            return {
                "classifier": fallback,
                "device": "cpu",
                "warnings": [*device_warnings, f"CUDA fallo durante fit ({exc.__class__.__name__}); se reintento en CPU."],
            }
        raise


def finalize_classifier_for_inference(classifier, model_key: str, trained_device: str) -> None:
    if model_key != "xgboost":
        return
    if trained_device != "cuda":
        return
    try:
        classifier.set_params(device="cpu")
    except Exception:
        pass
    try:
        classifier.get_booster().set_param({"device": "cpu"})
    except Exception:
        pass


def resolve_device(model_key: str, requested_device: str) -> Tuple[str, List[str]]:
    hardware = detect_hardware()
    warnings_out: List[str] = []
    if model_key == "ngboost":
        if requested_device == "cuda":
            warnings_out.append("NGBoost no usa CUDA en esta integracion; se entreno en CPU.")
        return "cpu", warnings_out
    if requested_device == "cpu":
        return "cpu", warnings_out
    if requested_device in {"auto", "cuda"} and hardware["cuda_available"]:
        return "cuda", warnings_out
    if requested_device == "cuda":
        warnings_out.append(f"CUDA no disponible ({hardware.get('cuda_error') or 'sin dispositivos'}); se entreno en CPU.")
    return "cpu", warnings_out


def build_worldcup_classifier(
        model_key: str,
        params: Dict[str, Any],
        n_jobs: int,
        device: str,
        seed: int,
        num_classes: int,
):
    if model_key == "xgboost":
        from xgboost import XGBClassifier

        kwargs = {
            "booster": "gbtree",
            "n_estimators": int(params.get("n_estimators", 100)),
            "learning_rate": float(params.get("learning_rate", 0.3)),
            "max_depth": int(params.get("max_depth", 6)),
            "min_child_weight": int(params.get("min_child_weight", 1)),
            "reg_lambda": float(params.get("lambda_regularization", 1.0)),
            "reg_alpha": float(params.get("alpha_regularization", 0.0)),
            "random_state": seed,
            "n_jobs": n_jobs,
            "eval_metric": "mlogloss" if num_classes > 2 else "logloss",
        }
        if num_classes > 2:
            kwargs.update({"objective": "multi:softprob", "num_class": num_classes})
        else:
            kwargs["objective"] = "binary:logistic"
        if device == "cuda":
            kwargs.update({"tree_method": "hist", "device": "cuda"})
        return XGBClassifier(**kwargs)
    if model_key == "lightgbm":
        from src.models.classifiers.boosting import WarningFreeLGBMClassifier

        if WarningFreeLGBMClassifier is None:
            raise WorldCupTrainingError("LightGBM no esta instalado. Ejecuta pip install -r requirements.txt.")
        return WarningFreeLGBMClassifier(
            objective="multiclass" if num_classes > 2 else "binary",
            n_estimators=int(params.get("n_estimators", 300)),
            num_leaves=int(params.get("num_leaves", 31)),
            max_depth=int(params.get("max_depth", -1)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            reg_lambda=float(params.get("lambda_regularization", 0.0)),
            reg_alpha=float(params.get("alpha_regularization", 0.0)),
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
            device_type="gpu" if device == "cuda" else "cpu",
        )
    if model_key == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=int(params.get("n_estimators", 300)),
            depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
            random_strength=float(params.get("random_strength", 1.0)),
            loss_function="MultiClass" if num_classes > 2 else "Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=n_jobs,
            task_type="GPU" if device == "cuda" else "CPU",
        )
    if model_key == "ngboost":
        from ngboost import NGBClassifier
        from ngboost.distns import k_categorical
        from sklearn.tree import DecisionTreeRegressor

        return NGBClassifier(
            Dist=k_categorical(num_classes),
            Base=DecisionTreeRegressor(max_depth=int(params.get("max_depth", 3)), random_state=seed),
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.02)),
            minibatch_frac=float(params.get("minibatch_frac", 1.0)),
            natural_gradient=bool(params.get("natural_gradient", True)),
            random_state=seed,
            verbose=False,
        )
    raise WorldCupTrainingError(f'Modelo "{model_key}" no soportado para Mundial.')


def tune_model_if_requested(
        config: Dict[str, Any],
        x_train: pd.DataFrame,
        y_train: pd.Series,
        progress_callback=None,
        market_label: str = "",
) -> Dict[str, Any]:
    if not config.get("tuning_enabled"):
        return {"enabled": False}
    tunables = selected_tunables(config)
    if not tunables:
        return {"enabled": False, "warning": "No hay parametros tunables para este modelo."}
    try:
        import optuna
    except ImportError as exc:
        raise WorldCupTrainingError("Optuna no esta instalado. Ejecuta pip install -r requirements.txt.") from exc

    x_fit, x_eval, y_fit, y_eval = safe_train_eval_split(
        x_train,
        y_train,
        test_size=0.25,
        random_state=config["seed"],
    )
    objective_name = normalize_metric_name(config["objective"])

    def objective(trial):
        trial_params = dict(config["params"])
        for name, spec in tunables.items():
            trial_params[name] = suggest_trial_value(trial, name, spec)
        try:
            fit_result = fit_configured_classifier(
                x_train=x_fit,
                y_train=y_fit,
                model_key=config["model_type"],
                params=trial_params,
                n_jobs=config["n_jobs"],
                requested_device=config["device"],
                seed=config["seed"],
                num_classes=int(pd.Series(y_train).nunique()),
            )
            pred = classifier_predict(fit_result["classifier"], x_eval)
            return metric_score(y_eval, pred, objective_name)
        except Exception as exc:
            trial.set_user_attr("error", f"{exc.__class__.__name__}: {exc}")
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=build_optuna_sampler(optuna, config["optuna_sampler"], config["seed"]),
        pruner=build_optuna_pruner(optuna, config["optuna_pruner"]),
    )
    total_trials = int(config["n_trials"])
    label = market_label or market_label_for_progress(config["training_target"])

    def progress_after_trial(study_obj, trial):
        current = min(len(study_obj.trials), total_trials)
        try:
            best_value = round(float(study_obj.best_value), 4)
            best_trial = int(study_obj.best_trial.number + 1)
        except ValueError:
            best_value = None
            best_trial = 0
        emit_training_progress(
            progress_callback,
            "tuning",
            current,
            total_trials,
            f"Fine-tuning {label}",
            best_value=best_value,
            best_trial=best_trial,
            last_state=getattr(trial.state, "name", str(trial.state)),
            market=label,
            model_type=config["model_type"],
        )

    emit_training_progress(
        progress_callback,
        "tuning",
        0,
        total_trials,
        f"Fine-tuning {label} iniciado",
        market=label,
        model_type=config["model_type"],
    )
    study.optimize(objective, n_trials=total_trials, show_progress_bar=False, callbacks=[progress_after_trial])
    try:
        best_value = round(float(study.best_value), 4)
        best_trial = int(study.best_trial.number + 1)
        best_params = dict(study.best_params)
    except ValueError:
        best_value = 0.0
        best_trial = 0
        best_params = {}
    return {
        "enabled": True,
        "trials": config["n_trials"],
        "sampler": config["optuna_sampler"],
        "pruner": config["optuna_pruner"],
        "objective": objective_name,
        "best_value": best_value,
        "best_trial": best_trial,
        "best_params": best_params,
    }


def selected_tunables(config: Dict[str, Any]) -> Dict[str, Any]:
    spec = MODEL_SPECS[config["model_type"]]
    raw = config.get("tune_params", "all")
    if isinstance(raw, str):
        params = ["all"] if raw.strip().lower() in {"", "all"} else [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, list):
        params = raw
    else:
        params = ["all"]
    candidates = tunable_param_names(spec) if params == ["all"] else params
    output = {}
    for param in candidates:
        if param not in MODEL_PARAM_KEYS:
            continue
        try:
            output[param] = spec.model_cls.get_suggest_param_values(param=param)
        except ValueError:
            continue
    return output


def suggest_trial_value(trial, name: str, spec: Any) -> Any:
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    low, high, step = spec["low"], spec["high"], spec["step"]
    if isinstance(low, int) and isinstance(high, int):
        return trial.suggest_int(name, low, high, step=step)
    return trial.suggest_float(name, float(low), float(high), step=float(step))


def build_optuna_sampler(optuna, name: str, seed: int):
    key = str(name or "tpe").strip().lower().replace("_", "-")
    if key == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if key in {"cmaes", "cma-es"}:
        return optuna.samplers.CmaEsSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed)


def build_optuna_pruner(optuna, name: str):
    key = str(name or "none").strip().lower().replace("_", "-")
    if key == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=5)
    if key in {"successive-halving", "successivehalving", "sha"}:
        return optuna.pruners.SuccessiveHalvingPruner()
    return optuna.pruners.NopPruner()


def normalize_metric_name(value: Any) -> str:
    key = str(value or "F1").strip().lower()
    if key == "accuracy":
        return "Accuracy"
    if key == "precision":
        return "Precision"
    if key == "recall":
        return "Recall"
    return "F1"


def metric_score(y_true, y_pred, metric: str) -> float:
    if metric == "Accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "Precision":
        return float(precision_score(y_true, y_pred, average="macro", zero_division=0.0))
    if metric == "Recall":
        return float(recall_score(y_true, y_pred, average="macro", zero_division=0.0))
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))


def top_feature_importances(clf, feature_columns: List[str], limit: int = 12) -> List[Dict[str, Any]]:
    values = None
    if hasattr(clf, "feature_importances_"):
        values = getattr(clf, "feature_importances_")
    elif hasattr(clf, "get_feature_importance"):
        try:
            values = clf.get_feature_importance()
        except Exception:
            values = None
    if values is None:
        return []
    vector = feature_importance_vector(values, len(feature_columns))
    pairs = []
    for column, value in zip(feature_columns, vector):
        pairs.append({"feature": column, "importance": round(float(value), 6)})
    return sorted(pairs, key=lambda item: item["importance"], reverse=True)[:limit]


def feature_importance_vector(values: Any, feature_count: int) -> np.ndarray:
    if feature_count <= 0:
        return np.asarray([], dtype=float)
    try:
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            output = array
        elif array.ndim > 1 and array.shape[-1] == feature_count:
            axes = tuple(range(array.ndim - 1))
            output = np.mean(np.abs(array), axis=axes)
        elif array.ndim > 1 and array.shape[0] == feature_count:
            axes = tuple(range(1, array.ndim))
            output = np.mean(np.abs(array), axis=axes)
        else:
            output = np.ravel(array)
    except (TypeError, ValueError):
        output = []
        for raw in list(values)[:feature_count]:
            try:
                nested = np.asarray(raw, dtype=float)
                scalar = float(np.mean(np.abs(nested))) if nested.ndim else float(nested.item())
            except Exception:
                scalar = 0.0
            output.append(scalar)
        output = np.asarray(output, dtype=float)
    output = np.nan_to_num(np.asarray(output, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if output.size < feature_count:
        output = np.pad(output, (0, feature_count - output.size), constant_values=0.0)
    return output[:feature_count]


def predict_ml_probs(base_model: WorldCupModel, home: str, away: str, model_id: Optional[str] = None, fixture_id: Optional[Any] = None) -> Tuple[Dict[str, float], str]:
    outputs = predict_ml_outputs(base_model, home, away, model_id=model_id, fixture_id=fixture_id)
    return outputs.get("result", {}), " - ".join(outputs.get("notes", []))


def predict_ml_outputs(base_model: WorldCupModel, home: str, away: str, model_id: Optional[str] = None, fixture_id: Optional[Any] = None) -> Dict[str, Any]:
    record = load_hybrid_model(model_id=model_id)
    if not record:
        return {"result": {}, "over_under_25": {}, "notes": ["Modelo Kaggle no entrenado."]}
    if record.get("bundle") and record.get("market_models"):
        return predict_bundle_ml_outputs(base_model, home, away, record, fixture_id=fixture_id)
    return predict_single_record_ml_outputs(base_model, home, away, record, fixture_id=fixture_id)


def predict_bundle_ml_outputs(base_model: WorldCupModel, home: str, away: str, record: Dict[str, Any], fixture_id: Optional[Any] = None) -> Dict[str, Any]:
    bundle_id = str(record.get("model_id") or active_worldcup_model_id() or "")
    bundle_name = str(record.get("model_name") or bundle_id or "Bundle Mundial")
    market_models = record.get("market_models") or {}
    result_output = {"result": {}, "notes": []}
    over_output = {"over_under_25": {}, "notes": []}
    market_model_names: Dict[str, str] = {}

    result_id = market_models.get("result")
    if result_id:
        result_record = load_hybrid_model(result_id)
        if result_record:
            result_output = predict_single_record_ml_outputs(base_model, home, away, result_record, fixture_id=fixture_id)
            if result_output.get("result"):
                market_model_names["result"] = result_output.get("model_name", "")

    over_id = market_models.get("over_under_25")
    if over_id:
        over_record = load_hybrid_model(over_id)
        if over_record:
            over_output = predict_single_record_ml_outputs(base_model, home, away, over_record, fixture_id=fixture_id)
            if over_output.get("over_under_25"):
                market_model_names["over_under_25"] = over_output.get("model_name", "")

    notes = []
    result_notes = result_output.get("notes", [])
    if over_output.get("over_under_25"):
        result_notes = [note for note in result_notes if "Over/Under 2.5 viene de Poisson" not in str(note)]
    notes.extend(result_notes)
    notes.extend(over_output.get("notes", []))
    if not over_output.get("over_under_25"):
        notes.append("O/U 2.5 viene de Poisson porque el bundle activo no tiene hijo O/U entrenado.")
    return {
        "result": result_output.get("result", {}),
        "over_under_25": over_output.get("over_under_25", {}),
        "model_id": bundle_id,
        "model_name": bundle_name,
        "market_model_ids": {key: value for key, value in market_models.items() if value},
        "market_model_names": market_model_names,
        "notes": unique_strings(notes),
    }


def predict_single_record_ml_outputs(base_model: WorldCupModel, home: str, away: str, record: Dict[str, Any], fixture_id: Optional[Any] = None) -> Dict[str, Any]:
    active_id = str(record.get("model_id") or active_worldcup_model_id() or "")
    model_name = str(record.get("model_name") or active_id or record.get("model_label", "Kaggle"))
    if record.get("mode") == "team_strength":
        home_strength = team_strength_score(record, home)
        away_strength = team_strength_score(record, away)
        diff = home_strength - away_strength
        draw = min(max(0.28 - abs(diff) * 0.12, 0.16), 0.34)
        home_share = 1.0 / (1.0 + np.exp(-diff * 4.0))
        home_prob = (1.0 - draw) * home_share
        away_prob = (1.0 - draw) * (1.0 - home_share)
        return {
            "result": {"H": float(home_prob), "D": float(draw), "A": float(away_prob)},
            "over_under_25": {},
            "model_id": active_id,
            "model_name": model_name,
            "notes": [
                f"Modelo {model_name} team-strength aplicado ({record.get('target_column', '')}).",
                "Over/Under 2.5 viene de Poisson porque el dataset de equipos no contiene goles de partido.",
            ],
        }
    team_features = pd.DataFrame(record.get("team_features", []))
    history_team_features = pd.DataFrame(record.get("history_team_features", []))
    matchup_features = pd.DataFrame(record.get("matchup_features", []))
    fixture_feature_rows = read_fixture_feature_rows()
    x = pd.DataFrame([
        match_feature_row(
            base_model,
            team_features,
            home,
            away,
            history_team_features=history_team_features,
            matchup_features=matchup_features,
            fixture_feature_rows=fixture_feature_rows,
            fixture_id=fixture_id,
        )
    ])
    feature_columns = record.get("feature_columns", BASE_FEATURE_COLUMNS)
    for column in feature_columns:
        if column not in x.columns:
            x[column] = 0.0
    x = x[feature_columns].fillna(0.0).astype(float)
    probabilities = np.asarray(classifier_predict_proba(record["classifier"], x)[0], dtype=float)
    labels = [str(label) for label in record.get("classes", [])]
    target = record.get("effective_target", "result")
    if target == "over_under_25":
        output = {"under25": 0.0, "over25": 0.0}
        for label, probability in zip(labels, probabilities):
            if str(label) in {"1", "True", "true"}:
                output["over25"] += float(probability)
            else:
                output["under25"] += float(probability)
        total = max(sum(output.values()), 1e-9)
        return {
            "result": {},
            "over_under_25": {key: value / total for key, value in output.items()},
            "model_id": active_id,
            "model_name": model_name,
            "notes": [f"Modelo {model_name} aplicado a Over/Under 2.5."],
        }
    output = {label: 0.0 for label in TARGET_LABELS}
    for label, probability in zip(labels, probabilities):
        if label in output:
            output[label] = float(probability)
    total = max(sum(output.values()), 1e-9)
    return {
        "result": {key: value / total for key, value in output.items()},
        "over_under_25": {},
        "model_id": active_id,
        "model_name": model_name,
        "notes": [
            f"Modelo {model_name} aplicado a 1X2.",
            "Over/Under 2.5 viene de Poisson cuando no existe hijo O/U entrenado.",
        ],
    }


def team_strength_score(record: Dict[str, Any], team: str) -> float:
    team_features = pd.DataFrame(record.get("team_features", []))
    if team_features.empty:
        return 0.5
    match = team_features[team_features["Team"].map(normalize_team_key) == normalize_team_key(team)]
    if match.empty:
        return 0.5
    x = match.drop(columns=["Team"], errors="ignore").copy()
    feature_columns = record.get("feature_columns", list(x.columns))
    for column in feature_columns:
        if column not in x.columns:
            x[column] = 0.0
    x = x[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    clf = record["classifier"]
    if hasattr(clf, "predict_proba"):
        probabilities = classifier_predict_proba(clf, x)[0]
        class_values = record.get("classes", [])
        for target_value in (1, "1", True, "True", "true"):
            if target_value in class_values:
                return float(probabilities[class_values.index(target_value)])
    encoded = int(classifier_predict(clf, x)[0])
    classes = record.get("classes", [])
    if 0 <= encoded < len(classes):
        return 1.0 if str(classes[encoded]) in {"1", "True", "true"} else 0.0
    return float(encoded)


def blend_probabilities(base_probs: Dict[str, float], ml_probs: Dict[str, float], ml_weight: float) -> Dict[str, float]:
    weight = min(max(float(ml_weight or 0.0), 0.0), 1.0)
    output = {}
    for label in TARGET_LABELS:
        output[label] = base_probs.get(label, 0.0) * (1.0 - weight) + ml_probs.get(label, base_probs.get(label, 0.0)) * weight
    total = max(sum(output.values()), 1e-9)
    return {label: value / total for label, value in output.items()}


def classifier_predict(classifier, x: pd.DataFrame) -> np.ndarray:
    if classifier.__class__.__module__.startswith("xgboost"):
        return np.asarray(classifier.predict(np.asarray(x, dtype=np.float32)))
    return np.asarray(classifier.predict(x))


def classifier_predict_proba(classifier, x: pd.DataFrame) -> np.ndarray:
    if classifier.__class__.__module__.startswith("xgboost"):
        return np.asarray(classifier.predict_proba(np.asarray(x, dtype=np.float32)))
    return np.asarray(classifier.predict_proba(x))


def blend_total_probabilities(base_probs: Dict[str, float], ml_probs: Dict[str, float], ml_weight: float) -> Dict[str, float]:
    weight = min(max(float(ml_weight or 0.0), 0.0), 1.0)
    output = {}
    for label in ["over25", "under25"]:
        output[label] = base_probs.get(label, 0.0) * (1.0 - weight) + ml_probs.get(label, base_probs.get(label, 0.0)) * weight
    total = max(sum(output.values()), 1e-9)
    return {label: value / total for label, value in output.items()}


def market_sources_payload(result_ml: Dict[str, float], over_under_ml: Dict[str, float], ml_outputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    model_name = str(ml_outputs.get("model_name") or "").strip()
    market_model_names = ml_outputs.get("market_model_names") or {}
    result_model_name = str(market_model_names.get("result") or model_name).strip()
    over_under_model_name = str(market_model_names.get("over_under_25") or model_name).strip()
    result_uses_ml = bool(result_ml)
    over_under_uses_ml = bool(over_under_ml)
    return {
        "result": {
            "label": "1X2",
            "source": "ML + Poisson" if result_uses_ml else "Poisson",
            "uses_ml": result_uses_ml,
            "model_name": result_model_name if result_uses_ml else "",
            "detail": f"1X2 mezcla el modelo {result_model_name} con Elo/Poisson." if result_uses_ml else "1X2 calculado solo con Elo/Poisson.",
        },
        "over_under_25": {
            "label": "O/U 2.5",
            "source": "ML + Poisson" if over_under_uses_ml else "Poisson",
            "uses_ml": over_under_uses_ml,
            "model_name": over_under_model_name if over_under_uses_ml else "",
            "detail": f"O/U 2.5 mezcla el modelo {over_under_model_name} con Poisson." if over_under_uses_ml else "O/U 2.5 calculado con Poisson; no hay modelo O/U activo para este target.",
        },
    }


def child_market_model_id(bundle_id: str, market: str) -> str:
    suffix = "__uo25" if market == "over_under_25" else "__result"
    return normalize_worldcup_model_id(f"{str(bundle_id)[:80 - len(suffix)]}{suffix}")


def market_training_summary(record: Dict[str, Any], result: Dict[str, Any], label: str) -> Dict[str, Any]:
    return {
        "label": label,
        "model_id": record.get("model_id", result.get("model_id", "")),
        "model_name": record.get("model_name", ""),
        "model_type": record.get("model_type", result.get("model_type", "")),
        "model_label": record.get("model_label", ""),
        "effective_target": record.get("effective_target", result.get("effective_target", "")),
        "requested_target": record.get("requested_target", result.get("requested_target", "")),
        "metrics": record.get("metrics", result.get("metrics", {})),
        "confusion_matrix": record.get("confusion_matrix", result.get("confusion_matrix", {})),
        "classes": record.get("classes", []),
        "eval_strategy": record.get("eval_strategy", result.get("eval_strategy", "")),
        "train_rows": int(result.get("train_rows", 0) or 0),
        "eval_rows": int(result.get("eval_rows", 0) or 0),
        "tuning": record.get("tuning", result.get("tuning", {})),
        "tuning_trace": record.get("tuning_trace", result.get("tuning_trace", {})),
        "top_features": record.get("top_features", []),
        "warnings": record.get("warnings", []),
        "hardware": record.get("hardware", result.get("hardware", {})),
        "feature_count": len(record.get("feature_columns", result.get("features", [])) or []),
        "final_test_year": record.get("final_test_year", result.get("final_test_year", "")),
        "split_policy": record.get("split_policy", result.get("split_policy", "")),
    }


def bundle_tuning_trace(markets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    steps = []
    enabled = False
    for key, market in markets.items():
        trace = market.get("tuning_trace") or market.get("tuning") or {}
        enabled = enabled or bool(trace.get("enabled"))
        detail = "desactivado"
        if trace.get("enabled"):
            detail = f"{trace.get('objective', 'F1')}={trace.get('best_value', '')} / trial {trace.get('best_trial', '')}"
        steps.append({
            "name": market.get("label") or key,
            "status": "ok" if market.get("model_id") else "pending",
            "detail": detail,
        })
    return {
        "enabled": enabled,
        "trials": sum(int((market.get("tuning_trace") or {}).get("trials", 0) or 0) for market in markets.values()),
        "sampler": "por mercado",
        "pruner": "por mercado",
        "objective": "por mercado",
        "best_value": "",
        "best_trial": "",
        "best_params": {},
        "steps": steps,
    }


def bundle_etl_steps(base_steps: List[Dict[str, Any]], market_models: Dict[str, str]) -> List[Dict[str, Any]]:
    steps = list(base_steps or [])
    steps.append({
        "name": "Bundle de mercados",
        "status": "ok" if "over_under_25" in market_models else "info",
        "detail": "Entrena 1X2 y O/U como clasificadores independientes bajo un solo modelo seleccionable.",
        "count": len(market_models),
    })
    return steps


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


def classification_metrics(clf, x_train, y_train, x_eval, y_eval) -> Dict[str, Dict[str, float]]:
    return classification_metrics_from_predictions(y_train, classifier_predict(clf, x_train), y_eval, classifier_predict(clf, x_eval))


def classification_metrics_from_predictions(y_train, y_train_pred, y_eval, y_eval_pred) -> Dict[str, Dict[str, float]]:
    return {
        "train": metric_row(y_train, y_train_pred),
        "eval": metric_row(y_eval, y_eval_pred),
    }


def metric_row(y_true, y_pred) -> Dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "F1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0.0)), 3),
        "Precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0.0)), 3),
        "Recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0.0)), 3),
    }


def confusion_matrix_payload(y_true, y_pred, classes: List[Any], target: str = "result") -> Dict[str, Any]:
    encoded_labels = list(range(len(classes)))
    labels = [display_class_label(label, target=target) for label in classes]
    matrix = confusion_matrix(y_true, y_pred, labels=encoded_labels).astype(int)
    total = int(matrix.sum())
    rows = []
    for row_index, actual in enumerate(labels):
        row_total = int(matrix[row_index].sum())
        for column_index, predicted in enumerate(labels):
            count = int(matrix[row_index, column_index])
            rows.append({
                "Actual": actual,
                "Predicho": predicted,
                "Conteo": count,
                "Porcentaje": round(count * 100.0 / max(row_total, 1), 2),
            })
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
        "rows": rows,
        "total": total,
    }


def display_class_label(label: Any, target: str = "result") -> str:
    text = str(label)
    if text == "H":
        return "1 Local"
    if text == "D":
        return "X Empate"
    if text == "A":
        return "2 Visita"
    if text in {"0", "False", "false"}:
        return "Under 2.5" if target == "over_under_25" else "No"
    if text in {"1", "True", "true"}:
        return "Over 2.5" if target == "over_under_25" else "Si"
    return text


def etl_steps(
        files: Iterable[Path],
        normalized: Dict[str, Any],
        eval_strategy: str,
        prepared: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    file_list = list(files)
    train_rows = labeled_train_row_count(normalized)
    test_rows = labeled_test_row_count(normalized)
    prediction_rows = int(normalized["team_prediction"].shape[0])
    feature_rows = int(normalized["team_features"].shape[0])
    walk_forward = walk_forward_status()
    prepared = prepared or {}
    prepared_ready = bool(prepared.get("ready"))
    prepared_stale = bool(prepared.get("stale"))
    prepared_label_source = str(prepared.get("label_source") or "")
    prepared_detail = "Artifact listo para entrenamiento." if prepared_ready and not prepared_stale else "Artifact desactualizado; vuelve a preparar ETL." if prepared_stale else "Aún no se ha preparado el artifact ETL."
    return [
        {
            "name": "Descarga Kaggle",
            "status": "ok" if file_list else "pending",
            "count": len(file_list),
            "detail": "Archivos CSV/XLS disponibles localmente.",
        },
        {
            "name": "Preparar ETL",
            "status": "ok" if prepared_ready and not prepared_stale else "pending" if not prepared_ready else "info",
            "count": train_rows,
            "detail": prepared_detail,
        },
        {
            "name": "Lectura y normalizacion",
            "status": "ok" if train_rows else "pending",
            "count": train_rows,
            "detail": f"Modo activo: {normalized.get('training_mode') or 'sin modo'}; fuente labels: {prepared_label_source or normalized.get('training_mode') or 'sin modo'}.",
        },
        {
            "name": "Split evaluacion",
            "status": "ok" if eval_strategy != "unavailable" else "pending",
            "count": test_rows if test_rows else planned_holdout_rows(train_rows),
            "detail": "Ultimo Mundial como test final" if eval_strategy == "final_worldcup_test" else "Test etiquetado" if eval_strategy == "test_file" else "Holdout temporal desde train" if eval_strategy == "holdout_temporal" else "Sin evaluacion.",
        },
        {
            "name": "Features seleccion",
            "status": "ok" if feature_rows else "pending",
            "count": feature_rows,
            "detail": "Features numericas por seleccion listas para el modelo.",
        },
        {
            "name": "Mercado O/U 2.5",
            "status": "ok" if has_over_under_target(normalized["train"]) else "pending",
            "count": int(pd.to_numeric(normalized["train"].get("OverUnder25"), errors="coerce").dropna().shape[0]) if "OverUnder25" in normalized["train"].columns else 0,
            "detail": "Solo usa goles reales observados; no se generan etiquetas artificiales.",
        },
        {
            "name": "Walk-forward XI",
            "status": "ok" if walk_forward["matches"] else "info",
            "count": walk_forward["matches"],
            "detail": f"{walk_forward['ready_for_retrain']} listos para reentreno, {walk_forward['pending_results']} esperando resultado final.",
        },
        {
            "name": "Prediccion 2026",
            "status": "ok" if prediction_rows else "info",
            "count": prediction_rows,
            "detail": "Filas sin label usadas como features futuras.",
        },
        {
            "name": "Anti-leakage",
            "status": "ok" if train_rows else "pending",
            "count": 0,
            "detail": "No se usan resultados del Mundial 2026 como target.",
        },
    ]


def tuning_trace(tuned: Dict[str, Any]) -> Dict[str, Any]:
    if not tuned or not tuned.get("enabled"):
        return {
            "enabled": False,
            "steps": [{
                "name": "Fine-tuning",
                "status": "off",
                "detail": "Optuna desactivado; se usaron parametros actuales del formulario.",
            }],
        }
    best_params = tuned.get("best_params", {}) or {}
    return {
        "enabled": True,
        "trials": int(tuned.get("trials", 0) or 0),
        "sampler": tuned.get("sampler", ""),
        "pruner": tuned.get("pruner", ""),
        "objective": tuned.get("objective", ""),
        "best_value": tuned.get("best_value", ""),
        "best_trial": tuned.get("best_trial", ""),
        "best_params": best_params,
        "steps": [
            {"name": "Sampler", "status": "ok", "detail": str(tuned.get("sampler", ""))},
            {"name": "Pruner", "status": "ok", "detail": str(tuned.get("pruner", ""))},
            {"name": "Trials", "status": "ok", "detail": str(tuned.get("trials", 0))},
            {"name": "Mejor valor", "status": "ok", "detail": str(tuned.get("best_value", ""))},
            {"name": "Parametros", "status": "ok" if best_params else "info", "detail": ", ".join(best_params.keys()) or "Sin cambios"},
        ],
    }


def worldcup_model_path(model_id: Any) -> Path:
    return WORLD_CUP_MODELS_ROOT / f"{normalize_worldcup_model_id(model_id)}.pkl"


def worldcup_model_meta_path(model_id: Any) -> Path:
    return WORLD_CUP_MODELS_ROOT / f"{normalize_worldcup_model_id(model_id)}.json"


def active_model_state_path() -> Path:
    return WORLD_CUP_MODELS_ROOT / ACTIVE_MODEL_STATE_FILE


def delete_model_files(model_id: Any) -> List[str]:
    removed = []
    for path in (worldcup_model_path(model_id), worldcup_model_meta_path(model_id)):
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def save_hybrid_model(record: Dict[str, Any], model_id: Optional[str] = None) -> None:
    WORLD_CUP_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    model_id = normalize_worldcup_model_id(model_id or record.get("model_id") or default_model_id(record.get("model_type", "xgboost"), record.get("requested_target", "result")))
    record["model_id"] = model_id
    record.setdefault("model_name", model_id)
    record["trained_at"] = record.get("trained_at") or datetime.now(timezone.utc).isoformat()
    with worldcup_model_path(model_id).open("wb") as handle:
        pickle.dump(record, handle)
    with HYBRID_MODEL_FILE.open("wb") as handle:
        pickle.dump(record, handle)
    meta = model_metadata_payload(record, model_id=model_id, model_path=worldcup_model_path(model_id))
    worldcup_model_meta_path(model_id).write_text(json.dumps(json_safe(meta), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    HYBRID_MODEL_META_FILE.write_text(json.dumps(json_safe(meta), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    set_active_worldcup_model(model_id)


def model_metadata_payload(record: Dict[str, Any], model_id: str, model_path: Path) -> Dict[str, Any]:
    meta = {
        "trained": True,
        "bundle": bool(record.get("bundle", False)),
        "market_mode": record.get("market_mode", record.get("requested_target", "")),
        "model_id": model_id,
        "model_name": record.get("model_name") or model_id,
        "model_path": str(model_path),
        "trained_at": record.get("trained_at", ""),
        "feature_count": len(record.get("feature_columns", [])),
        "classes": record.get("classes", []),
        "metrics": record.get("metrics", {}),
        "confusion_matrix": record.get("confusion_matrix", {}),
        "mode": record.get("mode", ""),
        "eval_strategy": record.get("eval_strategy", ""),
        "prediction_rows": record.get("prediction_rows", 0),
        "effective_target": record.get("effective_target", ""),
        "requested_target": record.get("requested_target", ""),
        "target_column": record.get("target_column", ""),
        "model_type": record.get("model_type", ""),
        "model_label": record.get("model_label", ""),
        "model_params": record.get("model_params", {}),
        "tuning": record.get("tuning", {}),
        "tuning_trace": record.get("tuning_trace", {}),
        "etl_steps": record.get("etl_steps", []),
        "hardware": record.get("hardware", {}),
        "warnings": record.get("warnings", []),
        "top_features": record.get("top_features", []),
        "kaggle_files": record.get("kaggle_files", []),
        "history_source": record.get("history_source", ""),
        "final_test_year": record.get("final_test_year", ""),
        "split_policy": record.get("split_policy", ""),
        "hidden_from_catalog": bool(record.get("hidden_from_catalog", False)),
        "markets": record.get("markets", {}),
        "market_models": record.get("market_models", {}),
        "walk_forward_mode": record.get("walk_forward_mode", "none"),
        "walk_forward_summary": record.get("walk_forward_summary", {}),
    }
    return meta


def load_hybrid_model(model_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    path = resolved_model_path(model_id)
    if path is None or not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def read_model_metadata(model_id: Optional[str] = None) -> Dict[str, Any]:
    meta_path = resolved_model_meta_path(model_id)
    if meta_path is not None and meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("model_id", active_worldcup_model_id() or LEGACY_MODEL_ID)
                data["active"] = data.get("model_id") == active_worldcup_model_id()
                return data
        except Exception:
            pass
    return {
        "trained": False,
        "bundle": False,
        "market_mode": "",
        "model_id": "",
        "model_name": "",
        "model_path": str(HYBRID_MODEL_FILE),
        "trained_at": "",
        "feature_count": 0,
        "classes": [],
        "metrics": {},
        "confusion_matrix": {},
        "model_type": "",
        "model_label": "",
        "effective_target": "",
        "requested_target": "",
        "eval_strategy": "",
        "prediction_rows": 0,
        "hardware": detect_hardware(),
        "tuning": {"enabled": False},
        "tuning_trace": tuning_trace({"enabled": False}),
        "etl_steps": [],
        "warnings": [],
        "top_features": [],
        "hidden_from_catalog": False,
        "markets": {},
        "market_models": {},
        "walk_forward_mode": "none",
        "walk_forward_summary": {},
        "final_test_year": "",
        "split_policy": "",
    }


def list_worldcup_models() -> Dict[str, Any]:
    WORLD_CUP_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    active_id = active_worldcup_model_id()
    items: Dict[str, Dict[str, Any]] = {}
    for path in sorted(WORLD_CUP_MODELS_ROOT.glob("*.json")):
        if path.name == ACTIVE_MODEL_STATE_FILE or path == HYBRID_MODEL_META_FILE:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if data.get("hidden_from_catalog"):
            continue
        model_id = normalize_worldcup_model_id(data.get("model_id") or path.stem)
        data["model_id"] = model_id
        data["active"] = model_id == active_id
        items[model_id] = data
    if HYBRID_MODEL_META_FILE.exists() and LEGACY_MODEL_ID not in items:
        try:
            legacy = json.loads(HYBRID_MODEL_META_FILE.read_text(encoding="utf-8"))
        except Exception:
            legacy = {}
        if isinstance(legacy, dict) and legacy:
            legacy.setdefault("model_id", LEGACY_MODEL_ID)
            legacy.setdefault("model_name", "Legacy hybrid")
            legacy["model_path"] = str(HYBRID_MODEL_FILE)
            legacy["legacy"] = True
            legacy["active"] = (active_id in {"", None}) or active_id == LEGACY_MODEL_ID
            items[LEGACY_MODEL_ID] = legacy
    models = sorted(
        items.values(),
        key=lambda item: (not bool(item.get("active")), str(item.get("trained_at", ""))),
        reverse=False,
    )
    return {
        "active_model_id": active_id or (models[0]["model_id"] if models else ""),
        "models": models,
        "model": read_model_metadata(),
    }


def set_active_worldcup_model(model_id: Any) -> Dict[str, Any]:
    model_id = normalize_worldcup_model_id(model_id)
    if model_id == LEGACY_MODEL_ID:
        if not HYBRID_MODEL_FILE.exists():
            raise WorldCupTrainingError(f'El modelo "{model_id}" no existe.')
    elif not worldcup_model_path(model_id).exists():
        raise WorldCupTrainingError(f'El modelo "{model_id}" no existe.')
    WORLD_CUP_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    active_model_state_path().write_text(json.dumps({"model_id": model_id}, indent=2) + "\n", encoding="utf-8")
    return read_model_metadata(model_id=model_id)


def delete_worldcup_model(model_id: Any) -> Dict[str, Any]:
    model_id = normalize_worldcup_model_id(model_id)
    if model_id == LEGACY_MODEL_ID:
        raise WorldCupTrainingError("El modelo legacy no se borra desde la interfaz; entrena un modelo nuevo y activalo.")
    was_active = active_worldcup_model_id() == model_id
    meta = read_model_metadata(model_id=model_id)
    removed = delete_model_files(model_id)
    if meta.get("bundle"):
        for child_id in (meta.get("market_models") or {}).values():
            if child_id:
                removed.extend(delete_model_files(child_id))
    if not removed:
        raise WorldCupTrainingError(f'El modelo "{model_id}" no existe.')
    if HYBRID_MODEL_META_FILE.exists():
        try:
            legacy_meta = json.loads(HYBRID_MODEL_META_FILE.read_text(encoding="utf-8"))
        except Exception:
            legacy_meta = {}
        if isinstance(legacy_meta, dict) and legacy_meta.get("model_id") == model_id:
            for path in (HYBRID_MODEL_FILE, HYBRID_MODEL_META_FILE):
                if path.exists():
                    path.unlink()
                    removed.append(str(path))
    if was_active:
        clear_active_worldcup_model()
    return {"deleted": model_id, "removed": removed, **list_worldcup_models()}


def active_worldcup_model_id() -> str:
    path = active_model_state_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            model_id = normalize_worldcup_model_id(data.get("model_id"))
            if model_id == LEGACY_MODEL_ID or worldcup_model_path(model_id).exists():
                return model_id
        except Exception:
            pass
    if HYBRID_MODEL_FILE.exists():
        return LEGACY_MODEL_ID
    return ""


def clear_active_worldcup_model() -> None:
    path = active_model_state_path()
    if path.exists():
        path.unlink()


def resolved_model_path(model_id: Optional[str]) -> Optional[Path]:
    target = normalize_worldcup_model_id(model_id) if model_id else active_worldcup_model_id()
    if target and target != LEGACY_MODEL_ID:
        path = worldcup_model_path(target)
        if path.exists():
            return path
    if target == LEGACY_MODEL_ID and HYBRID_MODEL_FILE.exists():
        return HYBRID_MODEL_FILE
    if not target and HYBRID_MODEL_FILE.exists():
        return HYBRID_MODEL_FILE
    return None


def resolved_model_meta_path(model_id: Optional[str]) -> Optional[Path]:
    target = normalize_worldcup_model_id(model_id) if model_id else active_worldcup_model_id()
    if target and target != LEGACY_MODEL_ID:
        path = worldcup_model_meta_path(target)
        if path.exists():
            return path
    if target == LEGACY_MODEL_ID and HYBRID_MODEL_META_FILE.exists():
        return HYBRID_MODEL_META_FILE
    if not target and HYBRID_MODEL_META_FILE.exists():
        return HYBRID_MODEL_META_FILE
    return None


def select_prediction_fixture(tournament: Dict[str, Any], fixture_id: Optional[Any] = None, home: Optional[str] = None, away: Optional[str] = None) -> pd.Series:
    fixtures = tournament_fixtures_dataframe(tournament)
    fixtures = fixtures[fixtures["Grupo"].astype(str) != ""].copy()
    if fixture_id not in {"", None}:
        match = fixtures[fixtures["No."].astype(str) == str(fixture_id)]
        if not match.empty:
            return match.iloc[0]
    if home and away:
        home_key = normalize_team_key(home)
        away_key = normalize_team_key(away)
        match = fixtures[
            (fixtures["Equipo 1"].map(normalize_team_key) == home_key) &
            (fixtures["Equipo 2"].map(normalize_team_key) == away_key)
        ]
        if not match.empty:
            return match.iloc[0]
    fixtures["_date"] = pd.to_datetime(fixtures["Fecha"], errors="coerce")
    today = pd.Timestamp.today().normalize()
    upcoming = fixtures[fixtures["_date"].notna() & (fixtures["_date"] >= today)]
    if upcoming.empty:
        upcoming = fixtures[fixtures["_date"].notna()]
    if upcoming.empty:
        return fixtures.iloc[0]
    return upcoming.sort_values(["_date", "No."], kind="stable").iloc[0]


def teams_from_tournament(tournament: Dict[str, Any]) -> List[str]:
    fixtures = tournament_fixtures_dataframe(tournament)
    teams = sorted(set(fixtures["Equipo 1"].dropna().astype(str)) | set(fixtures["Equipo 2"].dropna().astype(str)))
    return [team for team in teams if team and not re.match(r"^[123W][A-Z0-9/]+$", team)]


def discover_dataset_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls"})


def read_table(path: Path) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        return pd.read_excel(path)
    except Exception:
        return pd.DataFrame()


def first_existing(columns: Iterable[str], candidates: List[str]) -> str:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return ""


def label_from_goals(home_goals: Any, away_goals: Any) -> str:
    try:
        g_home = float(home_goals)
        g_away = float(away_goals)
    except (TypeError, ValueError):
        return ""
    if g_home > g_away:
        return "H"
    if g_away > g_home:
        return "A"
    return "D"


def label_from_target(value: Any, home: str, away: str) -> str:
    text = normalize_team_key(value)
    if text in {"h", "home", "local", "1", "win_home", "home_win"}:
        return "H"
    if text in {"a", "away", "visitante", "2", "win_away", "away_win"}:
        return "A"
    if text in {"d", "draw", "tie", "empate", "x", "0"}:
        return "D"
    if text == normalize_team_key(home):
        return "H"
    if text == normalize_team_key(away):
        return "A"
    return ""


def numeric_label(value: Any) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def label_display(label: str, home: str, away: str) -> str:
    if label == "H":
        return home
    if label == "A":
        return away
    return "Empate"


def preview_payload(df: pd.DataFrame, rows: int = 8) -> Dict[str, Any]:
    if df.empty:
        return {"columns": [], "rows": [], "total": 0}
    preview = df.head(rows).astype(object).where(pd.notna(df.head(rows)), "")
    return {
        "columns": [str(column) for column in preview.columns],
        "rows": json_safe(preview.to_dict(orient="records")),
        "total": int(df.shape[0]),
    }


def walk_forward_status() -> Dict[str, int]:
    matches = read_optional_csv(WALK_FORWARD_MATCHES_FILE)
    players = read_optional_csv(WALK_FORWARD_PLAYERS_FILE)
    team_features = read_optional_csv(WALK_FORWARD_TEAM_FEATURES_FILE)
    refresh_state = walk_forward_refresh_state()
    pending_results = int(refresh_state["completed_results"] - refresh_state["ready_result_only"]) if refresh_state["completed_results"] else 0
    ready_for_retrain = int(refresh_state["ready_result_only"])
    return {
        "matches": int(matches.shape[0]),
        "players": int(players.shape[0]),
        "team_rows": int(team_features.shape[0]),
        "pending_results": pending_results,
        "ready_for_retrain": ready_for_retrain,
        "ready_with_players": int(refresh_state["ready_with_players"]),
    }


def capture_walk_forward_snapshot(payload: Dict[str, Any]) -> Dict[str, int]:
    data = dict(payload or {})
    fixture_id = str(data.get("fixture_id") or "").strip()
    if not fixture_id:
        return walk_forward_status()
    WALK_FORWARD_ROOT.mkdir(parents=True, exist_ok=True)
    players = pd.DataFrame(data.get("players", []) or [])
    features = pd.DataFrame(data.get("features", []) or [])
    prediction_safe = bool(data.get("prediction_safe"))
    starters_home = 0
    starters_away = 0
    stats_known = 0
    if not players.empty:
        players = players.copy()
        players["fixture_id"] = fixture_id
        players["date"] = data.get("date", "")
        players["group"] = data.get("group", "")
        players["home"] = data.get("home", "")
        players["away"] = data.get("away", "")
        players["fetched_at"] = data.get("fetched_at", "")
        starters_home = int(players[(players.get("team", "") == data.get("home", "")) & players.get("starter", False)].shape[0])
        starters_away = int(players[(players.get("team", "") == data.get("away", "")) & players.get("starter", False)].shape[0])
        stats_known = int(players["stats"].apply(lambda value: isinstance(value, dict) and bool(value)).sum()) if "stats" in players.columns else 0
        write_deduped_csv(
            WALK_FORWARD_PLAYERS_FILE,
            players.applymap(json_safe),
            subset=["fixture_id", "team", "name"],
        )
    if not features.empty:
        features = features.copy()
        features["fixture_id"] = fixture_id
        features["fetched_at"] = data.get("fetched_at", "")
        write_deduped_csv(
            WALK_FORWARD_TEAM_FEATURES_FILE,
            features.applymap(json_safe),
            subset=["fixture_id", "Equipo"],
        )
    status = "ready_for_retrain" if prediction_safe and stats_known >= 22 else "pending_result"
    existing = read_optional_csv(WALK_FORWARD_MATCHES_FILE)
    included_result_only_at = ""
    included_with_players_at = ""
    if not existing.empty and "fixture_id" in existing.columns:
        current = existing[existing["fixture_id"].astype(str) == fixture_id]
        if not current.empty:
            included_result_only_at = str(current.iloc[-1].get("included_result_only_at", "") or "")
            included_with_players_at = str(current.iloc[-1].get("included_with_players_at", "") or "")
    match_row = pd.DataFrame([{
        "fixture_id": fixture_id,
        "date": data.get("date", ""),
        "group": data.get("group", ""),
        "home": data.get("home", ""),
        "away": data.get("away", ""),
        "source": data.get("source", ""),
        "match_url": data.get("match_url", ""),
        "fetched_at": data.get("fetched_at", ""),
        "prediction_safe": int(prediction_safe),
        "starters_home": starters_home,
        "starters_away": starters_away,
        "stats_known": stats_known,
        "status": status,
        "included_result_only_at": included_result_only_at,
        "included_with_players_at": included_with_players_at,
    }])
    write_deduped_csv(WALK_FORWARD_MATCHES_FILE, match_row, subset=["fixture_id"])
    return walk_forward_status()


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def write_deduped_csv(path: Path, frame: pd.DataFrame, subset: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_optional_csv(path)
    combined = pd.concat([existing, frame], ignore_index=True) if not existing.empty else frame.copy()
    valid_subset = [column for column in subset if column in combined.columns]
    if valid_subset:
        combined = combined.drop_duplicates(subset=valid_subset, keep="last")
    combined.to_csv(path, index=False)


def normalize_column(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_team_key(value: Any) -> str:
    text = str(value or "").lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return [json_safe(item) for item in value.astype(object).where(pd.notna(value), "").to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return {str(key): json_safe(val) for key, val in value.to_dict().items()}
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return value
