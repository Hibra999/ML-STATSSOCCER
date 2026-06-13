from __future__ import annotations

import json
import logging
import math
import os
import pickle
import re
import shutil
import subprocess
import hashlib
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score

from src.cli.model_specs import MODEL_SPECS, normalize_model_key, tunable_param_names
from src.models.classifiers.boosting import catboost_device_params, lightgbm_device_params, xgboost_cuda_params
from src.worldcup.api_football_provider import api_football_feature_table, load_api_football_data
from src.worldcup.data import CACHE_ROOT, clean_team_name, fallback_tournament_2026, group_letter, load_historical_matches, load_tournament_2026, tournament_fixtures_dataframe
from src.worldcup.international_provider import (
    INTERNATIONAL_DATASET_SLUG,
    INTERNATIONAL_MATCHES_FILE,
    INTERNATIONAL_ROOT,
    build_recent15_match_index,
    contextual_poisson_for_match,
    download_international_results,
    international_results_status,
    is_worldcup_tournament,
    load_international_matches,
    recent15_feature_table,
    tournament_weight,
)
from src.worldcup.market_provider import (
    load_market_data,
    market_feature_row as build_market_feature_row,
    market_for_match,
    market_source_priority,
    normalize_market_frame,
    qualifier_feature_table,
)
from src.worldcup.model import (
    HOST_TEAMS,
    TOTAL_GOAL_LINES,
    WorldCupModel,
    dixon_coles_probabilities,
    estimate_dixon_coles_rho,
    score_grid_features,
    total_line_suffix,
)


logger = logging.getLogger("uvicorn.error")
KAGGLE_DATASET_SLUG = "harrachimustapha/fifa-world-cup-team-dataset"
KAGGLE_ROOT = Path("storage") / "worldcup" / "kaggle"
WORLD_CUP_MODELS_ROOT = Path("storage") / "worldcup" / "models"
FEATURE_STORE_ROOT = Path("storage") / "worldcup" / "features"
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
TRAIN_TOTAL_GOAL_LINES = tuple(line for line in TOTAL_GOAL_LINES if line <= 3.5)
TOTAL_GOAL_LINE_SUFFIXES = tuple(total_line_suffix(line) for line in TRAIN_TOTAL_GOAL_LINES)
TOTAL_GOALS_CAP = 6
GOALS_DISTRIBUTION_TARGET = "goals_distribution"
OVER_UNDER_MARKET_TARGETS = tuple(f"over_under_{suffix}" for suffix in TOTAL_GOAL_LINE_SUFFIXES)
OVER_UNDER_TARGET_LINES = {
    f"over_under_{total_line_suffix(line)}": line
    for line in TRAIN_TOTAL_GOAL_LINES
}
MATCH_ROW_COLUMNS = [
    "FixtureId",
    "Date",
    "Year",
    "Home",
    "Away",
    "Label",
    "HG",
    "AG",
    "OverUnder05",
    "OverUnder15",
    "OverUnder25",
    "OverUnder35",
    "Source",
]
MATCH_METADATA_COLUMNS = [
    "is_worldcup_match",
    "tournament",
    "stage",
    "group",
    "knockout",
    "label_source",
    "sample_weight",
]
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
FUTURE_LABEL_EXCLUDED_YEAR = 2026
TARGET_WORLDCUP_YEAR = FUTURE_LABEL_EXCLUDED_YEAR
INTERNATIONAL_TRAINING_START_YEAR = 2014
INTERNATIONAL_TRAINING_START_DATE = f"{INTERNATIONAL_TRAINING_START_YEAR}-01-01"
PREPARED_SCHEMA_VERSION = "worldcup_2026_international_since_2014_v3"
FEATURE_STORE_SCHEMA_VERSION = "worldcup_feature_matrix_v2"
EVAL_STRATEGY_LAST_30 = "last_30_international_test"
SPLIT_POLICY_VALIDATION_LAST_30 = "temporal_since_2014_validation_last_30_test"
BENCHMARK_POLICY = EVAL_STRATEGY_LAST_30
WORLDCUP_XGBOOST_DEFAULTS = {
    "n_estimators": 450,
    "max_depth": 3,
    "min_child_weight": 3,
    "learning_rate": 0.045,
    "lambda_regularization": 3.0,
    "alpha_regularization": 0.25,
}
SAMPLE_WEIGHT_POLICY = {
    "worldcup": 1.65,
    "continental_or_world_official": 1.35,
    "qualifier": 1.25,
    "nations_league": 1.1,
    "other_official": 1.0,
    "friendly": 0.6,
}


class WorldCupTrainingError(RuntimeError):
    pass


class WorldCupFeatureBuildCache:
    def __init__(self):
        self.market_lookup: Dict[int, Dict[Tuple[str, str], pd.DataFrame]] = {}
        self.team_features_asof: Dict[Tuple[int, str], pd.DataFrame] = {}
        self.static_features: Dict[Tuple[Any, ...], Tuple[pd.DataFrame, pd.DataFrame]] = {}
        self.snapshots: Dict[Tuple[Any, ...], Tuple[WorldCupModel, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
        self.recent15_features: Dict[Tuple[Any, ...], pd.DataFrame] = {}
        self.table_lookups: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.matrices: Dict[Tuple[Any, ...], pd.DataFrame] = {}
        self.stats: Dict[str, int] = {
            "matrix_hits": 0,
            "matrix_misses": 0,
            "persistent_matrix_hits": 0,
            "persistent_matrix_misses": 0,
            "snapshot_hits": 0,
            "snapshot_misses": 0,
            "recent15_hits": 0,
            "recent15_misses": 0,
        }

    def summary(self) -> Dict[str, int]:
        return dict(self.stats)


class ConstantProbabilityClassifier:
    def __init__(self, constant_class: int = 0, classes: Optional[List[int]] = None):
        self.constant_class = int(constant_class)
        self.classes_ = np.asarray(classes if classes else [self.constant_class], dtype=int)

    def fit(self, x, y):
        values = pd.Series(y).dropna()
        if values.empty:
            self.constant_class = int(self.classes_[0])
        else:
            self.constant_class = int(values.mode().iloc[0])
        if self.constant_class not in set(self.classes_.tolist()):
            self.classes_ = np.asarray(sorted(set(self.classes_.tolist()) | {self.constant_class}), dtype=int)
        return self

    def predict(self, x):
        return np.full(len(x), self.constant_class, dtype=int)

    def predict_proba(self, x):
        columns = len(self.classes_)
        output = np.zeros((len(x), columns), dtype=float)
        try:
            index = list(self.classes_).index(self.constant_class)
        except ValueError:
            index = 0
        output[:, index] = 1.0
        return output


def emit_training_progress(callback, stage: str, current: int, total: int, message: str, **extra) -> None:
    total = max(int(total or 1), 1)
    current = min(max(int(current or 0), 0), total)
    percent = int(round(current * 100 / total))
    market = extra.get("market")
    market_label = f" [{market}]" if market else ""
    details = []
    if extra.get("model_type"):
        details.append(f"model={extra['model_type']}")
    if extra.get("model_id"):
        details.append(f"id={extra['model_id']}")
    if extra.get("best_value") not in {None, ""}:
        details.append(f"best={extra['best_value']}")
    if extra.get("best_trial") not in {None, "", 0}:
        details.append(f"best_trial={extra['best_trial']}")
    if extra.get("last_state"):
        details.append(f"state={extra['last_state']}")
    if extra.get("market_index") and extra.get("market_total"):
        details.append(f"markets={extra['market_index']}/{extra['market_total']}")
    if extra.get("trials_per_market"):
        details.append(f"trials_per_market={extra['trials_per_market']}")
    if extra.get("total_trial_budget"):
        details.append(f"total_trials={extra['total_trial_budget']}")
    if extra.get("overall_trial_current") not in {None, ""} and extra.get("total_trial_budget"):
        details.append(f"overall_trial={extra['overall_trial_current']}/{extra['total_trial_budget']}")
    if extra.get("elapsed_seconds") not in {None, ""}:
        details.append(f"elapsed={extra['elapsed_seconds']}s")
    if extra.get("rows_per_second") not in {None, ""}:
        details.append(f"rows_per_second={extra['rows_per_second']}")
    if extra.get("eta_seconds") not in {None, ""}:
        details.append(f"eta={extra['eta_seconds']}s")
    if extra.get("rows") not in {None, ""}:
        details.append(f"rows={extra['rows']}")
    if extra.get("features") not in {None, ""}:
        details.append(f"features={extra['features']}")
    if extra.get("progress_every") not in {None, ""}:
        details.append(f"progress_every={extra['progress_every']}")
    feature_cache_detail = format_feature_cache_for_progress(extra.get("feature_cache"))
    if feature_cache_detail:
        details.append(f"feature_cache={feature_cache_detail}")
    detail_label = f" - {' '.join(details)}" if details else ""
    line = f"[mundial-training]{market_label} {message} - {stage} {current}/{total} ({percent}%){detail_label}"
    print(line, flush=True)
    logger.info(line)
    if callback is None:
        return
    callback({
        "stage": stage,
        "current": current,
        "total": total,
        "current_trial": extra.get("overall_trial_current", current) if stage == "tuning" else "",
        "total_trials": extra.get("total_trial_budget", total) if stage == "tuning" else "",
        "percent": percent,
        "message": message,
        **extra,
    })


def format_feature_cache_for_progress(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, dict):
        keys = ("matrix_hits", "matrix_misses", "persistent_matrix_hits", "persistent_matrix_misses", "recent15_hits", "recent15_misses")
        return ",".join(f"{key}:{int(value.get(key, 0) or 0)}" for key in keys)
    return str(value)


def dynamic_feature_progress_every(row_count: int) -> int:
    rows = max(int(row_count or 0), 1)
    return max(500, min(2000, rows // 50))


def feature_progress_every_from_payload(payload: Optional[Dict[str, Any]], row_count: int) -> int:
    payload = payload or {}
    default = dynamic_feature_progress_every(row_count)
    raw_value = payload.get("feature_progress_every")
    if "feature_progress_every" not in payload or raw_value is None or raw_value == "":
        return default
    try:
        value = int(float(raw_value))
    except (TypeError, ValueError):
        return default
    return min(5000, max(100, value))


def market_label_for_progress(target: str) -> str:
    if target == GOALS_DISTRIBUTION_TARGET:
        return "Distribucion goles"
    if is_over_under_target(target):
        return goal_line_label_for_suffix(total_line_suffix(OVER_UNDER_TARGET_LINES[target]))
    return "1X2"


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
            "defaults": worldcup_model_defaults(key),
            "tunables": tunables,
            "supports_cuda": key in {"xgboost", "catboost", "lightgbm"},
        })
    return {
        "models": json_safe(models),
        "targets": [
            {"key": "dual_markets", "label": "1X2 + U/O 0.5-3.5 ML"},
        ],
        "hardware": detect_hardware(),
        "defaults": default_training_payload(),
    }


def detect_hardware() -> Dict[str, Any]:
    cpu_count = int(os.cpu_count() or 1)
    cuda_devices: List[str] = []
    cuda_sources: List[str] = []
    cuda_warnings: List[str] = []
    nvidia_smi = find_nvidia_smi()
    if nvidia_smi:
        devices, warning = query_nvidia_smi(nvidia_smi)
        if devices:
            cuda_devices.extend(devices)
            cuda_sources.append(f"nvidia-smi:{nvidia_smi}")
        elif warning:
            cuda_warnings.append(warning)
    else:
        cuda_warnings.append("nvidia-smi no disponible")
    if not cuda_devices:
        framework_devices, framework_source, framework_warning = framework_cuda_devices()
        if framework_devices:
            cuda_devices.extend(framework_devices)
            cuda_sources.append(framework_source)
        elif framework_warning:
            cuda_warnings.append(framework_warning)
    cuda_available = bool(cuda_devices)
    cuda_warning = "; ".join(dict.fromkeys(item for item in cuda_warnings if item))
    cuda_error = "" if cuda_available else (cuda_warning or "sin dispositivos CUDA detectados")
    return {
        "cpu_count": cpu_count,
        "default_n_jobs": -1,
        "cuda_available": cuda_available,
        "cuda_devices": cuda_devices,
        "cuda_device_names": cuda_device_names(cuda_devices),
        "cuda_detection_source": ", ".join(cuda_sources) if cuda_sources else "none",
        "cuda_detection_sources": cuda_sources,
        "cuda_error": cuda_error,
        "cuda_warning": cuda_warning,
        "device_default": "cuda" if cuda_available else "cpu",
    }


def find_nvidia_smi() -> str:
    path = shutil.which("nvidia-smi")
    if path:
        return str(path)
    for candidate in windows_nvidia_smi_candidates():
        try:
            if candidate.is_file():
                return str(candidate)
        except OSError:
            continue
    return ""


def windows_nvidia_smi_candidates() -> List[Path]:
    if os.name != "nt" and not sys.platform.startswith("win"):
        return []
    candidates: List[Path] = []
    program_roots = [
        os.environ.get("ProgramW6432"),
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
    ]
    for root in program_roots:
        if root:
            candidates.append(Path(root) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe")
    system_root = os.environ.get("SystemRoot")
    if system_root:
        candidates.append(Path(system_root) / "System32" / "nvidia-smi.exe")
    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def query_nvidia_smi(executable: str) -> Tuple[List[str], str]:
    try:
        result = subprocess.run(
            [executable, "-L"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception as exc:
        return [], f"{exc.__class__.__name__}: {exc}"
    if result.returncode == 0:
        devices = [line.strip() for line in str(result.stdout or "").splitlines() if line.strip()]
        if devices:
            return devices, ""
        return [], "nvidia-smi no reporto dispositivos CUDA"
    message = str(result.stderr or result.stdout or "").strip().splitlines()
    if message:
        return [], message[0]
    return [], f"nvidia-smi fallo con codigo {result.returncode}"


def framework_cuda_devices() -> Tuple[List[str], str, str]:
    devices, source, warning = tensorflow_cuda_devices()
    if devices:
        return devices, source, warning
    return [], "", warning


def tensorflow_cuda_devices() -> Tuple[List[str], str, str]:
    try:
        tf = sys.modules.get("tensorflow")
        if tf is None:
            return [], "", ""
        config = getattr(tf, "config", None)
        list_physical_devices = getattr(config, "list_physical_devices", None)
        if not callable(list_physical_devices):
            return [], "", "TensorFlow no expone deteccion de GPU"
        gpus = list_physical_devices("GPU") or []
        devices = [f"TensorFlow GPU: {getattr(gpu, 'name', str(gpu))}" for gpu in gpus]
        return devices, "tensorflow", ""
    except Exception as exc:
        return [], "", f"TensorFlow GPU check fallo ({exc.__class__.__name__}: {exc})"


def cuda_device_names(cuda_devices: Iterable[Any]) -> List[str]:
    names: List[str] = []
    for item in cuda_devices:
        name = str(item or "").strip()
        name = re.sub(r"^GPU\s+\d+\s*:\s*", "", name)
        name = re.sub(r"\s+\(UUID:.*\)$", "", name)
        if name:
            names.append(name)
    return names


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
        **worldcup_model_defaults("xgboost"),
    }


def worldcup_model_defaults(model_key: str) -> Dict[str, Any]:
    defaults = dict(MODEL_SPECS[model_key].defaults)
    if model_key == "xgboost":
        defaults.update(WORLDCUP_XGBOOST_DEFAULTS)
    return defaults


def download_kaggle_dataset(force: bool = False) -> Dict[str, Any]:
    try:
        international_status = download_international_results(force=bool(force))
    except Exception as exc:
        raise WorldCupTrainingError(f"No se pudo descargar all_matches.csv internacional: {exc}") from exc
    status = dataset_status()
    status["international_recent"] = international_status
    status["downloaded_path"] = international_status.get("downloaded_path", "")
    status["copied_files"] = international_status.get("copied_files", [])
    return status


def empty_international_training_dataset() -> Dict[str, Any]:
    return {
        "train": pd.DataFrame(),
        "validation": pd.DataFrame(),
        "test": pd.DataFrame(),
        "team_train": pd.DataFrame(),
        "team_test": pd.DataFrame(),
        "team_prediction": pd.DataFrame(),
        "team_features": pd.DataFrame(),
        "target_column": "",
        "team_columns": [],
        "training_mode": "",
        "trainable": False,
        "preview": {"columns": [], "rows": [], "total": 0},
    }


def dataset_status() -> Dict[str, Any]:
    international_status = international_results_status()
    files = [Path(str(international_status.get("source_path") or international_status.get("file_path") or INTERNATIONAL_MATCHES_FILE))] if international_status.get("exists") or international_status.get("available") else []
    normalized = empty_international_training_dataset()
    prepared = prepared_dataset_status(files=files, normalized=normalized)
    model_meta = read_model_metadata()
    active_dataset = prepared["dataset"] if prepared["ready"] else normalized
    train_rows = labeled_train_row_count(active_dataset)
    validation_rows = labeled_validation_row_count(active_dataset)
    test_rows = labeled_test_row_count(active_dataset)
    eval_strategy = evaluation_strategy(active_dataset)
    walk_forward = walk_forward_status()
    refresh_state = walk_forward_refresh_state()
    return {
        "dataset_slug": INTERNATIONAL_DATASET_SLUG,
        "local_path": str(INTERNATIONAL_ROOT),
        "files": [str(path) for path in files],
        "available": bool(international_status.get("available")),
        "etl_ready": bool(prepared["ready"]),
        "etl_stale": bool(prepared["stale"]),
        "etl_status": prepared["status"],
        "etl_artifact_path": str(PREPARED_DATASET_FILE),
        "prepared_at": prepared.get("prepared_at", ""),
        "prepared_mode": prepared.get("mode", ""),
        "prepared_label_source": prepared.get("label_source", ""),
        "prepared_schema_version": prepared.get("prepared_schema_version", ""),
        "target_worldcup_year": prepared.get("target_worldcup_year", str(TARGET_WORLDCUP_YEAR)),
        "benchmark_worldcup_year": prepared.get("benchmark_worldcup_year", prepared.get("final_test_year", "")),
        "benchmark_policy": prepared.get("benchmark_policy", BENCHMARK_POLICY),
        "label_policy_notes": prepared.get("label_policy_notes", []),
        "final_test_year": prepared.get("final_test_year", ""),
        "split_policy": prepared.get("split_policy", ""),
        "prepared_over_under_ready": bool(prepared.get("over_under_ready", False)),
        "prepared_goals_distribution_ready": bool(prepared.get("goals_distribution_ready", prepared.get("over_under_ready", False))),
        "prepared_warnings": prepared.get("warnings", []),
        "all_matches_rows": int(prepared.get("all_matches_rows", 0)),
        "worldcup_rows": int(prepared.get("worldcup_rows", 0)),
        "class_distribution": prepared.get("class_distribution", {}),
        "sample_weight_policy": prepared.get("sample_weight_policy", SAMPLE_WEIGHT_POLICY),
        "data_quality": prepared.get("data_quality", {}),
        "training_start_year": int(prepared.get("training_start_year", INTERNATIONAL_TRAINING_START_YEAR)),
        "max_label_date": prepared.get("max_label_date", ""),
        "market_rows": int(prepared.get("market_rows", 0)),
        "qualifier_feature_rows": int(prepared.get("qualifier_feature_rows", 0)),
        "market_status": prepared.get("market_status", {}),
        "market_warnings": prepared.get("market_warnings", []),
        "api_football_status": prepared.get("api_football_status", {}),
        "api_football_warnings": prepared.get("api_football_warnings", []),
        "api_football_fixture_rows": int(prepared.get("api_football_fixture_rows", 0)),
        "api_football_stat_rows": int(prepared.get("api_football_stat_rows", 0)),
        "api_football_market_rows": int(prepared.get("api_football_market_rows", 0)),
        "train_rows": train_rows,
        "validation_rows": validation_rows,
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
        "international_recent": international_status,
        "model": model_meta,
        "preview": active_dataset["preview"],
    }


def prepare_training_dataset(force: bool = False, refresh_history: bool = False) -> Dict[str, Any]:
    files: List[Path] = []
    normalized = empty_international_training_dataset()
    if PREPARED_DATASET_FILE.exists() and not force:
        current = prepared_dataset_status(files=files, normalized=normalized)
        if current["ready"] and not current["stale"]:
            return dataset_status()
    prepared = build_prepared_dataset(files=files, normalized=normalized, refresh_history=refresh_history)
    save_prepared_dataset(prepared)
    return dataset_status()


def ensure_prepared_dataset_current(payload: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
    files: List[Path] = []
    normalized = empty_international_training_dataset()
    current = prepared_dataset_status(files=files, normalized=normalized)
    if current["ready"] and not current["stale"]:
        dataset = current.get("dataset") or {}
        if prepared_dataset_schema_valid(dataset):
            return dataset
    international_status = international_results_status()
    if not international_status.get("available"):
        if current["ready"]:
            raise WorldCupTrainingError(
                "El artifact ETL Mundial esta desactualizado y no existe all_matches.csv para regenerarlo."
            )
        raise WorldCupTrainingError("Primero descarga o guarda all_matches.csv internacional y ejecuta Preparar ETL.")
    reason = "schema/targets desactualizados" if current["ready"] else "artifact ausente"
    emit_training_progress(
        progress_callback,
        "prepare_etl",
        0,
        1,
        f"Regenerando ETL Mundial antes de entrenar ({reason})",
    )
    prepare_training_dataset(force=True, refresh_history=bool(payload.get("refresh_history", False)))
    dataset = load_prepared_dataset(required=True)
    if not prepared_dataset_schema_valid(dataset or {}):
        raise WorldCupTrainingError("El ETL regenerado no contiene el schema actual de targets Mundial.")
    emit_training_progress(progress_callback, "prepare_etl", 1, 1, "ETL Mundial vigente")
    return dataset


def prepared_dataset_schema_valid(dataset: Dict[str, Any]) -> bool:
    if not isinstance(dataset, dict):
        return False
    if str(dataset.get("prepared_schema_version") or "") != PREPARED_SCHEMA_VERSION:
        return False
    if not bool(dataset.get("trainable", False)):
        return False
    train_rows = dataset.get("train", pd.DataFrame())
    if not isinstance(train_rows, pd.DataFrame) or train_rows.empty:
        return False
    required_columns = {"Label", "HG", "AG", *[f"OverUnder{suffix}" for suffix in TOTAL_GOAL_LINE_SUFFIXES]}
    if not required_columns.issubset(train_rows.columns):
        return False
    if str(dataset.get("target_column") or "") != "Label + GoalsDistribution + OverUnder05/15/25/35":
        return False
    if str(dataset.get("split_policy") or "") != SPLIT_POLICY_VALIDATION_LAST_30:
        return False
    if int(dataset.get("training_start_year", 0) or 0) != INTERNATIONAL_TRAINING_START_YEAR:
        return False
    validation_rows = dataset.get("validation", pd.DataFrame())
    if not isinstance(validation_rows, pd.DataFrame) or validation_rows.empty:
        return False
    test_rows = dataset.get("test", pd.DataFrame())
    if not isinstance(test_rows, pd.DataFrame) or int(test_rows.shape[0]) != 30:
        return False
    for frame in (train_rows, validation_rows, test_rows):
        dates = pd.to_datetime(frame.get("Date", pd.Series(index=frame.index, dtype=object)), errors="coerce")
        if dates.notna().any() and int(dates.dt.year.min()) < INTERNATIONAL_TRAINING_START_YEAR:
            return False
    return bool(dataset.get("over_under_ready", False)) and bool(dataset.get("goals_distribution_ready", False))


def train_hybrid_model(tournament: Dict[str, Any], payload: Optional[Dict[str, Any]] = None, progress_callback=None) -> Dict[str, Any]:
    payload = payload or {}
    train_config = training_config(payload)
    if train_config["market_mode"] != "dual_markets":
        raise WorldCupTrainingError("Mundial 2026 entrena el bundle principal 1X2 + U/O 0.5-3.5.")
    emit_training_progress(progress_callback, "preparing", 0, 8, "Preparando entrenamiento Mundial")
    ensure_prepared_dataset_current(payload, progress_callback=progress_callback)
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
        shared_context: Optional[Dict[str, Any]] = None,
        feature_cache: Optional[WorldCupFeatureBuildCache] = None,
) -> Dict[str, Any]:
    payload = payload or {}
    train_config = training_config(payload)
    model_id = train_config["model_id"]
    label = market_label or market_label_for_progress(train_config["training_target"])
    single_total_steps = 7
    emit_training_progress(progress_callback, "loading", 1, single_total_steps, f"Cargando artifact/contexto {label}", market=label, model_id=model_id)
    shared_context = shared_context or {}
    files = shared_context.get("files") or []
    normalized = shared_context.get("normalized") or load_prepared_dataset(required=True)
    train_rows = normalized["train"].copy()
    validation_rows = normalized.get("validation", pd.DataFrame()).copy()
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
    model_teams = sorted(
        set(group_teams)
        | set(teams_from_rows(train_rows))
        | set(teams_from_rows(validation_rows))
        | set(teams_from_rows(test_rows))
    )
    history_df = shared_context.get("history_df")
    history_source = shared_context.get("history_source", "")
    if history_df is None:
        history_df, history_source = load_historical_matches(refresh=bool(payload.get("refresh_history", False)))
    feature_store = normalized["team_features"]
    market_rows = normalized.get("market_data", pd.DataFrame())
    qualifier_rows = normalized.get("qualifier_matches", pd.DataFrame())
    api_football = normalized.get("api_football", {}) if isinstance(normalized.get("api_football", {}), dict) else {}
    dc_rho = float(normalized.get("dc_rho", 0.0) or 0.0)
    international_matches = shared_context.get("international_matches")
    if international_matches is None:
        international_matches = load_international_matches(required=False)
    recent15_match_index = shared_context.get("recent15_match_index")
    if recent15_match_index is None:
        recent15_match_index = build_recent15_match_index(international_matches)
    fixture_feature_rows = (
        shared_context.get("fixture_feature_rows")
        if "fixture_feature_rows" in shared_context
        else pd.DataFrame()
    )
    feature_cache = feature_cache or WorldCupFeatureBuildCache()
    target_warning = ""
    eval_strategy = "unavailable"
    effective_target = train_config["training_target"]
    if is_over_under_target(effective_target) and not has_over_under_target(train_rows, effective_target):
        raise WorldCupTrainingError(f"El ETL preparado no contiene goles suficientes para entrenar {market_label_for_progress(effective_target)}.")
    if effective_target == GOALS_DISTRIBUTION_TARGET and not has_goals_distribution_target(train_rows):
        raise WorldCupTrainingError("El ETL preparado no contiene goles suficientes para entrenar distribucion de goles.")
    eval_size = float(payload.get("eval_size", 0.25) or 0.25)
    train_rows = sort_match_rows(train_rows)
    validation_rows = sort_match_rows(validation_rows)
    fit_train_rows = train_rows
    emit_training_progress(
        progress_callback,
        "loading",
        2,
        single_total_steps,
        f"Contexto cargado {label}",
        market=label,
        model_id=model_id,
        train_rows=int(train_rows.shape[0]),
        validation_rows=int(validation_rows.shape[0]),
        test_rows=int(test_rows.shape[0]),
    )
    x_validation = pd.DataFrame()
    y_validation = pd.Series(dtype=object)
    if test_rows.empty:
        eval_strategy = "holdout_temporal"
        split_train_rows, split_eval_rows = safe_temporal_row_split(
            train_rows,
            test_size=eval_size,
        )
        fit_train_rows = split_train_rows
        train_progress_every = feature_progress_every_from_payload(payload, int(split_train_rows.shape[0]))
        emit_training_progress(progress_callback, "features_train", 3, single_total_steps, f"Construyendo features train {label}", market=label, model_id=model_id, rows=int(split_train_rows.shape[0]), progress_every=train_progress_every, feature_cache=feature_cache.summary())
        x_train, y_train, feature_columns = build_training_matrix(
            split_train_rows,
            history_df=history_df,
            teams=model_teams,
            team_features=feature_store,
            market_rows=market_rows,
            qualifier_rows=qualifier_rows,
            api_football=api_football,
            international_matches=international_matches,
            recent15_match_index=recent15_match_index,
            fixture_feature_rows=fixture_feature_rows,
            dc_rho=dc_rho,
            history_weight=float(payload.get("history_weight", 1.0) or 1.0),
            recency_weight=float(payload.get("recency_weight", 0.35) or 0.35),
            host_advantage=float(payload.get("host_advantage", 45.0) or 45.0),
            max_goals=int(payload.get("max_goals", 10) or 10),
            target=effective_target,
            feature_cache=feature_cache,
            progress_callback=progress_callback,
            progress_stage="features_train",
            progress_message=f"Construyendo features train {label}",
            progress_market=label,
            progress_model_id=model_id,
            progress_every=train_progress_every,
        )
        emit_training_progress(progress_callback, "features_train", 4, single_total_steps, f"Features train {label} listas", market=label, model_id=model_id, rows=int(x_train.shape[0]), features=int(x_train.shape[1]), progress_every=train_progress_every, feature_cache=feature_cache.summary())
        eval_progress_every = feature_progress_every_from_payload(payload, int(split_eval_rows.shape[0]))
        emit_training_progress(progress_callback, "features_eval", 5, single_total_steps, f"Construyendo features eval {label}", market=label, model_id=model_id, rows=int(split_eval_rows.shape[0]), progress_every=eval_progress_every, feature_cache=feature_cache.summary())
        x_eval, y_eval, _ = build_training_matrix(
            split_eval_rows,
            history_df=history_df,
            teams=model_teams,
            team_features=feature_store,
            market_rows=market_rows,
            qualifier_rows=qualifier_rows,
            api_football=api_football,
            international_matches=international_matches,
            recent15_match_index=recent15_match_index,
            fixture_feature_rows=fixture_feature_rows,
            dc_rho=dc_rho,
            feature_columns=feature_columns,
            frozen_years=years_from_rows(split_eval_rows),
            history_weight=float(payload.get("history_weight", 1.0) or 1.0),
            recency_weight=float(payload.get("recency_weight", 0.35) or 0.35),
            host_advantage=float(payload.get("host_advantage", 45.0) or 45.0),
            max_goals=int(payload.get("max_goals", 10) or 10),
            target=effective_target,
            feature_cache=feature_cache,
            progress_callback=progress_callback,
            progress_stage="features_eval",
            progress_message=f"Construyendo features eval {label}",
            progress_market=label,
            progress_model_id=model_id,
            progress_every=eval_progress_every,
        )
        emit_training_progress(progress_callback, "features_eval", 6, single_total_steps, f"Features eval {label} listas", market=label, model_id=model_id, rows=int(x_eval.shape[0]), features=int(x_eval.shape[1]), progress_every=eval_progress_every, feature_cache=feature_cache.summary())
    else:
        eval_strategy = evaluation_strategy(normalized)
        if eval_strategy == "test_file":
            eval_strategy = EVAL_STRATEGY_LAST_30
        test_rows = sort_match_rows(test_rows)
        fit_train_rows = train_rows
        train_progress_every = feature_progress_every_from_payload(payload, int(train_rows.shape[0]))
        emit_training_progress(progress_callback, "features_train", 3, single_total_steps, f"Construyendo features train {label}", market=label, model_id=model_id, rows=int(train_rows.shape[0]), progress_every=train_progress_every, feature_cache=feature_cache.summary())
        x_train, y_train, feature_columns = build_training_matrix(
            train_rows,
            history_df=history_df,
            teams=model_teams,
            team_features=feature_store,
            market_rows=market_rows,
            qualifier_rows=qualifier_rows,
            api_football=api_football,
            international_matches=international_matches,
            recent15_match_index=recent15_match_index,
            fixture_feature_rows=fixture_feature_rows,
            dc_rho=dc_rho,
            history_weight=float(payload.get("history_weight", 1.0) or 1.0),
            recency_weight=float(payload.get("recency_weight", 0.35) or 0.35),
            host_advantage=float(payload.get("host_advantage", 45.0) or 45.0),
            max_goals=int(payload.get("max_goals", 10) or 10),
            target=effective_target,
            feature_cache=feature_cache,
            progress_callback=progress_callback,
            progress_stage="features_train",
            progress_message=f"Construyendo features train {label}",
            progress_market=label,
            progress_model_id=model_id,
            progress_every=train_progress_every,
        )
        emit_training_progress(progress_callback, "features_train", 4, single_total_steps, f"Features train {label} listas", market=label, model_id=model_id, rows=int(x_train.shape[0]), features=int(x_train.shape[1]), progress_every=train_progress_every, feature_cache=feature_cache.summary())
        if not validation_rows.empty:
            validation_progress_every = feature_progress_every_from_payload(payload, int(validation_rows.shape[0]))
            emit_training_progress(progress_callback, "features_validation", 4, single_total_steps, f"Construyendo features validacion {label}", market=label, model_id=model_id, rows=int(validation_rows.shape[0]), progress_every=validation_progress_every, feature_cache=feature_cache.summary())
            x_validation, y_validation, _ = build_training_matrix(
                validation_rows,
                history_df=history_df,
                teams=model_teams,
                team_features=feature_store,
                market_rows=market_rows,
                qualifier_rows=qualifier_rows,
                api_football=api_football,
                international_matches=international_matches,
                recent15_match_index=recent15_match_index,
                fixture_feature_rows=fixture_feature_rows,
                dc_rho=dc_rho,
                feature_columns=feature_columns,
                frozen_years=years_from_rows(validation_rows),
                history_weight=float(payload.get("history_weight", 1.0) or 1.0),
                recency_weight=float(payload.get("recency_weight", 0.35) or 0.35),
                host_advantage=float(payload.get("host_advantage", 45.0) or 45.0),
                max_goals=int(payload.get("max_goals", 10) or 10),
                target=effective_target,
                feature_cache=feature_cache,
                progress_callback=progress_callback,
                progress_stage="features_validation",
                progress_message=f"Construyendo features validacion {label}",
                progress_market=label,
                progress_model_id=model_id,
                progress_every=validation_progress_every,
            )
            emit_training_progress(progress_callback, "features_validation", 4, single_total_steps, f"Features validacion {label} listas", market=label, model_id=model_id, rows=int(x_validation.shape[0]), features=int(x_validation.shape[1]), progress_every=validation_progress_every, feature_cache=feature_cache.summary())
        eval_progress_every = feature_progress_every_from_payload(payload, int(test_rows.shape[0]))
        emit_training_progress(progress_callback, "features_eval", 5, single_total_steps, f"Construyendo features eval {label}", market=label, model_id=model_id, rows=int(test_rows.shape[0]), progress_every=eval_progress_every, feature_cache=feature_cache.summary())
        x_eval, y_eval, _ = build_training_matrix(
            test_rows,
            history_df=history_df,
            teams=model_teams,
            team_features=feature_store,
            market_rows=market_rows,
            qualifier_rows=qualifier_rows,
            api_football=api_football,
            international_matches=international_matches,
            recent15_match_index=recent15_match_index,
            fixture_feature_rows=fixture_feature_rows,
            dc_rho=dc_rho,
            feature_columns=feature_columns,
            frozen_years=years_from_rows(test_rows),
            history_weight=float(payload.get("history_weight", 1.0) or 1.0),
            recency_weight=float(payload.get("recency_weight", 0.35) or 0.35),
            host_advantage=float(payload.get("host_advantage", 45.0) or 45.0),
            max_goals=int(payload.get("max_goals", 10) or 10),
            target=effective_target,
            feature_cache=feature_cache,
            progress_callback=progress_callback,
            progress_stage="features_eval",
            progress_message=f"Construyendo features eval {label}",
            progress_market=label,
            progress_model_id=model_id,
            progress_every=eval_progress_every,
        )
        emit_training_progress(progress_callback, "features_eval", 6, single_total_steps, f"Features eval {label} listas", market=label, model_id=model_id, rows=int(x_eval.shape[0]), features=int(x_eval.shape[1]), progress_every=eval_progress_every, feature_cache=feature_cache.summary())

    if x_train.empty or pd.Series(y_train).dropna().empty:
        raise WorldCupTrainingError("No hay filas entrenables para el objetivo seleccionado.")
    has_validation_matrix = isinstance(x_validation, pd.DataFrame) and not x_validation.empty and not pd.Series(y_validation).dropna().empty
    label_seed = pd.concat([pd.Series(y_train), pd.Series(y_validation)], ignore_index=True) if has_validation_matrix else pd.Series(y_train)
    _, label_classes = encode_target_labels(label_seed, effective_target)
    y_train_encoded = encode_existing_labels(y_train, label_classes)
    y_eval_encoded = encode_existing_labels(y_eval, label_classes)
    y_validation_encoded = encode_existing_labels(y_validation, label_classes) if has_validation_matrix else pd.Series(dtype=int)
    base_train_sample_weight = align_sample_weights(sample_weights_for_rows(fit_train_rows, effective_target), len(y_train))
    tuned = tune_model_if_requested(
        train_config,
        x_train,
        y_train_encoded,
        sample_weight=base_train_sample_weight,
        x_validation=x_validation if has_validation_matrix else None,
        y_validation=y_validation_encoded if not y_validation_encoded.empty else None,
        progress_callback=progress_callback,
        market_label=label,
    )
    if tuned.get("best_params"):
        train_config["params"].update(tuned["best_params"])
    x_fit_final = x_train
    y_fit_final = y_train_encoded
    fit_sample_weight = base_train_sample_weight
    if has_validation_matrix:
        x_fit_final = pd.concat([x_train, x_validation], ignore_index=True)
        y_fit_final = pd.concat([pd.Series(y_train_encoded), pd.Series(y_validation_encoded)], ignore_index=True)
        fit_train_rows = pd.concat([fit_train_rows, validation_rows], ignore_index=True)
        fit_sample_weight = align_sample_weights(sample_weights_for_rows(fit_train_rows, effective_target), len(y_fit_final))
    emit_training_progress(progress_callback, "fit", 6, single_total_steps, f"Entrenando clasificador final {label}", market=label, model_id=model_id)
    fit_result = fit_configured_classifier(
        x_train=x_fit_final,
        y_train=y_fit_final,
        model_key=train_config["model_type"],
        params=train_config["params"],
        n_jobs=train_config["n_jobs"],
        requested_device=train_config["device"],
        seed=train_config["seed"],
        num_classes=len(label_classes),
        sample_weight=fit_sample_weight,
    )
    clf = fit_result["classifier"]
    y_train_pred = classifier_predict(clf, x_fit_final)
    y_eval_pred = classifier_predict(clf, x_eval)
    y_train_proba = classifier_predict_proba(clf, x_fit_final)
    y_eval_proba = classifier_predict_proba(clf, x_eval)
    emit_training_progress(progress_callback, "metrics", 7, single_total_steps, f"Calculando metricas {label}", market=label, model_id=model_id)
    metrics = classification_metrics_from_predictions(
        y_fit_final,
        y_train_pred,
        y_eval_encoded,
        y_eval_pred,
        y_train_proba=y_train_proba,
        y_eval_proba=y_eval_proba,
        classes=label_classes,
        target=effective_target,
        x_train=x_fit_final,
        x_eval=x_eval,
    )
    calibration = calibration_payload(
        y_fit_final,
        y_train_proba,
        y_eval_encoded,
        y_eval_proba,
        classes=label_classes,
        target=effective_target,
        x_train=x_fit_final,
        x_eval=x_eval,
    )
    y_fit_labels = pd.concat([pd.Series(y_train), pd.Series(y_validation)], ignore_index=True) if has_validation_matrix else pd.Series(y_train)
    metrics["split_support"] = split_support_payload(y_fit_labels, y_eval, effective_target)
    metrics["validation_rows"] = int(x_validation.shape[0]) if isinstance(x_validation, pd.DataFrame) else 0
    confusion = confusion_matrix_payload(y_eval_encoded, y_eval_pred, label_classes, target=effective_target)
    derived_total_markets = derived_total_market_metrics(
        y_train=y_fit_labels,
        y_train_pred=decode_encoded_predictions(y_train_pred, label_classes),
        y_eval=y_eval,
        y_eval_pred=decode_encoded_predictions(y_eval_pred, label_classes),
    ) if effective_target == GOALS_DISTRIBUTION_TARGET else {}
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
        "market_data": market_rows.to_dict(orient="records") if isinstance(market_rows, pd.DataFrame) else [],
        "qualifier_matches": qualifier_rows.to_dict(orient="records") if isinstance(qualifier_rows, pd.DataFrame) else [],
        "api_football": api_football_records(api_football),
        "market_rows": int(normalized.get("market_rows", 0)),
        "qualifier_feature_rows": int(normalized.get("qualifier_feature_rows", 0)),
        "market_status": normalized.get("market_status", {}),
        "market_warnings": normalized.get("market_warnings", []),
        "api_football_status": normalized.get("api_football_status", {}),
        "api_football_warnings": normalized.get("api_football_warnings", []),
        "api_football_fixture_rows": int(normalized.get("api_football_fixture_rows", 0) or 0),
        "api_football_stat_rows": int(normalized.get("api_football_stat_rows", 0) or 0),
        "api_football_market_rows": int(normalized.get("api_football_market_rows", 0) or 0),
        "international_recent": normalized.get("international_recent", international_results_status()),
        "all_matches_rows": int(normalized.get("all_matches_rows", 0) or 0),
        "worldcup_rows": int(normalized.get("worldcup_rows", 0) or 0),
        "class_distribution": normalized.get("class_distribution", {}),
        "sample_weight_policy": normalized.get("sample_weight_policy", SAMPLE_WEIGHT_POLICY),
        "sample_weight_summary": sample_weight_summary(fit_sample_weight),
        "data_quality": normalized.get("data_quality", {}),
        "dc_rho": dc_rho,
        "source_files": [str(path) for path in normalized.get("source_files", [])],
        "kaggle_files": [],
        "history_source": normalized.get("history_source", history_source),
        "metrics": metrics,
        "confusion_matrix": confusion,
        "calibration": calibration,
        "derived_total_markets": derived_total_markets,
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
        "warnings": unique_strings([warning for warning in [target_warning, *normalized.get("warnings", []), *normalized.get("market_warnings", []), *normalized.get("api_football_warnings", []), *fit_result.get("warnings", [])] if warning]),
        "top_features": top_feature_importances(clf, feature_columns),
        "feature_inventory": feature_inventory_payload(feature_columns, x_train=x_fit_final, x_eval=x_eval),
        "feature_cache": feature_cache.summary(),
        "walk_forward_mode": walk_forward_mode,
        "walk_forward_summary": walk_forward_summary,
        "prepared_schema_version": normalized.get("prepared_schema_version", ""),
        "target_worldcup_year": str(normalized.get("target_worldcup_year") or TARGET_WORLDCUP_YEAR),
        "benchmark_worldcup_year": normalized.get("benchmark_worldcup_year", normalized.get("final_test_year", "")),
        "benchmark_policy": normalized.get("benchmark_policy", BENCHMARK_POLICY),
        "label_policy_notes": normalized.get("label_policy_notes", []),
        "final_test_year": normalized.get("final_test_year", ""),
        "split_policy": normalized.get("split_policy", ""),
        "training_start_year": int(normalized.get("training_start_year", INTERNATIONAL_TRAINING_START_YEAR) or INTERNATIONAL_TRAINING_START_YEAR),
        "max_label_date": normalized.get("max_label_date", ""),
        "validation_rows": int(x_validation.shape[0]) if isinstance(x_validation, pd.DataFrame) else 0,
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
        "calibration": calibration,
        "features": feature_columns,
        "feature_inventory": record["feature_inventory"],
        "train_rows": int(len(y_fit_final)),
        "validation_rows": int(x_validation.shape[0]) if isinstance(x_validation, pd.DataFrame) else 0,
        "eval_rows": int(len(y_eval)),
        "source": normalized.get("label_source", INTERNATIONAL_DATASET_SLUG),
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
        "prepared_schema_version": record["prepared_schema_version"],
        "target_worldcup_year": record["target_worldcup_year"],
        "benchmark_worldcup_year": record["benchmark_worldcup_year"],
        "benchmark_policy": record["benchmark_policy"],
        "label_policy_notes": record["label_policy_notes"],
        "final_test_year": normalized.get("final_test_year", ""),
        "split_policy": normalized.get("split_policy", ""),
        "training_start_year": record["training_start_year"],
        "max_label_date": record["max_label_date"],
        "all_matches_rows": int(normalized.get("all_matches_rows", 0) or 0),
        "worldcup_rows": int(normalized.get("worldcup_rows", 0) or 0),
        "class_distribution": normalized.get("class_distribution", {}),
        "sample_weight_summary": record["sample_weight_summary"],
        "data_quality": record["data_quality"],
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
    files: List[Path] = []
    normalized = load_prepared_dataset(required=True)
    if not prepared_dataset_schema_valid(normalized):
        normalized = ensure_prepared_dataset_current(payload, progress_callback=progress_callback)
        files = []
    if not normalized["trainable"]:
        raise WorldCupTrainingError("El ETL preparado no dejo filas entrenables para el modelo 1X2.")
    history_df, history_source = load_historical_matches(refresh=bool(payload.get("refresh_history", False)))
    international_matches = load_international_matches(required=False)
    shared_context = {
        "files": files,
        "normalized": normalized,
        "history_df": history_df,
        "history_source": history_source,
        "international_matches": international_matches,
        "recent15_match_index": build_recent15_match_index(international_matches),
        "fixture_feature_rows": pd.DataFrame(),
    }
    feature_cache = WorldCupFeatureBuildCache()

    common_payload = dict(payload)
    market_plan: List[Tuple[str, str]] = [
        ("result", "1X2"),
        *[(target, market_label_for_progress(target)) for target in OVER_UNDER_MARKET_TARGETS],
    ]
    if normalized.get("goals_distribution_ready", False):
        market_plan.append((GOALS_DISTRIBUTION_TARGET, "Distribucion goles"))

    warnings: List[str] = []
    market_results: Dict[str, Dict[str, Any]] = {}
    market_models: Dict[str, str] = {}
    result_result: Dict[str, Any] = {}
    result_record: Dict[str, Any] = {}
    goals_record: Dict[str, Any] = {}
    total_steps = len(market_plan) + 2
    trials_per_market = int(train_config.get("n_trials", 0) or 0) if train_config.get("tuning_enabled") else 0
    total_trial_budget = len(market_plan) * trials_per_market

    for index, (target, label) in enumerate(market_plan, start=1):
        child_id = child_market_model_id(bundle_id, target)
        child_payload = {
            **common_payload,
            "market_mode": target,
            "training_target": target,
            "model_id": child_id,
            "model_name": f"{bundle_name} - {label}",
            "hidden_from_catalog": True,
            "market_index": index,
            "market_total": len(market_plan),
            "trials_per_market": trials_per_market,
            "total_trial_budget": total_trial_budget,
            "trial_offset": (index - 1) * trials_per_market,
        }
        emit_training_progress(
            progress_callback,
            f"market-{target}",
            index,
            total_steps,
            (
                f"Entrenando mercado {label} ({index}/{len(market_plan)}; "
                f"fine-tuning {trials_per_market} trials por mercado, {total_trial_budget} total)"
                if trials_per_market
                else f"Entrenando mercado {label} ({index}/{len(market_plan)}; fine-tuning desactivado)"
            ),
            market=label,
            model_id=bundle_id,
            market_index=index,
            market_total=len(market_plan),
            trials_per_market=trials_per_market,
            total_trial_budget=total_trial_budget,
        )
        child_result = train_single_hybrid_model(
            tournament=tournament,
            payload=child_payload,
            progress_callback=progress_callback,
            market_label=label,
            shared_context=shared_context,
            feature_cache=feature_cache,
        )
        child_record = load_hybrid_model(child_id) or {}
        market_results[target] = market_training_summary(child_record, child_result, label)
        market_models[target] = child_id
        warnings.extend(child_record.get("warnings", []))
        if target == "result":
            result_result = child_result
            result_record = child_record
        elif target == GOALS_DISTRIBUTION_TARGET:
            goals_record = child_record

    if not result_record:
        raise WorldCupTrainingError("No se pudo entrenar el mercado principal 1X2.")

    warnings = unique_strings(warnings)
    trained_at = datetime.now(timezone.utc).isoformat()
    bundle_target_column = str(normalized.get("target_column") or result_record.get("target_column") or "result")
    derived_total_markets = goals_record.get("derived_total_markets", {}) if goals_record else {}
    bundle_record = {
        "bundle": True,
        "market_mode": "dual_markets",
        "classifier": None,
        "feature_columns": result_record.get("feature_columns", []),
        "team_features": result_record.get("team_features", []),
        "history_team_features": result_record.get("history_team_features", []),
        "matchup_features": result_record.get("matchup_features", []),
        "market_data": result_record.get("market_data", []),
        "qualifier_matches": result_record.get("qualifier_matches", []),
        "api_football": result_record.get("api_football", {}),
        "market_rows": int(result_record.get("market_rows", 0)),
        "qualifier_feature_rows": int(result_record.get("qualifier_feature_rows", 0)),
        "market_status": result_record.get("market_status", {}),
        "market_warnings": result_record.get("market_warnings", []),
        "api_football_status": result_record.get("api_football_status", {}),
        "api_football_warnings": result_record.get("api_football_warnings", []),
        "api_football_fixture_rows": int(result_record.get("api_football_fixture_rows", 0) or 0),
        "api_football_stat_rows": int(result_record.get("api_football_stat_rows", 0) or 0),
        "api_football_market_rows": int(result_record.get("api_football_market_rows", 0) or 0),
        "international_recent": result_record.get("international_recent", international_results_status()),
        "all_matches_rows": int(result_record.get("all_matches_rows", normalized.get("all_matches_rows", 0)) or 0),
        "worldcup_rows": int(result_record.get("worldcup_rows", normalized.get("worldcup_rows", 0)) or 0),
        "class_distribution": result_record.get("class_distribution", normalized.get("class_distribution", {})),
        "sample_weight_policy": result_record.get("sample_weight_policy", normalized.get("sample_weight_policy", SAMPLE_WEIGHT_POLICY)),
        "sample_weight_summary": result_record.get("sample_weight_summary", {}),
        "data_quality": result_record.get("data_quality", normalized.get("data_quality", {})),
        "dc_rho": float(result_record.get("dc_rho", 0.0) or 0.0),
        "source_files": result_record.get("source_files", normalized.get("source_files", [])),
        "kaggle_files": [],
        "history_source": result_record.get("history_source", ""),
        "metrics": result_record.get("metrics", {}),
        "confusion_matrix": result_record.get("confusion_matrix", {}),
        "calibration": result_record.get("calibration", {}),
        "classes": result_record.get("classes", []),
        "mode": normalized["training_mode"],
        "eval_strategy": result_record.get("eval_strategy", ""),
        "prediction_rows": int(normalized["team_prediction"].shape[0]),
        "effective_target": "result",
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
        "feature_inventory": result_record.get("feature_inventory", {}),
        "feature_cache": feature_cache.summary(),
        "derived_total_markets": derived_total_markets,
        "markets": market_results,
        "market_models": market_models,
        "over_under_preferred_sources": preferred_over_under_sources(market_results, goals_record),
        "trained_at": trained_at,
        "walk_forward_mode": result_record.get("walk_forward_mode", "none"),
        "walk_forward_summary": result_record.get("walk_forward_summary", {}),
        "prepared_schema_version": normalized.get("prepared_schema_version", ""),
        "target_worldcup_year": str(normalized.get("target_worldcup_year") or TARGET_WORLDCUP_YEAR),
        "benchmark_worldcup_year": normalized.get("benchmark_worldcup_year", normalized.get("final_test_year", "")),
        "benchmark_policy": normalized.get("benchmark_policy", BENCHMARK_POLICY),
        "label_policy_notes": normalized.get("label_policy_notes", []),
        "final_test_year": normalized.get("final_test_year", ""),
        "split_policy": normalized.get("split_policy", ""),
        "training_start_year": int(normalized.get("training_start_year", INTERNATIONAL_TRAINING_START_YEAR) or INTERNATIONAL_TRAINING_START_YEAR),
        "max_label_date": normalized.get("max_label_date", ""),
        "validation_rows": int(result_record.get("validation_rows", 0) or 0),
    }
    emit_training_progress(progress_callback, "saving", total_steps - 1, total_steps, "Guardando bundle de mercados", model_id=bundle_id)
    save_hybrid_model(bundle_record, model_id=bundle_id)
    model_meta = read_model_metadata(model_id=bundle_id)
    emit_training_progress(progress_callback, "complete", total_steps, total_steps, "Entrenamiento completado", model_id=bundle_id)
    return {
        "model": model_meta,
        "metrics": bundle_record["metrics"],
        "confusion_matrix": bundle_record["confusion_matrix"],
        "calibration": bundle_record["calibration"],
        "features": bundle_record["feature_columns"],
        "feature_inventory": bundle_record["feature_inventory"],
        "train_rows": int(result_result.get("train_rows", 0)),
        "validation_rows": int(result_result.get("validation_rows", 0)),
        "eval_rows": int(result_result.get("eval_rows", 0)),
        "source": normalized.get("label_source", INTERNATIONAL_DATASET_SLUG),
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
        "prepared_schema_version": bundle_record["prepared_schema_version"],
        "target_worldcup_year": bundle_record["target_worldcup_year"],
        "benchmark_worldcup_year": bundle_record["benchmark_worldcup_year"],
        "benchmark_policy": bundle_record["benchmark_policy"],
        "label_policy_notes": bundle_record["label_policy_notes"],
        "final_test_year": bundle_record["final_test_year"],
        "split_policy": bundle_record["split_policy"],
        "training_start_year": bundle_record["training_start_year"],
        "max_label_date": bundle_record["max_label_date"],
        "all_matches_rows": bundle_record["all_matches_rows"],
        "worldcup_rows": bundle_record["worldcup_rows"],
        "class_distribution": bundle_record["class_distribution"],
        "sample_weight_summary": bundle_record["sample_weight_summary"],
        "data_quality": bundle_record["data_quality"],
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
        poisson_recent_matches: int = 15,
) -> Dict[str, Any]:
    fixture = select_prediction_fixture(tournament, fixture_id=fixture_id, home=home, away=away)
    home_team = str(fixture.get("Equipo 1", home or ""))
    away_team = str(fixture.get("Equipo 2", away or ""))
    poisson = base_model.match_probabilities(home_team, away_team)
    base_probs = {"H": poisson["home"], "D": poisson["draw"], "A": poisson["away"]}
    base_totals = total_line_probabilities_from_probs(poisson)
    ml_outputs = {"result": {}, "over_under_ml": {}, "over_under_25": {}, "notes": ["Modelo internacional no entrenado."]}
    if use_ml_model:
        ml_outputs = predict_ml_outputs(base_model, home_team, away_team, model_id=model_id, fixture_id=fixture.get("No."))
    result_ml = ml_outputs.get("result", {})
    over_under_ml = ml_outputs.get("over_under_ml") or ml_outputs.get("over_under_25", {}) or {}
    result_weight = ml_weight if result_ml else 0.0
    totals_weight = ml_weight if over_under_ml else 0.0
    blended = blend_probabilities(base_probs, result_ml, result_weight)
    blended_totals = blend_total_probabilities(base_totals, over_under_ml, totals_weight)
    market_sources = market_sources_payload(result_ml, over_under_ml, ml_outputs)
    return_payload = {
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
            **{key: round(value * 100.0, 2) for key, value in blended_totals.items()},
        },
        "model_probs": {
            "poisson": {key: round(value * 100.0, 2) for key, value in base_probs.items()},
            "poisson_totals": {key: round(value * 100.0, 2) for key, value in base_totals.items()},
            "ml": {key: round(value * 100.0, 2) for key, value in result_ml.items()},
            "over_under_ml": {key: round(value * 100.0, 2) for key, value in over_under_ml.items()},
            "ml_weight": round(float(ml_weight if result_ml else 0.0), 3),
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
    return_payload["market_readout"] = prediction_market_readout(
        home=home_team,
        away=away_team,
        fixture=fixture.to_dict(),
        result_probs=blended,
        total_probs=blended_totals,
        model_id=model_id,
    )
    if return_payload["market_readout"].get("data_quality"):
        return_payload["data_quality"] = return_payload["market_readout"]["data_quality"]
    if ml_outputs.get("goal_distribution_ml"):
        return_payload["model_probs"]["goal_distribution_ml"] = {
            key: round(value * 100.0, 2)
            for key, value in (ml_outputs.get("goal_distribution_ml") or {}).items()
        }
    try:
        recent_match_limit = int(poisson_recent_matches or 15)
    except (TypeError, ValueError):
        recent_match_limit = 15
    recent_match_limit = max(3, min(50, recent_match_limit))
    return_payload["contextual_poisson"] = contextual_poisson_for_match(
        home_team,
        away_team,
        base_model=base_model,
        before_date=fixture.get("Fecha", HISTORY_REFERENCE_DATE),
        max_goals=base_model.max_goals,
        limit=recent_match_limit,
    )
    return return_payload


def prediction_market_readout(
        home: str,
        away: str,
        fixture: Dict[str, Any],
        result_probs: Dict[str, float],
        total_probs: Dict[str, float],
        model_id: Optional[str] = None,
) -> Dict[str, Any]:
    record = load_hybrid_model(model_id=model_id)
    if not record:
        return {"available": False, "missing_sources": ["modelo", "odds"], "lines": [], "data_quality": {}}
    market_rows = pd.DataFrame(record.get("market_data", []))
    match_date = fixture.get("Fecha") or fixture.get("date")
    market_match = market_for_match(
        market_rows,
        home,
        away,
        match_date=match_date,
        year=2026,
    )
    features = build_market_feature_row(
        market_match,
        model_probs=result_probs,
        model_totals=total_probs,
    )
    lines: List[Dict[str, Any]] = []
    if features.get("market_has_1x2"):
        for label, key, market_key in (
            ("1", "H", "market_prob_home"),
            ("X", "D", "market_prob_draw"),
            ("2", "A", "market_prob_away"),
        ):
            model_probability = float(result_probs.get(key, 0.0))
            market_probability = float(features.get(market_key, 0.0))
            lines.append({
                "market": "1X2",
                "label": label,
                "model_probability": round(model_probability * 100.0, 2),
                "market_probability": round(market_probability * 100.0, 2),
                "raw_edge": round((model_probability - market_probability) * 100.0, 2),
            })
    for line in TRAIN_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        if not features.get(f"market_has_ou{suffix}"):
            continue
        for side, key in (("Over", f"over{suffix}"), ("Under", f"under{suffix}")):
            market_key = f"market_prob_{key}"
            model_probability = float(total_probs.get(key, 0.0))
            market_probability = float(features.get(market_key, 0.0))
            lines.append({
                "market": f"U/O {line:.1f}",
                "label": side,
                "model_probability": round(model_probability * 100.0, 2),
                "market_probability": round(market_probability * 100.0, 2),
                "raw_edge": round((model_probability - market_probability) * 100.0, 2),
            })
    data_quality = record.get("data_quality", {}) or {}
    missing = list(data_quality.get("missing_sources", []))
    if not lines and "odds" not in missing:
        missing.append("odds")
    return {
        "available": bool(lines),
        "lines": lines,
        "missing_sources": unique_strings(missing),
        "data_quality": data_quality,
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
    market_bundle = load_market_data(force_download=bool(refresh_history), allow_download=bool(refresh_history), use_scraper=False)
    market_data = market_bundle.get("matches", pd.DataFrame()).copy()
    qualifier_matches = market_bundle.get("qualifiers", pd.DataFrame()).copy()
    api_football_bundle = load_api_football_data(force_download=bool(refresh_history), allow_download=bool(refresh_history))
    api_market_rows = api_football_bundle.get("market_rows", pd.DataFrame()).copy()
    try:
        international_matches = load_international_matches(required=True)
    except Exception as exc:
        raise WorldCupTrainingError(f"all_matches.csv es obligatorio para labels: {exc}") from exc
    international_status = international_results_status()
    all_matches_label_rows = international_match_rows(international_matches)
    if not api_market_rows.empty:
        market_data = pd.concat([market_data, api_market_rows], ignore_index=True) if not market_data.empty else api_market_rows
    combined_has_1x2 = bool(not market_data.empty and market_data[["market_odds_home", "market_odds_draw", "market_odds_away"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).any()) if {"market_odds_home", "market_odds_draw", "market_odds_away"}.issubset(market_data.columns) else False
    combined_has_ou25 = bool(not market_data.empty and market_data[["market_odds_over25", "market_odds_under25"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).any()) if {"market_odds_over25", "market_odds_under25"}.issubset(market_data.columns) else False
    warnings: List[str] = []
    label_policy_notes: List[str] = [
        f"Objetivo operativo: Mundial {TARGET_WORLDCUP_YEAR}.",
        "Labels obligatorios desde all_matches.csv; train.csv/test.csv y el Kaggle Mundial legacy se ignoran por completo.",
        f"Corpus entrenable: partidos internacionales desde {INTERNATIONAL_TRAINING_START_YEAR}, incluyendo FIFA World Cup y resultados 2026 ya jugados.",
        "Anti-leakage: split temporal con test final en los ultimos 30 partidos y validacion inmediatamente anterior.",
    ]
    label_source = "all_matches.csv"
    raw_mode = "international_only"
    team_features = pd.DataFrame()

    if international_status.get("warning"):
        warnings.append(str(international_status.get("warning")))
    if all_matches_label_rows.empty:
        raise WorldCupTrainingError("all_matches.csv no contiene partidos internacionales validos con marcador para construir labels.")

    labeled_rows = deduplicate_labeled_matches(sanitize_match_rows(all_matches_label_rows))
    labeled_rows, scope_stats = filter_international_training_scope(labeled_rows)
    if scope_stats["removed_before_start"]:
        label_policy_notes.append(
            f"{scope_stats['removed_before_start']} partidos anteriores a {INTERNATIONAL_TRAINING_START_YEAR} quedaron fuera del entrenamiento."
        )
    if scope_stats["removed_future"]:
        label_policy_notes.append(
            f"{scope_stats['removed_future']} partidos posteriores a {scope_stats['max_label_date']} quedaron fuera por fecha futura."
        )
    if labeled_rows.empty:
        raise WorldCupTrainingError(f"all_matches.csv no contiene partidos internacionales entrenables desde {INTERNATIONAL_TRAINING_START_YEAR}.")
    worldcup_rows_count = int(labeled_rows["is_worldcup_match"].map(coerce_bool_value).sum()) if "is_worldcup_match" in labeled_rows.columns else 0
    train_df, validation_df, test_df, split_warning = split_validation_last_30_international_test(labeled_rows)
    if split_warning:
        label_policy_notes.append(split_warning)
    over_under_ready = has_over_under_target(train_df)
    if not over_under_ready:
        raise WorldCupTrainingError("El ETL no pudo construir targets reales de U/O multi-linea con goles observados.")
    goals_distribution_ready = has_goals_distribution_target(train_df)

    prepared_at = datetime.now(timezone.utc).isoformat()
    preview_source = train_df if not train_df.empty else test_df if not test_df.empty else team_features
    dc_rho = estimate_dixon_coles_rho(history_df)
    class_distribution = split_class_distribution(train_df, test_df, validation_df=validation_df)
    all_matches_rows_count = int(all_matches_label_rows.shape[0])
    market_status_payload = {
        "status": "ok" if combined_has_1x2 or combined_has_ou25 else market_bundle.get("status", "missing"),
        "has_1x2": combined_has_1x2,
        "has_ou25": combined_has_ou25,
        "sources": [*market_bundle.get("sources", []), *api_football_bundle.get("sources", [])],
        "loaded_at": market_bundle.get("loaded_at", ""),
    }
    api_football_payload = {
        "fixtures": api_football_bundle.get("fixtures", pd.DataFrame()),
        "statistics": api_football_bundle.get("statistics", pd.DataFrame()),
        "team_stats": api_football_bundle.get("team_stats", pd.DataFrame()),
        "lineups": api_football_bundle.get("lineups", pd.DataFrame()),
        "injuries": api_football_bundle.get("injuries", pd.DataFrame()),
        "odds": api_football_bundle.get("odds", pd.DataFrame()),
        "market_rows": api_market_rows,
    }
    data_quality = data_quality_payload(
        prepared_at=prepared_at,
        international_status=international_status,
        all_matches_rows=all_matches_rows_count,
        worldcup_rows=worldcup_rows_count,
        market_rows=market_data,
        market_status=market_status_payload,
        api_football=api_football_payload,
        api_status={
            "status": api_football_bundle.get("status", "missing"),
            "sources": api_football_bundle.get("sources", []),
            "loaded_at": api_football_bundle.get("loaded_at", ""),
        },
    )
    return {
        "prepared_schema_version": PREPARED_SCHEMA_VERSION,
        "prepared_at": prepared_at,
        "source_files": prepared_source_files(files, international_status),
        "source_mode": raw_mode,
        "training_mode": "match_result",
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
        "team_train": pd.DataFrame(),
        "team_test": pd.DataFrame(),
        "team_prediction": pd.DataFrame(),
        "team_features": team_features,
        "market_data": market_data,
        "qualifier_matches": qualifier_matches,
        "market_rows": int(market_data.shape[0]),
        "qualifier_feature_rows": int(market_bundle.get("qualifier_rows", 0)),
        "market_status": market_status_payload,
        "market_warnings": market_bundle.get("warnings", []),
        "api_football": api_football_payload,
        "api_football_status": {
            "status": api_football_bundle.get("status", "missing"),
            "sources": api_football_bundle.get("sources", []),
            "loaded_at": api_football_bundle.get("loaded_at", ""),
        },
        "api_football_warnings": api_football_bundle.get("warnings", []),
        "api_football_fixture_rows": int(api_football_bundle.get("fixtures", pd.DataFrame()).shape[0]),
        "api_football_stat_rows": int(api_football_bundle.get("team_stats", pd.DataFrame()).shape[0]),
        "api_football_market_rows": int(api_market_rows.shape[0]),
        "international_recent": international_status,
        "all_matches_rows": all_matches_rows_count,
        "worldcup_rows": worldcup_rows_count,
        "class_distribution": class_distribution,
        "sample_weight_policy": SAMPLE_WEIGHT_POLICY,
        "data_quality": data_quality,
        "training_start_year": INTERNATIONAL_TRAINING_START_YEAR,
        "max_label_date": scope_stats["max_label_date"],
        "removed_before_start_rows": scope_stats["removed_before_start"],
        "removed_future_rows": scope_stats["removed_future"],
        "dc_rho": float(dc_rho),
        "target_column": "Label + GoalsDistribution + OverUnder05/15/25/35",
        "team_columns": [],
        "trainable": bool(not train_df.empty and train_df["Label"].isin(TARGET_LABELS).any()),
        "preview": preview_payload(preview_source),
        "warnings": unique_strings(warnings),
        "label_policy_notes": unique_strings(label_policy_notes),
        "label_source": label_source,
        "history_source": history_source,
        "target_worldcup_year": str(TARGET_WORLDCUP_YEAR),
        "benchmark_worldcup_year": "",
        "benchmark_policy": BENCHMARK_POLICY,
        "final_test_year": "",
        "split_policy": SPLIT_POLICY_VALIDATION_LAST_30,
        "over_under_ready": over_under_ready,
        "goals_distribution_ready": goals_distribution_ready,
        "result_ready": bool(not train_df.empty and train_df["Label"].isin(TARGET_LABELS).any()),
    }


def filter_international_training_scope(
        rows: pd.DataFrame,
        start_year: int = INTERNATIONAL_TRAINING_START_YEAR,
        max_date: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if rows.empty:
        return sanitize_match_rows(rows), {
            "removed_before_start": 0,
            "removed_future": 0,
            "max_label_date": str(max_date or datetime.now(timezone.utc).date()),
        }
    working = sort_match_rows(sanitize_match_rows(rows))
    date_values = pd.to_datetime(working.get("Date"), errors="coerce")
    start_ts = pd.Timestamp(f"{int(start_year)}-01-01")
    max_ts = pd.Timestamp(max_date if max_date is not None else datetime.now(timezone.utc).date())
    if max_ts.tzinfo is not None:
        max_ts = max_ts.tz_convert(None)
    date_only = date_values.dt.normalize()
    before_start = date_only.notna() & date_only.lt(start_ts)
    future = date_only.notna() & date_only.gt(max_ts.normalize())
    keep = date_only.notna() & ~before_start & ~future
    filtered = sort_match_rows(working[keep].copy()).reset_index(drop=True)
    return filtered, {
        "removed_before_start": int(before_start.sum()),
        "removed_future": int(future.sum()),
        "max_label_date": max_ts.date().isoformat(),
    }


def split_last_30_international_test(rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if rows.empty:
        raise WorldCupTrainingError("all_matches.csv no contiene partidos internacionales validos para train/test.")
    working = sort_match_rows(sanitize_match_rows(rows))
    date_values = pd.to_datetime(working.get("Date"), errors="coerce")
    valid = working[date_values.notna()].copy()
    valid = sort_match_rows(valid).reset_index(drop=True)
    if valid.shape[0] < 31:
        raise WorldCupTrainingError(
            f"all_matches.csv debe contener al menos 31 partidos internacionales validos; encontro {valid.shape[0]}."
        )
    train = valid.iloc[:-30].copy().reset_index(drop=True)
    test = valid.iloc[-30:].copy().reset_index(drop=True)
    worldcup_rows = int(valid["is_worldcup_match"].map(coerce_bool_value).sum()) if "is_worldcup_match" in valid.columns else 0
    warning = (
        f"{worldcup_rows} partidos FIFA World Cup incluidos en el split temporal."
        if worldcup_rows
        else ""
    )
    return train, test, warning


def split_validation_last_30_international_test(
        rows: pd.DataFrame,
        validation_fraction: float = 0.10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if rows.empty:
        raise WorldCupTrainingError("all_matches.csv no contiene partidos internacionales validos para train/validacion/test.")
    working = sort_match_rows(sanitize_match_rows(rows))
    date_values = pd.to_datetime(working.get("Date"), errors="coerce")
    valid = sort_match_rows(working[date_values.notna()].copy()).reset_index(drop=True)
    if valid.shape[0] < 32:
        raise WorldCupTrainingError(
            f"all_matches.csv debe contener al menos 32 partidos internacionales validos para train/validacion/test; encontro {valid.shape[0]}."
        )
    pre_test_rows = int(valid.shape[0] - 30)
    if pre_test_rows < 2:
        raise WorldCupTrainingError(
            f"all_matches.csv debe dejar al menos 2 partidos antes de los ultimos 30 para train/validacion; encontro {pre_test_rows}."
        )
    validation_size = max(1, int(math.ceil(pre_test_rows * float(validation_fraction))))
    validation_size = min(validation_size, pre_test_rows - 1)
    train_end = int(valid.shape[0] - 30 - validation_size)
    validation_end = int(valid.shape[0] - 30)
    train = valid.iloc[:train_end].copy().reset_index(drop=True)
    validation = valid.iloc[train_end:validation_end].copy().reset_index(drop=True)
    test = valid.iloc[-30:].copy().reset_index(drop=True)
    worldcup_rows = int(valid["is_worldcup_match"].map(coerce_bool_value).sum()) if "is_worldcup_match" in valid.columns else 0
    warning = (
        f"Split temporal desde {INTERNATIONAL_TRAINING_START_YEAR}: train={train.shape[0]}, validacion={validation.shape[0]}, test=30; {worldcup_rows} partidos FIFA World Cup incluidos."
    )
    return train, validation, test, warning


def split_latest_worldcup_test(rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    if rows.empty:
        return rows.copy(), pd.DataFrame(columns=[*MATCH_ROW_COLUMNS, *MATCH_METADATA_COLUMNS]), "", ""
    working = sort_match_rows(rows)
    years = pd.to_numeric(working.get("Year"), errors="coerce")
    if "is_worldcup_match" in working.columns:
        worldcup_mask = working["is_worldcup_match"].map(coerce_bool_value)
    else:
        worldcup_mask = working.get("tournament", pd.Series(index=working.index, dtype=object)).map(is_worldcup_tournament)
    valid_years = sorted({int(year) for year in years[worldcup_mask].dropna().tolist() if int(year) < FUTURE_LABEL_EXCLUDED_YEAR})
    if len(valid_years) < 2:
        return working.reset_index(drop=True), pd.DataFrame(columns=working.columns), "", "No hay al menos dos Mundiales FIFA masculinos completos etiquetados; se usara holdout temporal interno desde train."
    final_year = int(valid_years[-1])
    test_mask = worldcup_mask & years.eq(final_year)
    test = working[test_mask].copy()
    test_dates = pd.to_datetime(test.get("Date"), errors="coerce")
    first_test_date = test_dates.min() if not test.empty else pd.NaT
    row_dates = pd.to_datetime(working.get("Date"), errors="coerce")
    if pd.notna(first_test_date):
        train_mask = (~test_mask) & (row_dates.notna() & row_dates.lt(first_test_date))
        post_cutoff = int(((~test_mask) & row_dates.notna() & row_dates.ge(first_test_date)).sum())
    else:
        train_mask = (~test_mask) & years.lt(final_year)
        post_cutoff = int(((~test_mask) & years.ge(final_year)).sum())
    train = working[train_mask].copy()
    if train.empty or test.empty:
        return working.reset_index(drop=True), pd.DataFrame(columns=working.columns), "", "No se pudo aislar el ultimo Mundial FIFA masculino completo como benchmark; se usara holdout temporal interno desde train."
    warning = ""
    if post_cutoff:
        warning = f"{post_cutoff} partidos posteriores al inicio del Mundial {final_year} quedaron fuera de train/eval y solo sirven como contexto temporal."
    return train.reset_index(drop=True), test.reset_index(drop=True), str(final_year), warning


def sanitize_match_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=[*MATCH_ROW_COLUMNS, *MATCH_METADATA_COLUMNS])
    working = rows.copy()
    required = ["Home", "Away", "Label", "Source"]
    for column in required:
        if column not in working.columns:
            working[column] = ""
    for column in MATCH_METADATA_COLUMNS:
        if column not in working.columns:
            if column == "is_worldcup_match":
                working[column] = False
            elif column == "sample_weight":
                working[column] = 1.0
            elif column == "knockout":
                working[column] = False
            else:
                working[column] = ""
    if "FixtureId" not in working.columns:
        working["FixtureId"] = ""
    if "HG" not in working.columns:
        working["HG"] = np.nan
    if "AG" not in working.columns:
        working["AG"] = np.nan
    for suffix in TOTAL_GOAL_LINE_SUFFIXES:
        column = f"OverUnder{suffix}"
        if column not in working.columns:
            working[column] = np.nan
    if "Date" not in working.columns:
        working["Date"] = pd.NaT
    if "Year" not in working.columns:
        working["Year"] = np.nan
    working["Home"] = working["Home"].map(clean_team_name)
    working["Away"] = working["Away"].map(clean_team_name)
    working["Label"] = working["Label"].astype(str)
    working["HG"] = pd.to_numeric(working["HG"], errors="coerce")
    working["AG"] = pd.to_numeric(working["AG"], errors="coerce")
    goals_total = working["HG"] + working["AG"]
    for line in TRAIN_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        column = f"OverUnder{suffix}"
        needs_over = working[column].isna() & working["HG"].notna() & working["AG"].notna()
        if needs_over.any():
            working.loc[needs_over, column] = (goals_total.loc[needs_over] > line).astype(int)
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    inferred_year = working["Date"].dt.year
    working["Year"] = pd.to_numeric(working["Year"], errors="coerce").fillna(inferred_year)
    working["FixtureId"] = working["FixtureId"].astype(str)
    working["tournament"] = working["tournament"].astype(str)
    working["stage"] = working["stage"].astype(str)
    working["group"] = working["group"].astype(str)
    working["label_source"] = working["label_source"].astype(str)
    working["is_worldcup_match"] = working["is_worldcup_match"].map(coerce_bool_value)
    inferred_worldcup = working["tournament"].map(is_worldcup_tournament)
    working["is_worldcup_match"] = working["is_worldcup_match"] | inferred_worldcup
    working["knockout"] = working["knockout"].map(coerce_bool_value)
    working["sample_weight"] = pd.to_numeric(working["sample_weight"], errors="coerce")
    missing_weight = working["sample_weight"].isna()
    if missing_weight.any():
        working.loc[missing_weight, "sample_weight"] = working.loc[missing_weight].apply(
            lambda item: sample_weight_for_tournament(item.get("tournament"), is_worldcup=bool(item.get("is_worldcup_match"))),
            axis=1,
        )
    working = working[
        working["Home"].astype(str).str.len().gt(1) &
        working["Away"].astype(str).str.len().gt(1) &
        working["Label"].isin(TARGET_LABELS)
    ].copy()
    for column in MATCH_ROW_COLUMNS:
        if column not in working.columns:
            working[column] = np.nan
    ordered_columns = [*MATCH_ROW_COLUMNS, *MATCH_METADATA_COLUMNS]
    return sort_match_rows(working[ordered_columns + [column for column in working.columns if column not in ordered_columns]]).reset_index(drop=True)


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
        return pd.DataFrame(columns=[*MATCH_ROW_COLUMNS, *MATCH_METADATA_COLUMNS])
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
        stage = str(row.get("Round", "") or "")
        group = str(row.get("Group", "") or "")
        tournament = "FIFA World Cup"
        rows.append({
            "FixtureId": str(row.get("FixtureId", row.get("No.", index))),
            "Date": row.get("Date"),
            "Year": row.get("Year", pd.Timestamp(row.get("Date")).year if pd.notna(row.get("Date")) else np.nan),
            "Home": home,
            "Away": away,
            "Label": label_from_goals(goals_home, goals_away),
            "HG": goals_home,
            "AG": goals_away,
            **over_under_target_values(goals_home, goals_away),
            "Source": source,
            "is_worldcup_match": True,
            "tournament": tournament,
            "stage": stage,
            "group": group,
            "knockout": bool(stage and not group),
            "label_source": "historical_worldcup",
            "sample_weight": sample_weight_for_tournament(tournament, is_worldcup=True),
        })
    return sanitize_match_rows(pd.DataFrame(rows))


def international_match_rows(matches: pd.DataFrame) -> pd.DataFrame:
    if matches is None or matches.empty:
        return pd.DataFrame(columns=[*MATCH_ROW_COLUMNS, *MATCH_METADATA_COLUMNS])
    working = matches.copy()
    working["date"] = pd.to_datetime(working.get("date"), errors="coerce")
    working["home_score"] = pd.to_numeric(working.get("home_score"), errors="coerce")
    working["away_score"] = pd.to_numeric(working.get("away_score"), errors="coerce")
    working = working[
        working["date"].notna()
        & working.get("home_team", pd.Series(index=working.index, dtype=object)).astype(str).str.len().gt(1)
        & working.get("away_team", pd.Series(index=working.index, dtype=object)).astype(str).str.len().gt(1)
        & working["home_score"].notna()
        & working["away_score"].notna()
    ].copy()
    if working.empty:
        return pd.DataFrame(columns=[*MATCH_ROW_COLUMNS, *MATCH_METADATA_COLUMNS])

    rows: List[Dict[str, Any]] = []
    for index, row in working.iterrows():
        goals_home = float(row.get("home_score"))
        goals_away = float(row.get("away_score"))
        tournament = str(row.get("tournament", "") or "")
        worldcup = is_worldcup_tournament(tournament)
        match_date = pd.Timestamp(row.get("date"))
        rows.append({
            "FixtureId": str(row.get("fixture_id", index)),
            "Date": match_date,
            "Year": int(match_date.year),
            "Home": clean_team_name(row.get("home_team")),
            "Away": clean_team_name(row.get("away_team")),
            "Label": label_from_goals(goals_home, goals_away),
            "HG": goals_home,
            "AG": goals_away,
            **over_under_target_values(goals_home, goals_away),
            "Source": "all_matches.csv",
            "is_worldcup_match": bool(worldcup),
            "tournament": tournament,
            "stage": "",
            "group": "",
            "knockout": False,
            "label_source": "all_matches.csv",
            "sample_weight": sample_weight_for_tournament(tournament, is_worldcup=worldcup),
        })
    return sanitize_match_rows(pd.DataFrame(rows))


def deduplicate_labeled_matches(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return sanitize_match_rows(rows)
    working = sanitize_match_rows(rows)
    if working.empty:
        return working
    working["_date_key"] = pd.to_datetime(working["Date"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    working["_home_key"] = working["Home"].astype(str).str.lower().str.strip()
    working["_away_key"] = working["Away"].astype(str).str.lower().str.strip()
    working["_hg_key"] = pd.to_numeric(working["HG"], errors="coerce").round(3).astype(str)
    working["_ag_key"] = pd.to_numeric(working["AG"], errors="coerce").round(3).astype(str)
    worldcup_mask = working["is_worldcup_match"].astype(bool)
    historical_worldcup_mask = worldcup_mask & (
        working.get("label_source", "").astype(str).str.contains("historical_worldcup", case=False, na=False)
        | working.get("Source", "").astype(str).str.contains("worldcup", case=False, na=False)
    )
    working["_source_priority"] = np.select(
        [historical_worldcup_mask, worldcup_mask],
        [0, 1],
        default=2,
    )
    working["_row_order"] = np.arange(len(working), dtype=int)
    working = working.sort_values(
        ["_date_key", "_home_key", "_away_key", "_hg_key", "_ag_key", "_source_priority", "_row_order"],
        kind="stable",
    )
    working = working.drop_duplicates(["_date_key", "_home_key", "_away_key", "_hg_key", "_ag_key"], keep="first")
    return sort_match_rows(working.drop(columns=["_date_key", "_home_key", "_away_key", "_hg_key", "_ag_key", "_source_priority", "_row_order"]))


def drop_future_label_rows(rows: pd.DataFrame, cutoff_year: int = FUTURE_LABEL_EXCLUDED_YEAR) -> Tuple[pd.DataFrame, int]:
    if rows.empty or "Year" not in rows.columns:
        return rows.copy(), 0
    years = pd.to_numeric(rows["Year"], errors="coerce")
    keep = years.isna() | years.lt(int(cutoff_year))
    removed = int((~keep).sum())
    return rows[keep].copy().reset_index(drop=True), removed


def sample_weight_for_tournament(tournament: Any, is_worldcup: bool = False) -> float:
    text = str(tournament or "")
    normalized = normalize_text_key(text)
    if is_worldcup:
        return SAMPLE_WEIGHT_POLICY["worldcup"]
    if "friendly" in normalized:
        return SAMPLE_WEIGHT_POLICY["friendly"]
    if any(token in normalized for token in ("qualification", "qualifier", "qualifiers")):
        return SAMPLE_WEIGHT_POLICY["qualifier"]
    if "nations league" in normalized:
        return SAMPLE_WEIGHT_POLICY["nations_league"]
    weighted = tournament_weight(text)
    if weighted >= 1.25:
        return SAMPLE_WEIGHT_POLICY["continental_or_world_official"]
    if weighted >= 1.0:
        return SAMPLE_WEIGHT_POLICY["other_official"]
    return float(weighted)


def normalize_text_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def coerce_bool_value(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "si", "sí"}


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


def prepared_source_files(files: Iterable[Path], international_status: Optional[Dict[str, Any]] = None) -> List[str]:
    status = international_status or {}
    paths: List[str] = []
    source_path = str(status.get("source_path") or "")
    if status.get("available") and source_path:
        paths.append(source_path)
    elif INTERNATIONAL_MATCHES_FILE.exists():
        paths.append(str(INTERNATIONAL_MATCHES_FILE))
    return unique_strings(paths)


def split_class_distribution(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        validation_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    return {
        "train": class_distribution_for_rows(train_df),
        "validation": class_distribution_for_rows(validation_df if validation_df is not None else pd.DataFrame()),
        "test": class_distribution_for_rows(test_df),
    }


def class_distribution_for_rows(rows: pd.DataFrame) -> Dict[str, Any]:
    output: Dict[str, Any] = {"rows": int(rows.shape[0]) if isinstance(rows, pd.DataFrame) else 0, "markets": {}}
    if rows is None or rows.empty:
        return output
    if "Label" in rows.columns:
        output["markets"]["result"] = value_counts_payload(rows["Label"])
    for target in OVER_UNDER_MARKET_TARGETS:
        column = over_under_column_for_target(target)
        if column in rows.columns:
            output["markets"][target] = value_counts_payload(pd.to_numeric(rows[column], errors="coerce").dropna().astype(int))
    if {"HG", "AG"}.issubset(rows.columns):
        output["markets"][GOALS_DISTRIBUTION_TARGET] = value_counts_payload(total_goals_buckets(rows))
    return output


def value_counts_payload(values: Iterable[Any]) -> Dict[str, int]:
    series = pd.Series(values).dropna()
    counts = series.astype(str).value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


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
            "prepared_schema_version": "",
            "target_worldcup_year": str(TARGET_WORLDCUP_YEAR),
            "benchmark_worldcup_year": "",
            "benchmark_policy": BENCHMARK_POLICY,
            "label_policy_notes": [],
            "final_test_year": "",
            "split_policy": "",
            "over_under_ready": False,
            "goals_distribution_ready": False,
            "all_matches_rows": 0,
            "worldcup_rows": 0,
            "class_distribution": {},
            "sample_weight_policy": SAMPLE_WEIGHT_POLICY,
            "data_quality": {},
            "training_start_year": INTERNATIONAL_TRAINING_START_YEAR,
            "max_label_date": "",
            "warnings": [],
        }
    current_international_status = international_results_status()
    source_files = set(prepared_source_files(files, current_international_status))
    artifact_sources = set(dataset.get("source_files", []))
    artifact_time = PREPARED_DATASET_FILE.stat().st_mtime if PREPARED_DATASET_FILE.exists() else 0.0
    source_times = [Path(path).stat().st_mtime for path in source_files if Path(path).exists()]
    schema_version = str(dataset.get("prepared_schema_version") or "")
    schema_stale = schema_version != PREPARED_SCHEMA_VERSION or not prepared_dataset_schema_valid(dataset)
    stale = bool(source_times and max(source_times) > artifact_time) or bool(source_files != artifact_sources) or schema_stale
    status_dataset = normalized if schema_stale else dataset
    return {
        "ready": True,
        "stale": stale,
        "status": "stale" if stale else "ready",
        "dataset": status_dataset,
        "prepared_at": str(dataset.get("prepared_at") or ""),
        "mode": str(dataset.get("training_mode") or ""),
        "label_source": str(dataset.get("label_source") or ""),
        "prepared_schema_version": schema_version,
        "target_worldcup_year": str(dataset.get("target_worldcup_year") or TARGET_WORLDCUP_YEAR),
        "benchmark_worldcup_year": "" if schema_stale else str(dataset.get("benchmark_worldcup_year") or dataset.get("final_test_year") or ""),
        "benchmark_policy": str(dataset.get("benchmark_policy") or BENCHMARK_POLICY),
        "label_policy_notes": [] if schema_stale else dataset.get("label_policy_notes", []),
        "final_test_year": "" if schema_stale else str(dataset.get("final_test_year") or ""),
        "split_policy": "" if schema_stale else str(dataset.get("split_policy") or ""),
        "over_under_ready": bool(dataset.get("over_under_ready", False)),
        "goals_distribution_ready": bool(dataset.get("goals_distribution_ready", dataset.get("over_under_ready", False))),
        "market_rows": int(dataset.get("market_rows", 0)),
        "qualifier_feature_rows": int(dataset.get("qualifier_feature_rows", 0)),
        "market_status": dataset.get("market_status", {}),
        "market_warnings": dataset.get("market_warnings", []),
        "api_football_status": dataset.get("api_football_status", {}),
        "api_football_warnings": dataset.get("api_football_warnings", []),
        "api_football_fixture_rows": int(dataset.get("api_football_fixture_rows", 0)),
        "api_football_stat_rows": int(dataset.get("api_football_stat_rows", 0)),
        "api_football_market_rows": int(dataset.get("api_football_market_rows", 0)),
        "international_recent": current_international_status,
        "all_matches_rows": int(dataset.get("all_matches_rows", 0)),
        "worldcup_rows": int(dataset.get("worldcup_rows", 0)),
        "class_distribution": dataset.get("class_distribution", {}),
        "sample_weight_policy": dataset.get("sample_weight_policy", SAMPLE_WEIGHT_POLICY),
        "data_quality": {} if schema_stale else dataset.get("data_quality", {}),
        "training_start_year": int(dataset.get("training_start_year", INTERNATIONAL_TRAINING_START_YEAR) or INTERNATIONAL_TRAINING_START_YEAR),
        "max_label_date": "" if schema_stale else str(dataset.get("max_label_date") or ""),
        "warnings": [] if schema_stale else dataset.get("warnings", []),
    }


def prepared_dataset_metadata(dataset: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prepared_schema_version": dataset.get("prepared_schema_version", ""),
        "prepared_at": dataset.get("prepared_at", ""),
        "training_mode": dataset.get("training_mode", ""),
        "source_mode": dataset.get("source_mode", ""),
        "source_files": dataset.get("source_files", []),
        "label_source": dataset.get("label_source", ""),
        "warnings": dataset.get("warnings", []),
        "label_policy_notes": dataset.get("label_policy_notes", []),
        "target_worldcup_year": str(dataset.get("target_worldcup_year") or TARGET_WORLDCUP_YEAR),
        "benchmark_worldcup_year": dataset.get("benchmark_worldcup_year", dataset.get("final_test_year", "")),
        "benchmark_policy": dataset.get("benchmark_policy", BENCHMARK_POLICY),
        "target_column": dataset.get("target_column", ""),
        "team_columns": dataset.get("team_columns", []),
        "train_rows": labeled_train_row_count(dataset),
        "validation_rows": labeled_validation_row_count(dataset),
        "test_rows": labeled_test_row_count(dataset),
        "prediction_rows": int(dataset.get("team_prediction", pd.DataFrame()).shape[0]),
        "team_feature_rows": int(dataset.get("team_features", pd.DataFrame()).shape[0]),
        "market_rows": int(dataset.get("market_rows", 0)),
        "qualifier_feature_rows": int(dataset.get("qualifier_feature_rows", 0)),
        "market_status": dataset.get("market_status", {}),
        "market_warnings": dataset.get("market_warnings", []),
        "api_football_status": dataset.get("api_football_status", {}),
        "api_football_warnings": dataset.get("api_football_warnings", []),
        "api_football_fixture_rows": int(dataset.get("api_football_fixture_rows", 0)),
        "api_football_stat_rows": int(dataset.get("api_football_stat_rows", 0)),
        "api_football_market_rows": int(dataset.get("api_football_market_rows", 0)),
        "international_recent": dataset.get("international_recent", international_results_status()),
        "all_matches_rows": int(dataset.get("all_matches_rows", 0)),
        "worldcup_rows": int(dataset.get("worldcup_rows", 0)),
        "class_distribution": dataset.get("class_distribution", {}),
        "sample_weight_policy": dataset.get("sample_weight_policy", SAMPLE_WEIGHT_POLICY),
        "data_quality": dataset.get("data_quality", {}),
        "training_start_year": int(dataset.get("training_start_year", INTERNATIONAL_TRAINING_START_YEAR) or INTERNATIONAL_TRAINING_START_YEAR),
        "max_label_date": dataset.get("max_label_date", ""),
        "removed_before_start_rows": int(dataset.get("removed_before_start_rows", 0) or 0),
        "removed_future_rows": int(dataset.get("removed_future_rows", 0) or 0),
        "dc_rho": float(dataset.get("dc_rho", 0.0) or 0.0),
        "over_under_ready": bool(dataset.get("over_under_ready", False)),
        "goals_distribution_ready": bool(dataset.get("goals_distribution_ready", dataset.get("over_under_ready", False))),
        "result_ready": bool(dataset.get("result_ready", False)),
        "preview": dataset.get("preview", {"columns": [], "rows": [], "total": 0}),
        "history_source": dataset.get("history_source", ""),
        "final_test_year": dataset.get("final_test_year", ""),
        "split_policy": dataset.get("split_policy", ""),
    }


def data_quality_payload(
        prepared_at: str,
        international_status: Dict[str, Any],
        all_matches_rows: int,
        worldcup_rows: int,
        market_rows: pd.DataFrame,
        market_status: Dict[str, Any],
        api_football: Dict[str, pd.DataFrame],
        api_status: Dict[str, Any],
) -> Dict[str, Any]:
    api_football = api_football or {}
    market_rows = market_rows if isinstance(market_rows, pd.DataFrame) else pd.DataFrame()
    api_counts = {
        key: int(value.shape[0]) if isinstance(value, pd.DataFrame) else 0
        for key, value in api_football.items()
    }
    sources = {
        "all_matches": {
            "available": bool(international_status.get("available")),
            "rows": int(all_matches_rows or international_status.get("rows", 0) or 0),
            "worldcup_rows": int(worldcup_rows or international_status.get("worldcup_rows", 0) or 0),
            "updated_at": international_status.get("updated_at") or international_status.get("loaded_at") or "",
            "source_path": international_status.get("source_path") or international_status.get("file_path") or "",
        },
        "odds": {
            "available": bool((market_status or {}).get("has_1x2") or (market_status or {}).get("has_ou25")),
            "rows": int(market_rows.shape[0]),
            "has_1x2": bool((market_status or {}).get("has_1x2")),
            "has_ou25": bool((market_status or {}).get("has_ou25")),
            "updated_at": (market_status or {}).get("loaded_at", ""),
            "sources": (market_status or {}).get("sources", []),
        },
        "api_football": {
            "available": str((api_status or {}).get("status", "")).lower() == "ok" or any(api_counts.values()),
            "status": (api_status or {}).get("status", "missing"),
            "fixtures": api_counts.get("fixtures", 0),
            "team_stats": api_counts.get("team_stats", 0),
            "odds": api_counts.get("odds", 0),
            "market_rows": api_counts.get("market_rows", 0),
            "updated_at": (api_status or {}).get("loaded_at", ""),
            "sources": (api_status or {}).get("sources", []),
        },
        "lineups": {
            "available": api_counts.get("lineups", 0) > 0,
            "rows": api_counts.get("lineups", 0),
            "updated_at": (api_status or {}).get("loaded_at", ""),
        },
        "injuries": {
            "available": api_counts.get("injuries", 0) > 0,
            "rows": api_counts.get("injuries", 0),
            "updated_at": (api_status or {}).get("loaded_at", ""),
        },
        "player_stats": {
            "available": api_counts.get("team_stats", 0) > 0,
            "rows": api_counts.get("team_stats", 0),
            "updated_at": (api_status or {}).get("loaded_at", ""),
        },
    }
    missing = [key for key, value in sources.items() if not value.get("available")]
    strength = "fuerte" if not missing else "media" if sources["all_matches"]["available"] and sources["odds"]["available"] else "debil"
    return {
        "prepared_at": prepared_at,
        "strength": strength,
        "sources": sources,
        "missing_sources": missing,
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
                record.update(over_under_target_values(record["HG"], record["AG"]))
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


def total_goals_bucket_from_row(row: pd.Series, cap: int = TOTAL_GOALS_CAP) -> int:
    total_goals = pd.to_numeric(pd.Series([row.get("HG", np.nan)]), errors="coerce").iloc[0] + pd.to_numeric(pd.Series([row.get("AG", np.nan)]), errors="coerce").iloc[0]
    if pd.isna(total_goals):
        return 0
    return int(min(max(int(total_goals), 0), int(cap)))


def history_before_row(history_df: pd.DataFrame, row_date: Optional[pd.Timestamp], row_year: Optional[int], freeze_year: bool = False) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    working = history_df.copy()
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    working = working[working["Date"].notna()].copy()
    if row_date is not None:
        return working[working["Date"] < row_date].copy()
    if row_year:
        return working[working["Date"].dt.year < int(row_year)].copy()
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


def rows_for_training_target(rows: pd.DataFrame, target: str) -> pd.DataFrame:
    working = sort_match_rows(rows)
    if is_over_under_target(target):
        target_column = over_under_column_for_target(target)
        return working[working[target_column].notna()].copy() if target_column in working.columns else working.iloc[0:0].copy()
    if target == GOALS_DISTRIBUTION_TARGET:
        return working[working["HG"].notna() & working["AG"].notna()].copy()
    return working.copy()


def feature_rows_signature(rows: pd.DataFrame) -> Tuple[Any, ...]:
    if rows is None or rows.empty:
        return ("empty", 0)
    columns = [column for column in ("FixtureId", "Date", "Year", "Home", "Away") if column in rows.columns]
    if not columns:
        return ("rows", int(rows.shape[0]), tuple(rows.index.astype(str).tolist()))
    scoped = rows[columns].copy()
    for column in columns:
        if column == "Date":
            scoped[column] = pd.to_datetime(scoped[column], errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
        else:
            scoped[column] = scoped[column].astype(str).fillna("")
    return (
        "rows",
        int(rows.shape[0]),
        tuple(columns),
        tuple(tuple(item) for item in scoped.itertuples(index=False, name=None)),
    )


def dataframe_fingerprint(frame: Optional[pd.DataFrame]) -> str:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return "empty"
    working = frame.copy()
    meta = {
        "shape": [int(working.shape[0]), int(working.shape[1])],
        "columns": [str(column) for column in working.columns],
        "dtypes": [str(dtype) for dtype in working.dtypes],
    }
    digest = hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode("utf-8"))
    try:
        digest.update(pd.util.hash_pandas_object(working, index=True).values.tobytes())
    except Exception:
        digest.update(working.astype(str).to_csv(index=True).encode("utf-8", errors="ignore"))
    return digest.hexdigest()


def api_football_fingerprint(api_football: Optional[Dict[str, pd.DataFrame]]) -> Tuple[str, ...]:
    api_football = api_football or {}
    return tuple(
        dataframe_fingerprint(api_football.get(key))
        for key in ("team_stats", "lineups", "injuries")
    )


def worldcup_model_fingerprint(model: Optional[WorldCupModel]) -> str:
    if model is None:
        return "none"
    rows = []
    for team, profile in sorted(getattr(model, "_profiles", {}).items()):
        rows.append((
            str(team),
            round(float(profile.rating), 8),
            int(profile.matches),
            round(float(profile.attack), 8),
            round(float(profile.defense), 8),
        ))
    payload = {
        "profiles": rows,
        "global_g1": round(float(getattr(model, "global_g1", 0.0)), 8),
        "global_g2": round(float(getattr(model, "global_g2", 0.0)), 8),
        "host_advantage": round(float(getattr(model, "host_advantage", 0.0)), 8),
        "max_goals": int(getattr(model, "max_goals", 0) or 0),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def feature_matrix_cache_key(
        rows: pd.DataFrame,
        base_model: Optional[WorldCupModel],
        history_df: Optional[pd.DataFrame],
        team_features: pd.DataFrame,
        market_rows: pd.DataFrame,
        qualifier_rows: pd.DataFrame,
        api_football: Dict[str, pd.DataFrame],
        international_matches: pd.DataFrame,
        fixture_feature_rows: pd.DataFrame,
        teams: Iterable[str],
        frozen_years: set[int],
        dc_rho: float,
        history_weight: float,
        recency_weight: float,
        host_advantage: float,
        max_goals: int,
) -> Tuple[Any, ...]:
    api_football = api_football or {}
    return (
        FEATURE_STORE_SCHEMA_VERSION,
        feature_rows_signature(rows),
        worldcup_model_fingerprint(base_model),
        dataframe_fingerprint(history_df),
        dataframe_fingerprint(team_features),
        dataframe_fingerprint(market_rows),
        dataframe_fingerprint(qualifier_rows),
        *api_football_fingerprint(api_football),
        dataframe_fingerprint(international_matches),
        dataframe_fingerprint(fixture_feature_rows),
        tuple(sorted(str(team) for team in teams)),
        tuple(sorted(int(year) for year in frozen_years)),
        round(float(dc_rho or 0.0), 8),
        round(float(history_weight or 0.0), 8),
        round(float(recency_weight or 0.0), 8),
        round(float(host_advantage or 0.0), 8),
        int(max_goals),
    )


def feature_store_fingerprint(matrix_key: Tuple[Any, ...]) -> str:
    payload = json.dumps(json_safe(matrix_key), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_store_path(matrix_key: Tuple[Any, ...]) -> Path:
    return FEATURE_STORE_ROOT / f"{feature_store_fingerprint(matrix_key)}.pkl"


def load_feature_matrix_from_store(matrix_key: Tuple[Any, ...]) -> Optional[pd.DataFrame]:
    path = feature_store_path(matrix_key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("schema_version") != FEATURE_STORE_SCHEMA_VERSION:
        return None
    if payload.get("fingerprint") != feature_store_fingerprint(matrix_key):
        return None
    matrix = payload.get("matrix")
    if not isinstance(matrix, pd.DataFrame):
        return None
    return matrix


def save_feature_matrix_to_store(matrix_key: Tuple[Any, ...], matrix: pd.DataFrame) -> None:
    if matrix is None or matrix.empty:
        return
    FEATURE_STORE_ROOT.mkdir(parents=True, exist_ok=True)
    path = feature_store_path(matrix_key)
    payload = {
        "schema_version": FEATURE_STORE_SCHEMA_VERSION,
        "fingerprint": feature_store_fingerprint(matrix_key),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": int(matrix.shape[0]),
        "columns": list(matrix.columns),
        "matrix": matrix,
    }
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(path)


def normalize_build_progress_every(value: Optional[int], row_count: int) -> int:
    if value is None:
        return dynamic_feature_progress_every(row_count)
    try:
        return max(1, int(float(value)))
    except (TypeError, ValueError):
        return dynamic_feature_progress_every(row_count)


def recent15_feature_table_covers(features: Optional[pd.DataFrame], teams: Iterable[str]) -> bool:
    requested = {normalize_team_key(team) for team in teams if normalize_team_key(team)}
    if not requested:
        return True
    if features is None:
        return False
    if features.empty:
        return True
    if "Team" not in features.columns:
        return False
    available = set(features["Team"].dropna().astype(str).map(normalize_team_key).tolist())
    return requested.issubset(available)


def cached_recent15_feature_table(
        feature_cache: WorldCupFeatureBuildCache,
        international_matches: pd.DataFrame,
        working_teams: Iterable[str],
        reference_date: str,
        row_model: WorldCupModel,
        recent15_match_index: Optional[Dict[str, pd.DataFrame]],
) -> pd.DataFrame:
    cache_key = (
        "recent15-table",
        reference_date,
        str(id(row_model)),
        dataframe_cache_id(international_matches),
        id(recent15_match_index) if recent15_match_index is not None else 0,
    )
    teams = list(working_teams)
    cached = feature_cache.recent15_features.get(cache_key)
    if cached is not None and recent15_feature_table_covers(cached, teams):
        feature_cache.stats["recent15_hits"] += 1
        return cached
    feature_cache.stats["recent15_misses"] += 1
    features = recent15_feature_table(
        international_matches,
        teams=teams,
        before_date=reference_date,
        base_model=row_model,
        match_index=recent15_match_index,
    )
    feature_cache.recent15_features[cache_key] = features
    return features


def recent15_features_for_match(features: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    if features is None or features.empty or "Team" not in features.columns:
        return pd.DataFrame()
    requested = {normalize_team_key(home), normalize_team_key(away)}
    team_keys = features["Team"].map(normalize_team_key)
    return features.loc[team_keys.isin(requested)]


def dataframe_cache_id(frame: Optional[pd.DataFrame]) -> int:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return 0
    return id(frame)


def sample_weights_for_rows(rows: pd.DataFrame, target: str = "result") -> pd.Series:
    working = rows_for_training_target(rows, target)
    if working.empty:
        return pd.Series(dtype=float)
    if "sample_weight" in working.columns:
        weights = pd.to_numeric(working["sample_weight"], errors="coerce")
    else:
        weights = pd.Series(np.nan, index=working.index, dtype=float)
    missing = weights.isna()
    if missing.any():
        weights.loc[missing] = working.loc[missing].apply(
            lambda item: sample_weight_for_tournament(item.get("tournament"), is_worldcup=bool(item.get("is_worldcup_match", False))),
            axis=1,
        )
    return weights.astype(float).clip(lower=0.05).reset_index(drop=True)


def align_sample_weights(weights: pd.Series, expected_length: int) -> pd.Series:
    output = pd.to_numeric(pd.Series(weights).reset_index(drop=True), errors="coerce").fillna(1.0).astype(float)
    expected_length = int(expected_length)
    if len(output) > expected_length:
        output = output.iloc[:expected_length].copy()
    elif len(output) < expected_length:
        output = pd.concat([output, pd.Series([1.0] * (expected_length - len(output)))], ignore_index=True)
    return output.clip(lower=0.05).reset_index(drop=True)


def sample_weight_summary(weights: pd.Series) -> Dict[str, Any]:
    values = pd.to_numeric(pd.Series(weights), errors="coerce").dropna()
    if values.empty:
        return {"enabled": False, "rows": 0}
    return {
        "enabled": True,
        "rows": int(values.shape[0]),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
        "mean": round(float(values.mean()), 4),
        "policy": SAMPLE_WEIGHT_POLICY,
    }


def split_support_payload(y_train: Iterable[Any], y_eval: Iterable[Any], target: str) -> Dict[str, Any]:
    return {
        "target": str(target or "result"),
        "train": target_support_payload(y_train),
        "eval": target_support_payload(y_eval),
    }


def target_support_payload(values: Iterable[Any]) -> Dict[str, Any]:
    series = pd.Series(values).dropna()
    return {
        "rows": int(series.shape[0]),
        "class_distribution": value_counts_payload(series),
    }


def build_training_matrix(
        rows: pd.DataFrame,
        base_model: Optional[WorldCupModel] = None,
        team_features: Optional[pd.DataFrame] = None,
        history_team_features: Optional[pd.DataFrame] = None,
        matchup_features: Optional[pd.DataFrame] = None,
        market_rows: Optional[pd.DataFrame] = None,
        qualifier_rows: Optional[pd.DataFrame] = None,
        api_football: Optional[Dict[str, pd.DataFrame]] = None,
        international_matches: Optional[pd.DataFrame] = None,
        recent15_match_index: Optional[Dict[str, pd.DataFrame]] = None,
        fixture_feature_rows: Optional[pd.DataFrame] = None,
        feature_columns: Optional[List[str]] = None,
        target: str = "result",
        history_df: Optional[pd.DataFrame] = None,
        teams: Optional[Iterable[str]] = None,
        frozen_years: Optional[set[int]] = None,
        dc_rho: float = 0.0,
        history_weight: float = 1.0,
        recency_weight: float = 0.35,
        host_advantage: float = 45.0,
        max_goals: int = 10,
        feature_cache: Optional[WorldCupFeatureBuildCache] = None,
        progress_callback=None,
        progress_stage: str = "",
        progress_message: str = "",
        progress_market: str = "",
        progress_model_id: str = "",
        progress_every: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    working = rows_for_training_target(rows, target)
    if team_features is None:
        team_features = pd.DataFrame()
    if market_rows is None:
        market_rows = pd.DataFrame()
    if qualifier_rows is None:
        qualifier_rows = pd.DataFrame()
    elif not qualifier_rows.empty and not qualifier_rows.attrs.get("worldcup_market_normalized"):
        qualifier_rows = normalize_market_frame(qualifier_rows)
    if fixture_feature_rows is None:
        fixture_feature_rows = pd.DataFrame()
    api_football = api_football or {}
    if international_matches is None:
        international_matches = load_international_matches(required=False)
    if recent15_match_index is None:
        recent15_match_index = build_recent15_match_index(international_matches)
    static_model = base_model
    working_teams = list(teams or teams_from_rows(working))
    frozen_years = frozen_years or set()
    if static_model is None and history_df is None:
        static_model = WorldCupModel.from_history(pd.DataFrame(), teams=working_teams)
    feature_cache = feature_cache or WorldCupFeatureBuildCache()
    matrix_key = feature_matrix_cache_key(
        rows=working,
        base_model=static_model,
        history_df=history_df,
        team_features=team_features,
        market_rows=market_rows,
        qualifier_rows=qualifier_rows,
        api_football=api_football,
        international_matches=international_matches,
        fixture_feature_rows=fixture_feature_rows,
        teams=working_teams,
        frozen_years=frozen_years,
        dc_rho=dc_rho,
        history_weight=history_weight,
        recency_weight=recency_weight,
        host_advantage=host_advantage,
        max_goals=max_goals,
    )
    cached_x = feature_cache.matrices.get(matrix_key)
    if cached_x is not None:
        feature_cache.stats["matrix_hits"] += 1
        return finalize_training_matrix_from_features(
            cached_x.copy(),
            working,
            target,
            feature_columns,
            progress_callback=progress_callback,
            progress_stage=progress_stage or "features",
            progress_message=progress_message or "Features reutilizadas desde cache",
            progress_market=progress_market,
            progress_model_id=progress_model_id,
            feature_cache_state="hit",
            progress_every=normalize_build_progress_every(progress_every, int(max(working.shape[0], 1))),
        )
    persisted_x = load_feature_matrix_from_store(matrix_key)
    if persisted_x is not None:
        feature_cache.stats["matrix_hits"] += 1
        feature_cache.stats["persistent_matrix_hits"] += 1
        feature_cache.matrices[matrix_key] = persisted_x.copy()
        return finalize_training_matrix_from_features(
            persisted_x.copy(),
            working,
            target,
            feature_columns,
            progress_callback=progress_callback,
            progress_stage=progress_stage or "features",
            progress_message=progress_message or "Features reutilizadas desde feature store",
            progress_market=progress_market,
            progress_model_id=progress_model_id,
            feature_cache_state="persistent-hit",
            progress_every=normalize_build_progress_every(progress_every, int(max(working.shape[0], 1))),
        )

    feature_cache.stats["matrix_misses"] += 1
    feature_cache.stats["persistent_matrix_misses"] += 1
    market_key = dataframe_cache_id(market_rows)
    if market_key not in feature_cache.market_lookup:
        feature_cache.market_lookup[market_key] = build_market_lookup(market_rows)
    market_lookup = feature_cache.market_lookup[market_key]
    fixture_lookup = cached_fixture_feature_lookup(feature_cache, fixture_feature_rows)
    records = []
    progress_total = int(max(working.shape[0], 1))
    progress_every = normalize_build_progress_every(progress_every, progress_total)
    working_records = working.to_dict(orient="records")
    feature_started_at = time.monotonic()
    last_progress_at = feature_started_at
    for row_index, row in enumerate(working_records, start=1):
        row_year = match_year_from_row(row)
        row_date = match_date_from_row(row)
        reference_date = reference_date_for_row(row_date, row_year)
        home_team = str(row["Home"])
        away_team = str(row["Away"])
        team_features_key = (dataframe_cache_id(team_features), "" if row_year is None else str(row_year))
        if team_features_key not in feature_cache.team_features_asof:
            feature_cache.team_features_asof[team_features_key] = team_features_asof(team_features, row_year)
        if history_df is not None:
            frozen = row_year in frozen_years
            snapshot_key = (
                dataframe_cache_id(history_df),
                dataframe_cache_id(qualifier_rows),
                dataframe_cache_id(api_football.get("team_stats")),
                dataframe_cache_id(api_football.get("lineups")),
                dataframe_cache_id(api_football.get("injuries")),
                tuple(sorted(str(team) for team in working_teams)),
                "date" if row_date is not None else "year",
                reference_date if row_date is not None else str(int(row_year)) if row_year else reference_date,
                int(bool(frozen)),
                round(float(history_weight or 0.0), 8),
                round(float(recency_weight or 0.0), 8),
                round(float(host_advantage or 0.0), 8),
                int(max_goals),
            )
            if snapshot_key not in feature_cache.snapshots:
                feature_cache.stats["snapshot_misses"] += 1
                history_cutoff = history_before_row(history_df, row_date=row_date, row_year=row_year, freeze_year=frozen)
                snapshot_model = WorldCupModel.from_history(
                    history_cutoff,
                    teams=working_teams,
                    history_weight=history_weight,
                    recency_weight=recency_weight,
                    host_advantage=host_advantage,
                    max_goals=max_goals,
                )
                feature_cache.snapshots[snapshot_key] = (
                    snapshot_model,
                    build_history_feature_table(history_cutoff, reference_date=reference_date),
                    build_matchup_feature_table(history_cutoff, reference_date=reference_date),
                    qualifier_feature_table(qualifier_rows, reference_date=reference_date, teams=working_teams),
                    api_football_feature_table(
                        api_football.get("team_stats", pd.DataFrame()),
                        reference_date=reference_date,
                        teams=working_teams,
                        lineups=api_football.get("lineups", pd.DataFrame()),
                        injuries=api_football.get("injuries", pd.DataFrame()),
                    ),
                )
            else:
                feature_cache.stats["snapshot_hits"] += 1
            row_model, row_history_features, row_matchup_features, row_qualifier_features, row_api_football_features = feature_cache.snapshots[snapshot_key]
        else:
            row_model = static_model
            row_history_features = history_team_features
            row_matchup_features = matchup_features
            static_cache_key = (
                dataframe_cache_id(qualifier_rows),
                dataframe_cache_id(api_football.get("team_stats")),
                dataframe_cache_id(api_football.get("lineups")),
                dataframe_cache_id(api_football.get("injuries")),
                tuple(sorted(str(team) for team in working_teams)),
                reference_date,
                str(int(row_year)) if row_year else "",
            )
            if static_cache_key not in feature_cache.static_features:
                feature_cache.static_features[static_cache_key] = (
                    qualifier_feature_table(qualifier_rows, reference_date=reference_date, teams=working_teams),
                    api_football_feature_table(
                        api_football.get("team_stats", pd.DataFrame()),
                        reference_date=reference_date,
                        teams=working_teams,
                        lineups=api_football.get("lineups", pd.DataFrame()),
                        injuries=api_football.get("injuries", pd.DataFrame()),
                    ),
                )
            row_qualifier_features, row_api_football_features = feature_cache.static_features[static_cache_key]
        row_recent15_table = cached_recent15_feature_table(
            feature_cache=feature_cache,
            international_matches=international_matches,
            working_teams=working_teams,
            reference_date=reference_date,
            row_model=row_model,
            recent15_match_index=recent15_match_index,
        )
        row_feature_lookups = {
            "qualifier": cached_team_feature_lookup(feature_cache, row_qualifier_features),
            "api_football": cached_team_feature_lookup(feature_cache, row_api_football_features),
            "recent15": cached_team_feature_lookup(feature_cache, row_recent15_table),
            "kaggle": cached_team_feature_lookup(
                feature_cache,
                feature_cache.team_features_asof[team_features_key],
                limit=24,
            ),
            "history": cached_team_feature_lookup(feature_cache, row_history_features),
            "matchup": cached_matchup_feature_lookup(feature_cache, row_matchup_features),
            "fixture": fixture_lookup,
        }
        records.append(
            match_feature_row(
                row_model,
                feature_cache.team_features_asof[team_features_key],
                home_team,
                away_team,
                history_team_features=row_history_features,
                matchup_features=row_matchup_features,
                market_rows=market_rows,
                market_lookup=market_lookup,
                qualifier_features=row_qualifier_features,
                api_football_features=row_api_football_features,
                recent15_features=None,
                fixture_feature_rows=fixture_feature_rows,
                fixture_id=row.get("FixtureId"),
                match_date=row_date,
                match_year=row_year,
                fixture_context=row,
                dc_rho=dc_rho,
                feature_lookups=row_feature_lookups,
            )
        )
        now = time.monotonic()
        time_due = (now - last_progress_at) >= 12.0
        if progress_callback is not None and (row_index == 1 or row_index == progress_total or row_index % progress_every == 0 or time_due):
            elapsed = max(now - feature_started_at, 1e-9)
            rows_per_second = float(row_index / elapsed)
            eta_seconds = int(max((progress_total - row_index) / max(rows_per_second, 1e-9), 0.0))
            emit_training_progress(
                progress_callback,
                progress_stage or "features",
                row_index,
                progress_total,
                progress_message or "Construyendo features",
                market=progress_market,
                model_id=progress_model_id,
                rows=row_index,
                features=len(records[-1]) if records else 0,
                feature_cache="miss",
                progress_every=progress_every,
                elapsed_seconds=int(elapsed),
                rows_per_second=round(rows_per_second, 2),
                eta_seconds=eta_seconds,
            )
            last_progress_at = now
    x = pd.DataFrame(records).fillna(0.0)
    feature_cache.matrices[matrix_key] = x.copy()
    save_feature_matrix_to_store(matrix_key, x)
    return finalize_training_matrix_from_features(
        x,
        working,
        target,
        feature_columns,
        progress_callback=None,
        feature_cache_state="miss",
    )


def finalize_training_matrix_from_features(
        x: pd.DataFrame,
        working: pd.DataFrame,
        target: str,
        feature_columns: Optional[List[str]],
        progress_callback=None,
        progress_stage: str = "features",
        progress_message: str = "",
        progress_market: str = "",
        progress_model_id: str = "",
        feature_cache_state: str = "",
        progress_every: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if feature_columns is None:
        feature_columns = list(x.columns)
    for column in feature_columns:
        if column not in x.columns:
            x[column] = 0.0
    x = x[feature_columns].astype(float)
    if progress_callback is not None:
        emit_training_progress(
            progress_callback,
            progress_stage,
            int(working.shape[0]),
            int(max(working.shape[0], 1)),
            progress_message,
            market=progress_market,
            model_id=progress_model_id,
            rows=int(working.shape[0]),
            features=int(x.shape[1]),
            feature_cache=feature_cache_state,
            progress_every=progress_every,
        )
    if is_over_under_target(target):
        return x, working[over_under_column_for_target(target)].astype(int), feature_columns
    if target == GOALS_DISTRIBUTION_TARGET:
        return x, total_goals_buckets(working).astype(int), feature_columns
    return x, working["Label"].astype(str), feature_columns


def total_goals_buckets(rows: pd.DataFrame, cap: int = TOTAL_GOALS_CAP) -> pd.Series:
    home_goals = pd.to_numeric(rows.get("HG", pd.Series(index=rows.index, dtype=float)), errors="coerce")
    away_goals = pd.to_numeric(rows.get("AG", pd.Series(index=rows.index, dtype=float)), errors="coerce")
    totals = home_goals + away_goals
    buckets = pd.Series(0, index=rows.index, dtype=int)
    valid = totals.notna()
    if valid.any():
        buckets.loc[valid] = totals.loc[valid].astype(int).clip(lower=0, upper=int(cap))
    return buckets


def build_market_lookup(market_rows: Optional[pd.DataFrame]) -> Dict[Tuple[str, str], pd.DataFrame]:
    if market_rows is None or market_rows.empty:
        return {}
    working = normalize_market_frame(market_rows)
    if working.empty:
        return {}
    working = working.copy()
    working["_home_key"] = working["Home"].map(normalize_team_key)
    working["_away_key"] = working["Away"].map(normalize_team_key)
    working["_date_key"] = pd.to_datetime(working["Date"], errors="coerce").dt.date
    working["_year_key"] = pd.to_numeric(working["Year"], errors="coerce")
    working["_priority"] = working["market_source"].map(market_source_priority)
    return {
        (str(home_key), str(away_key)): frame.copy()
        for (home_key, away_key), frame in working.groupby(["_home_key", "_away_key"], sort=False)
    }


def market_for_match_lookup(
        lookup: Dict[Tuple[str, str], pd.DataFrame],
        home: str,
        away: str,
        match_date: Optional[Any] = None,
        year: Optional[int] = None,
) -> Dict[str, Any]:
    if not lookup:
        return {}
    scoped = lookup.get((normalize_team_key(home), normalize_team_key(away)))
    if scoped is None or scoped.empty:
        return {}
    selected = scoped
    date_ts = pd.to_datetime(match_date, errors="coerce") if match_date is not None else pd.NaT
    if pd.notna(date_ts):
        same_day = selected[selected["_date_key"] == pd.Timestamp(date_ts).date()]
        if same_day.empty:
            return {}
        selected = same_day
    elif year is not None:
        same_year = selected[selected["_year_key"] == int(year)]
        if not same_year.empty:
            selected = same_year
    if selected.empty:
        return {}
    return selected.sort_values("_priority", ascending=False, kind="stable").iloc[0].to_dict()


def cached_team_feature_lookup(
        feature_cache: WorldCupFeatureBuildCache,
        features: Optional[pd.DataFrame],
        team_column: str = "Team",
        exclude_columns: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
) -> Dict[str, Any]:
    excluded = tuple(exclude_columns or ())
    cache_key = ("team-feature-lookup", dataframe_cache_id(features), team_column, excluded, "" if limit is None else int(limit))
    cached = feature_cache.table_lookups.get(cache_key)
    if cached is not None:
        return cached
    lookup = build_team_feature_lookup(features, team_column=team_column, exclude_columns=excluded, limit=limit)
    feature_cache.table_lookups[cache_key] = lookup
    return lookup


def build_team_feature_lookup(
        features: Optional[pd.DataFrame],
        team_column: str = "Team",
        exclude_columns: Optional[Iterable[str]] = None,
        limit: Optional[int] = None,
) -> Dict[str, Any]:
    if features is None or features.empty or team_column not in features.columns:
        return {"columns": [], "rows": {}}
    excluded = set(exclude_columns or ())
    numeric_cols = [
        column for column in features.columns
        if column != team_column and column not in excluded and pd.api.types.is_numeric_dtype(features[column])
    ]
    if limit is not None:
        numeric_cols = numeric_cols[:int(limit)]
    rows: Dict[str, Dict[str, Any]] = {}
    for record in features[[team_column] + numeric_cols].to_dict(orient="records"):
        key = normalize_team_key(record.get(team_column))
        if key and key not in rows:
            rows[key] = {column: record.get(column) for column in numeric_cols}
    return {"columns": numeric_cols, "rows": rows}


def cached_matchup_feature_lookup(
        feature_cache: WorldCupFeatureBuildCache,
        matchup_features: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    cache_key = ("matchup-feature-lookup", dataframe_cache_id(matchup_features))
    cached = feature_cache.table_lookups.get(cache_key)
    if cached is not None:
        return cached
    lookup = build_matchup_feature_lookup(matchup_features)
    feature_cache.table_lookups[cache_key] = lookup
    return lookup


def build_matchup_feature_lookup(matchup_features: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if matchup_features is None or matchup_features.empty or not {"HomeKey", "AwayKey"}.issubset(matchup_features.columns):
        return {"columns": [], "rows": {}}
    columns = [column for column in matchup_features.columns if column not in {"HomeKey", "AwayKey"}]
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in matchup_features[["HomeKey", "AwayKey"] + columns].to_dict(orient="records"):
        key = (normalize_team_key(record.get("HomeKey")), normalize_team_key(record.get("AwayKey")))
        if key[0] and key[1] and key not in rows:
            rows[key] = {column: record.get(column) for column in columns}
    return {"columns": columns, "rows": rows}


def cached_fixture_feature_lookup(
        feature_cache: WorldCupFeatureBuildCache,
        fixture_feature_rows: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    cache_key = ("fixture-feature-lookup", dataframe_cache_id(fixture_feature_rows))
    cached = feature_cache.table_lookups.get(cache_key)
    if cached is not None:
        return cached
    lookup = build_fixture_feature_lookup(fixture_feature_rows)
    feature_cache.table_lookups[cache_key] = lookup
    return lookup


def build_fixture_feature_lookup(fixture_feature_rows: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if fixture_feature_rows is None or fixture_feature_rows.empty or not {"fixture_id", "Equipo"}.issubset(fixture_feature_rows.columns):
        return {"columns": [], "rows": {}}
    excluded = {"fixture_id", "Equipo", "Rival"}
    numeric_cols = [
        column for column in fixture_feature_rows.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(fixture_feature_rows[column])
    ]
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    fixture_ids = set()
    for record in fixture_feature_rows[["fixture_id", "Equipo"] + numeric_cols].to_dict(orient="records"):
        fixture_key = str(record.get("fixture_id"))
        team_key = normalize_team_key(record.get("Equipo"))
        if fixture_key:
            fixture_ids.add(fixture_key)
        if fixture_key and team_key and (fixture_key, team_key) not in rows:
            rows[(fixture_key, team_key)] = {column: record.get(column) for column in numeric_cols}
    return {"columns": numeric_cols, "rows": rows, "fixtures": fixture_ids}


def feature_float(value: Any, missing_default: float = 0.0) -> float:
    if value is None:
        return missing_default
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def merge_qualifier_feature_lookup(
        row: Dict[str, float],
        lookup: Optional[Dict[str, Any]],
        home_key: str,
        away_key: str,
) -> None:
    row["qualifier_context_available"] = 0.0
    if not lookup:
        return
    rows = lookup.get("rows", {})
    columns = lookup.get("columns", [])
    home_features = rows.get(home_key)
    away_features = rows.get(away_key)
    if home_features is not None or away_features is not None:
        row["qualifier_context_available"] = 1.0
    for column in columns:
        home_value = feature_float(home_features.get(column), 0.0) if home_features is not None else 0.0
        away_value = feature_float(away_features.get(column), 0.0) if away_features is not None else 0.0
        safe = normalize_column(column)
        row[f"{safe}_home"] = home_value
        row[f"{safe}_away"] = away_value
        row[f"{safe}_diff"] = home_value - away_value


def merge_team_feature_lookup(
        row: Dict[str, float],
        lookup: Optional[Dict[str, Any]],
        home_key: str,
        away_key: str,
        prefix: str,
) -> None:
    if not lookup:
        return
    rows = lookup.get("rows", {})
    columns = lookup.get("columns", [])
    home_features = rows.get(home_key)
    away_features = rows.get(away_key)
    for column in columns:
        home_value = feature_float(home_features.get(column), 0.0) if home_features is not None else 0.0
        away_value = feature_float(away_features.get(column), 0.0) if away_features is not None else 0.0
        safe = normalize_column(column)
        row[f"{prefix}_{safe}_home"] = home_value
        row[f"{prefix}_{safe}_away"] = away_value
        row[f"{prefix}_{safe}_diff"] = home_value - away_value


def merge_recent15_feature_lookup(
        row: Dict[str, float],
        lookup: Optional[Dict[str, Any]],
        home_key: str,
        away_key: str,
) -> None:
    if not lookup:
        return
    rows = lookup.get("rows", {})
    columns = lookup.get("columns", [])
    home_features = rows.get(home_key)
    away_features = rows.get(away_key)
    for column in columns:
        home_value = feature_float(home_features.get(column), 0.0) if home_features is not None else 0.0
        away_value = feature_float(away_features.get(column), 0.0) if away_features is not None else 0.0
        safe = normalize_column(column)
        key = safe if safe.startswith("recent15_") else f"recent15_{safe}"
        row[f"{key}_home"] = home_value
        row[f"{key}_away"] = away_value
        row[f"{key}_diff"] = home_value - away_value


def recent15_context_available_from_lookup(
        lookup: Optional[Dict[str, Any]],
        home_key: str,
        away_key: str,
) -> float:
    if not lookup:
        return 0.0
    rows = lookup.get("rows", {})
    for team_key in (home_key, away_key):
        record = rows.get(team_key)
        if record is not None and feature_float(record.get("recent15_matches"), 0.0) > 0.0:
            return 1.0
    return 0.0


def merge_matchup_feature_lookup(
        row: Dict[str, float],
        lookup: Optional[Dict[str, Any]],
        home_key: str,
        away_key: str,
) -> None:
    if not lookup:
        return
    record = lookup.get("rows", {}).get((home_key, away_key))
    if not record:
        return
    for column in lookup.get("columns", []):
        value = record.get(column)
        if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
            row[f"h2h_{normalize_column(column)}"] = float(value)


def merge_fixture_feature_lookup(
        row: Dict[str, float],
        lookup: Optional[Dict[str, Any]],
        home_key: str,
        away_key: str,
        fixture_id: Optional[Any],
        prefix: str,
) -> None:
    if not lookup or fixture_id in {"", None}:
        return
    rows = lookup.get("rows", {})
    columns = lookup.get("columns", [])
    fixture_key = str(fixture_id)
    if fixture_key not in lookup.get("fixtures", set()):
        return
    home_features = rows.get((fixture_key, home_key))
    away_features = rows.get((fixture_key, away_key))
    for column in columns:
        home_value = feature_float(home_features.get(column), 0.0) if home_features is not None else 0.0
        away_value = feature_float(away_features.get(column), 0.0) if away_features is not None else 0.0
        safe = normalize_column(column)
        row[f"{prefix}_{safe}_home"] = home_value
        row[f"{prefix}_{safe}_away"] = away_value
        row[f"{prefix}_{safe}_diff"] = home_value - away_value


def match_feature_row(
        base_model: WorldCupModel,
        team_features: pd.DataFrame,
        home: str,
        away: str,
        history_team_features: Optional[pd.DataFrame] = None,
        matchup_features: Optional[pd.DataFrame] = None,
        market_rows: Optional[pd.DataFrame] = None,
        qualifier_features: Optional[pd.DataFrame] = None,
        api_football_features: Optional[pd.DataFrame] = None,
        recent15_features: Optional[pd.DataFrame] = None,
        fixture_feature_rows: Optional[pd.DataFrame] = None,
        fixture_id: Optional[Any] = None,
        match_date: Optional[Any] = None,
        match_year: Optional[int] = None,
        fixture_context: Optional[Dict[str, Any]] = None,
        market_lookup: Optional[Dict[Tuple[str, str], pd.DataFrame]] = None,
        feature_lookups: Optional[Dict[str, Any]] = None,
        dc_rho: float = 0.0,
) -> Dict[str, float]:
    home_key = normalize_team_key(home)
    away_key = normalize_team_key(away)
    recent15_lookup = feature_lookups.get("recent15") if feature_lookups else None
    p_home = base_model.profile(home)
    p_away = base_model.profile(away)
    poisson = base_model.match_probabilities(home, away)
    lambda_home = float(poisson.get("lambda1", 0.0))
    lambda_away = float(poisson.get("lambda2", 0.0))
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
        "recent15_context_available": (
            recent15_context_available_from_lookup(recent15_lookup, home_key, away_key)
            if recent15_lookup is not None
            else recent15_context_available(recent15_features if recent15_features is not None else pd.DataFrame(), home, away)
        ),
    }
    row.update(shrinkage_feature_row(p_home, p_away))
    row.update(score_grid_features(lambda_home, lambda_away, max_goals=base_model.max_goals, score_cap=4))
    row.update(dixon_coles_probabilities(lambda_home, lambda_away, rho=dc_rho, max_goals=base_model.max_goals))
    row.update(model_calibration_features(poisson))
    market_match = (
        market_for_match_lookup(market_lookup, home, away, match_date=match_date, year=match_year)
        if market_lookup is not None
        else market_for_match(
            market_rows if market_rows is not None else pd.DataFrame(),
            home,
            away,
            match_date=match_date,
            year=match_year,
        )
    )
    row.update(build_market_feature_row(
        market_match,
        model_probs={"H": poisson.get("home", 0.0), "D": poisson.get("draw", 0.0), "A": poisson.get("away", 0.0)},
        model_totals=total_line_probabilities_from_probs(poisson),
    ))
    if feature_lookups is not None:
        merge_qualifier_feature_lookup(row, feature_lookups.get("qualifier"), home_key, away_key)
        merge_team_feature_lookup(row, feature_lookups.get("api_football"), home_key, away_key, prefix="api_football")
        merge_recent15_feature_lookup(row, feature_lookups.get("recent15"), home_key, away_key)
    else:
        merge_qualifier_feature_block(row, qualifier_features if qualifier_features is not None else pd.DataFrame(), home, away)
        merge_team_feature_block(row, api_football_features if api_football_features is not None else pd.DataFrame(), home, away, prefix="api_football")
        merge_recent15_feature_block(row, recent15_features if recent15_features is not None else pd.DataFrame(), home, away)
    row.update(fixture_context_features(fixture_context or {}, home=home, away=away, fixture_id=fixture_id, match_date=match_date, match_year=match_year))
    if feature_lookups is not None:
        merge_team_feature_lookup(row, feature_lookups.get("kaggle"), home_key, away_key, prefix="kaggle")
        merge_team_feature_lookup(row, feature_lookups.get("history"), home_key, away_key, prefix="history")
        merge_matchup_feature_lookup(row, feature_lookups.get("matchup"), home_key, away_key)
        merge_fixture_feature_lookup(row, feature_lookups.get("fixture"), home_key, away_key, fixture_id=fixture_id, prefix="xi")
    else:
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


def recent15_context_available(features: pd.DataFrame, home: str, away: str) -> float:
    if features is None or features.empty or "Team" not in features.columns:
        return 0.0
    home_features = features[features["Team"].map(normalize_team_key) == normalize_team_key(home)]
    away_features = features[features["Team"].map(normalize_team_key) == normalize_team_key(away)]
    for scoped in (home_features, away_features):
        if scoped.empty or "recent15_matches" not in scoped.columns:
            continue
        matches = pd.to_numeric(scoped["recent15_matches"], errors="coerce").fillna(0.0)
        if float(matches.iloc[0]) > 0.0:
            return 1.0
    return 0.0


def shrinkage_feature_row(p_home, p_away, prior_rating: float = 1500.0, prior_strength: float = 1.0, prior_matches: float = 6.0) -> Dict[str, float]:
    home_weight = float(p_home.matches / (p_home.matches + prior_matches)) if p_home.matches > 0 else 0.0
    away_weight = float(p_away.matches / (p_away.matches + prior_matches)) if p_away.matches > 0 else 0.0
    rating_home = prior_rating + (float(p_home.rating) - prior_rating) * home_weight
    rating_away = prior_rating + (float(p_away.rating) - prior_rating) * away_weight
    attack_home = prior_strength + (float(p_home.attack) - prior_strength) * home_weight
    attack_away = prior_strength + (float(p_away.attack) - prior_strength) * away_weight
    defense_home = prior_strength + (float(p_home.defense) - prior_strength) * home_weight
    defense_away = prior_strength + (float(p_away.defense) - prior_strength) * away_weight
    return {
        "shrinkage_weight_home": home_weight,
        "shrinkage_weight_away": away_weight,
        "rating_home_shrunk": rating_home,
        "rating_away_shrunk": rating_away,
        "rating_diff_shrunk": rating_home - rating_away,
        "attack_home_shrunk": attack_home,
        "attack_away_shrunk": attack_away,
        "attack_diff_shrunk": attack_home - attack_away,
        "defense_home_shrunk": defense_home,
        "defense_away_shrunk": defense_away,
        "defense_diff_shrunk": defense_home - defense_away,
    }


def model_calibration_features(poisson: Dict[str, float]) -> Dict[str, float]:
    result_probs = normalize_probability_vector([
        float(poisson.get("home", 0.0)),
        float(poisson.get("draw", 0.0)),
        float(poisson.get("away", 0.0)),
    ])
    total_probs = normalize_probability_vector([
        float(poisson.get("over25", 0.0)),
        float(poisson.get("under25", 0.0)),
    ])
    result_sorted = sorted(result_probs, reverse=True)
    total_sorted = sorted(total_probs, reverse=True)
    return {
        "model_entropy_1x2": probability_entropy(result_probs),
        "model_entropy_1x2_norm": probability_entropy(result_probs) / math.log(3.0),
        "model_entropy_ou25": probability_entropy(total_probs),
        "model_entropy_ou25_norm": probability_entropy(total_probs) / math.log(2.0),
        "model_max_prob_1x2": result_sorted[0],
        "model_second_prob_1x2": result_sorted[1],
        "model_gap_1x2": result_sorted[0] - result_sorted[1],
        "model_max_prob_ou25": total_sorted[0],
        "model_gap_ou25": total_sorted[0] - total_sorted[1],
        "model_sharpness_1x2": float(sum((prob - (1.0 / 3.0)) ** 2 for prob in result_probs)),
        "model_sharpness_ou25": float(sum((prob - 0.5) ** 2 for prob in total_probs)),
        "model_expected_brier_vs_uniform_1x2": float(sum((prob - (1.0 / 3.0)) ** 2 for prob in result_probs)),
        "model_expected_brier_vs_uniform_ou25": float(sum((prob - 0.5) ** 2 for prob in total_probs)),
    }


def normalize_probability_vector(values: Iterable[float]) -> List[float]:
    clean = [max(float(value), 0.0) for value in values]
    total = sum(clean)
    if total <= 0.0:
        return [1.0 / max(len(clean), 1)] * len(clean)
    return [value / total for value in clean]


def probability_entropy(values: Iterable[float]) -> float:
    return float(-sum(max(float(value), 1e-12) * math.log(max(float(value), 1e-12)) for value in values))


def merge_qualifier_feature_block(
        row: Dict[str, float],
        features: pd.DataFrame,
        home: str,
        away: str,
) -> None:
    row["qualifier_context_available"] = 0.0
    if features.empty or "Team" not in features.columns:
        return
    home_features = features[features["Team"].map(normalize_team_key) == normalize_team_key(home)]
    away_features = features[features["Team"].map(normalize_team_key) == normalize_team_key(away)]
    if not home_features.empty or not away_features.empty:
        row["qualifier_context_available"] = 1.0
    numeric_cols = [column for column in features.columns if column != "Team" and pd.api.types.is_numeric_dtype(features[column])]
    for column in numeric_cols:
        home_value = float(home_features[column].iloc[0]) if not home_features.empty else 0.0
        away_value = float(away_features[column].iloc[0]) if not away_features.empty else 0.0
        safe = normalize_column(column)
        row[f"{safe}_home"] = home_value
        row[f"{safe}_away"] = away_value
        row[f"{safe}_diff"] = home_value - away_value


def fixture_context_features(
        context: Dict[str, Any],
        home: str,
        away: str,
        fixture_id: Optional[Any] = None,
        match_date: Optional[Any] = None,
        match_year: Optional[int] = None,
) -> Dict[str, float]:
    round_name = str(
        context.get("Round", context.get("Ronda", context.get("round", context.get("stage", ""))))
        or ""
    )
    group_name = str(context.get("Group", context.get("Grupo", context.get("group", ""))) or "")
    venue = str(context.get("Venue", context.get("Sede", context.get("ground", ""))) or "")
    lowered_round = normalize_team_key(round_name)
    matchday_match = re.search(r"matchday\s+(\d+)", lowered_round)
    fixture_number = numeric_or_zero(context.get("FixtureId", context.get("No.", fixture_id)))
    date_ts = pd.to_datetime(match_date if match_date is not None else context.get("Date", context.get("Fecha", pd.NaT)), errors="coerce")
    year_value = float(match_year or context.get("Year", 0) or (date_ts.year if pd.notna(date_ts) else 0) or 0)
    is_group = bool(group_name) or "group" in lowered_round or "matchday" in lowered_round
    is_final = "final" in lowered_round and "semi" not in lowered_round and "quarter" not in lowered_round
    is_semi = "semi" in lowered_round
    is_quarter = "quarter" in lowered_round or "cuarto" in lowered_round
    is_round16 = "round of 16" in lowered_round or "last 16" in lowered_round or "octavo" in lowered_round
    is_round32 = "round of 32" in lowered_round
    is_knockout = any([is_final, is_semi, is_quarter, is_round16, is_round32]) or ("knockout" in lowered_round)
    venue_key = normalize_team_key(venue)
    home_key = normalize_team_key(home)
    away_key = normalize_team_key(away)
    return {
        "fixture_context_available": 1.0 if context else 0.0,
        "fixture_number": fixture_number,
        "fixture_year": year_value,
        "fixture_month": float(date_ts.month) if pd.notna(date_ts) else 0.0,
        "fixture_day": float(date_ts.day) if pd.notna(date_ts) else 0.0,
        "matchday_number": float(matchday_match.group(1)) if matchday_match else 0.0,
        "stage_group": 1.0 if is_group else 0.0,
        "stage_knockout": 1.0 if is_knockout and not is_group else 0.0,
        "stage_round_of_32": 1.0 if is_round32 else 0.0,
        "stage_round_of_16": 1.0 if is_round16 else 0.0,
        "stage_quarter_final": 1.0 if is_quarter else 0.0,
        "stage_semi_final": 1.0 if is_semi else 0.0,
        "stage_final": 1.0 if is_final else 0.0,
        "group_letter_ord": group_letter_index(group_name),
        "venue_mentions_home": 1.0 if home_key and home_key in venue_key else 0.0,
        "venue_mentions_away": 1.0 if away_key and away_key in venue_key else 0.0,
        "home_host_fixture": 1.0 if home in HOST_TEAMS else 0.0,
        "away_host_fixture": 1.0 if away in HOST_TEAMS else 0.0,
    }


def numeric_or_zero(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return float(number) if np.isfinite(number) else 0.0


def group_letter_index(group_name: Any) -> float:
    letter = str(group_letter(str(group_name or "")) or "").upper()[:1]
    if not letter or not ("A" <= letter <= "Z"):
        return 0.0
    return float(ord(letter) - ord("A") + 1)


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


def merge_recent15_feature_block(row: Dict[str, float], features: pd.DataFrame, home: str, away: str) -> None:
    if features.empty or "Team" not in features.columns:
        return
    home_features = features[features["Team"].map(normalize_team_key) == normalize_team_key(home)]
    away_features = features[features["Team"].map(normalize_team_key) == normalize_team_key(away)]
    numeric_cols = [column for column in features.columns if column != "Team" and pd.api.types.is_numeric_dtype(features[column])]
    for column in numeric_cols:
        home_value = float(home_features[column].iloc[0]) if not home_features.empty else 0.0
        away_value = float(away_features[column].iloc[0]) if not away_features.empty else 0.0
        safe = normalize_column(column)
        key = safe if safe.startswith("recent15_") else f"recent15_{safe}"
        row[f"{key}_home"] = home_value
        row[f"{key}_away"] = away_value
        row[f"{key}_diff"] = home_value - away_value


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
        base.update(linear_trend_features(team_df, prefix="all_trend"))
        base.update(linear_trend_features(team_df.tail(10), prefix="last_10_trend"))
        base.update(linear_trend_features(team_df.tail(5), prefix="last_5_trend"))
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
            f"{prefix}_goals_for_skew": 0.0,
            f"{prefix}_goals_for_kurtosis": 0.0,
            f"{prefix}_goals_against_skew": 0.0,
            f"{prefix}_goals_against_kurtosis": 0.0,
            f"{prefix}_goal_diff_skew": 0.0,
            f"{prefix}_goal_diff_kurtosis": 0.0,
            f"{prefix}_points_skew": 0.0,
            f"{prefix}_points_kurtosis": 0.0,
            f"{prefix}_high_scoring_rate": 0.0,
            f"{prefix}_low_scoring_rate": 0.0,
            f"{prefix}_blowout_rate": 0.0,
            f"{prefix}_big_win_rate": 0.0,
            f"{prefix}_heavy_loss_rate": 0.0,
        }
    recent = team_df.tail(int(window)).copy()
    weights = np.linspace(1.0, 1.0 + max(len(recent) - 1, 0) * 0.12, num=len(recent)) if len(recent) else np.array([1.0])
    total_goals = recent["GF"] + recent["GA"]
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
        f"{prefix}_goals_for_skew": series_skew(recent["GF"]),
        f"{prefix}_goals_for_kurtosis": series_kurtosis(recent["GF"]),
        f"{prefix}_goals_against_skew": series_skew(recent["GA"]),
        f"{prefix}_goals_against_kurtosis": series_kurtosis(recent["GA"]),
        f"{prefix}_goal_diff_skew": series_skew(recent["GoalDiff"]),
        f"{prefix}_goal_diff_kurtosis": series_kurtosis(recent["GoalDiff"]),
        f"{prefix}_points_skew": series_skew(recent["Points"]),
        f"{prefix}_points_kurtosis": series_kurtosis(recent["Points"]),
        f"{prefix}_high_scoring_rate": float((total_goals >= 4.0).mean()),
        f"{prefix}_low_scoring_rate": float((total_goals <= 1.0).mean()),
        f"{prefix}_blowout_rate": float((recent["GoalDiff"].abs() >= 3.0).mean()),
        f"{prefix}_big_win_rate": float((recent["GoalDiff"] >= 3.0).mean()),
        f"{prefix}_heavy_loss_rate": float((recent["GoalDiff"] <= -3.0).mean()),
    }


def linear_trend_features(team_df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for column, safe in (("Points", "points"), ("GF", "goals_for"), ("GA", "goals_against"), ("GoalDiff", "goal_diff")):
        slope, r2 = linear_slope_r2(team_df[column] if column in team_df.columns else pd.Series(dtype=float))
        output[f"{prefix}_{safe}_slope"] = slope
        output[f"{prefix}_{safe}_r2"] = r2
    return output


def linear_slope_r2(values: pd.Series) -> Tuple[float, float]:
    series = pd.to_numeric(values, errors="coerce").dropna().astype(float).reset_index(drop=True)
    if series.shape[0] < 2:
        return 0.0, 0.0
    x = np.arange(series.shape[0], dtype=float)
    y = series.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return float(slope), float(max(min(r2, 1.0), 0.0))


def series_skew(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    return float(series.skew()) if series.shape[0] >= 3 and np.isfinite(series.skew()) else 0.0


def series_kurtosis(values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    return float(series.kurt()) if series.shape[0] >= 4 and np.isfinite(series.kurt()) else 0.0


def ema_value(values: Iterable[float], alpha: float = 0.5) -> float:
    iterator = [float(value) for value in values]
    if not iterator:
        return 0.0
    ema = iterator[0]
    alpha = min(max(float(alpha), 0.0), 1.0)
    for value in iterator[1:]:
        ema = alpha * value + (1.0 - alpha) * ema
    return float(ema)


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
        goal_diffs = [g1 - g2 for _, g1, g2 in ordered_matches]
        points = [3.0 if g1 > g2 else 1.0 if g1 == g2 else 0.0 for _, g1, g2 in ordered_matches]
        total_goals = [g1 + g2 for _, g1, g2 in ordered_matches]
        recency_weights = np.linspace(1.0, 1.0 + max(len(ordered_matches) - 1, 0) * 0.2, num=len(ordered_matches)) if ordered_matches else np.array([1.0])
        result_counts = [
            sum(1 for _, g1, g2 in ordered_matches if g1 > g2),
            sum(1 for _, g1, g2 in ordered_matches if g1 == g2),
            sum(1 for _, g1, g2 in ordered_matches if g2 > g1),
        ]
        goal_total_counts = pd.Series(total_goals).value_counts().to_dict() if total_goals else {}
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
            "recency_weighted_points": float(np.average(points, weights=recency_weights)) if points else 0.0,
            "recency_weighted_goal_diff": float(np.average(goal_diffs, weights=recency_weights)) if goal_diffs else 0.0,
            "ema_points": float(ema_value(points, alpha=0.55)),
            "ema_goal_diff": float(ema_value(goal_diffs, alpha=0.55)),
            "result_entropy_1x2": probability_entropy(normalize_probability_vector(result_counts)),
            "result_entropy_1x2_norm": probability_entropy(normalize_probability_vector(result_counts)) / math.log(3.0),
            "goals_entropy": probability_entropy(normalize_probability_vector(goal_total_counts.values())),
            "high_scoring_rate": float(np.mean([goals >= 4.0 for goals in total_goals])) if total_goals else 0.0,
            "low_scoring_rate": float(np.mean([goals <= 1.0 for goals in total_goals])) if total_goals else 0.0,
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


def labeled_validation_row_count(normalized: Dict[str, Any]) -> int:
    validation = normalized.get("validation", pd.DataFrame())
    return int(validation.shape[0]) if isinstance(validation, pd.DataFrame) else 0


def labeled_test_row_count(normalized: Dict[str, Any]) -> int:
    return int(normalized["test"].shape[0] or normalized["team_test"].shape[0])


def evaluation_strategy(normalized: Dict[str, Any]) -> str:
    split_policy = str(normalized.get("split_policy") or "")
    if split_policy in {EVAL_STRATEGY_LAST_30, SPLIT_POLICY_VALIDATION_LAST_30}:
        return EVAL_STRATEGY_LAST_30
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
    params = worldcup_model_defaults(model_key)
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
        "market_index": int(payload.get("market_index") or 0),
        "market_total": int(payload.get("market_total") or 0),
        "trials_per_market": int(payload.get("trials_per_market") or 0),
        "total_trial_budget": int(payload.get("total_trial_budget") or 0),
        "trial_offset": int(payload.get("trial_offset") or 0),
    }


def default_model_id(model_key: str, target: str) -> str:
    short_model = {
        "xgboost": "xgb",
        "lightgbm": "lgbm",
        "catboost": "cat",
        "ngboost": "ngb",
    }.get(model_key, model_key)
    short_target = "hibrido" if target == "dual_markets" else "goals" if target == GOALS_DISTRIBUTION_TARGET else target.replace("over_under_", "uo") if is_over_under_target(target) else "result"
    return f"mundial-{short_model}-{short_target}"


def normalize_worldcup_model_id(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "-", text).strip("-._")
    if not text:
        raise WorldCupTrainingError("El nombre/id del modelo Mundial es obligatorio.")
    if len(text) > 80:
        raise WorldCupTrainingError("El id del modelo Mundial debe tener 80 caracteres o menos.")
    return text


def normalize_over_under_target_key(value: Any) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    compact = re.sub(r"[^a-z0-9]+", "", key)
    aliases = {
        "overunder": "over_under_25",
        "overunder25": "over_under_25",
        "ou25": "over_under_25",
        "uo25": "over_under_25",
        "uo025": "over_under_25",
        "u025": "over_under_25",
    }
    if compact in aliases:
        return aliases[compact]
    match = re.search(r"(?:over_under|overunder|ou|uo|u_o)_?0?([0-3])_?5$", key)
    if not match:
        match = re.search(r"(?:overunder|ou|uo)0?([0-3])5$", compact)
    if match:
        target = f"over_under_{match.group(1)}5"
        return target if target in OVER_UNDER_TARGET_LINES else ""
    return key if key in OVER_UNDER_TARGET_LINES else ""


def normalize_training_target(value: Any) -> str:
    key = str(value or "result").strip().lower().replace("-", "_")
    if key in {"goals", "goals_distribution", "goal_distribution", "total_goals", "distribucion_goles"}:
        return GOALS_DISTRIBUTION_TARGET
    normalized_over_under = normalize_over_under_target_key(key)
    if normalized_over_under:
        return normalized_over_under
    return "result"


def normalize_market_mode(value: Any, target: str = "result") -> str:
    key = str(value or target or "result").strip().lower().replace("-", "_")
    if key in {"dual", "dual_markets", "both", "ambos", "all", "result_over_under_25", "result_uo25", "result_over_under"}:
        return "dual_markets"
    if key in {"goals", "goals_distribution", "goal_distribution", "total_goals", "distribucion_goles"}:
        return GOALS_DISTRIBUTION_TARGET
    normalized_over_under = normalize_over_under_target_key(key)
    if normalized_over_under:
        return normalized_over_under
    return "result"


def normalize_walk_forward_mode(value: Any) -> str:
    key = str(value or "none").strip().lower().replace("-", "_")
    if key in {"players", "with_players", "player_features"}:
        return "players"
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


def has_over_under_target(rows: pd.DataFrame, target: str = "over_under_25") -> bool:
    target_column = over_under_column_for_target(target)
    if target_column not in rows.columns:
        return False
    values = pd.to_numeric(rows[target_column], errors="coerce").dropna()
    return values.shape[0] > 1


def has_goals_distribution_target(rows: pd.DataFrame) -> bool:
    if not {"HG", "AG"}.issubset(rows.columns):
        return False
    totals = pd.to_numeric(rows["HG"], errors="coerce") + pd.to_numeric(rows["AG"], errors="coerce")
    totals = totals.dropna().astype(int).clip(lower=0, upper=TOTAL_GOALS_CAP)
    return totals.shape[0] > 1 and totals.nunique() >= 2


def api_football_records(bundle: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    if not isinstance(bundle, dict):
        return {}
    output: Dict[str, List[Dict[str, Any]]] = {}
    for key in ("fixtures", "statistics", "team_stats", "lineups", "injuries", "odds", "market_rows"):
        value = bundle.get(key, pd.DataFrame())
        output[key] = value.to_dict(orient="records") if isinstance(value, pd.DataFrame) and not value.empty else []
    return output


def api_football_dataframes(bundle: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    if not isinstance(bundle, dict):
        return {}
    output: Dict[str, pd.DataFrame] = {}
    for key in ("fixtures", "statistics", "team_stats", "lineups", "injuries", "odds", "market_rows"):
        value = bundle.get(key, pd.DataFrame())
        output[key] = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value or [])
    return output


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
    working = worldcup_result_due_fixture_rows(tournament)
    if working.empty:
        return pd.DataFrame(columns=working.columns)
    working["HG"] = pd.to_numeric(working["Goles 1"], errors="coerce")
    working["AG"] = pd.to_numeric(working["Goles 2"], errors="coerce")
    working = working[working["HG"].notna() & working["AG"].notna()].copy()
    return working.sort_values(["_date", "No."], kind="stable")


def worldcup_result_due_fixture_rows(tournament: Dict[str, Any]) -> pd.DataFrame:
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
        return pd.DataFrame(columns=["FixtureId", "Home", "Away", "HG", "AG", "Label", "OverUnder05", "OverUnder15", "OverUnder25", "OverUnder35", "Source"])
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
        return pd.DataFrame(columns=["FixtureId", "Home", "Away", "HG", "AG", "Label", "OverUnder05", "OverUnder15", "OverUnder25", "OverUnder35", "Source"])
    working["Label"] = working.apply(lambda row: label_from_goals(row["HG"], row["AG"]), axis=1)
    for line in TRAIN_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        working[f"OverUnder{suffix}"] = ((working["HG"] + working["AG"]) > line).astype(int)
    return pd.DataFrame({
        "FixtureId": working["No."].astype(str),
        "Date": pd.to_datetime(working["Fecha"], errors="coerce"),
        "Year": 2026,
        "Home": working["Equipo 1"].map(clean_team_name),
        "Away": working["Equipo 2"].map(clean_team_name),
        "HG": working["HG"].astype(float),
        "AG": working["AG"].astype(float),
        "Label": working["Label"].astype(str),
        "OverUnder05": working["OverUnder05"].astype(int),
        "OverUnder15": working["OverUnder15"].astype(int),
        "OverUnder25": working["OverUnder25"].astype(int),
        "OverUnder35": working["OverUnder35"].astype(int),
        "Source": "worldcup_2026_walk_forward",
        "is_worldcup_match": True,
        "tournament": "FIFA World Cup",
        "stage": working["Ronda"].astype(str),
        "group": working["Grupo"].astype(str),
        "knockout": working["Grupo"].astype(str).str.len().eq(0),
        "label_source": "worldcup_2026_walk_forward",
        "sample_weight": SAMPLE_WEIGHT_POLICY["worldcup"],
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
    if normalized_mode == "players":
        empty["warnings"] = ["El modo con jugadores requiere actualizar snapshots; el reentreno base usa result_only."]
        return empty
    if dataset_mode != "match_result":
        empty["warnings"] = ["El reentreno walk-forward solo aplica cuando el dataset internacional esta preparado como partidos (match_result)."]
        return empty
    completed = completed_worldcup_training_rows(tournament)
    if completed.empty:
        empty["warnings"] = ["No hay partidos 2026 con marcador oficial disponible para incorporar al reentreno."]
        return empty
    completed = sanitize_match_rows(completed)
    return {
        "mode": normalized_mode,
        "rows": completed.reset_index(drop=True),
        "fixture_ids": completed["FixtureId"].astype(str).tolist(),
        "added_rows": int(completed.shape[0]),
        "warnings": [],
    }


def walk_forward_refresh_state() -> Dict[str, Any]:
    tournament = fallback_tournament_2026()
    cache_2026 = CACHE_ROOT / "worldcup_2026.json"
    if cache_2026.exists():
        try:
            tournament, _ = load_tournament_2026(refresh=False)
        except Exception:
            tournament = fallback_tournament_2026()
    due = worldcup_result_due_fixture_rows(tournament)
    played = worldcup_played_fixture_rows(tournament)
    completed = completed_worldcup_training_rows(tournament)
    matches = read_optional_csv(WALK_FORWARD_MATCHES_FILE)
    fixture_features = read_fixture_feature_rows()
    snapshot_ids = set(matches["fixture_id"].astype(str)) if not matches.empty and "fixture_id" in matches.columns else set()
    player_ready_ids = player_ready_fixture_ids(fixture_features)
    included_result_ids = set()
    if not matches.empty and "fixture_id" in matches.columns:
        if "included_result_only_at" in matches.columns:
            included_result_ids = set(matches[matches["included_result_only_at"].fillna("").astype(str).str.len() > 0]["fixture_id"].astype(str))
    due_ids = set(due["No."].astype(str)) if not due.empty else set()
    played_ids = set(played["No."].astype(str)) if not played.empty else set()
    completed_ids = set(completed["FixtureId"].astype(str)) if not completed.empty else set()
    needs_player_snapshot_ids = sorted(completed_ids - player_ready_ids)
    pending_result_ids = sorted(due_ids - completed_ids)
    ready_result_ids = sorted(completed_ids - included_result_ids)
    latest_fixture = ""
    if not played.empty:
        latest = played.iloc[-1]
        latest_fixture = f"{latest.get('Equipo 1', '')} vs {latest.get('Equipo 2', '')} ({latest.get('Fecha', '')})"
    note = ""
    if ready_result_ids:
        note = f"Hay {len(ready_result_ids)} partidos listos para reentreno base."
    elif needs_player_snapshot_ids:
        note = f"Hay {len(needs_player_snapshot_ids)} partidos con snapshot de jugadores pendiente."
    elif pending_result_ids:
        note = f"Hay {len(pending_result_ids)} partidos con fecha pasada esperando marcador final."
    return {
        "played_matches": int(len(played_ids)),
        "date_passed_matches": int(len(due_ids)),
        "completed_results": int(len(completed_ids)),
        "snapshot_matches": int(len(snapshot_ids)),
        "player_snapshot_matches": int(len(player_ready_ids)),
        "stale_match_ids": needs_player_snapshot_ids,
        "needs_player_snapshot_ids": needs_player_snapshot_ids,
        "pending_result_ids": pending_result_ids,
        "pending_results": int(len(pending_result_ids)),
        "ready_result_ids": ready_result_ids,
        "ready_result_only": int(len(ready_result_ids)),
        "needs_player_snapshot": int(len(needs_player_snapshot_ids)),
        "requires_reload": bool(needs_player_snapshot_ids),
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


def split_weights_like_safe_train_eval(sample_weight: Optional[pd.Series], total_length: int, fit_length: int) -> Optional[pd.Series]:
    if sample_weight is None:
        return None
    weights = align_sample_weights(pd.Series(sample_weight), total_length)
    if int(fit_length) >= int(total_length):
        return weights
    return weights.iloc[:int(fit_length)].reset_index(drop=True)


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


def encode_target_labels(y: pd.Series, target: str) -> Tuple[pd.Series, List[Any]]:
    if is_over_under_target(target):
        values = pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce").fillna(0).astype(int)
        values = values.clip(lower=0, upper=1)
        return values.astype(int), [0, 1]
    return encode_labels(y)


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
        sample_weight: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    y_values = pd.Series(y_train).dropna()
    if y_values.nunique() < 2:
        classes = list(range(max(int(num_classes or 1), 1)))
        constant = int(y_values.iloc[0]) if not y_values.empty else classes[0]
        classifier = ConstantProbabilityClassifier(constant_class=constant, classes=classes).fit(x_train, y_train)
        return {
            "classifier": classifier,
            "device": "cpu",
            "warnings": ["El target entrenado contiene una sola clase en train; se uso un clasificador constante para conservar el mercado."],
        }
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
        fit_warnings = fit_classifier(classifier, x_train, y_train, sample_weight=sample_weight)
        finalize_classifier_for_inference(classifier, model_key=model_key, trained_device=device)
        return {"classifier": classifier, "device": device, "warnings": [*device_warnings, *fit_warnings]}
    except Exception as exc:
        if device == "cuda":
            if requested_device == "cuda":
                raise WorldCupTrainingError(
                    f"CUDA fue solicitado explicitamente, pero el entrenamiento fallo "
                    f"({exc.__class__.__name__}: {exc})."
                ) from exc
            fallback = build_worldcup_classifier(
                model_key=model_key,
                params=params,
                n_jobs=n_jobs,
                device="cpu",
                seed=seed,
                num_classes=num_classes,
            )
            fit_warnings = fit_classifier(fallback, x_train, y_train, sample_weight=sample_weight)
            finalize_classifier_for_inference(fallback, model_key=model_key, trained_device="cpu")
            return {
                "classifier": fallback,
                "device": "cpu",
                "warnings": [*device_warnings, *fit_warnings, f"CUDA fallo durante fit ({exc.__class__.__name__}); se reintento en CPU."],
            }
        raise


def fit_classifier(classifier, x_train: pd.DataFrame, y_train: pd.Series, sample_weight: Optional[pd.Series] = None) -> List[str]:
    weights = pd.to_numeric(pd.Series(sample_weight), errors="coerce").fillna(1.0).astype(float) if sample_weight is not None else pd.Series(dtype=float)
    if sample_weight is None or weights.empty:
        classifier.fit(x_train, y_train)
        return []
    weights = align_sample_weights(weights, len(pd.Series(y_train)))
    try:
        classifier.fit(x_train, y_train, sample_weight=weights.to_numpy(dtype=float))
        return []
    except TypeError:
        classifier.fit(x_train, y_train)
        return ["El clasificador seleccionado no acepta sample_weight; se entreno sin ponderacion de filas."]


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
            "tree_method": "hist",
        }
        if num_classes > 2:
            kwargs.update({"objective": "multi:softprob", "num_class": num_classes})
        else:
            kwargs["objective"] = "binary:logistic"
        if device == "cuda":
            kwargs.update(xgboost_cuda_params())
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
            **lightgbm_device_params(device),
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
            **catboost_device_params(device),
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
        sample_weight: Optional[pd.Series] = None,
        x_validation: Optional[pd.DataFrame] = None,
        y_validation: Optional[pd.Series] = None,
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

    explicit_validation = (
        isinstance(x_validation, pd.DataFrame)
        and not x_validation.empty
        and y_validation is not None
        and not pd.Series(y_validation).dropna().empty
    )
    if explicit_validation:
        x_fit = x_train
        y_fit = pd.Series(y_train).reset_index(drop=True)
        x_eval = x_validation.copy()
        y_eval = pd.Series(y_validation).reset_index(drop=True)
        weight_fit = align_sample_weights(sample_weight, len(y_fit)) if sample_weight is not None else None
        validation_source = "temporal_validation"
    else:
        x_fit, x_eval, y_fit, y_eval = safe_train_eval_split(
            x_train,
            y_train,
            test_size=0.25,
            random_state=config["seed"],
        )
        weight_fit = split_weights_like_safe_train_eval(sample_weight, len(y_train), len(y_fit)) if sample_weight is not None else None
        validation_source = "internal_split"
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
                sample_weight=weight_fit,
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
    tuning_started_at = datetime.now(timezone.utc)
    total_trial_budget = int(config.get("total_trial_budget") or total_trials)
    trial_offset = int(config.get("trial_offset") or 0)

    def progress_after_trial(study_obj, trial):
        current = min(len(study_obj.trials), total_trials)
        overall_current = trial_offset + current
        elapsed_seconds = int((datetime.now(timezone.utc) - tuning_started_at).total_seconds())
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
            f"Fine-tuning {label} {current}/{total_trials}",
            best_value=best_value,
            best_trial=best_trial,
            last_state=getattr(trial.state, "name", str(trial.state)),
            market=label,
            model_type=config["model_type"],
            market_index=config.get("market_index"),
            market_total=config.get("market_total"),
            trials_per_market=config.get("trials_per_market") or total_trials,
            total_trial_budget=total_trial_budget,
            overall_trial_current=overall_current,
            elapsed_seconds=elapsed_seconds,
        )

    emit_training_progress(
        progress_callback,
        "tuning",
        0,
        total_trials,
        f"Fine-tuning {label} 0/{total_trials}",
        market=label,
        model_type=config["model_type"],
        market_index=config.get("market_index"),
        market_total=config.get("market_total"),
        trials_per_market=config.get("trials_per_market") or total_trials,
        total_trial_budget=total_trial_budget,
        overall_trial_current=trial_offset,
        elapsed_seconds=0,
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
        "market_index": config.get("market_index"),
        "market_total": config.get("market_total"),
        "trials_per_market": config.get("trials_per_market") or total_trials,
        "total_trial_budget": total_trial_budget,
        "trial_offset": trial_offset,
        "validation_source": validation_source,
        "elapsed_seconds": int((datetime.now(timezone.utc) - tuning_started_at).total_seconds()),
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


def feature_inventory_payload(
        feature_columns: List[str],
        x_train: Optional[pd.DataFrame] = None,
        x_eval: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    rows = []
    families: Dict[str, int] = {}
    for column in feature_columns:
        train_series = pd.to_numeric(x_train[column], errors="coerce") if x_train is not None and column in x_train.columns else pd.Series(dtype=float)
        eval_series = pd.to_numeric(x_eval[column], errors="coerce") if x_eval is not None and column in x_eval.columns else pd.Series(dtype=float)
        family = feature_family(column)
        families[family] = families.get(family, 0) + 1
        rows.append({
            "feature": column,
            "family": family,
            "train_non_zero_rate": round(float((train_series.fillna(0.0) != 0.0).mean()) if not train_series.empty else 0.0, 4),
            "eval_non_zero_rate": round(float((eval_series.fillna(0.0) != 0.0).mean()) if not eval_series.empty else 0.0, 4),
            "train_null_rate": round(float(train_series.isna().mean()) if not train_series.empty else 0.0, 4),
            "eval_null_rate": round(float(eval_series.isna().mean()) if not eval_series.empty else 0.0, 4),
            "train_variance": round(float(train_series.fillna(0.0).var(ddof=0)) if not train_series.empty else 0.0, 8),
        })
    return {
        "feature_count": len(feature_columns),
        "families": [{"family": key, "count": value} for key, value in sorted(families.items())],
        "features": rows,
    }


def feature_family(column: str) -> str:
    for prefix in (
        "api_football_",
        "history_",
        "market_",
        "model_vs_market_",
        "market_vs_model_",
        "model_market_",
        "poisson_",
        "prob_",
        "model_",
        "kaggle_",
        "h2h_",
        "fixture_",
        "stage_",
        "venue_",
        "rating_",
        "attack_",
        "defense_",
        "shrinkage_",
        "dc_",
        "total_",
        "goal_",
    ):
        if column.startswith(prefix):
            return prefix.rstrip("_")
    return str(column).split("_", 1)[0]


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
        return {"result": {}, "over_under_ml": {}, "over_under_25": {}, "notes": ["Modelo internacional no entrenado."]}
    if record.get("bundle") and record.get("market_models"):
        return predict_bundle_ml_outputs(base_model, home, away, record, fixture_id=fixture_id)
    return predict_single_record_ml_outputs(base_model, home, away, record, fixture_id=fixture_id)


def predict_bundle_ml_outputs(base_model: WorldCupModel, home: str, away: str, record: Dict[str, Any], fixture_id: Optional[Any] = None) -> Dict[str, Any]:
    bundle_id = str(record.get("model_id") or active_worldcup_model_id() or "")
    bundle_name = str(record.get("model_name") or bundle_id or "Bundle Mundial")
    market_models = record.get("market_models") or {}
    result_output = {"result": {}, "notes": []}
    over_under_ml: Dict[str, float] = {}
    over_under_line_sources: Dict[str, Dict[str, str]] = {}
    goal_distribution_ml: Dict[str, float] = {}
    goal_distribution_totals: Dict[str, float] = {}
    market_model_names: Dict[str, str] = {}
    preferred_sources = record.get("over_under_preferred_sources") or {}

    result_id = market_models.get("result")
    if result_id:
        result_record = load_hybrid_model(result_id)
        if result_record:
            result_output = predict_single_record_ml_outputs(base_model, home, away, result_record, fixture_id=fixture_id)
            if result_output.get("result"):
                market_model_names["result"] = result_output.get("model_name", "")

    notes = []
    notes.extend(result_output.get("notes", []))

    for target in OVER_UNDER_MARKET_TARGETS:
        child_id = market_models.get(target)
        if not child_id:
            continue
        child_record = load_hybrid_model(child_id)
        if not child_record:
            continue
        child_output = predict_single_record_ml_outputs(base_model, home, away, child_record, fixture_id=fixture_id)
        child_totals = child_output.get("over_under_ml") or child_output.get("over_under_25") or {}
        line = OVER_UNDER_TARGET_LINES[target]
        suffix = total_line_suffix(line)
        over_key = f"over{suffix}"
        under_key = f"under{suffix}"
        if over_key in child_totals and under_key in child_totals:
            over_under_ml[over_key] = float(child_totals[over_key])
            over_under_ml[under_key] = float(child_totals[under_key])
            market_model_names[target] = child_output.get("model_name", "")
            over_under_line_sources[target] = {
                "kind": "binary_ml",
                "model_name": str(child_output.get("model_name", "")),
            }
        notes.extend(child_output.get("notes", []))

    goals_id = market_models.get(GOALS_DISTRIBUTION_TARGET)
    if goals_id:
        goals_record = load_hybrid_model(goals_id)
        if goals_record:
            goals_output = predict_single_record_ml_outputs(base_model, home, away, goals_record, fixture_id=fixture_id)
            goal_distribution_ml = goals_output.get("goal_distribution") or {}
            goal_distribution_totals = goals_output.get("over_under_ml") or goals_output.get("over_under_25") or {}
            if goal_distribution_ml:
                market_model_names[GOALS_DISTRIBUTION_TARGET] = goals_output.get("model_name", "")
            notes.extend(goals_output.get("notes", []))
            for target in OVER_UNDER_MARKET_TARGETS:
                preferred_kind = (preferred_sources.get(target) or {}).get("kind")
                if target in over_under_line_sources and preferred_kind != "goal_distribution_ml":
                    continue
                line = OVER_UNDER_TARGET_LINES[target]
                suffix = total_line_suffix(line)
                over_key = f"over{suffix}"
                under_key = f"under{suffix}"
                if over_key in goal_distribution_totals and under_key in goal_distribution_totals:
                    over_under_ml[over_key] = float(goal_distribution_totals[over_key])
                    over_under_ml[under_key] = float(goal_distribution_totals[under_key])
                    market_model_names[target] = goals_output.get("model_name", "")
                    over_under_line_sources[target] = {
                        "kind": "goal_distribution_ml",
                        "model_name": str(goals_output.get("model_name", "")),
                    }

    return {
        "result": result_output.get("result", {}),
        "over_under_ml": over_under_ml,
        "over_under_25": over_under_ml,
        "goal_distribution_ml": goal_distribution_ml,
        "model_id": bundle_id,
        "model_name": bundle_name,
        "market_model_ids": {key: value for key, value in market_models.items() if value},
        "market_model_names": market_model_names,
        "over_under_line_sources": over_under_line_sources,
        "notes": unique_strings(notes),
    }


def predict_single_record_ml_outputs(base_model: WorldCupModel, home: str, away: str, record: Dict[str, Any], fixture_id: Optional[Any] = None) -> Dict[str, Any]:
    active_id = str(record.get("model_id") or active_worldcup_model_id() or "")
    model_name = str(record.get("model_name") or active_id or record.get("model_label", "Internacional"))
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
            "over_under_ml": {},
            "over_under_25": {},
            "model_id": active_id,
            "model_name": model_name,
            "notes": [
                f"Modelo {model_name} team-strength aplicado ({record.get('target_column', '')}).",
                "U/O 0.5-3.5 viene de Poisson porque el dataset de equipos no contiene goles de partido.",
            ],
        }
    team_features = pd.DataFrame(record.get("team_features", []))
    history_team_features = pd.DataFrame(record.get("history_team_features", []))
    matchup_features = pd.DataFrame(record.get("matchup_features", []))
    market_rows = pd.DataFrame(record.get("market_data", []))
    qualifier_rows = pd.DataFrame(record.get("qualifier_matches", []))
    api_football = api_football_dataframes(record.get("api_football", {}))
    qualifier_features = qualifier_feature_table(
        qualifier_rows,
        reference_date=HISTORY_REFERENCE_DATE,
        teams=[home, away],
    )
    api_football_features = api_football_feature_table(
        api_football.get("team_stats", pd.DataFrame()),
        reference_date=HISTORY_REFERENCE_DATE,
        teams=[home, away],
        lineups=api_football.get("lineups", pd.DataFrame()),
        injuries=api_football.get("injuries", pd.DataFrame()),
    )
    recent15_features = recent15_feature_table(
        load_international_matches(required=False),
        teams=[home, away],
        before_date=HISTORY_REFERENCE_DATE,
        base_model=base_model,
    )
    fixture_feature_rows = pd.DataFrame()
    x = pd.DataFrame([
        match_feature_row(
            base_model,
            team_features,
            home,
            away,
            history_team_features=history_team_features,
            matchup_features=matchup_features,
            market_rows=market_rows,
            qualifier_features=qualifier_features,
            api_football_features=api_football_features,
            recent15_features=recent15_features,
            fixture_feature_rows=fixture_feature_rows,
            fixture_id=fixture_id,
            match_date=None,
            match_year=2026,
            fixture_context={"FixtureId": fixture_id, "Year": 2026},
            dc_rho=float(record.get("dc_rho", 0.0) or 0.0),
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
    if target == GOALS_DISTRIBUTION_TARGET:
        distribution = goal_distribution_from_probabilities(labels, probabilities)
        over_under = total_line_probabilities_from_distribution(distribution)
        return {
            "result": {},
            "over_under_ml": over_under,
            "over_under_25": over_under,
            "goal_distribution": {key: value for key, value in distribution.items()},
            "model_id": active_id,
            "model_name": model_name,
            "over_under_line_sources": {
                target: {"kind": "goal_distribution_ml", "model_name": model_name}
                for target in OVER_UNDER_MARKET_TARGETS
            },
            "notes": [f"Modelo {model_name} aplicado a distribucion de goles y U/O multi-linea."],
        }
    if is_over_under_target(target):
        line = OVER_UNDER_TARGET_LINES[target]
        suffix = total_line_suffix(line)
        under_key = f"under{suffix}"
        over_key = f"over{suffix}"
        output = {under_key: 0.0, over_key: 0.0}
        for label, probability in zip(labels, probabilities):
            if str(label) in {"1", "True", "true"}:
                output[over_key] += float(probability)
            else:
                output[under_key] += float(probability)
        total = max(sum(output.values()), 1e-9)
        normalized_output = {key: value / total for key, value in output.items()}
        return {
            "result": {},
            "over_under_ml": normalized_output,
            "over_under_25": normalized_output if target == "over_under_25" else {},
            "model_id": active_id,
            "model_name": model_name,
            "over_under_line_sources": {target: {"kind": "binary_ml", "model_name": model_name}},
            "notes": [f"Modelo {model_name} aplicado a {market_label_for_progress(target)}."],
        }
    output = {label: 0.0 for label in TARGET_LABELS}
    for label, probability in zip(labels, probabilities):
        if label in output:
            output[label] = float(probability)
    total = max(sum(output.values()), 1e-9)
    return {
        "result": {key: value / total for key, value in output.items()},
        "over_under_ml": {},
        "over_under_25": {},
        "model_id": active_id,
        "model_name": model_name,
        "notes": [
            f"Modelo {model_name} aplicado a 1X2.",
            "U/O 0.5-3.5 viene de Poisson cuando no existen hijos U/O entrenados.",
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
    labels = sorted(set(base_probs) | set(ml_probs))
    for label in labels:
        output[label] = base_probs.get(label, 0.0) * (1.0 - weight) + ml_probs.get(label, base_probs.get(label, 0.0)) * weight
    for line in TRAIN_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        over_key = f"over{suffix}"
        under_key = f"under{suffix}"
        total = max(output.get(over_key, 0.0) + output.get(under_key, 0.0), 1e-9)
        output[over_key] = output.get(over_key, 0.0) / total
        output[under_key] = output.get(under_key, 0.0) / total
    return output


def total_line_probabilities_from_probs(probabilities: Dict[str, float]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for line in TRAIN_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        over_key = f"over{suffix}"
        under_key = f"under{suffix}"
        over_value = float(probabilities.get(over_key, 0.0) or 0.0)
        under_value = float(probabilities.get(under_key, 1.0 - over_value) or 0.0)
        total = max(over_value + under_value, 1e-9)
        output[over_key] = over_value / total
        output[under_key] = under_value / total
    return output


def goal_distribution_from_probabilities(labels: List[str], probabilities: np.ndarray) -> Dict[str, float]:
    output = {str(goal): 0.0 for goal in range(TOTAL_GOALS_CAP)}
    output[f"{TOTAL_GOALS_CAP}+"] = 0.0
    for label, probability in zip(labels, probabilities):
        try:
            goals = int(float(label))
        except (TypeError, ValueError):
            continue
        key = f"{TOTAL_GOALS_CAP}+" if goals >= TOTAL_GOALS_CAP else str(max(goals, 0))
        output[key] = output.get(key, 0.0) + float(probability)
    total = max(sum(output.values()), 1e-9)
    return {key: value / total for key, value in output.items()}


def total_line_probabilities_from_distribution(distribution: Dict[str, float]) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for line in TRAIN_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        over = 0.0
        under = 0.0
        for key, probability in distribution.items():
            goals = TOTAL_GOALS_CAP if str(key).endswith("+") else int(float(key))
            if goals > line:
                over += float(probability)
            else:
                under += float(probability)
        total = max(over + under, 1e-9)
        output[f"over{suffix}"] = over / total
        output[f"under{suffix}"] = under / total
    return output


def market_sources_payload(result_ml: Dict[str, float], over_under_ml: Dict[str, float], ml_outputs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    model_name = str(ml_outputs.get("model_name") or "").strip()
    market_model_names = ml_outputs.get("market_model_names") or {}
    result_model_name = str(market_model_names.get("result") or model_name).strip()
    result_uses_ml = bool(result_ml)
    sources = {
        "result": {
            "label": "1X2",
            "source": "ML + Poisson" if result_uses_ml else "Poisson",
            "uses_ml": result_uses_ml,
            "model_name": result_model_name if result_uses_ml else "",
            "detail": f"1X2 mezcla el modelo {result_model_name} con Elo/Poisson." if result_uses_ml else "1X2 calculado solo con Elo/Poisson.",
        },
    }
    line_sources = ml_outputs.get("over_under_line_sources") or {}
    for target in OVER_UNDER_MARKET_TARGETS:
        line = OVER_UNDER_TARGET_LINES[target]
        suffix = total_line_suffix(line)
        over_key = f"over{suffix}"
        under_key = f"under{suffix}"
        source_meta = line_sources.get(target, {}) if isinstance(line_sources, dict) else {}
        model_for_line = str(source_meta.get("model_name") or market_model_names.get(target) or model_name).strip()
        source_kind = str(source_meta.get("kind") or "").strip()
        uses_ml = over_key in over_under_ml and under_key in over_under_ml
        source_detail = "modelo binario" if source_kind == "binary_ml" else "distribucion auxiliar" if source_kind == "goal_distribution_ml" else "ML"
        sources[target] = {
            "label": f"U/O {line:.1f}",
            "source": "ML + Poisson" if uses_ml else "Poisson",
            "uses_ml": uses_ml,
            "model_name": model_for_line if uses_ml else "",
            "detail": f"U/O {line:.1f} mezcla {source_detail} {model_for_line} con Poisson." if uses_ml else f"U/O {line:.1f} calculado con Poisson.",
        }
    return sources


def child_market_model_id(bundle_id: str, market: str) -> str:
    suffix = "__goals" if market == GOALS_DISTRIBUTION_TARGET else f"__uo{total_line_suffix(OVER_UNDER_TARGET_LINES[market])}" if is_over_under_target(market) else "__result"
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
        "calibration": record.get("calibration", result.get("calibration", {})),
        "classes": record.get("classes", []),
        "eval_strategy": record.get("eval_strategy", result.get("eval_strategy", "")),
        "train_rows": int(result.get("train_rows", 0) or 0),
        "validation_rows": int(result.get("validation_rows", record.get("validation_rows", 0)) or 0),
        "eval_rows": int(result.get("eval_rows", 0) or 0),
        "tuning": record.get("tuning", result.get("tuning", {})),
        "tuning_trace": record.get("tuning_trace", result.get("tuning_trace", {})),
        "top_features": record.get("top_features", []),
        "warnings": record.get("warnings", []),
        "hardware": record.get("hardware", result.get("hardware", {})),
        "feature_count": len(record.get("feature_columns", result.get("features", [])) or []),
        "target_worldcup_year": record.get("target_worldcup_year", result.get("target_worldcup_year", "")),
        "benchmark_worldcup_year": record.get("benchmark_worldcup_year", result.get("benchmark_worldcup_year", result.get("final_test_year", ""))),
        "benchmark_policy": record.get("benchmark_policy", result.get("benchmark_policy", "")),
        "final_test_year": record.get("final_test_year", result.get("final_test_year", "")),
        "split_policy": record.get("split_policy", result.get("split_policy", "")),
    }


def preferred_over_under_sources(markets: Dict[str, Dict[str, Any]], goals_record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    derived = goals_record.get("derived_total_markets", {}) if goals_record else {}
    output: Dict[str, Dict[str, Any]] = {}
    for target in OVER_UNDER_MARKET_TARGETS:
        binary = markets.get(target, {})
        binary_score = eval_f1_from_metrics(binary.get("metrics", {}))
        derived_score = eval_f1_from_metrics((derived.get(target, {}) or {}).get("metrics", {}))
        if derived_score is not None and (binary_score is None or derived_score > binary_score):
            output[target] = {
                "kind": "goal_distribution_ml",
                "binary_eval_f1": binary_score,
                "goal_distribution_eval_f1": derived_score,
            }
        else:
            output[target] = {
                "kind": "binary_ml",
                "binary_eval_f1": binary_score,
                "goal_distribution_eval_f1": derived_score,
            }
    return output


def eval_f1_from_metrics(metrics: Dict[str, Any]) -> Optional[float]:
    try:
        value = (metrics.get("eval") or {}).get("F1")
        return float(value) if value is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def goal_line_training_summary(record: Dict[str, Any], result: Dict[str, Any], line_suffix: str = "25") -> Dict[str, Any]:
    derived_key = f"over_under_{line_suffix}"
    derived = (record.get("derived_total_markets") or {}).get(derived_key, {})
    summary = market_training_summary(record, result, derived.get("label") or goal_line_label_from_suffix(line_suffix))
    if derived:
        summary["metrics"] = derived.get("metrics", {})
        summary["confusion_matrix"] = derived.get("confusion_matrix", {})
    summary["effective_target"] = derived_key
    summary["derived_from"] = GOALS_DISTRIBUTION_TARGET
    summary["available_lines"] = sorted((record.get("derived_total_markets") or {}).keys())
    return summary


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
        "status": "ok" if "result" in market_models else "info",
        "detail": "Entrena 1X2, U/O 0.5-3.5 como clasificadores binarios y distribucion de goles como apoyo auxiliar cuando existe.",
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


def classification_metrics_from_predictions(
        y_train,
        y_train_pred,
        y_eval,
        y_eval_pred,
        y_train_proba: Optional[np.ndarray] = None,
        y_eval_proba: Optional[np.ndarray] = None,
        classes: Optional[List[Any]] = None,
        target: str = "result",
        x_train: Optional[pd.DataFrame] = None,
        x_eval: Optional[pd.DataFrame] = None,
) -> Dict[str, Dict[str, float]]:
    train_row = metric_row(y_train, y_train_pred)
    eval_row = metric_row(y_eval, y_eval_pred)
    if y_train_proba is not None and classes is not None:
        train_row.update(calibration_metric_row(y_train, y_train_proba, classes, target=target, x=x_train))
    if y_eval_proba is not None and classes is not None:
        eval_row.update(calibration_metric_row(y_eval, y_eval_proba, classes, target=target, x=x_eval))
    return {
        "train": train_row,
        "eval": eval_row,
    }


def metric_row(y_true, y_pred) -> Dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "F1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0.0)), 3),
        "Precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0.0)), 3),
        "Recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0.0)), 3),
    }


def calibration_metric_row(
        y_true,
        probabilities: np.ndarray,
        classes: List[Any],
        target: str = "result",
        x: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    payload = probability_calibration_summary(y_true, probabilities, classes)
    row = {
        "Brier": round(float(payload.get("brier", 0.0)), 4),
        "LogLoss": round(float(payload.get("log_loss", 0.0)), 4),
        "ECE": round(float(payload.get("ece", 0.0)), 4),
    }
    market = market_calibration_summary(y_true, classes, target=target, x=x)
    if market.get("available"):
        row["MarketBrier"] = round(float(market.get("brier", 0.0)), 4)
        row["ModelMinusMarketBrier"] = round(row["Brier"] - row["MarketBrier"], 4)
    return row


def calibration_payload(
        y_train,
        y_train_proba: np.ndarray,
        y_eval,
        y_eval_proba: np.ndarray,
        classes: List[Any],
        target: str,
        x_train: Optional[pd.DataFrame] = None,
        x_eval: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    train_summary = probability_calibration_summary(y_train, y_train_proba, classes)
    eval_summary = probability_calibration_summary(y_eval, y_eval_proba, classes)
    train_summary["market"] = market_calibration_summary(y_train, classes, target=target, x=x_train)
    eval_summary["market"] = market_calibration_summary(y_eval, classes, target=target, x=x_eval)
    return {
        "target": target,
        "classes": [str(item) for item in classes],
        "train": train_summary,
        "eval": eval_summary,
    }


def probability_calibration_summary(y_true, probabilities: np.ndarray, classes: List[Any], bins: int = 10) -> Dict[str, Any]:
    y_series = pd.to_numeric(pd.Series(y_true).reset_index(drop=True), errors="coerce")
    proba = normalize_probability_matrix(probabilities, len(classes))
    valid = y_series.notna() & y_series.astype(float).between(0, max(len(classes) - 1, 0))
    if not valid.any() or proba.shape[0] == 0:
        return {"available": False, "rows": 0, "brier": 0.0, "log_loss": 0.0, "ece": 0.0, "reliability_bins": []}
    y_int = y_series[valid].astype(int).to_numpy()
    proba = proba[valid.to_numpy()]
    one_hot = np.zeros_like(proba, dtype=float)
    one_hot[np.arange(len(y_int)), y_int] = 1.0
    brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))
    try:
        loss = float(log_loss(y_int, proba, labels=list(range(len(classes)))))
    except ValueError:
        loss = 0.0
    predicted = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    correct = (predicted == y_int).astype(float)
    reliability_bins = reliability_bins_payload(confidence, correct, bins=bins)
    ece = float(sum((item["count"] / max(len(y_int), 1)) * abs(item["accuracy"] - item["confidence"]) for item in reliability_bins))
    return {
        "available": True,
        "rows": int(len(y_int)),
        "brier": round(brier, 6),
        "log_loss": round(loss, 6),
        "ece": round(ece, 6),
        "reliability_bins": reliability_bins,
    }


def reliability_bins_payload(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    bins = max(1, int(bins or 10))
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        if index == bins - 1:
            mask = (confidence >= low) & (confidence <= high)
        else:
            mask = (confidence >= low) & (confidence < high)
        count = int(mask.sum())
        rows.append({
            "bin": index + 1,
            "low": round(low, 3),
            "high": round(high, 3),
            "count": count,
            "confidence": round(float(confidence[mask].mean()), 6) if count else 0.0,
            "accuracy": round(float(correct[mask].mean()), 6) if count else 0.0,
        })
    return rows


def normalize_probability_matrix(probabilities: np.ndarray, class_count: int) -> np.ndarray:
    class_count = max(int(class_count or 1), 1)
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim == 1:
        if class_count == 2:
            matrix = np.column_stack([1.0 - matrix, matrix])
        else:
            matrix = np.reshape(matrix, (-1, class_count))
    if matrix.shape[1] < class_count:
        matrix = np.pad(matrix, ((0, 0), (0, class_count - matrix.shape[1])), constant_values=0.0)
    matrix = matrix[:, :class_count]
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = np.clip(matrix, 0.0, None)
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals <= 0.0] = 1.0
    return matrix / totals


def market_calibration_summary(
        y_true,
        classes: List[Any],
        target: str,
        x: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    if x is None or x.empty:
        return {"available": False, "rows": 0}
    market_proba = market_probability_matrix_from_features(x, classes, target)
    if market_proba is None or market_proba.size == 0:
        return {"available": False, "rows": 0}
    has_market = np.sum(market_proba, axis=1) > 0.0
    y_series = pd.to_numeric(pd.Series(y_true).reset_index(drop=True), errors="coerce")
    valid = y_series.notna() & y_series.astype(float).between(0, max(len(classes) - 1, 0))
    valid_mask = valid.to_numpy() & has_market
    if not valid_mask.any():
        return {"available": False, "rows": 0}
    y_int = y_series[valid_mask].astype(int).to_numpy()
    proba = normalize_probability_matrix(market_proba[valid_mask], len(classes))
    one_hot = np.zeros_like(proba, dtype=float)
    one_hot[np.arange(len(y_int)), y_int] = 1.0
    brier = float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))
    try:
        loss = float(log_loss(y_int, proba, labels=list(range(len(classes)))))
    except ValueError:
        loss = 0.0
    return {
        "available": True,
        "rows": int(len(y_int)),
        "brier": round(brier, 6),
        "log_loss": round(loss, 6),
    }


def market_probability_matrix_from_features(x: pd.DataFrame, classes: List[Any], target: str) -> Optional[np.ndarray]:
    if target == "result":
        columns_by_label = {"H": "market_prob_home", "D": "market_prob_draw", "A": "market_prob_away"}
        if not all(column in x.columns for column in columns_by_label.values()):
            return None
        return np.column_stack([
            pd.to_numeric(x.get(columns_by_label.get(str(label), ""), pd.Series(0.0, index=x.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            for label in classes
        ])
    if is_over_under_target(target):
        suffix = total_line_suffix(OVER_UNDER_TARGET_LINES[target])
        columns_by_label = {"0": f"market_prob_under{suffix}", "1": f"market_prob_over{suffix}"}
        if not all(column in x.columns for column in columns_by_label.values()):
            return None
        return np.column_stack([
            pd.to_numeric(x.get(columns_by_label.get(str(label), ""), pd.Series(0.0, index=x.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            for label in classes
        ])
    return None


def decode_encoded_predictions(encoded: Iterable[Any], classes: List[Any]) -> pd.Series:
    values = pd.Series(encoded).reset_index(drop=True)
    decoded = []
    for value in values:
        try:
            index = int(value)
        except (TypeError, ValueError):
            decoded.append(value)
            continue
        decoded.append(classes[index] if 0 <= index < len(classes) else value)
    return pd.Series(decoded)


def derived_total_market_metrics(y_train, y_train_pred, y_eval, y_eval_pred) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for line in TRAIN_TOTAL_GOAL_LINES:
        suffix = total_line_suffix(line)
        target = f"over_under_{suffix}"
        train_actual = total_binary_for_line(y_train, line)
        train_pred = total_binary_for_line(y_train_pred, line)
        eval_actual = total_binary_for_line(y_eval, line)
        eval_pred = total_binary_for_line(y_eval_pred, line)
        output[target] = {
            "line": line,
            "label": f"U/O {line:.1f}",
            "metrics": classification_metrics_from_predictions(train_actual, train_pred, eval_actual, eval_pred),
            "confusion_matrix": binary_over_under_confusion(eval_actual, eval_pred, line),
        }
    return output


def is_over_under_target(target: Any) -> bool:
    return str(target or "") in OVER_UNDER_TARGET_LINES


def over_under_column_for_target(target: Any) -> str:
    line = OVER_UNDER_TARGET_LINES.get(str(target or ""), 2.5)
    return f"OverUnder{total_line_suffix(line)}"


def goal_line_label_from_suffix(line_suffix: str) -> str:
    text = str(line_suffix or "").strip()
    if len(text) == 2 and text.isdigit():
        return f"U/O {int(text[0])}.{text[1]}"
    return "U/O goles"


def goal_line_label_for_suffix(line_suffix: str) -> str:
    return goal_line_label_from_suffix(line_suffix)


def total_binary_for_line(values, line: float) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0).astype(int)
    return (numeric > line).astype(int)


def binary_over_under_confusion(y_true, y_pred, line: float) -> Dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int)
    labels = [f"Under {line:.1f}", f"Over {line:.1f}"]
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
    return {"labels": labels, "matrix": matrix.tolist(), "rows": rows, "total": int(matrix.sum())}


def confusion_matrix_payload(y_true, y_pred, classes: List[Any], target: str = "result") -> Dict[str, Any]:
    encoded_labels = list(range(len(classes)))
    labels = [display_class_label(label, target=target) for label in classes]
    true_values = pd.Series(y_true).reset_index(drop=True)
    pred_values = pd.Series(y_pred).reset_index(drop=True)
    known_actual_mask = true_values.isin(encoded_labels)
    unknown_actual_total = int((~known_actual_mask).sum())
    if known_actual_mask.any():
        matrix = confusion_matrix(
            true_values[known_actual_mask],
            pred_values[known_actual_mask],
            labels=encoded_labels,
        ).astype(int)
    else:
        matrix = np.zeros((len(encoded_labels), len(encoded_labels)), dtype=int)
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
        "unknown_actual_total": unknown_actual_total,
    }


def display_class_label(label: Any, target: str = "result") -> str:
    text = str(label)
    if target == GOALS_DISTRIBUTION_TARGET:
        try:
            goals = int(float(text))
            return f"{goals}+" if goals >= TOTAL_GOALS_CAP else f"{goals} goles"
        except (TypeError, ValueError):
            return text
    if text == "H":
        return "1 Local"
    if text == "D":
        return "X Empate"
    if text == "A":
        return "2 Visita"
    if text in {"0", "False", "false"}:
        if is_over_under_target(target):
            return f"Under {OVER_UNDER_TARGET_LINES[target]:.1f}"
        return "No"
    if text in {"1", "True", "true"}:
        if is_over_under_target(target):
            return f"Over {OVER_UNDER_TARGET_LINES[target]:.1f}"
        return "Si"
    return text


def etl_steps(
        files: Iterable[Path],
        normalized: Dict[str, Any],
        eval_strategy: str,
        prepared: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    file_list = list(files)
    train_rows = labeled_train_row_count(normalized)
    validation_rows = labeled_validation_row_count(normalized)
    test_rows = labeled_test_row_count(normalized)
    prediction_rows = int(normalized["team_prediction"].shape[0])
    feature_rows = int(normalized["team_features"].shape[0])
    walk_forward = walk_forward_status()
    prepared = prepared or {}
    prepared_ready = bool(prepared.get("ready"))
    prepared_stale = bool(prepared.get("stale"))
    prepared_label_source = str(prepared.get("label_source") or "")
    prepared_detail = "Artifact listo para entrenamiento." if prepared_ready and not prepared_stale else "Artifact desactualizado; vuelve a preparar ETL." if prepared_stale else "Aún no se ha preparado el artifact ETL."
    market_status = prepared.get("market_status", {}) if prepared else {}
    market_rows = int(prepared.get("market_rows", normalized.get("market_rows", 0)) or 0)
    qualifier_rows = int(prepared.get("qualifier_feature_rows", normalized.get("qualifier_feature_rows", 0)) or 0)
    api_status = prepared.get("api_football_status", {}) if prepared else {}
    api_fixture_rows = int(prepared.get("api_football_fixture_rows", normalized.get("api_football_fixture_rows", 0)) or 0)
    api_stat_rows = int(prepared.get("api_football_stat_rows", normalized.get("api_football_stat_rows", 0)) or 0)
    api_market_rows = int(prepared.get("api_football_market_rows", normalized.get("api_football_market_rows", 0)) or 0)
    international_status = prepared.get("international_recent") or normalized.get("international_recent") or international_results_status()
    all_matches_rows = int(prepared.get("all_matches_rows", normalized.get("all_matches_rows", international_status.get("all_matches_rows", 0))) or 0)
    worldcup_rows = int(prepared.get("worldcup_rows", normalized.get("worldcup_rows", international_status.get("worldcup_rows", 0))) or 0)
    benchmark_year = str(
        prepared.get("benchmark_worldcup_year")
        or normalized.get("benchmark_worldcup_year")
        or prepared.get("final_test_year")
        or normalized.get("final_test_year")
        or ""
    )
    international_detail = (
        f"all_matches.csv disponible como corpus principal ({all_matches_rows or international_status.get('rows', 0)} labels normalizados, {worldcup_rows} partidos FIFA World Cup incluidos si caen desde {INTERNATIONAL_TRAINING_START_YEAR})."
        if international_status.get("available")
        else f"All matches faltante: {international_status.get('reason') or 'sin CSV valido'} Ruta esperada: {international_status.get('file_path') or 'storage/worldcup/international/all_matches.csv'}."
    )
    if international_status.get("warning"):
        international_detail = f"{international_detail} {international_status.get('warning')}"
    return [
        {
            "name": "Dataset internacional",
            "status": "ok" if international_status.get("available") else "pending",
            "count": all_matches_rows or int(international_status.get("rows", 0) or 0),
            "detail": f"Fuente canonica: {international_status.get('source_path') or international_status.get('file_path') or INTERNATIONAL_MATCHES_FILE}.",
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
            "detail": f"Modo activo: {normalized.get('training_mode') or 'sin modo'}; fuente labels: {prepared_label_source or normalized.get('training_mode') or 'sin modo'}; train={train_rows}, validacion={validation_rows}, test={test_rows}.",
        },
        {
            "name": "Split evaluacion",
            "status": "ok" if eval_strategy != "unavailable" else "pending",
            "count": test_rows if test_rows else planned_holdout_rows(train_rows),
            "detail": f"Validacion temporal previa ({validation_rows}) + ultimos 30 partidos internacionales como test" if eval_strategy == EVAL_STRATEGY_LAST_30 else f"Benchmark historico: Mundial {benchmark_year}" if eval_strategy == "final_worldcup_test" and benchmark_year else "Benchmark historico Mundial" if eval_strategy == "final_worldcup_test" else "Test etiquetado" if eval_strategy == "test_file" else "Holdout temporal desde train" if eval_strategy == "holdout_temporal" else "Sin evaluacion.",
        },
        {
            "name": "Features dinamicas",
            "status": "ok" if train_rows else "pending",
            "count": feature_rows,
            "detail": "Elo/Poisson, historial, recent15, mercados y contexto temporal se calculan con cache por matriz.",
        },
        {
            "name": "Mercado 1X2 externo",
            "status": "ok" if market_status.get("has_1x2") else "info",
            "count": market_rows,
            "detail": "Odds Football-Data/manual normalizadas; faltantes se dejan en 0 con market_has_1x2=0.",
        },
        {
            "name": "Clasificatorios features",
            "status": "ok" if qualifier_rows else "info",
            "count": qualifier_rows,
            "detail": "Clasificatorios y partidos oficiales quedan en labels si estan en all_matches; este bloque aporta contexto externo cuando existe.",
        },
        {
            "name": "API-Football features",
            "status": "ok" if api_stat_rows or api_market_rows else "info",
            "count": api_stat_rows,
            "detail": f"Status {api_status.get('status', 'missing')}; fixtures={api_fixture_rows}, stats={api_stat_rows}, odds={api_market_rows}. Siempre se cortan por fecha del partido.",
        },
        {
            "name": "All matches recientes",
            "status": "ok" if international_status.get("available") else "info",
            "count": all_matches_rows or int(international_status.get("rows", 0) or 0),
            "detail": international_detail,
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
            "detail": "Features historicas/recent15 se cortan antes de la fecha del partido; validacion y test son bloques temporales posteriores.",
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
        "market_index": tuned.get("market_index", ""),
        "market_total": tuned.get("market_total", ""),
        "trials_per_market": tuned.get("trials_per_market", ""),
        "total_trial_budget": tuned.get("total_trial_budget", ""),
        "elapsed_seconds": tuned.get("elapsed_seconds", ""),
        "steps": [
            {"name": "Sampler", "status": "ok", "detail": str(tuned.get("sampler", ""))},
            {"name": "Pruner", "status": "ok", "detail": str(tuned.get("pruner", ""))},
            {"name": "Trials", "status": "ok", "detail": str(tuned.get("trials", 0))},
            {"name": "Presupuesto total", "status": "ok", "detail": str(tuned.get("total_trial_budget") or tuned.get("trials", 0))},
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
        "calibration": record.get("calibration", {}),
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
        "feature_inventory": record.get("feature_inventory", {}),
        "feature_cache": record.get("feature_cache", {}),
        "derived_total_markets": record.get("derived_total_markets", {}),
        "source_files": record.get("source_files", []),
        "kaggle_files": record.get("kaggle_files", []),
        "history_source": record.get("history_source", ""),
        "market_rows": int(record.get("market_rows", 0)),
        "qualifier_feature_rows": int(record.get("qualifier_feature_rows", 0)),
        "market_status": record.get("market_status", {}),
        "market_warnings": record.get("market_warnings", []),
        "api_football_status": record.get("api_football_status", {}),
        "api_football_warnings": record.get("api_football_warnings", []),
        "api_football_fixture_rows": int(record.get("api_football_fixture_rows", 0) or 0),
        "api_football_stat_rows": int(record.get("api_football_stat_rows", 0) or 0),
        "api_football_market_rows": int(record.get("api_football_market_rows", 0) or 0),
        "international_recent": record.get("international_recent", international_results_status()),
        "all_matches_rows": int(record.get("all_matches_rows", 0) or 0),
        "worldcup_rows": int(record.get("worldcup_rows", 0) or 0),
        "class_distribution": record.get("class_distribution", {}),
        "sample_weight_policy": record.get("sample_weight_policy", SAMPLE_WEIGHT_POLICY),
        "sample_weight_summary": record.get("sample_weight_summary", {}),
        "data_quality": record.get("data_quality", {}),
        "dc_rho": float(record.get("dc_rho", 0.0) or 0.0),
        "prepared_schema_version": record.get("prepared_schema_version", ""),
        "target_worldcup_year": str(record.get("target_worldcup_year") or TARGET_WORLDCUP_YEAR),
        "benchmark_worldcup_year": record.get("benchmark_worldcup_year", record.get("final_test_year", "")),
        "benchmark_policy": record.get("benchmark_policy", BENCHMARK_POLICY),
        "label_policy_notes": record.get("label_policy_notes", []),
        "final_test_year": record.get("final_test_year", ""),
        "split_policy": record.get("split_policy", ""),
        "training_start_year": int(record.get("training_start_year", INTERNATIONAL_TRAINING_START_YEAR) or INTERNATIONAL_TRAINING_START_YEAR),
        "max_label_date": record.get("max_label_date", ""),
        "validation_rows": int(record.get("validation_rows", 0) or 0),
        "hidden_from_catalog": bool(record.get("hidden_from_catalog", False)),
        "markets": record.get("markets", {}),
        "market_models": record.get("market_models", {}),
        "over_under_preferred_sources": record.get("over_under_preferred_sources", {}),
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
        "calibration": {},
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
        "api_football_status": {},
        "api_football_warnings": [],
        "api_football_fixture_rows": 0,
        "api_football_stat_rows": 0,
        "api_football_market_rows": 0,
        "international_recent": international_results_status(),
        "data_quality": {},
        "hidden_from_catalog": False,
        "markets": {},
        "market_models": {},
        "feature_cache": {},
        "walk_forward_mode": "none",
        "walk_forward_summary": {},
        "prepared_schema_version": "",
        "target_worldcup_year": str(TARGET_WORLDCUP_YEAR),
        "benchmark_worldcup_year": "",
        "benchmark_policy": BENCHMARK_POLICY,
        "label_policy_notes": [],
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


def over_under_target_values(home_goals: Any, away_goals: Any) -> Dict[str, int]:
    try:
        total_goals = float(home_goals) + float(away_goals)
    except (TypeError, ValueError):
        return {}
    if not np.isfinite(total_goals):
        return {}
    return {
        f"OverUnder{total_line_suffix(line)}": int(total_goals > line)
        for line in TRAIN_TOTAL_GOAL_LINES
    }


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
    pending_results = int(len(refresh_state.get("pending_result_ids", [])))
    ready_for_retrain = int(refresh_state["ready_result_only"])
    return {
        "matches": int(matches.shape[0]),
        "players": int(players.shape[0]),
        "team_rows": int(team_features.shape[0]),
        "completed_results": int(refresh_state.get("completed_results", 0)),
        "pending_results": pending_results,
        "needs_player_snapshot": int(refresh_state.get("needs_player_snapshot", 0)),
        "ready_result_only": ready_for_retrain,
        "ready_for_retrain": ready_for_retrain,
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
    if not existing.empty and "fixture_id" in existing.columns:
        current = existing[existing["fixture_id"].astype(str) == fixture_id]
        if not current.empty:
            included_result_only_at = str(current.iloc[-1].get("included_result_only_at", "") or "")
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
