from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from src.cli.model_specs import MODEL_SPECS, normalize_model_key, tunable_param_names
from src.worldcup.data import clean_team_name, load_historical_matches, tournament_fixtures_dataframe
from src.worldcup.model import HOST_TEAMS, WorldCupModel


KAGGLE_DATASET_SLUG = "harrachimustapha/fifa-world-cup-team-dataset"
KAGGLE_ROOT = Path("storage") / "worldcup" / "kaggle"
WORLD_CUP_MODELS_ROOT = Path("storage") / "worldcup" / "models"
HYBRID_MODEL_FILE = WORLD_CUP_MODELS_ROOT / "hybrid_worldcup_model.pkl"
HYBRID_MODEL_META_FILE = WORLD_CUP_MODELS_ROOT / "hybrid_worldcup_model.json"
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


class WorldCupTrainingError(RuntimeError):
    pass


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
            {"key": "result", "label": "Resultado 1/X/2"},
            {"key": "over_under_25", "label": "Over/Under 2.5"},
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
    model_meta = read_model_metadata()
    train_rows = labeled_train_row_count(normalized)
    test_rows = labeled_test_row_count(normalized)
    eval_strategy = evaluation_strategy(normalized)
    return {
        "dataset_slug": KAGGLE_DATASET_SLUG,
        "local_path": str(KAGGLE_ROOT),
        "files": [str(path) for path in files],
        "available": bool(files),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "eval_rows": test_rows if test_rows else planned_holdout_rows(train_rows),
        "prediction_rows": int(normalized["team_prediction"].shape[0]),
        "eval_strategy": eval_strategy,
        "team_feature_rows": int(normalized["team_features"].shape[0]),
        "target_column": normalized["target_column"],
        "team_columns": normalized["team_columns"],
        "training_mode": normalized["training_mode"],
        "trainable": bool(normalized["trainable"]),
        "model": model_meta,
        "preview": normalized["preview"],
    }


def train_hybrid_model(tournament: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    train_config = training_config(payload)
    files = list(discover_dataset_files(KAGGLE_ROOT))
    if not files:
        raise WorldCupTrainingError("No hay dataset Kaggle local. Primero descarga el dataset.")

    normalized = normalize_dataset_files(files)
    train_rows = normalized["train"].copy()
    test_rows = normalized["test"].copy()
    if not normalized["trainable"]:
        raise WorldCupTrainingError("El dataset no tiene columnas trainables de partido: equipos y resultado/winner.")

    group_teams = teams_from_tournament(tournament)
    history_df, history_source = load_historical_matches(refresh=bool(payload.get("refresh_history", False)))
    base_model = WorldCupModel.from_history(
        history_df,
        teams=group_teams,
        history_weight=float(payload.get("history_weight", 1.0) or 1.0),
        recency_weight=float(payload.get("recency_weight", 0.35) or 0.35),
        host_advantage=float(payload.get("host_advantage", 45.0) or 45.0),
        max_goals=int(payload.get("max_goals", 10) or 10),
    )
    feature_store = normalized["team_features"]
    target_warning = ""
    eval_strategy = "unavailable"
    if normalized["training_mode"] == "team_strength":
        effective_target = "team_strength"
        if train_config["training_target"] == "over_under_25":
            target_warning = "El dataset Kaggle de equipos no permite entrenar Over/Under 2.5; U/O queda con Poisson."
        x_train, y_train, feature_columns = build_team_training_matrix(normalized["team_train"])
        if normalized["team_test"].empty or "Label" not in normalized["team_test"].columns or normalized["team_test"]["Label"].dropna().empty:
            eval_strategy = "holdout_from_train"
            x_train, x_eval, y_train, y_eval = safe_train_eval_split(
                x_train,
                y_train,
                test_size=float(payload.get("eval_size", 0.25) or 0.25),
                random_state=train_config["seed"],
            )
        else:
            eval_strategy = "test_file"
            x_eval, y_eval, _ = build_team_training_matrix(normalized["team_test"], feature_columns=feature_columns)
    else:
        effective_target = train_config["training_target"]
        if effective_target == "over_under_25" and not has_over_under_target(train_rows):
            effective_target = "result"
            target_warning = "El dataset no tiene goles suficientes para entrenar Over/Under 2.5; U/O queda con Poisson."
        x_train, y_train, feature_columns = build_training_matrix(train_rows, base_model, feature_store, target=effective_target)
        if test_rows.empty:
            eval_strategy = "holdout_from_train"
            x_train, x_eval, y_train, y_eval = safe_train_eval_split(
                x_train,
                y_train,
                test_size=float(payload.get("eval_size", 0.25) or 0.25),
                random_state=train_config["seed"],
            )
        else:
            eval_strategy = "test_file"
            x_eval, y_eval, _ = build_training_matrix(test_rows, base_model, feature_store, feature_columns=feature_columns, target=effective_target)

    if x_train.empty or pd.Series(y_train).dropna().empty:
        raise WorldCupTrainingError("No hay filas entrenables para el objetivo seleccionado.")
    y_train_encoded, label_classes = encode_labels(y_train)
    y_eval_encoded = encode_existing_labels(y_eval, label_classes)
    tuned = tune_model_if_requested(train_config, x_train, y_train_encoded)
    if tuned.get("best_params"):
        train_config["params"].update(tuned["best_params"])
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
    metrics = classification_metrics(clf, x_train, y_train_encoded, x_eval, y_eval_encoded)
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
        "kaggle_files": [str(path) for path in files],
        "history_source": history_source,
        "metrics": metrics,
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
        "model_params": train_config["params"],
        "tuning": tuned,
        "hardware": hardware,
        "warnings": [warning for warning in [target_warning, *fit_result.get("warnings", [])] if warning],
        "top_features": top_feature_importances(clf, feature_columns),
    }
    save_hybrid_model(record)
    return {
        "model": read_model_metadata(),
        "metrics": metrics,
        "features": feature_columns,
        "train_rows": int(len(y_train)),
        "eval_rows": int(len(y_eval)),
        "source": KAGGLE_DATASET_SLUG,
        "mode": normalized["training_mode"],
        "eval_strategy": eval_strategy,
        "prediction_rows": int(normalized["team_prediction"].shape[0]),
        "effective_target": effective_target,
        "requested_target": train_config["training_target"],
        "model_type": train_config["model_type"],
        "hardware": hardware,
        "tuning": tuned,
        "warnings": record["warnings"],
    }


def predict_match_payload(
        tournament: Dict[str, Any],
        base_model: WorldCupModel,
        fixture_id: Optional[Any] = None,
        home: Optional[str] = None,
        away: Optional[str] = None,
        use_ml_model: bool = True,
        ml_weight: float = 0.5,
) -> Dict[str, Any]:
    fixture = select_prediction_fixture(tournament, fixture_id=fixture_id, home=home, away=away)
    home_team = str(fixture.get("Equipo 1", home or ""))
    away_team = str(fixture.get("Equipo 2", away or ""))
    poisson = base_model.match_probabilities(home_team, away_team)
    base_probs = {"H": poisson["home"], "D": poisson["draw"], "A": poisson["away"]}
    base_totals = {"over25": poisson["over25"], "under25": poisson["under25"]}
    ml_outputs = {"result": {}, "over_under_25": {}, "notes": ["Modelo Kaggle no entrenado."]}
    if use_ml_model:
        ml_outputs = predict_ml_outputs(base_model, home_team, away_team)
    result_ml = ml_outputs.get("result", {})
    over_under_ml = ml_outputs.get("over_under_25", {})
    result_weight = ml_weight if result_ml else 0.0
    totals_weight = ml_weight if over_under_ml else 0.0
    blended = blend_probabilities(base_probs, result_ml, result_weight)
    blended_totals = blend_total_probabilities(base_totals, over_under_ml, totals_weight)
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
        },
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
    if not home_col or not away_col:
        return pd.DataFrame()
    rows = []
    for _, row in clean.iterrows():
        home = clean_team_name(row.get(home_col))
        away = clean_team_name(row.get(away_col))
        if not home or not away:
            continue
        label = label_from_goals(row.get(goals_home), row.get(goals_away)) if goals_home and goals_away else ""
        if not label and target_col:
            label = label_from_target(row.get(target_col), home, away)
        if label not in TARGET_LABELS:
            continue
        record = {"Home": home, "Away": away, "Label": label, "Source": source}
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
    return output


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


def build_training_matrix(
        rows: pd.DataFrame,
        base_model: WorldCupModel,
        team_features: pd.DataFrame,
        feature_columns: Optional[List[str]] = None,
        target: str = "result",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    working = rows.copy()
    if target == "over_under_25":
        working = working[working["OverUnder25"].notna()] if "OverUnder25" in working.columns else working.iloc[0:0]
    records = [match_feature_row(base_model, team_features, row["Home"], row["Away"]) for _, row in working.iterrows()]
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


def match_feature_row(base_model: WorldCupModel, team_features: pd.DataFrame, home: str, away: str) -> Dict[str, float]:
    p_home = base_model.profile(home)
    p_away = base_model.profile(away)
    row = {
        "rating_home": p_home.rating,
        "rating_away": p_away.rating,
        "rating_diff": p_home.rating - p_away.rating,
        "attack_home": p_home.attack,
        "attack_away": p_away.attack,
        "attack_diff": p_home.attack - p_away.attack,
        "defense_home": p_home.defense,
        "defense_away": p_away.defense,
        "defense_diff": p_home.defense - p_away.defense,
        "matches_home": float(p_home.matches),
        "matches_away": float(p_away.matches),
        "home_is_host": 1.0 if home in HOST_TEAMS else 0.0,
        "away_is_host": 1.0 if away in HOST_TEAMS else 0.0,
    }
    if not team_features.empty and "Team" in team_features.columns:
        home_features = team_features[team_features["Team"].map(normalize_team_key) == normalize_team_key(home)]
        away_features = team_features[team_features["Team"].map(normalize_team_key) == normalize_team_key(away)]
        numeric_cols = [column for column in team_features.columns if column != "Team" and pd.api.types.is_numeric_dtype(team_features[column])]
        for column in numeric_cols[:24]:
            home_value = float(home_features[column].iloc[0]) if not home_features.empty else 0.0
            away_value = float(away_features[column].iloc[0]) if not away_features.empty else 0.0
            safe = normalize_column(column)
            row[f"kaggle_{safe}_home"] = home_value
            row[f"kaggle_{safe}_away"] = away_value
            row[f"kaggle_{safe}_diff"] = home_value - away_value
    return row


def labeled_train_row_count(normalized: Dict[str, Any]) -> int:
    return int(normalized["train"].shape[0] or normalized["team_train"].shape[0])


def labeled_test_row_count(normalized: Dict[str, Any]) -> int:
    return int(normalized["test"].shape[0] or normalized["team_test"].shape[0])


def evaluation_strategy(normalized: Dict[str, Any]) -> str:
    if labeled_test_row_count(normalized) > 0:
        return "test_file"
    if labeled_train_row_count(normalized) > 0:
        return "holdout_from_train"
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
    return {
        "model_type": model_key,
        "training_target": normalize_training_target(payload.get("training_target", payload.get("target", "result"))),
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


def normalize_training_target(value: Any) -> str:
    key = str(value or "result").strip().lower().replace("-", "_")
    if key in {"over_under", "over_under_25", "uo25", "u_o_25", "overunder25"}:
        return "over_under_25"
    return "result"


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
    return "OverUnder25" in rows.columns and rows["OverUnder25"].dropna().shape[0] > 0


def safe_train_eval_split(x: pd.DataFrame, y: pd.Series, test_size: float, random_state: int):
    y_series = pd.Series(y).reset_index(drop=True)
    if len(y_series) < 4 or y_series.nunique(dropna=True) < 2:
        return x.copy(), x.copy(), y_series.copy(), y_series.copy()
    test_count = max(1, int(round(len(y_series) * float(test_size))))
    n_classes = int(y_series.nunique(dropna=True))
    counts = y_series.value_counts()
    stratify = None
    if counts.min() >= 2 and test_count >= n_classes and (len(y_series) - test_count) >= n_classes:
        stratify = y_series
    try:
        return sklearn_train_test_split(
            x,
            y_series,
            test_size=float(test_size),
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return x.copy(), x.copy(), y_series.copy(), y_series.copy()


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
            return {
                "classifier": fallback,
                "device": "cpu",
                "warnings": [*device_warnings, f"CUDA fallo durante fit ({exc.__class__.__name__}); se reintento en CPU."],
            }
        raise


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


def tune_model_if_requested(config: Dict[str, Any], x_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
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
            pred = fit_result["classifier"].predict(x_eval)
            return metric_score(y_eval, pred, objective_name)
        except Exception as exc:
            trial.set_user_attr("error", f"{exc.__class__.__name__}: {exc}")
            return 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=build_optuna_sampler(optuna, config["optuna_sampler"], config["seed"]),
        pruner=build_optuna_pruner(optuna, config["optuna_pruner"]),
    )
    study.optimize(objective, n_trials=config["n_trials"], show_progress_bar=False)
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
    pairs = []
    for column, value in zip(feature_columns, np.asarray(values, dtype=float)):
        pairs.append({"feature": column, "importance": round(float(value), 6)})
    return sorted(pairs, key=lambda item: item["importance"], reverse=True)[:limit]


def predict_ml_probs(base_model: WorldCupModel, home: str, away: str) -> Tuple[Dict[str, float], str]:
    outputs = predict_ml_outputs(base_model, home, away)
    return outputs.get("result", {}), " - ".join(outputs.get("notes", []))


def predict_ml_outputs(base_model: WorldCupModel, home: str, away: str) -> Dict[str, Any]:
    record = load_hybrid_model()
    if not record:
        return {"result": {}, "over_under_25": {}, "notes": ["Modelo Kaggle no entrenado."]}
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
            "notes": [
                f"Modelo {record.get('model_label', 'Kaggle')} team-strength aplicado ({record.get('target_column', '')}).",
                "Over/Under 2.5 viene de Poisson porque el dataset de equipos no contiene goles de partido.",
            ],
        }
    team_features = pd.DataFrame(record.get("team_features", []))
    x = pd.DataFrame([match_feature_row(base_model, team_features, home, away)])
    feature_columns = record.get("feature_columns", BASE_FEATURE_COLUMNS)
    for column in feature_columns:
        if column not in x.columns:
            x[column] = 0.0
    x = x[feature_columns].fillna(0.0).astype(float)
    probabilities = np.asarray(record["classifier"].predict_proba(x)[0], dtype=float)
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
            "notes": [f"Modelo {record.get('model_label', 'Kaggle')} aplicado a Over/Under 2.5."],
        }
    output = {label: 0.0 for label in TARGET_LABELS}
    for label, probability in zip(labels, probabilities):
        if label in output:
            output[label] = float(probability)
    total = max(sum(output.values()), 1e-9)
    return {
        "result": {key: value / total for key, value in output.items()},
        "over_under_25": {},
        "notes": [
            f"Modelo {record.get('model_label', 'Kaggle')} aplicado a 1X2.",
            "Over/Under 2.5 viene de Poisson; entrena target U/O si el dataset trae goles.",
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
        probabilities = clf.predict_proba(x)[0]
        class_values = record.get("classes", [])
        for target_value in (1, "1", True, "True", "true"):
            if target_value in class_values:
                return float(probabilities[class_values.index(target_value)])
    encoded = int(clf.predict(x)[0])
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


def blend_total_probabilities(base_probs: Dict[str, float], ml_probs: Dict[str, float], ml_weight: float) -> Dict[str, float]:
    weight = min(max(float(ml_weight or 0.0), 0.0), 1.0)
    output = {}
    for label in ["over25", "under25"]:
        output[label] = base_probs.get(label, 0.0) * (1.0 - weight) + ml_probs.get(label, base_probs.get(label, 0.0)) * weight
    total = max(sum(output.values()), 1e-9)
    return {label: value / total for label, value in output.items()}


def classification_metrics(clf, x_train, y_train, x_eval, y_eval) -> Dict[str, Dict[str, float]]:
    return {
        "train": metric_row(y_train, clf.predict(x_train)),
        "eval": metric_row(y_eval, clf.predict(x_eval)),
    }


def metric_row(y_true, y_pred) -> Dict[str, float]:
    return {
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 3),
        "F1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0.0)), 3),
        "Precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0.0)), 3),
        "Recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0.0)), 3),
    }


def save_hybrid_model(record: Dict[str, Any]) -> None:
    WORLD_CUP_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    with HYBRID_MODEL_FILE.open("wb") as handle:
        pickle.dump(record, handle)
    meta = {
        "trained": True,
        "model_path": str(HYBRID_MODEL_FILE),
        "feature_count": len(record.get("feature_columns", [])),
        "classes": record.get("classes", []),
        "metrics": record.get("metrics", {}),
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
        "hardware": record.get("hardware", {}),
        "warnings": record.get("warnings", []),
        "top_features": record.get("top_features", []),
        "kaggle_files": record.get("kaggle_files", []),
        "history_source": record.get("history_source", ""),
    }
    HYBRID_MODEL_META_FILE.write_text(json.dumps(json_safe(meta), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_hybrid_model() -> Optional[Dict[str, Any]]:
    if not HYBRID_MODEL_FILE.exists():
        return None
    with HYBRID_MODEL_FILE.open("rb") as handle:
        return pickle.load(handle)


def read_model_metadata() -> Dict[str, Any]:
    if HYBRID_MODEL_META_FILE.exists():
        try:
            data = json.loads(HYBRID_MODEL_META_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {
        "trained": False,
        "model_path": str(HYBRID_MODEL_FILE),
        "feature_count": 0,
        "classes": [],
        "metrics": {},
        "model_type": "",
        "model_label": "",
        "effective_target": "",
        "requested_target": "",
        "eval_strategy": "",
        "prediction_rows": 0,
        "hardware": detect_hardware(),
        "tuning": {"enabled": False},
        "warnings": [],
        "top_features": [],
    }


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
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return value
