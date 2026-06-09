from __future__ import annotations

import json
import pickle
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split as sklearn_train_test_split

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


class WorldCupTrainingError(RuntimeError):
    pass


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
    return {
        "dataset_slug": KAGGLE_DATASET_SLUG,
        "local_path": str(KAGGLE_ROOT),
        "files": [str(path) for path in files],
        "available": bool(files),
        "train_rows": int(normalized["train"].shape[0] or normalized["team_train"].shape[0]),
        "test_rows": int(normalized["test"].shape[0] or normalized["team_test"].shape[0]),
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
    if normalized["training_mode"] == "team_strength":
        x_train, y_train, feature_columns = build_team_training_matrix(normalized["team_train"])
        if normalized["team_test"].empty or "Label" not in normalized["team_test"].columns or normalized["team_test"]["Label"].dropna().empty:
            x_train, x_eval, y_train, y_eval = sklearn_train_test_split(
                x_train,
                y_train,
                test_size=float(payload.get("eval_size", 0.25) or 0.25),
                random_state=int(payload.get("seed", 2026) or 2026),
                stratify=y_train if pd.Series(y_train).value_counts().min() >= 2 else None,
            )
        else:
            x_eval, y_eval, _ = build_team_training_matrix(normalized["team_test"], feature_columns=feature_columns)
    else:
        x_train, y_train, feature_columns = build_training_matrix(train_rows, base_model, feature_store)
        if test_rows.empty:
            stratify = y_train if pd.Series(y_train).value_counts().min() >= 2 else None
            x_train, x_eval, y_train, y_eval = sklearn_train_test_split(
                x_train,
                y_train,
                test_size=float(payload.get("eval_size", 0.25) or 0.25),
                random_state=int(payload.get("seed", 2026) or 2026),
                stratify=stratify,
            )
        else:
            x_eval, y_eval, _ = build_training_matrix(test_rows, base_model, feature_store, feature_columns=feature_columns)

    clf = RandomForestClassifier(
        n_estimators=int(payload.get("n_estimators", 240) or 240),
        random_state=int(payload.get("seed", 2026) or 2026),
        class_weight="balanced_subsample",
        min_samples_leaf=max(int(payload.get("min_samples_leaf", 2) or 2), 1),
    )
    clf.fit(x_train, y_train)
    metrics = classification_metrics(clf, x_train, y_train, x_eval, y_eval)
    record = {
        "classifier": clf,
        "feature_columns": feature_columns,
        "team_features": feature_store.to_dict(orient="records"),
        "kaggle_files": [str(path) for path in files],
        "history_source": history_source,
        "metrics": metrics,
        "classes": list(clf.classes_),
        "mode": normalized["training_mode"],
        "target_column": normalized["target_column"],
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
    ml_probs, ml_note = ({}, "Modelo Kaggle no entrenado.")
    if use_ml_model:
        ml_probs, ml_note = predict_ml_probs(base_model, home_team, away_team)
    blended = blend_probabilities(base_probs, ml_probs, ml_weight if ml_probs else 0.0)
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
            "over25": round(poisson["over25"] * 100.0, 2),
            "under25": round(poisson["under25"] * 100.0, 2),
        },
        "model_probs": {
            "poisson": {key: round(value * 100.0, 2) for key, value in base_probs.items()},
            "ml": {key: round(value * 100.0, 2) for key, value in ml_probs.items()},
            "ml_weight": round(float(ml_weight if ml_probs else 0.0), 3),
        },
        "expected_goals": {
            "home": round(poisson["lambda1"], 3),
            "away": round(poisson["lambda2"], 3),
        },
        "modal_score": f"{poisson['modal_g1']}-{poisson['modal_g2']}",
        "prediction": label_display(max(blended, key=blended.get), home_team, away_team),
        "notes": [ml_note],
    }


def normalize_dataset_files(files: Iterable[Path]) -> Dict[str, Any]:
    train_frames = []
    test_frames = []
    all_frames = []
    team_feature_frames = []
    team_train_frames = []
    team_test_frames = []
    target_column = ""
    team_columns: List[str] = []
    for path in files:
        raw = read_table(path)
        if raw.empty:
            continue
        standardized = standardize_match_rows(raw, source=str(path))
        team_features = extract_team_features(raw, source=str(path))
        if not team_features.empty:
            team_feature_frames.append(team_features)
        team_rows = standardize_team_target_rows(raw, source=str(path))
        if not team_rows.empty:
            if "test" in path.name.lower() or "eval" in path.name.lower():
                team_test_frames.append(team_rows)
            else:
                team_train_frames.append(team_rows)
        if not standardized.empty:
            all_frames.append(standardized)
            if "train" in path.name.lower():
                train_frames.append(standardized)
            elif "test" in path.name.lower() or "eval" in path.name.lower():
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
    team_features_df = merge_team_features(team_feature_frames)
    training_mode = "match_result" if not train_df.empty else "team_strength" if not team_train_df.empty else ""
    preview_source = train_df if not train_df.empty else team_train_df if not team_train_df.empty else pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    preview = preview_payload(preview_source)
    return {
        "train": train_df,
        "test": test_df,
        "team_train": team_train_df,
        "team_test": team_test_df,
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
        rows.append({"Home": home, "Away": away, "Label": label, "Source": source})
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
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    records = [match_feature_row(base_model, team_features, row["Home"], row["Away"]) for _, row in rows.iterrows()]
    x = pd.DataFrame(records).fillna(0.0)
    if feature_columns is None:
        feature_columns = list(x.columns)
    for column in feature_columns:
        if column not in x.columns:
            x[column] = 0.0
    x = x[feature_columns].astype(float)
    return x, rows["Label"].astype(str), feature_columns


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


def predict_ml_probs(base_model: WorldCupModel, home: str, away: str) -> Tuple[Dict[str, float], str]:
    record = load_hybrid_model()
    if not record:
        return {}, "Modelo Kaggle no entrenado."
    if record.get("mode") == "team_strength":
        home_strength = team_strength_score(record, home)
        away_strength = team_strength_score(record, away)
        diff = home_strength - away_strength
        draw = min(max(0.28 - abs(diff) * 0.12, 0.16), 0.34)
        home_share = 1.0 / (1.0 + np.exp(-diff * 4.0))
        home_prob = (1.0 - draw) * home_share
        away_prob = (1.0 - draw) * (1.0 - home_share)
        return {"H": float(home_prob), "D": float(draw), "A": float(away_prob)}, f"Modelo Kaggle team-strength aplicado ({record.get('target_column', '')})."
    team_features = pd.DataFrame(record.get("team_features", []))
    x = pd.DataFrame([match_feature_row(base_model, team_features, home, away)])
    feature_columns = record.get("feature_columns", BASE_FEATURE_COLUMNS)
    for column in feature_columns:
        if column not in x.columns:
            x[column] = 0.0
    x = x[feature_columns].fillna(0.0).astype(float)
    clf = record["classifier"]
    probabilities = clf.predict_proba(x)[0]
    output = {label: 0.0 for label in TARGET_LABELS}
    for label, probability in zip(clf.classes_, probabilities):
        if label in output:
            output[str(label)] = float(probability)
    total = max(sum(output.values()), 1e-9)
    return {key: value / total for key, value in output.items()}, "Modelo Kaggle aplicado."


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
        class_values = list(clf.classes_)
        if 1 in class_values:
            return float(probabilities[class_values.index(1)])
        if "1" in class_values:
            return float(probabilities[class_values.index("1")])
    return float(clf.predict(x)[0])


def blend_probabilities(base_probs: Dict[str, float], ml_probs: Dict[str, float], ml_weight: float) -> Dict[str, float]:
    weight = min(max(float(ml_weight or 0.0), 0.0), 1.0)
    output = {}
    for label in TARGET_LABELS:
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
        "target_column": record.get("target_column", ""),
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
    return {"trained": False, "model_path": str(HYBRID_MODEL_FILE), "feature_count": 0, "classes": [], "metrics": {}}


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
