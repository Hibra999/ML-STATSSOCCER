from __future__ import annotations

import json
import math
import os
import re
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mlstatssoccer-matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src.analysis import (
    BorutaAnalyzer,
    CoefficientAnalyzer,
    CorrelationAnalyzer,
    DescriptiveAnalyzer,
    DistributionAnalyzer,
    GiniImpurityAnalyzer,
    RuleExtractorAnalyzer,
    VarianceAnalyzer,
)
from src.cli.app import (
    EXPLAINER_BY_MODEL,
    _append_prediction_columns,
    _compute_probability_percentiles,
    _dataset_masks,
    _delete_stored_filter,
    _descriptive_table,
    _evaluation_mask,
    _fixture_selected_mask,
    _league_odd_mask,
    _parse_feature_pair,
    _profit_balance,
    _resolve_stats_columns,
    _seasonal_metrics,
    _store_filter,
    _study_to_dataframe,
    _valid_odd,
    _validate_fixture_rows,
)
from src.cli.common import (
    CLIError,
    COLORMAP_OPTIONS,
    export_dataframe,
    load_required_columns,
    parse_columns,
    parse_eval_odd_range,
    parse_odd_range,
    parse_target,
    save_figure,
    target_label,
    validate_identifier,
)
from src.cli.model_specs import MODEL_SPECS, build_model_params, normalize_model_key, tunable_param_names, tunable_params_for_args
from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.interpretability.explainers.decisiontree import DecisionTreeExplainer
from src.interpretability.explainers.discriminant import DiscriminantAnalysisExplainer
from src.interpretability.explainers.extremeboosting import ExtremeBoostingExplainer
from src.interpretability.explainers.knn import KNNExplainer
from src.interpretability.explainers.logistic import LogisticRegressionExplainer
from src.interpretability.explainers.nn import NeuralNetworkExplainer
from src.interpretability.explainers.randomforest import RandomForestExplainer
from src.interpretability.explainers.svm import SVMExplainer
from src.models.classifiers.neuralnets.nn import NeuralNetwork
from src.models.trainer import Trainer
from src.models.tuner import Tuner
from src.network.fixtures.footystats.scraper import FootyStatsScraper
from src.network.fixtures.utils import match_fixture_teams
from src.preprocessing.selection import train_test_split
from src.preprocessing.utils.inputs import construct_inputs_by_fixture
from src.preprocessing.utils.target import TargetType, construct_targets


OUTPUT_ROOT = Path("outputs") / "web"
COUNTRY_FLAGS_ROOT = Path("storage") / "graphics" / "countries"
SUPPORTED_BROWSERS = ("chrome", "firefox", "edge", "brave")
DEFAULT_BROWSER_CONFIG = {
    "application": "chrome",
    "headless": True,
    "brave_binary": "",
}
UPCOMING_FIXTURE_DAYS = 7
DASHBOARD_FIXTURE_LIMIT = 5
MEXICO_CITY_TZ = ZoneInfo("America/Mexico_City")
PREDICT_FIXTURE_COLUMNS = ["Date", "Hora MX", "Home", "Away", "1", "X", "2"]
DASHBOARD_FIXTURE_COLUMNS = ["Catalogo", "Liga", "Pais", *PREDICT_FIXTURE_COLUMNS]
MODEL_LABELS_ES = {
    "ngboost": "NGBoost",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
}


def jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return table_payload(value)
    if isinstance(value, pd.Series):
        return jsonable(value.to_dict())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (datetime, date, Path)):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, TargetType):
        return target_label(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return value


def table_payload(df: pd.DataFrame, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
    page = max(int(page or 1), 1)
    page_size = min(max(int(page_size or 50), 1), 500)
    total = int(df.shape[0])
    start = (page - 1) * page_size
    page_df = df.iloc[start:start + page_size].copy()
    page_df = page_df.astype(object).where(pd.notna(page_df), "")
    return {
        "columns": [str(column) for column in page_df.columns],
        "rows": jsonable(page_df.to_dict(orient="records")),
        "page": page,
        "page_size": page_size,
        "total": total,
        "pages": int(math.ceil(total / page_size)) if page_size else 0,
    }


def dashboard() -> Dict[str, Any]:
    leagues = saved_leagues()
    return {
        "leagues": len(leagues),
        "models": sum(item["models"] for item in leagues),
        "model_specs": model_specs(),
    }


def dashboard_fixtures(limit: int = DASHBOARD_FIXTURE_LIMIT, days: int = UPCOMING_FIXTURE_DAYS) -> Dict[str, Any]:
    db = LeagueDatabase()
    limit = min(max(int(limit or DASHBOARD_FIXTURE_LIMIT), 1), 25)
    days = min(max(int(days or UPCOMING_FIXTURE_DAYS), 1), 30)
    rows = []
    notes = []

    for league_index, league in enumerate(db.leagues, start=1):
        if len(rows) >= limit:
            break
        if not league.fixture:
            continue
        try:
            fixture_df = scrape_upcoming_fixtures(
                league=league,
                league_df=pd.DataFrame(),
                days=days,
                limit=limit - len(rows),
                headless=None,
                match_teams=False,
            )
        except Exception as exc:
            notes.append(f"{league_display_name(league)}: {clean_error_text(exc)}")
            continue
        if fixture_df.empty:
            continue
        fixture_df = fixture_df.copy()
        fixture_df.insert(0, "Pais", league.country)
        fixture_df.insert(0, "Liga", league_display_name(league))
        fixture_df.insert(0, "Catalogo", league_index)
        rows.extend(fixture_df[DASHBOARD_FIXTURE_COLUMNS].to_dict(orient="records"))

    output_df = pd.DataFrame(rows, columns=DASHBOARD_FIXTURE_COLUMNS)
    if not output_df.empty:
        output_df = output_df.sort_values(["Date", "Hora MX", "Liga"], kind="stable").head(limit).reset_index(drop=True)
    return {
        "fixtures": table_payload(output_df, page=1, page_size=limit),
        "notes": notes,
        "days": days,
        "limit": limit,
    }


def model_specs() -> List[Dict[str, Any]]:
    return [
        {
            "key": spec.key,
            "label": MODEL_LABELS_ES.get(spec.key, spec.label),
            "supports_calibration": spec.supports_calibration,
            "defaults": display_defaults(spec.defaults),
            "tunables": tunable_param_names(spec),
        }
        for spec in MODEL_SPECS.values()
    ]


def catalog_leagues() -> List[Dict[str, Any]]:
    db = LeagueDatabase()
    return [
        {
            "index": idx,
            "country": league.country,
            "name": league.name,
            "display_name": league_display_name(league),
            "default_league_id": default_league_id(league),
            "category": category_label(league.category),
            "start_year": int(league.start_year),
            "history_window": 3,
            "goal_margin": 2,
            "stats": "Todas",
            "fixture": league.fixture or "",
            "flag_url": country_flag_url(league.country),
        }
        for idx, league in enumerate(db.leagues, start=1)
    ]


def saved_leagues() -> List[Dict[str, Any]]:
    db = LeagueDatabase()
    leagues = []
    for league_id in db.get_league_ids():
        league = db.index[league_id]
        df = db.load_league(league_id)
        model_count = len(ModelDatabase(league_id=league_id).get_model_ids())
        leagues.append({
            **league_payload(league),
            "rows": 0 if df is None else int(df.shape[0]),
            "columns": 0 if df is None else int(df.shape[1]),
            "models": model_count,
        })
    return leagues


def league_detail(league_id: str, rows: int = 25, update: bool = False) -> Dict[str, Any]:
    _, league, df = load_league(league_id, update=update)
    return {
        "league": {
            **league_payload(league),
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "missing_rows": int(df.isna().any(axis=1).sum()),
            "models": len(ModelDatabase(league_id=league_id).get_model_ids()),
        },
        "preview": table_payload(df.head(rows), page=1, page_size=rows),
    }


def create_league(payload: Dict[str, Any]) -> Dict[str, Any]:
    db = LeagueDatabase()
    template = catalog_template(db, payload)
    league_id = validate_identifier(payload.get("league_id") or payload.get("id") or f"{template.name}-{template.country}-01", "league id")
    if db.league_exists(league_id):
        raise CLIError(f'La liga "{league_id}" ya existe.')

    current_year_threshold = date.today().year - 4
    start_year = int(payload.get("start_year") or template.start_year)
    if start_year < template.start_year:
        raise CLIError(f"{template.name} inicia en {template.start_year}; se solicito {start_year}.")
    if start_year > current_year_threshold:
        raise CLIError(f"El ano de inicio no puede ser mayor que {current_year_threshold}.")

    history_window = int(payload.get("history_window", 3))
    goal_margin = int(payload.get("goal_margin", 2))
    if history_window < 2 or history_window > 5:
        raise CLIError("El historial debe estar entre 2 y 5.")
    if goal_margin < 2 or goal_margin > 5:
        raise CLIError("El margen debe estar entre 2 y 5.")

    stats_columns = _resolve_stats_columns(template, str(payload.get("stats", "all")))
    league = template.clone(
        start_year=start_year,
        league_id=league_id,
        match_history_window=history_window,
        goal_diff_margin=goal_margin,
        stats_columns=stats_columns,
        odd_1_range=parse_odd_range(payload.get("odd_1")),
        odd_x_range=parse_odd_range(payload.get("odd_x")),
        odd_2_range=parse_odd_range(payload.get("odd_2")),
    )
    df = db.create_league(league=league)
    if df is None:
        raise CLIError("No se pudo descargar la liga. Revisa internet y disponibilidad de la fuente.")
    return {"league": league_payload(league), "preview": table_payload(df.head(25), page=1, page_size=25)}


def update_league(league_id: str) -> Dict[str, Any]:
    return league_detail(league_id, update=True)


def delete_league(league_id: str) -> Dict[str, str]:
    db = LeagueDatabase()
    if not db.league_exists(league_id):
        raise CLIError(f'La liga "{league_id}" no existe.')
    db.delete_league(league_id)
    return {"deleted": league_id}


def league_data(
        league_id: str,
        page: int = 1,
        page_size: int = 50,
        query: Optional[str] = None,
        column: Optional[str] = None,
        exact: bool = False,
        hide_missing: bool = False,
        columns: Optional[str] = None,
) -> Dict[str, Any]:
    _, _, df = load_league(league_id)
    df = filter_dataframe(df, query=query, column=column, exact=exact, hide_missing=hide_missing, columns=columns)
    return table_payload(df, page=page, page_size=page_size)


def export_league_data(
        league_id: str,
        fmt: str,
        query: Optional[str] = None,
        column: Optional[str] = None,
        exact: bool = False,
        hide_missing: bool = False,
        columns: Optional[str] = None,
) -> Dict[str, str]:
    _, _, df = load_league(league_id)
    df = filter_dataframe(df, query=query, column=column, exact=exact, hide_missing=hide_missing, columns=columns)
    suffix = ".xlsx" if str(fmt).lower() == "xlsx" else ".csv"
    output = output_path(f"{league_id}-dataset", suffix)
    export_dataframe(df, str(output))
    return {"path": str(output), "url": output_url(output)}


def list_models(league_id: str) -> List[Dict[str, Any]]:
    load_league(league_id)
    model_db = ModelDatabase(league_id=league_id)
    models = []
    for model_id in model_db.get_model_ids():
        config = model_db.load_model_config(model_id)
        if config is not None and is_supported_model_config(config):
            models.append(model_payload(model_id, config))
    return models


def train_model(league_id: str, payload: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
    model_key = normalize_model_key(payload.get("model_type", "xgboost"))
    spec = MODEL_SPECS[model_key]
    emit_training_progress(progress_callback, "preparing", 0, 1, "Preparando entrenamiento")
    _, league, df = load_league(league_id)
    df = df.dropna(ignore_index=True)
    if df.empty:
        raise CLIError("El dataset de entrenamiento no tiene filas completas despues de quitar faltantes.")

    model_db = ModelDatabase(league_id=league.league_id)
    model_id = validate_identifier(payload.get("model_id") or payload.get("id") or f"{league.league_id}-{model_key}", "model id")
    if model_db.model_exists(model_id):
        raise CLIError(f'El modelo "{model_id}" ya existe para la liga "{league.league_id}".')

    args = training_args(payload, league_id=league.league_id, model_id=model_id, model_key=model_key)
    if "tuning_enabled" in payload:
        if bool_payload(args.tuning_enabled):
            if str(args.tune or "").strip().lower() in {"", "none"}:
                args.tune = "all"
        else:
            args.tune = None
    args.trials = int(args.trials)
    args.eval_size = float(args.eval_size)
    args.optuna_sampler = normalize_optuna_choice(args.optuna_sampler, {"tpe", "random", "cmaes", "cma-es"}, "sampler")
    args.optuna_pruner = normalize_optuna_choice(args.optuna_pruner, {"none", "median", "successive-halving"}, "pruner")
    if args.eval_size < 5 or args.eval_size > 30:
        raise CLIError("El porcentaje de evaluacion debe estar entre 5 y 30.")
    if args.trials < 1:
        raise CLIError("N trials debe ser mayor o igual a 1.")
    if args.objective not in {"Accuracy", "F1", "Precision", "Recall"}:
        raise CLIError("El objetivo debe ser Accuracy, F1, Precision o Recall.")

    model_config = build_model_params(args=args, league_id=league.league_id, model_id=model_id, model_key=model_key)
    model_config["train"] = {"eval_samples_size": float(args.eval_size), "results": {}}

    trainer = Trainer()
    tunable_params = tunable_params_for_args(args, spec)
    optuna_summary = {
        "enabled": bool(tunable_params),
        "sampler": args.optuna_sampler,
        "pruner": args.optuna_pruner,
        "n_trials": args.trials if tunable_params else 0,
        "objective": args.objective,
        "best_score": "",
        "best_params": {},
    }
    if tunable_params:
        tuner = Tuner(
            model_cls=spec.model_cls,
            fixed_params=model_config,
            tunable_params=tunable_params,
            df=df,
            metric=args.objective,
            sampler=args.optuna_sampler,
            pruner=args.optuna_pruner,
            progress_callback=progress_callback,
        )
        study = tuner.tune(trials=args.trials)
        trials_df = _study_to_dataframe(study, args.objective)
        model_config["train"]["results"]["tune"] = trials_df
        model_config.update(**study.best_trial.params)
        optuna_summary["best_score"] = study.best_value
        optuna_summary["best_params"] = study.best_trial.params

    if args.cv:
        emit_training_progress(progress_callback, "cv", 0, 1, "Validacion cruzada")
        model = spec.model_cls(**model_config)
        cv_df = trainer.cross_validation(model=model, df=df)
        cv_df["Model"] = model_id
        cv_df["Model Type"] = model.__class__
        model_config["train"]["results"]["cv"] = cv_df
        emit_training_progress(progress_callback, "cv", 1, 1, "Validacion cruzada completada")

    if args.sliding_cv:
        emit_training_progress(progress_callback, "sliding-cv", 0, 1, "CV deslizante")
        model = spec.model_cls(**model_config)
        sliding_df = trainer.sliding_cross_validation(model=model, df=df, test_ratio=float(args.eval_size))
        sliding_df["Model"] = model_id
        sliding_df["Model Type"] = model.__class__
        model_config["train"]["results"]["sliding-cv"] = sliding_df
        emit_training_progress(progress_callback, "sliding-cv", 1, 1, "CV deslizante completado")

    emit_training_progress(progress_callback, "fit", 0, 1, "Entrenamiento final")
    train_df, eval_df = train_test_split(df=df, test_size=float(args.eval_size))
    model = spec.model_cls(**model_config)
    model, fit_df = trainer.train(model=model, train_df=train_df, eval_df=eval_df, check_nan=True)
    fit_df["Model"] = model_id
    fit_df["Model Type"] = model.__class__
    model_config["cls"] = model.__class__
    model_config["train"]["results"]["fit"] = fit_df

    if isinstance(model, NeuralNetwork):
        model_config.update({"input_size": model.input_size, "num_classes": model.num_classes})

    model_config["train"]["optuna"] = optuna_summary
    model_db.save_model(model=model, model_config=model_config)
    emit_training_progress(progress_callback, "complete", 1, 1, "Entrenamiento completado")
    return {
        "model": model_payload(model_id, model_config),
        "optuna": optuna_summary,
        "results": {name: table_payload(result_df, page=1, page_size=50) for name, result_df in model_config["train"]["results"].items()},
    }


def evaluate_model(league_id: str, model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    _, _, df = load_league(league_id)
    df = df.dropna(ignore_index=True)
    model_db = ModelDatabase(league_id=league_id)
    model, config = load_model(model_db, model_id)
    target_type = config["target_type"]
    dataset_key = str(payload.get("dataset", "all")).capitalize()
    odd_range = parse_eval_odd_range(payload.get("odd_filter"))

    if payload.get("delete_filter"):
        _delete_stored_filter(model_db, config, odd_range)
        return {"deleted_filter": str(odd_range)}

    y_prob = model.predict_proba(df=df)
    y_pred = y_prob.argmax(axis=1)
    y_prob = y_prob.round(2)
    y_true = construct_targets(df=df, target_type=target_type)

    dataset_masks = _dataset_masks(df=df, config=config)
    if dataset_key not in dataset_masks:
        raise CLIError("Los datos deben ser: todos, entrenamiento o evaluacion.")
    dataset_mask = dataset_masks[dataset_key]
    prob_percentiles = _compute_probability_percentiles(
        y_prob=y_prob[dataset_mask],
        target_type=target_type,
        p1=int(payload.get("p1", 0)),
        px=int(payload.get("px", 0)),
        p2=int(payload.get("p2", 0)),
        pu=int(payload.get("pu", 0)),
        po=int(payload.get("po", 0)),
    )
    filter_mask = _evaluation_mask(df=df, y_prob=y_prob, target_type=target_type, dataset_mask=dataset_mask, odd_range=odd_range, prob_percentiles=prob_percentiles)

    metrics = model.compute_metrics(y_true=y_true[filter_mask], y_pred=y_pred[filter_mask])
    metrics["Correct"] = int((y_true[filter_mask] == y_pred[filter_mask]).sum())
    metrics["Total"] = int(filter_mask.sum())
    metrics["Prof. Balance"] = _profit_balance(df=df, y_pred=y_pred[filter_mask], filter_mask=filter_mask, target_type=target_type)
    confusion_df = confusion_matrix_dataframe(target_type=target_type, y_true=y_true[filter_mask], y_pred=y_pred[filter_mask])

    output_df = prediction_output_dataframe(df=df, target_type=target_type, y_pred=y_pred, y_prob=y_prob)
    output_df = output_df[filter_mask].reset_index(drop=True)
    result = {
        "metrics": table_payload(metrics, page=1, page_size=10),
        "confusion_matrix": table_payload(confusion_df, page=1, page_size=10),
        "rows": table_payload(output_df, page=int(payload.get("page", 1)), page_size=int(payload.get("page_size", 50))),
        "percentiles": jsonable(prob_percentiles),
    }
    if payload.get("seasonal"):
        seasonal_df = _seasonal_metrics(df=df, target_type=target_type, y_true=y_true, y_pred=y_pred, filter_mask=filter_mask, model=model)
        result["seasonal"] = table_payload(seasonal_df, page=1, page_size=100)
    if payload.get("store_filter"):
        _store_filter(model_db=model_db, model_config=config, odd_range=odd_range, prob_percentiles=prob_percentiles)
        result["stored_filter"] = str(odd_range)
    return result


def delete_model(league_id: str, model_id: str) -> Dict[str, str]:
    load_league(league_id)
    model_db = ModelDatabase(league_id=league_id)
    if not model_db.model_exists(model_id):
        raise CLIError(f'El modelo "{model_id}" no existe.')
    model_db.delete_model(model_id)
    return {"deleted": model_id}


def upcoming_fixtures(league_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    _, league, df = load_league(league_id)
    days = min(max(int(payload.get("days") or UPCOMING_FIXTURE_DAYS), 1), 30)
    limit = min(max(int(payload.get("limit") or 100), 1), 500)
    fixture_df = scrape_upcoming_fixtures(
        league=league,
        league_df=df,
        days=days,
        limit=limit,
        headless=payload.get("headless"),
        match_teams=True,
    )
    return {
        "fixtures": table_payload(fixture_df[PREDICT_FIXTURE_COLUMNS], page=1, page_size=limit),
        "days": days,
        "limit": limit,
    }


def fixture_prediction(league_id: str, payload: Dict[str, Any], fixture_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    _, league, df = load_league(league_id)
    model_db = ModelDatabase(league_id=league_id)
    model, config = load_model(model_db, payload.get("model_id") or payload.get("model"))

    if fixture_df is None:
        if payload.get("fixtures"):
            fixture_df = fixture_rows_from_payload(payload.get("fixtures"))
        elif payload.get("date"):
            fixture_df = scrape_fixtures(league, str(payload.get("date")), payload.get("headless"))
        else:
            raise CLIError("Selecciona al menos un partido futuro para predecir.")

    fixture_df = _validate_fixture_rows(fixture_df, df)
    odd_mask = _league_odd_mask(league=league, fixture_df=fixture_df)
    prepared_df = construct_inputs_by_fixture(df=df, fixture_df=fixture_df)
    y_prob = model.predict_proba(df=prepared_df).round(2)
    y_pred = y_prob.argmax(axis=1)

    output_columns = [column for column in PREDICT_FIXTURE_COLUMNS if column in fixture_df.columns]
    output_df = fixture_df[output_columns].copy()
    output_df = _append_prediction_columns(output_df, target_type=config["target_type"], y_pred=y_pred, y_prob=y_prob)
    selected_mask = _fixture_selected_mask(
        output_df=output_df,
        y_prob=y_prob,
        target_type=config["target_type"],
        model_config=config,
        requested_filters=payload.get("filters"),
        base_mask=odd_mask,
    )
    output_df["Selected"] = selected_mask
    export_df = output_df if payload.get("include_all") else output_df[selected_mask].reset_index(drop=True)
    output = output_path(f"{league_id}-fixtures", ".csv")
    export_dataframe(export_df, str(output))
    return {"predictions": table_payload(output_df, page=1, page_size=100), "export": {"path": str(output), "url": output_url(output)}}


def analysis_plot(league_id: str, analysis_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    _, _, df = load_league(league_id)
    df = df.dropna()
    colormap = COLORMAP_OPTIONS.get(payload.get("colormap", "Blues"), "Blues")
    season = payload.get("season")
    season = None if season in {"", None} else int(season)
    output = output_path(f"{league_id}-{analysis_type}", ".png")

    if analysis_type == "descriptive":
        analyzer = DescriptiveAnalyzer(df=df)
        ax = analyzer.generate_plot(season=season, colormap=colormap, feature_type=payload.get("feature_type", "home"))
        save_figure(ax, str(output))
        table = _descriptive_table(df=df, feature_type=payload.get("feature_type", "home"))
        return {"image": image_payload(output), "table": table_payload(table.reset_index(names="metric"), page=1, page_size=50)}
    if analysis_type == "distributions":
        analyzer = DistributionAnalyzer(df=df)
        column = payload.get("column")
        if column not in analyzer.all_features:
            raise CLIError(f'Columna de distribucion desconocida: "{column}".')
        ax = analyzer.generate_plot(season=season, colormap=colormap, column=column)
    elif analysis_type == "variance":
        ax = VarianceAnalyzer(df=df).generate_plot(season=season, colormap=colormap)
    elif analysis_type == "correlation":
        ax = CorrelationAnalyzer(df=df).generate_plot(
            season=season,
            colormap=colormap,
            method=payload.get("method", "pearson"),
            feature_type=payload.get("feature_type", "home"),
        )
    elif analysis_type == "boruta":
        ax = BorutaAnalyzer(df=df).generate_plot(season=season, colormap=colormap, target_type=parse_target(payload.get("target", "result")))
    elif analysis_type == "coefficients":
        ax = CoefficientAnalyzer(df=df).generate_plot(season=season, colormap=colormap, target_type=parse_target(payload.get("target", "result")))
    elif analysis_type == "impurity":
        ax = GiniImpurityAnalyzer(df=df).generate_plot(season=season, colormap=colormap, target_type=parse_target(payload.get("target", "result")))
    elif analysis_type == "rules":
        ax = RuleExtractorAnalyzer(df=df).generate_plot(season=season, target_type=parse_target(payload.get("target", "result")), max_depth=int(payload.get("depth", 3)))
    else:
        raise CLIError(f'Tipo de analisis desconocido: "{analysis_type}".')

    save_figure(ax, str(output))
    return {"image": image_payload(output)}


def explain_plot(league_id: str, model_id: str, plot_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    compute_shap = plot_type in {"waterfall", "shap"}
    explainer, model = load_explainer(league_id, model_id, compute_shap=compute_shap)
    output = output_path(f"{league_id}-{model_id}-{plot_type}", ".png")

    if plot_type == "boundary":
        ax = explainer.boundary_plot(_parse_feature_pair(payload.get("features")))
    elif plot_type == "pdp":
        validate_target_label(model.target_type, payload.get("target"))
        ax = explainer.partial_dependence_plot(feature=payload.get("feature"), target=payload.get("target"))
    elif plot_type == "waterfall":
        validate_target_label(model.target_type, payload.get("target"))
        ax = explainer.instance_waterfall_plot(match_index=int(payload.get("match_index", 0)), target=payload.get("target"))
    elif plot_type == "shap":
        validate_target_label(model.target_type, payload.get("target"))
        ax = explainer.shap_bar_plot(target=payload.get("target"), clustering=bool(payload.get("cluster", True)))
        if ax is None:
            raise CLIError("Este explicador no entrega valores SHAP para el modelo seleccionado.")
    elif plot_type == "extra":
        ax = extra_explainer_plot(explainer, payload)
    else:
        raise CLIError(f'Grafico de explicacion desconocido: "{plot_type}".')

    save_figure(ax, str(output))
    return {"image": image_payload(output)}


def browser_config() -> Dict[str, Any]:
    data = DEFAULT_BROWSER_CONFIG.copy()
    with open("storage/network/browser.json", "r") as file:
        data.update(json.load(file))
    data["application"] = str(data.get("application") or DEFAULT_BROWSER_CONFIG["application"]).lower()
    data["headless"] = bool(data.get("headless", DEFAULT_BROWSER_CONFIG["headless"]))
    data["brave_binary"] = str(data.get("brave_binary") or "")
    return data


def update_browser_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = browser_config()
    application = payload.get("application")
    if application is not None:
        application = str(application).lower()
        if application not in SUPPORTED_BROWSERS:
            raise CLIError("El navegador debe ser chrome, firefox, edge o brave.")
        data["application"] = application
    if "headless" in payload:
        data["headless"] = bool(payload["headless"])
    if "brave_binary" in payload:
        data["brave_binary"] = str(payload.get("brave_binary") or "").strip()
    with open("storage/network/browser.json", "w") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    return data


def load_league(league_id: str, update: bool = False):
    db = LeagueDatabase()
    if not db.league_exists(league_id):
        raise CLIError(f'La liga "{league_id}" no existe.')
    df = db.update_league(league_id) if update else db.load_league(league_id)
    if df is None:
        raise CLIError(f'No se pudo cargar la liga "{league_id}".')
    return db, db.index[league_id], df.reset_index(drop=True)


def load_model(model_db: ModelDatabase, model_id: Optional[str]):
    if not model_id:
        raise CLIError("El id del modelo es obligatorio.")
    if not model_db.model_exists(model_id):
        raise CLIError(f'El modelo "{model_id}" no existe.')
    config = model_db.load_model_config(model_id=model_id)
    if config is None:
        raise CLIError(f'No se pudo cargar el modelo "{model_id}".')
    if not is_supported_model_config(config):
        raise CLIError(f'El modelo "{model_id}" usa un tipo no soportado en esta version.')
    model, config = model_db.load_model(model_id=model_id)
    if model is None or config is None:
        raise CLIError(f'No se pudo cargar el modelo "{model_id}".')
    return model, config


def read_fixture_upload(filename: str, content: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(BytesIO(content))
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(BytesIO(content))
    else:
        raise CLIError("El archivo de partidos debe ser .csv o .xlsx.")
    load_required_columns(df, ["Home", "Away", "1", "X", "2"], "Fixture input")
    return df[["Home", "Away", "1", "X", "2"]].copy()


def fixture_rows_from_payload(rows: Any) -> pd.DataFrame:
    if not isinstance(rows, list) or not rows:
        raise CLIError("Selecciona al menos un partido futuro para predecir.")
    df = pd.DataFrame(rows)
    load_required_columns(df, ["Home", "Away", "1", "X", "2"], "Fixture input")
    for column in ["1", "X", "2"]:
        df[column] = df[column].apply(_valid_odd)
    for column in ["Date", "Hora MX"]:
        if column not in df.columns:
            df[column] = ""
    return df[PREDICT_FIXTURE_COLUMNS].copy()


def scrape_fixtures(league, date_text: str, headless: Optional[bool]) -> pd.DataFrame:
    try:
        selected_date = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise CLIError("La fecha debe usar el formato AAAA-MM-DD.") from exc
    footystats_date = selected_date.strftime("%b %d").replace(" 0", " ")

    scraper = FootyStatsScraper(headless=headless)
    try:
        loaded = scraper.load_page(league.fixture)
        if not loaded:
            raise CLIError("No se pudo cargar FootyStats. Revisa internet, driver del navegador y URL de fixtures.")
        parsed = scraper.parse_fixture_table(date_str=footystats_date)
    finally:
        scraper.quit()
    if parsed is None or parsed.empty:
        raise CLIError(f"No se encontraron partidos para {date_text}.")
    parsed = parsed.copy()
    parsed.insert(0, "Date", selected_date.isoformat())
    if "Hora MX" not in parsed.columns:
        parsed["Hora MX"] = "No disponible"
    return match_fixture_teams(parsed_teams_df=parsed, league_df=load_league(league.league_id)[2])


def scrape_upcoming_fixtures(
        league,
        league_df: pd.DataFrame,
        days: int = UPCOMING_FIXTURE_DAYS,
        limit: Optional[int] = None,
        headless: Optional[bool] = None,
        match_teams: bool = True,
) -> pd.DataFrame:
    if not league.fixture:
        return pd.DataFrame(columns=PREDICT_FIXTURE_COLUMNS)

    today_mx = datetime.now(tz=MEXICO_CITY_TZ).date()
    target_dates = [today_mx + timedelta(days=offset) for offset in range(max(int(days or 1), 1))]
    footystats_dates = {
        target_date.strftime("%b %d").replace(" 0", " "): target_date
        for target_date in target_dates
    }

    scraper = FootyStatsScraper(headless=headless)
    try:
        loaded = scraper.load_page(league.fixture)
        if not loaded:
            raise CLIError("No se pudo cargar FootyStats. Revisa internet, driver del navegador y URL de fixtures.")
        parsed_by_date = scraper.parse_fixture_tables(date_strs=list(footystats_dates.keys()))
    finally:
        scraper.quit()

    frames = []
    for footystats_date, target_date in footystats_dates.items():
        parsed = parsed_by_date.get(footystats_date)
        if parsed is None or parsed.empty:
            continue
        parsed = parsed.copy()
        parsed.insert(0, "Date", target_date.isoformat())
        if "Hora MX" not in parsed.columns:
            parsed["Hora MX"] = "No disponible"
        frames.append(parsed)

    if not frames:
        return pd.DataFrame(columns=PREDICT_FIXTURE_COLUMNS)

    fixture_df = pd.concat(frames, ignore_index=True)
    if match_teams:
        fixture_df = match_fixture_teams(parsed_teams_df=fixture_df, league_df=league_df)

    for column in ["1", "X", "2"]:
        fixture_df[column] = pd.to_numeric(fixture_df[column], errors="coerce")
    fixture_df = fixture_df.dropna(subset=["Home", "Away", "1", "X", "2"]).reset_index(drop=True)
    fixture_df = fixture_df[PREDICT_FIXTURE_COLUMNS]
    if limit is not None:
        fixture_df = fixture_df.head(max(int(limit), 0))
    return fixture_df


def load_explainer(league_id: str, model_id: str, compute_shap: bool):
    _, _, df = load_league(league_id)
    df = df.dropna(ignore_index=True)
    model_db = ModelDatabase(league_id=league_id)
    model, _ = load_model(model_db, model_id)
    explainer_cls = None
    for model_cls, candidate in EXPLAINER_BY_MODEL.items():
        if isinstance(model, model_cls):
            explainer_cls = candidate
            break
    if explainer_cls is None:
        raise CLIError(f"No hay explicador registrado para {model.__class__.__name__}.")
    explainer = explainer_cls(model=model, df=df)
    if compute_shap:
        explainer.compute_shap_values()
        if explainer.shap_values is None:
            raise CLIError("Este modelo o explicador no entrega valores SHAP para este grafico.")
    return explainer, model


def prediction_output_dataframe(df: pd.DataFrame, target_type: TargetType, y_pred: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    base_columns = [col for col in ["Date", "Season", "Week", "Home", "Away", "1", "X", "2", "Result", "Result-U/O"] if col in df.columns]
    output_df = df[base_columns].copy()
    return _append_prediction_columns(output_df, target_type=target_type, y_pred=y_pred, y_prob=y_prob)


def validate_target_label(target_type: TargetType, target: Optional[str]):
    allowed = ["H", "D", "A"] if target_type == TargetType.RESULT else ["U", "O"]
    if target not in allowed:
        raise CLIError(f'Objetivo invalido "{target}". Objetivos validos para {target_label(target_type)}: {", ".join(allowed)}.')


def confusion_matrix_dataframe(target_type: TargetType, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    if target_type == TargetType.RESULT:
        labels = [(0, "H"), (1, "D"), (2, "A")]
    elif target_type == TargetType.OVER_UNDER:
        labels = [(0, "U"), (1, "O")]
    else:
        raise TypeError(f'Not supported target type: "{type(target_type)}"')

    rows = []
    for actual_value, actual_label in labels:
        row = {"Real": actual_label}
        actual_mask = y_true == actual_value
        for predicted_value, predicted_label in labels:
            row[f"Pred {predicted_label}"] = int((actual_mask & (y_pred == predicted_value)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def catalog_template(db: LeagueDatabase, payload: Dict[str, Any]):
    if payload.get("league_index") is not None:
        idx = int(payload["league_index"])
        if idx < 1 or idx > len(db.leagues):
            raise CLIError(f"El indice de liga debe estar entre 1 y {len(db.leagues)}.")
        return db.leagues[idx - 1]
    country = payload.get("country")
    name = payload.get("name")
    matches = [
        league for league in db.leagues
        if (not country or league.country.lower() == str(country).lower())
        and (not name or league.name.lower() == str(name).lower())
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise CLIError("Varias ligas del catalogo coinciden; usa league_index.")
    raise CLIError("No se encontro una liga del catalogo.")


def league_display_name(league) -> str:
    return f"{league.country} / {league.name}"


def default_league_id(league) -> str:
    return f"{slugify(league.country)}-{slugify(league.name)}-{league.start_year}"


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return slug or "liga"


def country_flag_url(country: str) -> str:
    flag_path = COUNTRY_FLAGS_ROOT / f"{country}.png"
    if flag_path.exists():
        return f"/assets/graphics/countries/{country}.png"
    return ""


def category_label(category: str) -> str:
    labels = {"main": "Principal", "extra": "Extra"}
    return labels.get(str(category), str(category) or "Principal")


def odd_range_payload(value) -> Dict[str, Any]:
    if not value:
        return {"active": False, "min": "", "max": "", "label": "Todas"}
    low, high = value
    return {"active": True, "min": float(low), "max": float(high), "label": f"{float(low):g}:{float(high):g}"}


def display_config_value(value: Any, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, type):
        return value.__name__
    text = str(value)
    return default if text in {"", "None"} else text


def display_defaults(defaults: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): "automatico" if value is None else jsonable(value)
        for key, value in defaults.items()
    }


def league_payload(league) -> Dict[str, Any]:
    return {
        "league_id": league.league_id or default_league_id(league),
        "country": league.country,
        "name": league.name,
        "display_name": league_display_name(league),
        "category": category_label(league.category),
        "start_year": int(league.start_year),
        "history_window": int(league.match_history_window or 3),
        "goal_margin": int(league.goal_diff_margin or 2),
        "stats": "Todas" if league.stats_columns is None else ", ".join(league.stats_columns),
        "odd_1": odd_range_payload(league.odd_1_range),
        "odd_x": odd_range_payload(league.odd_x_range),
        "odd_2": odd_range_payload(league.odd_2_range),
        "fixture": league.fixture or "",
        "flag_url": country_flag_url(league.country),
    }


def model_payload(model_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = config.get("train", {})
    return {
        "model_id": model_id,
        "class": display_config_value(config.get("cls"), default="Modelo"),
        "target": jsonable(config.get("target_type")),
        "normalizer": display_config_value(config.get("normalizer"), default="Ninguno"),
        "sampler": display_config_value(config.get("sampler"), default="Ninguno"),
        "eval_size": train_cfg.get("eval_samples_size", 20.0),
        "filters": jsonable(config.get("eval", {}).get("percentiles", {})),
    }


def is_supported_model_config(config: Dict[str, Any]) -> bool:
    supported_classes = {spec.model_cls for spec in MODEL_SPECS.values()}
    return config.get("cls") in supported_classes


def bool_payload(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "si", "sí", "on"}


def normalize_optuna_choice(value: Any, allowed: set[str], label: str) -> str:
    key = str(value or "").strip().lower().replace("_", "-")
    if key not in allowed:
        raise CLIError(f'Optuna {label} invalido: "{value}".')
    return key


def clean_error_text(exc: Exception) -> str:
    return re.sub(r"^(CLIError|ValueError|RuntimeError|NotImplementedError):\s*", "", f"{exc.__class__.__name__}: {exc}")


def emit_training_progress(callback, stage: str, current: int, total: int, message: str, **extra):
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


def filter_dataframe(
        df: pd.DataFrame,
        query: Optional[str],
        column: Optional[str],
        exact: bool,
        hide_missing: bool,
        columns: Optional[str],
) -> pd.DataFrame:
    if hide_missing:
        df = df.dropna(ignore_index=True)
    if query:
        if column:
            if column not in df.columns:
                raise CLIError(f'Columna desconocida: "{column}".')
            search_df = df[[column]]
        else:
            search_df = df
        if exact:
            mask = search_df.astype(str).eq(str(query)).any(axis=1)
        else:
            mask = search_df.astype(str).apply(lambda col: col.str.contains(str(query), case=False, na=False)).any(axis=1)
        df = df[mask].reset_index(drop=True)
    selected_columns = parse_columns(columns)
    if selected_columns:
        missing = [col for col in selected_columns if col not in df.columns]
        if missing:
            raise CLIError(f"Columnas desconocidas: {', '.join(missing)}")
        df = df[selected_columns]
    return df


def training_args(payload: Dict[str, Any], league_id: str, model_id: str, model_key: str) -> SimpleNamespace:
    defaults = {
        "league_id": league_id,
        "model_id": model_id,
        "model_type": model_key,
        "target": "result",
        "normalizer": "none",
        "sampler": "none",
        "calibrate": True,
        "eval_size": 20.0,
        "cv": True,
        "sliding_cv": True,
        "tuning_enabled": False,
        "tune": None,
        "trials": 25,
        "optuna_sampler": "tpe",
        "optuna_pruner": "none",
        "objective": "Accuracy",
        "export_metrics": None,
        "max_depth": None,
        "n_estimators": None,
        "min_child_weight": None,
        "learning_rate": None,
        "lambda_regularization": None,
        "alpha_regularization": None,
        "num_leaves": None,
        "min_child_samples": None,
        "minibatch_frac": None,
        "natural_gradient": None,
        "l2_leaf_reg": None,
        "random_strength": None,
    }
    aliases = {
        "eval-size": "eval_size",
        "sliding-cv": "sliding_cv",
        "model-id": "model_id",
        "n_trials": "trials",
        "tune_params": "tune",
        "optuna-sampler": "optuna_sampler",
        "optuna-pruner": "optuna_pruner",
    }
    for key, value in payload.items():
        defaults[aliases.get(key, key)] = value
    return SimpleNamespace(**defaults)


def extra_explainer_plot(explainer, payload: Dict[str, Any]):
    plot = payload.get("plot")
    if plot == "coefficients":
        if not isinstance(explainer, (LogisticRegressionExplainer, SVMExplainer)):
            raise CLIError("El grafico de coeficientes esta disponible para regresion logistica y SVM lineal.")
        return explainer.coefficients_bar_plot()
    if plot == "model":
        if isinstance(explainer, LogisticRegressionExplainer):
            return explainer.visualize_model(feature=payload.get("feature"))
        if isinstance(explainer, DiscriminantAnalysisExplainer):
            return explainer.visualize_model()
        if isinstance(explainer, SVMExplainer):
            return explainer.visualize_model(features=_parse_feature_pair(payload.get("features")))
        if isinstance(explainer, KNNExplainer):
            return explainer.visualize_model(features=_parse_feature_pair(payload.get("features")), match_index=int(payload.get("match_index", 0)))
        raise CLIError("La visualizacion del modelo no esta disponible para este tipo de modelo.")
    if plot == "impurity":
        if not isinstance(explainer, (DecisionTreeExplainer, RandomForestExplainer, ExtremeBoostingExplainer)):
            raise CLIError("El grafico de impureza esta disponible para modelos basados en arboles.")
        return explainer.feature_impurity_bar_plot()
    if plot == "tree":
        if isinstance(explainer, RandomForestExplainer):
            return explainer.plot_tree_rules(max_depth=int(payload.get("depth", 3)), estimator_id=int(payload.get("estimator_id", 0)))
        if isinstance(explainer, DecisionTreeExplainer):
            return explainer.plot_tree_rules(max_depth=int(payload.get("depth", 3)))
        raise CLIError("La visualizacion de reglas esta disponible para arbol de decision y Random Forest.")
    if plot == "attention":
        if not isinstance(explainer, NeuralNetworkExplainer):
            raise CLIError("El grafico de atencion esta disponible solo para modelos DNN con soporte VSN.")
        ax = explainer.plot_attention_scores()
        if ax is None:
            raise CLIError("Este modelo DNN no incluye puntajes de atencion o VSN.")
        return ax
    raise CLIError("La explicacion extra requiere plot: coefficients, model, impurity, tree o attention.")


def output_path(stem: str, suffix: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in stem)
    return OUTPUT_ROOT / f"{safe_stem}-{timestamp}{suffix}"


def output_url(path: Path) -> str:
    return "/" + str(path).replace("\\", "/")


def image_payload(path: Path) -> Dict[str, str]:
    return {"path": str(path), "url": output_url(path)}
