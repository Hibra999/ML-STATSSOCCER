from __future__ import annotations

import json
import math
import os
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

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
from src.cli.model_specs import MODEL_SPECS, build_model_params, normalize_model_key, tunable_params_for_args
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
from src.preprocessing.utils.inputs import construct_inputs_by_fixture, construct_inputs_by_teams
from src.preprocessing.utils.target import TargetType, construct_targets


OUTPUT_ROOT = Path("outputs") / "web"


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
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def table_payload(df: pd.DataFrame, page: int = 1, page_size: int = 50) -> Dict[str, Any]:
    page = max(int(page or 1), 1)
    page_size = min(max(int(page_size or 50), 1), 500)
    total = int(df.shape[0])
    start = (page - 1) * page_size
    page_df = df.iloc[start:start + page_size].copy()
    page_df = page_df.astype(object).where(pd.notna(page_df), None)
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


def model_specs() -> List[Dict[str, Any]]:
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "supports_calibration": spec.supports_calibration,
            "defaults": jsonable(spec.defaults),
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
            "category": league.category,
            "start_year": league.start_year,
            "fixture": league.fixture,
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
        raise CLIError(f'League "{league_id}" already exists.')

    current_year_threshold = date.today().year - 4
    start_year = int(payload.get("start_year") or template.start_year)
    if start_year < template.start_year:
        raise CLIError(f"{template.name} starts at {template.start_year}; requested {start_year}.")
    if start_year > current_year_threshold:
        raise CLIError(f"Start year cannot be newer than {current_year_threshold}.")

    history_window = int(payload.get("history_window", 3))
    goal_margin = int(payload.get("goal_margin", 2))
    if history_window < 2 or history_window > 5:
        raise CLIError("history_window must be between 2 and 5.")
    if goal_margin < 2 or goal_margin > 5:
        raise CLIError("goal_margin must be between 2 and 5.")

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
        raise CLIError("League download failed. Check internet access and source availability.")
    return {"league": league_payload(league), "preview": table_payload(df.head(25), page=1, page_size=25)}


def update_league(league_id: str) -> Dict[str, Any]:
    return league_detail(league_id, update=True)


def delete_league(league_id: str) -> Dict[str, str]:
    db = LeagueDatabase()
    if not db.league_exists(league_id):
        raise CLIError(f'League "{league_id}" does not exist.')
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
        if config is not None:
            models.append(model_payload(model_id, config))
    return models


def train_model(league_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    model_key = normalize_model_key(payload.get("model_type", "random-forest"))
    spec = MODEL_SPECS[model_key]
    _, league, df = load_league(league_id)
    df = df.dropna(ignore_index=True)
    if df.empty:
        raise CLIError("Training dataset has no complete rows after dropping missing values.")

    model_db = ModelDatabase(league_id=league.league_id)
    model_id = validate_identifier(payload.get("model_id") or payload.get("id") or f"{league.league_id}-{model_key}", "model id")
    if model_db.model_exists(model_id):
        raise CLIError(f'Model "{model_id}" already exists for league "{league.league_id}".')

    args = training_args(payload, league_id=league.league_id, model_id=model_id, model_key=model_key)
    if args.eval_size < 5 or args.eval_size > 30:
        raise CLIError("eval_size must be between 5 and 30 percent.")

    model_config = build_model_params(args=args, league_id=league.league_id, model_id=model_id, model_key=model_key)
    model_config["train"] = {"eval_samples_size": float(args.eval_size), "results": {}}

    trainer = Trainer()
    tunable_params = tunable_params_for_args(args, spec)
    if tunable_params:
        tuner = Tuner(model_cls=spec.model_cls, fixed_params=model_config, tunable_params=tunable_params, df=df, metric=args.objective)
        study = tuner.tune(trials=args.trials)
        trials_df = _study_to_dataframe(study, args.objective)
        model_config["train"]["results"]["tune"] = trials_df
        model_config.update(**study.best_trial.params)

    if args.cv:
        model = spec.model_cls(**model_config)
        cv_df = trainer.cross_validation(model=model, df=df)
        cv_df["Model"] = model_id
        cv_df["Model Type"] = model.__class__
        model_config["train"]["results"]["cv"] = cv_df

    if args.sliding_cv:
        model = spec.model_cls(**model_config)
        sliding_df = trainer.sliding_cross_validation(model=model, df=df, test_ratio=float(args.eval_size))
        sliding_df["Model"] = model_id
        sliding_df["Model Type"] = model.__class__
        model_config["train"]["results"]["sliding-cv"] = sliding_df

    train_df, eval_df = train_test_split(df=df, test_size=float(args.eval_size))
    model = spec.model_cls(**model_config)
    model, fit_df = trainer.train(model=model, train_df=train_df, eval_df=eval_df, check_nan=True)
    fit_df["Model"] = model_id
    fit_df["Model Type"] = model.__class__
    model_config["cls"] = model.__class__
    model_config["train"]["results"]["fit"] = fit_df

    if isinstance(model, NeuralNetwork):
        model_config.update({"input_size": model.input_size, "num_classes": model.num_classes})

    model_db.save_model(model=model, model_config=model_config)
    return {
        "model": model_payload(model_id, model_config),
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
        raise CLIError("dataset must be all, train or eval.")
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

    output_df = prediction_output_dataframe(df=df, target_type=target_type, y_pred=y_pred, y_prob=y_prob)
    output_df = output_df[filter_mask].reset_index(drop=True)
    result = {
        "metrics": table_payload(metrics, page=1, page_size=10),
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
        raise CLIError(f'Model "{model_id}" does not exist.')
    model_db.delete_model(model_id)
    return {"deleted": model_id}


def manual_prediction(league_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    _, _, df = load_league(league_id)
    model_db = ModelDatabase(league_id=league_id)
    model, config = load_model(model_db, payload.get("model_id") or payload.get("model"))

    home = payload.get("home")
    away = payload.get("away")
    home_teams = sorted(df["Home"].dropna().unique().tolist())
    away_teams = sorted(df["Away"].dropna().unique().tolist())
    if home not in home_teams:
        raise CLIError(f'Unknown home team "{home}".')
    if away not in away_teams:
        raise CLIError(f'Unknown away team "{away}".')
    if home == away:
        raise CLIError("Home and away teams must be different.")

    match_df = pd.DataFrame({
        "Date": [date.today().strftime("%Y-%m-%d")],
        "Home": [home],
        "Away": [away],
        "1": [_valid_odd(payload.get("odd_1"))],
        "X": [_valid_odd(payload.get("odd_x"))],
        "2": [_valid_odd(payload.get("odd_2"))],
    })
    prepared_df = construct_inputs_by_teams(df=df, match_df=match_df)
    y_prob = model.predict_proba(df=prepared_df).round(2)
    y_pred = y_prob.argmax(axis=1)
    output_df = prepared_df[["Date", "Season", "Week", "Home", "Away", "1", "X", "2"]].copy()
    output_df = _append_prediction_columns(output_df, target_type=config["target_type"], y_pred=y_pred, y_prob=y_prob)
    return {"prediction": table_payload(output_df, page=1, page_size=5)}


def fixture_prediction(league_id: str, payload: Dict[str, Any], fixture_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    _, league, df = load_league(league_id)
    model_db = ModelDatabase(league_id=league_id)
    model, config = load_model(model_db, payload.get("model_id") or payload.get("model"))

    if fixture_df is None:
        if not payload.get("date"):
            raise CLIError("date is required when no fixture file is uploaded.")
        fixture_df = scrape_fixtures(league, str(payload.get("date")), payload.get("headless"))

    fixture_df = _validate_fixture_rows(fixture_df, df)
    odd_mask = _league_odd_mask(league=league, fixture_df=fixture_df)
    prepared_df = construct_inputs_by_fixture(df=df, fixture_df=fixture_df)
    y_prob = model.predict_proba(df=prepared_df).round(2)
    y_pred = y_prob.argmax(axis=1)

    output_df = fixture_df[["Home", "Away", "1", "X", "2"]].copy()
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
            raise CLIError(f'Unknown distribution column "{column}".')
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
        raise CLIError(f'Unknown analysis type "{analysis_type}".')

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
            raise CLIError("This explainer does not provide SHAP values for the selected model.")
    elif plot_type == "extra":
        ax = extra_explainer_plot(explainer, payload)
    else:
        raise CLIError(f'Unknown explain plot "{plot_type}".')

    save_figure(ax, str(output))
    return {"image": image_payload(output)}


def browser_config() -> Dict[str, Any]:
    with open("storage/network/browser.json", "r") as file:
        return json.load(file)


def update_browser_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = browser_config()
    application = payload.get("application")
    if application is not None:
        if application not in {"chrome", "firefox", "edge"}:
            raise CLIError("application must be chrome, firefox or edge.")
        data["application"] = application
    if "headless" in payload:
        data["headless"] = bool(payload["headless"])
    with open("storage/network/browser.json", "w") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    return data


def load_league(league_id: str, update: bool = False):
    db = LeagueDatabase()
    if not db.league_exists(league_id):
        raise CLIError(f'League "{league_id}" does not exist.')
    df = db.update_league(league_id) if update else db.load_league(league_id)
    if df is None:
        raise CLIError(f'Could not load league "{league_id}".')
    return db, db.index[league_id], df.reset_index(drop=True)


def load_model(model_db: ModelDatabase, model_id: Optional[str]):
    if not model_id:
        raise CLIError("model_id is required.")
    if not model_db.model_exists(model_id):
        raise CLIError(f'Model "{model_id}" does not exist.')
    model, config = model_db.load_model(model_id=model_id)
    if model is None or config is None:
        raise CLIError(f'Model "{model_id}" could not be loaded.')
    return model, config


def read_fixture_upload(filename: str, content: bytes) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(BytesIO(content))
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(BytesIO(content))
    else:
        raise CLIError("Fixture input must be .csv or .xlsx.")
    load_required_columns(df, ["Home", "Away", "1", "X", "2"], "Fixture input")
    return df[["Home", "Away", "1", "X", "2"]].copy()


def scrape_fixtures(league, date_text: str, headless: Optional[bool]) -> pd.DataFrame:
    try:
        selected_date = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise CLIError("date must use YYYY-MM-DD format.") from exc
    footystats_date = selected_date.strftime("%b %d").replace(" 0", " ")

    scraper = FootyStatsScraper(headless=headless)
    try:
        loaded = scraper.load_page(league.fixture)
        if not loaded:
            raise CLIError("Could not load FootyStats page. Check internet, browser driver and fixture URL.")
        parsed = scraper.parse_fixture_table(date_str=footystats_date)
    finally:
        scraper.quit()
    if parsed is None or parsed.empty:
        raise CLIError(f"No fixtures found for {date_text}.")
    return match_fixture_teams(parsed_teams_df=parsed, league_df=load_league(league.league_id)[2])


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
        raise CLIError(f"No explainer registered for {model.__class__.__name__}.")
    explainer = explainer_cls(model=model, df=df)
    if compute_shap:
        explainer.compute_shap_values()
        if explainer.shap_values is None:
            raise CLIError("This model/explainer does not provide SHAP values for this plot.")
    return explainer, model


def prediction_output_dataframe(df: pd.DataFrame, target_type: TargetType, y_pred: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    base_columns = [col for col in ["Date", "Season", "Week", "Home", "Away", "1", "X", "2", "Result", "Result-U/O"] if col in df.columns]
    output_df = df[base_columns].copy()
    return _append_prediction_columns(output_df, target_type=target_type, y_pred=y_pred, y_prob=y_prob)


def validate_target_label(target_type: TargetType, target: Optional[str]):
    allowed = ["H", "D", "A"] if target_type == TargetType.RESULT else ["U", "O"]
    if target not in allowed:
        raise CLIError(f'Invalid target "{target}". Valid targets for {target_label(target_type)}: {", ".join(allowed)}.')


def catalog_template(db: LeagueDatabase, payload: Dict[str, Any]):
    if payload.get("league_index") is not None:
        idx = int(payload["league_index"])
        if idx < 1 or idx > len(db.leagues):
            raise CLIError(f"league_index must be between 1 and {len(db.leagues)}.")
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
        raise CLIError("Multiple catalog leagues matched; pass league_index.")
    raise CLIError("No catalog league matched.")


def league_payload(league) -> Dict[str, Any]:
    return {
        "league_id": league.league_id,
        "country": league.country,
        "name": league.name,
        "category": league.category,
        "start_year": league.start_year,
        "history_window": league.match_history_window,
        "goal_margin": league.goal_diff_margin,
        "stats": "all" if league.stats_columns is None else league.stats_columns,
        "odd_1": league.odd_1_range,
        "odd_x": league.odd_x_range,
        "odd_2": league.odd_2_range,
        "fixture": league.fixture,
    }


def model_payload(model_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = config.get("train", {})
    return {
        "model_id": model_id,
        "class": jsonable(config.get("cls")),
        "target": jsonable(config.get("target_type")),
        "normalizer": str(config.get("normalizer")),
        "sampler": str(config.get("sampler")),
        "eval_size": train_cfg.get("eval_samples_size"),
        "filters": jsonable(config.get("eval", {}).get("percentiles", {})),
    }


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
                raise CLIError(f'Unknown column "{column}".')
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
            raise CLIError(f"Unknown columns: {', '.join(missing)}")
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
        "tune": None,
        "trials": 25,
        "objective": "Accuracy",
        "export_metrics": None,
        "penalty": None,
        "oas": None,
        "decision_boundary": None,
        "criterion": None,
        "min_samples_leaf": None,
        "min_samples_split": None,
        "max_features": None,
        "max_depth": None,
        "class_weight": None,
        "n_estimators": None,
        "min_child_weight": None,
        "learning_rate": None,
        "lambda_regularization": None,
        "alpha_regularization": None,
        "n_neighbors": None,
        "weights": None,
        "p": None,
        "algorithm": None,
        "kernel": None,
        "degree": None,
        "gamma": None,
        "hidden_layers": None,
        "hidden_units": None,
        "hidden_activation": None,
        "vsn": None,
        "layer_normalization": None,
        "batch_normalization": None,
        "dropout_rate": None,
        "odd_noise_std": None,
        "optimizer": None,
        "lookahead": None,
        "label_smoothing": None,
        "batch_size": None,
        "epochs": None,
        "early_stopping_patience": None,
        "lr_decay_patience": None,
        "lr_decay_factor": None,
        "verbose": None,
    }
    aliases = {"eval-size": "eval_size", "sliding-cv": "sliding_cv", "model-id": "model_id"}
    for key, value in payload.items():
        defaults[aliases.get(key, key)] = value
    return SimpleNamespace(**defaults)


def extra_explainer_plot(explainer, payload: Dict[str, Any]):
    plot = payload.get("plot")
    if plot == "coefficients":
        if not isinstance(explainer, (LogisticRegressionExplainer, SVMExplainer)):
            raise CLIError("Coefficient plot is available for Logistic Regression and linear SVM models.")
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
        raise CLIError("Model visualization is not available for this model type.")
    if plot == "impurity":
        if not isinstance(explainer, (DecisionTreeExplainer, RandomForestExplainer, ExtremeBoostingExplainer)):
            raise CLIError("Impurity plot is available for tree-based models.")
        return explainer.feature_impurity_bar_plot()
    if plot == "tree":
        if isinstance(explainer, RandomForestExplainer):
            return explainer.plot_tree_rules(max_depth=int(payload.get("depth", 3)), estimator_id=int(payload.get("estimator_id", 0)))
        if isinstance(explainer, DecisionTreeExplainer):
            return explainer.plot_tree_rules(max_depth=int(payload.get("depth", 3)))
        raise CLIError("Tree rule visualization is available for Decision Tree and Random Forest.")
    if plot == "attention":
        if not isinstance(explainer, NeuralNetworkExplainer):
            raise CLIError("Attention plot is available only for DNN models with VSN support.")
        ax = explainer.plot_attention_scores()
        if ax is None:
            raise CLIError("This DNN model does not include attention/VSN scores.")
        return ax
    raise CLIError("extra explain requires plot: coefficients, model, impurity, tree or attention.")


def output_path(stem: str, suffix: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in stem)
    return OUTPUT_ROOT / f"{safe_stem}-{timestamp}{suffix}"


def output_url(path: Path) -> str:
    return "/" + str(path).replace("\\", "/")


def image_payload(path: Path) -> Dict[str, str]:
    return {"path": str(path), "url": output_url(path)}
