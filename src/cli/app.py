import argparse
import json
import math
import os
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

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
from src.cli.common import (
    CLIError,
    COLORMAP_OPTIONS,
    HELP_LINKS,
    console,
    confirm_or_abort,
    ensure_output_path,
    export_dataframe,
    load_required_columns,
    parse_columns,
    parse_eval_odd_range,
    parse_odd_range,
    parse_target,
    print_error,
    print_success,
    print_warning,
    prompt_choice,
    prompt_int,
    prompt_text,
    render_dataframe,
    render_mapping,
    save_figure,
    target_label,
    validate_identifier,
)
from src.cli.model_specs import (
    MODEL_SPECS,
    add_model_specific_arguments,
    build_model_params,
    normalize_model_key,
    tunable_params_for_args,
)
from src.database.league import LeagueDatabase
from src.database.model import ModelDatabase
from src.interpretability.explainers.decisiontree import DecisionTreeExplainer
from src.interpretability.explainers.discriminant import DiscriminantAnalysisExplainer
from src.interpretability.explainers.extremeboosting import ExtremeBoostingExplainer
from src.interpretability.explainers.knn import KNNExplainer
from src.interpretability.explainers.logistic import LogisticRegressionExplainer
from src.interpretability.explainers.naivebayes import NaiveBayesExplainer
from src.interpretability.explainers.nn import NeuralNetworkExplainer
from src.interpretability.explainers.randomforest import RandomForestExplainer
from src.interpretability.explainers.svm import SVMExplainer
from src.metrics.balance import compute_profit_balance
from src.models.classifiers.boosting import CatBoost, LightGBM, NGBoost
from src.models.classifiers.decisiontree import DecisionTree
from src.models.classifiers.discriminant import DiscriminantAnalysisClassifier
from src.models.classifiers.extremeboosting import XGBoost
from src.models.classifiers.knn import KNN
from src.models.classifiers.logistic import LogisticRegressor
from src.models.classifiers.naivebayes import NaiveBayes
from src.models.classifiers.neuralnets.nn import NeuralNetwork
from src.models.classifiers.randomforest import RandomForest
from src.models.classifiers.svm import SVM
from src.models.trainer import Trainer
from src.models.tuner import Tuner
from src.network.fixtures.footystats.scraper import FootyStatsScraper
from src.network.fixtures.utils import match_fixture_teams
from src.network.leagues.downloaders.extra import ExtraLeagueDownloader
from src.network.leagues.downloaders.main import MainLeagueDownloader
from src.network.leagues.league import League
from src.preprocessing.selection import train_test_split
from src.preprocessing.statistics import StatisticsEngine
from src.preprocessing.utils.inputs import construct_inputs_by_fixture
from src.preprocessing.utils.target import TargetType, construct_targets


ODD_RANGES = [
    "None",
    ("1", 1.00, 1.3), ("1", 1.31, 1.6), ("1", 1.61, 1.9), ("1", 1.91, 2.5), ("1", 2.5, 3.5), ("1", 3.51, 100),
    ("X", 1.00, 2.0), ("X", 2.0, 3.0), ("X", 3.01, 100),
    ("2", 1.00, 1.3), ("2", 1.31, 1.6), ("2", 1.61, 1.9), ("2", 1.91, 2.5), ("2", 2.5, 3.5), ("2", 3.51, 100),
]

RESULT_LABELS = np.array(["H", "D", "A"])
OVER_UNDER_LABELS = np.array(["U", "O"])

EXPLAINER_BY_MODEL = {
    LogisticRegressor: LogisticRegressionExplainer,
    DiscriminantAnalysisClassifier: DiscriminantAnalysisExplainer,
    DecisionTree: DecisionTreeExplainer,
    RandomForest: RandomForestExplainer,
    NGBoost: ExtremeBoostingExplainer,
    CatBoost: ExtremeBoostingExplainer,
    LightGBM: ExtremeBoostingExplainer,
    XGBoost: ExtremeBoostingExplainer,
    KNN: KNNExplainer,
    NaiveBayes: NaiveBayesExplainer,
    SVM: SVMExplainer,
    NeuralNetwork: NeuralNetworkExplainer,
}


def run(argv: Optional[List[str]] = None) -> int:
    warnings.filterwarnings("ignore")
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "handler"):
        parser.print_help()
        return 0

    try:
        args.handler(args)
    except CLIError as exc:
        print_error(str(exc))
        return 2
    except KeyboardInterrupt:
        print_warning("Operation interrupted.")
        return 130
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        print_error(f"{exc.__class__.__name__}: {exc}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prophitbet",
        description="Companion CLI for ML-STATSSOCCER. Run `python app.py` for the local web interface.",
    )
    parser.add_argument("--debug", action="store_true", help="Show full tracebacks for unexpected errors.")
    parser.set_defaults(handler=cmd_root_help, parser=parser)
    subparsers = parser.add_subparsers(dest="command")

    _build_league_parser(subparsers)
    _build_data_parser(subparsers)
    _build_model_parser(subparsers)
    _build_predict_parser(subparsers)
    _build_analysis_parser(subparsers)
    _build_explain_parser(subparsers)
    _build_config_parser(subparsers)

    resources = subparsers.add_parser("resources", help="Show learning, update, bug-report and donation links.")
    resources.set_defaults(handler=cmd_resources)

    return parser


def _build_league_parser(subparsers):
    league = subparsers.add_parser("league", help="Create, list, update, inspect and delete leagues.")
    league_sub = league.add_subparsers(dest="league_command", required=True)

    list_cmd = league_sub.add_parser("list", help="List saved leagues or available catalog leagues.")
    list_cmd.add_argument("--catalog", action="store_true", help="Show downloadable catalog instead of saved leagues.")
    list_cmd.set_defaults(handler=cmd_league_list)

    create = league_sub.add_parser("create", help="Create a new league by downloading historical data.")
    create.add_argument("--id", dest="league_id", help="Unique league id to store locally.")
    create.add_argument("--country", help="Catalog country, for example England.")
    create.add_argument("--name", help="Catalog league name, for example Premier-League.")
    create.add_argument("--template", help='Catalog selector "Country:League" or "Country-League".')
    create.add_argument("--league-index", type=int, help="1-based catalog index from `league list --catalog`.")
    create.add_argument("--start-year", type=int, help="Historical start year.")
    create.add_argument("--history-window", type=int, default=3, help="Previous matches used for statistics. Default: 3.")
    create.add_argument("--goal-margin", type=int, default=2, help="Goal difference margin. Default: 2.")
    create.add_argument(
        "--stats",
        default="all",
        help="Stats to compute: all, basic, extended, none or comma-separated columns.",
    )
    create.add_argument("--odd-1", help="Creation filter for odd 1 as MIN:MAX.")
    create.add_argument("--odd-x", help="Creation filter for odd X as MIN:MAX.")
    create.add_argument("--odd-2", help="Creation filter for odd 2 as MIN:MAX.")
    create.add_argument("--yes", action="store_true", help="Skip confirmation prompts.")
    create.set_defaults(handler=cmd_league_create)

    show = league_sub.add_parser("show", help="Show a saved league summary.")
    show.add_argument("league_id")
    show.add_argument("--rows", type=int, default=10, help="Preview rows.")
    show.add_argument("--update", action="store_true", help="Update before showing.")
    show.set_defaults(handler=cmd_league_show)

    update = league_sub.add_parser("update", help="Download newer historical data for a saved league.")
    update.add_argument("league_id")
    update.set_defaults(handler=cmd_league_update)

    delete = league_sub.add_parser("delete", help="Delete a saved league and its data.")
    delete.add_argument("league_id")
    delete.add_argument("--yes", action="store_true", help="Confirm deletion.")
    delete.set_defaults(handler=cmd_league_delete)


def _build_data_parser(subparsers):
    data = subparsers.add_parser("data", help="Inspect, search and export league datasets.")
    data_sub = data.add_subparsers(dest="data_command", required=True)

    show = data_sub.add_parser("show", help="Show league data as a terminal table.")
    show.add_argument("league_id")
    show.add_argument("--rows", type=int, default=25)
    show.add_argument("--columns", help="Comma-separated columns.")
    show.add_argument("--hide-missing", action="store_true", help="Drop rows with missing values.")
    show.set_defaults(handler=cmd_data_show)

    search = data_sub.add_parser("search", help="Search rows by keyword or exact value.")
    search.add_argument("league_id")
    search.add_argument("query")
    search.add_argument("--column", help="Restrict search to one column.")
    search.add_argument("--exact", action="store_true", help="Match exact cell values.")
    search.add_argument("--limit", type=int, default=25)
    search.set_defaults(handler=cmd_data_search)

    export = data_sub.add_parser("export", help="Export league data to CSV or Excel.")
    export.add_argument("league_id")
    export.add_argument("--output", required=True)
    export.add_argument("--columns", help="Comma-separated columns.")
    export.add_argument("--hide-missing", action="store_true")
    export.add_argument("--append", action="store_true")
    export.set_defaults(handler=cmd_data_export)


def _build_model_parser(subparsers):
    model = subparsers.add_parser("model", help="Train, evaluate and manage models.")
    model_sub = model.add_subparsers(dest="model_command", required=True)

    list_cmd = model_sub.add_parser("list", help="List stored models for a league.")
    list_cmd.add_argument("league_id")
    list_cmd.set_defaults(handler=cmd_model_list)

    train = model_sub.add_parser("train", help="Train a model for a saved league.")
    train.add_argument("league_id")
    train.add_argument("model_type", choices=sorted(MODEL_SPECS.keys()), help="Model type.")
    train.add_argument("--id", dest="model_id", help="Unique model id. Defaults to '<league>-<model>'.")
    train.add_argument("--target", default="result", help="result or over-under.")
    train.add_argument("--normalizer", default="none", help="none, standard, min-max or max-abs.")
    train.add_argument("--sampler", default="none", help="none, svm-smote, nearmiss or hardness-threshold.")
    train.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--eval-size", type=float, default=20.0, help="Most recent samples percent reserved for evaluation.")
    train.add_argument("--cv", action=argparse.BooleanOptionalAction, default=True, help="Run K-fold cross validation.")
    train.add_argument("--sliding-cv", action=argparse.BooleanOptionalAction, default=True, help="Run sliding CV.")
    train.add_argument("--tune", help="Comma-separated tunable params, 'all' or 'none'.")
    train.add_argument("--trials", type=int, default=25, help="Optuna trials if tuning is enabled.")
    train.add_argument("--optuna-sampler", choices=["tpe", "random", "cmaes", "cma-es"], default="tpe")
    train.add_argument("--optuna-pruner", choices=["none", "median", "successive-halving"], default="none")
    train.add_argument("--objective", choices=["Accuracy", "F1", "Precision", "Recall"], default="Accuracy")
    train.add_argument("--export-metrics", help="Directory where metrics CSV files are exported.")
    add_model_specific_arguments(train)
    train.set_defaults(handler=cmd_model_train)

    evaluate = model_sub.add_parser("evaluate", help="Evaluate a model with odds and percentile filters.")
    evaluate.add_argument("league_id")
    evaluate.add_argument("--model", dest="model_id", required=True)
    evaluate.add_argument("--dataset", choices=["all", "train", "eval"], default="all")
    evaluate.add_argument("--odd-filter", help='None or "ODD:MIN:MAX", for example "1:1.31:1.60".')
    evaluate.add_argument("--p1", type=int, default=0)
    evaluate.add_argument("--px", type=int, default=0)
    evaluate.add_argument("--p2", type=int, default=0)
    evaluate.add_argument("--pu", type=int, default=0)
    evaluate.add_argument("--po", type=int, default=0)
    evaluate.add_argument("--store-filter", action="store_true", help="Store selected filter on model config.")
    evaluate.add_argument("--delete-filter", action="store_true", help="Delete selected stored filter.")
    evaluate.add_argument("--seasonal", action="store_true", help="Show per-season metrics.")
    evaluate.add_argument("--output", help="Export evaluated rows to CSV/XLSX.")
    evaluate.add_argument("--append", action="store_true")
    evaluate.set_defaults(handler=cmd_model_evaluate)

    metrics = model_sub.add_parser("metrics", help="Show stored training/tuning metrics for a model.")
    metrics.add_argument("league_id")
    metrics.add_argument("model_id")
    metrics.add_argument("--export-dir", help="Export stored metric tables to this directory.")
    metrics.set_defaults(handler=cmd_model_metrics)

    delete = model_sub.add_parser("delete", help="Delete a saved model.")
    delete.add_argument("league_id")
    delete.add_argument("model_id")
    delete.add_argument("--yes", action="store_true")
    delete.set_defaults(handler=cmd_model_delete)


def _build_predict_parser(subparsers):
    predict = subparsers.add_parser("predict", help="Predict upcoming fixtures.")
    predict_sub = predict.add_subparsers(dest="predict_command", required=True)

    fixtures = predict_sub.add_parser("fixtures", help="Predict upcoming fixtures from FootyStats or a CSV file.")
    fixtures.add_argument("league_id")
    fixtures.add_argument("--model", dest="model_id", required=True)
    fixtures.add_argument("--date", help="Fixture date YYYY-MM-DD. Required when scraping.")
    fixtures.add_argument("--input", help="CSV/XLSX with Home,Away,1,X,2 columns. Skips scraping.")
    fixtures.add_argument("--output", default=None, help="CSV/XLSX export path.")
    fixtures.add_argument("--filters", default=None, help="Comma-separated stored filters, 'all' or 'none'.")
    fixtures.add_argument("--all", action="store_true", help="Export all predictions instead of selected/filter-passing rows.")
    fixtures.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None, help="Override browser headless mode.")
    fixtures.set_defaults(handler=cmd_predict_fixtures)


def _build_analysis_parser(subparsers):
    analysis = subparsers.add_parser("analysis", help="Run statistical analysis and save plots/tables.")
    analysis_sub = analysis.add_subparsers(dest="analysis_command", required=True)

    def add_common(p):
        p.add_argument("league_id")
        p.add_argument("--season", type=int, help="Season filter. Omit for all seasons.")
        p.add_argument("--colormap", choices=sorted(COLORMAP_OPTIONS), default="Blues")
        p.add_argument("--output", help="Output image/table path.")

    desc = analysis_sub.add_parser("descriptive", help="Descriptive statistics.")
    add_common(desc)
    desc.add_argument("--feature-type", choices=["home", "away"], default="home")
    desc.add_argument("--table-output", help="Optional CSV/XLSX table export.")
    desc.set_defaults(handler=cmd_analysis_descriptive)

    dist = analysis_sub.add_parser("distributions", help="Feature distributions.")
    add_common(dist)
    dist.add_argument("--column", required=True)
    dist.set_defaults(handler=cmd_analysis_distributions)

    variance = analysis_sub.add_parser("variance", help="Feature variance ranking.")
    add_common(variance)
    variance.set_defaults(handler=cmd_analysis_variance)

    corr = analysis_sub.add_parser("correlation", help="Feature correlation heatmap.")
    add_common(corr)
    corr.add_argument("--method", choices=["pearson", "kendall", "spearman"], default="pearson")
    corr.add_argument("--feature-type", choices=["home", "away"], default="home")
    corr.set_defaults(handler=cmd_analysis_correlation)

    boruta = analysis_sub.add_parser("boruta", help="Boruta feature selection ranking.")
    add_common(boruta)
    boruta.add_argument("--target", default="result")
    boruta.set_defaults(handler=cmd_analysis_boruta)

    coeff = analysis_sub.add_parser("coefficients", help="Logistic coefficient feature importance.")
    add_common(coeff)
    coeff.add_argument("--target", default="result")
    coeff.set_defaults(handler=cmd_analysis_coefficients)

    impurity = analysis_sub.add_parser("impurity", help="Gini impurity feature importance.")
    add_common(impurity)
    impurity.add_argument("--target", default="result")
    impurity.set_defaults(handler=cmd_analysis_impurity)

    rules = analysis_sub.add_parser("rules", help="Decision tree rule extraction.")
    add_common(rules)
    rules.add_argument("--target", default="result")
    rules.add_argument("--depth", type=int, default=3, choices=[3, 4, 5, 6, 7])
    rules.set_defaults(handler=cmd_analysis_rules)


def _build_explain_parser(subparsers):
    explain = subparsers.add_parser("explain", help="Generate interpretability plots for trained models.")
    explain_sub = explain.add_subparsers(dest="explain_command", required=True)

    def add_base(p):
        p.add_argument("league_id")
        p.add_argument("model_id")
        p.add_argument("--output", required=True, help="Output PNG path.")

    boundary = explain_sub.add_parser("boundary", help="Decision boundary plot for two features.")
    add_base(boundary)
    boundary.add_argument("--features", required=True, help="Two comma-separated features.")
    boundary.set_defaults(handler=cmd_explain_boundary)

    pdp = explain_sub.add_parser("pdp", help="Partial dependence plot.")
    add_base(pdp)
    pdp.add_argument("--feature", required=True)
    pdp.add_argument("--target", required=True, help="H/D/A or U/O depending on the model target.")
    pdp.set_defaults(handler=cmd_explain_pdp)

    waterfall = explain_sub.add_parser("waterfall", help="SHAP waterfall plot for a match index.")
    add_base(waterfall)
    waterfall.add_argument("--match-index", type=int, default=0)
    waterfall.add_argument("--target", required=True)
    waterfall.set_defaults(handler=cmd_explain_waterfall)

    shap_cmd = explain_sub.add_parser("shap", help="SHAP bar plot.")
    add_base(shap_cmd)
    shap_cmd.add_argument("--target", required=True)
    shap_cmd.add_argument("--cluster", action=argparse.BooleanOptionalAction, default=True)
    shap_cmd.set_defaults(handler=cmd_explain_shap)

    extra = explain_sub.add_parser("extra", help="Model-specific plots such as coefficients, tree rules or attention.")
    add_base(extra)
    extra.add_argument(
        "--plot",
        required=True,
        choices=["coefficients", "model", "impurity", "tree", "attention"],
        help="Model-specific plot.",
    )
    extra.add_argument("--feature", help="Feature for logistic visualization.")
    extra.add_argument("--features", help="Two comma-separated features for SVM/KNN visualization.")
    extra.add_argument("--match-index", type=int, default=0, help="Match index for KNN visualization.")
    extra.add_argument("--depth", type=int, default=3)
    extra.add_argument("--estimator-id", type=int, default=0, help="Random forest estimator id.")
    extra.set_defaults(handler=cmd_explain_extra)


def _build_config_parser(subparsers):
    config = subparsers.add_parser("config", help="Manage CLI/browser configuration.")
    config_sub = config.add_subparsers(dest="config_command", required=True)

    browser = config_sub.add_parser("browser", help="Show or update browser scraping config.")
    browser.add_argument("action", choices=["show", "set"])
    browser.add_argument("--application", choices=["chrome", "firefox", "edge", "brave"])
    browser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    browser.add_argument("--brave-binary", default=None, help="Ruta del ejecutable de Brave.")
    browser.set_defaults(handler=cmd_config_browser)


def cmd_league_list(args):
    db = LeagueDatabase()
    if args.catalog:
        table = Table(title="Available League Catalog")
        table.add_column("#", justify="right")
        table.add_column("Country")
        table.add_column("League")
        table.add_column("Category")
        table.add_column("Start")
        for idx, league in enumerate(db.leagues, start=1):
            table.add_row(str(idx), league.country, league.name, league.category, str(league.start_year))
        console.print(table)
        return

    ids = db.get_league_ids()
    if not ids:
        print_warning("No saved leagues. Use `league create` to download one.")
        return

    table = Table(title="Saved Leagues")
    table.add_column("ID")
    table.add_column("Country")
    table.add_column("League")
    table.add_column("Start")
    table.add_column("Stats")
    for league_id in ids:
        league = db.index[league_id]
        stats = "all" if league.stats_columns is None else str(len(league.stats_columns))
        table.add_row(league_id, league.country, league.name, str(league.start_year), stats)
    console.print(table)


def cmd_league_create(args):
    db = LeagueDatabase()
    template = _select_catalog_league(db=db, args=args)
    league_id = validate_identifier(args.league_id or prompt_text("League id", default=f"{template.name}-{template.country}-01"), "league id")

    if db.league_exists(league_id):
        raise CLIError(f'League "{league_id}" already exists.')

    current_year_threshold = date.today().year - 4
    start_year = args.start_year
    if start_year is None:
        start_year = prompt_int(
            "Start year",
            default=template.start_year,
            min_value=template.start_year,
            max_value=current_year_threshold,
        )
    if start_year < template.start_year:
        raise CLIError(f"{template.name} starts at {template.start_year}; requested {start_year}.")
    if start_year > current_year_threshold:
        raise CLIError(f"Start year cannot be newer than {current_year_threshold}.")

    if args.history_window < 2 or args.history_window > 5:
        raise CLIError("history-window must be between 2 and 5.")
    if args.goal_margin < 2 or args.goal_margin > 5:
        raise CLIError("goal-margin must be between 2 and 5.")

    stats_columns = _resolve_stats_columns(template, args.stats)
    league = template.clone(
        start_year=start_year,
        league_id=league_id,
        match_history_window=args.history_window,
        goal_diff_margin=args.goal_margin,
        stats_columns=stats_columns,
        odd_1_range=parse_odd_range(args.odd_1),
        odd_x_range=parse_odd_range(args.odd_x),
        odd_2_range=parse_odd_range(args.odd_2),
    )

    render_mapping(
        "League Creation",
        {
            "id": league.league_id,
            "catalog": f"{league.country} / {league.name}",
            "start_year": league.start_year,
            "history_window": league.match_history_window,
            "goal_margin": league.goal_diff_margin,
            "stats": "all" if stats_columns is None else ", ".join(stats_columns),
            "odd_1": league.odd_1_range or "all",
            "odd_x": league.odd_x_range or "all",
            "odd_2": league.odd_2_range or "all",
        },
    )
    confirm_or_abort("Download and create this league?", assume_yes=args.yes)

    with _spinner(f"Downloading and preparing {league_id}..."):
        df = db.create_league(league=league)
    if df is None:
        raise CLIError("League download failed. Check internet access and source availability.")

    print_success(f'League "{league_id}" created with {df.shape[0]} matches and {df.shape[1]} columns.')
    render_dataframe(df, title=f"{league_id} preview", max_rows=10)


def cmd_league_show(args):
    db, league, df = _load_league(args.league_id, update=args.update)
    render_mapping(
        f"League {league.league_id}",
        {
            "country": league.country,
            "name": league.name,
            "category": league.category,
            "start_year": league.start_year,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_rows": int(df.isna().any(axis=1).sum()),
            "models": len(ModelDatabase(league_id=league.league_id).get_model_ids()),
        },
    )
    render_dataframe(df, title="Dataset Preview", max_rows=args.rows)


def cmd_league_update(args):
    db, league, df = _load_league(args.league_id, update=True)
    print_success(f'League "{league.league_id}" updated. Rows: {df.shape[0]}.')
    render_dataframe(df, title="Updated Dataset Preview", max_rows=10)


def cmd_league_delete(args):
    db = LeagueDatabase()
    if not db.league_exists(args.league_id):
        raise CLIError(f'League "{args.league_id}" does not exist.')

    model_db = ModelDatabase(league_id=args.league_id)
    model_count = len(model_db.get_model_ids())
    confirm_or_abort(
        f'Delete league "{args.league_id}" and its local data/models ({model_count} models)?',
        assume_yes=args.yes,
    )
    db.delete_league(args.league_id)
    print_success(f'League "{args.league_id}" deleted.')


def cmd_data_show(args):
    _, _, df = _load_league(args.league_id)
    if args.hide_missing:
        df = df.dropna(ignore_index=True)
    columns = parse_columns(args.columns)
    render_dataframe(df, title=f"{args.league_id} Data", max_rows=args.rows, columns=columns, show_index=True)


def cmd_data_search(args):
    _, _, df = _load_league(args.league_id)
    if args.column:
        if args.column not in df.columns:
            raise CLIError(f'Unknown column "{args.column}".')
        search_df = df[[args.column]]
    else:
        search_df = df

    query = str(args.query)
    if args.exact:
        mask = search_df.astype(str).eq(query).any(axis=1)
    else:
        mask = search_df.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
    result = df[mask]
    render_dataframe(result, title=f'Search "{query}" ({result.shape[0]} matches)', max_rows=args.limit, show_index=True)


def cmd_data_export(args):
    _, _, df = _load_league(args.league_id)
    if args.hide_missing:
        df = df.dropna(ignore_index=True)
    columns = parse_columns(args.columns)
    if columns:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            raise CLIError(f"Unknown columns: {', '.join(missing)}")
        df = df[columns]
    export_dataframe(df, args.output, append=args.append)


def cmd_model_list(args):
    _, _, _ = _load_league(args.league_id)
    model_db = ModelDatabase(league_id=args.league_id)
    model_ids = model_db.get_model_ids()
    if not model_ids:
        print_warning(f'No models saved for league "{args.league_id}".')
        return

    table = Table(title=f"Models for {args.league_id}")
    table.add_column("Model ID")
    table.add_column("Type")
    table.add_column("Target")
    table.add_column("Normalizer")
    table.add_column("Sampler")
    table.add_column("Eval %")
    for model_id in model_ids:
        config = model_db.load_model_config(model_id=model_id)
        train_cfg = config.get("train", {})
        table.add_row(
            model_id,
            config["cls"].__name__,
            target_label(config["target_type"]),
            str(config.get("normalizer")),
            str(config.get("sampler")),
            str(train_cfg.get("eval_samples_size", "")),
        )
    console.print(table)


def cmd_model_train(args):
    model_key = normalize_model_key(args.model_type)
    spec = MODEL_SPECS[model_key]
    db, league, df = _load_league(args.league_id)
    df = df.dropna(ignore_index=True)
    if df.empty:
        raise CLIError("Training dataset has no complete rows after dropping missing values.")

    model_db = ModelDatabase(league_id=league.league_id)
    model_id = validate_identifier(args.model_id or f"{league.league_id}-{model_key}", "model id")
    if model_db.model_exists(model_id):
        raise CLIError(f'Model "{model_id}" already exists for league "{league.league_id}".')
    if args.eval_size < 5 or args.eval_size > 30:
        raise CLIError("eval-size must be between 5 and 30 percent.")
    if args.trials < 1:
        raise CLIError("trials must be greater than or equal to 1.")

    model_config = build_model_params(args=args, league_id=league.league_id, model_id=model_id, model_key=model_key)
    model_config["train"] = {"eval_samples_size": float(args.eval_size), "results": {}}

    render_mapping(
        "Training Plan",
        {
            "league": league.league_id,
            "rows": df.shape[0],
            "model_id": model_id,
            "model": spec.label,
            "target": target_label(model_config["target_type"]),
            "normalizer": model_config.get("normalizer"),
            "sampler": model_config.get("sampler"),
            "calibration": model_config.get("calibrate_probabilities", False),
            "cv": args.cv,
            "sliding_cv": args.sliding_cv,
            "tune": args.tune or "none",
            "optuna_sampler": args.optuna_sampler,
            "optuna_pruner": args.optuna_pruner,
        },
    )

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
        )
        with _spinner(f"Tuning {model_id} for {args.trials} trials..."):
            study = tuner.tune(trials=args.trials, show_progress_bar=True)
        trials_df = _study_to_dataframe(study, args.objective)
        model_config["train"]["results"]["tune"] = trials_df
        model_config.update(**study.best_trial.params)
        optuna_summary["best_score"] = study.best_value
        optuna_summary["best_params"] = study.best_trial.params
        render_dataframe(trials_df, "Hyperparameter Tuning Results", max_rows=10)

    if args.cv:
        model = spec.model_cls(**model_config)
        with _spinner("Running K-fold cross validation..."):
            cv_df = trainer.cross_validation(model=model, df=df)
        cv_df["Model"] = model_id
        cv_df["Model Type"] = model.__class__
        model_config["train"]["results"]["cv"] = cv_df
        render_dataframe(cv_df, "Cross Validation Results", max_rows=12)

    if args.sliding_cv:
        model = spec.model_cls(**model_config)
        with _spinner("Running sliding cross validation..."):
            sliding_df = trainer.sliding_cross_validation(model=model, df=df, test_ratio=float(args.eval_size))
        sliding_df["Model"] = model_id
        sliding_df["Model Type"] = model.__class__
        model_config["train"]["results"]["sliding-cv"] = sliding_df
        render_dataframe(sliding_df, "Sliding Cross Validation Results", max_rows=12)

    train_df, eval_df = train_test_split(df=df, test_size=float(args.eval_size))
    model = spec.model_cls(**model_config)
    with _spinner("Fitting final model..."):
        model, fit_df = trainer.train(model=model, train_df=train_df, eval_df=eval_df, check_nan=True)
    fit_df["Model"] = model_id
    fit_df["Model Type"] = model.__class__
    model_config["cls"] = model.__class__
    model_config["train"]["results"]["fit"] = fit_df

    if isinstance(model, NeuralNetwork):
        model_config.update({"input_size": model.input_size, "num_classes": model.num_classes})

    model_config["train"]["optuna"] = optuna_summary
    model_db.save_model(model=model, model_config=model_config)
    render_dataframe(fit_df, "Training Results", max_rows=10)

    if args.export_metrics:
        _export_model_result_tables(model_config, args.export_metrics)
    print_success(f'Model "{model_id}" trained and saved.')


def cmd_model_evaluate(args):
    _, _, df = _load_league(args.league_id)
    df = df.dropna(ignore_index=True)
    model_db = ModelDatabase(league_id=args.league_id)
    model, config = _load_model(model_db, args.model_id)
    target_type = config["target_type"]
    dataset_key = args.dataset.capitalize()
    odd_range = parse_eval_odd_range(args.odd_filter)

    if args.delete_filter:
        _delete_stored_filter(model_db, config, odd_range)
        print_success(f'Deleted stored filter "{odd_range}" from model "{args.model_id}".')
        return

    y_prob = model.predict_proba(df=df)
    y_pred = y_prob.argmax(axis=1)
    y_prob = y_prob.round(2)
    y_true = construct_targets(df=df, target_type=target_type)

    dataset_masks = _dataset_masks(df=df, config=config)
    dataset_mask = dataset_masks[dataset_key]
    prob_percentiles = _compute_probability_percentiles(
        y_prob=y_prob[dataset_mask],
        target_type=target_type,
        p1=args.p1,
        px=args.px,
        p2=args.p2,
        pu=args.pu,
        po=args.po,
    )
    filter_mask = _evaluation_mask(df=df, y_prob=y_prob, target_type=target_type, dataset_mask=dataset_mask, odd_range=odd_range, prob_percentiles=prob_percentiles)

    metrics = model.compute_metrics(y_true=y_true[filter_mask], y_pred=y_pred[filter_mask])
    correct = int((y_true[filter_mask] == y_pred[filter_mask]).sum())
    total = int(filter_mask.sum())
    profit_balance = _profit_balance(df=df, y_pred=y_pred[filter_mask], filter_mask=filter_mask, target_type=target_type)
    metrics["Correct"] = correct
    metrics["Total"] = total
    metrics["Prof. Balance"] = profit_balance
    render_dataframe(metrics, "Evaluation Metrics", max_rows=5)

    output_df = _prediction_output_dataframe(df=df, target_type=target_type, y_pred=y_pred, y_prob=y_prob)
    output_df = output_df[filter_mask].reset_index(drop=True)
    render_dataframe(output_df, "Filtered Evaluation Rows", max_rows=25)

    if args.seasonal:
        seasonal_df = _seasonal_metrics(df=df, target_type=target_type, y_true=y_true, y_pred=y_pred, filter_mask=filter_mask, model=model)
        render_dataframe(seasonal_df, "Seasonal Metrics", max_rows=50)

    if args.store_filter:
        _store_filter(model_db=model_db, model_config=config, odd_range=odd_range, prob_percentiles=prob_percentiles)
        print_success(f'Stored filter "{odd_range}" for model "{args.model_id}".')

    if args.output:
        export_dataframe(output_df, args.output, append=args.append)


def cmd_model_metrics(args):
    _, _, _ = _load_league(args.league_id)
    model_db = ModelDatabase(league_id=args.league_id)
    config = model_db.load_model_config(args.model_id)
    if config is None:
        raise CLIError(f'Model "{args.model_id}" does not exist.')
    results = config.get("train", {}).get("results", {})
    if not results:
        print_warning(f'Model "{args.model_id}" has no stored metrics.')
        return

    for name, df in results.items():
        render_dataframe(df, f"{args.model_id} / {name}", max_rows=15)
    if args.export_dir:
        _export_model_result_tables(config, args.export_dir)


def cmd_model_delete(args):
    _, _, _ = _load_league(args.league_id)
    model_db = ModelDatabase(league_id=args.league_id)
    if not model_db.model_exists(args.model_id):
        raise CLIError(f'Model "{args.model_id}" does not exist.')
    confirm_or_abort(f'Delete model "{args.model_id}" from league "{args.league_id}"?', assume_yes=args.yes)
    model_db.delete_model(args.model_id)
    print_success(f'Model "{args.model_id}" deleted.')


def cmd_predict_fixtures(args):
    _, league, df = _load_league(args.league_id)
    model_db = ModelDatabase(league_id=league.league_id)
    model, config = _load_model(model_db, args.model_id)

    if args.input:
        fixture_df = _read_fixture_file(args.input)
    else:
        if not args.date:
            raise CLIError("--date is required when --input is not provided.")
        fixture_df = _scrape_fixtures(league=league, date_text=args.date, headless=args.headless)

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
        requested_filters=args.filters,
        base_mask=odd_mask,
    )
    output_df["Selected"] = selected_mask
    render_dataframe(output_df, "Fixture Predictions", max_rows=50)

    export_df = output_df if args.all else output_df[selected_mask].reset_index(drop=True)
    if args.output:
        export_dataframe(export_df, args.output, append=Path(args.output).exists())
    else:
        default = f"outputs/{league.league_id}-fixtures.csv"
        export_dataframe(export_df, default, append=Path(default).exists())


def cmd_analysis_descriptive(args):
    _, _, df = _load_league(args.league_id)
    analyzer = DescriptiveAnalyzer(df=df.dropna())
    colormap = COLORMAP_OPTIONS[args.colormap]
    output = ensure_output_path(args.output, f"{args.league_id}-descriptive.png", ".png")
    ax = analyzer.generate_plot(season=args.season, colormap=colormap, feature_type=args.feature_type)
    save_figure(ax, str(output))
    if args.table_output:
        table_df = _descriptive_table(df=df.dropna(), feature_type=args.feature_type)
        export_dataframe(table_df.reset_index(names="metric"), args.table_output)


def cmd_analysis_distributions(args):
    _, _, df = _load_league(args.league_id)
    analyzer = DistributionAnalyzer(df=df.dropna())
    if args.column not in analyzer.all_features:
        raise CLIError(f'Unknown distribution column "{args.column}".')
    output = ensure_output_path(args.output, f"{args.league_id}-distribution-{args.column}.png", ".png")
    ax = analyzer.generate_plot(season=args.season, colormap=COLORMAP_OPTIONS[args.colormap], column=args.column)
    save_figure(ax, str(output))


def cmd_analysis_variance(args):
    _, _, df = _load_league(args.league_id)
    analyzer = VarianceAnalyzer(df=df.dropna())
    output = ensure_output_path(args.output, f"{args.league_id}-variance.png", ".png")
    ax = analyzer.generate_plot(season=args.season, colormap=COLORMAP_OPTIONS[args.colormap])
    save_figure(ax, str(output))


def cmd_analysis_correlation(args):
    _, _, df = _load_league(args.league_id)
    analyzer = CorrelationAnalyzer(df=df.dropna())
    output = ensure_output_path(args.output, f"{args.league_id}-correlation.png", ".png")
    ax = analyzer.generate_plot(
        season=args.season,
        colormap=COLORMAP_OPTIONS[args.colormap],
        method=args.method,
        feature_type=args.feature_type,
    )
    save_figure(ax, str(output))


def cmd_analysis_boruta(args):
    _, _, df = _load_league(args.league_id)
    analyzer = BorutaAnalyzer(df=df.dropna())
    output = ensure_output_path(args.output, f"{args.league_id}-boruta.png", ".png")
    ax = analyzer.generate_plot(
        season=args.season,
        colormap=COLORMAP_OPTIONS[args.colormap],
        target_type=parse_target(args.target),
    )
    save_figure(ax, str(output))


def cmd_analysis_coefficients(args):
    _, _, df = _load_league(args.league_id)
    analyzer = CoefficientAnalyzer(df=df.dropna())
    output = ensure_output_path(args.output, f"{args.league_id}-coefficients.png", ".png")
    ax = analyzer.generate_plot(
        season=args.season,
        colormap=COLORMAP_OPTIONS[args.colormap],
        target_type=parse_target(args.target),
    )
    save_figure(ax, str(output))


def cmd_analysis_impurity(args):
    _, _, df = _load_league(args.league_id)
    analyzer = GiniImpurityAnalyzer(df=df.dropna())
    output = ensure_output_path(args.output, f"{args.league_id}-impurity.png", ".png")
    ax = analyzer.generate_plot(
        season=args.season,
        colormap=COLORMAP_OPTIONS[args.colormap],
        target_type=parse_target(args.target),
    )
    save_figure(ax, str(output))


def cmd_analysis_rules(args):
    _, _, df = _load_league(args.league_id)
    analyzer = RuleExtractorAnalyzer(df=df.dropna())
    output = ensure_output_path(args.output, f"{args.league_id}-rules.png", ".png")
    ax = analyzer.generate_plot(season=args.season, target_type=parse_target(args.target), max_depth=args.depth)
    save_figure(ax, str(output))


def cmd_explain_boundary(args):
    explainer, _ = _load_explainer(args.league_id, args.model_id, compute_shap=False)
    features = _parse_feature_pair(args.features)
    ax = explainer.boundary_plot(features)
    save_figure(ax, args.output)


def cmd_explain_pdp(args):
    explainer, model = _load_explainer(args.league_id, args.model_id, compute_shap=False)
    _validate_target_label(model.target_type, args.target)
    ax = explainer.partial_dependence_plot(feature=args.feature, target=args.target)
    save_figure(ax, args.output)


def cmd_explain_waterfall(args):
    explainer, model = _load_explainer(args.league_id, args.model_id, compute_shap=True)
    _validate_target_label(model.target_type, args.target)
    ax = explainer.instance_waterfall_plot(match_index=args.match_index, target=args.target)
    save_figure(ax, args.output)


def cmd_explain_shap(args):
    explainer, model = _load_explainer(args.league_id, args.model_id, compute_shap=True)
    _validate_target_label(model.target_type, args.target)
    ax = explainer.shap_bar_plot(target=args.target, clustering=args.cluster)
    if ax is None:
        raise CLIError("This explainer does not provide SHAP values for the selected model.")
    save_figure(ax, args.output)


def cmd_explain_extra(args):
    explainer, model = _load_explainer(args.league_id, args.model_id, compute_shap=False)
    plot = args.plot
    ax = None

    if plot == "coefficients":
        if not isinstance(explainer, (LogisticRegressionExplainer, SVMExplainer)):
            raise CLIError("Coefficient plot is available for Logistic Regression and linear SVM models.")
        ax = explainer.coefficients_bar_plot()
    elif plot == "model":
        if isinstance(explainer, LogisticRegressionExplainer):
            if not args.feature:
                raise CLIError("--feature is required for logistic model visualization.")
            ax = explainer.visualize_model(feature=args.feature)
        elif isinstance(explainer, DiscriminantAnalysisExplainer):
            ax = explainer.visualize_model()
        elif isinstance(explainer, SVMExplainer):
            ax = explainer.visualize_model(features=_parse_feature_pair(args.features))
        elif isinstance(explainer, KNNExplainer):
            ax = explainer.visualize_model(features=_parse_feature_pair(args.features), match_index=args.match_index)
        else:
            raise CLIError("Model visualization is not available for this model type.")
    elif plot == "impurity":
        if not isinstance(explainer, (DecisionTreeExplainer, RandomForestExplainer, ExtremeBoostingExplainer)):
            raise CLIError("Impurity plot is available for tree-based models.")
        ax = explainer.feature_impurity_bar_plot()
    elif plot == "tree":
        if isinstance(explainer, RandomForestExplainer):
            ax = explainer.plot_tree_rules(max_depth=args.depth, estimator_id=args.estimator_id)
        elif isinstance(explainer, DecisionTreeExplainer):
            ax = explainer.plot_tree_rules(max_depth=args.depth)
        else:
            raise CLIError("Tree rule visualization is available for Decision Tree and Random Forest.")
    elif plot == "attention":
        if not isinstance(explainer, NeuralNetworkExplainer):
            raise CLIError("Attention plot is available only for DNN models with VSN support.")
        ax = explainer.plot_attention_scores()
        if ax is None:
            raise CLIError("This DNN model does not include attention/VSN scores.")

    save_figure(ax, args.output)


def cmd_config_browser(args):
    path = Path("storage/network/browser.json")
    with open(path, "r") as file:
        data = json.load(file)
    data.setdefault("application", "chrome")
    data.setdefault("headless", True)
    data.setdefault("brave_binary", "")

    if args.action == "show":
        render_mapping("Configuracion del navegador", data)
        return

    if args.application:
        data["application"] = args.application
    if args.headless is not None:
        data["headless"] = args.headless
    if args.brave_binary is not None:
        data["brave_binary"] = args.brave_binary.strip()
    if not args.application and args.headless is None and args.brave_binary is None:
        raise CLIError("Nada para actualizar. Usa --application, --headless/--no-headless o --brave-binary.")

    with open(path, "w") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    render_mapping("Configuracion del navegador actualizada", data)


def cmd_root_help(args):
    print_warning("The local web interface is now the primary app. Run `python app.py` and open http://127.0.0.1:5050.")
    args.parser.print_help()


def cmd_resources(args):
    table = Table(title="Resources")
    table.add_column("Topic")
    table.add_column("URL", overflow="fold")
    for topic, url in HELP_LINKS.items():
        table.add_row(topic, url)
    console.print(table)


def _load_league(league_id: str, update: bool = False) -> Tuple[LeagueDatabase, League, pd.DataFrame]:
    db = LeagueDatabase()
    if not db.league_exists(league_id):
        raise CLIError(f'League "{league_id}" does not exist. Use `league list`.')

    with _spinner(f"{'Updating' if update else 'Loading'} league {league_id}..."):
        df = db.update_league(league_id) if update else db.load_league(league_id)
    if df is None:
        raise CLIError(f'Could not load league "{league_id}".')
    return db, db.index[league_id], df.reset_index(drop=True)


def _load_model(model_db: ModelDatabase, model_id: str):
    if not model_db.model_exists(model_id):
        raise CLIError(f'Model "{model_id}" does not exist. Use `model list`.')
    with _spinner(f"Loading model {model_id}..."):
        model, config = model_db.load_model(model_id=model_id)
    if model is None or config is None:
        raise CLIError(f'Model "{model_id}" could not be loaded.')
    return model, config


def _select_catalog_league(db: LeagueDatabase, args) -> League:
    if args.league_index is not None:
        if args.league_index < 1 or args.league_index > len(db.leagues):
            raise CLIError(f"--league-index must be between 1 and {len(db.leagues)}.")
        return db.leagues[args.league_index - 1]

    if args.template:
        raw = args.template.strip()
        if ":" in raw:
            country, name = [part.strip() for part in raw.split(":", maxsplit=1)]
        else:
            parts = raw.rsplit("-", maxsplit=1)
            country, name = (None, raw) if len(parts) == 1 else (parts[0], parts[1])
        matches = [
            league for league in db.leagues
            if (country is None or league.country.lower() == country.lower())
            and league.name.lower() == name.lower()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return prompt_choice("Multiple matching leagues", matches, label_fn=lambda l: f"{l.country} / {l.name}")
        raise CLIError(f'No catalog league matches "{args.template}".')

    if args.country or args.name:
        matches = [
            league for league in db.leagues
            if (not args.country or league.country.lower() == args.country.lower())
            and (not args.name or league.name.lower() == args.name.lower())
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return prompt_choice("Matching Leagues", matches, label_fn=lambda l: f"{l.country} / {l.name}")
        raise CLIError("No catalog league matched --country/--name.")

    return prompt_choice("Available Leagues", db.leagues, label_fn=lambda l: f"{l.country} / {l.name} ({l.category})")


def _resolve_stats_columns(league: League, stats_arg: str) -> Optional[List[str]]:
    basic_stats = StatisticsEngine.get_basic_stat_columns()
    extended_stats = StatisticsEngine.get_extended_stat_columns()
    all_stats = basic_stats + extended_stats
    mandatory = {"Date", "Season", "Home", "Away", "HG", "AG", "Result", "1", "X", "2"}
    main_valid = set(MainLeagueDownloader().expected_columns + all_stats).difference(mandatory)
    extra_valid = set(ExtraLeagueDownloader().expected_columns + basic_stats).difference(mandatory)
    valid_stats = sorted((main_valid if league.category == "main" else extra_valid).intersection(all_stats))

    key = stats_arg.strip().lower()
    if key == "all":
        return valid_stats
    if key == "basic":
        return sorted(set(basic_stats).intersection(valid_stats))
    if key == "extended":
        return sorted(set(extended_stats).intersection(valid_stats))
    if key == "none":
        return []

    requested = [col.strip() for col in stats_arg.split(",") if col.strip()]
    invalid = [col for col in requested if col not in valid_stats]
    if invalid:
        raise CLIError(
            f"Invalid stats for {league.country}/{league.name}: {', '.join(invalid)}. "
            f"Valid stats: {', '.join(valid_stats)}"
        )
    return requested


def _study_to_dataframe(study, metric: str) -> pd.DataFrame:
    trials_df = study.trials_dataframe().drop(columns=["datetime_start", "datetime_complete"], errors="ignore")
    if "duration" in trials_df.columns:
        trials_df["duration"] = trials_df["duration"].dt.total_seconds() / 60
    trials_df = trials_df.rename(columns={
        "number": "Trial",
        "value": metric,
        "duration": "Duration(m)",
        **{col: col.split("_", 1)[1] for col in trials_df.columns if col.startswith("params_")},
    })
    return trials_df.sort_values(by=metric, ascending=False).round(3)


def _dataset_masks(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    eval_samples_size = config.get("train", {}).get("eval_samples_size", 20.0)
    num_eval = int(math.floor(df.shape[0] * eval_samples_size / 100))
    num_train = df.shape[0] - num_eval
    return {
        "All": np.array([True] * df.shape[0], dtype=bool),
        "Train": np.array([False] * num_eval + [True] * num_train, dtype=bool),
        "Eval": np.array([True] * num_eval + [False] * num_train, dtype=bool),
    }


def _compute_probability_percentiles(y_prob: np.ndarray, target_type: TargetType, p1: int, px: int, p2: int, pu: int, po: int) -> Dict[str, Tuple[int, float]]:
    for value in [p1, px, p2, pu, po]:
        if value < 0 or value > 100:
            raise CLIError("Percentile values must be between 0 and 100.")

    if target_type == TargetType.RESULT:
        quantiles = np.quantile(y_prob, [p1 / 100, px / 100, p2 / 100])
        return {"1": (p1, quantiles[0]), "X": (px, quantiles[1]), "2": (p2, quantiles[2]), "U": (0, 0.0), "O": (0, 0.0)}

    quantiles = np.quantile(y_prob, [pu / 100, po / 100])
    return {"1": (0, 0.0), "X": (0, 0.0), "2": (0, 0.0), "U": (pu, quantiles[0]), "O": (po, quantiles[1])}


def _evaluation_mask(
        df: pd.DataFrame,
        y_prob: np.ndarray,
        target_type: TargetType,
        dataset_mask: np.ndarray,
        odd_range,
        prob_percentiles: Dict[str, Tuple[int, float]],
) -> np.ndarray:
    if odd_range != "None":
        odd, low, high = odd_range
        mask = dataset_mask & ((low <= df[odd]) & (df[odd] <= high)).to_numpy()
    else:
        mask = dataset_mask.copy()

    if target_type == TargetType.RESULT:
        thresholds = np.float32([prob_percentiles["1"][1], prob_percentiles["X"][1], prob_percentiles["2"][1]])
    else:
        thresholds = np.float32([prob_percentiles["U"][1], prob_percentiles["O"][1]])
    percentile_mask = np.all(y_prob >= thresholds, axis=1)
    return mask & percentile_mask


def _prediction_output_dataframe(df: pd.DataFrame, target_type: TargetType, y_pred: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    base_columns = [col for col in ["Date", "Season", "Week", "Home", "Away", "1", "X", "2", "Result", "Result-U/O"] if col in df.columns]
    output_df = df[base_columns].copy()
    return _append_prediction_columns(output_df, target_type=target_type, y_pred=y_pred, y_prob=y_prob)


def _append_prediction_columns(df: pd.DataFrame, target_type: TargetType, y_pred: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    if target_type == TargetType.RESULT:
        df["Predicted"] = RESULT_LABELS.take(y_pred)
        df["Prob(1)"] = y_prob[:, 0]
        df["Prob(X)"] = y_prob[:, 1]
        df["Prob(2)"] = y_prob[:, 2]
    elif target_type == TargetType.OVER_UNDER:
        df["Predicted"] = OVER_UNDER_LABELS.take(y_pred)
        df["Prob(U)"] = y_prob[:, 0]
        df["Prob(O)"] = y_prob[:, 1]
    else:
        raise CLIError(f"Unsupported target type: {target_type}")
    return df


def _profit_balance(df: pd.DataFrame, y_pred: np.ndarray, filter_mask: np.ndarray, target_type: TargetType) -> float:
    if target_type != TargetType.RESULT or y_pred.shape[0] == 0:
        return 0.0
    odds_df = df.loc[filter_mask, ["1", "X", "2"]]
    odds = odds_df.values[np.arange(y_pred.shape[0]), y_pred]
    return compute_profit_balance(odds=odds)


def _seasonal_metrics(df: pd.DataFrame, target_type: TargetType, y_true: np.ndarray, y_pred: np.ndarray, filter_mask: np.ndarray, model) -> pd.DataFrame:
    filtered_df = pd.DataFrame({
        "y_true": y_true[filter_mask],
        "y_pred": y_pred[filter_mask],
        "Season": df.loc[filter_mask, "Season"],
    }, index=df.index[filter_mask])
    rows = []
    for season, season_df in filtered_df.groupby("Season"):
        metrics = model.compute_metrics(y_true=season_df["y_true"].to_numpy(), y_pred=season_df["y_pred"].to_numpy())
        metrics["Season"] = season
        metrics["Correct"] = (season_df["y_true"] == season_df["y_pred"]).sum()
        metrics["Total"] = season_df.shape[0]
        metrics["Prof. Balance"] = _profit_balance(
            df=df,
            y_pred=season_df["y_pred"].to_numpy(),
            filter_mask=season_df.index,
            target_type=target_type,
        )
        rows.append(metrics)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _store_filter(model_db: ModelDatabase, model_config: Dict[str, Any], odd_range, prob_percentiles: Dict[str, Tuple[int, float]]):
    if "eval" not in model_config:
        model_config["eval"] = {}
    if "percentiles" not in model_config["eval"]:
        model_config["eval"]["percentiles"] = {}
    model_config["eval"]["percentiles"][odd_range] = prob_percentiles
    model_db.update_model_config(model_config=model_config)


def _delete_stored_filter(model_db: ModelDatabase, model_config: Dict[str, Any], odd_range):
    percentiles = model_config.get("eval", {}).get("percentiles")
    if not percentiles or odd_range not in percentiles:
        raise CLIError(f'Filter "{odd_range}" is not stored on this model.')
    del percentiles[odd_range]
    model_db.update_model_config(model_config=model_config)


def _export_model_result_tables(model_config: Dict[str, Any], output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, result_df in model_config.get("train", {}).get("results", {}).items():
        export_dataframe(result_df, str(out / f"{model_config['model_id']}-{name}.csv"))


def _valid_odd(value: float) -> float:
    value = float(value)
    if value <= 1.0:
        raise CLIError(f"Odds must be greater than 1.00, got {value}.")
    return value


def _read_fixture_file(path: str) -> pd.DataFrame:
    input_path = Path(path)
    if not input_path.exists():
        raise CLIError(f"Fixture input file does not exist: {path}")
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(input_path)
    else:
        raise CLIError("Fixture input must be .csv or .xlsx.")
    load_required_columns(df, ["Home", "Away", "1", "X", "2"], "Fixture input")
    return df[["Home", "Away", "1", "X", "2"]].copy()


def _scrape_fixtures(league: League, date_text: str, headless: Optional[bool]) -> pd.DataFrame:
    try:
        selected_date = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise CLIError("--date must use YYYY-MM-DD format.") from exc
    footystats_date = selected_date.strftime("%b %d").replace(" 0", " ")

    scraper = FootyStatsScraper(headless=headless)
    try:
        with _spinner(f"Scraping fixtures for {footystats_date}..."):
            loaded = scraper.load_page(league.fixture)
            if not loaded:
                raise CLIError("Could not load FootyStats page. Check internet, browser driver and fixture URL.")
            parsed = scraper.parse_fixture_table(date_str=footystats_date)
    finally:
        scraper.quit()

    if parsed is None or parsed.empty:
        raise CLIError(f"No fixtures found for {date_text}.")
    return match_fixture_teams(parsed_teams_df=parsed, league_df=_load_league(league.league_id)[2])


def _validate_fixture_rows(fixture_df: pd.DataFrame, league_df: pd.DataFrame) -> pd.DataFrame:
    fixture_df = fixture_df.dropna(subset=["Home", "Away", "1", "X", "2"]).copy()
    if fixture_df.empty:
        raise CLIError("No fixture rows found.")
    home_teams = set(league_df["Home"].dropna().unique().tolist())
    away_teams = set(league_df["Away"].dropna().unique().tolist())
    valid_home = fixture_df["Home"].isin(home_teams)
    valid_away = fixture_df["Away"].isin(away_teams)
    dropped = int((~(valid_home & valid_away)).sum())
    if dropped:
        print_warning(f"Dropped {dropped} fixtures because teams were not found in historical data.")
    fixture_df = fixture_df[valid_home & valid_away].reset_index(drop=True)
    if fixture_df.empty:
        raise CLIError("All fixtures were dropped because teams could not be matched.")

    for column in ["1", "X", "2"]:
        fixture_df[column] = pd.to_numeric(fixture_df[column], errors="coerce")
    fixture_df = fixture_df.dropna(subset=["1", "X", "2"]).reset_index(drop=True)
    if (fixture_df[["1", "X", "2"]] <= 1.0).any().any():
        raise CLIError("Fixture odds must be greater than 1.00.")
    return fixture_df


def _league_odd_mask(league: League, fixture_df: pd.DataFrame) -> np.ndarray:
    mask = np.array([True] * fixture_df.shape[0], dtype=bool)
    for odd, odd_range in [("1", league.odd_1_range), ("X", league.odd_x_range), ("2", league.odd_2_range)]:
        if odd_range is None:
            continue
        low, high = odd_range
        mask = mask & ((fixture_df[odd] >= low) & (fixture_df[odd] <= high)).to_numpy()
    return mask


def _fixture_selected_mask(
        output_df: pd.DataFrame,
        y_prob: np.ndarray,
        target_type: TargetType,
        model_config: Dict[str, Any],
        requested_filters: Optional[str],
        base_mask: np.ndarray,
) -> np.ndarray:
    if requested_filters is None or requested_filters.strip().lower() in {"", "none"}:
        return base_mask

    percentiles = model_config.get("eval", {}).get("percentiles", {})
    if not percentiles:
        print_warning("No stored model filters found. Using only league odd filters.")
        return base_mask

    if requested_filters.strip().lower() == "all":
        selected_keys = list(percentiles.keys())
    else:
        selected_keys = []
        for raw in requested_filters.split(","):
            selected_keys.append(parse_eval_odd_range(raw.strip()))

    combined = np.zeros(shape=(output_df.shape[0],), dtype=bool)
    for key in selected_keys:
        if key not in percentiles:
            print_warning(f'Skipping unknown stored filter "{key}".')
            continue
        if key == "None":
            odd_mask = np.ones(shape=(output_df.shape[0],), dtype=bool)
        else:
            odd, low, high = key
            odd_mask = ((output_df[odd].astype(float) >= low) & (output_df[odd].astype(float) <= high)).to_numpy()

        prob_percentiles = percentiles[key]
        if target_type == TargetType.RESULT:
            thresholds = np.float32([prob_percentiles["1"][1], prob_percentiles["X"][1], prob_percentiles["2"][1]])
        else:
            thresholds = np.float32([prob_percentiles["U"][1], prob_percentiles["O"][1]])
        combined = combined | (odd_mask & np.all(y_prob >= thresholds, axis=1))
    return base_mask & combined


def _descriptive_table(df: pd.DataFrame, feature_type: str) -> pd.DataFrame:
    from src.preprocessing.dataset import DatasetPreprocessor

    trainable_features = df.columns.drop(DatasetPreprocessor().non_trainable_columns, errors="ignore").tolist()
    input_df = df[trainable_features + ["Result", "Result-U/O"]]
    if feature_type == "home":
        input_df = input_df[["1", "X", "2"] + [col for col in input_df.columns if col[0] == "H"]]
    elif feature_type == "away":
        input_df = input_df[["1", "X", "2"] + [col for col in input_df.columns if col[0] == "A"]]
    else:
        raise CLIError(f'Unknown feature type "{feature_type}".')
    return input_df.describe().round(decimals=2)


def _load_explainer(league_id: str, model_id: str, compute_shap: bool):
    _, _, df = _load_league(league_id)
    df = df.dropna(ignore_index=True)
    model_db = ModelDatabase(league_id=league_id)
    model, _ = _load_model(model_db, model_id)
    explainer_cls = None
    for model_cls, candidate in EXPLAINER_BY_MODEL.items():
        if isinstance(model, model_cls):
            explainer_cls = candidate
            break
    if explainer_cls is None:
        raise CLIError(f"No explainer registered for {model.__class__.__name__}.")
    explainer = explainer_cls(model=model, df=df)
    if compute_shap:
        with _spinner("Computing SHAP values..."):
            explainer.compute_shap_values()
        if explainer.shap_values is None:
            raise CLIError("This model/explainer does not provide SHAP values for this plot.")
    return explainer, model


def _parse_feature_pair(value: Optional[str]) -> List[str]:
    if not value:
        raise CLIError("--features is required and must contain two comma-separated features.")
    features = [feature.strip() for feature in value.split(",") if feature.strip()]
    if len(features) != 2:
        raise CLIError("--features must contain exactly two comma-separated features.")
    return features


def _validate_target_label(target_type: TargetType, target: str):
    allowed = ["H", "D", "A"] if target_type == TargetType.RESULT else ["U", "O"]
    if target not in allowed:
        raise CLIError(f'Invalid target "{target}". Valid targets for {target_label(target_type)}: {", ".join(allowed)}.')


class _spinner:
    def __init__(self, message: str):
        self.message = message
        self.progress = None

    def __enter__(self):
        self.progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True)
        self.progress.start()
        self.progress.add_task(self.message, total=None)

    def __exit__(self, exc_type, exc, tb):
        self.progress.stop()


def main():
    raise SystemExit(run())


if __name__ == "__main__":
    main()
