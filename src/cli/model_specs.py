from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from src.cli.common import CLIError, parse_bool, parse_normalizer, parse_sampler, parse_target, parse_tunable_params
from src.models.classifiers.decisiontree import DecisionTree
from src.models.classifiers.discriminant import DiscriminantAnalysisClassifier
from src.models.classifiers.extremeboosting import XGBoost
from src.models.classifiers.knn import KNN
from src.models.classifiers.logistic import LogisticRegressor
from src.models.classifiers.naivebayes import NaiveBayes
from src.models.classifiers.neuralnets.nn import NeuralNetwork
from src.models.classifiers.randomforest import RandomForest
from src.models.classifiers.svm import SVM


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_cls: Type
    supports_calibration: bool
    defaults: Dict[str, Any] = field(default_factory=dict)


MODEL_SPECS: Dict[str, ModelSpec] = {
    "logistic": ModelSpec(
        key="logistic",
        label="Logistic Regression",
        model_cls=LogisticRegressor,
        supports_calibration=True,
        defaults={"penalty": None},
    ),
    "discriminant": ModelSpec(
        key="discriminant",
        label="Discriminant Analysis (LDA/QDA)",
        model_cls=DiscriminantAnalysisClassifier,
        supports_calibration=False,
        defaults={"oas": True, "decision_boundary": "linear"},
    ),
    "decision-tree": ModelSpec(
        key="decision-tree",
        label="Decision Tree",
        model_cls=DecisionTree,
        supports_calibration=True,
        defaults={
            "criterion": "gini",
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": None,
            "max_depth": 0,
            "class_weight": True,
        },
    ),
    "random-forest": ModelSpec(
        key="random-forest",
        label="Random Forest",
        model_cls=RandomForest,
        supports_calibration=True,
        defaults={
            "n_estimators": 100,
            "criterion": "gini",
            "min_samples_leaf": 1,
            "min_samples_split": 2,
            "max_features": None,
            "max_depth": 0,
            "class_weight": True,
        },
    ),
    "xgboost": ModelSpec(
        key="xgboost",
        label="Extreme Boosting (XGBoost)",
        model_cls=XGBoost,
        supports_calibration=True,
        defaults={
            "n_estimators": 100,
            "max_depth": 6,
            "min_child_weight": 1,
            "learning_rate": 0.3,
            "lambda_regularization": 1.0,
            "alpha_regularization": 0.0,
        },
    ),
    "knn": ModelSpec(
        key="knn",
        label="K-Nearest Neighbors",
        model_cls=KNN,
        supports_calibration=True,
        defaults={"n_neighbors": 15, "weights": "distance", "p": 2},
    ),
    "naive-bayes": ModelSpec(
        key="naive-bayes",
        label="Naive Bayes",
        model_cls=NaiveBayes,
        supports_calibration=True,
        defaults={"algorithm": "gaussian"},
    ),
    "svm": ModelSpec(
        key="svm",
        label="Support Vector Machine",
        model_cls=SVM,
        supports_calibration=True,
        defaults={"kernel": "rbf", "degree": 3, "gamma": 1.0, "class_weight": True},
    ),
    "dnn": ModelSpec(
        key="dnn",
        label="Deep Neural Network",
        model_cls=NeuralNetwork,
        supports_calibration=False,
        defaults={
            "hidden_layers": 2,
            "hidden_units": 256,
            "hidden_activation": "gelu",
            "vsn": False,
            "layer_normalization": True,
            "batch_normalization": False,
            "dropout_rate": 0.1,
            "odd_noise_std": 0.1,
            "class_weight": True,
            "optimizer": "adam",
            "lookahead": True,
            "label_smoothing": 0.1,
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 50,
            "early_stopping_patience": 15,
            "lr_decay_patience": 10,
            "lr_decay_factor": 0.2,
            "verbose": "auto",
        },
    ),
}

MODEL_ALIASES = {
    "lr": "logistic",
    "lda": "discriminant",
    "qda": "discriminant",
    "tree": "decision-tree",
    "decisiontree": "decision-tree",
    "rf": "random-forest",
    "randomforest": "random-forest",
    "forest": "random-forest",
    "xgb": "xgboost",
    "extreme-boosting": "xgboost",
    "nb": "naive-bayes",
    "naivebayes": "naive-bayes",
    "neural-network": "dnn",
    "nn": "dnn",
}


def normalize_model_key(value: str) -> str:
    key = value.strip().lower().replace("_", "-")
    key = MODEL_ALIASES.get(key, key)
    if key not in MODEL_SPECS:
        raise CLIError(f'Invalid model type "{value}". Use one of: {", ".join(MODEL_SPECS)}.')
    return key


def add_model_specific_arguments(parser, model_key: Optional[str] = None):
    parser.add_argument("--penalty", choices=["none", "l1", "l2"], help="Logistic penalty.")
    parser.add_argument("--oas", type=parse_bool, help="Use OAS covariance estimator for discriminant models.")
    parser.add_argument("--decision-boundary", choices=["linear", "quadratic"], help="Discriminant boundary type.")
    parser.add_argument("--criterion", choices=["gini", "entropy", "log_loss"], help="Tree split criterion.")
    parser.add_argument("--min-samples-leaf", type=int, help="Minimum samples per leaf.")
    parser.add_argument("--min-samples-split", type=int, help="Minimum samples required to split a node.")
    parser.add_argument("--max-features", choices=["none", "sqrt", "log2"], help="Tree max features.")
    parser.add_argument("--max-depth", type=int, help="Maximum tree depth. Use 0 for unlimited.")
    parser.add_argument("--class-weight", type=parse_bool, help="Whether to use balanced class weights.")
    parser.add_argument("--n-estimators", type=int, help="Number of estimators.")
    parser.add_argument("--min-child-weight", type=int, help="XGBoost minimum child weight.")
    parser.add_argument("--learning-rate", type=float, help="Learning rate.")
    parser.add_argument("--lambda-regularization", type=float, help="XGBoost lambda regularization.")
    parser.add_argument("--alpha-regularization", type=float, help="XGBoost alpha regularization.")
    parser.add_argument("--n-neighbors", type=int, help="KNN neighbor count.")
    parser.add_argument("--weights", choices=["uniform", "distance"], help="KNN neighbor weighting.")
    parser.add_argument("--p", type=int, choices=[1, 2], help="KNN distance metric: 1 Manhattan, 2 Euclidean.")
    parser.add_argument("--algorithm", choices=["gaussian", "multinomial", "complement"], help="Naive Bayes algorithm.")
    parser.add_argument("--kernel", choices=["linear", "rbf", "poly", "sigmoid"], help="SVM kernel.")
    parser.add_argument("--degree", type=int, help="SVM polynomial degree.")
    parser.add_argument("--gamma", type=float, help="SVM C/gamma value used by the existing model wrapper.")
    parser.add_argument("--hidden-layers", type=int, help="DNN hidden layer count.")
    parser.add_argument("--hidden-units", type=int, help="DNN units per hidden layer.")
    parser.add_argument("--hidden-activation", choices=["tanh", "relu", "gelu"], help="DNN activation.")
    parser.add_argument("--vsn", type=parse_bool, help="DNN variable selection network.")
    parser.add_argument("--layer-normalization", type=parse_bool, help="DNN layer normalization.")
    parser.add_argument("--batch-normalization", type=parse_bool, help="DNN batch normalization.")
    parser.add_argument("--dropout-rate", type=float, help="DNN dropout rate.")
    parser.add_argument("--odd-noise-std", type=float, help="DNN odd noise standard deviation.")
    parser.add_argument("--optimizer", choices=["adam", "adabelief", "adan", "ranger25"], help="DNN optimizer.")
    parser.add_argument("--lookahead", type=parse_bool, help="DNN lookahead optimizer wrapper.")
    parser.add_argument("--label-smoothing", type=float, help="DNN label smoothing.")
    parser.add_argument("--batch-size", type=int, help="DNN batch size.")
    parser.add_argument("--epochs", type=int, help="DNN epochs.")
    parser.add_argument("--early-stopping-patience", type=int, help="DNN early stopping patience.")
    parser.add_argument("--lr-decay-patience", type=int, help="DNN learning-rate decay patience.")
    parser.add_argument("--lr-decay-factor", type=float, help="DNN learning-rate decay factor.")
    parser.add_argument("--verbose", default=None, help="DNN TensorFlow verbose mode.")


def build_model_params(args, league_id: str, model_id: str, model_key: str) -> Dict[str, Any]:
    spec = MODEL_SPECS[model_key]
    params = {
        "league_id": league_id,
        "model_id": model_id,
        "target_type": parse_target(args.target),
        "normalizer": parse_normalizer(args.normalizer),
        "sampler": parse_sampler(args.sampler),
    }
    if spec.supports_calibration:
        params["calibrate_probabilities"] = bool(args.calibrate)

    params.update(spec.defaults)
    overrides = {
        "penalty": _none_if_literal(getattr(args, "penalty", None)),
        "oas": getattr(args, "oas", None),
        "decision_boundary": getattr(args, "decision_boundary", None),
        "criterion": getattr(args, "criterion", None),
        "min_samples_leaf": getattr(args, "min_samples_leaf", None),
        "min_samples_split": getattr(args, "min_samples_split", None),
        "max_features": _none_if_literal(getattr(args, "max_features", None)),
        "max_depth": getattr(args, "max_depth", None),
        "class_weight": getattr(args, "class_weight", None),
        "n_estimators": getattr(args, "n_estimators", None),
        "min_child_weight": getattr(args, "min_child_weight", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "lambda_regularization": getattr(args, "lambda_regularization", None),
        "alpha_regularization": getattr(args, "alpha_regularization", None),
        "n_neighbors": getattr(args, "n_neighbors", None),
        "weights": getattr(args, "weights", None),
        "p": getattr(args, "p", None),
        "algorithm": getattr(args, "algorithm", None),
        "kernel": getattr(args, "kernel", None),
        "degree": getattr(args, "degree", None),
        "gamma": getattr(args, "gamma", None),
        "hidden_layers": getattr(args, "hidden_layers", None),
        "hidden_units": getattr(args, "hidden_units", None),
        "hidden_activation": getattr(args, "hidden_activation", None),
        "vsn": getattr(args, "vsn", None),
        "layer_normalization": getattr(args, "layer_normalization", None),
        "batch_normalization": getattr(args, "batch_normalization", None),
        "dropout_rate": getattr(args, "dropout_rate", None),
        "odd_noise_std": getattr(args, "odd_noise_std", None),
        "optimizer": getattr(args, "optimizer", None),
        "lookahead": getattr(args, "lookahead", None),
        "label_smoothing": getattr(args, "label_smoothing", None),
        "batch_size": getattr(args, "batch_size", None),
        "epochs": getattr(args, "epochs", None),
        "early_stopping_patience": getattr(args, "early_stopping_patience", None),
        "lr_decay_patience": getattr(args, "lr_decay_patience", None),
        "lr_decay_factor": getattr(args, "lr_decay_factor", None),
        "verbose": getattr(args, "verbose", None),
    }
    for key, value in overrides.items():
        if value is not None and key in params:
            params[key] = value

    _validate_model_params(spec=spec, params=params)
    return params


def tunable_params_for_args(args, spec: ModelSpec) -> Dict[str, Any]:
    params = parse_tunable_params(args.tune)
    if not params:
        return {}

    if params == ["all"]:
        candidate_params = _all_suggestable_params(spec)
    else:
        candidate_params = params

    tunables = {}
    for param in candidate_params:
        if param == "calibrate_probabilities" and not spec.supports_calibration:
            raise CLIError(f"{spec.label} does not support probability calibration tuning.")
        try:
            tunables[param] = spec.model_cls.get_suggest_param_values(param=param)
        except ValueError as exc:
            raise CLIError(f'Parameter "{param}" is not tunable for this model.') from exc
    return tunables


def _all_suggestable_params(spec: ModelSpec) -> List[str]:
    candidates = [
        "normalizer", "sampler", "calibrate_probabilities",
        "penalty", "oas", "decision_boundary",
        "criterion", "min_samples_leaf", "min_samples_split", "max_features", "max_depth", "class_weight",
        "n_estimators", "min_child_weight", "learning_rate", "lambda_regularization", "alpha_regularization",
        "n_neighbors", "weights", "p", "algorithm", "kernel", "degree", "gamma",
        "hidden_layers", "hidden_units", "hidden_activation", "vsn", "layer_normalization", "batch_normalization",
        "dropout_rate", "odd_noise_std", "optimizer", "lookahead", "label_smoothing", "batch_size", "epochs",
        "early_stopping_patience", "lr_decay_patience", "lr_decay_factor",
    ]
    valid = []
    for param in candidates:
        if param == "calibrate_probabilities" and not spec.supports_calibration:
            continue
        try:
            spec.model_cls.get_suggest_param_values(param=param)
        except ValueError:
            continue
        valid.append(param)
    return valid


def _none_if_literal(value):
    if isinstance(value, str) and value.lower() == "none":
        return None
    return value


def _validate_model_params(spec: ModelSpec, params: Dict[str, Any]):
    tunable_ranges = {
        key: spec.model_cls.get_suggest_param_values(key)
        for key in spec.defaults
        if _is_suggestable(spec.model_cls, key)
    }
    for key, range_cfg in tunable_ranges.items():
        if key not in params:
            continue
        value = params[key]
        if isinstance(range_cfg, list):
            if value not in range_cfg:
                raise CLIError(f'Invalid value for "{key}": {value}. Valid values: {range_cfg}')
        elif isinstance(range_cfg, dict) and value is not None:
            low, high = range_cfg["low"], range_cfg["high"]
            if value < low or value > high:
                raise CLIError(f'Invalid value for "{key}": {value}. Expected between {low} and {high}.')

    if params.get("calibrate_probabilities") is True and spec.supports_calibration is False:
        raise CLIError(f"{spec.label} does not support probability calibration.")


def _is_suggestable(model_cls: Type, key: str) -> bool:
    try:
        model_cls.get_suggest_param_values(key)
    except ValueError:
        return False
    return True
