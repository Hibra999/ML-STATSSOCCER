from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from src.cli.common import CLIError, parse_bool, parse_normalizer, parse_sampler, parse_target, parse_tunable_params
from src.models.classifiers.boosting import CatBoost, LightGBM, NGBoost
from src.models.classifiers.extremeboosting import XGBoost


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    model_cls: Type
    supports_calibration: bool
    defaults: Dict[str, Any] = field(default_factory=dict)


MODEL_SPECS: Dict[str, ModelSpec] = {
    "ngboost": ModelSpec(
        key="ngboost",
        label="NGBoost",
        model_cls=NGBoost,
        supports_calibration=False,
        defaults={
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.02,
            "minibatch_frac": 1.0,
            "natural_gradient": True,
        },
    ),
    "catboost": ModelSpec(
        key="catboost",
        label="CatBoost",
        model_cls=CatBoost,
        supports_calibration=True,
        defaults={
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "device": "auto",
        },
    ),
    "lightgbm": ModelSpec(
        key="lightgbm",
        label="LightGBM",
        model_cls=LightGBM,
        supports_calibration=True,
        defaults={
            "n_estimators": 300,
            "num_leaves": 31,
            "max_depth": -1,
            "learning_rate": 0.05,
            "min_child_samples": 20,
            "lambda_regularization": 0.0,
            "alpha_regularization": 0.0,
            "device": "auto",
        },
    ),
    "xgboost": ModelSpec(
        key="xgboost",
        label="XGBoost",
        model_cls=XGBoost,
        supports_calibration=True,
        defaults={
            "n_estimators": 100,
            "max_depth": 6,
            "min_child_weight": 1,
            "learning_rate": 0.3,
            "lambda_regularization": 1.0,
            "alpha_regularization": 0.0,
            "device": "auto",
        },
    ),
}

MODEL_ALIASES = {
    "ngb": "ngboost",
    "ng-boost": "ngboost",
    "cat": "catboost",
    "cat-boost": "catboost",
    "lgbm": "lightgbm",
    "light-gbm": "lightgbm",
    "xgb": "xgboost",
    "extreme-boosting": "xgboost",
}


def normalize_model_key(value: str) -> str:
    key = value.strip().lower().replace("_", "-")
    key = MODEL_ALIASES.get(key, key)
    if key not in MODEL_SPECS:
        raise CLIError(f'Invalid model type "{value}". Use one of: {", ".join(MODEL_SPECS)}.')
    return key


def add_model_specific_arguments(parser, model_key: Optional[str] = None):
    parser.add_argument("--n-estimators", type=int, help="Number of boosting estimators.")
    parser.add_argument("--max-depth", type=int, help="Maximum base learner depth. LightGBM accepts -1 for unlimited.")
    parser.add_argument("--learning-rate", type=float, help="Boosting learning rate.")
    parser.add_argument("--min-child-weight", type=int, help="XGBoost minimum child weight.")
    parser.add_argument("--lambda-regularization", type=float, help="L2 regularization for XGBoost/LightGBM.")
    parser.add_argument("--alpha-regularization", type=float, help="L1 regularization for XGBoost/LightGBM.")
    parser.add_argument("--num-leaves", type=int, help="LightGBM number of leaves.")
    parser.add_argument("--min-child-samples", type=int, help="LightGBM minimum child samples.")
    parser.add_argument("--minibatch-frac", type=float, help="NGBoost minibatch fraction.")
    parser.add_argument("--natural-gradient", type=parse_bool, help="NGBoost natural gradient.")
    parser.add_argument("--l2-leaf-reg", type=float, help="CatBoost L2 leaf regularization.")
    parser.add_argument("--random-strength", type=float, help="CatBoost random strength.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], help="Boosting device. Use auto to prefer CUDA when nvidia-smi is available.")


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
        "n_estimators": getattr(args, "n_estimators", None),
        "max_depth": getattr(args, "max_depth", None),
        "min_child_weight": getattr(args, "min_child_weight", None),
        "learning_rate": getattr(args, "learning_rate", None),
        "lambda_regularization": getattr(args, "lambda_regularization", None),
        "alpha_regularization": getattr(args, "alpha_regularization", None),
        "num_leaves": getattr(args, "num_leaves", None),
        "min_child_samples": getattr(args, "min_child_samples", None),
        "minibatch_frac": getattr(args, "minibatch_frac", None),
        "natural_gradient": getattr(args, "natural_gradient", None),
        "l2_leaf_reg": getattr(args, "l2_leaf_reg", None),
        "random_strength": getattr(args, "random_strength", None),
        "device": getattr(args, "device", None),
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

    candidate_params = tunable_param_names(spec) if params == ["all"] else params
    tunables = {}
    for param in candidate_params:
        if param == "calibrate_probabilities" and not spec.supports_calibration:
            raise CLIError(f"{spec.label} does not support probability calibration tuning.")
        try:
            tunables[param] = spec.model_cls.get_suggest_param_values(param=param)
        except ValueError as exc:
            raise CLIError(f'Parameter "{param}" is not tunable for this model.') from exc
    return tunables


def tunable_param_names(spec: ModelSpec) -> List[str]:
    candidates = [
        "normalizer",
        "sampler",
        "calibrate_probabilities",
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
        "device",
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
