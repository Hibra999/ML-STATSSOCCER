from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


OUTCOME_KEYS = ("home", "draw", "away")
LOSS_METRICS = {"log_loss", "brier", "score_log_loss"}
HIT_METRICS = {"hit"}


def build_prediction_statistical_audit(
        model_backtests: List[Dict[str, Any]] | None,
        baseline_key: str = "independent_poisson",
        alpha: float = 0.05,
        bootstrap_samples: int = 2000,
        seed: int = 2026,
) -> Dict[str, Any]:
    backtests = [dict(item) for item in (model_backtests or []) if item]
    if not backtests:
        return _empty_audit("Sin backtests de modelos para auditar.")

    rows_by_model = {
        str(model.get("model_key") or ""): _model_market_rows(model)
        for model in backtests
    }
    rows_by_model = {key: rows for key, rows in rows_by_model.items() if key and rows}
    if not rows_by_model:
        return _empty_audit("Los backtests no contienen filas por partido suficientes.")

    baseline = baseline_key if baseline_key in rows_by_model else next(iter(rows_by_model))
    warnings: List[str] = []
    if baseline != baseline_key:
        warnings.append(f"Baseline {baseline_key} no disponible; se uso {baseline}.")

    rng = np.random.default_rng(seed)
    model_metrics = [
        _model_metric_summary(key, rows, bootstrap_samples=bootstrap_samples, rng=rng)
        for key, rows in rows_by_model.items()
    ]
    calibration = [
        _model_calibration_summary(key, rows, bootstrap_samples=bootstrap_samples, rng=rng)
        for key, rows in rows_by_model.items()
    ]
    stability = [
        _temporal_stability_summary(key, rows)
        for key, rows in rows_by_model.items()
    ]
    comparisons = _paired_model_comparisons(
        rows_by_model,
        baseline=baseline,
        alpha=alpha,
        bootstrap_samples=bootstrap_samples,
        rng=rng,
    )
    recommendations = _audit_recommendations(
        model_metrics=model_metrics,
        calibration=calibration,
        stability=stability,
        comparisons=comparisons,
        baseline=baseline,
    )
    evaluated_matches = max((int(item.get("evaluated_matches") or 0) for item in backtests), default=0)
    if evaluated_matches < 30:
        warnings.append(
            f"Potencia limitada: {evaluated_matches} partidos evaluados. Trata p-values como exploratorios hasta tener >=30."
        )
    return {
        "available": True,
        "baseline_model_key": baseline,
        "alpha": float(alpha),
        "evaluated_models": len(rows_by_model),
        "evaluated_matches": int(evaluated_matches),
        "methods": [
            "Comparaciones pareadas por fixture contra baseline.",
            "Wilcoxon signed-rank para perdidas continuas no normales; t pareado solo diagnostico.",
            "McNemar/binomial exacto para aciertos binarios pareados.",
            "IC bootstrap percentil 95% para medias y diferencias pareadas.",
            "Correccion Holm-Bonferroni por familia de comparaciones.",
            "ECE y reliability bins para calibracion 1X2 y over/under agregado.",
        ],
        "model_metrics": model_metrics,
        "market_comparisons": comparisons,
        "calibration": calibration,
        "temporal_stability": stability,
        "recommendations": recommendations,
        "warnings": _unique_strings(warnings),
    }


def _empty_audit(reason: str) -> Dict[str, Any]:
    return {
        "available": False,
        "reason": reason,
        "methods": [],
        "model_metrics": [],
        "market_comparisons": [],
        "calibration": [],
        "temporal_stability": [],
        "recommendations": [
            "Genera primero un backtest walk-forward con al menos 30 partidos finalizados para poder estimar significancia.",
        ],
        "warnings": [reason],
    }


def _model_market_rows(model: Dict[str, Any]) -> List[Dict[str, Any]]:
    model_key = str(model.get("model_key") or "")
    rows: List[Dict[str, Any]] = []
    for index, sample in enumerate(model.get("matches") or []):
        fixture_id = _fixture_key(sample, index)
        date = str(sample.get("date") or "")
        actual = str(sample.get("actual_pick_key") or "").strip()
        probabilities = _percent_probability_dict(sample.get("probabilities") or {})
        actual_probability = _percent_value(sample.get("actual_probability"))
        confidence = _percent_value(sample.get("confidence"))
        if actual in OUTCOME_KEYS and probabilities:
            rows.append({
                "model_key": model_key,
                "fixture_id": fixture_id,
                "date": date,
                "market": "result_1x2",
                "metric": "log_loss",
                "value": -math.log(max(actual_probability, 1e-12)),
                "lower_is_better": True,
            })
            rows.append({
                "model_key": model_key,
                "fixture_id": fixture_id,
                "date": date,
                "market": "result_1x2",
                "metric": "brier",
                "value": _multiclass_brier(probabilities, actual),
                "lower_is_better": True,
            })
            rows.append({
                "model_key": model_key,
                "fixture_id": fixture_id,
                "date": date,
                "market": "result_1x2",
                "metric": "hit",
                "value": 1.0 if sample.get("pick_hit") else 0.0,
                "lower_is_better": False,
                "confidence": confidence,
            })
        score_probability = _percent_value(sample.get("score_probability"))
        if score_probability > 0:
            rows.append({
                "model_key": model_key,
                "fixture_id": fixture_id,
                "date": date,
                "market": "exact_score",
                "metric": "score_log_loss",
                "value": -math.log(max(score_probability, 1e-12)),
                "lower_is_better": True,
            })
        rows.append({
            "model_key": model_key,
            "fixture_id": fixture_id,
            "date": date,
            "market": "exact_score",
            "metric": "hit",
            "value": 1.0 if sample.get("score_hit") else 0.0,
            "lower_is_better": False,
        })
        rows.append({
            "model_key": model_key,
            "fixture_id": fixture_id,
            "date": date,
            "market": "top3_score",
            "metric": "hit",
            "value": 1.0 if sample.get("top3_score_hit") else 0.0,
            "lower_is_better": False,
        })
        for ou in sample.get("over_under") or []:
            line = str(ou.get("line") or "").strip()
            market = f"over_under_{line}" if line else "over_under"
            rows.extend(_over_under_metric_rows(model_key, fixture_id, date, market, ou))
    return rows


def _over_under_metric_rows(model_key: str, fixture_id: str, date: str, market: str, row: Dict[str, Any]) -> List[Dict[str, Any]]:
    output = []
    for metric in ("log_loss", "brier"):
        if row.get(metric) not in (None, ""):
            output.append({
                "model_key": model_key,
                "fixture_id": fixture_id,
                "date": date,
                "market": market,
                "metric": metric,
                "value": float(row.get(metric)),
                "lower_is_better": True,
            })
    output.append({
        "model_key": model_key,
        "fixture_id": fixture_id,
        "date": date,
        "market": market,
        "metric": "hit",
        "value": 1.0 if row.get("hit") else 0.0,
        "lower_is_better": False,
        "confidence": _percent_value(row.get("confidence")),
    })
    return output


def _fixture_key(sample: Dict[str, Any], index: int) -> str:
    explicit = str(sample.get("fixture_id") or "").strip()
    if explicit:
        return explicit
    return "|".join([
        str(sample.get("date") or index),
        str(sample.get("home") or ""),
        str(sample.get("away") or ""),
    ])


def _percent_probability_dict(payload: Dict[str, Any]) -> Dict[str, float]:
    values = {key: _percent_value(payload.get(key)) for key in OUTCOME_KEYS}
    total = sum(values.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in values.items()}


def _percent_value(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return float(np.clip(number / 100.0 if number > 1.0 else number, 0.0, 1.0))


def _multiclass_brier(probabilities: Dict[str, float], actual: str) -> float:
    return float(sum((float(probabilities.get(key, 0.0)) - (1.0 if key == actual else 0.0)) ** 2 for key in OUTCOME_KEYS))


def _model_metric_summary(
        model_key: str,
        rows: List[Dict[str, Any]],
        bootstrap_samples: int,
        rng: np.random.Generator,
) -> Dict[str, Any]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["market"]), str(row["metric"]))].append(float(row["value"]))
    markets = []
    for (market, metric), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        ci = _bootstrap_ci(arr, bootstrap_samples, rng)
        markets.append({
            "market": market,
            "metric": metric,
            "n": int(arr.size),
            "mean": round(float(np.mean(arr)), 6) if arr.size else None,
            "std": round(float(np.std(arr, ddof=1)), 6) if arr.size > 1 else 0.0,
            "ci95": ci,
        })
    return {"model_key": model_key, "markets": markets}


def _model_calibration_summary(
        model_key: str,
        rows: List[Dict[str, Any]],
        bootstrap_samples: int,
        rng: np.random.Generator,
) -> Dict[str, Any]:
    output = {"model_key": model_key, "markets": []}
    for label, predicate in (
            ("result_1x2", lambda item: item["market"] == "result_1x2" and item["metric"] == "hit"),
            ("over_under_all", lambda item: item["market"].startswith("over_under_") and item["metric"] == "hit"),
    ):
        items = [row for row in rows if predicate(row) and row.get("confidence") not in (None, "")]
        if not items:
            continue
        confidence = np.asarray([float(row.get("confidence") or 0.0) for row in items], dtype=float)
        hits = np.asarray([float(row["value"]) for row in items], dtype=float)
        ece, bins = _ece(confidence, hits)
        output["markets"].append({
            "market": label,
            "n": int(hits.size),
            "accuracy": round(float(np.mean(hits)), 6),
            "mean_confidence": round(float(np.mean(confidence)), 6),
            "confidence_minus_accuracy": round(float(np.mean(confidence) - np.mean(hits)), 6),
            "expected_calibration_error": round(float(ece), 6),
            "ece_ci95": _bootstrap_ece_ci(confidence, hits, bootstrap_samples, rng),
            "bins": bins,
        })
    return output


def _ece(confidence: np.ndarray, hits: np.ndarray, n_bins: int = 10) -> Tuple[float, List[Dict[str, Any]]]:
    ece = 0.0
    bins = []
    total = max(int(hits.size), 1)
    for index in range(n_bins):
        low = index / n_bins
        high = (index + 1) / n_bins
        mask = (confidence >= low) & (confidence <= high if index == n_bins - 1 else confidence < high)
        count = int(mask.sum())
        if count:
            avg_conf = float(np.mean(confidence[mask]))
            acc = float(np.mean(hits[mask]))
            gap = abs(avg_conf - acc)
            ece += (count / total) * gap
        else:
            avg_conf = 0.0
            acc = 0.0
            gap = 0.0
        bins.append({
            "bin": index + 1,
            "count": count,
            "confidence": round(avg_conf, 6),
            "accuracy": round(acc, 6),
            "gap": round(gap, 6),
        })
    return ece, bins


def _bootstrap_ece_ci(confidence: np.ndarray, hits: np.ndarray, samples: int, rng: np.random.Generator) -> List[float | None]:
    if hits.size < 2:
        value, _ = _ece(confidence, hits)
        return [round(float(value), 6), round(float(value), 6)]
    estimates = []
    for _ in range(max(int(samples), 1)):
        idx = rng.integers(0, hits.size, size=hits.size)
        estimates.append(_ece(confidence[idx], hits[idx])[0])
    return [round(float(np.percentile(estimates, 2.5)), 6), round(float(np.percentile(estimates, 97.5)), 6)]


def _paired_model_comparisons(
        rows_by_model: Dict[str, List[Dict[str, Any]]],
        baseline: str,
        alpha: float,
        bootstrap_samples: int,
        rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    baseline_index = _comparison_index(rows_by_model[baseline])
    raw: List[Dict[str, Any]] = []
    for model_key, rows in rows_by_model.items():
        if model_key == baseline:
            continue
        model_index = _comparison_index(rows)
        common = sorted(set(baseline_index) & set(model_index))
        for market, metric in sorted({(key[1], key[2]) for key in common}):
            paired_keys = [key for key in common if key[1] == market and key[2] == metric]
            if len(paired_keys) < 2:
                continue
            baseline_values = np.asarray([baseline_index[key] for key in paired_keys], dtype=float)
            model_values = np.asarray([model_index[key] for key in paired_keys], dtype=float)
            lower_is_better = metric in LOSS_METRICS
            comparison = (
                _continuous_paired_test(model_key, baseline, market, metric, model_values, baseline_values, lower_is_better, bootstrap_samples, rng)
                if metric in LOSS_METRICS
                else _binary_paired_test(model_key, baseline, market, metric, model_values, baseline_values)
            )
            raw.append(comparison)
    adjusted = _holm_adjust(raw, alpha)
    return adjusted


def _comparison_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], float]:
    return {
        (str(row["fixture_id"]), str(row["market"]), str(row["metric"])): float(row["value"])
        for row in rows
    }


def _continuous_paired_test(
        model_key: str,
        baseline: str,
        market: str,
        metric: str,
        model_values: np.ndarray,
        baseline_values: np.ndarray,
        lower_is_better: bool,
        bootstrap_samples: int,
        rng: np.random.Generator,
) -> Dict[str, Any]:
    diff = model_values - baseline_values
    mean_diff = float(np.mean(diff))
    improvement = -mean_diff if lower_is_better else mean_diff
    p_wilcoxon = _wilcoxon_pvalue(diff)
    p_ttest = _paired_ttest_pvalue(model_values, baseline_values)
    normality_p = _shapiro_pvalue(diff)
    effect = _paired_effect_size(diff)
    return {
        "model_key": model_key,
        "baseline_model_key": baseline,
        "market": market,
        "metric": metric,
        "test": "wilcoxon_signed_rank",
        "n_pairs": int(diff.size),
        "model_mean": round(float(np.mean(model_values)), 6),
        "baseline_mean": round(float(np.mean(baseline_values)), 6),
        "mean_difference_model_minus_baseline": round(mean_diff, 6),
        "improvement": round(float(improvement), 6),
        "ci95_difference": _bootstrap_ci(diff, bootstrap_samples, rng),
        "p_value": p_wilcoxon,
        "paired_t_p_value": p_ttest,
        "normality_shapiro_p": normality_p,
        "effect_size_paired_dz": effect,
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
    }


def _binary_paired_test(
        model_key: str,
        baseline: str,
        market: str,
        metric: str,
        model_values: np.ndarray,
        baseline_values: np.ndarray,
) -> Dict[str, Any]:
    model_hits = model_values >= 0.5
    baseline_hits = baseline_values >= 0.5
    model_only = int(np.sum(model_hits & ~baseline_hits))
    baseline_only = int(np.sum(~model_hits & baseline_hits))
    p_value = _mcnemar_exact_pvalue(model_only, baseline_only)
    n = max(int(model_values.size), 1)
    return {
        "model_key": model_key,
        "baseline_model_key": baseline,
        "market": market,
        "metric": metric,
        "test": "mcnemar_exact_binomial",
        "n_pairs": int(model_values.size),
        "model_mean": round(float(np.mean(model_values)), 6),
        "baseline_mean": round(float(np.mean(baseline_values)), 6),
        "mean_difference_model_minus_baseline": round(float(np.mean(model_values) - np.mean(baseline_values)), 6),
        "improvement": round(float(np.mean(model_values) - np.mean(baseline_values)), 6),
        "model_only_correct": model_only,
        "baseline_only_correct": baseline_only,
        "net_correct_delta": int(model_only - baseline_only),
        "net_correct_delta_rate": round(float((model_only - baseline_only) / n), 6),
        "p_value": p_value,
        "direction": "higher_is_better",
    }


def _wilcoxon_pvalue(diff: np.ndarray) -> float | None:
    if diff.size < 5 or np.allclose(diff, 0.0):
        return None
    try:
        from scipy import stats

        return round(float(stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided").pvalue), 6)
    except Exception:
        return None


def _paired_ttest_pvalue(model_values: np.ndarray, baseline_values: np.ndarray) -> float | None:
    if model_values.size < 3 or np.allclose(model_values, baseline_values):
        return None
    diff = model_values - baseline_values
    if float(np.std(diff, ddof=1)) <= 1e-12:
        return None
    try:
        from scipy import stats

        return round(float(stats.ttest_rel(model_values, baseline_values).pvalue), 6)
    except Exception:
        return None


def _shapiro_pvalue(values: np.ndarray) -> float | None:
    if values.size < 3 or values.size > 5000 or np.allclose(values, values[0]):
        return None
    try:
        from scipy import stats

        return round(float(stats.shapiro(values).pvalue), 6)
    except Exception:
        return None


def _paired_effect_size(diff: np.ndarray) -> float | None:
    if diff.size < 2:
        return None
    sd = float(np.std(diff, ddof=1))
    if sd <= 1e-12:
        return None
    return round(float(np.mean(diff) / sd), 6)


def _mcnemar_exact_pvalue(model_only: int, baseline_only: int) -> float | None:
    discordant = int(model_only + baseline_only)
    if discordant <= 0:
        return None
    try:
        from scipy import stats

        if hasattr(stats, "binomtest"):
            return round(float(stats.binomtest(min(model_only, baseline_only), n=discordant, p=0.5).pvalue), 6)
        return round(float(stats.binom_test(min(model_only, baseline_only), n=discordant, p=0.5)), 6)
    except Exception:
        return None


def _bootstrap_ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> List[float | None]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [None, None]
    if arr.size == 1:
        value = round(float(arr[0]), 6)
        return [value, value]
    estimates = []
    for _ in range(max(int(samples), 1)):
        idx = rng.integers(0, arr.size, size=arr.size)
        estimates.append(float(np.mean(arr[idx])))
    return [round(float(np.percentile(estimates, 2.5)), 6), round(float(np.percentile(estimates, 97.5)), 6)]


def _holm_adjust(items: List[Dict[str, Any]], alpha: float) -> List[Dict[str, Any]]:
    output = [dict(item) for item in items]
    valid = [(index, item) for index, item in enumerate(output) if item.get("p_value") is not None]
    valid.sort(key=lambda pair: float(pair[1]["p_value"]))
    m = len(valid)
    running = 0.0
    for rank, (index, item) in enumerate(valid, start=1):
        adjusted = min(float(item["p_value"]) * (m - rank + 1), 1.0)
        running = max(running, adjusted)
        output[index]["p_value_holm"] = round(running, 6)
        output[index]["significant_holm"] = bool(running < alpha)
    for item in output:
        item.setdefault("p_value_holm", None)
        item.setdefault("significant_holm", False)
    return sorted(output, key=lambda item: (
        str(item.get("market")),
        str(item.get("metric")),
        str(item.get("model_key")),
    ))


def _temporal_stability_summary(model_key: str, rows: List[Dict[str, Any]], folds: int = 3) -> Dict[str, Any]:
    result_rows = [
        row for row in rows
        if row["market"] == "result_1x2" and row["metric"] in {"log_loss", "hit", "brier"}
    ]
    if not result_rows:
        return {"model_key": model_key, "available": False, "reason": "Sin filas 1X2."}
    by_fixture: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for row in result_rows:
        by_fixture[str(row["fixture_id"])]["date"] = str(row.get("date") or "")
        by_fixture[str(row["fixture_id"])][str(row["metric"])] = float(row["value"])
    ordered = sorted(by_fixture.items(), key=lambda item: (item[1].get("date", ""), item[0]))
    if len(ordered) < folds:
        return {"model_key": model_key, "available": False, "reason": "Muestra insuficiente para splits temporales."}
    chunks = np.array_split(np.arange(len(ordered)), min(folds, len(ordered)))
    fold_rows = []
    for fold_index, indices in enumerate(chunks, start=1):
        items = [ordered[int(index)][1] for index in indices]
        fold_rows.append({
            "fold": fold_index,
            "n": len(items),
            "start_date": items[0].get("date", ""),
            "end_date": items[-1].get("date", ""),
            "log_loss": round(float(np.mean([item.get("log_loss", 0.0) for item in items])), 6),
            "brier": round(float(np.mean([item.get("brier", 0.0) for item in items])), 6),
            "accuracy": round(float(np.mean([item.get("hit", 0.0) for item in items])), 6),
        })
    return {
        "model_key": model_key,
        "available": True,
        "folds": fold_rows,
        "log_loss_range": _range_metric(fold_rows, "log_loss"),
        "brier_range": _range_metric(fold_rows, "brier"),
        "accuracy_range": _range_metric(fold_rows, "accuracy"),
        "last_minus_first_log_loss": round(float(fold_rows[-1]["log_loss"] - fold_rows[0]["log_loss"]), 6),
        "last_minus_first_accuracy": round(float(fold_rows[-1]["accuracy"] - fold_rows[0]["accuracy"]), 6),
    }


def _range_metric(rows: List[Dict[str, Any]], key: str) -> float:
    values = [float(row.get(key) or 0.0) for row in rows]
    return round(max(values) - min(values), 6) if values else 0.0


def _audit_recommendations(
        model_metrics: List[Dict[str, Any]],
        calibration: List[Dict[str, Any]],
        stability: List[Dict[str, Any]],
        comparisons: List[Dict[str, Any]],
        baseline: str,
) -> List[str]:
    recommendations: List[str] = []
    significant = [item for item in comparisons if item.get("significant_holm")]
    supported = [
        item for item in significant
        if item.get("improvement") is not None and float(item.get("improvement") or 0.0) > 0
    ]
    if supported:
        best = max(supported, key=lambda item: float(item.get("improvement") or 0.0))
        recommendations.append(
            f"Promueve {best['model_key']} solo para {best['market']}:{best['metric']} si mantiene mejora Holm-significativa en el siguiente corte temporal."
        )
    else:
        recommendations.append(
            f"No promociones modelos sobre {baseline} por p-value aislado; exige mejora pareada Holm-significativa y CI95 de diferencia que no cruce cero."
        )
    for item in calibration:
        for market in item.get("markets") or []:
            gap = float(market.get("confidence_minus_accuracy") or 0.0)
            ece = float(market.get("expected_calibration_error") or 0.0)
            if gap > 0.05 or ece > 0.08:
                recommendations.append(
                    f"Calibra {item['model_key']} en {market['market']}: sobreconfianza {gap:+.3f}, ECE={ece:.3f}; usa temperature/Platt con split temporal pre-test."
                )
    for item in stability:
        if not item.get("available"):
            continue
        if float(item.get("log_loss_range") or 0.0) > 0.25:
            recommendations.append(
                f"Reduce varianza temporal de {item['model_key']}: log-loss por fold varia {item['log_loss_range']:.3f}; sube regularizacion o usa ensemble por ventanas."
            )
    if not recommendations:
        recommendations.append("Mantén el ranking actual y acumula mas partidos; la evidencia todavia no justifica cambios estructurales.")
    recommendations.append(
        "Define antes del backtest una familia primaria de tests: 1X2 log-loss/Brier, exact-score log-loss/top3 y U/O por linea; aplica Holm o FDR."
    )
    return _unique_strings(recommendations)


def _unique_strings(values: Iterable[Any]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output
