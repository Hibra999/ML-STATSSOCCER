from src.worldcup.statistical_audit import build_prediction_statistical_audit


def _sample(index, *, hit, actual_probability, score_probability, top3=True, ou25_hit=True):
    actual = "home" if index % 3 == 0 else "draw" if index % 3 == 1 else "away"
    probs = {"home": 25.0, "draw": 25.0, "away": 50.0}
    probs[actual] = actual_probability
    remaining = max(100.0 - actual_probability, 0.0)
    others = [key for key in probs if key != actual]
    probs[others[0]] = remaining * 0.55
    probs[others[1]] = remaining * 0.45
    return {
        "fixture_id": str(index),
        "date": f"2026-06-{index + 1:02d}",
        "home": f"Home {index}",
        "away": f"Away {index}",
        "actual_pick_key": actual,
        "pick_hit": hit,
        "score_hit": index % 6 == 0,
        "top3_score_hit": top3,
        "actual_probability": actual_probability,
        "score_probability": score_probability,
        "confidence": max(probs.values()),
        "probabilities": probs,
        "over_under": [
            {
                "line": "2.5",
                "hit": ou25_hit,
                "log_loss": 0.45 if ou25_hit else 1.2,
                "brier": 0.12 if ou25_hit else 0.36,
                "confidence": 65.0 if ou25_hit else 58.0,
            }
        ],
    }


def test_statistical_audit_detects_paired_market_improvements():
    baseline = {
        "model_key": "independent_poisson",
        "evaluated_matches": 12,
        "matches": [
            _sample(index, hit=index % 3 == 0, actual_probability=42.0, score_probability=3.0, top3=index % 3 == 0, ou25_hit=index % 2 == 0)
            for index in range(12)
        ],
    }
    improved = {
        "model_key": "statsmodels_poisson_glm",
        "evaluated_matches": 12,
        "matches": [
            _sample(index, hit=True, actual_probability=68.0, score_probability=8.0, top3=True, ou25_hit=True)
            for index in range(12)
        ],
    }

    audit = build_prediction_statistical_audit([baseline, improved], bootstrap_samples=200, seed=7)

    assert audit["available"] is True
    assert audit["baseline_model_key"] == "independent_poisson"
    assert audit["market_comparisons"]
    result_hit = [
        item for item in audit["market_comparisons"]
        if item["model_key"] == "statsmodels_poisson_glm"
        and item["market"] == "result_1x2"
        and item["metric"] == "hit"
    ][0]
    assert result_hit["improvement"] > 0
    assert result_hit["test"] == "mcnemar_exact_binomial"
    assert audit["calibration"]
    assert audit["temporal_stability"]
    assert audit["recommendations"]


def test_statistical_audit_reports_unavailable_without_backtests():
    audit = build_prediction_statistical_audit([])

    assert audit["available"] is False
    assert audit["warnings"]
    assert "backtest" in audit["recommendations"][0]
