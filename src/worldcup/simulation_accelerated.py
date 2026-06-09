from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from src.worldcup.accelerators import cuda, njit, numba_cuda_available, prange


if njit is not None:
    @njit(parallel=True, fastmath=True, cache=True)
    def _poisson_scores_cpu(uniforms, lambdas, outcome_probs, enforce_outcome, max_score, output):
        iterations, matches, _ = uniforms.shape
        for iteration in prange(iterations):
            for match_index in range(matches):
                lam1 = lambdas[match_index, 0]
                lam2 = lambdas[match_index, 1]
                goals1 = _poisson_icdf(uniforms[iteration, match_index, 0], lam1, max_score)
                goals2 = _poisson_icdf(uniforms[iteration, match_index, 1], lam2, max_score)
                if enforce_outcome:
                    outcome = uniforms[iteration, match_index, 2]
                    home_p = outcome_probs[match_index, 0]
                    draw_p = outcome_probs[match_index, 1]
                    if outcome <= home_p and goals1 <= goals2:
                        goals1 = goals2 + 1
                    elif outcome <= home_p + draw_p:
                        draw_goals = int(round((goals1 + goals2) / 2.0))
                        goals1 = draw_goals
                        goals2 = draw_goals
                    elif goals2 <= goals1:
                        goals2 = goals1 + 1
                output[iteration, match_index, 0] = goals1
                output[iteration, match_index, 1] = goals2


    @njit(fastmath=True, cache=True)
    def _poisson_icdf(uniform_value, rate, max_score):
        rate = min(max(float(rate), 0.01), 8.0)
        probability = math.exp(-rate)
        cumulative = probability
        goals = 0
        while uniform_value > cumulative and goals < max_score:
            goals += 1
            probability *= rate / goals
            cumulative += probability
        return goals
else:
    _poisson_scores_cpu = None


if cuda is not None:
    @cuda.jit
    def _poisson_scores_cuda(uniforms, lambdas, outcome_probs, enforce_outcome, max_score, output):
        index = cuda.grid(1)
        iterations = uniforms.shape[0]
        matches = uniforms.shape[1]
        total = iterations * matches
        if index >= total:
            return
        iteration = index // matches
        match_index = index - iteration * matches
        lam1 = lambdas[match_index, 0]
        lam2 = lambdas[match_index, 1]
        goals1 = _poisson_icdf_cuda(uniforms[iteration, match_index, 0], lam1, max_score)
        goals2 = _poisson_icdf_cuda(uniforms[iteration, match_index, 1], lam2, max_score)
        if enforce_outcome:
            outcome = uniforms[iteration, match_index, 2]
            home_p = outcome_probs[match_index, 0]
            draw_p = outcome_probs[match_index, 1]
            if outcome <= home_p and goals1 <= goals2:
                goals1 = goals2 + 1
            elif outcome <= home_p + draw_p:
                draw_goals = int((goals1 + goals2 + 1) // 2)
                goals1 = draw_goals
                goals2 = draw_goals
            elif goals2 <= goals1:
                goals2 = goals1 + 1
        output[iteration, match_index, 0] = goals1
        output[iteration, match_index, 1] = goals2


    @cuda.jit(device=True)
    def _poisson_icdf_cuda(uniform_value, rate, max_score):
        rate = min(max(float(rate), 0.01), 8.0)
        probability = math.exp(-rate)
        cumulative = probability
        goals = 0
        while uniform_value > cumulative and goals < max_score:
            goals += 1
            probability *= rate / goals
            cumulative += probability
        return goals
else:
    _poisson_scores_cuda = None


def sample_group_scores(matches: List[Dict[str, Any]], model: Any, iterations: int, seed: int, prefer_cuda: bool = True) -> Tuple[np.ndarray | None, Dict[str, Any]]:
    if not matches:
        return None, {"backend": "python", "label": "Monte Carlo Python", "accelerated": False}
    lambdas = np.ascontiguousarray([
        model.expected_goals(str(match["team1"]), str(match["team2"]))
        for match in matches
    ], dtype=np.float32)
    enforce_outcome = bool(hasattr(model, "base_model"))
    outcome_probs = np.zeros((len(matches), 3), dtype=np.float32)
    if enforce_outcome:
        for index, match in enumerate(matches):
            probs = model.match_probabilities(str(match["team1"]), str(match["team2"]))
            outcome_probs[index] = [float(probs.get("home", 0.0)), float(probs.get("draw", 0.0)), float(probs.get("away", 0.0))]
    rng = np.random.default_rng(int(seed))
    uniforms = rng.random((int(iterations), len(matches), 3), dtype=np.float32)
    output = np.empty((int(iterations), len(matches), 2), dtype=np.int16)
    max_score = 18
    if prefer_cuda and numba_cuda_available() and _poisson_scores_cuda is not None:
        try:
            threads = 256
            blocks = (iterations * len(matches) + threads - 1) // threads
            d_uniforms = cuda.to_device(uniforms)
            d_lambdas = cuda.to_device(lambdas)
            d_outcome = cuda.to_device(outcome_probs)
            d_output = cuda.device_array(output.shape, dtype=np.int16)
            _poisson_scores_cuda[blocks, threads](d_uniforms, d_lambdas, d_outcome, enforce_outcome, max_score, d_output)
            cuda.synchronize()
            return d_output.copy_to_host(), {
                "backend": "cuda",
                "label": "Monte Carlo CUDA",
                "accelerated": True,
                "scope": "group_score_sampler",
            }
        except Exception as exc:
            cuda_error = f"{exc.__class__.__name__}: {exc}"
        else:
            cuda_error = ""
    else:
        cuda_error = ""
    if _poisson_scores_cpu is not None:
        _poisson_scores_cpu(uniforms, lambdas, outcome_probs, enforce_outcome, max_score, output)
        return output, {
            "backend": "cpu_numba",
            "label": "Monte Carlo CPU Numba",
            "accelerated": True,
            "scope": "group_score_sampler",
            "cuda_error": cuda_error,
        }
    poisson = rng.poisson(lambdas[None, :, :], size=(int(iterations), len(matches), 2)).astype(np.int16)
    return poisson, {
        "backend": "numpy",
        "label": "Monte Carlo NumPy",
        "accelerated": True,
        "scope": "group_score_sampler",
        "cuda_error": cuda_error,
    }
