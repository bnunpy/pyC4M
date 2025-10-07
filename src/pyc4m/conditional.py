"""Conditional causalized CCM implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike

from .cccm import causalized_ccm


@dataclass
class ConditionalPairResult:
    """Directional conditional CCM effects for a variable pair."""

    x_on_y: float
    y_on_x: float
    diagnostics: Dict[str, float]


@dataclass
class ConditionalCCMResult:
    """Aggregated conditional CCM outputs across multiple variable pairs."""

    pair_results: Dict[Tuple[int, int], ConditionalPairResult]
    base_correlations: np.ndarray
    settings: Dict[str, int]


def conditional_ccm(
    data: ArrayLike,
    tau: int,
    e_dim: int,
    pairs: Sequence[Sequence[int]],
    num_skip: int = 10,
    exclusion_radius: int = 0,
    causal: bool = True,
) -> ConditionalCCMResult:
    """Compute conditional causalized CCM for the requested column pairs.

    Parameters
    ----------
    data : array-like or pandas DataFrame
        Multivariate time-series with shape ``(n_samples, n_variables)``.
    tau : int
        Embedding delay shared across all variables.
    e_dim : int
        Embedding dimension ``E``.
    pairs : sequence of index pairs
        Each element specifies the two columns to evaluate. Indices can be
        zero-based or one-based; one-based indices are automatically
        converted.
    num_skip : int, optional
        Number of reconstructed points discarded before evaluating
        correlations and regression residuals (``N0`` in the MATLAB code).
    exclusion_radius : int, optional
        Temporal neighbourhood excluded for neighbour selection (mirrors
        ``exclusion_radius`` (alias ``exclusionRadius``) in :mod:`pyEDM`).
    causal : bool, optional
        If ``True`` (default) reuse causalized projections; set to ``False``
        to run conditional CCM with standard (non-causal) libraries.

    Returns
    -------
    ConditionalCCMResult
        Object containing directional causality ratios for each requested
        pair along with auxiliary diagnostics.
    """

    data_matrix = _as_2d_array(data)
    n_samples, n_variables = data_matrix.shape

    if n_variables < 3:
        raise ValueError(
            "conditional_ccm requires at least three variables so that the "
            "conditioning set is non-empty."
        )

    try:
        tau_value = int(tau)
    except (TypeError, ValueError) as exc:
        raise ValueError("tau must be an integer") from exc

    if tau_value == 0:
        raise ValueError("tau must be non-zero")
    if e_dim < 1:
        raise ValueError("e_dim must be positive")
    if num_skip < 1:
        raise ValueError("num_skip must be positive")

    if causal and tau_value >= 0:
        raise ValueError(
            "conditional_ccm(): causal projections require tau < 0 to embed into the past"
        )

    try:
        exclusion_radius = int(exclusion_radius)
    except (TypeError, ValueError) as exc:
        raise ValueError("conditional_ccm(): exclusion_radius must be an integer") from exc
    if exclusion_radius < 0:
        raise ValueError("conditional_ccm(): exclusion_radius must be non-negative")

    if not isinstance(causal, (bool, np.bool_)):
        raise ValueError("conditional_ccm(): causal must be a boolean")

    embed_gap = (e_dim - 1) * abs(tau_value)
    latent_length = n_samples - embed_gap
    if latent_length <= num_skip:
        raise ValueError(
            "Time series too short for the requested embedding and num_skip"
        )

    tail_start = embed_gap if tau_value < 0 else 0

    # Pre-compute causalized CCM reconstructions for all variable pairs.
    estimates = np.full((n_variables, n_variables, latent_length), np.nan)
    correlations = np.full((n_variables, n_variables), np.nan)

    for i in range(n_variables - 1):
        for j in range(i + 1, n_variables):
            result = causalized_ccm(
                data_matrix[:, i],
                data_matrix[:, j],
                tau=tau_value,
                e_dim=e_dim,
                num_skip=num_skip,
                exclusion_radius=exclusion_radius,
                causal=causal,
            )
            estimates[i, j, :] = result.y_estimates
            estimates[j, i, :] = result.x_estimates
            correlations[i, j] = result.correlation_y
            correlations[j, i] = result.correlation_x

    standardized_pairs = _normalise_pairs(pairs, n_variables)
    pair_results: Dict[Tuple[int, int], ConditionalPairResult] = {}

    # Actual data aligned with the reconstructed manifold tail.
    tail_data = data_matrix[tail_start : tail_start + latent_length, :]
    valid_slice = slice(num_skip - 1, None)

    for (i, j) in standardized_pairs:
        conditioning = [idx for idx in range(n_variables) if idx not in (i, j)]

        ccm_x_from_conditions = _gather_conditionals(
            estimates, conditioning, i
        )
        ccm_y_from_conditions = _gather_conditionals(
            estimates, conditioning, j
        )

        x_from_y = estimates[j, i, :][..., np.newaxis]
        y_from_x = estimates[i, j, :][..., np.newaxis]

        # Remove burn-in rows
        x_from_y = x_from_y[valid_slice]
        y_from_x = y_from_x[valid_slice]
        x_conditions = ccm_x_from_conditions[valid_slice]
        y_conditions = ccm_y_from_conditions[valid_slice]

        target_x = tail_data[valid_slice, i]
        target_y = tail_data[valid_slice, j]

        effect_y_to_x = _conditional_effect(
            x_from_y, x_conditions, target_x
        )
        effect_x_to_y = _conditional_effect(
            y_from_x, y_conditions, target_y
        )

        pair_results[(i, j)] = ConditionalPairResult(
            x_on_y=effect_x_to_y[0],
            y_on_x=effect_y_to_x[0],
            diagnostics={
                "var_x_with_cross": effect_y_to_x[1],
                "var_x_conditionals": effect_y_to_x[2],
                "var_y_with_cross": effect_x_to_y[1],
                "var_y_conditionals": effect_x_to_y[2],
            },
        )

    settings = {"tau": tau_value, "e_dim": e_dim, "num_skip": num_skip, "exclusion_radius": exclusion_radius, "causal": causal}
    return ConditionalCCMResult(
        pair_results=pair_results,
        base_correlations=correlations,
        settings=settings,
    )


def _as_2d_array(data: ArrayLike) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ValueError("data must be two-dimensional")
    return arr


def _normalise_pairs(
    pairs: Sequence[Sequence[int]], n_variables: int
) -> List[Tuple[int, int]]:
    if not pairs:
        raise ValueError("No pairs supplied")

    processed: List[Tuple[int, int]] = []
    use_one_based = all(min(pair) >= 1 for pair in pairs)

    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"Pair {pair} must contain exactly two indices")
        a, b = pair
        if use_one_based:
            a -= 1
            b -= 1
        if not (0 <= a < n_variables and 0 <= b < n_variables):
            raise ValueError(f"Pair {(a, b)} out of bounds for {n_variables} variables")
        if a == b:
            raise ValueError("Pair indices must differ")
        processed.append(tuple(sorted((a, b))))

    return processed


def _gather_conditionals(
    estimates: np.ndarray, conditional_idx: List[int], target: int
) -> np.ndarray:
    latent_length = estimates.shape[2]
    if not conditional_idx:
        raise ValueError(
            "Conditioning set is empty; provide at least one additional variable"
        )

    block = estimates[conditional_idx, target, :]
    block = np.asarray(block, dtype=float)
    block = block.reshape(len(conditional_idx), latent_length)
    return block.T


def _conditional_effect(
    cross_mapping: np.ndarray,
    conditioning: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float, float]:
    """Return causality ratio and residual variances."""

    predictors_all = np.hstack([conditioning, cross_mapping])

    finite_mask = np.isfinite(target) & np.all(np.isfinite(predictors_all), axis=1)
    if finite_mask.sum() <= predictors_all.shape[1]:
        return float('nan'), float('nan'), float('nan')

    predictors_all = predictors_all[finite_mask]
    conditioning = conditioning[finite_mask]
    cross_mapping = cross_mapping[finite_mask]
    target = target[finite_mask]

    coef_all, *_ = np.linalg.lstsq(predictors_all, target, rcond=None)
    fitted_all = predictors_all @ coef_all
    err_all = target - fitted_all

    if conditioning.shape[1] == 0:
        raise ValueError(
            "Conditioning matrix has zero columns; ensure at least one conditioning variable"
        )

    if conditioning.shape[0] <= conditioning.shape[1]:
        return float('nan'), float('nan'), float('nan')

    coef_cond, *_ = np.linalg.lstsq(conditioning, target, rcond=None)
    fitted_cond = conditioning @ coef_cond
    err_cond = target - fitted_cond

    var_all = float(np.var(err_all, ddof=1))
    var_cond = float(np.var(err_cond, ddof=1))

    effect = np.nan if var_cond == 0 else (var_cond - var_all) / var_cond

    return float(effect), var_all, var_cond
