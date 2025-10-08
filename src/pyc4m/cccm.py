"""Causalized Convergent Cross Mapping (cCCM) implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

# Detect whether the current SciPy build supports multi-threaded KD-tree queries.
try:
    _CKDTREE_QUERY_WORKERS = {"workers": -1}
    cKDTree(np.zeros((1, 1))).query(np.zeros(1), k=1, **_CKDTREE_QUERY_WORKERS)
except TypeError:  # pragma: no cover - depends on SciPy version
    _CKDTREE_QUERY_WORKERS = {}


@dataclass
class CausalizedCCMResult:
    """Container for causalized CCM outputs.

    Attributes
    ----------
    correlation_x : float
        Correlation between observed ``X`` and its cross-mapped estimate.
    correlation_y : float
        Correlation between observed ``Y`` and its cross-mapped estimate.
    x_estimates : np.ndarray
        Estimated ``X`` reconstructed from ``Y`` (length: ``L - T + 1``).
    y_estimates : np.ndarray
        Estimated ``Y`` reconstructed from ``X`` (length: ``L - T + 1``).
    """

    correlation_x: float
    correlation_y: float
    x_estimates: np.ndarray
    y_estimates: np.ndarray


@dataclass(frozen=True)
class VariableGeometry:
    """Pre-computed embedding geometry for a single variable."""

    manifold: np.ndarray
    tail: np.ndarray
    tree: cKDTree


def causalized_ccm(
    x: ArrayLike,
    y: ArrayLike,
    tau: int,
    e_dim: int,
    num_skip: int = 10,
    epsilon: float = 1e-12,
    tp: int = 0,
    library_indices: Optional[Sequence[int]] = None,
    exclusion_radius: int = 0,
    causal: bool = True,
    manifold_x: Optional[np.ndarray] = None,
    manifold_y: Optional[np.ndarray] = None,
    series_x_tail: Optional[ArrayLike] = None,
    series_y_tail: Optional[ArrayLike] = None,
    geometry_x: Optional[VariableGeometry] = None,
    geometry_y: Optional[VariableGeometry] = None,
) -> CausalizedCCMResult:
    """Compute causalized CCM between two scalar time series.

    Parameters
    ----------
    x, y : array-like
        Input time series of equal length.
    tau : int
        Time delay between embedding coordinates. Negative values embed
        into the past (pyEDM convention); positive values embed into the
        future. ``tau`` must be non-zero.
    e_dim : int
        Embedding dimension ``E``.
    num_skip : int, optional
        Number of initial reconstructed points ignored when computing
        correlations (``N0`` in the MATLAB reference implementation).
    epsilon : float, optional
        Small positive constant added to neighbor distances to avoid
        division by zero when calculating weights.
    tp : int, optional
        Prediction horizon. Positive values estimate future points
        (``Tp > 0``), negative values infer the past, matching the
        `Tp` semantics in :mod:`pyEDM`.
    library_indices : sequence of int, optional
        Embedding-row indices available to the library. The causalized
        search intersects these with ``[0, j]`` for each time step.
    exclusion_radius : int, optional
        Temporal neighbourhood excluded when selecting neighbours
        (mirrors ``exclusion_radius`` (alias ``exclusionRadius``) in :mod:`pyEDM`).
    causal : bool, optional
        If ``True`` (default) enforce causal libraries (only indices ``<= j``).
        Set to ``False`` to reproduce standard CCM behaviour that allows
        future library points.

    Returns
    -------
    CausalizedCCMResult
        Dataclass containing correlation coefficients and reconstructed
        time series.

    Notes
    -----
    This function follows the algorithm published with the
    "Causalized Convergent Cross Mapping" reference implementation.
    Only historical information (indices ``<= j``) is used when locating
    neighbors for the ``j``th point, matching the causal constraint
    of the MATLAB code.
    """

    x_arr = _as_float_array(x)
    y_arr = _as_float_array(y)

    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    if x_arr.ndim != 1:
        raise ValueError("x and y must be one-dimensional sequences")
    if e_dim < 1:
        raise ValueError("e_dim must be positive")
    if tau == 0:
        raise ValueError("tau must be non-zero")
    if num_skip < 1:
        raise ValueError("num_skip must be positive")

    try:
        tp = int(tp)
    except (TypeError, ValueError) as exc:
        raise ValueError("tp must be an integer") from exc

    try:
        exclusion_radius = int(exclusion_radius)
    except (TypeError, ValueError) as exc:
        raise ValueError("exclusion_radius must be an integer") from exc

    if exclusion_radius < 0:
        raise ValueError("exclusion_radius must be non-negative")

    if not isinstance(causal, (bool, np.bool_)):
        raise ValueError("causal must be a boolean")

    if causal and tau > 0:
        raise ValueError(
            "causalized_ccm(): causal projections require tau < 0 to embed into the past"
        )

    preembedded = manifold_x is not None and manifold_y is not None

    if geometry_x is not None:
        manifold_x = geometry_x.manifold
        xt = geometry_x.tail
        tree_x = geometry_x.tree
        n_vectors = manifold_x.shape[0]
    else:
        if preembedded:
            manifold_x = np.asarray(manifold_x, dtype=float)
            if manifold_x.ndim != 2:
                raise ValueError("Precomputed manifolds must be two-dimensional")
            if manifold_x.shape[1] != e_dim:
                raise ValueError(
                    f"Precomputed manifold dimension {manifold_x.shape[1]} does not match e_dim={e_dim}"
                )
            if series_x_tail is None:
                raise ValueError(
                    "series_x_tail must be provided when using precomputed manifolds"
                )
            xt = _as_float_array(series_x_tail)
            n_vectors = manifold_x.shape[0]
            if xt.size != n_vectors:
                raise ValueError("series_x_tail length must match manifold rows")
        else:
            series_length = x_arr.size
            embed_span = (e_dim - 1) * abs(tau)
            required = embed_span + 1
            if required > series_length:
                raise ValueError(
                    "Embedding exceeds series length: need at least "
                    f"{required} points, received {series_length}"
                )

            manifold_x, xt = _construct_manifold(x_arr, tau, e_dim)
            n_vectors = manifold_x.shape[0]
        tree_x = cKDTree(manifold_x)

    if geometry_y is not None:
        manifold_y = geometry_y.manifold
        yt = geometry_y.tail
        tree_y = geometry_y.tree
        if manifold_y.shape[0] != n_vectors:
            raise ValueError("geometry_y manifold length mismatch with geometry_x")
    else:
        if preembedded:
            manifold_y = np.asarray(manifold_y, dtype=float)
            if manifold_y.ndim != 2:
                raise ValueError("Precomputed manifolds must be two-dimensional")
            if manifold_y.shape[1] != e_dim:
                raise ValueError(
                    f"Precomputed manifold dimension {manifold_y.shape[1]} does not match e_dim={e_dim}"
                )
            if manifold_y.shape[0] != n_vectors:
                raise ValueError("manifold_y must match manifold_x row count")
            if series_y_tail is None:
                raise ValueError(
                    "series_y_tail must be provided when using precomputed manifolds"
                )
            yt = _as_float_array(series_y_tail)
            if yt.size != n_vectors:
                raise ValueError("series_y_tail length must match manifold rows")
        else:
            manifold_y, yt = _construct_manifold(y_arr, tau, e_dim)
            if manifold_y.shape[0] != n_vectors:
                raise ValueError("manifold_x and manifold_y must have the same shape")
        tree_y = cKDTree(manifold_y)

    if library_indices is not None:
        lib_idx = np.asarray(library_indices, dtype=int)
        if lib_idx.ndim != 1:
            raise ValueError("library_indices must be one-dimensional")
        if lib_idx.size == 0:
            raise ValueError("library_indices must contain at least one index")
        if np.any(lib_idx < 0) or np.any(lib_idx >= n_vectors):
            raise ValueError("library_indices out of range")
        lib_idx = np.unique(lib_idx)
    else:
        lib_idx = None

    base_indices = lib_idx if lib_idx is not None else np.arange(n_vectors, dtype=int)

    if abs(tp) >= n_vectors:
        raise ValueError(
            "tp magnitude leaves no usable reconstructed points: "
            f"|tp|={abs(tp)} >= available vectors {n_vectors}"
        )

    if num_skip >= n_vectors:
        raise ValueError(
            "num_skip prevents any points from being evaluated: "
            f"num_skip={num_skip}, available vectors={n_vectors}"
        )

    nn = e_dim + 1  # number of nearest neighbours to use
    abs_tp = abs(tp)
    x_est = np.full(n_vectors, np.nan, dtype=float)
    y_est = np.full(n_vectors, np.nan, dtype=float)

    for j in range(num_skip - 1, n_vectors):
        if causal:
            candidate_idx = base_indices[base_indices <= j]
        else:
            candidate_idx = base_indices

        if exclusion_radius > 0 and candidate_idx.size:
            candidate_idx = candidate_idx[
                np.abs(candidate_idx - j) > exclusion_radius
            ]

        candidate_idx = candidate_idx[candidate_idx != j]
        candidate_idx = np.ascontiguousarray(candidate_idx, dtype=int)
        if candidate_idx.size == 0:
            continue

        needed = min(candidate_idx.size, nn + abs_tp)
        neighbours_y_global, distances_y = _query_tree_neighbors(
            tree_x,
            manifold_x,
            j,
            candidate_idx,
            n_vectors,
            needed + abs_tp,
        )
        target_indices_y, distances_y = _apply_tp_to_neighbors(
            neighbours_y_global,
            distances_y,
            n_vectors,
            tp,
            nn,
        )
        if distances_y.size:
            weights_y = _compute_weights(distances_y, epsilon)
            y_est[j] = np.dot(weights_y, yt[target_indices_y])

        neighbours_x_global, distances_x = _query_tree_neighbors(
            tree_y,
            manifold_y,
            j,
            candidate_idx,
            n_vectors,
            needed + abs_tp,
        )
        target_indices_x, distances_x = _apply_tp_to_neighbors(
            neighbours_x_global,
            distances_x,
            n_vectors,
            tp,
            nn,
        )
        if distances_x.size:
            weights_x = _compute_weights(distances_x, epsilon)
            x_est[j] = np.dot(weights_x, xt[target_indices_x])

    x_est = _shift_predictions(x_est, tp)
    y_est = _shift_predictions(y_est, tp)

    corr_x = _pearson_correlation(xt[num_skip - 1 :], x_est[num_skip - 1 :])
    corr_y = _pearson_correlation(yt[num_skip - 1 :], y_est[num_skip - 1 :])

    return CausalizedCCMResult(corr_x, corr_y, x_est, y_est)


def _as_float_array(data: ArrayLike) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    return np.ravel(arr)


def _construct_manifold(
    series: np.ndarray, tau: int, e_dim: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Return delay embedding matrix and aligned series samples."""

    series_length = series.size
    lag = abs(tau)

    if e_dim == 0:
        raise ValueError("e_dim must be positive")

    if tau < 0:
        start = lag * (e_dim - 1)
        stop = series_length
    else:
        start = 0
        stop = series_length - lag * (e_dim - 1)

    if stop <= start:
        raise ValueError(
            "Embedding configuration leaves no usable state vectors: "
            f"E={e_dim}, tau={tau}, series length={series_length}"
        )

    base_indices = np.arange(start, stop, dtype=int)
    offsets = tau * np.arange(e_dim, dtype=int)

    manifold = series[np.add.outer(base_indices, offsets)]
    tail = series[base_indices]
    return manifold.astype(float, copy=False), tail.astype(float, copy=False)


def _query_tree_neighbors(
    tree: cKDTree,
    manifold: np.ndarray,
    query_index: int,
    candidate_indices: np.ndarray,
    n_vectors: int,
    requested: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Query a KDTree and filter results to the allowed candidate indices."""

    if candidate_indices.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    request = min(n_vectors - 1, max(requested, 1))
    point = manifold[query_index]

    while True:
        distances, indices = tree.query(point, k=request + 1, **_CKDTREE_QUERY_WORKERS)
        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)

        if indices.ndim > 1:
            indices = indices.flatten()
            distances = distances.flatten()

        match_positions = np.searchsorted(candidate_indices, indices, side="left")
        selection = np.zeros(indices.shape, dtype=bool)
        valid = match_positions < candidate_indices.size
        if np.any(valid):
            selection[valid] = candidate_indices[match_positions[valid]] == indices[valid]

        filtered_indices = indices[selection]
        filtered_distances = distances[selection]

        if filtered_indices.size >= requested or request >= n_vectors - 1:
            break

        request = min(n_vectors - 1, request * 2)

    return filtered_indices.astype(int), filtered_distances.astype(float)


def _apply_tp_to_neighbors(
    indices: np.ndarray,
    distances: np.ndarray,
    n_vectors: int,
    tp: int,
    limit: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if indices.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    shifted = indices + tp
    valid = (shifted >= 0) & (shifted < n_vectors)
    if not np.any(valid):
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    shifted = shifted[valid]
    distances = distances[valid]

    if shifted.size > limit:
        shifted = shifted[:limit]
        distances = distances[:limit]

    return shifted.astype(int), distances.astype(float)


def _shift_predictions(predictions: np.ndarray, tp: int) -> np.ndarray:
    if tp == 0:
        return predictions

    shifted = np.roll(predictions, tp)
    if tp > 0:
        shifted[:tp] = np.nan
    else:
        shifted[tp:] = np.nan
    return shifted


def _compute_weights(distances: np.ndarray, epsilon: float) -> np.ndarray:
    if distances.size == 0:
        raise ValueError("No distances provided for weight computation")

    adjusted = distances + epsilon
    reference = adjusted[0]
    if reference <= 0:
        reference = epsilon
    scales = np.exp(-adjusted / reference)
    weight_sum = np.sum(scales)
    if weight_sum == 0:
        return np.full_like(scales, fill_value=1.0 / scales.size)
    return scales / weight_sum


def _pearson_correlation(a: Iterable[float], b: Iterable[float]) -> float:
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    mask = np.isfinite(a_arr) & np.isfinite(b_arr)
    if mask.sum() < 2:
        return np.nan
    a_sel = a_arr[mask]
    b_sel = b_arr[mask]
    a_sel = a_sel - np.mean(a_sel)
    b_sel = b_sel - np.mean(b_sel)
    denom = np.sqrt(np.sum(a_sel**2) * np.sum(b_sel**2))
    if denom == 0:
        return np.nan
    return float(np.dot(a_sel, b_sel) / denom)


def prepare_variable_geometry(
    tau: int,
    e_dim: int,
    *,
    series: ArrayLike | None = None,
    manifold: Optional[np.ndarray] = None,
    tail: Optional[ArrayLike] = None,
) -> VariableGeometry:
    """Construct reusable manifold and KD-tree for a single variable."""

    if manifold is not None:
        manifold_arr = np.asarray(manifold, dtype=float)
        if manifold_arr.ndim != 2:
            raise ValueError("prepare_variable_geometry(): manifold must be two-dimensional")
        if manifold_arr.shape[1] != e_dim:
            raise ValueError(
                "prepare_variable_geometry(): manifold columns must match embedding dimension"
            )
        if tail is None:
            raise ValueError(
                "prepare_variable_geometry(): tail must be provided when manifold is supplied"
            )
        tail_arr = _as_float_array(tail)
        if tail_arr.size != manifold_arr.shape[0]:
            raise ValueError(
                "prepare_variable_geometry(): tail length must equal manifold rows"
            )
    else:
        if series is None:
            raise ValueError(
                "prepare_variable_geometry(): series must be provided when manifold is absent"
            )
        series_arr = _as_float_array(series)
        manifold_arr, tail_arr = _construct_manifold(series_arr, tau, e_dim)

    tree = cKDTree(manifold_arr)
    return VariableGeometry(
        manifold=manifold_arr,
        tail=tail_arr,
        tree=tree,
    )
