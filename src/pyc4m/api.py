"""Drop-in replacements for selected pyEDM API calls using causalized CCM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from numpy.random import default_rng
from matplotlib.pyplot import axhline, show

from pyEDM.AuxFunc import ComputeError, IsIterable

from .cccm import causalized_ccm
from .conditional import conditional_ccm


@dataclass
class CausalizedCCMRun:
    """Minimal object mimicking :class:`pyEDM.CCM` returnObject payload."""

    libMeans: pd.DataFrame
    PredictStats1: pd.DataFrame | None
    PredictStats2: pd.DataFrame | None

    # A few attributes kept for compatibility with user code that
    # inspects the CCM object.
    name: str = "CausalizedCCM"


def CCM(
    dataFrame = None,
    columns: str | Sequence[str] = "",
    target: str | Sequence[str] = "",
    libSizes: Iterable[int] | str = "",
    sample: int = 0,
    E: int = 0,
    Tp: int = 0,
    knn: int = 0,
    tau: int = -1,
    exclusionRadius: int | None = None,
    exclusion_radius: int | None = None,
    seed = None,
    embedded: bool = False,
    validLib: Sequence[int] | None = None,
    includeData: bool = False,
    noTime: bool = False,
    ignoreNan: bool = True,
    verbose: bool = False,
    showPlot: bool = False,
    returnObject: bool = False,
    causal: bool = True,
    **kwargs,
):
    """Causalized convergent cross mapping with a pyEDM-compatible API."""

    if dataFrame is None:
        raise RuntimeError("CCM(): dataFrame must be provided")

    if isinstance(dataFrame, pd.DataFrame):
        df = dataFrame.copy()
    elif hasattr(dataFrame, "to_frame"):
        df = dataFrame.to_frame()
    else:
        raise RuntimeError("CCM(): dataFrame must be pandas DataFrame like")

    if embedded:
        raise NotImplementedError("CCM(): embedded=True is not yet supported")
    if validLib:
        raise NotImplementedError("CCM(): validLib is not supported")
    tau = 1 if tau is None or tau <= 0 else tau
    if E <= 0:
        raise RuntimeError("CCM(): embedding dimension E must be positive")

    try:
        tp_value = 0 if Tp is None else int(Tp)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("CCM(): Tp must be an integer") from exc

    if exclusion_radius is not None and exclusionRadius is not None:
        raise RuntimeError("CCM(): specify only one of exclusion_radius or exclusionRadius")

    exclusion_param = exclusion_radius if exclusion_radius is not None else exclusionRadius

    try:
        exclusion_radius_value = 0 if exclusion_param is None else int(exclusion_param)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("CCM(): exclusion_radius must be an integer") from exc
    if exclusion_radius_value < 0:
        raise RuntimeError("CCM(): exclusion_radius must be non-negative")

    if not isinstance(causal, (bool, np.bool_)):
        raise RuntimeError("CCM(): causal must be a boolean")

    c_col = _first(columns, "columns")
    t_col = _first(target, "target")

    if c_col not in df.columns:
        raise RuntimeError(f"CCM(): column '{c_col}' not found in dataFrame")
    if t_col not in df.columns:
        raise RuntimeError(f"CCM(): target '{t_col}' not found in dataFrame")

    series_x = df[c_col].to_numpy(dtype=float)
    series_y = df[t_col].to_numpy(dtype=float)

    lib_sizes = _parse_lib_sizes(libSizes, len(df), E, tau, tp_value)

    if verbose:
        print(
            f"Causalized CCM evaluating libSizes={lib_sizes} with E={E}, tau={tau}, Tp={tp_value}"
        )

    num_skip = kwargs.pop("num_skip", 10)
    epsilon = kwargs.pop("epsilon", 1e-12)
    if kwargs:
        unexpected = ", ".join(kwargs.keys())
        raise TypeError(f"CCM(): unexpected keyword arguments: {unexpected}")

    lib_means: Dict[str, List[float]] = {
        "LibSize": [],
        f"{c_col}:{t_col}": [],
        f"{t_col}:{c_col}": [],
    }

    forward_stats: List[Dict[str, float]] = []
    reverse_stats: List[Dict[str, float]] = []

    total_points = len(series_x)
    if total_points != len(series_y):
        raise RuntimeError("CCM(): columns and target must be equal length")

    t_offset = 1 + (E - 1) * tau
    n_vectors_total = total_points - t_offset + 1
    if n_vectors_total <= 1:
        raise RuntimeError("CCM(): insufficient points for the requested embedding")

    try:
        sample_count = int(sample)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("CCM(): sample must be an integer") from exc
    if sample_count < 0:
        raise RuntimeError("CCM(): sample must be non-negative")
    if sample_count == 0:
        sample_count = 1

    rng = default_rng(seed)
    skip_base = max(1, num_skip)
    if skip_base >= n_vectors_total:
        skip_base = max(1, n_vectors_total - 1)
        if verbose:
            print(
                f"Adjusting num_skip to {skip_base} for dataset (available vectors={n_vectors_total})"
            )

    tail_x = series_x[t_offset - 1 :]
    tail_y = series_y[t_offset - 1 :]

    for lib_size in lib_sizes:
        lib_emb = lib_size - (E - 1) * tau
        if lib_emb <= 0:
            raise RuntimeError(
                f"CCM(): library size {lib_size} is too small for embedding E={E}, tau={tau}"
            )

        lib_emb = min(lib_emb, n_vectors_total)
        abs_tp = abs(tp_value)
        if lib_emb <= abs_tp:
            raise RuntimeError(
                f"CCM(): library size {lib_size} yields only {lib_emb} embedding vectors, "
                f"which is <= |Tp|={abs_tp}. Increase libSizes or reduce |Tp|."
            )

        rho_xy_samples = []
        rho_yx_samples = []

        for sample_index in range(sample_count):
            if sample_count == 1 or sample_index == 0:
                library_idx = np.arange(lib_emb, dtype=int)
            else:
                library_idx = np.sort(rng.choice(n_vectors_total, size=lib_emb, replace=False))

            skip = min(skip_base, max(1, lib_emb - 1))

            result = causalized_ccm(
                series_x,
                series_y,
                tau=tau,
                e_dim=E,
                num_skip=skip,
                epsilon=epsilon,
                tp=tp_value,
                library_indices=library_idx,
                exclusion_radius=exclusion_radius_value,
            
                causal=causal,
            )

            rho_xy_samples.append(result.correlation_y)
            rho_yx_samples.append(result.correlation_x)

            if includeData:
                valid_slice = slice(skip - 1, None)
                obs_y = tail_y[valid_slice]
                pred_y = result.y_estimates[valid_slice]
                err_y = ComputeError(obs_y, pred_y, digits=6)
                err_y["LibSize"] = lib_size
                err_y["Sample"] = sample_index

                obs_x = tail_x[valid_slice]
                pred_x = result.x_estimates[valid_slice]
                err_x = ComputeError(obs_x, pred_x, digits=6)
                err_x["LibSize"] = lib_size
                err_x["Sample"] = sample_index

                forward_stats.append(err_y)
                reverse_stats.append(err_x)

        lib_means["LibSize"].append(lib_size)
        lib_means[f"{c_col}:{t_col}"].append(float(np.nanmean(rho_xy_samples)))
        lib_means[f"{t_col}:{c_col}"].append(float(np.nanmean(rho_yx_samples)))

    lib_means_df = pd.DataFrame(lib_means)

    stats_forward_df = pd.DataFrame(forward_stats) if includeData else None
    stats_reverse_df = pd.DataFrame(reverse_stats) if includeData else None

    if showPlot:
        ax = lib_means_df.plot(
            "LibSize",
            [f"{c_col}:{t_col}", f"{t_col}:{c_col}"],
            title=f"E = {E}",
            linewidth=3,
        )
        ax.set(xlabel="Library Size", ylabel="CCM ρ")
        axhline(y=0, linewidth=1)
        show()

    if returnObject:
        return CausalizedCCMRun(
            libMeans=lib_means_df,
            PredictStats1=stats_forward_df,
            PredictStats2=stats_reverse_df,
        )

    if includeData:
        return {
            "LibMeans": lib_means_df,
            "PredictStats1": stats_forward_df,
            "PredictStats2": stats_reverse_df,
        }

    return lib_means_df


def conditional(
    dataFrame = None,
    tau: int = 1,
    e_dim: int = 3,
    pairs: Sequence[Sequence[int]] | None = None,
    num_skip: int = 10,
    exclusion_radius: int = 0,
    exclusionRadius: int | None = None,
    causal: bool = True,
):
    """Wrapper around :func:`conditional_ccm` using pandas column lookup."""

    if dataFrame is None:
        raise RuntimeError("conditional(): dataFrame must be provided")
    if pairs is None:
        raise RuntimeError("conditional(): pairs must be provided")

    if exclusionRadius is not None and exclusion_radius != 0:
        raise RuntimeError("conditional(): specify only one of exclusion_radius or exclusionRadius")

    if exclusionRadius is not None:
        exclusion_radius = exclusionRadius

    if not isinstance(dataFrame, pd.DataFrame):
        dataFrame = pd.DataFrame(dataFrame)

    matrix = dataFrame.to_numpy(dtype=float)
    return conditional_ccm(
        matrix,
        tau=tau,
        e_dim=e_dim,
        pairs=pairs,
        num_skip=num_skip,
        exclusion_radius=exclusion_radius,
        causal=causal,
    )


def _first(value, name: str) -> str:
    if isinstance(value, str):
        tokens = value.split()
        if not tokens:
            raise RuntimeError(f"CCM(): {name} must contain at least one column name")
        return tokens[0]

    if IsIterable(value):
        value_list = list(value)
        if not value_list:
            raise RuntimeError(f"CCM(): {name} must contain at least one column name")
        return str(value_list[0])

    if value:
        return str(value)

    raise RuntimeError(f"CCM(): {name} must be specified")


def _parse_lib_sizes(libSizes, data_length: int, E: int, tau: int, tp: int) -> List[int]:
    if not libSizes:
        lib_sizes = [data_length]
    elif isinstance(libSizes, str):
        parts = [int(x) for x in libSizes.split()]
        lib_sizes = _expand(parts)
    elif IsIterable(libSizes):
        lib_sizes = _expand([int(x) for x in libSizes])
    else:
        lib_sizes = [int(libSizes)]

    required = max(1 + (E - 1) * tau, E + 2)
    abs_tp = abs(tp)
    tp_adjusted_requirement = (E - 1) * tau + abs_tp + 1
    minimum_size = max(required, tp_adjusted_requirement)

    lib_sizes = sorted(set(lib_sizes))
    for size in lib_sizes:
        if size < minimum_size:
            raise RuntimeError(
                f"CCM(): library size {size} violates minimum {minimum_size} for E={E}, tau={tau}, Tp={tp}"
            )
        if size > data_length:
            raise RuntimeError(
                f"CCM(): library size {size} exceeds data length {data_length}"
            )

    return lib_sizes


def _expand(values: List[int]) -> List[int]:
    if len(values) == 3 and values[2] > 0 and values[0] < values[1]:
        start, stop, step = values
        return list(range(start, stop + 1, step))
    return values


__all__ = ["CCM", "CausalizedCCMRun", "conditional"]
