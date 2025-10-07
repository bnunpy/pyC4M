"""pyC4M public API: CCM wrappers compatible with pyEDM naming."""

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
    """Object returned when ``returnObject=True`` mimicking pyEDM layout."""

    libMeans: pd.DataFrame
    PredictStats1: pd.DataFrame | None
    PredictStats2: pd.DataFrame | None
    name: str = "CausalizedCCM"


def CCM(
    dataFrame = None,
    columns: str | Sequence[str] = "",
    target: str | Sequence[str] = "",
    libSizes: Iterable[int] | str | None = None,
    sample: int = 0,
    E: int = 0,
    Tp: int = 0,
    knn: int = 0,
    tau: int = -1,
    exclusionRadius: int | None = None,
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
    conditional: str | Sequence[str] | None = None,
    **kwargs,
):
    """Convergent cross mapping with optional causalisation and conditioning."""

    if dataFrame is None:
        raise RuntimeError("CCM(): dataFrame must be provided")

    if isinstance(dataFrame, pd.DataFrame):
        df = dataFrame.copy()
    elif hasattr(dataFrame, "to_frame"):
        df = dataFrame.to_frame()
    else:
        raise RuntimeError("CCM(): dataFrame must be a pandas DataFrame")

    if embedded:
        raise NotImplementedError("CCM(): embedded=True is not yet supported")
    if validLib:
        raise NotImplementedError("CCM(): validLib is not supported")

    try:
        tau_value = -1 if tau is None else int(tau)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("CCM(): tau must be an integer") from exc

    if tau_value == 0:
        raise RuntimeError("CCM(): tau must be non-zero")
    if E <= 0:
        raise RuntimeError("CCM(): embedding dimension E must be positive")

    tau = tau_value

    try:
        tp_value = 0 if Tp is None else int(Tp)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("CCM(): Tp must be an integer") from exc

    if exclusionRadius is None:
        exclusion_radius_value = 0
    else:
        try:
            exclusion_radius_value = int(exclusionRadius)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("CCM(): exclusionRadius must be an integer") from exc
        if exclusion_radius_value < 0:
            raise RuntimeError("CCM(): exclusionRadius must be non-negative")

    num_skip = kwargs.pop("num_skip", 10)
    epsilon = kwargs.pop("epsilon", 1e-12)
    if kwargs:
        unexpected = ", ".join(kwargs.keys())
        raise TypeError(f"CCM(): unexpected keyword arguments: {unexpected}")

    source_col = _first(columns, "columns")
    target_col = _first(target, "target")

    for column_name in [source_col, target_col]:
        if column_name not in df.columns:
            raise RuntimeError(f"CCM(): column '{column_name}' not found in dataFrame")

    if libSizes is None or libSizes == "":
        lib_sizes_arg = [len(df)]
    else:
        lib_sizes_arg = libSizes

    sample_count = int(sample)
    if sample_count < 0:
        raise RuntimeError("CCM(): sample must be non-negative")
    if sample_count == 0:
        sample_count = 1

    try:
        causal_flag = bool(causal)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError("CCM(): causal must be boolean") from exc

    if causal_flag and tau >= 0:
        raise RuntimeError(
            "CCM(): causal projections require tau < 0 to embed into the past"
        )

    if conditional is not None:
        if isinstance(conditional, str):
            conditional_cols = [conditional]
        elif IsIterable(conditional):
            conditional_cols = [str(col) for col in conditional]
        else:
            raise RuntimeError("CCM(): conditional must be a string or sequence of strings")

        for column_name in conditional_cols:
            if column_name not in df.columns:
                raise RuntimeError(f"CCM(): conditional column '{column_name}' not found in dataFrame")

        all_columns = [source_col, target_col] + conditional_cols
        subset = df[all_columns].to_numpy(dtype=float)

        cond_result = conditional_ccm(
            subset,
            tau=tau,
            e_dim=E,
            pairs=[(0, 1)],
            num_skip=num_skip,
            exclusion_radius=exclusion_radius_value,
            causal=causal_flag,
        )

        cond_result.settings.update(
            {
                "source": source_col,
                "target": target_col,
                "conditional": conditional_cols,
                "columns": all_columns,
            }
        )

        records = []
        for (src_idx, tgt_idx), pair_result in cond_result.pair_results.items():
            records.append(
                {
                    "source": all_columns[src_idx],
                    "target": all_columns[tgt_idx],
                    "conditional": conditional_cols,
                    "x_on_y": pair_result.x_on_y,
                    "y_on_x": pair_result.y_on_x,
                    "var_x_with_cross": pair_result.diagnostics["var_x_with_cross"],
                    "var_x_conditionals": pair_result.diagnostics["var_x_conditionals"],
                    "var_y_with_cross": pair_result.diagnostics["var_y_with_cross"],
                    "var_y_conditionals": pair_result.diagnostics["var_y_conditionals"],
                }
            )

        conditional_df = pd.DataFrame(records)
        conditional_df.attrs["Settings"] = cond_result.settings
        conditional_df.attrs["BaseCorrelations"] = cond_result.base_correlations

        return conditional_df

    series_x = df[source_col].to_numpy(dtype=float)
    series_y = df[target_col].to_numpy(dtype=float)

    parsed_lib_sizes = _parse_lib_sizes(lib_sizes_arg, len(df), E, tau, tp_value)

    embed_gap = (E - 1) * abs(tau)
    available_vectors = len(series_x) - embed_gap - abs(tp_value)
    if available_vectors <= 1:
        raise RuntimeError("CCM(): insufficient data for the requested embedding")

    total_vectors = np.arange(available_vectors, dtype=int)

    if verbose:
        print(
            f"Causalized CCM evaluating libSizes={parsed_lib_sizes} with E={E}, tau={tau}, Tp={tp_value}"
        )

    lib_means: Dict[str, List[float]] = {
        "LibSize": [],
        f"{source_col}:{target_col}": [],
        f"{target_col}:{source_col}": [],
    }

    forward_stats: List[Dict[str, float]] = []
    reverse_stats: List[Dict[str, float]] = []

    rng = default_rng(seed)
    skip_base = max(1, num_skip)
    if skip_base >= available_vectors:
        skip_base = max(1, available_vectors - 1)

    for lib_size in parsed_lib_sizes:
        lib_vectors = lib_size - embed_gap - abs(tp_value)
        if lib_vectors <= 1:
            raise RuntimeError(
                f"CCM(): library size {lib_size} is too small for embedding E={E}, tau={tau}, Tp={tp_value}"
            )
        if lib_vectors > available_vectors:
            raise RuntimeError(
                f"CCM(): library size {lib_size} exceeds available length {available_vectors + embed_gap + abs(tp_value)}"
            )

        rho_xy_samples = []
        rho_yx_samples = []

        for sample_index in range(sample_count):
            if sample_index == 0:
                library_idx = total_vectors[:lib_vectors]
            else:
                library_idx = np.sort(rng.choice(total_vectors, size=lib_vectors, replace=False))

            skip = min(skip_base, max(1, lib_vectors - 1))

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
                causal=causal_flag,
            )

            rho_xy_samples.append(result.correlation_y)
            rho_yx_samples.append(result.correlation_x)

            if includeData:
                start = skip - 1
                tail_start = embed_gap if tau < 0 else 0
                latent_length = len(result.y_estimates)
                tail_y = series_y[tail_start : tail_start + latent_length]
                tail_x = series_x[tail_start : tail_start + latent_length]

                obs_y = tail_y[start:]
                pred_y = result.y_estimates[start:]
                err_y = ComputeError(obs_y, pred_y, digits=6)
                err_y["LibSize"] = lib_size
                err_y["Sample"] = sample_index

                obs_x = tail_x[start:]
                pred_x = result.x_estimates[start:]
                err_x = ComputeError(obs_x, pred_x, digits=6)
                err_x["LibSize"] = lib_size
                err_x["Sample"] = sample_index

                forward_stats.append(err_y)
                reverse_stats.append(err_x)

        lib_means["LibSize"].append(lib_size)
        lib_means[f"{source_col}:{target_col}"].append(float(np.nanmean(rho_xy_samples)))
        lib_means[f"{target_col}:{source_col}"].append(float(np.nanmean(rho_yx_samples)))

    lib_means_df = pd.DataFrame(lib_means)

    stats_forward_df = pd.DataFrame(forward_stats) if includeData else None
    stats_reverse_df = pd.DataFrame(reverse_stats) if includeData else None

    if showPlot:
        ax = lib_means_df.plot(
            "LibSize",
            [f"{source_col}:{target_col}", f"{target_col}:{source_col}"],
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
    exclusionRadius: int = 0,
    causal: bool = True,
):
    """Wrapper around :func:`conditional_ccm` using pandas column lookup."""

    if dataFrame is None:
        raise RuntimeError("conditional(): dataFrame must be provided")
    if pairs is None:
        raise RuntimeError("conditional(): pairs must be provided")

    if not isinstance(dataFrame, pd.DataFrame):
        dataFrame = pd.DataFrame(dataFrame)

    matrix = dataFrame.to_numpy(dtype=float)
    return conditional_ccm(
        matrix,
        tau=tau,
        e_dim=e_dim,
        pairs=pairs,
        num_skip=num_skip,
        exclusion_radius=exclusionRadius,
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


def _parse_lib_sizes(lib_sizes, data_length: int, E: int, tau: int, tp: int) -> List[int]:
    if not lib_sizes:
        lib_sizes = [data_length]
    elif isinstance(lib_sizes, str):
        parts = [int(x) for x in lib_sizes.split()]
        lib_sizes = _expand(parts)
    elif IsIterable(lib_sizes):
        lib_sizes = _expand([int(x) for x in lib_sizes])
    else:
        lib_sizes = [int(lib_sizes)]

    embed_gap = (E - 1) * abs(tau)
    abs_tp = abs(tp)
    minimum_size = embed_gap + abs_tp + 2

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
