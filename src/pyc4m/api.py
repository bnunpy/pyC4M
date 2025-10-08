"""pyC4M public API: CCM wrappers compatible with pyEDM naming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.random import default_rng
from matplotlib.pyplot import axhline, show

from pyEDM.AuxFunc import ComputeError, IsIterable

from .cccm import VariableGeometry, causalized_ccm, prepare_variable_geometry
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
    E: int = 3,
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
    causal: bool = False,
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

    if validLib:
        raise NotImplementedError("CCM(): validLib is not supported")

    manifold_source = None
    manifold_target = None
    manifolds_map = None
    tails_map = None
    geometry_map: Dict[int, VariableGeometry] | None = None

    is_embedded = bool(embedded)

    try:
        tau_value = -1 if tau is None else int(tau)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("CCM(): tau must be an integer") from exc

    if tau_value == 0:
        raise RuntimeError("CCM(): tau must be non-zero")
    if not is_embedded and E <= 0:
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

    if libSizes is None or libSizes == "":
        lib_sizes_arg = [len(df)]
    else:
        lib_sizes_arg = libSizes

    if is_embedded:
        source_columns = _as_column_list(columns, "columns")
        target_columns = _as_column_list(target, "target")

        if E <= 0:
            E = len(source_columns)
        if len(source_columns) != E:
            raise RuntimeError(
                f"CCM(): number of source columns {len(source_columns)} does not match embedding dimension E={E}"
            )
        if len(target_columns) != E:
            raise RuntimeError(
                f"CCM(): number of target columns {len(target_columns)} does not match embedding dimension E={E}"
            )

        for column_name in source_columns + target_columns:
            if column_name not in df.columns:
                raise RuntimeError(f"CCM(): column '{column_name}' not found in dataFrame")

        source_col = source_columns[0]
        target_col = target_columns[0]
        manifold_source = df[source_columns].to_numpy(dtype=float)
        manifold_target = df[target_columns].to_numpy(dtype=float)
        series_x = df[source_columns[0]].to_numpy(dtype=float)
        series_y = df[target_columns[0]].to_numpy(dtype=float)
        embed_gap = (E - 1) * abs(tau)
        raw_length = len(df) + embed_gap
        parsed_lib_sizes = _parse_lib_sizes(lib_sizes_arg, raw_length, E, tau, tp_value)
        available_vectors = len(series_x) - abs(tp_value)
        if available_vectors <= 1:
            raise RuntimeError("CCM(): insufficient data for the requested embedding")

        total_vectors = np.arange(len(series_x), dtype=int)
    else:
        source_col = _first(columns, "columns")
        target_col = _first(target, "target")

        for column_name in [source_col, target_col]:
            if column_name not in df.columns:
                raise RuntimeError(f"CCM(): column '{column_name}' not found in dataFrame")
        manifold_source = None
        manifold_target = None

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
        if is_embedded:
            conditional_sets: List[List[str]] = []
            if isinstance(conditional, str) or isinstance(conditional, pd.Series):
                conditional_sets.append(_as_column_list(conditional, "conditional"))
            elif IsIterable(conditional):
                for idx, entry in enumerate(conditional):
                    conditional_sets.append(_as_column_list(entry, f"conditional[{idx}]"))
            else:
                raise RuntimeError("CCM(): conditional must be a string or sequence of strings")

            for cols in conditional_sets:
                if len(cols) != E:
                    raise RuntimeError(
                        f"CCM(): conditional embedding must supply {E} columns per variable"
                    )

            variable_column_sets = [source_columns, target_columns] + conditional_sets
            for col_set in variable_column_sets:
                for column_name in col_set:
                    if column_name not in df.columns:
                        raise RuntimeError(f"CCM(): column '{column_name}' not found in dataFrame")

            conditional_cols = [cols[0] for cols in conditional_sets]
            all_columns = [source_columns[0], target_columns[0]] + conditional_cols
            subset = np.column_stack(
                [df[col_set[0]].to_numpy(dtype=float) for col_set in variable_column_sets]
            )

            manifolds_map = {
                idx: df[col_set].to_numpy(dtype=float)
                for idx, col_set in enumerate(variable_column_sets)
            }
            tails_map = {
                idx: df[col_set[0]].to_numpy(dtype=float)
                for idx, col_set in enumerate(variable_column_sets)
            }
            geometry_map = {
                idx: prepare_variable_geometry(
                    tau=tau,
                    e_dim=E,
                    manifold=manifolds_map[idx],
                    tail=tails_map[idx],
                )
                for idx in range(len(variable_column_sets))
            }

            embed_gap = (E - 1) * abs(tau)
            latent_length = len(subset)
            raw_length = latent_length + embed_gap
            parsed_lib_sizes = _parse_lib_sizes(lib_sizes_arg, raw_length, E, tau, 0)
        else:
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
            embed_gap = (E - 1) * abs(tau)
            latent_length = len(subset) - embed_gap
            if latent_length <= num_skip:
                raise RuntimeError(
                    "CCM(): insufficient data for the requested embedding and num_skip"
                )

            parsed_lib_sizes = _parse_lib_sizes(lib_sizes_arg, len(subset), E, tau, 0)
            geometry_map = {
                idx: prepare_variable_geometry(
                    tau=tau,
                    e_dim=E,
                    series=subset[:, idx],
                )
                for idx in range(len(all_columns))
            }

        total_vectors = np.arange(latent_length, dtype=int)
        rng = default_rng(seed)

        metric_keys = [
            "x_on_y",
            "y_on_x",
            "var_x_with_cross",
            "var_x_conditionals",
            "var_y_with_cross",
            "var_y_conditionals",
        ]

        base_settings: Dict[str, object] | None = None
        pair_summaries: Dict[int, Dict[Tuple[int, int], Dict[str, List[float]]]] = {}
        base_corr_stats: Dict[int, Dict[str, np.ndarray]] = {}

        cached_reconstructions = {}

        for lib_size in parsed_lib_sizes:
            lib_vectors = lib_size - embed_gap
            if lib_vectors <= 1:
                raise RuntimeError(
                    f"CCM(): library size {lib_size} is too small for embedding E={E}, tau={tau}"
                )
            if lib_vectors > latent_length:
                raise RuntimeError(
                    f"CCM(): library size {lib_size} exceeds available length {latent_length + embed_gap}"
                )

            for sample_index in range(sample_count):
                if lib_vectors == latent_length and sample_index == 0:
                    library_idx = None
                else:
                    if sample_index == 0:
                        library_idx = total_vectors[:lib_vectors]
                    else:
                        library_idx = np.sort(
                            rng.choice(total_vectors, size=lib_vectors, replace=False)
                        )

                cache_key = (lib_size, ("__full__",) if library_idx is None else tuple(library_idx.tolist()))
                if cache_key in cached_reconstructions:
                    cond_result = cached_reconstructions[cache_key]
                else:
                    cond_result = conditional_ccm(
                        subset,
                        tau=tau,
                        e_dim=E,
                        pairs=[(0, 1)],
                        num_skip=num_skip,
                        exclusion_radius=exclusion_radius_value,
                        causal=causal_flag,
                        library_indices=None if library_idx is None else library_idx,
                        embedded=is_embedded,
                        manifolds=manifolds_map if is_embedded else None,
                        tails=tails_map if is_embedded else None,
                        geometries=geometry_map,
                    )
                    cached_reconstructions[cache_key] = cond_result

                if base_settings is None:
                    base_settings = dict(cond_result.settings)
                    base_settings.update(
                        {
                            "source": source_col,
                            "target": target_col,
                            "conditional": conditional_cols,
                            "columns": all_columns,
                        }
                    )

                pair_store = pair_summaries.setdefault(lib_size, {})
                for (src_idx, tgt_idx), pair_result in cond_result.pair_results.items():
                    metrics = pair_store.setdefault(
                        (src_idx, tgt_idx),
                        {key: [] for key in metric_keys},
                    )
                    metrics["x_on_y"].append(pair_result.x_on_y)
                    metrics["y_on_x"].append(pair_result.y_on_x)
                    metrics["var_x_with_cross"].append(
                        pair_result.diagnostics["var_x_with_cross"]
                    )
                    metrics["var_x_conditionals"].append(
                        pair_result.diagnostics["var_x_conditionals"]
                    )
                    metrics["var_y_with_cross"].append(
                        pair_result.diagnostics["var_y_with_cross"]
                    )
                    metrics["var_y_conditionals"].append(
                        pair_result.diagnostics["var_y_conditionals"]
                    )

                stats = base_corr_stats.setdefault(
                    lib_size,
                    {
                        "sum": np.zeros_like(cond_result.base_correlations, dtype=float),
                        "sumsq": np.zeros_like(cond_result.base_correlations, dtype=float),
                        "count": 0,
                    },
                )
                stats["sum"] += cond_result.base_correlations
                stats["sumsq"] += cond_result.base_correlations**2
                stats["count"] += 1

        if not pair_summaries:
            raise RuntimeError("CCM(): conditional results are empty")

        effect_lists: Dict[int, Dict[str, List[float]]] = {}
        effect_vars: Dict[int, Dict[str, float]] = {}
        sample_counts: Dict[int, Dict[str, int]] = {}
        diagnostics_lists: Dict[int, Dict[str, List[float]]] = {}
        diagnostics_mean: Dict[int, Dict[str, float]] = {}
        diagnostics_var: Dict[int, Dict[str, float]] = {}
        label_order: List[str] = []

        def register_label(label: str) -> None:
            if label not in label_order:
                label_order.append(label)

        for lib_size, pair_map in pair_summaries.items():
            effect_lists.setdefault(lib_size, {})
            sample_counts.setdefault(lib_size, {})
            diagnostics_lists.setdefault(lib_size, {})

            for pair, metrics in pair_map.items():
                src_idx, tgt_idx = pair
                forward_label = f"{all_columns[src_idx]}:{all_columns[tgt_idx]}"
                reverse_label = f"{all_columns[tgt_idx]}:{all_columns[src_idx]}"
                register_label(forward_label)
                register_label(reverse_label)

                effect_lists[lib_size][forward_label] = list(metrics["x_on_y"])
                effect_lists[lib_size][reverse_label] = list(metrics["y_on_x"])

                sample_counts[lib_size][forward_label] = len(metrics["x_on_y"])
                sample_counts[lib_size][reverse_label] = len(metrics["y_on_x"])

                diagnostics_lists[lib_size][
                    f"{forward_label}:var_with_cross"
                ] = list(metrics["var_y_with_cross"])
                diagnostics_lists[lib_size][
                    f"{forward_label}:var_conditionals"
                ] = list(metrics["var_y_conditionals"])
                diagnostics_lists[lib_size][
                    f"{reverse_label}:var_with_cross"
                ] = list(metrics["var_x_with_cross"])
                diagnostics_lists[lib_size][
                    f"{reverse_label}:var_conditionals"
                ] = list(metrics["var_x_conditionals"])

        rows: List[Dict[str, object]] = []
        for lib_size in sorted(effect_lists.keys()):
            row = {"LibSize": lib_size}
            effect_vars[lib_size] = {}
            diagnostics_mean[lib_size] = {}
            diagnostics_var[lib_size] = {}

            for label in label_order:
                values = np.array(effect_lists[lib_size].get(label, []), dtype=float)
                if values.size == 0:
                    continue
                row[label] = float(np.nanmean(values))
                effect_vars[lib_size][label] = float(np.nanvar(values))

            for diag_key, values in diagnostics_lists[lib_size].items():
                arr = np.array(values, dtype=float)
                diagnostics_mean[lib_size][diag_key] = float(np.nanmean(arr))
                diagnostics_var[lib_size][diag_key] = float(np.nanvar(arr))

            rows.append(row)

        conditional_df = pd.DataFrame(rows, columns=["LibSize", *label_order])

        if base_settings is None:
            base_settings = {
                "tau": tau,
                "e_dim": E,
                "num_skip": num_skip,
                "exclusion_radius": exclusion_radius_value,
                "causal": causal_flag,
                "source": source_col,
                "target": target_col,
                "conditional": conditional_cols,
                "columns": all_columns,
            }

        base_settings.update({"LibSizes": parsed_lib_sizes, "Sample": sample_count})

        aggregated_base_correlations: Dict[int, Dict[str, object]] = {}
        for lib_size, stats in base_corr_stats.items():
            count = stats["count"]
            mean = stats["sum"] / count
            var = stats["sumsq"] / count - mean**2
            var = np.maximum(var, 0.0)
            aggregated_base_correlations[lib_size] = {
                "mean": mean,
                "var": var,
                "count": count,
            }

        conditional_df.attrs["Settings"] = base_settings
        conditional_df.attrs["BaseCorrelations"] = aggregated_base_correlations
        conditional_df.attrs["SampleCount"] = sample_counts
        conditional_df.attrs["Variance"] = effect_vars
        conditional_df.attrs["DiagnosticsMean"] = diagnostics_mean
        conditional_df.attrs["DiagnosticsVar"] = diagnostics_var

        return conditional_df

    if not is_embedded:
        series_x = df[source_col].to_numpy(dtype=float)
        series_y = df[target_col].to_numpy(dtype=float)

        parsed_lib_sizes = _parse_lib_sizes(lib_sizes_arg, len(df), E, tau, tp_value)

        embed_gap = (E - 1) * abs(tau)
        geometry_source = prepare_variable_geometry(
            tau=tau,
            e_dim=E,
            series=series_x,
        )
        geometry_target = prepare_variable_geometry(
            tau=tau,
            e_dim=E,
            series=series_y,
        )
    else:
        geometry_source = prepare_variable_geometry(
            tau=tau,
            e_dim=E,
            manifold=manifold_source,
            tail=series_x,
        )
        geometry_target = prepare_variable_geometry(
            tau=tau,
            e_dim=E,
            manifold=manifold_target,
            tail=series_y,
        )

    n_vectors = geometry_source.manifold.shape[0]
    available_vectors = n_vectors - abs(tp_value)
    if available_vectors <= 1:
        raise RuntimeError("CCM(): insufficient data for the requested embedding")

    if is_embedded:
        total_vectors = np.arange(n_vectors, dtype=int)
    else:
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

    cached_results: Dict[Tuple[int, Tuple[int, ...]], Tuple[float, float, np.ndarray, np.ndarray]] = {}

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
                if lib_vectors == available_vectors:
                    library_idx = None
                else:
                    library_idx = total_vectors[:lib_vectors]
            else:
                library_idx = np.sort(rng.choice(total_vectors, size=lib_vectors, replace=False))

            skip = min(skip_base, max(1, lib_vectors - 1))

            if library_idx is None:
                cache_key = (lib_size, ("__full__",))
            else:
                cache_key = (lib_size, tuple(library_idx.tolist()))

            if cache_key in cached_results:
                corr_y, corr_x, est_y, est_x = cached_results[cache_key]
            else:
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
                    geometry_x=geometry_source,
                    geometry_y=geometry_target,
                )
                corr_y = result.correlation_y
                corr_x = result.correlation_x
                est_y = result.y_estimates
                est_x = result.x_estimates
                cached_results[cache_key] = (corr_y, corr_x, est_y, est_x)

            rho_xy_samples.append(corr_y)
            rho_yx_samples.append(corr_x)

            if includeData:
                start = skip - 1
                tail_start = embed_gap if tau < 0 else 0
                latent_length = len(est_y)
                tail_y = series_y[tail_start : tail_start + latent_length]
                tail_x = series_x[tail_start : tail_start + latent_length]

                obs_y = tail_y[start:]
                pred_y = est_y[start:]
                err_y = ComputeError(obs_y, pred_y, digits=6)
                err_y["LibSize"] = lib_size
                err_y["Sample"] = sample_index

                obs_x = tail_x[start:]
                pred_x = est_x[start:]
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
    causal: bool = False,
    embedded: bool = False,
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
        embedded=embedded,
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


def _as_column_list(value, name: str) -> List[str]:
    if isinstance(value, str):
        tokens = value.split()
        if not tokens:
            raise RuntimeError(f"CCM(): {name} must contain at least one column name")
        return tokens

    if IsIterable(value):
        value_list = [str(item) for item in value]
        if not value_list:
            raise RuntimeError(f"CCM(): {name} must contain at least one column name")
        return value_list

    if value:
        return [str(value)]

    raise RuntimeError(f"CCM(): {name} must be specified")


__all__ = ["CCM", "CausalizedCCMRun", "conditional"]
