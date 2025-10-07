#!/usr/bin/env python
"""Benchmark runtime scaling for pyC4M CCM variants.

Run from the repository root:

    python benchmarks/runtime_benchmarks.py

The script sweeps a grid of library sizes and sample counts for four CCM
variants (base, causalized, conditional, causalized+conditional), records
runtime measurements, and generates summary plots suitable for the README.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure the editable source tree is importable without installation.
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from pyc4m import CCM  # noqa: E402


DEFAULT_LIB_SIZES = [100, 300, 600, 900, 1200]
DEFAULT_SAMPLES = [1, 10, 25, 50]
DEFAULT_REPEATS = 3
DEFAULT_LENGTH = 1800
BASE_SEED = 20240519


@dataclass(frozen=True)
class BenchmarkVariant:
    """Descriptor for a CCM benchmark configuration."""

    name: str
    causal: bool
    conditional: Sequence[str] | None


VARIANTS: Sequence[BenchmarkVariant] = (
    BenchmarkVariant("Base CCM", causal=False, conditional=None),
    BenchmarkVariant("Causalized CCM", causal=True, conditional=None),
    BenchmarkVariant("Conditional CCM", causal=False, conditional=["z"]),
    BenchmarkVariant("Causalized Conditional CCM", causal=True, conditional=["z"]),
)


def parse_int_list(values: str, argument: str) -> List[int]:
    """Parse a comma-separated list of integers."""

    try:
        parsed = [int(value.strip()) for value in values.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{argument} must be a comma-separated list of integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError(f"{argument} must contain at least one integer")
    if any(value <= 0 for value in parsed):
        raise argparse.ArgumentTypeError(f"{argument} values must be positive")
    return parsed


def build_dataset(length: int, seed: int) -> pd.DataFrame:
    """Create a reproducible three-variable synthetic dataset for CCM."""

    rng = np.random.default_rng(seed)
    time_axis = np.linspace(0, 24 * np.pi, num=length)

    # Coupled oscillators with mild noise provide non-trivial embeddings.
    x = np.sin(time_axis) + 0.1 * rng.standard_normal(length)
    y = np.sin(time_axis + 0.6) + 0.1 * rng.standard_normal(length)
    z = 0.7 * np.sin(0.7 * time_axis + 1.2) + 0.1 * rng.standard_normal(length)

    return pd.DataFrame({"x": x, "y": y, "z": z})


def time_variant(
    data: pd.DataFrame,
    variant: BenchmarkVariant,
    lib_size: int,
    sample_count: int,
    repeats: int,
    start_seed: int,
) -> List[float]:
    """Measure the runtime of a CCM configuration."""

    timings: List[float] = []
    for repeat in range(repeats):
        seed = start_seed + repeat
        begin = time.perf_counter()
        CCM(
            dataFrame=data,
            columns="x",
            target="y",
            libSizes=[lib_size],
            sample=sample_count,
            E=3,
            tau=-1,
            Tp=0,
            exclusionRadius=0,
            seed=seed,
            causal=variant.causal,
            conditional=None if variant.conditional is None else list(variant.conditional),
            num_skip=5,
        )
        elapsed = time.perf_counter() - begin
        timings.append(elapsed)
    return timings


def summarise_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean runtime and variability per configuration."""

    grouped = (
        df.groupby(["variant", "lib_size", "sample"], as_index=False)
        .agg(
            mean_runtime=("elapsed", "mean"),
            std_runtime=("elapsed", "std"),
        )
        .sort_values(["variant", "lib_size", "sample"])
    )
    grouped["std_runtime"].fillna(0.0, inplace=True)
    return grouped


def plot_scaling(
    summary: pd.DataFrame,
    lib_sizes: Sequence[int],
    samples: Sequence[int],
    output_path: Path,
) -> None:
    """Create runtime scaling plots for library and sample sizes."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    median_lib_idx = len(lib_sizes) // 2
    median_sample_idx = len(samples) // 2
    lib_focus_sample = samples[median_sample_idx]
    sample_focus_lib = lib_sizes[median_lib_idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Runtime vs library size.
    subset_lib = summary[summary["sample"] == lib_focus_sample]
    for variant in subset_lib["variant"].unique():
        variant_data = subset_lib[subset_lib["variant"] == variant].sort_values("lib_size")
        axes[0].plot(
            variant_data["lib_size"],
            variant_data["mean_runtime"],
            marker="o",
            label=variant,
        )
    axes[0].set_title(f"Runtime vs library size (sample={lib_focus_sample})")
    axes[0].set_xlabel("Library size")
    axes[0].set_ylabel("Mean runtime [s]")
    axes[0].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    # Runtime vs sample count.
    subset_sample = summary[summary["lib_size"] == sample_focus_lib]
    for variant in subset_sample["variant"].unique():
        variant_data = subset_sample[subset_sample["variant"] == variant].sort_values("sample")
        axes[1].plot(
            variant_data["sample"],
            variant_data["mean_runtime"],
            marker="o",
            label=variant,
        )
    axes[1].set_title(f"Runtime vs sample count (lib size={sample_focus_lib})")
    axes[1].set_xlabel("Sample count")
    axes[1].grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def run_benchmarks(
    lib_sizes: Sequence[int],
    samples: Sequence[int],
    repeats: int,
    length: int,
    csv_path: Path,
    plot_path: Path,
) -> None:
    """Execute the benchmark suite and write artefacts to disk."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    data = build_dataset(length=length, seed=BASE_SEED)

    records = []
    for lib_idx, lib_size in enumerate(lib_sizes):
        for sample_idx, sample_count in enumerate(samples):
            for variant_idx, variant in enumerate(VARIANTS):
                seed_offset = BASE_SEED + 1000 * lib_idx + 100 * sample_idx + 10 * variant_idx
                timings = time_variant(
                    data=data,
                    variant=variant,
                    lib_size=lib_size,
                    sample_count=sample_count,
                    repeats=repeats,
                    start_seed=seed_offset,
                )
                for repeat_idx, elapsed in enumerate(timings):
                    records.append(
                        {
                            "variant": variant.name,
                            "lib_size": lib_size,
                            "sample": sample_count,
                            "repeat": repeat_idx,
                            "elapsed": elapsed,
                        }
                    )

    raw_df = pd.DataFrame.from_records(records)
    raw_df.to_csv(csv_path, index=False)

    summary = summarise_results(raw_df)
    plot_scaling(summary, lib_sizes=lib_sizes, samples=samples, output_path=plot_path)

    print(f"Wrote raw timings to {csv_path}")
    print(f"Wrote runtime plot to {plot_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """CLI argument parsing."""

    parser = argparse.ArgumentParser(description="Benchmark pyC4M CCM runtime scaling.")
    parser.add_argument(
        "--lib-sizes",
        type=lambda string: parse_int_list(string, "lib-sizes"),
        default=DEFAULT_LIB_SIZES,
        help=f"Comma-separated library sizes to benchmark (default: {','.join(map(str, DEFAULT_LIB_SIZES))})",
    )
    parser.add_argument(
        "--samples",
        type=lambda string: parse_int_list(string, "samples"),
        default=DEFAULT_SAMPLES,
        help=f"Comma-separated sample counts to benchmark (default: {','.join(map(str, DEFAULT_SAMPLES))})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of timing repeats per configuration (default: {DEFAULT_REPEATS})",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=DEFAULT_LENGTH,
        help=f"Length of the synthetic time series (default: {DEFAULT_LENGTH})",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("benchmarks") / "runtime_results.csv",
        help="Destination CSV for raw benchmark timings.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("docs") / "img" / "ccm_runtime_scaling.png",
        help="Destination for the runtime scaling plot.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    run_benchmarks(
        lib_sizes=args.lib_sizes,
        samples=args.samples,
        repeats=args.repeats,
        length=args.length,
        csv_path=args.csv_path,
        plot_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
