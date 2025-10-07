# pyC4M

pyC4M extends [pyEDM](https://github.com/SugiharaLab/pyEDM) with causalized convergent cross mapping (cCCM) and the multivariate conditional variant from the `CCM-Implement` MATLAB toolbox. Its public API mirrors `pyEDM.CCM()`, allowing existing code to switch to the causalised implementation with only an import change.

Key features:

- pyEDM-compatible CCM wrapper that defaults to the standard (non-causal) mode (`causal=False`); enable causalised dynamics by setting `causal=True` with a negative `tau`.
- Multivariate conditional CCM built on the same projections; invoke it either through the dedicated helper (`conditional_ccm`) or by passing `conditional=...` to `CCM`, with full support for `libSizes`, `sample`, and `seed`, and per-library means/variances aggregated across resamples. When you pass `embedded=True`, supply the delay-coordinate columns for each variable (e.g. `columns=['x0','x1']`, `conditional=[["z0","z1"]]`) so the precomputed manifolds are reused directly.
- Familiar keywords (`Tp`, `libSizes`, `exclusionRadius`, `includeData`, `sample`, `seed`) are honoured, the default embedding dimension is `E=3`, and you can bypass internal embedding with `embedded=True` when supplying pre-built delay blocks.
- Random library resampling automatically averages correlations and conditional ratios across replicates, mirroring pyEDM.

Install in editable mode for development:

```bash
pip install -e .
```

## Layout

```
pyC4M/
├── src/pyc4m/
│   ├── cccm.py          # causalized CCM translation
│   ├── conditional.py   # conditional CCM using causalized projections
│   ├── api.py           # pyEDM-compatible wrappers (CCM, conditional)
│   └── __init__.py
└── tests/               # smoke tests exercising the APIs
```

## Usage

After installation, import and use the APIs as shown below.

```python
import numpy as np
from pyc4m import CCM, causalized_ccm, conditional_ccm

# bivariate causalized CCM (low-level array API)
x = np.sin(np.linspace(0, 6 * np.pi, 500))
y = np.roll(x, 3)
ccm_result = causalized_ccm(x, y, tau=-1, e_dim=3)
print(ccm_result.correlation_x, ccm_result.correlation_y)

# pyEDM-compatible causalized CCM (drop-in replacement)
import pandas as pd

frame = pd.DataFrame({"x": x, "y": y})
lib_means = CCM(
    dataFrame=frame,
    columns="x",
    target="y",
    libSizes=[150, 250],
    E=3,
    tau=-1,
    Tp=1,
    exclusionRadius=12,
    causal=True,
)

# standard (non-causal) CCM for comparison
lib_means_noncausal = CCM(
    dataFrame=frame,
    columns="x",
    target="y",
    libSizes=[150, 250],
    E=3,
    tau=1,
    causal=False,
)
print(lib_means)

# conditional CCM on a 3-variable system
z = np.cos(np.linspace(0, 4 * np.pi, 500))
data = np.column_stack([x, y, z])
cond_result = conditional_ccm(data, tau=-1, e_dim=3, pairs=[(0, 1)])
print(cond_result.pair_results[(0, 1)].x_on_y)

# or via the CCM wrapper
cond_wrapper = CCM(
    dataFrame=pd.DataFrame({"x": x, "y": y, "z": z}),
    columns="x",
    target="y",
    conditional="z",
    tau=-1,
    E=3,
    num_skip=5,
)
print(cond_wrapper["x:y"].iloc[0])
print(cond_wrapper[["LibSize", "x:y", "y:x"]].head())

settings = cond_wrapper.attrs.get("Settings", {})
print(settings.get("conditional"))
print(cond_wrapper.attrs.get("SampleCount"))
print(cond_wrapper.attrs.get("Variance"))
```

The DataFrame returned by `CCM(..., conditional=...)` stores metadata in `df.attrs["Settings"]` and `df.attrs["BaseCorrelations"]`.

Column meanings:

- `LibSize`: library size corresponding to the aggregated statistics.
- `x:y`, `y:x`: conditional CCM causality ratios (mean over resamples) for each direction, using the same column naming as `pyEDM.CCM`.

Additional per-library details are stored in `df.attrs`:
- `SampleCount[lib_size]["x:y"]` / `SampleCount[lib_size]["y:x"]` give the number of resamples aggregated.
- `Variance[lib_size]["x:y"]` / `Variance[lib_size]["y:x"]` hold the variance across resamples.
- `DiagnosticsMean` / `DiagnosticsVar` contain residual variances with and without the cross-map for each direction (keys such as `"x:y:var_with_cross"`).


Monte-Carlo style diagnostics matching the MATLAB reference (`var_*` entries) are exposed via the `diagnostics` dictionaries returned for every pair.

## Tests

Run the included checks with:

```bash
python -m unittest discover -s tests -q
```

The tests cover basic regression scenarios to guard against regressions in the numerical translation.

## Compatibility Notes

The goal is API parity with `pyEDM.CCM()` while using the causalized dynamics from `CCM-Implement`. A few differences remain:

- `embedded=True` and `validLib` arguments are not yet supported; inputs must be raw time-series columns rather than pre-embedded blocks.
- `knn` and `noTime` keywords are currently ignored. Causalized CCM still uses `E+1` nearest neighbours without adaptive weighting; temporal exclusion via `exclusionRadius` is implemented.
- Causalised behaviour requires `causal=True` with negative `tau`; the default settings (`causal=False`, `tau=-1`) reproduce standard CCM while sharing the causalized internals. `embedded=True` is supported for both base and conditional CCM, letting you memoize delay embeddings manually when you provide the coordinate blocks yourself.
- Multiprocessing (`pyc4m.CCM(..., returnObject=True)` -> `.Project`) is run sequentially, so large batches may take longer than the native C++ routines.
- Conditional CCM replicates the MATLAB variance-based diagnostic, but additional options such as variable-specific `N0` or alternate metrics from the MATLAB scripts are not exposed.
- Random sampling always averages correlations across replicates; visualisation helpers present in `pyEDM` (e.g., `showPlot=True` plots) are generated via pandas/matplotlib only.

Planned enhancements include support for embedded input, time-aware exclusion radii, and alternative neighbour metrics to match the remaining pyEDM features.
