# pyC4M

pyC4M extends [pyEDM](https://github.com/SugiharaLab/pyEDM) with causalized convergent cross mapping (cCCM) and the multivariate conditional variant from the `CCM-Implement` MATLAB toolbox. Its public API mirrors `pyEDM.CCM()`, allowing existing code to switch to the causalised implementation with only an import change.

Key features:

- Causalised CCM with an optional standard (non-causal) mode via `causal=False`.
- Multivariate conditional CCM built on the same projections; invoke it either through the dedicated helper (`conditional_ccm`) or by passing `conditional=...` to `CCM`.
- Support for familiar keywords (`Tp`, `lib_sizes`, `exclusion_radius`, `include_data`, `sample`, `seed`). Default behaviour matches the causalised MATLAB implementation while camelCase aliases remain supported for compatibility.
- Random library resampling that automatically averages correlations across replicates, mirroring pyEDM.

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
ccm_result = causalized_ccm(x, y, tau=1, e_dim=3)
print(ccm_result.correlation_x, ccm_result.correlation_y)

# pyEDM-compatible causalized CCM (drop-in replacement)
import pandas as pd

frame = pd.DataFrame({"x": x, "y": y})
lib_means = CCM(
    dataFrame=frame,
    source="x",
    target="y",
    lib_sizes=[150, 250],
    E=3,
    tau=1,
    Tp=1,
    exclusion_radius=12,
    causal=True,
)

# standard (non-causal) CCM for comparison
lib_means_noncausal = CCM(
    dataFrame=frame,
    source="x",
    target="y",
    lib_sizes=[150, 250],
    E=3,
    tau=1,
    causal=False,
)
print(lib_means)

# conditional CCM on a 3-variable system
z = np.cos(np.linspace(0, 4 * np.pi, 500))
data = np.column_stack([x, y, z])
cond_result = conditional_ccm(data, tau=1, e_dim=3, pairs=[(0, 1)])
print(cond_result.pair_results[(0, 1)].x_on_y)

# or via the CCM wrapper
cond_wrapper = CCM(
    dataFrame=pd.DataFrame({"x": x, "y": y, "z": z}),
    source="x",
    target="y",
    conditional="z",
    tau=1,
    E=3,
    num_skip=5,
)
print(cond_wrapper.pair_results[(0, 1)].x_on_y)
```

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
- `knn` and `noTime` keywords are currently ignored. Causalized CCM still uses `E+1` nearest neighbours without adaptive weighting; temporal exclusion via `exclusion_radius` is implemented.
- Set `causal=False` in `CCM`/`conditional` to reproduce standard (non-causal) CCM alongside the default causalised mode.
- Prefer snake_case parameters (e.g., `exclusion_radius`); camelCase aliases remain available for compatibility but will be removed in a future release.
- Multiprocessing (`pyc4m.CCM(..., returnObject=True)` -> `.Project`) is run sequentially, so large batches may take longer than the native C++ routines.
- Conditional CCM replicates the MATLAB variance-based diagnostic, but additional options such as variable-specific `N0` or alternate metrics from the MATLAB scripts are not exposed.
- Random sampling always averages correlations across replicates; visualisation helpers present in `pyEDM` (e.g., `showPlot=True` plots) are generated via pandas/matplotlib only.

Planned enhancements include support for embedded input, time-aware exclusion radii, and alternative neighbour metrics to match the remaining pyEDM features.
