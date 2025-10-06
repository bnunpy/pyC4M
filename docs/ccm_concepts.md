# Causalized CCM and Conditional CCM

This note reviews the ideas behind *causalized convergent cross mapping* (cCCM) and the multivariate conditional variant we expose through `pyc4m`. It contrasts them with the standard CCM implementation popularised by Sugihara et al. and explains how the concepts are realised in the translation from the MATLAB `CCM-Implement` toolbox to Python.

---

## 1. From Standard CCM to Causalized CCM

### Reminder: what standard CCM does

Standard CCM reconstructs state-space embeddings (Takens delay vectors) for two scalar time series `X` and `Y`. For a chosen embedding dimension `E` and lag `tau`, it:

1. Builds manifolds `Mx` and `My` from delayed copies of each series.
2. For each prediction index `i`, finds the `E + 1` nearest neighbours of the point `Mx[i]` (or `My[i]`).
3. Uses the neighbour weights (exponentially decaying with distance) to cross-map the counterpart series; e.g. `Mx` predicts `Y`.
4. Compares the cross-mapped estimates to the observed data using Pearson correlation `rho`, typically across an increasing set of library sizes (`libSizes`).

Neighbour search typically includes the point itself, so to avoid the trivial neighbour, the smallest distance is ignored. CCM also treats library and prediction ranges independently, allowing “future” information (indices greater than `i`) to participate unless the user imposes an exclusion radius.

### What “causalized” changes conceptually

The cCCM variant introduced by Sun et al. explicitly enforces **temporal causality**: the cross-map at time `t` is allowed to use only information available up to and including `t`. This means:

- The library for the `j`-th embedding is restricted to indices `<= j`. No future embedding vectors can influence the prediction.
- When multiple sub-samples (random libraries) are used, each sampling step still honours this historical cut-off.
- The resulting correlations therefore serve as a consistent measure of causal influence in one direction, rather than symmetric mutual prediction skill.

Practically, cCCM retains the same distance-weighted averaging as CCM but employs a causal library schedule.

### Implementation notes (`src/pyc4m/cccm.py`)

The Python port mirrors the MATLAB code:

- **Embedding** – `_construct_manifold` replicates the sliding-window delay embedding used in standard CCM.
- **Causal library selection** – for each prediction index `j`, `candidate_idx = np.arange(j + 1)` (or the supplied indices intersected with `[0, j]`) create the time-respecting library.
- **Temporal exclusion** – the new `exclusion_radius` parameter removes neighbours whose timestamps are within the specified radius of `j`, aligning with the `exclusion_radius` option in `pyEDM`.
- **Prediction horizon (`Tp`)** – cCCM supports forward (`Tp > 0`) and backward (`Tp < 0`) horizons by shifting the neighbour indices through `_apply_tp_to_neighbors` and rolling the final projections.
- **Sampling (`sample`, `seed`)** – the API wrapper (`src/pyc4m/api.py`) randomly draws sets of embedding indices, maps them back into the causal routine, and averages the resulting correlations as pyEDM does.
- **Numerical stability** – weights are computed from scaled distances with a small `epsilon` to avoid zero-distance singularities; predictions are rolled and `num_skip` (aka `N0`) blanks the burn-in portion.

### Comparison to standard CCM

| Aspect                  | Standard CCM                      | Causalized CCM (`pyc4m`)                       |
|-------------------------|-----------------------------------|------------------------------------------------|
| Library membership      | Any index in library range        | Only indices `<= current j` (causal)           |
| Temporal exclusion      | Optional via `exclusion_radius`    | Supported; applied after causal filtering      |
| Prediction horizon      | `Tp` allowed                      | Same semantics                                 |
| Sampling (`sample`)     | Re-sample library with replacement| Random subsampling of embedding indices, causal per draw |
| Returned diagnostics    | Mean ρ per lib size (optional stats)| Identical structure; stats carry `Sample` ID  |

The rest of the computation (KD-tree neighbour lookup, exponential weighting, Pearson correlation) is unchanged. Setting `causal=False` in the Python API restores the standard CCM behaviour (libraries identical across time, no causal restriction), allowing side-by-side comparisons.

---

## 2. Multivariate Conditional CCM

### Why conditioning is needed

In multivariate systems, simple pairwise CCM can report spurious bidirectional causality when two variables share a common driver. Conditional CCM aims to quantify directed influence while **conditioning on the remaining variables**.

The approach used in `CCM-Implement` (and adopted here) is:

1. Run causalized CCM for every ordered pair `(i, j)` to obtain cross-mapped reconstructions (`YestX`, `XestY`).
2. For a target pair `(i, j)` with conditioning set `Z` (all other variables), collect:
   - The reconstructions of `i` from `j` and each `z ∈ Z`.
   - The reconstructions of `j` from `i` and each `z ∈ Z`.
3. Solve two least-squares problems:
   - `target_j ~ [Z reconstructions, i→j reconstructions]`.
   - `target_j ~ [Z reconstructions]`.
   The reduction in residual variance after adding the cross-map quantifies `i`’s conditional influence on `j` (and vice versa).
4. Normalise the variance reduction to obtain the conditional causality ratio (equivalent to the MATLAB `uX2Y`/`uY2X`).

### Implementation highlights (`src/pyc4m/conditional.py`)

- **Causal seeds** – Each pairwise reconstruction uses the same causalized CCM routine, so temporal directionality is preserved inside the conditional test.
- **Condition matrices** – `_gather_conditionals` stacks the reconstructions of conditioning variables into a design matrix aligned with the post burn-in samples.
- **Variance-based metric** – `_conditional_effect` runs two least squares fits and computes `(var_Z − var_Z_plus_cross) / var_Z`, matching the MATLAB formulation.
- **Robustness** – Before solving, the code filters out rows with `NaN` or insufficient degrees of freedom to avoid singular regressions (returning `NaN` when too few usable rows remain, which mirrors MATLAB’s guard via `N0`).

### Differences vs original MATLAB scripts

| Feature                       | MATLAB `CCM-Implement`          | `pyc4m` translation                          |
|-------------------------------|---------------------------------|----------------------------------------------|
| Base CCM routine              | cCCM with causal libraries      | Same logic (NumPy/SciPy implementation)      |
| Random sampling               | Optional, typically none        | Fully supported via API `sample`/`seed`      |
| `N0` burn-in (`num_skip`)     | Hard-coded (usually 10)         | Exposed parameter; adjusted per sample       |
| Temporal exclusion            | Built into MATLAB loops         | Controlled by `exclusion_radius`             |
| Diagnostics                   | Variance ratios                 | Same, in addition to raw residual variances in `diagnostics` |

Overall, the conditional extension is conceptually identical to the MATLAB version; the main additions are better handling of random subsets, guardrails for numerical stability, and an optional `causal=False` switch when the analyst wants to reproduce the standard (non-causal) conditional CCM.

---

### Quick reference

- Use `pyc4m.CCM(...)` with `exclusion_radius`, `Tp`, `sample`, and `seed` to reproduce causalized CCM experiments.
- Use `pyc4m.conditional(...)` for the conditional ratios; pass `exclusion_radius` if you need temporal exclusion in the pairwise reconstructions.
- Both APIs honour `num_skip` (`N0`) and will emit per-sample diagnostics when `includeData=True`.
