# pyC4M Verification Notes

- **Resolved – default `tau` compatibility**
  - `causalized_ccm` now accepts signed delays and the API enforces `tau < 0` only when causal mode is requested, matching pyEDM's default `tau=-1` convention.

- **Resolved – `includeData` alignment**
  - Error bookkeeping now compares reconstructed trajectories against the embedded tails returned by the cCCM core, keeping the statistics consistent with pyEDM and the MATLAB toolbox.

- **Resolved – zero-distance neighbour handling**
  - `_query_neighbors` drops only the self-match (distance zero) while retaining additional zero-distance ties, mirroring pyEDM's KDTree tie-handling.

- **Open – pyEDM interface gaps**
  - The wrapper still raises `NotImplementedError` for `embedded=True` and `validLib`. Existing pyEDM call-sites that rely on those parameters will need follow-up work before migration.
