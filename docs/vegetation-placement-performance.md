# Large-world forest vegetation performance

This branch improves the long vegetation-placement segment that can appear as
`66% Placing forest undergrowth across forest interiors` in Worldgen's GUI.

## Why the application seemed stuck

The message marks the **start** of the undergrowth candidate loop, not its
completion. After it, object placement runs the following passes in sequence:

1. Forest undergrowth across the interior grid.
2. Bushes on steep forested hills.
3. Vegetation clusters along mapped forest boundaries.
4. Vegetation along ditches and later object classes.

Previously there were no intermediate messages inside the first three loops.
A world with a fine undergrowth grid could remain on the same overall percent
for a long time while still consuming CPU.

## Safe changes

- **Cached cluster layouts**: reuse small read-only NumPy arrays for identical
  proxy layouts instead of recreating arrays for each undergrowth/border cluster.
- **Early cluster-mask checks**: test every proxy's exact forest/water/road/
  building raster cells before the expensive road-footprint and polygon-elevation
  queries. The same existing checks are retained for surviving candidates.
- **Steep-bush mask indexing**: compute one raster cell index for all four masks
  instead of mapping the same position four separate times.
- **Road-first steep-bush rejection**: skip terrain support sampling when an
  eligible hillside candidate intersects an indexed road corridor. Both checks
  still apply to any candidate that can be placed.
- **Progress and elapsed-time reporting**: report sampled candidate counts,
  added object counts, and wall time through undergrowth, steep-hill bushes, and
  forest borders. The progress callbacks receive periodic updates even for
  very large grids.
- **Forest-border enumeration reporting**: expose expensive forest-boundary
  candidate sampling and sorting before the border-placement loop begins.

Existing deterministic grid traversal, random seeds, object IDs, budgets,
cluster transformations, ground-fit limits, and sorting are intentionally
unchanged. The existing multicore candidate precomputation and NumPy terrain
cache remain active; this work avoids creating an additional process pool.

## Suggested verification

Run the same frozen source dataset, terrain and configuration on the baseline
and this branch, ideally with caching disabled or consistently warm caches.

- Compare total wall-clock time of each named vegetation stage and the total
  placement build.
- Compare object records (model, ID, location, heading and elevation), count
  summaries, and rejection counts.
- Check the `tests/test_forest_vector_performance_policy.py` and
  `tests/test_vegetation_pass_performance.py` regressions, along with the
  existing placement and forest tests.
- Measure peak memory and throughput on a dense 30 m undergrowth grid; array
  caching trades a small bounded layout cache for fewer allocations.

The performance benefit has **not** been measured on a full-sized island yet.
The work does not disable forest undergrowth or relax clearance constraints.
No release is created by this branch.
