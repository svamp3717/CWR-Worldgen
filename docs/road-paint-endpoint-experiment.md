# Experimental road-stroke endpoint alignment

This branch tests whether joining compatible road stroke endpoints *before* the
existing road model fitter can reduce small grass gaps without introducing
overlapping slabs or new junction geometry. It does not change the model
catalogue, generated road meshes, terrain materials, or fit heuristics.

## Enable and compare

The experiment is disabled by default. For the same source bundle, seed and
worldgen settings, build two separate worlds:

- Control: leave `CWR_ROAD_PAINT_SNAP_METRES` unset.
- Experiment: set `CWR_ROAD_PAINT_SNAP_METRES=2.5` in the worldgen process
  environment before running the same build. Values are capped at 3 metres.

The adjustment runs on the projected road lines after the existing synthetic
paved-junction pass. A candidate is accepted only when:

1. Two road features have different identities and compatible resolved surfaces.
2. Both are at-grade, supported roads, and their loose ends face one another
   along almost the same line.
3. The gap is within the configured reach *and* at most 20% of each adjacent
   segment's length.
4. Each endpoint has exactly one compatible candidate.

Both endpoints then meet at a common midpoint. All interior road vertices
remain unchanged. The normal fitter still chooses and emits the road P3Ds.

## Validation

Run `pytest -q tests/test_road_paint_snap_policy.py` for deterministic
geometry and safety tests, then compare generated WRPs with the existing
`cwr-road-inspector` road audit. Record endpoint-gap counts, maximum gaps,
failed connections, road object counts, and visual screenshots of the same
junctions and curves. Also check road-ground texture coverage and both paved
and gravel cases.

This is a limited geometry trial, not a proven in-game rendering fix. It can
only close near-collinear feature-to-feature gaps; it cannot correct P3D
texture seams, misaligned curved model connectors, T-junction paint order,
grade-separated roads, wide missing segments, or discontinuities *inside* a
road feature. The surface-paint pass still sees original source geometry,
so verify it does not leave visible terrain under the newly joined ends.
On a failed comparison, unset the environment variable to restore identical
default behavior.

Do not turn this on by default until an identical-world A/B render confirms
fewer visual seams without new false connections.
