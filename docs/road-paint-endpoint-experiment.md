# Experimental road-stroke endpoint alignment

This branch tests whether joining compatible road stroke endpoints *before* the
existing road model fitter can reduce small grass gaps while preserving
existing connection, terrain, and model-placement safeguards. It does not change the model
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

## Dense town and service roads

The same opt-in switch also enables two more narrowly gated topology repairs:

- **Short non-paved T misses:** a dirt or gravel street endpoint facing a
  transverse road of the same surface can move up to 3 m, at most one-fifth of
  its final segment, onto that segment. The through road receives the shared
  junction vertex. Existing connected junctions and parallel streets stay put.
- **Unconnected non-paved X intersections:** two transverse, at-grade dirt or
  gravel segments crossing geometrically receive a common vertex on both
  features. Ambiguous multiway intersections and near-parallel crossings are
  excluded.

These operations run *after* the existing synthetic paved-junction pass.
Paved T/X intersections are deliberately left to their existing fitter and
mixed paved/dirt or paved/gravel crossings are not rewritten. In both town
operations the fitter still owns P3D selection, generated junction caps and
terrain fit. No new model assets are introduced.

## Validation

Run
`pytest -q tests/test_road_paint_snap_policy.py tests/test_road_paint_town_policy.py tests/test_road_paint_worldgen_integration.py`
for deterministic geometry, safety, and full fitter A/B tests, then compare generated WRPs with the existing
`cwr-road-inspector` road audit. Record endpoint-gap counts, maximum gaps,
failed connections, road object counts, and visual screenshots of the same
junctions and curves. Also check road-ground texture coverage and both paved
and gravel cases.

This is a limited geometry trial, not a proven in-game rendering fix. It can
only close short near-collinear feature gaps and narrowly guarded same-surface
T/X town topology misses. It cannot correct P3D texture seams, curved model
connector mismatch, paved/gravel draw precedence, grade-separated roads,
wide missing segments, or discontinuities *inside* a road feature. The surface-paint pass still sees original source geometry,
so verify it does not leave visible terrain under the newly joined ends.
On a failed comparison, unset the environment variable to restore identical
default behavior.

Do not turn this on by default until an identical-world A/B render confirms
fewer visual seams without new false connections.

## Initial synthetic worldgen result

A 400 m dirt through-road with an approximately 2 m incomplete service-road
approach was processed by the real stock-piece road fitter on a flat 1 km map.
Without the switch: 22 emitted road objects, no junction caps and no reported
failed connections. With the switch: 27 emitted road objects, one junction
cap and no reported failed connections. This demonstrates that the new
topology reaches actual model placement rather than stopping at a map preview.
It does **not** prove a reduction in grass seams in CWA until game-rendered
A/B screenshots are compared.

A second flat-map A/B check uses two dirt through-roads intersecting
geometrically without an explicit OSM connection. Control emitted 32 road
objects and zero junction caps; the experiment emitted 39 road objects
and one junction cap. Both builds reported zero failed connections.

The branch-scoped GitHub Actions workflow also runs the paved-junction,
gravel-gap, and road-quality regression suites. The expanded suite passed
68 checks, with three additional end-to-end fixture checks passing.
