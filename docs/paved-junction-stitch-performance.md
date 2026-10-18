# Generated paved-junction seam stitching: large-world performance

## Why this change exists

A large build can reach `Paved-junction post-processing complete` and then
appear to hang. That message comes from the indexed paved-junction *apply*
operation. Immediately afterward, the fallback wrapper invokes
`_stitch_generated_hub_approaches()` before its next visible validation
message. The earlier stitcher scanned the entire road-object tuple separately
for **every generated paved hub**, without progress reporting.

With ~800,000 roads and many generated hubs, repeatedly checking every object
was a plausible hidden performance bottleneck. This change addresses that
specific pass; it does not assert that every stall has the same cause.

## Implementation

- Scan the input road list **once** and place each eligible paved segment in
  64 m spatial buckets based on its actual axis *bounding box*, not its centre.
- Retain unusually large segment bounding boxes in an overflow list, so a long
  segment crossing a junction cannot be missed.
- For each generated hub, collect nearby candidate road indices in their
  original object-list order, then use the **unchanged** narrow-phase seam,
  stale-piece, heading, and tie-break decisions.
- Skip building the spatial index when the current pass has no generated hubs.
- Emit progress during indexing and during hub stitching, including elapsed
  time and adjusted/removed counts. Fallback validation already emits
  separate indexing/validation progress messages.

Asymptotically, the old repeated scanning was proportional to
`road_objects * generated_hubs`; the new candidate-selection component is
approximately `road_objects + sum(nearby_candidates_per_hub)`, plus bucket
construction and queries. Actual speedup depends on road density, model mix,
number of generated hubs, and available memory.

The existing output order and final geometry rules are unchanged. Regressions
cover comparing indexed and unpruned output, long-axis intersections,
oversized-axis fallback, and progress emission.

## Important limits

- This fixes one *suspected* silent phase and does not claim to accelerate
  unrelated road-quality, fallback refits, cache serialization or packaging.
- No real-world 800,000-object run has yet been benchmarked in this branch.
- Validate using a reproducible large generated world and compare WRP and PBO
  output, successful connection counts, and phase-level wall-clock timings.
- The branch does not modify the public release workflow or publish a release.
