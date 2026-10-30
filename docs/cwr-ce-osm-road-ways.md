# CWR-CE native OSM road ways profile

The **cwr-ce-ways** game profile targets
[CWR-CE's isolated native OSM-style road implementation](https://github.com/svamp3717/CWR-CE/tree/feature/osm-style-road-ways-isolated).
Use that engine branch (or a CWR-CE build containing the feature). Ordinary
**cwa** and **cwr-ce** profiles are unchanged.

## Selecting the profile

In Worldgen's GUI, choose **Game profile > cwr-ce-ways** and build normally.
The same selector is exposed on the CLI:

    python -m cwr_worldgen milestone9 --source-dir source-data/map --output build/world --profile cwr-ce-ways

The normal DEM, OSM, terrain grading, buildings, terrain textures, and scenery
pipeline still runs. The additional output is **worldname.ways**, beside
**worldname.wrp** under **source/worldname/** and in the generated world PBO.

The file uses the engine's UTF-8 line format:

    node 1 100.0000 150.0000
    node 2 180.0000 150.0000
    node 3 220.0000 180.0000
    way 1 6.000 o\road\sil_new.paa 1 2
    # When a matching mounted donor P3D is available, instead:
    way 2 6.000 donor=o\road\sil25.p3d 2 3

Positions are *projected local world X/Z metres*, **not latitude/longitude**.
Node IDs are shared when projected/snapped road vertices coincide. This preserves
ordinary junctions and keeps separated geometry disconnected. Data is sorted and
clipped to the terrain bounds, with stable IDs for identical inputs.

The new profile **omits legacy road P3D objects from the WRP** so roads do not
appear twice. Worldgen still computes its original road fit as input for terrain
grading and building/vegetation clearance. Non-road P3Ds and separately
modelled bridge/other infrastructure are retained.

It also excludes those omitted road objects from **generated model usage**;
previously, they were removed from the WRP but still caused generated road
P3Ds to be packed unnecessarily. Builds using an existing output directory
now prune stale unused generated road P3Ds and road-only textures from
`source/<world>/i/` before PBO packing. Generated bridges, rocks, fences
and utilities are not deleted, nor are any generated road models actually
referenced by retained WRP objects. PBO verification rejects unexpected
unused generated road P3Ds in native-ways builds. A donor P3D referenced by
`donor=...p3d` is an external mounted asset, not a generated road object.

## Natural bends across rural roads

The exporter first **merges compatible end-to-end OSM ways** into continuous
native road chains, then rounds eligible bends across those original way
boundaries. Two ways are joinable only when their projected endpoint coordinates
match exactly, they have the same output width (to 0.001 m) and surface
material, and the shared point is not a branch or mapped interior crossing.
Nearby but distinct endpoints are *not* snapped, so the merge never creates
new connectivity. Junctions, bridge/tunnel approaches, width/material transitions, and loops
remain protected. The engine's existing 512-node output chunking still applies
to long merged chains.

Rural bends use sampled **cubic circular-arc approximations** with
tangent-continuity at both ends. Sharp turns between **18° and 160°** are now
eligible when both approach legs are at least **16 m**, or **7 m for an
isolated two-leg corner**. The fitter tries the widest natural radius first,
backs off around obstacles, then **binary-searches
the largest safe radius** between accepted/rejected sizes. This avoids abrupt
size jumps between otherwise similar bends. The minimum radius is **at least
5 m** and scales with exported
road width. Extremely tight hairpins or short legs that cannot accommodate
that radius remain unchanged rather than producing unrealistic kinks.

The maximum tangent trim is **52 m**. Two adjoining turns can each consume
up to **42%** of a shared segment; an isolated corner can use up to **62%**
of its approach when no neighboring bend needs that space. The cubic arc is
sampled at approximately **1.8 m** intervals and stays within **14 m**
of the original OSM corridor in both directions. Adjacent
curves are rejected together if their combined path would self-intersect.
This also handles previously unrounded turns above 135° without adding
roadway over unsafe shortcuts.

### P3D road-fitter geometry reused for native OSM ways

The native ways exporter now shares the existing P3D road fitter's **road-width
minimum-radius calculation, feasible tangent budget, and maximum local
heading-change sampling** (the stock road fit's 15° segment limit).

Before rounding ordinary rural corners, it can repair a tiny, short-leg
OSM zigzag that cannot physically accommodate the active road width.
As in the P3D road fitter, the unrepresentable intermediate vertex is
removed and the remaining road can be rounded. Unlike the original
P3D-only repair, native ways require the entire simplified centerline
to stay within **4.5 m** of the original OSM geometry, and the removed
point itself must be within the tighter width-dependent threshold.
These are short digitization defects, not genuine hairpins.

The repair is **blocked** at shared intersections, bridge and tunnel
connections, towns, buildings, water, nearby roads and map boundaries.
Long bends keep their existing adaptive cubic treatment, and the
original source still drives terrain grading. With **Keep donor road width**
enabled, the repair uses the measured P3D donor width rather than the
OSM width, just like the exported native ways.

The build manifest reports `road_ways.repaired_rural_kinks` separately from
`road_ways.smoothed_rural_bends`.

Corner rounding also works **near villages, residential landuse, and
T/X intersections**, where it previously left the road angular. It never moves
a shared junction or bridge/tunnel endpoint. Instead of forbidding every bend
within a broad 24 m radius of a junction, the exporter checks the *actual
candidate curve* and preserves a straight neck of at least **8 m** (or the
road width plus 2 m) before the connected junction.

A settlement marker or residential polygon alone no longer blocks a turn.
Inside settlements, the new arc must remain within **5 m of the original
road**, compared with the normal rural 14 m limit. The **building and roadway
clearance checks remain strict**, so narrow streets with nearby buildings,
water, neighboring roads, or no safe tangent space stay unchanged. Artificial
long S-curves and removal of tiny P3D digitization kinks remain conservative
around settlements. Endpoints always remain fixed.

Worldgen still grades terrain against the original OSM road centerline;
inspect steep slopes and built-up intersections in-game after regeneration.

### Fairing the remaining short S-bends

Longer roads sometimes contain **several closely spaced OSM vertices**.
Smoothing each separately can leave a visible angular zigzag, especially
on curves leading toward a T-junction. After regular corner fitting,
Worldgen now scans for remaining turns of **19° or more**. When possible,
it replaces a local **12–32 m window** with a tangent-matched cubic that
smooths several small corners *together*.

The new path must preserve all connected junction vertices and bridge/tunnel
approaches, pass the same road-width-aware curvature and clearance tests as
ordinary bends, and remain within **14 m of the original OSM centerline**
even after multiple smoothing passes. Individual fairing windows stay within
**10 m of the local source**, while urban bends retain the stricter
**5 m** limit. Wide donors are checked at their measured width.

This is not a universal spline that can cut across any intersection:
where a safe window cannot be found, the original geometry is retained.
The manifest records accepted fairings as
`road_ways.eased_residual_turns`. The original terrain grader still
uses the source OSM route, so compare the result in-game.

### Smooth paved through-roads at T/X junctions

The previous bend passes avoided **the exact shared graph vertex**. CWR-CE's
road renderer joins road-edge offsets at every centerline node, so an angular
through-road at that junction still showed a sharp, widened corner even after
nearby bends were rounded. The new pass handles this specific case.

For a clear T/X junction, Worldgen selects an unambiguous compatible
through-road pairing (same rendered material and width, preferring both
halves of the original OSM way), and shapes approximately the last/first
**30 m** on either side with cubic arcs whose *tangents meet* at the shared
node. **The junction point is not relocated**; the side road retains the
same exact node ID and its existing geometry. This works both when the
through-road is a single OSM feature and when it is split at the junction.

Curves are limited by the existing grade corridor (at most **10 m** local
deviation, or **5 m** within settlements). They are rejected where buildings,
water, nearby roads outside the normal junction overlap, map boundaries,
or other protected graph nodes make them unsafe. Bridge and tunnel joins
are excluded. Ambiguous multi-road junctions are not guessed.

Importantly, the later long-rural-straight S-curve pass **does not overwrite
these newly aligned approaches**. This prevents it from restoring the sharp
heading change at the shared point. The manifest reports accepted junctions
under `road_ways.smoothed_junction_through_roads`.

### Short bends immediately after connected junctions (Lundby fix)

A real `wg_osmways20.pbo` road sample exposed a remaining case: a
**39.4-degree paved turn** at approximately **X=4666, Z=4633** follows
a track T-junction by just **7.1 m**. The selected paved P3D donor makes the
road **9.1 m wide**. Previously the 16 m minimum-leg check and an
11.1 m junction buffer rejected this turn even though a short circular
fillet could fit while retaining the shared track endpoint.

The exporter now also considers **6 m+ legs immediately beside a genuine,
exactly shared OSM junction**. The shared node is kept in place; the
curve must pass the same width-aware minimum-radius, source-corridor,
building/water, and self-intersection checks as before. Its overlap with
a connected side road is allowed **only where the old roadway already
overlapped that side road**; the new road ribbon must create no extra
overlap. Nearby roads with separate or almost-matching endpoints still
block smoothing. These narrow changes do not relax genuine junction
topology and bridge/tunnel protection.

The regression test uses translated real coordinates from the provided
Lundby OSM input and `wg_osmways20.ways` output, including the gravel
branch that previously blocked the visible bend.

### Gentle curves on long straight sections

Corner rounding alone leaves long OSM straight lines completely straight.
For the **cwr-ce-ways** profile, the exporter now also considers straight
rural spans of at least **140 m**, including consecutive nearly collinear
OSM sections, for a smooth, low-frequency S-curve. Road sections adjoining
a sampled sharp-bend curve need **250 m** of room instead, to avoid
overcrowding turns.

The displacement is small and road-class dependent: secondary roads are
subtle (up to 1.4 m base amplitude), tertiary/unclassified/residential
roads have modest movement, and unsurfaced tracks can use up to **3.8 m
base amplitude**. The sinusoidal envelope reduces these maxima further
along the centerline and has zero lateral offset and lateral derivative
at both endpoints. This avoids sudden heading changes at road joins.
Curves have no random per-vertex jitter, are deterministic and
orientation-independent (reversing a road does not change its shape),
and are sampled approximately every 12 m. Motorways, trunks and primary
roads are not artificially bent.

The new line is checked against the original centerline (**at most 4 m
deviation**), towns and urban polygons, buildings, waterways, other
roads, shared junctions and map boundaries. Modeled bridge and tunnel
approaches stay exactly aligned. If checks fail, the original straight
section is exported unchanged. The existing road objects are still
used for terrain grading, so these curves are cosmetic refinements,
**not** an automatic regrading or a faithful edit to OSM itself.

The build manifest now includes **road_ways.smoothed_rural_bends**,
**road_ways.softened_rural_straights**, **road_ways.merged_source_runs**,
and **road_ways.merged_road_chains**.

## Optional donor road widths

In the GUI's **Advanced generation → Road model donors**, enable
**Keep donor road width (CWR-CE native .ways only)** to use the measured
straight donor P3D width instead of OSM's highway/width-derived values.
The option is **off by default**, is saved between GUI launches and in
saved GUI profiles, and can also be enabled from the CLI with
`--native-way-donor-width`.

When checked, the writer reads the **connector span** from each mounted
P3D donor's inspected geometry, falling back to its visual mesh width
only when connectors are absent. The result is emitted directly as the
width in each `way` record for its paved/dirt/gravel surface family.
This also affects which ways can merge (same surface and final width)
and the ribbon clearance checks. Widths and the active setting are
reported as `road_ways.surface_widths` and
`road_ways.keep_donor_road_width`.

To use this setting, build with **milestone4 or later** and add the PBO(s)
containing the configured straight road donors to **Asset roots**.
Milestone3 cannot inspect donor P3Ds and rejects the option explicitly. If a used road surface has no readable
donor or an invalid physical width, the build gives an explicit error
rather than silently keeping the OSM value. Unused surface families
don't require donor assets. With the checkbox **unchecked**, road width
calculation remains unchanged.

This only changes native CWR-CE `.ways` widths, **not** the actual P3D
mesh dimensions or the earlier OSM-based terrain grader. For very wide
custom donors, inspect roadside buildings, curves, and bridge approaches
in game, since the original graded corridor may be narrower.

## Road surface materials and isolated CWR-CE

The isolated branch extends the existing format to accept a **road P3D
material donor** in the third `way` field, e.g. `donor=o\road\sil25.p3d`.
Instead of using the model as a rigid placed segment, the engine extracts
the road's visible texture, UV mapping, multi-material shoulders, vertex
lighting/face flags and its separate Roadway contact texture.

For milestone4+ builds, Worldgen emits `donor=...p3d` if the configured
straight-road donor model is found in the specified **Asset roots**. Ensure
that P3D is also mounted/accessible to the game when loading the world.
If it is not found, Worldgen exports the donor's **visual texture** where
available, or a stock CWA texture. A plain `.paa` / `.pac` source remains
supported by the isolated CWR-CE engine but cannot recover custom-model
UVs and other donor-specific behavior.

The three *stock* fallback textures are recognized by the isolated engine,
which can automatically borrow matching stock P3D lighting/style when
the texture and donor match:

| Surface | Stock texture-only fallback | Matching P3D |
| --- | --- | --- |
| Paved | `o\road\sil_new.paa` | `o\road\sil25.p3d` |
| Dirt | `o\road\ces_hned.paa` | `o\road\ces25.p3d` |
| Gravel | `o\road\asf_new.paa` | `o\road\asf25.p3d` |

The emitted source is listed under **road_ways.surface_sources** in the
build manifest. **road_ways.surface_donors** (milestone4+) lists models
confirmed in Asset roots. These updates apply only to the **cwr-ce-ways**
profile and require the isolated CWR-CE engine branch or a build containing it.

## Current limitations

- **Bridges use a hybrid roadway.** The elevated, drivable stock bridge
  P3D supplies its own Roadway LOD over the water; a terrain-following
  native .ways ribbon must *not* be drawn on the water underneath. Worldgen
  now measures the *placed stock bridge's actual deck endpoints* (including
  object pitch) and splits OSM bridge-tagged roads at those endpoints.
  The dry approach at **each side of the bridge is exported as a native
  road** even when the OSM bridge=yes way continues hundreds of metres onto
  the banks. There is a 0.7 m intentional overlap into the bridge deck to
  conceal the visual seam, without overlaying native roads along the deck.
  Bridge models are retained in the WRP and the PBO.

  If final solved terrain rises above the bridge's visible road deck at an
  abutment, Worldgen also fits a **short elevated stock asphalt P3D cover**
  (from available `O\road\sil25.p3d`, `sil12.p3d` and `sil6.p3d`)
  over the buried end. It samples the finished terrain under the full road
  width, sets the model's road surface above the grass, slopes it toward the
  bridge deck and blends about 6 metres onto the dry native road and 4 metres
  onto visible stock roadway. Fully clear bridge ends get no duplicate road
  models. The same fitted stock P3Ds are included in the final WRP while
  ordinary terrain roads remain entirely native `.ways`.
  The manifest reports `road_ways.buried_bridge_road_covers`.
- **Unsupported bridge spans remain excluded.** If no stock bridge module can
  be matched to a bridge-tagged OSM road, the exporter does not guess an
  elevated deck or draw a submerged road. Check the build manifest's
  `road_ways.unmatched_bridge_features` count. The output also reports
  `road_ways.restored_bridge_approach_ways` and `road_ways.stock_bridge_modules`.
  Tunnel roads are still not supported by the terrain-following exporter.
- Shared junctions are inferred from coincident **normalized/projection**
  coordinates. Paths that nearly meet but have distinct vertices are not
  artificially connected. For complicated crossings, inspect the resulting
  network in game.
- The CWR-CE implementation limits files to 8 MiB, 100,000 nodes, 10,000 ways,
  200,000 lines, and 8,192 bytes per line. Worldgen splits long ways and rejects
  output exceeding remaining engine limits rather than writing an unreadable
  file.
- Road modelling is still performed for terrain-fitting and clearance purposes.
  This profile changes final road **serialization**, not the entire planner.
- The default CWA 1.99 game cannot read the ways sidecar; use **cwa** or **cwr-ce** for
  legacy P3D roads.

Build manifests include a **road_ways** summary (node count, way count,
textures, skipped bridge features and legacy-road object count). The build's
PBO layout verification also requires the ways entry for this profile.
