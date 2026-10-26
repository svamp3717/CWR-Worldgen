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

Rural bends now use sampled **cubic circular-arc approximations** with
tangent-continuity at both ends rather than the old quadratic interpolation.
The curvature is adaptive: open, long approaches receive larger radii; turns
with insufficient length for a 12 m minimum radius are left untouched. The
maximum tangent trim is 38 m and no more than 38% of each approach. Proposed
curves stay within 14 m of the original OSM road centerline to respect the
existing terrain-grading corridor.

The same cautious clearance checks remain: no smoothing at shared junctions,
inside or near towns and residential areas, or when the changed road ribbon
would touch buildings, other roads, bridge or tunnel approaches, water/watercourses,
or the terrain boundary. Endpoints are always preserved. Although curves
appear more natural, the generator still fits/grades terrain against the
original OSM source, so verify steep mountain roads in-game.

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
