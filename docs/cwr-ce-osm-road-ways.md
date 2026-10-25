# CWR-CE native OSM road ways profile

The **cwr-ce-ways** game profile targets
[CWR-CE's model-free OSM-style road implementation](https://github.com/svamp3717/CWR-CE/tree/feature/osm-style-road-ways).
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
    way 1 6.000 data\silnice.pac 1 2

Positions are *projected local world X/Z metres*, **not latitude/longitude**.
Node IDs are shared when projected/snapped road vertices coincide. This preserves
ordinary junctions and keeps separated geometry disconnected. Data is sorted and
clipped to the terrain bounds, with stable IDs for identical inputs.

The new profile **omits legacy road P3D objects from the WRP** so roads do not
appear twice. Worldgen still computes its original road fit as input for terrain
grading and building/vegetation clearance. Non-road P3Ds and separately
modelled bridge/other infrastructure are retained.

## Softer bends outside settlements

The native-ways profile also rounds eligible **rural** corners. Worldgen
replaces a sharp vertex with a short, tangent-continuous quadratic curve,
sampled into several shared-node way points for CWR-CE to render. The
underlying OSM source files and other game profiles are unchanged.

A corner is eligible only when both incoming and outgoing sections are at
least 18 m long, the direction change is between 25 and 135 degrees, and
there is room for a gentle bend. The curve is limited to an 18 m tangent
length and at most 30% of either adjoining section, so it stays close to
Worldgen's existing terrain-graded road corridor.

To preserve road connectivity and sensible placement, the writer **keeps
endpoints and shared OSM road junctions fixed**, and leaves bends unchanged
inside or near mapped urban land-use polygons, settlement polygons, and
city/town/village/hamlet places. It also checks the prospective road ribbon
against nearby road geometry (including bridge approaches), building
footprints/points, water and watercourses, and map boundaries. If a proposed rounding fails any check, that bend remains
exactly as supplied by OSM.

The build manifest's **road_ways.smoothed_rural_bends** field reports the
number of bends actually rounded. This is a geometric visual improvement;
it does not rebuild Worldgen's terrain solver or guarantee wider cuttings
on very steep slopes, so inspect sharply graded mountain roads in-game.

## Road surface textures

A native road way record takes a **Roadway texture path**, not a road P3D path.
When Worldgen successfully inspects a configured road donor's Roadway LOD,
it uses that donor's exact Roadway texture for paved/dirt/gravel ways.
If no donor Roadway texture is available, stock paths are used:

| Surface | Fallback |
| --- | --- |
| Paved | data\silnice.pac |
| Dirt | data\cesta.pac |
| Gravel | data\asfaltka.pac |

As in the CWR-CE design, the texture is also used by road surface queries.
For custom modded road donors, include the relevant model/texture assets in the
build's **Asset roots** to resolve the actual donor texture. Verify stock
fallbacks against the game's installed data and inspect the resulting
**road_ways.surface_textures** entry in the build manifest.

## Current limitations

- **Bridges and tunnels** are not exported as native ways. CWR-CE's ribbon
  follows the terrain, so elevated or underground routes need separate models.
  Existing non-road bridge assets are retained; bridge approaches may require
  manual road connection work.
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
