# CWR Worldgen

**CWR Worldgen** is a deterministic world-generation tool for creating playable **Operation Flashpoint / Cold War Assault** islands from real-world geographic data.

The goal is to turn real places into recognizable, playable OFP/CWA terrain with as little manual map-making work as possible.

CWR Worldgen combines elevation data with **OpenStreetMap** features and can use **Overture Maps** to enrich or fill gaps in building coverage. It then converts that source data into an OFP/CWA-compatible world, including terrain, roads, buildings, vegetation, surfaces and game-ready assets.

It supports both the original **Cold War Assault 1.99** runtime and **CWR-CE**.

## What CWR Worldgen generates

A generated world can include:

- Real-world elevation and terrain derived from DEM data
- Coastlines, sea, inland water, beaches and shoreline transitions
- Paved roads, dirt roads and gravel roads
- Bridges and road crossings
- Buildings based on OpenStreetMap and Overture building data
- Procedural buildings with regional/country-based architectural styles
- Original OFP/CWA and addon P3D building presets
- Forests, individual trees and rural vegetation
- Meadow grass, reeds, bushes and other vegetation passes
- Parks, agricultural areas and other mapped land-use features
- Terrain materials and configurable ground-texture styles
- Runway, parking and sports-field surface treatment
- Town and place information
- Overview-map and island presentation assets
- A finished RVW4 world and supporting files
- A packaged PBO placed into a CWA/OFP mod folder

Most parts of the generation process can be enabled, disabled or customized from the GUI.

## Real-world source data

CWR Worldgen builds its maps from several geographic sources.

**Elevation data** is used to create the game terrain grid. The normal source pipeline can obtain DEM data automatically and regrid it to the selected OFP/CWA world size.

**OpenStreetMap** provides roads, buildings, forests, water, land use, settlements, utilities and many of the other features used when constructing the world.

**Overture Maps** can optionally supplement OpenStreetMap building data. This is particularly useful in areas where OSM contains incomplete building coverage.

Source data can be cached and reused, allowing the same selected area to be rebuilt without downloading and processing everything again.

## Buildings

Buildings can be generated in several ways.

CWR Worldgen includes procedural building styles designed to adapt building appearance to the selected area or country. Building footprints and mapped information are used to choose suitable building types, dimensions and placement.

Original game and addon P3D models can also be used through selectable building catalogues. These catalogues contain measured model dimensions, semantic categories and placement information so the generator can choose models that fit the available footprint and settlement context.

Multiple building catalogues can be enabled together.

Building placement also takes terrain and nearby roads into account. The generator grades terrain where required, checks physical road clearance and can select smaller compatible models when a chosen stock building does not fit safely.

## Terrain and appearance

Terrain appearance is independent from vegetation appearance.

Ground textures can use several classic OFP/CWA terrain styles, including Nogova, Kolgujev, Malden and Everon, as well as generated or desert terrain materials. Terrain textures can also be disabled entirely when an unpainted WRP is desired.

**Stock forest undergrowth textures (on by default):** In **Common choices**,
**Stock forest undergrowth ground textures (incl. forest edges)** paints OSM
forest interiors with their original stock forest-floor texture and
automatically chooses **directional border and corner blend textures**.
The checkbox is on by default and does not control the separate 3D
**Forest undergrowth** vegetation option. It follows **Ground textures**, not
the selected vegetation profile:

| Ground texture preset | Forest floor | Example west-edge blend | Ordinary edge fallback |
| --- | --- | --- | --- |
| Nogova | `o\\j1.paa` | `o\\j1t1j1t1.pac` | `o\\t1.paa` |
| Everon | `eden\\pl.paa` | `eden\\pltnpltn.paa` | `eden\\tn.paa` |
| Malden | `abel\\pb.paa` | `abel\\pbttpbtt.pac` | `abel\\tt.paa` |
| Kolgujev/Cain | `cain\\u2.paa` | `cain\\u2j9u2j9.paa` | `cain\\j9.paa` |

The four two-letter parts of a blend filename describe its NE, NW, SE and
SW quadrants. Worldgen inspects the OSM forest shape and chooses a verified
stock path at each forest-edge cell, including corners. Everon and Malden
lack a few diagonal combinations in their original texture tables: Worldgen
substitutes a verified nearby blend rather than inventing a filename.
See [the original-island analysis](docs/stock-forest-border-texture-research.md).

Compatible ordinary grass cells immediately adjoining forest transitions can
also receive the appropriate verified stock blend. Protected roads, farmland,
water and urban surfaces, vegetation objects, and terrain heights remain intact.
Generated, desert and untextured profiles ignore the option. On the CLI,
the feature is enabled by default; use `--no-forest-undergrowth-textures` to
disable it or `--forest-undergrowth-textures` to enable explicitly.

**Malden and Kolgujev beach handling:** These two stock ground-texture
presets do not assign separate dry beach/sand classes. Dry shoreline and
mapped beach/sand areas use the same ordinary ground as the rest of the
island (`abel\\tt.paa` or `cain\\j9.paa`). Other ground-texture presets
retain their existing shoreline materials.
Original game texture artwork is **not** copied into the generated PBO.


Vegetation can likewise use different game styles independently of the terrain texture set. Forest blocks, individual trees, undergrowth, rural vegetation, meadow grass, wetland reeds and related vegetation systems can be configured separately or disabled with the **None** vegetation preset.

This makes it possible, for example, to use Nogova ground textures with Everon vegetation, or generate a terrain with no vegetation or painted terrain textures at all.

## Roads and terrain fitting

Road networks are generated from mapped road geometry and converted into OFP/CWA-compatible road objects.

The generator handles paved, dirt and gravel roads, junctions and bridges while adapting the surrounding terrain to keep roads usable. Road grade, terrain adjustment, connection tolerances and related limits are configurable.

**Default gravel donors:** Gravel-tagged OSM roads now prefer the stock
`o\\road\\asf25.p3d` straight family with
`o\\road\\asf10 25.p3d` as the selected curve-style donor, instead
of defaulting to Worldgen's procedural gravel texture and ribbons. Regular
stock straights and short repairs reuse available `asf` siblings; generated
connectors/junctions remain available when necessary for fitting, and use the
configured gravel donor's independent **Roadway LOD contact texture and face/
point flags**. This is a donor-surface choice: the stock `asf` family may
behave like asphalt rather than loose gravel. Paved roads use the selected
paved donor, and dirt roads use the selected dirt donor, rather than a common
fallback contact style.

Worldgen checks each generated road P3D's **emitted Roadway faces** against
the selected contact metadata before packing it, including after an asset-cache
restore. The `road-donors.json` report distinguishes inspected donor contact
data from a `visual-fallback` when source Roadway information is unavailable.
Where the donor is unreadable, matching in-game surface physics cannot be
guaranteed; add its PBO to the configured asset roots.

**Optional fence donor models:** Advanced generation settings now include two
fence P3D donors: one for ordinary/wooden/rural fences and one for metal fences.
Leave either model field blank to retain original OFP/CWA fences. Select a
**virtual P3D path** (for example `sfp\wood\fence.p3d`) and add its PBO or
directory under **Asset roots**. Do not enter a local PBO file path here.

Worldgen places these P3Ds directly along mapped OSM fences, farmland
boundaries and residential yards. It does **not** reshape donor meshes.
**Leave length and rotation blank:** the generator reads the first visual LOD
of the selected P3D from Asset roots, measures its longest horizontal
(X/Z) dimension, and infers alignment (long X: 90°; long Z: 0°).
This supports OFP/CWA MLOD and ODOL v6/v7 models. The two numeric inputs are
optional overrides for diagonal, unusual or unsupported meshes. An ambiguous
or unreadable model produces an explanatory error instead of silently
guessing. Origin/pivot offsets and curved fence geometry cannot be inferred
reliably from a rectangular bounding box, so donor meshes can still show gaps
at bends or short ends. Hedges and walls are unchanged.

**Traversable fence openings:** Worldgen now leaves seeded, walk-through
entrances in mapped OSM fences and selected fenced farmland perimeters. The
locations look random but remain reproducible with the same world seed and
OSM data. Boundaries at least 18 m long are eligible; long fences receive
roughly one gap per 120 m, with openings at least 5 m wide. The generator
removes complete fence panels and leaves their collision geometry out of
the WRP, rather than inserting a locked gate or a fake visual hole. Some
residential yard fences also omit one panel. Existing road/track crossings
continue to remain open. Custom fence donors participate automatically;
walls and hedges are not modified.

CLI example:

```powershell
python -m cwr_worldgen milestone9 --source-dir source-data/map --output build/world --asset-root "C:\Mods\sfp.pbo" --fence-donor-model "sfp\wood\fence.p3d"
```

Fence donor settings are remembered between GUI launches using the same
persistent preferences file as road donors.

**Wall, hedge and ditch vegetation donors:** Advanced generation settings also
accept one P3D each for mapped stone/retaining walls, mapped hedges, and
vegetation strips along mapped OSM ditches. All three are opt-in: blank
model paths leave stock walls/hedges and generated ditch grass unchanged.

Like fence donors, supply a virtual P3D path, add the model PBO or folder to
**Asset roots**, and leave length and rotation empty to measure the first
visual LOD automatically (legacy MLOD and ODOL v6/v7). The longer horizontal
axis is used to align straight pieces. Wall and hedge measurements determine
segment spacing; ditch strip length sets candidate spacing along the mapped
ditch. Ditch vegetation places the donor object directly, subject to road,
building and slope/float checks. Hedge grounding uses the model's measured
visual width and lowest vertex where inspection succeeds.

Example CLI flags:

```powershell
--wall-donor-model "addon\walls\stone.p3d" --hedge-donor-model "addon\hedges\thick.p3d" --ditch-vegetation-donor-model "addon\grass\reed_strip.p3d"
```

All length/rotation flags are optional manual overrides, for example
`--hedge-donor-length 5 --hedge-donor-heading-offset 0`.
These are direct placements, not stretched/generated meshes. A bounding
box cannot guarantee correct connectors, pivots or rounded bends.

**Existing GUI profiles:** Saved empty gravel donors from older releases
migrate to the stock `asf` defaults, but explicitly selected modded donors
are not replaced. Users who deliberately want procedural gravel may clear
both gravel donor fields; the GUI remembers that choice. The CLI can likewise
pass `--gravel-road-model "" --gravel-road-curve-model ""`.
This does not change post-build cleanup behavior.

Buildings and other objects are checked against the final road layout so that generated scenery does not simply occupy the same physical space because two datasets happened to disagree.

## Deterministic generation

CWR Worldgen is designed to produce reproducible worlds.

Generation uses deterministic seeds and stable processing rules so the same inputs and settings can be rebuilt consistently. Expensive stages can be cached, while regeneration verification can be used to compare repeated builds when reproducibility needs to be checked.

This also makes it practical to adjust one part of a world and rebuild it without manually recreating the entire terrain.

For faster repeat builds, leave the persistent build cache enabled and retain
the selected build directory between runs. `--no-cache` bypasses the stage
caches, `--cache-refresh` rebuilds matching cached stages, and reproducibility
verification deliberately performs additional work. The generator also batches
cache-fingerprint hashing for large elevation grids and avoids deep-copying
terrain/material/road arrays that are not included in the final diagnostic
JSON reports. These improvements preserve the generated world data and the
existing cache-key format.

## Output

The final build produces the files required for an OFP/CWA island, including the RVW4 world and generated supporting assets.

CWR Worldgen can package the finished world into a PBO and place it into the selected mod folder, reducing the process from real-world source data to a playable island to a largely automated workflow.

The project is intended for both **Cold War Assault 1.99** compatibility and the **CWR-CE** remastered engine, with runtime-specific behavior handled by the selected game profile.
