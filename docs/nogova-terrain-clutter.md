# Optional Nogova terrain clutter from the original Noe WRP

**Update:** This checkbox is now named **Original island terrain clutter (matching trees)** and also works with the Eden, Abel and Cain vegetation presets. It still selects the Nogova pool only for Resistance leaf/pine tree presets. See [Stock island terrain clutter](stock-island-terrain-clutter.md) for all four palettes.

## What the source shows

The user's Island Studio object export from **Noe.pbo / noe\noe.wrp**
contains **177,224** WRP objects, 345 model paths, and exact x/z
locations across the original 12.8 km x 12.8 km terrain.

| Original stock Nogova terrain-detail type | Count |
| --- | ---: |
| DD small bushes (`o\tree\dd_bush*.p3d`) | 101,099 |
| Young spruce (`o\tree\smrk_maly.p3d`) | 21,859 |
| Reeds (`o\tree\dd_rakosi*.p3d`) | 2,358 |
| All mature trees (selected spruce/maple/oak/acacia) | 34,551 |
| Forest polygons | 4,502 |

Eight compact DD shrub models plus additional small bush variants dominate
the scenery. Bushes and young spruce occur within 15,657 distinct 50-metre
cells; the occupied cell median contains 5 such small details. This is why
the original map is denser than ordinary generated rural/forest terrain.

The optional compositor only copies **verified model virtual paths and a
roughly source-proportional relative model weighting**. It does *not*
reproduce Noe's locations, total source density, forest polygons, buildings,
roads, or reeds. Existing dedicated forest and wetland stages already
handle those categories.

## Checkbox and tree-preset gating

**Common choices → Nogova terrain clutter (Resistance trees only)** is
**off by default** and is saved with other GUI settings.

The pass runs only when BOTH:
1. This checkbox (CLI flag `--nogova-terrain-clutter`) is enabled.
2. Actual active vegetation models are the Nogova Resistance **leaf**
   (`o\tree\les_nw_ctver_pruhozi_T1.p3d`, Javor01) or **pine**
   (`o\tree\les_nw_jehl_ctver_pruhozi.p3d`, smrkmaly) tree presets.

Nogova **ground textures with Everon trees do not enable this**.
Using Nogova trees with any ground texture preset can enable clutter.
The CLI also enforces this based on resolved WRP model paths, so manually
supplying the checkbox without the tree preset cannot bypass it.

The density is intentionally conservative, but **there is no fixed
clutter object-count cap**. Instead, 45-metre strata, 48% preliminary
sampling, and terrain/object-clearance checks decide how many objects
fit naturally. The obsolete `--nogova-clutter-max-objects` CLI argument
is still accepted for legacy scripts but is ignored, including a value
of zero. Use the checkbox or CLI enable flag to control clutter.
Candidates are seeded from the existing world seed and reproducible
across normal builds and regeneration audits.

## Safety and placement

- Checks final terrain height and local ground relief for floating shrubs.
- Leaves submerged land and a near-water height margin uncluttered.
- Rejects OSM water, forest, urban, field, road and building raster cells,
  including nearby mask boundaries.
- Rejects candidates too close to *already placed* world objects, and
  maintains clearance between the added shrubs.
- Keeps generated roads, existing forest, terrain materials and heights
  untouched, and registers the added O.pbo assets in the regular model
  usage/catalogue pipeline.

Tests: `tests/test_nogova_terrain_clutter.py`.
