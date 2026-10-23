# Stock terrain clutter: Eden, Abel, Cain and Nogova

## Source evidence

Three user-supplied Island Studio archives were analyzed from their
original OFP/CWA WRP object exports:

- `everon_objects.zip` from **Eden/Everon**, 56,740 WRP objects
- `malden_objects.zip` from **Abel/Malden**, 103,114 WRP objects
- `kolgujev_objects.zip` from **Cain/Kolgujev**, 70,770 WRP objects

The `object-models.csv` file in each archive provides the original model
path and exact appearance count; `object-placements.csv` provides
positions used only for **analysis**, not for generated-world placement.
No original object coordinates or terrain are copied.

We select small stock foliage/grass/sapling models and preserve their
**original *relative* frequency** as the generator's selection weights:

| Island | Selected models | Source small-object count | Visually dominant families |
| --- | ---: | ---: | --- |
| Eden | 14 | 17,805 | `krovi2/3/4`, small spruce, several grass tufts |
| Abel | 10 | 44,560 | Grass tufts, `str_fikovnik_ker`, `ker listnac`, bush types |
| Cain | 8 | 27,423 | Small spruce, grass tufts, `krovi2/3` |
| Nogova | 11 | 122,670 | DD bushes and `smrk_maly` |

These are model *weights*, not target counts. Large conifers, mature
trees, full forest polygons, road/farm objects, buildings, and wetland
reeds continue to belong to their separate placement systems.

Exact source counts and supported virtual paths are documented in
`src/cwr_worldgen/stock_terrain_clutter_profiles.py`; Nogova remains
in `src/cwr_worldgen/nogova_terrain_clutter.py`.

## Checkbox and tree-style gate

The existing GUI **Common choices** control is now called
**Original island terrain clutter (matching trees)**. The saved field
(`nogova_terrain_clutter`) remains intact for compatibility with prior
profiles, and the checkbox now defaults to **checked** for new worlds. Old
saved profiles that explicitly disabled clutter remain disabled.

When enabled (the new default), it selects ONE original-model pool based on the
actual forest-block *and* single-tree models in the active vegetation
preset:

- **Everon** trees → Eden pool
- **Malden** trees → Abel pool
- **Kolgujev** trees → Cain pool
- **Nogova Resistance leaf or pine** → Nogova pool
- **Vietnam, None, or nonmatching custom tree models** → no new clutter

The ground texture selector is completely independent. For example,
Nogova ground textures paired with Everon trees use the **Eden** model pool,
never DD Resistance bushes. Selecting just a matching texture cannot
activate the clutter. Direct CLI usage applies the same model-based gate.

CLI builds enable matching-tree clutter by default; pass
`--no-terrain-clutter` (or `--no-nogova-terrain-clutter`) to switch it
off. Explicit `--terrain-clutter` and `--nogova-terrain-clutter` flags
remain accepted. The former `--nogova-clutter-max-objects N` argument
is accepted for old saved build scripts but **ignored**; it cannot cap
or disable clutter. Uncheck the GUI box to disable it.

## Safety, distribution and compatibility

The shared deterministic object placer from the Nogova feature stays
conservative: 45-metre jittered candidate strata and about 48%
preliminary acceptance before safety filtering. There is **no numerical
maximum** for additional terrain clutter. Large maps can contain more
objects than small maps, subject to eligible natural terrain, collision
clearance, and density gates. Model selection is weighted by each
island's measured export counts. Separate hashed streams prevent the
four island palettes from accidentally sharing identical positions.

The pass ignores OSM water, forest interiors, urban, farmland, roads and
building areas, checks nearby boundaries, rejects submerged or steep
ground, and avoids objects already placed by the regular generator.
It changes no existing roads, world heights, terrain textures, forest
blocks or object placements, and only references game-provided original
P3Ds rather than repackaging them.

The option participates in Worldgen's normal object catalogue and
deterministic regeneration verification, using the existing placement
cache-key settings.

Regression tests:
`tests/test_stock_terrain_clutter.py` for Eden, Abel and Cain, and
`tests/test_nogova_terrain_clutter.py` for the original Nogova behavior.
