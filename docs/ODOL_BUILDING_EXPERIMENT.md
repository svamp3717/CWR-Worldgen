# Experimental CWA ODOL v7 procedural buildings

Branch: `feature/optional-odol-buildings`

## Enable

In the Worldgen GUI open **Advanced > Build behavior** and check
**Experimental: binarize procedural buildings as ODOL v7 (Python)**.

Equivalent CLI option for `milestone9`:

```console
python -m cwr_worldgen milestone9 ... --odol-buildings
```

The option defaults **off**. It affects only procedural building P3Ds written by
Worldgen, not road models, stock assets, textures, WRP placements or cleanup.

## Implementation

The first stage creates the usual MLOD building with the existing
`procedural_buildings.py` serializer. The optional Python postprocessor then
translates that MLOD to the older **ODOL version 7** wire format and atomically
replaces the model within the temporary source tree before the PBO is packaged.

`cwr_worldgen/odol_v7.py` implements this directly in Python:

- ODOL v7 shape and LOD headers and metadata
- vertex UVs, normals, point flags and face winding
- texture tables, aligned face records and contiguous draw sections
- named selections, properties and Geometry/Mass/Memory/Roadway/Paths LOD indices
- SSCompress-compatible LZSS streams (literal-only, with legacy checksum)

The writer uses documented structures and examples in the open source
[CWR-CE engine](https://github.com/ofpisnotdead-com/CWR-CE), especially
`engine/Poseidon/Graphics/Rendering/Shape/ShapeFile.cpp`,
`engine/Poseidon/IO/Streams/SsCompress.cpp`,
`engine/Poseidon/Asset/Formats/P3D/MLODLoader.hpp`, and
`tests/fixtures/p3d/generate_animated_morph_odol.py`.

This initial implementation favors correctness over the smallest binary size:
its LZSS output uses literal packets, and it duplicates per-face vertices
rather than aggressively optimizing them.

## Experimental limitations

- Not yet confirmed in-game. Test a small world first, particularly doors,
  house interiors, shadows, geometry collision, pathfinding and distant LODs.
- Does not yet support proxy tags or morph-animation tags. Those models fail
  explicitly instead of silently losing important model features.
- Some older engine semantics (mass/component handling and object materials)
  may still differ from a production CWA/Oxygen binarizer.
- Do **not** infer an FPS gain simply because the P3D signature is ODOL.
  Use two builds of the same world and compare load times, memory and FPS.
- Conversion failures abort the build and preserve the source MLOD model;
  generated files are not mislabeled.

The ODOL model cache is intentionally separate from the MLOD creation step.
Disabling the checkbox preserves the existing MLOD source/catalogue behavior.

## Validation before merging

1. Run `pytest tests/test_odol_v7.py` and the project's full test suite.
2. Generate identical small worlds with the checkbox off and on.
3. Verify `MLOD` vs `ODOL`/v7 signatures for generated building models.
4. Compare exterior walls/roofs, interiors, doors, stairs, Geometry and
   Roadway collision, AI pathfinding, shadows and distance LOD switching.
5. Record baseline and test load time, memory, and average/low FPS.
6. Check the source PBO still includes every texture and is clean on restart.

Do not merge the option into the main branch before completing these checks.
