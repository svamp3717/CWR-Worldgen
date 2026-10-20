# Experimental generated-world OPRW v3 binarization

The GUI exposes **Advanced generation, cache and PBO settings → Build behavior →
Experimental: binarize generated world as OPRW v3 (50 m grid only)**.
This checkbox is off by default and persists between GUI sessions, just like
the procedural-building ODOL option.

You can also pass `--binarize-world` to the `milestone9` CLI command.

## Outputs

When enabled, Worldgen generates and validates the original, editable 4WVR
world normally. After generation and reproducibility checks, it:

1. Writes an independent `<world>.binarized.wrp` (OPRW v3) in the build output.
2. Temporarily stages that WRP in the world source directory.
3. Builds a new game PBO containing the OPRW file, restoring the original
   4WVR to its source directory even when packing fails.
4. Replaces the final game PBO **only after packing succeeds**. If a pack fails,
   the previous archive is preserved and the command reports an error.
5. If a mod deployment is requested, deploys the newly binarized game PBO.

The `build-manifest.json` records whether binarization was enabled. Source
files are left unbinarized, so Island Studio can still edit the generated WRP.

## Compatibility and limitations

* The source grid **must be 50 metres per cell**. The CWR-CE engine currently
  forces `SetLandGrid(50)` when loading OFP/CWA OPRW v2/v3, independent of
  `CfgWorlds.LandGrid`. Enabling this on a 25 m world would scale the island
  incorrectly, so the builder refuses it before doing expensive generation.
* Output targets **CWA OPRW v3** only. It does not produce ArmA-era OPRW.
* This experimental writer preserves terrain heights, texture-slot selection,
  texture names, world dimensions, object IDs, model paths and transforms.
* Source `4WVR` files do **not** contain OPRW's serialized geography, random,
  or sound maps. The converter fills these with zeros. This may affect ambient
  sounds, vegetation/water-related terrain flags, or other engine systems
  until verified against stock CWA behavior. Do not treat it as a production
  replacement for the game's native binarizer without in-game testing.
* The writer uses bounded-time SSCompress-compatible tokens; there is no
  guarantee the resulting WRP will be smaller than a dedicated binarizer's
  optimized output, and **there is no measured FPS improvement**.
* The conversion is tested using independent WRP readers and synthetic
  terrain/object fixtures. Test a complete generated island in CWR-CE before
  shipping a release built with this option.

The option is separate from **binarize procedural buildings as ODOL v7**.
Enabling one does not enable the other.

## Separate config.cpp binarization

The same Advanced **Build behavior** section also has an independent
**Experimental: binarize addon config.cpp as CWA RAP v4 config.bin**
checkbox. It works with or without WRP binarization, preserves editable
`config.cpp` outside the game PBO, and avoids duplicate PBO repacks when
both switches are enabled. See
[CONFIG_BINARIZATION_EXPERIMENT.md](CONFIG_BINARIZATION_EXPERIMENT.md).

## Matching generation across packaging variants

In **Advanced generation, cache and PBO settings → Build behavior**, enter an
optional **Manual world seed (blank = automatic)**. For example, `314159`.
The same seed and the same normalized source data, island identity, terrain
parameters and procedural settings reproduce seeded tree, forest, fence,
undergrowth and surface decisions, regardless of whether you package the
runtime as editable 4WVR, experimental OPRW v3, or with RAP `config.bin`
and/or ODOL buildings. Packaging-only switches do not salt the seed.

CLI: `--world-seed 314159` on `milestone9`. The field is saved with the
GUI preferences and the effective seed is recorded under `world_seed` in
`build-manifest.json`. Leave it empty to retain the previous, automatically
derived `milestone9:<normalized-source-fingerprint>` seed. Both numeric and
text seeds are accepted (up to 128 characters, no control characters).

**Scope:** a seed controls procedural choices, not the imported geography,
heightmap, object placement rules, terrain grading parameters, P3D model
availability or engine runtime behavior. Different source data, configuration
options, Worldgen versions or island identity may still produce different
WRP files. In particular, a matching seed does **not** fix OPRW-only
vegetation grounding or rendering differences: those require separate
binarizer/engine checks. Compare the original loose 4WVR outputs or use
Island Studio's **Compare World** report to verify alignment.
