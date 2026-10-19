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
