# Everon underwater-only `ps` textures

The Everon Classic preset uses the original `eden\\ps.paa` ground
texture **only where final WRP elevation is at or below the configured
sea level**. The texture is not painted on exposed beaches or dry
shoreline cells. This restriction uses the world height samples, rather than
the OSM water raster or an unreliable `w` material category.

| Location | Everon ground texture |
| --- | --- |
| Submerged terrain | `eden\\ps.paa` |
| Submerged boundary against land | Original directional `ps`/`tn` quartet, on the **underwater cell only** |
| Above-water beach / shore / grass | `eden\\tn.paa` |
| Roads, forest, farmland, rock | Existing protected ground tiles |

Everon's 13 verified directional `ps`/`tn` PAA filenames come from the
original `eden.wrp` texture export. Geographic filename orientation is
rotated 180 degrees. Each shared vertex touching dry ordinary ground uses
the `tn` corner, allowing the underwater blended cell to meet an
unmodified `tn` dry cell. Unsupported diagonal combinations are resolved
at shared vertices without inventing a missing PAA.

This rule is applied in legacy and Milestone 9 Everon texture palettes
through the common WRP texture-table compositor, including validation and
deterministic regeneration. Water heights, coast shapes, gameplay objects
and all other island texture presets remain unchanged. No game artwork is
repackaged into the generated island.
