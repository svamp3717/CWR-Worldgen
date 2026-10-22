# Original Abel and Cain underwater ground and coast borders

## Source exports

The original `abel.wrp` (Malden) and `cain.wrp` (Kolgujev) Island
Studio texture exports have 256 × 256 cells at 50 metres per texture cell.
The corresponding object exports are independent of terrain material slots;
no object positions, orientations or model classes are changed.

| Island | Offshore original full tile | Cells | Map-perimeter cells | Inland background |
| --- | --- | ---: | ---: | --- |
| Abel / Malden | `abel\pi.paa` | 43,286 | 1,020 / 1,020 | `abel\tt.paa` (14,650 cells) |
| Cain / Kolgujev | `cain\l4.paa` | 52,209 | 1,020 / 1,020 | `cain\j9.paa` (3,921 cells) |

The original mixed-corner WRP texture tables contain **14 Abel
`pi`/`tt` quartet PAC paths** and **12 Cain `l4`/`j9` quartet
PAA paths**. Unlike Abel, Cain does *not* have either opposing diagonal
mixed-corner arrangement. The terrain compositor uses only those verified
filenames.

The original North-up cell atlases verify the filename-to-world
**180-degree quadrant rotation**. Across shared geographic corners where
both adjacent cells have two-material family texture paths:

| Source WRP family | Original filename order | 180-degree rotation |
| --- | --- | --- |
| Abel `pi` ↔ `tt` | 3,892 mismatches / 228,396 corners | **0 / 228,396** |
| Cain `l4` ↔ `j9` | 786 mismatches / 219,096 corners | **0 / 219,096** |

Cain also has a separate `l4`/`t9` transitional family in its original
WRP (13 mixed-corner tiles). `t9` is not the generated preset's normal
ground background; Worldgen intentionally keeps the `j9` dry-ground preset
and chooses the available `l4`/`j9` quartet family for this use case.
Original texture families must not be confused or mixed indiscriminately.

## Generator behavior

Only `w` (water/seabed) ground materials map to the new offshore
base texture, and **only where the FINAL terrain elevation is at or below
the user's water level**. Other above-water textures retain their normal
`tt` or `j9` ground assignment. Directional borders are applied to
the submerged side only, with shared dry-ground corners joining the normal
dry texture without a hard 50-metre square. Above-water areas, roads,
forests and special land-use materials are not repainted.

A water-classified cell that stays physically above the plane is changed
back to the ordinary land background. When an isolated submerged 50-metre
tile has four land corners, the original stock family cannot paint a
seamless mixed-corner water tile there; the ground falls back to the
ordinary land texture. Terrain heights, map water semantics and model
placements are unchanged.

This work reuses the normal WRP export and reproducibility/validation
texture-table creation code. The stock textures are not copied or
redistributed; the game must provide its usual original PBOs.

## Other islands

Everon/Eden is handled by the existing height-aware `ps`/`tn` composer.
Nogova/O remains unchanged pending the original Nogova WRP texture atlas:
the user mentioned a Nogova upload, but it was not present in the files
available for this review, so no exact sea/grass quartet family could be
validated from its original texture grid. No undocumented filenames or
unsourced textures have been substituted.
