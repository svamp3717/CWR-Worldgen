# Original Nogova (Noe) underwater ground and water borders

## Source inspection

The user-provided Island Studio exports `nogova_textures(1).zip` and
`nogova_objects(1).zip` are taken from stock `Noe.pbo/noe\noe.wrp`.
The ground-texture atlas is 256 × 256 cells at 50 metres per cell,
or 12,800 × 12,800 metres. This change reads texture metadata and cell
positions, not P3D placements; existing models/objects are unchanged.

| Ground texture | Stock WRP cells | Function |
| --- | ---: | --- |
| `o\ps.paa` | 20,057 | Offshore/seabed ground |
| `o\pt.paa` | 4,964 | Intermediate coastal ground |
| `o\t1.paa` | 14,838 | Main ordinary inland ground |

The original 1,020 perimeter cells include 903 `ps` and 31 `pt`
full tiles, and the rest are original `ps`/`pt` directional tiles.
This supports the offshore/coastal identification without assuming the
plain texture names reveal artwork contents.

There are **14 genuine ps↔pt directional PAC textures** and **14
genuine pt↔t1 directional PAC textures** in `texture-paths.csv`. Both
are complete sets of all mixed corner combinations. The original WRP
also contains some three-material and rare ps↔t1 texture filenames.
They are deliberately not used here: mixing unrelated source paths or
inventing a complete direct ps↔t1 family would create seams.

Using the same source-WRP geographic NE/NW/SE/SW convention validated for
the other islands, filename tokens require a **180° rotation**. On
paired-family original atlas cells:

| Original WRP family | Unrotated corner mismatches | Corrected 180° rotation |
| --- | ---: | ---: |
| ps/pt | 8,343 / 108,058 | 4 / 108,058 |
| pt/t1 | 3,775 / 71,192 | **0 / 71,192** |

The four remaining stock ps/pt mismatches arise in the original atlas:
the generator itself uses a globally shared vertex material grid and
tests all assigned neighboring quartet edges for exact continuity.

## Implementation

The generated **Nogova** ground-texture preset references:
- Offshore water: `o\ps.paa`.
- Underwater coast band: `o\pt.paa`.
- Land-facing/normal dry ground: `o\t1.paa`.

Ground texture assignment is based on **final WRP terrain elevation**:
the ps/pt directional and pt/t1 directional borders are applied only
to water-background cells at or below the configured sea level.
Dry shoreline and mapped beaches use `t1`, never `ps` or `pt`.

A two-cell underwater `pt` buffer separates the ps/pt quartet family
from the pt/t1 family. The geographic tile corner materials are derived
from a shared vertex lattice. No assigned tile can require both ps and
t1 at its corners, so source-family artwork never overlaps incorrectly.
An isolated submerged one-cell water background may use `t1` ground
if no existing stock sand/water texture can meet all dry corners. The
cell stays underwater; no height, map coast, or object placement changes.

Protected forest floor, farmland, rock and road textures retain their
existing slots. Other ground presets are not modified by Nogova's
three-material compositor. All texture artwork remains in the game's
installed original PBO; generated WRPs reference existing filenames.

The test suite `tests/test_nogova_water_blends.py` covers complete
source quartet names, both transitions, above/below water heights,
continuous straight/curved/narrow/random coasts, protected ground,
slot exhaustion and a written 4WVR round-trip.
