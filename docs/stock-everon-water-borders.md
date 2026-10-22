# Original Everon (Eden) water-ground borders

## Source evidence

The user's Island Studio export of **the original `eden.wrp`** contains
`texture-ids.png`, `texture-paths.csv`, `texture-runs.csv`,
`texture-adjacencies.csv` and `metadata.json`. The texture-layer raster
is **256 × 256 cells**, **50 m per cell**, world width **12,800 m**,
south-to-north game Z. The ZIP is a research input and is not packaged into
Worldgen or redistributed with this repository.

| Original ground texture | Used cells | Meaning in the coastline |
| --- | ---: | --- |
| `eden\ps.paa` | 42,634 | Offshore/underwater seabed |
| `eden\sz.paa` | 1,500 | Sandy shore |
| `eden\tn.paa` | 9,115 | Main inland grass |

The export also names **74 distinct directional shoreline quartet tiles**:

- **14** containing both `ps` and `sz`
- **14** containing both `sz` and `tn`
- **13** containing both `ps` and `tn`
- **33** containing all three codes

The one absent `ps/tn` quartet is not guessed. Every selected directional
path is checked against the original WRP path table before being added to a
new WRP. Original texture artwork is loaded from the installed game archives.

## Corner orientation, verified using original WRP cell placement

A quartet filename encodes its four graphic regions in nominal
NE/NW/SE/SW order, but the geographic corners rendered by the stock artwork
are rotated **180°** relative to that text.

Checking all adjoining original Eden texture cells with a supported
`ps/sz/tn` texture against the geographic corner arrangement yielded:

| Filename interpretation | Shared-corner mismatches |
| --- | ---: |
| Unrotated | 16,825 / 227,376 |
| **180° rotation** | **0 / 227,376** |

For example, geographic water on a tile's west side and sand on its east
side corresponds to **`eden\psszpssz.paa`**, while geographic sand
west and grass east corresponds to **`eden\sztnsztn.paa`**.

## Generator integration and limitations

The Everon preset now uses `ps` for water, `sz` for wet/dry
shoreline and mapped beaches, and `tn` for ordinary ground. This applies
to the legacy and Milestone 9 semantic palettes.

When writing the WRP, Worldgen assigns a *single shared material per corner*
from the neighboring eligible tiles, then selects the stock quartet whose
artwork displays those four geographic corners. It runs after the optional
forest-border overlay and does not alter existing special textures such as
forest floor, road underlays, rock and farmland. Unknown quartets, protected
corner intersections and exhausted WRP texture tables keep their original
cells instead of constructing unsafe filenames.

The generated coast still follows the **generated world's own shoreline**.
This recreates Everon's ground-texture *family and border orientation*, not
the physical shoreline outline or texture-placement coordinates of original
Everon. Sea-level heights, water masks and object placements are unchanged.

The tests in `tests/test_everon_water_blends.py` cover the exported stock
filenames, directional edges, seam-free adjacent tiles, protected land
materials and 4WVR texture table output.
