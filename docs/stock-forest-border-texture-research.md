# Original OFP/CWA forest-floor border textures

The original `cain.wrp` (Kolgujev), `abel.wrp` (Malden),
`eden.wrp` (Everon) and `Noe.pbo / noe\\noe.wrp` (Nogova) were
inspected using Island Studio's texture-location-map export. Each export
contains a north-up 256 × 256 ground-texture grid at 50 m per cell, original
WRP texture paths and directional neighbor counts.

**Key discovery: forest borders are not a single ground texture.** They are
directional four-quadrant blend tiles. The names consist of four two-character
base names. The original WRP texture-location maps establish where each
filename is used, but they do NOT prove how the artwork is oriented within
the file. Rechecking the actual artwork in Island Studio against generated WRPs
shows **both Nogova PAC and Everon PAA forest blends display corners
rotated 180 degrees** relative to their names. Worldgen continues to use
NE/NW/SE/SW geographic masks, but reverses filename masks for those two
stock artwork families. WRP texture-grid rows still increase northward.

| World | Solid forest floor | Distinct used forest-containing blend paths | Cells using blends | Dominant adjacent ground |
| --- | --- | ---: | ---: | --- |
| Nogova | `o\\j1.paa` | 80 | 2,046 | `t1`, `b1`; less often `s2` |
| Everon | `eden\\pl.paa` | 63 | 2,041 | `tn`; less often `st` |
| Malden | `abel\\pb.paa` | 12 | 592 | `tt` |
| Kolgujev/Cain | `cain\\u2.paa` | 94 | 2,099 | `j9`; less often `t9`, `k5` |

## Straight borders against each island's dominant surrounding terrain

"West edge" means the transition is on the western edge of the forest,
with **solid forest immediately east** of the transition cell. In contrast,
the eastern edge has solid forest to its west. The north edge has forest
south of it; the south edge has forest north of it. All names below
actually occur in the source islands.

| World | West edge (forest east) | East edge (forest west) | North edge (forest south) | South edge (forest north) |
| --- | --- | --- | --- | --- |
| Nogova | `o\\t1j1t1j1.pac` | `o\\j1t1j1t1.pac` | `o\\j1j1t1t1.pac` | `o\\t1t1j1j1.pac` |
| Everon | `eden\\tnpltnpl.paa` | `eden\\pltnpltn.paa` | `eden\\plpltntn.paa` | `eden\\tntnplpl.paa` |
| Malden | `abel\\pbttpbtt.pac` | `abel\\ttpbttpb.pac` | `abel\\ttttpbpb.pac` | `abel\\pbpbtttt.pac` |
| Kolgujev/Cain | `cain\\u2j9u2j9.paa` | `cain\\j9u2j9u2.paa` | `cain\\j9j9u2u2.paa` | `cain\\u2u2j9j9.paa` |

There are also **corner, diagonal and multi-terrain transitions**, not
just the four straight-edge examples. For example, Nogova has
`o\\j1j1b1b1.pac` and `o\\b1j1b1j1.pac` where forest meets `b1`
ground. The most common Everon forest blends are
`eden\\pltnpltn.paa`, `eden\\tnpltnpl.paa`,
`eden\\plpltntn.paa`, and `eden\\tntnplpl.paa`.

## Confirmed Nogova PAC artwork orientation (generated test 5)

The Island Studio viewport showed `o\\j1j1t1t1.pac` with its
forest-colored half to the **south**, not north; and
`o\\j1t1j1t1.pac` with forest toward the **west**, not east.
Quadrant-color comparisons of the uploaded WRP against the actual screenshot
favored a 180-degree transform over all other square symmetries. The earlier
table, based only on filenames and adjacency, was insufficient to establish
the intrinsic orientation of the stock PAC art.

**Worldgen now converts the requested geographic mask to the opposite
Nogova filename mask.** Island Studio continues to render all stock and
generated ground texture artwork identically; the original WRP is unchanged.

## Confirmed Everon PAA artwork orientation (generated test 9)

In the `wg_foresttest9.pbo` Island Studio screenshot, the same 180-degree
mismatch appears in Everon's `eden\\pl/tn` quartet textures. By
locating the 2×2 sports-pitch overlay to align screenshot pixels with
WRP cells, then sampling 55 unobscured Everon blend cells, comparing
each tile's four forest-colored corners with its eight possible
orientations gave:

- Original filename interpretation: **32.7%** correctly classified corners.
- 180-degree opposite filename interpretation: **95.5%** correctly
  classified corners, the best of the eight transforms.

The original Everon texture table lacks filename quadrant mask 9
(NE+SW). With its artwork rotated 180 degrees, the unavailable
**geographic** mask is 6 (NW+SE). Shared-corner placement removes a
shared SE vertex if that configuration appears, rather than emitting
an absent stock PAA or drawing a mismatched fallback.

This changes **only Worldgen's Everon filename selection**. The uniform
Island Studio renderer, existing Everon texture assets, and Nogova
filename selection remain unchanged. Tan farmland squares in the
screenshot come separately from `Eden\\zbh.paa` and are outside
this forest-border correction.

## Implications for the optional forest-undergrowth checkbox

1. **The provisional `o\\j2.paa` forest-border path must not be treated
   as verified.** It does not occur in Nogova's original WRP texture
   table, and the table instead names `.pac` quartet blend paths.
2. A single fixed "forest edge" slot for the entire world will not match
   the original islands. **Choose a blend on each edge cell** according to
   which sides/corners border forest and which stock ground tile surrounds
   it. Different orientations require different paths.
3. Populate the WRP texture table with the selected valid blend paths,
   selecting only the variants needed for that map. Preserve the existing
   per-island extension: Nogova/Malden blends use `.pac` while
   Everon/Cain blends use `.paa`.
4. There are rare three-or-more-terrain blend names, and not every possible
   orientation appears in every source WRP. Avoid inventing an asset path
   without source/asset confirmation; fall back conservatively.
5. Keep this feature **opt-in** and independent of 3D forest undergrowth.
   The implementation now assigns verified directional stock quartet blends
   per cell, including corners, based on the OSM forest raster. It preserves
   nonforest materials, adjusts the WRP texture table, and uses a verified
   nearest-quadrant blend if a rare diagonal has no matching stock tile.
6. These families use the dominant exterior land tile (`t1`, `tn`,
   `tt`, or `j9`). More complex stock transitions against alternate
   ground such as Nogova `b1` or Cain `t9` are documented here but are
   not yet selected automatically. Do not conflate the first implementation
   with full restoration of every multi-terrain transition in the originals.

Original island texture **names and grid adjacency data** establish
verified source paths; screenshots of stock artwork in Island Studio
also establish the Nogova and Everon visible corner orientation. The
original PAA/PAC artwork is not included or redistributed.
