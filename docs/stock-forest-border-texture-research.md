# Original OFP/CWA forest-floor border textures

The original `cain.wrp` (Kolgujev), `abel.wrp` (Malden),
`eden.wrp` (Everon) and `Noe.pbo / noe\\noe.wrp` (Nogova) were
inspected using Island Studio's texture-location-map export. Each export
contains a north-up 256 × 256 ground-texture grid at 50 m per cell, original
WRP texture paths and directional neighbor counts.

**Key discovery: forest borders are not a single ground texture.** They are
directional four-quadrant blend tiles. The names consist of four two-character
base names (in **SW, SE, NW, NE** order). This ordering is corroborated by the
actual neighbor locations in the exported texture grid.

| World | Solid forest floor | Distinct used forest-containing blend paths | Cells using blends | Dominant adjacent ground |
| --- | --- | ---: | ---: | --- |
| Nogova | `o\\j1.paa` | 80 | 2,046 | `t1`, `b1`; less often `s2` |
| Everon | `eden\\pl.paa` | 63 | 2,041 | `tn`; less often `st` |
| Malden | `abel\\pb.paa` | 12 | 592 | `tt` |
| Kolgujev/Cain | `cain\\u2.paa` | 94 | 2,099 | `j9`; less often `t9`, `k5` |

## Straight borders against each island's dominant surrounding terrain

"Forest to west" means the **solid forest cell directly west** of the
transition is the named forest-floor texture. The other compass directions
follow the same convention. All of these named blend paths appear and are
actively used in their respective exported source WRP.

| World | Forest to west | Forest to east | Forest to north | Forest to south |
| --- | --- | --- | --- | --- |
| Nogova | `o\\j1t1j1t1.pac` | `o\\t1j1t1j1.pac` | `o\\t1t1j1j1.pac` | `o\\j1j1t1t1.pac` |
| Everon | `eden\\pltnpltn.paa` | `eden\\tnpltnpl.paa` | `eden\\tntnplpl.paa` | `eden\\plpltntn.paa` |
| Malden | `abel\\pbttpbtt.pac` | `abel\\ttpbttpb.pac` | `abel\\ttttpbpb.pac` | `abel\\pbpbtttt.pac` |
| Kolgujev/Cain | `cain\\u2j9u2j9.paa` | `cain\\j9u2j9u2.paa` | `cain\\j9j9u2u2.paa` | `cain\\u2u2j9j9.paa` |

There are also **corner, diagonal and multi-terrain transitions**, not
just the four straight-edge examples. For example, Nogova has
`o\\j1j1b1b1.pac` and `o\\b1j1b1j1.pac` where forest meets `b1`
ground. The most common Everon forest blends are
`eden\\pltnpltn.paa`, `eden\\tnpltnpl.paa`,
`eden\\plpltntn.paa`, and `eden\\tntnplpl.paa`.

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
   Current PR implementation using one fixed `e` path is incomplete for
   authentic stock edge blending.

Only texture **names and grid adjacency data** were examined. The original
PAA/PAC artwork is not included or redistributed.
