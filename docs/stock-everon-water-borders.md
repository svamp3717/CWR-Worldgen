# Everon sandy ps-to-grass tn shoreline borders

## Why the texture changed

The original Eden WRP export (`everon_textures.zip`) contains a 256 × 256,
50-metre ground-texture grid with the original game texture paths.
However, the WRP names alone do not establish the artwork's appearance.
The previously selected `eden\sz.paa` is **stone**, not sandy shore.

The Everon Classic preset now uses:

| Surface | Actual game texture |
| --- | --- |
| Water / sandy seabed | `eden\ps.paa` |
| Wet shoreline / dry beach / mapped beach | `eden\ps.paa` |
| Ordinary inland grass | `eden\tn.paa` |

The original WRP has **13 actual ps/tn directional quartet assets**.
These follow the coastline's boundary between sand and grass rather than
painting a stone-colored ring. The geographic mask is rotated 180 degrees
relative to the original filename order. Examples:

- Sand west and grass east: `eden\pstnpstn.paa`
- Grass west and sand east: `eden\tnpstnps.paa`
- Sand south and grass north: `eden\pspstntn.paa`

The one unavailable original ps/tn diagonal (NE+SW grass, mask 9)
is resolved by modifying a *shared* SW vertex in all adjacent cells,
not by inventing a filename or leaving mismatched corners.

## Scope

This applies to both legacy and Milestone 9 **Everon** ground-texture
presets. The border compositor runs after forest blends and keeps roads,
farmland, rock and forest ground protected. Original game artwork is not
redistributed, and only WRP-confirmed paths may be referenced.
The generated world keeps its own coast shape, sea elevations and object
placement; other island ground presets are unchanged.

The relevant regression tests live in `tests/test_everon_water_blends.py`.
