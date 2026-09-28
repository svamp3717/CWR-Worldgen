# SEB Ia Drang forest preset measurements

Source analysed: `sebnam_ia_trang.pbo!seb_iatrang.wrp`, supplied for the
Vietnam forest-preset work.

The source world is an OFP/CWA `OPRW` version 2 world with a 256 x 256,
50 metre terrain grid (12.8 km x 12.8 km). It contains 115,881 object
placements referencing 79 model paths.

This document records the measurements used by the `vietnam` forest profile.
It does **not** copy Ia Drang's forest geometry. CWR-Worldgen still uses the
forest polygons from the selected source data; the preset borrows the source
world's vegetation family, block scale, density and mix.

## Formal jungle cells

The OPRW forest bit is set on 7,752 terrain cells:

- cell size: 50 m x 50 m
- formal jungle area: 19.38 km2
- `sebnam_obj\sebles_su_ctver_pruhozi.p3d`: 7,760 placements
- unique 50 m cells containing that jungle block: 7,752
- duplicate block placements: 8

Every forest-flagged cell therefore has the SEB jungle square. This is why the
Vietnam profile keeps the primary forest lattice at 50 m and uses
`sebles_su_ctver_pruhozi.p3d` as both the normal and steep source-family
forest block. Worldgen still applies its normal road clearance and terrain-fit
safety instead of reproducing source clipping/floating verbatim.

## Free trees inside formal jungle

There are 2,053 additional SEB tree objects inside the 7,752 formal jungle
cells, or about 105.93 free trees/km2.

| Model | Count | Share |
| --- | ---: | ---: |
| `sebstr borovice horska.p3d` | 639 | 31.1% |
| `sebstr_liskac.p3d` | 557 | 27.1% |
| `sebstr_fikovnik.p3d` | 389 | 18.9% |
| `sebstr osika.p3d` | 265 | 12.9% |
| `sebstr_fikovnik2.p3d` | 203 | 9.9% |

A regular point scatter matching that density has spacing

`sqrt(1,000,000 / 105.93) = 97.16 m`.

The preset therefore uses a 97 m free-tree lattice and a 20-slot weighted tree
pool of approximately 30/25/20/15/10 percent. The main free-tree scatter,
road-cut replacements, mapped OSM trees and gap infill all use this SEB family.

## Free shrubs inside formal jungle

There are 9,100 additional SEB shrub objects inside formal jungle, or about
469.56 shrubs/km2.

| Model | Count | Share |
| --- | ---: | ---: |
| `sebelekrovi2.p3d` | 5,428 | 59.6% |
| `sebstr krovisko vysoke.p3d` | 1,482 | 16.3% |
| `sebkrovi_long.p3d` | 1,064 | 11.7% |
| `sebstr_fikovnik_ker.p3d` | 693 | 7.6% |
| `sebkrovi4.p3d` | 433 | 4.8% |

Worldgen's three reusable undergrowth carriers contain 6, 5 and 4 visible
proxies, averaging five, and the placement pass deliberately keeps every second
valid carrier. The equivalent regular lattice is therefore

`sqrt(0.5 * 1,000,000 * 5 / 469.56) = 72.97 m`.

The preset uses 73 m undergrowth spacing. Steep-hill and border passes provide
additional irregularity rather than replacing this measured interior density.

## Wet vegetation

`sebnam_obj\sebker rakosi.p3d` occurs 6,779 times across the complete source
world, including 363 placements inside formal jungle. The Vietnam profile uses
that model for mapped wetland reeds and generated ditch/reed carriers instead
of introducing the default European/Resistance reeds.

## Runtime dependency

The vegetation models above are referenced from the `sebnam_obj` namespace.
The generated terrain does not redistribute those third-party P3Ds, so a world
using the Vietnam preset requires the user's `sebnam_obj.pbo` at runtime. The
GUI and generated terrain ReadMe call this dependency out explicitly.
