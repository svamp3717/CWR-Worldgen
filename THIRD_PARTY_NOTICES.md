# Third-party notices

## OSM House Modeler

This branch adapts procedural building-detail concepts and geometry behavior from
[`svamp3717/osm-house-modeler`](https://github.com/svamp3717/osm-house-modeler),
version 0.13.6.

The upstream project declares its license as MIT in `pyproject.toml`. The adapted
CWR implementation keeps CWR's existing GPL-3.0-or-later package licensing and
preserves the upstream source/version reference in
`src/cwr_worldgen/osm_house_modeler_upgrade.py`.

The integration intentionally retains CWR's OFP/CWA-specific P3D/MLOD, Geometry,
Roadway, Memory and Paths LOD implementation instead of copying the standalone
OBJ/viewer pipeline. It ports the architectural-detail behavior needed by the
world generator: entrance stairs for exterior-only models, porches/canopies,
balconies (including visual-only balconies on enterable variants), chimneys,
gutters and downspouts. Enterable variants keep CWR's collision-aware openings,
floors, stairs and animated entrance doors; no dedicated balcony door is required.

The 24 regional house-style profiles and the base set of 249 country profiles
originate from osm-house-modeler commit
`74c8049466875dc94409493bc77bfcad56e38a8d`. CWR extends every country/context
with explicit barn, shed, garage, warehouse, hangar and industrial wall/roof
material pools under revision `2026-09-country-utility-materials-v1`. These local
country-profile extensions preserve explicit OSM `building:material` and
`roof:material` tags as authoritative inputs.

The upstream country/region style-selection engine and procedural material texture
generator remain pinned to the same source commit. The vendored texture
implementation is stored as `src/cwr_worldgen/osm_house_modeler_textures.py`; CWR
wraps its wall, roof, foundation, window, door and secondary-material pixels in a
CWA/PAA bridge while retaining the existing MLOD asset pipeline.
