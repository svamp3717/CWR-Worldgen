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
OBJ/viewer pipeline. It ports entrance stairs, balconies, chimneys, gutters and
downspouts. Porch/canopy metadata is retained from the source profiles, but CWR
intentionally suppresses porch geometry because it does not read well in-game.
Enterable variants keep CWR's collision-aware openings, floors, stairs and animated
entrance doors; no dedicated balcony door is required. Generated CWR door textures
are intentionally solid and do not render glazed/window panels.

The upstream source commit `74c8049466875dc94409493bc77bfcad56e38a8d` provides
24 regional house-style profiles and the base set of 249 country profiles. CWR
deliberately bundles 23 regional profiles: upstream `24_sweden.json` is excluded so
Sweden is defined solely by `country_styles/SE_Sweden.json`, layered on the
`northern_europe` regional baseline like other countries.

CWR extends every country/context with explicit barn, shed, garage, warehouse,
hangar and industrial wall/roof material pools under revision
`2026-09-country-utility-materials-v1`. It also stores explicit ordinary wall
material and facade-colour distributions under revision
`2026-09-global-country-visual-balance-v1`; hand-tuned country weights take
precedence, while otherwise the country's own curated list receives a balanced
equal-weight baseline. Explicit OSM `building:material`, `building:colour` and
`roof:material` tags remain authoritative.

The upstream country/region style-selection engine and procedural material texture
generator remain pinned to the same source commit. The vendored texture
implementation is stored as `src/cwr_worldgen/osm_house_modeler_textures.py`; CWR
wraps its wall, roof, foundation, window, door and secondary-material pixels in a
CWA/PAA bridge while retaining the existing MLOD asset pipeline.
