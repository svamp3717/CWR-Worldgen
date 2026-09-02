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
balconies on non-enterable variants, chimneys, gutters and downspouts. Enterable
variants keep CWR's collision-aware openings, floors, stairs and animated doors.
