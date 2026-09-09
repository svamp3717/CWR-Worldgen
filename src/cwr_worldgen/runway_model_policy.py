# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate rotatable runway surface models from stock Resistance artwork.

RVW4 terrain cells only store a texture-table index; they do not carry a per-cell
UV rotation.  The Nogova runway artwork is cardinally authored, so writing
``o\\runtr_d.paa`` directly into arbitrary OSM runway cells necessarily leaves the
painted runway pointing the wrong way whenever the OSM bearing differs from the
stock island.

Use a very thin generated MLOD surface instead.  The P3D can rotate with the OSM
runway object and can rotate/repeat the stock texture through UV coordinates.
Nothing from O.pbo is copied into the generated world: the model keeps external
references to the verified stock ``o\\runtr_*`` / ``o\\runpi_*`` textures.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import json
import math
import re

from .procedural_buildings import _Face, _Lod, _MLOD_HEADER, _write_lod, inspect_mlod
from .procedural_infrastructure import (
    InfrastructureAssetResult,
    ProceduralInfrastructureLibrary as _BaseInfrastructureLibrary,
)


GRASS_RUNWAY_MIDDLE_TEXTURE = r"o\runtr_d.paa"
GRASS_RUNWAY_START_TEXTURE = r"o\runtr_z.paa"  # stock Nogova west end
GRASS_RUNWAY_END_TEXTURE = r"o\runtr_k.paa"    # stock Nogova east end
DESERT_RUNWAY_MIDDLE_TEXTURE = r"o\runpi_d.paa"
DESERT_RUNWAY_START_TEXTURE = r"o\runpi_z.paa"  # stock desert south end
DESERT_RUNWAY_END_TEXTURE = r"o\runpi_k.paa"    # stock desert north end

RUNWAY_TEXTURE_TILE_METRES = 50.0
RUNWAY_MODEL_WIDTH_METRES = 50.0
_VISUAL_LOD = 1.0
_LAND_CONTACT_LOD = 2.0e15
_ROADWAY_LOD = 3.0e15
_RUNWAY_PATTERN = re.compile(
    r"^runway_(grass|desert)_w(\d+)_l(\d+)\.p3d$", re.IGNORECASE
)
_INSTALLED = False


def runway_family(profile: object) -> str:
    return "desert" if str(profile or "").strip().casefold() == "desert" else "grass"


def runway_texture_triplet(profile: object) -> tuple[str, str, str]:
    """Return (start, repeating middle, end) stock textures for one profile."""
    if runway_family(profile) == "desert":
        return (
            DESERT_RUNWAY_START_TEXTURE,
            DESERT_RUNWAY_MIDDLE_TEXTURE,
            DESERT_RUNWAY_END_TEXTURE,
        )
    return (
        GRASS_RUNWAY_START_TEXTURE,
        GRASS_RUNWAY_MIDDLE_TEXTURE,
        GRASS_RUNWAY_END_TEXTURE,
    )


def runway_model_path(
    world_name: str,
    profile: object,
    length_metres: float,
    *,
    width_metres: float = RUNWAY_MODEL_WIDTH_METRES,
) -> str:
    """Return a deterministic world-local model path for one straight runway."""
    width_dm = max(10, int(round(float(width_metres) * 10.0)))
    length_dm = max(10, int(round(float(length_metres) * 10.0)))
    return (
        rf"{world_name}\i\runway_{runway_family(profile)}_"
        rf"w{width_dm}_l{length_dm}.p3d"
    )


def _runway_face(
    texture: str,
    point_start: int,
    *,
    axial_repeats: float,
    family: str,
) -> _Face:
    """Map the stock cardinal texture so its runway axis follows local +Z."""
    repeat = max(1.0e-6, float(axial_repeats))
    if family == "grass":
        # runtr_* is authored west/east: its U axis is the runway axis.  Mapping
        # U onto model Z is the 90-degree UV rotation missing from RVW4 terrain.
        uvs = ((0.0, 0.0), (repeat, 0.0), (repeat, 1.0), (0.0, 1.0))
    else:
        # runpi_* is authored south/north: its V axis already is the runway axis.
        uvs = ((0.0, 0.0), (0.0, repeat), (1.0, repeat), (1.0, 0.0))
    return _Face(
        texture,
        tuple((point_start + index, 0, uv[0], uv[1]) for index, uv in enumerate(uvs)),
    )


def _runway_lods(family: str, width_metres: float, length_metres: float) -> tuple[_Lod, ...]:
    """Build one straight runway with stock start/middle/end texture sections."""
    family = runway_family(family)
    width = max(1.0, float(width_metres))
    length = max(1.0, float(length_metres))
    half_width = width * 0.5
    half_length = length * 0.5
    start_texture, middle_texture, end_texture = runway_texture_triplet(family)

    # The stock end-cap artwork occupies one 50 m terrain tile.  Long runways
    # keep exactly one cap at each end and repeat runtr_d/runpi_d through the
    # middle.  Tiny synthetic/test runways split the available length between the
    # two end caps instead of overlapping faces.
    if length <= RUNWAY_TEXTURE_TILE_METRES:
        sections = ((-half_length, half_length, middle_texture, 1.0),)
    elif length < RUNWAY_TEXTURE_TILE_METRES * 2.0:
        midpoint = 0.0
        sections = (
            (-half_length, midpoint, start_texture, 1.0),
            (midpoint, half_length, end_texture, 1.0),
        )
    else:
        first_end = -half_length + RUNWAY_TEXTURE_TILE_METRES
        last_start = half_length - RUNWAY_TEXTURE_TILE_METRES
        middle_length = max(0.0, last_start - first_end)
        sections = (
            (-half_length, first_end, start_texture, 1.0),
            (
                first_end,
                last_start,
                middle_texture,
                max(1.0, middle_length / RUNWAY_TEXTURE_TILE_METRES),
            ),
            (last_start, half_length, end_texture, 1.0),
        )

    visual_points: list[tuple[float, float, float]] = []
    visual_faces: list[_Face] = []
    for z0, z1, texture, repeats in sections:
        point_start = len(visual_points)
        # Same winding as the generated gravel road ribbon: left start, left end,
        # right end, right start.  Local +Z is the runway's forward direction.
        visual_points.extend((
            (-half_width, 0.0, z0),
            (-half_width, 0.0, z1),
            (half_width, 0.0, z1),
            (half_width, 0.0, z0),
        ))
        visual_faces.append(
            _runway_face(
                texture,
                point_start,
                axial_repeats=repeats,
                family=family,
            )
        )

    visual = _Lod(
        tuple(visual_points),
        ((0.0, 1.0, 0.0),),
        tuple(visual_faces),
        _VISUAL_LOD,
        properties=(("autocenter", "0"), ("class", "road"), ("map", "road")),
    )

    plane_points = (
        (-half_width, 0.0, -half_length),
        (-half_width, 0.0, half_length),
        (half_width, 0.0, half_length),
        (half_width, 0.0, -half_length),
    )
    roadway = _Lod(
        plane_points,
        ((0.0, 1.0, 0.0),),
        (_Face("", ((0, 0, 0.0, 0.0), (1, 0, 0.0, 1.0), (2, 0, 1.0, 1.0), (3, 0, 1.0, 0.0))),),
        _ROADWAY_LOD,
    )
    land = _Lod(plane_points, (), (), _LAND_CONTACT_LOD)
    return visual, roadway, land


def write_runway_mlod(
    path: Path,
    family: str,
    width_metres: float,
    length_metres: float,
) -> None:
    lods = _runway_lods(family, width_metres, length_metres)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, len(lods)))
        for lod in lods:
            _write_lod(stream, lod)


class RunwayInfrastructureLibrary(_BaseInfrastructureLibrary):
    """Teach the existing generated-infrastructure stage about runway P3Ds."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._runway_usage: Counter[tuple[str, int, int]] = Counter()

    def register_model_usage(self, model_path: str, count: int = 1) -> None:
        count = max(0, int(count))
        if count == 0:
            return
        if self.is_generated_model(model_path):
            filename = model_path.replace("/", "\\").rsplit("\\", 1)[-1]
            match = _RUNWAY_PATTERN.fullmatch(filename)
            if match:
                family, width_dm, length_dm = match.groups()
                self._runway_usage[(family.casefold(), int(width_dm), int(length_dm))] += count
                return
        super().register_model_usage(model_path, count)

    def write_assets(self, source_dir: Path, catalogue_path: Path) -> InfrastructureAssetResult:
        base = super().write_assets(source_dir, catalogue_path)
        if not self._runway_usage:
            return base

        runway_models: list[dict[str, object]] = []
        runway_files: list[str] = []
        runway_placements = sum(self._runway_usage.values())
        for (family, width_dm, length_dm), usage_count in sorted(self._runway_usage.items()):
            wire = (
                rf"{self.world_name}\i\runway_{family}_"
                rf"w{width_dm}_l{length_dm}.p3d"
            )
            relative = wire.split("\\", 1)[1].replace("\\", "/")
            destination = source_dir / relative
            write_runway_mlod(
                destination,
                family,
                width_dm / 10.0,
                length_dm / 10.0,
            )
            summary = inspect_mlod(destination)
            if not any(math.isclose(value, _ROADWAY_LOD, rel_tol=1.0e-6) for value in summary.resolutions):
                raise ValueError("generated runway lost its Roadway LOD")
            runway_models.append({
                "key": {
                    "kind": "runway",
                    "subtype": family,
                    "width_dm": width_dm,
                    "length_dm": length_dm,
                },
                "model_path": wire,
                "relative_path": relative,
                "usage_count": usage_count,
                "sha256": sha256(destination.read_bytes()).hexdigest(),
                "lod_resolutions": summary.resolutions,
                "stock_textures": runway_texture_triplet(family),
            })
            runway_files.append(relative)

        document = json.loads(catalogue_path.read_text(encoding="utf-8"))
        document["placements"] = int(document.get("placements", 0)) + runway_placements
        document["generated_variants"] = int(document.get("generated_variants", 0)) + len(runway_models)
        document.setdefault("models", []).extend(runway_models)
        document["runway_texture_source"] = {
            "type": "stock-resistance-o-pbo",
            "grass": runway_texture_triplet("grass"),
            "desert": runway_texture_triplet("desert"),
            "tile_metres": RUNWAY_TEXTURE_TILE_METRES,
            "orientation": "P3D UV rotation plus object heading",
        }
        document.pop("catalogue_sha256", None)
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        document["catalogue_sha256"] = digest
        rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
        catalogue_path.write_text(rendered, encoding="utf-8")
        embedded = source_dir / "i" / "infrastructure.json"
        embedded.parent.mkdir(parents=True, exist_ok=True)
        embedded.write_text(rendered, encoding="utf-8")

        return InfrastructureAssetResult(
            placements=base.placements + runway_placements,
            generated_variants=base.generated_variants + len(runway_models),
            catalogue_sha256=digest,
            cache_hits=base.cache_hits,
            cache_misses=base.cache_misses + len(runway_models),
            model_files=tuple((*base.model_files, *runway_files)),
            texture_files=base.texture_files,
        )


def install_runway_model_policy() -> None:
    """Replace the generator's infrastructure library with runway-aware subclass."""
    global _INSTALLED
    if _INSTALLED:
        return
    from . import generator
    from . import procedural_infrastructure

    generator.ProceduralInfrastructureLibrary = RunwayInfrastructureLibrary
    procedural_infrastructure.ProceduralInfrastructureLibrary = RunwayInfrastructureLibrary
    _INSTALLED = True
