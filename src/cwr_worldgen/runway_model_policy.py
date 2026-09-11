# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate rotatable 50 m runway tiles from stock Resistance artwork.

RVW4 terrain cells only store a texture-table index; they do not carry a per-cell
UV rotation. The Nogova runway artwork is cardinally authored, so writing
``o\\runtr_d.paa`` directly into arbitrary OSM runway cells necessarily leaves the
painted runway pointing the wrong way whenever the OSM bearing differs from the
stock island.

Use thin generated MLOD tiles instead. Each P3D rotates with its OSM runway
segment and can rotate the stock texture through UV coordinates. Nothing from
O.pbo is copied into the generated world: the models keep external references to
the verified stock ``o\\runtr_*`` / ``o\\runpi_*`` textures.
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
    r"^runway_(grass|desert)_([dzk])\.p3d$", re.IGNORECASE
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


def runway_texture_for_role(profile: object, role: str) -> str:
    role = str(role).strip().casefold()
    textures = runway_texture_triplet(profile)
    try:
        return {"z": textures[0], "d": textures[1], "k": textures[2]}[role]
    except KeyError as exc:
        raise ValueError(f"unknown runway tile role: {role!r}") from exc


def runway_model_path(world_name: str, profile: object, role: str) -> str:
    """Return the reusable world-local P3D path for one runway tile role."""
    role = str(role).strip().casefold()
    if role not in {"z", "d", "k"}:
        raise ValueError(f"unknown runway tile role: {role!r}")
    return rf"{world_name}\i\runway_{runway_family(profile)}_{role}.p3d"


def _runway_face(texture: str, *, family: str) -> _Face:
    """Map the stock cardinal texture so its runway axis follows local +Z."""
    if family == "grass":
        # runtr_* is authored west/east: U is the runway axis. Mapping U onto
        # model Z is the 90-degree UV rotation RVW4 terrain cannot express.
        uvs = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    else:
        # runpi_* is authored south/north: V already is the runway axis.
        uvs = ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0))
    return _Face(
        texture,
        tuple((index, 0, uv[0], uv[1]) for index, uv in enumerate(uvs)),
    )


def _runway_lods(
    family: str,
    role: str,
    width_metres: float = RUNWAY_MODEL_WIDTH_METRES,
    length_metres: float = RUNWAY_TEXTURE_TILE_METRES,
) -> tuple[_Lod, ...]:
    """Build one stock-sized runway tile with Visual/Roadway/LandContact LODs."""
    family = runway_family(family)
    role = str(role).strip().casefold()
    width = max(1.0, float(width_metres))
    length = max(1.0, float(length_metres))
    half_width = width * 0.5
    half_length = length * 0.5
    texture = runway_texture_for_role(family, role)

    # Same winding as the generated gravel road ribbon: left start, left end,
    # right end, right start. Local +Z is the runway's canonical forward axis.
    plane_points = (
        (-half_width, 0.0, -half_length),
        (-half_width, 0.0, half_length),
        (half_width, 0.0, half_length),
        (half_width, 0.0, -half_length),
    )
    visual = _Lod(
        plane_points,
        ((0.0, 1.0, 0.0),),
        (_runway_face(texture, family=family),),
        _VISUAL_LOD,
        properties=(("autocenter", "0"), ("class", "road"), ("map", "road")),
    )
    roadway = _Lod(
        plane_points,
        ((0.0, 1.0, 0.0),),
        (_Face("", ((0, 0, 0.0, 0.0), (1, 0, 0.0, 1.0), (2, 0, 1.0, 1.0), (3, 0, 1.0, 0.0))),),
        _ROADWAY_LOD,
    )
    land = _Lod(plane_points, (), (), _LAND_CONTACT_LOD)
    return visual, roadway, land


def write_runway_mlod(path: Path, family: str, role: str) -> None:
    lods = _runway_lods(family, role)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, len(lods)))
        for lod in lods:
            _write_lod(stream, lod)


class RunwayInfrastructureLibrary(_BaseInfrastructureLibrary):
    """Teach the existing generated-infrastructure stage about runway P3Ds."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._runway_usage: Counter[tuple[str, str]] = Counter()

    def register_model_usage(self, model_path: str, count: int = 1) -> None:
        count = max(0, int(count))
        if count == 0:
            return
        if self.is_generated_model(model_path):
            filename = model_path.replace("/", "\\").rsplit("\\", 1)[-1]
            match = _RUNWAY_PATTERN.fullmatch(filename)
            if match:
                family, role = match.groups()
                self._runway_usage[(family.casefold(), role.casefold())] += count
                return
        super().register_model_usage(model_path, count)

    def write_assets(self, source_dir: Path, catalogue_path: Path) -> InfrastructureAssetResult:
        base = super().write_assets(source_dir, catalogue_path)
        if not self._runway_usage:
            return base

        runway_models: list[dict[str, object]] = []
        runway_files: list[str] = []
        runway_placements = sum(self._runway_usage.values())
        for (family, role), usage_count in sorted(self._runway_usage.items()):
            wire = rf"{self.world_name}\i\runway_{family}_{role}.p3d"
            relative = wire.split("\\", 1)[1].replace("\\", "/")
            destination = source_dir / relative
            write_runway_mlod(destination, family, role)
            summary = inspect_mlod(destination)
            if not any(math.isclose(value, _ROADWAY_LOD, rel_tol=1.0e-6) for value in summary.resolutions):
                raise ValueError("generated runway lost its Roadway LOD")
            runway_models.append({
                "key": {
                    "kind": "runway",
                    "subtype": f"{family}_{role}",
                    "width_dm": int(round(RUNWAY_MODEL_WIDTH_METRES * 10.0)),
                    "length_dm": int(round(RUNWAY_TEXTURE_TILE_METRES * 10.0)),
                },
                "model_path": wire,
                "relative_path": relative,
                "usage_count": usage_count,
                "sha256": sha256(destination.read_bytes()).hexdigest(),
                "lod_resolutions": summary.resolutions,
                "stock_texture": runway_texture_for_role(family, role),
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
