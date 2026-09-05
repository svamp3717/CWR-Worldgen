# SPDX-License-Identifier: GPL-3.0-or-later
"""Consume explicit per-country material, colour and utility-building pools.

Selection in this module is intentionally data-driven. Ordinary walls/facade
colours and utility classes come from the selected country/context JSON. Explicit
OSM ``building:material``, ``building:colour`` and ``roof:material`` tags remain
authoritative.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import random
from typing import Any, Mapping, Sequence

from . import osm_house_modeler_runtime as _runtime
from . import osm_house_modeler_styles as _styles
from . import osm_house_modeler_textures as _textures

_INSTALLED = False
_ORIGINAL_RESOLVE_STYLE = None
_ORIGINAL_CHOOSE_WALL_BASE = None
_ORIGINAL_RENDER_WALL = None
_ORIGINAL_CHOOSE_ROOF_BASE = None
_ORIGINAL_RENDER_ROOF = None


def _context_details(choice) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    context_name = str(getattr(choice, "context", "rural") or "rural")
    country_id = str(getattr(choice, "country_profile_identifier", "") or "")
    source = None
    if country_id:
        source = next((profile for profile in _styles.load_country_profiles() if profile.identifier == country_id), None)
    if source is None:
        region_id = str(getattr(choice, "region_identifier", "") or "")
        source = next((profile for profile in _styles.load_profiles() if profile.identifier == region_id), None)
    if source is None:
        return {}, {}
    context = source.contexts.get(context_name) or source.contexts.get("rural") or {}
    if not isinstance(context, Mapping):
        return {}, {}
    details = context.get("architectural_details") or {}
    if not isinstance(details, Mapping):
        return {}, {}
    materials = details.get("materials") or {}
    geometry = details.get("geometry_defaults") or {}
    return (
        materials if isinstance(materials, Mapping) else {},
        geometry if isinstance(geometry, Mapping) else {},
    )


def _override_name(choice, tags: Mapping[str, str]) -> str:
    building = str(tags.get("building", "") or "").casefold().strip()
    if building == "hangar":
        return "hangar"
    name = str(getattr(choice, "building_class", "") or "").casefold()
    if name in {"barn", "shed", "garage", "warehouse", "hangar", "industrial"}:
        return name
    family = str(getattr(choice, "family", "") or "").casefold()
    return {
        "agricultural": "barn",
        "outbuilding": "garage" if str(getattr(choice, "outbuilding_kind", "") or "").casefold() == "garage" else "shed",
        "industrial": "industrial",
    }.get(family, "")


def _weighted_pick(values: object, seed: str, fallback: str) -> str:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return str(fallback or "")
    choices: list[tuple[str, float]] = []
    for entry in values:
        if isinstance(entry, Mapping):
            material = str(entry.get("material", "") or "").strip()
            try:
                weight = max(0.0, float(entry.get("weight", 1.0)))
            except (TypeError, ValueError):
                weight = 1.0
        else:
            material = str(entry or "").strip()
            weight = 1.0
        if material and weight > 0.0:
            choices.append((material, weight))
    if not choices:
        return str(fallback or "")
    total = sum(weight for _material, weight in choices)
    unit = int.from_bytes(sha256(seed.encode("utf-8")).digest()[:8], "big") / 2**64
    target = unit * total
    running = 0.0
    for material, weight in choices:
        running += weight
        if target < running:
            return material
    return choices[-1][0]


def _weighted_colour(values: object, seed: str, fallback: str) -> str:
    """Pick a weighted facade colour from explicit country data."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return str(fallback or "")
    choices: list[tuple[str, float]] = []
    for entry in values:
        if isinstance(entry, Mapping):
            colour = str(entry.get("colour", entry.get("color", "")) or "").strip()
            try:
                weight = max(0.0, float(entry.get("weight", 1.0)))
            except (TypeError, ValueError):
                weight = 1.0
        else:
            colour = str(entry or "").strip()
            weight = 1.0
        if colour and weight > 0.0:
            choices.append((colour, weight))
    if not choices:
        return str(fallback or "")
    total = sum(weight for _colour, weight in choices)
    unit = int.from_bytes(sha256(seed.encode("utf-8")).digest()[:8], "big") / 2**64
    target = unit * total
    running = 0.0
    for colour, weight in choices:
        running += weight
        if target < running:
            return colour
    return choices[-1][0]


def _material_colour_distribution(source: object, wall_material: str):
    if not isinstance(source, Mapping):
        return None
    folded = str(wall_material or "").casefold().strip()
    for key, values in source.items():
        if str(key).casefold().strip() == folded:
            return values
    return None


def _facade_colour_distribution(
    materials: Mapping[str, Any],
    block: Mapping[str, Any],
    wall_material: str,
):
    # Building-class colour pools win first (a Swedish barn should be red far
    # more often than an ordinary rendered house), then a material-specific
    # pool, then the country's ordinary facade distribution.
    if block:
        values = block.get("facade_colour_distribution")
        if values:
            return values
        values = _material_colour_distribution(
            block.get("wall_material_colour_distributions"), wall_material
        )
        if values:
            return values
    values = _material_colour_distribution(
        materials.get("wall_material_colour_distributions"), wall_material
    )
    if values:
        return values
    return materials.get("facade_colour_distribution")


def apply_country_utility_materials(
    choice,
    tags: Mapping[str, str],
    *,
    seed: str,
    width_m: float,
    length_m: float,
):
    """Apply material and facade-colour pools explicitly stored in the profile."""
    materials, geometry = _context_details(choice)
    override_name = _override_name(choice, tags)
    overrides = materials.get("building_class_overrides") or {}
    block = {}
    if override_name and isinstance(overrides, Mapping):
        candidate = overrides.get(override_name) or {}
        if isinstance(candidate, Mapping):
            block = candidate

    signature = ":".join((
        str(seed),
        str(getattr(choice, "country_profile_identifier", "") or getattr(choice, "region_identifier", "")),
        str(getattr(choice, "context", "")),
        str(getattr(choice, "building_class", "")),
        str(getattr(choice, "family", "")),
        str(getattr(choice, "facade_style", "")),
        f"{float(width_m):.2f}",
        f"{float(length_m):.2f}",
    ))

    wall = str(getattr(choice, "wall_material", "") or "")
    roof = str(getattr(choice, "roof_material", "") or "")
    if not str(tags.get("building:material", "") or "").strip():
        if block:
            wall = _weighted_pick(block.get("wall_materials"), signature + ":wall", wall)
        else:
            wall = _weighted_pick(
                materials.get("common_wall_material_distribution"),
                signature + ":wall",
                wall,
            )
    if block and not str(tags.get("roof:material", "") or "").strip():
        roof = _weighted_pick(block.get("roof_materials"), signature + ":roof", roof)

    palette = tuple(str(value) for value in getattr(choice, "colour_palette", ()) if str(value).strip())
    explicit_colour = str(
        tags.get("building:colour")
        or tags.get("building:color")
        or ""
    ).strip()
    primary_colour = explicit_colour
    if not primary_colour:
        primary_colour = _weighted_colour(
            _facade_colour_distribution(materials, block, wall),
            signature + ":facade-colour",
            palette[0] if palette else "",
        )
    if primary_colour:
        primary_key = primary_colour.casefold()
        palette = (primary_colour,) + tuple(
            value for value in palette if value.casefold() != primary_key
        )

    thickness = float(getattr(choice, "wall_thickness_m", 0.22) or 0.22)
    if wall != getattr(choice, "wall_material", ""):
        try:
            thickness = float(_styles._wall_thickness_m(geometry, wall))
        except (AttributeError, TypeError, ValueError):
            pass
    return replace(
        choice,
        wall_material=wall,
        roof_material=roof,
        wall_thickness_m=thickness,
        colour_palette=palette,
    )


def _utility_kind(text: str) -> str:
    value = str(text or "").casefold()
    if not value.startswith("utility "):
        return ""
    if any(token in value for token in ("metal", "steel", "aluminium", "zinc", "corrugated", "sheet")):
        return "metal"
    if any(token in value for token in ("wood", "timber", "board", "plank", "bamboo")):
        return "wood"
    if any(token in value for token in ("concrete", "precast", "cement", "panel", "block")):
        return "concrete"
    if "brick" in value:
        return "brick"
    if any(token in value for token in ("stone", "granite", "masonry")):
        return "stone"
    if any(token in value for token in ("earth", "adobe", "mud", "laterite")):
        return "earth"
    if any(token in value for token in ("render", "stucco", "plaster")):
        return "stucco"
    return ""


def _utility_wall_base(region: str, facade: str, wall_material: str, palette: tuple[str, ...]):
    material_text = str(wall_material or "").casefold()
    if (
        not material_text.startswith("utility ")
        and "vertical" in material_text
        and any(token in material_text for token in ("timber", "wood"))
    ):
        base = (
            _textures._colour_from_name(palette[0], default=(148, 104, 70))
            if palette else (148, 104, 70)
        )
        return "cwr_vertical_timber", base
    utility_kind = _utility_kind(wall_material)
    if not utility_kind:
        return _ORIGINAL_CHOOSE_WALL_BASE(region, facade, wall_material, palette)
    if utility_kind == "metal":
        return "utility_metal", _textures._colour_from_name("painted/galvanised steel", default=(108, 114, 117))
    if utility_kind == "wood":
        base = _textures._colour_from_name(palette[0], default=(139, 96, 61)) if palette else (139, 96, 61)
        return "utility_wood", base
    if utility_kind == "concrete":
        return "utility_concrete", _textures._colour_from_name("concrete", default=(158, 161, 159))
    if utility_kind == "brick":
        return "utility_brick", _textures._colour_from_name("brick", default=(148, 76, 58))
    if utility_kind == "stone":
        return "utility_stone", _textures._colour_from_name("stone", default=(145, 140, 130))
    if utility_kind == "earth":
        return "utility_stucco", _textures._colour_from_name(palette[0], default=(151, 126, 91)) if palette else (151, 126, 91)
    base = _textures._colour_from_name(palette[0], default=(195, 181, 151)) if palette else (195, 181, 151)
    return "utility_stucco", base


def _utility_render_wall(kind: str, base, rng: random.Random, size: int):
    if str(kind) == "cwr_vertical_timber":
        # Swedish painted timber is vertical board-on-board/clapboard-like
        # cladding, not the generic modeler's broad horizontal wood courses.
        # Keep it chunky enough for CWA while adding subtle board-to-board
        # variation and sparse weathering instead of a flat colour slab.
        pixels = []
        board_width = max(16, int(round(size * 0.085)))
        for y in range(size):
            for x in range(size):
                board = x // board_width
                phase = x % board_width
                board_shift = ((board * 17) % 9) - 4
                colour = [
                    _textures._clamp(channel + board_shift + rng.randint(-4, 4))
                    for channel in base
                ]
                if phase < 2:
                    colour = [_textures._clamp(int(c * 0.57)) for c in colour]
                elif phase < 4:
                    colour = [_textures._clamp(int(c * 1.06)) for c in colour]
                # Sparse vertical grain and a very occasional butt joint keep
                # the material readable without turning it into stripy noise.
                if (y + board * 23) % 79 == 0 and phase > 4:
                    colour = [_textures._clamp(int(c * 0.91)) for c in colour]
                if y % max(96, int(size * 0.62)) < 2:
                    colour = [_textures._clamp(int(c * 0.88)) for c in colour]
                pixels.append(tuple(colour))
        return pixels
    if not str(kind).startswith("utility_"):
        return _ORIGINAL_RENDER_WALL(kind, base, rng, size)
    raw = str(kind)[8:]
    if raw == "metal":
        pixels = []
        for y in range(size):
            for x in range(size):
                colour = list(_textures._jitter(base, 7, rng))
                phase = x % 20
                if phase < 2:
                    colour = [_textures._clamp(int(c * 0.58)) for c in colour]
                elif phase in {2, 3}:
                    colour = [_textures._clamp(int(c * 1.08)) for c in colour]
                if y % 96 < 2:
                    colour = [_textures._clamp(int(c * 0.72)) for c in colour]
                pixels.append(tuple(colour))
        return pixels
    if raw == "wood":
        pixels = []
        for y in range(size):
            for x in range(size):
                colour = list(_textures._jitter(base, 9, rng))
                phase = x % 26
                if phase < 3:
                    colour = [_textures._clamp(int(c * 0.54)) for c in colour]
                elif phase == 3:
                    colour = [_textures._clamp(int(c * 0.82)) for c in colour]
                if y % 128 < 2:
                    colour = [_textures._clamp(int(c * 0.74)) for c in colour]
                pixels.append(tuple(colour))
        return pixels
    ordinary = {
        "concrete": "concrete",
        "brick": "brick",
        "stone": "stone",
        "stucco": "stucco",
    }.get(raw, "stucco")
    pixels = list(_ORIGINAL_RENDER_WALL(ordinary, base, rng, size))
    if raw == "concrete":
        for y in range(size):
            for x in range(size):
                if x % 128 < 3 or y % 96 < 3:
                    index = y * size + x
                    pixels[index] = tuple(_textures._clamp(int(c * 0.68)) for c in pixels[index])
    elif raw in {"brick", "stone"}:
        pixels = [tuple(_textures._clamp(int(c * 0.90)) for c in colour) for colour in pixels]
    return pixels


def _utility_roof_base(region: str, roof_material: str):
    material = str(roof_material or "")
    if not material.casefold().startswith("utility "):
        return _ORIGINAL_CHOOSE_ROOF_BASE(region, roof_material)
    kind, base = _ORIGINAL_CHOOSE_ROOF_BASE(region, material[8:].strip())
    return "utility_" + kind, base


def _utility_render_roof(kind: str, base, rng: random.Random, size: int):
    if not str(kind).startswith("utility_"):
        return _ORIGINAL_RENDER_ROOF(kind, base, rng, size)
    raw = str(kind)[8:]
    if raw == "metal":
        pixels = []
        for y in range(size):
            for x in range(size):
                colour = list(_textures._jitter(base, 8, rng))
                phase = x % 18
                if phase < 2:
                    colour = [_textures._clamp(int(c * 0.58)) for c in colour]
                elif phase == 2:
                    colour = [_textures._clamp(int(c * 1.08)) for c in colour]
                if y % 72 == 0 and x % 36 < 3:
                    colour = [_textures._clamp(int(c * 0.48)) for c in colour]
                pixels.append(tuple(colour))
        return pixels
    pixels = _ORIGINAL_RENDER_ROOF(raw, base, rng, size)
    return [tuple(_textures._clamp(int(c * 0.92)) for c in colour) for colour in pixels]


def install_country_utility_material_policy() -> None:
    """Install explicit country material selection and utility texture patterns."""
    global _INSTALLED, _ORIGINAL_RESOLVE_STYLE
    global _ORIGINAL_CHOOSE_WALL_BASE, _ORIGINAL_RENDER_WALL
    global _ORIGINAL_CHOOSE_ROOF_BASE, _ORIGINAL_RENDER_ROOF
    if _INSTALLED:
        return
    _ORIGINAL_RESOLVE_STYLE = _runtime.resolve_style
    _ORIGINAL_CHOOSE_WALL_BASE = _textures._choose_wall_base
    _ORIGINAL_RENDER_WALL = _textures._render_wall
    _ORIGINAL_CHOOSE_ROOF_BASE = _textures._choose_roof_base
    _ORIGINAL_RENDER_ROOF = _textures._render_roof

    def resolved(*args, **kwargs):
        choice = _ORIGINAL_RESOLVE_STYLE(*args, **kwargs)
        tags = kwargs.get("tags") or {}
        if not isinstance(tags, Mapping):
            tags = {}
        return apply_country_utility_materials(
            choice,
            tags,
            seed=str(kwargs.get("seed", "cwr-worldgen") or "cwr-worldgen"),
            width_m=float(kwargs.get("width_m", 0.0) or 0.0),
            length_m=float(kwargs.get("length_m", 0.0) or 0.0),
        )

    _runtime.resolve_style = resolved
    _textures._choose_wall_base = _utility_wall_base
    _textures._render_wall = _utility_render_wall
    _textures._choose_roof_base = _utility_roof_base
    _textures._render_roof = _utility_render_roof
    try:
        from . import osm_house_modeler_texture_bridge as bridge
        bridge._wall_material_image.cache_clear()
    except (ImportError, AttributeError):
        pass
    _INSTALLED = True
