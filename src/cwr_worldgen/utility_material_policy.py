# SPDX-License-Identifier: GPL-3.0-or-later
"""Give utility building classes country-aware material families of their own.

The upstream catalogue deliberately keeps a compact list of common materials per
country.  That is useful for houses, but barns/sheds/warehouses can otherwise end
up wearing exactly the same wall/roof finish as apartments.  This policy derives
utility-only variants from each selected country's own material vocabulary and
facade family, while supporting explicit JSON overrides when a country is curated
more deeply later.

Explicit OSM ``building:material`` and ``roof:material`` tags remain authoritative.
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

_UTILITY_FAMILIES = frozenset({"agricultural", "outbuilding", "industrial"})
_UTILITY_CLASSES = frozenset({"barn", "shed", "garage", "warehouse", "hangar", "industrial"})


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _tokens(value: str) -> set[str]:
    text = str(value or "").casefold().replace("-", " ").replace("_", " ")
    groups = {
        "metal": ("metal", "steel", "aluminium", "aluminum", "zinc", "galvan", "corrugated", "sheet", "tin"),
        "corrugated": ("corrugated", "sheet metal", "sheet steel", "profiled metal", "profiled steel"),
        "wood": ("wood", "timber", "clapboard", "board", "plank"),
        "concrete": ("concrete", "precast", "cement", "panel", "block"),
        "brick": ("brick",),
        "stone": ("stone", "granite", "limestone", "masonry"),
        "earth": ("adobe", "earth", "mud", "rammed"),
        "render": ("stucco", "plaster", "render"),
        "tile": ("tile", "clay"),
        "thatch": ("thatch",),
        "shingle": ("shingle", "asphalt"),
        "slate": ("slate",),
        "membrane": ("membrane", "bitumen", "bituminous", "felt", "tar"),
    }
    return {name for name, needles in groups.items() if any(needle in text for needle in needles)}


def _role(choice) -> str | None:
    building_class = str(getattr(choice, "building_class", "") or "").casefold()
    family = str(getattr(choice, "family", "") or "").casefold()
    outbuilding_kind = str(getattr(choice, "outbuilding_kind", "") or "").casefold()
    if building_class == "barn" or family == "agricultural":
        return "barn"
    if building_class == "garage" or outbuilding_kind == "garage":
        return "garage"
    if building_class in {"warehouse", "hangar", "industrial"} or family == "industrial":
        return "warehouse"
    if building_class == "shed" or family == "outbuilding":
        return "shed"
    return None


def _source_context(choice) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
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
    roof = geometry.get("roof") or {} if isinstance(geometry, Mapping) else {}
    return (
        materials if isinstance(materials, Mapping) else {},
        roof if isinstance(roof, Mapping) else {},
    )


def _override_block(materials: Mapping[str, Any], family: str, building_class: str) -> Mapping[str, Any]:
    class_overrides = materials.get("building_class_overrides") or {}
    if isinstance(class_overrides, Mapping):
        block = class_overrides.get(building_class) or {}
        if isinstance(block, Mapping) and block:
            return block
    family_overrides = materials.get("family_overrides") or {}
    if isinstance(family_overrides, Mapping):
        block = family_overrides.get(family) or {}
        if isinstance(block, Mapping) and block:
            return block
    return {}


def _weighted_extend(target: list[str], values: Sequence[str], weight: int, transform) -> None:
    for value in values:
        transformed = transform(value)
        target.extend([transformed] * max(1, int(weight)))


def _matching(values: Sequence[str], category: str) -> tuple[str, ...]:
    return tuple(value for value in values if category in _tokens(value))


def _synthetic_wall(value: str, kind: str) -> str:
    label = {
        "wood": "utility rough timber cladding",
        "metal": "utility corrugated metal cladding",
        "concrete": "utility precast concrete panels",
        "brick": "utility structural brick",
        "stone": "utility rough stone masonry",
        "earth": "utility earth masonry",
        "render": "utility service render",
    }.get(kind, "utility wall")
    return f"{label} ({value})"


def _facade_material(choice) -> tuple[str, str] | None:
    facade = str(getattr(choice, "facade_style", "") or "")
    categories = _tokens(facade)
    if "wood" in categories or "swed" in facade.casefold() or "nordic" in facade.casefold():
        return "painted timber", "wood"
    if "concrete" in categories or "panel" in facade.casefold():
        return "precast concrete", "concrete"
    if "brick" in categories:
        return "brick", "brick"
    if "stone" in categories:
        return "stone", "stone"
    if "earth" in categories:
        return "earth/adobe", "earth"
    if "render" in categories:
        return "stucco/render", "render"
    return None


def _derived_wall_pool(choice, role: str, common_walls: Sequence[str], common_roofs: Sequence[str]) -> tuple[str, ...]:
    pool: list[str] = []
    weights = {
        "barn": {"wood": 5, "metal": 3, "brick": 2, "stone": 1, "earth": 2, "concrete": 1, "render": 1},
        "shed": {"wood": 4, "metal": 5, "concrete": 2, "brick": 1, "stone": 1, "earth": 1, "render": 1},
        "garage": {"metal": 5, "concrete": 4, "brick": 2, "wood": 1, "stone": 1, "render": 1},
        "warehouse": {"metal": 6, "concrete": 6, "brick": 2, "stone": 1, "wood": 1, "render": 1},
    }[role]

    for category in ("wood", "concrete", "brick", "stone", "earth", "render"):
        values = _matching(common_walls, category)
        if values:
            _weighted_extend(pool, values, weights.get(category, 1), lambda value, c=category: _synthetic_wall(value, c))

    # Corrugated/profiled roof sheet is also a locally documented utility-cladding
    # cue. Standing-seam metal alone is not enough evidence to invent metal barns.
    metal_values = _matching(common_walls, "metal")
    corrugated_roofs = tuple(value for value in common_roofs if "corrugated" in _tokens(value))
    metal_source = metal_values or corrugated_roofs
    if metal_source:
        _weighted_extend(pool, metal_source, weights.get("metal", 1), lambda value: _synthetic_wall(value, "metal"))

    facade_hint = _facade_material(choice)
    if facade_hint is not None:
        value, category = facade_hint
        _weighted_extend(pool, (value,), max(2, weights.get(category, 1)), lambda item: _synthetic_wall(item, category))

    if pool:
        return tuple(pool)
    # Some countries intentionally have a very narrow vernacular palette. Keep
    # that local material rather than importing a foreign industrial stereotype.
    return tuple(common_walls)


def _derived_roof_pool(role: str, common_roofs: Sequence[str]) -> tuple[str, ...]:
    pool: list[str] = []
    weights = {
        "barn": {"metal": 5, "tile": 2, "thatch": 2, "shingle": 2, "slate": 1, "membrane": 1},
        "shed": {"metal": 6, "shingle": 2, "tile": 1, "slate": 1, "membrane": 2, "thatch": 1},
        "garage": {"metal": 6, "membrane": 3, "shingle": 2, "tile": 1, "slate": 1},
        "warehouse": {"metal": 8, "membrane": 5, "shingle": 1, "tile": 1, "slate": 1},
    }[role]
    for value in common_roofs:
        categories = _tokens(value)
        weight = max((weights.get(category, 0) for category in categories), default=0)
        if weight:
            pool.extend([f"utility {value}"] * weight)
    return tuple(pool or tuple(f"utility {value}" for value in common_roofs))


def _pick(values: Sequence[str], seed: str, fallback: str) -> str:
    if not values:
        return str(fallback or "")
    index = int.from_bytes(sha256(seed.encode("utf-8")).digest()[:8], "big") % len(values)
    return str(values[index])


def apply_utility_materials(choice, tags: Mapping[str, str], *, seed: str = "cwr-worldgen"):
    """Return *choice* with country-aware utility-only wall/roof materials."""
    role = _role(choice)
    if role is None:
        return choice

    materials, roof_detail = _source_context(choice)
    common_walls = _strings(materials.get("common_wall_materials") or ())
    common_roofs = _strings(roof_detail.get("materials") or materials.get("common_roof_materials") or ())
    family = str(getattr(choice, "family", "") or "")
    building_class = str(getattr(choice, "building_class", "") or "")
    override = _override_block(materials, family, building_class)

    wall_pool = _strings(override.get("wall_materials") or override.get("walls") or ())
    roof_pool = _strings(override.get("roof_materials") or override.get("roofs") or ())
    if not wall_pool:
        wall_pool = _derived_wall_pool(choice, role, common_walls, common_roofs)
    if not roof_pool:
        roof_pool = _derived_roof_pool(role, common_roofs)

    signature = ":".join((
        str(seed),
        str(getattr(choice, "country_profile_identifier", "") or getattr(choice, "region_identifier", "")),
        str(getattr(choice, "context", "")), role, building_class,
        str(getattr(choice, "facade_style", "")),
    ))
    wall = str(getattr(choice, "wall_material", "") or "")
    roof = str(getattr(choice, "roof_material", "") or "")
    if not str(tags.get("building:material", "") or "").strip():
        wall = _pick(wall_pool, signature + ":wall", wall)
    if not str(tags.get("roof:material", "") or "").strip():
        roof = _pick(roof_pool, signature + ":roof", roof)

    thickness = float(getattr(choice, "wall_thickness_m", 0.22) or 0.22)
    if wall != getattr(choice, "wall_material", ""):
        # Keep geometry consistent with the newly selected material class.
        _materials, _roof = _source_context(choice)
        source = None
        country_id = str(getattr(choice, "country_profile_identifier", "") or "")
        if country_id:
            source = next((p for p in _styles.load_country_profiles() if p.identifier == country_id), None)
        if source is None:
            source = next((p for p in _styles.load_profiles() if p.identifier == choice.region_identifier), None)
        if source is not None:
            context = source.contexts.get(choice.context) or source.contexts.get("rural") or {}
            details = context.get("architectural_details") or {} if isinstance(context, Mapping) else {}
            geometry = details.get("geometry_defaults") or {} if isinstance(details, Mapping) else {}
            if isinstance(geometry, Mapping):
                try:
                    thickness = float(_styles._wall_thickness_m(geometry, wall))
                except (AttributeError, TypeError, ValueError):
                    pass

    return replace(choice, wall_material=wall, roof_material=roof, wall_thickness_m=thickness)


def _utility_wall_base(region: str, facade: str, wall_material: str, palette: tuple[str, ...]):
    text = str(wall_material or "").casefold()
    if text.startswith("utility "):
        base_kind, base = _ORIGINAL_CHOOSE_WALL_BASE(region, facade, wall_material, palette)
        if "corrugated metal" in text:
            return "utility_metal", _textures._colour_from_name(wall_material, default=(105, 112, 117))
        if "rough timber" in text:
            return "utility_wood", base
        if "precast concrete" in text:
            return "utility_concrete", base
        if "structural brick" in text:
            return "utility_brick", base
        if "rough stone" in text:
            return "utility_stone", base
        return "utility_" + base_kind, base
    return _ORIGINAL_CHOOSE_WALL_BASE(region, facade, wall_material, palette)


def _utility_render_wall(kind: str, base, rng: random.Random, size: int):
    if not str(kind).startswith("utility_"):
        return _ORIGINAL_RENDER_WALL(kind, base, rng, size)
    raw_kind = str(kind)[8:]
    if raw_kind == "metal":
        pixels = []
        for y in range(size):
            for x in range(size):
                colour = list(_textures._jitter(base, 6, rng))
                phase = x % 20
                if phase < 2:
                    colour = [_textures._clamp(int(c * 0.58)) for c in colour]
                elif phase in {2, 3}:
                    colour = [_textures._clamp(int(c * 1.10)) for c in colour]
                if y % 96 < 2:
                    colour = [_textures._clamp(int(c * 0.72)) for c in colour]
                pixels.append(tuple(colour))
        return pixels
    if raw_kind == "wood":
        pixels = []
        for y in range(size):
            for x in range(size):
                colour = list(_textures._jitter(base, 8, rng))
                phase = x % 26
                if phase < 3:
                    colour = [_textures._clamp(int(c * 0.55)) for c in colour]
                elif phase == 3:
                    colour = [_textures._clamp(int(c * 0.82)) for c in colour]
                if y % 128 < 2:
                    colour = [_textures._clamp(int(c * 0.75)) for c in colour]
                pixels.append(tuple(colour))
        return pixels
    base_pixels = _ORIGINAL_RENDER_WALL(
        {"concrete": "concrete", "brick": "brick", "stone": "stone"}.get(raw_kind, raw_kind),
        base, rng, size,
    )
    pixels = list(base_pixels)
    if raw_kind == "concrete":
        for y in range(size):
            for x in range(size):
                if x % 128 < 3 or y % 96 < 3:
                    index = y * size + x
                    pixels[index] = tuple(_textures._clamp(int(c * 0.68)) for c in pixels[index])
    elif raw_kind in {"brick", "stone"}:
        # Utility masonry is deliberately a little darker/rougher than domestic
        # facade masonry while preserving the country's chosen base colour.
        pixels = [tuple(_textures._clamp(int(c * 0.90)) for c in colour) for colour in pixels]
    return pixels


def _utility_roof_base(region: str, roof_material: str):
    material = str(roof_material or "")
    if material.casefold().startswith("utility "):
        kind, base = _ORIGINAL_CHOOSE_ROOF_BASE(region, material[8:].strip())
        return "utility_" + kind, base
    return _ORIGINAL_CHOOSE_ROOF_BASE(region, roof_material)


def _utility_render_roof(kind: str, base, rng: random.Random, size: int):
    if not str(kind).startswith("utility_"):
        return _ORIGINAL_RENDER_ROOF(kind, base, rng, size)
    raw_kind = str(kind)[8:]
    if raw_kind == "metal":
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
    pixels = _ORIGINAL_RENDER_ROOF(raw_kind, base, rng, size)
    return [tuple(_textures._clamp(int(c * 0.92)) for c in colour) for colour in pixels]


def install_utility_material_policy() -> None:
    """Install class-specific material selection and utility texture variants."""
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
        tags = kwargs.get("tags")
        if tags is None and args:
            tags = args[0]
        tags = tags if isinstance(tags, Mapping) else {}
        seed = str(kwargs.get("seed", "cwr-worldgen") or "cwr-worldgen")
        return apply_utility_materials(choice, tags, seed=seed)

    _runtime.resolve_style = resolved
    _textures._choose_wall_base = _utility_wall_base
    _textures._render_wall = _utility_render_wall
    _textures._choose_roof_base = _utility_roof_base
    _textures._render_roof = _utility_render_roof

    # The bridge normally starts empty at package import. Clear defensively for
    # tests or embedded callers that imported it before installing policies.
    try:
        from . import osm_house_modeler_texture_bridge as bridge
        bridge._wall_material_image.cache_clear()
    except (ImportError, AttributeError):
        pass
    _INSTALLED = True
