# SPDX-License-Identifier: GPL-3.0-or-later
"""Translate OSM House Modeler's detailed StyleChoice into CWR-safe values.

The upstream modeler remains the authority for regional/country classification and
architectural policy. CWR remains the authority for P3D/MLOD geometry, collision,
Roadway/Memory/Paths LODs and its enterable-building implementation.
"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
import json
import math
from typing import Any, Mapping

from PIL import Image, ImageColor

from .osm_house_modeler_styles import StyleChoice, choose_style, load_profiles


def _number(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _text(mapping: Mapping[str, Any], name: str, default: str = "") -> str:
    value = mapping.get(name, default)
    return str(value or default)


@lru_cache(maxsize=1)
def modeler_profiles():
    return load_profiles()


def modeler_context(settlement_context: str) -> str:
    value = str(settlement_context or "rural").casefold()
    return "town_city" if value in {"city", "town", "town_city", "urban"} else "rural"


_STYLE_SEED_TAGS = frozenset({
    "building", "amenity", "shop", "man_made", "historic",
    "building:material", "building:colour", "building:color",
    "building:levels", "height", "roof:levels", "roof:shape",
    "roof:material", "roof:colour", "roof:color", "roof:height",
})


def stable_way_id(
    tags: Mapping[str, str], latitude: float, longitude: float, width_m: float, length_m: float,
) -> int:
    """Return a repeatable modeler seed without destroying CWR variant reuse.

    The standalone application has a real OSM way id available, so each way can
    legitimately sample a different style. CWR's building library often sees only
    semantic tags and dimensions, and its hard variant cap depends on repeated
    architectural requests collapsing to the same immutable key. Seeding from
    coordinates or labels such as ``name`` made identical houses unique and could
    spend every variant slot on cosmetic randomness. Country is still resolved from
    the real building coordinate; only the random style sample uses this compact
    architectural signature.
    """
    del latitude, longitude
    architectural_tags = {
        str(key): str(value)
        for key, value in tags.items()
        if str(key).casefold() in _STYLE_SEED_TAGS
    }
    payload = json.dumps(
        {
            "tags": dict(sorted(architectural_tags.items())),
            "width": round(float(width_m), 2),
            "length": round(float(length_m), 2),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return int.from_bytes(sha256(payload.encode("utf-8")).digest()[:8], "big") & 0x7FFFFFFF


def resolve_style(
    *,
    tags: Mapping[str, str],
    latitude: float,
    longitude: float,
    width_m: float,
    length_m: float,
    settlement_context: str,
    regional_preset: str = "auto",
    seed: str = "cwr-worldgen",
) -> StyleChoice:
    profiles = modeler_profiles()
    way_id = stable_way_id(tags, latitude, longitude, width_m, length_m)
    return choose_style(
        profiles,
        float(longitude),
        float(latitude),
        tags,
        way_id,
        modeler_context(settlement_context),
        str(regional_preset or "auto"),
        country_preset="auto",
        width_m=float(width_m),
        length_m=float(length_m),
        seed=f"{seed}:{way_id}",
    )


def requested_levels(tags: Mapping[str, str], choice: StyleChoice) -> int:
    raw = _number(tags.get("building:levels"), 0.0)
    if raw > 0.0:
        return max(1, min(12, int(round(raw))))
    if choice.default_levels > 0:
        levels = int(choice.default_levels)
    elif choice.building_class == "cabin":
        levels = 1
    elif choice.family in {"agricultural", "outbuilding", "industrial", "school", "shop"}:
        levels = 1
    elif choice.family == "urban":
        levels = 3
    else:
        levels = 2
    if choice.automatic_max_levels > 0:
        levels = min(levels, int(choice.automatic_max_levels))
    return max(1, min(12, levels))


def requested_height(tags: Mapping[str, str], choice: StyleChoice, fallback: float) -> float:
    explicit = _number(tags.get("height"), 0.0)
    if explicit > 1.0:
        return explicit
    levels_raw = _number(tags.get("building:levels"), 0.0)
    levels = requested_levels(tags, choice)
    storey = max(2.4, float(choice.storey_height_m or 3.0))
    if levels_raw > 0.0 or choice.default_levels > 0 or choice.building_class in {"cabin", "cottage"}:
        return max(2.4, levels * storey)
    return max(2.4, float(fallback))


def _json(value: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(value or {}), sort_keys=True, separators=(",", ":"))


def key_fields(choice: StyleChoice) -> dict[str, object]:
    window = dict(choice.window_spec or {})
    door = dict(choice.door_spec or {})
    roof_storey = dict(choice.roof_storey_spec or {})
    palette = tuple(str(value) for value in choice.colour_palette if str(value).strip())
    return {
        "building_class": str(choice.building_class or ""),
        "country_style_identifier": str(choice.country_profile_identifier or ""),
        "wall_material": str(choice.wall_material or ""),
        "roof_material": str(choice.roof_material or ""),
        "foundation_type": str(choice.foundation_type or ""),
        "storey_height_m": round(max(2.4, float(choice.storey_height_m or 3.0)), 3),
        "wall_thickness_m": round(max(0.08, min(0.80, float(choice.wall_thickness_m or 0.22))), 3),
        "style_foundation_depth_m": round(max(0.15, min(5.0, float(choice.foundation_depth_m or 1.0))), 3),
        "visible_plinth_m": round(max(0.0, min(1.5, float(choice.visible_plinth_m or 0.0))), 3),
        "roof_pitch_degrees": round(max(5.0, min(70.0, float(choice.roof_pitch_degrees or 35.0))), 3),
        "eave_overhang_m": round(max(0.0, min(1.5, float(choice.eave_overhang_m or 0.35))), 3),
        "colour_palette": palette,
        "window_width_m": round(max(0.0, _number(window.get("width_m"), 0.0)), 3),
        "window_height_m": round(max(0.0, _number(window.get("height_m"), 0.0)), 3),
        "window_sill_height_m": round(max(0.0, _number(window.get("sill_height_m"), 0.0)), 3),
        "window_edge_margin_m": round(max(0.0, _number(window.get("edge_margin_m"), 0.0)), 3),
        "window_bay_spacing_m": round(max(0.0, _number(window.get("target_bay_spacing_m"), 0.0)), 3),
        "window_density_multiplier": round(max(0.0, _number(window.get("density_multiplier"), 1.0)), 3),
        "window_type": _text(window, "type"),
        "window_placement_style": _text(window, "placement_style"),
        "window_frame_material": _text(window, "frame_material"),
        "door_width_m": round(max(0.0, _number(door.get("primary_width_m"), 0.0)), 3),
        "door_height_m": round(max(0.0, _number(door.get("primary_height_m"), 0.0)), 3),
        "door_corner_clearance_m": round(max(0.0, _number(door.get("corner_clearance_m"), 0.0)), 3),
        "door_window_clearance_m": round(max(0.0, _number(door.get("keep_clear_of_windows_m"), 0.0)), 3),
        "door_type": _text(door, "type"),
        "door_material": _text(door, "material"),
        "roof_storey": bool(choice.roof_storey),
        "roof_storey_probability": round(max(0.0, min(1.0, float(choice.roof_storey_probability or 0.0))), 4),
        "roof_storey_windows_per_gable": max(0, int(_number(roof_storey.get("windows_per_gable"), 0))),
        "roof_storey_spec_json": _json(roof_storey),
        "exterior_detail_spec_json": _json(choice.exterior_detail_spec),
        "texture_style_token": texture_style_token(choice),
    }


def texture_style_token(choice: StyleChoice) -> str:
    palette = ",".join(str(value).strip() for value in choice.colour_palette[:6])
    return "|".join((str(choice.facade_style or "default"), str(choice.wall_material or ""), palette))


def roof_texture_token(roof_style: str, roof_material: str, palette: tuple[str, ...] = ()) -> str:
    return "|".join((str(roof_style or "gabled"), str(roof_material or ""), ",".join(palette[:4])))


def split_texture_token(value: str) -> tuple[str, str, tuple[str, ...]]:
    parts = str(value or "default").split("|", 2)
    facade = parts[0] or "default"
    material = parts[1] if len(parts) > 1 else ""
    palette = tuple(v for v in (parts[2].split(",") if len(parts) > 2 else ()) if v)
    return facade, material, palette


def visual_style_alias(style: str, material: str = "") -> str:
    value = f"{style} {material}".casefold().replace("-", "_")
    if "half_timber" in value:
        return "western_half_timber"
    if "swed" in value or "nordic" in value:
        return "sweden_red" if "wood" in value or "timber" in value else "sweden_yellow"
    if "whitewash" in value:
        if any(token in value for token in ("africa", "earth", "mud")):
            return "africa_whitewash"
        if any(token in value for token in ("middle_east", "arab", "sand")):
            return "middle_east_whitewash"
        return "eastern_whitewash"
    if any(token in value for token in ("adobe", "earth", "mud")):
        return "middle_east_adobe" if "middle" in value or "arab" in value else "africa_earth"
    if "sandstone" in value:
        return "middle_east_sandstone"
    if "brick" in value:
        return "western_brick"
    if "stone" in value:
        return "western_stone"
    if any(token in value for token in ("concrete", "panel", "block", "cement")):
        return "eastern_panel"
    if any(token in value for token in ("stucco", "plaster", "render", "masonry")):
        return "western_stucco"
    if any(token in value for token in ("wood", "timber", "clapboard")):
        return "sweden_yellow"
    return style or "default"


def first_palette_rgb(palette: tuple[str, ...], fallback: tuple[int, int, int] | None = None):
    for value in palette:
        try:
            return ImageColor.getrgb(value)
        except (ValueError, TypeError):
            continue
    return fallback


def tint_texture(image: Image.Image, palette: tuple[str, ...], strength: float = 0.30) -> Image.Image:
    rgb = first_palette_rgb(palette)
    if rgb is None:
        return image
    base = image.convert("RGB")
    overlay = Image.new("RGB", base.size, rgb)
    return Image.blend(base, overlay, max(0.0, min(0.55, float(strength))))


def detail_spec_from_key(key: object) -> dict[str, Any]:
    raw = str(getattr(key, "exterior_detail_spec_json", "") or "")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}
