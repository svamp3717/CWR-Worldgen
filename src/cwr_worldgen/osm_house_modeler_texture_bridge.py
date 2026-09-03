# SPDX-License-Identifier: GPL-3.0-or-later
"""Render CWA PAA source images from OSM House Modeler's texture generator.

The upstream modeler emits PNG materials for its OBJ pipeline.  CWR needs the
same material logic in compact PAA atlases that fit its existing MLOD building
pipeline.  This module is the conversion layer: upstream owns material/palette
selection and pixels, CWR owns facade-atlas layout and PAA packaging.
"""
from __future__ import annotations

from hashlib import sha256
import random
from typing import Any, Mapping

from PIL import Image

from .osm_house_modeler_full_style import split_texture_token, texture_metadata_from_token
from . import osm_house_modeler_textures as _upstream

# CWA/OFP's fixed-function lighting makes diffuse textures read substantially
# brighter than the standalone modeler's neutral viewer.  Compensate once at
# export rather than corrupting every country palette independently.
CWA_EXTERIOR_EXPOSURE = 0.78
CWA_INTERIOR_EXPOSURE = 0.58


def _seed(text: str) -> int:
    return int.from_bytes(sha256(text.encode("utf-8")).digest()[:8], "big")


def _clamp_channel(value: int) -> int:
    return max(0, min(255, int(value)))


def cwa_exposure_compensate(image: Image.Image, factor: float = CWA_EXTERIOR_EXPOSURE) -> Image.Image:
    """Reduce diffuse brightness before CWA's legacy lighting is applied."""
    factor = max(0.35, min(1.0, float(factor)))
    table = [_clamp_channel(round(value * factor)) for value in range(256)]
    return image.convert("RGB").point(table * 3)


def _pixels_image(pixels: list[tuple[int, int, int]], size: int) -> Image.Image:
    image = Image.new("RGB", (size, size))
    image.putdata(pixels)
    return image


def _style_inputs(token: str) -> tuple[str, str, tuple[str, ...], dict[str, Any]]:
    facade, material, palette = split_texture_token(token)
    return facade, material, palette, texture_metadata_from_token(token)


def _wall_material_image(token: str, texture_variant: int, size: int) -> Image.Image:
    facade, material, palette, _metadata = _style_inputs(token)
    kind, base = _upstream._choose_wall_base(facade, facade, material, palette)
    rng = random.Random(_seed(f"wall:{token}:{texture_variant}"))
    return _pixels_image(_upstream._render_wall(kind, base, rng, size), size)


def _window_image(metadata: Mapping[str, Any], token: str, texture_variant: int, size: int) -> Image.Image:
    window = metadata.get("window") or {}
    if not isinstance(window, Mapping):
        window = {}
    spec = {
        "type": str(window.get("type", "casement") or "casement"),
        "frame_material": str(window.get("frame_material", "painted timber") or "painted timber"),
        "trim": str(window.get("trim", "white") or "white"),
    }
    rng = random.Random(_seed(f"window:{token}:{texture_variant}"))
    return _pixels_image(_upstream._render_window(spec, rng, size), size)


def _window_frame_image(metadata: Mapping[str, Any], token: str, texture_variant: int, size: int) -> Image.Image:
    window = metadata.get("window") or {}
    if not isinstance(window, Mapping):
        window = {}
    spec = {
        "type": str(window.get("type", "casement") or "casement"),
        "frame_material": str(window.get("frame_material", "painted timber") or "painted timber"),
        "trim": str(window.get("trim", "white") or "white"),
    }
    rng = random.Random(_seed(f"window-frame:{token}:{texture_variant}"))
    return _pixels_image(_upstream._render_window_frame(spec, rng, size), size)


def _door_image(
    metadata: Mapping[str, Any], token: str, texture_variant: int, size: int,
    *, family: str, outbuilding_kind: str = "",
) -> Image.Image:
    door = metadata.get("door") or {}
    if not isinstance(door, Mapping):
        door = {}
    material = str(door.get("material", "timber") or "timber")
    spec = {
        "type": str(door.get("type", "panel") or "panel"),
        "materials": [material],
    }
    rng = random.Random(_seed(f"door:{token}:{texture_variant}:{family}:{outbuilding_kind}"))
    return _pixels_image(
        _upstream._render_door(spec, family, outbuilding_kind, rng, size), size
    )


def _number(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _paste_scaled(base: Image.Image, source: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    base.paste(source.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR), (x0, y0))


def _with_openings(
    base: Image.Image,
    token: str,
    texture_variant: int,
    *, family: str,
    front: bool,
    outbuilding_kind: str = "",
) -> Image.Image:
    """Compose the modeler's real window/door textures into CWR's closed atlas."""
    image = base.convert("RGB").copy()
    size = image.width
    metadata = texture_metadata_from_token(token)
    window = metadata.get("window") or {}
    door = metadata.get("door") or {}
    if not isinstance(window, Mapping):
        window = {}
    if not isinstance(door, Mapping):
        door = {}

    door_w_m = max(0.0, _number(door.get("width_m"), 0.0))
    door_h_m = max(0.0, _number(door.get("height_m"), 0.0))
    door_w = int(round(size * max(0.14, min(0.34, (door_w_m or 0.95) / 4.0))))
    door_h = int(round(size * max(0.48, min(0.84, (door_h_m or 2.05) / 3.0))))
    door_x0 = (size - door_w) // 2
    door_box = (door_x0, size - door_h, door_x0 + door_w, size)

    window_w_m = max(0.0, _number(window.get("width_m"), 0.0))
    window_h_m = max(0.0, _number(window.get("height_m"), 0.0))
    sill_m = max(0.0, _number(window.get("sill_height_m"), 0.85))
    bay_m = max(1.2, _number(window.get("target_bay_spacing_m"), 4.0))
    density = max(0.0, _number(window.get("density_multiplier"), 1.0))
    if window_w_m > 0.0 and window_h_m > 0.0 and density > 0.0:
        count = max(1, min(4, int(round((4.0 / bay_m) * density))))
        win_w = int(round(size * max(0.12, min(0.38, window_w_m / 4.0))))
        win_h = int(round(size * max(0.16, min(0.52, window_h_m / 3.0))))
        bottom = int(round(size * (1.0 - min(2.4, sill_m) / 3.0)))
        top = max(2, bottom - win_h)
        source = _window_image(metadata, token, texture_variant, max(32, win_w, win_h))
        cell = size / count
        for index in range(count):
            cx = int(round((index + 0.5) * cell))
            x0 = max(int(round(index * cell + 2)), cx - win_w // 2)
            x1 = min(int(round((index + 1) * cell - 2)), x0 + win_w)
            if front and not (x1 < door_box[0] - 2 or x0 > door_box[2] + 2):
                continue
            _paste_scaled(image, source, (x0, top, x1, bottom))

    if front and (door_w_m > 0.0 or family in {
        "residential", "townhouse", "urban", "school", "shop",
        "agricultural", "industrial", "outbuilding",
    }):
        door_source = _door_image(
            metadata, token, texture_variant, max(32, door_w, door_h),
            family=family, outbuilding_kind=outbuilding_kind,
        )
        _paste_scaled(image, door_source, door_box)
    return image


def modeler_wall_texture_image(
    family: str, size: int = 128, regional_style: str = "default", texture_variant: int = 0,
) -> Image.Image:
    base = _wall_material_image(regional_style, texture_variant, int(size))
    composed = _with_openings(
        base, regional_style, texture_variant, family=family, front=False
    )
    return cwa_exposure_compensate(composed)


def modeler_open_wall_texture_image(
    family: str, size: int = 128, regional_style: str = "default", texture_variant: int = 0,
) -> Image.Image:
    del family
    return cwa_exposure_compensate(
        _wall_material_image(regional_style, texture_variant, int(size))
    )


def modeler_interior_wall_texture_image(
    family: str, size: int = 128, regional_style: str = "default", texture_variant: int = 0,
) -> Image.Image:
    del family
    return cwa_exposure_compensate(
        _wall_material_image(regional_style, texture_variant, int(size)),
        CWA_INTERIOR_EXPOSURE,
    )


def modeler_front_texture_image(
    family: str, size: int = 128, regional_style: str = "default",
    texture_variant: int = 0, outbuilding_kind: str = "",
) -> Image.Image:
    base = _wall_material_image(regional_style, texture_variant, int(size))
    composed = _with_openings(
        base, regional_style, texture_variant,
        family=family, front=True, outbuilding_kind=outbuilding_kind,
    )
    return cwa_exposure_compensate(composed)


def modeler_door_texture_image(
    size: int = 128, family: str = "residential", regional_style: str = "default",
    texture_variant: int = 0, outbuilding_kind: str = "",
) -> Image.Image:
    metadata = texture_metadata_from_token(regional_style)
    return cwa_exposure_compensate(
        _door_image(
            metadata, regional_style, texture_variant, int(size),
            family=family, outbuilding_kind=outbuilding_kind,
        ),
        0.84,
    )


def modeler_window_frame_texture_image(
    size: int = 128, regional_style: str = "default", texture_variant: int = 0,
) -> Image.Image:
    metadata = texture_metadata_from_token(regional_style)
    return cwa_exposure_compensate(
        _window_frame_image(metadata, regional_style, texture_variant, int(size)), 0.84
    )


def modeler_roof_texture_image(
    roof_style: str, size: int = 128, texture_variant: int = 0,
) -> Image.Image:
    parts = str(roof_style or "gabled").split("|", 2)
    shape = parts[0] or "gabled"
    material = parts[1] if len(parts) > 1 else ""
    palette = tuple(v for v in (parts[2].split(",") if len(parts) > 2 else ()) if v)
    kind, base = _upstream._choose_roof_base(shape, material)
    if palette:
        base = _upstream._colour_from_name(palette[0], default=base)
    rng = random.Random(_seed(f"roof:{roof_style}:{texture_variant}"))
    return cwa_exposure_compensate(
        _pixels_image(_upstream._render_roof(kind, base, rng, int(size)), int(size))
    )


def modeler_foundation_texture_image(size: int = 128, foundation_type: str = "concrete foundation") -> Image.Image:
    base = _upstream._colour_from_name(foundation_type, default=(131, 130, 126))
    rng = random.Random(_seed(f"foundation:{foundation_type}"))
    return cwa_exposure_compensate(
        _pixels_image(_upstream._render_foundation(rng, int(size), base), int(size))
    )


def modeler_detail_texture_image(kind: str, size: int = 128) -> Image.Image:
    kind = str(kind or "masonry").casefold()
    rng = random.Random(_seed(f"detail:{kind}"))
    size = int(size)
    if kind == "wood":
        pixels = _upstream._render_wall("wood", (139, 96, 61), rng, size)
    elif kind == "metal":
        pixels = _upstream._render_roof("metal", (100, 108, 112), rng, size)
    elif kind == "glass":
        pixels = _upstream._render_window(
            {"type": "fixed", "frame_material": "aluminium-clad timber", "trim": "grey"},
            rng, size,
        )
    elif kind == "balcony":
        pixels = _upstream._render_roof("metal", (92, 99, 102), rng, size)
    else:
        pixels = _upstream._render_foundation(rng, size, (139, 136, 128))
    return cwa_exposure_compensate(_pixels_image(pixels, size))
