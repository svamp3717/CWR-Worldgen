# SPDX-License-Identifier: GPL-3.0-or-later
"""Final visual-fidelity helpers for the OSM House Modeler bridge.

The detailed modeler catalogue carries architectural materials and opening
specifications that are richer than CWR's historical facade atlases.  This
module keeps those data useful without replacing CWR's mature P3D/LOD pipeline:
it emits a tiny set of dedicated detail PAA materials and paints closed facade
atlases from the resolved modeler window/door dimensions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw

from . import procedural_buildings as _pb
from .paa import write_rgb_dxt1_paa
from .osm_house_modeler_texture_bridge import modeler_detail_texture_image


DETAIL_MATERIAL_CODES: Mapping[str, str] = {
    "masonry": "qma",
    "wood": "qwo",
    "metal": "qme",
    "balcony": "qba",
    "glass": "qgl",
}


def detail_texture_path(reference_texture: str, kind: str) -> str:
    """Return a short CWA-safe texture path beside the normal building textures."""
    code = DETAIL_MATERIAL_CODES.get(str(kind).casefold(), DETAIL_MATERIAL_CODES["masonry"])
    reference = str(reference_texture or "")
    # Unit-level/legacy callers often pass a bare ``wall.paa`` instead of a
    # world-relative CWA path. Keep those calls byte-compatible and reserve the
    # dedicated material set for real generated addon paths.
    if "\\" not in reference:
        return reference
    prefix = reference.split("\\", 1)[0]
    return rf"{prefix}\d\{code}.paa"


def material_kind(material: object, fallback: str = "masonry") -> str:
    value = str(material or "").casefold().replace("-", "_")
    if any(token in value for token in ("steel", "metal", "aluminium", "aluminum", "zinc", "copper", "iron")):
        return "metal"
    if any(token in value for token in ("wood", "timber", "board", "cladding")):
        return "wood"
    if any(token in value for token in ("glass", "glazing")):
        return "glass"
    if any(token in value for token in ("brick", "stone", "masonry", "concrete", "cement", "stucco", "render", "adobe", "earth")):
        return "masonry"
    return fallback if fallback in DETAIL_MATERIAL_CODES else "masonry"


def material_texture_path(reference_texture: str, material: object, fallback: str = "masonry") -> str:
    return detail_texture_path(reference_texture, material_kind(material, fallback))


def _finish(image: Image.Image, size: int) -> Image.Image:
    return _pb._finish_pixel_texture(image, max(64, int(size)))


def _masonry_image(size: int) -> Image.Image:
    image = Image.new("RGB", (64, 64), (132, 126, 113))
    draw = ImageDraw.Draw(image)
    for row, y in enumerate(range(0, 64, 8)):
        draw.line((0, y, 64, y), fill=(86, 84, 78), width=1)
        offset = 0 if row % 2 == 0 else 8
        for x in range(offset, 64, 16):
            draw.line((x, y, x, min(63, y + 8)), fill=(91, 88, 81), width=1)
    for x, y in ((7, 15), (31, 5), (50, 31), (22, 52)):
        draw.line((x, y, x + 7, y + 2), fill=(151, 144, 128), width=1)
    return _finish(image, size)


def _wood_image(size: int) -> Image.Image:
    image = Image.new("RGB", (64, 64), (135, 103, 72))
    draw = ImageDraw.Draw(image)
    for x in range(0, 64, 6):
        fill = (145, 111, 77) if (x // 6) % 2 == 0 else (126, 94, 65)
        draw.rectangle((x, 0, min(63, x + 5), 63), fill=fill)
        draw.line((x, 0, x, 63), fill=(84, 64, 48), width=1)
    for y in (17, 43):
        draw.line((2, y, 62, y + 2), fill=(103, 76, 54), width=1)
    return _finish(image, size)


def _metal_image(size: int) -> Image.Image:
    image = Image.new("RGB", (64, 64), (119, 123, 121))
    draw = ImageDraw.Draw(image)
    for x in range(0, 64, 8):
        draw.line((x, 0, x, 63), fill=(75, 82, 82), width=1)
        if x + 1 < 64:
            draw.line((x + 1, 0, x + 1, 63), fill=(154, 157, 151), width=1)
    for x, y in ((10, 9), (35, 38), (53, 18)):
        draw.line((x, y, x + 2, y + 19), fill=(102, 78, 62), width=1)
    return _finish(image, size)


def _balcony_image(size: int) -> Image.Image:
    image = Image.new("RGB", (64, 64), (105, 107, 101))
    draw = ImageDraw.Draw(image)
    for y in range(0, 64, 8):
        draw.line((0, y, 63, y), fill=(73, 76, 73), width=1)
    for x in range(4, 64, 12):
        draw.line((x, 0, x, 63), fill=(151, 148, 136), width=2)
    draw.line((0, 57, 63, 57), fill=(66, 68, 66), width=3)
    return _finish(image, size)


def _glass_image(size: int) -> Image.Image:
    image = Image.new("RGB", (64, 64), (48, 64, 68))
    draw = ImageDraw.Draw(image)
    draw.rectangle((1, 1, 62, 62), outline=(123, 137, 134), width=2)
    for offset in (9, 31):
        draw.line((4, offset, 47, offset - 5), fill=(83, 105, 108), width=2)
    draw.line((14, 58, 55, 16), fill=(66, 88, 92), width=1)
    return _finish(image, size)


_DETAIL_IMAGE_FACTORIES = {
    "masonry": _masonry_image,
    "wood": _wood_image,
    "metal": _metal_image,
    "balcony": _balcony_image,
    "glass": _glass_image,
}


def emit_detail_material_textures(library: object, source_dir: Path) -> tuple[str, ...]:
    """Write the small shared material set referenced by modeler detail geometry."""
    size = max(64, int(getattr(library, "texture_size", 128) or 128))
    directory = Path(source_dir) / "d"
    directory.mkdir(parents=True, exist_ok=True)
    relative: list[str] = []
    for kind, code in DETAIL_MATERIAL_CODES.items():
        path = directory / f"{code}.paa"
        write_rgb_dxt1_paa(path, modeler_detail_texture_image(kind, size))
        relative.append(f"d/{code}.paa")
    return tuple(relative)


def _number(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def render_modeler_facade_texture(
    base_image: Image.Image,
    metadata: Mapping[str, Any],
    *,
    family: str,
    front: bool,
) -> Image.Image:
    """Paint modeler-sized windows/door onto a plain closed-facade material.

    CWR maps one facade atlas over roughly a four-metre horizontal bay and a
    three-metre storey.  Converting the modeler's metric dimensions into that
    atlas gives closed buildings the same opening proportions as enterable ones.
    """
    image = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    width_px, height_px = image.size
    sx = width_px / 64.0
    sy = height_px / 64.0

    window = metadata.get("window") or {}
    door = metadata.get("door") or {}
    if not isinstance(window, Mapping):
        window = {}
    if not isinstance(door, Mapping):
        door = {}

    window_width = max(0.0, _number(window.get("width_m"), 0.0))
    window_height = max(0.0, _number(window.get("height_m"), 0.0))
    sill = max(0.0, _number(window.get("sill_height_m"), 0.85))
    bay = max(1.2, _number(window.get("target_bay_spacing_m"), 4.0))
    density = max(0.0, _number(window.get("density_multiplier"), 1.0))
    frame_material = str(window.get("frame_material", "")).casefold()
    window_type = str(window.get("type", "")).casefold()

    if any(token in frame_material for token in ("upvc", "white", "painted timber", "painted wood")):
        trim = (218, 216, 200)
    elif any(token in frame_material for token in ("aluminium", "aluminum", "metal", "steel")):
        trim = (129, 133, 129)
    else:
        trim = (177, 163, 137)
    glass = (48, 63, 66)
    highlight = (80, 101, 103)

    door_width_m = max(0.0, _number(door.get("width_m"), 0.0))
    door_height_m = max(0.0, _number(door.get("height_m"), 0.0))
    door_width_logical = max(9.0, min(22.0, door_width_m / 4.0 * 64.0)) if door_width_m else 13.0
    door_height_logical = max(31.0, min(52.0, door_height_m / 3.0 * 64.0)) if door_height_m else 45.0
    door_x0 = 32.0 - door_width_logical * 0.5
    door_x1 = 32.0 + door_width_logical * 0.5
    door_y1 = 63.0
    door_y0 = max(4.0, door_y1 - door_height_logical)

    if window_width > 0.0 and window_height > 0.0 and density > 0.0:
        count = max(1, min(4, int(round((4.0 / bay) * density))))
        logical_width = max(7.0, min(24.0, window_width / 4.0 * 64.0))
        logical_height = max(9.0, min(36.0, window_height / 3.0 * 64.0))
        y1 = min(58.0, 64.0 * (1.0 - sill / 3.0))
        y0 = max(3.0, y1 - logical_height)
        cell = 64.0 / count
        for index in range(count):
            cx = (index + 0.5) * cell
            x0 = max(index * cell + 2.0, cx - logical_width * 0.5)
            x1 = min((index + 1) * cell - 2.0, cx + logical_width * 0.5)
            if front and not (x1 < door_x0 - 2.0 or x0 > door_x1 + 2.0):
                continue
            box = (
                round(x0 * sx), round(y0 * sy),
                round(x1 * sx), round(y1 * sy),
            )
            line_width = max(1, round(2 * min(sx, sy)))
            draw.rectangle(box, fill=glass, outline=trim, width=line_width)
            bx0, by0, bx1, by1 = box
            if any(token in window_type for token in ("casement", "multi", "paired", "triple", "mullion")):
                midx = (bx0 + bx1) // 2
                draw.line((midx, by0 + line_width, midx, by1 - line_width), fill=trim, width=max(1, line_width // 2))
            if any(token in window_type for token in ("multi", "triple", "paired")):
                midy = (by0 + by1) // 2
                draw.line((bx0 + line_width, midy, bx1 - line_width, midy), fill=trim, width=max(1, line_width // 2))
            draw.line((bx0 + line_width * 2, by0 + line_width * 2, bx1 - line_width * 2, by0 + line_width * 3), fill=highlight, width=max(1, line_width // 2))

    if front and family in {"residential", "townhouse", "urban", "school"}:
        material = str(door.get("material", "")).casefold()
        if any(token in material for token in ("metal", "steel", "aluminium", "aluminum")):
            door_colour = (83, 89, 87)
        elif any(token in material for token in ("glass", "glazed")):
            door_colour = (56, 73, 76)
        else:
            door_colour = (88, 68, 49)
        box = (
            round(door_x0 * sx), round(door_y0 * sy),
            round(door_x1 * sx), round(door_y1 * sy),
        )
        line_width = max(1, round(2 * min(sx, sy)))
        draw.rectangle(box, fill=door_colour, outline=trim, width=line_width)
        bx0, by0, bx1, by1 = box
        dtype = str(door.get("type", "")).casefold()
        if any(token in dtype for token in ("glazed", "glass")):
            glazed_bottom = by0 + max(line_width * 2, (by1 - by0) // 8)
            glazed_top = by0 + max(line_width * 4, (by1 - by0) // 2)
            draw.rectangle((bx0 + line_width * 2, glazed_bottom, bx1 - line_width * 2, glazed_top), fill=glass, outline=trim, width=max(1, line_width // 2))
        handle_y = by0 + round((by1 - by0) * 0.60)
        draw.rectangle((bx1 - line_width * 3, handle_y, bx1 - line_width, handle_y + max(1, line_width)), fill=(159, 131, 72))

    return image
