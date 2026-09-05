from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import random
import struct
import zlib
from typing import Any, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from .styles import StyleChoice

RGB = tuple[int, int, int]


def _clamp(v: int) -> int:
    return max(0, min(255, v))


def _png(path: Path, width: int, height: int, pixels: list[RGB]) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        for r, g, b in pixels[y * width:(y + 1) * width]:
            raw.extend((r, g, b))
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))


def _seed(text: str) -> int:
    return int.from_bytes(sha256(text.encode()).digest()[:8], "big")


def _colour_from_name(name: str, *, default: RGB = (180, 180, 180)) -> RGB:
    key = name.casefold().strip()
    mapping: dict[str, RGB] = {
        "falun red": (146, 61, 51), "ochre yellow": (181, 136, 58), "white": (226, 223, 213),
        "cream": (220, 208, 178), "dark green": (74, 96, 75), "grey": (140, 142, 144),
        "black": (64, 64, 66), "natural timber": (157, 112, 71), "brick": (151, 78, 59),
        "red brick": (153, 78, 58), "stone": (151, 145, 132), "granite": (125, 127, 131),
        "stucco": (213, 199, 164), "render": (213, 199, 164), "plaster": (216, 199, 172),
        "concrete": (166, 169, 166), "tile": (149, 72, 50), "clay/concrete tile": (149, 72, 50),
        "standing-seam metal": (99, 109, 116), "corrugated metal": (102, 109, 114),
        "slate": (67, 72, 76), "shingle": (89, 80, 71), "thatch": (156, 129, 74),
        "painted timber": (163, 84, 67), "timber": (147, 102, 69), "wood": (147, 102, 69),
        "uPVC": (230, 229, 223), "aluminium-clad timber": (208, 207, 200),
        "painted/galvanised steel": (113, 120, 124), "aluminium/composite": (142, 145, 149),
        "stone/granite plinth on older stock": (126, 126, 131), "crawlspace": (132, 126, 118),
        "frost-protected concrete basement": (126, 129, 132), "insulated slab-on-grade": (154, 155, 156),
    }
    if key in mapping:
        return mapping[key]
    for token, colour in mapping.items():
        if token in key:
            return colour
    return default


def _jitter(colour: RGB, amount: int, rng: random.Random) -> RGB:
    return tuple(_clamp(c + rng.randint(-amount, amount)) for c in colour)


def _resolve_style_inputs(
    region_or_style: str | "StyleChoice",
    facade: str | None,
    roof_material: str | None,
    seed_text: str,
) -> tuple[str, str, str, Mapping[str, Any], Mapping[str, Any], tuple[str, ...], str, str, str, str]:
    if not isinstance(region_or_style, str):
        style = region_or_style
        return (
            style.region_identifier,
            style.facade_style,
            style.roof_material,
            style.window_spec,
            style.door_spec,
            tuple(style.colour_palette),
            style.wall_material,
            style.family,
            style.outbuilding_kind,
            style.foundation_type,
        )
    return region_or_style, facade or "default", roof_material or "", {}, {}, (), "", "residential", "", "concrete foundation"


def _choose_wall_base(region: str, facade: str, wall_material: str, palette: tuple[str, ...]) -> tuple[str, RGB]:
    text = f"{facade} {wall_material}".casefold()
    if palette:
        base = _colour_from_name(palette[0], default=(188, 184, 171))
    else:
        base = (188, 184, 171)
    if "brick" in text:
        return "brick", _colour_from_name("brick")
    if any(t in text for t in ("stone", "granite", "limestone", "slate")):
        return "stone", _colour_from_name("stone")
    if any(t in text for t in ("concrete", "panel", "cement", "precast")):
        return "concrete", _colour_from_name("concrete")
    if any(t in text for t in ("wood", "timber")):
        return "wood", _colour_from_name(palette[0], default=(148, 104, 70)) if palette else (148, 104, 70)
    if any(t in text for t in ("stucco", "plaster", "render")):
        return "stucco", _colour_from_name(palette[0], default=(213, 199, 164)) if palette else (213, 199, 164)
    if region in {"sweden", "northern_europe"}:
        return "wood", _colour_from_name(palette[0], default=(164, 77, 58)) if palette else (164, 77, 58)
    return "stucco", base


def _choose_roof_base(region: str, roof_material: str) -> tuple[str, RGB]:
    mat = (roof_material or "").casefold()
    if "tile" in mat or "clay" in mat:
        return "tile", _colour_from_name("tile")
    if "slate" in mat:
        return "slate", _colour_from_name("slate")
    if any(t in mat for t in ("metal", "steel", "zinc", "aluminium", "standing-seam", "corrugated")):
        return "metal", _colour_from_name("standing-seam metal")
    if "thatch" in mat:
        return "thatch", _colour_from_name("thatch")
    if region in {"mediterranean_europe", "south_america", "mexico_central_america"}:
        return "tile", (153, 70, 45)
    if region in {"northern_europe", "sweden", "western_europe"}:
        return "slate", (68, 73, 78)
    return "shingle", (87, 79, 70)


def _render_wall(kind: str, base: RGB, rng: random.Random, size: int) -> list[RGB]:
    wall: list[RGB] = []
    for y in range(size):
        for x in range(size):
            r, g, b = _jitter(base, 8, rng)
            if kind == "brick":
                row = y // 24
                bx = (x + (12 if row % 2 else 0)) % 48
                if y % 24 < 3 or bx < 3:
                    r, g, b = (196, 188, 170)
            elif kind == "stone":
                row = y // 32
                sx = (x + (16 if row % 2 else 0)) % 64
                if y % 32 < 3 or sx < 3:
                    r, g, b = (111, 108, 101)
            elif kind == "wood":
                if y % 28 < 3:
                    r, g, b = tuple(_clamp(int(c * 0.58)) for c in base)
                elif x % 96 < 2:
                    r, g, b = tuple(_clamp(int(c * 0.78)) for c in base)
            elif kind == "concrete" and (x % 96 < 2 or y % 96 < 2):
                r, g, b = (119, 122, 120)
            wall.append((r, g, b))
    return wall


def _render_roof(kind: str, base: RGB, rng: random.Random, size: int) -> list[RGB]:
    roof: list[RGB] = []
    for y in range(size):
        for x in range(size):
            r, g, b = _jitter(base, 10, rng)
            if kind in {"tile", "shingle", "slate"}:
                row_h = 22 if kind == "tile" else 18
                cell_w = 38 if kind == "tile" else 48
                offset = cell_w // 2 if (y // row_h) % 2 else 0
                if y % row_h < 2 or (x + offset) % cell_w < 2:
                    r, g, b = tuple(_clamp(int(c * 0.62)) for c in base)
            elif kind == "metal" and x % 28 < 2:
                r, g, b = tuple(_clamp(int(c * 0.7)) for c in base)
            elif kind == "thatch" and x % 7 == 0:
                r, g, b = tuple(_clamp(int(c * 0.75)) for c in base)
            roof.append((r, g, b))
    return roof


def _render_foundation(rng: random.Random, size: int, base: RGB) -> list[RGB]:
    pixels: list[RGB] = []
    for y in range(size):
        for x in range(size):
            r, g, b = _jitter(base, 7, rng)
            row = y // 26
            sx = (x + (10 if row % 2 else 0)) % 52
            if y % 26 < 2 or sx < 2:
                r, g, b = tuple(_clamp(int(c * 0.7)) for c in base)
            pixels.append((r, g, b))
    return pixels


def _render_window(window_spec: Mapping[str, Any], rng: random.Random, size: int, *, no_glass: bool = False) -> list[RGB]:
    frame = _colour_from_name(str(window_spec.get("frame_material", "painted timber")), default=(220, 220, 214))
    trim = _colour_from_name(str(window_spec.get("trim", "white")), default=frame)
    wtype = str(window_spec.get("type", "casement")).casefold()
    pixels: list[RGB] = []
    border = 16 if "small" not in wtype else 22
    mullion_x = [size // 2]
    mullion_y = []
    if any(k in wtype for k in ("multi_light", "paired", "casement", "triple")):
        mullion_y = [size // 2]
    if "triple" in wtype:
        mullion_x = [size // 3, 2 * size // 3]
    elif "fixed_plus" in wtype:
        mullion_x = [int(size * 0.38)]
    elif any(k in wtype for k in ("clerestory", "strip")):
        border = 10
        mullion_y = [size // 2]
        mullion_x = [size // 4, size // 2, 3 * size // 4]
    for y in range(size):
        for x in range(size):
            is_frame = x < border or x >= size - border or y < border or y >= size - border
            if not no_glass:
                is_frame = is_frame or any(abs(x - m) < 4 for m in mullion_x) or any(abs(y - m) < 4 for m in mullion_y)
            if is_frame:
                colour = _jitter(trim if x < border or y < border else frame, 4, rng)
            else:
                if no_glass:
                    base = tuple(_clamp(int((frame[i] * 0.78 + trim[i] * 0.22))) for i in range(3))
                    colour = tuple(_clamp(c + rng.randint(-4, 4)) for c in base)
                else:
                    t = y / max(1, size - 1)
                    base = (70 + int(22 * (1 - t)), 99 + int(30 * (1 - t)), 122 + int(36 * (1 - t)))
                    highlight = 26 if abs((x + y) - 225) < 9 or abs((x + y) - 285) < 5 else 0
                    colour = tuple(_clamp(c + rng.randint(-5, 5) + highlight) for c in base)
            pixels.append(colour)
    return pixels


def _render_window_frame(window_spec: Mapping[str, Any], rng: random.Random, size: int) -> list[RGB]:
    """Render a seamless frame-only texture for actual 3D window casing.

    Frame geometry should never sample the decorative full-window atlas because
    mullions and glazing bands become stretched across wide townhouse/apartment
    casings.  This atlas intentionally contains only the selected trim/frame
    finish, with tiny material variation so every UV region is safe to sample.
    """
    frame = _colour_from_name(str(window_spec.get("frame_material", "painted timber")), default=(220, 220, 214))
    trim = _colour_from_name(str(window_spec.get("trim", "white")), default=frame)
    base = tuple(_clamp(int(frame[i] * 0.72 + trim[i] * 0.28)) for i in range(3))
    material = str(window_spec.get("frame_material", "")).casefold()
    pixels: list[RGB] = []
    for y in range(size):
        for x in range(size):
            colour = list(_jitter(base, 4, rng))
            if any(token in material for token in ("timber", "wood")) and x % 53 < 2:
                colour = [_clamp(int(c * 0.90)) for c in colour]
            pixels.append(tuple(colour))
    return pixels


def _render_door(door_spec: Mapping[str, Any], family: str, outbuilding_kind: str, rng: random.Random, size: int, *, no_glass: bool = False) -> list[RGB]:
    dtype = str(door_spec.get("type", "panel")).casefold()
    materials = door_spec.get("materials") or ["timber"]
    base = _colour_from_name(str(materials[0]), default=(118, 79, 49))
    if family == "industrial" or "roll" in dtype or "sectional" in dtype:
        base = _colour_from_name("painted/galvanised steel")
    elif family == "garage" or outbuilding_kind == "garage":
        base = _colour_from_name("aluminium/composite")
    pixels: list[RGB] = []
    for y in range(size):
        for x in range(size):
            r, g, b = _jitter(base, 7, rng)
            edge = x < 13 or x >= size - 13 or y < 13 or y >= size - 13
            if family in {"industrial"} or any(k in dtype for k in ("sectional", "roll", "warehouse")):
                if y % 36 < 2:
                    r, g, b = tuple(_clamp(int(c * 0.72)) for c in base)
                if x < 10 or x >= size - 10:
                    r, g, b = tuple(_clamp(int(c * 0.6)) for c in base)
            elif any(k in dtype for k in ("barn", "plank", "sliding", "double_leaf")):
                if x % 31 < 2:
                    r, g, b = tuple(_clamp(int(c * 0.62)) for c in base)
                if abs(x - size // 2) < 2:
                    r, g, b = tuple(_clamp(int(c * 0.58)) for c in base)
                if y < 16 or y >= size - 16:
                    r, g, b = tuple(_clamp(int(c * 0.52)) for c in base)
            else:
                panel_edge = (
                    28 <= x < 34 or size - 34 <= x < size - 28 or
                    45 <= y < 51 or 126 <= y < 132 or size - 48 <= y < size - 42
                )
                grain = x % 31 < 2
                if panel_edge:
                    r, g, b = tuple(_clamp(int(c * 0.68)) for c in base)
                elif grain:
                    r, g, b = tuple(_clamp(int(c * 0.88)) for c in base)
            if edge:
                r, g, b = tuple(_clamp(int(c * 0.55)) for c in base)
            if not no_glass and "glazed" in dtype and size * 0.15 < x < size * 0.85 and 26 < y < size * 0.42:
                r, g, b = (111, 137, 156)
            if (x - 205) ** 2 + (y - 137) ** 2 <= 8 ** 2 and "roll" not in dtype and "sectional" not in dtype:
                r, g, b = (190, 166, 92)
            pixels.append((r, g, b))
    return pixels


def make_textures(
    out_dir: Path,
    region_or_style: str | "StyleChoice",
    facade: str | None = None,
    roof_material: str | None = None,
    seed_text: str = "seed",
    *,
    no_glass: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    region, facade_style, roof_mat, window_spec, door_spec, palette, wall_material, family, outbuilding_kind, foundation_type = _resolve_style_inputs(
        region_or_style, facade, roof_material, seed_text
    )
    rng = random.Random(_seed(seed_text))
    wall_kind, wall_base = _choose_wall_base(region, facade_style, wall_material, palette)
    roof_kind, roof_base = _choose_roof_base(region, roof_mat)
    foundation_base = _colour_from_name(foundation_type, default=(131, 130, 126))
    if foundation_base == (131, 130, 126) and "concrete" in wall_material.casefold():
        foundation_base = _colour_from_name("insulated slab-on-grade")
    size = 256

    wall = _render_wall(wall_kind, wall_base, rng, size)
    roof = _render_roof(roof_kind, roof_base, rng, size)
    foundation = _render_foundation(rng, size, foundation_base)
    window = _render_window(window_spec, rng, size, no_glass=no_glass)
    window_frame = _render_window_frame(window_spec, rng, size)
    door = _render_door(door_spec, family, outbuilding_kind, rng, size, no_glass=no_glass)
    # Secondary architecture uses three deliberately reusable material atlases.
    # This keeps OBJ/MTL material counts sane while allowing
    # chimneys, stoops, porches, balconies and rainwater hardware to read as
    # actual geometry rather than flat facade decals.
    detail_masonry = _render_foundation(rng, size, (139, 136, 128))
    detail_wood = _render_wall("wood", (139, 96, 61), rng, size)
    detail_metal = _render_roof("metal", (100, 108, 112), rng, size)

    balcony_spec: Mapping[str, Any] = {}
    if not isinstance(region_or_style, str):
        raw_balcony = (getattr(region_or_style, "exterior_detail_spec", {}) or {}).get("balconies") or {}
        if isinstance(raw_balcony, Mapping):
            balcony_spec = raw_balcony
    balcony_text = f"{balcony_spec.get('material', '')} {balcony_spec.get('type', '')}".casefold()
    if any(token in balcony_text for token in ("wood", "timber", "plank", "board")):
        balcony = _render_wall("wood", _colour_from_name(str(balcony_spec.get("material", "timber")), default=(139, 96, 61)), rng, size)
    elif any(token in balcony_text for token in ("metal", "steel", "iron", "aluminium", "galvan", "zinc")):
        balcony = _render_roof("metal", _colour_from_name(str(balcony_spec.get("material", "painted/galvanised steel")), default=(105, 112, 117)), rng, size)
    else:
        balcony = _render_foundation(rng, size, _colour_from_name(str(balcony_spec.get("material", "concrete")), default=(145, 145, 142)))

    wall_path = out_dir / "wall.png"
    roof_path = out_dir / "roof.png"
    foundation_path = out_dir / "foundation.png"
    window_path = out_dir / "window.png"
    window_frame_path = out_dir / "window_frame.png"
    door_path = out_dir / "door.png"
    balcony_path = out_dir / "balcony.png"
    detail_masonry_path = out_dir / "detail_masonry.png"
    detail_wood_path = out_dir / "detail_wood.png"
    detail_metal_path = out_dir / "detail_metal.png"
    _png(wall_path, size, size, wall)
    _png(roof_path, size, size, roof)
    _png(foundation_path, size, size, foundation)
    _png(window_path, size, size, window)
    _png(window_frame_path, size, size, window_frame)
    _png(door_path, size, size, door)
    _png(balcony_path, size, size, balcony)
    _png(detail_masonry_path, size, size, detail_masonry)
    _png(detail_wood_path, size, size, detail_wood)
    _png(detail_metal_path, size, size, detail_metal)
    return wall_path, roof_path, foundation_path, window_path, door_path
