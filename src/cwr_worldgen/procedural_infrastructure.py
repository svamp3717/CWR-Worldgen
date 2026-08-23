# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import json
import math
import re
from typing import Iterable

from PIL import Image, ImageDraw, ImageFilter

from .cache import cache_key, restore_or_create_file
from .paa import inspect_paa, write_rgb_dxt1_paa, write_rgba_dxt1_paa
from .parallel_assets import process_asset_tasks
from .procedural_buildings import (
    _Face,
    _Lod,
    _MLOD_HEADER,
    _NamedSelection,
    _write_lod,
    inspect_mlod,
)

_VISUAL_LOD = 1.0
_GEOMETRY_LOD = 1.0e13
_LAND_CONTACT_LOD = 2.0e15
_ROADWAY_LOD = 3.0e15

# Generated gravel is a terrain-hugging surface ribbon, not a raised slab.
# Its visible skin and Roadway LOD are coplanar and are placed directly on the
# graded terrain; the Geometry LOD carries map metadata only and has no faces.
GENERATED_GRAVEL_VISUAL_TOP_METRES = 0.025
GENERATED_GRAVEL_ROADWAY_HEIGHT_METRES = 0.025
GENERATED_GRAVEL_HALF_WIDTH_METRES = 2.30
GENERATED_GRAVEL_SURFACE_CLEARANCE_METRES = 0.0
GENERATED_GRAVEL_VISUAL_OVERLAP_METRES = 0.90
GENERATED_GRAVEL_OVERLAP_DROP_METRES = 0.040
GENERATED_GRAVEL_CURVE_BUCKETS = (5, 10, 15, 20, 30, 45)
GENERATED_GRAVEL_RIBBON_SECTIONS = 6
GENERATED_GRAVEL_TEXTURE_REPEAT_METRES = 3.0
GENERATED_GRAVEL_EDGE_WIDTH_METRES = 0.18
GENERATED_GRAVEL_EDGE_JITTER_METRES = 0.06
GENERATED_GRAVEL_EDGE_SECTION_METRES = 0.65
GENERATED_BRIDGE_MAXIMUM_DEPTH_METRES = 0.8
GENERATED_BRIDGE_RAIL_OVERHANG_METRES = 0.16
GENERATED_BRIDGE_ROADWAY_HEIGHT_METRES = 0.20

_EDEN_GRAVEL_SURFACES = Path(__file__).resolve().parent / "data" / "eden_gravel" / "surfaces.txt"
_GRAVEL_REFERENCE_TEXTURE = Path(__file__).resolve().parent / "data" / "gravel_reference.png"


def create_gravel_road_texture_image(size: int = 512) -> Image.Image:
    """Build the photo-based gravel texture with a clean terrain-visible edge.

    OFP/CWA DXT1 only supports one-bit alpha. The old wide ordered-dither verge
    therefore rendered as rows of obvious dots. Keep transparency only in a
    very narrow outer strip and let the model's smoothly irregular physical edge
    provide the terrain transition instead.
    """

    size = int(size)
    if size < 32:
        raise ValueError("gravel road texture size must be at least 32 pixels")

    source = Image.open(_GRAVEL_REFERENCE_TEXTURE).convert("RGBA")
    image = source.resize((size, size), Image.Resampling.LANCZOS)
    pixels = image.load()
    # One clean binary edge avoids the coarse halftone/dotted pattern produced
    # by DXT1 alpha dithering and its mipmaps. At 512 px this is only a few
    # centimetres on the generated 4.6 m road.
    transparent_edge = 0.018
    for y in range(size):
        for x in range(size):
            xf = (x + 0.5) / size
            r, g, b, _a = pixels[x, y]
            alpha = 0 if min(xf, 1.0 - xf) <= transparent_edge else 255
            pixels[x, y] = (r, g, b, alpha)
    return image


def create_gravel_junction_texture_image(size: int = 512) -> Image.Image:
    """Build an opaque gravel texture for generated junction polygons.

    Straight/curved gravel ribbons intentionally use transparent texture edges so
    the world terrain can blend into their outside verges. Junction meshes tile
    UVs over a two-dimensional polygon, however, so those repeating alpha edges
    become narrow strips of visible terrain *inside* the road surface. Reuse the
    exact same gravel photograph for junctions, but keep every texel opaque.
    """

    size = int(size)
    if size < 32:
        raise ValueError("gravel junction texture size must be at least 32 pixels")
    source = Image.open(_GRAVEL_REFERENCE_TEXTURE).convert("RGB")
    return source.resize((size, size), Image.Resampling.LANCZOS)

def _eden_gravel_surface_rules() -> dict[str, float]:
    rules: dict[str, float] = {}
    for raw_line in _EDEN_GRAVEL_SURFACES.read_text(encoding="ascii").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            rules[fields[0]] = float(fields[1])
        except ValueError:
            continue
    return rules


@dataclass(frozen=True, order=True, slots=True)
class InfrastructureModelKey:
    kind: str
    subtype: str
    width_dm: int
    length_dm: int

    @property
    def width_m(self) -> float:
        return self.width_dm / 10.0

    @property
    def length_m(self) -> float:
        return self.length_dm / 10.0


@dataclass(frozen=True, slots=True)
class InfrastructureAssetResult:
    placements: int
    generated_variants: int
    catalogue_sha256: str
    cache_hits: int
    cache_misses: int
    model_files: tuple[str, ...]
    texture_files: tuple[str, ...]

    def to_manifest(self) -> dict[str, object]:
        return {
            "placements": self.placements,
            "generated_variants": self.generated_variants,
            "catalogue_sha256": self.catalogue_sha256,
            "model_files": self.model_files,
            "texture_files": self.texture_files,
        }


_TEXTURE_FILE_STEMS = {
    # MLOD stores texture references in a 32-byte C string (31 bytes plus NUL).
    # World names may legally use all 20 characters, so every world-local
    # infrastructure texture must use a compact filename. 0.9.176 only shortened
    # utility names, leaving gravel/bridge/fence/hedge capable of overflowing the
    # P3D field for long world names.
    "fence": "f",
    "wall": "w",
    "hedge": "h",
    "bridge": "b",
    "gravel": "g",
    "gravel_junction": "gj",
    "power_pole": "up",
    "power_tower": "ut",
    "water_tower": "uw",
}


def _texture_file_stem(kind: str) -> str:
    return _TEXTURE_FILE_STEMS.get(kind, kind)


def _texture_image(kind: str, size: int = 128) -> Image.Image:
    colours = {
        "fence": (125, 102, 72),
        "wall": (133, 132, 123),
        "hedge": (74, 102, 55),
        "bridge": (112, 105, 91),
        "rock": (118, 116, 107),
        "gravel": (126, 119, 103),
        "gravel_junction": (126, 119, 103),
        "power_pole": (116, 102, 78),
        "power_tower": (118, 120, 119),
        "water_tower": (142, 148, 151),
    }
    base = colours[kind]
    image = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(image)
    if kind == "fence":
        for y in range(0, size, 16):
            draw.line((0, y, size, y), fill=(91, 72, 49), width=2)
        for x in range(0, size, 24):
            draw.line((x, 0, x, size), fill=(151, 124, 85), width=3)
    elif kind == "wall":
        for y in range(0, size, 20):
            draw.line((0, y, size, y), fill=(91, 91, 86), width=2)
            offset = 0 if (y // 20) % 2 == 0 else 16
            for x in range(offset, size, 32):
                draw.line((x, y, x, min(size, y + 20)), fill=(95, 95, 89), width=2)
    elif kind == "hedge":
        for y in range(size):
            for x in range(size):
                n = ((x * 37 + y * 19 + (x ^ y) * 7) % 25) - 12
                image.putpixel((x, y), tuple(max(0, min(255, c + n)) for c in base))
        draw.ellipse((12, 12, 54, 68), fill=(63, 91, 47))
        draw.ellipse((46, 20, 104, 80), fill=(83, 113, 60))
        draw.ellipse((74, 8, 124, 70), fill=(67, 96, 49))
    elif kind == "bridge":
        for x in range(0, size, 16):
            draw.line((x, 0, x, size), fill=(82, 78, 70), width=2)
        for y in range(0, size, 32):
            draw.line((0, y, size, y), fill=(145, 136, 117), width=2)
    elif kind == "gravel":
        return create_gravel_road_texture_image(size)
    elif kind == "gravel_junction":
        return create_gravel_junction_texture_image(size)
    else:
        for y in range(size):
            for x in range(size):
                n = ((x * 13 + y * 29 + (x * y) % 17) % 31) - 15
                image.putpixel((x, y), tuple(max(0, min(255, c + n)) for c in base))
    return image


def _quad(texture: str, a: int, b: int, c: int, d: int, normal: int = 0) -> _Face:
    return _Face(texture, ((a, normal, 0.0, 1.0), (b, normal, 0.0, 0.0), (c, normal, 1.0, 0.0), (d, normal, 1.0, 1.0)))


def _append_box(
    points: list[tuple[float, float, float]],
    faces: list[_Face],
    *,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z0: float,
    z1: float,
    texture: str,
) -> tuple[range, range]:
    p = len(points)
    f = len(faces)
    points.extend(((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)))
    faces.extend((
        _quad(texture, p + 0, p + 4, p + 5, p + 1),
        _quad(texture, p + 1, p + 5, p + 6, p + 2),
        _quad(texture, p + 2, p + 6, p + 7, p + 3),
        _quad(texture, p + 3, p + 7, p + 4, p + 0),
        _quad(texture, p + 4, p + 7, p + 6, p + 5),
        _quad(texture, p + 0, p + 1, p + 2, p + 3),
    ))
    return range(p, p + 8), range(f, f + 6)


def _selection(name: str, point_count: int, face_count: int, point_range: range, face_range: range) -> _NamedSelection:
    point_weights = bytearray(point_count)
    face_flags = bytearray(face_count)
    for index in point_range:
        point_weights[index] = 1
    for index in face_range:
        face_flags[index] = 1
    return _NamedSelection(name, bytes(point_weights), bytes(face_flags))


def _geometry_from_boxes(boxes: tuple[tuple[float, float, float, float, float, float], ...], *, map_value: str = "") -> _Lod:
    points: list[tuple[float, float, float]] = []
    faces: list[_Face] = []
    ranges: list[tuple[range, range]] = []
    for box in boxes:
        ranges.append(_append_box(points, faces, x0=box[0], x1=box[1], y0=box[2], y1=box[3], z0=box[4], z1=box[5], texture=""))
    selections = tuple(_selection(f"component{index:02d}", len(points), len(faces), pr, fr) for index, (pr, fr) in enumerate(ranges, start=1))
    mass = tuple(max(20.0, 2500.0 / max(1, len(points))) for _ in points)
    properties = (("map", map_value),) if map_value else ()
    return _Lod(tuple(points), (), tuple(faces), _GEOMETRY_LOD, mass, selections, properties)


def _barrier_lods(key: InfrastructureModelKey, texture: str) -> tuple[_Lod, ...]:
    length = key.length_m
    half = length * 0.5
    points: list[tuple[float, float, float]] = []
    faces: list[_Face] = []
    boxes: list[tuple[float, float, float, float, float, float]] = []
    if key.subtype == "wall":
        boxes.append((-0.28, 0.28, 0.0, 1.65, -half, half))
        _append_box(points, faces, x0=-0.28, x1=0.28, y0=0.0, y1=1.65, z0=-half, z1=half, texture=texture)
    elif key.subtype == "hedge":
        boxes.append((-0.5, 0.5, 0.0, 1.45, -half, half))
        # Crossed foliage planes. Double-sided faces are explicit.
        points.extend(((-0.45, 0.0, -half), (-0.45, 1.5, -half), (0.45, 1.5, half), (0.45, 0.0, half), (0.45, 0.0, -half), (0.45, 1.5, -half), (-0.45, 1.5, half), (-0.45, 0.0, half)))
        faces.extend((_quad(texture, 0, 1, 2, 3), _quad(texture, 3, 2, 1, 0), _quad(texture, 4, 5, 6, 7), _quad(texture, 7, 6, 5, 4)))
    else:
        post_positions = (-half, 0.0, half)
        for z in post_positions:
            boxes.append((-0.10, 0.10, 0.0, 1.45, z - 0.10, z + 0.10))
            _append_box(points, faces, x0=-0.10, x1=0.10, y0=0.0, y1=1.45, z0=z - 0.10, z1=z + 0.10, texture=texture)
        for y in (0.45, 1.05):
            boxes.append((-0.08, 0.08, y - 0.07, y + 0.07, -half, half))
            _append_box(points, faces, x0=-0.08, x1=0.08, y0=y - 0.07, y1=y + 0.07, z0=-half, z1=half, texture=texture)
    visual = _Lod(tuple(points), ((0.0, 1.0, 0.0),), tuple(faces), _VISUAL_LOD, properties=(("autocenter", "0"),))
    geometry = _geometry_from_boxes(tuple(boxes), map_value="fence")
    land = _Lod(((-0.25, 0.0, -half), (0.25, 0.0, -half), (-0.25, 0.0, half), (0.25, 0.0, half)), (), (), _LAND_CONTACT_LOD)
    return visual, geometry, land


def _bridge_lods(key: InfrastructureModelKey, texture: str) -> tuple[_Lod, ...]:
    """Build one bridge model of the requested full span length.

    The WRP now places one procedural bridge object per bridge way. Internally
    this P3D still splits a long span into modest deck/rail/collision sections so
    legacy CWA never has to solve one enormous convex component. Visually and in
    the world file it remains a single continuous bridge model.
    """

    width = key.width_m
    length = key.length_m
    half_w = width * 0.5
    half_l = length * 0.5
    deck_bottom = -0.30
    deck_top = 0.18
    component_span = 24.0
    segment_count = max(1, int(math.ceil(length / component_span)))
    segment_length = length / segment_count

    points: list[tuple[float, float, float]] = []
    faces: list[_Face] = []
    geometry_boxes: list[tuple[float, float, float, float, float, float]] = []
    rail_x = half_w + 0.08
    for index in range(segment_count):
        z0 = -half_l + index * segment_length
        z1 = -half_l + (index + 1) * segment_length
        _append_box(
            points, faces,
            x0=-half_w, x1=half_w,
            y0=deck_bottom, y1=deck_top,
            z0=z0, z1=z1,
            texture=texture,
        )
        geometry_boxes.append((-half_w, half_w, deck_bottom, deck_top, z0, z1))
        for sign in (-1.0, 1.0):
            x0 = sign * rail_x - 0.08
            x1 = sign * rail_x + 0.08
            _append_box(
                points, faces,
                x0=min(x0, x1), x1=max(x0, x1),
                y0=deck_top, y1=1.28,
                z0=z0, z1=z1,
                texture=texture,
            )

    # End abutments close the one-piece span visually without forcing their full
    # depth into the terrain. The object origin still owns the final bridge Y.
    if key.subtype in {"start", "single"}:
        _append_box(
            points, faces,
            x0=-half_w - 0.35, x1=half_w + 0.35,
            y0=-0.55, y1=deck_top,
            z0=-half_l, z1=-half_l + min(0.50, length),
            texture=texture,
        )
    if key.subtype in {"end", "single"}:
        _append_box(
            points, faces,
            x0=-half_w - 0.35, x1=half_w + 0.35,
            y0=-0.55, y1=deck_top,
            z0=half_l - min(0.50, length), z1=half_l,
            texture=texture,
        )

    visual = _Lod(
        tuple(points), ((0.0, 1.0, 0.0),), tuple(faces), _VISUAL_LOD,
        properties=(("autocenter", "0"),),
    )
    raw_geometry = _geometry_from_boxes(tuple(geometry_boxes))
    geometry = _Lod(
        raw_geometry.points, raw_geometry.normals, raw_geometry.faces,
        raw_geometry.resolution, raw_geometry.mass_per_point, raw_geometry.selections,
        (("autocenter", "0"), ("canbeoccluded", "0"), ("canocclude", "0")),
    )

    roadway_y = GENERATED_BRIDGE_ROADWAY_HEIGHT_METRES
    roadway_points: list[tuple[float, float, float]] = []
    roadway_faces: list[_Face] = []
    for index in range(segment_count):
        z0 = -half_l + index * segment_length
        z1 = -half_l + (index + 1) * segment_length
        start_index = len(roadway_points)
        roadway_points.extend((
            (-half_w, roadway_y, z0),
            (half_w, roadway_y, z0),
            (half_w, roadway_y, z1),
            (-half_w, roadway_y, z1),
        ))
        roadway_faces.append(_quad("", start_index + 0, start_index + 3, start_index + 2, start_index + 1))
    roadway = _Lod(
        tuple(roadway_points), ((0.0, 1.0, 0.0),), tuple(roadway_faces), _ROADWAY_LOD
    )
    land = _Lod(
        (
            (-half_w, deck_bottom, -half_l),
            (half_w, deck_bottom, -half_l),
            (-half_w, deck_bottom, half_l),
            (half_w, deck_bottom, half_l),
        ),
        (), (), _LAND_CONTACT_LOD,
    )
    return visual, geometry, roadway, land

def _gravel_curve_degrees(subtype: str) -> int:
    match = re.fullmatch(r"gravel(?:25|12|6|3)(?:_([lr])(05|10|15|20|30|45))?", subtype, re.IGNORECASE)
    if not match or not match.group(1):
        return 0
    amount = int(match.group(2))
    return amount if match.group(1).casefold() == "r" else -amount


def _road_ribbon_sections(
    length: float,
    half_width: float,
    curve_degrees: int,
    *,
    overhang: float,
    section_count: int | None = None,
) -> tuple[tuple[float, float, float, float], ...]:
    """Return left/right cross-section coordinates for a smooth road ribbon.

    Values are ``(left_x, left_z, right_x, right_z)``.  The nominal endpoints
    remain on the local Z axis, while curved variants bow sideways between them.
    Short lowered overhangs hide the triangular cracks that otherwise appear
    where independently rotated CWA road objects meet on a bend.
    """

    half_length = length * 0.5
    signed = float(curve_degrees)
    if abs(signed) < 1e-9:
        sagitta = 0.0
    else:
        theta = math.radians(abs(signed))
        radius = length / max(1e-9, 2.0 * math.sin(theta * 0.5))
        sagitta = math.copysign(radius * (1.0 - math.cos(theta * 0.5)), signed)
    control_x = sagitta * 2.0

    centres: list[tuple[float, float, float, float]] = []
    section_count = max(2, GENERATED_GRAVEL_RIBBON_SECTIONS if section_count is None else int(section_count))
    for index in range(section_count + 1):
        t = index / section_count
        one_minus = 1.0 - t
        x = 2.0 * one_minus * t * control_x
        z = -half_length * one_minus * one_minus + half_length * t * t
        dx = 2.0 * (one_minus * control_x + t * (-control_x))
        dz = length
        tangent_length = max(1e-9, math.hypot(dx, dz))
        tx, tz = dx / tangent_length, dz / tangent_length
        centres.append((x, z, tx, tz))

    if overhang > 0.0:
        first_x, first_z, first_tx, first_tz = centres[0]
        last_x, last_z, last_tx, last_tz = centres[-1]
        centres.insert(0, (first_x - first_tx * overhang, first_z - first_tz * overhang, first_tx, first_tz))
        centres.append((last_x + last_tx * overhang, last_z + last_tz * overhang, last_tx, last_tz))

    sections: list[tuple[float, float, float, float]] = []
    for x, z, tx, tz in centres:
        # Local +X is the right side of a north-facing (+Z) model.
        rx, rz = tz, -tx
        sections.append((x - rx * half_width, z - rz * half_width, x + rx * half_width, z + rz * half_width))
    return tuple(sections)


def _ribbon_lod(
    sections: tuple[tuple[float, float, float, float], ...],
    *,
    texture: str,
    resolution: float,
    height: float,
    lowered_overlap: bool,
    double_sided: bool,
    u_span_override: float | None = None,
) -> _Lod:
    points: list[tuple[float, float, float]] = []
    cumulative = [0.0]
    centres = [((section[0] + section[2]) * 0.5, (section[1] + section[3]) * 0.5) for section in sections]
    for first, second in zip(centres, centres[1:]):
        cumulative.append(cumulative[-1] + math.dist(first, second))
    texture_scale = GENERATED_GRAVEL_TEXTURE_REPEAT_METRES
    section_width = math.dist((sections[0][0], sections[0][1]), (sections[0][2], sections[0][3]))
    u_span = section_width / texture_scale if u_span_override is None else float(u_span_override)
    last_index = len(sections) - 1
    for index, (lx, lz, rx, rz) in enumerate(sections):
        y = height
        if lowered_overlap and index in {0, last_index}:
            y -= GENERATED_GRAVEL_OVERLAP_DROP_METRES
        points.extend(((lx, y, lz), (rx, y, rz)))

    faces: list[_Face] = []
    for index in range(len(sections) - 1):
        ls, rs = index * 2, index * 2 + 1
        le, re = (index + 1) * 2, (index + 1) * 2 + 1
        v0 = cumulative[index] / texture_scale
        v1 = cumulative[index + 1] / texture_scale
        top = _Face(texture, ((ls, 0, 0.0, v0), (le, 0, 0.0, v1), (re, 0, u_span, v1), (rs, 0, u_span, v0)))
        faces.append(top)
        if double_sided:
            faces.append(_Face(texture, ((rs, 0, u_span, v0), (re, 0, u_span, v1), (le, 0, 0.0, v1), (ls, 0, 0.0, v0))))
    return _Lod(tuple(points), ((0.0, 1.0, 0.0),), tuple(faces), resolution, properties=((('autocenter', '0'), ('class', 'road'), ('map', 'road')) if resolution == _VISUAL_LOD else ()))


def _gravel_visual_lod(length: float, half_width: float, curve_degrees: int, texture: str) -> _Lod:
    """Build a softly irregular gravel ribbon with terrain-visible outer verges."""

    visual_section_count = max(6, min(20, int(math.ceil(length / GENERATED_GRAVEL_EDGE_SECTION_METRES))))
    full_sections = _road_ribbon_sections(
        length, half_width, curve_degrees, overhang=GENERATED_GRAVEL_VISUAL_OVERLAP_METRES,
        section_count=visual_section_count,
    )
    centres = [((sec[0] + sec[2]) * 0.5, (sec[1] + sec[3]) * 0.5) for sec in full_sections]
    sections: list[tuple[float, float, float, float]] = []
    last = max(1, len(full_sections) - 1)
    for index, (section, centre) in enumerate(zip(full_sections, centres)):
        lx, lz, rx, rz = section
        cx, cz = centre
        # Keep the physical edge smooth. Earlier per-section random-looking
        # insets created triangular points at piece joins. A low-amplitude wave
        # returns to the same inset at both ends, while texture alpha supplies
        # the fine irregular blend into the game's own terrain.
        t = index / last
        # Fade all edge wandering out very strongly near piece ends. Adjacent
        # objects then meet on the same full-width cross section, while the
        # irregular gravel verge returns gradually farther inside each piece.
        envelope = math.sin(math.pi * t) ** 4
        left_wave = 0.55 * math.sin(math.tau * t * 1.35 + 0.35) + 0.30 * math.sin(math.tau * t * 2.7 + 1.05)
        right_wave = 0.55 * math.sin(math.tau * t * 1.35 + 2.15) + 0.30 * math.sin(math.tau * t * 2.7 + 0.55)
        edge_base = 0.060 * envelope
        left_inset = edge_base + envelope * GENERATED_GRAVEL_EDGE_JITTER_METRES * (0.65 + left_wave)
        right_inset = edge_base + envelope * GENERATED_GRAVEL_EDGE_JITTER_METRES * (0.65 + right_wave)

        # Do not narrow the overlap tips. The old inward taper produced the very
        # visible bites/notches where two rotated road pieces met. Full-width,
        # buried overlap tips let the two meshes cover one another cleanly.

        left_len = max(1e-9, math.hypot(lx - cx, lz - cz))
        right_len = max(1e-9, math.hypot(rx - cx, rz - cz))
        left_inset = min(left_len * 0.15, max(0.0, left_inset))
        right_inset = min(right_len * 0.15, max(0.0, right_inset))
        lx -= (lx - cx) / left_len * left_inset
        lz -= (lz - cz) / left_len * left_inset
        rx -= (rx - cx) / right_len * right_inset
        rz -= (rz - cz) / right_len * right_inset
        sections.append((lx, lz, rx, rz))

    visual = _ribbon_lod(
        tuple(sections), texture=texture, resolution=_VISUAL_LOD,
        height=GENERATED_GRAVEL_VISUAL_TOP_METRES, lowered_overlap=True,
        double_sided=True, u_span_override=1.0,
    )
    return _Lod(
        visual.points, visual.normals, visual.faces, visual.resolution,
        visual.mass_per_point, visual.selections,
        (("autocenter", "0"), ("class", "road"), ("map", "road")),
    )

def _gravel_junction_lods(key: InfrastructureModelKey, texture: str) -> tuple[_Lod, ...]:
    """Build a compact terrain-coplanar gravel hub for 3/4-way junctions."""

    half_w = max(1.8, key.width_m * 0.5)
    half_l = max(half_w + 0.25, key.length_m * 0.5)
    # A small opaque core plus four feathered arms covers the branch ends while
    # retaining the same terrain-showthrough texture at the outside edges. The
    # 3-way variant is slightly smaller; its symmetry avoids brittle heading
    # guesses at skewed OSM T-junctions.
    degree = 3 if key.subtype.casefold() == "gravel_j3" else 4
    extent = half_l - (0.25 if degree == 3 else 0.0)
    core = min(half_w * 0.78, extent * 0.72)
    y = GENERATED_GRAVEL_VISUAL_TOP_METRES
    points: list[tuple[float, float, float]] = []
    faces: list[_Face] = []

    def quad(x0: float, z0: float, x1: float, z1: float, uv: tuple[tuple[float, float], ...]) -> None:
        start = len(points)
        points.extend(((x0, y, z0), (x0, y, z1), (x1, y, z1), (x1, y, z0)))
        faces.append(_Face(texture, tuple((start + index, 0, u, v) for index, (u, v) in enumerate(uv))))

    # Centre samples only the fully opaque portion of the gravel artwork.
    quad(-core, -core, core, core, ((0.20, 0.20), (0.20, 0.80), (0.80, 0.80), (0.80, 0.20)))
    # Four short arms. Give every rendered face a non-zero UV area. Two arms in
    # 0.9.202-0.9.204 collapsed all four UVs onto one line, a malformed input for
    # the original Poseidon renderer and the strongest candidate for ClipDraw's
    # reported `orIn` assertion.
    quad(-half_w, core, half_w, extent, ((0.00, 0.45), (0.00, 0.80), (1.00, 0.80), (1.00, 0.45)))
    quad(-half_w, -extent, half_w, -core, ((0.00, 0.20), (0.00, 0.55), (1.00, 0.55), (1.00, 0.20)))
    quad(core, -half_w, extent, half_w, ((0.45, 0.00), (0.45, 1.00), (0.80, 1.00), (0.80, 0.00)))
    quad(-extent, -half_w, -core, half_w, ((0.20, 0.00), (0.20, 1.00), (0.55, 1.00), (0.55, 0.00)))

    visual = _Lod(
        tuple(points), ((0.0, 1.0, 0.0),), tuple(faces), _VISUAL_LOD,
        properties=(("autocenter", "0"), ("class", "road"), ("map", "road")),
    )
    map_geometry = _Lod(
        ((-extent, 0.0, -extent), (extent, 0.0, -extent), (extent, 0.0, extent), (-extent, 0.0, extent)),
        (), (), _GEOMETRY_LOD, properties=(("map", "road"),),
    )
    roadway_y = GENERATED_GRAVEL_ROADWAY_HEIGHT_METRES
    roadway_points = ((-extent, roadway_y, -extent), (extent, roadway_y, -extent), (extent, roadway_y, extent), (-extent, roadway_y, extent))
    roadway = _Lod(roadway_points, ((0.0, 1.0, 0.0),), (_quad("", 0, 3, 2, 1),), _ROADWAY_LOD)
    land = _Lod(roadway_points, (), (), _LAND_CONTACT_LOD)
    return visual, map_geometry, roadway, land


def _road_lods(key: InfrastructureModelKey, texture: str) -> tuple[_Lod, ...]:
    if key.subtype.casefold() in {"gravel_j3", "gravel_j4"}:
        return _gravel_junction_lods(key, texture)
    width = key.width_m
    length = key.length_m
    half_w = width * 0.5
    curve_degrees = _gravel_curve_degrees(key.subtype)

    visual = _gravel_visual_lod(length, half_w, curve_degrees, texture)

    # Keep road simulation on the first Resolution LOD, but also emit a face-less
    # Geometry LOD carrying map=road. OFP/CWA reliably reads the 2D map symbol
    # from Geometry, while zero faces mean this metadata LOD adds no collision.
    half_l = length * 0.5
    map_geometry = _Lod(
        ((-half_w, 0.0, -half_l), (half_w, 0.0, -half_l),
         (half_w, 0.0, half_l), (-half_w, 0.0, half_l)),
        (), (), _GEOMETRY_LOD, properties=(("map", "road"),),
    )
    roadway_sections = _road_ribbon_sections(length, half_w, curve_degrees, overhang=0.0)
    roadway = _ribbon_lod(
        roadway_sections, texture=texture, resolution=_ROADWAY_LOD,
        height=GENERATED_GRAVEL_ROADWAY_HEIGHT_METRES, lowered_overlap=False,
        double_sided=False, u_span_override=1.0,
    )
    first = roadway_sections[0]
    last = roadway_sections[-1]
    land = _Lod(
        ((first[0], 0.0, first[1]), (first[2], 0.0, first[3]),
         (last[0], 0.0, last[1]), (last[2], 0.0, last[3])),
        (), (), _LAND_CONTACT_LOD,
    )
    return visual, map_geometry, roadway, land


def _rock_lods(key: InfrastructureModelKey, texture: str) -> tuple[_Lod, ...]:
    width = key.width_m
    length = key.length_m
    variant = int(key.subtype.rsplit("_", 1)[-1]) if "_" in key.subtype else 0
    offsets = ((-0.22, -0.18, 0.50), (0.23, 0.16, 0.38), (0.05, -0.02, 0.30))
    points: list[tuple[float, float, float]] = []
    faces: list[_Face] = []
    boxes: list[tuple[float, float, float, float, float, float]] = []
    for index, (ox, oz, scale) in enumerate(offsets[: 2 + variant % 2]):
        cx = ox * width
        cz = oz * length
        rx = max(0.5, width * scale * 0.28)
        rz = max(0.5, length * scale * 0.28)
        h = max(0.7, (rx + rz) * 0.65)
        p = len(points)
        points.extend(((cx - rx, 0.0, cz - rz), (cx + rx, 0.0, cz - rz), (cx + rx, 0.0, cz + rz), (cx - rx, 0.0, cz + rz), (cx, h, cz)))
        faces.extend((
            _Face(texture, ((p + 0, 0, 0.0, 1.0), (p + 4, 0, 0.5, 0.0), (p + 1, 0, 1.0, 1.0))),
            _Face(texture, ((p + 1, 0, 0.0, 1.0), (p + 4, 0, 0.5, 0.0), (p + 2, 0, 1.0, 1.0))),
            _Face(texture, ((p + 2, 0, 0.0, 1.0), (p + 4, 0, 0.5, 0.0), (p + 3, 0, 1.0, 1.0))),
            _Face(texture, ((p + 3, 0, 0.0, 1.0), (p + 4, 0, 0.5, 0.0), (p + 0, 0, 1.0, 1.0))),
            _Face(texture, ((p + 0, 0, 0.0, 0.0), (p + 1, 0, 1.0, 0.0), (p + 2, 0, 1.0, 1.0), (p + 3, 0, 0.0, 1.0))),
        ))
        boxes.append((cx - rx * 0.75, cx + rx * 0.75, 0.0, h * 0.7, cz - rz * 0.75, cz + rz * 0.75))
    visual = _Lod(tuple(points), ((0.0, 1.0, 0.0),), tuple(faces), _VISUAL_LOD, properties=(("autocenter", "0"),))
    geometry = _geometry_from_boxes(tuple(boxes))
    land = _Lod(tuple((box[0], 0.0, box[4]) for box in boxes) + tuple((box[1], 0.0, box[5]) for box in boxes), (), (), _LAND_CONTACT_LOD)
    return visual, geometry, land


def _utility_lods(key: InfrastructureModelKey, texture: str) -> tuple[_Lod, ...]:
    points: list[tuple[float, float, float]] = []
    faces: list[_Face] = []
    boxes: list[tuple[float, float, float, float, float, float]] = []

    def box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> None:
        boxes.append((x0, x1, y0, y1, z0, z1))
        _append_box(points, faces, x0=x0, x1=x1, y0=y0, y1=y1, z0=z0, z1=z1, texture=texture)

    if key.subtype == "power_pole":
        box(-0.14, 0.14, 0.0, 9.0, -0.14, 0.14)
        box(-2.0, 2.0, 7.8, 8.05, -0.12, 0.12)
        for x in (-1.65, 0.0, 1.65):
            box(x - 0.05, x + 0.05, 8.0, 8.45, -0.05, 0.05)
    elif key.subtype == "power_tower":
        for x in (-1.25, 1.25):
            for z in (-1.25, 1.25):
                box(x - 0.12, x + 0.12, 0.0, 13.0, z - 0.12, z + 0.12)
        for y, span in ((6.0, 2.6), (10.0, 3.4), (12.2, 4.4)):
            box(-span, span, y, y + 0.22, -0.12, 0.12)
    elif key.subtype == "water_tower":
        for x in (-1.5, 1.5):
            for z in (-1.5, 1.5):
                box(x - 0.18, x + 0.18, 0.0, 10.5, z - 0.18, z + 0.18)
        box(-2.8, 2.8, 10.0, 14.2, -2.8, 2.8)
        box(-3.05, 3.05, 14.2, 14.55, -3.05, 3.05)
    else:
        raise ValueError(f"unknown utility subtype: {key.subtype}")

    visual = _Lod(tuple(points), ((0.0, 1.0, 0.0),), tuple(faces), _VISUAL_LOD, properties=(("autocenter", "0"),))
    geometry = _geometry_from_boxes(tuple(boxes))
    half = 3.1 if key.subtype == "water_tower" else 1.5
    land = _Lod(((-half, 0.0, -half), (half, 0.0, -half), (-half, 0.0, half), (half, 0.0, half)), (), (), _LAND_CONTACT_LOD)
    return visual, geometry, land


def utility_model_path(world_name: str, subtype: str) -> str:
    if subtype not in {"power_pole", "power_tower", "water_tower"}:
        raise ValueError(f"unknown utility subtype: {subtype}")
    return rf"{world_name}\i\util_{subtype}.p3d"


def write_infrastructure_mlod(path: Path, key: InfrastructureModelKey, texture_path: str) -> None:
    if key.kind == "barrier":
        lods = _barrier_lods(key, texture_path)
    elif key.kind == "bridge":
        lods = _bridge_lods(key, texture_path)
    elif key.kind == "road":
        lods = _road_lods(key, texture_path)
    elif key.kind == "rock":
        lods = _rock_lods(key, texture_path)
    elif key.kind == "utility":
        lods = _utility_lods(key, texture_path)
    else:
        raise ValueError(f"unknown infrastructure model kind: {key.kind}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, len(lods)))
        for lod in lods:
            _write_lod(stream, lod)


def gravel_road_model_path(world_name: str, nominal_length: int = 25, curve_degrees: int = 0) -> str:
    if nominal_length not in {3, 6, 12, 25}:
        raise ValueError("gravel road nominal length must be 3, 6, 12, or 25 metres")
    if curve_degrees == 0:
        filename = f"gravel{nominal_length}.p3d"
    else:
        amount = abs(int(curve_degrees))
        if amount not in GENERATED_GRAVEL_CURVE_BUCKETS:
            raise ValueError(f"gravel road curve must be one of {GENERATED_GRAVEL_CURVE_BUCKETS}")
        side = "r" if curve_degrees > 0 else "l"
        filename = f"gravel{nominal_length}_{side}{amount:02d}.p3d"
    return rf"{world_name}\i\{filename}"


def gravel_junction_model_path(world_name: str, degree: int) -> str:
    if degree not in {3, 4}:
        raise ValueError("gravel junction degree must be 3 or 4")
    return rf"{world_name}\i\gravel_j{degree}.p3d"


def is_generated_gravel_junction_model(model_path: str) -> bool:
    filename = model_path.replace("/", "\\").rsplit("\\", 1)[-1]
    return re.fullmatch(r"gravel_j[34]\.p3d", filename, re.IGNORECASE) is not None


def gravel_curve_model_path(model_path: str, curve_degrees: int) -> str:
    normalized = model_path.replace("/", "\\")
    prefix, filename = normalized.rsplit("\\", 1)
    match = re.fullmatch(r"gravel(25|12|6|3)(?:_[lr](?:05|10|15|20|30|45))?\.p3d", filename, re.IGNORECASE)
    if not match:
        return model_path
    nominal = int(match.group(1))
    world_name = prefix.split("\\", 1)[0]
    return gravel_road_model_path(world_name, nominal, curve_degrees)


def is_generated_gravel_road_model(model_path: str) -> bool:
    filename = model_path.replace("/", "\\").rsplit("\\", 1)[-1]
    return re.fullmatch(r"gravel(?:25|12|6|3)(?:_[lr](?:05|10|15|20|30|45))?\.p3d", filename, re.IGNORECASE) is not None


@dataclass(frozen=True, slots=True)
class _InfrastructureAssetTask:
    key: InfrastructureModelKey
    wire: str
    relative: str
    destination: Path
    texture: str
    cache_path: Path | None
    cache_enabled: bool
    cache_refresh: bool
    usage_count: int


def _write_infrastructure_asset_task(task: _InfrastructureAssetTask) -> tuple[dict[str, object], bool]:
    hit = restore_or_create_file(
        cache_path=task.cache_path,
        destination=task.destination,
        producer=lambda target: write_infrastructure_mlod(target, task.key, task.texture),
        enabled=task.cache_enabled,
        refresh=task.cache_refresh,
    )
    summary = inspect_mlod(task.destination)
    if task.key.kind in {"bridge", "road"} and not any(
        math.isclose(value, _ROADWAY_LOD, rel_tol=1e-6) for value in summary.resolutions
    ):
        raise ValueError(f"generated {task.key.kind} lost its Roadway LOD")
    return ({
        "key": asdict(task.key),
        "model_path": task.wire,
        "relative_path": task.relative,
        "usage_count": task.usage_count,
        "sha256": sha256(task.destination.read_bytes()).hexdigest(),
        "lod_resolutions": summary.resolutions,
    }, hit)


def _infrastructure_texture_kind(key: InfrastructureModelKey) -> str:
    """Return the generated texture family used by one infrastructure model."""
    if key.kind == "road":
        # Junction polygons tile their UV coordinates in two dimensions. They
        # must not use the alpha-edged ribbon texture, otherwise each texture
        # wrap reveals a thin strip of grass through the middle of the junction.
        return "gravel_junction" if key.subtype.casefold().startswith("gravel_j") else "gravel"
    return key.subtype if key.kind in {"barrier", "utility"} else key.kind


class ProceduralInfrastructureLibrary:
    _PATTERN = re.compile(r"^(bar|br|rock)_([a-z0-9_]+)_w(\d+)_l(\d+)\.p3d$", re.IGNORECASE)
    _GRAVEL_PATTERN = re.compile(r"^gravel(25|12|6|3)(?:_[lr](?:05|10|15|20|30|45))?\.p3d$", re.IGNORECASE)
    _GRAVEL_JUNCTION_PATTERN = re.compile(r"^gravel_j([34])\.p3d$", re.IGNORECASE)
    _UTILITY_PATTERN = re.compile(r"^util_(power_pole|power_tower|water_tower)\.p3d$", re.IGNORECASE)

    def __init__(self, world_name: str, *, road_segment_length: float = 24.5, cache_dir: Path | None = None, cache_enabled: bool = True, cache_refresh: bool = False) -> None:
        self.world_name = world_name
        self.road_segment_length = float(road_segment_length)
        if not math.isfinite(self.road_segment_length) or self.road_segment_length <= 0.0:
            raise ValueError("road segment length must be positive and finite")
        self.cache_dir = cache_dir
        self.cache_enabled = cache_enabled
        self.cache_refresh = cache_refresh
        self.cache_hits = 0
        self.cache_misses = 0
        self._usage: Counter[InfrastructureModelKey] = Counter()

    def model_path(self, key: InfrastructureModelKey) -> str:
        if key.kind == "road":
            return rf"{self.world_name}\i\{key.subtype}.p3d"
        if key.kind == "utility":
            return utility_model_path(self.world_name, key.subtype)
        prefix = {"barrier": "bar", "bridge": "br", "rock": "rock"}[key.kind]
        return rf"{self.world_name}\i\{prefix}_{key.subtype}_w{key.width_dm:03d}_l{key.length_dm:03d}.p3d"

    def barrier_model(self, subtype: str, length_m: float = 6.0) -> str:
        if subtype.casefold() == "fence":
            raise ValueError("fences must use stock OFP/CWA models")
        key = InfrastructureModelKey("barrier", subtype, 0, max(10, int(round(length_m * 10.0))))
        self._usage[key] += 1
        return self.model_path(key)

    def bridge_model(self, subtype: str, width_m: float, length_m: float) -> str:
        if subtype not in {"start", "middle", "end", "single"}:
            raise ValueError(f"unknown bridge subtype: {subtype}")
        key = InfrastructureModelKey("bridge", subtype, max(35, int(round(width_m * 10.0))), max(30, int(round(length_m * 10.0))))
        self._usage[key] += 1
        return self.model_path(key)

    def rock_model(self, subtype: str, width_m: float, length_m: float) -> str:
        key = InfrastructureModelKey("rock", subtype, max(40, int(round(width_m * 10.0))), max(40, int(round(length_m * 10.0))))
        self._usage[key] += 1
        return self.model_path(key)

    def utility_model(self, subtype: str) -> str:
        dimensions = {
            "power_pole": (4.0, 4.0),
            "power_tower": (9.0, 9.0),
            "water_tower": (6.5, 6.5),
        }
        if subtype not in dimensions:
            raise ValueError(f"unknown utility subtype: {subtype}")
        width, length = dimensions[subtype]
        key = InfrastructureModelKey("utility", subtype, int(round(width * 10.0)), int(round(length * 10.0)))
        self._usage[key] += 1
        return self.model_path(key)

    def is_generated_model(self, model_path: str) -> bool:
        prefix = (self.world_name + "\\i\\").casefold()
        return model_path.casefold().startswith(prefix) and model_path.casefold().endswith(".p3d")

    def register_model_usage(self, model_path: str, count: int = 1) -> None:
        count = max(0, int(count))
        if count == 0:
            return
        if not self.is_generated_model(model_path):
            return
        filename = model_path.rsplit("\\", 1)[-1]
        junction_match = self._GRAVEL_JUNCTION_PATTERN.fullmatch(filename)
        if junction_match:
            degree = int(junction_match.group(1))
            self._usage[InfrastructureModelKey(
                "road", filename[:-4].casefold(),
                int(round(GENERATED_GRAVEL_HALF_WIDTH_METRES * 20.0)),
                54 if degree == 3 else 60,
            )] += count
            return
        gravel_match = self._GRAVEL_PATTERN.fullmatch(filename)
        if gravel_match:
            nominal = int(gravel_match.group(1))
            actual_length = self.road_segment_length * nominal / 25.0
            self._usage[InfrastructureModelKey(
                "road", filename[:-4].casefold(), int(round(GENERATED_GRAVEL_HALF_WIDTH_METRES * 20.0)), max(10, int(round(actual_length * 10.0)))
            )] += count
            return
        utility_match = self._UTILITY_PATTERN.fullmatch(filename)
        if utility_match:
            subtype = utility_match.group(1).casefold()
            dimensions = {"power_pole": (40, 40), "power_tower": (90, 90), "water_tower": (65, 65)}
            width, length = dimensions[subtype]
            self._usage[InfrastructureModelKey("utility", subtype, width, length)] += count
            return
        match = self._PATTERN.fullmatch(filename)
        if not match:
            raise ValueError(f"invalid generated infrastructure model path: {model_path}")
        prefix, subtype, width, length = match.groups()
        kind = {"bar": "barrier", "br": "bridge", "rock": "rock"}[prefix.casefold()]
        self._usage[InfrastructureModelKey(kind, subtype.casefold(), int(width), int(length))] += count

    def register_model(self, model_path: str) -> None:
        self.register_model_usage(model_path, 1)

    def register_models(self, model_paths: Iterable[str]) -> None:
        for model_path in model_paths:
            self.register_model(model_path)

    def _texture_path(self, key: InfrastructureModelKey) -> str:
        if key.kind == "rock":
            # Reuse the actual Resistance O.pbo rock terrain artwork instead of
            # generating a mismatched grey object texture. Alternate the two
            # verified rock tiles across deterministic rock-group variants.
            return r"o\lom2.paa" if key.subtype.casefold().endswith("_1") else r"o\l1.paa"
        kind = _infrastructure_texture_kind(key)
        return rf"{self.world_name}\i\{_texture_file_stem(kind)}.paa"

    def write_assets(self, source_dir: Path, catalogue_path: Path) -> InfrastructureAssetResult:
        used_texture_kind_set = {
            _infrastructure_texture_kind(key)
            for key in self._usage
            if key.kind != "rock"
        }
        used_texture_kinds = sorted(used_texture_kind_set)
        texture_files: list[str] = []
        for kind in used_texture_kinds:
            wire = rf"{self.world_name}\i\{_texture_file_stem(kind)}.paa"
            relative = wire.split("\\", 1)[1].replace("\\", "/")
            destination = source_dir / relative
            if kind == "gravel":
                asset_key = cache_key(
                    "procedural-infrastructure-texture-v15-reference-gravel-clean-edge",
                    {"kind": kind, "size": 512, "recipe": "reference-gravel-photo-clean-edge-v3"},
                )
                producer = lambda target: write_rgba_dxt1_paa(
                    target, create_gravel_road_texture_image(512)
                )
            elif kind == "gravel_junction":
                asset_key = cache_key(
                    "procedural-infrastructure-texture-v16-reference-gravel-junction-opaque",
                    {"kind": kind, "size": 512, "recipe": "reference-gravel-photo-opaque-junction-v1"},
                )
                producer = lambda target: write_rgb_dxt1_paa(
                    target, create_gravel_junction_texture_image(512)
                )
            else:
                texture_size = 256 if kind == "bridge" else 128
                texture_cache_version = (
                    "procedural-infrastructure-texture-v6-procedural-bridge"
                    if kind == "bridge"
                    else "procedural-infrastructure-texture-v5-osm-utilities"
                )
                asset_key = cache_key(
                    texture_cache_version,
                    {"kind": kind, "size": texture_size},
                )
                producer = lambda target, kind=kind, size=texture_size: write_rgb_dxt1_paa(target, _texture_image(kind, size))
            cached = self.cache_dir / "procedural-assets" / f"{asset_key}.paa" if self.cache_dir else None
            hit = restore_or_create_file(
                cache_path=cached,
                destination=destination,
                producer=producer,
                enabled=self.cache_enabled,
                refresh=self.cache_refresh,
            )
            self.cache_hits += int(hit)
            self.cache_misses += int(not hit)
            inspect_paa(destination)
            texture_files.append(relative)

        gravel_source: dict[str, object] | None = None
        if {"gravel", "gravel_junction"} & set(used_texture_kinds):
            stale_edge = source_dir / "i" / "ge.paa"
            if stale_edge.exists():
                stale_edge.unlink()
            source_rules_bytes = _EDEN_GRAVEL_SURFACES.read_bytes()
            source_rules = _eden_gravel_surface_rules()
            source_rules_path = source_dir / "i" / "gravel-source-surfaces.txt"
            source_rules_path.parent.mkdir(parents=True, exist_ok=True)
            source_rules_path.write_bytes(source_rules_bytes)
            gravel_source = {
                "type": "bundled-reference",
                "texture": f"i/{_texture_file_stem('gravel')}.paa",
                "texture_recipe": "reference-gravel-photo-clean-edge-v3",
                "texture_size": 512,
                "edge_blend": "clean DXT1 cutout plus smoothly irregular model edge",
                "map_symbol": "road",
                "map_symbol_lod": "face-less Geometry",
                "surface_rules": "Eden_roadtype_textures/surfaces.txt",
                "surface_rules_sha256": sha256(source_rules_bytes).hexdigest(),
                "surface_rule_values": source_rules,
                "embedded_surface_rules": "i/gravel-source-surfaces.txt",
            }
            if "gravel_junction" in used_texture_kinds:
                gravel_source.update({
                    "junction_texture": f"i/{_texture_file_stem('gravel_junction')}.paa",
                    "junction_texture_recipe": "reference-gravel-photo-opaque-junction-v1",
                    "junction_texture_alpha": "opaque",
                })

        model_tasks: list[_InfrastructureAssetTask] = []
        for key in sorted(self._usage):
            wire = self.model_path(key)
            relative = wire.split("\\", 1)[1].replace("\\", "/")
            destination = source_dir / relative
            texture = self._texture_path(key)
            model_cache_version = (
                "procedural-infrastructure-model-v15-safe-junction-uvs"
                if key.kind == "road"
                else "procedural-infrastructure-model-v17-single-span-segmented-collision"
                if key.kind == "bridge"
                else "procedural-infrastructure-model-v5-osm-utilities"
            )
            asset_key = cache_key(
                model_cache_version,
                {"world": self.world_name, "key": asdict(key), "texture": texture},
            )
            cached = self.cache_dir / "procedural-assets" / f"{asset_key}.p3d" if self.cache_dir else None
            model_tasks.append(_InfrastructureAssetTask(
                key=key, wire=wire, relative=relative, destination=destination,
                texture=texture, cache_path=cached, cache_enabled=self.cache_enabled,
                cache_refresh=self.cache_refresh, usage_count=self._usage[key],
            ))

        model_results = process_asset_tasks(_write_infrastructure_asset_task, model_tasks)
        models: list[dict[str, object]] = []
        model_files: list[str] = []
        for model, hit in model_results:
            self.cache_hits += int(hit)
            self.cache_misses += int(not hit)
            models.append(model)
            model_files.append(str(model["relative_path"]))

        document: dict[str, object] = {
            "schema": 1,
            "generator": "cwr-worldgen procedural infrastructure",
            "placements": sum(self._usage.values()),
            "generated_variants": len(models),
            "textures": texture_files,
            "models": models,
        }
        if gravel_source is not None:
            document["gravel_texture_source"] = gravel_source
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        document["catalogue_sha256"] = digest
        catalogue_path.parent.mkdir(parents=True, exist_ok=True)
        catalogue_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        embedded = source_dir / "i" / "infrastructure.json"
        embedded.parent.mkdir(parents=True, exist_ok=True)
        embedded.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return InfrastructureAssetResult(
            placements=sum(self._usage.values()),
            generated_variants=len(models),
            catalogue_sha256=digest,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            model_files=tuple(model_files),
            texture_files=tuple(texture_files),
        )
