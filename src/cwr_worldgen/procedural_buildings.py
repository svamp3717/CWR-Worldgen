# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
import json
import math
import re
import shutil
import struct
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFilter
from shapely.geometry import Point, Polygon

from .cache import cache_key, restore_or_create_file
from .building_semantics import detect_region, is_actual_church
from .paa import inspect_paa, write_rgb_dxt1_paa

_MLOD_HEADER = struct.Struct("<4sBBHI")
_SP3X_HEADER = struct.Struct("<4siiiiii")
_POINT = struct.Struct("<fffi")
_NORMAL = struct.Struct("<fff")
_FACE_VERTEX = struct.Struct("<iiff")
_FACE_TRAILER = struct.Struct("<i")
_TAG_HEADER = struct.Struct("<64si")
_FLOAT = struct.Struct("<f")

_GEOMETRY_LOD = 1.0e13
_LAND_CONTACT_LOD = 2.0e15
_MAX_GEOMETRY_COMPONENT_SPAN_M = 40.0
DEFAULT_BUILDING_TEXTURE_VARIANTS = 10
BUILDING_ASSET_TEXTURE_SIZE = 128
HIGH_QUALITY_BUILDING_ASSET_TEXTURE_SIZE = 256
_BUILDING_TEXTURE_LOGICAL_SIZE = 64
# Reused procedural variants should stay close to the requested OSM footprint.
# 0.9.90 intentionally started broad; 0.9.92 tightens the envelope so physical
# size wins over cosmetic roof/palette matching when the variant cap is hit.
BUILDING_REUSE_MIN_DIMENSION_RATIO = 0.70
BUILDING_REUSE_MAX_DIMENSION_RATIO = 1.15
BUILDING_REUSE_MIN_AREA_RATIO = 0.65
BUILDING_REUSE_MAX_AREA_RATIO = 1.20
INTERIOR_ELIGIBLE_FAMILIES = frozenset({
    "residential", "townhouse", "urban", "school", "shop",
})
WHITE_WINDOW_TRIM_REGIONAL_STYLES = frozenset({
    "sweden_red",
    "sweden_yellow",
    "eastern_whitewash",
    "africa_whitewash",
    "middle_east_whitewash",
    "western_stucco",
})
INTERIOR_MAXIMUM_WIDTH_M = 30.0
INTERIOR_MAXIMUM_LENGTH_M = 40.0
INTERIOR_MAXIMUM_HEIGHT_M = 15.0
INTERIOR_DOOR_HEIGHT_M = 2.2
INTERIOR_WINDOW_SILL_M = 0.9
INTERIOR_WINDOW_TOP_M = 2.05
FACADE_TILE_HEIGHT_M = 3.0
PAINTED_WINDOW_MINIMUM_SILL_M = 1.0
_PAINTED_WINDOW_FAMILIES = frozenset({
    "residential", "townhouse", "urban", "church", "school",
})
_DISTANCE_VALUE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)")

PointXZ = tuple[float, float]


@dataclass(frozen=True, order=True, slots=True)
class BuildingVariantKey:
    family: str
    roof_style: str
    width_m: float
    length_m: float
    height_m: float
    foundation_depth_m: float = 0.5
    regional_style: str = "default"
    texture_variant: int = 0
    interiors: bool = False

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return sha256(self.canonical().encode("ascii")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class BuildingPlacement:
    model_path: str
    heading_degrees: float
    requested: BuildingVariantKey
    selected: BuildingVariantKey


@dataclass(frozen=True, slots=True)
class GeneratedBuildingAsset:
    key: BuildingVariantKey
    model_path: str
    relative_path: str
    usage_count: int
    sha256: str
    lod_count: int
    point_count: int
    face_count: int
    texture_paths: tuple[str, ...]
    visual_face_count: int
    geometry_component_count: int
    geometry_mass_point_count: int
    map_symbol: str


@dataclass(frozen=True, slots=True)
class BuildingGenerationResult:
    enabled: bool
    placements: int
    unique_requested_variants: int
    generated_variants: int
    reused_placements: int
    reuse_ratio: float
    capped_variants: int
    model_assets: tuple[GeneratedBuildingAsset, ...]
    texture_files: tuple[str, ...]
    catalogue_sha256: str
    cache_hits: int = 0
    cache_misses: int = 0

    def to_manifest(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "placements": self.placements,
            "unique_requested_variants": self.unique_requested_variants,
            "generated_variants": self.generated_variants,
            "reused_placements": self.reused_placements,
            "reuse_ratio": self.reuse_ratio,
            "capped_variants": self.capped_variants,
            "texture_files": self.texture_files,
            "catalogue_sha256": self.catalogue_sha256,
            "models": [asdict(asset) for asset in self.model_assets],
        }


@dataclass(frozen=True, slots=True)
class MlodSummary:
    version_major: int
    version_minor: int
    lod_count: int
    resolutions: tuple[float, ...]
    point_count: int
    normal_count: int
    face_count: int
    texture_paths: tuple[str, ...]
    point_counts: tuple[int, ...] = ()
    face_counts: tuple[int, ...] = ()
    selection_names: tuple[tuple[str, ...], ...] = ()
    mass_point_counts: tuple[int, ...] = ()
    named_properties: tuple[tuple[tuple[str, str], ...], ...] = ()


@dataclass(frozen=True, slots=True)
class _Footprint:
    width_m: float
    length_m: float
    heading_degrees: float


@dataclass(frozen=True, slots=True)
class _Face:
    texture: str
    vertices: tuple[tuple[int, int, float, float], ...]
    flags: int = 0


@dataclass(frozen=True, slots=True)
class _NamedSelection:
    name: str
    point_weights: bytes
    face_flags: bytes


@dataclass(frozen=True, slots=True)
class _Lod:
    points: tuple[tuple[float, float, float], ...]
    normals: tuple[tuple[float, float, float], ...]
    faces: tuple[_Face, ...]
    resolution: float
    mass_per_point: tuple[float, ...] = ()
    selections: tuple[_NamedSelection, ...] = ()
    properties: tuple[tuple[str, str], ...] = ()


_FAMILY_COLOURS: dict[str, tuple[int, int, int]] = {
    # Muted, slightly dirty OFP/CWA-era material palette. These are deliberately
    # low-saturation and mid-value so buildings sit inside Everon/Eden rather
    # than glowing like freshly exported CAD models.
    "residential": (156, 150, 132),
    "townhouse": (151, 144, 129),
    "urban": (139, 139, 132),
    "industrial": (116, 119, 111),
    "agricultural": (112, 96, 74),
    "outbuilding": (118, 104, 82),
    "church": (166, 160, 145),
    "school": (151, 142, 123),
    "shop": (143, 132, 116),
}
_ROOF_COLOURS: dict[str, tuple[int, int, int]] = {
    "flat": (76, 78, 73),
    "gabled": (104, 61, 46),
}


def _shade(colour: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    return tuple(max(0, min(255, channel + amount)) for channel in colour)


def _deterministic_grain(image: Image.Image, strength: int, salt: int) -> None:
    pixels = image.load()
    width, height = image.size
    for y in range(height):
        for x in range(width):
            value = ((x * 1103515245 + y * 12345 + salt * 2654435761) >> 16) & 0xFF
            delta = (value % (strength * 2 + 1)) - strength
            pixels[x, y] = _shade(tuple(pixels[x, y]), delta)


def _weather_stains(image: Image.Image, salt: int, amount: int = 16) -> None:
    """Add broad deterministic stains rather than modern high-frequency noise."""
    draw = ImageDraw.Draw(image, "RGB")
    width, height = image.size
    for index in range(8):
        value = (salt * 2654435761 + index * 2246822519) & 0xFFFFFFFF
        x = int((value & 0xFF) / 255 * width)
        y = int(((value >> 8) & 0xFF) / 255 * height)
        rx = 5 + ((value >> 16) & 0x0F)
        ry = 7 + ((value >> 20) & 0x1F)
        sample = image.getpixel((min(width - 1, x), min(height - 1, y)))
        colour = _shade(sample, -amount + index % 3 * 5)
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=colour)


class _ScaledImageDraw:
    """Scale the established 64px facade coordinates onto a larger canvas."""

    def __init__(self, image: Image.Image, scale: float) -> None:
        self._draw = ImageDraw.Draw(image)
        self._scale = max(1.0, float(scale))

    def _coords(self, values: Sequence[float | int]) -> tuple[int, ...]:
        return tuple(round(float(value) * self._scale) for value in values)

    def rectangle(self, xy: Sequence[float | int], *, fill=None, outline=None, width: int = 1) -> None:
        self._draw.rectangle(
            self._coords(xy), fill=fill, outline=outline,
            width=max(1, round(width * self._scale)),
        )

    def line(self, xy: Sequence[float | int], *, fill=None, width: int = 1) -> None:
        self._draw.line(
            self._coords(xy), fill=fill, width=max(1, round(width * self._scale))
        )

    def ellipse(self, xy: Sequence[float | int], *, fill=None, outline=None, width: int = 1) -> None:
        self._draw.ellipse(
            self._coords(xy), fill=fill, outline=outline,
            width=max(1, round(width * self._scale)),
        )


def _pixel_canvas(
    base: tuple[int, int, int], size: int, salt: int, grain: int = 4
) -> tuple[Image.Image, ImageDraw.ImageDraw | _ScaledImageDraw]:
    # Façade composition remains a stable 64x64 logical design. For HQ assets,
    # draw that design directly onto a 128px working canvas before the final
    # 256px export. Frames, board gaps, mortar and roof edges therefore receive
    # actual extra samples instead of merely stretching a finished 64px image.
    work = max(_BUILDING_TEXTURE_LOGICAL_SIZE, size // 2)
    image = Image.new("RGB", (work, work), base)
    _deterministic_grain(image, grain, salt)
    _weather_stains(image, salt, 10)
    scale = work / _BUILDING_TEXTURE_LOGICAL_SIZE
    if scale <= 1.0:
        return image, ImageDraw.Draw(image)
    return image, _ScaledImageDraw(image, scale)

def _fine_texture_overlay(size: int, salt: int) -> Image.Image:
    """Return low-amplitude deterministic material variation for HQ exports.

    A tiny smoothly enlarged noise field adds real high-resolution information
    instead of merely stretching a 64px source.  Keeping the amplitude small is
    important for DXT1 and for the old engine's mipmaps: visible static is not
    detail, regardless of how enthusiastically computers produce it.
    """

    field_size = max(16, min(48, size // 8))
    state = (int(salt) ^ 0x9E3779B9) & 0xFFFFFFFF
    values: list[int] = []
    for _ in range(field_size * field_size):
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        values.append(124 + ((state >> 28) & 0x07))
    field = Image.new("L", (field_size, field_size))
    field.putdata(values)
    field = field.resize((size, size), Image.Resampling.BICUBIC)
    return Image.merge("RGB", (field, field, field))


def _finish_pixel_texture(image: Image.Image, size: int) -> Image.Image:
    # Preserve the legacy 128px helper output used by compatibility tests and
    # previews.  Generated game assets use 256px and take the higher-quality
    # path below.
    if size <= 128:
        return image.resize((size, size), Image.Resampling.NEAREST)

    enlarged = image.resize((size, size), Image.Resampling.LANCZOS)
    digest = sha256(image.tobytes()).digest()
    salt = int.from_bytes(digest[:4], "little")
    enlarged = ImageChops.add(
        enlarged, _fine_texture_overlay(size, salt), scale=1.0, offset=-127
    )
    # A restrained unsharp pass restores frame/plank definition after Lanczos
    # without returning to staircase edges.
    return enlarged.filter(ImageFilter.UnsharpMask(radius=0.7, percent=65, threshold=3))


def _raise_painted_windows_above_ground(image: Image.Image) -> Image.Image:
    """Raise painted windows without baking a foundation into every wall tile.

    Closed procedural buildings repeat the wall atlas once per roughly three
    metres of wall height.  Older atlases put a dark stone/plaster footing in
    their bottom 12/64 rows.  Stretching that footing to obtain a one-metre
    window sill made the footing repeat at *every storey*, producing the broad
    grey bands visible between floors.

    Keep the upper facade/window artwork, compress it enough to reserve the
    requested sill, then fill the lower metre with a window-free sample of the
    same wall material.  The actual foundation is separate model geometry and
    therefore appears only at terrain level.
    """

    width, height = image.size
    if width <= 0 or height <= 1:
        return image
    source_footing_top = max(1, min(height - 1, round(height * 52.0 / 64.0)))
    target_window_bottom = max(
        1,
        min(
            source_footing_top,
            round(
                height
                * (1.0 - PAINTED_WINDOW_MINIMUM_SILL_M / FACADE_TILE_HEIGHT_M)
            ),
        ),
    )
    if target_window_bottom >= source_footing_top:
        return image

    upper = image.crop((0, 0, width, source_footing_top)).resize(
        (width, target_window_bottom), Image.Resampling.NEAREST
    )

    # The first ~5/64 rows precede all current painted window artwork and
    # contain only the regional wall material/pattern. Reusing that strip for
    # the lower metre preserves timber boards, brick courses, render, etc.
    # without reintroducing a fake stone plinth into every repeated storey.
    clean_band_height = max(1, min(source_footing_top, round(height * 5.0 / 64.0)))
    clean_wall = image.crop((0, 0, width, clean_band_height)).resize(
        (width, height - target_window_bottom), Image.Resampling.NEAREST
    )
    raised = Image.new(image.mode, image.size)
    raised.paste(upper, (0, 0))
    raised.paste(clean_wall, (0, target_window_bottom))
    return raised


def _draw_old_window(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], frame: tuple[int, int, int] = (151, 145, 128)) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=(44, 51, 51), outline=_shade(frame, -35), width=2)
    draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), fill=(54, 67, 69), outline=frame, width=1)
    midx, midy = (x0 + x1) // 2, (y0 + y1) // 2
    draw.line((midx, y0 + 2, midx, y1 - 2), fill=_shade(frame, -12), width=1)
    draw.line((x0 + 2, midy, x1 - 2, midy), fill=_shade(frame, -12), width=1)
    draw.line((x0 + 4, y0 + 4, midx - 2, midy - 2), fill=(84, 96, 93), width=1)


_TEXTURE_VARIANT_COLOUR_OFFSETS: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (7, 3, -2),
    (-6, -2, 4),
    (3, -5, -7),
    (-3, 5, 6),
    (9, -1, -5),
    (-8, 3, 1),
    (4, 6, -3),
    (-4, -6, 5),
    (6, -4, 7),
)


def _normalise_texture_variant(texture_variant: int, count: int = DEFAULT_BUILDING_TEXTURE_VARIANTS) -> int:
    count = max(1, int(count))
    return int(texture_variant) % count


def _variant_colour(colour: tuple[int, int, int], texture_variant: int) -> tuple[int, int, int]:
    variant = _normalise_texture_variant(texture_variant, len(_TEXTURE_VARIANT_COLOUR_OFFSETS))
    offsets = _TEXTURE_VARIANT_COLOUR_OFFSETS[variant]
    return tuple(max(0, min(255, channel + offsets[index])) for index, channel in enumerate(colour))


def _placement_texture_variant(
    tags: Mapping[str, str],
    coordinates: Sequence[PointXZ],
    *,
    variant_count: int = DEFAULT_BUILDING_TEXTURE_VARIANTS,
) -> int:
    """Select a stable façade variant from the building's tags and position."""

    variant_count = max(1, int(variant_count))
    document = {
        "tags": sorted((str(key), str(value)) for key, value in tags.items()),
        "coordinates": [
            [round(float(x), 3), round(float(z), 3)] for x, z in coordinates
        ],
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    digest = sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % variant_count


def _regional_wall_base(family: str, regional_style: str) -> tuple[int, int, int]:
    if regional_style == "sweden_red" and family in {"residential", "townhouse", "agricultural", "outbuilding"}:
        return (123, 53, 43)
    if regional_style == "sweden_yellow" and family in {"residential", "townhouse", "outbuilding"}:
        return (174, 139, 67)
    if regional_style == "eastern_plaster":
        return {
            "residential": (166, 151, 126), "townhouse": (157, 146, 128),
            "urban": (145, 145, 137), "agricultural": (145, 127, 101),
            "school": (159, 149, 126), "shop": (149, 138, 119),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "eastern_brick":
        return {
            "residential": (126, 72, 54), "townhouse": (119, 68, 53),
            "urban": (113, 75, 64), "agricultural": (112, 68, 49),
            "school": (127, 79, 61), "shop": (122, 74, 57),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "eastern_whitewash":
        return {
            "residential": (188, 181, 158), "townhouse": (181, 174, 151),
            "urban": (168, 166, 154), "agricultural": (177, 170, 143),
            "school": (184, 177, 151), "shop": (177, 169, 146),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "eastern_panel":
        return {
            "townhouse": (142, 143, 139), "urban": (132, 136, 135),
            "school": (143, 145, 138), "shop": (139, 139, 132),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "western_stucco":
        return {
            "residential": (202, 187, 151), "townhouse": (194, 181, 151),
            "urban": (181, 174, 157), "agricultural": (190, 174, 139),
            "school": (204, 192, 163), "shop": (195, 181, 149),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "western_brick":
        return {
            "residential": (137, 78, 57), "townhouse": (130, 73, 55),
            "urban": (123, 76, 61), "agricultural": (126, 72, 50),
            "school": (142, 84, 62), "shop": (134, 77, 57),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "western_stone":
        return {
            "residential": (151, 145, 128), "townhouse": (145, 140, 126),
            "urban": (139, 138, 130), "agricultural": (137, 129, 111),
            "school": (157, 151, 135), "shop": (149, 143, 127),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "western_half_timber":
        return {
            "residential": (191, 181, 155), "townhouse": (184, 176, 152),
            "urban": (176, 170, 153), "agricultural": (177, 163, 134),
            "school": (191, 182, 157), "shop": (185, 175, 149),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "africa_earth":
        return {
            "residential": (166, 112, 68), "townhouse": (157, 105, 65),
            "urban": (151, 112, 82), "agricultural": (143, 91, 53),
            "school": (177, 131, 80), "shop": (166, 117, 73),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "africa_whitewash":
        return {
            "residential": (205, 194, 162), "townhouse": (199, 190, 164),
            "urban": (183, 181, 168), "agricultural": (190, 178, 145),
            "school": (207, 197, 169), "shop": (197, 187, 157),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "africa_block":
        return {
            "residential": (143, 137, 119), "townhouse": (136, 134, 122),
            "urban": (126, 130, 126), "agricultural": (130, 122, 104),
            "school": (145, 142, 128), "shop": (138, 133, 117),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "africa_colour":
        return {
            "residential": (91, 151, 144), "townhouse": (174, 117, 117),
            "urban": (118, 143, 164), "agricultural": (151, 126, 74),
            "school": (167, 148, 79), "shop": (93, 151, 137),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "middle_east_sandstone":
        return {
            "residential": (184, 157, 116), "townhouse": (177, 151, 113),
            "urban": (164, 149, 126), "agricultural": (164, 132, 91),
            "school": (190, 168, 131), "shop": (181, 157, 118),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "middle_east_whitewash":
        return {
            "residential": (215, 205, 177), "townhouse": (207, 199, 176),
            "urban": (192, 191, 180), "agricultural": (197, 185, 153),
            "school": (215, 207, 184), "shop": (206, 197, 170),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "middle_east_adobe":
        return {
            "residential": (160, 116, 77), "townhouse": (154, 111, 76),
            "urban": (151, 120, 91), "agricultural": (144, 98, 61),
            "school": (171, 131, 89), "shop": (161, 118, 77),
        }.get(family, _FAMILY_COLOURS[family])
    if regional_style == "middle_east_concrete":
        return {
            "residential": (160, 156, 143), "townhouse": (151, 151, 144),
            "urban": (139, 143, 142), "agricultural": (145, 137, 120),
            "school": (163, 161, 150), "shop": (154, 151, 139),
        }.get(family, _FAMILY_COLOURS[family])
    return _FAMILY_COLOURS[family]


def _wall_texture_image(
    family: str, size: int = 128, regional_style: str = "default",
    texture_variant: int = 0,
) -> Image.Image:
    texture_variant = _normalise_texture_variant(texture_variant)
    base = _variant_colour(_regional_wall_base(family, regional_style), texture_variant)
    family_salts = {
        "residential": 11, "townhouse": 17, "urban": 23, "industrial": 37,
        "agricultural": 39, "outbuilding": 40, "church": 41, "school": 43, "shop": 47,
    }
    style_salt = {
        "default": 0, "sweden_red": 101, "sweden_yellow": 149,
        "eastern_plaster": 211, "eastern_brick": 263,
        "eastern_whitewash": 307, "eastern_panel": 359,
        "africa_earth": 401, "africa_whitewash": 457,
        "africa_block": 503, "africa_colour": 557,
        "middle_east_sandstone": 601, "middle_east_whitewash": 653,
        "middle_east_adobe": 701, "middle_east_concrete": 751,
        "western_stucco": 809, "western_brick": 863,
        "western_stone": 911, "western_half_timber": 967,
    }.get(regional_style, 1013)
    salt = family_salts[family] + style_salt + texture_variant * 977
    image, draw = _pixel_canvas(
        base, size, salt, 5 if family in {"industrial", "agricultural", "outbuilding"} else 3
    )
    w = h = _BUILDING_TEXTURE_LOGICAL_SIZE

    if family == "outbuilding":
        # Small sheds and garages use plain windowless cladding. The large gate
        # is painted only on the front atlas so side/back walls stay believable.
        for x in range(0, w, 6):
            plank = _shade(base, 6 if (x // 6) % 3 == 0 else (-5 if (x // 6) % 3 == 1 else 1))
            draw.rectangle((x, 0, min(w, x + 5), h), fill=plank)
            draw.line((x, 0, x, h), fill=_shade(base, -22), width=1)
        draw.rectangle((0, 56, w, h), fill=(82, 78, 68))
        draw.line((0, 53, w, 53), fill=_shade(base, -15), width=2)
    elif regional_style == "sweden_red" and family == "agricultural":
        # Traditional red timber barn with pale trim and large braced doors.
        for x in range(0, w, 6):
            plank = _shade(base, 6 if (x // 6) % 3 == 0 else -6)
            draw.rectangle((x, 0, min(w, x + 5), h), fill=plank)
            draw.line((x, 0, x, h), fill=_shade(base, -25), width=1)
        trim = (205, 197, 169)
        draw.rectangle((9, 15, 55, 58), fill=_shade(base, -18), outline=trim, width=2)
        draw.line((11, 17, 53, 56), fill=trim, width=2)
        draw.line((53, 17, 11, 56), fill=trim, width=2)
        draw.line((32, 16, 32, 58), fill=trim, width=2)
        draw.rectangle((0, 57, w, h), fill=(82, 78, 69))
    elif regional_style in {"sweden_red", "sweden_yellow"} and family in {"residential", "townhouse"}:
        # Swedish timber cladding: vertical boards, pale corner boards and a
        # dark stone footing. Red is deliberately dominant, while yellow keeps
        # the regional palette from becoming one enormous paint bucket.
        for x in range(0, w, 5):
            plank = _shade(base, 5 if (x // 5) % 3 == 0 else -5)
            draw.rectangle((x, 0, min(w, x + 4), h), fill=plank)
            draw.line((x, 0, x, h), fill=_shade(base, -24), width=1)
        trim = (204, 196, 166)
        draw.rectangle((0, 0, 3, h), fill=trim)
        draw.rectangle((w - 4, 0, w - 1, h), fill=trim)
        window_count = 2 if family == "townhouse" else 1
        for index in range(window_count):
            centre = int((index + 1) * w / (window_count + 1))
            _draw_old_window(draw, (centre - 9, 13, centre + 9, 39), trim)
        draw.rectangle((0, 52, w, h), fill=(82, 79, 70))
        draw.line((0, 50, w, 50), fill=trim, width=2)
    elif regional_style == "eastern_brick" and family in {
        "residential", "townhouse", "urban", "agricultural", "outbuilding", "school", "shop"
    }:
        # Uneven dark brick with pale mortar, patched render, and a heavy base.
        draw.rectangle((0, 0, w, h), fill=base)
        brick_height = 6
        brick_width = 11
        for row, y in enumerate(range(0, h, brick_height)):
            draw.line((0, y, w, y), fill=(78, 64, 57), width=1)
            offset = 0 if row % 2 == 0 else brick_width // 2
            for x in range(offset, w, brick_width):
                draw.line((x, y, x, min(h, y + brick_height)), fill=(82, 65, 57), width=1)
        draw.rectangle((0, 54, w, h), fill=(85, 82, 74))
        draw.rectangle((2, 5, 17, 13), fill=(156, 145, 124))
        draw.rectangle((45, 35, 62, 48), fill=(148, 137, 119))
        if family == "agricultural":
            draw.rectangle((12, 18, 52, 57), fill=(70, 55, 43), outline=(47, 43, 38), width=2)
            draw.line((14, 20, 50, 55), fill=(104, 93, 77), width=2)
            draw.line((50, 20, 14, 55), fill=(104, 93, 77), width=2)
        else:
            columns = 3 if family in {"urban", "townhouse", "school"} else 2
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                _draw_old_window(draw, (centre - 7, 13, centre + 7, 37), (143, 136, 120))
    elif regional_style == "eastern_panel" and family in {"townhouse", "urban", "school", "shop"}:
        # Prefabricated concrete panels, visible seams, and a darker stair bay.
        draw.rectangle((0, 0, w, h), fill=base)
        for x in range(0, w, 16):
            draw.line((x, 0, x, h), fill=_shade(base, -19), width=1)
        for y in range(0, h, 18):
            draw.line((0, y, w, y), fill=_shade(base, -16), width=1)
        draw.rectangle((27, 0, 37, 54), fill=_shade(base, -13))
        for y in (7, 27):
            for x in (4, 18, 42, 56):
                _draw_old_window(draw, (x - 5, y, x + 5, y + 13), (121, 124, 120))
        draw.rectangle((0, 54, w, h), fill=(87, 88, 83))
        draw.line((2, 48, 62, 44), fill=(106, 72, 55), width=1)
    elif regional_style == "eastern_whitewash" and family in {
        "residential", "townhouse", "urban", "agricultural", "school", "shop"
    }:
        # Whitewashed masonry with cool painted trim and an irregular stone base.
        draw.rectangle((0, 0, w, h), fill=base)
        for y in (11, 29, 47):
            draw.line((0, y, w, y + 1), fill=_shade(base, -7), width=1)
        trim = (79, 105, 103)
        draw.rectangle((0, 52, w, h), fill=(91, 87, 76))
        if family == "agricultural":
            draw.rectangle((11, 18, 53, 58), fill=(86, 73, 55), outline=trim, width=2)
            draw.line((13, 20, 51, 56), fill=trim, width=2)
            draw.line((51, 20, 13, 56), fill=trim, width=2)
        else:
            columns = 3 if family in {"urban", "townhouse", "school"} else 2
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                _draw_old_window(draw, (centre - 7, 13, centre + 7, 38), trim)
                draw.rectangle((centre - 10, 12, centre - 8, 40), fill=trim)
                draw.rectangle((centre + 8, 12, centre + 10, 40), fill=trim)
    elif regional_style == "eastern_plaster" and family in {
        "residential", "townhouse", "urban", "agricultural", "school", "shop"
    }:
        # Faded rendered masonry with exposed brick scars and damp lower walls.
        draw.rectangle((0, 0, w, h), fill=base)
        for x, y, width, height in ((5, 7, 13, 6), (43, 27, 16, 8), (20, 47, 11, 5)):
            draw.rectangle((x, y, x + width, y + height), fill=_shade(base, -12))
            draw.line((x, y + height, x + width, y + height), fill=(107, 76, 60), width=1)
        draw.rectangle((0, 53, w, h), fill=(92, 89, 80))
        draw.line((0, 50, w, 50), fill=_shade(base, -18), width=2)
        if family == "agricultural":
            draw.rectangle((10, 17, 54, 58), fill=(76, 64, 49), outline=(112, 101, 82), width=2)
            draw.line((12, 19, 52, 56), fill=(111, 98, 78), width=2)
            draw.line((52, 19, 12, 56), fill=(111, 98, 78), width=2)
        else:
            columns = 3 if family in {"urban", "townhouse", "school"} else 2
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                _draw_old_window(draw, (centre - 7, 13, centre + 7, 38), (145, 138, 120))
    elif regional_style == "western_stucco" and family in {
        "residential", "townhouse", "urban", "agricultural", "school", "shop"
    }:
        # Warm lime render, pale stone surrounds and a dark rain-splashed plinth.
        draw.rectangle((0, 0, w, h), fill=base)
        for x, y, width in ((4, 8, 15), (39, 25, 18), (18, 46, 12)):
            draw.line((x, y, x + width, y + 1), fill=_shade(base, -8), width=1)
        trim = (218, 211, 190)
        draw.rectangle((0, 53, w, h), fill=(104, 99, 87))
        draw.rectangle((0, 0, 3, 53), fill=trim)
        draw.rectangle((w - 4, 0, w - 1, 53), fill=trim)
        if family == "agricultural":
            draw.rectangle((11, 18, 53, 58), fill=(78, 65, 49), outline=trim, width=2)
            draw.line((13, 20, 51, 56), fill=(118, 103, 81), width=2)
        else:
            columns = 3 if family in {"urban", "townhouse", "school", "shop"} else 2
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                _draw_old_window(draw, (centre - 7, 13, centre + 7, 38), trim)
                draw.rectangle((centre - 9, 11, centre + 9, 13), fill=trim)
                draw.rectangle((centre - 9, 38, centre + 9, 41), fill=trim)
    elif regional_style == "western_brick" and family in {
        "residential", "townhouse", "urban", "agricultural", "school", "shop"
    }:
        # Flemish/British-style red brick with light lintels and stone sills.
        draw.rectangle((0, 0, w, h), fill=base)
        for row, y in enumerate(range(0, h, 6)):
            draw.line((0, y, w, y), fill=(94, 75, 65), width=1)
            offset = 0 if row % 2 == 0 else 6
            for x in range(offset, w, 12):
                draw.line((x, y, x, min(h, y + 6)), fill=(91, 69, 60), width=1)
        trim = (178, 169, 145)
        draw.rectangle((0, 55, w, h), fill=(84, 81, 74))
        if family == "agricultural":
            draw.rectangle((10, 17, 54, 58), fill=(72, 55, 43), outline=trim, width=2)
            draw.line((12, 19, 52, 56), fill=(105, 91, 72), width=2)
        else:
            columns = 3 if family in {"urban", "townhouse", "school", "shop"} else 2
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                _draw_old_window(draw, (centre - 7, 13, centre + 7, 38), trim)
                draw.rectangle((centre - 9, 10, centre + 9, 13), fill=trim)
                draw.rectangle((centre - 9, 38, centre + 9, 41), fill=trim)
    elif regional_style == "western_stone" and family in {
        "residential", "townhouse", "urban", "agricultural", "school", "shop"
    }:
        # Irregular grey/tan rural stonework with restrained dressed surrounds.
        draw.rectangle((0, 0, w, h), fill=base)
        for row, y in enumerate(range(0, h, 9)):
            draw.line((0, y, w, y), fill=_shade(base, -18), width=1)
            offset = (row * 7) % 17
            for x in range(offset - 17, w, 17):
                draw.line((x, y, x + 2, min(h, y + 9)), fill=_shade(base, -14), width=1)
        trim = (184, 178, 157)
        draw.rectangle((0, 54, w, h), fill=(91, 88, 79))
        if family == "agricultural":
            draw.rectangle((11, 18, 53, 58), fill=(72, 61, 49), outline=trim, width=2)
        else:
            columns = 3 if family in {"urban", "townhouse", "school", "shop"} else 2
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                _draw_old_window(draw, (centre - 7, 13, centre + 7, 38), trim)
    elif regional_style == "western_half_timber" and family in {
        "residential", "townhouse", "urban", "agricultural", "school", "shop"
    }:
        # Pale plaster panels carried by a dark exposed timber frame.
        draw.rectangle((0, 0, w, h), fill=base)
        timber = (75, 59, 45)
        for y in (5, 29, 51):
            draw.rectangle((0, y, w, y + 3), fill=timber)
        for x in (3, 21, 42, 59):
            draw.rectangle((x, 0, x + 3, 54), fill=timber)
        draw.line((4, 5, 21, 29), fill=timber, width=3)
        draw.line((24, 29, 42, 8), fill=timber, width=3)
        draw.line((45, 29, 59, 51), fill=timber, width=3)
        draw.rectangle((0, 54, w, h), fill=(91, 86, 75))
        if family == "agricultural":
            draw.rectangle((12, 18, 52, 58), fill=(70, 54, 42), outline=timber, width=2)
        else:
            columns = 3 if family in {"urban", "townhouse", "school", "shop"} else 2
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                _draw_old_window(draw, (centre - 6, 14, centre + 6, 38), (144, 133, 111))
    elif regional_style == "africa_earth" and family in {
        "residential", "townhouse", "urban", "agricultural", "school", "shop"
    }:
        # Sun-baked earth render with hand-smoothed patches, a darker splash
        # zone, and small deeply shaded openings. The texture remains chunky
        # enough for the original OFP renderer rather than becoming a photo.
        draw.rectangle((0, 0, w, h), fill=base)
        for x, y, rx, ry in ((7, 9, 15, 7), (43, 18, 17, 9), (22, 42, 13, 6)):
            draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=_shade(base, -7))
        draw.rectangle((0, 52, w, h), fill=_shade(base, -24))
        draw.line((0, 49, w, 51), fill=_shade(base, -13), width=2)
        if family == "agricultural":
            draw.rectangle((13, 18, 51, 58), fill=(77, 57, 40), outline=(116, 88, 59), width=2)
            draw.line((15, 20, 49, 56), fill=(120, 91, 62), width=2)
            draw.line((49, 20, 15, 56), fill=(120, 91, 62), width=2)
        else:
            columns = 3 if family in {"urban", "townhouse", "school"} else 2
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                draw.rectangle((centre - 6, 16, centre + 6, 36), fill=(42, 47, 45), outline=(106, 88, 65), width=2)
                draw.line((centre - 5, 18, centre + 5, 18), fill=(70, 80, 75), width=1)
    elif regional_style == "africa_whitewash" and family in {
        "residential", "townhouse", "urban", "agricultural", "school", "shop"
    }:
        # Bright limewash, coloured trim, and dusty lower walls.
        draw.rectangle((0, 0, w, h), fill=base)
        trim_options = ((55, 110, 125), (72, 119, 91), (150, 91, 68), (79, 94, 145))
        trim = trim_options[texture_variant % len(trim_options)]
        for y in (15, 34):
            draw.line((0, y, w, y), fill=_shade(base, -8), width=1)
        draw.rectangle((0, 52, w, h), fill=(128, 109, 82))
        columns = 3 if family in {"urban", "townhouse", "school"} else 2
        if family == "agricultural":
            draw.rectangle((11, 18, 53, 58), fill=(88, 71, 51), outline=trim, width=2)
            draw.line((13, 20, 51, 56), fill=trim, width=2)
            draw.line((51, 20, 13, 56), fill=trim, width=2)
        else:
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                _draw_old_window(draw, (centre - 7, 14, centre + 7, 38), trim)
                draw.rectangle((centre - 9, 12, centre + 9, 14), fill=trim)
    elif regional_style == "africa_block" and family in {
        "residential", "townhouse", "urban", "agricultural", "school", "shop"
    }:
        # Unfinished cement block, visible mortar joints, and patched openings.
        draw.rectangle((0, 0, w, h), fill=base)
        block_width, block_height = 14, 8
        for row, y in enumerate(range(0, h, block_height)):
            draw.line((0, y, w, y), fill=_shade(base, -18), width=1)
            offset = 0 if row % 2 == 0 else block_width // 2
            for x in range(offset, w, block_width):
                draw.line((x, y, x, min(h, y + block_height)), fill=_shade(base, -16), width=1)
        draw.rectangle((0, 55, w, h), fill=(105, 99, 84))
        columns = 3 if family in {"urban", "townhouse", "school"} else 2
        if family == "agricultural":
            draw.rectangle((10, 18, 54, 58), fill=(81, 69, 54), outline=(68, 65, 58), width=2)
        else:
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                draw.rectangle((centre - 7, 14, centre + 7, 38), fill=(49, 57, 56), outline=(111, 108, 95), width=2)
    elif regional_style == "africa_colour" and family in {
        "residential", "townhouse", "urban", "agricultural", "school", "shop"
    }:
        # Painted render with a contrasting lower band and sun-faded shop or
        # school colours. Variant selection changes the base and trim together.
        draw.rectangle((0, 0, w, h), fill=base)
        accents = ((210, 168, 65), (72, 111, 148), (185, 93, 83), (64, 133, 111))
        accent = accents[texture_variant % len(accents)]
        draw.rectangle((0, 48, w, h), fill=_shade(base, -25))
        draw.rectangle((0, 44, w, 48), fill=accent)
        columns = 3 if family in {"urban", "townhouse", "school", "shop"} else 2
        if family == "agricultural":
            draw.rectangle((12, 18, 52, 58), fill=(82, 65, 48), outline=accent, width=2)
            draw.line((14, 20, 50, 56), fill=accent, width=2)
        else:
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                _draw_old_window(draw, (centre - 7, 13, centre + 7, 37), (199, 188, 158))
                draw.rectangle((centre - 9, 10, centre + 9, 13), fill=accent)
    elif regional_style == "middle_east_sandstone" and family in {
        "residential", "townhouse", "urban", "agricultural", "school", "shop"
    }:
        # Warm cut-stone courses with recessed arched openings and a heavy base.
        draw.rectangle((0, 0, w, h), fill=base)
        for y in range(0, h, 8):
            draw.line((0, y, w, y), fill=_shade(base, -15), width=1)
        for row, y in enumerate(range(0, h, 8)):
            offset = 0 if row % 2 == 0 else 9
            for x in range(offset, w, 18):
                draw.line((x, y, x, min(h, y + 8)), fill=_shade(base, -11), width=1)
        draw.rectangle((0, 54, w, h), fill=(118, 100, 78))
        columns = 3 if family in {"urban", "townhouse", "school", "shop"} else 2
        if family == "agricultural":
            draw.rectangle((11, 18, 53, 58), fill=(80, 63, 46), outline=(136, 116, 84), width=2)
        else:
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                draw.ellipse((centre - 7, 10, centre + 7, 28), fill=(45, 52, 52), outline=(137, 121, 91), width=2)
                draw.rectangle((centre - 7, 19, centre + 7, 39), fill=(45, 52, 52), outline=(137, 121, 91), width=2)
    elif regional_style == "middle_east_whitewash" and family in {
        "residential", "townhouse", "urban", "agricultural", "school", "shop"
    }:
        # Pale plaster, blue/green shutters, and a dusty stone plinth.
        draw.rectangle((0, 0, w, h), fill=base)
        trim = (58, 105, 112) if texture_variant % 2 == 0 else (71, 111, 83)
        for y in (18, 39):
            draw.line((0, y, w, y), fill=_shade(base, -8), width=1)
        draw.rectangle((0, 53, w, h), fill=(126, 110, 86))
        columns = 3 if family in {"urban", "townhouse", "school"} else 2
        if family == "agricultural":
            draw.rectangle((12, 19, 52, 58), fill=(91, 72, 52), outline=trim, width=2)
        else:
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                _draw_old_window(draw, (centre - 7, 14, centre + 7, 38), trim)
                draw.rectangle((centre - 10, 12, centre - 8, 40), fill=trim)
                draw.rectangle((centre + 8, 12, centre + 10, 40), fill=trim)
    elif regional_style == "middle_east_adobe" and family in {
        "residential", "townhouse", "urban", "agricultural", "school", "shop"
    }:
        # Earthen plaster with timber lintels and small openings for hot climates.
        draw.rectangle((0, 0, w, h), fill=base)
        for x, y, width, height in ((3, 5, 19, 7), (39, 24, 20, 8), (18, 43, 13, 6)):
            draw.rectangle((x, y, x + width, y + height), fill=_shade(base, -7))
        draw.rectangle((0, 53, w, h), fill=_shade(base, -20))
        columns = 3 if family in {"urban", "townhouse", "school"} else 2
        if family == "agricultural":
            draw.rectangle((13, 19, 51, 58), fill=(73, 55, 40), outline=(116, 87, 58), width=2)
        else:
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                draw.rectangle((centre - 6, 18, centre + 6, 36), fill=(42, 47, 45), outline=(106, 82, 55), width=2)
                draw.rectangle((centre - 8, 15, centre + 8, 18), fill=(91, 62, 41))
    elif regional_style == "middle_east_concrete" and family in {
        "residential", "townhouse", "urban", "agricultural", "school", "shop"
    }:
        # Pale reinforced-concrete frame with block infill, balcony bands, and
        # darker service cores for larger urban buildings.
        draw.rectangle((0, 0, w, h), fill=base)
        for x in range(0, w, 16):
            draw.line((x, 0, x, h), fill=_shade(base, -14), width=2)
        for y in (20, 42):
            draw.line((0, y, w, y), fill=_shade(base, -15), width=2)
        draw.rectangle((0, 53, w, h), fill=(103, 101, 91))
        columns = 3 if family in {"urban", "townhouse", "school", "shop"} else 2
        if family == "agricultural":
            draw.rectangle((11, 19, 53, 58), fill=(82, 72, 58), outline=(116, 112, 101), width=2)
        else:
            for index in range(columns):
                centre = int((index + 1) * w / (columns + 1))
                draw.rectangle((centre - 7, 10, centre + 7, 34), fill=(48, 55, 55), outline=(117, 117, 108), width=2)
                draw.rectangle((centre - 9, 35, centre + 9, 39), fill=_shade(base, -22))
    elif family == "residential":
        for y in (13, 31, 47):
            draw.line((0, y, w, y), fill=_shade(base, -5), width=1)
        draw.rectangle((0, 52, w, h), fill=(94, 89, 77))
        for x in range(-4, w, 9):
            draw.line((x, 53, x + 7, h), fill=(73, 70, 62), width=1)
        _draw_old_window(draw, (18, 13, 46, 43), (157, 149, 129))
        draw.rectangle((15, 43, 49, 47), fill=(105, 97, 81))
        draw.line((0, 50, w, 50), fill=(120, 112, 95), width=2)
    elif family == "townhouse":
        # Denser two-storey town façade. Repeated windows and a rendered plinth
        # distinguish it from the single rural farmhouse atlas.
        draw.rectangle((0, 52, w, h), fill=(94, 90, 82))
        for y in (8, 31):
            for x in (5, 25, 45):
                _draw_old_window(draw, (x, y, x + 14, y + 18), (156, 149, 132))
        draw.line((0, 28, w, 28), fill=_shade(base, -12), width=2)
        draw.line((0, 50, w, 50), fill=_shade(base, -18), width=2)
    elif family == "urban":
        for x in range(0, w, 16):
            draw.line((x, 0, x, h), fill=_shade(base, -9), width=1)
        for y in range(0, h, 16):
            draw.line((0, y, w, y), fill=_shade(base, -8), width=1)
        _draw_old_window(draw, (12, 11, 52, 42), (137, 136, 126))
        draw.rectangle((9, 42, 55, 46), fill=(104, 104, 97))
        draw.rectangle((0, 54, w, h), fill=(95, 94, 85))
    elif family == "agricultural":
        for x in range(0, w, 6):
            plank = _shade(base, 7 if (x // 6) % 3 == 0 else (-7 if (x // 6) % 3 == 1 else 0))
            draw.rectangle((x, 0, min(w, x + 5), h), fill=plank)
            draw.line((x, 0, x, h), fill=(54, 49, 40), width=1)
        draw.rectangle((10, 17, 54, 57), fill=(71, 61, 47), outline=(43, 40, 34), width=2)
        draw.line((12, 19, 52, 55), fill=(45, 42, 36), width=2)
        draw.line((52, 19, 12, 55), fill=(45, 42, 36), width=2)
        draw.line((32, 18, 32, 57), fill=(43, 40, 34), width=2)
        draw.rectangle((0, 57, w, h), fill=(83, 78, 67))
    elif family == "church":
        for y in range(0, h, 9):
            draw.line((0, y, w, y), fill=_shade(base, -7), width=1)
        draw.rectangle((0, 53, w, h), fill=(91, 88, 81))
        draw.ellipse((20, 8, 44, 32), fill=(45, 53, 57), outline=(115, 108, 94), width=2)
        draw.rectangle((20, 20, 44, 47), fill=(45, 53, 57), outline=(115, 108, 94), width=2)
        draw.line((32, 11, 32, 45), fill=(131, 121, 101), width=1)
        draw.line((22, 30, 42, 30), fill=(131, 121, 101), width=1)
    elif family == "school":
        draw.rectangle((0, 50, w, h), fill=(100, 88, 70))
        for x in (4, 24, 44):
            _draw_old_window(draw, (x, 14, x + 16, 40), (156, 148, 128))
        draw.line((0, 46, w, 46), fill=_shade(base, -18), width=2)
    elif family == "shop":
        draw.rectangle((3, 10, 61, 45), fill=(51, 55, 55), outline=(141, 132, 113), width=2)
        draw.rectangle((6, 13, 58, 42), fill=(57, 69, 69))
        draw.rectangle((25, 17, 40, 48), fill=(50, 45, 39), outline=(137, 128, 110), width=2)
        draw.rectangle((2, 3, 62, 11), fill=(99, 61, 44))
        draw.rectangle((0, 50, w, h), fill=(89, 84, 73))
    else:
        for x in range(0, w, 4):
            colour = _shade(base, -12 if (x // 4) % 2 else 7)
            draw.rectangle((x, 0, min(w, x + 3), h), fill=colour)
            draw.line((x, 0, x, h), fill=(75, 76, 70), width=1)
        draw.rectangle((6, 12, 58, 31), fill=(46, 51, 51), outline=(103, 106, 99), width=2)
        for x in range(17, 58, 12):
            draw.line((x, 14, x, 29), fill=(111, 116, 108), width=1)
        draw.rectangle((0, 53, w, h), fill=(83, 78, 67))
        for x in (7, 22, 43, 57):
            draw.line((x, 7, x + 3, 24), fill=(102, 58, 39), width=2)
    if family in _PAINTED_WINDOW_FAMILIES:
        image = _raise_painted_windows_above_ground(image)
    return _finish_pixel_texture(image, size)


def _open_wall_texture_image(
    family: str, size: int = 128, regional_style: str = "default",
    texture_variant: int = 0,
) -> Image.Image:
    """Return matching wall material without any painted door or window art."""

    texture_variant = _normalise_texture_variant(texture_variant)
    base = _variant_colour(
        _regional_wall_base(family, regional_style), texture_variant
    )
    salt = 1201 + sum(ord(character) for character in family + regional_style)
    salt += texture_variant * 977
    image, draw = _pixel_canvas(base, size, salt, 3)
    width = height = _BUILDING_TEXTURE_LOGICAL_SIZE

    if regional_style in {"sweden_red", "sweden_yellow"}:
        for x in range(0, width, 5):
            draw.line((x, 0, x, height), fill=_shade(base, -22), width=1)
    elif regional_style in {"eastern_brick", "africa_block", "western_brick"}:
        course_height = 8 if regional_style == "africa_block" else 6
        block_width = 14 if regional_style == "africa_block" else (12 if regional_style == "western_brick" else 11)
        for row, y in enumerate(range(0, height, course_height)):
            draw.line((0, y, width, y), fill=_shade(base, -18), width=1)
            offset = 0 if row % 2 == 0 else block_width // 2
            for x in range(offset, width, block_width):
                draw.line(
                    (x, y, x, min(height, y + course_height)),
                    fill=_shade(base, -16), width=1,
                )
    elif regional_style in {"eastern_panel", "middle_east_concrete"}:
        for x in range(0, width, 16):
            draw.line((x, 0, x, height), fill=_shade(base, -17), width=1)
        for y in range(0, height, 18):
            draw.line((0, y, width, y), fill=_shade(base, -15), width=1)
    elif regional_style == "western_stone":
        for row, y in enumerate(range(0, height, 9)):
            draw.line((0, y, width, y), fill=_shade(base, -18), width=1)
            offset = (row * 7) % 17
            for x in range(offset - 17, width, 17):
                draw.line((x, y, x + 2, min(height, y + 9)), fill=_shade(base, -14), width=1)
    elif regional_style == "western_half_timber":
        timber = (75, 59, 45)
        for y in (5, 29, 51):
            draw.rectangle((0, y, width, y + 3), fill=timber)
        for x in (3, 21, 42, 59):
            draw.rectangle((x, 0, x + 3, 54), fill=timber)
        draw.line((4, 5, 21, 29), fill=timber, width=3)
        draw.line((24, 29, 42, 8), fill=timber, width=3)
        draw.line((45, 29, 59, 51), fill=timber, width=3)
    elif regional_style == "western_stucco":
        for y in (15, 33, 49):
            draw.line((0, y, width, y), fill=_shade(base, -6), width=1)
        draw.rectangle((0, 0, 3, 52), fill=_shade(base, 13))
        draw.rectangle((width - 4, 0, width - 1, 52), fill=_shade(base, 13))
    else:
        for y in (15, 33, 49):
            draw.line((0, y, width, y), fill=_shade(base, -7), width=1)

    if family != "church":
        draw.rectangle((0, 54, width, height), fill=_shade(base, -27))
        draw.line((0, 52, width, 52), fill=_shade(base, -15), width=2)
    return _finish_pixel_texture(image, size)


def _interior_wall_texture_image(
    family: str, size: int = 128, regional_style: str = "default",
    texture_variant: int = 0,
) -> Image.Image:
    """Return a subdued inner-wall material that makes open bays readable."""

    exterior = _variant_colour(
        _regional_wall_base(family, regional_style),
        _normalise_texture_variant(texture_variant),
    )
    base = _shade(exterior, -48)
    salt = 1601 + sum(ord(character) for character in family + regional_style)
    salt += texture_variant * 991
    image, draw = _pixel_canvas(base, size, salt, 4)
    width = height = _BUILDING_TEXTURE_LOGICAL_SIZE
    for y in (18, 37, 54):
        draw.line((0, y, width, y), fill=_shade(base, -7), width=1)
    draw.rectangle((0, 57, width, height), fill=_shade(base, -14))
    return _finish_pixel_texture(image, size)


def _white_trim_texture_image(size: int = 128) -> Image.Image:
    """Return slightly weathered white paint for exterior opening surrounds."""

    base = (218, 215, 198)
    image, draw = _pixel_canvas(base, size, 1901, 2)
    width = height = _BUILDING_TEXTURE_LOGICAL_SIZE
    for x in range(0, width, 16):
        draw.line((x, 0, x, height), fill=_shade(base, -5), width=1)
    draw.line((0, height - 4, width, height - 4), fill=_shade(base, -9), width=1)
    return _finish_pixel_texture(image, size)


def _front_texture_image(
    family: str, size: int = 128, regional_style: str = "default",
    texture_variant: int = 0,
) -> Image.Image:
    texture_variant = _normalise_texture_variant(texture_variant)
    # Compose the entrance on a 64px logical atlas. HQ exports keep a 128px
    # working facade so door frames and boards receive the same supersampling as
    # the wall instead of being painted after the quality pass.
    if size <= 128:
        image = _wall_texture_image(
            family, 128, regional_style, texture_variant
        ).resize(
            (_BUILDING_TEXTURE_LOGICAL_SIZE, _BUILDING_TEXTURE_LOGICAL_SIZE),
            Image.Resampling.NEAREST,
        )
        draw: ImageDraw.ImageDraw | _ScaledImageDraw = ImageDraw.Draw(image)
    else:
        work = max(_BUILDING_TEXTURE_LOGICAL_SIZE, size // 2)
        image = _wall_texture_image(
            family, size, regional_style, texture_variant
        ).resize((work, work), Image.Resampling.LANCZOS)
        draw = _ScaledImageDraw(image, work / _BUILDING_TEXTURE_LOGICAL_SIZE)
    if family == "outbuilding":
        # A small OSM footprint is much more likely to be a shed/garage than a
        # miniature house. Give it one broad vehicle gate and no fake windows.
        width = 40
        x0 = (64 - width) // 2
        y0 = 18
        door_bottom = 63
        if regional_style in {"sweden_red", "sweden_yellow"}:
            door = (79, 70, 54)
            trim = (203, 196, 169)
        else:
            door = _shade(_regional_wall_base(family, regional_style), -22)
            trim = (151, 143, 123)
        door = _variant_colour(door, (texture_variant * 3) % DEFAULT_BUILDING_TEXTURE_VARIANTS)
        draw.rectangle((x0, y0, x0 + width, door_bottom), fill=door, outline=trim, width=2)
        for x in range(x0 + 5, x0 + width, 6):
            draw.line((x, y0 + 2, x, door_bottom - 2), fill=_shade(door, -10), width=1)
        draw.line((x0 + width // 2, y0 + 1, x0 + width // 2, door_bottom - 1), fill=_shade(door, -17), width=2)
        draw.rectangle((x0 - 2, door_bottom - 2, x0 + width + 2, door_bottom), fill=(86, 82, 72))
    elif family not in {"shop", "agricultural"}:
        width = 15 if family in {"urban", "townhouse", "school", "church"} else 13
        x0 = (64 - width) // 2
        # Closed facades use painted doors, but their visual threshold follows
        # the same local Y=0 contract as real enterable openings.  Draw a full
        # approximately 2.2 m door down to the bottom of the three-metre tile
        # after raising the surrounding window artwork.
        y0 = round(64 * (1.0 - INTERIOR_DOOR_HEIGHT_M / FACADE_TILE_HEIGHT_M))
        door_bottom = 63
        if regional_style in {"sweden_red", "sweden_yellow"}:
            door = (73, 66, 51)
        elif regional_style.startswith("eastern_"):
            door = (62, 73, 65) if regional_style == "eastern_whitewash" else (68, 55, 43)
        elif regional_style.startswith("africa_"):
            door = (55, 91, 88) if regional_style in {"africa_whitewash", "africa_colour"} else (73, 54, 38)
        elif regional_style.startswith("middle_east_"):
            door = (54, 88, 91) if regional_style == "middle_east_whitewash" else (72, 54, 39)
        elif regional_style.startswith("western_"):
            western_doors = ((61, 76, 61), (71, 57, 44), (57, 68, 82))
            door = western_doors[texture_variant % len(western_doors)]
        else:
            door = (62, 50, 38) if family == "residential" else (66, 58, 47)
        door = _variant_colour(door, (texture_variant * 3) % DEFAULT_BUILDING_TEXTURE_VARIANTS)
        draw.rectangle((x0, y0, x0 + width, door_bottom), fill=door, outline=(180, 170, 145), width=2)
        for x in range(x0 + 3, x0 + width, 4):
            draw.line((x, y0 + 2, x, door_bottom - 2), fill=_shade(door, -9), width=1)
        handle_y = y0 + round((door_bottom - y0) * 0.60)
        draw.rectangle((x0 + width - 4, handle_y, x0 + width - 2, handle_y + 2), fill=(155, 126, 65))
        draw.rectangle((x0 - 2, door_bottom - 2, x0 + width + 2, door_bottom), fill=(90, 84, 72))
    return _finish_pixel_texture(image, size)


def _roof_texture_image(
    roof_style: str, size: int = 128, texture_variant: int = 0
) -> Image.Image:
    texture_variant = _normalise_texture_variant(texture_variant)
    base = _variant_colour(_ROOF_COLOURS[roof_style], texture_variant)
    image, draw = _pixel_canvas(
        base, size, (51 if roof_style == "flat" else 67) + texture_variant * 991, 4
    )
    w = h = _BUILDING_TEXTURE_LOGICAL_SIZE
    if roof_style == "flat":
        # Weathered sheet roofing, useful for schools, shops and industrial halls.
        for x in range(0, w, 5):
            draw.line((x, 0, x, h), fill=_shade(base, -13 if (x // 5) % 2 else 8), width=1)
        rust_shift = texture_variant % 5
        for x in (8 + rust_shift, 27 - rust_shift, 49 + rust_shift):
            draw.line((x, 3, x + 2, 47), fill=(104, 57, 40), width=2)
        draw.rectangle((0, 56, w, h), fill=_shade(base, -12))
    else:
        # Chunky offset clay tiles, deliberately irregular and darkened by age.
        row_height = 8
        tile_width = 12
        for row, y in enumerate(range(-2, h, row_height)):
            offset = (texture_variant * 2) % tile_width
            if row % 2:
                offset = (offset + tile_width // 2) % tile_width
            draw.line((0, y, w, y), fill=(63, 42, 35), width=1)
            for x in range(offset - tile_width, w, tile_width):
                tile = _shade(base, ((x + row * 7) % 11) - 6)
                draw.rectangle((x + 1, y + 1, x + tile_width - 1, y + row_height - 1), fill=tile)
                draw.line((x + 1, y + 2, x + tile_width - 2, y + 2), fill=_shade(tile, 11), width=1)
                draw.line((x + tile_width - 1, y + 1, x + tile_width - 1, y + row_height), fill=(67, 43, 34), width=1)
        for x in (9, 38, 55):
            draw.line((x, 0, x + 5, h), fill=(72, 50, 41), width=1)
    return _finish_pixel_texture(image, size)


def _foundation_texture_image(size: int = 128) -> Image.Image:
    """Return a coarse Everon-era stone foundation texture."""
    base = (96, 91, 80)
    image, draw = _pixel_canvas(base, size, 83, 5)
    width = height = _BUILDING_TEXTURE_LOGICAL_SIZE
    row_height = 10
    for row, y in enumerate(range(-2, height, row_height)):
        offset = 0 if row % 2 == 0 else 8
        draw.line((0, y, width, y), fill=(62, 61, 56), width=1)
        for x in range(offset - 16, width, 16):
            shade = _shade(base, ((x + row * 5) % 13) - 6)
            draw.rectangle((x + 1, y + 1, x + 15, y + row_height - 1), fill=shade)
            draw.line((x + 15, y + 1, x + 15, y + row_height), fill=(66, 64, 58), width=1)
    return _finish_pixel_texture(image, size)


def _parse_number(value: str | None) -> float | None:
    if not value:
        return None
    match = _DISTANCE_VALUE.search(value.replace(",", "."))
    if not match:
        return None
    try:
        parsed = float(match.group(0))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _family(
    tags: Mapping[str, str],
    width_m: float | None = None,
    length_m: float | None = None,
    *,
    settlement_context: str = "rural",
) -> str:
    """Classify a building from explicit semantics, then settlement context.

    Large generic footprints are only treated as town buildings within one
    source-ground kilometre of an explicit ``place=town`` or ``place=city``
    point. Residential land-use polygons alone do not establish that context.
    """

    building = tags.get("building", "").casefold()
    amenity = tags.get("amenity", "").casefold()
    man_made = tags.get("man_made", "").casefold()
    if is_actual_church(tags):
        return "church"
    if amenity == "place_of_worship":
        return "urban"
    if amenity == "school" or building in {"school", "kindergarten", "college", "university"}:
        return "school"
    if amenity == "social_facility" or bool(tags.get("social_facility")):
        # Amenity semantics are authoritative even when the physical footprint
        # is tagged warehouse/barn or is large enough to trigger rural fallback.
        return "urban"
    if tags.get("shop") or building in {"retail", "supermarket", "kiosk"}:
        return "shop"
    if building in {"industrial", "warehouse", "hangar", "factory", "manufacture"} or man_made in {
        "works", "storage_tank", "silo"
    }:
        return "industrial"
    # Exact isolated-dwelling membership outranks the tiny-building shed
    # heuristic. A lone mapped shed-shaped footprint can be the dwelling itself;
    # explicit garage/carport/outbuilding tags remain non-residential.
    if settlement_context == "isolated_dwelling_single" and building in {
        "", "yes", "house", "residential", "detached", "cabin", "shed"
    }:
        return "residential"
    if building in {"garage", "garages", "shed", "carport", "outbuilding"}:
        return "outbuilding"
    if building in {
        "barn", "farm_auxiliary", "agricultural", "cowshed", "stable",
        "sty", "greenhouse", "storage"
    }:
        return "agricultural"
    if building in {
        "apartments", "commercial", "office", "hotel", "hospital",
        "civic", "public", "government"
    }:
        return "urban"

    minor = major = area = 0.0
    if width_m is not None and length_m is not None:
        minor, major = sorted((max(0.1, float(width_m)), max(0.1, float(length_m))))
        area = minor * major

    # Tiny generic footprints are overwhelmingly sheds, garages, pump houses,
    # or similar outbuildings. Do this before settlement promotion so a 5x7 m
    # polygon in a town does not become a tiny window-covered townhouse.
    if (
        building in {"", "yes"}
        and area > 0.0
        and area <= 72.0
        and minor <= 8.5
        and major <= 12.0
    ):
        return "outbuilding"

    levels = _parse_number(tags.get("building:levels")) or 0.0
    if settlement_context in {"urban", "town", "city"}:
        # City centres become dense at a smaller footprint than towns; both
        # still obey the hard one-kilometre settlement gate above.  Explicit
        # apartment/commercial semantics were already handled before this
        # generic-footprint branch.
        if settlement_context == "city":
            apartment_scale = levels >= 3.0 or major >= 24.0 or minor >= 16.0 or area >= 320.0
        else:
            apartment_scale = levels >= 3.0 or major >= 28.0 or minor >= 18.0 or area >= 420.0
        return "urban" if apartment_scale else "townhouse"

    if settlement_context == "village":
        # Villages are deliberately not towns.  Even a large generic footprint
        # stays residential unless OSM explicitly describes agricultural use.
        return "residential"

    if width_m is not None and length_m is not None:
        aspect = major / minor
        oversized_rural_footprint = (
            (major >= 32.0 and minor >= 8.0)
            or minor >= 20.0
            or area >= 600.0
            or (aspect >= 3.0 and major >= 24.0)
        )
        if oversized_rural_footprint:
            return "agricultural"
    return "residential"


def _regional_style(
    region_identifier: str | None,
    family: str,
    tags: Mapping[str, str],
    width_m: float,
    length_m: float,
) -> str:
    material = str(tags.get("building:material", "")).casefold()
    colour = str(tags.get("building:colour", "")).casefold()
    supported = {
        "residential", "townhouse", "urban", "agricultural", "outbuilding", "school", "shop"
    }

    identity = json.dumps(
        {
            "building": tags.get("building", ""),
            "name": tags.get("name", ""),
            "levels": tags.get("building:levels", ""),
            "width": round(width_m, 2),
            "length": round(length_m, 2),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    value = int.from_bytes(sha256(identity.encode("utf-8")).digest()[:2], "little") % 100

    if region_identifier == "sweden" and family in {"residential", "townhouse", "urban", "agricultural", "outbuilding"}:
        if colour in {"red", "dark_red", "brown-red", "brown_red"}:
            return "sweden_red"
        if colour in {"yellow", "ochre", "cream"}:
            return "sweden_yellow"
        if material in {"brick", "stone", "concrete"}:
            return "default"
        if family == "urban":
            return "sweden_yellow" if value < 18 else "default"
        if family == "townhouse" and value < 14:
            return "sweden_yellow"
        if family in {"agricultural", "outbuilding"}:
            return "sweden_red"
        # 0.9.80 reduces the automatic red-house share by one third:
        # 72% -> 48%. The yellow share remains 16%; the released share
        # becomes neutral/default regional façades.
        if value < 48:
            return "sweden_red"
        if value < 64:
            return "sweden_yellow"
        return "default"

    if region_identifier == "eastern_europe" and family in supported:
        if material in {"prefabricated", "precast_concrete", "concrete", "cement_block"}:
            return "eastern_panel" if family in {"urban", "townhouse", "school", "shop"} else "default"
        if material in {"brick", "bricks"} or colour in {
            "red", "dark_red", "brown", "brown-red", "brown_red", "terracotta"
        }:
            return "eastern_brick"
        if material in {"plaster", "stucco", "render", "masonry"} or colour in {
            "cream", "beige", "yellow", "ochre"
        }:
            return "eastern_plaster"
        if colour in {"white", "off_white", "light_grey", "light_gray"}:
            return "eastern_whitewash"
        if material in {"stone", "wood", "timber", "metal", "glass"}:
            return "default"
        if family == "urban":
            if value < 58:
                return "eastern_panel"
            if value < 86:
                return "eastern_plaster"
            if value < 95:
                return "eastern_brick"
            return "default"
        if family in {"school", "shop"}:
            if value < 52:
                return "eastern_plaster"
            if value < 82:
                return "eastern_panel"
            if value < 94:
                return "eastern_brick"
            return "default"
        if family == "agricultural":
            if value < 46:
                return "eastern_brick"
            if value < 77:
                return "eastern_whitewash"
            if value < 93:
                return "eastern_plaster"
            return "default"
        if value < 45:
            return "eastern_plaster"
        if value < 68:
            return "eastern_brick"
        if value < 90:
            return "eastern_whitewash"
        return "default"

    if region_identifier == "western_europe" and family in supported:
        if material in {
            "timber_framing", "timber_frame", "half_timbered", "half-timbered",
            "fachwerk"
        }:
            return "western_half_timber"
        if material in {"brick", "bricks"} or colour in {
            "red", "dark_red", "brown", "brown-red", "brown_red", "terracotta"
        }:
            return "western_brick"
        if material in {"stone", "limestone", "sandstone", "granite", "slate"}:
            return "western_stone"
        if material in {"plaster", "stucco", "render", "masonry"} or colour in {
            "white", "off_white", "cream", "ivory", "beige", "yellow",
            "ochre", "light_grey", "light_gray"
        }:
            return "western_stucco"
        if material in {"wood", "timber", "metal", "glass", "concrete", "precast_concrete"}:
            return "default"
        if family == "urban":
            if value < 45:
                return "western_stucco"
            if value < 76:
                return "western_brick"
            if value < 90:
                return "western_stone"
            return "default"
        if family in {"townhouse", "shop", "school"}:
            if value < 42:
                return "western_stucco"
            if value < 73:
                return "western_brick"
            if value < 87:
                return "western_stone"
            if value < 95:
                return "western_half_timber"
            return "default"
        if family == "agricultural":
            if value < 38:
                return "western_stone"
            if value < 68:
                return "western_brick"
            if value < 84:
                return "western_half_timber"
            if value < 95:
                return "western_stucco"
            return "default"
        if value < 38:
            return "western_stucco"
        if value < 65:
            return "western_brick"
        if value < 84:
            return "western_stone"
        if value < 95:
            return "western_half_timber"
        return "default"

    if region_identifier == "africa" and family in supported:
        if material in {"mud", "adobe", "earth", "clay", "rammed_earth"}:
            return "africa_earth"
        if material in {"concrete", "cement_block", "concrete_blocks", "cinder_block"}:
            return "africa_block"
        if colour in {"white", "off_white", "cream", "ivory", "light_grey", "light_gray"}:
            return "africa_whitewash"
        if colour in {
            "blue", "green", "turquoise", "cyan", "pink", "orange", "yellow",
            "purple", "teal"
        }:
            return "africa_colour"
        if material in {"plaster", "stucco", "render", "masonry", "limestone"}:
            return "africa_whitewash"
        if material in {"brick", "stone", "sandstone"} or colour in {
            "brown", "red", "ochre", "beige", "sand"
        }:
            return "africa_earth"
        if material in {"wood", "timber", "metal", "glass"}:
            return "default"
        if family == "urban":
            if value < 55:
                return "africa_block"
            if value < 78:
                return "africa_whitewash"
            if value < 94:
                return "africa_colour"
            return "default"
        if family in {"school", "shop", "townhouse"}:
            if value < 34:
                return "africa_whitewash"
            if value < 63:
                return "africa_block"
            if value < 90:
                return "africa_colour"
            return "africa_earth"
        if family == "agricultural":
            if value < 52:
                return "africa_earth"
            if value < 77:
                return "africa_block"
            if value < 92:
                return "africa_whitewash"
            return "default"
        if value < 43:
            return "africa_earth"
        if value < 68:
            return "africa_whitewash"
        if value < 88:
            return "africa_colour"
        return "africa_block"

    if region_identifier == "middle_east" and family in supported:
        if material in {"mud", "adobe", "earth", "clay", "rammed_earth"}:
            return "middle_east_adobe"
        if material in {"stone", "sandstone", "limestone", "masonry"}:
            return "middle_east_sandstone"
        if material in {"concrete", "cement_block", "concrete_blocks", "precast_concrete"}:
            return "middle_east_concrete"
        if colour in {"white", "off_white", "cream", "ivory", "light_grey", "light_gray"}:
            return "middle_east_whitewash"
        if colour in {"beige", "sand", "yellow", "ochre", "brown", "terracotta"}:
            return "middle_east_sandstone"
        if material in {"plaster", "stucco", "render"}:
            return "middle_east_whitewash"
        if material in {"wood", "timber", "metal", "glass"}:
            return "default"
        if family == "urban":
            if value < 58:
                return "middle_east_concrete"
            if value < 82:
                return "middle_east_sandstone"
            if value < 94:
                return "middle_east_whitewash"
            return "default"
        if family in {"school", "shop", "townhouse"}:
            if value < 36:
                return "middle_east_sandstone"
            if value < 66:
                return "middle_east_concrete"
            if value < 89:
                return "middle_east_whitewash"
            return "middle_east_adobe"
        if family == "agricultural":
            if value < 51:
                return "middle_east_adobe"
            if value < 81:
                return "middle_east_sandstone"
            if value < 95:
                return "middle_east_concrete"
            return "default"
        if value < 40:
            return "middle_east_sandstone"
        if value < 69:
            return "middle_east_adobe"
        if value < 90:
            return "middle_east_whitewash"
        return "middle_east_concrete"

    return "default"


def _roof_style(
    tags: Mapping[str, str], family: str, regional_style: str = "default"
) -> str:
    value = tags.get("roof:shape", "").casefold()
    if value in {"flat", "terrace", "skillion", "shed"}:
        return "flat"
    if value in {"gabled", "gable", "half-hipped", "hipped", "pyramidal", "gambrel", "mansard"}:
        return "gabled"
    if regional_style.startswith("middle_east_"):
        return "flat"
    if regional_style == "africa_earth" and family in {"residential", "townhouse", "urban", "school", "shop"}:
        return "flat"
    if regional_style.startswith("africa_") and family in {"townhouse", "urban", "school", "shop"}:
        return "flat"
    return "flat" if family in {"urban", "industrial", "school", "shop"} else "gabled"


def _height(tags: Mapping[str, str], family: str, level_height: float) -> float:
    # Rural homes are capped at two storeys. Schools and shops are deliberately
    # one-storey buildings, while barns/warehouses remain one tall usable level.
    maximum_heights = {
        "residential": 2.0 * level_height,
        "townhouse": 3.0 * level_height,
        "school": 1.0 * level_height,
        "shop": 1.0 * level_height,
        "agricultural": 2.0 * level_height,
        "outbuilding": 1.35 * level_height,
    }
    explicit = _parse_number(tags.get("height"))
    if explicit is not None and explicit > 0:
        return min(explicit, maximum_heights.get(family, explicit))
    levels = _parse_number(tags.get("building:levels"))
    if levels is not None and levels > 0:
        if family == "agricultural":
            return maximum_heights[family]
        level_caps = {"residential": 2.0, "townhouse": 3.0, "school": 1.0, "shop": 1.0, "outbuilding": 1.0}
        levels = min(levels, level_caps.get(family, levels))
        return levels * level_height
    defaults = {
        "residential": 6.0,
        "townhouse": 9.0,
        "urban": 12.0,
        "industrial": 7.0,
        "agricultural": 6.0,
        "outbuilding": 3.0,
        "church": 12.0,
        "school": 3.0,
        "shop": 3.0,
    }
    return defaults[family]


def _quantize(value: float, quantum: float, minimum: float, maximum: float) -> float:
    clamped = min(maximum, max(minimum, value))
    return round(round(clamped / quantum) * quantum, 3)


def footprint_from_polygon(points: Sequence[PointXZ]) -> _Footprint:
    if len(points) < 3:
        raise ValueError("building footprint requires at least three points")
    polygon = Polygon(points)
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError("building footprint has no usable area")
    rectangle = polygon.minimum_rotated_rectangle
    coordinates = list(rectangle.exterior.coords)[:4]
    edges: list[tuple[float, float, float]] = []
    for start, end in zip(coordinates, coordinates[1:] + coordinates[:1]):
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        edges.append((math.hypot(dx, dz), dx, dz))
    length, dx, dz = max(edges, key=lambda edge: (edge[0], abs(edge[2]), abs(edge[1])))
    width = min(edge[0] for edge in edges)
    heading = math.degrees(math.atan2(dx, dz)) % 180.0
    return _Footprint(max(0.1, width), max(0.1, length), heading)


def _cstring(value: str, size: int, label: str) -> bytes:
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be ASCII") from exc
    if len(encoded) >= size:
        raise ValueError(f"{label} exceeds {size - 1} bytes")
    return encoded + bytes(size - len(encoded))


def _quad(
    texture: str,
    points: tuple[int, int, int, int],
    normal: int,
    u_scale: float = 1.0,
    v_scale: float = 1.0,
) -> _Face:
    return _Face(texture, (
        (points[0], normal, 0.0, v_scale),
        (points[1], normal, 0.0, 0.0),
        (points[2], normal, u_scale, 0.0),
        (points[3], normal, u_scale, v_scale),
    ))


def _quad_uv(
    texture: str,
    points: tuple[int, int, int, int],
    normal: int,
    *,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
) -> _Face:
    """Return a textured quad with explicit UV bounds.

    Closed painted wall atlases include a footing/plinth strip near the bottom.
    Explicit UVs let upper wall bands reuse only the body of the atlas instead
    of repeating that footing halfway up a tall facade.
    """

    return _Face(texture, (
        (points[0], normal, u0, v1),
        (points[1], normal, u0, v0),
        (points[2], normal, u1, v0),
        (points[3], normal, u1, v1),
    ))


def _triangle(texture: str, points: tuple[int, int, int], normal: int) -> _Face:
    return _Face(texture, (
        (points[0], normal, 0.0, 1.0),
        (points[1], normal, 0.5, 0.0),
        (points[2], normal, 1.0, 1.0),
    ))


def _reverse_face(face: _Face) -> _Face:
    """Return the same polygon with opposite MLOD winding and preserved UVs.

    CWA's SP3X renderer is aggressively one-sided.  Emitting the reverse face
    makes the generated shell robust against the legacy engine's coordinate and
    winding conversions while keeping the exterior UV mapping identical.
    """

    return _Face(face.texture, tuple(reversed(face.vertices)), face.flags)


def _double_sided_faces(faces: Sequence[_Face]) -> tuple[_Face, ...]:
    doubled: list[_Face] = []
    for face in faces:
        doubled.append(face)
        doubled.append(_reverse_face(face))
    return tuple(doubled)


def _add_foundation_skirt(
    points: tuple[tuple[float, float, float], ...],
    faces: tuple[_Face, ...],
    *,
    half_width: float,
    half_length: float,
    texture: str,
    depth: float,
    top_height: float = 0.0,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[_Face, ...]]:
    if depth <= 0.0 and top_height <= 0.0:
        return points, faces
    start = len(points)
    # The skirt is a terrain-gap mask, not a decorative belt around the wall.
    # Stop it exactly at the model origin (or church plinth top) so it cannot
    # overlap the facade and create a visible above-ground foundation stripe.
    top = max(0.0, float(top_height))
    bottom = -max(0.0, float(depth))
    points = points + (
        (-half_width, bottom, -half_length), (half_width, bottom, -half_length),
        (half_width, bottom, half_length), (-half_width, bottom, half_length),
        (-half_width, top, -half_length), (half_width, top, -half_length),
        (half_width, top, half_length), (-half_width, top, half_length),
    )
    vertical_scale = max(0.25, (top - bottom) / 1.5)
    skirt = (
        _quad(texture, (start + 0, start + 4, start + 5, start + 1), 0, half_width / 2.0, vertical_scale),
        _quad(texture, (start + 1, start + 5, start + 6, start + 2), 1, half_length / 2.0, vertical_scale),
        _quad(texture, (start + 2, start + 6, start + 7, start + 3), 2, half_width / 2.0, vertical_scale),
        _quad(texture, (start + 3, start + 7, start + 4, start + 0), 3, half_length / 2.0, vertical_scale),
    )
    return points, faces + skirt


def _front_faces_with_doorway(
    points: tuple[tuple[float, float, float], ...],
    *,
    half_width: float,
    front_z: float,
    wall_top: float,
    outer_bottom_left: int,
    outer_top_left: int,
    outer_top_right: int,
    outer_bottom_right: int,
    texture: str,
    normal: int,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[_Face, ...]]:
    """Split the front wall around its door and every procedural window bay."""

    door_half = min(0.8, max(0.6, half_width * 0.18))
    door_height = min(INTERIOR_DOOR_HEIGHT_M, max(1.9, wall_top - 0.25))
    openings = _window_openings(
        -half_width,
        half_width,
        wall_top,
        ground_exclusions=((-door_half - 0.35, door_half + 0.35),),
    )
    openings = openings + (
        (-door_half, door_half, 0.0, door_height),
    )
    return _wall_faces_with_openings(
        points,
        horizontal_min=-half_width,
        horizontal_max=half_width,
        plane=front_z,
        wall_top=wall_top,
        horizontal_axis="x",
        openings=openings,
        texture=texture,
        normal=normal,
    )


def _window_openings(
    horizontal_min: float,
    horizontal_max: float,
    wall_top: float,
    *,
    ground_exclusions: Sequence[tuple[float, float]] = (),
) -> tuple[tuple[float, float, float, float], ...]:
    """Return repeated window bays shared by visual and collision LODs."""

    span = horizontal_max - horizontal_min
    bay_count = max(1, min(12, round(span / 3.8)))
    bay_width = span / bay_count
    opening_half = min(0.75, max(0.52, bay_width * 0.20))
    storeys = max(1, min(5, round(wall_top / 3.0)))
    openings: list[tuple[float, float, float, float]] = []
    for storey in range(storeys):
        sill = storey * 3.0 + INTERIOR_WINDOW_SILL_M
        top = min(storey * 3.0 + INTERIOR_WINDOW_TOP_M, wall_top - 0.25)
        if top - sill < 0.45:
            continue
        for bay in range(bay_count):
            centre = horizontal_min + (bay + 0.5) * bay_width
            opening_min = centre - opening_half
            opening_max = centre + opening_half
            if storey == 0 and any(
                opening_min < excluded_max and opening_max > excluded_min
                for excluded_min, excluded_max in ground_exclusions
            ):
                continue
            openings.append((opening_min, opening_max, sill, top))
    return tuple(openings)


def _solid_wall_rectangles(
    horizontal_min: float,
    horizontal_max: float,
    wall_top: float,
    openings: Sequence[tuple[float, float, float, float]],
) -> tuple[tuple[float, float, float, float], ...]:
    """Merge the solid cells around apertures into a compact rectangle set."""

    horizontal_breaks = sorted({
        horizontal_min, horizontal_max,
        *(value for opening in openings for value in opening[:2]),
    })
    vertical_breaks = sorted({
        0.0, wall_top,
        *(value for opening in openings for value in opening[2:]),
    })
    columns = len(horizontal_breaks) - 1
    rows = len(vertical_breaks) - 1
    solid = [[True] * columns for _ in range(rows)]
    for row in range(rows):
        vertical_centre = (vertical_breaks[row] + vertical_breaks[row + 1]) * 0.5
        for column in range(columns):
            horizontal_centre = (
                horizontal_breaks[column] + horizontal_breaks[column + 1]
            ) * 0.5
            solid[row][column] = not any(
                opening_min < horizontal_centre < opening_max
                and opening_bottom < vertical_centre < opening_top
                for opening_min, opening_max, opening_bottom, opening_top in openings
            )

    rectangles: list[tuple[float, float, float, float]] = []
    used = [[False] * columns for _ in range(rows)]
    for row in range(rows):
        for column in range(columns):
            if not solid[row][column] or used[row][column]:
                continue
            column_end = column + 1
            while (
                column_end < columns
                and solid[row][column_end]
                and not used[row][column_end]
            ):
                column_end += 1
            row_end = row + 1
            while row_end < rows and all(
                solid[row_end][index] and not used[row_end][index]
                for index in range(column, column_end)
            ):
                row_end += 1
            for used_row in range(row, row_end):
                for used_column in range(column, column_end):
                    used[used_row][used_column] = True
            rectangles.append((
                horizontal_breaks[column],
                horizontal_breaks[column_end],
                vertical_breaks[row],
                vertical_breaks[row_end],
            ))
    return tuple(rectangles)


def _wall_faces_with_openings(
    points: tuple[tuple[float, float, float], ...],
    *,
    horizontal_min: float,
    horizontal_max: float,
    plane: float,
    wall_top: float,
    horizontal_axis: str,
    openings: Sequence[tuple[float, float, float, float]],
    texture: str,
    normal: int,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[_Face, ...]]:
    """Tessellate one wall while omitting every requested open aperture."""

    def wall_point(horizontal: float, vertical: float) -> tuple[float, float, float]:
        if horizontal_axis == "x":
            return (horizontal, vertical, plane)
        if horizontal_axis == "z":
            return (plane, vertical, horizontal)
        raise ValueError(f"unsupported wall horizontal axis: {horizontal_axis}")

    faces: list[_Face] = []
    for cell_min, cell_max, vertical_min, vertical_max in _solid_wall_rectangles(
        horizontal_min, horizontal_max, wall_top, openings
    ):
        start = len(points)
        points = points + (
            wall_point(cell_min, vertical_min),
            wall_point(cell_min, vertical_max),
            wall_point(cell_max, vertical_max),
            wall_point(cell_max, vertical_min),
        )
        u0 = (cell_min - horizontal_min) / 4.0
        u1 = (cell_max - horizontal_min) / 4.0
        v0 = (wall_top - vertical_max) / 3.0
        v1 = (wall_top - vertical_min) / 3.0
        faces.append(_Face(texture, (
            (start + 0, normal, u0, v1),
            (start + 1, normal, u0, v0),
            (start + 2, normal, u1, v0),
            (start + 3, normal, u1, v1),
        )))
    return points, tuple(faces)


def _opening_reveal_faces(
    points: tuple[tuple[float, float, float], ...],
    *,
    openings: Sequence[tuple[float, float, float, float]],
    outer_plane: float,
    inner_plane: float,
    horizontal_axis: str,
    texture: str,
    normal: int,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[_Face, ...]]:
    """Add recessed jambs, sill, and lintel while leaving each bay empty."""

    def reveal_point(
        horizontal: float, vertical: float, plane: float
    ) -> tuple[float, float, float]:
        if horizontal_axis == "x":
            return (horizontal, vertical, plane)
        if horizontal_axis == "z":
            return (plane, vertical, horizontal)
        raise ValueError(f"unsupported reveal horizontal axis: {horizontal_axis}")

    faces: list[_Face] = []
    depth = max(0.05, abs(inner_plane - outer_plane))
    for opening_min, opening_max, opening_bottom, opening_top in openings:
        start = len(points)
        points = points + (
            reveal_point(opening_min, opening_bottom, outer_plane),
            reveal_point(opening_min, opening_top, outer_plane),
            reveal_point(opening_max, opening_top, outer_plane),
            reveal_point(opening_max, opening_bottom, outer_plane),
            reveal_point(opening_min, opening_bottom, inner_plane),
            reveal_point(opening_min, opening_top, inner_plane),
            reveal_point(opening_max, opening_top, inner_plane),
            reveal_point(opening_max, opening_bottom, inner_plane),
        )
        opening_width = max(0.2, (opening_max - opening_min) / 2.0)
        opening_height = max(0.2, (opening_top - opening_bottom) / 2.0)
        faces.extend((
            _quad(texture, (start + 0, start + 4, start + 5, start + 1), normal,
                  depth, opening_height),
            _quad(texture, (start + 3, start + 2, start + 6, start + 7), normal,
                  opening_height, depth),
            _quad(texture, (start + 0, start + 3, start + 7, start + 4), normal,
                  opening_width, depth),
            _quad(texture, (start + 1, start + 5, start + 6, start + 2), normal,
                  depth, opening_width),
        ))
    return points, tuple(faces)


def _add_interior_wall_shell(
    key: BuildingVariantKey,
    points: tuple[tuple[float, float, float], ...],
    faces: tuple[_Face, ...],
    *,
    wall_top: float,
    texture: str,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[_Face, ...]]:
    """Add a darker recessed shell around the same genuine open apertures."""

    half_width = key.width_m * 0.5
    half_length = key.length_m * 0.5
    inset = min(0.30, max(0.18, min(key.width_m, key.length_m) * 0.025))
    door_half = min(0.8, max(0.6, half_width * 0.18))
    door_height = min(INTERIOR_DOOR_HEIGHT_M, max(1.9, wall_top - 0.25))
    front_openings = _window_openings(
        -half_width,
        half_width,
        wall_top,
        ground_exclusions=((-door_half - 0.35, door_half + 0.35),),
    ) + ((-door_half, door_half, 0.0, door_height),)
    back_openings = _window_openings(-half_width, half_width, wall_top)
    side_openings = _window_openings(-half_length, half_length, wall_top)
    walls = (
        ("x", -half_length, -half_length + inset, -half_width, half_width, front_openings, 0),
        ("z", half_width, half_width - inset, -half_length, half_length, side_openings, 1),
        ("x", half_length, half_length - inset, -half_width, half_width, back_openings, 2),
        ("z", -half_width, -half_width + inset, -half_length, half_length, side_openings, 3),
    )
    for axis, outer_plane, inner_plane, wall_min, wall_max, openings, normal in walls:
        points, inner_faces = _wall_faces_with_openings(
            points,
            horizontal_min=wall_min,
            horizontal_max=wall_max,
            plane=inner_plane,
            wall_top=wall_top,
            horizontal_axis=axis,
            openings=openings,
            texture=texture,
            normal=normal,
        )
        points, reveal_faces = _opening_reveal_faces(
            points,
            openings=openings,
            outer_plane=outer_plane,
            inner_plane=inner_plane,
            horizontal_axis=axis,
            texture=texture,
            normal=normal,
        )
        faces = faces + inner_faces + reveal_faces
    return points, faces


def _add_white_window_trim(
    key: BuildingVariantKey,
    points: tuple[tuple[float, float, float], ...],
    faces: tuple[_Face, ...],
    *,
    wall_top: float,
    texture: str,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[_Face, ...]]:
    """Add projecting white rectangular surrounds without closing the holes."""

    half_width = key.width_m * 0.5
    half_length = key.length_m * 0.5
    door_half = min(0.8, max(0.6, half_width * 0.18))
    front_windows = _window_openings(
        -half_width,
        half_width,
        wall_top,
        ground_exclusions=((-door_half - 0.35, door_half + 0.35),),
    )
    back_windows = _window_openings(-half_width, half_width, wall_top)
    side_windows = _window_openings(-half_length, half_length, wall_top)
    trim_width = 0.16
    projection = 0.07

    def add_trim_box(
        horizontal_min: float,
        horizontal_max: float,
        vertical_min: float,
        vertical_max: float,
        wall_plane: float,
        projected_plane: float,
        axis: str,
        normal: int,
    ) -> None:
        nonlocal points, faces
        plane_min, plane_max = sorted((wall_plane, projected_plane))
        if axis == "x":
            x0, x1 = horizontal_min, horizontal_max
            z0, z1 = plane_min, plane_max
        elif axis == "z":
            x0, x1 = plane_min, plane_max
            z0, z1 = horizontal_min, horizontal_max
        else:
            raise ValueError(f"unsupported trim horizontal axis: {axis}")
        start = len(points)
        points = points + (
            (x0, vertical_min, z0), (x1, vertical_min, z0),
            (x1, vertical_min, z1), (x0, vertical_min, z1),
            (x0, vertical_max, z0), (x1, vertical_max, z0),
            (x1, vertical_max, z1), (x0, vertical_max, z1),
        )
        faces = faces + (
            _quad(texture, (start + 0, start + 4, start + 5, start + 1), normal),
            _quad(texture, (start + 1, start + 5, start + 6, start + 2), normal),
            _quad(texture, (start + 2, start + 6, start + 7, start + 3), normal),
            _quad(texture, (start + 3, start + 7, start + 4, start + 0), normal),
            _quad(texture, (start + 4, start + 7, start + 6, start + 5), normal),
            _quad(texture, (start + 0, start + 1, start + 2, start + 3), normal),
        )

    walls = (
        ("x", -half_length, -half_length - projection, front_windows, 0),
        ("z", half_width, half_width + projection, side_windows, 1),
        ("x", half_length, half_length + projection, back_windows, 2),
        ("z", -half_width, -half_width - projection, side_windows, 3),
    )
    for axis, wall_plane, projected_plane, openings, normal in walls:
        for opening_min, opening_max, opening_bottom, opening_top in openings:
            add_trim_box(
                opening_min - trim_width,
                opening_min,
                opening_bottom - trim_width,
                opening_top + trim_width,
                wall_plane,
                projected_plane,
                axis,
                normal,
            )
            add_trim_box(
                opening_max,
                opening_max + trim_width,
                opening_bottom - trim_width,
                opening_top + trim_width,
                wall_plane,
                projected_plane,
                axis,
                normal,
            )
            add_trim_box(
                opening_min,
                opening_max,
                opening_bottom - trim_width,
                opening_bottom,
                wall_plane,
                projected_plane,
                axis,
                normal,
            )
            add_trim_box(
                opening_min,
                opening_max,
                opening_top,
                opening_top + trim_width,
                wall_plane,
                projected_plane,
                axis,
                normal,
            )
    return points, faces


def _add_simple_interior_visuals(
    key: BuildingVariantKey,
    points: tuple[tuple[float, float, float], ...],
    normals: tuple[tuple[float, float, float], ...],
    faces: tuple[_Face, ...],
    *,
    wall_texture: str,
    floor_texture: str,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
    tuple[_Face, ...],
]:
    """Add a visible ground-floor room and one doorway partition."""

    inset = min(0.28, max(0.16, min(key.width_m, key.length_m) * 0.02))
    half_width = max(0.5, key.width_m * 0.5 - inset)
    half_length = max(0.5, key.length_m * 0.5 - inset)
    ceiling = min(2.85, max(2.35, key.height_m - 0.15))
    normal_start = len(normals)
    normals = normals + (
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )

    floor_start = len(points)
    points = points + (
        (-half_width, 0.025, -half_length),
        (half_width, 0.025, -half_length),
        (half_width, 0.025, half_length),
        (-half_width, 0.025, half_length),
        (-half_width, ceiling, -half_length),
        (half_width, ceiling, -half_length),
        (half_width, ceiling, half_length),
        (-half_width, ceiling, half_length),
    )
    interior_faces: list[_Face] = [
        _quad(
            floor_texture,
            (floor_start + 0, floor_start + 3, floor_start + 2, floor_start + 1),
            normal_start,
            key.width_m / 2.0,
            key.length_m / 2.0,
        ),
        _quad(
            wall_texture,
            (floor_start + 4, floor_start + 5, floor_start + 6, floor_start + 7),
            normal_start + 1,
            key.width_m / 2.0,
            key.length_m / 2.0,
        ),
    ]

    if key.width_m >= 6.0 and key.length_m >= 7.0:
        door_half = min(0.75, max(0.55, half_width * 0.16))
        door_height = min(INTERIOR_DOOR_HEIGHT_M, ceiling - 0.15)
        partition_start = len(points)
        points = points + (
            (-half_width, 0.0, 0.0),
            (-half_width, ceiling, 0.0),
            (half_width, ceiling, 0.0),
            (half_width, 0.0, 0.0),
            (-door_half, 0.0, 0.0),
            (-door_half, door_height, 0.0),
            (door_half, door_height, 0.0),
            (door_half, 0.0, 0.0),
        )
        interior_faces.extend((
            _quad(
                wall_texture,
                (partition_start + 0, partition_start + 1, partition_start + 5, partition_start + 4),
                normal_start + 2,
                max(0.25, half_width - door_half),
                ceiling / 2.0,
            ),
            _quad(
                wall_texture,
                (partition_start + 7, partition_start + 6, partition_start + 2, partition_start + 3),
                normal_start + 2,
                max(0.25, half_width - door_half),
                ceiling / 2.0,
            ),
            _quad(
                wall_texture,
                (partition_start + 5, partition_start + 1, partition_start + 2, partition_start + 6),
                normal_start + 2,
                door_half,
                max(0.2, ceiling - door_height),
            ),
        ))

    return points, normals, faces + tuple(interior_faces)


def _main_building_height(key: BuildingVariantKey) -> float:
    """Return the wall/roof height used by the main building mass.

    Churches keep their existing tower proportions, but the nave itself is one
    conventional 3 m storey shorter. This is deliberately a model-only change:
    footprint, grounding, foundation depth and tower placement stay unchanged.
    """

    if key.family == "church":
        return max(3.0, key.height_m - 3.0)
    return key.height_m


def _visual_lod(
    key: BuildingVariantKey,
    wall_texture: str,
    roof_texture: str,
    roof_pitch_degrees: float,
    front_texture: str | None = None,
    foundation_texture: str | None = None,
    foundation_depth: float = 0.0,
    church_plinth_height: float = 0.0,
    interior_texture: str | None = None,
    window_trim_texture: str | None = None,
    plain_wall_texture: str | None = None,
) -> _Lod:
    front_texture = front_texture or wall_texture
    foundation_texture = foundation_texture or wall_texture
    interior_texture = interior_texture or wall_texture
    plain_wall_texture = plain_wall_texture or wall_texture
    half_width = key.width_m / 2.0
    half_length = key.length_m / 2.0
    main_height = _main_building_height(key)
    ground_floor_height = min(3.0, main_height)

    if key.roof_style == "flat":
        height = main_height
        points = (
            (-half_width, 0.0, -half_length), (half_width, 0.0, -half_length),
            (half_width, 0.0, half_length), (-half_width, 0.0, half_length),
            (-half_width, height, -half_length), (half_width, height, -half_length),
            (half_width, height, half_length), (-half_width, height, half_length),
        )
        normals = (
            (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0, 0.0),
        )
        front_faces: list[_Face] = []
        if key.interiors:
            points, doorway_faces = _front_faces_with_doorway(
                points,
                half_width=half_width,
                front_z=-half_length,
                wall_top=height,
                outer_bottom_left=0,
                outer_top_left=4,
                outer_top_right=5,
                outer_bottom_right=1,
                texture=wall_texture,
                normal=0,
            )
            front_faces.extend(doorway_faces)
        elif ground_floor_height < height - 1e-6:
            ground_left = len(points)
            ground_right = ground_left + 1
            points = points + (
                (-half_width, ground_floor_height, -half_length),
                (half_width, ground_floor_height, -half_length),
            )
            # The entrance atlas is used once across one physical ground-floor
            # strip. Upper storeys use the ordinary wall texture, so a door can
            # neither repeat vertically nor stretch into the second floor.
            front_faces.extend((
                _quad(front_texture, (0, ground_left, ground_right, 1), 0, 1.0, 1.0),
                _quad(
                    wall_texture,
                    (ground_left, 4, 5, ground_right),
                    0,
                    key.width_m / 4.0,
                    (height - ground_floor_height) / 3.0,
                ),
            ))
        else:
            front_faces.append(_quad(front_texture, (0, 4, 5, 1), 0, 1.0, 1.0))
        if key.interiors:
            side_openings = _window_openings(-half_length, half_length, height)
            back_openings = _window_openings(-half_width, half_width, height)
            points, right_faces = _wall_faces_with_openings(
                points, horizontal_min=-half_length, horizontal_max=half_length,
                plane=half_width, wall_top=height, horizontal_axis="z",
                openings=side_openings, texture=wall_texture, normal=1,
            )
            points, back_faces = _wall_faces_with_openings(
                points, horizontal_min=-half_width, horizontal_max=half_width,
                plane=half_length, wall_top=height, horizontal_axis="x",
                openings=back_openings, texture=wall_texture, normal=2,
            )
            points, left_faces = _wall_faces_with_openings(
                points, horizontal_min=-half_length, horizontal_max=half_length,
                plane=-half_width, wall_top=height, horizontal_axis="z",
                openings=side_openings, texture=wall_texture, normal=3,
            )
            exterior_side_faces = right_faces + back_faces + left_faces
        else:
            exterior_side_faces = (
                _quad(wall_texture, (1, 5, 6, 2), 1, key.length_m / 4.0, height / 3.0),
                _quad(wall_texture, (2, 6, 7, 3), 2, key.width_m / 4.0, height / 3.0),
                _quad(wall_texture, (3, 7, 4, 0), 3, key.length_m / 4.0, height / 3.0),
            )
        faces = tuple(front_faces) + exterior_side_faces + (
            _quad(roof_texture, (4, 7, 6, 5), 4, key.width_m / 4.0, key.length_m / 4.0),
            _quad("", (0, 1, 2, 3), 5),
        )
        if key.interiors:
            points, faces = _add_interior_wall_shell(
                key,
                points,
                faces,
                wall_top=height,
                texture=interior_texture,
            )
            if (
                window_trim_texture
                and key.regional_style in WHITE_WINDOW_TRIM_REGIONAL_STYLES
            ):
                points, faces = _add_white_window_trim(
                    key,
                    points,
                    faces,
                    wall_top=height,
                    texture=window_trim_texture,
                )
            points, normals, faces = _add_simple_interior_visuals(
                key,
                points,
                normals,
                faces,
                wall_texture=interior_texture,
                floor_texture=foundation_texture,
            )
        plinth_height = max(0.0, float(church_plinth_height)) if key.family == "church" else 0.0
        if plinth_height > 0.0:
            points = tuple((x, y + plinth_height, z) for x, y, z in points)
        points, faces = _add_foundation_skirt(
            points, faces, half_width=half_width, half_length=half_length,
            texture=foundation_texture, depth=foundation_depth, top_height=plinth_height,
        )
        return _Lod(points, normals, _double_sided_faces(faces), 1.0)

    maximum_rise = half_width * math.tan(math.radians(roof_pitch_degrees))
    roof_rise = min(maximum_rise, max(1.0, main_height * 0.35))
    eave_height = max(2.5, main_height - roof_rise)
    roof_rise = main_height - eave_height
    points = (
        (-half_width, 0.0, -half_length), (half_width, 0.0, -half_length),
        (half_width, 0.0, half_length), (-half_width, 0.0, half_length),
        (-half_width, eave_height, -half_length), (half_width, eave_height, -half_length),
        (half_width, eave_height, half_length), (-half_width, eave_height, half_length),
        (0.0, main_height, -half_length), (0.0, main_height, half_length),
    )
    slope_length = math.hypot(half_width, roof_rise)
    left_normal = (-roof_rise / slope_length, half_width / slope_length, 0.0)
    right_normal = (roof_rise / slope_length, half_width / slope_length, 0.0)
    normals = (
        (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0), left_normal, right_normal, (0.0, -1.0, 0.0),
    )

    # Churches use one painted window level only. Their upper nave and tower
    # shaft use a matching windowless wall material so the three-metre facade
    # atlas is never repeated vertically into several rows of windows.
    church_tower_half = 0.0
    church_tower_depth = 0.0
    church_tower_front_z = 0.0
    church_tower_back_z = 0.0
    if key.family == "church":
        church_tower_half = min(4.0, max(1.5, key.width_m * 0.22))
        church_tower_depth = min(6.0, max(3.0, key.length_m * 0.24))
        # Put the tower front a few centimetres ahead of the nave facade. The
        # previous version placed both faces on exactly the same plane, which
        # created classic depth-buffer flicker/clipping in CWA.
        church_tower_front_z = -half_length - 0.04
        church_tower_back_z = min(
            half_length - 0.02, -half_length + church_tower_depth
        )

    front_faces: list[_Face] = []
    ground_floor_height = min(3.0, eave_height)
    if key.interiors:
        points, doorway_faces = _front_faces_with_doorway(
            points,
            half_width=half_width,
            front_z=-half_length,
            wall_top=eave_height,
            outer_bottom_left=0,
            outer_top_left=4,
            outer_top_right=5,
            outer_bottom_right=1,
            texture=wall_texture,
            normal=0,
        )
        front_faces.extend(doorway_faces)
    elif ground_floor_height < eave_height - 1e-6:
        ground_left = len(points)
        ground_right = ground_left + 1
        points = points + (
            (-half_width, ground_floor_height, -half_length),
            (half_width, ground_floor_height, -half_length),
        )
        front_faces.extend((
            _quad(front_texture, (0, ground_left, ground_right, 1), 0, 1.0, 1.0),
            _quad(
                plain_wall_texture if key.family == "church" else wall_texture,
                (ground_left, 4, 5, ground_right),
                0,
                key.width_m / 4.0,
                max(1.0, (eave_height - ground_floor_height) / 3.0),
            ),
        ))
    else:
        front_faces.append(_quad(front_texture, (0, 4, 5, 1), 0, 1.0, 1.0))

    if key.interiors:
        side_openings = _window_openings(-half_length, half_length, eave_height)
        back_openings = _window_openings(-half_width, half_width, eave_height)
        points, right_faces = _wall_faces_with_openings(
            points, horizontal_min=-half_length, horizontal_max=half_length,
            plane=half_width, wall_top=eave_height, horizontal_axis="z",
            openings=side_openings, texture=wall_texture, normal=1,
        )
        points, back_faces = _wall_faces_with_openings(
            points, horizontal_min=-half_width, horizontal_max=half_width,
            plane=half_length, wall_top=eave_height, horizontal_axis="x",
            openings=back_openings, texture=wall_texture, normal=2,
        )
        points, left_faces = _wall_faces_with_openings(
            points, horizontal_min=-half_length, horizontal_max=half_length,
            plane=-half_width, wall_top=eave_height, horizontal_axis="z",
            openings=side_openings, texture=wall_texture, normal=3,
        )
    elif key.family == "church" and ground_floor_height < eave_height - 1e-6:
        mid = len(points)
        points = points + (
            (-half_width, ground_floor_height, -half_length),
            (half_width, ground_floor_height, -half_length),
            (half_width, ground_floor_height, half_length),
            (-half_width, ground_floor_height, half_length),
        )
        m0, m1, m2, m3 = mid, mid + 1, mid + 2, mid + 3
        right_faces = (
            _quad(wall_texture, (1, m1, m2, 2), 1, key.length_m / 4.0, 1.0),
            _quad(plain_wall_texture, (m1, 5, 6, m2), 1, key.length_m / 4.0, max(1.0, (eave_height-ground_floor_height)/3.0)),
        )
        back_faces = (
            _quad(wall_texture, (2, m2, m3, 3), 2, key.width_m / 4.0, 1.0),
            _quad(plain_wall_texture, (m2, 6, 7, m3), 2, key.width_m / 4.0, max(1.0, (eave_height-ground_floor_height)/3.0)),
        )
        left_faces = (
            _quad(wall_texture, (3, m3, m0, 0), 3, key.length_m / 4.0, 1.0),
            _quad(plain_wall_texture, (m3, 7, 4, m0), 3, key.length_m / 4.0, max(1.0, (eave_height-ground_floor_height)/3.0)),
        )
    else:
        right_faces = (_quad(
            wall_texture, (1, 5, 6, 2), 1,
            key.length_m / 4.0, eave_height / 3.0,
        ),)
        back_faces = (_quad(
            wall_texture, (2, 6, 7, 3), 2,
            key.width_m / 4.0, eave_height / 3.0,
        ),)
        left_faces = (_quad(
            wall_texture, (3, 7, 4, 0), 3,
            key.length_m / 4.0, eave_height / 3.0,
        ),)

    gable_texture = plain_wall_texture if key.family == "church" else wall_texture
    if key.family == "church":
        # Cut the nave roof around the tower volume. Previously the complete roof
        # continued straight through the tower, so the two surfaces intersected
        # and could clip/flicker depending on camera angle. The tower remains
        # part of this same P3D, but the mesh now joins it instead of overlapping.
        z_join = church_tower_back_z
        roof_y_at_tower = eave_height + roof_rise * max(
            0.0, 1.0 - church_tower_half / max(1e-6, half_width)
        )
        rp = len(points)
        points = points + (
            (-half_width, eave_height, z_join),
            (0.0, key.height_m, z_join),
            (half_width, eave_height, z_join),
            (-church_tower_half, roof_y_at_tower, -half_length),
            (-church_tower_half, roof_y_at_tower, z_join),
            (church_tower_half, roof_y_at_tower, -half_length),
            (church_tower_half, roof_y_at_tower, z_join),
        )
        jl, jr, rr = rp, rp + 2, rp + 1
        lfront, ljoin, rfront, rjoin = rp + 3, rp + 4, rp + 5, rp + 6
        roof_faces = (
            # Back roof, full width from tower rear to nave rear.
            _quad(roof_texture, (jl, 7, 9, rr), 4, key.length_m / 4.0, slope_length / 4.0),
            _quad(roof_texture, (rr, 9, 6, jr), 5, key.length_m / 4.0, slope_length / 4.0),
            # Front shoulders beside the tower. No face exists beneath tower.
            _quad(roof_texture, (4, jl, ljoin, lfront), 4, max(0.5, church_tower_depth / 4.0), max(0.5, (half_width-church_tower_half)/4.0)),
            _quad(roof_texture, (rfront, rjoin, jr, 5), 5, max(0.5, church_tower_depth / 4.0), max(0.5, (half_width-church_tower_half)/4.0)),
        )
    else:
        roof_faces = (
            _quad(roof_texture, (4, 7, 9, 8), 4, key.length_m / 4.0, slope_length / 4.0),
            _quad(roof_texture, (8, 9, 6, 5), 5, key.length_m / 4.0, slope_length / 4.0),
        )

    faces = tuple(front_faces) + (
        _triangle(gable_texture, (4, 8, 5), 0),
    ) + right_faces + back_faces + (
        _triangle(gable_texture, (6, 9, 7), 2),
    ) + left_faces + roof_faces + (
        _quad("", (0, 1, 2, 3), 6),
    )

    if key.interiors:
        points, faces = _add_interior_wall_shell(
            key,
            points,
            faces,
            wall_top=eave_height,
            texture=interior_texture,
        )
        if (
            window_trim_texture
            and key.regional_style in WHITE_WINDOW_TRIM_REGIONAL_STYLES
        ):
            points, faces = _add_white_window_trim(
                key,
                points,
                faces,
                wall_top=eave_height,
                texture=window_trim_texture,
            )
        points, normals, faces = _add_simple_interior_visuals(
            key,
            points,
            normals,
            faces,
            wall_texture=interior_texture,
            floor_texture=foundation_texture,
        )

    if key.family == "church":
        # The tower is not a separate object. It is integrated into the church
        # P3D and shares the same world transform. Build it as a non-overlapping
        # continuation of the nave mesh to avoid CWA depth-buffer clipping.
        tower_half = church_tower_half
        z0 = church_tower_front_z
        z1 = church_tower_back_z
        tower_height = max(18.0, key.height_m + 8.0, eave_height + 10.0)
        apex_height = tower_height + min(8.0, max(5.0, tower_half * 2.0))
        tower_ground_height = min(3.0, tower_height)
        # Give the belfry one compact upper window band. The nave still keeps
        # only its single ground-floor window row; the tall tower shaft stays
        # mostly plain so the facade atlas never repeats into a stack of windows.
        tower_window_height = min(2.8, max(2.2, tower_half * 0.75))
        tower_window_top = max(tower_ground_height + 3.0, tower_height - 1.8)
        tower_window_bottom = max(
            tower_ground_height + 2.0, tower_window_top - tower_window_height
        )
        start = len(points)
        points = points + (
            # Bottom ring.
            (-tower_half, 0.0, z0), (tower_half, 0.0, z0),
            (tower_half, 0.0, z1), (-tower_half, 0.0, z1),
            # Ground-floor painted level.
            (-tower_half, tower_ground_height, z0), (tower_half, tower_ground_height, z0),
            (tower_half, tower_ground_height, z1), (-tower_half, tower_ground_height, z1),
            # Upper belfry window band.
            (-tower_half, tower_window_bottom, z0), (tower_half, tower_window_bottom, z0),
            (tower_half, tower_window_bottom, z1), (-tower_half, tower_window_bottom, z1),
            (-tower_half, tower_window_top, z0), (tower_half, tower_window_top, z0),
            (tower_half, tower_window_top, z1), (-tower_half, tower_window_top, z1),
            # Plain wall beneath the spire.
            (-tower_half, tower_height, z0), (tower_half, tower_height, z0),
            (tower_half, tower_height, z1), (-tower_half, tower_height, z1),
            (0.0, apex_height, (z0 + z1) * 0.5),
        )
        b0,b1,b2,b3 = start,start+1,start+2,start+3
        m0,m1,m2,m3 = start+4,start+5,start+6,start+7
        w0,w1,w2,w3 = start+8,start+9,start+10,start+11
        q0,q1,q2,q3 = start+12,start+13,start+14,start+15
        t0,t1,t2,t3,apex = start+16,start+17,start+18,start+19,start+20
        lower_plain_v = max(1.0, (tower_window_bottom-tower_ground_height)/3.0)
        upper_plain_v = max(1.0, (tower_height-tower_window_top)/3.0)
        tower_faces = (
            _quad(front_texture, (b0, m0, m1, b1), 0, 1.0, 1.0),
            _quad(plain_wall_texture, (m0, w0, w1, m1), 0, tower_half / 2.0, lower_plain_v),
            _quad(wall_texture, (w0, q0, q1, w1), 0, tower_half / 2.0, 1.0),
            _quad(plain_wall_texture, (q0, t0, t1, q1), 0, tower_half / 2.0, upper_plain_v),
            _quad(wall_texture, (b1, m1, m2, b2), 1, church_tower_depth / 4.0, 1.0),
            _quad(plain_wall_texture, (m1, w1, w2, m2), 1, church_tower_depth / 4.0, lower_plain_v),
            _quad(wall_texture, (w1, q1, q2, w2), 1, church_tower_depth / 4.0, 1.0),
            _quad(plain_wall_texture, (q1, t1, t2, q2), 1, church_tower_depth / 4.0, upper_plain_v),
            _quad(wall_texture, (b2, m2, m3, b3), 2, tower_half / 2.0, 1.0),
            _quad(plain_wall_texture, (m2, w2, w3, m3), 2, tower_half / 2.0, lower_plain_v),
            _quad(wall_texture, (w2, q2, q3, w3), 2, tower_half / 2.0, 1.0),
            _quad(plain_wall_texture, (q2, t2, t3, q3), 2, tower_half / 2.0, upper_plain_v),
            _quad(wall_texture, (b3, m3, m0, b0), 3, church_tower_depth / 4.0, 1.0),
            _quad(plain_wall_texture, (m3, w3, w0, m0), 3, church_tower_depth / 4.0, lower_plain_v),
            _quad(wall_texture, (w3, q3, q0, w0), 3, church_tower_depth / 4.0, 1.0),
            _quad(plain_wall_texture, (q3, t3, t0, q0), 3, church_tower_depth / 4.0, upper_plain_v),
            _triangle(roof_texture, (t0, apex, t1), 0),
            _triangle(roof_texture, (t1, apex, t2), 1),
            _triangle(roof_texture, (t2, apex, t3), 2),
            _triangle(roof_texture, (t3, apex, t0), 3),
        )
        faces = faces + tower_faces
    plinth_height = max(0.0, float(church_plinth_height)) if key.family == "church" else 0.0
    if plinth_height > 0.0:
        points = tuple((x, y + plinth_height, z) for x, y, z in points)
    points, faces = _add_foundation_skirt(
        points, faces, half_width=half_width, half_length=half_length,
        texture=foundation_texture, depth=foundation_depth, top_height=plinth_height,
    )
    return _Lod(points, normals, _double_sided_faces(faces), 1.0)

def _land_contact_lod(key: BuildingVariantKey) -> _Lod:
    half_width = key.width_m / 2.0
    half_length = key.length_m / 2.0
    return _Lod((
        (-half_width, 0.0, -half_length), (half_width, 0.0, -half_length),
        (half_width, 0.0, half_length), (-half_width, 0.0, half_length),
    ), (), (), _LAND_CONTACT_LOD)


def _map_symbol_for_family(family: str) -> str:
    """Return the CWA 2D-map classification for a procedural building."""

    return "house" if family in {"residential", "townhouse"} else "building"


def _hollow_geometry_lod(key: BuildingVariantKey) -> _Lod:
    """Build an enterable ground-floor collision shell from convex boxes."""

    half_width = key.width_m * 0.5
    half_length = key.length_m * 0.5
    wall_thickness = min(0.30, max(0.18, min(key.width_m, key.length_m) * 0.025))
    door_half = min(0.8, max(0.6, half_width * 0.18))
    door_height = min(INTERIOR_DOOR_HEIGHT_M, max(1.9, key.height_m - 0.25))
    ceiling = min(2.85, max(2.35, key.height_m - 0.15))

    points: list[tuple[float, float, float]] = []
    faces: list[_Face] = []
    component_ranges: list[tuple[range, range]] = []

    def add_box(
        x0: float, y0: float, z0: float,
        x1: float, y1: float, z1: float,
    ) -> None:
        if x1 - x0 <= 1.0e-6 or y1 - y0 <= 1.0e-6 or z1 - z0 <= 1.0e-6:
            return
        point_start = len(points)
        face_start = len(faces)
        points.extend((
            (x0, y0, z0), (x1, y0, z0),
            (x1, y0, z1), (x0, y0, z1),
            (x0, y1, z0), (x1, y1, z0),
            (x1, y1, z1), (x0, y1, z1),
        ))
        faces.extend(
            _Face("", tuple((point_start + point, -1, 0.0, 0.0) for point in indices))
            for indices in (
                (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3),
                (3, 7, 4, 0), (4, 7, 6, 5), (0, 1, 2, 3),
            )
        )
        component_ranges.append((
            range(point_start, point_start + 8),
            range(face_start, face_start + 6),
        ))

    def add_wall(
        *,
        horizontal_min: float,
        horizontal_max: float,
        plane_min: float,
        plane_max: float,
        horizontal_axis: str,
        openings: Sequence[tuple[float, float, float, float]],
    ) -> None:
        """Fill a wall grid with merged convex boxes outside its apertures."""

        for h0, h1, y0, y1 in _solid_wall_rectangles(
            horizontal_min, horizontal_max, key.height_m, openings
        ):
            if horizontal_axis == "x":
                add_box(h0, y0, plane_min, h1, y1, plane_max)
            elif horizontal_axis == "z":
                add_box(plane_min, y0, h0, plane_max, y1, h1)
            else:
                raise ValueError(
                    f"unsupported collision wall horizontal axis: {horizontal_axis}"
                )

    # Every façade bay uses the same openings in the visual and collision LODs.
    # The front adds a centred door and leaves its ground-floor bay window-free.
    front_windows = _window_openings(
        -half_width,
        half_width,
        key.height_m,
        ground_exclusions=((-door_half - 0.35, door_half + 0.35),),
    )
    front_openings = front_windows + (
        (-door_half, door_half, 0.0, door_height),
    )
    back_openings = _window_openings(-half_width, half_width, key.height_m)
    side_openings = _window_openings(-half_length, half_length, key.height_m)
    add_wall(
        horizontal_min=-half_width, horizontal_max=half_width,
        plane_min=-half_length, plane_max=-half_length + wall_thickness,
        horizontal_axis="x", openings=front_openings,
    )
    add_wall(
        horizontal_min=-half_width, horizontal_max=half_width,
        plane_min=half_length - wall_thickness, plane_max=half_length,
        horizontal_axis="x", openings=back_openings,
    )
    add_wall(
        horizontal_min=-half_length + wall_thickness,
        horizontal_max=half_length - wall_thickness,
        plane_min=-half_width, plane_max=-half_width + wall_thickness,
        horizontal_axis="z", openings=side_openings,
    )
    add_wall(
        horizontal_min=-half_length + wall_thickness,
        horizontal_max=half_length - wall_thickness,
        plane_min=half_width - wall_thickness, plane_max=half_width,
        horizontal_axis="z", openings=side_openings,
    )

    # Floor and a low ground-floor ceiling complete the bounded shell.
    add_box(-half_width, -0.18, -half_length, half_width, 0.0, half_length)
    add_box(-half_width, ceiling, -half_length, half_width, min(key.height_m, ceiling + 0.18), half_length)

    # One partition makes two useful rooms without creating a heavy maze. Its
    # doorway is aligned with the exterior entrance for predictable navigation.
    if key.width_m >= 6.0 and key.length_m >= 7.0:
        partition_half = wall_thickness * 0.5
        interior_door_half = min(0.75, max(0.55, half_width * 0.16))
        add_box(-half_width + wall_thickness, 0.0, -partition_half, -interior_door_half, ceiling, partition_half)
        add_box(interior_door_half, 0.0, -partition_half, half_width - wall_thickness, ceiling, partition_half)
        add_box(-interior_door_half, door_height, -partition_half, interior_door_half, ceiling, partition_half)

    selections: list[_NamedSelection] = []
    for index, (point_range, face_range) in enumerate(component_ranges, start=1):
        point_weights = bytearray(len(points))
        face_flags = bytearray(len(faces))
        for point_index in point_range:
            point_weights[point_index] = 1
        for face_index in face_range:
            face_flags[face_index] = 1
        selections.append(_NamedSelection(
            name=f"component{index:02d}",
            point_weights=bytes(point_weights),
            face_flags=bytes(face_flags),
        ))

    volume = key.width_m * key.length_m * max(1.0, _main_building_height(key))
    total_mass = max(1000.0, volume * 60.0)
    mass_per_point = tuple(total_mass / len(points) for _ in points)
    return _Lod(
        tuple(points),
        (),
        tuple(faces),
        _GEOMETRY_LOD,
        mass_per_point,
        tuple(selections),
        (("map", _map_symbol_for_family(key.family)), ("class", "house")),
    )


def _geometry_lod(key: BuildingVariantKey, church_plinth_height: float = 0.0) -> _Lod:
    """Build closed convex collision components with CWA component selections.

    Geometry LODs are split into bounded boxes so very large OSM footprints do
    not exceed the legacy engine's practical component-size limit.  Every box
    is a closed convex component and receives a consecutive ``componentXX``
    named selection, which is required for character collision.
    """

    if key.interiors:
        return _hollow_geometry_lod(key)

    x_segments = max(1, math.ceil(key.width_m / _MAX_GEOMETRY_COMPONENT_SPAN_M))
    z_segments = max(1, math.ceil(key.length_m / _MAX_GEOMETRY_COMPONENT_SPAN_M))
    x_min = -key.width_m / 2.0
    z_min = -key.length_m / 2.0
    x_step = key.width_m / x_segments
    z_step = key.length_m / z_segments

    points: list[tuple[float, float, float]] = []
    faces: list[_Face] = []
    component_ranges: list[tuple[range, range]] = []
    height = _main_building_height(key) + (max(0.0, float(church_plinth_height)) if key.family == "church" else 0.0)

    for z_index in range(z_segments):
        z0 = z_min + z_index * z_step
        z1 = z_min + (z_index + 1) * z_step
        for x_index in range(x_segments):
            x0 = x_min + x_index * x_step
            x1 = x_min + (x_index + 1) * x_step
            point_start = len(points)
            face_start = len(faces)
            points.extend((
                (x0, 0.0, z0), (x1, 0.0, z0),
                (x1, 0.0, z1), (x0, 0.0, z1),
                (x0, height, z0), (x1, height, z0),
                (x1, height, z1), (x0, height, z1),
            ))
            faces.extend(
                _Face("", tuple((point_start + point, -1, 0.0, 0.0) for point in indices))
                for indices in (
                    (0, 4, 5, 1),
                    (1, 5, 6, 2),
                    (2, 6, 7, 3),
                    (3, 7, 4, 0),
                    (4, 7, 6, 5),
                    (0, 1, 2, 3),
                )
            )
            component_ranges.append((
                range(point_start, point_start + 8),
                range(face_start, face_start + 6),
            ))

    selections: list[_NamedSelection] = []
    for index, (point_range, face_range) in enumerate(component_ranges, start=1):
        point_weights = bytearray(len(points))
        face_flags = bytearray(len(faces))
        for point_index in point_range:
            point_weights[point_index] = 1
        for face_index in face_range:
            face_flags[face_index] = 1
        selections.append(_NamedSelection(
            name=f"component{index:02d}",
            point_weights=bytes(point_weights),
            face_flags=bytes(face_flags),
        ))

    volume = key.width_m * key.length_m * max(1.0, key.height_m)
    total_mass = max(1000.0, volume * 60.0)
    mass_per_point = tuple(total_mass / len(points) for _ in points)
    geometry_properties = [("map", _map_symbol_for_family(key.family))]
    if key.family == "church":
        # Churches have a tower/spire far above the nave while their object
        # origin, land-contact LOD and visible wall base all remain at Y=0.
        # Explicitly disable automatic model centering so CWA uses that authored
        # origin instead of deriving a centre from the unusually tall church
        # silhouette. Ordinary houses keep their historical model metadata.
        geometry_properties.append(("autocenter", "0"))
    return _Lod(
        tuple(points),
        (),
        tuple(faces),
        _GEOMETRY_LOD,
        mass_per_point,
        tuple(selections),
        tuple(geometry_properties),
    )


def _write_tag(stream, name: str, payload: bytes) -> None:
    stream.write(_TAG_HEADER.pack(_cstring(name, 64, "MLOD tag name"), len(payload)))
    stream.write(payload)


def _write_lod(stream, lod: _Lod) -> None:
    stream.write(_SP3X_HEADER.pack(b"SP3X", 28, 1, len(lod.points), len(lod.normals), len(lod.faces), 0))
    for point in lod.points:
        stream.write(_POINT.pack(*point, 0))
    for normal in lod.normals:
        stream.write(_NORMAL.pack(*normal))
    for face in lod.faces:
        if len(face.vertices) not in {3, 4}:
            raise ValueError("MLOD faces must contain three or four vertices")
        stream.write(_cstring(face.texture, 32, "P3D texture path"))
        stream.write(struct.pack("<i", len(face.vertices)))
        padded = face.vertices + tuple((-1, -1, 0.0, 0.0) for _ in range(4 - len(face.vertices)))
        for vertex in padded:
            stream.write(_FACE_VERTEX.pack(*vertex))
        stream.write(_FACE_TRAILER.pack(face.flags))
    stream.write(b"TAGG")
    for selection in lod.selections:
        if len(selection.point_weights) != len(lod.points):
            raise ValueError(f"selection {selection.name!r} has the wrong point count")
        if len(selection.face_flags) != len(lod.faces):
            raise ValueError(f"selection {selection.name!r} has the wrong face count")
        _write_tag(stream, selection.name, selection.point_weights + selection.face_flags)
    for name, value in lod.properties:
        payload = _cstring(name.casefold(), 64, "MLOD property name") + _cstring(
            value.casefold(), 64, "MLOD property value"
        )
        _write_tag(stream, "#Property#", payload)
    if lod.mass_per_point:
        if len(lod.mass_per_point) != len(lod.points):
            raise ValueError("MLOD mass table must contain one value per point")
        _write_tag(stream, "#Mass#", b"".join(_FLOAT.pack(value) for value in lod.mass_per_point))
    _write_tag(stream, "#EndOfFile#", b"")
    stream.write(_FLOAT.pack(lod.resolution))


def write_building_mlod(
    path: Path,
    key: BuildingVariantKey,
    *,
    wall_texture: str,
    roof_texture: str,
    roof_pitch_degrees: float = 35.0,
    front_texture: str | None = None,
    foundation_texture: str | None = None,
    foundation_depth: float = 0.0,
    church_plinth_height: float = 0.0,
    interior_texture: str | None = None,
    window_trim_texture: str | None = None,
    plain_wall_texture: str | None = None,
) -> None:
    lods = (
        _visual_lod(
            key, wall_texture, roof_texture, roof_pitch_degrees, front_texture,
            foundation_texture, foundation_depth, church_plinth_height,
            interior_texture, window_trim_texture, plain_wall_texture,
        ),
        _geometry_lod(key, church_plinth_height),
        _land_contact_lod(key),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, len(lods)))
        for lod in lods:
            _write_lod(stream, lod)


def inspect_mlod(path: Path) -> MlodSummary:
    data = path.read_bytes()
    offset = 0
    if len(data) < _MLOD_HEADER.size:
        raise ValueError("truncated MLOD header")
    signature, major, minor, padding, lod_count = _MLOD_HEADER.unpack_from(data, offset)
    offset += _MLOD_HEADER.size
    if signature != b"MLOD" or major < 1 or padding != 0 or not 1 <= lod_count <= 100:
        raise ValueError("invalid MLOD header")
    resolutions: list[float] = []
    textures: set[str] = set()
    total_points = total_normals = total_faces = 0
    point_counts: list[int] = []
    face_counts: list[int] = []
    selection_names: list[tuple[str, ...]] = []
    mass_point_counts: list[int] = []
    named_properties: list[tuple[tuple[str, str], ...]] = []
    for _ in range(lod_count):
        if offset + _SP3X_HEADER.size > len(data):
            raise ValueError("truncated SP3X header")
        sp3x, head_size, version, points, normals, faces, flags = _SP3X_HEADER.unpack_from(data, offset)
        if sp3x != b"SP3X" or head_size < _SP3X_HEADER.size or min(points, normals, faces) < 0:
            raise ValueError("invalid SP3X header")
        offset += head_size
        total_points += points
        total_normals += normals
        total_faces += faces
        point_counts.append(points)
        face_counts.append(faces)
        offset += points * _POINT.size + normals * _NORMAL.size
        for _face in range(faces):
            if offset + 32 + 4 + 4 * _FACE_VERTEX.size + _FACE_TRAILER.size > len(data):
                raise ValueError("truncated MLOD face")
            texture = data[offset:offset + 32].split(b"\0", 1)[0]
            if texture:
                textures.add(texture.decode("ascii"))
            offset += 32
            vertex_count = struct.unpack_from("<i", data, offset)[0]
            if vertex_count not in {3, 4}:
                raise ValueError("invalid MLOD face vertex count")
            offset += 4 + 4 * _FACE_VERTEX.size + _FACE_TRAILER.size
        if data[offset:offset + 4] != b"TAGG":
            raise ValueError("MLOD LOD is missing TAGG section")
        offset += 4
        lod_selections: list[str] = []
        lod_properties: list[tuple[str, str]] = []
        lod_mass_points = 0
        while True:
            if offset + _TAG_HEADER.size > len(data):
                raise ValueError("truncated MLOD TAGG entry")
            raw_name, size = _TAG_HEADER.unpack_from(data, offset)
            offset += _TAG_HEADER.size
            if size < 0 or offset + size > len(data):
                raise ValueError("invalid MLOD TAGG size")
            name = raw_name.split(b"\0", 1)[0].decode("ascii")
            payload = data[offset:offset + size]
            offset += size
            if name == "#EndOfFile#":
                break
            if not name.startswith(("#", "-", ".")):
                if size != points + faces:
                    raise ValueError(f"invalid named selection size for {name!r}")
                lod_selections.append(name)
            elif name == "#Property#":
                if size != 128:
                    raise ValueError("invalid MLOD property size")
                property_name = payload[:64].split(b"\0", 1)[0].decode("ascii")
                property_value = payload[64:].split(b"\0", 1)[0].decode("ascii")
                if not property_name:
                    raise ValueError("empty MLOD property name")
                lod_properties.append((property_name, property_value))
            elif name == "#Mass#":
                if size != points * _FLOAT.size:
                    raise ValueError("invalid MLOD mass table size")
                if any(not math.isfinite(value[0]) or value[0] < 0.0 for value in struct.iter_unpack("<f", payload)):
                    raise ValueError("invalid MLOD mass value")
                lod_mass_points = points
        if offset + 4 > len(data):
            raise ValueError("truncated MLOD resolution")
        resolutions.append(_FLOAT.unpack_from(data, offset)[0])
        selection_names.append(tuple(lod_selections))
        mass_point_counts.append(lod_mass_points)
        named_properties.append(tuple(lod_properties))
        offset += 4
    if offset != len(data):
        raise ValueError("unexpected trailing MLOD bytes")
    return MlodSummary(
        major,
        minor,
        lod_count,
        tuple(resolutions),
        total_points,
        total_normals,
        total_faces,
        tuple(sorted(textures)),
        tuple(point_counts),
        tuple(face_counts),
        tuple(selection_names),
        tuple(mass_point_counts),
        tuple(named_properties),
    )


class ProceduralBuildingLibrary:
    def __init__(
        self,
        *,
        world_name: str,
        width_quantum: float = 2.0,
        length_quantum: float = 2.0,
        height_quantum: float = 3.0,
        minimum_width: float = 4.0,
        maximum_width: float = 80.0,
        minimum_length: float = 4.0,
        maximum_length: float = 160.0,
        minimum_height: float = 3.0,
        maximum_height: float = 48.0,
        default_level_height: float = 3.0,
        maximum_variants: int = 128,
        roof_pitch_degrees: float = 35.0,
        foundation_depth: float = 0.5,
        maximum_foundation_depth: float = 2.5,
        foundation_depth_quantum: float = 0.25,
        church_plinth_height: float = 0.0,
        generate_interiors: bool = False,
        high_quality_textures: bool = False,
        texture_variants: int = DEFAULT_BUILDING_TEXTURE_VARIANTS,
        cache_dir: Path | None = None,
        cache_enabled: bool = True,
        cache_refresh: bool = False,
    ) -> None:
        self.world_name = world_name
        self.width_quantum = width_quantum
        self.length_quantum = length_quantum
        self.height_quantum = height_quantum
        self.minimum_width = minimum_width
        self.maximum_width = maximum_width
        self.minimum_length = minimum_length
        self.maximum_length = maximum_length
        self.minimum_height = minimum_height
        self.maximum_height = maximum_height
        self.default_level_height = default_level_height
        self.maximum_variants = maximum_variants
        self.roof_pitch_degrees = roof_pitch_degrees
        self.foundation_depth = foundation_depth
        self.maximum_foundation_depth = max(foundation_depth, maximum_foundation_depth)
        self.foundation_depth_quantum = max(0.05, foundation_depth_quantum)
        self.church_plinth_height = max(0.0, float(church_plinth_height))
        self.generate_interiors = bool(generate_interiors)
        self.high_quality_textures = bool(high_quality_textures)
        self.texture_size = (
            HIGH_QUALITY_BUILDING_ASSET_TEXTURE_SIZE
            if self.high_quality_textures else BUILDING_ASSET_TEXTURE_SIZE
        )
        self.texture_variants = max(1, min(10, int(texture_variants)))
        self.cache_dir = cache_dir
        self.cache_enabled = cache_enabled
        self.cache_refresh = cache_refresh
        self.cache_hits = 0
        self.cache_misses = 0
        self._request_counts: Counter[BuildingVariantKey] = Counter()
        self._mapping: dict[BuildingVariantKey, BuildingVariantKey] = {}
        self._usage: Counter[BuildingVariantKey] = Counter()
        self._prepared = False
        self.region_identifier: str | None = None
        self._urban_polygons: tuple[Polygon, ...] = ()
        self._settlement_points: tuple[tuple[float, float, float, str], ...] = ()
        self._isolated_dwelling_areas: tuple[Polygon, ...] = ()
        self._settlement_scale_x = 1.0
        self._settlement_scale_z = 1.0

    def _prepare_geographic_context(self, dataset: Any, projection: Any) -> None:
        tag_sources = [feature.tags for feature in getattr(dataset, "places", ())]
        tag_sources.extend(feature.tags for feature in getattr(dataset, "building_polygons", ()))
        profile = detect_region(
            (projection.south, projection.west, projection.north, projection.east),
            tag_sources,
        )
        self.region_identifier = profile.identifier if profile is not None else None

        # Residential, commercial, and industrial land-use polygons are not
        # reliable evidence that a footprint belongs to a town. They occur in
        # villages, isolated subdivisions, and mapped farm compounds as well.
        # Town architecture is therefore enabled only near explicit place nodes.
        self._urban_polygons = ()

        self._settlement_scale_x = max(1.0e-9, float(projection.scale_x))
        self._settlement_scale_z = max(1.0e-9, float(projection.scale_z))
        # Keep the one-kilometre town/city rule introduced in 0.9.81, but
        # retain village/hamlet context as a weaker architectural hint.  A
        # village never enables townhouse/apartment promotion; it simply keeps
        # ambiguous footprints in the residential family instead of letting a
        # large rectangle become a guessed farm building.
        radii = {
            "city": 1000.0, "town": 1000.0, "village": 1000.0,
            "hamlet": 1000.0,
        }

        # Preserve exact isolated-dwelling membership. A mapped
        # place=isolated_dwelling polygon defines the property/settlement area;
        # using a radius around its centroid can easily capture a neighbouring
        # property. Only a building whose representative point is inside a
        # qualifying polygon can receive the lone-cabin promotion.
        building_geometries: list[tuple[Polygon, Mapping[str, str]]] = []
        for building_feature in getattr(dataset, "building_polygons", ()):
            for geo_polygon in building_feature.polygons:
                outer = [projection.to_world(point) for point in geo_polygon.outer]
                holes = [
                    [projection.to_world(point) for point in hole]
                    for hole in geo_polygon.holes
                ]
                if len(outer) >= 4:
                    geometry = Polygon(outer, holes)
                    if not geometry.is_empty:
                        building_geometries.append((geometry, building_feature.tags))

        isolated_dwelling_cabins: list[Polygon] = []
        for place_feature in getattr(dataset, "place_areas", ()):
            if str(place_feature.tags.get("place", "")).casefold() != "isolated_dwelling":
                continue
            for geo_polygon in place_feature.polygons:
                outer = [projection.to_world(point) for point in geo_polygon.outer]
                holes = [
                    [projection.to_world(point) for point in hole]
                    for hole in geo_polygon.holes
                ]
                if len(outer) < 4:
                    continue
                area = Polygon(outer, holes)
                if area.is_empty:
                    continue
                inside: list[tuple[Polygon, str]] = []
                for building_geometry, building_tags in building_geometries:
                    if not area.covers(building_geometry.representative_point()):
                        continue
                    inside.append((
                        building_geometry,
                        str(building_tags.get("building", "")).casefold(),
                    ))
                plausible = [
                    geometry for geometry, building_kind in inside
                    if building_kind in {"", "yes", "house", "residential", "detached", "cabin"}
                ]
                if len(plausible) == 1:
                    # One plausible dwelling may coexist with explicit garages or
                    # sheds; only the selected footprint receives cabin context.
                    isolated_dwelling_cabins.append(plausible[0])
                elif len(plausible) == 0 and len(inside) == 1 and inside[0][1] == "shed":
                    # Sparse OSM sometimes labels the only actual dwelling as a
                    # shed. When it is literally the sole building inside the
                    # mapped isolated-dwelling polygon, prefer the area semantic.
                    isolated_dwelling_cabins.append(inside[0][0])
        self._isolated_dwelling_cabins = tuple(isolated_dwelling_cabins)

        settlement_points: list[tuple[float, float, float, str]] = []
        for feature in getattr(dataset, "places", ()):
            kind = str(feature.tags.get("place", "")).casefold()
            radius = radii.get(kind)
            if radius is None:
                continue
            x, z = projection.to_world(feature.point)
            settlement_points.append((x, z, radius, kind))
        self._settlement_points = tuple(settlement_points)

    def _settlement_context(self, x: float | None, z: float | None) -> str:
        if x is None or z is None:
            return "rural"
        matches: list[tuple[int, float, str]] = []
        priority = {"city": 0, "town": 1, "village": 2, "hamlet": 3}
        point = Point(float(x), float(z))
        # Exact polygon membership is authoritative. 0.9.205 added the cabin
        # footprint to the match list with lower priority than a nearby hamlet,
        # so real isolated-dwelling buildings such as ways 788104416/420 were
        # still classified as tiny outbuildings. A building selected by the
        # containing place=isolated_dwelling polygon must win over any radius-
        # based settlement hint.
        if any(footprint.covers(point) for footprint in self._isolated_dwelling_cabins):
            return "isolated_dwelling_single"
        for centre_x, centre_z, radius, kind in self._settlement_points:
            distance = math.hypot(
                (float(x) - centre_x) / self._settlement_scale_x,
                (float(z) - centre_z) / self._settlement_scale_z,
            )
            if distance <= radius:
                matches.append((priority.get(kind, 99), distance, kind))
        if not matches:
            return "rural"
        # A city/town within the allowed kilometre outranks a nearby village,
        # otherwise choose the closest settlement of the strongest class.
        return min(matches)[2]

    def _foundation_depth(
        self,
        value: float | None = None,
        *,
        allow_above_configured_maximum: bool = False,
    ) -> float:
        requested = self.foundation_depth if value is None else float(value)
        if allow_above_configured_maximum:
            requested = max(self.foundation_depth, requested)
        else:
            requested = max(
                self.foundation_depth,
                min(self.maximum_foundation_depth, requested),
            )
        steps = math.ceil((requested - 1e-9) / self.foundation_depth_quantum)
        quantized = max(self.foundation_depth, steps * self.foundation_depth_quantum)
        return (
            quantized
            if allow_above_configured_maximum
            else min(self.maximum_foundation_depth, quantized)
        )

    def key_for(
        self,
        tags: Mapping[str, str],
        width_m: float,
        length_m: float,
        *,
        foundation_depth_m: float | None = None,
        settlement_context: str = "rural",
    ) -> BuildingVariantKey:
        family = _family(
            tags, width_m, length_m, settlement_context=settlement_context
        )
        width, length = sorted((max(0.1, width_m), max(0.1, length_m)))
        regional_style = _regional_style(
            self.region_identifier, family, tags, width, length
        )
        roof = _roof_style(tags, family, regional_style)
        requested_height = _height(tags, family, self.default_level_height)
        if (
            settlement_context == "isolated_dwelling_single"
            and family == "residential"
            and _parse_number(tags.get("height")) is None
            and _parse_number(tags.get("building:levels")) is None
        ):
            # One-storey cabin rather than a six-metre generic house. Explicit
            # OSM height/levels still win when present.
            requested_height = self.default_level_height
        quantized_height = _quantize(
            requested_height,
            self.height_quantum,
            self.minimum_height,
            self.maximum_height,
        )
        interiors = (
            self.generate_interiors
            and family in INTERIOR_ELIGIBLE_FAMILIES
            and width <= INTERIOR_MAXIMUM_WIDTH_M
            and length <= INTERIOR_MAXIMUM_LENGTH_M
            and quantized_height <= INTERIOR_MAXIMUM_HEIGHT_M
        )
        return BuildingVariantKey(
            family=family,
            roof_style=roof,
            width_m=_quantize(width, self.width_quantum, self.minimum_width, self.maximum_width),
            length_m=_quantize(length, self.length_quantum, self.minimum_length, self.maximum_length),
            height_m=quantized_height,
            foundation_depth_m=self._foundation_depth(foundation_depth_m),
            regional_style=regional_style,
            interiors=interiors,
        )

    def _iter_dataset_keys(self, dataset: Any, projection: Any, point_footprint: float) -> Iterable[BuildingVariantKey]:
        for feature in dataset.building_polygons:
            for polygon in feature.polygons:
                projected = [projection.to_world(point) for point in polygon.outer[:-1]]
                if len(projected) >= 3:
                    footprint = footprint_from_polygon(projected)
                    centre_x = sum(point[0] for point in projected) / len(projected)
                    centre_z = sum(point[1] for point in projected) / len(projected)
                    yield self.key_for(
                        feature.tags, footprint.width_m, footprint.length_m,
                        settlement_context=self._settlement_context(centre_x, centre_z),
                    )
        for feature in dataset.building_points:
            x, z = projection.to_world(feature.point)
            yield self.key_for(
                feature.tags, point_footprint, point_footprint,
                settlement_context=self._settlement_context(x, z),
            )

    @staticmethod
    def _variant_within_fit_envelope(
        requested: BuildingVariantKey, candidate: BuildingVariantKey
    ) -> bool:
        rw, rl = sorted((max(0.1, requested.width_m), max(0.1, requested.length_m)))
        cw, cl = sorted((max(0.1, candidate.width_m), max(0.1, candidate.length_m)))
        area_ratio = (cw * cl) / max(0.01, rw * rl)
        return (
            rw * BUILDING_REUSE_MIN_DIMENSION_RATIO <= cw <= rw * BUILDING_REUSE_MAX_DIMENSION_RATIO
            and rl * BUILDING_REUSE_MIN_DIMENSION_RATIO <= cl <= rl * BUILDING_REUSE_MAX_DIMENSION_RATIO
            and BUILDING_REUSE_MIN_AREA_RATIO <= area_ratio <= BUILDING_REUSE_MAX_AREA_RATIO
        )

    @staticmethod
    def _variant_fit_score(requested: BuildingVariantKey, candidate: BuildingVariantKey) -> tuple[float, ...]:
        """Rank a reusable model primarily by physical footprint fit.

        Width/length are canonicalized, so a 90-degree rotation is implicit.
        Since 0.9.92, footprint fit is deliberately stricter and carries more
        weight than roof or regional cosmetics when the variant cap forces reuse.
        """

        rw, rl = sorted((max(0.1, requested.width_m), max(0.1, requested.length_m)))
        cw, cl = sorted((max(0.1, candidate.width_m), max(0.1, candidate.length_m)))
        requested_area = rw * rl
        candidate_area = cw * cl
        requested_aspect = rl / rw
        candidate_aspect = cl / cw

        within_fit_envelope = ProceduralBuildingLibrary._variant_within_fit_envelope(
            requested, candidate
        )
        dimension_error = abs(cw - rw) / rw + abs(cl - rl) / rl
        area_error = abs(math.log(candidate_area / requested_area))
        aspect_error = abs(math.log(candidate_aspect / requested_aspect))
        height_error = abs(candidate.height_m - requested.height_m) / max(3.0, requested.height_m)
        return (
            0.0 if within_fit_envelope else 1.0,
            dimension_error * 0.50 + area_error * 0.30 + aspect_error * 0.15 + height_error * 0.05,
            dimension_error,
            area_error,
            aspect_error,
            height_error,
        )

    @staticmethod
    def _compatible_families(family: str) -> tuple[str, ...]:
        # If the variant cap forces reuse, stay within the nearest architectural
        # class rather than mapping a townhouse to a church or warehouse.
        return {
            "residential": ("residential", "townhouse", "urban"),
            "townhouse": ("townhouse", "urban", "residential"),
            "urban": ("urban", "townhouse", "residential"),
            "agricultural": ("agricultural", "outbuilding", "industrial", "residential"),
            "outbuilding": ("outbuilding", "agricultural", "industrial"),
            "industrial": ("industrial", "agricultural", "urban"),
        }.get(family, (family,))

    def _reuse_candidates(
        self, requested: BuildingVariantKey, candidates: Sequence[BuildingVariantKey]
    ) -> list[BuildingVariantKey]:
        """Prefer a strict physical fit before preserving roof/palette details."""

        same_mode = [
            candidate for candidate in candidates
            if candidate.interiors == requested.interiors
        ]
        if same_mode:
            candidates = same_mode
        compatible = set(self._compatible_families(requested.family))
        # Churches, schools and shops carry explicit semantics. Even if the
        # variant cap cannot reserve an exact physical fit, reuse a model from
        # the same semantic family rather than silently turning a school wing
        # into a Swedish barn, which the supplied 0.9.205 build demonstrated.
        if requested.family in {"church", "school", "shop"}:
            same_family = [
                candidate for candidate in candidates
                if candidate.family == requested.family
            ]
            if same_family:
                strict = [
                    candidate for candidate in same_family
                    if self._variant_within_fit_envelope(requested, candidate)
                ]
                return strict or same_family
        tiers = (
            [candidate for candidate in candidates
             if candidate.family == requested.family
             and candidate.roof_style == requested.roof_style
             and candidate.regional_style == requested.regional_style],
            [candidate for candidate in candidates
             if candidate.family == requested.family
             and candidate.regional_style == requested.regional_style],
            [candidate for candidate in candidates if candidate.family == requested.family],
            [candidate for candidate in candidates if candidate.family in compatible],
        )
        # Preserve architectural family before considering an unrelated exact
        # physical fit. 0.9.205's final all-candidates strict tier is why a
        # correctly requested 3 m isolated cabin was still rendered with the
        # exact-size outbuilding model. The same mechanism also helped school
        # footprints drift into barns. Search only compatible families first;
        # use an unrelated model solely when the selected variant set contains
        # no compatible family at all.
        for tier in tiers:
            strict = [
                candidate for candidate in tier
                if self._variant_within_fit_envelope(requested, candidate)
            ]
            if strict:
                return strict
        for tier in tiers:
            if tier:
                return tier
        return list(candidates)

    def _best_variant(
        self, requested: BuildingVariantKey, candidates: Sequence[BuildingVariantKey]
    ) -> BuildingVariantKey:
        if not candidates:
            raise ValueError("building variant selection requires candidates")
        compatible = self._compatible_families(requested.family)
        return min(
            candidates,
            key=lambda candidate: (
                compatible.index(candidate.family) if candidate.family in compatible else 99,
                self._variant_fit_score(requested, candidate),
                candidate.regional_style != requested.regional_style,
                candidate.roof_style != requested.roof_style,
                candidate,
            ),
        )

    def prepare(self, dataset: Any, projection: Any, point_footprint: float) -> None:
        self._mapping.clear()
        self._usage.clear()
        self._prepared = False
        self._prepare_geographic_context(dataset, projection)
        self._request_counts = Counter(self._iter_dataset_keys(dataset, projection, point_footprint))
        ordered = sorted(self._request_counts, key=lambda key: (-self._request_counts[key], key))

        # Frequency-only selection can use every variant slot on common houses,
        # leaving a church, school, or shop mapped onto an unrelated residential
        # model. Keep the most common baseline variant, then reserve one model for
        # each present semantic family before frequency fills the remaining slots.
        representatives: dict[str, BuildingVariantKey] = {}
        for key in ordered:
            representatives.setdefault(key.family, key)
        selected: list[BuildingVariantKey] = ordered[:1]
        # Semantic landmark/campus families must never be squeezed down to one
        # tiny representative and then reused as a barn/house when the variant
        # cap is reached. Reserve a useful spread of their requested dimensions
        # before general frequency filling. Twelve per semantic family is far
        # above normal map demand but still bounded on pathological datasets.
        protected_families = ("church", "school", "shop")
        for family in protected_families:
            family_keys = [key for key in ordered if key.family == family]
            for key in family_keys[:12]:
                if key in selected:
                    continue
                if len(selected) >= self.maximum_variants:
                    break
                selected.append(key)
        # Exact isolated-dwelling promotion requests a one-storey residential
        # key. Reserve those compact 3 m variants too; otherwise the family is
        # preserved but the nearest selected house can still be a 6 m two-storey
        # model, defeating the cabin intent visible in-game.
        cabin_keys = [
            key for key in ordered
            if key.family == "residential"
            and key.height_m <= self.default_level_height + 1e-6
        ]
        for key in cabin_keys[:12]:
            if key in selected:
                continue
            if len(selected) >= self.maximum_variants:
                break
            selected.append(key)
        for family in ("outbuilding", "townhouse", "urban", "agricultural", "industrial"):
            representative = representatives.get(family)
            if representative is None or representative in selected:
                continue
            if len(selected) >= self.maximum_variants:
                break
            selected.append(representative)
        for key in ordered:
            if len(selected) >= self.maximum_variants:
                break
            if key not in selected:
                selected.append(key)
        if not selected and ordered:
            selected = [ordered[0]]
        for requested in ordered:
            if requested in selected:
                self._mapping[requested] = requested
                continue
            candidates = self._reuse_candidates(requested, selected)
            self._mapping[requested] = self._best_variant(requested, candidates)
        self._prepared = True

    def model_path(self, key: BuildingVariantKey) -> str:
        return rf"{self.world_name}\g\b_{key.digest}.p3d"

    def is_generated_model(self, path: str) -> bool:
        return path.casefold().startswith((self.world_name + r"\g\b_").casefold()) and path.casefold().endswith(".p3d")

    def _selected(self, requested: BuildingVariantKey) -> BuildingVariantKey:
        if not self._prepared:
            raise RuntimeError("procedural building library must be prepared before placement")
        if requested in self._mapping:
            return self._mapping[requested]
        if not self._mapping:
            self._mapping[requested] = requested
            return requested
        candidates = self._reuse_candidates(
            requested, sorted(set(self._mapping.values()))
        )
        return self._best_variant(requested, candidates)

    def plan_polygon(
        self, tags: Mapping[str, str], points: Sequence[PointXZ], *, road_point: PointXZ | None = None
    ) -> BuildingPlacement:
        footprint = footprint_from_polygon(points)
        centre_x = sum(point[0] for point in points) / len(points)
        centre_z = sum(point[1] for point in points) / len(points)
        requested = self.key_for(
            tags, footprint.width_m, footprint.length_m,
            settlement_context=self._settlement_context(centre_x, centre_z),
        )
        selected = replace(
            self._selected(requested),
            texture_variant=_placement_texture_variant(
                tags, points, variant_count=self.texture_variants
            ),
        )
        heading = footprint.heading_degrees
        if road_point is not None:
            centre_x = sum(point[0] for point in points) / len(points)
            centre_z = sum(point[1] for point in points) / len(points)
            road_vector = (road_point[0] - centre_x, road_point[1] - centre_z)
            angle = math.radians(heading)
            front_vector = (-math.sin(angle), -math.cos(angle))
            if front_vector[0] * road_vector[0] + front_vector[1] * road_vector[1] < 0.0:
                heading = (heading + 180.0) % 360.0
        return BuildingPlacement(self.model_path(selected), heading, requested, selected)

    def plan_point(
        self, tags: Mapping[str, str], footprint: float, heading_degrees: float,
        *, x: float | None = None, z: float | None = None,
    ) -> BuildingPlacement:
        requested = self.key_for(
            tags, footprint, footprint,
            settlement_context=self._settlement_context(x, z),
        )
        coordinates = ((float(x), float(z)),) if x is not None and z is not None else ((float(footprint), float(heading_degrees)),)
        selected = replace(
            self._selected(requested),
            texture_variant=_placement_texture_variant(
                tags, coordinates, variant_count=self.texture_variants
            ),
        )
        return BuildingPlacement(self.model_path(selected), heading_degrees, requested, selected)

    def register_placement(
        self, placement: BuildingPlacement, *, foundation_depth_m: float | None = None
    ) -> BuildingPlacement:
        selected = replace(
            placement.selected,
            foundation_depth_m=self._foundation_depth(
                foundation_depth_m,
                allow_above_configured_maximum=foundation_depth_m is not None,
            ),
        )
        self._usage[selected] += 1
        return BuildingPlacement(self.model_path(selected), placement.heading_degrees, placement.requested, selected)

    def place_polygon(
        self, tags: Mapping[str, str], points: Sequence[PointXZ], *, road_point: PointXZ | None = None
    ) -> BuildingPlacement:
        return self.register_placement(self.plan_polygon(tags, points, road_point=road_point))

    def place_point(
        self, tags: Mapping[str, str], footprint: float, heading_degrees: float,
        *, x: float | None = None, z: float | None = None,
    ) -> BuildingPlacement:
        return self.register_placement(
            self.plan_point(tags, footprint, heading_degrees, x=x, z=z)
        )

    @staticmethod
    def _family_texture_index(family: str) -> int:
        return {
            "residential": 0, "urban": 1, "industrial": 2, "church": 3,
            "school": 4, "shop": 5, "agricultural": 6, "townhouse": 7, "outbuilding": 8,
        }[family]

    @staticmethod
    def _regional_style_index(regional_style: str) -> int:
        return {
            "default": 0, "sweden_red": 1, "sweden_yellow": 2,
            "eastern_plaster": 3, "eastern_brick": 4,
            "eastern_whitewash": 5, "eastern_panel": 6,
            # A world has one detected region, so Western Europe safely reuses
            # the four regional atlas slots without expanding the two-character
            # OFP texture-name budget.
            "western_stucco": 3, "western_brick": 4,
            "western_stone": 5, "western_half_timber": 6,
            "africa_earth": 7, "africa_whitewash": 8,
            "africa_block": 9, "africa_colour": 10,
            "middle_east_sandstone": 11, "middle_east_whitewash": 12,
            "middle_east_adobe": 13, "middle_east_concrete": 14,
        }.get(regional_style, 15)

    @staticmethod
    def _base36_code(value: int) -> str:
        """Encode a compact two- or three-character building texture id.

        Adding the outbuilding family pushed the palette above the old 36^2
        ceiling for styles such as Sweden red when all ten texture variants are
        emitted.  A third base36 digit still keeps a maximum-length 20-character
        world name inside OFP/CWA's 31-byte P3D texture-path field.
        """
        alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        value = max(0, int(value))
        if value >= len(alphabet) ** 3:
            raise ValueError("building texture code exceeds three base36 digits")
        if value < len(alphabet) ** 2:
            return alphabet[value // len(alphabet)] + alphabet[value % len(alphabet)]
        return (
            alphabet[value // (len(alphabet) ** 2)]
            + alphabet[(value // len(alphabet)) % len(alphabet)]
            + alphabet[value % len(alphabet)]
        )

    def _palette_texture_code(
        self, family: str, regional_style: str, texture_variant: int
    ) -> str:
        family_index = self._family_texture_index(family)
        style_index = self._regional_style_index(regional_style)
        value = (family_index * 16 + style_index) * self.texture_variants
        value += _normalise_texture_variant(texture_variant, self.texture_variants)
        return self._base36_code(value)

    def _wall_texture(
        self, family: str, regional_style: str = "default", texture_variant: int = 0
    ) -> str:
        code = self._palette_texture_code(family, regional_style, texture_variant)
        return rf"{self.world_name}\d\w{code}.paa"

    def _open_wall_texture(
        self, family: str, regional_style: str = "default", texture_variant: int = 0
    ) -> str:
        code = self._palette_texture_code(family, regional_style, texture_variant)
        return rf"{self.world_name}\d\o{code}.paa"

    def _interior_wall_texture(
        self, family: str, regional_style: str = "default", texture_variant: int = 0
    ) -> str:
        code = self._palette_texture_code(family, regional_style, texture_variant)
        return rf"{self.world_name}\d\i{code}.paa"

    def _white_window_trim_texture(self) -> str:
        return rf"{self.world_name}\d\t.paa"

    def _front_texture(
        self, family: str, regional_style: str = "default", texture_variant: int = 0
    ) -> str:
        code = self._palette_texture_code(family, regional_style, texture_variant)
        return rf"{self.world_name}\d\e{code}.paa"

    def _roof_texture(self, roof_style: str, texture_variant: int = 0) -> str:
        roof_index = {"flat": 0, "gabled": 1}[roof_style]
        value = roof_index * self.texture_variants
        value += _normalise_texture_variant(texture_variant, self.texture_variants)
        return rf"{self.world_name}\d\r{self._base36_code(value)}.paa"

    def _foundation_texture(self) -> str:
        return rf"{self.world_name}\d\f.paa"

    def write_assets(self, source_dir: Path, catalogue_path: Path) -> BuildingGenerationResult:
        selected = sorted(self._usage)
        model_assets: list[GeneratedBuildingAsset] = []
        texture_files: list[str] = []
        used_palettes = sorted({(key.family, key.regional_style) for key in selected})
        used_open_palettes = sorted({
            (key.family, key.regional_style) for key in selected
            if key.interiors or key.family == "church"
        })
        used_interior_palettes = sorted({
            (key.family, key.regional_style) for key in selected if key.interiors
        })
        uses_white_window_trim = any(
            key.interiors
            and key.regional_style in WHITE_WINDOW_TRIM_REGIONAL_STYLES
            for key in selected
        )
        used_roofs = sorted({key.roof_style for key in selected})
        if any(key.foundation_depth_m > 0.0 for key in selected):
            texture_path = self._foundation_texture()
            relative = texture_path.split("\\", 1)[1].replace("\\", "/")
            file_path = source_dir / relative
            key = cache_key("procedural-building-foundation-v4-selectable-quality", {"texture": "stone", "texture_size": self.texture_size})
            cached = self.cache_dir / "procedural-assets" / f"{key}.paa" if self.cache_dir else None
            hit = restore_or_create_file(
                cache_path=cached, destination=file_path,
                producer=lambda target: write_rgb_dxt1_paa(target, _foundation_texture_image(self.texture_size)),
                enabled=self.cache_enabled, refresh=self.cache_refresh,
            )
            self.cache_hits += int(hit); self.cache_misses += int(not hit)
            inspect_paa(file_path); texture_files.append(relative)
        if uses_white_window_trim:
            texture_path = self._white_window_trim_texture()
            relative = texture_path.split("\\", 1)[1].replace("\\", "/")
            file_path = source_dir / relative
            key = cache_key(
                "procedural-building-white-window-trim-v4-selectable-quality",
                {"material": "weathered-white-painted-wood", "texture_size": self.texture_size},
            )
            cached = self.cache_dir / "procedural-assets" / f"{key}.paa" if self.cache_dir else None
            hit = restore_or_create_file(
                cache_path=cached,
                destination=file_path,
                producer=lambda target: write_rgb_dxt1_paa(
                    target, _white_trim_texture_image(self.texture_size)
                ),
                enabled=self.cache_enabled,
                refresh=self.cache_refresh,
            )
            self.cache_hits += int(hit)
            self.cache_misses += int(not hit)
            inspect_paa(file_path)
            texture_files.append(relative)
        for family, regional_style in used_palettes:
            for texture_variant in range(self.texture_variants):
                texture_path = self._wall_texture(
                    family, regional_style, texture_variant
                )
                relative = texture_path.split("\\", 1)[1].replace("\\", "/")
                file_path = source_dir / relative
                key = cache_key(
                    "procedural-building-wall-v9-selectable-quality",
                    {
                        "family": family,
                        "regional_style": regional_style,
                        "texture_variant": texture_variant,
                        "texture_size": self.texture_size,
                    },
                )
                cached = self.cache_dir / "procedural-assets" / f"{key}.paa" if self.cache_dir else None
                hit = restore_or_create_file(
                    cache_path=cached,
                    destination=file_path,
                    producer=lambda target, family=family, regional_style=regional_style, texture_variant=texture_variant: write_rgb_dxt1_paa(
                        target,
                        _wall_texture_image(
                            family,
                            size=self.texture_size,
                            regional_style=regional_style,
                            texture_variant=texture_variant,
                        ),
                    ),
                    enabled=self.cache_enabled, refresh=self.cache_refresh,
                )
                self.cache_hits += int(hit)
                self.cache_misses += int(not hit)
                inspect_paa(file_path)
                texture_files.append(relative)
        for family, regional_style in used_open_palettes:
            for texture_variant in range(self.texture_variants):
                texture_path = self._open_wall_texture(
                    family, regional_style, texture_variant
                )
                relative = texture_path.split("\\", 1)[1].replace("\\", "/")
                file_path = source_dir / relative
                key = cache_key(
                    "procedural-building-open-wall-v3-selectable-quality",
                    {
                        "family": family,
                        "regional_style": regional_style,
                        "texture_variant": texture_variant,
                        "texture_size": self.texture_size,
                    },
                )
                cached = self.cache_dir / "procedural-assets" / f"{key}.paa" if self.cache_dir else None
                hit = restore_or_create_file(
                    cache_path=cached,
                    destination=file_path,
                    producer=lambda target, family=family, regional_style=regional_style, texture_variant=texture_variant: write_rgb_dxt1_paa(
                        target,
                        _open_wall_texture_image(
                            family,
                            size=self.texture_size,
                            regional_style=regional_style,
                            texture_variant=texture_variant,
                        ),
                    ),
                    enabled=self.cache_enabled,
                    refresh=self.cache_refresh,
                )
                self.cache_hits += int(hit)
                self.cache_misses += int(not hit)
                inspect_paa(file_path)
                texture_files.append(relative)
        for family, regional_style in used_interior_palettes:
            for texture_variant in range(self.texture_variants):
                texture_path = self._interior_wall_texture(
                    family, regional_style, texture_variant
                )
                relative = texture_path.split("\\", 1)[1].replace("\\", "/")
                file_path = source_dir / relative
                key = cache_key(
                    "procedural-building-interior-wall-v3-selectable-quality",
                    {
                        "family": family,
                        "regional_style": regional_style,
                        "texture_variant": texture_variant,
                        "texture_size": self.texture_size,
                    },
                )
                cached = self.cache_dir / "procedural-assets" / f"{key}.paa" if self.cache_dir else None
                hit = restore_or_create_file(
                    cache_path=cached,
                    destination=file_path,
                    producer=lambda target, family=family, regional_style=regional_style, texture_variant=texture_variant: write_rgb_dxt1_paa(
                        target,
                        _interior_wall_texture_image(
                            family,
                            size=self.texture_size,
                            regional_style=regional_style,
                            texture_variant=texture_variant,
                        ),
                    ),
                    enabled=self.cache_enabled,
                    refresh=self.cache_refresh,
                )
                self.cache_hits += int(hit)
                self.cache_misses += int(not hit)
                inspect_paa(file_path)
                texture_files.append(relative)
        for family, regional_style in used_palettes:
            for texture_variant in range(self.texture_variants):
                texture_path = self._front_texture(
                    family, regional_style, texture_variant
                )
                relative = texture_path.split("\\", 1)[1].replace("\\", "/")
                file_path = source_dir / relative
                key = cache_key(
                    "procedural-building-front-v9-selectable-quality",
                    {
                        "family": family,
                        "regional_style": regional_style,
                        "texture_variant": texture_variant,
                        "texture_size": self.texture_size,
                    },
                )
                cached = self.cache_dir / "procedural-assets" / f"{key}.paa" if self.cache_dir else None
                hit = restore_or_create_file(
                    cache_path=cached, destination=file_path,
                    producer=lambda target, family=family, regional_style=regional_style, texture_variant=texture_variant: write_rgb_dxt1_paa(
                        target,
                        _front_texture_image(
                            family,
                            size=self.texture_size,
                            regional_style=regional_style,
                            texture_variant=texture_variant,
                        ),
                    ),
                    enabled=self.cache_enabled, refresh=self.cache_refresh,
                )
                self.cache_hits += int(hit); self.cache_misses += int(not hit)
                inspect_paa(file_path); texture_files.append(relative)

        for roof in used_roofs:
            for texture_variant in range(self.texture_variants):
                texture_path = self._roof_texture(roof, texture_variant)
                relative = texture_path.split("\\", 1)[1].replace("\\", "/")
                file_path = source_dir / relative
                key = cache_key(
                    "procedural-building-roof-v5-selectable-quality",
                    {"roof": roof, "texture_variant": texture_variant, "texture_size": self.texture_size},
                )
                cached = self.cache_dir / "procedural-assets" / f"{key}.paa" if self.cache_dir else None
                hit = restore_or_create_file(
                    cache_path=cached,
                    destination=file_path,
                    producer=lambda target, roof=roof, texture_variant=texture_variant: write_rgb_dxt1_paa(
                        target, _roof_texture_image(
                            roof, size=self.texture_size,
                            texture_variant=texture_variant
                        )
                    ),
                    enabled=self.cache_enabled, refresh=self.cache_refresh,
                )
                self.cache_hits += int(hit)
                self.cache_misses += int(not hit)
                inspect_paa(file_path)
                texture_files.append(relative)

        for key in selected:
            model_path = self.model_path(key)
            relative = model_path.split("\\", 1)[1].replace("\\", "/")
            file_path = source_dir / relative
            asset_key = cache_key(
                "procedural-building-model-v30-church-tower-window-band",
                {
                    "world_name": self.world_name,
                    "variant": asdict(key),
                    "roof_pitch_degrees": self.roof_pitch_degrees,
                    "foundation_depth": key.foundation_depth_m,
                    "church_plinth_height": self.church_plinth_height,
                },
            )
            cached = self.cache_dir / "procedural-assets" / f"{asset_key}.p3d" if self.cache_dir else None
            hit = restore_or_create_file(
                cache_path=cached,
                destination=file_path,
                producer=lambda target, key=key: write_building_mlod(
                    target,
                    key,
                    wall_texture=(
                        self._open_wall_texture(
                            key.family, key.regional_style, key.texture_variant
                        )
                        if key.interiors else
                        self._wall_texture(
                            key.family, key.regional_style, key.texture_variant
                        )
                    ),
                    front_texture=self._front_texture(
                        key.family, key.regional_style, key.texture_variant
                    ),
                    roof_texture=self._roof_texture(
                        key.roof_style, key.texture_variant
                    ),
                    roof_pitch_degrees=self.roof_pitch_degrees,
                    foundation_texture=self._foundation_texture() if key.foundation_depth_m > 0.0 or key.family == "church" else None,
                    foundation_depth=key.foundation_depth_m,
                    church_plinth_height=self.church_plinth_height,
                    interior_texture=(
                        self._interior_wall_texture(
                            key.family, key.regional_style, key.texture_variant
                        )
                        if key.interiors else None
                    ),
                    window_trim_texture=(
                        self._white_window_trim_texture()
                        if key.interiors
                        and key.regional_style in WHITE_WINDOW_TRIM_REGIONAL_STYLES
                        else None
                    ),
                    plain_wall_texture=(
                        self._open_wall_texture(
                            key.family, key.regional_style, key.texture_variant
                        )
                        if key.family == "church" else None
                    ),
                ),
                enabled=self.cache_enabled,
                refresh=self.cache_refresh,
            )
            self.cache_hits += int(hit)
            self.cache_misses += int(not hit)
            summary = inspect_mlod(file_path)
            model_assets.append(GeneratedBuildingAsset(
                key=key,
                model_path=model_path,
                relative_path=relative,
                usage_count=self._usage[key],
                sha256=sha256(file_path.read_bytes()).hexdigest(),
                lod_count=summary.lod_count,
                point_count=summary.point_count,
                face_count=summary.face_count,
                texture_paths=summary.texture_paths,
                visual_face_count=summary.face_counts[0],
                geometry_component_count=len(summary.selection_names[1]),
                geometry_mass_point_count=summary.mass_point_counts[1],
                map_symbol=dict(summary.named_properties[1])["map"],
            ))

        placements = sum(self._usage.values())
        generated = len(model_assets)
        document = {
            "schema": 9,
            "generator": "cwr-worldgen procedural MLOD building library",
            "region": self.region_identifier or "default",
            "settings": {
                "width_quantum_m": self.width_quantum,
                "length_quantum_m": self.length_quantum,
                "height_quantum_m": self.height_quantum,
                "maximum_variants": self.maximum_variants,
                "texture_variants_per_style": self.texture_variants,
                "texture_size_px": self.texture_size,
                "high_quality_textures": self.high_quality_textures,
                "texture_variant_selection": "deterministic-building-tags-and-position",
                "roof_pitch_degrees": self.roof_pitch_degrees,
                "minimum_foundation_depth_m": self.foundation_depth,
                "maximum_foundation_depth_m": self.maximum_foundation_depth,
                "enterable_foundation_limit_m": self.maximum_foundation_depth,
                "foundation_depth_quantum_m": self.foundation_depth_quantum,
                "foundation_texture": self._foundation_texture() if selected else None,
                "foundation_strategy": "per-placement-depth-with-non-enterable-door-fallback",
                "church_plinth_height_m": self.church_plinth_height,
                "procedural_interiors_enabled": self.generate_interiors,
                "white_window_trim_styles": sorted(WHITE_WINDOW_TRIM_REGIONAL_STYLES),
                "white_window_trim_texture": (
                    self._white_window_trim_texture()
                    if uses_white_window_trim else None
                ),
                "interior_eligible_families": sorted(INTERIOR_ELIGIBLE_FAMILIES),
                "interior_maximum_dimensions_m": [
                    INTERIOR_MAXIMUM_WIDTH_M,
                    INTERIOR_MAXIMUM_LENGTH_M,
                    INTERIOR_MAXIMUM_HEIGHT_M,
                ],
                "interior_layout": "enterable-ground-floor-two-room-when-space-allows",
                "double_sided_visual_shell": True,
                "maximum_geometry_component_span_m": _MAX_GEOMETRY_COMPONENT_SPAN_M,
                "grounding_footprint": "selected_model_oriented_rectangle",
                "grounding_margin_m": 0.5,
                "settlement_building_context": {
                    "town_city_radius_m": 1000.0,
                    "village_hamlet_context_radius_m": 1000.0,
                    "residential_landuse_implies_town": False,
                    "village_enables_town_buildings": False,
                },
                "variant_reuse_fit": {
                    "dimension_fit_range": [BUILDING_REUSE_MIN_DIMENSION_RATIO, BUILDING_REUSE_MAX_DIMENSION_RATIO],
                    "area_fit_range": [BUILDING_REUSE_MIN_AREA_RATIO, BUILDING_REUSE_MAX_AREA_RATIO],
                    "score_weights": {"dimensions": 0.50, "area": 0.30, "aspect": 0.15, "height": 0.05},
                    "fit_precedes_roof_and_palette": True,
                    "orientation": "best-90-degree-fit-via-canonical-width-length",
                },
                "map_symbols": {
                    "residential": "house",
                    "townhouse": "house",
                    "urban": "building",
                    "industrial": "building",
                    "agricultural": "building",
                    "church": "building",
                    "school": "building",
                    "shop": "building",
                },
            },
            "placements": placements,
            "unique_requested_variants": len(self._request_counts),
            "generated_variants": generated,
            "reused_placements": max(0, placements - generated),
            "reuse_ratio": round((placements - generated) / placements, 6) if placements else 0.0,
            "capped_variants": max(0, len(self._request_counts) - generated),
            "textures": sorted(texture_files),
            "request_mapping": [
                {
                    "requested": asdict(requested),
                    "selected": asdict(selected_key),
                    "request_count": self._request_counts[requested],
                }
                for requested, selected_key in sorted(self._mapping.items())
            ],
            "models": [asdict(asset) for asset in model_assets],
        }
        canonical = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        document["catalogue_sha256"] = digest
        catalogue_path.parent.mkdir(parents=True, exist_ok=True)
        catalogue_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        embedded_catalogue = source_dir / "g" / "buildings.json"
        embedded_catalogue.parent.mkdir(parents=True, exist_ok=True)
        embedded_catalogue.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        return BuildingGenerationResult(
            enabled=True,
            placements=placements,
            unique_requested_variants=len(self._request_counts),
            generated_variants=generated,
            reused_placements=max(0, placements - generated),
            reuse_ratio=round((placements - generated) / placements, 6) if placements else 0.0,
            capped_variants=max(0, len(self._request_counts) - generated),
            model_assets=tuple(model_assets),
            texture_files=tuple(sorted(texture_files)),
            catalogue_sha256=digest,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
        )
