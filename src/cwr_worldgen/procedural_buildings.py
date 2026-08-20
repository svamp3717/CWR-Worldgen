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
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import triangulate

try:
    # Shapely 2.1+ exposes GEOS constrained Delaunay triangulation. Ordinary
    # Delaunay triangulation does not preserve concave polygon edges, so merely
    # discarding triangles outside a footprint can leave valid OSM buildings
    # with uncovered roof slivers. Keep this optional so Shapely 2.0 installs
    # retain the deterministic ear-clipping fallback below.
    from shapely import constrained_delaunay_triangles as _constrained_delaunay_triangles
except ImportError:  # pragma: no cover - exercised only on older Shapely 2.0
    _constrained_delaunay_triangles = None

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
_ROADWAY_LOD = 3.0e15
_MEMORY_LOD = 1.0e15
_PATHS_LOD = 4.0e15
INTERIOR_DISTANCE_LOD_RESOLUTION = 20.0
_MAX_GEOMETRY_COMPONENT_SPAN_M = 40.0
DEFAULT_BUILDING_TEXTURE_VARIANTS = 10
POLYGON_NATIVE_RECTANGULAR_FILL_THRESHOLD = 0.90
POLYGON_NATIVE_SIMPLIFY_TOLERANCE_M = 0.50
POLYGON_NATIVE_VERTEX_QUANTUM_M = 0.50
POLYGON_NATIVE_MAXIMUM_VERTICES = 32
POLYGON_NATIVE_MAXIMUM_VARIANTS = 2048
POLYGON_NATIVE_ENTRANCE_FRACTION_QUANTUM = 0.01
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
    "industrial", "agricultural", "outbuilding",
})
UTILITY_INTERIOR_FAMILIES = frozenset({
    "industrial", "agricultural", "outbuilding",
})
SECOND_STOREY_INTERIOR_FAMILIES = frozenset({
    "residential", "townhouse", "urban",
})
HOUSE_ROAD_FACING_FAMILIES = SECOND_STOREY_INTERIOR_FAMILIES
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
# Utility buildings use a much cheaper single-hall interior, so ordinary barns,
# sheds/garages, and warehouses can remain enterable at substantially larger
# footprint sizes without applying the house-style window/partition cost.
INTERIOR_FAMILY_MAXIMUM_DIMENSIONS_M = {
    "industrial": (80.0, 160.0, 24.0),
    "agricultural": (80.0, 160.0, 18.0),
    "outbuilding": (40.0, 60.0, 9.0),
}
INTERIOR_DOOR_HEIGHT_M = 2.2
INTERIOR_WINDOW_SILL_M = 0.9
INTERIOR_WINDOW_TOP_M = 2.05
INTERIOR_WINDOW_CROSS_BAR_WIDTH_M = 0.09
INTERIOR_DOOR_THICKNESS_M = 0.08
INTERIOR_PATH_HALF_WIDTH_M = 0.32
INTERIOR_ROADWAY_Y_M = 0.06
INTERIOR_ROADWAY_WALL_CLEARANCE_M = 0.12
INTERIOR_ROADWAY_TILE_SPAN_M = 20.0
INTERIOR_VISUAL_FLOOR_Y_M = 0.025
INTERIOR_COLLISION_DOOR_SIDE_CLEARANCE_M = 0.08
INTERIOR_COLLISION_DOOR_TOP_CLEARANCE_M = 0.18
INTERIOR_MODEL_TEXTURE_VARIANTS = 3
INTERIOR_FOUNDATION_DEPTH_QUANTUM_M = 0.5
INTERIOR_STAIR_TARGET_RISE_M = 0.18
INTERIOR_STAIR_TREAD_M = 0.30
INTERIOR_STAIR_SIDE_MARGIN_M = 0.35
INTERIOR_STAIR_MAXIMUM_STEPS = 64
# Vehicle-scale entrances should meet terrain with a ramp rather than a little
# flight of domestic stairs. A 4:1 run is gentle enough for infantry/vehicles in
# CWA while remaining compact around ordinary 0.5-1.0 m foundation skirts.
INTERIOR_VEHICLE_RAMP_RUN_PER_RISE = 4.0
INTERIOR_VEHICLE_RAMP_MINIMUM_RUN_M = 1.6
# The visible second-floor steps remain lightweight visual geometry. Matching
# horizontal Roadway treads sit over a slightly lower solid stepped Geometry
# staircase so the old infantry collider has no sloped or empty surface to fall
# through.
INTERIOR_SECOND_STOREY_MINIMUM_WIDTH_M = 6.0
INTERIOR_SECOND_STOREY_MINIMUM_LENGTH_M = 8.0
INTERIOR_SECOND_STOREY_MINIMUM_HEIGHT_M = 5.6
INTERIOR_SECOND_STOREY_FLOOR_Y_M = 2.55
INTERIOR_SECOND_STOREY_MINIMUM_HEADROOM_M = 2.20
INTERIOR_SECOND_STOREY_CEILING_THICKNESS_M = 0.10
INTERIOR_SECOND_STOREY_STAIR_MINIMUM_RUN_M = 3.0
INTERIOR_SECOND_STOREY_STAIR_MAXIMUM_RUN_M = 4.5
INTERIOR_SECOND_STOREY_STAIR_MINIMUM_WIDTH_M = 0.90
INTERIOR_SECOND_STOREY_STAIR_MAXIMUM_WIDTH_M = 1.15
INTERIOR_SECOND_STOREY_STAIR_STEPS = 16
# Once the facade system has decided that a house visibly has two storeys, an
# enterable model that physically fits the upper floor should match that facade.
# Older builds used a 75% lottery here, which produced two-storey exteriors with
# a blank upper facade whenever the interior happened to be selected as one level.
INTERIOR_SECOND_STOREY_DEFAULT_SHARE_PERCENT = 100
# Keep a small visible plinth above the local model origin. Without a reveal,
# correctly grounded buildings can still *look* as if the wall material simply
# stops in mid-air on uneven terrain, especially once the world placement lifts
# the shell by a small safety clearance.
FOUNDATION_VISIBLE_REVEAL_M = 0.10
# Short polygon-native wall runs should not squeeze an atlas window down until
# it resembles a mail slot. Switch those runs to a plain wall treatment.
MIN_NATIVE_EDGE_WINDOW_TEXTURE_SPAN_M = 3.25
# Closed painted-window facades with very shallow upper bands or very narrow
# spans produce absurdly tiny second-floor windows. In those cases fall back to
# the matching plain facade instead of squeezing the atlas into nonsense.
CLOSED_WINDOW_TEXTURE_MIN_SPAN_M = 6.25
CLOSED_WINDOW_TEXTURE_MIN_UPPER_BAND_HEIGHT_M = 2.55
# Upper-storey painted windows need more room than a full-height ground-floor
# band. Otherwise the repeated atlas degenerates into tiny second-floor slots,
# exactly the artifact visible on distant red houses in testing.
CLOSED_WINDOW_TEXTURE_MIN_UPPER_BAND_SPAN_M = 7.50
# Isolated cabins/dwellings are often legitimately narrower than suburban
# houses. One normal ground-floor window still reads correctly at this span;
# below it the side/back wall becomes plain, while the selected front keeps its
# entrance atlas regardless.
ISOLATED_DWELLING_WINDOW_TEXTURE_MIN_SPAN_M = 3.25
ISOLATED_DWELLING_MINIMUM_FACADE_HEIGHT_M = 2.30
VISIBLE_FACADE_STOREY_HEIGHT_M = 3.0
MINIMUM_VISIBLE_FACADE_STOREY_HEIGHT_M = 2.55
FACADE_WINDOW_UV_INSET = 1.0 / 256.0
# Almost-square footprints can safely rotate freely to face a nearby road
# without materially changing the occupied footprint. Longer rectangles keep
# their source-aligned long axis and may only flip front/back.
HOUSE_ROAD_FACING_FREE_ROTATION_ASPECT_RATIO = 1.08
# Outbuildings use footprint size to decide whether they are a vehicle garage or
# a pedestrian shed.  The dimensions are deliberately a little larger than a
# typical passenger car so the *inside* of the procedural shell can plausibly
# contain one after wall thickness/door jambs are accounted for.
OUTBUILDING_GARAGE_MINIMUM_WIDTH_M = 2.4
OUTBUILDING_GARAGE_MINIMUM_LENGTH_M = 4.8
FACADE_TILE_HEIGHT_M = 3.0
PAINTED_WINDOW_MINIMUM_SILL_M = 1.15
_PAINTED_WINDOW_FAMILIES = frozenset({
    "residential", "townhouse", "urban", "church", "school",
})
# These atlases contain artwork that only makes sense at ground level. In
# particular, agricultural wall textures paint a full-size barn door into the
# 3 m facade tile. Repeating that atlas above the first storey creates a row of
# impossible second-floor barn doors, so upper wall bands must switch to the
# matching windowless/open-wall material.
_GROUND_FLOOR_ONLY_FACADE_FAMILIES = frozenset({"agricultural"})
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
    second_storey: bool = False
    outbuilding_kind: str = ""
    # Quantized local X/Z outline for one polygon-native exterior shell. Empty
    # means the long-standing rectangular procedural model. Keeping the shape
    # in the immutable variant key makes model reuse/cache identity deterministic.
    footprint_vertices: tuple[PointXZ, ...] = ()
    # Interior rings are authored courtyard/open-air holes.  They live in the
    # same local coordinate system as ``footprint_vertices`` and are deliberately
    # part of the cache key so two otherwise identical shells cannot accidentally
    # share a roof/floor that fills the wrong courtyard.
    footprint_holes: tuple[tuple[PointXZ, ...], ...] = ()
    # Polygon-native facades have no universal "front" axis.  Store the actual
    # exterior edge selected by a mapped entrance (or nearest road fallback) and
    # a quantized lateral position along that edge.  Closed models can therefore
    # paint the entrance where OSM put it instead of centring it by decree.
    entrance_edge: int = -1
    entrance_fraction: float = 0.5
    # Number of window-bearing facade storeys. Zero means derive a conservative
    # count from the model height for legacy/manual keys. Carrying this in the
    # immutable model key prevents the renderer from inventing a third row of
    # windows merely because a wall happens to be tall enough to repeat an
    # atlas again.
    facade_storeys: int = 0
    # True when the footprint belongs to an explicit place=isolated_dwelling
    # context. These small one-storey homes need a guaranteed entrance facade;
    # generic narrow-wall anti-window heuristics must not erase the only door.
    isolated_dwelling: bool = False

    def canonical(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return sha256(self.canonical().encode("ascii")).hexdigest()[:12]


@dataclass(frozen=True, slots=True)
class _SecondStoreyLayout:
    floor_y: float
    stair_x0: float
    stair_x1: float
    stair_z0: float
    stair_z1: float
    opening_x0: float
    opening_x1: float
    opening_z0: float
    opening_z1: float

    @property
    def stair_center_x(self) -> float:
        return (self.stair_x0 + self.stair_x1) * 0.5


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
    "shop": (184, 179, 164),
}
_ROOF_COLOURS: dict[str, tuple[int, int, int]] = {
    "flat": (76, 78, 73),
    "gabled": (104, 61, 46),
    "hipped": (104, 61, 46),
    "pyramidal": (104, 61, 46),
    "dome": (94, 86, 71),
    "onion": (91, 84, 69),
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


def _placement_hash_u32(
    tags: Mapping[str, str], coordinates: Sequence[PointXZ]
) -> int:
    """Return the stable per-placement hash used by visual variation rules.

    Planning previously serialized and hashed the same tags/coordinates twice for
    most houses: once for second-storey selection and again for texture choice.
    Large worlds turn that innocent duplication into millions of JSON encodes.
    """

    document = {
        "tags": sorted((str(key), str(value)) for key, value in tags.items()),
        "coordinates": [
            [round(float(x), 3), round(float(z), 3)] for x, z in coordinates
        ],
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    digest = sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _placement_texture_variant(
    tags: Mapping[str, str],
    coordinates: Sequence[PointXZ],
    *,
    variant_count: int = DEFAULT_BUILDING_TEXTURE_VARIANTS,
    placement_hash: int | None = None,
) -> int:
    """Select a stable façade variant from the building's tags and position."""

    variant_count = max(1, int(variant_count))
    if placement_hash is None:
        placement_hash = _placement_hash_u32(tags, coordinates)
    return int(placement_hash) % variant_count


def _placement_uses_second_storey(
    tags: Mapping[str, str],
    coordinates: Sequence[PointXZ],
    key: "BuildingVariantKey",
    *,
    placement_hash: int | None = None,
) -> bool:
    """Choose whether an eligible house receives an upper interior floor.

    Explicit source levels remain authoritative. Inferred two-storey facades now
    receive a matching upper interior whenever the footprint can support one;
    visual and walkable storey counts should not disagree by random chance.
    """

    levels = _parse_number(tags.get("building:levels"))
    if levels is not None:
        # Explicit OSM levels are authoritative in both directions. A mapped
        # one-storey house must never fall through to the old 75% random upper
        # floor selector merely because the shell is tall enough to hold one.
        if levels < 2.0:
            return False
        return _supports_second_storey(key)
    if _facade_storey_count(key, _main_building_height(key)) < 2:
        return False
    if not _supports_second_storey(key):
        return False
    if key.second_storey:
        return True
    return (
        _placement_texture_variant(
            tags, coordinates, variant_count=100, placement_hash=placement_hash
        ) < INTERIOR_SECOND_STOREY_DEFAULT_SHARE_PERCENT
    )


def _front_vector_for_heading(heading_degrees: float) -> PointXZ:
    angle = math.radians(heading_degrees)
    return (-math.sin(angle), -math.cos(angle))


def _heading_directly_towards_vector(dx: float, dz: float) -> float:
    """Return a model heading whose local front (-Z) points at ``(dx, dz)``."""

    return math.degrees(math.atan2(-dx, -dz)) % 360.0


def _house_heading_towards_road(
    heading_degrees: float,
    *,
    centre_x: float,
    centre_z: float,
    road_point: PointXZ | None,
    width_m: float,
    length_m: float,
) -> float:
    """Face a house entrance toward the closest road when footprint fit allows.

    Long rectangular footprints preserve their source/model long-axis alignment
    and only flip 180 degrees. Near-square footprints can rotate freely because
    doing so does not meaningfully change their occupied footprint.
    """

    if road_point is None:
        return heading_degrees % 360.0
    dx = float(road_point[0]) - float(centre_x)
    dz = float(road_point[1]) - float(centre_z)
    if math.hypot(dx, dz) <= 1.0e-6:
        return heading_degrees % 360.0

    short = max(0.1, min(width_m, length_m))
    long = max(0.1, max(width_m, length_m))
    if long / short <= HOUSE_ROAD_FACING_FREE_ROTATION_ASPECT_RATIO:
        return _heading_directly_towards_vector(dx, dz)

    heading = heading_degrees % 360.0
    front_x, front_z = _front_vector_for_heading(heading)
    if front_x * dx + front_z * dz < 0.0:
        heading = (heading + 180.0) % 360.0
    return heading


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
        # Sheds and garages use plain windowless cladding. Their size-selected
        # pedestrian/vehicle door is painted only on the front atlas so side/back
        # walls stay believable.
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
        door, trim = _barn_door_colours(regional_style, texture_variant)
        _draw_barn_door(draw, (9, 15, 55, 58), door, trim)
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
        door, trim = _barn_door_colours(regional_style, texture_variant)
        _draw_barn_door(draw, (10, 17, 54, 57), door, trim)
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
        # Intact small-town storefront.  The old atlas used one enormous nearly
        # black rectangle across most of the facade, which reads as a blasted-
        # out wall at CWA distances.  Keep the glazing suitably dark for the
        # engine, but break it into framed display windows, a glazed timber door,
        # a proper sign fascia, and a clean rendered surround.
        draw.rectangle((0, 0, w, h), fill=base)
        for y in (14, 31, 48):
            draw.line((0, y, w, y), fill=_shade(base, -6), width=1)
        render_trim = _shade(base, 14)
        frame = (172, 164, 143)
        glass = (67, 82, 81)
        glass_lit = (83, 99, 96)
        fascia_options = (
            (111, 72, 52), (79, 91, 72), (76, 82, 96), (126, 91, 57),
        )
        fascia = fascia_options[texture_variant % len(fascia_options)]
        draw.rectangle((2, 3, 62, 12), fill=_shade(fascia, -8), outline=frame, width=1)
        draw.rectangle((4, 5, 60, 10), fill=fascia)
        # A few pale blocks suggest painted shop lettering without baking a
        # language-specific name into every generated storefront.
        for x0, x1 in ((9, 17), (21, 31), (35, 44), (48, 56)):
            draw.rectangle((x0, 7, x1, 8), fill=_shade(frame, 24))

        def storefront_window(box: tuple[int, int, int, int]) -> None:
            x0, y0, x1, y1 = box
            draw.rectangle(box, fill=_shade(frame, -34), outline=render_trim, width=2)
            draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), fill=glass, outline=frame, width=1)
            midx = (x0 + x1) // 2
            transom = y0 + 8
            draw.line((midx, y0 + 2, midx, y1 - 2), fill=_shade(frame, -8), width=1)
            draw.line((x0 + 2, transom, x1 - 2, transom), fill=_shade(frame, -8), width=1)
            draw.line((x0 + 4, y0 + 4, midx - 2, transom - 2), fill=glass_lit, width=1)
            # Low display backing keeps the windows from reading as empty holes.
            draw.rectangle((x0 + 3, y1 - 7, x1 - 3, y1 - 3), fill=(111, 103, 86))

        storefront_window((4, 15, 25, 45))
        storefront_window((39, 15, 60, 45))

        door = (91, 72, 52)
        draw.rectangle((27, 14, 37, 51), fill=_shade(door, -22), outline=render_trim, width=2)
        draw.rectangle((29, 16, 35, 35), fill=glass, outline=frame, width=1)
        draw.line((32, 17, 32, 34), fill=_shade(frame, -8), width=1)
        draw.rectangle((29, 37, 35, 49), fill=door)
        draw.rectangle((34, 39, 35, 40), fill=(157, 128, 67))
        draw.rectangle((0, 53, w, h), fill=_shade(base, -24))
        draw.line((0, 51, w, 51), fill=_shade(base, -14), width=2)
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

    if family == "outbuilding":
        # Match the closed shed/garage cladding exactly, but omit the painted
        # front door because an enterable model carries a real animated panel.
        for x in range(0, width, 6):
            plank = _shade(
                base,
                6 if (x // 6) % 3 == 0 else (-5 if (x // 6) % 3 == 1 else 1),
            )
            draw.rectangle((x, 0, min(width, x + 5), height), fill=plank)
            draw.line((x, 0, x, height), fill=_shade(base, -22), width=1)
        draw.rectangle((0, 56, width, height), fill=(82, 78, 68))
        draw.line((0, 53, width, 53), fill=_shade(base, -15), width=2)
    elif regional_style == "sweden_red" and family == "agricultural":
        # Same Swedish barn boards/footing as the non-enterable wall atlas, only
        # without the baked-in double barn door. This makes the real-door and
        # painted-door variants read as the same building rather than cousins.
        for x in range(0, width, 6):
            plank = _shade(base, 6 if (x // 6) % 3 == 0 else -6)
            draw.rectangle((x, 0, min(width, x + 5), height), fill=plank)
            draw.line((x, 0, x, height), fill=_shade(base, -25), width=1)
        draw.rectangle((0, 57, width, height), fill=(82, 78, 69))
    elif family == "shop" and regional_style == "default":
        # Enterable shops use real geometry for doors/windows, so this matching
        # material intentionally contains no fake openings.  Keep the pale
        # render and restrained course lines from the closed storefront instead
        # of the old soot-dark surround.
        for y in (14, 31, 48):
            draw.line((0, y, width, y), fill=_shade(base, -6), width=1)
        draw.rectangle((0, 0, 3, 52), fill=_shade(base, 10))
        draw.rectangle((width - 4, 0, width - 1, 52), fill=_shade(base, 10))
    elif regional_style in {"sweden_red", "sweden_yellow"}:
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


def _barn_door_colours(
    regional_style: str,
    texture_variant: int = 0,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return door/trim colours shared by painted and real barn doors."""

    base = _variant_colour(
        _regional_wall_base("agricultural", regional_style),
        _normalise_texture_variant(texture_variant),
    )
    if regional_style == "sweden_red":
        return _shade(base, -18), (205, 197, 169)
    if regional_style == "sweden_yellow":
        return (105, 80, 56), (211, 204, 177)
    return (71, 61, 47), (48, 44, 37)


def _garage_door_colours(
    family: str,
    regional_style: str,
    texture_variant: int = 0,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return a recognisable sectional-door palette for garages/warehouses."""

    if regional_style in {"sweden_red", "sweden_yellow"}:
        door = _variant_colour((188, 187, 174), texture_variant)
        trim = (216, 208, 180)
    elif family == "industrial":
        door = _variant_colour((143, 146, 141), texture_variant)
        trim = (96, 99, 95)
    else:
        wall = _variant_colour(
            _regional_wall_base(family, regional_style), texture_variant
        )
        door = _shade(wall, 18)
        trim = _shade(wall, -24)
    return door, trim


def _draw_barn_door(
    draw: ImageDraw.ImageDraw | _ScaledImageDraw,
    box: tuple[int, int, int, int],
    door: tuple[int, int, int],
    trim: tuple[int, int, int],
) -> None:
    """Draw a double-leaf plank barn door with separate X braces per leaf."""

    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=door, outline=trim, width=2)
    inner_x0, inner_x1 = x0 + 2, x1 - 2
    inner_y0, inner_y1 = y0 + 2, y1 - 2
    for x in range(inner_x0 + 4, inner_x1, 5):
        draw.line((x, inner_y0, x, inner_y1), fill=_shade(door, -10), width=1)
    mid = (x0 + x1) // 2
    draw.line((mid, y0 + 1, mid, y1 - 1), fill=trim, width=2)
    draw.line((x0 + 2, (y0 + y1) // 2, x1 - 2, (y0 + y1) // 2), fill=trim, width=2)
    for leaf0, leaf1 in ((x0 + 3, mid - 2), (mid + 2, x1 - 3)):
        draw.line((leaf0, y0 + 4, leaf1, y1 - 4), fill=trim, width=2)
        draw.line((leaf1, y0 + 4, leaf0, y1 - 4), fill=trim, width=2)


def _draw_garage_door(
    draw: ImageDraw.ImageDraw | _ScaledImageDraw,
    box: tuple[int, int, int, int],
    door: tuple[int, int, int],
    trim: tuple[int, int, int],
) -> None:
    """Draw an overhead sectional garage/warehouse door, not barn planks."""

    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=door, outline=trim, width=2)
    height = max(1, y1 - y0)
    width = max(1, x1 - x0)
    # Strong horizontal panel breaks are the visual cue that was missing from
    # the old vertically-planked 'garage' frontage.
    for panel in range(1, 5):
        y = y0 + round(height * panel / 5)
        draw.line((x0 + 2, y, x1 - 2, y), fill=_shade(door, -15), width=2)
    for x in (x0 + width // 4, x0 + width // 2, x0 + (3 * width) // 4):
        draw.line((x, y0 + 2, x, y1 - 2), fill=_shade(door, -6), width=1)
    handle_y = y0 + round(height * 0.72)
    handle_x = (x0 + x1) // 2
    draw.rectangle((handle_x - 3, handle_y, handle_x + 3, handle_y + 1), fill=_shade(trim, -15))
    draw.line((x0 + 1, y1 - 2, x1 - 1, y1 - 2), fill=_shade(trim, -22), width=2)


def _door_texture_image(
    size: int = 128,
    family: str = "residential",
    regional_style: str = "default",
    texture_variant: int = 0,
    outbuilding_kind: str = "",
) -> Image.Image:
    """Return the animated entrance material appropriate to the building type."""

    texture_variant = _normalise_texture_variant(texture_variant)
    if family == "agricultural":
        door, trim = _barn_door_colours(regional_style, texture_variant)
        image, draw = _pixel_canvas(door, size, 2129 + texture_variant * 31, 3)
        _draw_barn_door(draw, (2, 2, 61, 61), door, trim)
        return _finish_pixel_texture(image, size)

    if family == "industrial" or (family == "outbuilding" and outbuilding_kind == "garage"):
        door, trim = _garage_door_colours(family, regional_style, texture_variant)
        image, draw = _pixel_canvas(door, size, 2177 + texture_variant * 37, 3)
        _draw_garage_door(draw, (2, 2, 61, 61), door, trim)
        return _finish_pixel_texture(image, size)

    if family == "outbuilding":
        base = _variant_colour((103, 78, 55), texture_variant)
        image, draw = _pixel_canvas(base, size, 2203 + texture_variant * 41, 3)
        frame = _shade(base, -22)
        draw.rectangle((3, 2, 60, 61), outline=frame, width=2)
        for x in range(10, 59, 8):
            draw.line((x, 4, x, 59), fill=_shade(base, -10), width=1)
        draw.line((5, 10, 58, 55), fill=frame, width=2)
        draw.ellipse((52, 34, 55, 37), fill=(96, 91, 77))
        return _finish_pixel_texture(image, size)

    base = (103, 78, 55)
    image, draw = _pixel_canvas(base, size, 2129, 3)
    width = height = _BUILDING_TEXTURE_LOGICAL_SIZE
    frame = _shade(base, -22)
    draw.rectangle((3, 2, width - 4, height - 3), outline=frame, width=2)
    for x in (16, 32, 48):
        draw.line((x, 4, x, height - 5), fill=_shade(base, -10), width=1)
    draw.line((4, 31, width - 5, 31), fill=frame, width=2)
    draw.ellipse((51, 29, 54, 32), fill=(96, 91, 77))
    return _finish_pixel_texture(image, size)

def _front_texture_image(
    family: str, size: int = 128, regional_style: str = "default",
    texture_variant: int = 0, outbuilding_kind: str = "",
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
        # Outbuilding frontage follows the same car-fit decision as the actual
        # enterable doorway. A true shed gets a small personnel door; a footprint
        # large enough for a car gets a broad garage gate.
        garage = outbuilding_kind == "garage"
        width = 40 if garage else 14
        x0 = (64 - width) // 2
        y0 = 18 if garage else 22
        door_bottom = 63
        if garage:
            door, trim = _garage_door_colours(
                family, regional_style, texture_variant
            )
            _draw_garage_door(
                draw, (x0, y0, x0 + width, door_bottom), door, trim
            )
        else:
            if regional_style in {"sweden_red", "sweden_yellow"}:
                door = (79, 70, 54)
                trim = (203, 196, 169)
            else:
                door = _shade(_regional_wall_base(family, regional_style), -22)
                trim = (151, 143, 123)
            door = _variant_colour(
                door, (texture_variant * 3) % DEFAULT_BUILDING_TEXTURE_VARIANTS
            )
            draw.rectangle((x0, y0, x0 + width, door_bottom), fill=door, outline=trim, width=2)
            draw.line((x0 + 2, y0 + 14, x0 + width - 2, y0 + 14), fill=_shade(door, -12), width=1)
            draw.ellipse((x0 + width - 4, y0 + 20, x0 + width - 2, y0 + 22), fill=(96, 91, 77))
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
    # Exact isolated-dwelling membership may promote only an *untyped/generic*
    # footprint that CWR would otherwise have to guess. Explicit OSM building
    # semantics stay authoritative: a mapped shed is a shed, a garage is a
    # garage, and a house remains a house rather than being silently relabelled
    # as a cabin because of surrounding settlement context.
    if settlement_context == "isolated_dwelling_single" and building in {
        "", "yes"
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
    if building in {"terrace", "terraced_house"}:
        return "townhouse"
    if building in {
        "house", "residential", "detached", "semidetached_house",
        "bungalow", "cabin"
    }:
        return "residential"

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


def _outbuilding_kind_from_dimensions(width_m: float, length_m: float) -> str:
    """Return ``garage`` only when an outbuilding can plausibly fit a car."""

    minor, major = sorted((max(0.1, float(width_m)), max(0.1, float(length_m))))
    if (
        minor >= OUTBUILDING_GARAGE_MINIMUM_WIDTH_M
        and major >= OUTBUILDING_GARAGE_MINIMUM_LENGTH_M
    ):
        return "garage"
    return "shed"


def _outbuilding_kind(
    tags: Mapping[str, str], width_m: float, length_m: float
) -> str:
    """Return the outbuilding subtype without overriding explicit OSM tags.

    Dimensions are useful only when CWR itself is inferring an accessory
    building from ``building=yes``/an untyped footprint.  They must not turn an
    explicit shed into a garage or vice versa.
    """

    building = str(tags.get("building", "")).casefold()
    if building in {"garage", "garages", "carport"}:
        return "garage"
    if building == "shed":
        return "shed"
    return _outbuilding_kind_from_dimensions(width_m, length_m)


def _outbuilding_is_garage(key: BuildingVariantKey) -> bool:
    if key.family != "outbuilding":
        return False
    if key.outbuilding_kind:
        return key.outbuilding_kind == "garage"
    # Backward-compatible fallback for hand-created/test keys that predate the
    # explicit subtype field.
    return _outbuilding_kind_from_dimensions(key.width_m, key.length_m) == "garage"


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
    if value in {"hipped", "half-hipped"}:
        return "hipped"
    if value in {"pyramidal", "pyramid"}:
        return "pyramidal"
    if value in {"dome", "domed"}:
        return "dome"
    if value in {"onion", "onion_dome"}:
        return "onion"
    if value in {"gabled", "gable", "gambrel", "mansard"}:
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
        # Generic town-context houses should not silently become three-storey
        # blocks. Three floors are still produced when OSM explicitly supplies
        # building:levels=3 or a matching height.
        "townhouse": 6.0,
        "urban": 12.0,
        "industrial": 7.0,
        "agricultural": 6.0,
        "outbuilding": 3.0,
        "church": 12.0,
        "school": 3.0,
        "shop": 3.0,
    }
    return defaults[family]


def _requested_facade_storeys(
    tags: Mapping[str, str],
    family: str,
    requested_height: float,
    level_height: float,
) -> int:
    """Return the intended number of visible window-bearing storeys.

    OSM ``building:levels`` is authoritative when present. Explicit height is
    converted conservatively. Untagged buildings use family defaults rather
    than deriving an unlimited number of floors from a tall wall shell.
    """

    levels = _parse_number(tags.get("building:levels"))
    if levels is not None and levels > 0.0:
        return max(1, min(12, int(round(levels))))

    explicit_height = _parse_number(tags.get("height"))
    if explicit_height is not None and explicit_height > 0.0:
        return max(
            1,
            min(
                12,
                int(math.floor((explicit_height + 0.15) / max(2.5, level_height))),
            ),
        )

    defaults = {
        "residential": 2,
        "townhouse": 2,
        "urban": 3,
        "industrial": 1,
        "agricultural": 1,
        "outbuilding": 1,
        "church": 1,
        "school": 1,
        "shop": 1,
    }
    return defaults.get(
        family,
        max(1, int(math.floor(requested_height / max(2.5, level_height)))),
    )


def _facade_storey_count(key: BuildingVariantKey, wall_height: float) -> int:
    """Return the number of full realistic window bands that fit this wall."""

    requested = int(key.facade_storeys)
    if requested <= 0:
        requested = max(
            1,
            int(
                math.floor(
                    (float(key.height_m) + 0.15) / VISIBLE_FACADE_STOREY_HEIGHT_M
                )
            ),
        )
        if key.family == "townhouse" and requested > 2:
            requested = 2
    maximum_that_fits = max(
        1,
        int(
            math.floor(
                (max(0.0, float(wall_height)) + 1.0e-6)
                / MINIMUM_VISIBLE_FACADE_STOREY_HEIGHT_M
            )
        ),
    )
    return max(1, min(requested, maximum_that_fits))


def _quantize(value: float, quantum: float, minimum: float, maximum: float) -> float:
    clamped = min(maximum, max(minimum, value))
    return round(round(clamped / quantum) * quantum, 3)


def _footprint_from_polygon_object(polygon: Polygon) -> _Footprint:
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


def _polygon_with_footprint(
    points: Sequence[PointXZ],
    holes: Sequence[Sequence[PointXZ]] = (),
) -> tuple[Polygon, _Footprint]:
    if len(points) < 3:
        raise ValueError("building footprint requires at least three points")
    polygon = Polygon(points, [tuple(hole) for hole in holes if len(hole) >= 3])
    return polygon, _footprint_from_polygon_object(polygon)


def footprint_from_polygon(points: Sequence[PointXZ]) -> _Footprint:
    """Fit the minimum rotated rectangle for one mapped building polygon."""

    return _polygon_with_footprint(points)[1]


def _polygon_signed_area(points: Sequence[PointXZ]) -> float:
    return 0.5 * sum(
        float(a[0]) * float(b[1]) - float(b[0]) * float(a[1])
        for a, b in zip(points, points[1:] + points[:1])
    )


def _canonical_cycle(points: Sequence[PointXZ]) -> tuple[PointXZ, ...]:
    """Return a stable cyclic representation without changing the geometry."""

    if not points:
        return ()
    values = tuple((round(float(x), 3), round(float(z), 3)) for x, z in points)
    rotations = [values[index:] + values[:index] for index in range(len(values))]
    return min(rotations)


def _compact_quantized_ring(
    points: Sequence[PointXZ], quantum: float, *, clockwise: bool,
) -> tuple[PointXZ, ...]:
    """Quantize one ring, remove collapsed corners, and canonicalize winding/start."""

    compact: list[PointXZ] = []
    for x, z in points:
        point = (round(float(x) / quantum) * quantum, round(float(z) / quantum) * quantum)
        if not compact or math.hypot(point[0] - compact[-1][0], point[1] - compact[-1][1]) > 1.0e-6:
            compact.append(point)
    if len(compact) >= 2 and math.hypot(
        compact[0][0] - compact[-1][0], compact[0][1] - compact[-1][1]
    ) <= 1.0e-6:
        compact.pop()
    if len(compact) < 3:
        return ()
    signed = _polygon_signed_area(compact)
    if (clockwise and signed > 0.0) or (not clockwise and signed < 0.0):
        compact.reverse()
    return _canonical_cycle(compact)


def _native_polygon_profile(
    points: Sequence[PointXZ],
    holes: Sequence[Sequence[PointXZ]] = (),
    *,
    footprint: _Footprint | None = None,
    polygon: Polygon | None = None,
    rectangular_fill_threshold: float = POLYGON_NATIVE_RECTANGULAR_FILL_THRESHOLD,
    simplify_tolerance: float = POLYGON_NATIVE_SIMPLIFY_TOLERANCE_M,
    vertex_quantum: float = POLYGON_NATIVE_VERTEX_QUANTUM_M,
    maximum_vertices: int = POLYGON_NATIVE_MAXIMUM_VERTICES,
) -> tuple[tuple[PointXZ, ...], tuple[tuple[PointXZ, ...], ...], float, float, float] | None:
    """Return a reusable local outline, including courtyard holes, for one building.

    Near-rectangles without holes deliberately stay on the mature rectangular
    path.  Meaningfully irregular valid polygons are simplified and quantized in
    model space.  Courtyard rings are preserved rather than filled, and a 180°
    canonicalization lets rotated copies reuse the same P3D.
    """

    if len(points) < 4:
        return None
    if polygon is None:
        polygon = Polygon(points, [tuple(hole) for hole in holes if len(hole) >= 3])
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 4.0:
        return None
    if footprint is None:
        footprint = _footprint_from_polygon_object(polygon)

    # Holes are intrinsically non-rectangular from the authored-building point of
    # view, so never throw them away merely because the outer ring is rectangular.
    envelope_area = max(0.01, footprint.width_m * footprint.length_m)
    if not holes and polygon.area / envelope_area >= max(
        0.0, min(1.0, rectangular_fill_threshold)
    ):
        return None

    simplified = polygon.simplify(
        max(0.0, float(simplify_tolerance)), preserve_topology=True
    )
    if simplified.is_empty or simplified.geom_type != "Polygon" or not simplified.is_valid:
        return None

    exterior = [(float(x), float(z)) for x, z in list(simplified.exterior.coords)[:-1]]
    interiors = [
        [(float(x), float(z)) for x, z in list(ring.coords)[:-1]]
        for ring in simplified.interiors
    ]
    total_vertices = len(exterior) + sum(len(ring) for ring in interiors)
    if len(exterior) < 4 or total_vertices > max(4, int(maximum_vertices)):
        return None

    centre = simplified.centroid
    centre_x, centre_z = float(centre.x), float(centre.y)
    heading = footprint.heading_degrees % 360.0
    quantum = max(0.05, float(vertex_quantum))

    def localize_ring(ring: Sequence[PointXZ], use_heading: float) -> list[PointXZ]:
        a = math.radians(use_heading)
        wx, wz = math.cos(a), -math.sin(a)
        lx, lz = math.sin(a), math.cos(a)
        result: list[PointXZ] = []
        for world_x, world_z in ring:
            dx = float(world_x) - centre_x
            dz = float(world_z) - centre_z
            result.append((dx * wx + dz * wz, dx * lx + dz * lz))
        return result

    outer = _compact_quantized_ring(localize_ring(exterior, heading), quantum, clockwise=False)
    local_holes = tuple(
        ring for ring in (
            _compact_quantized_ring(localize_ring(interior, heading), quantum, clockwise=True)
            for interior in interiors
        ) if ring
    )
    if len(outer) < 4 or len(local_holes) != len(interiors):
        return None

    local_polygon = Polygon(outer, local_holes)
    if local_polygon.is_empty or not local_polygon.is_valid or local_polygon.area <= 4.0:
        return None

    # Quantization can move the centroid slightly. Re-centre all rings once so
    # the P3D origin and the world placement centroid remain aligned.
    lc = local_polygon.centroid
    shift_x, shift_z = float(lc.x), float(lc.y)
    outer = _compact_quantized_ring(
        [(x - shift_x, z - shift_z) for x, z in outer], quantum, clockwise=False
    )
    local_holes = tuple(
        _compact_quantized_ring(
            [(x - shift_x, z - shift_z) for x, z in ring], quantum, clockwise=True
        )
        for ring in local_holes
    )
    if not outer or any(not ring for ring in local_holes):
        return None

    def shape_signature(
        candidate_outer: Sequence[PointXZ], candidate_holes: Sequence[Sequence[PointXZ]]
    ) -> tuple[tuple[PointXZ, ...], tuple[tuple[PointXZ, ...], ...]]:
        normalized_outer = _compact_quantized_ring(candidate_outer, quantum, clockwise=False)
        normalized_holes = tuple(sorted(
            _compact_quantized_ring(ring, quantum, clockwise=True)
            for ring in candidate_holes
        ))
        return normalized_outer, normalized_holes

    canonical_outer, canonical_holes = shape_signature(outer, local_holes)
    rotated_outer, rotated_holes = shape_signature(
        [(-x, -z) for x, z in outer],
        [[(-x, -z) for x, z in ring] for ring in local_holes],
    )
    if (rotated_outer, rotated_holes) < (canonical_outer, canonical_holes):
        canonical_outer, canonical_holes = rotated_outer, rotated_holes
        heading = (heading + 180.0) % 360.0

    final_polygon = Polygon(canonical_outer, canonical_holes)
    if final_polygon.is_empty or not final_polygon.is_valid or final_polygon.area <= 4.0:
        return None
    min_x, min_z, max_x, max_z = final_polygon.bounds
    return (
        canonical_outer,
        canonical_holes,
        heading,
        max(0.1, max_x - min_x),
        max(0.1, max_z - min_z),
    )


def _triangulate_polygon_coordinates(
    outer: Sequence[PointXZ], holes: Sequence[Sequence[PointXZ]] = (),
) -> tuple[tuple[PointXZ, PointXZ, PointXZ], ...]:
    """Triangulate a valid polygon, retaining courtyard holes exactly.

    The historical implementation used unconstrained Delaunay triangles and
    discarded every triangle crossing the footprint boundary. That is not a
    valid triangulation strategy for strongly concave polygons: a Delaunay
    diagonal can cross a notch, causing otherwise valid roof area to disappear.
    Keep the old path first so existing simple assets remain byte-stable, then
    use constrained triangulation or deterministic ear clipping as fallbacks.
    """

    normalized_holes = tuple(tuple(hole) for hole in holes if len(hole) >= 3)
    shape = Polygon(outer, normalized_holes)
    if shape.is_empty or not shape.is_valid or shape.area <= 0.0:
        return ()

    tolerance = max(0.05, shape.area * 1.0e-5)

    def verified(geometries: Iterable[Any]) -> tuple[tuple[PointXZ, PointXZ, PointXZ], ...]:
        result: list[tuple[PointXZ, PointXZ, PointXZ]] = []
        for candidate in geometries:
            if getattr(candidate, "geom_type", "") != "Polygon" or candidate.area <= 1.0e-8:
                continue
            coords = list(candidate.exterior.coords)[:-1]
            if len(coords) != 3 or not shape.covers(candidate):
                continue
            result.append(tuple((float(x), float(z)) for x, z in coords))
        covered = sum(abs(_polygon_signed_area(triangle)) for triangle in result)
        if result and abs(covered - shape.area) <= tolerance:
            return tuple(result)
        return ()

    # Preserve the old output for the common case. This succeeds for rectangles
    # and many mildly concave footprints and avoids gratuitously changing every
    # generated P3D merely because the robust fallback now exists.
    legacy = verified(triangulate(shape))
    if legacy:
        return legacy

    # GEOS constrained triangulation preserves the polygon boundary and holes,
    # which is exactly what a roof/collision mesh needs. This fixes valid deeply
    # concave OSM footprints that ordinary Delaunay triangulation cannot cover.
    if _constrained_delaunay_triangles is not None:
        constrained = _constrained_delaunay_triangles(shape)
        robust = verified(getattr(constrained, "geoms", (constrained,)))
        if robust:
            return robust

    # Shapely 2.0 lacks constrained_delaunay_triangles. For simple polygons,
    # deterministic ear clipping still gives an exact triangulation using the
    # source vertices and prevents one unusual building from aborting the build.
    if not normalized_holes:
        indices = _triangulate_simple_polygon(tuple((float(x), float(z)) for x, z in outer))
        if indices:
            points = tuple((float(x), float(z)) for x, z in outer)
            result = tuple(tuple(points[index] for index in triangle) for triangle in indices)
            covered = sum(abs(_polygon_signed_area(triangle)) for triangle in result)
            if abs(covered - shape.area) <= tolerance:
                return result

    return ()

def _triangulate_simple_polygon(points: Sequence[PointXZ]) -> tuple[tuple[int, int, int], ...]:
    """Ear-clip a simple polygon into triangles using only existing vertices."""

    if len(points) < 3:
        return ()
    values = list(points)
    if _polygon_signed_area(values) < 0.0:
        values.reverse()
    # Map any reversed working order back to the caller's coordinate tuple by
    # using coordinates as the stable identity. Native profiles reject duplicate
    # adjacent vertices, and practical OSM building rings do not repeat corners.
    lookup = {point: index for index, point in enumerate(points)}
    indices = [lookup[point] for point in values]

    def cross(a: PointXZ, b: PointXZ, c: PointXZ) -> float:
        return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])

    def point_in_triangle(p: PointXZ, a: PointXZ, b: PointXZ, c: PointXZ) -> bool:
        def side(p1: PointXZ, p2: PointXZ, p3: PointXZ) -> float:
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        d1, d2, d3 = side(p, a, b), side(p, b, c), side(p, c, a)
        has_neg = d1 < -1.0e-7 or d2 < -1.0e-7 or d3 < -1.0e-7
        has_pos = d1 > 1.0e-7 or d2 > 1.0e-7 or d3 > 1.0e-7
        return not (has_neg and has_pos)

    remaining = indices[:]
    triangles: list[tuple[int, int, int]] = []
    guard = len(remaining) * len(remaining) * 2
    while len(remaining) > 3 and guard > 0:
        guard -= 1
        clipped = False
        for position, current in enumerate(remaining):
            previous = remaining[position - 1]
            following = remaining[(position + 1) % len(remaining)]
            a, b, c = points[previous], points[current], points[following]
            if cross(a, b, c) <= 1.0e-7:
                continue
            if any(
                other not in {previous, current, following}
                and point_in_triangle(points[other], a, b, c)
                for other in remaining
            ):
                continue
            triangles.append((previous, current, following))
            del remaining[position]
            clipped = True
            break
        if not clipped:
            return ()
    if len(remaining) == 3:
        triangles.append(tuple(remaining))
    return tuple(triangles)


def decompose_footprint_rectangles(
    points: Sequence[PointXZ],
    *,
    max_parts: int = 4,
    minimum_part_size: float = 2.0,
    rectangular_fill_threshold: float = 0.88,
) -> tuple[tuple[PointXZ, ...], ...]:
    """Legacy geometry helper for decomposing an orthogonal polygon.

    Active procedural building placement no longer calls this helper: one source
    footprint is always one world object. It remains available for compatibility
    and geometry tests while older callers migrate away from multipart placement.

    The decomposition is deliberately conservative. Buildings whose outline is
    already close to its minimum rotated rectangle, contains too many distinct
    edge coordinates, or would require more than ``max_parts`` keep the legacy
    one-rectangle model. This targets the common mapped L/T/U cases without
    trying to turn every survey-grade polygon into a tiny model-per-corner maze.
    """

    if len(points) < 4 or max_parts < 2:
        return ()
    polygon = Polygon(points)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 1.0:
        return ()
    footprint = footprint_from_polygon(points)
    envelope_area = max(0.01, footprint.width_m * footprint.length_m)
    if polygon.area / envelope_area >= rectangular_fill_threshold:
        return ()

    centre = polygon.centroid
    centre_x, centre_z = float(centre.x), float(centre.y)
    angle = math.radians(footprint.heading_degrees)
    width_axis = (math.cos(angle), -math.sin(angle))
    length_axis = (math.sin(angle), math.cos(angle))

    def to_local(point: PointXZ) -> PointXZ:
        dx, dz = point[0] - centre_x, point[1] - centre_z
        return (
            dx * width_axis[0] + dz * width_axis[1],
            dx * length_axis[0] + dz * length_axis[1],
        )

    def to_world(point: PointXZ) -> PointXZ:
        return (
            centre_x + point[0] * width_axis[0] + point[1] * length_axis[0],
            centre_z + point[0] * width_axis[1] + point[1] * length_axis[1],
        )

    local = Polygon([to_local(point) for point in points])
    # Remove survey jitter that otherwise creates a dozen microscopic coordinate
    # bands along a wall that is visually straight in CWA anyway.
    local = local.simplify(0.35, preserve_topology=True)
    if local.is_empty or local.geom_type != "Polygon":
        return ()
    coords = list(local.exterior.coords)[:-1]

    def clustered(values: Sequence[float], tolerance: float = 0.55) -> list[float]:
        result: list[list[float]] = []
        for value in sorted(float(item) for item in values):
            if result and abs(value - sum(result[-1]) / len(result[-1])) <= tolerance:
                result[-1].append(value)
            else:
                result.append([value])
        return [sum(group) / len(group) for group in result]

    xs = clustered([point[0] for point in coords])
    zs = clustered([point[1] for point in coords])
    if len(xs) < 3 or len(zs) < 3 or len(xs) > 8 or len(zs) > 8:
        return ()

    filled: set[tuple[int, int]] = set()
    for zi in range(len(zs) - 1):
        z0, z1 = zs[zi], zs[zi + 1]
        if z1 - z0 < 0.2:
            continue
        for xi in range(len(xs) - 1):
            x0, x1 = xs[xi], xs[xi + 1]
            if x1 - x0 < 0.2:
                continue
            cell = Polygon(((x0, z0), (x1, z0), (x1, z1), (x0, z1)))
            intersection = local.intersection(cell).area
            if intersection >= cell.area * 0.80:
                filled.add((xi, zi))
    if not filled:
        return ()

    remaining = set(filled)
    rectangles: list[tuple[float, float, float, float]] = []
    while remaining:
        start_x, start_z = min(remaining, key=lambda item: (item[1], item[0]))
        end_x = start_x
        while (end_x + 1, start_z) in remaining:
            end_x += 1
        end_z = start_z
        while True:
            candidate_z = end_z + 1
            if all((xi, candidate_z) in remaining for xi in range(start_x, end_x + 1)):
                end_z = candidate_z
            else:
                break
        cells = {
            (xi, zi)
            for zi in range(start_z, end_z + 1)
            for xi in range(start_x, end_x + 1)
        }
        remaining.difference_update(cells)
        x0, x1 = xs[start_x], xs[end_x + 1]
        z0, z1 = zs[start_z], zs[end_z + 1]
        if x1 - x0 >= minimum_part_size and z1 - z0 >= minimum_part_size:
            rectangles.append((x0, z0, x1, z1))
        if len(rectangles) > max_parts:
            return ()

    if len(rectangles) < 2:
        return ()
    rectangle_area = sum((x1 - x0) * (z1 - z0) for x0, z0, x1, z1 in rectangles)
    if rectangle_area < polygon.area * 0.72:
        return ()

    result: list[tuple[PointXZ, ...]] = []
    for x0, z0, x1, z1 in rectangles:
        result.append(tuple(map(to_world, (
            (x0, z0), (x1, z0), (x1, z1), (x0, z1),
        ))))
    result.sort(
        key=lambda rectangle: -Polygon(rectangle).area
    )
    return tuple(result)


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


def _wall_quad_ground_anchored(
    texture: str,
    points: tuple[int, int, int, int],
    normal: int,
    *,
    u_scale: float,
    vertical_min: float,
    vertical_max: float,
) -> _Face:
    """Map a repeating facade atlas from the ground/storey boundary upward.

    Pitched roofs commonly leave an eave height that is not an exact multiple
    of the 3 m facade tile.  Mapping ``0..height / 3`` from the roof downward
    puts that fractional remainder at local Y=0.  Because the upper part of the
    atlas contains the painted windows, the wrap then paints a second fragment
    of a window directly into the ground.

    Anchor every complete 3 m repeat to the lower storey boundary instead.  Any
    fractional atlas remainder is consumed at the *top* of the wall, where the
    lower, window-free portion of the texture is harmless.
    """

    lower = float(vertical_min)
    upper = float(vertical_max)
    if upper <= lower:
        return _quad(texture, points, normal, u_scale, 1.0)

    tile = FACADE_TILE_HEIGHT_M
    # Subtract a tiny epsilon so exact 3 m boundaries do not round up to an
    # unnecessary extra repeat because of floating-point noise.
    anchor = math.ceil(upper / tile - 1e-9)
    v_bottom = anchor - lower / tile
    v_top = anchor - upper / tile
    return _quad_uv(
        texture, points, normal,
        u0=0.0, u1=u_scale, v0=v_top, v1=v_bottom,
    )


def _closed_facade_texture(
    key: BuildingVariantKey,
    texture: str,
    plain_texture: str,
    *,
    span_m: float,
    height_m: float,
    upper_band: bool = False,
    gable: bool = False,
) -> str:
    """Prefer plain materials when painted windows would become implausible."""

    if key.interiors:
        return texture
    if key.family not in _PAINTED_WINDOW_FAMILIES:
        return texture
    if gable:
        return plain_texture
    if (
        key.isolated_dwelling
        and not upper_band
        and span_m >= ISOLATED_DWELLING_WINDOW_TEXTURE_MIN_SPAN_M
    ):
        return texture
    if span_m < CLOSED_WINDOW_TEXTURE_MIN_SPAN_M:
        return plain_texture
    if upper_band and span_m < CLOSED_WINDOW_TEXTURE_MIN_UPPER_BAND_SPAN_M:
        return plain_texture
    if upper_band and height_m < CLOSED_WINDOW_TEXTURE_MIN_UPPER_BAND_HEIGHT_M:
        return plain_texture
    return texture


def _closed_facade_bands(
    key: BuildingVariantKey,
    wall_height: float,
    *,
    span_m: float,
    ground_texture: str,
    upper_texture: str,
    plain_texture: str,
    preserve_ground_texture: bool = False,
) -> tuple[tuple[float, float, str, bool], ...]:
    """Return explicit non-repeating vertical facade bands.

    The boolean marks a window-bearing atlas band. Any leftover wall above the
    requested storeys is deliberately plain instead of becoming a fractional
    extra floor through texture repetition.
    """

    height = max(0.0, float(wall_height))
    if height <= 1.0e-6:
        return ()
    if key.interiors or key.family not in _PAINTED_WINDOW_FAMILIES:
        return ((0.0, height, ground_texture, False),)

    storeys = _facade_storey_count(key, height)
    if storeys <= 0:
        return ((0.0, height, plain_texture, False),)
    band_height = min(
        VISIBLE_FACADE_STOREY_HEIGHT_M,
        height / max(1, storeys),
    )
    minimum_band_height = (
        ISOLATED_DWELLING_MINIMUM_FACADE_HEIGHT_M
        if key.isolated_dwelling and storeys == 1
        else MINIMUM_VISIBLE_FACADE_STOREY_HEIGHT_M
    )
    if band_height < minimum_band_height - 1.0e-6:
        return ((0.0, height, plain_texture, False),)

    bands: list[tuple[float, float, str, bool]] = []
    for storey in range(storeys):
        y0 = storey * band_height
        y1 = min(height, (storey + 1) * band_height)
        requested_texture = ground_texture if storey == 0 else upper_texture
        texture = (
            requested_texture
            if storey == 0 and preserve_ground_texture
            else _closed_facade_texture(
                key,
                requested_texture,
                plain_texture,
                span_m=span_m,
                height_m=y1 - y0,
                upper_band=storey > 0,
            )
        )
        bands.append((y0, y1, texture, texture != plain_texture))

    used_top = min(height, storeys * band_height)
    if height - used_top > 1.0e-5:
        bands.append((used_top, height, plain_texture, False))
    return tuple(bands)


def _closed_wall_storey_faces(
    key: BuildingVariantKey,
    points: tuple[tuple[float, float, float], ...],
    *,
    lower_left: int,
    upper_left: int,
    upper_right: int,
    lower_right: int,
    wall_height: float,
    span_m: float,
    ground_texture: str,
    upper_texture: str,
    plain_texture: str,
    normal: int,
    u_scale: float,
    ground_u_scale: float | None = None,
    preserve_ground_texture: bool = False,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[_Face, ...]]:
    """Split one closed wall into real storeys with non-wrapping window UVs."""

    left = points[lower_left]
    right = points[lower_right]
    faces: list[_Face] = []
    for y0, y1, texture, windowed in _closed_facade_bands(
        key,
        wall_height,
        span_m=span_m,
        ground_texture=ground_texture,
        upper_texture=upper_texture,
        plain_texture=plain_texture,
        preserve_ground_texture=preserve_ground_texture,
    ):
        base = len(points)
        points = points + (
            (left[0], y0, left[2]),
            (left[0], y1, left[2]),
            (right[0], y1, right[2]),
            (right[0], y0, right[2]),
        )
        indices = (base, base + 1, base + 2, base + 3)
        band_u_scale = (
            float(ground_u_scale)
            if ground_u_scale is not None and y0 <= 1.0e-6
            else float(u_scale)
        )
        if windowed:
            inset = FACADE_WINDOW_UV_INSET
            faces.append(_quad_uv(
                texture,
                indices,
                normal,
                u0=0.0,
                u1=max(1.0, band_u_scale),
                v0=inset,
                v1=1.0 - inset,
            ))
        else:
            faces.append(_quad(
                texture,
                indices,
                normal,
                max(1.0, band_u_scale),
                max(0.25, (y1 - y0) / FACADE_TILE_HEIGHT_M),
            ))
    return points, tuple(faces)


def _split_closed_wall_at_height(
    points: tuple[tuple[float, float, float], ...],
    *,
    lower_left: int,
    upper_left: int,
    upper_right: int,
    lower_right: int,
    split_height: float,
    wall_top: float,
    lower_texture: str,
    upper_texture: str,
    normal: int,
    u_scale: float,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[_Face, _Face]]:
    """Split a closed vertical wall into ground and upper texture bands."""

    left = points[lower_left]
    right = points[lower_right]
    mid_left = len(points)
    mid_right = mid_left + 1
    points = points + (
        (left[0], split_height, left[2]),
        (right[0], split_height, right[2]),
    )
    return points, (
        _wall_quad_ground_anchored(
            lower_texture,
            (lower_left, mid_left, mid_right, lower_right),
            normal,
            u_scale=u_scale,
            vertical_min=0.0,
            vertical_max=split_height,
        ),
        _wall_quad_ground_anchored(
            upper_texture,
            (mid_left, upper_left, upper_right, mid_right),
            normal,
            u_scale=u_scale,
            vertical_min=split_height,
            vertical_max=wall_top,
        ),
    )


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
    # The skirt primarily hides terrain gaps, but a shallow visible reveal also
    # helps grounded buildings read as properly founded rather than floating.
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


def _interior_wall_thickness(key: BuildingVariantKey) -> float:
    """Return one shared wall thickness for visual reveals, collision, and roadway inset."""

    return min(0.30, max(0.18, min(key.width_m, key.length_m) * 0.025))


def _supports_second_storey(
    key: BuildingVariantKey,
    *,
    wall_top: float | None = None,
) -> bool:
    """Return whether the shell can physically hold a usable upper floor."""

    if not key.interiors or key.family not in SECOND_STOREY_INTERIOR_FAMILIES:
        return False
    if (
        key.width_m < INTERIOR_SECOND_STOREY_MINIMUM_WIDTH_M
        or key.length_m < INTERIOR_SECOND_STOREY_MINIMUM_LENGTH_M
        or _main_building_height(key) < INTERIOR_SECOND_STOREY_MINIMUM_HEIGHT_M
    ):
        return False
    minimum_wall_top = (
        INTERIOR_SECOND_STOREY_FLOOR_Y_M
        + INTERIOR_SECOND_STOREY_MINIMUM_HEADROOM_M
        + 0.15
    )
    if wall_top is not None and wall_top < minimum_wall_top - 1.0e-6:
        return False
    return True


def _interior_storey_count(
    key: BuildingVariantKey,
    *,
    wall_top: float | None = None,
) -> int:
    """Return the number of genuinely usable procedural interior storeys.

    Utility shells deliberately stay one open level. Residential/town/city
    shells only gain an upper floor when the placement selected one *and* the
    footprint/height can hold a stable staircase with real headroom.
    """

    visible_wall_top = (
        _main_building_height(key) if wall_top is None else float(wall_top)
    )
    if _facade_storey_count(key, visible_wall_top) < 2:
        return 1
    return 2 if key.second_storey and _supports_second_storey(key, wall_top=wall_top) else 1


def _visible_window_storey_count(
    key: BuildingVariantKey,
    *,
    wall_top: float,
) -> int:
    """Return visible window rows independently from walkable upper floors.

    ``facade_storeys`` describes the exterior CWR/OSM decided to build. A
    staircase only decides whether an upper level is walkable, not whether the
    exterior second storey mysteriously loses all of its windows.
    """

    if key.family in UTILITY_INTERIOR_FAMILIES:
        return 1
    return max(1, _facade_storey_count(key, wall_top))


def _second_storey_layout(key: BuildingVariantKey) -> _SecondStoreyLayout | None:
    """Return the shared stair/stairwell layout for a two-level house interior."""

    if _interior_storey_count(key) < 2:
        return None
    wall_clearance = _interior_wall_thickness(key) + INTERIOR_ROADWAY_WALL_CLEARANCE_M
    half_width = max(0.5, key.width_m * 0.5 - wall_clearance)
    half_length = max(0.5, key.length_m * 0.5 - wall_clearance)
    stair_width = min(
        INTERIOR_SECOND_STOREY_STAIR_MAXIMUM_WIDTH_M,
        max(INTERIOR_SECOND_STOREY_STAIR_MINIMUM_WIDTH_M, key.width_m * 0.09),
    )
    stair_run = min(
        INTERIOR_SECOND_STOREY_STAIR_MAXIMUM_RUN_M,
        max(INTERIOR_SECOND_STOREY_STAIR_MINIMUM_RUN_M, key.length_m * 0.32),
        max(INTERIOR_SECOND_STOREY_STAIR_MINIMUM_RUN_M, half_length * 2.0 - 1.2),
    )
    side_margin = min(0.45, max(0.30, half_width * 0.10))
    stair_x1 = half_width - side_margin
    stair_x0 = stair_x1 - stair_width
    stair_z0 = -stair_run * 0.5
    stair_z1 = stair_run * 0.5
    opening_margin_x = 0.12
    return _SecondStoreyLayout(
        floor_y=INTERIOR_SECOND_STOREY_FLOOR_Y_M,
        stair_x0=stair_x0,
        stair_x1=stair_x1,
        stair_z0=stair_z0,
        stair_z1=stair_z1,
        opening_x0=max(-half_width, stair_x0 - opening_margin_x),
        opening_x1=min(half_width, stair_x1 + opening_margin_x),
        opening_z0=max(-half_length, stair_z0 - 0.18),
        # End the opening at the top tread so the upper Roadway immediately
        # provides a landing instead of leaving a small fall-through gap.
        opening_z1=min(half_length, stair_z1 - 0.03),
    )


def _interior_painted_facade_from_y(key: BuildingVariantKey) -> float:
    """Height above which distant-style painted windows may resume."""

    return FACADE_TILE_HEIGHT_M * _interior_storey_count(key)


def _gabled_profile(
    key: BuildingVariantKey,
    roof_pitch_degrees: float,
    *,
    interior_storeys_override: int | None = None,
) -> tuple[float, float, float]:
    """Return ``(eave_height, roof_rise, slope_length)`` for a gabled shell.

    A normal 6 m procedural house used to devote roughly two metres to roof
    rise, leaving too little vertical wall for a real upper room.  Enterable
    two-storey houses use a shallower gable so the second floor has usable
    standing height while preserving the overall model height.
    """

    half_width = key.width_m * 0.5
    main_height = _main_building_height(key)
    maximum_rise = half_width * math.tan(math.radians(roof_pitch_degrees))
    roof_rise = min(maximum_rise, max(1.0, main_height * 0.35))
    interior_storeys = (
        _interior_storey_count(key)
        if interior_storeys_override is None
        else max(1, int(interior_storeys_override))
    )
    visible_storeys = _facade_storey_count(key, main_height)
    if visible_storeys >= 2:
        minimum_visible_eave = (
            visible_storeys * MINIMUM_VISIBLE_FACADE_STOREY_HEIGHT_M
        )
        roof_rise = min(
            roof_rise,
            max(0.55, main_height - minimum_visible_eave),
        )
    if interior_storeys >= 2:
        minimum_eave = (
            INTERIOR_SECOND_STOREY_FLOOR_Y_M
            + INTERIOR_SECOND_STOREY_MINIMUM_HEADROOM_M
            + 0.15
        )
        maximum_interior_roof_rise = max(0.55, main_height - minimum_eave)
        roof_rise = min(roof_rise, maximum_interior_roof_rise)
    eave_height = max(2.5, main_height - roof_rise)
    roof_rise = main_height - eave_height
    slope_length = math.hypot(half_width, roof_rise)
    return eave_height, roof_rise, slope_length


def _interior_window_openings(
    key: BuildingVariantKey,
    horizontal_min: float,
    horizontal_max: float,
    wall_top: float,
    *,
    ground_exclusions: Sequence[tuple[float, float]] = (),
) -> tuple[tuple[float, float, float, float], ...]:
    """Return real window apertures for room-like interiors only.

    Barns, warehouses, sheds and garages use a cheap open-hall layout. Their
    facade material may imply vents or glazing, but cutting dozens of physical
    window holes into large utility shells is expensive and makes the collision
    mesh needlessly fragile.
    """

    if key.family in UTILITY_INTERIOR_FAMILIES:
        return ()
    storeys = _visible_window_storey_count(key, wall_top=wall_top)
    walkable_storeys = _interior_storey_count(key, wall_top=wall_top)
    return _window_openings(
        horizontal_min,
        horizontal_max,
        wall_top,
        ground_exclusions=ground_exclusions,
        maximum_storeys=storeys,
        storey_height_m=(
            INTERIOR_SECOND_STOREY_FLOOR_Y_M
            if walkable_storeys >= 2
            else VISIBLE_FACADE_STOREY_HEIGHT_M
        ),
    )


def _front_faces_with_doorway(
    key: BuildingVariantKey,
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
    upper_texture: str | None = None,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[_Face, ...]]:
    """Split the front wall around its door and any real procedural windows."""

    door_half, door_height, _pivot_z = _door_dimensions(key)
    door_height = min(door_height, max(1.9, wall_top - 0.25))
    openings = _interior_window_openings(
        key,
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
        upper_texture=upper_texture,
        upper_texture_from_y=_interior_painted_facade_from_y(key),
    )


def _window_openings(
    horizontal_min: float,
    horizontal_max: float,
    wall_top: float,
    *,
    ground_exclusions: Sequence[tuple[float, float]] = (),
    maximum_storeys: int | None = 1,
    storey_height_m: float = FACADE_TILE_HEIGHT_M,
) -> tuple[tuple[float, float, float, float], ...]:
    """Return repeated window bays shared by visual and collision LODs.

    Callers cap genuine apertures to the number of actually usable interior
    levels; higher decorative storeys remain cheap painted facade windows.
    """

    span = horizontal_max - horizontal_min
    bay_count = max(1, min(12, round(span / 3.8)))
    bay_width = span / bay_count
    opening_half = min(0.75, max(0.52, bay_width * 0.20))
    storeys = max(1, min(5, round(wall_top / 3.0)))
    if maximum_storeys is not None:
        storeys = min(storeys, max(1, int(maximum_storeys)))
    openings: list[tuple[float, float, float, float]] = []
    for storey in range(storeys):
        sill = storey * storey_height_m + INTERIOR_WINDOW_SILL_M
        top = min(
            storey * storey_height_m + INTERIOR_WINDOW_TOP_M,
            wall_top - 0.25,
        )
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
    upper_texture: str | None = None,
    upper_texture_from_y: float = 3.0,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[_Face, ...]]:
    """Tessellate one wall while omitting every requested open aperture.

    When ``upper_texture`` is provided, the solid wall above the genuinely
    enterable storeys uses that atlas. This keeps real cut apertures on usable
    levels while higher decorative storeys fall back to cheap painted windows.
    """

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
        bands = [(vertical_min, vertical_max)]
        if (
            upper_texture
            and vertical_min < upper_texture_from_y < vertical_max
        ):
            bands = [
                (vertical_min, upper_texture_from_y),
                (upper_texture_from_y, vertical_max),
            ]
        for band_min, band_max in bands:
            if band_max - band_min <= 1.0e-6:
                continue
            start = len(points)
            points = points + (
                wall_point(cell_min, band_min),
                wall_point(cell_min, band_max),
                wall_point(cell_max, band_max),
                wall_point(cell_max, band_min),
            )
            u0 = (cell_min - horizontal_min) / 4.0
            u1 = (cell_max - horizontal_min) / 4.0
            v0 = (wall_top - band_max) / 3.0
            v1 = (wall_top - band_min) / 3.0
            face_texture = (
                upper_texture
                if upper_texture and band_min >= upper_texture_from_y - 1.0e-6
                else texture
            )
            faces.append(_Face(face_texture, (
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
    inset = _interior_wall_thickness(key)
    door_half, door_height, _pivot_z = _door_dimensions(key)
    door_height = min(door_height, max(1.9, wall_top - 0.25))
    front_openings = _interior_window_openings(
        key,
        -half_width,
        half_width,
        wall_top,
        ground_exclusions=((-door_half - 0.35, door_half + 0.35),),
    ) + ((-door_half, door_half, 0.0, door_height),)
    back_openings = _interior_window_openings(key, -half_width, half_width, wall_top)
    side_openings = _interior_window_openings(key, -half_length, half_length, wall_top)
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
    """Add cheap flat white surrounds around genuine enterable-storey windows.

    Older builds modelled every trim strip as a six-faced box and then doubled
    every face for CWA. A modest house could spend hundreds of polygons on trim
    alone. Flat double-sided strips look the same at normal play distance while
    cutting that cost by roughly six times.
    """

    half_width = key.width_m * 0.5
    half_length = key.length_m * 0.5
    door_half, _door_height, _pivot_z = _door_dimensions(key)
    front_windows = _interior_window_openings(
        key,
        -half_width,
        half_width,
        wall_top,
        ground_exclusions=((-door_half - 0.35, door_half + 0.35),),
    )
    back_windows = _interior_window_openings(key, -half_width, half_width, wall_top)
    side_windows = _interior_window_openings(key, -half_length, half_length, wall_top)
    trim_width = 0.14
    projection = 0.035

    def add_strip(
        horizontal_min: float,
        horizontal_max: float,
        vertical_min: float,
        vertical_max: float,
        plane: float,
        axis: str,
        normal: int,
    ) -> None:
        nonlocal points, faces
        start = len(points)
        if axis == "x":
            points = points + (
                (horizontal_min, vertical_min, plane),
                (horizontal_min, vertical_max, plane),
                (horizontal_max, vertical_max, plane),
                (horizontal_max, vertical_min, plane),
            )
        elif axis == "z":
            points = points + (
                (plane, vertical_min, horizontal_min),
                (plane, vertical_max, horizontal_min),
                (plane, vertical_max, horizontal_max),
                (plane, vertical_min, horizontal_max),
            )
        else:
            raise ValueError(f"unsupported trim horizontal axis: {axis}")
        faces = faces + (_quad(texture, (start, start + 1, start + 2, start + 3), normal),)

    walls = (
        ("x", -half_length - projection, front_windows, 0),
        ("z", half_width + projection, side_windows, 1),
        ("x", half_length + projection, back_windows, 2),
        ("z", -half_width - projection, side_windows, 3),
    )
    for axis, plane, openings, normal in walls:
        for opening_min, opening_max, opening_bottom, opening_top in openings:
            add_strip(
                opening_min - trim_width,
                opening_min,
                opening_bottom - trim_width,
                opening_top + trim_width,
                plane,
                axis,
                normal,
            )
            add_strip(
                opening_max,
                opening_max + trim_width,
                opening_bottom - trim_width,
                opening_top + trim_width,
                plane,
                axis,
                normal,
            )
            add_strip(
                opening_min,
                opening_max,
                opening_bottom - trim_width,
                opening_bottom,
                plane,
                axis,
                normal,
            )
            add_strip(
                opening_min,
                opening_max,
                opening_top,
                opening_top + trim_width,
                plane,
                axis,
                normal,
            )
    return points, faces


def _add_window_crosses(
    key: BuildingVariantKey,
    points: tuple[tuple[float, float, float], ...],
    faces: tuple[_Face, ...],
    *,
    wall_top: float,
    texture: str,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[_Face, ...]]:
    """Add flat plus-shaped mullions to genuine procedural window openings."""

    half_width = key.width_m * 0.5
    half_length = key.length_m * 0.5
    door_half, _door_height, _pivot_z = _door_dimensions(key)
    front_windows = _interior_window_openings(
        key,
        -half_width,
        half_width,
        wall_top,
        ground_exclusions=((-door_half - 0.35, door_half + 0.35),),
    )
    back_windows = _interior_window_openings(key, -half_width, half_width, wall_top)
    side_windows = _interior_window_openings(key, -half_length, half_length, wall_top)
    half_bar = INTERIOR_WINDOW_CROSS_BAR_WIDTH_M * 0.5
    inset = 0.015

    def add_strip(
        horizontal_min: float,
        horizontal_max: float,
        vertical_min: float,
        vertical_max: float,
        plane: float,
        axis: str,
        normal: int,
    ) -> None:
        nonlocal points, faces
        start = len(points)
        if axis == "x":
            points = points + (
                (horizontal_min, vertical_min, plane),
                (horizontal_min, vertical_max, plane),
                (horizontal_max, vertical_max, plane),
                (horizontal_max, vertical_min, plane),
            )
        elif axis == "z":
            points = points + (
                (plane, vertical_min, horizontal_min),
                (plane, vertical_max, horizontal_min),
                (plane, vertical_max, horizontal_max),
                (plane, vertical_min, horizontal_max),
            )
        else:
            raise ValueError(f"unsupported window-cross horizontal axis: {axis}")
        faces = faces + (_quad(texture, (start, start + 1, start + 2, start + 3), normal),)

    walls = (
        ("x", -half_length - inset, front_windows, 0),
        ("z", half_width + inset, side_windows, 1),
        ("x", half_length + inset, back_windows, 2),
        ("z", -half_width - inset, side_windows, 3),
    )
    for axis, plane, openings, normal in walls:
        for opening_min, opening_max, opening_bottom, opening_top in openings:
            horizontal_mid = (opening_min + opening_max) * 0.5
            vertical_mid = (opening_bottom + opening_top) * 0.5
            add_strip(
                horizontal_mid - half_bar,
                horizontal_mid + half_bar,
                opening_bottom,
                opening_top,
                plane,
                axis,
                normal,
            )
            add_strip(
                opening_min,
                opening_max,
                vertical_mid - half_bar,
                vertical_mid + half_bar,
                plane,
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
    wall_top: float,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
    tuple[_Face, ...],
]:
    """Add visible floors/ceilings and simple room structure.

    Utility families intentionally use one open hall with a high ceiling. Small
    room-like buildings keep the original single-floor partition. Residential,
    townhouse and urban variants that have enough height and footprint receive
    a real upper floor with a stairwell and a straight visible staircase.
    """

    inset = _interior_wall_thickness(key) + 0.015
    half_width = max(0.5, key.width_m * 0.5 - inset)
    half_length = max(0.5, key.length_m * 0.5 - inset)
    layout = _second_storey_layout(key)
    if key.family in UTILITY_INTERIOR_FAMILIES:
        ceiling = max(2.35, wall_top - 0.15)
    elif layout is not None:
        ceiling = min(
            wall_top - 0.15,
            layout.floor_y + 2.45,
        )
        ceiling = max(layout.floor_y + 2.05, ceiling)
    else:
        ceiling = min(2.85, max(2.35, wall_top - 0.15))
    normal_start = len(normals)
    normals = normals + (
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )

    floor_start = len(points)
    points = points + (
        (-half_width, INTERIOR_VISUAL_FLOOR_Y_M, -half_length),
        (half_width, INTERIOR_VISUAL_FLOOR_Y_M, -half_length),
        (half_width, INTERIOR_VISUAL_FLOOR_Y_M, half_length),
        (-half_width, INTERIOR_VISUAL_FLOOR_Y_M, half_length),
    )
    interior_faces: list[_Face] = [
        _quad(
            floor_texture,
            (floor_start + 0, floor_start + 3, floor_start + 2, floor_start + 1),
            normal_start,
            key.width_m / 2.0,
            key.length_m / 2.0,
        ),
    ]

    def add_horizontal_rect(
        x0: float,
        z0: float,
        x1: float,
        z1: float,
        y: float,
        *,
        texture: str,
        upward: bool,
    ) -> None:
        nonlocal points
        if x1 - x0 <= 1.0e-6 or z1 - z0 <= 1.0e-6:
            return
        start = len(points)
        points = points + ((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1))
        if upward:
            indices = (start + 0, start + 3, start + 2, start + 1)
            normal = normal_start
        else:
            indices = (start + 0, start + 1, start + 2, start + 3)
            normal = normal_start + 1
        interior_faces.append(_quad(
            texture,
            indices,
            normal,
            max(0.25, (x1 - x0) / 2.0),
            max(0.25, (z1 - z0) / 2.0),
        ))

    if layout is not None and _interior_storey_count(key, wall_top=wall_top) >= 2:
        # The intermediate floor is visual only. Collision/standing is handled
        # exclusively by the upper Roadway surface, avoiding the competing-floor
        # contact that caused soldiers to become embedded in earlier interiors.
        hx0 = max(-half_width, layout.opening_x0)
        hx1 = min(half_width, layout.opening_x1)
        hz0 = max(-half_length, layout.opening_z0)
        hz1 = min(half_length, layout.opening_z1)
        rectangles = (
            (-half_width, -half_length, hx0, half_length),
            (hx1, -half_length, half_width, half_length),
            (hx0, -half_length, hx1, hz0),
            (hx0, hz1, hx1, half_length),
        )
        upper_floor_y = layout.floor_y + INTERIOR_VISUAL_FLOOR_Y_M
        lower_ceiling_y = layout.floor_y - INTERIOR_SECOND_STOREY_CEILING_THICKNESS_M
        for x0, z0, x1, z1 in rectangles:
            add_horizontal_rect(
                x0, z0, x1, z1, upper_floor_y,
                texture=floor_texture, upward=True,
            )
            add_horizontal_rect(
                x0, z0, x1, z1, lower_ceiling_y,
                texture=wall_texture, upward=False,
            )

        # Thin fascia around the stair opening hides the gap between floor top
        # and ceiling underside without introducing any Geometry collision.
        fascia_top = upper_floor_y
        fascia_bottom = lower_ceiling_y
        fascia_start = len(points)
        points = points + (
            (hx0, fascia_bottom, hz0), (hx0, fascia_top, hz0),
            (hx1, fascia_top, hz0), (hx1, fascia_bottom, hz0),
            (hx1, fascia_bottom, hz1), (hx1, fascia_top, hz1),
            (hx0, fascia_top, hz1), (hx0, fascia_bottom, hz1),
        )
        interior_faces.extend((
            _quad(floor_texture, (fascia_start + 0, fascia_start + 1, fascia_start + 2, fascia_start + 3), normal_start + 2),
            _quad(floor_texture, (fascia_start + 3, fascia_start + 2, fascia_start + 5, fascia_start + 4), normal_start + 3),
            _quad(floor_texture, (fascia_start + 4, fascia_start + 5, fascia_start + 6, fascia_start + 7), normal_start + 2),
            _quad(floor_texture, (fascia_start + 7, fascia_start + 6, fascia_start + 1, fascia_start + 0), normal_start + 3),
        ))

        add_horizontal_rect(
            -half_width, -half_length, half_width, half_length, ceiling,
            texture=wall_texture, upward=False,
        )

        # Visible steps. The Roadway LOD mirrors these as horizontal walkable
        # treads. They are not Geometry boxes, so the player receives stable step
        # heights without adding solid risers that can snag legacy collision.
        step_count = INTERIOR_SECOND_STOREY_STAIR_STEPS
        step_run = (layout.stair_z1 - layout.stair_z0) / step_count
        step_rise = layout.floor_y / step_count
        previous_y = INTERIOR_VISUAL_FLOOR_Y_M
        for index in range(step_count):
            z0 = layout.stair_z0 + index * step_run
            z1 = layout.stair_z0 + (index + 1) * step_run
            top_y = (index + 1) * step_rise
            start = len(points)
            points = points + (
                (layout.stair_x0, top_y, z0),
                (layout.stair_x1, top_y, z0),
                (layout.stair_x1, top_y, z1),
                (layout.stair_x0, top_y, z1),
                (layout.stair_x0, previous_y, z0),
                (layout.stair_x1, previous_y, z0),
            )
            interior_faces.extend((
                _quad(
                    floor_texture,
                    (start + 0, start + 3, start + 2, start + 1),
                    normal_start,
                    max(0.25, (layout.stair_x1 - layout.stair_x0) / 2.0),
                    max(0.10, step_run / 2.0),
                ),
                _quad(
                    wall_texture,
                    (start + 4, start + 0, start + 1, start + 5),
                    normal_start + 3,
                    max(0.25, (layout.stair_x1 - layout.stair_x0) / 2.0),
                    max(0.08, (top_y - previous_y) / 2.0),
                ),
            ))
            previous_y = top_y
    else:
        ceiling_start = len(points)
        points = points + (
            (-half_width, ceiling, -half_length),
            (half_width, ceiling, -half_length),
            (half_width, ceiling, half_length),
            (-half_width, ceiling, half_length),
        )
        interior_faces.append(_quad(
            wall_texture,
            (ceiling_start + 0, ceiling_start + 1, ceiling_start + 2, ceiling_start + 3),
            normal_start + 1,
            key.width_m / 2.0,
            key.length_m / 2.0,
        ))

        if (
            key.family not in UTILITY_INTERIOR_FAMILIES
            and key.width_m >= 6.0
            and key.length_m >= 7.0
        ):
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

def _entrance_uses_vehicle_ramp(key: BuildingVariantKey) -> bool:
    """Return whether the entrance is a vehicle-scale bay rather than a porch."""

    return (
        key.family in {"agricultural", "industrial"}
        or (key.family == "outbuilding" and _outbuilding_is_garage(key))
    )


def _interior_vehicle_ramp_profile(
    key: BuildingVariantKey,
    foundation_depth: float,
) -> tuple[float, float, float, float] | None:
    """Return ``(outer_z, inner_z, outer_y, inner_y)`` for a vehicle ramp."""

    if not _entrance_uses_vehicle_ramp(key):
        return None
    depth = max(0.0, float(foundation_depth))
    if depth <= 1.0e-6:
        return None
    front_z = -key.length_m * 0.5
    run = max(
        INTERIOR_VEHICLE_RAMP_MINIMUM_RUN_M,
        depth * INTERIOR_VEHICLE_RAMP_RUN_PER_RISE,
    )
    return front_z - run, front_z, -depth, 0.0


def _interior_stair_profile(
    key: BuildingVariantKey,
    foundation_depth: float,
) -> tuple[tuple[float, float, float, float], ...]:
    """Return exterior stair treads from the door down the foundation skirt.

    Each tuple is ``(outer_z, inner_z, top_y, bottom_y)``.  The first tread is
    flush with the door threshold at local Y=0 and subsequent treads descend
    away from the front wall.  The last riser reaches the full foundation depth,
    so some lower steps may harmlessly disappear into terrain on a slope while
    still guaranteeing a continuous route to the doorway.
    """

    depth = max(0.0, float(foundation_depth))
    if depth <= 1.0e-6:
        return ()
    half_length = key.length_m * 0.5
    count = max(
        1,
        min(
            INTERIOR_STAIR_MAXIMUM_STEPS,
            int(math.ceil(depth / INTERIOR_STAIR_TARGET_RISE_M)),
        ),
    )
    rise = depth / count
    front_z = -half_length
    profile: list[tuple[float, float, float, float]] = []
    for index in range(count):
        inner_z = front_z - index * INTERIOR_STAIR_TREAD_M
        outer_z = inner_z - INTERIOR_STAIR_TREAD_M
        top_y = -index * rise
        bottom_y = -min(depth, (index + 1) * rise)
        profile.append((outer_z, inner_z, top_y, bottom_y))
    return tuple(profile)


def _add_entrance_stairs(
    key: BuildingVariantKey,
    points: tuple[tuple[float, float, float], ...],
    normals: tuple[tuple[float, float, float], ...],
    faces: tuple[_Face, ...],
    *,
    foundation_depth: float,
    texture: str,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
    tuple[_Face, ...],
]:
    """Add the visible terrain transition for an enterable entrance.

    Pedestrian doors keep small foundation stairs. Barns, warehouses, and
    car-sized garages use a broad sloped ramp so a vehicle-scale door does not
    absurdly terminate in a domestic porch.
    """

    door_half, _door_height, _pivot_z = _door_dimensions(key)
    transition_half = min(
        key.width_m * 0.45,
        door_half + INTERIOR_STAIR_SIDE_MARGIN_M,
    )
    ramp = _interior_vehicle_ramp_profile(key, foundation_depth)
    if ramp is not None:
        outer_z, inner_z, outer_y, inner_y = ramp
        normal_start = len(normals)
        normals = normals + (
            (0.0, 1.0, 0.0),
            (-1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
        )
        start = len(points)
        points = points + (
            (-transition_half, outer_y, outer_z),
            (transition_half, outer_y, outer_z),
            (transition_half, inner_y, inner_z),
            (-transition_half, inner_y, inner_z),
            (-transition_half, outer_y - 0.08, outer_z),
            (transition_half, outer_y - 0.08, outer_z),
        )
        ramp_faces = (
            _quad(texture, (start + 0, start + 3, start + 2, start + 1), normal_start),
            _triangle(texture, (start + 4, start + 3, start + 0), normal_start + 1),
            _triangle(texture, (start + 1, start + 2, start + 5), normal_start + 2),
            _quad(texture, (start + 4, start + 0, start + 1, start + 5), normal_start + 3),
        )
        return points, normals, faces + ramp_faces

    profile = _interior_stair_profile(key, foundation_depth)
    if not profile:
        return points, normals, faces
    top_normal = len(normals)
    front_normal = top_normal + 1
    normals = normals + ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0))
    stair_faces: list[_Face] = []
    for outer_z, inner_z, top_y, bottom_y in profile:
        start = len(points)
        points = points + (
            (-transition_half, top_y, inner_z),
            (transition_half, top_y, inner_z),
            (transition_half, top_y, outer_z),
            (-transition_half, top_y, outer_z),
            (-transition_half, bottom_y, outer_z),
            (transition_half, bottom_y, outer_z),
        )
        stair_faces.extend((
            _quad(texture, (start + 0, start + 3, start + 2, start + 1), top_normal),
            _quad(texture, (start + 4, start + 3, start + 2, start + 5), front_normal),
        ))
    return points, normals, faces + tuple(stair_faces)

def _interior_roadway_lod(
    key: BuildingVariantKey,
    foundation_depth: float,
) -> _Lod | None:
    """Return stable walkable floors, doorway bridge, stairs, and vehicle ramps.

    Two-storey houses use a continuous segmented Roadway slope beneath the
    visible stair treads, backed by a thin Geometry ramp. That combination is
    deliberately redundant: older CWA builds can intermittently miss Roadway
    tread contacts, while a Geometry-only staircase is prone to snagging.
    """

    if not key.interiors:
        return None

    wall_inset = _interior_wall_thickness(key) + INTERIOR_ROADWAY_WALL_CLEARANCE_M
    half_width = max(0.35, key.width_m * 0.5 - wall_inset)
    half_length = max(0.35, key.length_m * 0.5 - wall_inset)
    roadway_y = INTERIOR_ROADWAY_Y_M
    points: list[tuple[float, float, float]] = []
    faces: list[_Face] = []

    def add_roadway_quad(x0: float, z0: float, x1: float, z1: float, y: float) -> None:
        if x1 - x0 <= 1.0e-6 or z1 - z0 <= 1.0e-6:
            return
        start = len(points)
        points.extend(((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z1)))
        faces.append(_Face("", (
            (start + 0, 0, 0.0, 0.0),
            (start + 3, 0, 0.0, 1.0),
            (start + 2, 0, 1.0, 1.0),
            (start + 1, 0, 1.0, 0.0),
        )))

    def add_roadway_slope(
        x0: float, z0: float, x1: float, z1: float, y0: float, y1: float
    ) -> None:
        if x1 - x0 <= 1.0e-6 or z1 - z0 <= 1.0e-6:
            return
        start = len(points)
        points.extend(((x0, y0, z0), (x1, y0, z0), (x1, y1, z1), (x0, y1, z1)))
        faces.append(_Face("", (
            (start + 0, 0, 0.0, 0.0),
            (start + 3, 0, 0.0, 1.0),
            (start + 2, 0, 1.0, 1.0),
            (start + 1, 0, 1.0, 0.0),
        )))

    def tile_rect(x0: float, z0: float, x1: float, z1: float, y: float) -> None:
        if x1 - x0 <= 1.0e-6 or z1 - z0 <= 1.0e-6:
            return
        x_segments = max(1, int(math.ceil((x1 - x0) / INTERIOR_ROADWAY_TILE_SPAN_M)))
        z_segments = max(1, int(math.ceil((z1 - z0) / INTERIOR_ROADWAY_TILE_SPAN_M)))
        x_step = (x1 - x0) / x_segments
        z_step = (z1 - z0) / z_segments
        for z_index in range(z_segments):
            sz0 = z0 + z_index * z_step
            sz1 = z0 + (z_index + 1) * z_step
            for x_index in range(x_segments):
                sx0 = x0 + x_index * x_step
                sx1 = x0 + (x_index + 1) * x_step
                add_roadway_quad(sx0, sz0, sx1, sz1, y)

    layout = _second_storey_layout(key)
    if layout is None:
        # Large warehouse/barn floors are tiled so no single Roadway polygon
        # spans a huge legacy-engine coordinate range. Houses usually remain one
        # ground tile.
        tile_rect(-half_width, -half_length, half_width, half_length, roadway_y)
    else:
        # Do not leave a second horizontal Roadway directly under the inclined
        # stair surface. Even though both are Roadway rather than Geometry,
        # overlapping walkable surfaces can make the legacy character controller
        # choose the wrong contact as the unit climbs. Cut only the stair footprint
        # from the ground level and let the stair treads be the sole walkable
        # surfaces there.
        sx0 = max(-half_width, layout.stair_x0)
        sx1 = min(half_width, layout.stair_x1)
        sz0 = max(-half_length, layout.stair_z0)
        sz1 = min(half_length, layout.stair_z1)
        for x0, z0, x1, z1 in (
            (-half_width, -half_length, sx0, half_length),
            (sx1, -half_length, half_width, half_length),
            (sx0, -half_length, sx1, sz0),
            (sx0, sz1, sx1, half_length),
        ):
            tile_rect(x0, z0, x1, z1, roadway_y)

        hx0 = max(-half_width, layout.opening_x0)
        hx1 = min(half_width, layout.opening_x1)
        hz0 = max(-half_length, layout.opening_z0)
        hz1 = min(half_length, layout.opening_z1)
        upper_y = layout.floor_y + roadway_y
        for x0, z0, x1, z1 in (
            (-half_width, -half_length, hx0, half_length),
            (hx1, -half_length, half_width, half_length),
            (hx0, -half_length, hx1, hz0),
            (hx0, hz1, hx1, half_length),
        ):
            tile_rect(x0, z0, x1, z1, upper_y)
        # CWA is much happier with conventional horizontal stair contacts than
        # with an inclined Roadway face. Mirror every visible tread in Roadway
        # and overlap adjacent treads by a few centimetres so precision loss at
        # the riser edge cannot create a fall-through crack. Geometry supplies a
        # second, slightly lower solid staircase beneath these surfaces.
        step_count = INTERIOR_SECOND_STOREY_STAIR_STEPS
        stair_run = layout.stair_z1 - layout.stair_z0
        step_run = stair_run / step_count
        step_rise = layout.floor_y / step_count
        overlap = min(0.035, step_run * 0.18)
        for index in range(step_count):
            z0 = layout.stair_z0 + index * step_run
            z1 = layout.stair_z0 + (index + 1) * step_run
            if index > 0:
                z0 -= overlap
            if index < step_count - 1:
                z1 += overlap
            top_y = roadway_y + (index + 1) * step_rise
            add_roadway_quad(
                layout.stair_x0, z0, layout.stair_x1, z1, top_y
            )

    door_half, _door_height, _pivot_z = _door_dimensions(key)
    transition_half = min(key.width_m * 0.45, door_half + INTERIOR_STAIR_SIDE_MARGIN_M)
    threshold_half = max(0.25, door_half - 0.08)
    front_z = -key.length_m * 0.5
    vehicle_ramp = _interior_vehicle_ramp_profile(key, foundation_depth)
    stair_profile = () if vehicle_ramp is not None else _interior_stair_profile(key, foundation_depth)

    # Bridge the complete wall thickness into the safely inset floor. Keep this
    # strip narrower than the real doorway so the Roadway surface never enters a
    # solid jamb component.
    has_transition = vehicle_ramp is not None or bool(stair_profile)
    threshold_outer_z = front_z - (0.04 if has_transition else INTERIOR_STAIR_TREAD_M)
    threshold_inner_z = -half_length + 0.03
    add_roadway_quad(
        -threshold_half,
        threshold_outer_z,
        threshold_half,
        threshold_inner_z,
        roadway_y,
    )

    if vehicle_ramp is not None:
        outer_z, inner_z, outer_y, inner_y = vehicle_ramp
        segments = max(2, int(math.ceil((inner_z - outer_z) / 1.5)))
        for index in range(segments):
            t0 = index / segments
            t1 = (index + 1) / segments
            z0 = outer_z + (inner_z - outer_z) * t0
            z1 = outer_z + (inner_z - outer_z) * t1
            y0 = outer_y + (inner_y - outer_y) * t0 + roadway_y
            y1 = outer_y + (inner_y - outer_y) * t1 + roadway_y
            add_roadway_slope(
                -transition_half, z0, transition_half, z1, y0, y1
            )
    else:
        for outer_z, inner_z, top_y, _bottom_y in stair_profile:
            add_roadway_quad(
                -transition_half,
                outer_z,
                transition_half,
                inner_z,
                top_y + roadway_y,
            )

    return _Lod(
        tuple(points),
        ((0.0, 1.0, 0.0),),
        tuple(faces),
        _ROADWAY_LOD,
    )

def _main_building_height(key: BuildingVariantKey) -> float:
    """Return the wall/roof height used by the main building mass.

    Churches keep their existing tower proportions, but the nave itself is one
    conventional 3 m storey shorter. This is deliberately a model-only change:
    footprint, grounding, foundation depth and tower placement stay unchanged.
    """

    if key.family == "church":
        return max(3.0, key.height_m - 3.0)
    return key.height_m


def _door_dimensions(key: BuildingVariantKey) -> tuple[float, float, float]:
    """Return entrance half-width, height, and front-wall pivot depth.

    Houses keep a normal pedestrian door. Enterable agricultural/industrial
    shells and car-sized outbuildings use vehicle-scale openings. Outbuildings
    too small to contain a passenger car are sheds and keep a small door.
    """

    half_width = key.width_m * 0.5
    main_height = _main_building_height(key)
    if key.family == "agricultural":
        door_half = min(2.2, max(1.45, half_width * 0.30))
        target_height = 3.4
    elif key.family == "industrial":
        door_half = min(2.6, max(1.65, half_width * 0.24))
        target_height = 3.8
    elif key.family == "outbuilding" and _outbuilding_is_garage(key):
        # Car-capable outbuildings are garages and receive a proper vehicle bay.
        # Make the opening read as an actual garage door rather than an enlarged
        # person door, while still leaving structural jambs at either side.
        door_half = min(1.80, max(1.40, half_width * 0.45))
        target_height = 2.6
    elif key.family == "outbuilding":
        # Shed-class outbuildings keep a human-sized entrance. Explicit OSM
        # garage/garages/carport tags are resolved to the garage subtype before
        # model generation and therefore take the vehicle-bay branch above.
        door_half = min(0.75, max(0.55, half_width * 0.18))
        target_height = 2.05
    elif key.family in {"school", "shop", "urban"}:
        # Public/commercial pedestrian doors are wider than domestic doors, but
        # still remain recognisably human-scale rather than inheriting a fixed
        # fraction of a broad facade.
        door_half = min(0.70, max(0.58, half_width * 0.08))
        target_height = 2.20
    elif key.family == "townhouse":
        door_half = min(0.58, max(0.50, half_width * 0.08))
        target_height = 2.12
    else:
        # Typical detached-house entrance: roughly 0.96-1.08 m clear width.
        door_half = min(0.54, max(0.48, half_width * 0.07))
        target_height = 2.08

    # Always leave enough wall for stable jamb collision components. Vehicle
    # bays can use narrower jambs than pedestrian doors; this is important for
    # the smallest footprints that genuinely fit a car.
    if key.family == "outbuilding" and _outbuilding_is_garage(key):
        jamb = 0.25
    elif key.family in {"agricultural", "industrial"}:
        jamb = 0.35
    else:
        jamb = 0.45
    door_half = min(door_half, max(0.55, half_width - jamb))
    door_height = min(target_height, max(1.9, main_height - 0.25))
    pivot_z = -key.length_m * 0.5 + 0.04
    return door_half, door_height, pivot_z


def _add_animated_door_visual(
    lod: _Lod,
    key: BuildingVariantKey,
    texture: str,
) -> _Lod:
    """Add one selected, double-sided door panel to an enterable visual LOD."""

    if not key.interiors:
        return lod
    door_half, door_height, pivot_z = _door_dimensions(key)
    point_start = len(lod.points)
    points = lod.points + (
        (-door_half, 0.03, pivot_z),
        (-door_half, door_height - 0.03, pivot_z),
        (door_half, door_height - 0.03, pivot_z),
        (door_half, 0.03, pivot_z),
    )
    normal_start = len(lod.normals)
    normals = lod.normals + ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0))
    face = _Face(texture, (
        (point_start + 0, normal_start, 0.0, 1.0),
        (point_start + 1, normal_start, 0.0, 0.0),
        (point_start + 2, normal_start, 1.0, 0.0),
        (point_start + 3, normal_start, 1.0, 1.0),
    ))
    faces = lod.faces + (face, _reverse_face(face))
    point_weights = bytearray(len(points))
    for index in range(point_start, point_start + 4):
        point_weights[index] = 1
    face_flags = bytearray(len(faces))
    face_flags[-2] = 1
    face_flags[-1] = 1
    selection = _NamedSelection("door1", bytes(point_weights), bytes(face_flags))
    return replace(
        lod,
        points=points,
        normals=normals,
        faces=faces,
        selections=lod.selections + (selection,),
    )


def _interior_memory_lod(key: BuildingVariantKey) -> _Lod | None:
    """Return the hinge axis and player action point for the animated entrance."""

    if not key.interiors:
        return None
    door_half, door_height, pivot_z = _door_dimensions(key)
    points = (
        (-door_half, 0.02, pivot_z),
        (-door_half, door_height - 0.02, pivot_z),
        (0.0, min(1.15, door_height * 0.55), -key.length_m * 0.5 - 0.30),
    )
    axis_weights = bytes((1, 1, 0))
    action_weights = bytes((0, 0, 1))
    selections = (
        _NamedSelection("door1_axis", axis_weights, b""),
        _NamedSelection("door1_action", action_weights, b""),
    )
    return _Lod(points, (), (), _MEMORY_LOD, selections=selections)


def _two_storey_paths_lod(
    key: BuildingVariantKey,
    foundation_depth: float,
    layout: _SecondStoreyLayout,
) -> _Lod:
    """Return an AI path strip from terrain through both interior levels."""

    half_length = key.length_m * 0.5
    door_half, _door_height, _pivot_z = _door_dimensions(key)
    half_path = min(INTERIOR_PATH_HALF_WIDTH_M, max(0.22, door_half * 0.55))
    stations: list[tuple[float, float, float]] = []  # x, y, z
    vehicle_ramp = _interior_vehicle_ramp_profile(key, foundation_depth)
    stair_profile = () if vehicle_ramp is not None else _interior_stair_profile(key, foundation_depth)
    if vehicle_ramp is not None:
        outer_z, inner_z, outer_y, inner_y = vehicle_ramp
        for index in range(4):
            t = index / 3.0
            stations.append((
                0.0,
                outer_y + (inner_y - outer_y) * t + 0.11,
                outer_z + (inner_z - outer_z) * t,
            ))
    elif stair_profile:
        for outer_z, inner_z, top_y, _bottom_y in reversed(stair_profile):
            stations.append((0.0, top_y + 0.11, (outer_z + inner_z) * 0.5))
    else:
        stations.append((0.0, 0.11, -half_length - 0.45))

    threshold_index = len(stations)
    stations.append((0.0, 0.11, -half_length + 0.20))
    ground_front_index = len(stations)
    ground_front_z = min(layout.stair_z0 - 0.55, -0.35)
    stations.append((0.0, 0.11, ground_front_z))
    stair_approach_index = len(stations)
    stations.append((layout.stair_center_x, 0.11, layout.stair_z0 - 0.22))

    stair_station_count = 5
    for index in range(stair_station_count):
        t = index / (stair_station_count - 1)
        stations.append((
            layout.stair_center_x,
            0.11 + layout.floor_y * t,
            layout.stair_z0 + (layout.stair_z1 - layout.stair_z0) * t,
        ))
    upper_top_index = len(stations)
    upper_z = min(half_length - 0.65, layout.stair_z1 + 0.35)
    stations.append((layout.stair_center_x, layout.floor_y + 0.11, upper_z))
    upper_centre_index = len(stations)
    stations.append((0.0, layout.floor_y + 0.11, upper_z))
    upper_rear_index = upper_centre_index
    rear_z = max(upper_z, half_length - 0.85)
    if rear_z > upper_z + 0.35:
        upper_rear_index = len(stations)
        stations.append((0.0, layout.floor_y + 0.11, rear_z))

    points: list[tuple[float, float, float]] = []
    for index, (x, y, z) in enumerate(stations):
        if index == 0:
            tx = stations[1][0] - x
            tz = stations[1][2] - z
        elif index == len(stations) - 1:
            tx = x - stations[index - 1][0]
            tz = z - stations[index - 1][2]
        else:
            tx = stations[index + 1][0] - stations[index - 1][0]
            tz = stations[index + 1][2] - stations[index - 1][2]
        length = math.hypot(tx, tz)
        if length <= 1.0e-6:
            px, pz = half_path, 0.0
        else:
            px = -tz / length * half_path
            pz = tx / length * half_path
        points.extend(((x + px, y, z + pz), (x, y, z), (x - px, y, z - pz)))

    faces: list[_Face] = []
    for station in range(len(stations) - 1):
        a = station * 3
        b = (station + 1) * 3
        for indices in (
            (a + 0, a + 1, b + 1),
            (a + 0, b + 1, b + 0),
            (a + 1, a + 2, b + 2),
            (a + 1, b + 2, b + 1),
        ):
            faces.append(_Face("", tuple((point, 0, 0.0, 0.0) for point in indices)))

    selections: list[_NamedSelection] = []
    point_count = len(points)
    face_count = len(faces)

    def selected_point(name: str, station_index: int) -> None:
        weights = bytearray(point_count)
        weights[station_index * 3 + 1] = 1
        selections.append(_NamedSelection(name, bytes(weights), bytes(face_count)))

    selected_point("In1", 0)
    selected_point("In2", threshold_index)
    for name, station_index in (
        ("Pos1", ground_front_index),
        ("Pos2", stair_approach_index),
        ("Pos3", upper_top_index),
        ("Pos4", upper_centre_index),
        ("Pos5", upper_rear_index),
    ):
        if not any(selection.name == name for selection in selections):
            selected_point(name, station_index)

    return _Lod(
        tuple(points),
        ((0.0, 1.0, 0.0),),
        tuple(faces),
        _PATHS_LOD,
        selections=tuple(selections),
    )


def _interior_paths_lod(
    key: BuildingVariantKey,
    foundation_depth: float,
) -> _Lod | None:
    """Return a compact AI path strip through the entrance and interior rooms.

    The path remains usable regardless of door animation state. CWA AI ignores
    Geometry while following a Paths LOD, so agents do not depend on invoking a
    player UserAction to traverse the selected door panel.
    """

    if not key.interiors:
        return None
    layout = _second_storey_layout(key)
    if layout is not None:
        return _two_storey_paths_lod(key, foundation_depth, layout)
    half_length = key.length_m * 0.5
    door_half, _door_height, _pivot_z = _door_dimensions(key)
    half_path = min(INTERIOR_PATH_HALF_WIDTH_M, max(0.22, door_half * 0.55))

    stations: list[tuple[float, float]] = []  # (y, z), outside -> inside
    vehicle_ramp = _interior_vehicle_ramp_profile(key, foundation_depth)
    stair_profile = () if vehicle_ramp is not None else _interior_stair_profile(key, foundation_depth)
    if vehicle_ramp is not None:
        outer_z, inner_z, outer_y, inner_y = vehicle_ramp
        for index in range(4):
            t = index / 3.0
            stations.append((
                outer_y + (inner_y - outer_y) * t + 0.11,
                outer_z + (inner_z - outer_z) * t,
            ))
    elif stair_profile:
        for outer_z, inner_z, top_y, _bottom_y in reversed(stair_profile):
            stations.append((top_y + 0.11, (outer_z + inner_z) * 0.5))
    else:
        stations.append((0.11, -half_length - 0.45))

    interior_stations = [
        (0.11, -half_length + 0.20),
        (0.11, -min(1.0, half_length * 0.28)),
    ]
    if key.width_m >= 6.0 and key.length_m >= 7.0:
        interior_stations.append((0.11, min(1.0, half_length * 0.28)))
    rear_z = max(0.25, half_length - 0.85)
    if rear_z > interior_stations[-1][1] + 0.35:
        interior_stations.append((0.11, rear_z))
    for station in interior_stations:
        if not stations or abs(station[1] - stations[-1][1]) > 0.05:
            stations.append(station)

    points: list[tuple[float, float, float]] = []
    for y, z in stations:
        points.extend((
            (-half_path, y, z),
            (0.0, y, z),
            (half_path, y, z),
        ))
    faces: list[_Face] = []
    for station in range(len(stations) - 1):
        a = station * 3
        b = (station + 1) * 3
        for indices in (
            (a + 0, a + 1, b + 1),
            (a + 0, b + 1, b + 0),
            (a + 1, a + 2, b + 2),
            (a + 1, b + 2, b + 1),
        ):
            faces.append(_Face("", tuple((index, 0, 0.0, 0.0) for index in indices)))

    selections: list[_NamedSelection] = []
    point_count = len(points)
    face_count = len(faces)

    def selected_point(name: str, point_index: int) -> None:
        weights = bytearray(point_count)
        weights[point_index] = 1
        selections.append(_NamedSelection(name, bytes(weights), bytes(face_count)))

    # Multiple access points make steep foundation stairs more tolerant of local
    # terrain variation. At least the threshold access point is always above the
    # terrain because procedural buildings are grounded from their highest
    # footprint support sample.
    selected_point("In1", 1)
    if len(stations) > 1:
        selected_point("In2", (len(stair_profile) if stair_profile else 1) * 3 + 1)
    first_interior = len(stair_profile) if stair_profile else 1
    for position_index, station_index in enumerate(
        range(first_interior, len(stations)), start=1
    ):
        selected_point(f"Pos{position_index}", station_index * 3 + 1)

    return _Lod(
        tuple(points),
        ((0.0, 1.0, 0.0),),
        tuple(faces),
        _PATHS_LOD,
        selections=tuple(selections),
    )



def _surface_normal(
    points: Sequence[tuple[float, float, float]], indices: Sequence[int]
) -> tuple[float, float, float]:
    """Return a stable unit normal for one generated roof polygon."""

    if len(indices) < 3:
        return (0.0, 1.0, 0.0)
    a = points[indices[0]]
    b = points[indices[1]]
    c = points[indices[2]]
    ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length <= 1.0e-9:
        return (0.0, 1.0, 0.0)
    return (nx / length, ny / length, nz / length)


def _append_special_roof(
    base: _Lod,
    key: BuildingVariantKey,
    roof_texture: str,
    *,
    eave_height: float,
    roof_top_height: float,
    vertical_offset: float = 0.0,
) -> _Lod:
    """Add hipped/pyramidal/dome/onion visual geometry above a flat wall shell.

    The base flat top intentionally remains beneath the pitched/curved roof. It
    acts as an attic ceiling and avoids introducing holes into enterable models.
    """

    points = list(base.points)
    normals = list(base.normals)
    faces = list(base.faces)
    half_width = key.width_m * 0.5
    half_length = key.length_m * 0.5
    base_y = eave_height + vertical_offset
    top_y = max(base_y + 0.25, roof_top_height + vertical_offset)
    roof_faces: list[_Face] = []

    def add_face(indices: tuple[int, ...]) -> None:
        normal_index = len(normals)
        normals.append(_surface_normal(points, indices))
        if len(indices) == 3:
            face = _triangle(roof_texture, indices, normal_index)
        elif len(indices) == 4:
            face = _quad(roof_texture, indices, normal_index, 1.0, 1.0)
        else:
            raise ValueError("special roof faces must be triangles or quads")
        roof_faces.extend(_double_sided_faces((face,)))

    # Eave corners follow the rectangular building shell exactly.
    corner_start = len(points)
    points.extend((
        (-half_width, base_y, -half_length),
        (half_width, base_y, -half_length),
        (half_width, base_y, half_length),
        (-half_width, base_y, half_length),
    ))
    c0, c1, c2, c3 = range(corner_start, corner_start + 4)

    if key.roof_style == "hipped":
        # For a long rectangle, shorten the ridge by roughly one half-width at
        # each end. Near-square footprints naturally collapse to a pyramid.
        ridge_half = max(0.0, half_length - half_width)
        if ridge_half <= 0.25:
            apex = len(points)
            points.append((0.0, top_y, 0.0))
            for tri in ((c0, apex, c1), (c1, apex, c2), (c2, apex, c3), (c3, apex, c0)):
                add_face(tri)
        else:
            r0 = len(points)
            r1 = r0 + 1
            points.extend(((0.0, top_y, -ridge_half), (0.0, top_y, ridge_half)))
            add_face((c0, r0, c1))
            add_face((c1, r0, r1, c2))
            add_face((c2, r1, c3))
            add_face((c3, r1, r0, c0))
    elif key.roof_style == "pyramidal":
        apex = len(points)
        points.append((0.0, top_y, 0.0))
        for tri in ((c0, apex, c1), (c1, apex, c2), (c2, apex, c3), (c3, apex, c0)):
            add_face(tri)
    elif key.roof_style in {"dome", "onion"}:
        sectors = 12
        if key.roof_style == "dome":
            # Bottom ring through three quarter-sphere bands to the crown.
            bands = (
                (0.0, 1.00),
                (0.34, 0.94),
                (0.66, 0.75),
                (0.88, 0.42),
            )
        else:
            # Characteristic onion silhouette: narrow foot, strong lower bulge,
            # then a rapidly tapering shoulder into a pointed crown.
            bands = (
                (0.0, 0.62),
                (0.18, 0.96),
                (0.46, 0.86),
                (0.72, 0.55),
                (0.90, 0.24),
            )
        ring_indices: list[list[int]] = []
        roof_rise = top_y - base_y
        for vertical_fraction, radius_fraction in bands:
            ring: list[int] = []
            y = base_y + roof_rise * vertical_fraction
            for sector in range(sectors):
                angle = 2.0 * math.pi * sector / sectors
                ring.append(len(points))
                points.append((
                    math.cos(angle) * half_width * radius_fraction,
                    y,
                    math.sin(angle) * half_length * radius_fraction,
                ))
            ring_indices.append(ring)
        for lower, upper in zip(ring_indices, ring_indices[1:]):
            for sector in range(sectors):
                nxt = (sector + 1) % sectors
                add_face((lower[sector], upper[sector], upper[nxt], lower[nxt]))
        apex = len(points)
        # Onion roofs get a slightly pointed finial-like top without becoming a
        # separate decorative object. Domes retain a softer cap.
        apex_extra = roof_rise * (0.10 if key.roof_style == "onion" else 0.0)
        points.append((0.0, top_y + apex_extra, 0.0))
        last = ring_indices[-1]
        for sector in range(sectors):
            nxt = (sector + 1) % sectors
            add_face((last[sector], apex, last[nxt]))
    else:
        raise ValueError(f"unsupported special roof style: {key.roof_style}")

    faces.extend(roof_faces)
    return _Lod(
        tuple(points), tuple(normals), tuple(faces), base.resolution,
        base.mass_per_point, base.selections, base.properties,
    )


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
    interior_storeys_override: int | None = None,
    _main_height_override: float | None = None,
) -> _Lod:
    front_texture = front_texture or wall_texture
    foundation_texture = foundation_texture or wall_texture
    interior_texture = interior_texture or wall_texture
    plain_wall_texture = plain_wall_texture or wall_texture
    # Enterable utility fronts must use *plain cladding* around the real door.
    # Cropping the closed-building front atlas around a physical doorway leaves
    # pieces of its painted garage/barn door on the jambs, which is exactly the
    # strange pale side-panel effect seen in CWA. The animated door itself uses
    # the same family-specific door art as the closed frontage, while the wall
    # around it uses the matching door-free material.
    interior_front_texture = (
        plain_wall_texture
        if key.family in UTILITY_INTERIOR_FAMILIES
        or key.family in _GROUND_FLOOR_ONLY_FACADE_FAMILIES
        else wall_texture
    )
    half_width = key.width_m / 2.0
    half_length = key.length_m / 2.0
    main_height = (
        _main_building_height(key)
        if _main_height_override is None
        else max(0.5, float(_main_height_override))
    )
    ground_floor_height = min(3.0, main_height)

    if key.roof_style in {"hipped", "pyramidal", "dome", "onion"}:
        # Reuse the mature flat-wall/interior implementation for the occupied
        # shell, then add the requested roof as separate visual geometry. This
        # preserves doors, windows, Roadway/Geometry compatibility and foundation
        # treatment instead of duplicating several hundred lines of wall logic.
        eave_height, _roof_rise, _slope_length = _gabled_profile(
            key,
            roof_pitch_degrees,
            interior_storeys_override=interior_storeys_override,
        )
        wall_key = replace(key, roof_style="flat")
        base = _visual_lod(
            wall_key, wall_texture, roof_texture, roof_pitch_degrees,
            front_texture=front_texture,
            foundation_texture=foundation_texture,
            foundation_depth=foundation_depth,
            church_plinth_height=church_plinth_height,
            interior_texture=interior_texture,
            window_trim_texture=window_trim_texture,
            plain_wall_texture=plain_wall_texture,
            interior_storeys_override=interior_storeys_override,
            _main_height_override=eave_height,
        )
        plinth_offset = (
            max(0.0, float(church_plinth_height)) if key.family == "church" else 0.0
        )
        return _append_special_roof(
            base, key, roof_texture,
            eave_height=eave_height,
            roof_top_height=_main_building_height(key),
            vertical_offset=plinth_offset,
        )

    if key.roof_style == "flat":
        height = main_height
        closed_front_ground_texture = _closed_facade_texture(
            key,
            front_texture,
            plain_wall_texture,
            span_m=key.width_m,
            height_m=ground_floor_height,
        )
        closed_front_upper_texture = _closed_facade_texture(
            key,
            (
                plain_wall_texture
                if key.family in _GROUND_FLOOR_ONLY_FACADE_FAMILIES
                or (
                    key.family in _PAINTED_WINDOW_FAMILIES
                    and plain_wall_texture != wall_texture
                )
                else wall_texture
            ),
            plain_wall_texture,
            span_m=key.width_m,
            height_m=max(0.0, height - ground_floor_height),
            upper_band=True,
        )
        closed_side_texture = _closed_facade_texture(
            key, wall_texture, plain_wall_texture,
            span_m=key.length_m, height_m=height,
        )
        closed_back_texture = _closed_facade_texture(
            key, wall_texture, plain_wall_texture,
            span_m=key.width_m, height_m=height,
        )
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
                key,
                points,
                half_width=half_width,
                front_z=-half_length,
                wall_top=height,
                outer_bottom_left=0,
                outer_top_left=4,
                outer_top_right=5,
                outer_bottom_right=1,
                texture=interior_front_texture,
                normal=0,
                upper_texture=(
                    plain_wall_texture if plain_wall_texture != wall_texture else None
                ),
            )
            front_faces.extend(doorway_faces)
        elif key.family in _PAINTED_WINDOW_FAMILIES and key.family != "church" and plain_wall_texture != wall_texture:
            points, closed_front_faces = _closed_wall_storey_faces(
                key,
                points,
                lower_left=0,
                upper_left=4,
                upper_right=5,
                lower_right=1,
                wall_height=height,
                span_m=key.width_m,
                ground_texture=front_texture,
                upper_texture=wall_texture,
                plain_texture=plain_wall_texture,
                normal=0,
                u_scale=key.width_m / 4.0,
                ground_u_scale=1.0,
                preserve_ground_texture=key.isolated_dwelling,
            )
            front_faces.extend(closed_front_faces)
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
                _wall_quad_ground_anchored(
                    closed_front_ground_texture, (0, ground_left, ground_right, 1), 0,
                    u_scale=1.0, vertical_min=0.0, vertical_max=ground_floor_height,
                ),
                _wall_quad_ground_anchored(
                    closed_front_upper_texture,
                    (ground_left, 4, 5, ground_right), 0,
                    u_scale=key.width_m / 4.0,
                    vertical_min=ground_floor_height, vertical_max=height,
                ),
            ))
        else:
            front_faces.append(_wall_quad_ground_anchored(
                closed_front_ground_texture, (0, 4, 5, 1), 0,
                u_scale=1.0, vertical_min=0.0, vertical_max=height,
            ))
        if key.interiors:
            side_openings = _interior_window_openings(key, -half_length, half_length, height)
            back_openings = _interior_window_openings(key, -half_width, half_width, height)
            points, right_faces = _wall_faces_with_openings(
                points, horizontal_min=-half_length, horizontal_max=half_length,
                plane=half_width, wall_top=height, horizontal_axis="z",
                openings=side_openings, texture=wall_texture, normal=1,
                upper_texture=(
                    plain_wall_texture if plain_wall_texture != wall_texture else None
                ),
                upper_texture_from_y=_interior_painted_facade_from_y(key),
            )
            points, back_faces = _wall_faces_with_openings(
                points, horizontal_min=-half_width, horizontal_max=half_width,
                plane=half_length, wall_top=height, horizontal_axis="x",
                openings=back_openings, texture=wall_texture, normal=2,
                upper_texture=(
                    plain_wall_texture if plain_wall_texture != wall_texture else None
                ),
                upper_texture_from_y=_interior_painted_facade_from_y(key),
            )
            points, left_faces = _wall_faces_with_openings(
                points, horizontal_min=-half_length, horizontal_max=half_length,
                plane=-half_width, wall_top=height, horizontal_axis="z",
                openings=side_openings, texture=wall_texture, normal=3,
                upper_texture=(
                    plain_wall_texture if plain_wall_texture != wall_texture else None
                ),
                upper_texture_from_y=_interior_painted_facade_from_y(key),
            )
            exterior_side_faces = right_faces + back_faces + left_faces
        elif key.family in _PAINTED_WINDOW_FAMILIES and key.family != "church" and plain_wall_texture != wall_texture:
            points, right_faces = _closed_wall_storey_faces(
                key,
                points,
                lower_left=1,
                upper_left=5,
                upper_right=6,
                lower_right=2,
                wall_height=height,
                span_m=key.length_m,
                ground_texture=wall_texture,
                upper_texture=wall_texture,
                plain_texture=plain_wall_texture,
                normal=1,
                u_scale=key.length_m / 4.0,
            )
            points, back_faces = _closed_wall_storey_faces(
                key,
                points,
                lower_left=2,
                upper_left=6,
                upper_right=7,
                lower_right=3,
                wall_height=height,
                span_m=key.width_m,
                ground_texture=wall_texture,
                upper_texture=wall_texture,
                plain_texture=plain_wall_texture,
                normal=2,
                u_scale=key.width_m / 4.0,
            )
            points, left_faces = _closed_wall_storey_faces(
                key,
                points,
                lower_left=3,
                upper_left=7,
                upper_right=4,
                lower_right=0,
                wall_height=height,
                span_m=key.length_m,
                ground_texture=wall_texture,
                upper_texture=wall_texture,
                plain_texture=plain_wall_texture,
                normal=3,
                u_scale=key.length_m / 4.0,
            )
            exterior_side_faces = right_faces + back_faces + left_faces
        elif (
            (
                key.family in _GROUND_FLOOR_ONLY_FACADE_FAMILIES
                or (
                    key.family in _PAINTED_WINDOW_FAMILIES
                    and plain_wall_texture != wall_texture
                )
            )
            and ground_floor_height < height - 1e-6
        ):
            points, right_faces = _split_closed_wall_at_height(
                points, lower_left=1, upper_left=5, upper_right=6, lower_right=2,
                split_height=ground_floor_height, wall_top=height,
                lower_texture=closed_side_texture, upper_texture=plain_wall_texture,
                normal=1, u_scale=key.length_m / 4.0,
            )
            points, back_faces = _split_closed_wall_at_height(
                points, lower_left=2, upper_left=6, upper_right=7, lower_right=3,
                split_height=ground_floor_height, wall_top=height,
                lower_texture=closed_back_texture, upper_texture=plain_wall_texture,
                normal=2, u_scale=key.width_m / 4.0,
            )
            points, left_faces = _split_closed_wall_at_height(
                points, lower_left=3, upper_left=7, upper_right=4, lower_right=0,
                split_height=ground_floor_height, wall_top=height,
                lower_texture=closed_side_texture, upper_texture=plain_wall_texture,
                normal=3, u_scale=key.length_m / 4.0,
            )
            exterior_side_faces = right_faces + back_faces + left_faces
        else:
            exterior_side_faces = (
                _wall_quad_ground_anchored(
                    closed_side_texture, (1, 5, 6, 2), 1,
                    u_scale=key.length_m / 4.0, vertical_min=0.0, vertical_max=height,
                ),
                _wall_quad_ground_anchored(
                    closed_back_texture, (2, 6, 7, 3), 2,
                    u_scale=key.width_m / 4.0, vertical_min=0.0, vertical_max=height,
                ),
                _wall_quad_ground_anchored(
                    closed_side_texture, (3, 7, 4, 0), 3,
                    u_scale=key.length_m / 4.0, vertical_min=0.0, vertical_max=height,
                ),
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
            points, faces = _add_window_crosses(
                key,
                points,
                faces,
                wall_top=height,
                texture=(
                    window_trim_texture
                    if window_trim_texture
                    and key.regional_style in WHITE_WINDOW_TRIM_REGIONAL_STYLES
                    else interior_texture
                ),
            )
            points, normals, faces = _add_simple_interior_visuals(
                key,
                points,
                normals,
                faces,
                wall_texture=interior_texture,
                floor_texture=foundation_texture,
                wall_top=height,
            )
            points, normals, faces = _add_entrance_stairs(
                key, points, normals, faces,
                foundation_depth=foundation_depth, texture=foundation_texture,
            )
        plinth_height = max(0.0, float(church_plinth_height)) if key.family == "church" else 0.0
        if plinth_height > 0.0:
            points = tuple((x, y + plinth_height, z) for x, y, z in points)
        foundation_top = plinth_height + (
            FOUNDATION_VISIBLE_REVEAL_M if foundation_depth > 0.0 else 0.0
        )
        points, faces = _add_foundation_skirt(
            points, faces, half_width=half_width, half_length=half_length,
            texture=foundation_texture, depth=foundation_depth, top_height=foundation_top,
        )
        return _Lod(points, normals, _double_sided_faces(faces), 1.0)

    eave_height, roof_rise, slope_length = _gabled_profile(
        key,
        roof_pitch_degrees,
        interior_storeys_override=interior_storeys_override,
    )
    points = (
        (-half_width, 0.0, -half_length), (half_width, 0.0, -half_length),
        (half_width, 0.0, half_length), (-half_width, 0.0, half_length),
        (-half_width, eave_height, -half_length), (half_width, eave_height, -half_length),
        (half_width, eave_height, half_length), (-half_width, eave_height, half_length),
        (0.0, main_height, -half_length), (0.0, main_height, half_length),
    )
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
    closed_front_ground_texture = _closed_facade_texture(
        key,
        front_texture,
        plain_wall_texture,
        span_m=key.width_m,
        height_m=ground_floor_height,
    )
    closed_front_upper_texture = _closed_facade_texture(
        key,
        (
            plain_wall_texture
            if key.family == "church"
            or key.family in _GROUND_FLOOR_ONLY_FACADE_FAMILIES
            or (
                key.family in _PAINTED_WINDOW_FAMILIES
                and plain_wall_texture != wall_texture
            )
            else wall_texture
        ),
        plain_wall_texture,
        span_m=key.width_m,
        height_m=max(0.0, eave_height - ground_floor_height),
        upper_band=True,
    )
    closed_side_texture = _closed_facade_texture(
        key, wall_texture, plain_wall_texture,
        span_m=key.length_m, height_m=eave_height,
    )
    closed_back_texture = _closed_facade_texture(
        key, wall_texture, plain_wall_texture,
        span_m=key.width_m, height_m=eave_height,
    )
    if key.interiors:
        points, doorway_faces = _front_faces_with_doorway(
            key,
            points,
            half_width=half_width,
            front_z=-half_length,
            wall_top=eave_height,
            outer_bottom_left=0,
            outer_top_left=4,
            outer_top_right=5,
            outer_bottom_right=1,
            texture=interior_front_texture,
            normal=0,
        )
        front_faces.extend(doorway_faces)
    elif (
        key.family in _PAINTED_WINDOW_FAMILIES
        and key.family != "church"
        and plain_wall_texture != wall_texture
    ):
        points, closed_front_faces = _closed_wall_storey_faces(
            key,
            points,
            lower_left=0,
            upper_left=4,
            upper_right=5,
            lower_right=1,
            wall_height=eave_height,
            span_m=key.width_m,
            ground_texture=front_texture,
            upper_texture=wall_texture,
            plain_texture=plain_wall_texture,
            normal=0,
            u_scale=key.width_m / 4.0,
            ground_u_scale=1.0,
            preserve_ground_texture=key.isolated_dwelling,
        )
        front_faces.extend(closed_front_faces)
    elif ground_floor_height < eave_height - 1e-6:
        ground_left = len(points)
        ground_right = ground_left + 1
        points = points + (
            (-half_width, ground_floor_height, -half_length),
            (half_width, ground_floor_height, -half_length),
        )
        front_faces.extend((
            _wall_quad_ground_anchored(
                closed_front_ground_texture, (0, ground_left, ground_right, 1), 0,
                u_scale=1.0, vertical_min=0.0, vertical_max=ground_floor_height,
            ),
            _wall_quad_ground_anchored(
                closed_front_upper_texture,
                (ground_left, 4, 5, ground_right), 0,
                u_scale=key.width_m / 4.0,
                vertical_min=ground_floor_height, vertical_max=eave_height,
            ),
        ))
    else:
        front_faces.append(_wall_quad_ground_anchored(
            closed_front_ground_texture, (0, 4, 5, 1), 0,
            u_scale=1.0, vertical_min=0.0, vertical_max=eave_height,
        ))

    if key.interiors:
        side_openings = _interior_window_openings(key, -half_length, half_length, eave_height)
        back_openings = _interior_window_openings(key, -half_width, half_width, eave_height)
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
    elif (
        key.family in _PAINTED_WINDOW_FAMILIES
        and key.family != "church"
        and plain_wall_texture != wall_texture
    ):
        points, right_faces = _closed_wall_storey_faces(
            key,
            points,
            lower_left=1,
            upper_left=5,
            upper_right=6,
            lower_right=2,
            wall_height=eave_height,
            span_m=key.length_m,
            ground_texture=wall_texture,
            upper_texture=wall_texture,
            plain_texture=plain_wall_texture,
            normal=1,
            u_scale=key.length_m / 4.0,
        )
        points, back_faces = _closed_wall_storey_faces(
            key,
            points,
            lower_left=2,
            upper_left=6,
            upper_right=7,
            lower_right=3,
            wall_height=eave_height,
            span_m=key.width_m,
            ground_texture=wall_texture,
            upper_texture=wall_texture,
            plain_texture=plain_wall_texture,
            normal=2,
            u_scale=key.width_m / 4.0,
        )
        points, left_faces = _closed_wall_storey_faces(
            key,
            points,
            lower_left=3,
            upper_left=7,
            upper_right=4,
            lower_right=0,
            wall_height=eave_height,
            span_m=key.length_m,
            ground_texture=wall_texture,
            upper_texture=wall_texture,
            plain_texture=plain_wall_texture,
            normal=3,
            u_scale=key.length_m / 4.0,
        )
    elif (
        (
            key.family in _GROUND_FLOOR_ONLY_FACADE_FAMILIES
            or (
                key.family in _PAINTED_WINDOW_FAMILIES
                and plain_wall_texture != wall_texture
            )
        )
        and ground_floor_height < eave_height - 1e-6
    ):
        points, right_faces = _split_closed_wall_at_height(
            points, lower_left=1, upper_left=5, upper_right=6, lower_right=2,
            split_height=ground_floor_height, wall_top=eave_height,
            lower_texture=closed_side_texture, upper_texture=plain_wall_texture,
            normal=1, u_scale=key.length_m / 4.0,
        )
        points, back_faces = _split_closed_wall_at_height(
            points, lower_left=2, upper_left=6, upper_right=7, lower_right=3,
            split_height=ground_floor_height, wall_top=eave_height,
            lower_texture=closed_back_texture, upper_texture=plain_wall_texture,
            normal=2, u_scale=key.width_m / 4.0,
        )
        points, left_faces = _split_closed_wall_at_height(
            points, lower_left=3, upper_left=7, upper_right=4, lower_right=0,
            split_height=ground_floor_height, wall_top=eave_height,
            lower_texture=closed_side_texture, upper_texture=plain_wall_texture,
            normal=3, u_scale=key.length_m / 4.0,
        )
    else:
        right_faces = (_wall_quad_ground_anchored(
            closed_side_texture, (1, 5, 6, 2), 1,
            u_scale=key.length_m / 4.0, vertical_min=0.0, vertical_max=eave_height,
        ),)
        back_faces = (_wall_quad_ground_anchored(
            closed_back_texture, (2, 6, 7, 3), 2,
            u_scale=key.width_m / 4.0, vertical_min=0.0, vertical_max=eave_height,
        ),)
        left_faces = (_wall_quad_ground_anchored(
            closed_side_texture, (3, 7, 4, 0), 3,
            u_scale=key.length_m / 4.0, vertical_min=0.0, vertical_max=eave_height,
        ),)

    gable_texture = (
        _closed_facade_texture(
            key,
            wall_texture,
            plain_wall_texture,
            span_m=key.width_m,
            height_m=roof_rise,
            upper_band=True,
            gable=(
                key.family != "church"
                and key.family not in _GROUND_FLOOR_ONLY_FACADE_FAMILIES
            ),
        )
        if not (
            key.family == "church"
            or key.family in _GROUND_FLOOR_ONLY_FACADE_FAMILIES
        )
        else plain_wall_texture
    )
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
        points, faces = _add_window_crosses(
            key,
            points,
            faces,
            wall_top=eave_height,
            texture=(
                window_trim_texture
                if window_trim_texture
                and key.regional_style in WHITE_WINDOW_TRIM_REGIONAL_STYLES
                else interior_texture
            ),
        )
        points, normals, faces = _add_simple_interior_visuals(
            key,
            points,
            normals,
            faces,
            wall_texture=interior_texture,
            floor_texture=foundation_texture,
            wall_top=eave_height,
        )
        points, normals, faces = _add_entrance_stairs(
            key, points, normals, faces,
            foundation_depth=foundation_depth, texture=foundation_texture,
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
    foundation_top = plinth_height + (
        FOUNDATION_VISIBLE_REVEAL_M if foundation_depth > 0.0 else 0.0
    )
    points, faces = _add_foundation_skirt(
        points, faces, half_width=half_width, half_length=half_length,
        texture=foundation_texture, depth=foundation_depth, top_height=foundation_top,
    )
    return _Lod(points, normals, _double_sided_faces(faces), 1.0)


def _polygon_native_shape(key: BuildingVariantKey) -> Polygon:
    return Polygon(
        tuple(key.footprint_vertices),
        [tuple(ring) for ring in key.footprint_holes if len(ring) >= 3],
    )


def _iter_polygonal_geometries(geometry: Any) -> Iterable[Polygon]:
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    for part in getattr(geometry, "geoms", ()):
        yield from _iter_polygonal_geometries(part)


def _ring_coordinates_from_geometry(geometry: Any) -> list[PointXZ]:
    points: list[PointXZ] = []
    for polygon in _iter_polygonal_geometries(geometry):
        points.extend((float(x), float(z)) for x, z in list(polygon.exterior.coords)[:-1])
        for interior in polygon.interiors:
            points.extend((float(x), float(z)) for x, z in list(interior.coords)[:-1])
        representative = polygon.representative_point()
        points.append((float(representative.x), float(representative.y)))
    return points


def _line_sample_points_inside_shape(line_geometry: Any, spacing: float = 4.0) -> list[PointXZ]:
    result: list[PointXZ] = []
    geometries = (
        tuple(getattr(line_geometry, "geoms", ()))
        if getattr(line_geometry, "geom_type", "") in {"MultiLineString", "GeometryCollection"}
        else (line_geometry,)
    )
    for geometry in geometries:
        if getattr(geometry, "geom_type", "") == "LineString":
            length = float(geometry.length)
            if length <= 1.0e-8:
                continue
            count = max(1, int(math.ceil(length / max(0.5, spacing))))
            for index in range(count + 1):
                point = geometry.interpolate(length * index / count)
                result.append((float(point.x), float(point.y)))
        elif getattr(geometry, "geom_type", "") == "Point":
            result.append((float(geometry.x), float(geometry.y)))
    return result


def _triangulate_polygon_with_samples(
    shape: Polygon,
    samples: Sequence[PointXZ],
) -> tuple[tuple[PointXZ, PointXZ, PointXZ], ...]:
    points: list[PointXZ] = [
        (float(x), float(z)) for x, z in list(shape.exterior.coords)[:-1]
    ]
    for ring in shape.interiors:
        points.extend((float(x), float(z)) for x, z in list(ring.coords)[:-1])
    points.extend((float(x), float(z)) for x, z in samples)
    # Stable de-duplication avoids feeding GEOS hundreds of coincident offset
    # corners on small footprints.
    unique = tuple(dict.fromkeys((round(x, 6), round(z, 6)) for x, z in points))
    if len(unique) < 3:
        return ()
    triangles: list[tuple[PointXZ, PointXZ, PointXZ]] = []
    for candidate in triangulate(MultiPoint(unique)):
        if candidate.area <= 1.0e-8:
            continue
        clipped = candidate.intersection(shape)
        for part in _iter_polygonal_geometries(clipped):
            if part.area <= 1.0e-8:
                continue
            part_outer = tuple(
                (float(x), float(z)) for x, z in list(part.exterior.coords)[:-1]
            )
            part_holes = tuple(
                tuple((float(x), float(z)) for x, z in list(ring.coords)[:-1])
                for ring in part.interiors
            )
            triangles.extend(
                _triangulate_polygon_coordinates(part_outer, part_holes)
            )
    covered = sum(abs(_polygon_signed_area(triangle)) for triangle in triangles)
    if not triangles or abs(covered - shape.area) > max(0.05, shape.area * 1.0e-5):
        return ()
    return tuple(triangles)


def _polygon_native_sectioned_roof(
    key: BuildingVariantKey,
    shape: Polygon,
    *,
    eave_height: float,
    roof_rise: float,
    roof_pitch_degrees: float,
) -> tuple[
    tuple[tuple[PointXZ, PointXZ, PointXZ], ...],
    Callable[[PointXZ], float],
] | None:
    """Build connected wing roofs for common orthogonal L/T/U footprints.

    The legacy rectangle decomposition is used only as a *roof construction
    guide*. The world still receives one semantic building/P3D. Taking the
    maximum of each wing's local roof field naturally creates valley lines at
    intersections without throwing a single ridge through the entire concave
    outline.
    """

    if key.footprint_holes or key.roof_style not in {"gabled", "hipped", "pyramidal"}:
        return None
    rectangles = decompose_footprint_rectangles(
        key.footprint_vertices,
        max_parts=6,
        minimum_part_size=2.0,
        rectangular_fill_threshold=0.96,
    )
    if len(rectangles) < 2:
        return None
    bounds: list[tuple[float, float, float, float]] = []
    coverage: Polygon | None = None
    for rectangle in rectangles:
        part = Polygon(rectangle).intersection(shape)
        if part.is_empty or part.area < 3.0:
            continue
        bx0, bz0, bx1, bz1 = part.bounds
        if bx1 - bx0 < 1.0 or bz1 - bz0 < 1.0:
            continue
        bounds.append((float(bx0), float(bz0), float(bx1), float(bz1)))
        coverage = part if coverage is None else coverage.union(part)
    if len(bounds) < 2 or coverage is None:
        return None
    # Do not use sectioned fields when the conservative decomposition missed a
    # material part of the building. The generic sampled roof remains safer.
    if shape.difference(coverage).area > max(0.5, shape.area * 0.035):
        return None

    pitch = math.tan(math.radians(max(1.0, roof_pitch_degrees)))
    samples: list[PointXZ] = []
    for x0, z0, x1, z1 in bounds:
        sx, sz = x1 - x0, z1 - z0
        cx, cz = (x0 + x1) * 0.5, (z0 + z1) * 0.5
        half_short = max(0.05, min(sx, sz) * 0.5)
        if sx <= sz:
            if key.roof_style == "gabled":
                ridge = LineString(((cx, z0), (cx, z1)))
            else:
                ridge = LineString(((cx, min(z1, z0 + half_short)), (cx, max(z0, z1 - half_short))))
        else:
            if key.roof_style == "gabled":
                ridge = LineString(((x0, cz), (x1, cz)))
            else:
                ridge = LineString(((min(x1, x0 + half_short), cz), (max(x0, x1 - half_short), cz)))
        samples.extend(_line_sample_points_inside_shape(ridge.intersection(shape), spacing=2.5))
        samples.append((cx, cz))

    tolerance = 0.08

    def sectioned_height(point: PointXZ) -> float:
        px, pz = float(point[0]), float(point[1])
        heights: list[float] = []
        for x0, z0, x1, z1 in bounds:
            if not (x0 - tolerance <= px <= x1 + tolerance and z0 - tolerance <= pz <= z1 + tolerance):
                continue
            sx, sz = x1 - x0, z1 - z0
            cx, cz = (x0 + x1) * 0.5, (z0 + z1) * 0.5
            if key.roof_style == "gabled":
                if sx <= sz:
                    half = max(0.05, sx * 0.5)
                    fraction = 1.0 - abs(px - cx) / half
                else:
                    half = max(0.05, sz * 0.5)
                    fraction = 1.0 - abs(pz - cz) / half
                local_rise = min(roof_rise, half * pitch)
                heights.append(eave_height + local_rise * max(0.0, min(1.0, fraction)))
            else:
                distance = max(0.0, min(px - x0, x1 - px, pz - z0, z1 - pz))
                local_rise = min(roof_rise, min(sx, sz) * 0.5 * pitch)
                heights.append(eave_height + min(local_rise, distance * pitch))
        return max(heights, default=eave_height)

    triangles = _triangulate_polygon_with_samples(shape, samples)
    if not triangles:
        return None
    return triangles, sectioned_height


def _polygon_native_roof_mesh(
    key: BuildingVariantKey,
    roof_pitch_degrees: float,
) -> tuple[float, tuple[tuple[PointXZ, PointXZ, PointXZ], ...], Callable[[PointXZ], float]]:
    """Return eave height, roof triangles and a deterministic roof height field."""

    shape = _polygon_native_shape(key)
    if shape.is_empty or not shape.is_valid:
        raise ValueError("polygon-native roof requires a valid footprint")
    main_height = _main_building_height(key)
    if key.roof_style == "flat":
        triangles = _triangulate_polygon_coordinates(
            key.footprint_vertices, key.footprint_holes
        )
        return main_height, triangles, lambda _point: main_height

    eave_height, roof_rise, _slope = _gabled_profile(key, roof_pitch_degrees)
    if roof_rise <= 1.0e-6:
        triangles = _triangulate_polygon_coordinates(
            key.footprint_vertices, key.footprint_holes
        )
        return eave_height, triangles, lambda _point: eave_height

    sectioned = _polygon_native_sectioned_roof(
        key,
        shape,
        eave_height=eave_height,
        roof_rise=roof_rise,
        roof_pitch_degrees=roof_pitch_degrees,
    )
    if sectioned is not None:
        triangles, height_field = sectioned
        return eave_height, triangles, height_field

    min_x, min_z, max_x, max_z = shape.bounds
    if key.roof_style == "gabled":
        ridge_x = (min_x + max_x) * 0.5
        half_span = max(0.1, (max_x - min_x) * 0.5)

        def roof_height(point: PointXZ) -> float:
            fraction = 1.0 - abs(float(point[0]) - ridge_x) / half_span
            return eave_height + roof_rise * max(0.0, min(1.0, fraction))

        ridge_line = LineString(((ridge_x, min_z - 2.0), (ridge_x, max_z + 2.0)))
        samples = _line_sample_points_inside_shape(
            ridge_line.intersection(shape), spacing=3.5
        )
        triangles = _triangulate_polygon_with_samples(shape, samples)
        if triangles:
            return eave_height, triangles, roof_height
        triangles = _triangulate_polygon_coordinates(
            key.footprint_vertices, key.footprint_holes
        )
        return main_height, triangles, lambda _point: main_height

    pitch_tangent = math.tan(math.radians(max(1.0, roof_pitch_degrees)))
    run_to_top = roof_rise / max(1.0e-6, pitch_tangent)

    # A true single-apex pyramid is safe for convex footprints. Concave shells
    # and courtyards use the same inward-distance field as a hip roof, which
    # creates connected ridges/valleys instead of throwing triangles across air.
    if key.roof_style == "pyramidal" and not key.footprint_holes and shape.equals(shape.convex_hull):
        apex = shape.centroid
        apex_point = (float(apex.x), float(apex.y))

        def pyramidal_height(point: PointXZ) -> float:
            if math.hypot(point[0] - apex_point[0], point[1] - apex_point[1]) <= 1.0e-5:
                return eave_height + roof_rise
            return eave_height

        triangles = _triangulate_polygon_with_samples(shape, (apex_point,))
        if triangles:
            return eave_height, triangles, pyramidal_height

    boundary = shape.boundary
    samples: list[PointXZ] = []
    for fraction in (0.25, 0.50, 0.75, 1.0):
        distance = run_to_top * fraction
        if distance <= 0.05:
            continue
        inset = shape.buffer(-distance, join_style=2)
        samples.extend(_ring_coordinates_from_geometry(inset))
    representative = shape.representative_point()
    samples.append((float(representative.x), float(representative.y)))
    maximum_sample_distance = max(
        (
            float(boundary.distance(Point(float(x), float(z))))
            for x, z in samples
        ),
        default=0.0,
    )
    # Thin courtyard rings can have a smaller inradius than the nominal fitted
    # width used to choose roof rise. Preserve the source/model total height by
    # steepening the sampled hip just enough to reach the intended ridge rather
    # than silently producing a building several metres too short.
    effective_tangent = max(
        pitch_tangent,
        roof_rise / max(1.0e-6, maximum_sample_distance),
    )

    def hipped_height(point: PointXZ) -> float:
        distance = float(boundary.distance(Point(float(point[0]), float(point[1]))))
        return eave_height + min(roof_rise, distance * effective_tangent)

    triangles = _triangulate_polygon_with_samples(shape, samples)
    if triangles:
        return eave_height, triangles, hipped_height

    # Geometry that defeats the sampled roof mesh still stays one faithful
    # building. A flat top is intentionally preferable to resurrecting multiple
    # overlapping rectangular wings.
    triangles = _triangulate_polygon_coordinates(
        key.footprint_vertices, key.footprint_holes
    )
    return main_height, triangles, lambda _point: main_height


def _polygon_native_front_edge(key: BuildingVariantKey) -> int:
    """Return the selected exterior entrance edge for one native footprint."""

    outer = tuple(key.footprint_vertices)
    if not outer:
        return -1
    selected = int(key.entrance_edge)
    if 0 <= selected < len(outer):
        return selected
    # Stable legacy fallback: favour the lowest local-Z wall, then its length.
    return min(
        range(len(outer)),
        key=lambda index: (
            (outer[index][1] + outer[(index + 1) % len(outer)][1]) * 0.5,
            -math.hypot(
                outer[(index + 1) % len(outer)][0] - outer[index][0],
                outer[(index + 1) % len(outer)][1] - outer[index][1],
            ),
        ),
    )


def _polygon_native_edge_frame(
    start: PointXZ, end: PointXZ
) -> tuple[float, float, float, float, float]:
    """Return span, tangent X/Z and filled-side inward normal X/Z."""

    dx = float(end[0]) - float(start[0])
    dz = float(end[1]) - float(start[1])
    span = math.hypot(dx, dz)
    if span <= 1.0e-8:
        return 0.0, 1.0, 0.0, 0.0, 1.0
    tx, tz = dx / span, dz / span
    # Native outer rings are CCW and courtyard rings CW. In both cases the
    # occupied building material lies to the left of the authored ring edge.
    return span, tx, tz, -tz, tx


def _polygon_native_door_opening(
    key: BuildingVariantKey,
    edge_index: int,
    span: float,
    wall_top: float,
) -> tuple[float, float, float, float] | None:
    if edge_index != _polygon_native_front_edge(key) or span < 1.35:
        return None
    door_half, door_height, _pivot = _door_dimensions(key)
    jamb = 0.22 if key.family in UTILITY_INTERIOR_FAMILIES else 0.28
    usable_half = max(0.42, min(door_half, span * 0.5 - jamb))
    fraction = max(0.0, min(1.0, float(key.entrance_fraction)))
    centre = fraction * span
    centre = max(usable_half + jamb, min(span - usable_half - jamb, centre))
    top = min(door_height, max(1.9, wall_top - 0.18))
    return (centre - usable_half, centre + usable_half, 0.0, top)


def _polygon_native_edge_openings(
    key: BuildingVariantKey,
    edge_index: int,
    span: float,
    wall_top: float,
    *,
    courtyard: bool = False,
) -> tuple[tuple[float, float, float, float], ...]:
    """Return realistic physical apertures along one arbitrary facade edge."""

    if not key.interiors:
        return ()
    door = None if courtyard else _polygon_native_door_opening(
        key, edge_index, span, wall_top
    )
    exclusions: tuple[tuple[float, float], ...] = ()
    if door is not None:
        exclusions = ((max(0.0, door[0] - 0.38), min(span, door[1] + 0.38)),)
    windows: tuple[tuple[float, float, float, float], ...] = ()
    if key.family not in UTILITY_INTERIOR_FAMILIES and span >= 2.4:
        visible_storeys = _visible_window_storey_count(key, wall_top=wall_top)
        windows = _window_openings(
            0.0,
            span,
            wall_top,
            ground_exclusions=exclusions,
            maximum_storeys=visible_storeys,
            storey_height_m=VISIBLE_FACADE_STOREY_HEIGHT_M,
        )
    return windows + ((door,) if door is not None else ())


def _polygon_native_partition_segments(key: BuildingVariantKey) -> tuple[tuple[PointXZ, PointXZ], ...]:
    """Return a conservative room divider clipped to the actual footprint."""

    if key.family in UTILITY_INTERIOR_FAMILIES or not key.interiors:
        return ()
    shape = _polygon_native_shape(key)
    inset = _interior_wall_thickness(key) + 0.08
    inner = shape.buffer(-inset, join_style=2)
    if inner.is_empty or inner.area < 24.0:
        return ()
    representative = inner.representative_point()
    min_x, min_z, max_x, max_z = inner.bounds
    # The native model's X axis follows the minimum rectangle's width axis, so
    # a horizontal divider normally separates front/back rooms while remaining
    # stable across L/T/U footprints. Clip it rather than inventing walls across
    # courtyards or concave notches.
    line = LineString(((min_x - 2.0, representative.y), (max_x + 2.0, representative.y)))
    cut = inner.intersection(line)
    geometries = (
        tuple(getattr(cut, "geoms", ()))
        if getattr(cut, "geom_type", "") in {"MultiLineString", "GeometryCollection"}
        else (cut,)
    )
    segments: list[tuple[PointXZ, PointXZ]] = []
    for geometry in geometries:
        if getattr(geometry, "geom_type", "") != "LineString" or geometry.length < 3.0:
            continue
        coords = list(geometry.coords)
        start = (float(coords[0][0]), float(coords[0][1]))
        end = (float(coords[-1][0]), float(coords[-1][1]))
        if math.hypot(end[0] - start[0], end[1] - start[1]) >= 3.0:
            segments.append((start, end))
    return tuple(segments[:2])


def _polygon_native_visual_lod(
    key: BuildingVariantKey,
    wall_texture: str,
    roof_texture: str,
    *,
    roof_pitch_degrees: float = 35.0,
    front_texture: str | None = None,
    foundation_texture: str | None = None,
    foundation_depth: float = 0.0,
    interior_texture: str | None = None,
    window_trim_texture: str | None = None,
    plain_wall_texture: str | None = None,
    door_texture: str | None = None,
) -> _Lod:
    """Build one footprint-faithful shell, including a usable ground interior."""

    outer = tuple(key.footprint_vertices)
    holes = tuple(tuple(ring) for ring in key.footprint_holes)
    if len(outer) < 3:
        raise ValueError("polygon-native building requires an exterior footprint")
    shape = _polygon_native_shape(key)
    if shape.is_empty or not shape.is_valid or shape.area <= 0.0:
        raise ValueError("polygon-native building requires a valid footprint")

    wall_texture = wall_texture or ""
    front_texture = front_texture or wall_texture
    foundation_texture = foundation_texture or wall_texture
    interior_texture = interior_texture or plain_wall_texture or wall_texture
    door_texture = door_texture or front_texture
    eave_height, roof_triangles, roof_height = _polygon_native_roof_mesh(
        key, roof_pitch_degrees
    )
    if not roof_triangles:
        raise ValueError("polygon-native building requires a triangulatable roof")

    points: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    faces: list[_Face] = []
    selections: list[_NamedSelection] = []
    selected_front_edge = _polygon_native_front_edge(key)
    wall_thickness = _interior_wall_thickness(key) if key.interiors else 0.0

    def edge_point(
        start: PointXZ,
        tx: float,
        tz: float,
        inward_x: float,
        inward_z: float,
        horizontal: float,
        vertical: float,
        inset: float = 0.0,
    ) -> tuple[float, float, float]:
        return (
            float(start[0]) + tx * horizontal + inward_x * inset,
            vertical,
            float(start[1]) + tz * horizontal + inward_z * inset,
        )

    def add_edge_wall(
        start: PointXZ,
        end: PointXZ,
        *,
        ring_index: int,
        edge_index: int,
        bottom: float = 0.0,
        top: float = eave_height,
        openings: Sequence[tuple[float, float, float, float]] | None = None,
        exterior_texture: str | None = None,
        interior_wall_texture: str | None = None,
    ) -> None:
        span, tx, tz, inward_x, inward_z = _polygon_native_edge_frame(start, end)
        if span <= 1.0e-5 or top <= bottom + 1.0e-6:
            return
        if openings is None:
            openings = _polygon_native_edge_openings(
                key, edge_index, span, top, courtyard=ring_index > 0
            )
        texture = exterior_texture or wall_texture
        outer_normal = len(normals)
        normals.append((-inward_x, 0.0, -inward_z))
        inner_normal = len(normals)
        normals.append((inward_x, 0.0, inward_z))

        if (
            not key.interiors
            and not openings
            and bottom >= -1.0e-6
            and key.family in _PAINTED_WINDOW_FAMILIES
            and plain_wall_texture
        ):
            ground_texture = wall_texture if ring_index == 0 else plain_wall_texture
            upper_texture = wall_texture if ring_index == 0 else plain_wall_texture
            for y0, y1, band_texture, windowed in _closed_facade_bands(
                key,
                top,
                span_m=span,
                ground_texture=ground_texture,
                upper_texture=upper_texture,
                plain_texture=plain_wall_texture,
            ):
                base = len(points)
                points.extend((
                    edge_point(start, tx, tz, inward_x, inward_z, 0.0, y0),
                    edge_point(start, tx, tz, inward_x, inward_z, 0.0, y1),
                    edge_point(start, tx, tz, inward_x, inward_z, span, y1),
                    edge_point(start, tx, tz, inward_x, inward_z, span, y0),
                ))
                if windowed:
                    inset = FACADE_WINDOW_UV_INSET
                    faces.append(_Face(band_texture, (
                        (base + 0, outer_normal, 0.0, 1.0 - inset),
                        (base + 1, outer_normal, 0.0, inset),
                        (base + 2, outer_normal, max(1.0, span / 4.0), inset),
                        (base + 3, outer_normal, max(1.0, span / 4.0), 1.0 - inset),
                    )))
                else:
                    faces.append(_Face(band_texture, (
                        (base + 0, outer_normal, 0.0, max(0.25, (y1 - y0) / 3.0)),
                        (base + 1, outer_normal, 0.0, 0.0),
                        (base + 2, outer_normal, max(1.0, span / 4.0), 0.0),
                        (base + 3, outer_normal, max(1.0, span / 4.0), max(0.25, (y1 - y0) / 3.0)),
                    )))
            return
        if (
            not openings
            and plain_wall_texture
            and texture == wall_texture
            and ring_index == 0
            and span < MIN_NATIVE_EDGE_WINDOW_TEXTURE_SPAN_M
            and key.family in _PAINTED_WINDOW_FAMILIES
        ):
            texture = plain_wall_texture
        if not openings and bottom < 0.0:
            solid_rectangles = ((0.0, span, bottom, top),)
        else:
            solid_rectangles = _solid_wall_rectangles(0.0, span, top, openings)
        for h0, h1, y0, y1 in solid_rectangles:
            y0 = max(bottom, y0)
            y1 = min(top, y1)
            if h1 - h0 <= 1.0e-6 or y1 - y0 <= 1.0e-6:
                continue
            base = len(points)
            points.extend((
                edge_point(start, tx, tz, inward_x, inward_z, h0, y0),
                edge_point(start, tx, tz, inward_x, inward_z, h0, y1),
                edge_point(start, tx, tz, inward_x, inward_z, h1, y1),
                edge_point(start, tx, tz, inward_x, inward_z, h1, y0),
            ))
            faces.append(_Face(texture, (
                (base + 0, outer_normal, h0 / 4.0, (top - y0) / 3.0),
                (base + 1, outer_normal, h0 / 4.0, (top - y1) / 3.0),
                (base + 2, outer_normal, h1 / 4.0, (top - y1) / 3.0),
                (base + 3, outer_normal, h1 / 4.0, (top - y0) / 3.0),
            )))
            if key.interiors:
                inner_base = len(points)
                points.extend((
                    edge_point(start, tx, tz, inward_x, inward_z, h0, y0, wall_thickness),
                    edge_point(start, tx, tz, inward_x, inward_z, h1, y0, wall_thickness),
                    edge_point(start, tx, tz, inward_x, inward_z, h1, y1, wall_thickness),
                    edge_point(start, tx, tz, inward_x, inward_z, h0, y1, wall_thickness),
                ))
                faces.append(_Face(interior_wall_texture or interior_texture, (
                    (inner_base + 0, inner_normal, h0 / 4.0, (top - y0) / 3.0),
                    (inner_base + 1, inner_normal, h1 / 4.0, (top - y0) / 3.0),
                    (inner_base + 2, inner_normal, h1 / 4.0, (top - y1) / 3.0),
                    (inner_base + 3, inner_normal, h0 / 4.0, (top - y1) / 3.0),
                )))

        if key.interiors and wall_thickness > 0.0:
            # Add jamb/sill/lintel reveals for every genuine aperture. Doorways
            # deliberately have no sill across the threshold.
            for h0, h1, y0, y1 in openings:
                reveal_normal = len(normals)
                normals.append((0.0, 1.0, 0.0))
                for horizontal, reverse in ((h0, False), (h1, True)):
                    base = len(points)
                    outer_low = edge_point(start, tx, tz, inward_x, inward_z, horizontal, y0)
                    outer_high = edge_point(start, tx, tz, inward_x, inward_z, horizontal, y1)
                    inner_low = edge_point(start, tx, tz, inward_x, inward_z, horizontal, y0, wall_thickness)
                    inner_high = edge_point(start, tx, tz, inward_x, inward_z, horizontal, y1, wall_thickness)
                    quad = (outer_low, outer_high, inner_high, inner_low)
                    points.extend(reversed(quad) if reverse else quad)
                    faces.append(_Face(interior_texture, tuple(
                        (base + index, reveal_normal, (index & 1), (index >> 1))
                        for index in range(4)
                    )))
                # Lintel.
                base = len(points)
                points.extend((
                    edge_point(start, tx, tz, inward_x, inward_z, h0, y1),
                    edge_point(start, tx, tz, inward_x, inward_z, h1, y1),
                    edge_point(start, tx, tz, inward_x, inward_z, h1, y1, wall_thickness),
                    edge_point(start, tx, tz, inward_x, inward_z, h0, y1, wall_thickness),
                ))
                faces.append(_Face(interior_texture, tuple(
                    (base + index, reveal_normal, (index & 1), (index >> 1))
                    for index in range(4)
                )))
                if y0 > 0.05:
                    base = len(points)
                    points.extend((
                        edge_point(start, tx, tz, inward_x, inward_z, h0, y0),
                        edge_point(start, tx, tz, inward_x, inward_z, h0, y0, wall_thickness),
                        edge_point(start, tx, tz, inward_x, inward_z, h1, y0, wall_thickness),
                        edge_point(start, tx, tz, inward_x, inward_z, h1, y0),
                    ))
                    faces.append(_Face(interior_texture, tuple(
                        (base + index, reveal_normal, (index & 1), (index >> 1))
                        for index in range(4)
                    )))

            # Native walls use physical window apertures, so give them physical
            # frames too instead of relying on a rectangular facade atlas to
            # imply scale. The strips are intentionally shallow and cheap.
            trim_texture = window_trim_texture or interior_texture
            frame_width = 0.075
            mullion_width = 0.045
            for h0, h1, y0, y1 in openings:
                if y0 <= 0.05 or y1 - y0 < 0.55 or h1 - h0 < 0.55:
                    continue
                normal_index = len(normals)
                normals.append((-inward_x, 0.0, -inward_z))

                def add_trim_rect(th0: float, th1: float, ty0: float, ty1: float) -> None:
                    if th1 <= th0 or ty1 <= ty0:
                        return
                    base = len(points)
                    outward = -0.022
                    points.extend((
                        edge_point(start, tx, tz, inward_x, inward_z, th0, ty0, outward),
                        edge_point(start, tx, tz, inward_x, inward_z, th0, ty1, outward),
                        edge_point(start, tx, tz, inward_x, inward_z, th1, ty1, outward),
                        edge_point(start, tx, tz, inward_x, inward_z, th1, ty0, outward),
                    ))
                    faces.append(_Face(trim_texture, (
                        (base + 0, normal_index, 0.0, 1.0),
                        (base + 1, normal_index, 0.0, 0.0),
                        (base + 2, normal_index, 1.0, 0.0),
                        (base + 3, normal_index, 1.0, 1.0),
                    )))

                add_trim_rect(h0 - frame_width, h0 + frame_width, y0 - frame_width, y1 + frame_width)
                add_trim_rect(h1 - frame_width, h1 + frame_width, y0 - frame_width, y1 + frame_width)
                add_trim_rect(h0, h1, y0 - frame_width, y0 + frame_width)
                add_trim_rect(h0, h1, y1 - frame_width, y1 + frame_width)
                centre_h = (h0 + h1) * 0.5
                centre_y = (y0 + y1) * 0.5
                add_trim_rect(centre_h - mullion_width, centre_h + mullion_width, y0, y1)
                add_trim_rect(h0, h1, centre_y - mullion_width, centre_y + mullion_width)

    rings = (outer, *holes)
    for ring_index, ring in enumerate(rings):
        for edge_index in range(len(ring)):
            start = ring[edge_index]
            end = ring[(edge_index + 1) % len(ring)]
            add_edge_wall(start, end, ring_index=ring_index, edge_index=edge_index)

            # Close any gable/hip rise above the common eave. Sampling the edge
            # midpoint as well as its ends handles connected wing roofs much more
            # reliably than the old one-global-ridge special case.
            if key.roof_style != "flat":
                span, tx, tz, _ix, _iz = _polygon_native_edge_frame(start, end)
                sample_distances = [0.0, span * 0.5, span]
                samples = [
                    (start[0] + tx * distance, start[1] + tz * distance)
                    for distance in sample_distances
                ]
                heights = [roof_height(sample) for sample in samples]
                for index in range(2):
                    if max(heights[index], heights[index + 1]) <= eave_height + 1.0e-5:
                        continue
                    a, b = samples[index], samples[index + 1]
                    span2, tx2, tz2, inward_x2, inward_z2 = _polygon_native_edge_frame(a, b)
                    if span2 <= 1.0e-6:
                        continue
                    base = len(points)
                    points.extend((
                        (a[0], eave_height, a[1]),
                        (a[0], heights[index], a[1]),
                        (b[0], heights[index + 1], b[1]),
                        (b[0], eave_height, b[1]),
                    ))
                    normal_index = len(normals)
                    normals.append((-inward_x2, 0.0, -inward_z2))
                    gable_wall_texture = (
                        plain_wall_texture
                        if not key.interiors
                        and key.family in _PAINTED_WINDOW_FAMILIES
                        and plain_wall_texture
                        else wall_texture
                    )
                    faces.append(_Face(gable_wall_texture, (
                        (base + 0, normal_index, 0.0, 1.0),
                        (base + 1, normal_index, 0.0, 0.0),
                        (base + 2, normal_index, span2 / 4.0, 0.0),
                        (base + 3, normal_index, span2 / 4.0, 1.0),
                    )))

    # Roof triangles follow the complete footprint and preserve courtyard holes.
    for triangle in roof_triangles:
        base = len(points)
        roof_points = tuple(
            (float(x), float(roof_height((x, z))), float(z)) for x, z in triangle
        )
        points.extend(roof_points)
        normal_index = len(normals)
        normals.append(_surface_normal(points, (base, base + 1, base + 2)))
        faces.append(_Face(roof_texture, (
            (base, normal_index, triangle[0][0] / 4.0, triangle[0][1] / 4.0),
            (base + 1, normal_index, triangle[1][0] / 4.0, triangle[1][1] / 4.0),
            (base + 2, normal_index, triangle[2][0] / 4.0, triangle[2][1] / 4.0),
        )))

    # Keep the shell watertight even if a sampled pitched-roof mesh leaves a
    # tiny uncovered sliver near a ridge or valley. The cap sits just beneath
    # the eaves, so it is invisible in normal cases and only appears when it is
    # preventing a genuine hole from exposing the sky.
    if key.roof_style != "flat":
        cap_texture = interior_texture if key.interiors else (plain_wall_texture or wall_texture)
        cap_y = max(0.05, eave_height - 0.03)
        for triangle in _triangulate_polygon_coordinates(outer, holes):
            base = len(points)
            points.extend(((x, cap_y, z) for x, z in triangle))
            up = len(normals)
            normals.append((0.0, 1.0, 0.0))
            faces.append(_Face(cap_texture, (
                (base + 0, up, triangle[0][0] / 3.0, triangle[0][1] / 3.0),
                (base + 1, up, triangle[1][0] / 3.0, triangle[1][1] / 3.0),
                (base + 2, up, triangle[2][0] / 3.0, triangle[2][1] / 3.0),
            )))

    if key.interiors:
        inner = shape.buffer(-max(0.08, wall_thickness), join_style=2)
        ceiling_y = min(eave_height - 0.06, max(2.35, min(2.75, eave_height - 0.06)))
        if not inner.is_empty and ceiling_y > 2.1:
            for part in _iter_polygonal_geometries(inner):
                part_outer = tuple((float(x), float(z)) for x, z in list(part.exterior.coords)[:-1])
                part_holes = tuple(
                    tuple((float(x), float(z)) for x, z in list(ring.coords)[:-1])
                    for ring in part.interiors
                )
                for triangle in _triangulate_polygon_coordinates(part_outer, part_holes):
                    # Walkable-looking visual floor. The actual character contact
                    # is supplied by the Roadway LOD below.
                    base = len(points)
                    points.extend(tuple((x, INTERIOR_VISUAL_FLOOR_Y_M, z) for x, z in triangle))
                    up = len(normals); normals.append((0.0, 1.0, 0.0))
                    faces.append(_Face(foundation_texture, (
                        (base + 0, up, triangle[0][0] / 3.0, triangle[0][1] / 3.0),
                        (base + 1, up, triangle[1][0] / 3.0, triangle[1][1] / 3.0),
                        (base + 2, up, triangle[2][0] / 3.0, triangle[2][1] / 3.0),
                    )))
                    base = len(points)
                    points.extend(tuple((x, ceiling_y, z) for x, z in reversed(triangle)))
                    down = len(normals); normals.append((0.0, -1.0, 0.0))
                    faces.append(_Face(interior_texture, (
                        (base + 0, down, 0.0, 0.0),
                        (base + 1, down, 1.0, 0.0),
                        (base + 2, down, 1.0, 1.0),
                    )))

            # A restrained room divider gives houses an actual interior layout
            # without pretending we know survey-grade room boundaries from OSM.
            for partition_start, partition_end in _polygon_native_partition_segments(key):
                span, _tx, _tz, _ix, _iz = _polygon_native_edge_frame(partition_start, partition_end)
                if span < 3.0:
                    continue
                doorway_half = min(0.48, max(0.42, span * 0.10))
                centre = span * 0.5
                openings = ((centre - doorway_half, centre + doorway_half, 0.0, 2.05),)
                add_edge_wall(
                    partition_start,
                    partition_end,
                    ring_index=0,
                    edge_index=-999,
                    top=ceiling_y,
                    openings=openings,
                    exterior_texture=interior_texture,
                    interior_wall_texture=interior_texture,
                )

    # Add a correctly scaled entrance panel on the actual mapped facade. Closed
    # native buildings use it as a shallow overlay; enterable models use the same
    # panel as the animated door selected by door1.
    if selected_front_edge >= 0:
        start = outer[selected_front_edge]
        end = outer[(selected_front_edge + 1) % len(outer)]
        span, tx, tz, inward_x, inward_z = _polygon_native_edge_frame(start, end)
        door = _polygon_native_door_opening(key, selected_front_edge, span, eave_height)
        if door is None and span >= 1.35:
            # Closed models still need a human-scale door even though they have
            # no physical doorway aperture.
            door_half, door_height, _ = _door_dimensions(key)
            jamb = 0.28
            usable_half = max(0.42, min(door_half, span * 0.5 - jamb))
            centre = max(usable_half + jamb, min(span - usable_half - jamb, key.entrance_fraction * span))
            door = (centre - usable_half, centre + usable_half, 0.0, min(door_height, eave_height - 0.12))
        if door is not None:
            h0, h1, _y0, y1 = door
            outward = -0.018 if key.interiors else -0.025
            base = len(points)
            points.extend((
                edge_point(start, tx, tz, inward_x, inward_z, h0, 0.03, outward),
                edge_point(start, tx, tz, inward_x, inward_z, h0, y1 - 0.03, outward),
                edge_point(start, tx, tz, inward_x, inward_z, h1, y1 - 0.03, outward),
                edge_point(start, tx, tz, inward_x, inward_z, h1, 0.03, outward),
            ))
            normal_index = len(normals)
            normals.append((-inward_x, 0.0, -inward_z))
            face = _Face(door_texture, (
                (base + 0, normal_index, 0.0, 1.0),
                (base + 1, normal_index, 0.0, 0.0),
                (base + 2, normal_index, 1.0, 0.0),
                (base + 3, normal_index, 1.0, 1.0),
            ))
            faces.append(face)
            if key.interiors:
                point_weights = bytearray(len(points))
                for index in range(base, base + 4):
                    point_weights[index] = 1
                # Selection face flags are finalized after double-sided faces are
                # created below, so retain the original front-face index here.
                selected_face_index = len(faces) - 1

    depth = max(0.0, float(foundation_depth))
    if depth > 0.0:
        foundation_top = FOUNDATION_VISIBLE_REVEAL_M
        for ring_index, ring in enumerate(rings):
            for edge_index in range(len(ring)):
                add_edge_wall(
                    ring[edge_index], ring[(edge_index + 1) % len(ring)],
                    ring_index=ring_index, edge_index=edge_index,
                    bottom=-depth, top=foundation_top, openings=(),
                    exterior_texture=foundation_texture,
                    interior_wall_texture=foundation_texture,
                )

    doubled = _double_sided_faces(faces)
    if key.interiors and 'selected_face_index' in locals():
        # _double_sided_faces stores each original followed by its reverse.
        face_flags = bytearray(len(doubled))
        face_flags[selected_face_index * 2] = 1
        face_flags[selected_face_index * 2 + 1] = 1
        point_weights = bytearray(len(points))
        for index in range(base, base + 4):
            point_weights[index] = 1
        selections.append(_NamedSelection("door1", bytes(point_weights), bytes(face_flags)))
    return _Lod(tuple(points), tuple(normals), doubled, 1.0, selections=tuple(selections))

def _polygon_native_hollow_geometry_lod(key: BuildingVariantKey) -> _Lod:
    """Build an enterable arbitrary-footprint collision shell.

    Each wall cell is one convex oriented box, so concave footprints and
    courtyards never become one illegal concave Geometry component. Genuine
    window/door openings use the same aperture layout as the visual LOD.
    """

    shape = _polygon_native_shape(key)
    if shape.is_empty or not shape.is_valid:
        raise ValueError("polygon-native interior collision requires a valid footprint")
    eave_height, _triangles, _roof_height = _polygon_native_roof_mesh(key, 35.0)
    wall_top = max(2.4, eave_height)
    wall_thickness = _interior_wall_thickness(key)
    points: list[tuple[float, float, float]] = []
    faces: list[_Face] = []
    component_ranges: list[tuple[range, range]] = []

    def add_prism(
        start: PointXZ,
        end: PointXZ,
        h0: float,
        h1: float,
        y0: float,
        y1: float,
        *,
        thickness: float = wall_thickness,
    ) -> tuple[range, range] | None:
        span, tx, tz, inward_x, inward_z = _polygon_native_edge_frame(start, end)
        if span <= 1.0e-6 or h1 <= h0 + 1.0e-6 or y1 <= y0 + 1.0e-6:
            return None
        # Keep arbitrary wall components inside the same practical legacy-engine
        # span target as the old rectangular collision path.
        count = max(1, int(math.ceil((h1 - h0) / (_MAX_GEOMETRY_COMPONENT_SPAN_M - 2.0))))
        result: tuple[range, range] | None = None
        for piece in range(count):
            ph0 = h0 + (h1 - h0) * piece / count
            ph1 = h0 + (h1 - h0) * (piece + 1) / count
            outer0 = (start[0] + tx * ph0, start[1] + tz * ph0)
            outer1 = (start[0] + tx * ph1, start[1] + tz * ph1)
            inner0 = (outer0[0] + inward_x * thickness, outer0[1] + inward_z * thickness)
            inner1 = (outer1[0] + inward_x * thickness, outer1[1] + inward_z * thickness)
            point_start = len(points)
            face_start = len(faces)
            points.extend((
                (outer0[0], y0, outer0[1]), (outer1[0], y0, outer1[1]),
                (inner1[0], y0, inner1[1]), (inner0[0], y0, inner0[1]),
                (outer0[0], y1, outer0[1]), (outer1[0], y1, outer1[1]),
                (inner1[0], y1, inner1[1]), (inner0[0], y1, inner0[1]),
            ))
            for indices in (
                (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3),
                (3, 7, 4, 0), (4, 7, 6, 5), (0, 1, 2, 3),
            ):
                faces.append(_Face("", tuple(
                    (point_start + index, -1, 0.0, 0.0) for index in indices
                )))
            result = (
                range(point_start, point_start + 8),
                range(face_start, face_start + 6),
            )
            component_ranges.append(result)
        return result

    outer = tuple(key.footprint_vertices)
    rings = (outer, *tuple(tuple(ring) for ring in key.footprint_holes))
    selected_front = _polygon_native_front_edge(key)
    for ring_index, ring in enumerate(rings):
        for edge_index in range(len(ring)):
            start = ring[edge_index]
            end = ring[(edge_index + 1) % len(ring)]
            span, _tx, _tz, _ix, _iz = _polygon_native_edge_frame(start, end)
            openings = _polygon_native_edge_openings(
                key, edge_index, span, wall_top, courtyard=ring_index > 0
            )
            for h0, h1, y0, y1 in _solid_wall_rectangles(0.0, span, wall_top, openings):
                add_prism(start, end, h0, h1, y0, y1)

    # Add one conservative room divider for non-utility buildings. Its visual
    # doorway and collision doorway share the same 0.84-0.96 m clear opening.
    for partition_start, partition_end in _polygon_native_partition_segments(key):
        span, _tx, _tz, _ix, _iz = _polygon_native_edge_frame(partition_start, partition_end)
        doorway_half = min(0.48, max(0.42, span * 0.10))
        centre = span * 0.5
        partition_top = min(wall_top - 0.10, 2.70)
        openings = ((centre - doorway_half, centre + doorway_half, 0.0, 2.10),)
        for h0, h1, y0, y1 in _solid_wall_rectangles(0.0, span, partition_top, openings):
            add_prism(partition_start, partition_end, h0, h1, y0, y1, thickness=max(0.10, wall_thickness * 0.75))

    # Door is a separate selected collision component so the engine animation
    # opens the physical doorway as well as the visual panel.
    door_component: tuple[range, range] | None = None
    if selected_front >= 0:
        start = outer[selected_front]
        end = outer[(selected_front + 1) % len(outer)]
        span, _tx, _tz, _ix, _iz = _polygon_native_edge_frame(start, end)
        door = _polygon_native_door_opening(key, selected_front, span, wall_top)
        if door is not None:
            door_component = add_prism(
                start, end, door[0], door[1], 0.03, door[3] - 0.03,
                thickness=INTERIOR_DOOR_THICKNESS_M,
            )

    selections: list[_NamedSelection] = []
    for index, (point_range, face_range) in enumerate(component_ranges, start=1):
        point_weights = bytearray(len(points))
        face_flags = bytearray(len(faces))
        for point_index in point_range:
            point_weights[point_index] = 1
        for face_index in face_range:
            face_flags[face_index] = 1
        selections.append(_NamedSelection(
            f"component{index:02d}", bytes(point_weights), bytes(face_flags)
        ))
    if door_component is not None:
        point_weights = bytearray(len(points))
        face_flags = bytearray(len(faces))
        for point_index in door_component[0]:
            point_weights[point_index] = 1
        for face_index in door_component[1]:
            face_flags[face_index] = 1
        selections.append(_NamedSelection("door1", bytes(point_weights), bytes(face_flags)))

    total_mass = max(1000.0, float(shape.area) * max(1.0, wall_top) * 35.0)
    mass_per_point = tuple(total_mass / max(1, len(points)) for _ in points)
    return _Lod(
        tuple(points), (), tuple(faces), _GEOMETRY_LOD,
        mass_per_point, tuple(selections),
        (("map", _map_symbol_for_family(key.family)), ("autocenter", "0")),
    )


def _polygon_native_geometry_lod(key: BuildingVariantKey) -> _Lod:
    """Build collision from bounded convex pieces, preserving courtyards."""

    if key.interiors:
        return _polygon_native_hollow_geometry_lod(key)

    shape = _polygon_native_shape(key)
    triangles: list[tuple[PointXZ, PointXZ, PointXZ]] = []
    min_x, min_z, max_x, max_z = shape.bounds
    # A triangle inside a square can span the square diagonal. Keep the clipping
    # cells below 40/sqrt(2) so every convex prism remains inside the same
    # practical component span used by the rectangular collision generator.
    cell_span = _MAX_GEOMETRY_COMPONENT_SPAN_M / math.sqrt(2.0)
    x_segments = max(1, int(math.ceil((max_x - min_x) / cell_span)))
    z_segments = max(1, int(math.ceil((max_z - min_z) / cell_span)))
    x_step = max(1.0e-6, (max_x - min_x) / x_segments)
    z_step = max(1.0e-6, (max_z - min_z) / z_segments)
    for z_index in range(z_segments):
        z0 = min_z + z_index * z_step
        z1 = min_z + (z_index + 1) * z_step
        for x_index in range(x_segments):
            x0 = min_x + x_index * x_step
            x1 = min_x + (x_index + 1) * x_step
            clip = Polygon(((x0, z0), (x1, z0), (x1, z1), (x0, z1)))
            clipped = shape.intersection(clip)
            for part in _iter_polygonal_geometries(clipped):
                part_outer = tuple(
                    (float(x), float(z)) for x, z in list(part.exterior.coords)[:-1]
                )
                part_holes = tuple(
                    tuple((float(x), float(z)) for x, z in list(ring.coords)[:-1])
                    for ring in part.interiors
                )
                triangles.extend(_triangulate_polygon_coordinates(part_outer, part_holes))
    if not triangles:
        raise ValueError("polygon-native collision requires a triangulatable footprint")
    height = _main_building_height(key)
    points: list[tuple[float, float, float]] = []
    faces: list[_Face] = []
    component_ranges: list[tuple[range, range]] = []

    for triangle in triangles:
        point_start = len(points)
        face_start = len(faces)
        points.extend((
            (triangle[0][0], 0.0, triangle[0][1]),
            (triangle[1][0], 0.0, triangle[1][1]),
            (triangle[2][0], 0.0, triangle[2][1]),
            (triangle[0][0], height, triangle[0][1]),
            (triangle[1][0], height, triangle[1][1]),
            (triangle[2][0], height, triangle[2][1]),
        ))
        faces.extend((
            _Face("", tuple((point_start + point, -1, 0.0, 0.0) for point in (0, 3, 4, 1))),
            _Face("", tuple((point_start + point, -1, 0.0, 0.0) for point in (1, 4, 5, 2))),
            _Face("", tuple((point_start + point, -1, 0.0, 0.0) for point in (2, 5, 3, 0))),
            _Face("", tuple((point_start + point, -1, 0.0, 0.0) for point in (3, 5, 4))),
            _Face("", tuple((point_start + point, -1, 0.0, 0.0) for point in (0, 1, 2))),
        ))
        component_ranges.append((
            range(point_start, point_start + 6),
            range(face_start, face_start + 5),
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
            f"component{index:02d}", bytes(point_weights), bytes(face_flags)
        ))

    polygon_area = max(1.0, float(shape.area))
    total_mass = max(1000.0, polygon_area * max(1.0, height) * 60.0)
    mass_per_point = tuple(total_mass / len(points) for _ in points)
    return _Lod(
        tuple(points), (), tuple(faces), _GEOMETRY_LOD,
        mass_per_point,
        tuple(selections),
        (("map", _map_symbol_for_family(key.family)), ("autocenter", "0")),
    )


def _polygon_native_land_contact_lod(key: BuildingVariantKey) -> _Lod:
    points: list[tuple[float, float, float]] = []

    def append_ring(ring: Sequence[PointXZ]) -> None:
        if not ring:
            return
        count = len(ring)
        for index, start in enumerate(ring):
            end = ring[(index + 1) % count]
            points.append((float(start[0]), 0.0, float(start[1])))
            mid_x = (float(start[0]) + float(end[0])) * 0.5
            mid_z = (float(start[1]) + float(end[1])) * 0.5
            points.append((mid_x, 0.0, mid_z))

    append_ring(key.footprint_vertices)
    for ring in key.footprint_holes:
        append_ring(ring)
    shape = _polygon_native_shape(key)
    representative = shape.representative_point()
    points.append((float(representative.x), 0.0, float(representative.y)))
    # Stable de-duplication keeps the LOD compact while retaining useful edge
    # support points for irregular outlines.
    points = list(dict.fromkeys((round(x, 4), round(y, 4), round(z, 4)) for x, y, z in points))
    return _Lod(tuple(points), (), (), _LAND_CONTACT_LOD)


def _polygon_native_roadway_lod(
    key: BuildingVariantKey, foundation_depth: float
) -> _Lod | None:
    """Return a walkable floor and entrance threshold for native interiors."""

    if not key.interiors:
        return None
    shape = _polygon_native_shape(key)
    inset = _interior_wall_thickness(key) + INTERIOR_ROADWAY_WALL_CLEARANCE_M
    inner = shape.buffer(-inset, join_style=2)
    if inner.is_empty:
        inner = shape.buffer(-max(0.05, inset * 0.5), join_style=2)
    points: list[tuple[float, float, float]] = []
    faces: list[_Face] = []
    roadway_y = INTERIOR_ROADWAY_Y_M
    for part in _iter_polygonal_geometries(inner):
        outer = tuple((float(x), float(z)) for x, z in list(part.exterior.coords)[:-1])
        holes = tuple(
            tuple((float(x), float(z)) for x, z in list(ring.coords)[:-1])
            for ring in part.interiors
        )
        for triangle in _triangulate_polygon_coordinates(outer, holes):
            start = len(points)
            points.extend(tuple((x, roadway_y, z) for x, z in triangle))
            faces.append(_Face("", tuple(
                (start + index, 0, 0.0, 0.0) for index in (0, 2, 1)
            )))

    outer_ring = tuple(key.footprint_vertices)
    edge_index = _polygon_native_front_edge(key)
    if edge_index >= 0 and outer_ring:
        start_edge = outer_ring[edge_index]
        end_edge = outer_ring[(edge_index + 1) % len(outer_ring)]
        span, tx, tz, inward_x, inward_z = _polygon_native_edge_frame(start_edge, end_edge)
        door = _polygon_native_door_opening(key, edge_index, span, _main_building_height(key))
        if door is not None:
            h0, h1, _bottom, _top = door
            margin = min(0.18, max(0.08, (h1 - h0) * 0.08))
            h0 += margin
            h1 -= margin
            run = max(0.55, max(0.0, float(foundation_depth)) * INTERIOR_VEHICLE_RAMP_RUN_PER_RISE)
            inside = _interior_wall_thickness(key) + 0.38
            outside_y = roadway_y - max(0.0, float(foundation_depth))
            base = len(points)
            points.extend((
                (
                    start_edge[0] + tx * h0 - inward_x * run,
                    outside_y,
                    start_edge[1] + tz * h0 - inward_z * run,
                ),
                (
                    start_edge[0] + tx * h1 - inward_x * run,
                    outside_y,
                    start_edge[1] + tz * h1 - inward_z * run,
                ),
                (
                    start_edge[0] + tx * h1 + inward_x * inside,
                    roadway_y,
                    start_edge[1] + tz * h1 + inward_z * inside,
                ),
                (
                    start_edge[0] + tx * h0 + inward_x * inside,
                    roadway_y,
                    start_edge[1] + tz * h0 + inward_z * inside,
                ),
            ))
            faces.append(_Face("", tuple(
                (base + index, 0, 0.0, 0.0) for index in (0, 3, 2, 1)
            )))

    if not faces:
        return None
    return _Lod(tuple(points), ((0.0, 1.0, 0.0),), tuple(faces), _ROADWAY_LOD)


def _polygon_native_memory_lod(key: BuildingVariantKey) -> _Lod | None:
    """Return door hinge/action points aligned to the mapped polygon facade."""

    if not key.interiors or not key.footprint_vertices:
        return None
    outer = tuple(key.footprint_vertices)
    edge_index = _polygon_native_front_edge(key)
    if edge_index < 0:
        return None
    start = outer[edge_index]
    end = outer[(edge_index + 1) % len(outer)]
    span, tx, tz, inward_x, inward_z = _polygon_native_edge_frame(start, end)
    door = _polygon_native_door_opening(key, edge_index, span, _main_building_height(key))
    if door is None:
        return None
    h0, h1, _bottom, top = door
    hinge_h = h0
    action_h = (h0 + h1) * 0.5
    points = (
        (start[0] + tx * hinge_h, 0.02, start[1] + tz * hinge_h),
        (start[0] + tx * hinge_h, top - 0.02, start[1] + tz * hinge_h),
        (
            start[0] + tx * action_h - inward_x * 0.32,
            min(1.15, top * 0.55),
            start[1] + tz * action_h - inward_z * 0.32,
        ),
    )
    return _Lod(
        points, (), (), _MEMORY_LOD,
        selections=(
            _NamedSelection("door1_axis", bytes((1, 1, 0)), b""),
            _NamedSelection("door1_action", bytes((0, 0, 1)), b""),
        ),
    )


def _polygon_native_paths_lod(
    key: BuildingVariantKey, foundation_depth: float
) -> _Lod | None:
    """Return a small AI strip through the arbitrary entrance into the shell."""

    if not key.interiors or not key.footprint_vertices:
        return None
    shape = _polygon_native_shape(key)
    outer = tuple(key.footprint_vertices)
    edge_index = _polygon_native_front_edge(key)
    if edge_index < 0:
        return None
    start = outer[edge_index]
    end = outer[(edge_index + 1) % len(outer)]
    span, tx, tz, inward_x, inward_z = _polygon_native_edge_frame(start, end)
    door = _polygon_native_door_opening(key, edge_index, span, _main_building_height(key))
    if door is None:
        return None
    centre_h = (door[0] + door[1]) * 0.5
    centre = (start[0] + tx * centre_h, start[1] + tz * centre_h)
    half_path = min(INTERIOR_PATH_HALF_WIDTH_M, max(0.22, (door[1] - door[0]) * 0.28))
    run = max(0.45, max(0.0, float(foundation_depth)) * 2.0)
    stations: list[tuple[float, float, float]] = [
        (
            centre[0] - inward_x * run,
            0.11 - max(0.0, float(foundation_depth)),
            centre[1] - inward_z * run,
        ),
        (
            centre[0] + inward_x * (_interior_wall_thickness(key) + 0.35),
            0.11,
            centre[1] + inward_z * (_interior_wall_thickness(key) + 0.35),
        ),
    ]
    representative = shape.representative_point()
    candidate = (float(representative.x), float(representative.y))
    if shape.covers(LineString(((stations[-1][0], stations[-1][2]), candidate))):
        stations.append((candidate[0], 0.11, candidate[1]))

    points: list[tuple[float, float, float]] = []
    for x, y, z in stations:
        points.extend((
            (x - tx * half_path, y, z - tz * half_path),
            (x, y, z),
            (x + tx * half_path, y, z + tz * half_path),
        ))
    faces: list[_Face] = []
    for station_index in range(len(stations) - 1):
        a = station_index * 3
        b = (station_index + 1) * 3
        for indices in ((a, a + 1, b + 1), (a, b + 1, b), (a + 1, a + 2, b + 2), (a + 1, b + 2, b + 1)):
            faces.append(_Face("", tuple((index, 0, 0.0, 0.0) for index in indices)))
    point_count = len(points)
    face_count = len(faces)
    selections: list[_NamedSelection] = []
    for name, point_index in (("In1", 1), ("Pos1", 4), ("Pos2", len(points) - 2)):
        weights = bytearray(point_count)
        weights[point_index] = 1
        selections.append(_NamedSelection(name, bytes(weights), bytes(face_count)))
    return _Lod(
        tuple(points), ((0.0, 1.0, 0.0),), tuple(faces), _PATHS_LOD,
        selections=tuple(selections),
    )

def _land_contact_lod(key: BuildingVariantKey) -> _Lod:
    half_width = key.width_m / 2.0
    half_length = key.length_m / 2.0
    return _Lod((
        (-half_width, 0.0, -half_length), (half_width, 0.0, -half_length),
        (half_width, 0.0, half_length), (-half_width, 0.0, half_length),
        (0.0, 0.0, -half_length), (half_width, 0.0, 0.0),
        (0.0, 0.0, half_length), (-half_width, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ), (), (), _LAND_CONTACT_LOD)


def _map_symbol_for_family(family: str) -> str:
    """Return the CWA 2D-map classification for a procedural building."""

    return "house" if family in {"residential", "townhouse"} else "building"


def _hollow_geometry_lod(key: BuildingVariantKey) -> _Lod:
    """Build an enterable collision shell from bounded convex wall boxes.

    Walkable floors belong exclusively to the Roadway LOD. Keeping a second
    solid Geometry slab at virtually the same height caused occasional CWA
    character penetration/sticking. The former low collision ceiling is also
    omitted; it could trap a unit pushed upward against the visual ceiling.
    """

    half_width = key.width_m * 0.5
    half_length = key.length_m * 0.5
    main_height = _main_building_height(key)
    wall_thickness = _interior_wall_thickness(key)
    door_half, door_height, _pivot_z = _door_dimensions(key)
    partition_ceiling = min(2.85, max(2.35, main_height - 0.15))

    points: list[tuple[float, float, float]] = []
    faces: list[_Face] = []
    component_ranges: list[tuple[range, range]] = []

    def add_raw_box(
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

    def add_sloped_box(
        x0: float, z0: float, x1: float, z1: float,
        top_y0: float, top_y1: float, thickness: float,
    ) -> None:
        """Add one thin convex sloped collision slab."""

        if x1 - x0 <= 1.0e-6 or z1 - z0 <= 1.0e-6 or thickness <= 1.0e-6:
            return
        bottom_y0 = top_y0 - thickness
        bottom_y1 = top_y1 - thickness
        point_start = len(points)
        face_start = len(faces)
        points.extend((
            (x0, bottom_y0, z0), (x1, bottom_y0, z0),
            (x1, bottom_y1, z1), (x0, bottom_y1, z1),
            (x0, top_y0, z0), (x1, top_y0, z0),
            (x1, top_y1, z1), (x0, top_y1, z1),
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

    def add_box(
        x0: float, y0: float, z0: float,
        x1: float, y1: float, z1: float,
    ) -> None:
        """Add one box, splitting long utility walls into legacy-safe components."""

        if x1 - x0 <= 1.0e-6 or y1 - y0 <= 1.0e-6 or z1 - z0 <= 1.0e-6:
            return
        x_segments = max(1, int(math.ceil((x1 - x0) / _MAX_GEOMETRY_COMPONENT_SPAN_M)))
        z_segments = max(1, int(math.ceil((z1 - z0) / _MAX_GEOMETRY_COMPONENT_SPAN_M)))
        x_step = (x1 - x0) / x_segments
        z_step = (z1 - z0) / z_segments
        for z_index in range(z_segments):
            sz0 = z0 + z_index * z_step
            sz1 = z0 + (z_index + 1) * z_step
            for x_index in range(x_segments):
                sx0 = x0 + x_index * x_step
                sx1 = x0 + (x_index + 1) * x_step
                add_raw_box(sx0, y0, sz0, sx1, y1, sz1)

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
            horizontal_min, horizontal_max, main_height, openings
        ):
            if horizontal_axis == "x":
                add_box(h0, y0, plane_min, h1, y1, plane_max)
            elif horizontal_axis == "z":
                add_box(plane_min, y0, h0, plane_max, y1, h1)
            else:
                raise ValueError(
                    f"unsupported collision wall horizontal axis: {horizontal_axis}"
                )

    front_windows = _interior_window_openings(
        key,
        -half_width,
        half_width,
        main_height,
        ground_exclusions=((-door_half - 0.35, door_half + 0.35),),
    )
    collision_door_half = min(half_width - wall_thickness, door_half + INTERIOR_COLLISION_DOOR_SIDE_CLEARANCE_M)
    collision_door_height = min(
        main_height, door_height + INTERIOR_COLLISION_DOOR_TOP_CLEARANCE_M
    )
    front_openings = front_windows + (
        (-collision_door_half, collision_door_half, 0.0, collision_door_height),
    )
    back_openings = _interior_window_openings(key, -half_width, half_width, main_height)
    side_openings = _interior_window_openings(key, -half_length, half_length, main_height)
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

    # Room-like buildings keep one cheap partition. Utility buildings stay as a
    # single open hall and therefore avoid both unnecessary collision components
    # and a comically domestic wall through the middle of a warehouse.
    if (
        key.family not in UTILITY_INTERIOR_FAMILIES
        and _interior_storey_count(key) < 2
        and key.width_m >= 6.0
        and key.length_m >= 7.0
    ):
        partition_half = wall_thickness * 0.5
        interior_door_half = min(0.75, max(0.55, half_width * 0.16))
        collision_partition_half = min(
            half_width - wall_thickness,
            interior_door_half + INTERIOR_COLLISION_DOOR_SIDE_CLEARANCE_M,
        )
        add_box(
            -half_width + wall_thickness, 0.0, -partition_half,
            -collision_partition_half, partition_ceiling, partition_half,
        )
        add_box(
            collision_partition_half, 0.0, -partition_half,
            half_width - wall_thickness, partition_ceiling, partition_half,
        )
        # No collision lintel above the interior doorway. The visual wall still
        # has one, but leaving head clearance in Geometry avoids old-engine
        # snagging when a soldier steps or is pushed through the opening.

    # Back the visible second-floor staircase with a *solid stepped* Geometry
    # mass. The previous sloped support could still be penetrated by OFP/CWA's
    # character controller. Sixteen shallow convex boxes are more old-engine
    # friendly: Roadway provides the intended foot contact and, if that contact
    # is missed, Geometry is only a few centimetres below it instead of letting
    # the player fall through the staircase. Slight Z overlap removes cracks
    # between adjacent components.
    second_layout = _second_storey_layout(key)
    if second_layout is not None:
        step_count = INTERIOR_SECOND_STOREY_STAIR_STEPS
        step_run = (second_layout.stair_z1 - second_layout.stair_z0) / step_count
        step_rise = second_layout.floor_y / step_count
        overlap = min(0.04, step_run * 0.20)
        support_bottom = -0.20
        for index in range(step_count):
            z0 = second_layout.stair_z0 + index * step_run
            z1 = second_layout.stair_z0 + (index + 1) * step_run
            if index > 0:
                z0 -= overlap
            if index < step_count - 1:
                z1 += overlap
            top_y = (index + 1) * step_rise + INTERIOR_ROADWAY_Y_M - 0.035
            add_raw_box(
                second_layout.stair_x0,
                support_bottom,
                z0,
                second_layout.stair_x1,
                top_y,
                z1,
            )

    # Exterior pedestrian foundation steps need Geometry too. Earlier builds
    # only drew them and provided Roadway treads, so the player could sometimes
    # clip through the visible porch staircase before the Roadway contact was
    # established. Fill every exposed tread with a shallow convex block.
    pedestrian_stairs = (
        ()
        if _entrance_uses_vehicle_ramp(key)
        else _interior_stair_profile(key, key.foundation_depth_m)
    )
    if pedestrian_stairs:
        transition_half = min(
            key.width_m * 0.45,
            door_half + INTERIOR_STAIR_SIDE_MARGIN_M,
        )
        stair_bottom = -max(0.20, key.foundation_depth_m + 0.08)
        for outer_z, inner_z, top_y, _bottom_y in pedestrian_stairs:
            add_box(
                -transition_half, stair_bottom, outer_z,
                transition_half, top_y - 0.02, inner_z,
            )

    vehicle_ramp = _interior_vehicle_ramp_profile(key, key.foundation_depth_m)
    if vehicle_ramp is not None:
        outer_z, inner_z, outer_y, inner_y = vehicle_ramp
        ramp_half = min(
            key.width_m * 0.45,
            door_half + INTERIOR_STAIR_SIDE_MARGIN_M,
        )
        add_sloped_box(
            -ramp_half, outer_z, ramp_half, inner_z,
            outer_y + INTERIOR_ROADWAY_Y_M - 0.04,
            inner_y + INTERIOR_ROADWAY_Y_M - 0.04,
            0.12,
        )

    # The entrance door remains an animated collision component. It is narrow
    # enough that add_box never splits it, so the last component is the door.
    door_point_range: range | None = None
    door_face_range: range | None = None
    door_z0 = -half_length + 0.02
    door_z1 = door_z0 + INTERIOR_DOOR_THICKNESS_M
    add_box(
        -door_half, 0.03, door_z0,
        door_half, door_height - 0.03, door_z1,
    )
    if component_ranges:
        door_point_range, door_face_range = component_ranges[-1]

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

    if door_point_range is not None and door_face_range is not None:
        point_weights = bytearray(len(points))
        face_flags = bytearray(len(faces))
        for point_index in door_point_range:
            point_weights[point_index] = 1
        for face_index in door_face_range:
            face_flags[face_index] = 1
        selections.append(_NamedSelection(
            name="door1",
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
        (("map", _map_symbol_for_family(key.family)), ("class", "house"), ("autocenter", "0")),
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
    # Every procedural building is authored with its wall base and LandContact
    # at local Y=0 and its foundation extending downward.  The supplied debug
    # world showed Way 783739728 serialized about 0.1 m above the highest terrain
    # under its complete 12 x 12 m footprint, yet CWA still drew the doorway
    # partly underground. Churches had the same symptom until autocenter=0 was
    # added. Apply the same authored-origin rule to all closed buildings so the
    # engine cannot recenter the model independently of WRP grounding.
    geometry_properties = [
        ("map", _map_symbol_for_family(key.family)),
        ("autocenter", "0"),
    ]
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
    door_texture: str | None = None,
    distance_wall_texture: str | None = None,
) -> None:
    if key.footprint_vertices and not _triangulate_polygon_coordinates(
        key.footprint_vertices, key.footprint_holes
    ):
        # Last-resort build continuity: a malformed/GEOS-hostile native polygon
        # must not destroy an otherwise complete seven-minute world build. The
        # robust triangulator above handles valid concave footprints in normal
        # operation; this path retains the building's semantic family, fitted
        # dimensions, height, roof style and interior settings while using its
        # fitted rectangle for the one pathological asset.
        key = replace(
            key,
            footprint_vertices=(),
            footprint_holes=(),
            entrance_edge=-1,
            entrance_fraction=0.5,
        )

    if key.footprint_vertices:
        detail = _polygon_native_visual_lod(
            key, wall_texture, roof_texture,
            roof_pitch_degrees=roof_pitch_degrees,
            front_texture=front_texture,
            foundation_texture=foundation_texture,
            foundation_depth=foundation_depth,
            interior_texture=interior_texture,
            window_trim_texture=window_trim_texture,
            plain_wall_texture=plain_wall_texture,
            door_texture=door_texture,
        )
        lods_list = [detail]
        if key.interiors:
            distance = _polygon_native_visual_lod(
                replace(key, interiors=False, second_storey=False),
                distance_wall_texture or plain_wall_texture or wall_texture,
                roof_texture,
                roof_pitch_degrees=roof_pitch_degrees,
                front_texture=front_texture,
                foundation_texture=foundation_texture,
                foundation_depth=foundation_depth,
                plain_wall_texture=(
                    plain_wall_texture
                    or distance_wall_texture
                    or wall_texture
                ),
                door_texture=door_texture,
            )
            lods_list.append(replace(
                distance, resolution=INTERIOR_DISTANCE_LOD_RESOLUTION,
                selections=(),
            ))
        lods_list.append(_polygon_native_geometry_lod(key))
        roadway = _polygon_native_roadway_lod(key, foundation_depth)
        if roadway is not None:
            lods_list.append(roadway)
        memory = _polygon_native_memory_lod(key)
        if memory is not None:
            lods_list.append(memory)
        paths = _polygon_native_paths_lod(key, foundation_depth)
        if paths is not None:
            lods_list.append(paths)
        lods_list.append(_polygon_native_land_contact_lod(key))
        lods = tuple(lods_list)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, len(lods)))
            for lod in lods:
                _write_lod(stream, lod)
        return

    detail = _visual_lod(
        key, wall_texture, roof_texture, roof_pitch_degrees, front_texture,
        foundation_texture, foundation_depth, church_plinth_height,
        interior_texture, window_trim_texture, plain_wall_texture,
    )
    if key.interiors:
        detail = _add_animated_door_visual(
            detail, key, door_texture or interior_texture or wall_texture
        )

    lods_list = [detail]
    if key.interiors:
        # At distance the engine should not pay for room shells, cut windows,
        # trim, mullions, or stairs on every visible house. The normal painted
        # facade supplies a compact silhouette until the detailed LOD is close
        # enough to matter.
        distance = _visual_lod(
            replace(key, interiors=False, second_storey=False),
            distance_wall_texture or wall_texture,
            roof_texture,
            roof_pitch_degrees,
            front_texture=front_texture,
            foundation_texture=foundation_texture,
            foundation_depth=foundation_depth,
            church_plinth_height=church_plinth_height,
            plain_wall_texture=(
                plain_wall_texture
                or distance_wall_texture
                or wall_texture
            ),
        )
        lods_list.append(replace(distance, resolution=INTERIOR_DISTANCE_LOD_RESOLUTION))

    lods_list.append(_geometry_lod(key, church_plinth_height))
    roadway = _interior_roadway_lod(key, foundation_depth)
    if roadway is not None:
        lods_list.append(roadway)
    memory = _interior_memory_lod(key)
    if memory is not None:
        lods_list.append(memory)
    paths = _interior_paths_lod(key, foundation_depth)
    if paths is not None:
        lods_list.append(paths)
    lods_list.append(_land_contact_lod(key))
    lods = tuple(lods_list)
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
        maximum_polygon_variants: int = POLYGON_NATIVE_MAXIMUM_VARIANTS,
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
        self.maximum_polygon_variants = max(0, int(maximum_polygon_variants))
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
        self._selection_cache: dict[BuildingVariantKey, BuildingVariantKey] = {}
        self._model_path_cache: dict[BuildingVariantKey, str] = {}
        self._usage: Counter[BuildingVariantKey] = Counter()
        self._prepared = False
        self.region_identifier: str | None = None
        self._urban_polygons: tuple[Polygon, ...] = ()
        self._settlement_points: tuple[tuple[float, float, float, str], ...] = ()
        self._settlement_bucket_size = 1.0
        self._settlement_buckets: dict[tuple[int, int], tuple[int, ...]] = {}
        self._isolated_dwelling_areas: tuple[Polygon, ...] = ()
        self._isolated_dwelling_cabins: tuple[Polygon, ...] = ()
        self._settlement_scale_x = 1.0
        self._settlement_scale_z = 1.0
        self._polygon_native_keys: set[BuildingVariantKey] = set()

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
                    if building_kind in {"", "yes"}
                ]
                if len(plausible) == 1:
                    # One generic footprint may coexist with explicit garages,
                    # sheds, houses, etc. Only the footprint whose semantic type
                    # CWR actually has to infer receives cabin context.
                    isolated_dwelling_cabins.append(plausible[0])
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
        # Settlement classification used to scan every place node for every
        # building. Bucket the fixed-radius hints in world space and retain the
        # exact distance test below, so semantics stay identical while dense
        # regional extracts stop paying O(buildings * places).
        self._settlement_bucket_size = max(
            1.0, 1000.0 * self._settlement_scale_x, 1000.0 * self._settlement_scale_z
        )
        mutable_buckets: dict[tuple[int, int], list[int]] = {}
        for index, (centre_x, centre_z, _radius, _kind) in enumerate(self._settlement_points):
            key = (
                math.floor(centre_x / self._settlement_bucket_size),
                math.floor(centre_z / self._settlement_bucket_size),
            )
            mutable_buckets.setdefault(key, []).append(index)
        self._settlement_buckets = {key: tuple(values) for key, values in mutable_buckets.items()}

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
        bucket_size = self._settlement_bucket_size
        bucket_x = math.floor(float(x) / bucket_size)
        bucket_z = math.floor(float(z) / bucket_size)
        candidate_indices: list[int] = []
        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                candidate_indices.extend(self._settlement_buckets.get((bucket_x + dx, bucket_z + dz), ()))
        for index in candidate_indices:
            centre_x, centre_z, radius, kind = self._settlement_points[index]
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
        explicit_levels = _parse_number(tags.get("building:levels"))
        if (
            str(tags.get("building", "")).casefold() == "cabin"
            and _parse_number(tags.get("height")) is None
            and _parse_number(tags.get("building:levels")) is None
        ):
            requested_height = self.default_level_height
        if (
            settlement_context == "isolated_dwelling_single"
            and family == "residential"
            and _parse_number(tags.get("height")) is None
            and _parse_number(tags.get("building:levels")) is None
        ):
            # One-storey cabin rather than a six-metre generic house. Explicit
            # OSM height/levels still win when present.
            requested_height = self.default_level_height
        facade_storeys = _requested_facade_storeys(
            tags, family, requested_height, self.default_level_height
        )
        if (
            explicit_levels is None
            and _parse_number(tags.get("height")) is None
            and requested_height <= self.default_level_height + 1.0e-6
        ):
            facade_storeys = 1
        quantized_height = _quantize(
            requested_height,
            self.height_quantum,
            self.minimum_height,
            self.maximum_height,
        )
        interior_max_width, interior_max_length, interior_max_height = (
            INTERIOR_FAMILY_MAXIMUM_DIMENSIONS_M.get(
                family,
                (
                    INTERIOR_MAXIMUM_WIDTH_M,
                    INTERIOR_MAXIMUM_LENGTH_M,
                    INTERIOR_MAXIMUM_HEIGHT_M,
                ),
            )
        )
        interiors = (
            self.generate_interiors
            and family in INTERIOR_ELIGIBLE_FAMILIES
            and width <= interior_max_width
            and length <= interior_max_length
            and quantized_height <= interior_max_height
        )
        isolated_dwelling = (
            settlement_context == "isolated_dwelling_single"
            and family == "residential"
        )
        physically_supports_second_storey = (
            width >= INTERIOR_SECOND_STOREY_MINIMUM_WIDTH_M
            and length >= INTERIOR_SECOND_STOREY_MINIMUM_LENGTH_M
            and quantized_height >= INTERIOR_SECOND_STOREY_MINIMUM_HEIGHT_M
        )
        second_storey = (
            interiors
            and family in SECOND_STOREY_INTERIOR_FAMILIES
            and facade_storeys >= 2
            and physically_supports_second_storey
            and not (explicit_levels is not None and explicit_levels < 2.0)
        )
        outbuilding_kind = (
            _outbuilding_kind(tags, width, length)
            if family == "outbuilding"
            else ""
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
            second_storey=second_storey,
            outbuilding_kind=outbuilding_kind,
            facade_storeys=facade_storeys,
            isolated_dwelling=isolated_dwelling,
        )

    def _iter_dataset_keys(self, dataset: Any, projection: Any, point_footprint: float) -> Iterable[BuildingVariantKey]:
        for feature in dataset.building_polygons:
            for polygon in feature.polygons:
                projected = [projection.to_world(point) for point in polygon.outer[:-1]]
                if len(projected) >= 3:
                    projected_holes = tuple(
                        tuple(projection.to_world(point) for point in ring[:-1])
                        for ring in polygon.holes
                        if len(ring) >= 4
                    )
                    # One mapped footprint always requests one building variant.
                    # Irregular polygons may later receive one polygon-native
                    # model, but they are never split into independent wings.
                    polygon_geometry, footprint = _polygon_with_footprint(
                        projected, projected_holes
                    )
                    centre = polygon_geometry.centroid
                    yield self.key_for(
                        feature.tags, footprint.width_m, footprint.length_m,
                        settlement_context=self._settlement_context(float(centre.x), float(centre.y)),
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
        if requested.family == "outbuilding" and requested.outbuilding_kind:
            same_outbuilding_kind = [
                candidate for candidate in candidates
                if candidate.family == "outbuilding"
                and candidate.outbuilding_kind == requested.outbuilding_kind
            ]
            if same_outbuilding_kind:
                candidates = same_outbuilding_kind
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
                (
                    candidate.outbuilding_kind != requested.outbuilding_kind
                    if requested.family == "outbuilding"
                    else False
                ),
                self._variant_fit_score(requested, candidate),
                candidate.regional_style != requested.regional_style,
                candidate.roof_style != requested.roof_style,
                candidate,
            ),
        )

    def prepare(self, dataset: Any, projection: Any, point_footprint: float) -> None:
        self._mapping.clear()
        self._selection_cache.clear()
        self._model_path_cache.clear()
        self._usage.clear()
        self._polygon_native_keys.clear()
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
        # Preserve both shed and garage frontages when both occur. Without this,
        # a tight variant budget can reserve one generic outbuilding model and
        # make every small shed inherit a vehicle-sized garage opening.
        for outbuilding_kind in ("shed", "garage"):
            representative = next((
                key for key in ordered
                if key.family == "outbuilding"
                and key.outbuilding_kind == outbuilding_kind
            ), None)
            if representative is None or representative in selected:
                continue
            if len(selected) >= self.maximum_variants:
                break
            selected.append(representative)
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
        # Land_<p3dName> config classes are global in CWA. Include a compact
        # world hash in the filename so two generated islands can coexist
        # without their interactive-building classes colliding. Model paths are
        # requested for every placement but there are only a bounded number of
        # variant keys, so cache the JSON/SHA work rather than redoing it 100k+.
        cached = self._model_path_cache.get(key)
        if cached is not None:
            return cached
        world_code = sha256(self.world_name.encode("ascii")).hexdigest()[:4]
        path = rf"{self.world_name}\g\b_{world_code}_{key.digest}.p3d"
        self._model_path_cache[key] = path
        return path

    def is_generated_model(self, path: str) -> bool:
        return path.casefold().startswith((self.world_name + r"\g\b_").casefold()) and path.casefold().endswith(".p3d")

    def _selected(self, requested: BuildingVariantKey) -> BuildingVariantKey:
        if not self._prepared:
            raise RuntimeError("procedural building library must be prepared before placement")
        if requested in self._mapping:
            return self._mapping[requested]
        cached = self._selection_cache.get(requested)
        if cached is not None:
            return cached
        if not self._mapping:
            self._mapping[requested] = requested
            return requested
        candidates = self._reuse_candidates(
            requested, sorted(set(self._mapping.values()))
        )
        selected = self._best_variant(requested, candidates)
        self._selection_cache[requested] = selected
        return selected

    def plan_polygon(
        self, tags: Mapping[str, str], points: Sequence[PointXZ], *,
        holes: Sequence[Sequence[PointXZ]] = (),
        road_point: PointXZ | None = None, entrance_point: PointXZ | None = None,
        allow_native_polygon: bool = True,
    ) -> BuildingPlacement:
        polygon, footprint = _polygon_with_footprint(points, holes)
        centre = polygon.centroid
        centre_x, centre_z = float(centre.x), float(centre.y)
        hash_coordinates = tuple(points) + tuple(
            point for ring in holes for point in ring
        )
        placement_hash = _placement_hash_u32(tags, hash_coordinates)
        requested = self.key_for(
            tags, footprint.width_m, footprint.length_m,
            settlement_context=self._settlement_context(centre_x, centre_z),
        )

        native_profile = (
            _native_polygon_profile(
                points, holes, footprint=footprint, polygon=polygon
            )
            if allow_native_polygon
            and requested.family != "church"
            else None
        )
        if native_profile is not None:
            (
                native_vertices, native_holes, native_heading,
                native_width, native_length,
            ) = native_profile
            native_requested = self.key_for(
                tags, native_width, native_length,
                settlement_context=self._settlement_context(centre_x, centre_z),
            )
            # Gabled/hipped/pyramidal roofs now follow the polygon. More exotic
            # roof families still keep the exact building outline but deliberately
            # fall back to a flat top; one slightly conservative roof is much less
            # wrong than turning one source building into overlapping rectangles.
            native_roof_style = (
                native_requested.roof_style
                if native_requested.roof_style in {"flat", "gabled", "hipped", "pyramidal"}
                else "flat"
            )

            entrance_edge = -1
            entrance_fraction = 0.5
            frontage_point = entrance_point if entrance_point is not None else road_point
            if frontage_point is not None and native_vertices:
                angle = math.radians(native_heading)
                dx = float(frontage_point[0]) - centre_x
                dz = float(frontage_point[1]) - centre_z
                local_frontage = (
                    dx * math.cos(angle) - dz * math.sin(angle),
                    dx * math.sin(angle) + dz * math.cos(angle),
                )
                ranked_edges: list[tuple[float, int, float]] = []
                for edge_index, start in enumerate(native_vertices):
                    end = native_vertices[(edge_index + 1) % len(native_vertices)]
                    edge_x, edge_z = end[0] - start[0], end[1] - start[1]
                    length_sq = edge_x * edge_x + edge_z * edge_z
                    if length_sq <= 1.0e-8:
                        continue
                    fraction = max(0.0, min(1.0, (
                        (local_frontage[0] - start[0]) * edge_x
                        + (local_frontage[1] - start[1]) * edge_z
                    ) / length_sq))
                    nearest_x = start[0] + edge_x * fraction
                    nearest_z = start[1] + edge_z * fraction
                    distance_sq = (
                        (local_frontage[0] - nearest_x) ** 2
                        + (local_frontage[1] - nearest_z) ** 2
                    )
                    ranked_edges.append((distance_sq, edge_index, fraction))
                if ranked_edges:
                    _distance, entrance_edge, entrance_fraction = min(ranked_edges)
                    quantum = POLYGON_NATIVE_ENTRANCE_FRACTION_QUANTUM
                    entrance_fraction = max(0.0, min(1.0,
                        round(entrance_fraction / quantum) * quantum
                    ))

            native_requested = replace(
                native_requested,
                roof_style=native_roof_style,
                # Native shells now have a footprint-following ground-floor
                # interior. Keep the normal family/dimension eligibility gate;
                # second-storey polygon partitioning is deliberately deferred
                # until stairs can be generated without crossing concave walls.
                interiors=native_requested.interiors,
                second_storey=False,
                footprint_vertices=native_vertices,
                footprint_holes=native_holes,
                entrance_edge=entrance_edge,
                entrance_fraction=entrance_fraction,
            )
            selected = replace(
                native_requested,
                texture_variant=_placement_texture_variant(
                    tags, hash_coordinates, variant_count=(
                        min(self.texture_variants, INTERIOR_MODEL_TEXTURE_VARIANTS)
                        if native_requested.interiors else self.texture_variants
                    ),
                    placement_hash=placement_hash,
                ),
            )
            if (
                selected in self._polygon_native_keys
                or len(self._polygon_native_keys) < self.maximum_polygon_variants
            ):
                self._polygon_native_keys.add(selected)
                return BuildingPlacement(
                    self.model_path(selected),
                    native_heading,
                    native_requested,
                    selected,
                )

        # Rectangle fallback. This is intentionally ONE model even for L/T/U
        # footprints. If the polygon is close enough to rectangular this is also
        # the preferred path, avoiding bespoke P3Ds for trivial survey notches.
        requested = replace(
            requested,
            second_storey=_placement_uses_second_storey(
                tags, points, requested, placement_hash=placement_hash
            ),
        )
        selected = replace(
            self._selected(requested),
            texture_variant=_placement_texture_variant(
                tags, points, variant_count=(
                    min(self.texture_variants, INTERIOR_MODEL_TEXTURE_VARIANTS)
                    if requested.interiors else self.texture_variants
                ),
                placement_hash=placement_hash,
            ),
            second_storey=requested.second_storey,
        )
        heading = footprint.heading_degrees
        frontage_point = entrance_point if entrance_point is not None else road_point
        if selected.family in HOUSE_ROAD_FACING_FAMILIES:
            heading = _house_heading_towards_road(
                heading,
                centre_x=centre_x,
                centre_z=centre_z,
                road_point=frontage_point,
                width_m=selected.width_m,
                length_m=selected.length_m,
            )
        elif frontage_point is not None:
            dx = frontage_point[0] - centre_x
            dz = frontage_point[1] - centre_z
            front_x, front_z = _front_vector_for_heading(heading)
            if front_x * dx + front_z * dz < 0.0:
                heading = (heading + 180.0) % 360.0
        return BuildingPlacement(self.model_path(selected), heading, requested, selected)

    def plan_point(
        self, tags: Mapping[str, str], footprint: float, heading_degrees: float,
        *, x: float | None = None, z: float | None = None,
        road_point: PointXZ | None = None,
    ) -> BuildingPlacement:
        requested = self.key_for(
            tags, footprint, footprint,
            settlement_context=self._settlement_context(x, z),
        )
        coordinates = ((float(x), float(z)),) if x is not None and z is not None else ((float(footprint), float(heading_degrees)),)
        placement_hash = _placement_hash_u32(tags, coordinates)
        requested = replace(
            requested,
            second_storey=_placement_uses_second_storey(
                tags, coordinates, requested, placement_hash=placement_hash
            ),
        )
        selected = replace(
            self._selected(requested),
            texture_variant=_placement_texture_variant(
                tags, coordinates, variant_count=(
                    min(self.texture_variants, INTERIOR_MODEL_TEXTURE_VARIANTS)
                    if requested.interiors else self.texture_variants
                ),
                placement_hash=placement_hash,
            ),
            second_storey=requested.second_storey,
        )
        heading = heading_degrees
        if (
            selected.family in HOUSE_ROAD_FACING_FAMILIES
            and x is not None
            and z is not None
        ):
            heading = _house_heading_towards_road(
                heading,
                centre_x=float(x),
                centre_z=float(z),
                road_point=road_point,
                width_m=selected.width_m,
                length_m=selected.length_m,
            )
        return BuildingPlacement(self.model_path(selected), heading, requested, selected)

    def register_placement(
        self, placement: BuildingPlacement, *, foundation_depth_m: float | None = None
    ) -> BuildingPlacement:
        foundation_depth = self._foundation_depth(
            foundation_depth_m,
            allow_above_configured_maximum=foundation_depth_m is not None,
        )
        if placement.selected.interiors:
            # Foundation depth is mostly buried geometry, but putting every 25 cm
            # terrain variation into the model key creates many near-identical
            # P3Ds. Round enterable buildings upward to 50 cm buckets so stairs
            # still reach the terrain while the engine loads far fewer models.
            foundation_depth = (
                math.ceil(foundation_depth / INTERIOR_FOUNDATION_DEPTH_QUANTUM_M)
                * INTERIOR_FOUNDATION_DEPTH_QUANTUM_M
            )
        selected = replace(
            placement.selected,
            foundation_depth_m=foundation_depth,
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
        road_point: PointXZ | None = None,
    ) -> BuildingPlacement:
        return self.register_placement(
            self.plan_point(
                tags, footprint, heading_degrees, x=x, z=z, road_point=road_point
            )
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

    def _door_texture(
        self, family: str = "residential", regional_style: str = "default",
        texture_variant: int = 0, outbuilding_kind: str = "",
    ) -> str:
        """Return a compact animated-door texture path for the building type."""

        if family == "agricultural":
            prefix = "b"
        elif family == "industrial" or (
            family == "outbuilding" and outbuilding_kind == "garage"
        ):
            prefix = "c"
        elif family == "outbuilding":
            prefix = "p"
        else:
            return rf"{self.world_name}\d\dr.paa"
        code = self._palette_texture_code(family, regional_style, texture_variant)
        return rf"{self.world_name}\d\{prefix}{code}.paa"

    def _front_texture(
        self, family: str, regional_style: str = "default", texture_variant: int = 0,
        outbuilding_kind: str = "",
    ) -> str:
        code = self._palette_texture_code(family, regional_style, texture_variant)
        # Keep the compact filename budget while allowing both outbuilding
        # frontages to coexist for the same regional palette variant.
        prefix = ("g" if outbuilding_kind == "garage" else "s") if family == "outbuilding" else "e"
        return rf"{self.world_name}\d\{prefix}{code}.paa"

    def _roof_texture(self, roof_style: str, texture_variant: int = 0) -> str:
        roof_index = {
            "flat": 0, "gabled": 1, "hipped": 2,
            "pyramidal": 3, "dome": 4, "onion": 5,
        }[roof_style]
        value = roof_index * self.texture_variants
        value += _normalise_texture_variant(texture_variant, self.texture_variants)
        return rf"{self.world_name}\d\r{self._base36_code(value)}.paa"

    def _foundation_texture(self) -> str:
        return rf"{self.world_name}\d\f.paa"

    def write_assets(self, source_dir: Path, catalogue_path: Path) -> BuildingGenerationResult:
        selected = sorted(self._usage)
        model_assets: list[GeneratedBuildingAsset] = []
        texture_files: list[str] = []
        # Emit only texture variants actually referenced by generated P3Ds.
        # Previous releases wrote all ten palette variants for every used style
        # even when the selected model set referenced only one or two. That made
        # interior-heavy PBOs unnecessarily large and slower to enumerate/load.
        used_palette_variants = sorted({
            (key.family, key.regional_style, key.texture_variant)
            for key in selected
        })
        used_open_palette_variants = sorted({
            (key.family, key.regional_style, key.texture_variant) for key in selected
            if key.interiors
            or key.family == "church"
            or key.family in _GROUND_FLOOR_ONLY_FACADE_FAMILIES
            or key.family in _PAINTED_WINDOW_FAMILIES
        })
        used_interior_palette_variants = sorted({
            (key.family, key.regional_style, key.texture_variant)
            for key in selected if key.interiors
        })
        used_front_palette_variants = sorted({
            (key.family, key.regional_style, key.texture_variant, key.outbuilding_kind)
            for key in selected
        })
        uses_white_window_trim = any(
            key.interiors
            and key.family not in UTILITY_INTERIOR_FAMILIES
            and key.regional_style in WHITE_WINDOW_TRIM_REGIONAL_STYLES
            for key in selected
        )
        used_door_variants: dict[str, tuple[str, str, int, str]] = {}
        for key in selected:
            if not key.interiors and not key.footprint_vertices:
                continue
            path = self._door_texture(
                key.family, key.regional_style, key.texture_variant,
                key.outbuilding_kind,
            )
            used_door_variants.setdefault(
                path,
                (key.family, key.regional_style, key.texture_variant, key.outbuilding_kind),
            )
        used_roof_variants = sorted({
            (key.roof_style, key.texture_variant) for key in selected
        })
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
        for texture_path, door_variant in sorted(used_door_variants.items()):
            family, regional_style, texture_variant, outbuilding_kind = door_variant
            relative = texture_path.split("\\", 1)[1].replace("\\", "/")
            file_path = source_dir / relative
            cache_id = cache_key(
                "procedural-building-door-v3-clean-utility-aperture-selectable-quality",
                {
                    "family": family,
                    "regional_style": regional_style,
                    "texture_variant": texture_variant,
                    "outbuilding_kind": outbuilding_kind,
                    "texture_size": self.texture_size,
                },
            )
            cached = self.cache_dir / "procedural-assets" / f"{cache_id}.paa" if self.cache_dir else None
            hit = restore_or_create_file(
                cache_path=cached,
                destination=file_path,
                producer=lambda target, family=family, regional_style=regional_style, texture_variant=texture_variant, outbuilding_kind=outbuilding_kind: write_rgb_dxt1_paa(
                    target,
                    _door_texture_image(
                        self.texture_size,
                        family=family,
                        regional_style=regional_style,
                        texture_variant=texture_variant,
                        outbuilding_kind=outbuilding_kind,
                    ),
                ),
                enabled=self.cache_enabled,
                refresh=self.cache_refresh,
            )
            self.cache_hits += int(hit)
            self.cache_misses += int(not hit)
            inspect_paa(file_path)
            texture_files.append(relative)
        for family, regional_style, texture_variant in used_palette_variants:
                texture_path = self._wall_texture(
                    family, regional_style, texture_variant
                )
                relative = texture_path.split("\\", 1)[1].replace("\\", "/")
                file_path = source_dir / relative
                key = cache_key(
                    "procedural-building-wall-v12-window-sill-selectable-quality",
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
        for family, regional_style, texture_variant in used_open_palette_variants:
                texture_path = self._open_wall_texture(
                    family, regional_style, texture_variant
                )
                relative = texture_path.split("\\", 1)[1].replace("\\", "/")
                file_path = source_dir / relative
                key = cache_key(
                    "procedural-building-open-wall-v4-utility-cladding-match-selectable-quality",
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
        for family, regional_style, texture_variant in used_interior_palette_variants:
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
        for family, regional_style, texture_variant, outbuilding_kind in used_front_palette_variants:
                texture_path = self._front_texture(
                    family, regional_style, texture_variant, outbuilding_kind
                )
                relative = texture_path.split("\\", 1)[1].replace("\\", "/")
                file_path = source_dir / relative
                key = cache_key(
                    "procedural-building-front-v13-window-sill-selectable-quality",
                    {
                        "family": family,
                        "regional_style": regional_style,
                        "texture_variant": texture_variant,
                        "outbuilding_kind": outbuilding_kind,
                        "texture_size": self.texture_size,
                    },
                )
                cached = self.cache_dir / "procedural-assets" / f"{key}.paa" if self.cache_dir else None
                hit = restore_or_create_file(
                    cache_path=cached, destination=file_path,
                    producer=lambda target, family=family, regional_style=regional_style, texture_variant=texture_variant, outbuilding_kind=outbuilding_kind: write_rgb_dxt1_paa(
                        target,
                        _front_texture_image(
                            family,
                            size=self.texture_size,
                            regional_style=regional_style,
                            texture_variant=texture_variant,
                            outbuilding_kind=outbuilding_kind,
                        ),
                    ),
                    enabled=self.cache_enabled, refresh=self.cache_refresh,
                )
                self.cache_hits += int(hit); self.cache_misses += int(not hit)
                inspect_paa(file_path); texture_files.append(relative)

        for roof, texture_variant in used_roof_variants:
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
                "procedural-building-model-v49-robust-polygon-roof-triangulation",
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
                        self._wall_texture(
                            key.family, key.regional_style, key.texture_variant
                        )
                        if key.interiors and key.family in UTILITY_INTERIOR_FAMILIES
                        else self._open_wall_texture(
                            key.family, key.regional_style, key.texture_variant
                        )
                        if key.interiors
                        else self._wall_texture(
                            key.family, key.regional_style, key.texture_variant
                        )
                    ),
                    front_texture=self._front_texture(
                        key.family, key.regional_style, key.texture_variant,
                        key.outbuilding_kind,
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
                        and key.family not in UTILITY_INTERIOR_FAMILIES
                        and key.regional_style in WHITE_WINDOW_TRIM_REGIONAL_STYLES
                        else None
                    ),
                    plain_wall_texture=(
                        self._open_wall_texture(
                            key.family, key.regional_style, key.texture_variant
                        )
                        if (
                            key.interiors
                            or key.family in _GROUND_FLOOR_ONLY_FACADE_FAMILIES
                            or key.family == "church"
                            or key.family in _PAINTED_WINDOW_FAMILIES
                        )
                        else None
                    ),
                    door_texture=(
                        self._door_texture(
                            key.family, key.regional_style, key.texture_variant,
                            key.outbuilding_kind,
                        )
                        if key.interiors or key.footprint_vertices else None
                    ),
                    distance_wall_texture=(
                        self._wall_texture(
                            key.family, key.regional_style, key.texture_variant
                        )
                        if key.interiors else None
                    ),
                ),
                enabled=self.cache_enabled,
                refresh=self.cache_refresh,
            )
            self.cache_hits += int(hit)
            self.cache_misses += int(not hit)
            summary = inspect_mlod(file_path)
            geometry_index = next(
                index for index, resolution in enumerate(summary.resolutions)
                if math.isclose(resolution, _GEOMETRY_LOD, rel_tol=1.0e-6)
            )
            geometry_selection_names = summary.selection_names[geometry_index]
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
                geometry_component_count=sum(
                    name.startswith("component") for name in geometry_selection_names
                ),
                geometry_mass_point_count=summary.mass_point_counts[geometry_index],
                map_symbol=dict(summary.named_properties[geometry_index])["map"],
            ))

        placements = sum(self._usage.values())
        generated = len(model_assets)
        polygon_native_variants = sum(1 for key in selected if key.footprint_vertices)
        polygon_native_placements = sum(
            self._usage[key] for key in selected if key.footprint_vertices
        )
        document = {
            "schema": 18,
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
                "foundation_strategy": "shared-placement-depth-with-enterable-foundation-stairs",
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
                "interior_family_maximum_dimensions_m": {
                    family: list(dimensions)
                    for family, dimensions in sorted(
                        INTERIOR_FAMILY_MAXIMUM_DIMENSIONS_M.items()
                    )
                },
                "interior_layout": "one-or-two-level-room-shell-or-open-utility-hall-with-roadway-only-floors-and-stairs",
                "interior_second_storey_families": sorted(SECOND_STOREY_INTERIOR_FAMILIES),
                "interior_second_storey_minimum_dimensions_m": [
                    INTERIOR_SECOND_STOREY_MINIMUM_WIDTH_M,
                    INTERIOR_SECOND_STOREY_MINIMUM_LENGTH_M,
                    INTERIOR_SECOND_STOREY_MINIMUM_HEIGHT_M,
                ],
                "interior_second_storey_floor_y_m": INTERIOR_SECOND_STOREY_FLOOR_Y_M,
                "interior_second_storey_stair_steps": INTERIOR_SECOND_STOREY_STAIR_STEPS,
                "interior_second_storey_default_share_percent": INTERIOR_SECOND_STOREY_DEFAULT_SHARE_PERCENT,
                "interior_second_storey_collision": "roadway-treads-with-solid-geometry-step-support-no-geometry-floor-or-ceiling",
                "utility_vehicle_scale_entrance_doors": True,
                "outbuilding_door_selection": {
                    "garage_minimum_width_m": OUTBUILDING_GARAGE_MINIMUM_WIDTH_M,
                    "garage_minimum_length_m": OUTBUILDING_GARAGE_MINIMUM_LENGTH_M,
                    "smaller": "shed-small-door",
                    "car_capable": "garage-vehicle-door",
                },
                "house_entrance_road_facing": "nearest-road-with-footprint-safe-rotation",
                "animated_entrance_doors": True,
                "ai_paths_lod": True,
                "interior_distance_lod_resolution": INTERIOR_DISTANCE_LOD_RESOLUTION,
                "interior_model_texture_variants": INTERIOR_MODEL_TEXTURE_VARIANTS,
                "interior_foundation_depth_quantum_m": INTERIOR_FOUNDATION_DEPTH_QUANTUM_M,
                "texture_asset_emission": "referenced-variants-only",
                "genuine_window_storeys": 2,
                "window_trim_geometry": "flat-strips",
                "window_mullion_geometry": "flat-strips",
                "double_sided_visual_shell": True,
                "maximum_geometry_component_span_m": _MAX_GEOMETRY_COMPONENT_SPAN_M,
                "polygon_native_buildings": {
                    "enabled": True,
                    "roof_support": [
                        "flat", "gabled", "hipped", "pyramidal",
                        "unsupported-shapes-as-flat",
                    ],
                    "courtyard_holes": True,
                    "mapped_entrance_lateral_position": True,
                    "entrance_fraction_quantum": POLYGON_NATIVE_ENTRANCE_FRACTION_QUANTUM,
                    "interiors": False,
                    "maximum_vertices": POLYGON_NATIVE_MAXIMUM_VERTICES,
                    "maximum_variants": self.maximum_polygon_variants,
                    "rectangular_fill_threshold": POLYGON_NATIVE_RECTANGULAR_FILL_THRESHOLD,
                    "simplify_tolerance_m": POLYGON_NATIVE_SIMPLIFY_TOLERANCE_M,
                    "vertex_quantum_m": POLYGON_NATIVE_VERTEX_QUANTUM_M,
                    "collision_component_span_m": _MAX_GEOMETRY_COMPONENT_SPAN_M,
                    "whole_building_span_limit_m": None,
                    "fallback": "one-fitted-rectangle-never-multipart",
                },
                "grounding_footprint": "selected_model_exact_polygon_or_oriented_rectangle",
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
            "polygon_native_variants": polygon_native_variants,
            "polygon_native_placements": polygon_native_placements,
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
