# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from hashlib import blake2s
from pathlib import Path
import math
from typing import Callable, Mapping, Sequence

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np

from .osm import (
    BboxProjection,
    BuildingPlacementPlan,
    OsmDataset,
    OsmLineFeature,
    OsmPolygonFeature,
    OsmRaster,
    road_is_dirt,
    road_is_gravel,
    road_is_supported,
    road_width_metres,
)
from .paa import write_rgb_dxt1_paa
from .procedural_infrastructure import create_gravel_road_texture_image
from .terrain import MaterialDefinition, NOGOVA_GROUND_TEXTURE_CYCLE


@dataclass(frozen=True, slots=True)
class SurfaceMaterialDefinition(MaterialDefinition):
    everon_path: str | None = None


# One-character codes keep generated texture paths inside RVW4's 31-byte limit
# even when the world name uses all twenty permitted characters.
# The ``everon`` profile is deliberately a complete stock-texture palette.
# OSM still supplies semantic placement masks, but no generated OSM-themed ground
# artwork is written when this profile is selected.  Rock and farmland classes
# use their dedicated textures from O.pbo; the remaining semantic classes reuse
# the compact verified Eden palette.
MILESTONE9_MATERIALS: tuple[SurfaceMaterialDefinition, ...] = (
    SurfaceMaterialDefinition("w", "seabed/water", (48, 75, 94), r"Eden\tn.paa"),
    SurfaceMaterialDefinition("q", "wet shoreline", (116, 104, 75), r"Eden\tn.paa"),
    SurfaceMaterialDefinition("s", "dry shoreline sand", (184, 168, 111), r"Eden\bak\bah.pac"),
    SurfaceMaterialDefinition("g", "grass", (83, 121, 67), r"Eden\zbh.paa"),
    SurfaceMaterialDefinition("h", "dry grass", (119, 126, 73), r"Eden\zbh.paa"),
    SurfaceMaterialDefinition("r", "rock", (109, 106, 101), r"o\l1.paa"),
    SurfaceMaterialDefinition("k", "steep rock/scree", (86, 84, 82), r"o\lom2.paa"),
    SurfaceMaterialDefinition("f", "forest interior", (47, 83, 44), r"Eden\zbh.paa"),
    SurfaceMaterialDefinition("e", "forest edge", (67, 99, 53), r"Eden\zbh.paa"),
    SurfaceMaterialDefinition("a", "farmland light", (154, 144, 78), r"o\pole1.paa"),
    SurfaceMaterialDefinition("b", "farmland dark", (126, 121, 64), r"o\pole2.paa"),
    SurfaceMaterialDefinition("c", "field boundary", (91, 92, 52), r"Eden\zbh.paa"),
    SurfaceMaterialDefinition("u", "urban surface", (139, 136, 130), r"Eden\tn.paa"),
    SurfaceMaterialDefinition("i", "industrial surface", (112, 113, 109), r"Eden\tn.paa"),
    SurfaceMaterialDefinition("p", "paved road", (57, 57, 55), r"Eden\tn.paa"),
    SurfaceMaterialDefinition("o", "road shoulder", (112, 105, 91), r"Eden\bak\bah.pac"),
    SurfaceMaterialDefinition("d", "dirt road", (113, 86, 55), r"Eden\bak\bah.pac"),
    SurfaceMaterialDefinition("t", "dirt-road blend", (104, 101, 64), r"Eden\zbh.paa"),
    # Dedicated gravel underlay uses the generated gravel artwork while the
    # artwork as the road P3D. It hides sub-cell grass cracks beneath object joins
    # instead of letting a curved road reveal green triangles between cards.
    SurfaceMaterialDefinition("v", "gravel road", (80, 76, 68), r"Eden\bak\bah.pac"),
    SurfaceMaterialDefinition("j", "park", (96, 127, 67), r"Eden\zbh.paa"),
    SurfaceMaterialDefinition("y", "sports field", (103, 132, 74), r"Eden\zbh.paa"),
    SurfaceMaterialDefinition("x", "mapped beach", (194, 174, 116), r"Eden\bak\bah.pac"),
)

# Nogova is intentionally independent from Everon. General semantic terrain
# slots resolve to the same three-texture Nogova family, with explicit farmland
# overrides below. Object/tree profiles remain independently configurable.
EVERON_SURFACE_TEXTURES: dict[str, str] = {
    material.code: material.everon_path
    for material in MILESTONE9_MATERIALS
    if material.everon_path is not None
}

NOGOVA_SURFACE_TEXTURES: dict[str, str] = {
    material.code: NOGOVA_GROUND_TEXTURE_CYCLE[index % len(NOGOVA_GROUND_TEXTURE_CYCLE)]
    for index, material in enumerate(MILESTONE9_MATERIALS)
}
NOGOVA_SURFACE_TEXTURES.update({
    "a": r"o\pole1.paa",
    "b": r"o\pole2.paa",
    # Keep sports turf on the same standard green grass texture as ordinary
    # grass.  The former o\b1.paa override is visibly lighter and creates pale
    # pitch rectangles that do not match the surrounding Nogova terrain.
    "y": NOGOVA_SURFACE_TEXTURES["g"],
    "x": r"o\ps.paa",
    # Nogova has no rock tile that blends cleanly with this terrain palette.
    # Rock/scree semantic cells therefore keep their ordinary Nogova cycle
    # texture; 3D stone objects may still be placed independently.
})

STOCK_SURFACE_TEXTURES: Mapping[str, Mapping[str, str]] = {
    "everon": EVERON_SURFACE_TEXTURES,
    "nogova": NOGOVA_SURFACE_TEXTURES,
}

MATERIAL_INDEX: Mapping[str, int] = {
    material.code: index for index, material in enumerate(MILESTONE9_MATERIALS)
}


MALDEN_SURFACE_COLOURS: Mapping[str, tuple[int, int, int]] = {
    "w": (48, 73, 91),
    "q": (124, 111, 82),
    "s": (184, 162, 109),
    "g": (109, 118, 70),
    "h": (132, 126, 77),
    "r": (110, 105, 96),
    "k": (88, 85, 80),
    "f": (64, 86, 48),
    "e": (82, 101, 57),
    # No assumed Malden farmland artwork: all farm semantics resolve to the
    # same basic grass terrain in the WRP texture table.
    "a": (109, 118, 70),
    "b": (109, 118, 70),
    "c": (109, 118, 70),
    "u": (132, 126, 115),
    "i": (115, 112, 105),
    "p": (61, 59, 55),
    "o": (118, 108, 91),
    "d": (120, 91, 59),
    "t": (108, 103, 69),
    "v": (100, 94, 80),
    "j": (111, 127, 73),
    "y": (111, 133, 78),
    "x": (190, 169, 111),
}

DESERT_SURFACE_COLOURS: Mapping[str, tuple[int, int, int]] = {
    "w": (49, 82, 98),
    "q": (153, 133, 91),
    "s": (205, 184, 120),
    "g": (155, 145, 88),
    "h": (174, 157, 96),
    "r": (128, 113, 91),
    "k": (102, 93, 80),
    "f": (96, 116, 66),
    "e": (124, 134, 75),
    "a": (184, 163, 96),
    "b": (157, 137, 83),
    "c": (125, 118, 69),
    "u": (145, 133, 116),
    "i": (121, 113, 103),
    "p": (67, 62, 55),
    "o": (144, 128, 94),
    "d": (135, 100, 65),
    "t": (139, 124, 78),
    "v": (101, 92, 74),
}


@dataclass(frozen=True, slots=True)
class SurfacePassReport:
    indices: tuple[int, ...]
    shoreline_cells: int
    softened_landuse_cells: int
    wet_shoreline_cells: int
    dry_shoreline_cells: int
    forest_edge_cells: int
    farmland_light_cells: int
    farmland_dark_cells: int
    field_boundary_cells: int
    urban_cells: int
    industrial_cells: int
    paved_road_cells: int
    paved_shoulder_cells: int
    dirt_road_cells: int
    dirt_blend_cells: int
    gravel_road_cells: int
    gravel_blend_cells: int
    mapped_grassland_cells: int
    mapped_park_cells: int
    mapped_sports_cells: int
    mapped_sand_cells: int
    aeroway_surface_cells: int
    rock_cells: int
    steep_rock_cells: int
    colour_reference_cells: int
    feature_seed_count: int

    def to_manifest(self) -> dict[str, object]:
        document = asdict(self)
        document.pop("indices", None)
        document["material_cells"] = {
            material.code: self.indices.count(index)
            for index, material in enumerate(MILESTONE9_MATERIALS)
        }
        return document


def _stable_u32(seed: str, feature_id: str, label: str, *values: int) -> int:
    payload = ":".join((seed, feature_id, label, *(str(value) for value in values))).encode("utf-8")
    return int.from_bytes(blake2s(payload, digest_size=4).digest(), "little")


def _stable_fraction(seed: str, feature_id: str, label: str, *values: int) -> float:
    return _stable_u32(seed, feature_id, label, *values) / 0xFFFFFFFF




def _stable_grid_fraction(seed: str, feature_id: str, label: str, cells: int) -> np.ndarray:
    """Return a deterministic [0, 1] pseudo-random field without per-cell hashing.

    Surface classification used to call BLAKE2 once for every natural terrain
    cell. On 2048² maps that means millions of Python hash constructions before
    land-use work even begins. Seed one 64-bit field from BLAKE2, then use a
    SplitMix-style integer mixer over x/z coordinates entirely in NumPy.
    """
    seed_bytes = blake2s(f"{seed}:{feature_id}:{label}".encode("utf-8"), digest_size=8).digest()
    base = np.uint64(int.from_bytes(seed_bytes, "little"))
    x = np.arange(cells, dtype=np.uint64)[None, :]
    z = np.arange(cells, dtype=np.uint64)[:, None]
    value = base ^ (x * np.uint64(0x9E3779B97F4A7C15)) ^ (z * np.uint64(0xD1B54A32D192ED03))
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return value.astype(np.float64) / float(np.iinfo(np.uint64).max)


def _feature_boundary_mask(grid: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of :func:`_feature_boundary` for a whole grid."""
    selected = grid != 0
    boundary = np.zeros(grid.shape, dtype=np.bool_)
    boundary[0, :] |= selected[0, :]
    boundary[-1, :] |= selected[-1, :]
    boundary[:, 0] |= selected[:, 0]
    boundary[:, -1] |= selected[:, -1]
    boundary[:, 1:] |= selected[:, 1:] & (grid[:, 1:] != grid[:, :-1])
    boundary[:, :-1] |= selected[:, :-1] & (grid[:, :-1] != grid[:, 1:])
    boundary[1:, :] |= selected[1:, :] & (grid[1:, :] != grid[:-1, :])
    boundary[:-1, :] |= selected[:-1, :] & (grid[:-1, :] != grid[1:, :])
    return boundary

def _draw_polygon(
    draw: ImageDraw.ImageDraw,
    feature: OsmPolygonFeature,
    projection: BboxProjection,
    resolution: int,
    fill: int,
) -> None:
    for polygon in feature.polygons:
        outer = [projection.to_pixel(point, resolution) for point in polygon.outer]
        if len(outer) >= 3:
            draw.polygon(outer, fill=fill)
        for hole in polygon.holes:
            points = [projection.to_pixel(point, resolution) for point in hole]
            if len(points) >= 3:
                draw.polygon(points, fill=0)


def _image_values(image: Image.Image):
    getter = getattr(image, "get_flattened_data", None)
    return getter() if getter is not None else image.getdata()


def _image_to_wrp_values(image: Image.Image, cells: int) -> np.ndarray:
    if image.size != (cells, cells):
        image = image.resize((cells, cells), Image.Resampling.NEAREST)
    values = np.asarray(image, dtype=np.int32)
    return np.flipud(values).reshape(-1).copy()


def _image_to_wrp_mask(image: Image.Image, cells: int, *, threshold: int = 64) -> np.ndarray:
    if image.size != (cells, cells):
        image = image.resize((cells, cells), Image.Resampling.BOX)
    values = np.asarray(image, dtype=np.uint8)
    return (np.flipud(values) >= threshold).reshape(-1).copy()


def _feature_grid(
    features: Sequence[OsmPolygonFeature],
    projection: BboxProjection,
    cells: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    image = Image.new("I", (cells, cells), 0)
    draw = ImageDraw.Draw(image)
    feature_ids = [""]
    for value, feature in enumerate(features, start=1):
        feature_ids.append(feature.osm_key)
        _draw_polygon(draw, feature, projection, cells, value)
    return _image_to_wrp_values(image, cells), tuple(feature_ids)


def _polygon_mask(
    features: Sequence[OsmPolygonFeature],
    projection: BboxProjection,
    cells: int,
) -> np.ndarray:
    image = Image.new("L", (cells * 4, cells * 4), 0)
    draw = ImageDraw.Draw(image)
    for feature in features:
        _draw_polygon(draw, feature, projection, cells * 4, 255)
    return _image_to_wrp_mask(image, cells, threshold=32)


def _aeroway_mask(
    dataset: OsmDataset,
    projection: BboxProjection,
    cells: int,
) -> np.ndarray:
    resolution = cells * 8
    image = Image.new("L", (resolution, resolution), 0)
    draw = ImageDraw.Draw(image)
    for feature in dataset.aeroway_areas:
        if feature.tags.get("aeroway", "").casefold() == "aerodrome":
            continue
        _draw_polygon(draw, feature, projection, resolution, 255)
    for feature in dataset.aeroway_lines:
        kind = feature.tags.get("aeroway", "").casefold()
        if kind not in {"runway", "taxiway"}:
            continue
        points = [projection.to_pixel(point, resolution) for point in feature.points]
        if len(points) < 2:
            continue
        try:
            tagged_width = float(str(feature.tags.get("width", "") or 0).replace(",", "."))
        except ValueError:
            tagged_width = 0.0
        default_width = 36.0 if kind == "runway" else 12.0
        width_metres = max(3.0, tagged_width or default_width)
        width_pixels = max(1, int(round(width_metres / projection.world_size * resolution)))
        draw.line(points, fill=255, width=width_pixels, joint="curve")
    return _image_to_wrp_mask(image, cells, threshold=20)


def _road_masks(
    dataset: OsmDataset,
    projection: BboxProjection,
    cells: int,
    cell_size: float,
    include_minor: bool,
    paved_shoulder_metres: float,
    dirt_blend_metres: float,
) -> tuple[
    tuple[bool, ...], tuple[bool, ...], tuple[bool, ...], tuple[bool, ...],
    tuple[bool, ...], tuple[bool, ...],
]:
    resolution = cells * 8
    images = [Image.new("L", (resolution, resolution), 0) for _ in range(6)]
    (
        paved_draw, paved_expanded_draw, dirt_draw, dirt_expanded_draw,
        gravel_draw, gravel_expanded_draw,
    ) = map(ImageDraw.Draw, images)
    for feature in dataset.roads:
        if not road_is_supported(feature.tags, include_minor=include_minor):
            continue
        points = [projection.to_pixel(point, resolution) for point in feature.points]
        if len(points) < 2:
            continue
        width_metres = road_width_metres(feature.tags)
        is_gravel = road_is_gravel(feature.tags)
        is_dirt = road_is_dirt(feature.tags) and not is_gravel
        expansion = dirt_blend_metres if is_dirt else (0.0 if is_gravel else paved_shoulder_metres)
        centre_pixels = max(1, int(round(width_metres / projection.world_size * resolution)))
        expanded_pixels = max(
            centre_pixels,
            int(round((width_metres + expansion * 2.0) / projection.world_size * resolution)),
        )
        if is_gravel:
            gravel_expanded_draw.line(points, fill=255, width=expanded_pixels, joint="curve")
            gravel_draw.line(points, fill=255, width=centre_pixels, joint="curve")
        elif is_dirt:
            dirt_expanded_draw.line(points, fill=255, width=expanded_pixels, joint="curve")
            dirt_draw.line(points, fill=255, width=centre_pixels, joint="curve")
        else:
            paved_expanded_draw.line(points, fill=255, width=expanded_pixels, joint="curve")
            paved_draw.line(points, fill=255, width=centre_pixels, joint="curve")
    return tuple(_image_to_wrp_mask(image, cells, threshold=20) for image in images)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class _OwnerField:
    ranks: np.ndarray
    labels: tuple[str, ...]

    def __getitem__(self, index: int) -> str:
        rank = int(self.ranks[index])
        return self.labels[rank] if rank >= 0 else ""


def _distance_array_from_mask(
    mask: Sequence[bool], cells: int, maximum: int
) -> np.ndarray:
    """Bounded Manhattan distance using vectorized wavefront dilation."""
    infinity = cells * cells + 1
    selected = np.asarray(mask, dtype=np.bool_).reshape((cells, cells))
    distances = np.full((cells, cells), infinity, dtype=np.int32)
    distances[selected] = 0
    frontier = selected.copy()
    unseen = ~selected
    for distance in range(1, maximum + 1):
        expanded = np.zeros_like(frontier)
        expanded[:, 1:] |= frontier[:, :-1]
        expanded[:, :-1] |= frontier[:, 1:]
        expanded[1:, :] |= frontier[:-1, :]
        expanded[:-1, :] |= frontier[1:, :]
        frontier = expanded & unseen
        if not np.any(frontier):
            break
        distances[frontier] = distance
        unseen[frontier] = False
    return distances.reshape(-1)


def _nearest_feature_owners(
    mask: Sequence[bool],
    feature_grid: Sequence[int],
    feature_ids: Sequence[str],
    cells: int,
    maximum: int,
    fallback_feature_id: str,
) -> _OwnerField:
    """Propagate nearest source ownership across a shallow transition band.

    Ownership used to be a Python string plus deque entry for every selected
    terrain cell. Large forest/water masks made that one of the heaviest parts
    of the surface pass. Propagate compact lexical ranks over the already-bounded
    distance field instead; ties still choose the lexicographically smaller ID.
    """
    selected = np.asarray(mask, dtype=np.bool_).reshape(-1)
    grid = np.asarray(feature_grid, dtype=np.int32).reshape(-1)
    distances = _distance_array_from_mask(selected, cells, maximum).reshape((cells, cells))

    labels = tuple(sorted(set(feature_ids[1:]) | {fallback_feature_id}))
    label_rank = {label: rank for rank, label in enumerate(labels)}
    fallback_rank = label_rank[fallback_feature_id]
    feature_ranks = np.full(max(1, len(feature_ids)), fallback_rank, dtype=np.int32)
    for feature_number in range(1, len(feature_ids)):
        feature_ranks[feature_number] = label_rank[feature_ids[feature_number]]

    ranks = np.full(cells * cells, -1, dtype=np.int32)
    if np.any(selected):
        selected_numbers = grid[selected]
        valid = (selected_numbers > 0) & (selected_numbers < len(feature_ranks))
        seeded = np.full(selected_numbers.shape, fallback_rank, dtype=np.int32)
        seeded[valid] = feature_ranks[selected_numbers[valid]]
        ranks[selected] = seeded
    ranks_grid = ranks.reshape((cells, cells))
    sentinel = np.iinfo(np.int32).max

    for distance in range(1, maximum + 1):
        current = distances == distance
        if not np.any(current):
            continue
        best = np.full((cells, cells), sentinel, dtype=np.int32)

        valid = distances[:, :-1] == distance - 1
        candidate = np.where(valid, ranks_grid[:, :-1], sentinel)
        np.minimum(best[:, 1:], candidate, out=best[:, 1:])

        valid = distances[:, 1:] == distance - 1
        candidate = np.where(valid, ranks_grid[:, 1:], sentinel)
        np.minimum(best[:, :-1], candidate, out=best[:, :-1])

        valid = distances[:-1, :] == distance - 1
        candidate = np.where(valid, ranks_grid[:-1, :], sentinel)
        np.minimum(best[1:, :], candidate, out=best[1:, :])

        valid = distances[1:, :] == distance - 1
        candidate = np.where(valid, ranks_grid[1:, :], sentinel)
        np.minimum(best[:-1, :], candidate, out=best[:-1, :])

        assign = current & (best != sentinel)
        ranks_grid[assign] = best[assign]

    return _OwnerField(ranks_grid.reshape(-1), labels)


def _distance_from_mask(mask: Sequence[bool], cells: int, maximum: int | None = None) -> tuple[int, ...]:
    infinity = cells * cells + 1
    if maximum is not None:
        return tuple(int(value) for value in _distance_array_from_mask(mask, cells, maximum))

    distances = [infinity] * len(mask)
    queue: deque[int] = deque()
    for index, selected in enumerate(mask):
        if selected:
            distances[index] = 0
            queue.append(index)
    while queue:
        index = queue.popleft()
        distance = distances[index]
        x, z = index % cells, index // cells
        for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
            if not (0 <= nx < cells and 0 <= nz < cells):
                continue
            neighbour = nz * cells + nx
            if distances[neighbour] > distance + 1:
                distances[neighbour] = distance + 1
                queue.append(neighbour)
    return tuple(distances)


def _boundary(mask: Sequence[bool], cells: int, index: int) -> bool:
    x, z = index % cells, index // cells
    for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
        if not (0 <= nx < cells and 0 <= nz < cells):
            return True
        if not mask[nz * cells + nx]:
            return True
    return False


def _feature_boundary(grid: Sequence[int], cells: int, index: int) -> bool:
    value = grid[index]
    if value == 0:
        return False
    x, z = index % cells, index // cells
    for nx, nz in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
        if not (0 <= nx < cells and 0 <= nz < cells):
            return True
        if grid[nz * cells + nx] != value:
            return True
    return False


def _load_reference(path: Path | None, cells: int) -> tuple[tuple[int, int, int], ...] | None:
    if path is None:
        return None
    if not path.is_file():
        raise ValueError(f"colour reference does not exist: {path}")
    with Image.open(path) as source:
        source.load()
        image = source.convert("RGB").resize((cells, cells), Image.Resampling.BILINEAR)
        pixels = tuple(tuple(int(channel) for channel in pixel) for pixel in _image_values(image))
    return tuple(pixels[(cells - 1 - z) * cells + x] for z in range(cells) for x in range(cells))


def _reference_natural_material(
    base_index: int,
    reference: tuple[int, int, int],
    strength: float,
) -> int:
    if strength <= 0:
        return base_index
    candidates = (MATERIAL_INDEX["s"], MATERIAL_INDEX["g"], MATERIAL_INDEX["h"], MATERIAL_INDEX["r"])
    base = MILESTONE9_MATERIALS[base_index].colour
    target = tuple(round(base[channel] * (1.0 - strength) + reference[channel] * strength) for channel in range(3))
    return min(
        candidates,
        key=lambda candidate: (
            sum((target[channel] - MILESTONE9_MATERIALS[candidate].colour[channel]) ** 2 for channel in range(3)),
            candidate,
        ),
    )


def _texture_material_for_profile(
    material: SurfaceMaterialDefinition,
    profile: str,
) -> SurfaceMaterialDefinition:
    if profile == "malden":
        return SurfaceMaterialDefinition(
            material.code,
            material.name,
            MALDEN_SURFACE_COLOURS.get(material.code, material.colour),
            material.everon_path,
        )
    if profile == "desert":
        return SurfaceMaterialDefinition(
            material.code,
            material.name,
            DESERT_SURFACE_COLOURS.get(material.code, material.colour),
            material.everon_path,
        )
    return material


def build_surface_pass(
    dataset: OsmDataset,
    projection: BboxProjection,
    raster: OsmRaster,
    elevations: Sequence[float],
    slopes: Sequence[float],
    spec,
) -> SurfacePassReport:
    cells = spec.cells
    expected = cells * cells
    if len(elevations) != expected or len(slopes) != expected:
        raise ValueError("surface pass terrain grids have the wrong size")

    # The OSM mask may contain elevated ponds that the terrain solver correctly
    # preserved because CWA has no elevated water surfaces. Treat mapped water as
    # actual surface water only where the solved terrain reaches the global plane.
    surface_water = (
        np.asarray(raster.water, dtype=np.bool_)
        & (np.asarray(elevations, dtype=np.float64) <= float(spec.sea_level) + 1.0e-7)
    )

    wet_cells = int(getattr(spec, "surface_shoreline_wet_cells", 1))
    sand_cells = int(getattr(spec, "surface_shoreline_sand_cells", 2))
    transition_cells = int(getattr(spec, "surface_transition_cells", 2))
    forest_edge_cells = int(getattr(spec, "surface_forest_edge_cells", 1))
    field_width = int(getattr(spec, "surface_farmland_strip_cells", 4))
    shoulder_metres = float(getattr(spec, "surface_road_shoulder_metres", max(2.0, spec.cell_size * 0.18)))
    dirt_blend_metres = float(getattr(spec, "surface_dirt_blend_metres", max(2.0, spec.cell_size * 0.22)))
    steep_slope = float(getattr(spec, "surface_steep_slope_degrees", max(spec.rock_slope_degrees + 8.0, 36.0)))
    reference_strength = float(getattr(spec, "surface_colour_reference_strength", 0.0))
    reference_path = getattr(spec, "surface_colour_reference_path", None)
    seed = str(spec.deterministic_seed)

    water_grid, water_ids = _feature_grid(dataset.water, projection, cells)
    farmland_grid, farmland_ids = _feature_grid(dataset.farmland, projection, cells)
    forest_grid, forest_ids = _feature_grid(dataset.forests, projection, cells)
    urban_grid, urban_ids = _feature_grid(dataset.urban, projection, cells)
    industrial_features = tuple(
        feature for feature in dataset.urban
        if feature.tags.get("landuse", "").casefold() in {"industrial", "commercial", "railway", "construction"}
        or feature.tags.get("category", "").casefold() == "industrial"
    )
    industrial_mask = _polygon_mask(industrial_features, projection, cells)
    industrial_buildings = tuple(
        feature for feature in dataset.building_polygons
        if feature.tags.get("building", "").casefold() in {"industrial", "warehouse", "hangar", "factory", "manufacture"}
        or feature.tags.get("man_made", "").casefold() in {"works", "silo", "storage_tank"}
    )
    industrial_building_mask = _polygon_mask(industrial_buildings, projection, cells)
    grassland_mask = _polygon_mask(tuple(
        feature for feature in dataset.surface_areas if feature.tags.get("surface_kind", feature.tags.get("natural", "")).casefold() == "grassland"
    ), projection, cells)
    park_mask = _polygon_mask(tuple(
        feature for feature in dataset.surface_areas
        if feature.tags.get("surface_kind", "").casefold() == "park"
        or feature.tags.get("leisure", "").casefold() == "park"
    ), projection, cells)
    sports_mask = _polygon_mask(tuple(
        feature for feature in dataset.surface_areas
        if feature.tags.get("surface_kind", "").casefold() == "sports_pitch"
        or feature.tags.get("leisure", "").casefold() == "pitch"
        or bool(feature.tags.get("sport", "").strip())
    ), projection, cells)
    mapped_beach_mask = _polygon_mask(tuple(
        feature for feature in dataset.surface_areas
        if feature.tags.get("surface_kind", "").casefold() == "beach"
        or feature.tags.get("natural", "").casefold() == "beach"
    ), projection, cells)
    mapped_sand_mask = _polygon_mask(tuple(
        feature for feature in dataset.surface_areas
        if feature.tags.get("surface_kind", feature.tags.get("natural", "")).casefold() == "sand"
    ), projection, cells)
    aeroway_mask = _aeroway_mask(dataset, projection, cells)

    paved, paved_expanded, dirt, dirt_expanded, gravel, gravel_expanded = _road_masks(
        dataset,
        projection,
        cells,
        spec.cell_size,
        spec.include_minor_roads,
        shoulder_metres,
        dirt_blend_metres,
    )
    maximum_shore_distance = wet_cells + sand_cells + transition_cells
    water_distance = _distance_array_from_mask(surface_water, cells, maximum_shore_distance)
    coastline_signature = "coastline:" + "|".join(sorted(feature.osm_key for feature in dataset.coastlines))
    if coastline_signature == "coastline:":
        coastline_signature = "water-mask"
    water_owners = _nearest_feature_owners(
        surface_water, water_grid, water_ids, cells, maximum_shore_distance, coastline_signature
    )
    outside_forest = np.logical_not(np.asarray(raster.forest, dtype=np.bool_))
    forest_inside_distance = _distance_array_from_mask(outside_forest, cells, forest_edge_cells + 1)
    forest_distance = _distance_array_from_mask(raster.forest, cells, forest_edge_cells + 1)
    forest_owners = _nearest_feature_owners(
        raster.forest, forest_grid, forest_ids, cells, forest_edge_cells + 1, "forest-mask"
    )
    reference = _load_reference(reference_path, cells)

    # Keep classification grids in compact NumPy arrays. The old implementation
    # repeatedly walked several-million-element Python tuples for every material
    # layer; most of these layers are simple boolean masks and are far cheaper in
    # vectorized form.
    result = np.full(expected, MATERIAL_INDEX["g"], dtype=np.int16)
    water_np = np.asarray(surface_water, dtype=np.bool_)
    elevations_np = np.asarray(elevations, dtype=np.float64)
    slopes_np = np.asarray(slopes, dtype=np.float64)
    buildings_np = np.asarray(raster.buildings, dtype=np.bool_)
    roads_np = np.asarray(raster.roads, dtype=np.bool_)
    farmland_raster_np = np.asarray(raster.farmland, dtype=np.bool_)
    forest_raster_np = np.asarray(raster.forest, dtype=np.bool_)
    urban_raster_np = np.asarray(raster.urban, dtype=np.bool_)
    farmland_grid_np = np.asarray(farmland_grid, dtype=np.int32)
    forest_grid_np = np.asarray(forest_grid, dtype=np.int32)
    urban_grid_np = np.asarray(urban_grid, dtype=np.int32)
    feature_seed_ids: set[str] = set()

    # Natural base. The visual dry-grass breakup remains deterministic, but is
    # generated as one vectorized integer field rather than millions of BLAKE2
    # calls. This intentionally changes only the pseudo-random micro-pattern.
    base_water = water_np | (elevations_np <= spec.sea_level)
    steep_mask = (~base_water) & (slopes_np >= steep_slope)
    rock_mask = (~base_water) & (~steep_mask) & (slopes_np >= spec.rock_slope_degrees)
    natural_mask = ~(base_water | steep_mask | rock_mask)
    result[base_water] = MATERIAL_INDEX["w"]
    result[steep_mask] = MATERIAL_INDEX["k"]
    result[rock_mask] = MATERIAL_INDEX["r"]
    dryness = _stable_grid_fraction(seed, "world", "dry-grass", cells).reshape(-1)
    result[natural_mask] = np.where(
        dryness[natural_mask] < 0.18, MATERIAL_INDEX["h"], MATERIAL_INDEX["g"]
    )
    reference_cells = 0
    if reference is not None and reference_strength > 0:
        for index in np.flatnonzero(natural_mask):
            result[index] = _reference_natural_material(
                int(result[index]), reference[int(index)], reference_strength
            )
            reference_cells += 1

    # Farmland subdivisions. Work feature-by-feature so orientation/phase hashing
    # occurs once per OSM field rather than once per cell.
    x_grid = np.tile(np.arange(cells, dtype=np.int32), cells)
    z_grid = np.repeat(np.arange(cells, dtype=np.int32), cells)
    farmland_boundary = _feature_boundary_mask(farmland_grid_np.reshape((cells, cells))).reshape(-1)
    farmland_valid = (
        (farmland_grid_np > 0)
        & (~water_np)
        & (result != MATERIAL_INDEX["r"])
        & (result != MATERIAL_INDEX["k"])
    )
    for feature_number in np.unique(farmland_grid_np[farmland_valid]):
        if feature_number <= 0:
            continue
        feature_id = farmland_ids[int(feature_number)]
        feature_seed_ids.add(feature_id)
        selected = farmland_valid & (farmland_grid_np == feature_number)
        boundary = selected & farmland_boundary
        result[boundary] = MATERIAL_INDEX["c"]
        interior = selected & (~farmland_boundary)
        if not np.any(interior):
            continue
        orientation = _stable_u32(seed, feature_id, "field-orientation") % 4
        if orientation == 0:
            coordinate = x_grid
        elif orientation == 1:
            coordinate = z_grid
        elif orientation == 2:
            coordinate = x_grid + z_grid
        else:
            coordinate = x_grid - z_grid
        phase = _stable_u32(seed, feature_id, "field-phase") % max(1, field_width * 2)
        stripe_even = ((coordinate + phase) // max(1, field_width)) % 2 == 0
        result[interior & stripe_even] = MATERIAL_INDEX["a"]
        result[interior & (~stripe_even)] = MATERIAL_INDEX["b"]

    # Forest interior and ground-edge band.
    forest_inside_np = np.asarray(forest_inside_distance, dtype=np.int32)
    forest_valid = (
        (forest_grid_np > 0)
        & (~water_np)
        & (result != MATERIAL_INDEX["r"])
        & (result != MATERIAL_INDEX["k"])
    )
    result[forest_valid & (forest_inside_np <= forest_edge_cells)] = MATERIAL_INDEX["e"]
    result[forest_valid & (forest_inside_np > forest_edge_cells)] = MATERIAL_INDEX["f"]
    for feature_number in np.unique(forest_grid_np[forest_valid]):
        if feature_number > 0:
            feature_seed_ids.add(forest_ids[int(feature_number)])
    forest_distance_np = np.asarray(forest_distance, dtype=np.int32)
    outer_candidates = np.flatnonzero(
        (forest_distance_np == 1)
        & (~water_np)
        & (~forest_raster_np)
        & (~farmland_raster_np)
        & (~urban_raster_np)
    )
    for raw_index in outer_candidates:
        index = int(raw_index)
        feature_id = forest_owners[index] or "forest-mask"
        feature_seed_ids.add(feature_id)
        if _stable_fraction(seed, feature_id, "forest-outer-edge", index % cells, index // cells) < 0.68:
            result[index] = MATERIAL_INDEX["e"]

    # Urban areas. Solid interiors are vectorized; only stochastic polygon-edge
    # breakup requires per-cell hashing.
    industrial_np = np.asarray(industrial_mask, dtype=np.bool_) | np.asarray(industrial_building_mask, dtype=np.bool_)
    urban_valid = (urban_grid_np > 0) & (~water_np)
    result[urban_valid & industrial_np] = MATERIAL_INDEX["i"]
    result[urban_valid & (~industrial_np)] = MATERIAL_INDEX["u"]
    for feature_number in np.unique(urban_grid_np[urban_valid]):
        if feature_number > 0:
            feature_seed_ids.add(urban_ids[int(feature_number)])
    softened = 0
    urban_boundary = _feature_boundary_mask(urban_grid_np.reshape((cells, cells))).reshape(-1)
    for raw_index in np.flatnonzero(urban_valid & urban_boundary):
        index = int(raw_index)
        feature_id = urban_ids[int(urban_grid_np[index])]
        if _stable_fraction(seed, feature_id, "urban-edge", index % cells, index // cells) < 0.24:
            result[index] = MATERIAL_INDEX["g"]
            softened += 1
    nonurban_buildings = buildings_np & (~water_np) & (urban_grid_np == 0)
    result[nonurban_buildings & industrial_np] = MATERIAL_INDEX["i"]
    result[nonurban_buildings & (~industrial_np)] = MATERIAL_INDEX["u"]

    # Explicit OSM natural/leisure polygons, preserving the historical priority.
    eligible = (~water_np) & (~buildings_np)
    beach_np = np.asarray(mapped_beach_mask, dtype=np.bool_)
    sand_np = np.asarray(mapped_sand_mask, dtype=np.bool_)
    sports_np = np.asarray(sports_mask, dtype=np.bool_)
    park_np = np.asarray(park_mask, dtype=np.bool_)
    grass_np = np.asarray(grassland_mask, dtype=np.bool_)
    beach_selected = eligible & beach_np
    sand_selected = eligible & (~beach_np) & sand_np
    sports_selected = eligible & (~beach_np) & (~sand_np) & sports_np
    park_selected = eligible & (~beach_np) & (~sand_np) & (~sports_np) & park_np
    grass_selected = eligible & (~beach_np) & (~sand_np) & (~sports_np) & (~park_np) & grass_np
    result[grass_selected] = MATERIAL_INDEX["g"]
    result[park_selected] = MATERIAL_INDEX["j"]
    result[sports_selected] = MATERIAL_INDEX["y"]
    result[sand_selected] = MATERIAL_INDEX["s"]
    result[beach_selected] = MATERIAL_INDEX["x"]
    grassland_count = int(np.count_nonzero(grass_selected))
    park_count = int(np.count_nonzero(park_selected))
    sports_count = int(np.count_nonzero(sports_selected))
    mapped_sand_count = int(np.count_nonzero(beach_selected | sand_selected))

    aeroway_np = np.asarray(aeroway_mask, dtype=np.bool_)
    aeroway_selected = aeroway_np & (~water_np) & (~buildings_np)
    result[aeroway_selected] = MATERIAL_INDEX["p"]
    aeroway_count = int(np.count_nonzero(aeroway_selected))

    # Multi-band shoreline. This region is normally only a thin perimeter, so
    # keep source-owner hashing on the candidate cells while avoiding a full-grid
    # Python scan.
    wet_count = 0
    dry_count = 0
    water_distance_np = np.asarray(water_distance, dtype=np.int32)
    shore_candidates = np.flatnonzero(
        (~water_np) & (~roads_np) & (~buildings_np)
        & (water_distance_np > 0)
        & (water_distance_np <= maximum_shore_distance)
    )
    for raw_index in shore_candidates:
        index = int(raw_index)
        distance = int(water_distance_np[index])
        feature_id = water_owners[index] or coastline_signature
        feature_seed_ids.add(feature_id)
        jitter = _stable_fraction(seed, feature_id, "shoreline-band", index % cells, index // cells) - 0.5
        adjusted = distance + jitter * 0.70
        if adjusted <= wet_cells:
            result[index] = MATERIAL_INDEX["q"]
            wet_count += 1
        elif adjusted <= wet_cells + sand_cells:
            result[index] = MATERIAL_INDEX["s"]
            dry_count += 1
        elif adjusted <= wet_cells + sand_cells + transition_cells:
            if _stable_fraction(seed, feature_id, "shoreline-outer", index % cells, index // cells) < 0.48:
                result[index] = MATERIAL_INDEX["h"]
                softened += 1

    # Roads and shoulders are pure mask overlays.
    paved_np = np.asarray(paved, dtype=np.bool_)
    paved_expanded_np = np.asarray(paved_expanded, dtype=np.bool_)
    dirt_np = np.asarray(dirt, dtype=np.bool_)
    dirt_expanded_np = np.asarray(dirt_expanded, dtype=np.bool_)
    gravel_np = np.asarray(gravel, dtype=np.bool_)
    valid_road_ground = (~water_np) & (~buildings_np)
    paved_shoulder = valid_road_ground & paved_expanded_np & (~paved_np) & (~dirt_np)
    dirt_shoulder = valid_road_ground & dirt_expanded_np & (~dirt_np) & (~paved_np) & (~gravel_np)
    result[paved_shoulder] = MATERIAL_INDEX["o"]
    result[dirt_shoulder] = MATERIAL_INDEX["t"]
    shoulder_count = int(np.count_nonzero(paved_shoulder))
    dirt_blend_count = int(np.count_nonzero(dirt_shoulder))
    gravel_blend_count = 0

    dirt_centres = (~water_np) & dirt_np & (~paved_np)
    paved_centres = (~water_np) & paved_np
    result[dirt_centres] = MATERIAL_INDEX["d"]
    result[paved_centres] = MATERIAL_INDEX["p"]
    result[water_np] = MATERIAL_INDEX["w"]
    paved_count = int(np.count_nonzero((~water_np) & paved_np))
    dirt_count = int(np.count_nonzero((~water_np) & (~paved_np) & dirt_np))
    gravel_count = int(np.count_nonzero((~water_np) & (~paved_np) & (~dirt_np) & gravel_np))

    counts_array = np.bincount(result.astype(np.int32), minlength=len(MILESTONE9_MATERIALS))
    counts = {code: int(counts_array[material_index]) for code, material_index in MATERIAL_INDEX.items()}
    return SurfacePassReport(
        indices=tuple(int(value) for value in result),
        shoreline_cells=wet_count + dry_count,
        softened_landuse_cells=softened,
        wet_shoreline_cells=wet_count,
        dry_shoreline_cells=dry_count,
        forest_edge_cells=counts["e"],
        farmland_light_cells=counts["a"],
        farmland_dark_cells=counts["b"],
        field_boundary_cells=counts["c"],
        urban_cells=counts["u"],
        industrial_cells=counts["i"],
        paved_road_cells=paved_count,
        paved_shoulder_cells=shoulder_count,
        dirt_road_cells=dirt_count,
        dirt_blend_cells=dirt_blend_count,
        gravel_road_cells=gravel_count,
        gravel_blend_cells=gravel_blend_count,
        mapped_grassland_cells=grassland_count,
        mapped_park_cells=park_count,
        mapped_sports_cells=sports_count,
        mapped_sand_cells=mapped_sand_count,
        aeroway_surface_cells=aeroway_count,
        rock_cells=counts["r"],
        steep_rock_cells=counts["k"],
        colour_reference_cells=reference_cells,
        feature_seed_count=len(feature_seed_ids),
    )


def surface_texture_wire_paths(world_name: str, profile: str) -> tuple[str, ...]:
    if profile not in {"generated", "everon", "nogova", "malden", "desert"}:
        raise ValueError("ground texture profile must be nogova, malden, everon, desert or generated")
    paths: list[str] = []
    stock_paths = STOCK_SURFACE_TEXTURES.get(profile, {})
    for material in MILESTONE9_MATERIALS:
        if material.code in stock_paths:
            paths.append(stock_paths[material.code])
        elif profile == "malden" and material.code in {"a", "b", "c"}:
            # Malden has farming, but this preset intentionally does not invent a
            # dedicated field texture. Reuse the basic ground tile instead.
            paths.append(rf"{world_name}\data\g.paa")
        else:
            paths.append(rf"{world_name}\data\{material.code}.paa")
    return tuple(paths)

def external_surface_texture_paths(profile: str) -> tuple[str, ...]:
    stock_paths = STOCK_SURFACE_TEXTURES.get(profile)
    if stock_paths is None:
        return ()
    return tuple(
        stock_paths[material.code]
        for material in MILESTONE9_MATERIALS
        if material.code in stock_paths
    )


def _noise(seed_value: int, x: int, y: int) -> int:
    value = (seed_value ^ (x * 0x9E3779B1) ^ (y * 0x85EBCA77)) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    value ^= value >> 16
    return value


def _clamp_colour(colour: Sequence[int]) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(value))) for value in colour)  # type: ignore[return-value]


def create_surface_texture(material: SurfaceMaterialDefinition, seed: str, size: int = 128) -> Image.Image:
    """Build a deterministic high-frequency terrain tile.

    The old generator used one fine grain value plus one broad 8-pixel value.
    At Resistance-era 512 px sizes that looked like enlarged coloured noise.
    Use several octaves and material-specific microstructure so extra resolution
    produces actual detail rather than merely more pixels.
    """
    if size < 16 or size & (size - 1):
        raise ValueError("surface texture size must be a power of two of at least 16")
    seed_value = _stable_u32(seed, material.code, "surface-texture-hq-v2")
    image = Image.new("RGB", (size, size), material.colour)
    pixels = image.load()
    scale = max(1, size // 128)
    for y in range(size):
        for x in range(size):
            fine = ((_noise(seed_value, x, y) & 31) - 15) * 0.42
            small = ((_noise(seed_value ^ 0x7A31D4E9, x // max(1, scale), y // max(1, scale)) & 31) - 15) * 0.34
            medium = ((_noise(seed_value ^ 0xA5A5A5A5, x // (4 * scale), y // (4 * scale)) & 31) - 15) * 0.28
            broad = ((_noise(seed_value ^ 0x1F123BB5, x // (16 * scale), y // (16 * scale)) & 31) - 15) * 0.22
            base = material.colour
            variation = int(round(fine + small + medium + broad))
            colour = [base[channel] + variation for channel in range(3)]
            code = material.code
            value = _noise(seed_value ^ 0xD00DFEED, x, y)

            if code in {"a", "b"}:
                row = (x + (y // max(1, 4 * scale)) * 3) % max(6, 18 * scale)
                if row < max(1, 2 * scale):
                    colour = [channel - 18 for channel in colour]
                if value % 97 < 3:
                    colour = [colour[0] + 8, colour[1] + 6, colour[2] - 2]
            elif code == "c":
                if (x + y) % max(4, 12 * scale) < max(2, 5 * scale):
                    colour = [colour[0] - 12, colour[1] + 7, colour[2] - 8]
            elif code in {"p", "o", "d", "t", "v"}:
                aggregate = _noise(seed_value ^ 0xC3C3C3C3, x // max(1, 2 * scale), y // max(1, 2 * scale)) & 63
                if aggregate < 9:
                    colour = [channel + (20 if code in {"p", "o"} else 12) for channel in colour]
                elif aggregate > 57:
                    colour = [channel - 12 for channel in colour]
                if code in {"d", "t"} and y % max(8, 24 * scale) < max(1, 2 * scale):
                    colour = [channel - 10 for channel in colour]
            elif code in {"r", "k"}:
                # Fine mineral flecks plus broader fissures. The steep variant is
                # darker and more fractured so large bare hills do not read as a
                # single flat grey sheet.
                fleck = value & 127
                if fleck < (14 if code == "k" else 9):
                    lift = 19 if (value >> 8) & 1 else -22
                    colour = [channel + lift for channel in colour]
                fissure_period = max(19, (41 if code == "k" else 53) * scale)
                fissure = (x * 3 + y * 5 + (seed_value & 63)) % fissure_period
                if fissure < max(1, 2 * scale):
                    colour = [channel - (31 if code == "k" else 24) for channel in colour]
            elif code in {"f", "e", "g", "h", "j", "y"}:
                if value % 43 < 5:
                    colour = [colour[0] - 8, colour[1] + 12, colour[2] - 5]
                if (value >> 9) % 113 < 3:
                    colour = [colour[0] + 5, colour[1] - 7, colour[2] + 2]
            elif code in {"u", "i"}:
                seam_period = max(16, 32 * scale)
                seam = x % seam_period < max(1, scale) or y % seam_period < max(1, scale)
                if seam:
                    colour = [channel - 17 for channel in colour]
                elif value % 89 < 4:
                    colour = [channel + 9 for channel in colour]
            elif code in {"q", "s", "x"}:
                if value % 31 < 5:
                    colour = [colour[0] - 9, colour[1] - 6, colour[2] + 2]
                elif value % 67 < 3:
                    colour = [colour[0] + 10, colour[1] + 8, colour[2] + 2]
            pixels[x, y] = _clamp_colour(colour)

    if size >= 128:
        image = image.filter(ImageFilter.UnsharpMask(radius=0.55, percent=45, threshold=4))
    return image


def write_surface_textures(
    source_dir: Path,
    world_name: str,
    profile: str,
    seed: str,
    size: int,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    stock_paths = STOCK_SURFACE_TEXTURES.get(profile, {})
    for material in MILESTONE9_MATERIALS:
        if material.code in stock_paths:
            continue
        texture_material = _texture_material_for_profile(material, profile)
        path = source_dir / "data" / f"{material.code}.paa"
        if material.code == "v":
            write_rgb_dxt1_paa(path, create_gravel_road_texture_image(size))
        else:
            write_rgb_dxt1_paa(path, create_surface_texture(texture_material, seed, size))
        paths.append(path)
    return tuple(paths)


def _surface_image(indices: Sequence[int], elevations: Sequence[float], slopes: Sequence[float], cells: int) -> Image.Image:
    minimum = min(elevations)
    maximum = max(elevations)
    span = max(1e-6, maximum - minimum)
    image = Image.new("RGB", (cells, cells))
    pixels = image.load()
    for z in range(cells):
        y = cells - 1 - z
        for x in range(cells):
            index = z * cells + x
            base = MILESTONE9_MATERIALS[indices[index]].colour
            height = (elevations[index] - minimum) / span
            shade = 0.72 + height * 0.30 - min(50.0, slopes[index]) / 230.0
            pixels[x, y] = _clamp_colour(channel * max(0.48, min(1.12, shade)) for channel in base)
    return image


def render_overview_map(
    path: Path,
    indices: Sequence[int],
    elevations: Sequence[float],
    slopes: Sequence[float],
    dataset: OsmDataset,
    projection: BboxProjection,
    cells: int,
    size: int,
    *,
    towns: Sequence[object] = (),
    reference_path: Path | None = None,
    building_mask: Sequence[bool] | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> Image.Image:
    if size < 128 or size > 4096 or size & (size - 1):
        raise ValueError("overview size must be a power of two within 128..4096")
    if building_mask is not None and len(building_mask) != cells * cells:
        raise ValueError("overview building mask size does not match terrain grid")
    if progress_callback is not None:
        progress_callback(0, f"Rendering {cells:,}×{cells:,} terrain overview base")
    base = _surface_image(indices, elevations, slopes, cells).resize((size, size), Image.Resampling.BILINEAR)
    base = ImageEnhance.Contrast(base).enhance(1.08)
    if progress_callback is not None:
        progress_callback(18, "Applying overview contrast and colour reference")
    if reference_path is not None and reference_path.is_file():
        with Image.open(reference_path) as reference_source:
            reference = reference_source.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        reference = ImageEnhance.Color(reference).enhance(0.50).filter(ImageFilter.GaussianBlur(max(1, size // 512)))
        base = Image.blend(base, reference, 0.15)

    draw = ImageDraw.Draw(base)
    scale = size / projection.world_size
    road_total = len(dataset.roads)
    road_step = max(1, road_total // 20)
    if progress_callback is not None:
        progress_callback(28, f"Drawing {road_total:,} overview road lines")
    for road_index, feature in enumerate(dataset.roads, start=1):
        if progress_callback is not None and (road_index == road_total or road_index % road_step == 0):
            progress_callback(28 + min(37, round(road_index / max(1, road_total) * 37)), f"Drew overview roads {road_index:,}/{road_total:,}")
        if not road_is_supported(feature.tags, include_minor=True):
            continue
        points = []
        for point in feature.points:
            x, z = projection.to_world(point)
            points.append((x * scale, size - z * scale))
        width = max(1, round(road_width_metres(feature.tags) * scale))
        if road_is_gravel(feature.tags):
            fill = (92, 88, 80)
        else:
            fill = (112, 82, 50) if road_is_dirt(feature.tags) else (54, 54, 54)
        if len(points) >= 2:
            draw.line(points, fill=fill, width=width, joint="curve")
    if building_mask is not None:
        if progress_callback is not None:
            progress_callback(68, f"Rasterizing overview buildings from {cells:,}×{cells:,} placement mask")
        mask_bytes = bytearray(cells * cells)
        for z in range(cells):
            source = z * cells
            target = (cells - 1 - z) * cells
            mask_bytes[target : target + cells] = bytes(255 if value else 0 for value in building_mask[source : source + cells])
        mask = Image.frombytes("L", (cells, cells), bytes(mask_bytes)).resize((size, size), Image.Resampling.NEAREST)
        building_colour = Image.new("RGB", (size, size), (213, 205, 182))
        base.paste(building_colour, (0, 0), mask)
        draw = ImageDraw.Draw(base)
    else:
        building_total = len(dataset.building_polygons)
        building_step = max(1, building_total // 20)
        if progress_callback is not None:
            progress_callback(68, f"Drawing {building_total:,} overview building polygons")
        for building_index, feature in enumerate(dataset.building_polygons, start=1):
            if progress_callback is not None and (building_index == building_total or building_index % building_step == 0):
                progress_callback(68 + min(20, round(building_index / max(1, building_total) * 20)), f"Drew overview buildings {building_index:,}/{building_total:,}")
            for polygon in feature.polygons:
                points = []
                for point in polygon.outer:
                    x, z = projection.to_world(point)
                    points.append((x * scale, size - z * scale))
                if len(points) >= 3:
                    draw.polygon(points, fill=(213, 205, 182), outline=(92, 82, 72))
    if progress_callback is not None:
        progress_callback(90, f"Drawing {len(towns):,} town labels and map furniture")
    for town in towns:
        x = int(round(float(getattr(town, "x")) * scale))
        y = int(round(size - float(getattr(town, "z")) * scale))
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(240, 235, 218), outline=(25, 25, 25))
        name = str(getattr(town, "name", ""))
        if name:
            draw.text((x + 5, y - 6), name, fill=(20, 20, 20), stroke_width=1, stroke_fill=(235, 230, 210))

    border = max(2, size // 256)
    draw.rectangle((border, border, size - border - 1, size - border - 1), outline=(30, 30, 30), width=border)
    arrow_x = size - max(22, size // 18)
    arrow_y = max(24, size // 18)
    arrow = max(10, size // 42)
    draw.polygon(((arrow_x, arrow_y - arrow), (arrow_x - arrow // 2, arrow_y + arrow), (arrow_x + arrow // 2, arrow_y + arrow)), fill=(240, 240, 230), outline=(20, 20, 20))
    draw.text((arrow_x - 4, arrow_y + arrow + 2), "N", fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    base.save(path, format="PNG", optimize=False)
    if progress_callback is not None:
        progress_callback(100, "Overview PNG rendered")
    return base


BUILDING_SOURCE_REFERENCE_COLOURS: Mapping[str, tuple[int, int, int]] = {
    "osm": (74, 156, 255),
    "overture": (255, 178, 58),
    "generated": (116, 224, 120),
}


def render_building_source_reference(
    path: Path,
    dataset: OsmDataset,
    projection: BboxProjection,
    building_placement_plans: Sequence[BuildingPlacementPlan],
    size: int,
) -> Image.Image:
    if size < 128 or size > 4096 or size & (size - 1):
        raise ValueError("building source reference size must be a power of two within 128..4096")
    image = Image.new("RGB", (size, size), (30, 32, 36))
    draw = ImageDraw.Draw(image)
    scale = size / projection.world_size

    def screen(point: tuple[float, float]) -> tuple[float, float]:
        x, z = point
        return x * scale, size - z * scale

    def draw_world_polygon(points: Sequence[tuple[float, float]], colour: tuple[int, int, int]) -> None:
        if len(points) < 3:
            return
        screen_points = [screen(point) for point in points]
        outline = tuple(max(0, channel - 55) for channel in colour)
        draw.polygon(screen_points, fill=colour, outline=outline)

    for feature in dataset.roads:
        if not road_is_supported(feature.tags, include_minor=True):
            continue
        points = [screen(projection.to_world(point)) for point in feature.points]
        if len(points) >= 2:
            width = max(1, round(road_width_metres(feature.tags) * scale))
            draw.line(points, fill=(86, 88, 94), width=width, joint="curve")

    counts = {"osm": 0, "overture": 0, "generated": 0}
    for feature in dataset.building_polygons:
        source = "overture" if feature.tags.get("source") == "overturemaps" else "osm"
        colour = BUILDING_SOURCE_REFERENCE_COLOURS[source]
        for polygon in feature.polygons:
            draw_world_polygon([projection.to_world(point) for point in polygon.outer[:-1]], colour)
            counts[source] += 1

    point_colour = BUILDING_SOURCE_REFERENCE_COLOURS["osm"]
    point_half = max(2.0, size / max(1.0, projection.world_size) * 4.0)
    for feature in dataset.building_points:
        if feature.tags.get("source") == "overturemaps":
            continue
        x, y = screen(projection.to_world(feature.point))
        draw.rectangle((x - point_half, y - point_half, x + point_half, y + point_half), fill=point_colour, outline=(19, 92, 164))
        counts["osm"] += 1

    generated_colour = BUILDING_SOURCE_REFERENCE_COLOURS["generated"]
    for plan in building_placement_plans:
        if not plan.synthetic_infill:
            continue
        draw_world_polygon(plan.support_polygon, generated_colour)
        counts["generated"] += 1

    legend_rows = (
        ("OSM", "osm"),
        ("Overture", "overture"),
        ("Generated", "generated"),
    )
    padding = max(8, size // 96)
    swatch = max(10, size // 80)
    row_height = max(16, size // 48)
    legend_width = max(138, size // 5)
    legend_height = padding * 2 + row_height * len(legend_rows)
    draw.rectangle((padding, padding, padding + legend_width, padding + legend_height), fill=(18, 19, 22), outline=(210, 210, 200))
    for index, (label, key) in enumerate(legend_rows):
        y = padding * 2 + index * row_height
        draw.rectangle((padding * 2, y, padding * 2 + swatch, y + swatch), fill=BUILDING_SOURCE_REFERENCE_COLOURS[key])
        draw.text((padding * 2 + swatch + 6, y - 1), f"{label}: {counts[key]}", fill=(238, 238, 230))

    border = max(2, size // 256)
    draw.rectangle((border, border, size - border - 1, size - border - 1), outline=(210, 210, 200), width=border)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return image


def write_world_icon(path: Path, overview: Image.Image, *, size: int = 128) -> Image.Image:
    icon = overview.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    icon = ImageEnhance.Color(icon).enhance(1.15)
    icon = ImageEnhance.Contrast(icon).enhance(1.20)
    draw = ImageDraw.Draw(icon)
    border = max(2, size // 32)
    draw.rectangle((border // 2, border // 2, size - border // 2 - 1, size - border // 2 - 1), outline=(230, 225, 205), width=border)
    write_rgb_dxt1_paa(path, icon)
    return icon
