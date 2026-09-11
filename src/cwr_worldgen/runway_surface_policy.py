# SPDX-License-Identifier: GPL-3.0-or-later
"""Render OSM runways as generated, correctly oriented WRP terrain textures.

RVW4 terrain cells carry a texture-table index but no per-cell UV transform. A
single stock ``o\\runtr_d.paa`` therefore cannot follow an arbitrary OSM bearing.
Generate compact world-local PAA slices in world coordinates instead. Adjacent
cells form one continuous runway at the mapped bearing and the runway remains
visible in the editor terrain view.

A generated slice replaces the whole 50 m terrain cell, so its background must
also resemble the stock terrain it replaces. Nogova therefore uses path-aware
approximations of its ``o\\t1`` / ``o\\trava*`` / field palette rather than the
brighter generic semantic colours. Identical rendered slices are hashed and
reused, allowing many cells to share one WRP texture slot.

The WRP table has 512 slots. Generated cell textures are preferred when the
worst-case cell count fits; otherwise the previous rotatable P3D runway remains a
bounded fallback.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import json
import math
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .paa import write_rgb_dxt1_paa
from .runway_model_policy import (
    DESERT_RUNWAY_END_TEXTURE,
    DESERT_RUNWAY_MIDDLE_TEXTURE,
    DESERT_RUNWAY_START_TEXTURE,
    GRASS_RUNWAY_END_TEXTURE,
    GRASS_RUNWAY_MIDDLE_TEXTURE,
    GRASS_RUNWAY_START_TEXTURE,
    install_runway_model_policy,
    runway_family,
    runway_model_path,
    runway_texture_triplet,
)


GRASS_RUNWAY_TEXTURE = GRASS_RUNWAY_MIDDLE_TEXTURE
DESERT_RUNWAY_TEXTURE = DESERT_RUNWAY_MIDDLE_TEXTURE
RVW4_TEXTURE_LIMIT = 512
RUNWAY_TEXTURE_SIZE = 128
RUNWAY_TEXTURE_PREFIX = "rw"
RUNWAY_SURFACE_OFFSET_METRES = 0.060
_SURFACE_CACHE_V11 = "surface-pipeline-v11-vectorized-material-pass"
_SURFACE_CACHE_V18 = "surface-pipeline-v18-nogova-runway-blend-dedup"
_INSTALLED = False
_ORIGINAL_FIT_ROAD_OBJECTS = None
_ORIGINAL_CACHE_KEY = None
_ORIGINAL_AEROWAY_MASK = None
_ORIGINAL_WRITE_RVW4 = None
_ORIGINAL_VALIDATE_MILESTONE4 = None
_ORIGINAL_GROUND_TEXTURE_PATHS = None

# Approximate average colours of the stock Nogova terrain families. These are
# intentionally darker and less saturated than the semantic overview palette.
# We cannot composite BI's PAA bytes unless the user supplies/extracts them, but
# matching the actual selected stock path removes the glaring rectangular colour
# jump visible around generated runway cells.
_NOGOVA_STOCK_COLOURS: dict[str, tuple[int, int, int]] = {
    r"o\t1.paa": (58, 66, 45),
    r"o\trava2.paa": (77, 88, 54),
    r"o\trava3.paa": (96, 105, 65),
    r"o\pole1.paa": (151, 143, 91),
    r"o\pole2.paa": (126, 121, 72),
    r"o\ps.paa": (177, 161, 110),
    r"o\l1.paa": (100, 98, 90),
    r"o\lom2.paa": (81, 79, 73),
}


@dataclass(frozen=True, slots=True)
class _RunwayGeometry:
    osm_key: str
    start_x: float
    start_z: float
    end_x: float
    end_z: float
    length: float
    half_width: float
    ux: float
    uz: float
    px: float
    pz: float


@dataclass(frozen=True, slots=True)
class _RunwayBuildContext:
    world_name: str
    cells: int
    dataset: object
    projection: object
    spec: object
    cell_indices: tuple[int, ...]
    mode: str
    base_texture_entries: int


_RUNWAY_CONTEXT: ContextVar[_RunwayBuildContext | None] = ContextVar(
    "cwr_runway_texture_context", default=None
)
_GENERATED_RUNWAY_PATHS: ContextVar[tuple[str, ...]] = ContextVar(
    "cwr_generated_runway_paths", default=()
)


def runway_texture_for_profile(profile: object) -> str:
    """Return the historical stock middle texture for compatibility."""
    return runway_texture_triplet(profile)[1]


def _tagged_width_metres(feature, default: float = 36.0) -> float:
    raw = feature.tags.get("width", "")
    try:
        value = float(str(raw or 0).replace(",", "."))
    except (TypeError, ValueError):
        value = 0.0
    if not math.isfinite(value) or value <= 0.0:
        value = default
    return max(3.0, value)


def _clean_projected_points(feature, projection) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for point in feature.points:
        x, z = projection.to_world(point)
        value = (float(x), float(z))
        if not points or math.dist(points[-1], value) > 0.05:
            points.append(value)
    return tuple(points)


def _canonical_runway_points(
    points: Sequence[tuple[float, float]], profile: object
) -> tuple[tuple[float, float], ...]:
    """Orient west->east for grass and south->north for Desert end roles."""
    values = tuple(points)
    if len(values) < 2:
        return values
    first, last = values[0], values[-1]
    if runway_family(profile) == "desert":
        reverse = (last[1], last[0]) < (first[1], first[0])
    else:
        reverse = (last[0], last[1]) < (first[0], first[1])
    return tuple(reversed(values)) if reverse else values


def _canonical_runway_endpoints(
    points: Sequence[tuple[float, float]], profile: object
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    values = _canonical_runway_points(points, profile)
    if len(values) < 2:
        return None
    return values[0], values[-1]


def _runway_geometries(dataset, projection, profile: object) -> tuple[_RunwayGeometry, ...]:
    geometries: list[_RunwayGeometry] = []
    for feature in dataset.aeroway_lines:
        if str(feature.tags.get("aeroway", "")).strip().casefold() != "runway":
            continue
        points = _canonical_runway_points(
            _clean_projected_points(feature, projection), profile
        )
        if len(points) < 2:
            continue
        start, end = points[0], points[-1]
        dx, dz = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dz)
        if length < 1.0:
            continue
        ux, uz = dx / length, dz / length
        geometries.append(_RunwayGeometry(
            osm_key=str(getattr(feature, "osm_key", "")),
            start_x=start[0], start_z=start[1], end_x=end[0], end_z=end[1],
            length=length, half_width=_tagged_width_metres(feature) * 0.5,
            ux=ux, uz=uz, px=-uz, pz=ux,
        ))
    return tuple(geometries)


def runway_texture_cell_indices(dataset, projection, spec) -> tuple[int, ...]:
    """Return every WRP cell touched by a line runway at its mapped width."""
    from . import surface_pass as surface

    cells = int(spec.cells)
    if cells <= 0:
        return ()
    resolution = cells * 8
    image = Image.new("L", (resolution, resolution), 0)
    draw = ImageDraw.Draw(image)
    any_runway = False
    for feature in dataset.aeroway_lines:
        if str(feature.tags.get("aeroway", "")).strip().casefold() != "runway":
            continue
        points = [projection.to_pixel(point, resolution) for point in feature.points]
        if len(points) < 2:
            continue
        width_pixels = max(1, int(round(
            _tagged_width_metres(feature) / float(projection.world_size) * resolution
        )))
        draw.line(points, fill=255, width=width_pixels, joint="curve")
        any_runway = True
    if not any_runway:
        return ()
    mask = surface._image_to_wrp_mask(image, cells, threshold=1)
    return tuple(int(value) for value in np.flatnonzero(mask))


def _runway_texture_budget(spec, runway_cells: int) -> tuple[bool, int, int]:
    """Return the conservative worst-case texture budget before deduplication."""
    from . import generator

    paths = (
        _ORIGINAL_GROUND_TEXTURE_PATHS(spec)
        if _ORIGINAL_GROUND_TEXTURE_PATHS is not None
        else generator._ground_texture_paths(spec)
    )
    base_entries = 1 + len(paths)
    final_entries = base_entries + max(0, int(runway_cells))
    return final_entries <= RVW4_TEXTURE_LIMIT, base_entries, final_entries


def _canonical_texture_path(value: object) -> str:
    path = str(value or "").replace("/", "\\").strip().lstrip("\\").casefold()
    while "\\\\" in path:
        path = path.replace("\\\\", "\\")
    return path


def _ground_path_for_material(spec, material_index: int) -> str:
    """Return the actual stock/local WRP path replaced by a generated slice."""
    from . import generator

    paths = (
        tuple(_ORIGINAL_GROUND_TEXTURE_PATHS(spec))
        if _ORIGINAL_GROUND_TEXTURE_PATHS is not None
        else tuple(generator._ground_texture_paths(spec))
    )
    if 0 <= material_index < len(paths):
        return paths[material_index]
    return ""


def _profile_surface_colour(
    material, profile: object, *, ground_path: str = ""
) -> tuple[int, int, int]:
    from . import surface_pass as surface

    name = str(profile or "").strip().casefold()
    code = str(getattr(material, "code", "g"))
    if name == "nogova":
        stock = _NOGOVA_STOCK_COLOURS.get(_canonical_texture_path(ground_path))
        if stock is not None:
            return stock
    if name == "desert":
        return tuple(surface.DESERT_SURFACE_COLOURS.get(code, material.colour))
    if name == "malden":
        return tuple(surface.MALDEN_SURFACE_COLOURS.get(code, material.colour))
    return tuple(material.colour)


def _background_texture(
    material,
    profile: object,
    seed: str,
    size: int,
    *,
    ground_path: str = "",
) -> Image.Image:
    """Generate a local background tuned to the actual replaced terrain path."""
    from . import surface_pass as surface

    canonical = _canonical_texture_path(ground_path)
    colour = _profile_surface_colour(material, profile, ground_path=canonical)
    proxy = surface.SurfaceMaterialDefinition(
        str(getattr(material, "code", "g")),
        str(getattr(material, "name", "ground")),
        colour,
        None,
    )
    # Include the selected stock path in the seed so t1/trava2/trava3 retain
    # distinct deterministic grain even when semantic material codes coincide.
    return surface.create_surface_texture(proxy, f"{seed}:{canonical}", size)


def _runway_palette(profile: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return runway deck, wear and marking colours.

    The former grass-family palette was literally green. ``runtr_d`` is runway
    artwork, not an instruction to camouflage the landing strip as a lawn.
    """
    if runway_family(profile) == "desert":
        return (
            np.asarray((121, 116, 100), dtype=np.float32),
            np.asarray((91, 88, 78), dtype=np.float32),
            np.asarray((226, 219, 194), dtype=np.float32),
        )
    return (
        np.asarray((108, 110, 104), dtype=np.float32),
        np.asarray((78, 80, 76), dtype=np.float32),
        np.asarray((224, 221, 207), dtype=np.float32),
    )


def _render_runway_cell(
    *,
    cell_index: int,
    original_wrp_texture_index: int,
    geometries: Sequence[_RunwayGeometry],
    materials: Sequence[object],
    spec,
    size: int = RUNWAY_TEXTURE_SIZE,
    background_cache: dict[tuple[int, str], Image.Image] | None = None,
) -> Image.Image:
    """Render one globally aligned runway slice into one terrain-cell texture."""
    cells, cell_size = int(spec.cells), float(spec.cell_size)
    if cells <= 0 or cell_size <= 0.0:
        raise ValueError("runway texture generation requires a positive terrain grid")
    if size < 16 or size & (size - 1):
        raise ValueError("runway texture size must be a power of two of at least 16")

    material_index = int(original_wrp_texture_index) - 1
    if not 0 <= material_index < len(materials):
        material_index = next((
            index for index, material in enumerate(materials)
            if str(getattr(material, "code", "")).casefold() == "g"
        ), 0)

    ground_path = _ground_path_for_material(spec, material_index)
    cache = background_cache if background_cache is not None else {}
    cache_key = (material_index, _canonical_texture_path(ground_path))
    base = cache.get(cache_key)
    if base is None:
        base = _background_texture(
            materials[material_index],
            getattr(spec, "ground_texture_profile", "generated"),
            str(getattr(spec, "deterministic_seed", "cwr-worldgen")),
            size,
            ground_path=ground_path,
        )
        cache[cache_key] = base

    image = np.asarray(base, dtype=np.float32).copy()
    cz, cx = divmod(int(cell_index), cells)
    pixel_scale = cell_size / size
    xs = cx * cell_size + (np.arange(size, dtype=np.float64) + 0.5) * pixel_scale
    zs = cz * cell_size + (np.arange(size, dtype=np.float64) + 0.5) * pixel_scale
    world_x, world_z = np.meshgrid(xs, zs)

    body_colour, wear_colour, marking_colour = _runway_palette(
        getattr(spec, "ground_texture_profile", "generated")
    )
    marking_half_width = max(0.55, pixel_scale * 1.1)
    threshold_half_depth = max(0.85, pixel_scale * 1.5)

    for geometry in geometries:
        rel_x, rel_z = world_x - geometry.start_x, world_z - geometry.start_z
        along = rel_x * geometry.ux + rel_z * geometry.uz
        lateral = rel_x * geometry.px + rel_z * geometry.pz
        inside = (
            (along >= 0.0) & (along <= geometry.length)
            & (np.abs(lateral) <= geometry.half_width)
        )
        if not bool(np.any(inside)):
            continue

        # Runway deck. Keep a little source grain so it does not become a flat
        # vector rectangle beside legacy low-frequency stock terrain.
        image[inside] = image[inside] * 0.12 + body_colour * 0.88
        track_offset = min(geometry.half_width * 0.34, 6.0)
        tracks = inside & (
            np.abs(np.abs(lateral) - track_offset) <= max(0.75, pixel_scale * 1.5)
        )
        image[tracks] = image[tracks] * 0.42 + wear_colour * 0.58

        centreline = inside & (np.abs(lateral) <= marking_half_width)
        image[centreline] = marking_colour

        # Procedural equivalents of the stock z/k end-cap intent.
        threshold_span = max(2.0, geometry.half_width * 0.72)
        start_at = min(4.0, geometry.length * 0.12)
        end_at = max(geometry.length - 4.0, geometry.length * 0.88)
        thresholds = inside & (
            ((np.abs(along - start_at) <= threshold_half_depth)
             | (np.abs(along - end_at) <= threshold_half_depth))
            & (np.abs(lateral) <= threshold_span)
        )
        image[thresholds] = marking_colour

        edge = inside & (
            np.abs(np.abs(lateral) - geometry.half_width) <= max(0.45, pixel_scale)
        )
        image[edge] = image[edge] * 0.28 + wear_colour * 0.72

    return Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8), mode="RGB")


def apply_generated_runway_texture_table(
    source_dir: Path,
    dataset,
    projection,
    spec,
    texture_indices: Sequence[int],
    texture_paths: Sequence[str],
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    """Render runway cells, deduplicate identical images, and extend the WRP table."""
    from . import generator

    cell_indices = runway_texture_cell_indices(dataset, projection, spec)
    if not cell_indices:
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()
    if len(texture_paths) + len(cell_indices) > RVW4_TEXTURE_LIMIT:
        # The install wrapper chooses the P3D path before this point. Keep this
        # defensive check for direct callers and tests.
        raise ValueError("generated runway textures exceed the RVW4 512-entry texture table")
    if len(texture_indices) != int(spec.cells) * int(spec.cells):
        raise ValueError("runway texture generation received a mismatched WRP grid")

    geometries = _runway_geometries(
        dataset, projection, getattr(spec, "ground_texture_profile", "generated")
    )
    if not geometries:
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()

    source_dir = Path(source_dir)
    for stale in source_dir.glob(f"{RUNWAY_TEXTURE_PREFIX}[0-9a-f][0-9a-f][0-9a-f].paa"):
        stale.unlink()
    report_path = source_dir / "runway-textures.json"
    if report_path.exists():
        report_path.unlink()

    materials = tuple(generator._material_definitions(spec))
    revised_indices = [int(value) for value in texture_indices]
    revised_paths = [str(value) for value in texture_paths]
    generated_paths: list[str] = []
    background_cache: dict[tuple[int, str], Image.Image] = {}
    # digest -> (raw RGB bytes, WRP slot). Keep bytes to make even the theoretical
    # SHA collision harmless rather than letting mathematics ruin an airport.
    reusable: dict[bytes, tuple[bytes, int]] = {}
    reused_cells = 0

    for cell_index in cell_indices:
        image = _render_runway_cell(
            cell_index=cell_index,
            original_wrp_texture_index=revised_indices[cell_index],
            geometries=geometries,
            materials=materials,
            spec=spec,
            size=RUNWAY_TEXTURE_SIZE,
            background_cache=background_cache,
        )
        raw = image.tobytes()
        digest = sha256(raw).digest()
        prior = reusable.get(digest)
        if prior is not None and prior[0] == raw:
            revised_indices[cell_index] = prior[1]
            reused_cells += 1
            continue

        slot = len(revised_paths)
        if slot >= RVW4_TEXTURE_LIMIT:
            raise ValueError("deduplicated runway textures exceed the RVW4 texture table")
        serial = len(generated_paths)
        filename = f"{RUNWAY_TEXTURE_PREFIX}{serial:03x}.paa"
        wire_path = rf"{spec.name}\{filename}"
        write_rgb_dxt1_paa(source_dir / filename, image)
        revised_paths.append(wire_path)
        revised_indices[cell_index] = slot
        generated_paths.append(wire_path)
        reusable[digest] = (raw, slot)

    report_path.write_text(json.dumps({
        "schema": 2,
        "mode": "generated-terrain-textures",
        "texture_limit": RVW4_TEXTURE_LIMIT,
        "texture_size": RUNWAY_TEXTURE_SIZE,
        "base_texture_entries": len(texture_paths),
        "runway_cells": len(cell_indices),
        "generated_runway_textures": len(generated_paths),
        "reused_runway_cell_assignments": reused_cells,
        "reuse_ratio": (reused_cells / len(cell_indices)) if cell_indices else 0.0,
        "final_texture_entries": len(revised_paths),
        "runway_cell_indices": list(cell_indices),
        "texture_paths": generated_paths,
        "stock_reference_family": {
            "grass": list(runway_texture_triplet("grass")),
            "desert": list(runway_texture_triplet("desert")),
        },
        "nogova_background_mode": "selected-stock-path-colour-match",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return tuple(revised_indices), tuple(revised_paths), tuple(generated_paths)


def _runway_tile_plan(points: Sequence[tuple[float, float]], profile: object):
    """Keep the previous 50 m P3D planner solely for table-overflow fallback."""
    from . import playability as _playability
    from .runway_model_policy import RUNWAY_TEXTURE_TILE_METRES

    canonical = _canonical_runway_points(points, profile)
    if len(canonical) < 2:
        return ()
    measure = _playability._PolylineMeasure.create(canonical)
    if measure.total < 1.0:
        return ()
    tile = float(RUNWAY_TEXTURE_TILE_METRES)
    tile_count = max(1, int(math.ceil(measure.total / tile - 1.0e-9)))
    first_centre = (measure.total - tile_count * tile) * 0.5 + tile * 0.5
    half = tile * 0.5
    planned = []
    for index in range(tile_count):
        centre_x, centre_z, heading = measure.point(first_centre + index * tile)
        angle = math.radians(heading)
        dx, dz = math.sin(angle) * half, math.cos(angle) * half
        role = (
            "d" if tile_count == 1 else
            "z" if index == 0 else
            "k" if index == tile_count - 1 else "d"
        )
        planned.append((
            role,
            (centre_x - dx, centre_z - dz),
            (centre_x + dx, centre_z + dz),
        ))
    return tuple(planned)


def runway_overlay_objects(
    dataset, projection, elevations: Sequence[float], spec, *, starting_id: int = 1
):
    """Create rotatable P3D tiles only when generated textures cannot fit."""
    from . import playability as _playability

    objects, next_id = [], int(starting_id)
    profile = getattr(spec, "ground_texture_profile", "generated")
    world_size = float(getattr(spec, "world_size", getattr(projection, "world_size", 0.0)))
    for feature in dataset.aeroway_lines:
        if str(feature.tags.get("aeroway", "")).strip().casefold() != "runway":
            continue
        for role, start, end in _runway_tile_plan(
            _clean_projected_points(feature, projection), profile
        ):
            centre_x = (start[0] + end[0]) * 0.5
            centre_z = (start[1] + end[1]) * 0.5
            if world_size > 0.0 and not (
                0.0 <= centre_x < world_size and 0.0 <= centre_z < world_size
            ):
                continue
            objects.append(_playability._road_object_on_slope(
                next_id,
                runway_model_path(spec.name, profile, role),
                start,
                end,
                elevations,
                spec,
                vertical_offset=RUNWAY_SURFACE_OFFSET_METRES,
            ))
            next_id += 1
    return tuple(objects)


def _validation_with_generated_runway_textures(*args, **kwargs):
    """Temporarily extend validation's expected terrain paths with runway PAAs."""
    from . import generator

    generated_paths = _GENERATED_RUNWAY_PATHS.get()
    if not generated_paths:
        return _ORIGINAL_VALIDATE_MILESTONE4(*args, **kwargs)

    current_ground_paths = generator._ground_texture_paths

    def validation_ground_paths(spec):
        base = tuple(_ORIGINAL_GROUND_TEXTURE_PATHS(spec))
        return (*base, *generated_paths)

    generator._ground_texture_paths = validation_ground_paths
    try:
        lines = _ORIGINAL_VALIDATE_MILESTONE4(*args, **kwargs)
    finally:
        generator._ground_texture_paths = current_ground_paths

    revised = list(lines)
    for index, line in enumerate(revised):
        if "WRP terrain texture profile" in line:
            context = _RUNWAY_CONTEXT.get()
            final_count = (
                context.base_texture_entries + len(generated_paths)
                if context is not None else len(generated_paths)
            )
            revised[index] = (
                "[PASS] WRP terrain texture profile: "
                f"unique generated runway textures={len(generated_paths)}, "
                f"texture table={final_count}/{RVW4_TEXTURE_LIMIT}"
            )
            break
    return revised


def install_runway_surface_policy() -> None:
    """Install generated runway terrain textures with P3D overflow fallback."""
    global _INSTALLED, _ORIGINAL_FIT_ROAD_OBJECTS, _ORIGINAL_CACHE_KEY
    global _ORIGINAL_AEROWAY_MASK, _ORIGINAL_WRITE_RVW4
    global _ORIGINAL_VALIDATE_MILESTONE4, _ORIGINAL_GROUND_TEXTURE_PATHS
    if _INSTALLED:
        return

    from . import generator
    from . import playability
    from . import surface_pass as surface

    install_runway_model_policy()
    _ORIGINAL_FIT_ROAD_OBJECTS = generator.fit_road_objects
    _ORIGINAL_CACHE_KEY = generator.cache_key
    _ORIGINAL_AEROWAY_MASK = surface._aeroway_mask
    _ORIGINAL_WRITE_RVW4 = generator.write_rvw4
    _ORIGINAL_VALIDATE_MILESTONE4 = generator._validate_milestone4
    _ORIGINAL_GROUND_TEXTURE_PATHS = generator._ground_texture_paths

    def aeroway_mask_without_line_runways(dataset, projection, cells):
        # Generated runway cells contain their own local background. Start from
        # natural terrain instead of also painting the old rectangular pavement.
        lines = tuple(
            feature for feature in dataset.aeroway_lines
            if str(feature.tags.get("aeroway", "")).strip().casefold() != "runway"
        )
        if len(lines) == len(dataset.aeroway_lines):
            return _ORIGINAL_AEROWAY_MASK(dataset, projection, cells)
        return _ORIGINAL_AEROWAY_MASK(
            replace(dataset, aeroway_lines=lines), projection, cells
        )

    def fit_road_objects_with_runways(
        dataset, projection, elevations, spec, *, starting_id: int = 1,
        progress_callback=None,
    ):
        report = _ORIGINAL_FIT_ROAD_OBJECTS(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )
        cell_indices = runway_texture_cell_indices(dataset, projection, spec)
        fits, base_entries, _final_entries = _runway_texture_budget(
            spec, len(cell_indices)
        )
        mode = (
            "textures" if cell_indices and fits else
            "p3d-fallback" if cell_indices else "none"
        )
        _RUNWAY_CONTEXT.set(_RunwayBuildContext(
            world_name=str(spec.name),
            cells=int(spec.cells),
            dataset=dataset,
            projection=projection,
            spec=spec,
            cell_indices=cell_indices,
            mode=mode,
            base_texture_entries=base_entries,
        ))
        _GENERATED_RUNWAY_PATHS.set(())
        if mode != "p3d-fallback":
            return report
        next_id = max(
            (int(obj.object_id) for obj in report.objects),
            default=int(starting_id) - 1,
        ) + 1
        runways = runway_overlay_objects(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=next_id,
        )
        return report if not runways else replace(
            report, objects=tuple((*report.objects, *runways))
        )

    def write_rvw4_with_runway_textures(
        path, width, height, elevations, texture_indices, texture_paths, objects, **kwargs
    ):
        context = _RUNWAY_CONTEXT.get()
        if (
            context is not None
            and context.mode == "textures"
            and context.world_name.casefold() == Path(path).stem.casefold()
            and context.cells == int(width) == int(height)
        ):
            revised_indices, revised_paths, generated_paths = (
                apply_generated_runway_texture_table(
                    Path(path).parent,
                    context.dataset,
                    context.projection,
                    context.spec,
                    texture_indices,
                    texture_paths,
                )
            )
            _GENERATED_RUNWAY_PATHS.set(generated_paths)
            infrastructure = Path(path).parent / "i"
            if infrastructure.is_dir():
                for stale in infrastructure.glob("runway_*.p3d"):
                    stale.unlink()
            return _ORIGINAL_WRITE_RVW4(
                path,
                width,
                height,
                elevations,
                revised_indices,
                revised_paths,
                objects,
                **kwargs,
            )
        _GENERATED_RUNWAY_PATHS.set(())
        return _ORIGINAL_WRITE_RVW4(
            path,
            width,
            height,
            elevations,
            texture_indices,
            texture_paths,
            objects,
            **kwargs,
        )

    def runway_cache_key(namespace: str, payload):
        if namespace == _SURFACE_CACHE_V11:
            namespace = _SURFACE_CACHE_V18
        return _ORIGINAL_CACHE_KEY(namespace, payload)

    surface._aeroway_mask = aeroway_mask_without_line_runways
    generator.fit_road_objects = fit_road_objects_with_runways
    playability.fit_road_objects = fit_road_objects_with_runways
    generator.write_rvw4 = write_rvw4_with_runway_textures
    generator._validate_milestone4 = _validation_with_generated_runway_textures
    generator.cache_key = runway_cache_key
    _INSTALLED = True
