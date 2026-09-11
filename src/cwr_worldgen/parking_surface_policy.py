# SPDX-License-Identifier: GPL-3.0-or-later
"""Render OSM parking lots as cached world-aligned terrain overlays.

Parking remains terrain, not a floating slab. Each mapped parking polygon is
classified from its nearest supported OSM road: paved roads produce asphalt,
while gravel/dirt roads produce gravel. The overlay is composited over the same
preset grass background used by sports pitches, rendered in world coordinates,
and cached per WRP cell.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import math
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon, box
from shapely.strtree import STRtree

from .cache import cache_key, restore_or_create_file
from .shared_cache_policy import SHARED_CACHE_DIRNAME, shared_cache_root


PARKING_TEXTURE_PREFIX = "pk"
PARKING_TEXTURE_CACHE_DIRNAME = "parking-lot-textures"
PARKING_TEXTURE_CACHE_SCHEMA = 1
RVW4_TEXTURE_LIMIT = 512
_INSTALLED = False
_GENERATED_PARKING_PATHS: ContextVar[tuple[str, ...]] = ContextVar(
    "cwr_generated_parking_paths", default=()
)


@dataclass(frozen=True, slots=True)
class _ParkingGeometry:
    osm_key: str
    outer: tuple[tuple[float, float], ...]
    holes: tuple[tuple[tuple[float, float], ...], ...]
    centre_x: float
    centre_z: float
    half_length: float
    half_width: float
    ux: float
    uz: float
    px: float
    pz: float
    surface: str


def _canonical(value: object) -> str:
    path = str(value or "").replace("/", "\\").strip().lstrip("\\").casefold()
    while "\\\\" in path:
        path = path.replace("\\\\", "\\")
    return path


def _road_surface_catalogue(dataset, projection):
    from .osm import road_is_dirt, road_is_gravel, road_is_supported

    lines: list[LineString] = []
    surfaces: list[str] = []
    for feature in getattr(dataset, "roads", ()):
        tags = getattr(feature, "tags", {}) or {}
        if not road_is_supported(tags, include_minor=True):
            continue
        points = [projection.to_world(point) for point in getattr(feature, "points", ())]
        cleaned: list[tuple[float, float]] = []
        for point in points:
            value = (float(point[0]), float(point[1]))
            if not cleaned or math.dist(cleaned[-1], value) > 0.05:
                cleaned.append(value)
        if len(cleaned) < 2:
            continue
        line = LineString(cleaned)
        if line.is_empty or line.length <= 0.05:
            continue
        lines.append(line)
        surfaces.append("gravel" if (road_is_gravel(tags) or road_is_dirt(tags)) else "paved")
    if not lines:
        return (), (), None
    return tuple(lines), tuple(surfaces), STRtree(lines)


def _nearest_road_surface(shape: Polygon, lines, surfaces, tree) -> str:
    if not lines or tree is None:
        return "paved"
    try:
        nearest = tree.nearest(shape)
        index = int(nearest.item() if hasattr(nearest, "item") else nearest)
        if 0 <= index < len(surfaces):
            return surfaces[index]
    except (TypeError, ValueError, AttributeError):
        pass
    index = min(range(len(lines)), key=lambda item: shape.distance(lines[item]))
    return surfaces[index]


def _parking_geometries(dataset, projection) -> tuple[_ParkingGeometry, ...]:
    lines, surfaces, tree = _road_surface_catalogue(dataset, projection)
    result: list[_ParkingGeometry] = []
    for feature in getattr(dataset, "sites", ()):
        tags = getattr(feature, "tags", {}) or {}
        if str(tags.get("site", "")).casefold() != "parking":
            continue
        for polygon in getattr(feature, "polygons", ()):
            outer = tuple(
                (float(x), float(z))
                for x, z in (projection.to_world(point) for point in polygon.outer[:-1])
            )
            if len(outer) < 3:
                continue
            holes = tuple(
                tuple(
                    (float(x), float(z))
                    for x, z in (projection.to_world(point) for point in ring[:-1])
                )
                for ring in getattr(polygon, "holes", ())
                if len(ring) >= 4
            )
            shape = Polygon(outer, holes)
            if shape.is_empty or not shape.is_valid or shape.area < 20.0:
                continue
            rectangle = shape.minimum_rotated_rectangle
            coordinates = list(rectangle.exterior.coords)[:4]
            if len(coordinates) != 4:
                continue
            edges: list[tuple[float, float, float]] = []
            for index in range(2):
                ax, az = coordinates[index]
                bx, bz = coordinates[index + 1]
                dx, dz = bx - ax, bz - az
                edges.append((math.hypot(dx, dz), dx, dz))
            long_index = 0 if edges[0][0] >= edges[1][0] else 1
            length, dx, dz = edges[long_index]
            width = edges[1 - long_index][0]
            if length < 4.0 or width < 3.0:
                continue
            ux, uz = dx / length, dz / length
            centre = rectangle.centroid
            result.append(_ParkingGeometry(
                osm_key=str(getattr(feature, "osm_key", "")),
                outer=outer,
                holes=holes,
                centre_x=float(centre.x),
                centre_z=float(centre.y),
                half_length=float(length) * 0.5,
                half_width=float(width) * 0.5,
                ux=ux,
                uz=uz,
                px=-uz,
                pz=ux,
                surface=_nearest_road_surface(shape, lines, surfaces, tree),
            ))
    return tuple(result)


def _parking_cells(
    geometries: Sequence[_ParkingGeometry], cells: int, cell_size: float
) -> dict[int, tuple[_ParkingGeometry, ...]]:
    by_cell: dict[int, list[_ParkingGeometry]] = {}
    if cells <= 0 or cell_size <= 0.0:
        return {}
    for geometry in geometries:
        shape = Polygon(geometry.outer, geometry.holes)
        min_x, min_z, max_x, max_z = shape.bounds
        col0 = max(0, int(math.floor(min_x / cell_size)))
        row0 = max(0, int(math.floor(min_z / cell_size)))
        col1 = min(cells - 1, int(math.floor(max(0.0, max_x - 1.0e-9) / cell_size)))
        row1 = min(cells - 1, int(math.floor(max(0.0, max_z - 1.0e-9) / cell_size)))
        for row in range(row0, row1 + 1):
            z0 = row * cell_size
            for col in range(col0, col1 + 1):
                x0 = col * cell_size
                if shape.intersects(box(x0, z0, x0 + cell_size, z0 + cell_size)):
                    by_cell.setdefault(row * cells + col, []).append(geometry)
    return {index: tuple(values) for index, values in by_cell.items()}


def _polygon_mask_for_cell(
    geometry: _ParkingGeometry,
    *,
    cell_index: int,
    cells: int,
    cell_size: float,
    size: int,
) -> np.ndarray:
    row, col = divmod(int(cell_index), cells)
    x0, z0 = col * cell_size, row * cell_size
    scale = size / cell_size

    def pixels(points):
        return [((x - x0) * scale, (z - z0) * scale) for x, z in points]

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(pixels(geometry.outer), fill=255)
    for hole in geometry.holes:
        draw.polygon(pixels(hole), fill=0)
    return np.asarray(mask, dtype=np.uint8) != 0


def _world_noise(world_x: np.ndarray, world_z: np.ndarray, *, salt: int) -> np.ndarray:
    # Deterministic coordinate hash. It is based on world position rather than
    # cell-local pixels so adjacent parking textures share the same material grain.
    x = np.floor(world_x * 2.0).astype(np.uint64)
    z = np.floor(world_z * 2.0).astype(np.uint64)
    value = (
        x * np.uint64(0x9E3779B97F4A7C15)
        ^ z * np.uint64(0xD1B54A32D192ED03)
        ^ np.uint64(salt)
    )
    value ^= value >> np.uint64(30)
    value *= np.uint64(0xBF58476D1CE4E5B9)
    value ^= value >> np.uint64(27)
    value *= np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    return (value & np.uint64(0xFFFF)).astype(np.float32) / 65535.0


def _render_parking_cell(
    *,
    cell_index: int,
    geometries: Sequence[_ParkingGeometry],
    spec,
    base: Image.Image,
) -> Image.Image:
    size = int(base.width)
    cells = int(spec.cells)
    cell_size = float(spec.cell_size)
    if base.height != size or size < 16:
        raise ValueError("parking background must be a square terrain texture")

    image = np.asarray(base.convert("RGB"), dtype=np.float32).copy()
    row, col = divmod(int(cell_index), cells)
    pixel_scale = cell_size / size
    xs = col * cell_size + (np.arange(size, dtype=np.float64) + 0.5) * pixel_scale
    zs = row * cell_size + (np.arange(size, dtype=np.float64) + 0.5) * pixel_scale
    world_x, world_z = np.meshgrid(xs, zs)

    for geometry in geometries:
        mask = _polygon_mask_for_cell(
            geometry,
            cell_index=cell_index,
            cells=cells,
            cell_size=cell_size,
            size=size,
        )
        if not bool(np.any(mask)):
            continue
        if geometry.surface == "gravel":
            noise = _world_noise(world_x, world_z, salt=0x5A17)
            base_colour = np.asarray((104.0, 96.0, 80.0), dtype=np.float32)
            material = base_colour[None, None, :] + (noise[:, :, None] - 0.5) * 28.0
        else:
            noise = _world_noise(world_x, world_z, salt=0xA551)
            base_colour = np.asarray((66.0, 67.0, 66.0), dtype=np.float32)
            material = base_colour[None, None, :] + (noise[:, :, None] - 0.5) * 12.0
        image[mask] = image[mask] * 0.06 + material[mask] * 0.94

        # Paved lots get restrained parking-bay markings. Gravel lots remain
        # unpainted, matching the coarse rural lots they are intended to model.
        if geometry.surface == "paved" and geometry.half_length >= 7.5 and geometry.half_width >= 4.5:
            dx = world_x - geometry.centre_x
            dz = world_z - geometry.centre_z
            along = dx * geometry.ux + dz * geometry.uz
            lateral = dx * geometry.px + dz * geometry.pz
            stall = 2.7
            line_half = max(0.18, pixel_scale * 0.72)
            phase = np.mod(along + geometry.half_length + stall * 0.5, stall) - stall * 0.5
            edge_depth = min(5.2, geometry.half_width * 0.42)
            edge_band = np.abs(lateral) >= geometry.half_width - edge_depth
            bay_lines = mask & edge_band & (np.abs(phase) <= line_half)
            if bool(np.any(bay_lines)):
                white = np.asarray((214.0, 211.0, 194.0), dtype=np.float32)
                image[bay_lines] = image[bay_lines] * 0.12 + white * 0.88

    return Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8), mode="RGB")


def _cache_root(source_dir: Path, spec) -> Path:
    shared = shared_cache_root(getattr(spec, "cache_dir", None))
    if shared is not None:
        return shared / PARKING_TEXTURE_CACHE_DIRNAME
    return source_dir.parent / SHARED_CACHE_DIRNAME / PARKING_TEXTURE_CACHE_DIRNAME


def _geometry_fingerprint(geometry: _ParkingGeometry) -> dict[str, object]:
    return {
        "osm_key": geometry.osm_key,
        "outer": [[round(x, 4), round(z, 4)] for x, z in geometry.outer],
        "holes": [
            [[round(x, 4), round(z, 4)] for x, z in ring]
            for ring in geometry.holes
        ],
        "axis": [round(geometry.ux, 7), round(geometry.uz, 7)],
        "surface": geometry.surface,
    }


def _is_generated_cell(cell_index: int, texture_indices, texture_paths, prefixes: tuple[str, ...]) -> bool:
    slot = int(texture_indices[cell_index])
    if not 0 <= slot < len(texture_paths):
        return False
    filename = Path(str(texture_paths[slot]).replace("\\", "/")).name.casefold()
    return any(filename.startswith(prefix) for prefix in prefixes) and filename.endswith(".paa")


def apply_parking_textures(
    source_dir: Path,
    dataset,
    projection,
    spec,
    texture_indices: Sequence[int],
    texture_paths: Sequence[str],
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    from . import generator
    from . import runway_exact_background_policy as exact
    from . import runway_surface_policy as runway
    from . import sports_pitch_surface_policy as sports

    geometries = _parking_geometries(dataset, projection)
    if not geometries:
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()
    cell_map = _parking_cells(geometries, int(spec.cells), float(spec.cell_size))
    if not cell_map:
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()

    # Priority is runway > sports pitch > parking. Avoid allocating parking
    # textures for cells that a higher-priority surface owns.
    sports_cells = set(
        sports._pitch_cells(
            sports._pitch_geometries(dataset, projection),
            int(spec.cells),
            float(spec.cell_size),
        )
    )
    skipped_runway = 0
    skipped_sports = 0
    for index in tuple(cell_map):
        if _is_generated_cell(index, texture_indices, texture_paths, ("rw",)):
            cell_map.pop(index, None)
            skipped_runway += 1
        elif index in sports_cells or _is_generated_cell(index, texture_indices, texture_paths, ("sp",)):
            cell_map.pop(index, None)
            skipped_sports += 1
    if not cell_map:
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()

    source_dir = Path(source_dir)
    for stale in source_dir.glob(f"{PARKING_TEXTURE_PREFIX}[0-9a-f][0-9a-f][0-9a-f].paa"):
        stale.unlink()
    for stale in source_dir.glob(f".{PARKING_TEXTURE_PREFIX}cell*.paa"):
        stale.unlink()
    report_path = source_dir / "parking-lot-textures.json"
    report_path.unlink(missing_ok=True)

    if len(texture_paths) + len(cell_map) > RVW4_TEXTURE_LIMIT:
        report_path.write_text(json.dumps({
            "schema": 1,
            "mode": "skipped-texture-table-budget",
            "parking_count": len(geometries),
            "parking_cells": len(cell_map),
            "runway_overlap_cells_skipped": skipped_runway,
            "sports_overlap_cells_skipped": skipped_sports,
            "base_texture_entries": len(texture_paths),
            "texture_limit": RVW4_TEXTURE_LIMIT,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()

    grass_path = sports._grass_background_path(generator, spec)
    exact_source = exact._load_exact_texture(source_dir, spec, grass_path)
    if exact_source is not None:
        base = exact_source.top_image.copy()
        background_sha = sha256(exact_source.data).hexdigest()
        background_source = exact_source.source
    else:
        size = int(getattr(runway, "RUNWAY_TEXTURE_SIZE", 128))
        base = sports._fallback_background(generator, runway, spec, grass_path, size)
        background_sha = sha256(base.tobytes()).hexdigest()
        background_source = "generated-fallback"

    cache_dir = _cache_root(source_dir, spec)
    cache_enabled = bool(getattr(spec, "cache_enabled", True))
    cache_refresh = bool(getattr(spec, "cache_refresh", False))
    revised_indices = [int(value) for value in texture_indices]
    revised_paths = [str(value) for value in texture_paths]
    generated_paths: list[str] = []
    reusable: dict[bytes, tuple[bytes, int]] = {}
    cache_hits = cache_misses = reused_cells = 0

    for cell_index in sorted(cell_map):
        geometries_here = cell_map[cell_index]
        payload = {
            "schema": PARKING_TEXTURE_CACHE_SCHEMA,
            "renderer": "parking-world-aligned-v1",
            "profile": str(getattr(spec, "ground_texture_profile", "generated")),
            "background_path": _canonical(grass_path),
            "background_sha256": background_sha,
            "cells": int(spec.cells),
            "cell_size": float(spec.cell_size),
            "cell_index": int(cell_index),
            "parking": [_geometry_fingerprint(value) for value in geometries_here],
        }
        cached = cache_dir / f"{cache_key('parking-cell-v1', payload)}.paa"
        candidate = source_dir / f".{PARKING_TEXTURE_PREFIX}cell{cell_index:06x}.paa"

        def paint(destination: Path, *, index=cell_index, values=geometries_here):
            image = _render_parking_cell(
                cell_index=index,
                geometries=values,
                spec=spec,
                base=base,
            )
            if exact_source is not None and image.size == exact_source.top_image.size:
                exact._write_with_preserved_background(destination, image, exact_source)
            else:
                runway.write_rgb_dxt1_paa(destination, image)

        hit = restore_or_create_file(
            cache_path=cached,
            destination=candidate,
            producer=paint,
            enabled=cache_enabled,
            refresh=cache_refresh,
        )
        cache_hits += int(hit)
        cache_misses += int(not hit)
        data = candidate.read_bytes()
        digest = sha256(data).digest()
        prior = reusable.get(digest)
        if prior is not None and prior[0] == data:
            revised_indices[cell_index] = prior[1]
            reused_cells += 1
            candidate.unlink(missing_ok=True)
            continue

        slot = len(revised_paths)
        if slot >= RVW4_TEXTURE_LIMIT:
            candidate.unlink(missing_ok=True)
            raise ValueError("parking textures exceed the RVW4 texture table")
        filename = f"{PARKING_TEXTURE_PREFIX}{len(generated_paths):03x}.paa"
        candidate.replace(source_dir / filename)
        wire_path = rf"{spec.name}\{filename}"
        revised_paths.append(wire_path)
        revised_indices[cell_index] = slot
        generated_paths.append(wire_path)
        reusable[digest] = (data, slot)

    paved = sum(value.surface == "paved" for value in geometries)
    gravel = sum(value.surface == "gravel" for value in geometries)
    report_path.write_text(json.dumps({
        "schema": 1,
        "mode": "generated-terrain-textures",
        "strategy": "nearest-road-surface-world-aligned-overlay",
        "parking_count": len(geometries),
        "paved_parking_count": paved,
        "gravel_parking_count": gravel,
        "parking_cells": len(cell_map),
        "runway_overlap_cells_skipped": skipped_runway,
        "sports_overlap_cells_skipped": skipped_sports,
        "background_path": grass_path,
        "background_source": background_source,
        "generated_parking_textures": len(generated_paths),
        "reused_parking_cell_assignments": reused_cells,
        "cache_directory": str(cache_dir),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "base_texture_entries": len(texture_paths),
        "final_texture_entries": len(revised_paths),
        "texture_limit": RVW4_TEXTURE_LIMIT,
        "texture_paths": generated_paths,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return tuple(revised_indices), tuple(revised_paths), tuple(generated_paths)


def install_parking_surface_policy() -> None:
    """Install parking preparation beneath the runway write/validation hooks."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import generator
    from . import runway_surface_policy as runway
    from .progress import report_progress

    original_writer = runway._ORIGINAL_WRITE_RVW4
    if not callable(original_writer):
        return

    def write_rvw4_with_parking(
        path, width, height, elevations, texture_indices, texture_paths, objects, **kwargs
    ):
        context = runway._RUNWAY_CONTEXT.get()
        if (
            context is None
            or context.world_name.casefold() != Path(path).stem.casefold()
            or context.cells != int(width)
            or context.cells != int(height)
        ):
            _GENERATED_PARKING_PATHS.set(())
            return original_writer(
                path, width, height, elevations, texture_indices, texture_paths, objects, **kwargs
            )

        geometries = _parking_geometries(context.dataset, context.projection)
        if not geometries:
            _GENERATED_PARKING_PATHS.set(())
            return original_writer(
                path, width, height, elevations, texture_indices, texture_paths, objects, **kwargs
            )

        started = perf_counter()
        report_progress(86, f"Preparing parking lot terrain textures ({len(geometries)} lots)")
        revised_indices, revised_paths, generated = apply_parking_textures(
            Path(path).parent,
            context.dataset,
            context.projection,
            context.spec,
            texture_indices,
            texture_paths,
        )
        _GENERATED_PARKING_PATHS.set(generated)
        elapsed = perf_counter() - started
        report_path = Path(path).parent / "parking-lot-textures.json"
        detail = ""
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                detail = (
                    f"{int(report.get('parking_cells', 0))} cells; "
                    f"cache {int(report.get('cache_hits', 0))} hits/"
                    f"{int(report.get('cache_misses', 0))} rendered; "
                    f"{int(report.get('generated_parking_textures', 0))} unique PAAs"
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        report_progress(
            86,
            "Parking lot terrain textures ready"
            + (f" ({detail}; {elapsed:.2f}s)" if detail else f" ({elapsed:.2f}s)"),
        )
        return original_writer(
            path, width, height, elevations, revised_indices, revised_paths, objects, **kwargs
        )

    # Installed after sports. The outer runway wrapper delegates to parking,
    # parking delegates to sports, and sports delegates to the fast WRP writer.
    # Parking itself skips all sports-owned cells, giving runway > sports > parking.
    runway._ORIGINAL_WRITE_RVW4 = write_rvw4_with_parking

    original_validate = runway._ORIGINAL_VALIDATE_MILESTONE4

    def validate_with_parking(*args, **kwargs):
        generated = _GENERATED_PARKING_PATHS.get()
        if not generated:
            return original_validate(*args, **kwargs)
        current_paths = generator._ground_texture_paths

        def validation_paths(spec):
            return (*tuple(current_paths(spec)), *generated)

        generator._ground_texture_paths = validation_paths
        try:
            return original_validate(*args, **kwargs)
        finally:
            generator._ground_texture_paths = current_paths

    runway._ORIGINAL_VALIDATE_MILESTONE4 = validate_with_parking
    _INSTALLED = True
