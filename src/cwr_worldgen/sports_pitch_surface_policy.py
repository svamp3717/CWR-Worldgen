# SPDX-License-Identifier: GPL-3.0-or-later
"""Render OSM sports pitches into world-aligned terrain textures.

Sports pitches are terrain semantics, not floating slab objects.  This policy
uses the preset's ordinary grass texture as the exact background and paints
field markings only where an OSM pitch crosses a WRP terrain cell.  Like the
runway implementation, every cell is rendered in world coordinates and cached
persistently, so rotated pitches stay aligned and unchanged stock DXT1 blocks
remain byte-identical where possible.
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
from PIL import Image
from shapely.geometry import Polygon, box

from .cache import cache_key, restore_or_create_file
from .shared_cache_policy import SHARED_CACHE_DIRNAME, shared_cache_root


SPORTS_TEXTURE_PREFIX = "sp"
SPORTS_TEXTURE_CACHE_DIRNAME = "sports-pitch-textures"
SPORTS_TEXTURE_CACHE_SCHEMA = 1
RVW4_TEXTURE_LIMIT = 512
_INSTALLED = False
_GENERATED_SPORTS_PATHS: ContextVar[tuple[str, ...]] = ContextVar(
    "cwr_generated_sports_pitch_paths", default=()
)


@dataclass(frozen=True, slots=True)
class _PitchGeometry:
    osm_key: str
    centre_x: float
    centre_z: float
    half_length: float
    half_width: float
    ux: float
    uz: float
    px: float
    pz: float
    corners: tuple[tuple[float, float], ...]
    soccer: bool


def _canonical(value: object) -> str:
    path = str(value or "").replace("/", "\\").strip().lstrip("\\").casefold()
    while "\\\\" in path:
        path = path.replace("\\\\", "\\")
    return path


def _pitch_geometries(dataset, projection) -> tuple[_PitchGeometry, ...]:
    result: list[_PitchGeometry] = []
    for feature in getattr(dataset, "sites", ()):
        tags = getattr(feature, "tags", {}) or {}
        if str(tags.get("site", "")).casefold() != "sports_pitch":
            continue
        sport = str(tags.get("sport", "")).strip().casefold()
        soccer = sport in {"", "soccer", "football", "association_football"}
        for polygon in getattr(feature, "polygons", ()):
            points = [projection.to_world(point) for point in polygon.outer[:-1]]
            if len(points) < 3:
                continue
            shape = Polygon(points)
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
            if length < 6.0 or width < 4.0:
                continue
            ux, uz = dx / length, dz / length
            centre = rectangle.centroid
            result.append(_PitchGeometry(
                osm_key=str(getattr(feature, "osm_key", "")),
                centre_x=float(centre.x),
                centre_z=float(centre.y),
                half_length=float(length) * 0.5,
                half_width=float(width) * 0.5,
                ux=ux,
                uz=uz,
                px=-uz,
                pz=ux,
                corners=tuple((float(x), float(z)) for x, z in coordinates),
                soccer=soccer,
            ))
    return tuple(result)


def _pitch_cells(
    geometries: Sequence[_PitchGeometry], cells: int, cell_size: float
) -> dict[int, tuple[_PitchGeometry, ...]]:
    by_cell: dict[int, list[_PitchGeometry]] = {}
    if cells <= 0 or cell_size <= 0.0:
        return {}
    for geometry in geometries:
        shape = Polygon(geometry.corners)
        min_x, min_z, max_x, max_z = shape.bounds
        col0 = max(0, int(math.floor(min_x / cell_size)))
        row0 = max(0, int(math.floor(min_z / cell_size)))
        col1 = min(cells - 1, int(math.floor(max(0.0, max_x - 1.0e-9) / cell_size)))
        row1 = min(cells - 1, int(math.floor(max(0.0, max_z - 1.0e-9) / cell_size)))
        for row in range(row0, row1 + 1):
            z0 = row * cell_size
            for col in range(col0, col1 + 1):
                x0 = col * cell_size
                if not shape.intersects(box(x0, z0, x0 + cell_size, z0 + cell_size)):
                    continue
                by_cell.setdefault(row * cells + col, []).append(geometry)
    return {index: tuple(values) for index, values in by_cell.items()}


def _grass_background_path(generator, spec) -> str:
    materials = tuple(generator._material_definitions(spec))
    paths = tuple(generator._ground_texture_paths(spec))
    grass_index = next(
        (index for index, material in enumerate(materials)
         if str(getattr(material, "code", "")).casefold() == "g"),
        0,
    )
    if 0 <= grass_index < len(paths):
        return str(paths[grass_index])
    return rf"{getattr(spec, 'name', '')}\data\g.paa"


def _fallback_background(generator, runway, spec, path: str, size: int) -> Image.Image:
    materials = tuple(generator._material_definitions(spec))
    grass = next(
        (material for material in materials
         if str(getattr(material, "code", "")).casefold() == "g"),
        materials[0],
    )
    return runway._background_texture(
        grass,
        getattr(spec, "ground_texture_profile", "generated"),
        str(getattr(spec, "deterministic_seed", "cwr-worldgen")),
        size,
        ground_path=path,
    )


def _mark(mask: np.ndarray, image: np.ndarray, colour: np.ndarray) -> None:
    if bool(np.any(mask)):
        image[mask] = image[mask] * 0.08 + colour * 0.92


def _render_pitch_cell(
    *,
    cell_index: int,
    geometries: Sequence[_PitchGeometry],
    spec,
    base: Image.Image,
) -> Image.Image:
    size = int(base.width)
    cells = int(spec.cells)
    cell_size = float(spec.cell_size)
    if base.height != size or size < 16:
        raise ValueError("sports pitch background must be a square terrain texture")

    image = np.asarray(base.convert("RGB"), dtype=np.float32).copy()
    row, col = divmod(int(cell_index), cells)
    pixel_scale = cell_size / size
    xs = col * cell_size + (np.arange(size, dtype=np.float64) + 0.5) * pixel_scale
    zs = row * cell_size + (np.arange(size, dtype=np.float64) + 0.5) * pixel_scale
    world_x, world_z = np.meshgrid(xs, zs)
    white = np.asarray((232, 232, 216), dtype=np.float32)
    line_half = max(0.32, pixel_scale * 0.95)

    for geometry in geometries:
        dx = world_x - geometry.centre_x
        dz = world_z - geometry.centre_z
        along = dx * geometry.ux + dz * geometry.uz
        lateral = dx * geometry.px + dz * geometry.pz
        inside = (
            (np.abs(along) <= geometry.half_length)
            & (np.abs(lateral) <= geometry.half_width)
        )
        if not bool(np.any(inside)):
            continue

        boundary = inside & (
            (np.abs(np.abs(along) - geometry.half_length) <= line_half)
            | (np.abs(np.abs(lateral) - geometry.half_width) <= line_half)
        )
        centre_line = inside & (np.abs(along) <= line_half)
        radius = min(9.15, geometry.half_width * 0.35, geometry.half_length * 0.22)
        radial = np.hypot(along, lateral)
        centre_circle = inside & (np.abs(radial - radius) <= line_half)
        centre_spot = inside & (radial <= max(0.38, line_half))
        markings = boundary | centre_line | centre_circle | centre_spot

        if geometry.soccer:
            penalty_depth = min(16.5, geometry.half_length * 0.34)
            penalty_half_width = min(20.16, geometry.half_width * 0.82)
            goal_depth = min(5.5, geometry.half_length * 0.14)
            goal_half_width = min(9.16, geometry.half_width * 0.45)
            left_distance = along + geometry.half_length
            right_distance = geometry.half_length - along

            for distance in (left_distance, right_distance):
                penalty_box = inside & (
                    ((np.abs(distance - penalty_depth) <= line_half)
                     & (np.abs(lateral) <= penalty_half_width))
                    | ((distance <= penalty_depth + line_half)
                       & (np.abs(np.abs(lateral) - penalty_half_width) <= line_half))
                )
                goal_box = inside & (
                    ((np.abs(distance - goal_depth) <= line_half)
                     & (np.abs(lateral) <= goal_half_width))
                    | ((distance <= goal_depth + line_half)
                       & (np.abs(np.abs(lateral) - goal_half_width) <= line_half))
                )
                penalty_spot = inside & (
                    np.hypot(distance - min(11.0, penalty_depth * 0.67), lateral)
                    <= max(0.38, line_half)
                )
                markings |= penalty_box | goal_box | penalty_spot

        _mark(markings, image, white)

    return Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8), mode="RGB")


def _cache_root(source_dir: Path, spec) -> Path:
    shared = shared_cache_root(getattr(spec, "cache_dir", None))
    if shared is not None:
        return shared / SPORTS_TEXTURE_CACHE_DIRNAME
    return source_dir.parent / SHARED_CACHE_DIRNAME / SPORTS_TEXTURE_CACHE_DIRNAME


def _geometry_fingerprint(geometry: _PitchGeometry) -> dict[str, object]:
    return {
        "osm_key": geometry.osm_key,
        "centre": [round(geometry.centre_x, 5), round(geometry.centre_z, 5)],
        "half_length": round(geometry.half_length, 5),
        "half_width": round(geometry.half_width, 5),
        "axis": [round(geometry.ux, 7), round(geometry.uz, 7)],
        "soccer": geometry.soccer,
    }


def apply_sports_pitch_textures(
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

    geometries = _pitch_geometries(dataset, projection)
    if not geometries:
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()
    cell_map = _pitch_cells(geometries, int(spec.cells), float(spec.cell_size))
    if not cell_map:
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()

    source_dir = Path(source_dir)
    for stale in source_dir.glob(f"{SPORTS_TEXTURE_PREFIX}[0-9a-f][0-9a-f][0-9a-f].paa"):
        stale.unlink()
    for stale in source_dir.glob(f".{SPORTS_TEXTURE_PREFIX}cell*.paa"):
        stale.unlink()
    report_path = source_dir / "sports-pitch-textures.json"
    report_path.unlink(missing_ok=True)

    if len(texture_paths) + len(cell_map) > RVW4_TEXTURE_LIMIT:
        report_path.write_text(json.dumps({
            "schema": 1,
            "mode": "skipped-texture-table-budget",
            "pitch_count": len(geometries),
            "pitch_cells": len(cell_map),
            "base_texture_entries": len(texture_paths),
            "texture_limit": RVW4_TEXTURE_LIMIT,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return tuple(map(int, texture_indices)), tuple(map(str, texture_paths)), ()

    grass_path = _grass_background_path(generator, spec)
    exact_source = exact._load_exact_texture(source_dir, spec, grass_path)
    if exact_source is not None:
        base = exact_source.top_image.copy()
        background_sha = sha256(exact_source.data).hexdigest()
        background_source = exact_source.source
    else:
        size = int(getattr(runway, "RUNWAY_TEXTURE_SIZE", 128))
        base = _fallback_background(generator, runway, spec, grass_path, size)
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
            "schema": SPORTS_TEXTURE_CACHE_SCHEMA,
            "renderer": "sports-pitch-world-aligned-v1",
            "profile": str(getattr(spec, "ground_texture_profile", "generated")),
            "background_path": _canonical(grass_path),
            "background_sha256": background_sha,
            "cells": int(spec.cells),
            "cell_size": float(spec.cell_size),
            "cell_index": int(cell_index),
            "pitches": [_geometry_fingerprint(value) for value in geometries_here],
        }
        cached = cache_dir / f"{cache_key('sports-pitch-cell-v1', payload)}.paa"
        candidate = source_dir / f".{SPORTS_TEXTURE_PREFIX}cell{cell_index:06x}.paa"

        def paint(destination: Path, *, index=cell_index, values=geometries_here):
            image = _render_pitch_cell(
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
            raise ValueError("sports pitch textures exceed the RVW4 texture table")
        filename = f"{SPORTS_TEXTURE_PREFIX}{len(generated_paths):03x}.paa"
        candidate.replace(source_dir / filename)
        wire_path = rf"{spec.name}\{filename}"
        revised_paths.append(wire_path)
        revised_indices[cell_index] = slot
        generated_paths.append(wire_path)
        reusable[digest] = (data, slot)

    report_path.write_text(json.dumps({
        "schema": 1,
        "mode": "generated-terrain-textures",
        "strategy": "default-grass-plus-world-aligned-markings",
        "pitch_count": len(geometries),
        "pitch_cells": len(cell_map),
        "background_path": grass_path,
        "background_source": background_source,
        "generated_pitch_textures": len(generated_paths),
        "reused_pitch_cell_assignments": reused_cells,
        "cache_directory": str(cache_dir),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "base_texture_entries": len(texture_paths),
        "final_texture_entries": len(revised_paths),
        "texture_limit": RVW4_TEXTURE_LIMIT,
        "texture_paths": generated_paths,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return tuple(revised_indices), tuple(revised_paths), tuple(generated_paths)


def install_sports_pitch_surface_policy() -> None:
    """Install pitch preparation between runway compositing and RVW4 writing."""
    global _INSTALLED
    if _INSTALLED:
        return

    from . import generator
    from . import runway_surface_policy as runway
    from .progress import report_progress

    original_writer = runway._ORIGINAL_WRITE_RVW4
    if not callable(original_writer):
        return

    def write_rvw4_with_sports(
        path, width, height, elevations, texture_indices, texture_paths, objects, **kwargs
    ):
        context = runway._RUNWAY_CONTEXT.get()
        if (
            context is None
            or context.world_name.casefold() != Path(path).stem.casefold()
            or context.cells != int(width)
            or context.cells != int(height)
        ):
            _GENERATED_SPORTS_PATHS.set(())
            return original_writer(
                path, width, height, elevations, texture_indices, texture_paths, objects, **kwargs
            )

        geometries = _pitch_geometries(context.dataset, context.projection)
        if not geometries:
            _GENERATED_SPORTS_PATHS.set(())
            return original_writer(
                path, width, height, elevations, texture_indices, texture_paths, objects, **kwargs
            )

        started = perf_counter()
        report_progress(86, f"Preparing sports pitch terrain textures ({len(geometries)} pitches)")
        revised_indices, revised_paths, generated = apply_sports_pitch_textures(
            Path(path).parent,
            context.dataset,
            context.projection,
            context.spec,
            texture_indices,
            texture_paths,
        )
        _GENERATED_SPORTS_PATHS.set(generated)
        elapsed = perf_counter() - started
        report_path = Path(path).parent / "sports-pitch-textures.json"
        detail = ""
        if report_path.is_file():
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
                detail = (
                    f"{int(report.get('pitch_cells', 0))} cells; "
                    f"cache {int(report.get('cache_hits', 0))} hits/"
                    f"{int(report.get('cache_misses', 0))} rendered; "
                    f"{int(report.get('generated_pitch_textures', 0))} unique PAAs"
                )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
        report_progress(
            86,
            "Sports pitch terrain textures ready"
            + (f" ({detail}; {elapsed:.2f}s)" if detail else f" ({elapsed:.2f}s)"),
        )
        return original_writer(
            path, width, height, elevations, revised_indices, revised_paths, objects, **kwargs
        )

    # The runway wrapper remains the outer write hook.  When a runway exists it
    # first revises its own cells, then delegates here; without a runway it still
    # delegates here directly.  The fast RVW4 writer remains underneath us.
    runway._ORIGINAL_WRITE_RVW4 = write_rvw4_with_sports

    # Validation is similarly layered beneath the runway validator.  At call time
    # it sees either the normal ground paths or the runway-extended paths, and then
    # appends the generated sports texture paths for the base validator.
    original_validate = runway._ORIGINAL_VALIDATE_MILESTONE4

    def validate_with_sports(*args, **kwargs):
        generated = _GENERATED_SPORTS_PATHS.get()
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

    runway._ORIGINAL_VALIDATE_MILESTONE4 = validate_with_sports
    _INSTALLED = True
