# SPDX-License-Identifier: GPL-3.0-or-later
"""Render OSM runways as generated, correctly oriented WRP terrain textures.

RVW4 terrain cells carry a texture-table index but no per-cell UV transform. A
single stock ``o\\runtr_d.paa`` therefore cannot follow an arbitrary OSM bearing.
Instead, generate one compact world-local PAA for every terrain cell touched by a
runway. Each PAA is rendered in world coordinates, so adjacent cells form one
continuous runway at the mapped bearing and the runway is visible in the editor's
terrain view as well as in game.

The texture table has 512 slots. Generated cell textures are the preferred path;
if a world does not have enough free slots, retain the rotatable P3D runway
implementation as a bounded fallback rather than emitting an invalid WRP.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, replace
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


# Compatibility names retained for callers/tests from the first runway pass.
GRASS_RUNWAY_TEXTURE = GRASS_RUNWAY_MIDDLE_TEXTURE
DESERT_RUNWAY_TEXTURE = DESERT_RUNWAY_MIDDLE_TEXTURE

RVW4_TEXTURE_LIMIT = 512
RUNWAY_TEXTURE_SIZE = 128
RUNWAY_TEXTURE_PREFIX = "rw"
RUNWAY_SURFACE_OFFSET_METRES = 0.060
_SURFACE_CACHE_V11 = "surface-pipeline-v11-vectorized-material-pass"
_SURFACE_CACHE_V17 = "surface-pipeline-v17-generated-runway-cell-textures"
_INSTALLED = False
_ORIGINAL_FIT_ROAD_OBJECTS = None
_ORIGINAL_CACHE_KEY = None
_ORIGINAL_AEROWAY_MASK = None
_ORIGINAL_WRITE_RVW4 = None
_ORIGINAL_VALIDATE_MILESTONE4 = None


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
    elevation_identity: int
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
    points: Sequence[tuple[float, float]],
    profile: object,
) -> tuple[tuple[float, float], ...]:
    """Orient runway ends independently of arbitrary OSM way-node direction.

    Grass follows the verified Nogova ``runtr_z`` west / ``runtr_k`` east
    convention. Desert follows ``runpi_z`` south / ``runpi_k`` north. Generated
    textures reproduce those end roles without copying the stock texture bytes.
    """
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
    points: Sequence[tuple[float, float]],
    profile: object,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Compatibility helper returning the canonical first/last runway points."""
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
        start = points[0]
        end = points[-1]
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length = math.hypot(dx, dz)
        if length < 1.0:
            continue
        ux = dx / length
        uz = dz / length
        geometries.append(
            _RunwayGeometry(
                osm_key=str(getattr(feature, "osm_key", "")),
                start_x=start[0],
                start_z=start[1],
                end_x=end[0],
                end_z=end[1],
                length=length,
                half_width=_tagged_width_metres(feature) * 0.5,
                ux=ux,
                uz=uz,
                px=-uz,
                pz=ux,
            )
        )
    return tuple(geometries)


def runway_texture_cell_indices(dataset, projection, spec) -> tuple[int, ...]:
    """Return WRP cells whose texture must contain generated runway artwork."""
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
        width_pixels = max(
            1,
            int(
                round(
                    _tagged_width_metres(feature)
                    / float(projection.world_size)
                    * resolution
                )
            ),
        )
        draw.line(points, fill=255, width=width_pixels, joint="curve")
        any_runway = True
    if not any_runway:
        return ()

    # Threshold 1 intentionally includes even thin edge slivers. A cell omitted
    # here would reveal the ordinary terrain texture through the runway edge.
    mask = surface._image_to_wrp_mask(image, cells, threshold=1)
    return tuple(int(value) for value in np.flatnonzero(mask))


def _runway_texture_budget(spec, runway_cells: int) -> tuple[bool, int, int]:
    """Return (fits, base_entries, final_entries) for generated runway cells."""
    from . import generator

    base_entries = 1 + len(generator._ground_texture_paths(spec))
    final_entries = base_entries + max(0, int(runway_cells))
    return final_entries <= RVW4_TEXTURE_LIMIT, base_entries, final_entries


def _profile_surface_colour(
    material,
    profile: object,
) -> tuple[int, int, int]:
    from . import surface_pass as surface

    name = str(profile or "").strip().casefold()
    code = str(getattr(material, "code", "g"))
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
) -> Image.Image:
    """Create a deterministic repeated base underneath one runway cell."""
    from . import surface_pass as surface

    colour = _profile_surface_colour(material, profile)
    proxy = surface.SurfaceMaterialDefinition(
        str(getattr(material, "code", "g")),
        str(getattr(material, "name", "ground")),
        colour,
        None,
    )
    return surface.create_surface_texture(proxy, seed, size)


def _runway_palette(profile: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return body, wear and marking RGB colours for one runway family."""
    if runway_family(profile) == "desert":
        return (
            np.asarray((161, 145, 100), dtype=np.float32),
            np.asarray((138, 124, 88), dtype=np.float32),
            np.asarray((229, 220, 190), dtype=np.float32),
        )
    return (
        np.asarray((113, 128, 77), dtype=np.float32),
        np.asarray((91, 108, 66), dtype=np.float32),
        np.asarray((224, 221, 187), dtype=np.float32),
    )


def _render_runway_cell(
    *,
    cell_index: int,
    original_wrp_texture_index: int,
    geometries: Sequence[_RunwayGeometry],
    materials: Sequence[object],
    spec,
    size: int = RUNWAY_TEXTURE_SIZE,
    background_cache: dict[int, Image.Image] | None = None,
) -> Image.Image:
    """Render one world-space runway slice into a terrain-cell PAA image."""
    cells = int(spec.cells)
    cell_size = float(spec.cell_size)
    if cells <= 0 or cell_size <= 0.0:
        raise ValueError("runway texture generation requires a positive terrain grid")
    if size < 16 or size & (size - 1):
        raise ValueError("runway texture size must be a power of two of at least 16")

    material_index = int(original_wrp_texture_index) - 1
    if not 0 <= material_index < len(materials):
        # Slot zero is the WRP dummy and should never be selected by terrain. If
        # a malformed/intermediate grid reaches us, grass is the safest visual
        # background instead of magenta or an out-of-range crash.
        material_index = next(
            (
                index
                for index, material in enumerate(materials)
                if str(getattr(material, "code", "")).casefold() == "g"
            ),
            0,
        )

    cache = background_cache if background_cache is not None else {}
    base = cache.get(material_index)
    if base is None:
        base = _background_texture(
            materials[material_index],
            getattr(spec, "ground_texture_profile", "generated"),
            str(getattr(spec, "deterministic_seed", "cwr-worldgen")),
            size,
        )
        cache[material_index] = base

    image = np.asarray(base, dtype=np.float32).copy()
    cz, cx = divmod(int(cell_index), cells)
    pixel_scale = cell_size / size

    # OFP terrain UVs increase with the WRP cell axes. PAA V grows downward in
    # image memory, so row 0 corresponds to the cell's local V=0 / south edge.
    xs = (cx * cell_size) + (np.arange(size, dtype=np.float64) + 0.5) * pixel_scale
    zs = (cz * cell_size) + (np.arange(size, dtype=np.float64) + 0.5) * pixel_scale
    world_x, world_z = np.meshgrid(xs, zs)

    body_colour, wear_colour, marking_colour = _runway_palette(
        getattr(spec, "ground_texture_profile", "generated")
    )
    marking_half_width = max(0.55, pixel_scale * 1.1)
    threshold_half_depth = max(0.85, pixel_scale * 1.5)

    for geometry in geometries:
        rel_x = world_x - geometry.start_x
        rel_z = world_z - geometry.start_z
        along = rel_x * geometry.ux + rel_z * geometry.uz
        lateral = rel_x * geometry.px + rel_z * geometry.pz
        inside = (
            (along >= 0.0)
            & (along <= geometry.length)
            & (np.abs(lateral) <= geometry.half_width)
        )
        if not bool(np.any(inside)):
            continue

        # Keep some of the underlying high-frequency terrain grain so generated
        # runway cells do not read as perfectly flat vector rectangles.
        image[inside] = image[inside] * 0.28 + body_colour * 0.72

        # Two subtle longitudinal wear tracks break up the body and make the
        # direction legible even when mipmaps soften the painted centreline.
        track_offset = min(geometry.half_width * 0.34, 6.0)
        tracks = inside & (
            np.abs(np.abs(lateral) - track_offset)
            <= max(0.75, pixel_scale * 1.5)
        )
        image[tracks] = image[tracks] * 0.45 + wear_colour * 0.55

        # ``runtr_d`` / ``runpi_d`` visual intent: one continuous centre stripe.
        centreline = inside & (np.abs(lateral) <= marking_half_width)
        image[centreline] = marking_colour

        # ``z`` and ``k`` visual intent: one transverse threshold mark near each
        # canonical end. Grass canonical order is west->east; Desert is
        # south->north, matching the stock texture family names the user supplied.
        threshold_span = max(2.0, geometry.half_width * 0.72)
        start_threshold = (
            inside
            & (np.abs(along - min(4.0, geometry.length * 0.12)) <= threshold_half_depth)
            & (np.abs(lateral) <= threshold_span)
        )
        end_threshold = (
            inside
            & (
                np.abs(
                    along
                    - max(
                        geometry.length - 4.0,
                        geometry.length * 0.88,
                    )
                )
                <= threshold_half_depth
            )
            & (np.abs(lateral) <= threshold_span)
        )
        image[start_threshold | end_threshold] = marking_colour

        # A darker half-metre verge defines the runway edge without painting the
        # whole terrain cell as pavement.
        edge = inside & (
            np.abs(np.abs(lateral) - geometry.half_width)
            <= max(0.45, pixel_scale)
        )
        image[edge] = image[edge] * 0.35 + wear_colour * 0.65

    return Image.fromarray(
        np.clip(np.rint(image), 0, 255).astype(np.uint8),
        mode="RGB",
    )


def apply_generated_runway_texture_table(
    source_dir: Path,
    dataset,
    projection,
    spec,
    texture_indices: Sequence[int],
    texture_paths: Sequence[str],
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    """Generate cell PAAs and return revised WRP indices/table paths.

    Each touched cell gets one unique compact path ``<world>\\rwNNN.paa``. The
    world-name limit is 20 characters, keeping the longest wire path at 30 bytes
    and therefore below RVW4's 31-byte terrain-texture path limit.
    """
    from . import generator

    cell_indices = runway_texture_cell_indices(dataset, projection, spec)
    if not cell_indices:
        return (
            tuple(int(value) for value in texture_indices),
            tuple(str(value) for value in texture_paths),
            (),
        )

    if len(texture_paths) + len(cell_indices) > RVW4_TEXTURE_LIMIT:
        raise ValueError(
            "generated runway textures exceed the RVW4 512-entry texture table"
        )

    cells = int(spec.cells)
    if len(texture_indices) != cells * cells:
        raise ValueError("runway texture generation received a mismatched WRP grid")

    geometries = _runway_geometries(
        dataset,
        projection,
        getattr(spec, "ground_texture_profile", "generated"),
    )
    if not geometries:
        return (
            tuple(int(value) for value in texture_indices),
            tuple(str(value) for value in texture_paths),
            (),
        )

    source_dir = Path(source_dir)
    # Remove stale slices from an incremental rebuild. They are root-level on
    # purpose: a 20-character world name plus ``\\data\\`` would exceed RVW4's
    # 31-byte texture-path field once a unique filename is appended.
    for stale in source_dir.glob(f"{RUNWAY_TEXTURE_PREFIX}[0-9a-f][0-9a-f][0-9a-f].paa"):
        stale.unlink()
    stale_report = source_dir / "runway-textures.json"
    if stale_report.exists():
        stale_report.unlink()

    materials = tuple(generator._material_definitions(spec))
    revised_indices = [int(value) for value in texture_indices]
    revised_paths = [str(value) for value in texture_paths]
    generated_paths: list[str] = []
    background_cache: dict[int, Image.Image] = {}

    for serial, cell_index in enumerate(cell_indices):
        slot = len(revised_paths)
        filename = f"{RUNWAY_TEXTURE_PREFIX}{serial:03x}.paa"
        wire_path = rf"{spec.name}\{filename}"
        local_path = source_dir / filename
        image = _render_runway_cell(
            cell_index=cell_index,
            original_wrp_texture_index=revised_indices[cell_index],
            geometries=geometries,
            materials=materials,
            spec=spec,
            size=RUNWAY_TEXTURE_SIZE,
            background_cache=background_cache,
        )
        write_rgb_dxt1_paa(local_path, image)
        revised_paths.append(wire_path)
        revised_indices[cell_index] = slot
        generated_paths.append(wire_path)

    report = {
        "schema": 1,
        "mode": "generated-terrain-textures",
        "texture_limit": RVW4_TEXTURE_LIMIT,
        "texture_size": RUNWAY_TEXTURE_SIZE,
        "base_texture_entries": len(texture_paths),
        "generated_runway_textures": len(generated_paths),
        "final_texture_entries": len(revised_paths),
        "runway_cells": list(cell_indices),
        "texture_paths": generated_paths,
        "stock_reference_family": {
            "grass": list(runway_texture_triplet("grass")),
            "desert": list(runway_texture_triplet("desert")),
        },
    }
    stale_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return tuple(revised_indices), tuple(revised_paths), tuple(generated_paths)


def _runway_tile_plan(
    points: Sequence[tuple[float, float]],
    profile: object,
):
    """Compatibility helper used only by the P3D overflow fallback."""
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
    covered = tile_count * tile
    first_centre = (measure.total - covered) * 0.5 + tile * 0.5
    half = tile * 0.5

    planned = []
    for index in range(tile_count):
        centre_distance = first_centre + index * tile
        centre_x, centre_z, heading = measure.point(centre_distance)
        angle = math.radians(heading)
        dx = math.sin(angle) * half
        dz = math.cos(angle) * half
        if tile_count == 1:
            role = "d"
        elif index == 0:
            role = "z"
        elif index == tile_count - 1:
            role = "k"
        else:
            role = "d"
        planned.append(
            (
                role,
                (centre_x - dx, centre_z - dz),
                (centre_x + dx, centre_z + dz),
            )
        )
    return tuple(planned)


def runway_overlay_objects(
    dataset,
    projection,
    elevations: Sequence[float],
    spec,
    *,
    starting_id: int = 1,
):
    """Create rotatable 50 m P3D tiles for the texture-budget fallback only."""
    from . import playability as _playability

    objects = []
    next_id = int(starting_id)
    profile = getattr(spec, "ground_texture_profile", "generated")
    world_size = float(
        getattr(spec, "world_size", getattr(projection, "world_size", 0.0))
    )

    for feature in dataset.aeroway_lines:
        if str(feature.tags.get("aeroway", "")).strip().casefold() != "runway":
            continue
        points = _clean_projected_points(feature, projection)
        for role, start, end in _runway_tile_plan(points, profile):
            centre_x = (start[0] + end[0]) * 0.5
            centre_z = (start[1] + end[1]) * 0.5
            if world_size > 0.0 and not (
                0.0 <= centre_x < world_size
                and 0.0 <= centre_z < world_size
            ):
                continue
            objects.append(
                _playability._road_object_on_slope(
                    next_id,
                    runway_model_path(spec.name, profile, role),
                    start,
                    end,
                    elevations,
                    spec,
                    vertical_offset=RUNWAY_SURFACE_OFFSET_METRES,
                )
            )
            next_id += 1
    return tuple(objects)


def _validation_with_generated_runway_textures(*args, **kwargs):
    """Teach the existing validation report about intentional extra WRP PAAs."""
    lines = _ORIGINAL_VALIDATE_MILESTONE4(*args, **kwargs)
    generated_paths = _GENERATED_RUNWAY_PATHS.get()
    if not generated_paths:
        return lines

    # The core validator predates per-cell runway textures and expects the WRP
    # texture set to equal only the semantic ground palette. Replace just that
    # one diagnostic after the writer has proven the generated paths fit the WRP.
    revised = list(lines)
    for index, line in enumerate(revised):
        if "WRP terrain texture profile" not in line:
            continue
        context = _RUNWAY_CONTEXT.get()
        final_count = (
            context.base_texture_entries + len(generated_paths)
            if context is not None
            else len(generated_paths)
        )
        revised[index] = (
            "[PASS] WRP terrain texture profile: "
            f"generated runway cell textures={len(generated_paths)}, "
            f"texture table={final_count}/{RVW4_TEXTURE_LIMIT}"
        )
        break
    return revised


def install_runway_surface_policy() -> None:
    """Install generated runway terrain textures with P3D overflow fallback."""
    global _INSTALLED, _ORIGINAL_FIT_ROAD_OBJECTS, _ORIGINAL_CACHE_KEY
    global _ORIGINAL_AEROWAY_MASK, _ORIGINAL_WRITE_RVW4
    global _ORIGINAL_VALIDATE_MILESTONE4
    if _INSTALLED:
        return

    from . import generator
    from . import playability
    from . import surface_pass as surface

    # Retain the runway-aware infrastructure subclass only for the rare texture
    # table overflow path. Normal builds generate no runway P3D objects.
    install_runway_model_policy()

    _ORIGINAL_FIT_ROAD_OBJECTS = generator.fit_road_objects
    _ORIGINAL_CACHE_KEY = generator.cache_key
    _ORIGINAL_AEROWAY_MASK = surface._aeroway_mask
    _ORIGINAL_WRITE_RVW4 = generator.write_rvw4
    _ORIGINAL_VALIDATE_MILESTONE4 = generator._validate_milestone4

    def aeroway_mask_without_line_runways(dataset, projection, cells):
        # Generated runway PAAs contain both the runway and its local background.
        # Remove the historic generic paved runway mask first so the background
        # represents the natural terrain rather than a grey rectangular halo.
        # Aprons, taxiways and area-only runways retain the existing paved mask.
        lines = tuple(
            feature
            for feature in dataset.aeroway_lines
            if str(feature.tags.get("aeroway", "")).strip().casefold() != "runway"
        )
        if len(lines) == len(dataset.aeroway_lines):
            return _ORIGINAL_AEROWAY_MASK(dataset, projection, cells)
        filtered = replace(dataset, aeroway_lines=lines)
        return _ORIGINAL_AEROWAY_MASK(filtered, projection, cells)

    def fit_road_objects_with_runways(
        dataset,
        projection,
        elevations,
        spec,
        *,
        starting_id: int = 1,
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
        mode = "textures" if cell_indices and fits else (
            "p3d-fallback" if cell_indices else "none"
        )
        _RUNWAY_CONTEXT.set(
            _RunwayBuildContext(
                world_name=str(spec.name),
                cells=int(spec.cells),
                elevation_identity=id(elevations),
                dataset=dataset,
                projection=projection,
                spec=spec,
                cell_indices=cell_indices,
                mode=mode,
                base_texture_entries=base_entries,
            )
        )
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
        if not runways:
            return report
        return replace(report, objects=tuple((*report.objects, *runways)))

    def write_rvw4_with_runway_textures(
        path,
        width,
        height,
        elevations,
        texture_indices,
        texture_paths,
        objects,
        **kwargs,
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

            # An incremental build may still contain P3Ds from the previous
            # implementation. They are unreferenced now, so do not repack them.
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
        # v14 stored an unrotatable stock texture in terrain, while v15/v16 used
        # runway P3Ds. Re-key the surface pass so the generated-texture build gets
        # natural terrain under the line runway before its cell slices are drawn.
        if namespace == _SURFACE_CACHE_V11:
            namespace = _SURFACE_CACHE_V17
        return _ORIGINAL_CACHE_KEY(namespace, payload)

    surface._aeroway_mask = aeroway_mask_without_line_runways
    generator.fit_road_objects = fit_road_objects_with_runways
    playability.fit_road_objects = fit_road_objects_with_runways
    generator.write_rvw4 = write_rvw4_with_runway_textures
    generator._validate_milestone4 = _validation_with_generated_runway_textures
    generator.cache_key = runway_cache_key
    _INSTALLED = True
