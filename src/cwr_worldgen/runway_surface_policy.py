# SPDX-License-Identifier: GPL-3.0-or-later
"""Render OSM runway centre-lines as oriented stock-texture surface models.

RVW4 terrain cells contain only a texture-table index. They cannot rotate one
terrain texture per cell, which is why the verified Nogova ``runtr_d`` artwork
appeared ninety degrees wrong on a north/south OSM runway.

Keep the ordinary terrain underneath the airport and place thin generated 50 m
runway P3Ds over each OSM runway centre-line. The models own UV orientation and
therefore rotate cleanly with the OSM bearing. The first/last tiles use the
verified stock end caps: runtr_z/runtr_k for grass-family worlds and
runpi_z/runpi_k for Desert, with runtr_d/runpi_d between them.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Sequence

from .runway_model_policy import (
    DESERT_RUNWAY_END_TEXTURE,
    DESERT_RUNWAY_MIDDLE_TEXTURE,
    DESERT_RUNWAY_START_TEXTURE,
    GRASS_RUNWAY_END_TEXTURE,
    GRASS_RUNWAY_MIDDLE_TEXTURE,
    GRASS_RUNWAY_START_TEXTURE,
    RUNWAY_TEXTURE_TILE_METRES,
    install_runway_model_policy,
    runway_family,
    runway_model_path,
    runway_texture_triplet,
)


# Compatibility names retained for callers/tests from the first runway pass.
GRASS_RUNWAY_TEXTURE = GRASS_RUNWAY_MIDDLE_TEXTURE
DESERT_RUNWAY_TEXTURE = DESERT_RUNWAY_MIDDLE_TEXTURE
RUNWAY_SURFACE_OFFSET_METRES = 0.060
_SURFACE_CACHE_V11 = "surface-pipeline-v11-vectorized-material-pass"
_SURFACE_CACHE_V16 = "surface-pipeline-v16-oriented-runway-tiles"
_INSTALLED = False
_ORIGINAL_FIT_ROAD_OBJECTS = None
_ORIGINAL_CACHE_KEY = None
_ORIGINAL_AEROWAY_MASK = None


def runway_texture_for_profile(profile: object) -> str:
    """Return the repeating stock runway texture for one ground preset."""
    return runway_texture_triplet(profile)[1]


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
    """Orient a runway independently of the arbitrary OSM way-node direction.

    ``runtr_z``/``runtr_k`` are the stock grass west/east end textures, while
    ``runpi_z``/``runpi_k`` are the stock desert south/north pair. For diagonal
    runways the same primary axis chooses a stable end ordering, with the other
    coordinate used only as a deterministic tie-breaker.
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


def _runway_tile_plan(
    points: Sequence[tuple[float, float]],
    profile: object,
) -> tuple[tuple[str, tuple[float, float], tuple[float, float]], ...]:
    """Return adjacent 50 m runway tiles covering one mapped centre-line.

    Tiles stay at the stock 50 m size so the BI artwork is never stretched.
    When the mapped runway length is not an exact multiple of 50 m, the excess
    coverage is split equally between both ends. This keeps every tile exactly
    adjacent to its neighbours, avoiding either visible gaps or coplanar overlap.
    """
    from . import playability as _playability

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

    planned: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
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
    """Create rotatable terrain-following 50 m surface tiles for runway ways."""
    from . import playability as _playability

    objects = []
    next_id = int(starting_id)
    profile = getattr(spec, "ground_texture_profile", "generated")
    world_size = float(getattr(spec, "world_size", getattr(projection, "world_size", 0.0)))

    for feature in dataset.aeroway_lines:
        if str(feature.tags.get("aeroway", "")).strip().casefold() != "runway":
            continue
        points = _clean_projected_points(feature, projection)
        for role, start, end in _runway_tile_plan(points, profile):
            centre_x = (start[0] + end[0]) * 0.5
            centre_z = (start[1] + end[1]) * 0.5
            if world_size > 0.0 and not (
                0.0 <= centre_x < world_size and 0.0 <= centre_z < world_size
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


def install_runway_surface_policy() -> None:
    """Install oriented runway tiles after the final road/surface wrappers."""
    global _INSTALLED, _ORIGINAL_FIT_ROAD_OBJECTS, _ORIGINAL_CACHE_KEY
    global _ORIGINAL_AEROWAY_MASK
    if _INSTALLED:
        return

    from . import generator
    from . import playability
    from . import surface_pass as surface

    install_runway_model_policy()
    _ORIGINAL_FIT_ROAD_OBJECTS = generator.fit_road_objects
    _ORIGINAL_CACHE_KEY = generator.cache_key
    _ORIGINAL_AEROWAY_MASK = surface._aeroway_mask

    def aeroway_mask_without_line_runways(dataset, projection, cells):
        # The P3D contains the stock runway surface itself. Leaving the historic
        # generic paved runway underlay visible around it produces a grey halo.
        # Aprons, taxiways and area-only runways keep their existing paved mask.
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

    def runway_cache_key(namespace: str, payload):
        # v14 cached sideways runway artwork directly into WRP terrain. v15 used
        # one obsolete long runway model. Re-key the surface stage so testing a
        # fixed branch cannot resurrect either representation from an old build.
        if namespace == _SURFACE_CACHE_V11:
            namespace = _SURFACE_CACHE_V16
        return _ORIGINAL_CACHE_KEY(namespace, payload)

    surface._aeroway_mask = aeroway_mask_without_line_runways
    generator.fit_road_objects = fit_road_objects_with_runways
    playability.fit_road_objects = fit_road_objects_with_runways
    generator.cache_key = runway_cache_key
    _INSTALLED = True
