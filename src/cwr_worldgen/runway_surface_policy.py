# SPDX-License-Identifier: GPL-3.0-or-later
"""Render OSM runway centre-lines as oriented stock-texture surface models.

RVW4 terrain cells contain only a texture-table index.  They cannot rotate one
terrain texture per cell, which is why the verified Nogova ``runtr_d`` artwork
appeared ninety degrees wrong on a north/south OSM runway.

Keep the ordinary terrain underneath the airport and place one thin generated
runway P3D over each OSM runway centre-line.  The model owns UV orientation and
therefore rotates cleanly with the OSM bearing.  It also uses the verified stock
end caps: runtr_z/runtr_k for grass-family worlds and runpi_z/runpi_k for Desert.
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
    RUNWAY_MODEL_WIDTH_METRES,
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
_SURFACE_CACHE_V15 = "surface-pipeline-v15-oriented-runway-models"
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


def _canonical_runway_endpoints(
    points: Sequence[tuple[float, float]],
    profile: object,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Choose the stock end-cap direction independent of OSM node order.

    Nogova names runtr_z/runtr_k are west/east ends.  The desert runpi_z/runpi_k
    pair is south/north.  Canonicalising the chord before creating the object
    keeps those end roles correct even when an OSM way happens to be digitised in
    the reverse direction.
    """
    if len(points) < 2:
        return None
    first, last = points[0], points[-1]
    if runway_family(profile) == "desert":
        reverse = (last[1], last[0]) < (first[1], first[0])
    else:
        reverse = (last[0], last[1]) < (first[0], first[1])
    return (last, first) if reverse else (first, last)


def runway_overlay_objects(
    dataset,
    projection,
    elevations: Sequence[float],
    spec,
    *,
    starting_id: int = 1,
):
    """Create one rotatable, terrain-following surface object per runway way."""
    from . import playability as _playability

    objects = []
    next_id = int(starting_id)
    for feature in dataset.aeroway_lines:
        if str(feature.tags.get("aeroway", "")).strip().casefold() != "runway":
            continue
        points = _clean_projected_points(feature, projection)
        endpoints = _canonical_runway_endpoints(points, spec.ground_texture_profile)
        if endpoints is None:
            continue
        start, end = endpoints
        length = math.dist(start, end)
        if length < 1.0:
            continue
        centre_x = (start[0] + end[0]) * 0.5
        centre_z = (start[1] + end[1]) * 0.5
        world_size = float(getattr(spec, "world_size", getattr(projection, "world_size", 0.0)))
        if world_size > 0.0 and not (
            0.0 <= centre_x < world_size and 0.0 <= centre_z < world_size
        ):
            continue
        model_path = runway_model_path(
            spec.name,
            spec.ground_texture_profile,
            length,
            width_metres=RUNWAY_MODEL_WIDTH_METRES,
        )
        objects.append(
            _playability._road_object_on_slope(
                next_id,
                model_path,
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
    """Install oriented runway models after the final road/surface wrappers."""
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
        # The P3D contains its own grass/sand shoulders.  Leaving the historical
        # generic paved runway underlay visible outside that 50 m stock tile
        # produces a grey halo, so only line runways are removed here.  Aprons,
        # taxiways and area-only runways retain the old paved fallback.
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
        # v14 cached the sideways stock texture directly into WRP terrain cells.
        # Force the ordinary surface pass to rebuild without that material before
        # the oriented P3D overlay is emitted.
        if namespace == _SURFACE_CACHE_V11:
            namespace = _SURFACE_CACHE_V15
        return _ORIGINAL_CACHE_KEY(namespace, payload)

    surface._aeroway_mask = aeroway_mask_without_line_runways
    generator.fit_road_objects = fit_road_objects_with_runways
    playability.fit_road_objects = fit_road_objects_with_runways
    generator.cache_key = runway_cache_key
    _INSTALLED = True
