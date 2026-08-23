# SPDX-License-Identifier: GPL-3.0-or-later
"""Fill acute visual corner wedges at generated-gravel road junctions."""
from __future__ import annotations

from dataclasses import replace
from functools import wraps
import math

from . import generator as _generator
from . import playability as _p
from . import road_quality_policy as _rq
from .procedural_infrastructure import (
    GENERATED_GRAVEL_HALF_WIDTH_METRES,
    GENERATED_GRAVEL_VISUAL_OVERLAP_METRES,
    gravel_road_model_path,
)

# A generic plus-shaped hub is a good fit near 90 degrees. Below this angle the
# intersection of the two road-width strips extends far enough along the inside
# bisector that a triangular terrain wedge can remain visible in CWA.
_GRAVEL_CORNER_MAX_ANGLE_DEGREES = 78.0
# Very small angular separations are normally duplicate/parallel OSM incidents,
# not a junction corner that should be paved over.
_GRAVEL_CORNER_MIN_ANGLE_DEGREES = 30.0
_GRAVEL_CORNER_MARGIN_METRES = 0.25
# Generated gravel is normally exactly coplanar with the graded terrain. Raise
# the tiny patch only enough to make the old engine's depth buffer deterministic.
_GRAVEL_CORNER_PATCH_RAISE_METRES = 0.008

_ORIGINAL_FIT = _p.fit_road_objects
_INSTALLED = False


def _unit(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(vector[0], vector[1])
    if length <= 1.0e-9:
        return (0.0, 1.0)
    return (vector[0] / length, vector[1] / length)


def _direction_heading(direction: tuple[float, float]) -> float:
    return math.degrees(math.atan2(direction[0], direction[1])) % 360.0


def _is_generated_gravel_hub(junction) -> bool:
    return math.isclose(
        float(junction.half_width),
        float(GENERATED_GRAVEL_HALF_WIDTH_METRES),
        rel_tol=0.0,
        abs_tol=1.0e-6,
    )


def _piece_length(spec, nominal: int) -> float:
    return float(spec.road_segment_length) * float(nominal) / 25.0


def _corner_fillers(spec, context):
    """Yield minimal short-ribbon patches for acute gravel junction sectors."""

    half_width = float(GENERATED_GRAVEL_HALF_WIDTH_METRES)
    for junction in context.junctions.values():
        if not _is_generated_gravel_hub(junction) or len(junction.directions) < 3:
            continue

        directions = sorted(junction.directions, key=_direction_heading)
        headings = tuple(_direction_heading(direction) for direction in directions)
        for index, left in enumerate(directions):
            right = directions[(index + 1) % len(directions)]
            gap = (headings[(index + 1) % len(directions)] - headings[index]) % 360.0
            if not (_GRAVEL_CORNER_MIN_ANGLE_DEGREES <= gap < _GRAVEL_CORNER_MAX_ANGLE_DEGREES):
                continue

            half_angle = math.radians(gap * 0.5)
            sine = math.sin(half_angle)
            if sine <= 1.0e-6:
                continue

            # For two strips of half-width w meeting at angle A, the two inside
            # edges intersect on the bisector at w/sin(A/2). Cover from well
            # inside the existing hub to just beyond that apex.
            apex_radius = half_width / sine
            near_radius = min(float(junction.half_length), float(junction.half_width)) * 0.25
            far_radius = apex_radius + _GRAVEL_CORNER_MARGIN_METRES
            required_visual_span = max(0.0, far_radius - near_radius)

            nominal = None
            for candidate in (3, 6, 12):
                visual_length = _piece_length(spec, candidate) + 2.0 * GENERATED_GRAVEL_VISUAL_OVERLAP_METRES
                if visual_length >= required_visual_span:
                    nominal = candidate
                    break
            if nominal is None:
                continue

            bisector = _unit((left[0] + right[0], left[1] + right[1]))
            centre_radius = (near_radius + far_radius) * 0.5
            centre = (
                junction.point[0] + bisector[0] * centre_radius,
                junction.point[1] + bisector[1] * centre_radius,
            )
            model_length = _piece_length(spec, nominal)
            start = (
                centre[0] - bisector[0] * model_length * 0.5,
                centre[1] - bisector[1] * model_length * 0.5,
            )
            end = (
                centre[0] + bisector[0] * model_length * 0.5,
                centre[1] + bisector[1] * model_length * 0.5,
            )
            yield gravel_road_model_path(spec.name, nominal), start, end


def _fill_skewed_gravel_corners(report, elevations, spec, context):
    fillers = tuple(_corner_fillers(spec, context))
    if not fillers:
        return report, 0

    objects = list(report.objects)
    next_id = max((obj.object_id for obj in objects), default=0) + 1
    added = 0
    for model_path, start, end in fillers:
        centre_x = (start[0] + end[0]) * 0.5
        centre_z = (start[1] + end[1]) * 0.5
        if not (0.0 <= centre_x < spec.world_size and 0.0 <= centre_z < spec.world_size):
            continue
        patch = _p._road_object_on_slope(
            next_id,
            model_path,
            start,
            end,
            elevations,
            spec,
            vertical_offset=_p._STOCK_GRAVEL_VERTICAL_OFFSET_METRES,
        )
        # The surrounding generated gravel visual is exactly terrain-coplanar.
        # Eight millimetres is enough to make the patch win the depth test while
        # remaining visually flush and far below any meaningful clipping height.
        objects.append(replace(patch, y=patch.y + _GRAVEL_CORNER_PATCH_RAISE_METRES))
        next_id += 1
        added += 1

    if not added:
        return report, 0
    return replace(
        report,
        objects=tuple(objects),
        short_piece_objects=report.short_piece_objects + added,
    ), added


@wraps(_ORIGINAL_FIT)
def _fit(dataset, projection, elevations, spec, *, starting_id: int = 1, progress_callback=None):
    if not bool(getattr(spec, "stock_road_piece_fitting", False)):
        return _ORIGINAL_FIT(
            dataset,
            projection,
            elevations,
            spec,
            starting_id=starting_id,
            progress_callback=progress_callback,
        )

    deferred_completion: tuple[int, str] | None = None

    def progress(value: int, message: str) -> None:
        nonlocal deferred_completion
        if value >= 100 and message.startswith("Stock road fitting complete:"):
            deferred_completion = (value, message)
            return
        if progress_callback is not None:
            progress_callback(value, message)

    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress if progress_callback is not None else None,
    )
    context = _rq._Context(elevations, spec, _rq._junction_geometry(dataset, projection, spec))
    report, added = _fill_skewed_gravel_corners(report, elevations, spec, context)
    if progress_callback is not None and added:
        progress_callback(99, f"Filled {added:,} skewed gravel junction corners")
    if progress_callback is not None and deferred_completion is not None:
        progress_callback(
            deferred_completion[0],
            f"Stock road fitting complete: {len(report.objects):,} objects in {report.chain_count:,} chains",
        )
    return report


def install_gravel_corner_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
