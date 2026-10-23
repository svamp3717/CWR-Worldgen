# SPDX-License-Identifier: GPL-3.0-or-later
"""Terrain-aware cross-slope fitting for generated roads meeting pavement.

The user's real wg_belowtest23.wrp at (3302.524, 3167.709) shows why
flat-ground tests missed an in-game gravel cutoff: generated gravel P3D
visual faces on the uphill side sit 7-12 cm *below the WRP terrain*, even
though the stock-paved visual surface is geometrically above gravel.
A rigid pitched road tracks its centreline but cannot track lateral grade.

Near actual paved-road intersections, bank only generated dirt/gravel approach
ribbons using the RVW4's native affine terrain-shear transform (also used by
stock Nogova forests). Keep the visible centre and outer edges above terrain,
and limit uplift so lower-priority surfaces cannot paint over the paved road.
Explicit deeply pitched underlay terminals remain untouched.
"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Sequence

from .model import TerrainShearedWorldObject
from .osm import _sample_elevation, _triangle_elevation_bounds

_GROUND_CLEARANCE_METRES = 0.02
_PAVED_VISIBLE_CLEARANCE_METRES = 0.035
_MAX_LIFT_METRES = 0.18
_MAX_BANK_SLOPE = 0.10
_NEAR_PAVED_MARGIN_METRES = 4.0


def _contains(polygon: tuple[tuple[float, float], ...], x: float, z: float) -> bool:
    """Convex point-in-polygon including the boundary, without new dependencies."""
    signs = []
    for first, second in zip(polygon, (*polygon[1:], polygon[0])):
        cross = ((second[0] - first[0]) * (z - first[1])
                 - (second[1] - first[1]) * (x - first[0]))
        if abs(cross) > 1.0e-8:
            signs.append(cross > 0)
    return not signs or all(value == signs[0] for value in signs)


def ground_generated_approaches(
    report: object, spec: object, elevations: Sequence[float],
) -> object:
    """Correct only dirt/gravel generated approaches near paved road footprints."""
    # Import lazily to avoid a cycle: the final fitter calls this after its
    # normal geometry deduplication and junction/cap corrections are complete.
    from . import final_road_dedup_policy as road

    objects = tuple(report.objects)
    if not objects:
        return report
    cells = int(spec.cells)
    size = float(spec.cell_size)
    if cells <= 1 or size <= 0 or len(elevations) != cells * cells:
        return report
    paved = road._paved_blockers(report, spec)
    if not paved:
        return report

    paved_buckets: dict[tuple[int, int], set[int]] = {}
    for index, blocker in enumerate(paved):
        for bucket in road._polygon_buckets(blocker.polygon):
            paved_buckets.setdefault(bucket, set()).add(index)

    def terrain(x: float, z: float) -> float:
        return _sample_elevation(elevations, cells, size, x, z)

    updated: list[object] = list(objects)
    for index, obj in enumerate(objects):
        path = str(obj.model_path).replace("/", "\\").casefold()
        if "\\i\\road_gravel_w" not in path and "\\i\\road_dirt_w" not in path:
            continue
        # Terminal ribbons intentionally dive underneath paving; do not bank
        # them back upward or overwrite the seam's deliberate pitched end.
        if abs(float(obj.pitch_degrees)) > 2.0:
            continue
        axis = road._road_axis(obj, index, spec)
        if axis is None or axis.family not in {"gravel", "ces"}:
            continue

        # Expanded buckets find intersecting paved curves and approaches even
        # when the short connector's centre is outside the paved rectangle.
        min_x, min_z, max_x, max_z = axis.bounds
        margin = axis.half_width + _NEAR_PAVED_MARGIN_METRES
        possible: set[int] = set()
        for bucket in road._polygon_buckets((
            (min_x - margin, min_z - margin),
            (max_x + margin, min_z - margin),
            (max_x + margin, max_z + margin),
            (min_x - margin, max_z + margin),
        )):
            possible.update(paved_buckets.get(bucket, ()))
        if not possible:
            continue
        nearby = []
        for bidx in possible:
            blocker = paved[bidx]
            bx0, bz0, bx1, bz1 = blocker.bounds
            if (
                min_x - margin <= bx1 and max_x + margin >= bx0
                and min_z - margin <= bz1 and max_z + margin >= bz0
            ):
                nearby.append(blocker)
        if not nearby:
            continue

        x, z = float(obj.x), float(obj.z)
        delta = min(3.0, size * 0.25)
        gradient_x = (terrain(x + delta, z) - terrain(x - delta, z)) / (2 * delta)
        gradient_z = (terrain(x, z + delta) - terrain(x, z - delta)) / (2 * delta)
        if max(abs(gradient_x), abs(gradient_z)) > _MAX_BANK_SLOPE:
            continue

        heading = math.radians(float(obj.heading_degrees))
        cos_h, sin_h = math.cos(heading), math.sin(heading)
        half_length, half_width = axis.length * 0.5, axis.half_width
        # The generated P3D's ordinary visual skin is 0.025 m above origin.
        visual_base = float(obj.y) + float(road._p.GENERATED_GRAVEL_VISUAL_TOP_METRES)
        needed_lift = 0.0
        allowed_lift = _MAX_LIFT_METRES
        has_exposed = False
        for local_z in (-half_length, 0.0, half_length):
            for local_x in (-half_width, 0.0, half_width):
                world_x = x + cos_h * local_x + sin_h * local_z
                world_z = z - sin_h * local_x + cos_h * local_z
                bank = gradient_x * (world_x - x) + gradient_z * (world_z - z)
                visual_y = visual_base + bank
                covering = [
                    blocker for blocker in nearby
                    if _contains(blocker.polygon, world_x, world_z)
                ]
                if not covering:
                    has_exposed = True
                    needed_lift = max(
                        needed_lift,
                        max(_triangle_elevation_bounds(
                            elevations, cells, size, world_x, world_z,
                        )) + _GROUND_CLEARANCE_METRES - visual_y,
                    )

        # The nine terrain samples do not bound the entire paved overlap:
        # narrow or oblique asphalt can fit between every sample point. Both
        # visual surfaces are affine planes, so their smallest separation on
        # a convex intersection occurs at one of its polygon vertices.
        # Testing every overlap vertex also accounts for stock versus generated
        # P3Ds having different visible offsets from their WRP object origins.
        lower_polygon = road._surface_polygon(axis)
        for blocker in nearby:
            overlap = road._clip_convex_polygon(
                lower_polygon, blocker.polygon,
            )
            if road._polygon_area(overlap) <= 1.0e-5:
                continue
            for point_x, point_z in overlap:
                bank = (
                    gradient_x * (point_x - x)
                    + gradient_z * (point_z - z)
                )
                gravel_y = visual_base + bank
                paved_y = (
                    road._blocker_height_at(blocker, (point_x, point_z))
                    + blocker.visual_offset
                )
                allowed_lift = min(
                    allowed_lift,
                    paved_y - _PAVED_VISIBLE_CLEARANCE_METRES - gravel_y,
                )
        if not has_exposed:
            continue
        lift = max(0.0, min(needed_lift, allowed_lift, _MAX_LIFT_METRES))
        # Even zero lift can matter: shear itself fixes lateral clipping.
        if lift < 1.0e-5 and abs(gradient_x) + abs(gradient_z) < 1.0e-5:
            continue
        updated[index] = TerrainShearedWorldObject(
            int(obj.object_id), str(obj.model_path),
            x, float(obj.y) + lift, z,
            float(obj.heading_degrees), 0.0,
            gradient_x, gradient_z,
        )
    if all(a is b for a, b in zip(objects, updated)):
        return report
    return replace(report, objects=tuple(updated))
