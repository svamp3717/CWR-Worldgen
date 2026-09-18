# SPDX-License-Identifier: GPL-3.0-or-later
"""Bound final bridge approach filler searches with a road-centre spatial index.

The historical final-alignment pass searched every fitted road object for each
bridge endpoint, then searched every fitted road object again for every candidate
while checking whether the road continued away from the bridge.  On million-piece
worlds that turns a tiny abutment repair into a quadratic global scan.

Keep the exact endpoint/heading/3D join predicates, but use road-object centres as
a conservative broad phase.  Every straight stock-road endpoint lies within half
its nominal model length of the object centre, so querying by
``half_length + exact_tolerance`` cannot discard a historically eligible object.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

from . import bridge_final_alignment_policy as _alignment

_BUCKET_METRES = 32.0
_DIAGNOSTIC_OBJECT_THRESHOLD = 100_000
_INSTALLED = False


@dataclass(slots=True)
class _RoadCentreIndex:
    buckets: dict[tuple[int, int], list[Any]]
    maximum_half_length: float
    indexed_objects: int

    @classmethod
    def build(cls, road_objects: Sequence[Any]) -> "_RoadCentreIndex":
        buckets: dict[tuple[int, int], list[Any]] = {}
        maximum_half_length = 0.0
        indexed = 0
        for obj in road_objects:
            length = _alignment._straight_road_nominal_length(obj.model_path)
            if length is None or length <= 1.0e-9:
                continue
            maximum_half_length = max(maximum_half_length, float(length) * 0.5)
            key = (
                math.floor(float(obj.x) / _BUCKET_METRES),
                math.floor(float(obj.z) / _BUCKET_METRES),
            )
            buckets.setdefault(key, []).append(obj)
            indexed += 1
        return cls(buckets, maximum_half_length, indexed)

    def query(self, point: tuple[float, float], radius: float):
        radius = max(0.0, float(radius))
        x, z = float(point[0]), float(point[1])
        bx0 = math.floor((x - radius) / _BUCKET_METRES)
        bx1 = math.floor((x + radius) / _BUCKET_METRES)
        bz0 = math.floor((z - radius) / _BUCKET_METRES)
        bz1 = math.floor((z + radius) / _BUCKET_METRES)
        for bx in range(bx0, bx1 + 1):
            for bz in range(bz0, bz1 + 1):
                yield from self.buckets.get((bx, bz), ())


def _indexed_road_has_outward_neighbour(
    obj,
    bridge_point,
    road_index: _RoadCentreIndex,
) -> bool:
    """Exact historical continuation test after a conservative centre query."""
    endpoints = _alignment._road_endpoints(obj)
    if endpoints is None:
        return False
    far = max(
        endpoints,
        key=lambda point: math.dist(
            (point[0], point[2]),
            (float(bridge_point[0]), float(bridge_point[1])),
        ),
    )
    radius = (
        road_index.maximum_half_length
        + float(_alignment._APPROACH_CHAIN_JOIN_TOLERANCE_METRES)
    )
    for other in road_index.query((far[0], far[2]), radius):
        if int(other.object_id) == int(obj.object_id):
            continue
        other_endpoints = _alignment._road_endpoints(other)
        if other_endpoints is None:
            continue
        if (
            _alignment._axial_heading_difference(
                obj.heading_degrees,
                other.heading_degrees,
            )
            > _alignment._APPROACH_FILL_HEADING_TOLERANCE_DEGREES
        ):
            continue
        if (
            min(math.dist(far, point) for point in other_endpoints)
            <= _alignment._APPROACH_CHAIN_JOIN_TOLERANCE_METRES
        ):
            return True
    return False


def _indexed_approach_candidate(
    bridge_point,
    bridge_heading,
    road_index: _RoadCentreIndex,
):
    """Return exactly the historical best candidate without the global scans."""
    radius = (
        road_index.maximum_half_length
        + float(_alignment._APPROACH_FILL_MAX_GAP_METRES)
    )
    candidates = []
    for obj in road_index.query(
        (float(bridge_point[0]), float(bridge_point[1])),
        radius,
    ):
        if _alignment._cleanup._paved._family(obj.model_path) is None:
            continue
        endpoints = _alignment._road_endpoints(obj)
        if endpoints is None or _alignment._six_metre_sibling(obj.model_path) is None:
            continue
        if (
            _alignment._axial_heading_difference(
                obj.heading_degrees,
                bridge_heading,
            )
            > _alignment._APPROACH_FILL_HEADING_TOLERANCE_DEGREES
        ):
            continue
        near = min(
            endpoints,
            key=lambda point: math.dist(
                (point[0], point[2]),
                (float(bridge_point[0]), float(bridge_point[1])),
            ),
        )
        gap = math.dist(
            (near[0], near[2]),
            (float(bridge_point[0]), float(bridge_point[1])),
        )
        if gap > _alignment._APPROACH_FILL_MAX_GAP_METRES:
            continue
        connected = _indexed_road_has_outward_neighbour(
            obj,
            bridge_point,
            road_index,
        )
        candidates.append(
            (
                0 if connected else 1,
                gap,
                int(obj.object_id),
                obj,
                near,
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:3])
    return candidates[0][3], candidates[0][4], candidates[0][1]


def _fast_add_bridge_approach_fillers(report, spans, elevations, spec):
    """Indexed equivalent of the historical bridge-approach filler pass."""
    if not spans or not getattr(report, "objects", ()):
        return report, 0

    objects = list(report.objects)
    road_objects = tuple(objects)
    diagnostic = len(road_objects) >= _DIAGNOSTIC_OBJECT_THRESHOLD
    if diagnostic:
        print(
            "[bridge-road-alignment] indexing final straight roads for "
            f"{len(spans):,} bridge span(s): {len(road_objects):,} fitted objects",
            flush=True,
        )
    road_index = _RoadCentreIndex.build(road_objects)
    if diagnostic:
        print(
            "[bridge-road-alignment] road index ready: "
            f"{road_index.indexed_objects:,} straight pieces in "
            f"{len(road_index.buckets):,} spatial buckets",
            flush=True,
        )

    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    added = 0
    endpoints_checked = 0

    for span in spans:
        if len(span.points) < 2:
            continue
        start = (float(span.points[0][0]), float(span.points[0][1]))
        end = (float(span.points[-1][0]), float(span.points[-1][1]))
        dx = end[0] - start[0]
        dz = end[1] - start[1]
        length = math.hypot(dx, dz)
        if length <= 0.1:
            continue
        axis = (dx / length, dz / length)
        bridge_heading = math.degrees(math.atan2(axis[0], axis[1])) % 360.0

        for bridge_point, outward in (
            (start, (-axis[0], -axis[1])),
            (end, axis),
        ):
            endpoints_checked += 1
            selected = _indexed_approach_candidate(
                bridge_point,
                bridge_heading,
                road_index,
            )
            if selected is None:
                continue
            candidate, candidate_near, gap = selected
            deck_y = _alignment._bridge_endpoint_deck_height(
                bridge_point,
                outward,
                float(candidate_near[1]),
                elevations,
                spec,
            )
            filler = _alignment._approach_filler(
                next_id,
                bridge_point,
                outward,
                bridge_heading,
                candidate,
                candidate_near,
                gap,
                deck_y,
            )
            if filler is None:
                continue
            objects.append(filler)
            next_id += 1
            added += 1

    if diagnostic:
        print(
            "[bridge-road-alignment] bridge approach search complete: "
            f"{endpoints_checked:,} endpoint(s), {added:,} connector(s) added",
            flush=True,
        )

    if not added:
        return report, 0

    # Preserve the historical contiguous-ID repack exactly.
    first_id = min(int(obj.object_id) for obj in objects)
    renumbered = tuple(
        replace(obj, object_id=first_id + index)
        for index, obj in enumerate(objects)
    )
    return replace(report, objects=renumbered), added


def install_bridge_final_alignment_performance_policy() -> None:
    """Replace only the quadratic final bridge approach scan."""
    global _INSTALLED
    if _INSTALLED:
        return
    _alignment._add_bridge_approach_fillers = _fast_add_bridge_approach_fillers
    _INSTALLED = True
