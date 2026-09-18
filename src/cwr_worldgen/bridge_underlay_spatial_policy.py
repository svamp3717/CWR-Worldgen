# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatial candidate pruning for bridge road-underlay cleanup.

The bridge cleanup's exact geometric predicates are intentionally unchanged.
This layer only prevents obviously distant road objects from testing every bridge
span, and prevents terminal-underlay checks from scanning the full road object
collection for each synthesized segment.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import math
from typing import Any, Sequence

from . import bridge_underlay_cleanup_policy as _cleanup

_INSTALLED = False
_ORIGINAL_REMOVE_BRIDGE_UNDERLAYS: Any = None
_ORIGINAL_ADD_TERMINAL_UNDERLAYS: Any = None

_SPAN_BUCKET_METRES = 64.0
_OBJECT_BUCKET_METRES = 16.0
_MATCH_RADIUS_METRES = 4.0


def _bucket_range(minimum: float, maximum: float, bucket_size: float) -> range:
    return range(
        math.floor(float(minimum) / bucket_size),
        math.floor(float(maximum) / bucket_size) + 1,
    )


def _expanded_bounds(
    points: Sequence[tuple[float, float]],
    padding: float,
) -> tuple[float, float, float, float] | None:
    if len(points) < 2:
        return None
    xs = tuple(float(point[0]) for point in points)
    zs = tuple(float(point[1]) for point in points)
    padding = max(0.0, float(padding))
    return (
        min(xs) - padding,
        min(zs) - padding,
        max(xs) + padding,
        max(zs) + padding,
    )


@dataclass(slots=True)
class _BridgeSpanIndex:
    spans: tuple[Any, ...]
    bucket_size: float
    buckets: dict[tuple[int, int], tuple[int, ...]]

    @classmethod
    def create(
        cls,
        spans: Sequence[Any],
        *,
        bucket_size: float = _SPAN_BUCKET_METRES,
    ) -> "_BridgeSpanIndex":
        frozen = tuple(spans)
        bucket_size = max(8.0, float(bucket_size))
        mutable: dict[tuple[int, int], set[int]] = defaultdict(set)

        for index, span in enumerate(frozen):
            corridor = max(1.25, float(span.road_width) * 0.40)
            point_sets = [span.points]
            if (
                span.source_points
                and span.source_end_measure > span.source_start_measure
            ):
                point_sets.append(span.source_points)

            for points in point_sets:
                bounds = _expanded_bounds(points, corridor)
                if bounds is None:
                    continue
                minimum_x, minimum_z, maximum_x, maximum_z = bounds
                for bz in _bucket_range(minimum_z, maximum_z, bucket_size):
                    for bx in _bucket_range(minimum_x, maximum_x, bucket_size):
                        mutable[(bx, bz)].add(index)

        return cls(
            frozen,
            bucket_size,
            {
                key: tuple(sorted(indices))
                for key, indices in mutable.items()
            },
        )

    def candidates(self, x: float, z: float) -> tuple[Any, ...]:
        key = (
            math.floor(float(x) / self.bucket_size),
            math.floor(float(z) / self.bucket_size),
        )
        indices = self.buckets.get(key, ())
        return tuple(self.spans[index] for index in indices)


@dataclass(slots=True)
class _ObjectPointIndex:
    objects: list[Any]
    bucket_size: float
    buckets: dict[tuple[int, int], list[int]]

    @classmethod
    def create(
        cls,
        objects: list[Any],
        *,
        bucket_size: float = _OBJECT_BUCKET_METRES,
    ) -> "_ObjectPointIndex":
        result = cls(objects, max(8.0, float(bucket_size)), defaultdict(list))
        for index, obj in enumerate(objects):
            result._insert(index, obj)
        return result

    def _insert(self, index: int, obj: Any) -> None:
        key = (
            math.floor(float(obj.x) / self.bucket_size),
            math.floor(float(obj.z) / self.bucket_size),
        )
        self.buckets[key].append(index)

    def append(self, obj: Any) -> None:
        index = len(self.objects)
        self.objects.append(obj)
        self._insert(index, obj)

    def candidate_indices(
        self,
        x: float,
        z: float,
        radius: float,
    ) -> tuple[int, ...]:
        radius = max(0.0, float(radius))
        minimum_x = float(x) - radius
        minimum_z = float(z) - radius
        maximum_x = float(x) + radius
        maximum_z = float(z) + radius
        result: list[int] = []
        for bz in _bucket_range(minimum_z, maximum_z, self.bucket_size):
            for bx in _bucket_range(minimum_x, maximum_x, self.bucket_size):
                result.extend(self.buckets.get((bx, bz), ()))
        if len(result) <= 1:
            return tuple(result)
        # Objects live in exactly one point bucket. Sorting restores source order
        # when a radius query touches multiple neighboring buckets.
        result.sort()
        return tuple(result)


def _remove_bridge_underlays(report, spans):
    """Exact bridge-underlay removal after conservative spatial pruning."""

    if not spans or not getattr(report, "objects", ()):
        return report, 0

    index = _BridgeSpanIndex.create(spans)
    kept = []
    removed = 0
    for obj in report.objects:
        candidates = index.candidates(float(obj.x), float(obj.z))
        if candidates and _cleanup._road_object_under_bridge(obj, candidates):
            removed += 1
        else:
            kept.append(obj)

    if not removed:
        return report, 0
    return replace(report, objects=tuple(kept)), removed


def _has_matching_underlay(
    index: _ObjectPointIndex,
    start: tuple[float, float],
    end: tuple[float, float],
    model_path: str,
) -> bool:
    midpoint = (
        (float(start[0]) + float(end[0])) * 0.5,
        (float(start[1]) + float(end[1])) * 0.5,
    )
    heading = math.degrees(
        math.atan2(
            float(end[0]) - float(start[0]),
            float(end[1]) - float(start[1]),
        )
    ) % 360.0
    target_family = _cleanup._paved._family(model_path)

    for object_index in index.candidate_indices(
        midpoint[0],
        midpoint[1],
        _MATCH_RADIUS_METRES,
    ):
        obj = index.objects[object_index]
        if _cleanup._paved._family(obj.model_path) != target_family:
            continue
        if math.dist((float(obj.x), float(obj.z)), midpoint) > _MATCH_RADIUS_METRES:
            continue
        if (
            _cleanup._undirected_heading_difference(
                obj.heading_degrees,
                heading,
            )
            <= 12.0
        ):
            return True
    return False


def _add_terminal_underlays(report, spans, elevations, spec):
    """Synthesize missing terminal pieces using a local road-object point index."""

    if not spans:
        return report, 0

    objects = list(getattr(report, "objects", ()))
    index = _ObjectPointIndex.create(objects)
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    added = 0

    for span in spans:
        model = _cleanup._terminal_model(span.road_model_path)
        if model is None:
            continue
        for start, end in _cleanup._terminal_segments(span):
            if _has_matching_underlay(index, start, end, model):
                continue
            obj = _cleanup._terminal_underlay_object(
                next_id,
                model,
                span,
                start,
                end,
                elevations,
                spec,
            )
            index.append(obj)
            next_id += 1
            added += 1

    if not added:
        return report, 0
    return replace(report, objects=tuple(objects)), added


def install_bridge_underlay_spatial_policy() -> None:
    """Install spatial pruning behind the existing exact cleanup predicates."""

    global _INSTALLED
    global _ORIGINAL_REMOVE_BRIDGE_UNDERLAYS, _ORIGINAL_ADD_TERMINAL_UNDERLAYS
    if _INSTALLED:
        return

    _ORIGINAL_REMOVE_BRIDGE_UNDERLAYS = _cleanup._remove_bridge_underlays
    _ORIGINAL_ADD_TERMINAL_UNDERLAYS = _cleanup._add_terminal_underlays
    _cleanup._remove_bridge_underlays = _remove_bridge_underlays
    _cleanup._add_terminal_underlays = _add_terminal_underlays
    _INSTALLED = True
