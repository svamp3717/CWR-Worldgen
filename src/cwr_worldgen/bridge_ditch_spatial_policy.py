# SPDX-License-Identifier: GPL-3.0-or-later
"""Spatially bound explicit-bridge ditch-crossing checks.

The historical bridge classifier compares every segment of an explicit bridge
road with every segment of every mapped watercourse.  On large extracts that
turns a single long bridge-tagged way into an O(road_segments * all_watercourse_segments)
scan.  Keep the exact segment-distance predicate, but index projected
watercourse segments once per dataset/projection and only test nearby buckets.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from . import osm as _osm
from . import playability as _playability
from . import terrain_solver as _terrain


_BUCKET_SIZE_METRES = 128.0
_CACHE_LIMIT = 4
_INSTALLED = False
_ORIGINAL_DITCH_TEST = _osm.road_bridge_crosses_ditch_only

PointXZ = tuple[float, float]


@dataclass(frozen=True, slots=True)
class _WatercourseSegment:
    start: PointXZ
    end: PointXZ
    kind: str


@dataclass(slots=True)
class _WatercourseIndex:
    dataset: Any
    projection: Any
    segments: tuple[_WatercourseSegment, ...]
    buckets: dict[tuple[int, int], tuple[int, ...]]


_INDEX_CACHE: dict[tuple[int, int], _WatercourseIndex] = {}


def _build_index(dataset: Any, projection: Any) -> _WatercourseIndex:
    segments: list[_WatercourseSegment] = []
    mutable: dict[tuple[int, int], list[int]] = {}
    bucket = _BUCKET_SIZE_METRES

    for feature in getattr(dataset, "watercourses", ()):
        points = tuple(projection.to_world(point) for point in getattr(feature, "points", ()))
        if len(points) < 2:
            continue
        kind = str(getattr(feature, "tags", {}).get("waterway", "stream")).strip().casefold()
        for start, end in zip(points, points[1:]):
            index = len(segments)
            segments.append(_WatercourseSegment(start, end, kind))
            min_x = min(float(start[0]), float(end[0]))
            min_z = min(float(start[1]), float(end[1]))
            max_x = max(float(start[0]), float(end[0]))
            max_z = max(float(start[1]), float(end[1]))
            for bz in range(math.floor(min_z / bucket), math.floor(max_z / bucket) + 1):
                for bx in range(math.floor(min_x / bucket), math.floor(max_x / bucket) + 1):
                    mutable.setdefault((bx, bz), []).append(index)

    return _WatercourseIndex(
        dataset=dataset,
        projection=projection,
        segments=tuple(segments),
        buckets={key: tuple(sorted(set(values))) for key, values in mutable.items()},
    )


def _watercourse_index(dataset: Any, projection: Any) -> _WatercourseIndex:
    key = (id(dataset), id(projection))
    cached = _INDEX_CACHE.get(key)
    if cached is not None and cached.dataset is dataset and cached.projection is projection:
        return cached
    if len(_INDEX_CACHE) >= _CACHE_LIMIT:
        _INDEX_CACHE.clear()
    result = _build_index(dataset, projection)
    _INDEX_CACHE[key] = result
    return result


def _candidate_indices(
    index: _WatercourseIndex,
    start: PointXZ,
    end: PointXZ,
    tolerance: float,
) -> tuple[int, ...]:
    bucket = _BUCKET_SIZE_METRES
    minimum_x = min(float(start[0]), float(end[0])) - tolerance
    minimum_z = min(float(start[1]), float(end[1])) - tolerance
    maximum_x = max(float(start[0]), float(end[0])) + tolerance
    maximum_z = max(float(start[1]), float(end[1])) + tolerance
    found: set[int] = set()
    for bz in range(math.floor(minimum_z / bucket), math.floor(maximum_z / bucket) + 1):
        for bx in range(math.floor(minimum_x / bucket), math.floor(maximum_x / bucket) + 1):
            found.update(index.buckets.get((bx, bz), ()))
    return tuple(sorted(found))


def _indexed_road_bridge_crosses_ditch_only(
    feature: Any,
    dataset: Any,
    projection: Any,
    *,
    tolerance_metres: float = _osm.BRIDGE_DITCH_CROSSING_TOLERANCE_METRES,
) -> bool:
    tags = getattr(feature, "tags", {}) or {}
    bridge = str(tags.get("bridge", "")).strip().casefold()
    explicit = (
        bridge not in {"", "no", "false", "0", "none"}
        or str(tags.get("man_made", "")).strip().casefold() == "bridge"
        or str(tags.get("special", "")).strip().casefold() == "bridge"
    )
    feature_points = getattr(feature, "points", ())
    if not explicit or len(feature_points) < 2:
        return False

    points = tuple(projection.to_world(point) for point in feature_points)
    index = _watercourse_index(dataset, projection)
    if not index.segments:
        return False

    tolerance = max(0.0, float(tolerance_metres))
    tolerance_squared = tolerance * tolerance
    found_ditch = False

    for road_start, road_end in zip(points, points[1:]):
        for candidate_index in _candidate_indices(index, road_start, road_end, tolerance):
            water = index.segments[candidate_index]
            # Buckets are only a conservative broad phase. Preserve the exact
            # historical segment-distance predicate for the final decision.
            if _osm._segment_distance_squared(
                road_start,
                road_end,
                water.start,
                water.end,
            ) > tolerance_squared:
                continue
            if water.kind != "ditch":
                # Historical semantics are false if any crossed mapped
                # watercourse is not a ditch. We can therefore stop here.
                return False
            found_ditch = True

    return found_ditch


def install_bridge_ditch_spatial_policy() -> None:
    """Install indexed ditch classification into every imported live binding."""
    global _INSTALLED
    if _INSTALLED:
        return
    _osm.road_bridge_crosses_ditch_only = _indexed_road_bridge_crosses_ditch_only
    _terrain.road_bridge_crosses_ditch_only = _indexed_road_bridge_crosses_ditch_only
    _playability.road_bridge_crosses_ditch_only = _indexed_road_bridge_crosses_ditch_only
    _INSTALLED = True
