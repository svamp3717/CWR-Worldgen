# SPDX-License-Identifier: GPL-3.0-or-later
"""Use finer exact road-corridor buckets for vegetation placement queries.

The canonical spatial road index uses 100 m buckets, which is a good general
purpose size but coarse for the thousands of 1-5 m road-clearance queries made
by forest fallback trees and bushes.  Re-bucket the already projected corridor
segments without changing their geometry or intersection predicate.
"""
from __future__ import annotations

from collections import defaultdict
import math
import os
from typing import Any

from . import osm as _osm

_INSTALLED = False
_BASE_PROJECT: Any = None
_REBUCKETED: dict[tuple[int, float], tuple[Any, Any]] = {}
_DEFAULT_BUCKET_METRES = 32.0
_MINIMUM_BUCKET_METRES = 12.0
_MAX_CACHED_INDEXES = 16


def _configured_bucket(source_bucket: float) -> float:
    raw = os.environ.get("CWR_WORLDGEN_FOREST_ROAD_BUCKET", "").strip()
    if raw:
        try:
            requested = float(raw)
        except ValueError:
            requested = _DEFAULT_BUCKET_METRES
    else:
        requested = _DEFAULT_BUCKET_METRES
    if not math.isfinite(requested) or requested <= 0.0:
        requested = _DEFAULT_BUCKET_METRES
    return min(float(source_bucket), max(_MINIMUM_BUCKET_METRES, requested))


def _rebucket_index(
    source: Any,
    bucket_size: float,
):
    """Return an exact ``IndexedRoadCorridors`` copy with smaller buckets."""

    if not isinstance(source, _osm.IndexedRoadCorridors):
        return source
    bucket_size = max(_MINIMUM_BUCKET_METRES, float(bucket_size))
    if bucket_size >= float(source.bucket_size) - 1.0e-9:
        return source

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (start, end, radius) in enumerate(source.corridors):
        radius = max(0.0, float(radius))
        minimum_x = min(float(start[0]), float(end[0])) - radius
        minimum_z = min(float(start[1]), float(end[1])) - radius
        maximum_x = max(float(start[0]), float(end[0])) + radius
        maximum_z = max(float(start[1]), float(end[1])) + radius
        for bz in range(
            math.floor(minimum_z / bucket_size),
            math.floor(maximum_z / bucket_size) + 1,
        ):
            for bx in range(
                math.floor(minimum_x / bucket_size),
                math.floor(maximum_x / bucket_size) + 1,
            ):
                buckets[(bx, bz)].append(index)

    # ``index`` grows monotonically, so each bucket is already sorted and no
    # temporary set/sort pass is needed while freezing it.
    return _osm.IndexedRoadCorridors(
        source.corridors,
        bucket_size,
        {key: tuple(values) for key, values in buckets.items()},
    )


def _project_road_corridors(*args, **kwargs):
    source = _BASE_PROJECT(*args, **kwargs)
    if not isinstance(source, _osm.IndexedRoadCorridors) or not source.corridors:
        return source

    bucket_size = _configured_bucket(float(source.bucket_size))
    if bucket_size >= float(source.bucket_size) - 1.0e-9:
        return source

    key = (id(source), bucket_size)
    cached = _REBUCKETED.get(key)
    if cached is not None and cached[0] is source:
        return cached[1]

    result = _rebucket_index(source, bucket_size)
    _REBUCKETED[key] = (source, result)
    while len(_REBUCKETED) > _MAX_CACHED_INDEXES:
        _REBUCKETED.pop(next(iter(_REBUCKETED)))
    return result


def install_forest_corridor_performance_policy() -> None:
    global _INSTALLED, _BASE_PROJECT
    if _INSTALLED:
        return
    _BASE_PROJECT = _osm.project_road_corridors
    _osm.project_road_corridors = _project_road_corridors
    _INSTALLED = True
