# SPDX-License-Identifier: GPL-3.0-or-later
"""Experimental join-aware road strokes for the world generation fitter.

Only close, unambiguous, collinear loose ends of compatible roads are joined.
The normal road fitter still selects the actual road models and junctions.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Sequence

from . import playability as _playability

_ENVIRONMENT_FLAG = "CWR_ROAD_PAINT_SNAP_METRES"
_MAXIMUM_REACH_METRES = 3.0
_MINIMUM_SEGMENT_METRES = 2.0
_MINIMUM_FORWARD_COSINE = math.cos(math.radians(20.0))
_MINIMUM_OPPOSITION_COSINE = math.cos(math.radians(25.0))
_ORIGINAL_PROJECT = _playability._paved_junction_augmented_polylines
_INSTALLED = False


def _at_grade(tags) -> bool:
    for key in ("bridge", "tunnel"):
        if str(tags.get(key, "")).casefold() not in ("", "no", "false", "0"):
            return False
    try:
        return int(tags.get("layer", 0)) == 0
    except (TypeError, ValueError):
        return False


def _outgoing(points: Sequence[tuple[float, float]], start: bool):
    end, adjacent = (points[0], points[1]) if start else (points[-1], points[-2])
    dx, dz = end[0] - adjacent[0], end[1] - adjacent[1]
    length = math.hypot(dx, dz)
    if length < _MINIMUM_SEGMENT_METRES:
        return None
    return (dx / length, dz / length), length


def snap_loose_road_ends(
    dataset, polylines: Sequence[Sequence[tuple[float, float]]], spec, *,
    reach_metres: float,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Close only mutual, axis-aligned loose-end gaps, without moving inner nodes.

    Restrict movement to 20% of the adjacent road segment and avoid grade-separated
    or mixed-surface joins. Ambiguous multiway ends are intentionally left alone.
    """
    original = tuple(tuple(line) for line in polylines)
    requested = float(reach_metres)
    if not math.isfinite(requested) or requested <= 0.0 or len(original) != len(dataset.roads):
        return original
    reach = min(_MAXIMUM_REACH_METRES, requested)

    endpoints = []
    buckets = defaultdict(list)
    surface_cache = {}
    for feature_index, (feature, line) in enumerate(zip(dataset.roads, original)):
        if len(line) < 2 or not _at_grade(feature.tags):
            continue
        if not _playability.road_is_supported(
            feature.tags, include_minor=spec.include_minor_roads,
        ):
            continue
        model = _playability.road_model_for_tags(spec, feature.tags)
        if model not in surface_cache:
            surface_cache[model] = _playability.road_model_surface(spec, model)
        surface = surface_cache[model]
        if surface is None:
            continue
        for start in (True, False):
            outgoing = _outgoing(line, start)
            if outgoing is None:
                continue
            direction, segment_length = outgoing
            point = line[0] if start else line[-1]
            if not all(math.isfinite(v) for v in point):
                continue
            key = (math.floor(point[0] / reach), math.floor(point[1] / reach))
            index = len(endpoints)
            endpoints.append((feature_index, start, point, direction, segment_length, surface))
            buckets[key].append(index)

    candidates = defaultdict(list)
    for index, (feature, _, point, outward, length, surface) in enumerate(endpoints):
        bx, bz = math.floor(point[0] / reach), math.floor(point[1] / reach)
        for x in range(bx - 1, bx + 2):
            for z in range(bz - 1, bz + 2):
                for other_index in buckets.get((x, z), ()):
                    if other_index <= index:
                        continue
                    other_feature, _, other_point, other_outward, other_length, other_surface = endpoints[other_index]
                    if other_feature == feature or surface != other_surface:
                        continue
                    dx, dz = other_point[0] - point[0], other_point[1] - point[1]
                    gap = math.hypot(dx, dz)
                    if gap <= 1.0e-8 or gap > min(reach, 0.20 * length, 0.20 * other_length):
                        continue
                    ux, uz = dx / gap, dz / gap
                    if (
                        outward[0] * ux + outward[1] * uz < _MINIMUM_FORWARD_COSINE
                        or -other_outward[0] * ux - other_outward[1] * uz < _MINIMUM_FORWARD_COSINE
                        or outward[0] * other_outward[0] + outward[1] * other_outward[1] > -_MINIMUM_OPPOSITION_COSINE
                    ):
                        continue
                    candidates[index].append(other_index)
                    candidates[other_index].append(index)

    if not candidates:
        return original
    corrected = [list(line) for line in original]
    for index, matches in sorted(candidates.items()):
        if len(matches) != 1:
            continue
        other = matches[0]
        if other <= index or candidates.get(other) != [index]:
            continue
        first_feature, first_start, first_point, *_ = endpoints[index]
        other_feature, other_start, other_point, *_ = endpoints[other]
        shared = (
            (first_point[0] + other_point[0]) * 0.5,
            (first_point[1] + other_point[1]) * 0.5,
        )
        corrected[first_feature][0 if first_start else -1] = shared
        corrected[other_feature][0 if other_start else -1] = shared
    return tuple(tuple(line) for line in corrected)


def _project_with_joined_strokes(dataset, projection, spec):
    original = _ORIGINAL_PROJECT(dataset, projection, spec)
    requested = os.environ.get(_ENVIRONMENT_FLAG, "").strip()
    if not requested:
        return original
    try:
        reach = float(requested)
    except ValueError:
        return original
    if not math.isfinite(reach) or reach <= 0.0:
        return original
    return snap_loose_road_ends(dataset, original, spec, reach_metres=reach)


def install_road_paint_snap_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _playability._paved_junction_augmented_polylines = _project_with_joined_strokes
    _INSTALLED = True
