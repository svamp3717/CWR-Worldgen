# SPDX-License-Identifier: GPL-3.0-or-later
"""Join-aware road strokes at small non-paved town T-junctions.

This is an opt-in geometry prepass. Actual model fitting and junction rendering
remain owned by the existing world generation pipeline.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Sequence

from . import playability as _p


_BUCKET_METRES = 32.0
_MAXIMUM_REACH_METRES = 3.0
_MINIMUM_APPROACH_METRES = 2.0
_MAXIMUM_PARALLEL_COSINE = math.cos(math.radians(55.0))
_MINIMUM_FORWARD_COSINE = math.cos(math.radians(25.0))
_MAXIMUM_MOVEMENT_FRACTION = 0.20
_MAX_BUCKETS_PER_SEGMENT = 256


def _surface(spec, tags, cache):
    model = _p.road_model_for_tags(spec, tags)
    if model not in cache:
        cache[model] = _p.road_model_surface(spec, model)
    return cache[model]


def _supported(spec, tags) -> bool:
    if not _p.road_is_supported(tags, include_minor=spec.include_minor_roads):
        return False
    if str(tags.get("highway", "")).casefold() == "raceway":
        return False
    if _p._road_is_grade_separated_for_synthetic_junction(tags):
        return False
    return True


def _segment_nearest(point, start, end):
    dx, dz = end[0] - start[0], end[1] - start[1]
    length_squared = dx * dx + dz * dz
    if length_squared < 1.0:
        return None
    fraction = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dz) / length_squared
    if not (0.0 < fraction < 1.0):
        return None
    near = (start[0] + dx * fraction, start[1] + dz * fraction)
    length = math.sqrt(length_squared)
    return fraction, near, length, (dx / length, dz / length)


def snap_town_road_mouths(
    dataset, polylines: Sequence[Sequence[tuple[float, float]]], spec, *,
    reach_metres: float,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Join only isolated dirt/gravel street endpoints to a transverse road.

    Preserve endpoints already connected to another road. Never invent paved
    intersections, move an interior vertex, or join across surface families.
    """
    original = tuple(tuple(line) for line in polylines)
    reach = float(reach_metres)
    if not math.isfinite(reach) or reach <= 0.0 or len(original) != len(dataset.roads):
        return original
    reach = min(_MAXIMUM_REACH_METRES, reach)

    surface_cache = {}
    surfaces = []
    vertex_owners = defaultdict(set)
    for index, (feature, line) in enumerate(zip(dataset.roads, original)):
        surface = (
            _surface(spec, feature.tags, surface_cache)
            if len(line) >= 2 and _supported(spec, feature.tags) else None
        )
        surfaces.append(surface)
        if surface is not None:
            for point in line:
                vertex_owners[_p._road_node_key(point)].add(index)

    segments = []
    buckets = defaultdict(list)
    oversized = []
    for feature, (line, surface) in enumerate(zip(original, surfaces)):
        if surface not in ("gravel", "dirt"):
            continue
        for segment, (start, end) in enumerate(zip(line, line[1:])):
            if not all(math.isfinite(v) for v in (*start, *end)):
                continue
            if math.dist(start, end) < 4.0:
                continue
            index = len(segments)
            segments.append((feature, segment, start, end, surface))
            bx0 = math.floor((min(start[0], end[0]) - reach) / _BUCKET_METRES)
            bx1 = math.floor((max(start[0], end[0]) + reach) / _BUCKET_METRES)
            bz0 = math.floor((min(start[1], end[1]) - reach) / _BUCKET_METRES)
            bz1 = math.floor((max(start[1], end[1]) + reach) / _BUCKET_METRES)
            if (bx1 - bx0 + 1) * (bz1 - bz0 + 1) > _MAX_BUCKETS_PER_SEGMENT:
                oversized.append(index)
            else:
                for bx in range(bx0, bx1 + 1):
                    for bz in range(bz0, bz1 + 1):
                        buckets[(bx, bz)].append(index)

    proposals = {}
    for feature, (line, surface) in enumerate(zip(original, surfaces)):
        if surface not in ("dirt", "gravel") or len(line) < 2:
            continue
        for is_start in (True, False):
            vertex = 0 if is_start else len(line) - 1
            point = line[vertex]
            neighbor = line[1] if is_start else line[-2]
            if not all(math.isfinite(v) for v in (*point, *neighbor)):
                continue
            if len(vertex_owners[_p._road_node_key(point)]) != 1:
                continue
            dx, dz = point[0] - neighbor[0], point[1] - neighbor[1]
            length = math.hypot(dx, dz)
            if length < _MINIMUM_APPROACH_METRES:
                continue
            outward = dx / length, dz / length
            bx, bz = math.floor(point[0] / _BUCKET_METRES), math.floor(point[1] / _BUCKET_METRES)
            nearby = set(oversized)
            for x in range(bx - 1, bx + 2):
                for z in range(bz - 1, bz + 2):
                    nearby.update(buckets.get((x, z), ()))
            matches = []
            for candidate in sorted(nearby):
                target_feature, target_segment, start, end, target_surface = segments[candidate]
                if target_feature == feature or target_surface != surface:
                    continue
                projected = _segment_nearest(point, start, end)
                if projected is None:
                    continue
                fraction, nearest, segment_length, axis = projected
                distance = math.dist(point, nearest)
                if distance > min(reach, length * _MAXIMUM_MOVEMENT_FRACTION):
                    continue
                margin = max(0.2, min(2.0, segment_length * 0.10))
                if min(fraction, 1.0 - fraction) * segment_length < margin:
                    continue
                if abs(outward[0] * axis[0] + outward[1] * axis[1]) > _MAXIMUM_PARALLEL_COSINE:
                    continue
                if distance > 1.0e-8:
                    toward = ((nearest[0] - point[0]) / distance,
                              (nearest[1] - point[1]) / distance)
                    if outward[0] * toward[0] + outward[1] * toward[1] < _MINIMUM_FORWARD_COSINE:
                        continue
                matches.append((target_feature, target_segment, fraction, nearest))
            if len(matches) == 1:
                proposals[(feature, vertex)] = matches[0]

    if not proposals:
        return original

    # Two nearby branches aimed at one road segment may represent a crossroads
    # or parallel access tracks. Do not turn an ambiguous cluster into a new hub.
    by_target = defaultdict(list)
    for source, (target, segment, _, point) in proposals.items():
        by_target[(target, segment)].append((source, point))
    excluded = set()
    for choices in by_target.values():
        for index, (source, point) in enumerate(choices):
            for other_source, other_point in choices[index + 1:]:
                if math.dist(point, other_point) <= 2.0 * reach:
                    excluded.add(source)
                    excluded.add(other_source)

    corrected = [list(line) for line in original]
    insertions = defaultdict(list)
    changed = False
    for source, (target, segment, fraction, point) in sorted(proposals.items()):
        if source in excluded:
            continue
        feature, vertex = source
        corrected[feature][vertex] = point
        insertions[(target, segment)].append((fraction, point))
        changed = True

    if not changed:
        return original
    for target in sorted({key[0] for key in insertions}):
        original_line = original[target]
        rebuilt = [corrected[target][0]]
        for segment in range(len(original_line) - 1):
            for _, point in sorted(insertions.get((target, segment), ())):
                if math.dist(rebuilt[-1], point) > 0.05:
                    rebuilt.append(point)
            end = corrected[target][segment + 1]
            if math.dist(rebuilt[-1], end) > 0.05:
                rebuilt.append(end)
        corrected[target] = rebuilt
    return tuple(tuple(line) for line in corrected)
