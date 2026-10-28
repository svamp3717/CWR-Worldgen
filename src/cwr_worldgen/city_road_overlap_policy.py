# SPDX-License-Identifier: GPL-3.0-or-later
"""Conservative overlap prevention for dense projected streets and town caps."""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import replace
from functools import wraps
import math

from . import generator as _generator
from . import playability as _p
from . import final_road_dedup_policy as _final

_MAXIMUM_CENTRELINE_SEPARATION_METRES = 0.65
_ENDPOINT_CELL_METRES = 1.0
_MAXIMUM_CAP_LATERAL_METRES = 0.20
_CAP_BUCKET_METRES = 12.0
_ORIGINAL_FIT = None
_INSTALLED = False


def _road_identity(tags):
    """Never merge different lanes, road classes, elevations or surface tags."""
    if _p._road_is_grade_separated_for_synthetic_junction(tags):
        return None
    return tuple(
        str(tags.get(key, "")).strip().casefold()
        for key in ("highway", "surface", "oneway", "lanes", "width",
                    "junction", "access", "bridge", "tunnel", "layer")
    )


def _samples(points):
    cumulative = [0.0]
    for start, end in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + math.dist(start, end))
    total = cumulative[-1]
    if total < 2.0:
        return total, ()
    count = max(8, min(128, math.ceil(total / 4.0)))
    result = []
    for index in range(count + 1):
        distance = total * index / count
        segment = min(len(points) - 2, max(0, bisect_left(cumulative, distance, lo=1) - 1))
        first, last = cumulative[segment:segment + 2]
        fraction = 0.0 if last - first < 1.0e-9 else (distance - first) / (last - first)
        result.append((
            points[segment][0] + (points[segment + 1][0] - points[segment][0]) * fraction,
            points[segment][1] + (points[segment + 1][1] - points[segment][1]) * fraction,
        ))
    return total, tuple(result)


def suppress_coincident_street_strokes(dataset, polylines, spec, *, tolerance_metres=0.65):
    """Do not fit duplicate near-identical whole streets twice.

    Only whole-feature, same-tagged, co-level traces whose *both* ends and
    uniformly sampled centreline agree can be removed. Partial overlaps,
    divided roads, distinct street classes and unshared branch nodes stay.
    """
    original = tuple(tuple(line) for line in polylines)
    tolerance = float(tolerance_metres)
    if (len(original) != len(dataset.roads)
            or not math.isfinite(tolerance) or tolerance <= 0.0):
        return original
    tolerance = min(_MAXIMUM_CENTRELINE_SEPARATION_METRES, tolerance)
    candidates = []
    cache = {}
    owners = defaultdict(set)
    for index, (feature, line) in enumerate(zip(dataset.roads, original)):
        identity = _road_identity(feature.tags)
        if (len(line) < 2 or identity is None
                or not _p.road_is_supported(feature.tags, include_minor=spec.include_minor_roads)
                or not all(math.isfinite(v) for point in line for v in point)):
            continue
        model = _p.road_model_for_tags(spec, feature.tags)
        if model not in cache:
            cache[model] = _p.road_model_surface(spec, model)
        surface = cache[model]
        if surface is None:
            continue
        length, samples = _samples(line)
        if not samples:
            continue
        candidates.append((index, identity, surface, length, samples))
        for point in line:
            owners[_p._road_node_key(point)].add(index)

    # Prefer a better-vertex-resolved trace, then the first feature. This also
    # retains junction vertices when one duplicate has an explicit branch.
    candidates.sort(key=lambda entry: (-len(original[entry[0]]), entry[0]))
    by_endpoint = defaultdict(list)
    dropped = set()

    def bucket(point):
        return (math.floor(point[0] / _ENDPOINT_CELL_METRES),
                math.floor(point[1] / _ENDPOINT_CELL_METRES))

    for index, identity, surface, length, samples in candidates:
        source = original[index]
        nearby = set()
        bx, bz = bucket(source[0])
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nearby.update(by_endpoint.get((bx + dx, bz + dz), ()))
        duplicate = False
        for other_index, other_identity, other_surface, other_length, other_samples in sorted(nearby, key=lambda item: item[0]):
            if identity != other_identity or surface != other_surface:
                continue
            other = original[other_index]
            reverse = math.dist(source[0], other[-1]) < math.dist(source[0], other[0])
            if (
                math.dist(source[0], other[-1] if reverse else other[0]) > tolerance
                or math.dist(source[-1], other[0] if reverse else other[-1]) > tolerance
                or abs(length - other_length) > min(1.0, 0.02 * min(length, other_length))
            ):
                continue
            other_order = other_samples[::-1] if reverse else other_samples
            if len(samples) != len(other_order):
                # Sample counts are tied to length and may differ by one near a
                # 4 m boundary. Compare both at equal normalized fractions.
                _, other_order = _samples_at_count(other, len(samples) - 1)
                if reverse:
                    other_order = other_order[::-1]
            if any(math.dist(p, q) > tolerance for p, q in zip(samples, other_order)):
                continue
            # Preserve any special intermediate vertex touched by a third way
            # if the retained trace has no *exact* corresponding junction node.
            retained_keys = {_p._road_node_key(point) for point in other}
            unsafe = any(
                (owners[_p._road_node_key(point)] - {index, other_index})
                and _p._road_node_key(point) not in retained_keys
                for point in source
            )
            if unsafe:
                continue
            duplicate = True
            break
        if duplicate:
            dropped.add(index)
        else:
            entry = (index, identity, surface, length, samples)
            by_endpoint[bucket(source[0])].append(entry)
            by_endpoint[bucket(source[-1])].append(entry)
    if not dropped:
        return original
    return tuple(() if index in dropped else line for index, line in enumerate(original))


def _samples_at_count(points, count):
    distances = [0.0]
    for start, end in zip(points, points[1:]):
        distances.append(distances[-1] + math.dist(start, end))
    total = distances[-1]
    result = []
    for index in range(count + 1):
        distance = total * index / count
        segment = min(len(points) - 2, max(0, bisect_left(distances, distance, lo=1) - 1))
        first, last = distances[segment:segment + 2]
        fraction = (distance - first) / (last - first) if last > first + 1.0e-9 else 0.0
        a, b = points[segment:segment + 2]
        result.append((a[0] + (b[0] - a[0]) * fraction,
                       a[1] + (b[1] - a[1]) * fraction))
    return total, tuple(result)


def remove_duplicate_town_cap_slabs(report, spec):
    """Discard only near-identical dirt/gravel pieces already covered by a cap.

    Preserve actual approach overlap, junction caps and vertically separated
    streets. Existing final road deduplication handles paved roads separately.
    """
    count = min(max(0, int(report.junction_cap_objects)), len(report.objects))
    if count == 0:
        return report
    buckets = defaultdict(list)
    for index in range(count):
        obj = report.objects[index]
        axis = _final._road_axis(obj, index, spec, junction_cap=True)
        if (axis is None or axis.family not in ("ces", "gravel")
                or axis.curved_model or axis.length > 8.0):
            continue
        key = (math.floor(obj.x / _CAP_BUCKET_METRES),
               math.floor(obj.z / _CAP_BUCKET_METRES))
        buckets[key].append(axis)

    if not buckets:
        return report
    removed = set()
    cosine = math.cos(math.radians(3.0))
    for index in range(count, len(report.objects)):
        obj = report.objects[index]
        axis = _final._road_axis(obj, index, spec)
        if (axis is None or axis.family not in ("ces", "gravel")
                or axis.curved_model or axis.length > 8.0):
            continue
        bx = math.floor(obj.x / _CAP_BUCKET_METRES)
        bz = math.floor(obj.z / _CAP_BUCKET_METRES)
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for cap in buckets.get((bx + dx, bz + dz), ()):
                    if (cap.family != axis.family
                            or abs(cap.elevation - axis.elevation) > 0.20
                            or abs(cap.half_width - axis.half_width) > 0.05
                            or abs(cap.ux * axis.ux + cap.uz * axis.uz) < cosine):
                        continue
                    lateral = max(
                        abs((point[0] - cap.start[0]) * cap.uz -
                            (point[1] - cap.start[1]) * cap.ux)
                        for point in (axis.start, axis.end)
                    )
                    if lateral > _MAXIMUM_CAP_LATERAL_METRES:
                        continue
                    projections = (
                        (point[0] - cap.start[0]) * cap.ux
                        + (point[1] - cap.start[1]) * cap.uz
                        for point in (axis.start, axis.end)
                    )
                    low, high = sorted(projections)
                    # Require complete containment, not simply large footprint
                    # overlap. Removing a partly exposed approach creates gaps.
                    if low >= -0.05 and high <= cap.length + 0.05:
                        removed.add(index)
                        break
                if index in removed:
                    break
            if index in removed:
                break
    if not removed:
        return report
    return replace(
        report,
        objects=tuple(obj for index, obj in enumerate(report.objects)
                      if index not in removed),
    )


def install_city_overlap_policy() -> None:
    global _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects

    @wraps(_ORIGINAL_FIT)
    def fitting_without_duplicate_town_caps(dataset, projection, elevations, spec,
                                            *, starting_id=1, progress_callback=None):
        report = _ORIGINAL_FIT(
            dataset, projection, elevations, spec,
            starting_id=starting_id, progress_callback=progress_callback,
        )
        import os
        value = os.environ.get("CWR_ROAD_PAINT_SNAP_METRES", "").strip()
        try:
            enabled = math.isfinite(float(value)) and float(value) > 0.0
        except ValueError:
            enabled = False
        return remove_duplicate_town_cap_slabs(report, spec) if enabled else report

    _p.fit_road_objects = fitting_without_duplicate_town_caps
    _generator.fit_road_objects = fitting_without_duplicate_town_caps
    _INSTALLED = True
