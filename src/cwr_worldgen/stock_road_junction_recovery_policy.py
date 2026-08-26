# SPDX-License-Identifier: GPL-3.0-or-later
"""Recover native stock junctions from the road objects that were actually fitted.

The primary junction pass reconstructs incidents from source road data. A source
lookup can occasionally miss a cap even though three/four fitted road arms are
unambiguous, leaving the historical six-metre straight rectangle at the node.
This final pass uses the emitted arm geometry as a second source of truth. It is
strictly local: it never moves the junction centre or surrounding objects.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import playability as _p
from . import road_quality_policy as _quality
from . import stock_road_curve_policy as _curve
from . import stock_road_junction_policy as _junction

_SEARCH_RADIUS_METRES = 20.0
_MAXIMUM_NEAR_ENDPOINT_METRES = 4.5
_DUPLICATE_DIRECTION_DEGREES = 12.0
_BUCKET_METRES = 24.0
_ORIGINAL_REPLACE = None
_INSTALLED = False


def _fitted_family(model_path: str) -> str | None:
    family = _junction._family(model_path)
    if family is not None:
        return family
    curve = _curve._curve_match(str(model_path))
    return curve.group("family").casefold() if curve is not None else None


def _object_axis(obj, spec):
    length = _quality._piece_length(obj.model_path, spec.road_segment_length)
    if not math.isfinite(length) or length <= 0.05:
        return None
    return _p._model_axis(obj, length)


def _incident_from_object(node, obj, spec):
    family = _fitted_family(obj.model_path)
    if family is None:
        return None
    axis = _object_axis(obj, spec)
    if axis is None:
        return None
    first, second = axis
    first_distance = math.dist(node, first)
    second_distance = math.dist(node, second)
    if first_distance <= second_distance:
        near, far, near_distance = first, second, first_distance
    else:
        near, far, near_distance = second, first, second_distance
    if near_distance > _MAXIMUM_NEAR_ENDPOINT_METRES:
        return None
    if math.dist(node, far) <= near_distance + 0.50:
        return None
    dx, dz = far[0] - near[0], far[1] - near[1]
    length = math.hypot(dx, dz)
    if length <= 1.0e-9:
        return None
    return (
        near_distance,
        _junction._Incident((dx / length, dz / length), family, obj.model_path),
    )


def _deduplicate_incidents(candidates):
    cosine_limit = math.cos(math.radians(_DUPLICATE_DIRECTION_DEGREES))
    result = []
    for _distance, incident in sorted(candidates, key=lambda item: item[0]):
        duplicate = False
        for existing in result:
            dot = (
                incident.direction[0] * existing.direction[0]
                + incident.direction[1] * existing.direction[1]
            )
            if dot >= cosine_limit and incident.family == existing.family:
                duplicate = True
                break
        if not duplicate:
            result.append(incident)
    return tuple(result)


def _bucket_key(x: float, z: float) -> tuple[int, int]:
    return int(math.floor(x / _BUCKET_METRES)), int(math.floor(z / _BUCKET_METRES))


def _road_buckets(objects):
    buckets = {}
    for obj in objects:
        if _fitted_family(obj.model_path) is None:
            continue
        buckets.setdefault(_bucket_key(obj.x, obj.z), []).append(obj)
    return buckets


def _nearby_objects(node, buckets):
    bx, bz = _bucket_key(*node)
    radius = int(math.ceil(_SEARCH_RADIUS_METRES / _BUCKET_METRES)) + 1
    for dx in range(-radius, radius + 1):
        for dz in range(-radius, radius + 1):
            for obj in buckets.get((bx + dx, bz + dz), ()):
                if math.dist(node, (obj.x, obj.z)) <= _SEARCH_RADIUS_METRES:
                    yield obj


def _native_from_fitted_arms(node, objects, spec):
    candidates = []
    for obj in objects:
        candidate = _incident_from_object(node, obj, spec)
        if candidate is not None:
            candidates.append(candidate)
    incidents = _deduplicate_incidents(candidates)
    if len(incidents) not in {3, 4}:
        return None
    return _junction._native_junction_for_incidents(incidents)


def _recover_remaining_caps(report, elevations, spec):
    count = int(getattr(report, "junction_cap_objects", 0))
    if count <= 0 or not report.objects:
        return report
    objects = list(report.objects)
    branch_objects = tuple(objects[min(count, len(objects)):])
    buckets = _road_buckets(branch_objects)
    changed = False

    for index in range(min(count, len(objects))):
        old = objects[index]
        cap_match = _junction._STOCK_CAP_MODEL.fullmatch(
            old.model_path.replace("/", "\\")
        )
        if cap_match is None:
            continue
        node = (old.x, old.z)
        native = _native_from_fitted_arms(
            node, tuple(_nearby_objects(node, buckets)), spec
        )
        if native is None:
            continue
        cap_family = cap_match.group("family").casefold()
        if native.cap_family != cap_family:
            continue
        objects[index] = _junction._native_junction_object(
            old, native, elevations, spec
        )
        changed = True

    return replace(report, objects=tuple(objects)) if changed else report


def _replace_with_recovery(report, dataset, projection, elevations, spec):
    if _ORIGINAL_REPLACE is None:
        raise RuntimeError("stock road junction recovery policy is not installed")
    report = _ORIGINAL_REPLACE(report, dataset, projection, elevations, spec)
    return _recover_remaining_caps(report, elevations, spec)


def _exact_connector_half_extent(_spec) -> float:
    # Native stock T/X models connect to real six-metre road pieces. Their
    # connector radius is therefore 3.0 m, independent of the old 24.5 spacing
    # preference used by the generator.
    return 3.0


def install_stock_road_junction_recovery_policy() -> None:
    global _ORIGINAL_REPLACE, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_REPLACE = _junction._replace_stock_junction_caps
    _junction._replace_stock_junction_caps = _replace_with_recovery
    _junction._connector_half_extent = _exact_connector_half_extent
    _INSTALLED = True
