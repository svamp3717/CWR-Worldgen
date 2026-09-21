# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep fallback paved-intersection fill below the visible approaches.

The stock T/X models cannot represent every skewed road node.  When fitting one
of those junctions fails, the ordinary road fitter restores a short straight cap
at the node.  That cap is useful central fill, but leaving it coplanar with the
three or four source-aligned approaches produces z-fighting and a rectangular
slab that visibly clips their borders.

This final road pass keeps the cap as a shallow underlay.  Same-family junctions
also receive one short, lower tongue on each otherwise-uncovered road axis.  The
real approaches remain the highest surface and therefore own the visible edges;
the added pieces only cover the small wedges between them.  Native stock T/X
junctions, mixed surfaces, dirt and gravel are unchanged.
"""
from __future__ import annotations

from dataclasses import replace
import math
import re

from . import generator as _generator
from . import paved_junction_policy as _paved
from . import playability as _p
from . import road_quality_policy as _quality


_NODE_BUCKET_METRES = 1.0
_NODE_MATCH_METRES = 0.75
_AXIS_DUPLICATE_TOLERANCE_DEGREES = 1.0
_CAP_UNDERLAY_DROP_METRES = 0.006
_TONGUE_UNDERLAY_DROP_METRES = 0.005
_SHORT_STRAIGHT = re.compile(r"(?<!\d)6\.p3d$", re.IGNORECASE)

_ORIGINAL_FIT = None
_INSTALLED = False


def _normalise_model_path(model_path: str) -> str:
    return str(model_path).replace("/", "\\").casefold()


def _bucket(point: tuple[float, float]) -> tuple[int, int]:
    return (
        math.floor(float(point[0]) / _NODE_BUCKET_METRES),
        math.floor(float(point[1]) / _NODE_BUCKET_METRES),
    )


def _nearby_objects(buckets, point: tuple[float, float]):
    bx, bz = _bucket(point)
    for nx in range(bx - 1, bx + 2):
        for nz in range(bz - 1, bz + 2):
            yield from buckets.get((nx, nz), ())


def _ordinary_short_paved_family(obj) -> str | None:
    path = _normalise_model_path(obj.model_path)
    filename = path.rsplit("\\", 1)[-1]
    if _SHORT_STRAIGHT.search(filename) is None:
        return None
    family = _paved._family(path)
    if family is None or _paved._kind(family) != "paved":
        return None
    return family


def _axis_heading_difference(first: float, second: float) -> float:
    difference = abs((float(first) - float(second) + 180.0) % 360.0 - 180.0)
    return min(difference, abs(180.0 - difference))


def _uncovered_axis_headings(plan, cap_heading: float) -> tuple[float, ...]:
    result: list[float] = []
    for arm in plan.arms:
        heading = _paved._heading(arm.source_direction)
        if (
            _axis_heading_difference(heading, cap_heading)
            <= _AXIS_DUPLICATE_TOLERANCE_DEGREES
        ):
            continue
        if any(
            _axis_heading_difference(heading, existing)
            <= _AXIS_DUPLICATE_TOLERANCE_DEGREES
            for existing in result
        ):
            continue
        result.append(heading)
    return tuple(result)


def _same_surface_family(plan, cap_family: str) -> bool:
    expected = _paved._junction_family(cap_family)
    return all(
        _paved._junction_family(arm.family) == expected
        for arm in plan.arms
    )


def _tongue_object(
    object_id: int,
    model_path: str,
    point: tuple[float, float],
    heading: float,
    elevations,
    spec,
):
    length = _quality._piece_length(
        model_path,
        float(spec.road_segment_length),
    )
    half = length * 0.5
    direction = _paved._direction(heading)
    start = (
        point[0] - direction[0] * half,
        point[1] - direction[1] * half,
    )
    end = (
        point[0] + direction[0] * half,
        point[1] + direction[1] * half,
    )
    return _p._road_object_on_slope(
        object_id,
        model_path,
        start,
        end,
        elevations,
        spec,
        vertical_offset=(
            _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
            - _TONGUE_UNDERLAY_DROP_METRES
        ),
    )


def _lower_cap(cap, point, elevations, spec):
    length = _quality._piece_length(
        str(cap.model_path),
        float(spec.road_segment_length),
    )
    half = length * 0.5
    direction = _paved._direction(float(cap.heading_degrees))
    start = (
        point[0] - direction[0] * half,
        point[1] - direction[1] * half,
    )
    end = (
        point[0] + direction[0] * half,
        point[1] + direction[1] * half,
    )
    lowered = _p._road_object_on_slope(
        int(cap.object_id),
        str(cap.model_path),
        start,
        end,
        elevations,
        spec,
        vertical_offset=(
            _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
            - _CAP_UNDERLAY_DROP_METRES
        ),
    )
    return replace(
        lowered,
        x=float(point[0]),
        z=float(point[1]),
        heading_degrees=float(cap.heading_degrees) % 360.0,
    )


def finish_paved_intersection_overlaps(
    report,
    plans,
    elevations,
    spec,
):
    """Lower ordinary paved caps left by failed stock-junction plans."""

    objects = list(getattr(report, "objects", ()))
    if not objects or not plans or int(getattr(report, "junction_cap_objects", 0)) <= 0:
        return report

    buckets: dict[tuple[int, int], list[object]] = {}
    for obj in objects:
        buckets.setdefault(_bucket((float(obj.x), float(obj.z))), []).append(obj)

    index_by_id = {
        int(obj.object_id): index
        for index, obj in enumerate(objects)
    }
    cap_count = min(int(report.junction_cap_objects), len(objects))
    cap_ids = {
        int(obj.object_id)
        for obj in objects[:cap_count]
    }
    lowered_ids: set[int] = set()
    additions = []
    next_id = max(index_by_id, default=0) + 1

    for key in sorted(plans):
        plan = plans[key]
        nearby = tuple(
            obj
            for obj in _nearby_objects(buckets, plan.point)
            if math.dist((float(obj.x), float(obj.z)), plan.point)
            <= _NODE_MATCH_METRES
        )
        native_model = _normalise_model_path(plan.model_path)
        if any(
            _normalise_model_path(obj.model_path) == native_model
            for obj in nearby
        ):
            continue

        candidates = []
        for obj in nearby:
            family = _ordinary_short_paved_family(obj)
            if (
                family is None
                or int(obj.object_id) not in cap_ids
                or int(obj.object_id) in lowered_ids
            ):
                continue
            candidates.append((
                math.dist((float(obj.x), float(obj.z)), plan.point),
                int(obj.object_id),
                family,
                obj,
            ))
        if not candidates:
            continue

        _distance, object_id, family, cap = min(candidates)
        objects[index_by_id[object_id]] = _lower_cap(
            cap,
            plan.point,
            elevations,
            spec,
        )
        lowered_ids.add(object_id)

        # Mixed paved families can have different widths and edge textures.  The
        # lowered central cap is still safe, but a tongue borrowed from one family
        # could visibly leak beneath another, so only seal uniform-family nodes.
        if not _same_surface_family(plan, family):
            continue
        for heading in _uncovered_axis_headings(
            plan,
            float(cap.heading_degrees),
        ):
            additions.append(
                _tongue_object(
                    next_id,
                    str(cap.model_path),
                    plan.point,
                    heading,
                    elevations,
                    spec,
                )
            )
            next_id += 1

    if not lowered_ids and not additions:
        return report

    objects.extend(additions)
    maximum = int(getattr(spec, "max_road_objects", len(objects)))
    if (
        len(objects) > maximum
        and not bool(getattr(spec, "advisory_object_limits", False))
    ):
        raise ValueError(
            "road object budget is too small after paved-intersection overlap "
            f"repair: requires {len(objects):,} objects, limit is {maximum:,}"
        )

    updates = {"objects": tuple(objects)}
    if hasattr(report, "short_piece_objects"):
        updates["short_piece_objects"] = (
            int(report.short_piece_objects) + len(additions)
        )
    if hasattr(report, "maximum_road_pitch_degrees") and additions:
        updates["maximum_road_pitch_degrees"] = max(
            float(report.maximum_road_pitch_degrees),
            max(abs(float(obj.pitch_degrees)) for obj in additions),
        )
    return replace(report, **updates)


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id: int = 1,
    progress_callback=None,
):
    if _ORIGINAL_FIT is None:
        raise RuntimeError("paved intersection overlap policy is not installed")
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    if (
        not bool(getattr(spec, "stock_road_piece_fitting", False))
        or int(getattr(report, "junction_cap_objects", 0)) <= 0
    ):
        return report
    plans = _paved._plans(dataset, projection, spec)
    return finish_paved_intersection_overlaps(
        report,
        plans,
        elevations,
        spec,
    )


def install_paved_intersection_overlap_policy() -> None:
    """Install the final source-aware paved-intersection surface pass."""

    global _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
