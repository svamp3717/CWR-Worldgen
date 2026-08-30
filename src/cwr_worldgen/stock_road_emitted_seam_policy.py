# SPDX-License-Identifier: GPL-3.0-or-later
"""Close paved seams after every other stock-road wrapper has finished.

Several road policies legitimately replace or append objects after the older
visual-finish seam hook runs.  A seam that is perfect in the intermediate report
can also open in the actual WRP because a pitched rigid P3D has only
``length*cos(pitch)`` of horizontal connector span.  The final Road Inspector
therefore used to find asphalt gaps that no generator-side seam pass had ever
seen.

This policy is deliberately the outermost stock-road fit wrapper.  It inspects
the final ``WorldObject`` geometry through ``_p._model_axis`` (which already uses
measured 3D stock connectors), ignores gaps already covered by another
same-family paved straight, and adds only low six-metre underlays.  Straight
mitres use the same 0.20 m physical connector radius as Road Inspector.  Residual
curve seams may bridge up to 1.50 m when the mutually-nearest connector pair has
a modest tangent error; the underlay follows the straight-side tangent when one
is available, avoiding a diagonal slab across the carriageway.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import generator as _generator
from . import playability as _p
from . import stock_road_model_geometry as _geometry
from . import stock_road_visual_finish_policy as _finish
from .procedural_infrastructure import paved_fill_model_path

MAXIMUM_EMITTED_STRAIGHT_GAP_METRES = 0.20
MAXIMUM_EMITTED_CURVE_GAP_METRES = 1.50
MINIMUM_EMITTED_TANGENT_ERROR_DEGREES = 0.75
MAXIMUM_EMITTED_STRAIGHT_TANGENT_ERROR_DEGREES = 35.0
MAXIMUM_EMITTED_CURVE_TANGENT_ERROR_DEGREES = 12.0
EMITTED_SEAM_UNDERLAY_BIAS_METRES = -0.010
_ENDPOINT_BUCKET_METRES = MAXIMUM_EMITTED_CURVE_GAP_METRES
_PAIR_UNIQUENESS_MARGIN_METRES = 0.03
_SURFACE_MARGIN_METRES = 0.08
_VERTICAL_NEIGHBOURHOOD_METRES = 0.20
_SAMPLE_FRACTIONS = (0.20, 0.40, 0.60, 0.80)
_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})

_ORIGINAL_FIT = None
_INSTALLED = False


def _bucket(point: tuple[float, float]) -> tuple[int, int]:
    return (
        math.floor(float(point[0]) / _ENDPOINT_BUCKET_METRES),
        math.floor(float(point[1]) / _ENDPOINT_BUCKET_METRES),
    )


def _endpoint_half_width(endpoint) -> float:
    return float(_geometry.STOCK_HALF_WIDTHS_METRES[endpoint.family])


def _cross_section_edges(endpoint):
    heading = math.radians(float(endpoint.tangent_axis_degrees))
    normal = (math.cos(heading), -math.sin(heading))
    width = _endpoint_half_width(endpoint)
    return (
        (
            float(endpoint.point[0]) + normal[0] * width,
            float(endpoint.point[1]) + normal[1] * width,
        ),
        (
            float(endpoint.point[0]) - normal[0] * width,
            float(endpoint.point[1]) - normal[1] * width,
        ),
    )


def _segment_samples(start, end):
    return tuple(
        (
            float(start[0]) + (float(end[0]) - float(start[0])) * fraction,
            float(start[1]) + (float(end[1]) - float(start[1])) * fraction,
        )
        for fraction in _SAMPLE_FRACTIONS
    )


def _gap_samples(first, second):
    first_edges = _cross_section_edges(first)
    second_edges = _cross_section_edges(second)
    direct = (
        (first_edges[0], second_edges[0]),
        (first_edges[1], second_edges[1]),
    )
    crossed = (
        (first_edges[0], second_edges[1]),
        (first_edges[1], second_edges[0]),
    )
    pairs = (
        direct
        if sum(math.dist(a, b) for a, b in direct)
        <= sum(math.dist(a, b) for a, b in crossed)
        else crossed
    )
    samples = []
    for start, end in pairs:
        samples.extend(_segment_samples(start, end))
    samples.extend(_segment_samples(first.point, second.point))
    return tuple(samples)


def _straight_contains(obj, point: tuple[float, float]) -> bool:
    match = _geometry.stock_straight_match(str(obj.model_path))
    if match is None:
        return False
    family = match.group("family").casefold()
    if family not in _PAVED_FAMILIES:
        return False
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[int(match.group("length"))])
    start, end = _p._model_axis(obj, length)
    dx = float(end[0]) - float(start[0])
    dz = float(end[1]) - float(start[1])
    horizontal = math.hypot(dx, dz)
    if horizontal <= 1.0e-9:
        return False
    ux, uz = dx / horizontal, dz / horizontal
    px = float(point[0]) - float(start[0])
    pz = float(point[1]) - float(start[1])
    along = px * ux + pz * uz
    lateral = abs(px * -uz + pz * ux)
    return (
        -_SURFACE_MARGIN_METRES <= along <= horizontal + _SURFACE_MARGIN_METRES
        and lateral
        <= float(_geometry.STOCK_HALF_WIDTHS_METRES[family]) + _SURFACE_MARGIN_METRES
    )


def _covered_by_existing_surface(report, first, second) -> bool:
    samples = _gap_samples(first, second)
    if not samples:
        return False
    object_ids = {int(first.object_id), int(second.object_id)}
    road_by_id = {int(obj.object_id): obj for obj in report.objects}
    involved = [road_by_id.get(object_id) for object_id in object_ids]
    involved = [obj for obj in involved if obj is not None]
    if len(involved) != 2:
        return False
    minimum_y = min(float(obj.y) for obj in involved) - _VERTICAL_NEIGHBOURHOOD_METRES
    maximum_y = max(float(obj.y) for obj in involved) + _VERTICAL_NEIGHBOURHOOD_METRES
    candidates = tuple(
        obj
        for obj in report.objects
        if (
            int(obj.object_id) not in object_ids
            and minimum_y <= float(obj.y) <= maximum_y
            and (
                (match := _geometry.stock_straight_match(str(obj.model_path)))
                is not None
            )
            and match.group("family").casefold() == first.family
            and first.family in _PAVED_FAMILIES
        )
    )
    if not candidates:
        return False
    return all(any(_straight_contains(obj, sample) for obj in candidates) for sample in samples)


def _nearest_endpoint_pairs(endpoints):
    buckets: dict[tuple[str, int, int], list[int]] = {}
    for index, endpoint in enumerate(endpoints):
        bx, bz = _bucket(endpoint.point)
        buckets.setdefault((endpoint.family, bx, bz), []).append(index)

    nearest = []
    for index, endpoint in enumerate(endpoints):
        bx, bz = _bucket(endpoint.point)
        best = None
        for dx in (-1, 0, 1):
            for dz in (-1, 0, 1):
                for candidate_index in buckets.get(
                    (endpoint.family, bx + dx, bz + dz), ()
                ):
                    if candidate_index == index:
                        continue
                    candidate = endpoints[candidate_index]
                    if candidate.object_id == endpoint.object_id:
                        continue
                    distance = math.dist(endpoint.point, candidate.point)
                    if distance > MAXIMUM_EMITTED_CURVE_GAP_METRES + 1.0e-9:
                        continue
                    score = (
                        distance,
                        int(candidate.object_id),
                        int(candidate.endpoint_index),
                        candidate_index,
                    )
                    if best is None or score < best:
                        best = score
        nearest.append(best)

    pairs = []
    used = set()
    for index, best in enumerate(nearest):
        if best is None:
            continue
        other_index = int(best[-1])
        reverse = nearest[other_index]
        if reverse is None or int(reverse[-1]) != index:
            continue
        pair_key = tuple(sorted((index, other_index)))
        if pair_key in used:
            continue
        used.add(pair_key)
        first, second = endpoints[pair_key[0]], endpoints[pair_key[1]]
        pairs.append((float(best[0]), first, second, pair_key))
    return tuple(pairs)


def _pair_is_unambiguous(endpoints, first, second, distance: float) -> bool:
    limit = float(distance) + _PAIR_UNIQUENESS_MARGIN_METRES
    for candidate in endpoints:
        key = (int(candidate.object_id), int(candidate.endpoint_index))
        if key in {
            (int(first.object_id), int(first.endpoint_index)),
            (int(second.object_id), int(second.endpoint_index)),
        }:
            continue
        if candidate.family != first.family:
            continue
        if candidate.object_id in {first.object_id, second.object_id}:
            continue
        if (
            math.dist(candidate.point, first.point) <= limit
            or math.dist(candidate.point, second.point) <= limit
        ):
            return False
    return True


def _plan_heading(first, second) -> float:
    if first.is_curve != second.is_curve:
        straight = second if first.is_curve else first
        return float(straight.tangent_axis_degrees) % 180.0
    return _finish._average_axis_heading(
        first.tangent_axis_degrees,
        second.tangent_axis_degrees,
    )


def _emitted_seam_cover_plans(report):
    endpoints = tuple(
        endpoint
        for endpoint in _finish._seam_endpoints(report)
        if endpoint.family in _PAVED_FAMILIES
    )
    if not endpoints:
        return ()

    plans = []
    for distance, first, second, _pair_key in _nearest_endpoint_pairs(endpoints):
        if first.family != second.family:
            continue
        if not _pair_is_unambiguous(endpoints, first, second, distance):
            continue
        tangent_error = _finish._axis_heading_difference(
            first.tangent_axis_degrees,
            second.tangent_axis_degrees,
        )
        if tangent_error < MINIMUM_EMITTED_TANGENT_ERROR_DEGREES:
            continue

        curve_seam = bool(first.is_curve or second.is_curve)
        if curve_seam:
            if (
                distance > MAXIMUM_EMITTED_CURVE_GAP_METRES + 1.0e-9
                or tangent_error > MAXIMUM_EMITTED_CURVE_TANGENT_ERROR_DEGREES
            ):
                continue
        else:
            if (
                distance > MAXIMUM_EMITTED_STRAIGHT_GAP_METRES + 1.0e-9
                or tangent_error > MAXIMUM_EMITTED_STRAIGHT_TANGENT_ERROR_DEGREES
            ):
                continue

        if _covered_by_existing_surface(report, first, second):
            continue
        plans.append(
            _finish._SeamCoverPlan(
                model_path=rf"o\road\{first.family}6.p3d",
                centre=(
                    (float(first.point[0]) + float(second.point[0])) * 0.5,
                    (float(first.point[1]) + float(second.point[1])) * 0.5,
                ),
                tangent_axis_degrees=_plan_heading(first, second),
            )
        )
    return tuple(plans)


def _apply_emitted_seam_covers(report, elevations, spec):
    plans = _emitted_seam_cover_plans(report)
    if not plans:
        return report

    required = len(report.objects) + len(plans)
    if (
        required > int(spec.max_road_objects)
        and not bool(getattr(spec, "advisory_object_limits", False))
    ):
        raise ValueError(
            "road object budget is too small after final emitted seam coverage: "
            f"requires {required:,} objects, limit is {int(spec.max_road_objects):,}"
        )

    objects = list(report.objects)
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    half = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6]) * 0.5
    for plan in plans:
        angle = math.radians(float(plan.tangent_axis_degrees))
        direction = (math.sin(angle), math.cos(angle))
        start = (
            float(plan.centre[0]) - direction[0] * half,
            float(plan.centre[1]) - direction[1] * half,
        )
        end = (
            float(plan.centre[0]) + direction[0] * half,
            float(plan.centre[1]) + direction[1] * half,
        )
        model_path = plan.model_path
        if str(model_path).replace("/", "\\").casefold() == r"o\road\sil6.p3d":
            model_path = paved_fill_model_path(
                str(getattr(spec, "name", "cwr_worldgen"))
            )
        objects.append(
            _p._road_object_on_slope(
                next_id,
                model_path,
                start,
                end,
                elevations,
                spec,
                vertical_offset=(
                    _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
                    + EMITTED_SEAM_UNDERLAY_BIAS_METRES
                ),
            )
        )
        next_id += 1

    return replace(
        report,
        objects=tuple(objects),
        short_piece_objects=(
            int(getattr(report, "short_piece_objects", 0)) + len(plans)
        ),
    )


def _fit(
    dataset,
    projection,
    elevations,
    spec,
    *,
    starting_id=1,
    progress_callback=None,
):
    if _ORIGINAL_FIT is None:
        raise RuntimeError("final emitted seam policy is not installed")
    report = _ORIGINAL_FIT(
        dataset,
        projection,
        elevations,
        spec,
        starting_id=starting_id,
        progress_callback=progress_callback,
    )
    if not bool(getattr(spec, "stock_road_piece_fitting", False)):
        return report
    return _apply_emitted_seam_covers(report, elevations, spec)


def install_stock_road_emitted_seam_policy() -> None:
    """Install the final post-fit paved seam audit after intersection coverage."""

    global _ORIGINAL_FIT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_FIT = _p.fit_road_objects
    _p.fit_road_objects = _fit
    _generator.fit_road_objects = _fit
    _INSTALLED = True
