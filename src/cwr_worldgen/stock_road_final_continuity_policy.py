# SPDX-License-Identifier: GPL-3.0-or-later
"""Final continuity rules for stock curves and strongly skewed paved T nodes.

Two engine-visible failures remain after centreline fitting:

* stock curves have a rigid ten-degree connector frame.  Measuring source
  headings from one polyline segment at a time can make a smooth sampled arc
  look like a sequence of different radii, producing mixed curve/straight
  pieces whose painted borders do not line up; and
* a same-family paved T can be too skewed for the conservative native-junction
  matcher even when the physical branch road still fully covers the native
  connector.  Falling back to a six-metre straight cap then looks like two
  crossing roads rather than an intersection.

Use a short symmetric tangent window while curve candidates are chosen and while
final curve turn error is audited.  This removes sampling quantisation without
moving the source centreline.  Curve seam underlays are disabled: a wrong curve
choice must be fixed at selection time rather than hidden under another visible
road slab.

For a fallback same-family paved T, keep the dominant through-road axis exact and
use the native T mesh only when its branch connector centre still lies inside the
actual branch road width.  The ordinary fitted approaches already continue under
the cap, so the overlap closes the skew connector without inventing a lateral
repair piece.
"""
from __future__ import annotations

import math

from . import playability as _p
from . import stock_road_curve_policy as _curve
from . import stock_road_geometry_policy as _geometry
from . import stock_road_junction_policy as _junction
from . import stock_road_model_geometry as _model_geometry
from . import stock_road_visual_finish_policy as _finish

SMOOTHED_TANGENT_HALF_WINDOW_METRES = 2.5
MAXIMUM_FINAL_CURVE_TURN_ERROR_DEGREES = 1.75
SKEW_T_CONNECTOR_EDGE_MARGIN_METRES = 0.05
MAXIMUM_SKEW_T_MAIN_AXIS_ERROR_DEGREES = 7.5

_ORIGINAL_CURVE_CHAIN = None
_ORIGINAL_REALIGN_LEGACY_CAPS = None
_INSTALLED = False


class _SmoothedHeadingMeasure:
    """Delegate polyline geometry while returning a stable local tangent."""

    def __init__(self, measure):
        self._measure = measure

    def __getattr__(self, name):
        return getattr(self._measure, name)

    def point(self, distance: float):
        x, z, _heading = self._measure.point(distance)
        heading = _smoothed_measure_heading(self._measure, distance)
        return x, z, heading


def _heading(start, end) -> float:
    return math.degrees(
        math.atan2(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    ) % 360.0


def _smoothed_measure_heading(measure, distance: float) -> float:
    half = SMOOTHED_TANGENT_HALF_WINDOW_METRES
    before = measure.point(float(distance) - half)
    after = measure.point(float(distance) + half)
    if math.dist((before[0], before[1]), (after[0], after[1])) <= 1.0e-9:
        return float(measure.point(distance)[2]) % 360.0
    return _heading(before, after)


def _distance_on_measure(measure, point) -> float:
    """Return cumulative distance of the nearest projection of ``point``."""

    best = None
    for index, (start, end) in enumerate(zip(measure.points, measure.points[1:])):
        dx = float(end[0]) - float(start[0])
        dz = float(end[1]) - float(start[1])
        denominator = dx * dx + dz * dz
        if denominator <= 1.0e-12:
            continue
        t = (
            (float(point[0]) - float(start[0])) * dx
            + (float(point[1]) - float(start[1])) * dz
        ) / denominator
        t = max(0.0, min(1.0, t))
        projected = (
            float(start[0]) + dx * t,
            float(start[1]) + dz * t,
        )
        distance = math.dist((float(point[0]), float(point[1])), projected)
        along = float(measure.cumulative[index]) + math.hypot(dx, dz) * t
        candidate = (distance, index, along)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return 0.0 if best is None else float(best[2])


def _smoothed_curve_turn_error_degrees(run, start, end) -> float:
    measure = _p._PolylineMeasure.create(run)
    start_distance = _distance_on_measure(measure, start)
    end_distance = _distance_on_measure(measure, end)
    start_heading = _smoothed_measure_heading(measure, start_distance)
    end_heading = _smoothed_measure_heading(measure, end_distance)
    source_turn = abs(_p._signed_heading_delta(start_heading, end_heading))
    return abs(source_turn - _model_geometry.STOCK_CURVE_ANGLE_DEGREES)


def _smoothed_curve_chain(measure, pieces, **kwargs):
    if _ORIGINAL_CURVE_CHAIN is None:
        raise RuntimeError("final stock-road continuity policy is not installed")
    return _ORIGINAL_CURVE_CHAIN(_SmoothedHeadingMeasure(measure), pieces, **kwargs)


def _same_family_paved_skew_t(incidents, family: str):
    if len(incidents) != 3 or family not in {"sil", "asf", "kos"}:
        return None
    if any(incident.family != family for incident in incidents):
        return None
    model = _junction._T_JUNCTION_MODELS.get((family, family))
    if model is None:
        return None

    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return None
    first, second = pair
    branch = next(index for index in range(3) if index not in pair)
    branch_heading = _junction._heading(incidents[branch].direction)

    candidates = []
    for zero, opposite in ((first, second), (second, first)):
        rotation, main_error = _junction._best_rotation(
            (
                (0.0, _junction._heading(incidents[zero].direction)),
                (180.0, _junction._heading(incidents[opposite].direction)),
            )
        )
        branch_error = _junction._angular_distance(
            (rotation + 90.0) % 360.0, branch_heading
        )
        candidates.append((branch_error, main_error, rotation))

    branch_error, main_error, rotation = min(candidates)
    if main_error > MAXIMUM_SKEW_T_MAIN_AXIS_ERROR_DEGREES:
        return None

    half_width = float(_model_geometry.STOCK_HALF_WIDTHS_METRES[family])
    lateral = (
        _model_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
        * math.sin(math.radians(branch_error))
    )
    if lateral > half_width - SKEW_T_CONNECTOR_EDGE_MARGIN_METRES:
        return None

    return _junction._NativeJunction(
        model_path=model,
        heading_degrees=rotation % 360.0,
        maximum_heading_error_degrees=max(main_error, branch_error),
        cap_family=family,
    )


def _replace_physically_covered_skew_t_caps(report, dataset, projection, elevations, spec):
    if _ORIGINAL_REALIGN_LEGACY_CAPS is None:
        raise RuntimeError("final stock-road continuity policy is not installed")
    report = _ORIGINAL_REALIGN_LEGACY_CAPS(
        report, dataset, projection, elevations, spec
    )

    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    if cap_count <= 0:
        return report
    incident_map = _finish._junction_incident_map(dataset, projection, spec)
    if not incident_map:
        return report

    objects = list(report.objects)
    changed = False
    for index in range(cap_count):
        old = objects[index]
        match = _model_geometry.stock_straight_match(str(old.model_path))
        if match is None or int(match.group("length")) != 6:
            continue
        family = match.group("family").casefold()
        key = _p._road_node_key((float(old.x), float(old.z)))
        junction = incident_map.get(key)
        if junction is None:
            continue
        node, incidents = junction
        if math.dist((float(old.x), float(old.z)), node) > 0.25:
            continue
        native = _same_family_paved_skew_t(incidents, family)
        if native is None:
            continue
        objects[index] = _junction._native_junction_object(
            old, native, elevations, spec
        )
        changed = True

    if not changed:
        return report
    from dataclasses import replace

    return replace(report, objects=tuple(objects))


def _disable_curve_seam_underlays(report, elevations, spec):
    """Curve borders are fixed by coherent piece selection, not repair slabs."""

    return report


def install_stock_road_final_continuity_policy() -> None:
    global _ORIGINAL_CURVE_CHAIN, _ORIGINAL_REALIGN_LEGACY_CAPS, _INSTALLED
    if _INSTALLED:
        return

    _ORIGINAL_CURVE_CHAIN = _geometry._ORIGINAL_CURVE_CHAIN
    _ORIGINAL_REALIGN_LEGACY_CAPS = _finish._realign_legacy_caps
    if _ORIGINAL_CURVE_CHAIN is None or _ORIGINAL_REALIGN_LEGACY_CAPS is None:
        raise RuntimeError("stock road geometry and visual-finish policies must install first")

    # The geometry wrapper calls this captured chain at run time, so replacing
    # the capture preserves every outer terrain/connector wrapper while giving
    # curve selection stable tangent observations.
    _geometry._ORIGINAL_CURVE_CHAIN = _smoothed_curve_chain
    _geometry._curve_turn_error_degrees = _smoothed_curve_turn_error_degrees
    _geometry._MAXIMUM_TANGENT_TURN_ERROR_DEGREES = (
        MAXIMUM_FINAL_CURVE_TURN_ERROR_DEGREES
    )

    _finish._apply_curve_seam_covers = _disable_curve_seam_underlays
    _finish._realign_legacy_caps = _replace_physically_covered_skew_t_caps
    _INSTALLED = True
