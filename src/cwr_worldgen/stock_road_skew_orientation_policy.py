# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep late skew-T replacement aligned with measured Memory-LOD geometry.

Resistance T junctions use local 0/180 degrees for the through road and local
-X (270 degrees) for the branch. That rigid geometry is useful only while a
source T remains reasonably close to perpendicular. Once the branch is strongly
skewed, sliding the model can put its connector centre on the source line but
cannot rotate the visible asphalt tongue to match that line. The resulting
surface is the broad rectangular slab seen in RoadLab.

A second residual case occurs when the *through* road itself turns at the
intersection. Aligning a legacy six-metre cap to only one arm puts the entire
heading change on the opposite road edge. For bounded paved T nodes, use the
otherwise-unused native same-family T mesh and choose a minimax rotation across
all three measured connectors. This splits a modest through-road bend across the
intersection instead of leaving one large edge mismatch. Strongly skewed nodes
still retain the small legacy cap.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import playability as _p
from . import stock_road_final_continuity_policy as _final
from . import stock_road_junction_policy as _junction
from . import stock_road_model_geometry as _model_geometry
from . import stock_road_visual_finish_policy as _finish

MAXIMUM_NATIVE_T_BRANCH_ERROR_DEGREES = 20.0
MINIMUM_TURNING_T_MAIN_BEND_DEGREES = 2.0
MAXIMUM_TURNING_T_MAIN_BEND_DEGREES = 25.0
MAXIMUM_TURNING_T_CONNECTOR_ERROR_DEGREES = 12.5
_ORIGINAL_FINAL_REALIGN = None
_INSTALLED = False


def _turning_main_bend_degrees(incidents, pair) -> float:
    first, second = pair
    first_heading = _junction._heading(incidents[first].direction)
    second_heading = _junction._heading(incidents[second].direction)
    separation = _junction._angular_distance(first_heading, second_heading)
    return abs(180.0 - separation)


def _balanced_turning_t(incidents, family: str, model: str, pair, branch):
    """Fit a native T by sharing a modest through-road bend over all connectors."""

    bend = _turning_main_bend_degrees(incidents, pair)
    if not (
        MINIMUM_TURNING_T_MAIN_BEND_DEGREES
        <= bend
        <= MAXIMUM_TURNING_T_MAIN_BEND_DEGREES
    ):
        return None

    first, second = pair
    branch_heading = _junction._heading(incidents[branch].direction)
    candidates = []
    for zero, opposite in ((first, second), (second, first)):
        actual = (
            _junction._heading(incidents[zero].direction),
            _junction._heading(incidents[opposite].direction),
            branch_heading,
        )
        rotation, _maximum = _junction._best_rotation(
            (
                (0.0, actual[0]),
                (180.0, actual[1]),
                (270.0, actual[2]),
            )
        )
        errors = (
            _junction._angular_distance(rotation, actual[0]),
            _junction._angular_distance((rotation + 180.0) % 360.0, actual[1]),
            _junction._angular_distance((rotation + 270.0) % 360.0, actual[2]),
        )
        candidates.append(
            (
                max(errors),
                sum(error * error for error in errors),
                rotation % 360.0,
                errors,
            )
        )

    maximum_error, _sum_squared, rotation, errors = min(candidates)
    if maximum_error > MAXIMUM_TURNING_T_CONNECTOR_ERROR_DEGREES:
        return None

    # Heading error is only useful when the physical connector still lands well
    # inside the source carriageway. Check all three arms, not merely the branch.
    half_width = float(_model_geometry.STOCK_HALF_WIDTHS_METRES[family])
    usable_half_width = half_width - _final.SKEW_T_CONNECTOR_EDGE_MARGIN_METRES
    radius = float(_model_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES)
    if any(
        radius * math.sin(math.radians(error)) > usable_half_width
        for error in errors
    ):
        return None

    return _junction._NativeJunction(
        model_path=model,
        heading_degrees=rotation,
        maximum_heading_error_degrees=maximum_error,
        cap_family=family,
    )


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

    # When the through road bends at the node, a cap aligned to one side leaves
    # the whole angle on the other road edge. A balanced native T is visibly
    # better when every rigid connector remains within a small, bounded error.
    turning = _balanced_turning_t(incidents, family, model, pair, branch)
    if turning is not None:
        return turning

    branch_heading = _junction._heading(incidents[branch].direction)
    candidates = []
    for zero, opposite in ((first, second), (second, first)):
        rotation, main_error = _junction._best_rotation(
            (
                (0.0, _junction._heading(incidents[zero].direction)),
                (180.0, _junction._heading(incidents[opposite].direction)),
            )
        )
        # Memory LOD measurements put the T branch on local -X, not +X.
        branch_error = _junction._angular_distance(
            (rotation + 270.0) % 360.0, branch_heading
        )
        candidates.append((branch_error, main_error, rotation))

    branch_error, main_error, rotation = min(candidates)
    if main_error > _final.MAXIMUM_SKEW_T_MAIN_AXIS_ERROR_DEGREES:
        return None

    # A connector centre can still lie inside the road strip at much larger
    # angles, but the rendered T tongue cannot. Cap the visible angular error
    # explicitly instead of treating half-width containment as sufficient.
    if branch_error > MAXIMUM_NATIVE_T_BRANCH_ERROR_DEGREES:
        return None

    half_width = float(_model_geometry.STOCK_HALF_WIDTHS_METRES[family])
    lateral = (
        _model_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
        * math.sin(math.radians(branch_error))
    )
    if lateral > half_width - _final.SKEW_T_CONNECTOR_EDGE_MARGIN_METRES:
        return None

    return _junction._NativeJunction(
        model_path=model,
        heading_degrees=rotation % 360.0,
        maximum_heading_error_degrees=max(main_error, branch_error),
        cap_family=family,
    )


def _unit_heading(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(float(heading_degrees))
    return math.sin(angle), math.cos(angle)


def _skew_t_longitudinal_shift(incidents, native) -> float:
    """Return signed main-axis shift that puts the T branch on the source line."""

    if len(incidents) != 3:
        return 0.0
    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return 0.0
    branch = next(index for index in range(3) if index not in pair)

    bx, bz = incidents[branch].direction
    length = math.hypot(float(bx), float(bz))
    if length <= 1.0e-9:
        return 0.0
    branch_unit = (float(bx) / length, float(bz) / length)

    main_unit = _unit_heading(float(native.heading_degrees))
    connector_unit = _unit_heading((float(native.heading_degrees) + 270.0) % 360.0)
    along_main = branch_unit[0] * main_unit[0] + branch_unit[1] * main_unit[1]
    along_connector = (
        branch_unit[0] * connector_unit[0]
        + branch_unit[1] * connector_unit[1]
    )
    if along_connector <= 1.0e-6:
        return 0.0

    radius = float(_model_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES)
    return radius * along_main / along_connector


def _same_family_for_native_t(model_path: str) -> str | None:
    normalized = str(model_path).replace("/", "\\").casefold()
    for (main, branch), candidate in _junction._T_JUNCTION_MODELS.items():
        if main == branch and candidate.casefold() == normalized:
            return main
    return None


def _logical_intersection(obj) -> tuple[float, float] | None:
    local = _model_geometry.native_junction_intersection_offset(str(obj.model_path))
    if local is None:
        return None
    return _model_geometry.transform_local(
        local,
        (float(obj.x), float(obj.z)),
        float(obj.heading_degrees),
    )


def _realign_and_shift_skew_t_caps(report, dataset, projection, elevations, spec):
    """Run final cap selection, then align accepted near-orthogonal native Ts."""

    if _ORIGINAL_FINAL_REALIGN is None:
        raise RuntimeError("skew orientation policy is not installed")
    report = _ORIGINAL_FINAL_REALIGN(report, dataset, projection, elevations, spec)

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
        current = objects[index]
        family = _same_family_for_native_t(str(current.model_path))
        if family is None:
            continue
        node = _logical_intersection(current)
        if node is None:
            continue
        junction = incident_map.get(_p._road_node_key(node))
        if junction is None:
            # Projection normalization can move a node by a few millimetres.
            nearest = min(
                incident_map.values(),
                key=lambda value: math.dist(node, value[0]),
                default=None,
            )
            if nearest is None or math.dist(node, nearest[0]) > 0.05:
                continue
            junction = nearest
        source_node, incidents = junction
        native = _same_family_paved_skew_t(incidents, family)
        if native is None:
            continue
        if native.model_path.casefold() != str(current.model_path).casefold():
            continue
        if _junction._angular_distance(
            native.heading_degrees, float(current.heading_degrees)
        ) > 0.05:
            continue

        shift = _skew_t_longitudinal_shift(incidents, native)
        if abs(shift) <= 0.02:
            continue
        main_unit = _unit_heading(native.heading_degrees)
        shifted_node = (
            float(source_node[0]) + main_unit[0] * shift,
            float(source_node[1]) + main_unit[1] * shift,
        )

        # Re-run measured placement at the shifted logical center so terrain
        # height/pitch and the model's asymmetric origin offset stay correct.
        seed = replace(current, x=shifted_node[0], z=shifted_node[1])
        objects[index] = _junction._native_junction_object(
            seed, native, elevations, spec
        )
        changed = True

    return replace(report, objects=tuple(objects)) if changed else report


def install_stock_road_skew_orientation_policy() -> None:
    """Patch the final skew chooser and its visual placement after all layers."""

    global _ORIGINAL_FINAL_REALIGN, _INSTALLED
    if _INSTALLED:
        return
    _final._same_family_paved_skew_t = _same_family_paved_skew_t
    _ORIGINAL_FINAL_REALIGN = _finish._realign_legacy_caps
    if _ORIGINAL_FINAL_REALIGN is None:
        raise RuntimeError("final stock-road continuity policy must install first")
    _finish._realign_legacy_caps = _realign_and_shift_skew_t_caps
    _INSTALLED = True
