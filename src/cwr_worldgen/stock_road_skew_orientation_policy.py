# SPDX-License-Identifier: GPL-3.0-or-later
"""Own late paved T-junction orientation and fallback decisions.

Resistance T junctions use local 0/180 degrees for the through road and local
-X (270 degrees) for the branch. That rigid geometry is useful only while a
source T remains reasonably close to perpendicular. Once the branch is strongly
skewed, sliding the model can put its connector centre on the source line but
cannot rotate the visible asphalt tongue to match that line.

A second residual case occurs when the through road itself turns at the
intersection. A balanced native T can split a modest bend over all three rigid
connectors, but larger bends are safer on the low central-fill path so the fitted
approaches remain authoritative. Both decisions are one late junction-output
responsibility, but their installers remain separately timed to preserve the
historical pipeline order.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import playability as _p
from . import stock_road_final_continuity_policy as _final
from . import stock_road_junction_policy as _junction
from . import stock_road_model_geometry as _model_geometry
from . import stock_road_visual_finish_policy as _finish
from .procedural_infrastructure import paved_fill_model_path

MAXIMUM_NATIVE_T_BRANCH_ERROR_DEGREES = 20.0
MINIMUM_TURNING_T_MAIN_BEND_DEGREES = 2.0
MAXIMUM_TURNING_T_MAIN_BEND_DEGREES = 25.0
MAXIMUM_TURNING_T_CONNECTOR_ERROR_DEGREES = 12.5
MAXIMUM_BALANCED_NATIVE_MAIN_BEND_DEGREES = 15.0
MAXIMUM_NATIVE_NODE_RECOVERY_DISTANCE_METRES = 2.5

_ORIGINAL_FINAL_REALIGN = None
_ORIGINAL_TURNING_REALIGN = None
_INSTALLED = False
_TURNING_FALLBACK_INSTALLED = False


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
        branch_error = _junction._angular_distance(
            (rotation + 270.0) % 360.0, branch_heading
        )
        candidates.append((branch_error, main_error, rotation))

    branch_error, main_error, rotation = min(candidates)
    if main_error > _final.MAXIMUM_SKEW_T_MAIN_AXIS_ERROR_DEGREES:
        return None
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

        seed = replace(current, x=shifted_node[0], z=shifted_node[1])
        objects[index] = _junction._native_junction_object(
            seed, native, elevations, spec
        )
        changed = True

    return replace(report, objects=tuple(objects)) if changed else report


def _nearest_source_junction(incident_map, point: tuple[float, float]):
    direct = incident_map.get(_p._road_node_key(point))
    if direct is not None:
        return direct
    nearest = min(
        incident_map.values(),
        key=lambda value: math.dist(point, value[0]),
        default=None,
    )
    if (
        nearest is None
        or math.dist(point, nearest[0])
        > MAXIMUM_NATIVE_NODE_RECOVERY_DISTANCE_METRES
    ):
        return None
    return nearest


def _legacy_cap_for_turning_t(current, source_node, incidents, family, elevations, spec):
    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return current
    heading = _junction._heading(incidents[pair[0]].direction)
    length = float(_model_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
    half = length * 0.5
    angle = math.radians(heading)
    direction = (math.sin(angle), math.cos(angle))
    start = (
        float(source_node[0]) - direction[0] * half,
        float(source_node[1]) - direction[1] * half,
    )
    end = (
        float(source_node[0]) + direction[0] * half,
        float(source_node[1]) + direction[1] * half,
    )
    model_path = (
        paved_fill_model_path(str(getattr(spec, "name", "cwr_worldgen")))
        if family == "sil"
        else rf"o\road\{family}6.p3d"
    )
    fixed = _p._road_object_on_slope(
        int(current.object_id),
        model_path,
        start,
        end,
        elevations,
        spec,
        vertical_offset=_p._STOCK_ROAD_VERTICAL_OFFSET_METRES,
    )
    return replace(
        fixed,
        x=float(source_node[0]),
        z=float(source_node[1]),
        heading_degrees=heading % 360.0,
    )


def _demote_over_bent_native_ts(report, dataset, projection, elevations, spec):
    if _ORIGINAL_TURNING_REALIGN is None:
        raise RuntimeError("turning-T fallback stage is not installed")
    report = _ORIGINAL_TURNING_REALIGN(report, dataset, projection, elevations, spec)

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
        normalized_model = str(current.model_path).replace("/", "\\").casefold()
        family = _same_family_for_native_t(normalized_model)
        legacy_straight = _model_geometry.stock_straight_match(normalized_model)
        legacy_family = None
        if legacy_straight is not None and int(legacy_straight.group("length")) == 6:
            candidate_family = legacy_straight.group("family").casefold()
            if candidate_family == "sil":
                legacy_family = candidate_family
        if family is None:
            family = legacy_family
        if family is None:
            continue
        logical = (
            (float(current.x), float(current.z))
            if legacy_family is not None
            else _logical_intersection(current)
        )
        if logical is None:
            continue
        junction = _nearest_source_junction(incident_map, logical)
        if junction is None:
            continue
        source_node, incidents = junction
        if (
            len(incidents) != 3
            or any(incident.family != family for incident in incidents)
        ):
            continue
        if legacy_family is not None:
            objects[index] = _legacy_cap_for_turning_t(
                current,
                source_node,
                incidents,
                family,
                elevations,
                spec,
            )
            changed = True
            continue
        pair = _junction._dominant_pair(incidents)
        if pair is None:
            continue
        bend = _turning_main_bend_degrees(incidents, pair)
        if bend <= MAXIMUM_BALANCED_NATIVE_MAIN_BEND_DEGREES + 1.0e-9:
            continue

        objects[index] = _legacy_cap_for_turning_t(
            current,
            source_node,
            incidents,
            family,
            elevations,
            spec,
        )
        changed = True

    return replace(report, objects=tuple(objects)) if changed else report


def install_stock_road_skew_orientation_policy() -> None:
    """Patch the final skew chooser and its visual placement after continuity."""

    global _ORIGINAL_FINAL_REALIGN, _INSTALLED
    if _INSTALLED:
        return
    _final._same_family_paved_skew_t = _same_family_paved_skew_t
    _ORIGINAL_FINAL_REALIGN = _finish._realign_legacy_caps
    if _ORIGINAL_FINAL_REALIGN is None:
        raise RuntimeError("final stock-road continuity policy must install first")
    _finish._realign_legacy_caps = _realign_and_shift_skew_t_caps
    _INSTALLED = True


def install_stock_road_turning_t_fallback_policy() -> None:
    """Demote over-bent T meshes at the historical post-skew stage."""

    global _ORIGINAL_TURNING_REALIGN, _TURNING_FALLBACK_INSTALLED
    global MAXIMUM_TURNING_T_MAIN_BEND_DEGREES
    if _TURNING_FALLBACK_INSTALLED:
        return
    if not _INSTALLED:
        raise RuntimeError("stock road skew-orientation policy must install first")

    MAXIMUM_TURNING_T_MAIN_BEND_DEGREES = (
        MAXIMUM_BALANCED_NATIVE_MAIN_BEND_DEGREES
    )
    _ORIGINAL_TURNING_REALIGN = _finish._realign_legacy_caps
    _finish._realign_legacy_caps = _demote_over_bent_native_ts
    _TURNING_FALLBACK_INSTALLED = True
