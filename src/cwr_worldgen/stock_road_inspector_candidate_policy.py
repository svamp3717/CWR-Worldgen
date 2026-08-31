# SPDX-License-Identifier: GPL-3.0-or-later
"""Apply Road Inspector's paved and mixed-junction repair candidates.

The candidate text in Road Inspector is treated as executable policy here rather
than as a vague suggestion:

* native T/X meshes are accepted only when their measured connector directions
  are within the Inspector's visible tolerance;
* otherwise the fitted approaches remain authoritative and the junction is
  reduced to a low stock-family centre fill, with low stock tongues only where a
  real approach stops short of the logical node;
* purpose-built native junctions own their complete measured connector footprint,
  including the stock ``ces`` branch of a mixed paved/ces T;
* paved bends get one final connector-locked stock-curve search that may replace
  an equally curved but tangent-mismatched baseline chain;
* turn wedges are not hidden with full overlapping road strips, and this policy
  does not emit generated paved helper P3Ds.

Standalone stock ``ces``/generated-gravel bends remain outside this policy.
``ces`` is touched only at a mixed junction whose main road is paved.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import playability as _p
from . import stock_road_curve_usage_policy as _curve_usage
from . import stock_road_emitted_seam_policy as _emitted
from . import stock_road_junction_policy as _junction
from . import stock_road_micro_bend_policy as _micro
from . import stock_road_model_geometry as _geometry
from . import stock_road_native_junction_ownership_policy as _ownership
from . import stock_road_paved_junction_completion_policy as _paved
from . import stock_road_sharp_exact_policy as _exact
from . import stock_road_sharp_turn_policy as _sharp
from . import stock_road_single_vertex_bend_policy as _single
from . import stock_road_stock_paved_only_policy as _stock_only
from . import stock_road_surface_overlap_policy as _surface
from . import stock_road_visual_finish_policy as _finish


MEASURED_T_BRANCH_LOCAL_HEADING_DEGREES = 270.0
INSPECTOR_NATIVE_CONNECTOR_TOLERANCE_DEGREES = 0.90
INSPECTOR_CURVE_MINIMUM_TURN_DEGREES = 4.50
INSPECTOR_CURVE_MAXIMUM_TURN_DEGREES = 70.0
INSPECTOR_CURVE_TRANSITION_ERROR_DEGREES = 0.75
INSPECTOR_CURVE_MAXIMUM_EXTRA_PIECES = 2

JUNCTION_TONGUE_MINIMUM_GAP_METRES = 0.35
JUNCTION_TONGUE_MAXIMUM_ENDPOINT_DISTANCE_METRES = 7.25
JUNCTION_TONGUE_ALIGNMENT_COSINE = math.cos(math.radians(35.0))
JUNCTION_TONGUE_VERTICAL_BIAS_METRES = -0.009

_NATIVE_OWNED_STOCK_FAMILIES = frozenset({"sil", "asf", "kos", "ces"})
_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})

_ORIGINAL_STOCK_APPLY = None
_ORIGINAL_OWNER_REALIGN = None
_ORIGINAL_PIECE_CHAIN = None
_INSTALLED = False


def _normalised(vector: tuple[float, float]) -> tuple[float, float]:
    length = math.hypot(float(vector[0]), float(vector[1]))
    if length <= 1.0e-9:
        return (0.0, 1.0)
    return float(vector[0]) / length, float(vector[1]) / length


def _eligible_paved_or_mixed_incidents(incidents) -> bool:
    """Return True for all-paved T/X or a paved-main T with a stock ces branch."""

    if len(incidents) not in {3, 4}:
        return False
    if all(incident.family in _PAVED_FAMILIES for incident in incidents):
        return True
    if len(incidents) != 3:
        return False

    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return False
    first, second = pair
    branch = next(index for index in range(3) if index not in pair)
    main_family = incidents[first].family
    return (
        main_family in _PAVED_FAMILIES
        and incidents[second].family == main_family
        and incidents[branch].family in (_PAVED_FAMILIES | {"ces"})
    )


def _measured_native_t_junction(incidents):
    """Fit a T from measured connectors and reject visible connector rotation."""

    if len(incidents) != 3:
        return None
    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return None
    first, second = pair
    branch = next(index for index in range(3) if index not in pair)
    main_family = incidents[first].family
    if main_family is None or incidents[second].family != main_family:
        return None
    branch_family = incidents[branch].family
    if branch_family is None:
        return None
    model = _junction._T_JUNCTION_MODELS.get((main_family, branch_family))
    if model is None:
        return None

    main_a = _junction._heading(incidents[first].direction)
    main_b = _junction._heading(incidents[second].direction)
    branch_heading = _junction._heading(incidents[branch].direction)
    fits = []
    for actual_zero, actual_180 in ((main_a, main_b), (main_b, main_a)):
        pairs = (
            (0.0, actual_zero),
            (180.0, actual_180),
            (MEASURED_T_BRANCH_LOCAL_HEADING_DEGREES, branch_heading),
        )
        rotation, maximum_error = _junction._best_rotation(pairs)
        fits.append((maximum_error, rotation))
    maximum_error, rotation = min(fits)
    if maximum_error > INSPECTOR_NATIVE_CONNECTOR_TOLERANCE_DEGREES + 1.0e-9:
        return None
    return _junction._NativeJunction(model, rotation, maximum_error, main_family)


def _measured_native_junction_object(old, native, elevations, spec):
    """Put the measured logical intersection, rather than the P3D origin, on node."""

    node = (float(old.x), float(old.z))
    radius = float(_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES)
    angle = math.radians(float(native.heading_degrees))
    direction = (math.sin(angle), math.cos(angle))
    start = (
        node[0] - direction[0] * radius,
        node[1] - direction[1] * radius,
    )
    end = (
        node[0] + direction[0] * radius,
        node[1] + direction[1] * radius,
    )
    fitted = _p._road_object_on_slope(
        int(old.object_id),
        native.model_path,
        start,
        end,
        elevations,
        spec,
        vertical_offset=(
            _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
            + _junction.NATIVE_JUNCTION_VERTICAL_BIAS_METRES
        ),
    )

    local_center = _geometry.native_junction_intersection_offset(native.model_path)
    if local_center is None:
        return fitted
    offset = _geometry.rotate_local(local_center, float(native.heading_degrees))
    return replace(
        fitted,
        x=node[0] - float(offset[0]),
        z=node[1] - float(offset[1]),
        heading_degrees=float(native.heading_degrees) % 360.0,
    )


def _trim_one_native_center(objects, cap, *, cap_count, elevations, spec, next_id):
    """Terminate stock approaches at the connector that actually owns their arm."""

    center = _paved._logical_center(cap)
    if center is None:
        return objects, next_id, False
    connectors = _surface._native_cap_connectors(cap)
    if not connectors:
        return objects, next_id, False

    radius = float(_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES)
    output = list(objects[:cap_count])
    changed = False
    for obj in objects[cap_count:]:
        match = _geometry.stock_straight_match(str(obj.model_path))
        if match is None:
            output.append(obj)
            continue
        family = match.group("family").casefold()
        if family not in _NATIVE_OWNED_STOCK_FAMILIES:
            output.append(obj)
            continue
        axis = _ownership._physical_straight_axis(obj)
        if axis is None or (
            _p._point_segment_distance(center, axis[0], axis[1])
            > _ownership._NATIVE_AXIS_INTRUSION_METRES
        ):
            output.append(obj)
            continue

        outside = [
            endpoint
            for endpoint in axis
            if math.dist(center, endpoint)
            > radius + _ownership._NATIVE_FOOTPRINT_MARGIN_METRES
        ]
        if not outside:
            changed = True
            continue

        replacements = []
        candidate_next_id = next_id
        first_id_available = int(obj.object_id)
        matched_any = False
        failed_matched_span = False
        for outer in outside:
            connector = _ownership._matching_connector(
                connectors, family, center, outer
            )
            if connector is None:
                # A mixed T has only one ces branch connector. A stale ces
                # through-piece on the other side is inside the wrong arm.
                continue
            matched_any = True
            built = _ownership._build_stock_span(
                family,
                connector,
                outer,
                first_object_id=first_id_available,
                next_object_id=candidate_next_id,
                elevations=elevations,
                spec=spec,
            )
            if built is None:
                failed_matched_span = True
                break
            pieces, candidate_next_id = built
            if pieces:
                replacements.extend(pieces)
                first_id_available = candidate_next_id
                candidate_next_id += 1

        if failed_matched_span or not matched_any:
            output.append(obj)
            continue
        output.extend(replacements)
        next_id = candidate_next_id
        changed = True

    return output, next_id, changed


def _coherent_candidate_bend(points) -> tuple[int, float] | None:
    """Return a safe paved bend sign for the final Inspector curve search."""

    sign = 0
    total = 0.0
    count = 0
    for previous, point, following in zip(points, points[1:], points[2:]):
        turn = float(_sharp._signed_turn(previous, point, following))
        magnitude = abs(turn)
        if magnitude < 0.35:
            continue
        if magnitude > 35.0:
            return None
        current_sign = 1 if turn > 0.0 else -1
        if sign and current_sign != sign:
            if magnitude <= 1.50:
                continue
            return None
        if not sign:
            sign = current_sign
        total += turn
        count += 1

    magnitude = abs(total)
    if (
        sign == 0
        or count < 1
        or magnitude < INSPECTOR_CURVE_MINIMUM_TURN_DEGREES
        or magnitude > INSPECTOR_CURVE_MAXIMUM_TURN_DEGREES
    ):
        return None
    return sign, magnitude


def _candidate_exact_curve_chain(
    measure,
    pieces,
    *,
    start_distance,
    preferred_end_distance,
    minimum_end_distance,
    maximum_end_distance,
):
    """Final paved-only exact-chain attempt for Inspector miter/transition candidates."""

    if _ORIGINAL_PIECE_CHAIN is None:
        raise RuntimeError("Road Inspector candidate curve policy is not installed")

    baseline = _ORIGINAL_PIECE_CHAIN(
        measure,
        pieces,
        start_distance=start_distance,
        preferred_end_distance=preferred_end_distance,
        minimum_end_distance=minimum_end_distance,
        maximum_end_distance=maximum_end_distance,
    )
    if _sharp._paved_family(pieces) is None:
        return baseline
    if float(measure.total) > 180.0:
        return baseline

    bend = _coherent_candidate_bend(measure.points)
    if bend is None:
        return baseline
    turn_sign, _total_turn = bend

    start = max(0.0, min(float(measure.total), float(start_distance)))
    end = max(start, min(float(measure.total), float(preferred_end_distance)))
    if end <= start + 1.0:
        return baseline

    source_points, entry_heading, source_exit_heading = _exact._measure_slice(
        measure, start, end
    )
    stock_exit_heading = _exact._quantised_stock_exit_heading(
        entry_heading,
        source_exit_heading,
        turn_sign,
    )

    # If the end is exposed, preserve the existing tight exit-heading gate.
    end_cover = float(measure.total) - float(preferred_end_distance)
    if (
        end_cover < 0.40
        and _p._heading_difference(stock_exit_heading, source_exit_heading) > 1.50
    ):
        return baseline

    locked_path = _sharp._beam_stock_path(
        source_points,
        turn_sign,
        entry_heading,
        stock_exit_heading,
        pieces,
    )
    if locked_path is None:
        return baseline
    exact = _exact._recover_exact_actions(locked_path, pieces, turn_sign)
    if exact is None or _exact._curve_count(exact) < 1:
        return baseline
    if len(exact) > len(baseline) + INSPECTOR_CURVE_MAXIMUM_EXTRA_PIECES:
        return baseline

    exact_tangent = _curve_usage._maximum_internal_tangent_error(exact, turn_sign)
    if exact_tangent > 1.0e-4:
        return baseline
    baseline_tangent = (
        _curve_usage._maximum_internal_tangent_error(baseline, turn_sign)
        if baseline
        else math.inf
    )
    baseline_curves = _exact._curve_count(baseline)
    exact_curves = _exact._curve_count(exact)
    baseline_short = _exact._baseline_short_straights(baseline)

    # This is the key difference from the older promotion layer: an exact chain
    # may replace a baseline with the same curve count when the old curve/straight
    # transition itself is the Inspector finding.
    if (
        exact_curves <= baseline_curves
        and baseline_tangent <= INSPECTOR_CURVE_TRANSITION_ERROR_DEGREES
        and baseline_short < 2
    ):
        return baseline

    end_projection = _sharp._nearest_forward(
        measure,
        locked_path[-1],
        start,
        float(maximum_end_distance),
    )
    if end_projection is None:
        return baseline
    if end_projection[0] > _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES + 1.0e-9:
        return baseline
    if end_projection[1] < float(minimum_end_distance) - 0.20:
        return baseline
    start_point = measure.point(start)[:2]
    if math.dist(locked_path[0], start_point) > 1.0e-6:
        return baseline
    return exact


def _physical_piece_endpoints(obj):
    straight = _geometry.stock_straight_match(str(obj.model_path))
    if straight is not None:
        length = float(
            _geometry.STOCK_STRAIGHT_LENGTHS_METRES[int(straight.group("length"))]
        )
        return (
            _surface._world_point(obj, (0.0, -length * 0.5)),
            _surface._world_point(obj, (0.0, length * 0.5)),
        )
    curve = _geometry.stock_curve_connectors(str(obj.model_path))
    if curve is None:
        return None
    return (
        _surface._world_point(obj, curve.begin),
        _surface._world_point(obj, curve.end),
    )


def _stock_family(model_path: str) -> str | None:
    match = _geometry.stock_straight_match(str(model_path))
    if match is not None:
        return match.group("family").casefold()
    match = _geometry.stock_curve_match(str(model_path))
    return match.group("family").casefold() if match is not None else None


def _nearest_incident_endpoint_distance(objects, node, incident) -> float | None:
    family = incident.family
    if family not in _PAVED_FAMILIES:
        return None
    incident_direction = _normalised(incident.direction)
    best = math.inf
    for obj in objects:
        if _stock_family(str(obj.model_path)) != family:
            continue
        endpoints = _physical_piece_endpoints(obj)
        if endpoints is None:
            continue
        for endpoint, other in ((endpoints[0], endpoints[1]), (endpoints[1], endpoints[0])):
            distance = math.dist(node, endpoint)
            if distance > JUNCTION_TONGUE_MAXIMUM_ENDPOINT_DISTANCE_METRES:
                continue
            outward = _normalised((other[0] - node[0], other[1] - node[1]))
            alignment = (
                outward[0] * incident_direction[0]
                + outward[1] * incident_direction[1]
            )
            if alignment < JUNCTION_TONGUE_ALIGNMENT_COSINE:
                continue
            best = min(best, distance)
    return None if not math.isfinite(best) else best


def _low_incident_tongue(object_id, node, incident, elevations, spec):
    family = incident.family
    if family not in _PAVED_FAMILIES:
        return None
    direction = _normalised(incident.direction)
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
    end = (
        float(node[0]) + direction[0] * length,
        float(node[1]) + direction[1] * length,
    )
    return _p._road_object_on_slope(
        int(object_id),
        rf"o\road\{family}6.p3d",
        (float(node[0]), float(node[1])),
        end,
        elevations,
        spec,
        vertical_offset=(
            _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
            + JUNCTION_TONGUE_VERTICAL_BIAS_METRES
        ),
    )


def _add_low_fallback_tongues(report, dataset, projection, elevations, spec):
    """Bridge only uncovered paved fallback arms, below the visible approaches."""

    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    if cap_count <= 0:
        return report
    incident_map = _finish._junction_incident_map(dataset, projection, spec)
    if not incident_map:
        return report

    caps = list(report.objects[:cap_count])
    chains = list(report.objects[cap_count:])
    additions = []
    next_id = max((int(obj.object_id) for obj in report.objects), default=0) + 1
    seen = set()

    for cap in caps:
        if _paved._native_signature(str(cap.model_path)) is not None:
            continue
        match = _geometry.stock_straight_match(str(cap.model_path))
        if match is None or int(match.group("length")) != 6:
            continue
        cap_family = match.group("family").casefold()
        if cap_family not in _PAVED_FAMILIES:
            continue

        junction = _paved._matching_junction(
            incident_map, (float(cap.x), float(cap.z))
        )
        if junction is None:
            continue
        node, incidents = junction
        if len(incidents) not in {3, 4, 5}:
            continue
        if not any(incident.family in _PAVED_FAMILIES for incident in incidents):
            continue
        if any(
            incident.family not in (_PAVED_FAMILIES | {"ces"})
            for incident in incidents
        ):
            continue

        for incident in incidents:
            if incident.family not in _PAVED_FAMILIES:
                continue
            gap = _nearest_incident_endpoint_distance(chains, node, incident)
            if gap is None or gap <= JUNCTION_TONGUE_MINIMUM_GAP_METRES:
                continue
            heading = _junction._heading(incident.direction) % 360.0
            key = (
                round(float(node[0]), 3),
                round(float(node[1]), 3),
                incident.family,
                round(heading, 2),
            )
            if key in seen:
                continue
            tongue = _low_incident_tongue(
                next_id, node, incident, elevations, spec
            )
            if tongue is None:
                continue
            additions.append(tongue)
            chains.append(tongue)
            seen.add(key)
            next_id += 1

    if not additions:
        return report
    required = len(report.objects) + len(additions)
    if (
        required > int(spec.max_road_objects)
        and not bool(getattr(spec, "advisory_object_limits", False))
    ):
        raise ValueError(
            "road object budget is too small after Inspector junction tongues: "
            f"requires {required:,} objects, limit is {int(spec.max_road_objects):,}"
        )
    return replace(
        report,
        objects=tuple((*report.objects, *additions)),
        short_piece_objects=(
            int(getattr(report, "short_piece_objects", 0)) + len(additions)
        ),
    )


def _native_owner_realign(report, dataset, projection, elevations, spec):
    """Apply the Inspector's native-or-low-fill decision at final WRP geometry."""

    if _ORIGINAL_OWNER_REALIGN is None:
        raise RuntimeError("Road Inspector candidate policy is not installed")
    report = _ORIGINAL_OWNER_REALIGN(
        report, dataset, projection, elevations, spec
    )

    cap_count = min(
        int(getattr(report, "junction_cap_objects", 0)), len(report.objects)
    )
    if cap_count <= 0:
        return report
    incident_map = _finish._junction_incident_map(dataset, projection, spec)
    if not incident_map:
        return _ownership._trim_native_center_intruders(report, elevations, spec)

    objects = list(report.objects)
    changed = False
    for index in range(cap_count):
        current = objects[index]
        logical = _paved._logical_center(current)
        if logical is None:
            continue
        junction = _paved._matching_junction(incident_map, logical)
        if junction is None:
            continue
        node, incidents = junction
        if not _eligible_paved_or_mixed_incidents(incidents):
            continue

        signature = _paved._native_signature(str(current.model_path))
        if signature is not None:
            family, local_headings = signature
            error = _paved._connector_error_degrees(
                current, incidents, local_headings
            )
            if error <= INSPECTOR_NATIVE_CONNECTOR_TOLERANCE_DEGREES + 1.0e-9:
                continue
            replacement = _paved._low_stock_cap(
                current,
                node,
                incidents,
                family,
                elevations,
                spec,
            )
            if replacement != current:
                objects[index] = replacement
                changed = True
            continue

        match = _geometry.stock_straight_match(str(current.model_path))
        if match is None or int(match.group("length")) != 6:
            continue
        family = match.group("family").casefold()
        if family not in _PAVED_FAMILIES:
            continue

        native = _junction._native_junction_for_incidents(incidents)
        if (
            native is not None
            and native.cap_family == family
            and float(native.maximum_heading_error_degrees)
            <= INSPECTOR_NATIVE_CONNECTOR_TOLERANCE_DEGREES + 1.0e-9
        ):
            aligned_cap = replace(current, x=float(node[0]), z=float(node[1]))
            objects[index] = _measured_native_junction_object(
                aligned_cap, native, elevations, spec
            )
            changed = True
            continue

        replacement = _paved._low_stock_cap(
            current,
            node,
            incidents,
            family,
            elevations,
            spec,
        )
        if replacement != current:
            objects[index] = replacement
            changed = True

    if changed:
        report = replace(report, objects=tuple(objects))
    report = _ownership._trim_native_center_intruders(report, elevations, spec)
    return _add_low_fallback_tongues(
        report, dataset, projection, elevations, spec
    )


def _apply_wedge_candidates(report, elevations, spec):
    """Do not hide Inspector turn wedges with generated or full-strip overlays."""

    if _ORIGINAL_STOCK_APPLY is None:
        raise RuntimeError("Road Inspector candidate policy is not installed")

    original_wedge_planner = _emitted._terrain_wedge_cover_plans
    original_seam_planner = _emitted._emitted_seam_cover_plans

    def non_turn_seams(current):
        return tuple(
            plan
            for plan in original_seam_planner(current)
            if (
                getattr(plan, "outer_miter_apex", None) is None
                or float(getattr(plan, "turn_degrees", 0.0))
                < INSPECTOR_CURVE_MINIMUM_TURN_DEGREES
            )
        )

    _emitted._terrain_wedge_cover_plans = lambda *_args, **_kwargs: ()
    _emitted._emitted_seam_cover_plans = non_turn_seams
    try:
        # The exact curve-chain wrapper has already had first refusal. Keep only
        # non-turn stock seam fallbacks here. A remaining turn stays reportable
        # instead of being buried under a second painted road or a generated P3D.
        return _ORIGINAL_STOCK_APPLY(report, elevations, spec)
    finally:
        _emitted._terrain_wedge_cover_plans = original_wedge_planner
        _emitted._emitted_seam_cover_plans = original_seam_planner


def install_stock_road_inspector_candidate_policy() -> None:
    """Install the paved/mixed repairs described by Road Inspector candidates."""

    global _ORIGINAL_STOCK_APPLY, _ORIGINAL_OWNER_REALIGN
    global _ORIGINAL_PIECE_CHAIN, _INSTALLED
    if _INSTALLED:
        return
    if not _stock_only._INSTALLED:
        raise RuntimeError("stock paved-only policy must install first")

    # Candidate: exact stock curve/curve-chain before any visual overlap fallback.
    _micro.MINIMUM_MICRO_BEND_TOTAL_TURN_DEGREES = (
        INSPECTOR_CURVE_MINIMUM_TURN_DEGREES
    )
    _single.MINIMUM_SINGLE_VERTEX_TURN_DEGREES = (
        INSPECTOR_CURVE_MINIMUM_TURN_DEGREES
    )
    _curve_usage._MINIMUM_TOTAL_TURN_DEGREES = (
        INSPECTOR_CURVE_MINIMUM_TURN_DEGREES
    )
    _curve_usage._MINIMUM_PROMOTED_CURVES = 1
    _curve_usage._MINIMUM_BASELINE_SHORT_STRAIGHTS = 2
    _ORIGINAL_PIECE_CHAIN = _p._stock_piece_chain
    _p._stock_piece_chain = _candidate_exact_curve_chain

    # Candidate: use a measured native model only inside its visible connector
    # tolerance. Mixed paved/ces T junctions follow the same rule as all-paved.
    _paved._all_paved_incidents = _eligible_paved_or_mixed_incidents
    _junction._native_t_junction = _measured_native_t_junction
    _junction._native_junction_object = _measured_native_junction_object

    # Candidate: one native centre owns all ordinary approach geometry, including
    # its stock ces branch.
    _ownership._trim_one_native_center = _trim_one_native_center
    _ORIGINAL_OWNER_REALIGN = _ownership._native_owner_realign
    _ownership._native_owner_realign = _native_owner_realign

    # Candidate: do not reintroduce a full road strip or generated paved wedge
    # after exact curve fitting.
    _ORIGINAL_STOCK_APPLY = _stock_only._apply_stock_emitted_seam_covers
    _emitted._apply_emitted_seam_covers = _apply_wedge_candidates

    _INSTALLED = True
