# SPDX-License-Identifier: GPL-3.0-or-later
"""Apply the Road Inspector's measured paved-road repair candidates.

Lundby47 made three production gaps explicit:

* T-junction fitting treated the branch as local +X even though the measured
  Memory-LOD branch connector is local -X;
* asymmetric T meshes were placed by their model origin instead of compensating
  the measured logical-intersection offset;
* the native-centre cleanup deliberately skipped stock ``ces`` approaches, so
  mixed sil/ces T junctions kept the exact under-cap roads the Inspector flagged.

The same report also showed paved outside wedges again because the later
stock-only fallback replaced the borderless paved-wedge overlay with a complete
``sil6`` strip. Keep ordinary late seam fallbacks stock-only, but follow the
Inspector candidate for terrain wedges: append only the bounded borderless
sil/asf/kos wedge polygon when the strict physical audit says terrain is exposed.

Generated gravel is not changed. Stock ``ces`` is touched only where a measured
mixed native T already owns the intersection centre.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import playability as _p
from . import stock_road_emitted_seam_policy as _emitted
from . import stock_road_junction_policy as _junction
from . import stock_road_model_geometry as _geometry
from . import stock_road_native_junction_ownership_policy as _ownership
from . import stock_road_paved_junction_completion_policy as _paved
from . import stock_road_stock_paved_only_policy as _stock_only
from . import stock_road_surface_overlap_policy as _surface
from . import stock_road_visual_finish_policy as _finish


MEASURED_T_BRANCH_LOCAL_HEADING_DEGREES = 270.0
_NATIVE_OWNED_STOCK_FAMILIES = frozenset({"sil", "asf", "kos", "ces"})

_ORIGINAL_STOCK_APPLY = None
_ORIGINAL_OWNER_REALIGN = None
_INSTALLED = False


def _measured_native_t_junction(incidents):
    """Fit a T from the actual local -X branch connector, not a guessed +X arm."""

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
    if maximum_error > _junction.MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES:
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
            # A short piece wholly inside a purpose-built junction has no visible
            # approach surface to preserve. Removing it is the candidate fix.
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
                # A T has only one branch-family connector. The opposite half of
                # a stale through-centre branch is inside the wrong side of the T
                # and is intentionally discarded.
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


def _native_owner_realign(report, dataset, projection, elevations, spec):
    """Re-run measured native selection after late fitting, then trim its centre."""

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
        if _paved._native_signature(str(current.model_path)) is not None:
            continue
        match = _geometry.stock_straight_match(str(current.model_path))
        if match is None or int(match.group("length")) != 6:
            continue
        family = match.group("family").casefold()
        if family not in _paved._PAVED_FAMILIES:
            continue

        junction = _paved._matching_junction(
            incident_map, (float(current.x), float(current.z))
        )
        if junction is None:
            continue
        node, incidents = junction
        native = _junction._native_junction_for_incidents(incidents)
        if native is None or native.cap_family != family:
            continue

        aligned_cap = replace(current, x=float(node[0]), z=float(node[1]))
        objects[index] = _measured_native_junction_object(
            aligned_cap, native, elevations, spec
        )
        changed = True

    if changed:
        report = replace(report, objects=tuple(objects))
    return _ownership._trim_native_center_intruders(report, elevations, spec)


def _apply_wedge_candidates(report, elevations, spec):
    """Keep stock seam fallbacks, but use only borderless overlays for grass wedges."""

    if _ORIGINAL_STOCK_APPLY is None:
        raise RuntimeError("Road Inspector candidate policy is not installed")

    wedge_planner = _emitted._terrain_wedge_cover_plans
    # The stock-only wrapper treats terrain-wedge plans like ordinary seam plans
    # and adds a complete painted road strip. Suppress only that input while it
    # performs its normal non-wedge work, then run the strict wedge planner on
    # the resulting geometry and append only the borderless triangle.
    _emitted._terrain_wedge_cover_plans = lambda *_args, **_kwargs: ()
    try:
        base = _ORIGINAL_STOCK_APPLY(report, elevations, spec)
    finally:
        _emitted._terrain_wedge_cover_plans = wedge_planner

    wedge_plans = tuple(wedge_planner(base, elevations, spec))
    if not wedge_plans:
        return base

    native_centres = _stock_only._native_junction_centres(base)
    objects = list(base.objects)
    next_id = max((int(obj.object_id) for obj in objects), default=0) + 1
    half = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6]) * 0.5
    seen = set()
    added = 0

    for plan in wedge_plans:
        key = _stock_only._plan_key(plan)
        if key in seen:
            continue
        seen.add(key)
        if _stock_only._plan_hits_native_junction(plan, native_centres):
            continue

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
        reference = _p._road_object_on_slope(
            next_id,
            _emitted.paved_miter_model_path(
                str(getattr(spec, "name", "cwr_worldgen")),
                float(plan.turn_degrees),
            ),
            start,
            end,
            elevations,
            spec,
            vertical_offset=(
                _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
                + _emitted.EMITTED_SEAM_UNDERLAY_BIAS_METRES
            ),
        )
        overlay = _emitted._terrain_clear_wedge_overlay(
            plan,
            reference,
            next_id,
            elevations,
            spec,
            force=True,
        )
        if overlay is None:
            continue
        objects.append(overlay)
        next_id += 1
        added += 1

    if added == 0:
        return base
    required = len(objects)
    if (
        required > int(spec.max_road_objects)
        and not bool(getattr(spec, "advisory_object_limits", False))
    ):
        raise ValueError(
            "road object budget is too small after Inspector paved-wedge "
            f"coverage: requires {required:,} objects, "
            f"limit is {int(spec.max_road_objects):,}"
        )
    return replace(
        base,
        objects=tuple(objects),
        short_piece_objects=(
            int(getattr(base, "short_piece_objects", 0)) + added
        ),
    )


def install_stock_road_inspector_candidate_policy() -> None:
    """Install measured junction and borderless paved-wedge candidate fixes."""

    global _ORIGINAL_STOCK_APPLY, _ORIGINAL_OWNER_REALIGN, _INSTALLED
    if _INSTALLED:
        return
    if not _stock_only._INSTALLED:
        raise RuntimeError("stock paved-only policy must install first")

    _junction._native_t_junction = _measured_native_t_junction
    _junction._native_junction_object = _measured_native_junction_object

    _ownership._trim_one_native_center = _trim_one_native_center
    _ORIGINAL_OWNER_REALIGN = _ownership._native_owner_realign
    _ownership._native_owner_realign = _native_owner_realign

    _ORIGINAL_STOCK_APPLY = _stock_only._apply_stock_emitted_seam_covers
    _emitted._apply_emitted_seam_covers = _apply_wedge_candidates

    _INSTALLED = True
