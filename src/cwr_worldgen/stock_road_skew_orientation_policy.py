# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep late skew-T replacement aligned with measured Memory-LOD geometry.

Resistance T junctions use local 0/180 degrees for the through road and local
-X (270 degrees) for the branch.  A strongly skewed source T therefore has two
independent placement requirements: choose the correct side of the model, then
slide the rigid T along the through road until its fixed branch connector lies
on the diagonal source branch centreline.

The longitudinal slide does not move the T away from the through-road axis.  It
only uses the degree of freedom that a straight main road already provides.  At
a 45-degree branch the required shift is exactly one measured connector radius,
6.25 m; a near-perpendicular branch needs almost no shift.  The fitted source
approaches continue underneath the raised junction mesh, so the shifted native
surface covers the real merge instead of leaving its tongue behind the branch.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import playability as _p
from . import stock_road_final_continuity_policy as _final
from . import stock_road_junction_policy as _junction
from . import stock_road_model_geometry as _model_geometry
from . import stock_road_visual_finish_policy as _finish

_ORIGINAL_FINAL_REALIGN = None
_INSTALLED = False


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
        # Memory LOD measurements put the T branch on local -X, not +X.
        branch_error = _junction._angular_distance(
            (rotation + 270.0) % 360.0, branch_heading
        )
        candidates.append((branch_error, main_error, rotation))

    branch_error, main_error, rotation = min(candidates)
    if main_error > _final.MAXIMUM_SKEW_T_MAIN_AXIS_ERROR_DEGREES:
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
    """Run the final cap replacement, then align skew T connectors longitudinally."""

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
