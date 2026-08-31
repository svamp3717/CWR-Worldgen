# SPDX-License-Identifier: GPL-3.0-or-later
"""Finish paved T/X junctions from their actual fitted approach geometry.

Rigid Resistance junction meshes look good only when their measured connectors
nearly match the source-road directions.  A few degrees of error is already
visible at the painted road edges, and trying to bend the approach polylines onto
that rigid mesh merely moves the defect away from the logical intersection.

Keep native ``sil/asf/kos`` T/X models when every connector is essentially exact.
For a visibly mismatched all-paved junction, keep the source-aligned fitted
approaches, replace the rigid mesh with one stock six-metre main-family piece at
the logical node, and place that piece slightly below the approaches.  The cap is
therefore central fill only; the real approach pieces own the visible road edges.
No generated paved helper P3Ds are introduced, and dirt/gravel junction behavior
is deliberately untouched.
"""
from __future__ import annotations

from dataclasses import replace
import itertools
import math

from . import playability as _p
from . import stock_road_connector_policy as _connector
from . import stock_road_junction_policy as _junction
from . import stock_road_model_geometry as _geometry
from . import stock_road_visual_finish_policy as _finish


_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})
MAXIMUM_VISIBLE_NATIVE_CONNECTOR_ERROR_DEGREES = 0.90
PAVED_JUNCTION_UNDERLAY_BIAS_METRES = -0.006
JUNCTION_NODE_RECOVERY_METRES = 0.40

_ORIGINAL_NATIVE_T_TARGETS = None
_ORIGINAL_REALIGN = None
_INSTALLED = False


def _all_paved_incidents(incidents) -> bool:
    return (
        len(incidents) in {3, 4}
        and all(incident.family in _PAVED_FAMILIES for incident in incidents)
    )


def _native_signature(model_path: str):
    normalized = str(model_path).replace("/", "\\").casefold()
    for (main, _branch), candidate in _junction._T_JUNCTION_MODELS.items():
        if str(candidate).casefold() == normalized:
            return main, (0.0, 180.0, 270.0)
    for family, candidate in _junction._X_JUNCTION_MODELS.items():
        if str(candidate).casefold() == normalized:
            return family, (0.0, 90.0, 180.0, 270.0)
    return None


def _connector_error_degrees(cap, incidents, local_headings) -> float:
    source = tuple(_junction._heading(incident.direction) for incident in incidents)
    connectors = tuple(
        (float(cap.heading_degrees) + float(local)) % 360.0
        for local in local_headings
    )
    if len(source) != len(connectors):
        return 180.0
    best = math.inf
    for assignment in itertools.permutations(connectors):
        error = max(
            _junction._angular_distance(actual, connector)
            for actual, connector in zip(source, assignment)
        )
        best = min(best, error)
    return float(best)


def _native_t_targets(incidents, native):
    """Do not bend paved source approaches toward a junction we will reject."""

    if _ORIGINAL_NATIVE_T_TARGETS is None:
        raise RuntimeError("paved junction completion policy is not installed")
    if (
        _all_paved_incidents(incidents)
        and float(native.maximum_heading_error_degrees)
        > MAXIMUM_VISIBLE_NATIVE_CONNECTOR_ERROR_DEGREES
    ):
        return None
    return _ORIGINAL_NATIVE_T_TARGETS(incidents, native)


def _logical_center(cap) -> tuple[float, float] | None:
    if _geometry.stock_straight_match(str(cap.model_path)) is not None:
        return float(cap.x), float(cap.z)
    local = _geometry.native_junction_intersection_offset(str(cap.model_path))
    if local is None:
        return None
    return _geometry.transform_local(
        local,
        (float(cap.x), float(cap.z)),
        float(cap.heading_degrees),
    )


def _matching_junction(incident_map, point: tuple[float, float]):
    direct = incident_map.get(_p._road_node_key(point))
    if direct is not None and math.dist(point, direct[0]) <= JUNCTION_NODE_RECOVERY_METRES:
        return direct
    nearest = min(
        incident_map.values(),
        key=lambda value: math.dist(point, value[0]),
        default=None,
    )
    if nearest is None or math.dist(point, nearest[0]) > JUNCTION_NODE_RECOVERY_METRES:
        return None
    return nearest


def _cap_family(cap) -> str | None:
    straight = _geometry.stock_straight_match(str(cap.model_path))
    if straight is not None and int(straight.group("length")) == 6:
        family = straight.group("family").casefold()
        return family if family in _PAVED_FAMILIES else None
    signature = _native_signature(str(cap.model_path))
    if signature is None:
        return None
    family, _local = signature
    return family if family in _PAVED_FAMILIES else None


def _dominant_heading(incidents, family: str) -> float | None:
    """Prefer the most-opposed pair in the cap family, then any through pair."""

    candidates = []
    for first in range(len(incidents)):
        if incidents[first].family != family:
            continue
        first_heading = _junction._heading(incidents[first].direction)
        for second in range(first + 1, len(incidents)):
            if incidents[second].family != family:
                continue
            second_heading = _junction._heading(incidents[second].direction)
            separation = _junction._angular_distance(first_heading, second_heading)
            candidates.append((abs(180.0 - separation), first, second))
    if candidates:
        _error, first, _second = min(candidates)
        return _junction._heading(incidents[first].direction)

    pair = _junction._dominant_pair(incidents)
    if pair is None:
        return None
    return _junction._heading(incidents[pair[0]].direction)


def _low_stock_cap(current, node, incidents, family, elevations, spec):
    heading = _dominant_heading(incidents, family)
    if heading is None:
        return current
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
    half = length * 0.5
    angle = math.radians(heading)
    direction = (math.sin(angle), math.cos(angle))
    start = (
        float(node[0]) - direction[0] * half,
        float(node[1]) - direction[1] * half,
    )
    end = (
        float(node[0]) + direction[0] * half,
        float(node[1]) + direction[1] * half,
    )
    fixed = _p._road_object_on_slope(
        int(current.object_id),
        rf"o\road\{family}6.p3d",
        start,
        end,
        elevations,
        spec,
        vertical_offset=(
            _p._STOCK_ROAD_VERTICAL_OFFSET_METRES
            + PAVED_JUNCTION_UNDERLAY_BIAS_METRES
        ),
    )
    return replace(
        fixed,
        x=float(node[0]),
        z=float(node[1]),
        heading_degrees=heading % 360.0,
    )


def _finish_paved_junctions(report, dataset, projection, elevations, spec):
    """Keep exact native junctions; demote visible paved mismatches to low fill."""

    if _ORIGINAL_REALIGN is None:
        raise RuntimeError("paved junction completion policy is not installed")
    report = _ORIGINAL_REALIGN(report, dataset, projection, elevations, spec)

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
        family = _cap_family(current)
        if family is None:
            continue
        logical = _logical_center(current)
        if logical is None:
            continue
        junction = _matching_junction(incident_map, logical)
        if junction is None:
            continue
        node, incidents = junction
        if not _all_paved_incidents(incidents):
            continue

        signature = _native_signature(str(current.model_path))
        if signature is not None:
            _main_family, local_headings = signature
            error = _connector_error_degrees(current, incidents, local_headings)
            if error <= MAXIMUM_VISIBLE_NATIVE_CONNECTOR_ERROR_DEGREES + 1.0e-9:
                continue

        replacement = _low_stock_cap(
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

    return replace(report, objects=tuple(objects)) if changed else report


def install_stock_road_paved_junction_completion_policy() -> None:
    """Install the final stock-only all-paved junction pass."""

    global _ORIGINAL_NATIVE_T_TARGETS, _ORIGINAL_REALIGN, _INSTALLED
    if _INSTALLED:
        return
    if not _finish._INSTALLED:
        raise RuntimeError("stock road visual-finish policy must install first")

    # Make the same visible-error bound authoritative during *selection*, not
    # merely during the later cap audit. The measured dispatcher resolves T/X
    # selectors dynamically, so skewed rigid junctions never get to steer the
    # approach fitter before being demoted again.
    _junction.MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES = (
        MAXIMUM_VISIBLE_NATIVE_CONNECTOR_ERROR_DEGREES
    )
    _ORIGINAL_NATIVE_T_TARGETS = _connector._native_t_targets
    _ORIGINAL_REALIGN = _finish._realign_legacy_caps
    _connector._native_t_targets = _native_t_targets
    _finish._realign_legacy_caps = _finish_paved_junctions
    _INSTALLED = True
