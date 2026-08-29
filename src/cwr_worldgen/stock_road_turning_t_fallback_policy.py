# SPDX-License-Identifier: GPL-3.0-or-later
"""Keep strongly turning paved T nodes on the low fallback-cap path.

A Resistance T junction is rigid: its two main connectors are exactly opposite.
The balanced turning-T chooser can split a modest source-road bend across that
mesh, but Lundby23 demonstrates that accepting a 20.66-degree through-road bend
still leaves the native connectors roughly one to two metres from the fitted
approaches. At that point the existing fallback is visually safer: the actual
approach pieces remain the top road surface and the intersection-edge policy uses
only low same-family fill underneath them.

There are two native-T entry paths in the layered fitter. The late skew chooser
can promote a legacy cap, but the earlier stock-junction policy may already have
replaced the cap before the late chooser runs. Merely tightening the chooser is
therefore insufficient. This policy also audits the final cap object after skew
placement and demotes an already-native same-family T when the measured source
through road bends beyond the accepted limit.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import playability as _p
from . import stock_road_junction_policy as _junction
from . import stock_road_model_geometry as _geometry
from . import stock_road_skew_orientation_policy as _skew
from . import stock_road_visual_finish_policy as _finish

MAXIMUM_BALANCED_NATIVE_MAIN_BEND_DEGREES = 15.0
MAXIMUM_NATIVE_NODE_RECOVERY_DISTANCE_METRES = 2.5

_ORIGINAL_REALIGN = None
_INSTALLED = False


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
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
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
    fixed = _p._road_object_on_slope(
        int(current.object_id),
        rf"o\road\{family}6.p3d",
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
    if _ORIGINAL_REALIGN is None:
        raise RuntimeError("turning-T fallback policy is not installed")
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
        family = _skew._same_family_for_native_t(str(current.model_path))
        if family is None:
            continue
        logical = _skew._logical_intersection(current)
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
        pair = _junction._dominant_pair(incidents)
        if pair is None:
            continue
        bend = _skew._turning_main_bend_degrees(incidents, pair)
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


def install_stock_road_turning_t_fallback_policy() -> None:
    global _ORIGINAL_REALIGN, _INSTALLED
    if _INSTALLED:
        return
    if not _skew._INSTALLED:
        raise RuntimeError("stock road skew-orientation policy must install first")

    _skew.MAXIMUM_TURNING_T_MAIN_BEND_DEGREES = (
        MAXIMUM_BALANCED_NATIVE_MAIN_BEND_DEGREES
    )
    _ORIGINAL_REALIGN = _finish._realign_legacy_caps
    _finish._realign_legacy_caps = _demote_over_bent_native_ts
    _INSTALLED = True
