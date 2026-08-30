# SPDX-License-Identifier: GPL-3.0-or-later
"""Report ordinary stock roads that remain visible through a native junction.

A purpose-built Resistance T/X owns the complete intersection surface between
its measured connectors. An ordinary sil/asf/kos/ces piece whose centreline still
crosses the logical junction centre is therefore not an approach: it is a stale
under-cap road that can expose its borders and create the layered overlap visible
in game.

This diagnostic is intentionally geometric. It does not care which policy made
the extra road and it does not hide the existing connector or catalogue issues.
"""
from __future__ import annotations

import math

from . import road_inspector as _core
from . import stock_road_model_geometry as _geometry


NATIVE_CENTRE_MATCH_METRES = 0.90
INTRUDING_AXIS_DISTANCE_METRES = 0.75
CONNECTOR_FOOTPRINT_MARGIN_METRES = 0.35

_ORIGINAL_SOURCE_INTERSECTION_ISSUES = None
_INSTALLED = False


def _native_cap(roads, node):
    candidates = [
        road
        for road in roads
        if road.kind in {"junction_t", "junction_x"}
        and math.dist(road.logical_center, node) <= NATIVE_CENTRE_MATCH_METRES
    ]
    return min(
        candidates,
        key=lambda road: (math.dist(road.logical_center, node), road.object_id),
        default=None,
    )


def _axis_intrudes_native_center(road, node) -> bool:
    if road.kind not in {"straight", "curve"} or len(road.endpoints) < 2:
        return False
    first = road.endpoints[0].point
    second = road.endpoints[1].point
    if (
        _core._point_segment_distance(node, first, second)
        > INTRUDING_AXIS_DISTANCE_METRES
    ):
        return False

    # A correct approach ends around the native connector radius and never puts
    # its axis through the logical centre. Require at least one physical endpoint
    # to penetrate well inside that footprint before calling this an overlap.
    radius = float(_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES)
    return min(math.dist(node, first), math.dist(node, second)) < (
        radius - CONNECTOR_FOOTPRINT_MARGIN_METRES
    )


def _source_intersection_issues(roads, junctions, *, match_tolerance):
    if _ORIGINAL_SOURCE_INTERSECTION_ISSUES is None:
        raise RuntimeError("native junction overlap inspector is not installed")

    issues = list(
        _ORIGINAL_SOURCE_INTERSECTION_ISSUES(
            roads,
            junctions,
            match_tolerance=match_tolerance,
        )
    )
    existing = {
        (issue.category, round(float(issue.x), 2), round(float(issue.z), 2))
        for issue in issues
    }

    for source in junctions:
        cap = _native_cap(roads, source.point)
        if cap is None:
            continue
        intruders = [
            road
            for road in roads
            if road.object_id != cap.object_id
            and _axis_intrudes_native_center(road, source.point)
        ]
        if not intruders:
            continue

        key = (
            "intersection_native_overlap",
            round(float(source.point[0]), 2),
            round(float(source.point[1]), 2),
        )
        if key in existing:
            continue
        existing.add(key)

        object_ids = tuple(sorted({cap.object_id, *(road.object_id for road in intruders)}))
        models = tuple(sorted({cap.model_path, *(road.model_path for road in intruders)}))
        score = 90.0
        issues.append(
            _core.RoadIssue(
                "",
                _core._severity(score),
                score,
                "intersection_native_overlap",
                float(source.point[0]),
                float(source.point[1]),
                object_ids,
                models,
                (
                    f"Native junction {cap.model_path} owns this intersection centre, "
                    f"but {len(intruders)} ordinary road object(s) still cross inside "
                    "its measured connector footprint."
                ),
                (
                    "Keep one purpose-built native T/X at the centre and terminate "
                    "ordinary approaches at its measured Memory-LOD connectors."
                ),
                {
                    "native_model": cap.model_path,
                    "intruding_road_count": len(intruders),
                    "connector_radius_metres": round(
                        float(_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES), 5
                    ),
                },
            )
        )
    return issues


def install() -> None:
    global _ORIGINAL_SOURCE_INTERSECTION_ISSUES, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_SOURCE_INTERSECTION_ISSUES = _core._source_intersection_issues
    _core._source_intersection_issues = _source_intersection_issues
    _INSTALLED = True
