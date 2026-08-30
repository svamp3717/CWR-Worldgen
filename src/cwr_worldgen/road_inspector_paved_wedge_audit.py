# SPDX-License-Identifier: GPL-3.0-or-later
"""Make paved grass-wedge reporting independent of the base seam thresholds.

The legacy grass-wedge layer only reclassified seam findings already emitted by
the core inspector.  A shallow but real outside triangle can therefore vanish
when its centre/edge discontinuity falls below the ordinary seam thresholds.
This final read-only audit scans physical paved endpoints directly, preserves the
same source-junction exclusion, and asks the surface-coverage layer whether the
triangle is actually hidden by visible asphalt before adding a finding.

Only ``sil``, ``asf`` and ``kos`` participate.  Dirt/gravel is intentionally
outside this audit.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from . import road_inspector as _core
from . import road_inspector_grass_wedge as _grass
from . import road_inspector_surface_coverage as _coverage


_PAVED_FAMILIES = frozenset({"sil", "asf", "kos"})
_SCAN_TOLERANCE_METRES = _grass.MAXIMUM_GRASS_WEDGE_CENTER_GAP_METRES
_ORIGINAL_INSPECT = None
_INSTALLED = False


def _candidate_pairs(roads):
    endpoints = tuple(
        endpoint
        for road in roads
        if road.family in _PAVED_FAMILIES
        and not road.kind.startswith("junction_")
        for endpoint in road.endpoints
    )
    clusters = _core._endpoint_clusters(endpoints, _SCAN_TOLERANCE_METRES)
    pairs = []
    for cluster in clusters:
        unique = {
            (int(endpoint.object_id), int(endpoint.endpoint_index)): endpoint
            for endpoint in cluster
        }
        if len(unique) != 2:
            continue
        first, second = sorted(
            unique.values(),
            key=lambda endpoint: (int(endpoint.object_id), int(endpoint.endpoint_index)),
        )
        if first.object_id == second.object_id or first.family != second.family:
            continue
        pairs.append((first, second))
    return tuple(pairs)


def _provisional_issue(first, second):
    center_gap = math.dist(first.point, second.point)
    tangent_error = _core._axis_heading_difference(
        first.tangent_axis_degrees,
        second.tangent_axis_degrees,
    )
    edge_max, edge_min, edge_mean = _core._edge_discontinuity(first, second)
    score = _core._score_geometry(
        center_gap=center_gap,
        edge_gap=edge_max,
        tangent_error=tangent_error,
    )
    return _core.RoadIssue(
        issue_id="",
        severity=_core._severity(score),
        score=score,
        category="connector_gap",
        x=(float(first.point[0]) + float(second.point[0])) * 0.5,
        z=(float(first.point[1]) + float(second.point[1])) * 0.5,
        object_ids=tuple(sorted((int(first.object_id), int(second.object_id)))),
        models=(first.model_path, second.model_path),
        message="Direct paved outside-miter audit candidate.",
        candidate_fix="Refit the paved seam so its physical outside edges remain continuous.",
        metrics={
            "center_gap_metres": round(center_gap, 5),
            "tangent_error_degrees": round(tangent_error, 5),
            "edge_gap_max_metres": round(edge_max, 5),
            "edge_gap_min_metres": round(edge_min, 5),
            "edge_gap_mean_metres": round(edge_mean, 5),
        },
    )


def _scan_missing_grass_wedges(result, source_junctions, match_tolerance, terrain=None):
    existing_pairs = {
        tuple(sorted(int(value) for value in issue.object_ids))
        for issue in result.issues
        if issue.category == "grass_wedge" and len(issue.object_ids) == 2
    }
    additions = []
    for first, second in _candidate_pairs(result.road_objects):
        pair = tuple(sorted((int(first.object_id), int(second.object_id))))
        if pair in existing_pairs:
            continue
        midpoint = (
            (float(first.point[0]) + float(second.point[0])) * 0.5,
            (float(first.point[1]) + float(second.point[1])) * 0.5,
        )
        if _grass._near_source_junction(midpoint, source_junctions, match_tolerance):
            continue
        if _grass._grass_wedge_geometry(first, second) is None:
            continue

        provisional = _provisional_issue(first, second)
        if _coverage._covered_by_other_paved_surface(
            provisional,
            result.road_objects,
            terrain,
        ):
            continue
        classified = _grass._classify_grass_wedge(
            provisional,
            result.road_objects,
            source_junctions,
            match_tolerance,
        )
        if classified.category != "grass_wedge":
            continue
        additions.append(classified)
        existing_pairs.add(pair)

    if not additions:
        return result
    return replace(
        result,
        issues=_core._number_issues(tuple(result.issues) + tuple(additions)),
    )


def inspect_road_geometry(
    input_path: Path,
    *,
    roads_geojson: Path | None = None,
    endpoint_tolerance: float = _core.DEFAULT_ENDPOINT_TOLERANCE_METRES,
    minimum_edge_gap: float = _core.DEFAULT_MINIMUM_EDGE_GAP_METRES,
    minimum_tangent_error: float = _core.DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES,
    junction_match_tolerance: float = _core.DEFAULT_JUNCTION_MATCH_TOLERANCE_METRES,
):
    if _ORIGINAL_INSPECT is None:
        raise RuntimeError("road inspector paved-wedge audit is not installed")
    result = _ORIGINAL_INSPECT(
        input_path,
        roads_geojson=roads_geojson,
        endpoint_tolerance=endpoint_tolerance,
        minimum_edge_gap=minimum_edge_gap,
        minimum_tangent_error=minimum_tangent_error,
        junction_match_tolerance=junction_match_tolerance,
    )
    source_junctions = _core._source_junctions(roads_geojson) if roads_geojson else ()
    terrain = _coverage._terrain_context(Path(input_path))
    return _scan_missing_grass_wedges(
        result,
        source_junctions,
        junction_match_tolerance,
        terrain,
    )


def install() -> None:
    global _ORIGINAL_INSPECT, _INSTALLED
    if _INSTALLED:
        return
    # Connector-gap seams are legitimate grass-wedge candidates too; the unit
    # test for the original classifier already expects this behaviour.
    _grass._SEAM_CATEGORIES = frozenset(
        set(_grass._SEAM_CATEGORIES) | {"connector_gap"}
    )
    _ORIGINAL_INSPECT = _core.inspect_road_geometry
    _core.inspect_road_geometry = inspect_road_geometry
    _INSTALLED = True
