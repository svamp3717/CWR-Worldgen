# SPDX-License-Identifier: GPL-3.0-or-later
"""Stable command entry for the read-only post-build Road Inspector."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from . import road_inspector as _core


_ORIGINAL_SOURCE_INTERSECTION_ISSUES = _core._source_intersection_issues


def _source_intersection_issues(roads, junctions, *, match_tolerance):
    """Compare source outward headings with the direction from node into a piece.

    ``RoadEndpoint.outward_heading_degrees`` is outward from the *piece*.  At an
    approach endpoint sitting on an intersection that direction points toward
    the node, so the source road's incident direction is its opposite.  Native
    junction connectors already point outward from the junction and are left
    unchanged.
    """
    corrected = []
    for road in roads:
        if road.kind in {"junction_t", "junction_x"}:
            corrected.append(road)
            continue
        endpoints = tuple(
            replace(
                endpoint,
                outward_heading_degrees=(float(endpoint.outward_heading_degrees) + 180.0) % 360.0,
            )
            for endpoint in road.endpoints
        )
        corrected.append(replace(road, endpoints=endpoints))
    return _ORIGINAL_SOURCE_INTERSECTION_ISSUES(
        tuple(corrected),
        junctions,
        match_tolerance=match_tolerance,
    )


# Patch only the inspector's own diagnostic function. This module changes no
# generator, road fitter, terrain or object-placement behavior.
_core._source_intersection_issues = _source_intersection_issues

RoadIssue = _core.RoadIssue
InspectionResult = _core.InspectionResult
write_inspection_report = _core.write_inspection_report


def inspect_road_geometry(
    input_path: Path,
    *,
    roads_geojson: Path | None = None,
    endpoint_tolerance: float = _core.DEFAULT_ENDPOINT_TOLERANCE_METRES,
    minimum_edge_gap: float = _core.DEFAULT_MINIMUM_EDGE_GAP_METRES,
    minimum_tangent_error: float = _core.DEFAULT_MINIMUM_TANGENT_ERROR_DEGREES,
    junction_match_tolerance: float = _core.DEFAULT_JUNCTION_MATCH_TOLERANCE_METRES,
):
    return _core.inspect_road_geometry(
        input_path,
        roads_geojson=roads_geojson,
        endpoint_tolerance=endpoint_tolerance,
        minimum_edge_gap=minimum_edge_gap,
        minimum_tangent_error=minimum_tangent_error,
        junction_match_tolerance=junction_match_tolerance,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return _core.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
