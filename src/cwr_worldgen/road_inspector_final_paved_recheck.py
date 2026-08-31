# SPDX-License-Identifier: GPL-3.0-or-later
"""Revalidate final paved grass-wedge findings against the actual WRP surfaces.

The early Inspector classifier can label a stock seam as ``grass_wedge`` before
later layers have parsed an embedded historical wedge helper or recognised a
Kodiak-style overlap.  The direct paved audit used to skip any pair that was
already labelled, so a false positive could survive even when the final physical
surface audit proved that no terrain was visible.

New production worlds no longer emit generated road models at all.  This layer
is still useful for old PBOs and for stock-overlap seams: an existing finding is
kept unless the complete outside triangle is physically covered by the real
embedded/stock surfaces and, when terrain is available, those surfaces are
visibly above it.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from . import road_inspector as _core
from . import road_inspector_embedded_paved_geometry as _embedded
from . import road_inspector_grass_wedge as _grass
from . import road_inspector_paved_wedge_audit as _audit
from . import road_inspector_surface_coverage as _coverage


_ORIGINAL_INSPECT = None
_INSTALLED = False
_MISSING = object()


def _actual_wedge_contains_factory(input_path: Path):
    footprints = _embedded._embedded_wedge_footprints(Path(input_path))
    fallback = _core._paved_wedge_contains

    def contains(road, point, *, margin: float = 0.0):
        filename = _embedded._basename(road.model_path)
        triangle = footprints.get(filename, _MISSING)
        if triangle is _MISSING:
            return fallback(road, point, margin=margin)
        if triangle is None:
            return False
        return _embedded._embedded_wedge_contains(
            road,
            point,
            triangle,
            margin=margin,
        )

    return contains


def _recheck_existing_grass_wedges(result, input_path: Path):
    grass_pairs = {
        tuple(sorted(int(value) for value in issue.object_ids))
        for issue in result.issues
        if issue.category == "grass_wedge" and len(issue.object_ids) == 2
    }
    if not grass_pairs:
        return result

    candidates = {}
    for first, second in _audit._candidate_pairs(result.road_objects):
        pair = tuple(sorted((int(first.object_id), int(second.object_id))))
        if pair in grass_pairs:
            candidates[pair] = (first, second)
    if not candidates:
        return result

    terrain = _coverage._terrain_context(Path(input_path))
    original_contains = _core._paved_wedge_contains
    _core._paved_wedge_contains = _actual_wedge_contains_factory(Path(input_path))
    try:
        covered_pairs = set()
        for pair, (first, second) in candidates.items():
            geometry = _grass._grass_wedge_geometry(first, second)
            if geometry is None:
                continue
            if _audit._strictly_covered_by_other_paved_surface(
                first,
                second,
                geometry,
                result.road_objects,
                terrain,
            ):
                covered_pairs.add(pair)
    finally:
        _core._paved_wedge_contains = original_contains

    if not covered_pairs:
        return result
    remaining = tuple(
        issue
        for issue in result.issues
        if not (
            issue.category == "grass_wedge"
            and len(issue.object_ids) == 2
            and tuple(sorted(int(value) for value in issue.object_ids)) in covered_pairs
        )
    )
    return replace(result, issues=_core._number_issues(remaining))


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
        raise RuntimeError("final paved Inspector recheck is not installed")
    result = _ORIGINAL_INSPECT(
        input_path,
        roads_geojson=roads_geojson,
        endpoint_tolerance=endpoint_tolerance,
        minimum_edge_gap=minimum_edge_gap,
        minimum_tangent_error=minimum_tangent_error,
        junction_match_tolerance=junction_match_tolerance,
    )
    return _recheck_existing_grass_wedges(result, Path(input_path))


def install() -> None:
    global _ORIGINAL_INSPECT, _INSTALLED
    if _INSTALLED:
        return
    if not _embedded._INSTALLED:
        raise RuntimeError("embedded paved geometry audit must install first")
    _ORIGINAL_INSPECT = _core.inspect_road_geometry
    _core.inspect_road_geometry = inspect_road_geometry
    _INSTALLED = True
