# SPDX-License-Identifier: GPL-3.0-or-later
"""Teach Road Inspector the bounded stock-road overlap seen in ``kodiak2.wrp``.

Kodiak's paved road pieces are usually tangent-aligned but their physical
connectors commonly overlap by roughly half a metre.  That is different from a
repair slab placed across a turn: the two intended stock pieces themselves own
the seam.  The direct grass-wedge audit must therefore inspect those wider
endpoint separations and count the involved stock road surfaces as valid cover.

This module is read-only and paved-only.  It does not alter generated geometry.
"""
from __future__ import annotations

from dataclasses import replace
import math

from . import road_inspector_grass_wedge as _grass
from . import road_inspector_paved_wedge_audit as _audit
from . import road_inspector_surface_coverage as _coverage


KODIAK_PAVED_SEAM_SCAN_METRES = 1.10
_ORIGINAL_STRICT_COVERAGE = None
_ORIGINAL_CLASSIFY = None
_INSTALLED = False


def _local_point(road, point: tuple[float, float]) -> tuple[float, float] | None:
    dx = float(point[0]) - float(road.x)
    dz = float(point[1]) - float(road.z)
    heading = math.radians(float(road.heading_degrees))
    local_x = dx * math.cos(heading) - dz * math.sin(heading)
    cosine_pitch = math.cos(math.radians(float(road.pitch_degrees)))
    if abs(cosine_pitch) <= 1.0e-9:
        return None
    local_z = (dx * math.sin(heading) + dz * math.cos(heading)) / cosine_pitch
    return local_x, local_z


def _strict_curve_contains(road, point: tuple[float, float]) -> bool:
    """Return True inside the actual ten-degree stock curve ribbon."""

    if road.kind != "curve":
        return False
    geometry = _audit._core._geometry.stock_curve_connectors(road.model_path)
    if geometry is None:
        return False
    local = _local_point(road, point)
    if local is None:
        return False

    radius = float(geometry.radius_metres)
    half_width = float(_audit._core._geometry.STOCK_HALF_WIDTHS_METRES[geometry.family])
    margin = float(_audit._STRICT_SURFACE_MARGIN_METRES)
    center = (float(geometry.begin[0]) + radius, float(geometry.begin[1]))
    vector = (float(local[0]) - center[0], float(local[1]) - center[1])
    radial = math.hypot(*vector)
    if not (
        radius - half_width - margin
        <= radial
        <= radius + half_width + margin
    ):
        return False

    # The verified stock curves turn right by ten degrees from begin to end.
    # In model X/Z space the radius vector therefore rotates clockwise from 180
    # degrees to 170 degrees.
    angle = math.degrees(math.atan2(vector[1], vector[0])) % 360.0
    clockwise = (180.0 - angle) % 360.0
    angular_margin = math.degrees(margin / max(1.0, radius))
    return clockwise <= 10.0 + angular_margin or clockwise >= 360.0 - angular_margin


def _strict_surface_contains(road, point: tuple[float, float]) -> bool:
    if road.kind == "curve":
        return _strict_curve_contains(road, point)
    return _audit._strict_surface_contains(road, point)


def _visible_at_sample(road, sample, terrain) -> bool:
    if not _strict_surface_contains(road, sample):
        return False
    if terrain is None:
        return True
    return (
        _coverage._surface_height(road, sample)
        - _coverage._terrain_height(terrain, sample)
        >= _audit._MINIMUM_VISIBLE_CLEARANCE_METRES
    )


def _strictly_covered_by_other_paved_surface(
    first,
    second,
    geometry,
    roads,
    terrain,
) -> bool:
    """Accept real stock overlap before looking for a third repair surface."""

    triangle = _audit._wedge_triangle(first, second, geometry)
    if triangle is not None:
        samples = _audit.triangle_samples(*triangle)
        road_by_id = {int(road.object_id): road for road in roads}
        involved = tuple(
            road_by_id.get(value)
            for value in (int(first.object_id), int(second.object_id))
        )
        if all(road is not None for road in involved):
            # Kodiak-style overlap is healthy only if the intended pair itself
            # physically covers every sample.  This does not forgive a mere
            # connector-distance discrepancy or an unrelated overlapping slab.
            if all(
                any(_visible_at_sample(road, sample, terrain) for road in involved)
                for sample in samples
            ):
                return True

    if _ORIGINAL_STRICT_COVERAGE is None:
        raise RuntimeError("Kodiak Inspector overlap policy is not installed")
    return _ORIGINAL_STRICT_COVERAGE(
        first,
        second,
        geometry,
        roads,
        terrain,
    )


def _classify_grass_wedge(issue, roads, source_junctions, match_tolerance: float):
    if _ORIGINAL_CLASSIFY is None:
        raise RuntimeError("Kodiak Inspector overlap policy is not installed")
    classified = _ORIGINAL_CLASSIFY(
        issue,
        roads,
        source_junctions,
        match_tolerance,
    )
    if classified.category != "grass_wedge":
        return classified
    return replace(
        classified,
        candidate_fix=(
            "First refit the bend with connector-locked native stock curves. "
            "If the intended stock pieces are already tangent-aligned, a bounded "
            "longitudinal overlap of those same pieces (Kodiak-style, roughly "
            "0.45 m) is valid when their real surfaces cover the outside seam. "
            "Do not add a cross-axis road slab; use the borderless wedge only "
            "when stock geometry still leaves terrain exposed."
        ),
    )


def install() -> None:
    global _ORIGINAL_STRICT_COVERAGE, _ORIGINAL_CLASSIFY, _INSTALLED
    if _INSTALLED:
        return
    if not _audit._INSTALLED:
        raise RuntimeError("paved wedge audit must install first")

    _ORIGINAL_STRICT_COVERAGE = _audit._strictly_covered_by_other_paved_surface
    _ORIGINAL_CLASSIFY = _grass._classify_grass_wedge

    # Scan the full overlap range measured in Kodiak instead of assuming every
    # legitimate seam has coincident endpoints within 0.35 m.
    _grass.MAXIMUM_GRASS_WEDGE_CENTER_GAP_METRES = KODIAK_PAVED_SEAM_SCAN_METRES
    _audit._SCAN_TOLERANCE_METRES = KODIAK_PAVED_SEAM_SCAN_METRES
    _audit._strictly_covered_by_other_paved_surface = (
        _strictly_covered_by_other_paved_surface
    )
    _grass._classify_grass_wedge = _classify_grass_wedge
    _INSTALLED = True
