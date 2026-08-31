# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

# Import the stable entry first so the read-only Inspector layer stack is
# installed exactly as it is for the CLI/GUI, including the Kodiak overlap pass.
from cwr_worldgen import road_inspector_entry as _entry  # noqa: F401
from cwr_worldgen import road_inspector as _core
from cwr_worldgen import road_inspector_grass_wedge as _grass
from cwr_worldgen import road_inspector_kodiak_overlap as _kodiak
from cwr_worldgen import road_inspector_paved_wedge_audit as _audit


def _straight(object_id: int, start, heading_degrees: float):
    length = 6.25
    angle = math.radians(heading_degrees)
    direction = (math.sin(angle), math.cos(angle))
    end = (
        float(start[0]) + direction[0] * length,
        float(start[1]) + direction[1] * length,
    )
    center = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
    endpoints = (
        _core._endpoint(
            object_id=object_id,
            model_path=r"o\road\sil6.p3d",
            family="sil",
            kind="straight",
            endpoint_index=0,
            point=start,
            tangent_axis_degrees=heading_degrees,
            outward_heading_degrees=heading_degrees + 180.0,
        ),
        _core._endpoint(
            object_id=object_id,
            model_path=r"o\road\sil6.p3d",
            family="sil",
            kind="straight",
            endpoint_index=1,
            point=end,
            tangent_axis_degrees=heading_degrees,
            outward_heading_degrees=heading_degrees,
        ),
    )
    road = _core.RoadObject(
        object_id,
        r"o\road\sil6.p3d",
        center[0],
        0.10,
        center[1],
        heading_degrees,
        0.0,
        "sil",
        "straight",
        length,
        center,
        endpoints,
    )
    return road


def test_kodiak_overlap_scan_range_is_installed() -> None:
    assert _kodiak._INSTALLED
    assert math.isclose(
        _audit._SCAN_TOLERANCE_METRES,
        _kodiak.KODIAK_PAVED_SEAM_SCAN_METRES,
        abs_tol=1.0e-12,
    )
    assert math.isclose(
        _grass.MAXIMUM_GRASS_WEDGE_CENTER_GAP_METRES,
        _kodiak.KODIAK_PAVED_SEAM_SCAN_METRES,
        abs_tol=1.0e-12,
    )


def test_intended_overlapping_straights_count_as_real_surface_cover() -> None:
    first = _straight(1, (0.0, 0.0), 0.0)
    # Start the next stock road 0.45 m before the first connector and turn only
    # one degree. This is the characteristic Kodiak longitudinal overlap, not a
    # cross-axis repair slab.
    second = _straight(2, (0.0, 5.80), 1.0)
    first_endpoint = first.endpoints[1]
    second_endpoint = second.endpoints[0]
    geometry = _grass._grass_wedge_geometry(first_endpoint, second_endpoint)
    assert geometry is not None

    assert _kodiak._strictly_covered_by_other_paved_surface(
        first_endpoint,
        second_endpoint,
        geometry,
        (first, second),
        None,
    )


def test_stock_curve_ribbon_is_accepted_as_physical_cover() -> None:
    geometry = _core._geometry.stock_curve_connectors(r"o\road\sil10 100.p3d")
    assert geometry is not None
    radius = float(geometry.radius_metres)
    center = (float(geometry.begin[0]) + radius, float(geometry.begin[1]))
    angle = math.radians(175.0)
    local = (
        center[0] + math.cos(angle) * radius,
        center[1] + math.sin(angle) * radius,
    )
    road = _core.RoadObject(
        3,
        r"o\road\sil10 100.p3d",
        0.0,
        0.10,
        0.0,
        0.0,
        0.0,
        "sil",
        "curve",
        float(geometry.chord_length_metres),
        (0.0, 0.0),
        (),
    )

    assert _kodiak._strict_curve_contains(road, local)
