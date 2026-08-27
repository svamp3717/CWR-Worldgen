# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from pathlib import Path

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_junction_policy as _junction
from cwr_worldgen import stock_road_skew_orientation_policy as _skew
from cwr_worldgen import stock_road_visual_finish_policy as _finish
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(heading_degrees)
    return math.sin(angle), math.cos(angle)


def _incident(heading_degrees: float, family: str = "sil"):
    return _junction._Incident(
        _direction(heading_degrees),
        family,
        rf"o\road\{family}25.p3d",
    )


def _distance_to_line(point, origin, direction) -> float:
    dx = float(point[0]) - float(origin[0])
    dz = float(point[1]) - float(origin[1])
    length = math.hypot(float(direction[0]), float(direction[1]))
    ux, uz = float(direction[0]) / length, float(direction[1]) / length
    projection = dx * ux + dz * uz
    nearest = (float(origin[0]) + ux * projection, float(origin[1]) + uz * projection)
    return math.dist((float(point[0]), float(point[1])), nearest)


def test_45_degree_t_uses_measured_branch_side_and_one_radius_slide() -> None:
    incidents = (
        _incident(90.0),
        _incident(270.0),
        _incident(45.0),
    )

    native = _skew._same_family_paved_skew_t(incidents, "sil")

    assert native is not None
    assert native.model_path == r"o\road\kr_new_sil_sil_t.p3d"
    assert math.isclose(native.heading_degrees, 90.0, abs_tol=1.0e-9)
    assert math.isclose(
        _skew._skew_t_longitudinal_shift(incidents, native),
        6.25,
        abs_tol=1.0e-9,
    )


def test_longitudinal_slide_puts_native_branch_connector_on_source_centerline() -> None:
    incidents = (
        _incident(90.0),
        _incident(270.0),
        _incident(45.0),
    )
    native = _skew._same_family_paved_skew_t(incidents, "sil")
    assert native is not None

    node = (100.0, 200.0)
    shift = _skew._skew_t_longitudinal_shift(incidents, native)
    main = _skew._unit_heading(native.heading_degrees)
    connector = _skew._unit_heading((native.heading_degrees + 270.0) % 360.0)
    center = (node[0] + main[0] * shift, node[1] + main[1] * shift)
    radius = 6.25
    connector_point = (
        center[0] + connector[0] * radius,
        center[1] + connector[1] * radius,
    )

    assert _distance_to_line(connector_point, node, incidents[2].direction) < 1.0e-9


def test_production_fit_shifts_diagonal_t_without_moving_off_main_axis() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    main = OsmLineFeature(
        "way/main",
        {"highway": "residential"},
        tuple(
            projection.to_latlon(point)
            for point in ((500.0, 300.0), (500.0, 500.0), (500.0, 700.0))
        ),
    )
    branch = OsmLineFeature(
        "way/branch",
        {"highway": "residential"},
        tuple(
            projection.to_latlon(point)
            for point in ((500.0, 500.0), (650.0, 650.0))
        ),
    )
    dataset = OsmDataset(
        source_generator="skew-t-longitudinal-alignment",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(main, branch),
    )
    spec = _Milestone9PlayabilitySpec(
        name="road_quality",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=40,
        cell_size=25.0,
        max_road_objects=10000,
        strict_assets=False,
    )

    report = _p.fit_road_objects(dataset, projection, [0.0] * (40 * 40), spec)
    cap = report.objects[0]
    assert cap.model_path.casefold() == r"o\road\kr_new_sil_sil_t.p3d"

    incident_map = _finish._junction_incident_map(dataset, projection, spec)
    source_node, incidents = next(iter(incident_map.values()))
    native = _skew._same_family_paved_skew_t(incidents, "sil")
    assert native is not None
    shift = _skew._skew_t_longitudinal_shift(incidents, native)
    main_unit = _skew._unit_heading(native.heading_degrees)
    expected_center = (
        source_node[0] + main_unit[0] * shift,
        source_node[1] + main_unit[1] * shift,
    )
    actual_center = _skew._logical_intersection(cap)
    assert actual_center is not None
    assert math.dist(actual_center, expected_center) < 1.0e-6

    # Sliding along the dominant through-road is allowed; lateral displacement is not.
    main_axis = incidents[_junction._dominant_pair(incidents)[0]].direction
    normal = (-main_axis[1], main_axis[0])
    lateral = (
        (actual_center[0] - source_node[0]) * normal[0]
        + (actual_center[1] - source_node[1]) * normal[1]
    )
    assert abs(lateral) < 1.0e-6

    connector_unit = _skew._unit_heading((native.heading_degrees + 270.0) % 360.0)
    connector_point = (
        actual_center[0] + connector_unit[0] * 6.25,
        actual_center[1] + connector_unit[1] * 6.25,
    )
    pair = _junction._dominant_pair(incidents)
    assert pair is not None
    branch_index = next(index for index in range(3) if index not in pair)
    assert _distance_to_line(
        connector_point, source_node, incidents[branch_index].direction
    ) < 1.0e-6
