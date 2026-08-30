# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from pathlib import Path

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_junction_policy as _junction
from cwr_worldgen import stock_road_skew_orientation_policy as _skew
from cwr_worldgen import stock_road_turning_t_fallback_policy as _turning_fallback
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


def test_45_degree_t_rejects_rigid_native_surface() -> None:
    incidents = (
        _incident(90.0),
        _incident(270.0),
        _incident(45.0),
    )

    assert _skew._same_family_paved_skew_t(incidents, "sil") is None


def test_near_orthogonal_t_keeps_measured_branch_side_and_small_slide() -> None:
    incidents = (
        _incident(90.0),
        _incident(270.0),
        _incident(10.0),
    )

    native = _skew._same_family_paved_skew_t(incidents, "sil")

    assert native is not None
    assert native.model_path == r"o\road\kr_new_sil_sil_t.p3d"
    assert math.isclose(native.heading_degrees, 90.0, abs_tol=1.0e-9)
    assert math.isclose(native.maximum_heading_error_degrees, 10.0, abs_tol=1.0e-9)
    assert math.isclose(
        _skew._skew_t_longitudinal_shift(incidents, native),
        6.25 * math.tan(math.radians(10.0)),
        abs_tol=1.0e-9,
    )


def test_lundby_turning_main_t_rejects_rigid_native_surface() -> None:
    # Real incident headings at Lundby's all-asphalt T near 3223.50/3181.50.
    # Lundby23 proves that balancing this 20.66-degree through bend over the
    # rigid T still leaves its measured connectors roughly 1-2 m from the actual
    # approach pieces. Keep the visible approaches and low fallback fill instead.
    incidents = (
        _incident(93.732),
        _incident(340.710),
        _incident(253.070),
    )

    pair = _junction._dominant_pair(incidents)
    assert pair is not None
    main_bend = _skew._turning_main_bend_degrees(incidents, pair)
    assert 20.65 <= main_bend <= 20.67
    assert main_bend > _turning_fallback.MAXIMUM_BALANCED_NATIVE_MAIN_BEND_DEGREES
    assert _skew._same_family_paved_skew_t(incidents, "sil") is None


def test_moderate_turning_main_t_can_still_use_balanced_native_surface() -> None:
    incidents = (
        _incident(84.0),
        _incident(276.0),
        _incident(0.0),
    )

    pair = _junction._dominant_pair(incidents)
    assert pair is not None
    main_bend = _skew._turning_main_bend_degrees(incidents, pair)
    assert math.isclose(main_bend, 12.0, abs_tol=1.0e-9)
    native = _skew._same_family_paved_skew_t(incidents, "sil")
    assert native is not None
    assert native.model_path == r"o\road\kr_new_sil_sil_t.p3d"


def test_turning_main_t_still_rejects_excessive_connector_error() -> None:
    incidents = (
        _incident(110.0),
        _incident(350.0),
        _incident(250.0),
    )

    assert _skew._same_family_paved_skew_t(incidents, "sil") is None


def test_longitudinal_slide_puts_accepted_branch_connector_on_source_centerline() -> None:
    incidents = (
        _incident(90.0),
        _incident(270.0),
        _incident(10.0),
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


def test_production_fit_uses_low_fallback_for_lundby_turning_main_geometry() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    node = (500.0, 500.0)

    roads = []
    for index, heading in enumerate((93.732, 340.710, 253.070)):
        direction = _direction(heading)
        endpoint = (
            node[0] + direction[0] * 80.0,
            node[1] + direction[1] * 80.0,
        )
        roads.append(
            OsmLineFeature(
                f"way/turning-{index}",
                {"highway": "tertiary", "surface": "asphalt"},
                tuple(projection.to_latlon(point) for point in (node, endpoint)),
            )
        )

    dataset = OsmDataset(
        source_generator="lundby-turning-t",
        element_count=3,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=tuple(roads),
    )
    spec = _Milestone9PlayabilitySpec(
        name="lundby_turning_t",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=40,
        cell_size=25.0,
        max_road_objects=10000,
        strict_assets=False,
    )

    report = _p.fit_road_objects(dataset, projection, [0.0] * (40 * 40), spec)
    assert report.junction_cap_objects >= 1
    cap = report.objects[0]
    assert cap.model_path.casefold() == rf"{spec.name}\i\paved_fill.p3d"

    # The borderless fallback fill remains at the actual source node while the
    # approach pieces own every visible road edge.
    assert math.dist((cap.x, cap.z), node) < 0.05
    assert all(
        obj.model_path.casefold() != r"o\road\kr_new_sil_sil_t.p3d"
        for obj in report.objects
    )


def test_production_fit_keeps_small_main_axis_fallback_for_45_degree_t() -> None:
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
        source_generator="skew-t-fallback",
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
    assert report.junction_cap_objects >= 1
    cap = report.objects[0]
    assert cap.model_path.casefold() == rf"{spec.name}\i\paved_fill.p3d"

    incident_map = _finish._junction_incident_map(dataset, projection, spec)
    source_node, incidents = next(iter(incident_map.values()))
    assert math.dist((cap.x, cap.z), source_node) < 0.05
    pair = _junction._dominant_pair(incidents)
    assert pair is not None
    main_heading = _junction._heading(incidents[pair[0]].direction)
    assert _finish._axis_heading_difference(cap.heading_degrees, main_heading) < 0.05

    # Most importantly, the visible 90-degree T slab never enters the report.
    assert all(
        obj.model_path.casefold() != r"o\road\kr_new_sil_sil_t.p3d"
        for obj in report.objects
    )
