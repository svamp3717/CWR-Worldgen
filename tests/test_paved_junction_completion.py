# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_connector_policy as _connector
from cwr_worldgen import stock_road_junction_policy as _junction
from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen import stock_road_paved_junction_completion_policy as _paved
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(float(heading_degrees))
    return math.sin(angle), math.cos(angle)


def _incident(heading_degrees: float, family: str = "sil"):
    return _junction._Incident(
        _direction(heading_degrees),
        family,
        rf"o\road\{family}25.p3d",
    )


def _feature(projection, key: str, points, *, surface: str = "asphalt"):
    return OsmLineFeature(
        key,
        {"highway": "tertiary", "surface": surface},
        tuple(projection.to_latlon(point) for point in points),
    )


def _spec(bbox):
    return _Milestone9PlayabilitySpec(
        name="paved_junction_completion",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=40,
        cell_size=25.0,
        max_road_objects=10000,
        strict_assets=False,
    )


def _stock_endpoints(obj):
    straight = _geometry.stock_straight_match(str(obj.model_path))
    if straight is not None:
        length = float(
            _geometry.STOCK_STRAIGHT_LENGTHS_METRES[int(straight.group("length"))]
        )
        return _p._model_axis(obj, length)
    curve = _geometry.stock_curve_connectors(str(obj.model_path))
    if curve is None:
        return ()
    origin = (float(obj.x), float(obj.z))
    return (
        _geometry.transform_local(curve.begin, origin, float(obj.heading_degrees)),
        _geometry.transform_local(curve.end, origin, float(obj.heading_degrees)),
    )


def test_visibly_skewed_paved_t_is_rejected_before_connector_relaxation() -> None:
    incidents = (
        _incident(0.0),
        _incident(180.0),
        _incident(276.0),
    )

    # Raw source geometry does not get a rigid T. A complete fit may still use
    # one only after the transaction moves every eligible approach onto its
    # measured connector and the strict matcher accepts the edited geometry.
    assert _junction._native_junction_for_incidents(incidents) is None


def test_nearly_exact_paved_t_can_still_use_native_connector_targets() -> None:
    incidents = (
        _incident(0.0),
        _incident(180.0),
        _incident(270.5),
    )
    native = _junction._native_junction_for_incidents(incidents)

    assert native is not None
    assert native.maximum_heading_error_degrees <= _paved.MAXIMUM_VISIBLE_NATIVE_CONNECTOR_ERROR_DEGREES
    assert _connector._native_t_targets(incidents, native) is not None


def test_dirt_or_gravel_incident_is_outside_paved_completion_policy() -> None:
    incidents = (
        _incident(0.0, "sil"),
        _incident(180.0, "sil"),
        _incident(270.0, "ces"),
    )

    assert not _paved._all_paved_incidents(incidents)


def test_skewed_side_arm_can_transactionally_align_native_t() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    node = (500.0, 500.0)
    main = _feature(
        projection,
        "way/main",
        ((500.0, 300.0), node, (500.0, 700.0)),
    )
    branch_direction = _direction(96.0)
    branch_end = (
        node[0] + branch_direction[0] * 160.0,
        node[1] + branch_direction[1] * 160.0,
    )
    branch = _feature(projection, "way/branch", (node, branch_end))
    dataset = OsmDataset(
        source_generator="paved-skew-t-transaction",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(main, branch),
    )

    report = _p.fit_road_objects(
        dataset,
        projection,
        [0.0] * (40 * 40),
        _spec(bbox),
    )

    assert report.junction_cap_objects >= 1
    cap = report.objects[0]
    assert cap.model_path.casefold() == r"o\road\kr_new_sil_sil_t.p3d"

    # The purpose-built mesh owns the centre. Ordinary approaches terminate at
    # its measured 6.25 m connector footprint instead of crossing the node.
    assert all(
        not (
            _geometry.stock_straight_match(str(obj.model_path)) is not None
            and math.dist((float(obj.x), float(obj.z)), node) < 1.0
        )
        for obj in report.objects[report.junction_cap_objects :]
    )
    approach_geometry = sorted(
        math.dist(node, endpoint)
        for obj in report.objects[report.junction_cap_objects :]
        for endpoint in _stock_endpoints(obj)
    )
    assert approach_geometry
    assert 5.5 <= approach_geometry[0] <= 7.0


def test_exact_paved_t_keeps_native_stock_junction() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    node = (500.0, 500.0)
    dataset = OsmDataset(
        source_generator="paved-exact-t",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(
            _feature(
                projection,
                "way/main",
                ((500.0, 300.0), node, (500.0, 700.0)),
            ),
            _feature(projection, "way/branch", (node, (300.0, 500.0))),
        ),
    )

    report = _p.fit_road_objects(
        dataset,
        projection,
        [0.0] * (40 * 40),
        _spec(bbox),
    )

    assert report.junction_cap_objects >= 1
    assert report.objects[0].model_path.casefold() == r"o\road\kr_new_sil_sil_t.p3d"
    assert all(
        not (
            _geometry.stock_straight_match(str(obj.model_path)) is not None
            and math.dist((float(obj.x), float(obj.z)), node) < 1.0
        )
        for obj in report.objects[report.junction_cap_objects :]
    )


def test_skewed_paved_x_exceeds_visible_native_connector_limit() -> None:
    incidents = tuple(_incident(value) for value in (0.0, 180.0, 92.0, 272.0))
    cap = SimpleNamespace(heading_degrees=0.0)
    signature = _paved._native_signature(r"o\road\kr_new_silxsil.p3d")

    assert signature is not None
    _family, local_headings = signature
    assert (
        _paved._connector_error_degrees(cap, incidents, local_headings)
        > _paved.MAXIMUM_VISIBLE_NATIVE_CONNECTOR_ERROR_DEGREES
    )
