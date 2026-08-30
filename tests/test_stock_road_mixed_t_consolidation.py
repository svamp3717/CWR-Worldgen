# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from pathlib import Path

from cwr_worldgen import playability as _p
from cwr_worldgen import gravel_asphalt_transition_policy as _mixed
from cwr_worldgen import stock_road_junction_policy as _junction
from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature


_LUNDBY44_MAIN_EAST = 73.0702750877
_LUNDBY44_MAIN_WEST = 264.4737193943
_LUNDBY44_CES_BRANCH = 192.3600816056


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(float(heading_degrees))
    return math.sin(angle), math.cos(angle)


def _incident(heading_degrees: float, family: str):
    return _junction._Incident(
        _direction(heading_degrees),
        family,
        rf"o\road\{family}25.p3d",
    )


def _point(node: tuple[float, float], heading_degrees: float, distance: float):
    direction = _direction(heading_degrees)
    return (
        node[0] + direction[0] * distance,
        node[1] + direction[1] * distance,
    )


def test_lundby44_stock_ces_t_prefers_one_native_mixed_junction() -> None:
    incidents = (
        _incident(_LUNDBY44_MAIN_EAST, "sil"),
        _incident(_LUNDBY44_MAIN_WEST, "sil"),
        _incident(_LUNDBY44_CES_BRANCH, "ces"),
    )

    native = _mixed._native_t_junction(incidents)

    assert native is not None
    assert native.model_path.casefold() == r"o\road\kr_new_sil_ces_t.p3d"
    assert 14.64 < native.maximum_heading_error_degrees < 14.65
    assert native.maximum_heading_error_degrees <= (
        _mixed.MAXIMUM_STOCK_CES_NATIVE_HEADING_ERROR_DEGREES
    )
    # The existing connector policy then has permission to steer the short
    # hidden approach segment onto this one rigid native T.
    assert _mixed._relaxation_eligible(incidents)


def test_lundby44_geometry_emits_native_t_instead_of_visible_sil6_cap() -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    projection = BboxProjection.create(bbox, 1000.0)
    node = (500.0, 500.0)

    west = _point(node, _LUNDBY44_MAIN_WEST, 100.0)
    east = _point(node, _LUNDBY44_MAIN_EAST, 100.0)
    branch_end = _point(node, _LUNDBY44_CES_BRANCH, 100.0)

    main = OsmLineFeature(
        "way/lundby44-main",
        {"highway": "tertiary", "surface": "asphalt"},
        tuple(projection.to_latlon(point) for point in (west, node, east)),
    )
    branch = OsmLineFeature(
        "way/lundby44-track",
        {"highway": "track"},
        tuple(projection.to_latlon(point) for point in (node, branch_end)),
    )
    dataset = OsmDataset(
        source_generator="lundby44-mixed-t",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(main, branch),
    )
    spec = _Milestone9PlayabilitySpec(
        name="lundby44_mixed_t",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=40,
        cell_size=25.0,
        max_road_objects=10000,
        strict_assets=False,
    )

    report = _p.fit_road_objects(dataset, projection, [0.0] * (40 * 40), spec)

    assert report.junction_cap_objects == 1
    cap = report.objects[0]
    assert cap.model_path.casefold() == r"o\road\kr_new_sil_ces_t.p3d"

    # The visible centre is one purpose-built T P3D. A final seam pass must not
    # sneak a second co-centred sil6 rectangle back under/through that junction.
    assert all(
        not (
            obj.model_path.casefold() == r"o\road\sil6.p3d"
            and math.dist((obj.x, obj.z), node) < 0.25
        )
        for obj in report.objects
    )

    # Stock ces must terminate at the native T's measured connector rather than
    # running all the way to the logical node underneath the junction. The old
    # overlay rule did exactly that and left a second visible strip across the T.
    nearest_ces_endpoint = math.inf
    for obj in report.objects[report.junction_cap_objects :]:
        match = _geometry.stock_straight_match(str(obj.model_path))
        if match is None or match.group("family").casefold() != "ces":
            continue
        length = float(
            _geometry.STOCK_STRAIGHT_LENGTHS_METRES[int(match.group("length"))]
        )
        axis = _p._model_axis(obj, length)
        nearest_ces_endpoint = min(
            nearest_ces_endpoint,
            math.dist(node, axis[0]),
            math.dist(node, axis[1]),
        )

    assert math.isfinite(nearest_ces_endpoint)
    assert nearest_ces_endpoint > 5.5
