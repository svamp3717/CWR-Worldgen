# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from pathlib import Path

from cwr_worldgen import playability as _p
from cwr_worldgen import gravel_asphalt_transition_policy as _mixed
from cwr_worldgen import stock_road_inspector_candidate_policy as _candidate
from cwr_worldgen import stock_road_junction_policy as _junction
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


def test_lundby44_historical_native_candidate_is_rejected_by_inspector_tolerance() -> None:
    incidents = (
        _incident(_LUNDBY44_MAIN_EAST, "sil"),
        _incident(_LUNDBY44_MAIN_WEST, "sil"),
        _incident(_LUNDBY44_CES_BRANCH, "ces"),
    )

    # The older mixed-road fitter could geometrically manufacture a native T by
    # steering the approaches roughly fifteen degrees. Keep that measurement as
    # evidence, but the Inspector candidate says not to force a rigid mesh here.
    historical = _mixed._native_t_junction(incidents)
    assert historical is not None
    assert historical.model_path.casefold() == r"o\road\kr_new_sil_ces_t.p3d"
    assert 14.64 < historical.maximum_heading_error_degrees < 14.65
    assert historical.maximum_heading_error_degrees <= (
        _mixed.MAXIMUM_STOCK_CES_NATIVE_HEADING_ERROR_DEGREES
    )

    assert _candidate._measured_native_t_junction(incidents) is None
    assert _junction._native_junction_for_incidents(incidents) is None


def test_lundby44_geometry_keeps_turning_approaches_over_low_sil_fill() -> None:
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
    assert cap.model_path.casefold() == r"o\road\sil6.p3d"
    assert cap.y < _p._STOCK_ROAD_VERTICAL_OFFSET_METRES

    # This is intentionally the opposite of the older consolidation regression:
    # a ~14.65-degree connector disagreement must not survive as a rigid native T.
    assert all(
        obj.model_path.casefold() != r"o\road\kr_new_sil_ces_t.p3d"
        for obj in report.objects[: report.junction_cap_objects]
    )
    assert any(
        obj.model_path.casefold().startswith(r"o\road\sil")
        for obj in report.objects[report.junction_cap_objects :]
    )
    assert any(
        obj.model_path.casefold().startswith(r"o\road\ces")
        for obj in report.objects[report.junction_cap_objects :]
    )
