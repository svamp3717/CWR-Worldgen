# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen import stock_road_straight_seam_policy as _straight


def _object(object_id: int, x: float, z: float, heading: float):
    return SimpleNamespace(
        object_id=object_id,
        model_path=r"o\road\sil6.p3d",
        x=float(x),
        y=0.0,
        z=float(z),
        heading_degrees=float(heading),
        pitch_degrees=0.0,
    )


def _gapped_miter(gap_metres: float, heading: float = 8.0):
    half = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6]) * 0.5
    first = _object(1, 0.0, -half, 0.0)
    direction = (
        math.sin(math.radians(heading)),
        math.cos(math.radians(heading)),
    )
    # First endpoint is (0, 0); second starts at (gap, 0).
    second = _object(
        2,
        float(gap_metres) + direction[0] * half,
        direction[1] * half,
        heading,
    )
    return SimpleNamespace(objects=(first, second), junction_cap_objects=0)


def test_lundby_sized_paved_connector_gap_gets_underlay():
    plans = _straight._straight_seam_cover_plans(_gapped_miter(0.18))

    assert len(plans) == 1
    assert plans[0].model_path.casefold() == r"o\road\sil6.p3d"
    assert math.dist(plans[0].centre, (0.09, 0.0)) < 1.0e-9


def test_gap_outside_inspector_seam_radius_is_not_bridged():
    plans = _straight._straight_seam_cover_plans(_gapped_miter(0.21))

    assert plans == ()
