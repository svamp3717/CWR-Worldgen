# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen.model import WorldObject
from cwr_worldgen import stock_road_curve_seam_fallback_policy as _fallback
from cwr_worldgen import stock_road_visual_finish_policy as _finish


def _report(*objects: WorldObject):
    return SimpleNamespace(objects=objects, junction_cap_objects=0)


def _lundby18_object(
    object_id: int,
    model_path: str,
    x: float,
    z: float,
    heading_degrees: float,
) -> WorldObject:
    return WorldObject(
        object_id=object_id,
        model_path=model_path,
        x=x,
        y=0.0,
        z=z,
        heading_degrees=heading_degrees,
    )


def test_lundby18_curve_to_straight_wedges_receive_paved_underlays() -> None:
    """Cover the exact two curve seams visible in the supplied Lundby18 shots."""

    # Compiled WRP objects around player positions roughly
    # (2754.47,3453.58) and (2739.82,3457.89).  Their physical connector
    # positions coincide, but the rendered tangent mismatches are about 6.01
    # and 2.56 degrees respectively.  Both are large enough to expose a grass
    # triangle across part of the 9.1 m sil carriageway.
    cases = (
        (
            _lundby18_object(
                9085,
                r"o\road\sil10 50.p3d",
                2767.2724609375,
                3454.5625,
                103.9245217859275,
            ),
            _lundby18_object(
                9086,
                r"o\road\sil6.p3d",
                2759.6630859375,
                3456.467529296875,
                277.9126483088498,
            ),
        ),
        (
            _lundby18_object(
                9087,
                r"o\road\sil10 100.p3d",
                2748.243408203125,
                3457.0087890625,
                85.29014191508348,
            ),
            _lundby18_object(
                9088,
                r"o\road\sil6.p3d",
                2736.01416015625,
                3456.868896484375,
                267.8506195775699,
            ),
        ),
    )

    assert _finish.MAXIMUM_CURVE_SEAM_TANGENT_ERROR_DEGREES == (
        _fallback.MAXIMUM_PAVED_CURVE_SEAM_TANGENT_ERROR_DEGREES
    )

    for curve, straight in cases:
        plans = _fallback._paved_curve_seam_cover_plans(_report(curve, straight))
        assert len(plans) == 1, (curve, straight, plans)
        plan = plans[0]
        assert plan.model_path.casefold() == r"o\road\sil6.p3d"

        # The helper belongs exactly at the shared physical connector, not at
        # the curve object's visual bounding-box centre.
        endpoints = _finish._seam_endpoints(_report(curve, straight))
        first, second = sorted(
            endpoints,
            key=lambda item: math.dist(item.point, plan.centre),
        )[:2]
        assert math.dist(first.point, second.point) <= _finish.CURVE_SEAM_ENDPOINT_TOLERANCE_METRES
        assert math.dist(plan.centre, first.point) <= _finish.CURVE_SEAM_ENDPOINT_TOLERANCE_METRES


def test_curve_seam_fallback_is_paved_only() -> None:
    assert _fallback._PAVED_COVER.fullmatch(r"o\road\sil6.p3d") is not None
    assert _fallback._PAVED_COVER.fullmatch(r"o\road\asf6.p3d") is not None
    assert _fallback._PAVED_COVER.fullmatch(r"o\road\kos6.p3d") is not None
    assert _fallback._PAVED_COVER.fullmatch(r"o\road\ces6.p3d") is None
