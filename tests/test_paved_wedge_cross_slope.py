# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen.model import WorldObject
from cwr_worldgen import stock_road_emitted_seam_policy as _emitted
from cwr_worldgen import stock_road_visual_finish_policy as _finish


def test_apex_clear_miter_still_gets_wedge_when_base_is_buried_by_cross_slope():
    turn = 20.0
    centre = (10.0, 10.0)
    apex = (
        centre[0],
        centre[1]
        + _emitted.GENERATED_PAVED_FILL_RADIUS_METRES
        / math.cos(math.radians(turn * 0.5)),
    )
    plan = _finish._SeamCoverPlan(
        model_path=r"o\road\sil6.p3d",
        centre=centre,
        tangent_axis_degrees=0.0,
        turn_degrees=turn,
        outer_miter_apex=apex,
    )
    # At the apex, x=10 and terrain is 0.0, so this 40 mm-high miter passes the
    # old apex-only 30 mm clearance test. Across the wedge base, however, terrain
    # rises with x and buries one side of the road surface.
    low_miter = WorldObject(
        10,
        r"wg_test\i\paved_miter_q080.p3d",
        centre[0],
        0.040,
        centre[1],
        0.0,
        0.0,
    )
    spec = SimpleNamespace(name="wg_test", cells=3, cell_size=10.0)
    elevations = (
        -1.0, 0.0, 1.0,
        -1.0, 0.0, 1.0,
        -1.0, 0.0, 1.0,
    )

    overlay = _emitted._terrain_clear_wedge_overlay(
        plan,
        low_miter,
        11,
        elevations,
        spec,
    )

    assert overlay is not None
    assert overlay.model_path.casefold() == r"wg_test\i\paved_wedge_q080.p3d"
    assert overlay.y > low_miter.y
