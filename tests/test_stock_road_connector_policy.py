# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen.stock_road_connector_policy import (
    MAXIMUM_APPROACH_LATERAL_RELAXATION_METRES,
    _native_t_targets,
    _relaxed_arm_point,
)
from cwr_worldgen.stock_road_junction_policy import _Incident, _NativeJunction
from cwr_worldgen.stock_road_model_geometry import STOCK_JUNCTION_CONNECTOR_RADIUS_METRES


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(heading_degrees)
    return math.sin(angle), math.cos(angle)


def test_skewed_mixed_t_gets_exact_measured_connector_targets():
    incidents = (
        _Incident(_direction(0.0), "sil", r"O\Road\sil25.p3d"),
        _Incident(_direction(150.0), "sil", r"O\Road\sil25.p3d"),
        _Incident(_direction(90.0), "ces", r"O\Road\ces25.p3d"),
    )
    native = _NativeJunction(
        r"o\road\kr_new_sil_ces_t.p3d",
        345.0,
        15.0,
        "sil",
    )

    targets = _native_t_targets(incidents, native)

    assert targets is not None
    assert len(targets) == 3
    local_targets = sorted((target - native.heading_degrees) % 360.0 for target in targets)
    assert local_targets == [0.0, 180.0, 270.0]
    # The measured Resistance T branch is model-local -X (270 degrees). Keep
    # this signed direction exact; treating it as +X would mirror the branch
    # through the junction.
    assert math.isclose(
        (targets[2] - native.heading_degrees) % 360.0,
        270.0,
        abs_tol=1.0e-9,
    )


def test_relaxed_approach_reaches_real_connector_inside_existing_road_corridor():
    node = (0.0, 0.0)
    neighbour = (0.0, 20.0)
    connector_half_extent = STOCK_JUNCTION_CONNECTOR_RADIUS_METRES

    point = _relaxed_arm_point(node, neighbour, 17.0, connector_half_extent)

    assert point is not None
    assert abs(point[0]) <= MAXIMUM_APPROACH_LATERAL_RELAXATION_METRES + 1.0e-9
    assert math.dist(node, point) > connector_half_extent
    assert math.isclose(MAXIMUM_APPROACH_LATERAL_RELAXATION_METRES, 2.0, abs_tol=1.0e-12)


def test_very_short_arm_is_not_forced_into_relaxation():
    connector_half_extent = STOCK_JUNCTION_CONNECTOR_RADIUS_METRES

    assert _relaxed_arm_point(
        (0.0, 0.0),
        (0.0, connector_half_extent + 0.1),
        15.0,
        connector_half_extent,
    ) is None
