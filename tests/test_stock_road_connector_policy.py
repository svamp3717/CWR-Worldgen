# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen.stock_road_connector_policy import (
    MAXIMUM_APPROACH_LATERAL_RELAXATION_METRES,
    _native_t_targets,
    _relaxed_arm_point,
)
from cwr_worldgen.stock_road_junction_policy import _Incident
from cwr_worldgen.stock_road_model_geometry import STOCK_JUNCTION_CONNECTOR_RADIUS_METRES
from cwr_worldgen.stock_road_skew_policy import _native_junction_with_bounded_mixed_skew


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(heading_degrees)
    return math.sin(angle), math.cos(angle)


def test_skewed_mixed_t_gets_exact_measured_connector_targets():
    incidents = (
        _Incident(_direction(0.0), "sil", r"O\Road\sil25.p3d"),
        _Incident(_direction(150.0), "sil", r"O\Road\sil25.p3d"),
        _Incident(_direction(90.0), "ces", r"wg_test\i\gravel6.p3d"),
    )
    native = _native_junction_with_bounded_mixed_skew(incidents)

    assert native is not None
    targets = _native_t_targets(incidents, native)
    assert targets is not None
    assert len(targets) == 3
    local_targets = sorted((target - native.heading_degrees) % 360.0 for target in targets)
    assert all(
        min(abs(local - value) for value in (0.0, 90.0, 180.0, 360.0)) < 1.0e-9
        for local in local_targets
    )
    # The native T matcher defines the branch on model-local +90. This must be
    # signed, not merely equal modulo 180, otherwise the branch is folded across
    # the junction and creates a large visible overlap in game.
    pair = (0, 1)
    branch = next(index for index in range(3) if index not in pair)
    assert math.isclose(
        (targets[branch] - native.heading_degrees) % 360.0,
        90.0,
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
