# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import stock_road_junction_policy as _junction
from cwr_worldgen.stock_road_skew_policy import (
    MAXIMUM_RELAXED_JUNCTION_HEADING_ERROR_DEGREES,
    _family_with_generated_gravel,
)


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(heading_degrees)
    return math.sin(angle), math.cos(angle)


def test_generated_gravel_is_a_dirt_connector_for_mixed_native_t():
    assert _family_with_generated_gravel(r"wg_test\i\gravel3.p3d") == "ces"
    assert _family_with_generated_gravel(r"wg_test\i\gravel6_l15.p3d") == "ces"
    assert _family_with_generated_gravel(r"O\Road\sil25.p3d") == "sil"


def test_skewed_mixed_t_uses_native_paved_gravel_junction():
    # Deliberately synthetic headings exercise a large but bounded skew without
    # encoding geometry from any user-supplied world.
    incidents = (
        _junction._Incident(_direction(280.0), "sil", r"O\Road\sil25.p3d"),
        _junction._Incident(_direction(135.0), "sil", r"O\Road\sil25.p3d"),
        _junction._Incident(_direction(40.0), "ces", r"wg_test\i\gravel3.p3d"),
    )

    native = _junction._native_junction_for_incidents(incidents)

    assert native is not None
    assert native.model_path == r"o\road\kr_new_sil_ces_t.p3d"
    assert native.cap_family == "sil"
    assert 17.0 < native.maximum_heading_error_degrees < 18.0
    assert native.maximum_heading_error_degrees <= MAXIMUM_RELAXED_JUNCTION_HEADING_ERROR_DEGREES

    lateral = 3.0 * math.sin(math.radians(native.maximum_heading_error_degrees))
    assert lateral < 1.0


def test_more_extreme_skew_still_keeps_safe_fallback():
    incidents = (
        _junction._Incident(_direction(0.0), "sil", r"O\Road\sil25.p3d"),
        _junction._Incident(_direction(145.0), "sil", r"O\Road\sil25.p3d"),
        _junction._Incident(_direction(70.0), "ces", r"wg_test\i\gravel3.p3d"),
    )

    assert _junction._native_junction_for_incidents(incidents) is None
