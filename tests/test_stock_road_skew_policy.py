# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math

from cwr_worldgen import stock_road_junction_policy as _junction
from cwr_worldgen.gravel_asphalt_transition_policy import (
    MAXIMUM_LAYERED_MAIN_HEADING_ERROR_DEGREES,
    _relaxation_eligible,
)
from cwr_worldgen.stock_road_skew_policy import _family_with_generated_gravel


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(heading_degrees)
    return math.sin(angle), math.cos(angle)


def test_generated_gravel_uses_dirt_family_for_connector_geometry_only():
    assert _family_with_generated_gravel(r"synthetic\i\gravel3.p3d") == "ces"
    assert _family_with_generated_gravel(r"synthetic\i\gravel6_l15.p3d") == "ces"
    assert _family_with_generated_gravel(r"O\Road\sil25.p3d") == "sil"


def test_skewed_mixed_t_keeps_visible_apron_paved():
    incidents = (
        _junction._Incident(_direction(280.0), "sil", r"O\Road\sil25.p3d"),
        _junction._Incident(_direction(135.0), "sil", r"O\Road\sil25.p3d"),
        _junction._Incident(_direction(40.0), "ces", r"synthetic\i\gravel3.p3d"),
    )

    native = _junction._native_junction_for_incidents(incidents)

    # Generated gravel borrows ces connector semantics internally, but the
    # visible mixed node is deliberately one ordinary paved short slab. It must
    # never expose a stock dirt-transition T mesh.
    assert native is not None
    assert native.model_path == r"o\road\sil6.p3d"
    assert native.cap_family == "sil"
    assert 17.0 < native.maximum_heading_error_degrees < 18.0
    assert native.maximum_heading_error_degrees <= MAXIMUM_LAYERED_MAIN_HEADING_ERROR_DEGREES
    # A straight surface overlay has no native side connector to snap the gravel
    # arm onto, so connector relaxation must remain disabled for this node.
    assert not _relaxation_eligible(incidents)


def test_more_extreme_skew_still_uses_safe_unrelaxed_paved_overlay():
    incidents = (
        _junction._Incident(_direction(0.0), "sil", r"O\Road\sil25.p3d"),
        _junction._Incident(_direction(140.0), "sil", r"O\Road\sil25.p3d"),
        _junction._Incident(_direction(70.0), "ces", r"synthetic\i\gravel3.p3d"),
    )

    native = _junction._native_junction_for_incidents(incidents)

    assert native is not None
    assert native.model_path == r"o\road\sil6.p3d"
    assert math.isclose(native.maximum_heading_error_degrees, 20.0, abs_tol=1.0e-9)
    assert not _relaxation_eligible(incidents)
