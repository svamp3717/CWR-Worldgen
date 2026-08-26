# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen.stock_road_junction_policy import (
    MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES,
    NATIVE_JUNCTION_VERTICAL_BIAS_METRES,
    _Incident,
    _NativeJunction,
    _native_asset_paths,
    _native_junction_for_incidents,
    _native_junction_object,
)


def _direction(heading_degrees: float) -> tuple[float, float]:
    angle = math.radians(heading_degrees)
    return math.sin(angle), math.cos(angle)


def _incident(heading_degrees: float, family: str) -> _Incident:
    return _Incident(
        _direction(heading_degrees),
        family,
        rf"O\Road\{family}25.p3d",
    )


def test_highway_t_junction_uses_native_resistance_model():
    native = _native_junction_for_incidents(
        (_incident(0.0, "sil"), _incident(180.0, "sil"), _incident(90.0, "sil"))
    )

    assert native is not None
    assert native.model_path == r"o\road\kr_new_sil_sil_t.p3d"
    assert native.cap_family == "sil"
    assert native.maximum_heading_error_degrees < 1.0e-9


def test_highway_crossroads_use_native_resistance_model():
    native = _native_junction_for_incidents(
        tuple(_incident(value, "sil") for value in (13.0, 103.0, 193.0, 283.0))
    )

    assert native is not None
    assert native.model_path == r"o\road\kr_new_silxsil.p3d"
    assert native.cap_family == "sil"
    assert native.maximum_heading_error_degrees < 1.0e-9


def test_mixed_highway_dirt_t_uses_surface_correct_native_model():
    native = _native_junction_for_incidents(
        (_incident(0.0, "sil"), _incident(180.0, "sil"), _incident(90.0, "ces"))
    )

    assert native is not None
    assert native.model_path == r"o\road\kr_new_sil_ces_t.p3d"
    assert native.cap_family == "sil"


def test_small_skew_is_shared_between_connectors_inside_road_corridor():
    # Representative mild skew: rotating the native T by 6.5 degrees makes all
    # three connector errors no greater than 6.5 degrees.
    native = _native_junction_for_incidents(
        (
            _incident(84.6, "sil"),
            _incident(264.6, "sil"),
            _incident(161.6, "sil"),
        )
    )

    assert native is not None
    assert math.isclose(native.maximum_heading_error_degrees, 6.5, abs_tol=1.0e-9)
    half_extent = 24.5 * 6.0 / 25.0 * 0.5
    lateral_deviation = half_extent * math.sin(
        math.radians(native.maximum_heading_error_degrees)
    )
    assert lateral_deviation < 0.40


def test_excessively_skewed_t_falls_back_instead_of_forcing_native_mesh():
    native = _native_junction_for_incidents(
        (_incident(0.0, "sil"), _incident(180.0, "sil"), _incident(120.0, "sil"))
    )

    assert native is None


def test_all_dirt_t_keeps_fallback_because_cwa_has_no_dirt_dirt_t_asset():
    native = _native_junction_for_incidents(
        (_incident(0.0, "ces"), _incident(180.0, "ces"), _incident(90.0, "ces"))
    )

    assert native is None


def test_native_junction_assets_are_exposed_to_strict_asset_validation():
    paths = _native_asset_paths(r"O\Road\sil25.p3d")

    assert r"o\road\kr_new_sil_sil_t.p3d" in paths
    assert r"o\road\kr_new_silxsil.p3d" in paths
    assert _native_asset_paths(r"custom\road25.p3d") == ()


def test_native_junction_uses_only_tiny_vertical_bias_over_stock_roads():
    spec = SimpleNamespace(cells=2, cell_size=10.0, road_segment_length=24.5)
    old = _p.WorldObject(7, r"o\road\sil6.p3d", 10.0, 0.060, 10.0, 0.0)
    native = _NativeJunction(
        r"o\road\kr_new_sil_sil_t.p3d", 0.0, 0.0, "sil"
    )

    obj = _native_junction_object(old, native, (0.0, 0.0, 0.0, 0.0), spec)

    assert math.isclose(
        obj.y,
        _p._STOCK_ROAD_VERTICAL_OFFSET_METRES + NATIVE_JUNCTION_VERTICAL_BIAS_METRES,
        abs_tol=1.0e-12,
    )
    assert obj.y < old.y
    assert MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES <= 7.5
