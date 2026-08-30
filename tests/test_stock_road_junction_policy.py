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
    _connector_half_extent,
    _native_asset_paths,
    _native_junction_for_incidents,
    _native_junction_object,
)
from cwr_worldgen.stock_road_model_geometry import (
    native_junction_intersection_offset,
    transform_local,
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


def test_highway_t_junction_uses_measured_negative_x_branch():
    native = _native_junction_for_incidents(
        (_incident(0.0, "sil"), _incident(180.0, "sil"), _incident(270.0, "sil"))
    )

    assert native is not None
    assert native.model_path == r"o\road\kr_new_sil_sil_t.p3d"
    assert native.cap_family == "sil"
    assert native.maximum_heading_error_degrees < 1.0e-9
    assert math.isclose(native.heading_degrees % 360.0, 0.0, abs_tol=1.0e-9)


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
        (_incident(0.0, "sil"), _incident(180.0, "sil"), _incident(270.0, "ces"))
    )

    assert native is not None
    assert native.model_path == r"o\road\kr_new_sil_ces_t.p3d"
    assert native.cap_family == "sil"


def test_nearly_aligned_mixed_highway_dirt_t_keeps_native_transition():
    native = _native_junction_for_incidents(
        (_incident(0.0, "sil"), _incident(180.0, "sil"), _incident(268.0, "ces"))
    )

    assert native is not None
    assert native.model_path == r"o\road\kr_new_sil_ces_t.p3d"
    assert native.maximum_heading_error_degrees <= 1.5


def test_visibly_skewed_mixed_highway_dirt_t_uses_stock_main_overlay():
    native = _native_junction_for_incidents(
        (_incident(0.0, "sil"), _incident(180.0, "sil"), _incident(266.0, "ces"))
    )

    assert native is not None
    assert native.model_path == r"o\road\sil6.p3d"
    assert native.cap_family == "sil"
    assert native.maximum_heading_error_degrees < 1.0e-9


def test_small_skew_is_shared_between_measured_connectors_inside_road_surface():
    native = _native_junction_for_incidents(
        (
            _incident(84.6, "sil"),
            _incident(264.6, "sil"),
            _incident(161.6, "sil"),
        )
    )

    assert native is not None
    assert math.isclose(native.maximum_heading_error_degrees, 6.5, abs_tol=1.0e-9)
    lateral_deviation = _connector_half_extent(None) * math.sin(
        math.radians(native.maximum_heading_error_degrees)
    )
    assert lateral_deviation < 0.75


def test_excessively_skewed_t_falls_back_instead_of_forcing_native_mesh():
    native = _native_junction_for_incidents(
        (_incident(0.0, "sil"), _incident(180.0, "sil"), _incident(240.0, "sil"))
    )

    assert native is None


def test_all_dirt_t_keeps_fallback_because_cwa_has_no_dirt_dirt_t_asset():
    native = _native_junction_for_incidents(
        (_incident(0.0, "ces"), _incident(180.0, "ces"), _incident(270.0, "ces"))
    )

    assert native is None


def test_native_junction_assets_are_exposed_to_strict_asset_validation():
    paths = _native_asset_paths(r"O\Road\sil25.p3d")

    assert r"o\road\kr_new_sil_sil_t.p3d" in paths
    assert r"o\road\kr_new_silxsil.p3d" in paths
    assert _native_asset_paths(r"custom\road25.p3d") == ()


def test_native_junction_origin_is_shifted_so_logical_center_stays_on_node():
    spec = SimpleNamespace(cells=2, cell_size=10.0, road_segment_length=24.5)
    node = (10.0, 10.0)
    old = _p.WorldObject(7, r"o\road\sil6.p3d", node[0], 0.060, node[1], 0.0)
    native = _NativeJunction(
        r"o\road\kr_new_sil_sil_t.p3d", 0.0, 0.0, "sil"
    )

    obj = _native_junction_object(old, native, (0.0, 0.0, 0.0, 0.0), spec)
    local_center = native_junction_intersection_offset(native.model_path)
    assert local_center is not None
    mapped_center = transform_local(local_center, (obj.x, obj.z), obj.heading_degrees)

    assert math.dist(mapped_center, node) < 1.0e-9
    assert math.isclose(_connector_half_extent(spec), 6.25, abs_tol=1.0e-12)
    assert math.isclose(
        obj.y,
        _p._STOCK_ROAD_VERTICAL_OFFSET_METRES + NATIVE_JUNCTION_VERTICAL_BIAS_METRES,
        abs_tol=1.0e-12,
    )
    assert obj.y < old.y
    assert MAXIMUM_NATIVE_JUNCTION_HEADING_ERROR_DEGREES <= 7.5
