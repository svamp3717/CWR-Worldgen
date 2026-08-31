# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_3d_connector_policy as _three_d
from cwr_worldgen import stock_road_curve_usage_policy as _curve_usage
from cwr_worldgen import stock_road_inspector_candidate_policy as _candidate
from cwr_worldgen import stock_road_reference_wrp_policy as _reference


def _piece(model_path: str):
    return SimpleNamespace(model_path=model_path)


def test_reference_wrp_policy_is_active_in_production_stack() -> None:
    assert _reference._INSTALLED
    assert _p._road_object_on_slope is _reference._road_object_on_slope
    assert (
        _three_d._uses_measured_rigid_connectors
        is _reference._uses_measured_rigid_connectors
    )


def test_paved_stock_chains_use_full_planar_connector_lengths() -> None:
    paved = (
        _piece(r"o\road\sil25.p3d"),
        _piece(r"o\road\sil10 100.p3d"),
    )
    dirt = (
        _piece(r"o\road\ces25.p3d"),
        _piece(r"o\road\ces6.p3d"),
    )

    # The reference WRP keeps paved stock P3Ds horizontal, so fitting must not
    # shorten their X/Z chord by the old pitch cosine before placement.
    assert not _reference._uses_measured_rigid_connectors(paved)
    # Keep the scope promised by the paved-road work: stock dirt still uses the
    # existing terrain-following 3D connector semantics.
    assert _reference._uses_measured_rigid_connectors(dirt)


def test_reference_surface_classifier_covers_paved_curves_and_native_junctions() -> None:
    assert _reference._is_paved_stock_surface(r"o\road\sil6.p3d")
    assert _reference._is_paved_stock_surface(r"o\road\asf10 75.p3d")
    assert _reference._is_paved_stock_surface(r"o\road\kos10 100.p3d")
    assert _reference._is_paved_stock_surface(r"o\road\kr_new_sil_ces_t.p3d")
    assert _reference._is_paved_stock_surface(r"o\road\kr_new_silxsil.p3d")
    assert not _reference._is_paved_stock_surface(r"o\road\ces6.p3d")


def test_paved_stock_object_is_flattened_without_moving_its_center(monkeypatch) -> None:
    original = _p.WorldObject(
        7,
        r"o\road\sil6.p3d",
        100.0,
        12.5,
        200.0,
        37.0,
        8.0,
    )
    monkeypatch.setattr(
        _reference,
        "_ORIGINAL_ROAD_OBJECT_ON_SLOPE",
        lambda *args, **kwargs: original,
    )

    fixed = _reference._road_object_on_slope(
        7,
        r"o\road\sil6.p3d",
        (0.0, 0.0),
        (1.0, 1.0),
        (),
        object(),
    )

    assert fixed.pitch_degrees == 0.0
    assert fixed.x == original.x
    assert fixed.y == original.y
    assert fixed.z == original.z
    assert fixed.heading_degrees == original.heading_degrees


def test_stock_dirt_object_keeps_terrain_pitch(monkeypatch) -> None:
    original = _p.WorldObject(
        8,
        r"o\road\ces6.p3d",
        100.0,
        12.5,
        200.0,
        37.0,
        8.0,
    )
    monkeypatch.setattr(
        _reference,
        "_ORIGINAL_ROAD_OBJECT_ON_SLOPE",
        lambda *args, **kwargs: original,
    )

    fixed = _reference._road_object_on_slope(
        8,
        r"o\road\ces6.p3d",
        (0.0, 0.0),
        (1.0, 1.0),
        (),
        object(),
    )

    assert math.isclose(fixed.pitch_degrees, 8.0, abs_tol=1.0e-12)


def test_reference_curve_preference_uses_native_curves_before_three_short_facets() -> None:
    assert (
        _curve_usage._MINIMUM_BASELINE_SHORT_STRAIGHTS
        == _reference.REFERENCE_MINIMUM_BASELINE_SHORT_STRAIGHTS
        == 2
    )
    assert math.isclose(
        _curve_usage._MINIMUM_TOTAL_TURN_DEGREES,
        _reference.REFERENCE_MINIMUM_TOTAL_TURN_DEGREES,
        abs_tol=1.0e-12,
    )
    assert _curve_usage._MINIMUM_PROMOTED_CURVES == 1
    assert _curve_usage._MAXIMUM_EXTRA_PIECES == 3
    assert math.isclose(
        _candidate.INSPECTOR_CURVE_MINIMUM_TURN_DEGREES,
        _reference.REFERENCE_INSPECTOR_CURVE_MINIMUM_TURN_DEGREES,
        abs_tol=1.0e-12,
    )
