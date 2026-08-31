# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from contextlib import contextmanager
import math
from types import SimpleNamespace

from cwr_worldgen import asset_mapping as _asset_mapping
from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_junction_policy as _junction
from cwr_worldgen import stock_road_relaxation_transaction_policy as _transaction
from cwr_worldgen import stock_road_sharp_turn_policy as _sharp
from cwr_worldgen import stock_road_stock_assets_only_policy as _stock_only
from cwr_worldgen.procedural_infrastructure import gravel_road_model_path


def _spec():
    return SimpleNamespace(
        name="stock_only_test",
        procedural_gravel_roads=True,
        dirt_road_model=r"o\road\ces25.p3d",
        paved_road_model=r"o\road\sil25.p3d",
        barriers_enabled=False,
        bus_stops_enabled=False,
        cemeteries_enabled=False,
        wetland_reeds_enabled=False,
    )


def _direction(heading: float) -> tuple[float, float]:
    angle = math.radians(float(heading))
    return math.sin(angle), math.cos(angle)


@contextmanager
def _planning_junction():
    token = _transaction._PLANNING_RELAXED_JUNCTION.set(True)
    try:
        yield
    finally:
        _transaction._PLANNING_RELAXED_JUNCTION.reset(token)


def test_service_and_track_roads_ignore_procedural_gravel_flag() -> None:
    spec = _spec()
    assert _p.road_model_for_tags(
        spec,
        {"highway": "track", "surface": "gravel"},
    ) == r"o\road\ces25.p3d"
    assert _p.road_model_for_tags(
        spec,
        {"highway": "service", "surface": "compacted"},
    ) == r"o\road\ces25.p3d"


def test_asset_mapping_never_requests_generated_gravel_road_family() -> None:
    mapping = _asset_mapping.default_osm_asset_mapping(_spec(), 9)
    gravel = next(rule for rule in mapping.rules if rule.rule_id == "road-gravel")
    assert gravel.models == (r"o\road\ces25.p3d",)
    assert not gravel.textures


def test_generated_road_models_are_forbidden_by_final_guard() -> None:
    assert _stock_only._generated_road_model(
        gravel_road_model_path("stock_only_test", 6)
    )
    assert _stock_only._generated_road_model(r"stock_only_test\i\paved_fill.p3d")
    assert _stock_only._generated_road_model(r"stock_only_test\i\paved_miter_q020.p3d")
    assert _stock_only._generated_road_model(r"stock_only_test\i\paved_wedge_q020.p3d")
    assert not _stock_only._generated_road_model(r"o\road\sil6.p3d")
    assert not _stock_only._generated_road_model(r"o\road\ces6.p3d")
    assert not _stock_only._generated_road_model(r"o\road\kr_new_sil_ces_t.p3d")


def test_eleven_degree_mixed_t_can_be_planned_but_not_forced_unmodified() -> None:
    incidents = (
        _junction._Incident(_direction(0.0), "sil", r"o\road\sil25.p3d"),
        _junction._Incident(_direction(168.6), "sil", r"o\road\sil25.p3d"),
        _junction._Incident(_direction(270.0), "ces", r"o\road\ces25.p3d"),
    )

    # Raw source geometry is visibly too skewed for the final 0.90-degree
    # matcher, so a native T is not simply stamped over the original roads.
    assert _stock_only._stock_native_t_dispatch(incidents) is None

    # The obstacle-checked transaction is allowed to consider the real stock T
    # and insert connector-aligned local approach points.  The transaction later
    # re-runs the strict matcher on that edited geometry before committing it.
    with _planning_junction():
        native = _stock_only._stock_native_t_dispatch(incidents)
    assert native is not None
    assert native.model_path.casefold() == r"o\road\kr_new_sil_ces_t.p3d"


def test_reference_curve_corridor_is_installed() -> None:
    assert math.isclose(
        _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES,
        _stock_only.STOCK_CURVE_SOURCE_CORRIDOR_METRES,
        abs_tol=1.0e-12,
    )
    assert _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES > 0.60
