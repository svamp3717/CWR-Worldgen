# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from contextlib import contextmanager
import math
from types import SimpleNamespace

from cwr_worldgen import asset_mapping as _asset_mapping
from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_junction_policy as _junction
from cwr_worldgen import stock_road_local_fit_policy as _local
from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen import stock_road_sharp_turn_policy as _sharp
from cwr_worldgen import stock_road_stock_assets_only_policy as _stock_only
from cwr_worldgen.procedural_infrastructure import is_generated_gravel_road_model


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


def _object(object_id, model, x, z, heading, *, y=0.0, pitch=0.0):
    return SimpleNamespace(
        object_id=int(object_id),
        model_path=model,
        x=float(x),
        y=float(y),
        z=float(z),
        heading_degrees=float(heading),
        pitch_degrees=float(pitch),
    )


def _straight_from_start(object_id, family, start, heading):
    length = float(_geometry.STOCK_STRAIGHT_LENGTHS_METRES[6])
    half = length * 0.5
    direction = (
        math.sin(math.radians(float(heading))),
        math.cos(math.radians(float(heading))),
    )
    centre = (
        float(start[0]) + direction[0] * half,
        float(start[1]) + direction[1] * half,
    )
    return _object(
        object_id,
        rf"o\road\{family}6.p3d",
        centre[0],
        centre[1],
        heading,
    )


@contextmanager
def _planning_junction():
    token = _local._PLANNING_RELAXED_JUNCTION.set(True)
    try:
        yield
    finally:
        _local._PLANNING_RELAXED_JUNCTION.reset(token)


def test_stock_paved_helper_stage_and_final_guard_share_one_owner() -> None:
    assert _stock_only._PAVED_HELPERS_INSTALLED
    assert _stock_only._INSTALLED


def test_final_paved_turn_fallback_serializes_stock_assets_only():
    first = _object(1, r"o\road\sil6.p3d", 0.0, -3.125, 0.0)
    second = _straight_from_start(2, "sil", (0.0, 0.0), 12.0)
    report = _p.RoadFitReport(
        objects=(first, second),
        chain_count=1,
        connection_count=1,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
    )
    spec = SimpleNamespace(
        name="wg_stock_only",
        cells=2,
        cell_size=25.0,
        max_road_objects=100,
        advisory_object_limits=False,
    )

    fixed = _stock_only._apply_stock_emitted_seam_covers(
        report,
        [0.0] * 4,
        spec,
    )

    assert len(fixed.objects) > len(report.objects)
    added = fixed.objects[len(report.objects):]
    assert all(obj.model_path.casefold().startswith("o\\road\\") for obj in added)
    assert all("paved_fill" not in obj.model_path.casefold() for obj in added)
    assert all("paved_miter" not in obj.model_path.casefold() for obj in added)
    assert all("paved_wedge" not in obj.model_path.casefold() for obj in added)
    assert all(obj.model_path.casefold() == r"o\road\sil6.p3d" for obj in added)


def test_gravel_still_uses_generated_family_when_enabled() -> None:
    spec = _spec()
    model = _p.road_model_for_tags(
        spec,
        {"highway": "track", "surface": "gravel"},
    )
    assert is_generated_gravel_road_model(model)
    assert not _stock_only._generated_road_model(model)


def test_asset_mapping_still_requests_generated_gravel_family() -> None:
    mapping = _asset_mapping.default_osm_asset_mapping(_spec(), 9)
    gravel = next(rule for rule in mapping.rules if rule.rule_id == "road-gravel")
    assert gravel.models
    assert any("gravel" in model.casefold() for model in gravel.models)
    assert gravel.textures


def test_only_generated_paved_or_dirt_models_are_forbidden_by_final_guard() -> None:
    assert _stock_only._generated_road_model(r"stock_only_test\i\paved_fill.p3d")
    assert _stock_only._generated_road_model(r"stock_only_test\i\paved_miter_q020.p3d")
    assert _stock_only._generated_road_model(r"stock_only_test\i\paved_wedge_q020.p3d")
    assert _stock_only._generated_road_model(r"stock_only_test\i\dirt6.p3d")
    assert not _stock_only._generated_road_model(r"stock_only_test\i\gravel6.p3d")
    assert not _stock_only._generated_road_model(r"o\road\sil6.p3d")
    assert not _stock_only._generated_road_model(r"o\road\ces6.p3d")
    assert not _stock_only._generated_road_model(r"o\road\kr_new_sil_ces_t.p3d")


def test_eleven_degree_mixed_t_can_be_planned_but_not_forced_unmodified() -> None:
    incidents = (
        _junction._Incident(_direction(0.0), "sil", r"o\road\sil25.p3d"),
        _junction._Incident(_direction(168.6), "sil", r"o\road\sil25.p3d"),
        _junction._Incident(_direction(270.0), "ces", r"o\road\ces25.p3d"),
    )

    assert _stock_only._stock_native_t_dispatch(incidents) is None

    with _planning_junction():
        native = _stock_only._stock_native_t_dispatch(incidents)
    assert native is not None
    assert native.model_path.casefold() == r"o\road\kr_new_sil_ces_t.p3d"


def test_family_first_planner_does_not_reject_t_from_raw_heading_error() -> None:
    incidents = (
        _junction._Incident(_direction(0.0), "sil", r"o\road\sil25.p3d"),
        _junction._Incident(_direction(150.0), "sil", r"o\road\sil25.p3d"),
        _junction._Incident(_direction(270.0), "ces", r"o\road\ces25.p3d"),
    )

    with _planning_junction():
        native = _stock_only._stock_native_t_dispatch(incidents)
    assert native is not None
    assert native.model_path.casefold() == r"o\road\kr_new_sil_ces_t.p3d"
    assert native.maximum_heading_error_degrees > 5.0


def test_tangent_stock_straights_receive_real_longitudinal_overlap() -> None:
    measure = _p._PolylineMeasure.create(((0.0, 0.0), (0.0, 60.0)))
    piece = _p._RoadPiece(r"o\road\sil25.p3d", 25.0, 25)
    baseline = (
        (piece, (0.0, 0.0), (0.0, 25.0)),
        (piece, (0.0, 25.0), (0.0, 50.0)),
    )

    fitted = _stock_only._overlapped_stock_chain(
        measure,
        (piece,),
        baseline,
        start_distance=0.0,
        preferred_end_distance=50.0,
        minimum_end_distance=49.5,
        maximum_end_distance=60.0,
    )

    assert fitted != baseline
    assert len(fitted) == 2
    assert math.isclose(math.dist(fitted[0][1], fitted[0][2]), 25.0, abs_tol=1.0e-9)
    assert math.isclose(math.dist(fitted[1][1], fitted[1][2]), 25.0, abs_tol=1.0e-9)
    assert math.isclose(
        math.dist(fitted[0][2], fitted[1][1]),
        _stock_only.ORDINARY_PAVED_OVERLAP_METRES,
        abs_tol=1.0e-9,
    )
    assert math.isclose(fitted[-1][2][1], 49.55, abs_tol=1.0e-9)


def test_reference_curve_corridor_is_installed() -> None:
    assert math.isclose(
        _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES,
        _stock_only.STOCK_CURVE_SOURCE_CORRIDOR_METRES,
        abs_tol=1.0e-12,
    )
    assert _sharp._MAXIMUM_LOCKED_CORRIDOR_METRES > 0.60
