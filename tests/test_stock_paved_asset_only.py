# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
from cwr_worldgen import road_inspector as _core
from cwr_worldgen import road_inspector_stock_paved_only as _inspector_stock
from cwr_worldgen import stock_road_junction_policy as _junction
from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen import stock_road_stock_paved_only_policy as _stock


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

    fixed = _stock._apply_stock_emitted_seam_covers(report, [0.0] * 4, spec)

    assert len(fixed.objects) > len(report.objects)
    added = fixed.objects[len(report.objects):]
    assert all(obj.model_path.casefold().startswith(r"o\road\") for obj in added)
    assert all("paved_fill" not in obj.model_path.casefold() for obj in added)
    assert all("paved_miter" not in obj.model_path.casefold() for obj in added)
    assert all("paved_wedge" not in obj.model_path.casefold() for obj in added)
    assert all(obj.model_path.casefold() == r"o\road\sil6.p3d" for obj in added)


def test_turning_t_fallback_uses_stock_sil_cap_not_generated_fill():
    current = _object(7, r"o\road\kr_new_sil_sil_t.p3d", 10.0, 20.0, 0.0)
    incidents = (
        _junction._Incident((0.0, -1.0), "sil", r"o\road\sil6.p3d"),
        _junction._Incident((0.2, 1.0), "sil", r"o\road\sil6.p3d"),
        _junction._Incident((1.0, 0.0), "sil", r"o\road\sil6.p3d"),
    )
    spec = SimpleNamespace(cells=2, cell_size=25.0)

    fixed = _stock._legacy_stock_cap_for_turning_t(
        current,
        (10.0, 20.0),
        incidents,
        "sil",
        [0.0] * 4,
        spec,
    )

    assert fixed.model_path.casefold() == r"o\road\sil6.p3d"
    assert "wg_" not in fixed.model_path.casefold()


def test_retired_generated_paved_helpers_cannot_hide_inspector_wedge():
    generated = _core.RoadObject(
        object_id=99,
        model_path=r"wg_old\i\paved_wedge_q048.p3d",
        x=0.0,
        y=0.03,
        z=0.0,
        heading_degrees=0.0,
        pitch_degrees=0.0,
        family="sil",
        kind="paved_wedge",
        nominal_length_metres=1.0,
        logical_center=(0.0, 0.0),
        endpoints=(),
    )

    assert not _inspector_stock._stock_surface_contains(generated, (0.0, 0.0))
