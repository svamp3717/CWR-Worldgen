# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen import road_quality_policy as _quality
from cwr_worldgen import stock_road_connector_policy as _connector
from cwr_worldgen import stock_road_local_fit_policy as _local
from cwr_worldgen import stock_road_model_geometry as _geometry
from cwr_worldgen import stock_road_paved_junction_completion_policy as _paved


def _junction(extent: float):
    return _quality._Junction(
        point=(0.0, 0.0),
        axis=(0.0, 1.0),
        half_length=extent,
        half_width=extent,
        directions=((0.0, 1.0), (0.0, -1.0), (1.0, 0.0)),
    )


def test_measured_native_junction_is_distinguished_from_legacy_cap() -> None:
    native = _junction(float(_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES))
    legacy = _junction(3.125)

    assert _paved._is_measured_native_junction(native)
    assert not _paved._is_measured_native_junction(legacy)


def test_native_ownership_restores_pre_local_fit_connector_trim(monkeypatch) -> None:
    measure = SimpleNamespace(points=((0.0, 0.0), (0.0, 100.0)))
    pieces = (SimpleNamespace(model_path=r"o\road\sil6.p3d", length_metres=6.25),)
    extent = float(_geometry.STOCK_JUNCTION_CONNECTOR_RADIUS_METRES)
    context = SimpleNamespace(
        junctions={
            _local._p._road_node_key(measure.points[0]): _junction(extent),
            _local._p._road_node_key(measure.points[-1]): _junction(extent),
        }
    )

    monkeypatch.setattr(
        _paved,
        "_ORIGINAL_QUALITY_WINDOW",
        lambda *args: (0.0, 100.0, 99.9, 103.0),
    )
    monkeypatch.setattr(
        _local,
        "_ORIGINAL_QUALITY_WINDOW",
        lambda *args: (6.03, 93.97, 93.61, 94.19),
    )

    result = _paved._native_ownership_quality_window(
        measure,
        pieces,
        0.0,
        100.0,
        99.9,
        103.0,
        context,
    )

    assert result == (6.03, 93.97, 93.61, 94.19)


def test_legacy_straight_cap_keeps_under_cap_extension(monkeypatch) -> None:
    measure = SimpleNamespace(points=((0.0, 0.0), (0.0, 100.0)))
    pieces = (SimpleNamespace(model_path=r"o\road\sil6.p3d", length_metres=6.25),)
    context = SimpleNamespace(
        junctions={
            _local._p._road_node_key(measure.points[-1]): _junction(3.125),
        }
    )
    current = (0.0, 100.0, 99.9, 103.0)
    monkeypatch.setattr(_paved, "_ORIGINAL_QUALITY_WINDOW", lambda *args: current)
    monkeypatch.setattr(
        _local,
        "_ORIGINAL_QUALITY_WINDOW",
        lambda *args: (0.0, 93.97, 93.61, 94.19),
    )

    assert _paved._native_ownership_quality_window(
        measure,
        pieces,
        0.0,
        100.0,
        99.9,
        103.0,
        context,
    ) == current


def test_production_restores_measured_paved_t_connector_planner() -> None:
    assert _paved._NATIVE_OWNERSHIP_INSTALLED
    assert _paved._ORIGINAL_NATIVE_T_TARGETS is not None
    assert _connector._native_t_targets is _paved._ORIGINAL_NATIVE_T_TARGETS
