# SPDX-License-Identifier: GPL-3.0-or-later
from types import SimpleNamespace

from cwr_worldgen import stock_road_curve_usage_policy as _usage


def _pieces():
    return (
        SimpleNamespace(model_path=r"o\road\sil25.p3d", length_metres=25.0, nominal_length=25),
        SimpleNamespace(model_path=r"o\road\sil12.p3d", length_metres=12.5, nominal_length=12),
        SimpleNamespace(model_path=r"o\road\sil6.p3d", length_metres=6.25, nominal_length=6),
    )


def test_repeated_stock_beam_search_uses_cached_result(monkeypatch):
    calls = []
    expected = ((0.0, 0.0), (0.0, 10.0))

    def beam(source, sign, entry, exit, pieces, **kwargs):
        calls.append(dict(kwargs))
        assert all(hasattr(piece, "model_path") for piece in pieces)
        return expected

    source = ((0.0, 0.0), (0.0, 10.0))

    monkeypatch.setattr(_usage, "_ORIGINAL_BEAM", beam)
    _usage._cached_micro_beam_stock_path.cache_clear()
    try:
        first = _usage._micro_beam_stock_path(source, 1, 0.0, 10.0, _pieces())
        second = _usage._micro_beam_stock_path(source, 1, 0.0, 10.0, _pieces())
    finally:
        _usage._cached_micro_beam_stock_path.cache_clear()

    assert first == expected
    assert second == expected
    assert len(calls) == 1
    assert calls[0]["minimum_curve_count"] == 1


def test_reachable_twenty_degree_target_still_tries_strict_beam_first(monkeypatch):
    calls = []
    expected = ((0.0, 0.0), (0.0, 20.0))

    def beam(source, sign, entry, exit, pieces, **kwargs):
        calls.append(dict(kwargs))
        return expected

    monkeypatch.setattr(_usage, "_ORIGINAL_BEAM", beam)
    _usage._cached_micro_beam_stock_path.cache_clear()
    try:
        result = _usage._micro_beam_stock_path(
            ((0.0, 0.0), (0.0, 20.0)),
            1,
            0.0,
            20.0,
            _pieces(),
        )
    finally:
        _usage._cached_micro_beam_stock_path.cache_clear()

    assert result == expected
    assert calls == [{}]
