# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from cwr_worldgen import playability as _p
from cwr_worldgen import stock_road_junction_endpoint_policy as _endpoint


def _measure():
    return _p._PolylineMeasure.create(((0.0, 0.0), (0.0, 20.0)))


def _piece():
    return _p._RoadPiece(r"o\road\sil6.p3d", 6.25, 6)


def _run(chain):
    measure = _measure()
    return chain(
        measure,
        (_piece(),),
        start_distance=2.425,
        preferred_end_distance=17.575,
        minimum_end_distance=16.725,
        maximum_end_distance=20.70,
    )


def test_trimmed_late_result_is_retried_with_effective_junction_window(monkeypatch) -> None:
    calls = []

    def fake_chain(
        measure,
        pieces,
        *,
        start_distance,
        preferred_end_distance,
        minimum_end_distance,
        maximum_end_distance,
    ):
        calls.append((float(start_distance), float(preferred_end_distance)))
        piece = pieces[0]
        if float(start_distance) > 0.1:
            return ((piece, (0.0, 2.425), (0.0, 17.575)),)
        return ((piece, (0.0, 0.0), (0.0, 20.0)),)

    monkeypatch.setattr(_endpoint, "_ORIGINAL_CHAIN", fake_chain)
    monkeypatch.setattr(
        _endpoint,
        "_effective_window",
        lambda *_args: (0.0, 20.0, 19.9, 20.70),
    )

    fitted = _run(_endpoint._junction_endpoint_chain)

    assert calls == [(2.425, 17.575), (0.0, 20.0)]
    assert fitted[0][1] == (0.0, 0.0)
    assert fitted[-1][2] == (0.0, 20.0)


def test_exact_result_that_already_reaches_junction_is_kept(monkeypatch) -> None:
    calls = []

    def fake_chain(
        measure,
        pieces,
        *,
        start_distance,
        preferred_end_distance,
        minimum_end_distance,
        maximum_end_distance,
    ):
        calls.append((float(start_distance), float(preferred_end_distance)))
        return ((pieces[0], (0.0, 0.0), (0.0, 20.0)),)

    monkeypatch.setattr(_endpoint, "_ORIGINAL_CHAIN", fake_chain)
    monkeypatch.setattr(
        _endpoint,
        "_effective_window",
        lambda *_args: (0.0, 20.0, 19.9, 20.70),
    )

    fitted = _run(_endpoint._junction_endpoint_chain)

    assert calls == [(2.425, 17.575)]
    assert fitted[-1][2] == (0.0, 20.0)


def test_uncovered_run_does_not_trigger_endpoint_retry(monkeypatch) -> None:
    calls = []

    def fake_chain(
        measure,
        pieces,
        *,
        start_distance,
        preferred_end_distance,
        minimum_end_distance,
        maximum_end_distance,
    ):
        calls.append((float(start_distance), float(preferred_end_distance)))
        return ((pieces[0], (0.0, 2.425), (0.0, 17.575)),)

    monkeypatch.setattr(_endpoint, "_ORIGINAL_CHAIN", fake_chain)
    monkeypatch.setattr(
        _endpoint,
        "_effective_window",
        lambda _measure, _pieces, start, preferred, minimum, maximum: (
            float(start), float(preferred), float(minimum), float(maximum)
        ),
    )

    _run(_endpoint._junction_endpoint_chain)

    assert calls == [(2.425, 17.575)]


def test_junction_endpoint_policy_is_outermost_stock_chain_on_package_import() -> None:
    assert _endpoint._INSTALLED
    assert _p._stock_piece_chain is _endpoint._junction_endpoint_chain
