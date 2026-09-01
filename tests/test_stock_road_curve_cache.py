# SPDX-License-Identifier: GPL-3.0-or-later
import math
from types import SimpleNamespace

from cwr_worldgen import playability as _p
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


def _measure():
    return _p._PolylineMeasure.create(
        ((0.0, 0.0), (0.0, 12.0), (4.0, 24.0), (12.0, 31.0), (18.0, 42.0))
    )


def test_fast_measure_point_matches_original_interpolation_and_extrapolation():
    measure = _measure()
    for distance in (-4.0, 0.0, 4.5, 12.0, 17.5, 29.0, measure.total, measure.total + 6.0):
        expected = _usage._ORIGINAL_POINT(measure, distance)
        actual = _usage._fast_measure_point(measure, distance)
        for observed, wanted in zip(actual, expected):
            assert math.isclose(observed, wanted, rel_tol=0.0, abs_tol=1.0e-12)


def test_fast_nearest_forward_matches_original_bounded_projection():
    measure = _measure()
    cases = (
        ((1.0, 4.0), 0.0, measure.total),
        ((3.0, 18.0), 5.0, 28.0),
        ((10.0, 27.0), 15.0, 35.0),
        ((19.0, 44.0), 30.0, measure.total),
        ((-2.0, -1.0), 0.0, 8.0),
    )

    _usage._segment_table.cache_clear()
    for point, minimum, maximum in cases:
        expected = _usage._ORIGINAL_NEAREST_FORWARD(measure, point, minimum, maximum)
        actual = _usage._fast_nearest_forward(measure, point, minimum, maximum)
        assert expected is not None and actual is not None
        assert math.isclose(actual[0], expected[0], rel_tol=0.0, abs_tol=1.0e-12)
        assert math.isclose(actual[1], expected[1], rel_tol=0.0, abs_tol=1.0e-12)


def test_fast_chord_endpoint_matches_original_without_full_vertex_scan():
    measure = _measure()
    cases = (
        (0.0, 6.25, measure.total),
        (3.0, 12.5, 30.0),
        (11.0, 6.25, 25.0),
        (20.0, 12.5, measure.total),
    )

    for start, chord, maximum in cases:
        expected = _usage._ORIGINAL_CHORD_ENDPOINT(measure, start, chord, maximum)
        actual = _usage._fast_chord_endpoint(measure, start, chord, maximum)
        assert (expected is None) == (actual is None)
        if expected is None:
            continue
        assert actual is not None
        for observed, wanted in zip(actual, expected):
            assert math.isclose(observed, wanted, rel_tol=0.0, abs_tol=1.0e-12)


def test_fast_maximum_chord_deviation_matches_original_span_filtering():
    measure = _measure()
    cases = (
        (0.0, 20.0),
        (5.0, 30.0),
        (12.0, measure.total),
    )

    for start_distance, end_distance in cases:
        start = measure.point(start_distance)[:2]
        end = measure.point(end_distance)[:2]
        expected = _usage._ORIGINAL_MAXIMUM_CHORD_DEVIATION(
            measure, start_distance, end_distance, start, end
        )
        actual = _usage._fast_maximum_chord_deviation(
            measure, start_distance, end_distance, start, end
        )
        assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
