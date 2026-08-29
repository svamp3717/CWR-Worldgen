# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen import stock_road_emitted_seam_policy as _emitted
from cwr_worldgen import stock_road_fit_first_policy as _fit_first
from cwr_worldgen import stock_road_intersection_edge_policy as _intersection_edge
from cwr_worldgen import stock_road_visual_finish_policy as _finish


def test_fit_first_guard_owns_all_post_fit_overlap_hooks() -> None:
    assert _fit_first._INSTALLED
    assert _finish._apply_curve_seam_covers is _fit_first._preserve_fitted_visual_seam
    assert (
        _intersection_edge._seal_legacy_paved_intersections
        is _fit_first._preserve_fitted_intersection
    )
    assert (
        _emitted._apply_emitted_seam_covers
        is _fit_first._preserve_fitted_emitted_seam
    )


def test_previous_overlap_hooks_remain_recorded_for_regression_analysis() -> None:
    assert _fit_first._ORIGINAL_VISUAL_SEAM_APPLY is not None
    assert _fit_first._ORIGINAL_INTERSECTION_EDGE_APPLY is not None
    assert _fit_first._ORIGINAL_EMITTED_SEAM_APPLY is not None
    assert _fit_first._ORIGINAL_VISUAL_SEAM_APPLY.__name__ == (
        "_apply_paved_curve_seam_fallback"
    )
    assert _fit_first._ORIGINAL_INTERSECTION_EDGE_APPLY.__name__ == (
        "_seal_legacy_paved_intersections"
    )
    assert _fit_first._ORIGINAL_EMITTED_SEAM_APPLY.__name__ == (
        "_apply_emitted_seam_covers"
    )


def test_visual_seam_guard_does_not_append_road_objects() -> None:
    report = SimpleNamespace(objects=("existing-road",), short_piece_objects=7)

    result = _fit_first._preserve_fitted_visual_seam(report, (), None)

    assert result is report
    assert result.objects == ("existing-road",)
    assert result.short_piece_objects == 7


def test_intersection_guard_does_not_append_overlap_tongues() -> None:
    report = SimpleNamespace(objects=("fitted-junction",), short_piece_objects=3)

    result = _fit_first._preserve_fitted_intersection(
        report,
        None,
        None,
        (),
        None,
    )

    assert result is report
    assert result.objects == ("fitted-junction",)
    assert result.short_piece_objects == 3


def test_emitted_seam_guard_does_not_append_final_underlays() -> None:
    report = SimpleNamespace(objects=("fitted-road",), short_piece_objects=2)

    result = _fit_first._preserve_fitted_emitted_seam(report, (), None)

    assert result is report
    assert result.objects == ("fitted-road",)
    assert result.short_piece_objects == 2
