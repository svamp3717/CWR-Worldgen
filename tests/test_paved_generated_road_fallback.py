from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen import playability
from cwr_worldgen import procedural_infrastructure as infrastructure
from cwr_worldgen import paved_road_generated_fallback_policy as fallback
from cwr_worldgen import paved_junction_policy as paved_junctions
from cwr_worldgen import road_quality_policy as quality
from cwr_worldgen.milestone9 import Milestone9Spec, _Milestone9PlayabilitySpec


def _spec(*, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        procedural_paved_road_fallback=enabled,
        name="paved_fallback_test",
        paved_road_model=r"o\road\sil25.p3d",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
    )


def _single_piece_result(measure, piece):
    endpoint = measure.chord_endpoint(0.0, piece.length_metres, measure.total)
    assert endpoint is not None
    _distance, end_x, end_z, _heading = endpoint
    start_x, start_z, _heading = measure.point(0.0)
    return ((piece, (start_x, start_z), (end_x, end_z)),)


def _upgrade(measure, pieces, piece, spec):
    token = quality._CONTEXT.set(quality._Context((), spec, {}))
    try:
        return fallback._upgrade_stock_result(
            _single_piece_result(measure, piece),
            measure,
            pieces,
            start_distance=0.0,
            preferred_end_distance=measure.total,
            minimum_end_distance=0.0,
            maximum_end_distance=measure.total,
        )
    finally:
        quality._CONTEXT.reset(token)


def test_milestone9_enables_only_the_new_paved_fallback_by_default() -> None:
    assert Milestone9Spec(
        source_dir=Path(".")
    ).procedural_paved_road_fallback is True
    assert _Milestone9PlayabilitySpec(
        heightmap_path=Path("unused.png")
    ).procedural_paved_road_fallback is True


def test_straight_paved_piece_keeps_stock_p3d() -> None:
    spec = _spec()
    pieces = playability.road_model_variants(
        spec.paved_road_model, spec.road_segment_length
    )
    piece = next(item for item in pieces if item.nominal_length == 6)
    measure = playability._PolylineMeasure.create(((0.0, 0.0), (0.0, 20.0)))

    upgraded = _upgrade(measure, pieces, piece, spec)

    assert upgraded[0][0].model_path == piece.model_path
    assert not infrastructure.is_generated_paved_road_model(
        upgraded[0][0].model_path
    )


def _two_piece_result(measure, piece):
    first = measure.chord_endpoint(0.0, piece.length_metres, measure.total)
    assert first is not None
    first_distance, first_x, first_z, _heading = first
    second = measure.chord_endpoint(
        first_distance, piece.length_metres, measure.total
    )
    assert second is not None
    _second_distance, second_x, second_z, _heading = second
    start_x, start_z, _heading = measure.point(0.0)
    return (
        (piece, (start_x, start_z), (first_x, first_z)),
        (piece, (first_x, first_z), (second_x, second_z)),
    )


def _upgrade_result(measure, pieces, result, spec):
    token = quality._CONTEXT.set(quality._Context((), spec, {}))
    try:
        return fallback._upgrade_stock_result(
            result,
            measure,
            pieces,
            start_distance=0.0,
            preferred_end_distance=measure.total,
            minimum_end_distance=0.0,
            maximum_end_distance=measure.total,
        )
    finally:
        quality._CONTEXT.reset(token)


def test_multiple_straight_paved_pieces_remain_stock() -> None:
    spec = _spec()
    pieces = playability.road_model_variants(
        spec.paved_road_model, spec.road_segment_length
    )
    piece = next(item for item in pieces if item.nominal_length == 6)
    measure = playability._PolylineMeasure.create(
        ((0.0, 0.0), (0.0, 12.0))
    )

    upgraded = _upgrade_result(
        measure, pieces, _two_piece_result(measure, piece), spec
    )

    assert len(upgraded) == 2
    assert all(item[0].model_path == piece.model_path for item in upgraded)
    assert not any(
        infrastructure.is_generated_paved_road_model(item[0].model_path)
        for item in upgraded
    )


def test_clipping_stock_joint_generates_only_the_offending_piece() -> None:
    spec = _spec()
    pieces = playability.road_model_variants(
        spec.paved_road_model, spec.road_segment_length
    )
    piece = next(item for item in pieces if item.nominal_length == 6)
    angle = math.radians(20.0)
    measure = playability._PolylineMeasure.create(
        (
            (0.0, 0.0),
            (0.0, 6.0),
            (math.sin(angle) * 6.0, 6.0 + math.cos(angle) * 6.0),
        )
    )

    upgraded = _upgrade_result(
        measure, pieces, _two_piece_result(measure, piece), spec
    )

    assert upgraded[0][0].model_path == piece.model_path
    assert infrastructure.is_generated_paved_road_model(
        upgraded[1][0].model_path
    )
    assert sum(
        infrastructure.is_generated_paved_road_model(item[0].model_path)
        for item in upgraded
    ) == 1


def test_tight_paved_bend_uses_generated_fallback_when_stock_piece_fails() -> None:
    spec = _spec()
    pieces = playability.road_model_variants(
        spec.paved_road_model, spec.road_segment_length
    )
    piece = next(item for item in pieces if item.nominal_length == 6)
    measure = playability._PolylineMeasure.create(
        ((0.0, 0.0), (2.0, 3.0), (5.0, 4.0), (7.0, 1.0))
    )

    upgraded = _upgrade(measure, pieces, piece, spec)
    model_path = upgraded[0][0].model_path

    assert infrastructure.is_generated_paved_road_model(model_path)
    assert model_path.casefold().startswith(r"paved_fallback_test\i\paved_")
    assert r"\road\sil6.p3d" not in model_path.casefold()


def test_tight_dirt_bend_never_uses_generated_paved_fallback() -> None:
    spec = _spec()
    pieces = playability.road_model_variants(
        spec.dirt_road_model, spec.road_segment_length
    )
    piece = next(item for item in pieces if item.nominal_length == 6)
    measure = playability._PolylineMeasure.create(
        ((0.0, 0.0), (2.0, 3.0), (5.0, 4.0), (7.0, 1.0))
    )

    upgraded = _upgrade(measure, pieces, piece, spec)

    assert upgraded[0][0].model_path == piece.model_path
    assert not infrastructure.is_generated_paved_road_model(
        upgraded[0][0].model_path
    )


def test_disabled_fallback_never_replaces_stock_piece() -> None:
    spec = _spec(enabled=False)
    pieces = playability.road_model_variants(
        spec.paved_road_model, spec.road_segment_length
    )
    piece = next(item for item in pieces if item.nominal_length == 6)
    measure = playability._PolylineMeasure.create(
        ((0.0, 0.0), (2.0, 3.0), (5.0, 4.0), (7.0, 1.0))
    )

    upgraded = _upgrade(measure, pieces, piece, spec)

    assert upgraded[0][0].model_path == piece.model_path


def test_generated_paved_model_names_quantize_for_reuse() -> None:
    first = infrastructure.paved_fallback_model_path(
        "reuse_world", 9.10, 6.24, 19.0
    )
    second = infrastructure.paved_fallback_model_path(
        "reuse_world", 9.06, 6.23, 21.0
    )

    assert first == second
    assert first.endswith(r"paved_w091_l0062_r20.p3d")


def test_generated_paved_asset_is_written_once_and_has_roadway_lod(
    tmp_path: Path,
) -> None:
    model = infrastructure.paved_fallback_model_path(
        "reuse_world", 9.10, 6.24, -19.0
    )
    library = infrastructure.ProceduralInfrastructureLibrary(
        "reuse_world",
        cache_enabled=False,
    )
    library.register_model_usage(model, 3)
    result = library.write_assets(
        tmp_path,
        tmp_path / "infrastructure.json",
    )

    assert result.placements == 3
    assert result.generated_variants == 1
    assert "i/pv.paa" in result.texture_files
    assert len(result.model_files) == 1

    document = json.loads(
        (tmp_path / "infrastructure.json").read_text(encoding="utf-8")
    )
    assert document["models"][0]["usage_count"] == 3
    assert any(
        abs(value - infrastructure._ROADWAY_LOD) < 1.0
        for value in document["models"][0]["lod_resolutions"]
    )


def test_generated_paved_road_matches_requested_stock_surface_height() -> None:
    spec = SimpleNamespace(cells=4, cell_size=10.0)
    model = infrastructure.paved_fallback_model_path(
        "grounded_world", 9.10, 6.20, 10.0
    )
    obj = playability._road_object_on_slope(
        1,
        model,
        (10.0, 10.0),
        (10.0, 16.2),
        [0.0] * 16,
        spec,
        vertical_offset=0.060,
    )

    # The local paved skin is 0.025 m above its model origin. Keep its world-space
    # top on the same +6 cm plane requested for the surrounding stock road.
    expected_origin = 0.060 - infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES
    assert abs(obj.y - expected_origin) < 1.0e-9


def test_generated_paved_curve_has_square_nonoverhanging_connection_planes() -> None:
    length = 6.2
    half_width = 4.55
    sections = infrastructure._road_ribbon_sections(
        length,
        half_width,
        45,
        overhang=0.0,
        square_ends=True,
    )
    first = sections[0]
    last = sections[-1]

    assert math.isclose(first[1], -length * 0.5, abs_tol=1.0e-9)
    assert math.isclose(first[3], -length * 0.5, abs_tol=1.0e-9)
    assert math.isclose(last[1], length * 0.5, abs_tol=1.0e-9)
    assert math.isclose(last[3], length * 0.5, abs_tol=1.0e-9)


def test_stock_paved_joint_limit_is_based_on_visible_edge_error() -> None:
    piece = playability._RoadPiece(r"o\road\sil6.p3d", 6.25, 6)
    small = quality._stock_paved_joint_edge_discontinuity(
        piece, 0.0, piece, 1.0
    )
    visible = quality._stock_paved_joint_edge_discontinuity(
        piece, 0.0, piece, 3.0
    )

    assert small < quality._STOCK_PAVED_MAX_EDGE_DISCONTINUITY_METRES
    assert visible > quality._STOCK_PAVED_MAX_EDGE_DISCONTINUITY_METRES


def test_parallel_quality_wrapper_keeps_generated_paved_fallback_live() -> None:
    from cwr_worldgen import road_quality_parallel_compat_policy as parallel_quality

    assert parallel_quality._batched_quality_chain is fallback._parallel_chain
    assert playability._stock_piece_chain is fallback._parallel_chain


def test_generated_paved_length_is_available_to_road_audits() -> None:
    model = infrastructure.paved_fallback_model_path(
        "audit_world", 9.10, 6.20, -20.0
    )
    assert math.isclose(quality._piece_length(model, 25.0), 6.2, abs_tol=1.0e-9)


def test_generated_paved_width_maps_back_to_stock_junction_family() -> None:
    wide = infrastructure.paved_fallback_model_path(
        "junction_world", 9.10, 6.20, 0.0
    )
    narrow = infrastructure.paved_fallback_model_path(
        "junction_world", 7.00, 6.20, 0.0
    )
    assert paved_junctions._family(wide) == "sil"
    assert paved_junctions._family(narrow) == "asf"
