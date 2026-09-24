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
        ((0.0, 0.0), (0.0, 12.5))
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


def test_clipping_stock_joint_collapses_pair_into_one_generated_bend() -> None:
    spec = _spec()
    pieces = playability.road_model_variants(
        spec.paved_road_model, spec.road_segment_length
    )
    piece = next(item for item in pieces if item.nominal_length == 6)
    angle = math.radians(20.0)
    measure = playability._PolylineMeasure.create(
        (
            (0.0, 0.0),
            (0.0, 6.25),
            (
                math.sin(angle) * 6.25,
                6.25 + math.cos(angle) * 6.25,
            ),
        )
    )

    upgraded = _upgrade_result(
        measure, pieces, _two_piece_result(measure, piece), spec
    )

    # Remove the bad seam itself rather than painting over one side of it.
    assert len(upgraded) == 1
    generated, start_point, end_point = upgraded[0]
    assert infrastructure.is_generated_paved_road_model(generated.model_path)
    assert start_point == (0.0, 0.0)
    assert math.isclose(end_point[0], math.sin(angle) * 6.25, abs_tol=1.0e-9)
    assert math.isclose(
        end_point[1], 6.25 + math.cos(angle) * 6.25, abs_tol=1.0e-9
    )

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


def test_legacy_paved_stock_families_share_stock_first_geometry_rules() -> None:
    silnice = playability.road_model_variants(r"data3d\silnice25.p3d", 24.5)
    asfaltka = playability.road_model_variants(r"data3d\asfaltka25.p3d", 24.5)

    assert [piece.length_metres for piece in silnice] == [25.0, 12.5, 6.25]
    assert [piece.length_metres for piece in asfaltka] == [25.0, 12.5, 6.25]
    assert quality._stock_paved_family(silnice[0]) == "silnice"
    assert quality._stock_paved_family(asfaltka[0]) == "asfaltka"
    assert math.isclose(
        fallback._generated_width(asfaltka, SimpleNamespace(paved_road_model=r"data3d\asfaltka25.p3d")),
        7.0,
        abs_tol=1.0e-9,
    )

def test_vanilla_stock_variants_use_physical_lengths_at_default_24_5_setting() -> None:
    pieces = playability.road_model_variants(r"o\road\sil25.p3d", 24.5)
    lengths = {
        piece.nominal_length: piece.length_metres
        for piece in pieces
    }

    assert lengths == {25: 25.0, 12: 12.5, 6: 6.25}
    assert math.isclose(
        quality._piece_length(r"o\road\sil25.p3d", 24.5),
        25.0,
        abs_tol=1.0e-9,
    )


def test_two_stock_25m_pieces_meet_without_axial_overlap_at_default_setting() -> None:
    pieces = playability.road_model_variants(r"o\road\sil25.p3d", 24.5)
    piece = next(value for value in pieces if value.nominal_length == 25)
    measure = playability._PolylineMeasure.create(((0.0, 0.0), (0.0, 50.0)))
    token = quality._CONTEXT.set(
        quality._Context(
            (),
            SimpleNamespace(
                cells=4,
                cell_size=25.0,
                road_connection_tolerance=0.35,
            ),
            {},
        )
    )
    try:
        fitted = quality._quality_chain(
            measure,
            pieces,
            start_distance=0.0,
            preferred_end_distance=50.0,
            minimum_end_distance=50.0,
            maximum_end_distance=50.0,
        )
    finally:
        quality._CONTEXT.reset(token)

    assert [item[0].nominal_length for item in fitted] == [25, 25]
    first_axis = fitted[0][1], fitted[0][2]
    second_axis = fitted[1][1], fitted[1][2]
    assert first_axis[1] == second_axis[0] == (0.0, 25.0)
    assert math.isclose(piece.length_metres, 25.0, abs_tol=1.0e-9)


def test_two_degree_generated_curve_is_supported_for_wide_paved_seams() -> None:
    model = infrastructure.paved_fallback_model_path(
        "fine_curve_world", 9.10, 6.25, 2.0
    )
    assert model.endswith(r"paved_w091_l0062_r02.p3d")
    assert infrastructure.is_generated_paved_road_model(model)


def test_generated_paved_model_names_quantize_for_reuse() -> None:
    first = infrastructure.paved_fallback_model_path(
        "reuse_world", 9.10, 6.24, 19.0
    )
    second = infrastructure.paved_fallback_model_path(
        "reuse_world", 9.06, 6.23, 21.0
    )

    assert first == second
    assert first.endswith(r"paved_w091_l0062_r20.p3d")


def test_generated_paved_asset_reuses_stock_texture_and_has_roadway_lod(
    tmp_path: Path,
) -> None:
    model = infrastructure.paved_fallback_model_path(
        "reuse_world", 9.10, 6.24, -19.0
    )
    stale = tmp_path / "i" / "pv.paa"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"obsolete-generated-asphalt")
    stock_texture = r"landtext\silnice.pac"
    library = infrastructure.ProceduralInfrastructureLibrary(
        "reuse_world",
        paved_texture_path=stock_texture,
        cache_enabled=False,
    )
    library.register_model_usage(model, 3)
    result = library.write_assets(
        tmp_path,
        tmp_path / "infrastructure.json",
    )

    assert result.placements == 3
    assert result.generated_variants == 1
    assert result.texture_files == ()
    assert not stale.exists()
    assert len(result.model_files) == 1

    model_summary = infrastructure.inspect_mlod(
        tmp_path / result.model_files[0]
    )
    assert stock_texture in model_summary.textures

    document = json.loads(
        (tmp_path / "infrastructure.json").read_text(encoding="utf-8")
    )
    assert document["models"][0]["usage_count"] == 3
    assert document["paved_texture_source"] == {
        "type": "external-stock-texture",
        "texture": stock_texture,
        "generated_texture": False,
    }
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


def test_generated_paved_curve_keeps_endpoint_centres_without_overhang() -> None:
    length = 6.2
    half_width = 4.55
    sections = infrastructure._road_ribbon_sections(
        length,
        half_width,
        20,
        overhang=0.0,
        square_ends=False,
    )
    first = sections[0]
    last = sections[-1]

    # The edge rotates to the curve tangent, but its midpoint remains exactly on
    # the nominal connector. No extra mesh is pushed beyond the centreline ends.
    assert math.isclose((first[1] + first[3]) * 0.5, -length * 0.5, abs_tol=1.0e-9)
    assert math.isclose((last[1] + last[3]) * 0.5, length * 0.5, abs_tol=1.0e-9)
    assert not math.isclose(first[1], first[3], abs_tol=1.0e-6)
    assert not math.isclose(last[1], last[3], abs_tol=1.0e-6)


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
    for curve in (-20.0, 2.0):
        model = infrastructure.paved_fallback_model_path(
            "audit_world", 9.10, 6.20, curve
        )
        assert math.isclose(
            quality._piece_length(model, 24.5),
            6.2,
            abs_tol=1.0e-9,
        )


def test_generated_paved_width_maps_back_to_stock_junction_family() -> None:
    wide = infrastructure.paved_fallback_model_path(
        "junction_world", 9.10, 6.20, 0.0
    )
    narrow = infrastructure.paved_fallback_model_path(
        "junction_world", 7.00, 6.20, 0.0
    )
    assert paved_junctions._family(wide) == "sil"
    assert paved_junctions._family(narrow) == "asf"
