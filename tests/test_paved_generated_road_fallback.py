from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen import playability
from cwr_worldgen import procedural_infrastructure as infrastructure
from cwr_worldgen import paved_road_generated_fallback_policy as fallback
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
    assert Milestone9Spec().procedural_paved_road_fallback is True
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
        "reuse_world", 4.55, 6.24, 19.0
    )
    second = infrastructure.paved_fallback_model_path(
        "reuse_world", 4.56, 6.23, 21.0
    )

    assert first == second
    assert first.endswith(r"paved_w046_l0062_r20.p3d")


def test_generated_paved_asset_is_written_once_and_has_roadway_lod(
    tmp_path: Path,
) -> None:
    model = infrastructure.paved_fallback_model_path(
        "reuse_world", 4.55, 6.24, -19.0
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
