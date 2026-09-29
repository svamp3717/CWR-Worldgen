from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cwr_worldgen import paved_road_generated_fallback_policy as fallback
from cwr_worldgen import playability
from cwr_worldgen import procedural_infrastructure as infrastructure
from cwr_worldgen import road_quality_policy as quality
from cwr_worldgen.milestone9 import Milestone9Spec


def _spec() -> SimpleNamespace:
    return SimpleNamespace(
        name="customroads",
        road_segment_length=25.0,
        paved_road_model=r"o\road\sil25.p3d",
        dirt_road_model=r"o\road\ces25.p3d",
        procedural_paved_road_fallback=True,
        custom_road_shapes=True,
    )


@pytest.mark.parametrize(
    ("surface", "recognizer"),
    (
        ("paved", infrastructure.is_generated_paved_road_model),
        ("gravel", infrastructure.is_generated_gravel_road_model),
        ("dirt", infrastructure.is_generated_dirt_road_model),
    ),
)
def test_custom_road_model_encodes_arbitrary_shape(surface, recognizer) -> None:
    model = infrastructure.custom_road_model_path(
        "customroads",
        surface,
        width_metres=7.4,
        length_metres=13.7,
        curve_degrees=27.0,
    )

    assert model.endswith(
        rf"\i\road_{surface}_w074_l0137_r027.p3d"
    )
    assert recognizer(model)
    assert infrastructure.custom_road_model_signature(model) == (
        surface,
        7.4,
        13.7,
        27,
    )


def test_custom_road_curve_is_not_limited_to_stock_buckets() -> None:
    assert 27 not in infrastructure.GENERATED_PAVED_CURVE_BUCKETS
    model = infrastructure.custom_road_model_path(
        "customroads", "paved", 9.1, 11.3, -27.0
    )
    assert model.endswith(r"\i\road_paved_w091_l0113_l027.p3d")


def test_custom_road_curve_is_safely_clamped() -> None:
    model = infrastructure.custom_road_model_path(
        "customroads", "dirt", 3.5, 6.0, 140.0
    )
    signature = infrastructure.custom_road_model_signature(model)
    assert signature is not None
    assert signature[3] == infrastructure.CUSTOM_ROAD_MAX_CURVE_DEGREES


def test_milestone9_enables_custom_road_shapes_by_default() -> None:
    assert Milestone9Spec().custom_road_shapes is True


def test_custom_road_library_emits_all_three_surface_families(tmp_path: Path) -> None:
    models = (
        infrastructure.custom_road_model_path("customroads", "paved", 9.1, 12.4, 17.0),
        infrastructure.custom_road_model_path("customroads", "gravel", 4.6, 9.2, -23.0),
        infrastructure.custom_road_model_path("customroads", "dirt", 3.5, 7.8, 31.0),
    )
    library = infrastructure.ProceduralInfrastructureLibrary(
        "customroads",
        paved_texture_path=r"o\road\sil_new.paa",
        cache_enabled=False,
    )
    library.register_models(models)
    result = library.write_assets(tmp_path, tmp_path / "infrastructure.json")

    assert result.generated_variants == 3
    assert all((tmp_path / model.split("\\", 1)[1].replace("\\", "/")).is_file() for model in models)
    assert (tmp_path / "i" / "g.paa").is_file()
    assert (tmp_path / "i" / "dt.paa").is_file()
    assert not (tmp_path / "i" / "pv.paa").exists()


@pytest.mark.parametrize(
    ("surface", "base_model", "pieces"),
    (
        (
            "paved",
            r"o\road\sil25.p3d",
            (playability._RoadPiece(r"o\road\sil6.p3d", 6.0, 6),),
        ),
        (
            "gravel",
            r"customroads\i\gravel25.p3d",
            (playability._RoadPiece(r"customroads\i\gravel6.p3d", 6.0, 6),),
        ),
        (
            "dirt",
            r"o\road\ces25.p3d",
            (playability._RoadPiece(r"o\road\ces6.p3d", 6.0, 6),),
        ),
    ),
)
def test_sharp_chain_can_be_replaced_by_custom_surface(
    surface: str,
    base_model: str,
    pieces,
) -> None:
    spec = _spec()
    if surface == "gravel":
        # The generated gravel family is identified directly from its model names.
        pass
    elif surface == "dirt":
        assert base_model == spec.dirt_road_model
    else:
        assert base_model == spec.paved_road_model

    measure = playability._PolylineMeasure.create(
        ((0.0, 0.0), (2.5, 0.0), (4.5, 2.5), (6.5, 5.0))
    )
    piece = pieces[0]
    endpoint = measure.chord_endpoint(0.0, piece.length_metres, measure.total)
    assert endpoint is not None
    end_distance, end_x, end_z, _heading = endpoint
    result = ((piece, (0.0, 0.0), (end_x, end_z)),)

    context = quality._Context((), spec, {})
    token = quality._CONTEXT.set(context)
    try:
        upgraded = fallback._upgrade_stock_result(
            result,
            measure,
            pieces,
            start_distance=0.0,
            preferred_end_distance=end_distance,
            minimum_end_distance=end_distance,
            maximum_end_distance=measure.total,
        )
    finally:
        quality._CONTEXT.reset(token)

    assert len(upgraded) == 1
    generated = upgraded[0][0].model_path
    signature = infrastructure.custom_road_model_signature(generated)
    assert signature is not None
    assert signature[0] == surface
    assert abs(signature[3]) not in infrastructure.GENERATED_PAVED_CURVE_BUCKETS


def test_custom_road_model_reuses_identical_shape() -> None:
    first = infrastructure.custom_road_model_path(
        "customroads", "gravel", 4.62, 10.04, 26.6
    )
    second = infrastructure.custom_road_model_path(
        "customroads", "gravel", 4.61, 10.01, 26.8
    )
    assert first == second
