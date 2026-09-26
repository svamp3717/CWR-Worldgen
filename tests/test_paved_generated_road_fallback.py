from __future__ import annotations

import json
import math
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
    assert first.endswith(r"paved_w091_l0062_r20.p3d")


def test_generated_paved_junction_quantizes_heading_and_reuses_stock_texture(
    tmp_path: Path,
) -> None:
    first = infrastructure.paved_junction_model_path(
        "reuse_world", 9.10, 9.10, 258.0
    )
    second = infrastructure.paved_junction_model_path(
        "reuse_world", 9.08, 9.12, 261.0
    )
    assert first == second
    assert first.endswith(r"\paved_j3_m091_b091_a260.p3d")
    assert infrastructure.is_generated_paved_junction_model(first)

    stock_texture = r"landtext\silnice.pac"
    library = infrastructure.ProceduralInfrastructureLibrary(
        "reuse_world",
        paved_texture_path=stock_texture,
        cache_enabled=False,
    )
    library.register_model_usage(first, 2)
    result = library.write_assets(
        tmp_path,
        tmp_path / "infrastructure.json",
    )

    assert result.placements == 2
    assert result.generated_variants == 1
    assert result.texture_files == ()
    summary = infrastructure.inspect_mlod(
        tmp_path / result.model_files[0]
    )
    assert stock_texture in summary.textures
    assert any(
        abs(value - infrastructure._ROADWAY_LOD) < 1.0
        for value in summary.resolutions
    )

    key = infrastructure.InfrastructureModelKey(
        "road",
        "paved_j3_m091_b091_a260",
        91,
        int(round(
            infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES * 20.0
        )),
    )
    visual, _map_geometry, roadway, _land = infrastructure._road_lods(
        key,
        stock_texture,
    )
    assert min(point[0] for point in visual.points) < -5.5
    assert max(point[2] for point in visual.points) > 6.0
    assert min(point[2] for point in visual.points) < -6.0
    assert visual.faces
    assert roadway.faces


def test_generated_paved_hub_and_ribbon_match_stock_surface_height() -> None:
    spec = SimpleNamespace(cells=4, cell_size=10.0)
    elevations = (0.0,) * 16
    models = (
        infrastructure.paved_fallback_model_path(
            "height_world", 9.10, 6.20, 10.0
        ),
        infrastructure.paved_junction_model_path(
            "height_world", 9.10, 9.10, 260.0
        ),
    )
    requested_surface_height = 0.035

    for index, model in enumerate(models, start=1):
        obj = playability._road_object_on_slope(
            index,
            model,
            (0.0, 0.0),
            (0.0, 6.25),
            elevations,
            spec,
            vertical_offset=requested_surface_height,
        )
        assert math.isclose(
            obj.y + infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES,
            requested_surface_height,
            abs_tol=1.0e-9,
        )


def test_generated_curved_paved_piece_has_square_nominal_seam_ends() -> None:
    key = infrastructure.InfrastructureModelKey(
        "road",
        "paved_w091_l0062_r20",
        91,
        62,
    )
    visual, _map_geometry, roadway, _land = infrastructure._road_lods(
        key,
        r"o\road\sil_new.paa",
    )

    half_length = key.length_m * 0.5
    for lod in (visual, roadway):
        first_left, first_right = lod.points[0], lod.points[1]
        last_left, last_right = lod.points[-2], lod.points[-1]
        assert math.isclose(first_left[2], first_right[2], abs_tol=1.0e-9)
        assert math.isclose(last_left[2], last_right[2], abs_tol=1.0e-9)
        assert min(point[2] for point in lod.points) >= -half_length - 1.0e-9
        assert max(point[2] for point in lod.points) <= half_length + 1.0e-9

    assert infrastructure.GENERATED_PAVED_VISUAL_OVERLAP_METRES == 0.0
    assert infrastructure.GENERATED_PAVED_TEXTURE_REPEAT_METRES == 6.25


def test_generated_paved_junction_visual_overhang_covers_logical_seam() -> None:
    key = infrastructure.InfrastructureModelKey(
        "road",
        "paved_j3_m091_b091_a260",
        91,
        int(round(
            infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES * 20.0
        )),
    )
    visual, _map_geometry, roadway, _land = infrastructure._road_lods(
        key,
        r"o\road\sil_new.paa",
    )

    branch_heading = math.radians(260.0)
    branch_direction = (math.sin(branch_heading), math.cos(branch_heading))
    visual_reach = max(
        point[0] * branch_direction[0] + point[2] * branch_direction[1]
        for point in visual.points
    )
    roadway_reach = max(
        point[0] * branch_direction[0] + point[2] * branch_direction[1]
        for point in roadway.points
    )

    assert math.isclose(
        roadway_reach,
        infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES,
        abs_tol=0.01,
    )
    assert visual_reach >= (
        infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES
        + infrastructure.GENERATED_PAVED_JUNCTION_VISUAL_OVERHANG_METRES
        - 0.01
    )
    assert (
        infrastructure.GENERATED_PAVED_JUNCTION_APPROACH_CLEARANCE_METRES
        < infrastructure.GENERATED_PAVED_JUNCTION_VISUAL_OVERHANG_METRES
    )


def test_generated_paved_junction_visual_overhang_covers_terrtest41_skew() -> None:
    # terrtest41 places a generated T correctly, but one real paved approach
    # reaches the averaged main axis about 10.7 degrees off the hub heading.
    # A square 9.10 m road end rotates its outer corner by roughly 0.86 m.
    # Keep enough visual-only hub coverage beyond the 6.45 m approach start
    # to hide that corner while Roadway remains on the stock 6.25 m radius.
    measured_skew_degrees = 10.7
    required_corner_sweep = (
        infrastructure.GENERATED_PAVED_HALF_WIDTH_METRES
        * math.tan(math.radians(measured_skew_degrees))
    )
    visible_overlap = (
        infrastructure.GENERATED_PAVED_JUNCTION_VISUAL_OVERHANG_METRES
        - infrastructure.GENERATED_PAVED_JUNCTION_APPROACH_CLEARANCE_METRES
    )

    assert required_corner_sweep > 0.85
    assert visible_overlap >= required_corner_sweep + 0.10


def test_generated_paved_asset_reuses_stock_texture_and_has_roadway_lod(
    tmp_path: Path,
) -> None:
    model = infrastructure.paved_fallback_model_path(
        "reuse_world", 4.55, 6.24, -19.0
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


def test_generated_paved_and_gravel_roads_use_native_render_metadata() -> None:
    keys = (
        infrastructure.InfrastructureModelKey(
            "road", "paved_w091_l0062", 91, 62
        ),
        infrastructure.InfrastructureModelKey(
            "road",
            "paved_j3_m091_b091_a260",
            91,
            int(round(
                infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES * 20.0
            )),
        ),
        infrastructure.InfrastructureModelKey(
            "road",
            "gravel6",
            int(round(infrastructure.GENERATED_GRAVEL_HALF_WIDTH_METRES * 20.0)),
            60,
        ),
    )
    assert infrastructure._ROAD_SURFACE_POINT_FLAG == 0x0000013F
    assert infrastructure._ROAD_SURFACE_FACE_FLAG == 0x0002C102
    assert infrastructure._ROAD_SURFACE_NORMAL == (0.0, -1.0, 0.0)

    for key in keys:
        visual, _map_geometry, roadway, _land = infrastructure._road_lods(
            key,
            r"o\road\sil_new.paa",
        )
        assert visual.point_flags == (
            infrastructure._ROAD_SURFACE_POINT_FLAG,
        ) * len(visual.points)
        assert roadway.point_flags == (
            infrastructure._ROAD_SURFACE_POINT_FLAG,
        ) * len(roadway.points)
        assert visual.normals == (infrastructure._ROAD_SURFACE_NORMAL,)
        assert roadway.normals == (infrastructure._ROAD_SURFACE_NORMAL,)
        assert visual.faces
        assert roadway.faces
        assert all(
            face.flags == infrastructure._ROAD_SURFACE_FACE_FLAG
            for face in visual.faces
        )
        assert all(
            face.flags == infrastructure._ROAD_SURFACE_FACE_FLAG
            for face in roadway.faces
        )
