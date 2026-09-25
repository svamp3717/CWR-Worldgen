from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen import generator
from cwr_worldgen import gravel_junction_policy as gravel_junctions
from cwr_worldgen import playability
from cwr_worldgen import procedural_infrastructure as infrastructure
from cwr_worldgen import paved_road_generated_fallback_policy as fallback
from cwr_worldgen import paved_junction_policy as paved_junctions
from cwr_worldgen import road_quality_policy as quality
from cwr_worldgen.assets import AssetRecord, model_texture_dependencies
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


def test_stock_paved_texture_dependency_resolves_relative_to_model_directory() -> None:
    records = (
        AssetRecord(
            path=r"o\road\sil25.p3d",
            source="unused.pbo",
            size=1,
            sha256=None,
            dependencies=("silnice.pac",),
        ),
        AssetRecord(
            path=r"o\road\silnice.pac",
            source="unused.pbo",
            size=1,
            sha256=None,
        ),
    )

    assert model_texture_dependencies(
        records,
        r"o\road\sil25.p3d",
    ) == (r"o\road\silnice.pac",)


def test_stock_paved_texture_chooser_prefers_configured_road_family() -> None:
    selected = generator._preferred_stock_paved_texture(
        r"o\road\sil25.p3d",
        (
            r"o\road\detail.paa",
            r"o\road\silnice.pac",
            r"o\road\asfaltka.pac",
        ),
    )

    assert selected == r"o\road\silnice.pac"
    assert generator._preferred_stock_paved_texture(
        r"o\road\sil25.p3d",
        (),
    ) == r"landtext\silnice.pac"


def test_generated_paved_and_gravel_roads_use_native_onsurface_vertex_lighting() -> None:
    cases = (
        (
            infrastructure.InfrastructureModelKey(
                "road", "paved_w091_l0062_r20", 91, 62
            ),
            r"o\road\silnice.pac",
        ),
        (
            infrastructure.InfrastructureModelKey("road", "gravel6", 46, 62),
            r"lighting_world\i\g.paa",
        ),
        (
            infrastructure.InfrastructureModelKey("road", "gravel_j3", 46, 54),
            r"lighting_world\i\gj.paa",
        ),
    )

    for key, texture in cases:
        lods = infrastructure._road_lods(key, texture)
        visual = lods[0]
        roadway = next(
            lod
            for lod in lods
            if abs(lod.resolution - infrastructure._ROADWAY_LOD) < 1.0
        )
        expected_visual = (infrastructure._ROAD_SURFACE_POINT_FLAG,) * len(
            visual.points
        )
        expected_roadway = (infrastructure._ROAD_SURFACE_POINT_FLAG,) * len(
            roadway.points
        )
        assert infrastructure._ROAD_SURFACE_POINT_FLAG == 0x0000013F
        assert infrastructure._ROAD_SURFACE_FACE_FLAG == 0x0002C102
        assert infrastructure._ROAD_SURFACE_NORMAL == (0.0, -1.0, 0.0)
        assert visual.point_flags == expected_visual
        assert roadway.point_flags == expected_roadway
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

def test_generated_paved_junction_signature_reuses_rotation_invariant_shape() -> None:
    first_signature, first_axis = (
        infrastructure.paved_junction_signature_for_directions(
            ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0))
        )
    )
    angle = math.radians(27.0)

    def rotate(direction: tuple[float, float]) -> tuple[float, float]:
        x, z = direction
        return (
            x * math.cos(angle) + z * math.sin(angle),
            -x * math.sin(angle) + z * math.cos(angle),
        )

    second_signature, second_axis = (
        infrastructure.paved_junction_signature_for_directions(
            tuple(
                rotate(direction)
                for direction in ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0))
            )
        )
    )

    assert first_signature == (0, 90, 180)
    assert second_signature == first_signature
    assert abs(math.hypot(*first_axis) - 1.0) < 1.0e-9
    assert abs(math.hypot(*second_axis) - 1.0) < 1.0e-9


def test_generated_paved_junction_asset_uses_stock_texture_and_road_metadata(
    tmp_path: Path,
) -> None:
    signature, _axis = infrastructure.paved_junction_signature_for_directions(
        ((0.0, 1.0), (1.0, 0.0), (0.0, -1.0))
    )
    model = infrastructure.paved_junction_model_path(
        "junction_world",
        9.10,
        signature,
    )
    assert model.endswith(r"\paved_j3_w091_h000_090_180.p3d")
    assert infrastructure.is_generated_paved_road_model(model)

    stock_texture = r"o\road\sil_new.paa"
    library = infrastructure.ProceduralInfrastructureLibrary(
        "junction_world",
        paved_texture_path=stock_texture,
        cache_enabled=False,
    )
    library.register_model_usage(model, 2)
    result = library.write_assets(
        tmp_path,
        tmp_path / "infrastructure.json",
    )

    assert result.placements == 2
    assert result.generated_variants == 1
    assert result.texture_files == ()

    model_summary = infrastructure.inspect_mlod(
        tmp_path / result.model_files[0]
    )
    assert stock_junction_texture in model_summary.textures
    assert stock_texture not in model_summary.textures

    document = json.loads(
        (tmp_path / "infrastructure.json").read_text(encoding="utf-8")
    )
    entry = document["models"][0]
    assert entry["key"]["subtype"] == "paved_j3_w091_h000_090_180"
    assert entry["key"]["width_dm"] == 91
    assert entry["usage_count"] == 2
    assert any(
        abs(value - infrastructure._ROADWAY_LOD) < 1.0
        for value in entry["lod_resolutions"]
    )

    key = infrastructure.InfrastructureModelKey(
        "road",
        "paved_j3_w091_h000_090_180",
        91,
        int(
            round(
                infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES
                * 20.0
            )
        ),
    )
    visual, _map_geometry, roadway, _land = infrastructure._road_lods(
        key,
        stock_texture,
    )
    assert visual.normals == (infrastructure._ROAD_SURFACE_NORMAL,)
    assert roadway.normals == (infrastructure._ROAD_SURFACE_NORMAL,)
    assert visual.point_flags == (
        infrastructure._ROAD_SURFACE_POINT_FLAG,
    ) * len(visual.points)
    assert roadway.point_flags == (
        infrastructure._ROAD_SURFACE_POINT_FLAG,
    ) * len(roadway.points)
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


def test_generated_paved_junction_width_tracks_stock_family() -> None:
    assert infrastructure.paved_junction_width_for_models(
        (r"o\road\sil6.p3d", r"o\road\sil12.p3d")
    ) == 9.1
    assert infrastructure.paved_junction_width_for_models(
        (r"o\road\asf6.p3d", r"o\road\asf12.p3d")
    ) == 7.0
    assert infrastructure.paved_junction_width_for_models(
        (
            r"junction_world\i\paved_w083_l0060.p3d",
            r"o\road\asf6.p3d",
        )
    ) == 8.3

def test_generated_paved_junction_exit_distance_uses_full_arm_extent() -> None:
    junction = quality._Junction(
        point=(0.0, 0.0),
        axis=(0.0, 1.0),
        half_length=infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES,
        half_width=4.55,
        directions=((0.0, 1.0), (0.0, -1.0), (1.0, 0.0)),
    )
    for direction in junction.directions:
        assert math.isclose(
            quality._exit_distance(junction, direction),
            infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES,
            abs_tol=1.0e-9,
        )

def test_generated_paved_junction_visual_has_continuous_uvs() -> None:
    key = infrastructure.InfrastructureModelKey(
        "road",
        "paved_j3_w091_h000_090_180",
        91,
        int(
            round(
                infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES
                * 20.0
            )
        ),
    )
    visual = infrastructure._road_lods(
        key,
        r"o\road\sil_new.paa",
    )[0]

    uvs_by_point: dict[int, set[tuple[float, float]]] = {}
    for face in visual.faces:
        for point_index, _normal_index, u, v in face.vertices:
            uvs_by_point.setdefault(point_index, set()).add(
                (round(float(u), 7), round(float(v), 7))
            )

    # A single generated hub may be triangulated internally, but shared points
    # must keep one UV coordinate. Otherwise triangle boundaries become visible
    # as wedge-shaped "overlapping P3Ds" in CWA.
    assert uvs_by_point
    assert all(len(values) == 1 for values in uvs_by_point.values())


def test_generated_paved_junction_quality_window_has_no_coplanar_overlap() -> None:
    junction = quality._Junction(
        point=(0.0, 0.0),
        axis=(0.0, 1.0),
        half_length=infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES,
        half_width=4.55,
        directions=((0.0, 1.0), (0.0, -1.0), (1.0, 0.0)),
    )
    measure = playability._PolylineMeasure.create(
        ((0.0, 0.0), (0.0, 40.0))
    )
    piece = playability._RoadPiece(r"o\road\sil6.p3d", 6.25, 6)
    context = quality._Context(
        (),
        SimpleNamespace(cells=4, cell_size=10.0),
        {playability._road_node_key((0.0, 0.0)): junction},
    )

    start, _preferred, _minimum, _maximum = quality._quality_window(
        measure,
        (piece,),
        0.0,
        measure.total,
        measure.total,
        measure.total,
        context,
    )

    assert infrastructure.GENERATED_PAVED_JUNCTION_APPROACH_OVERLAP_METRES == 0.0
    assert math.isclose(
        start,
        infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES,
        abs_tol=1.0e-9,
    )

def test_gravel_quality_wrapper_uses_generated_paved_overlap_constant() -> None:
    junction = quality._Junction(
        point=(0.0, 0.0),
        axis=(0.0, 1.0),
        half_length=infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES,
        half_width=4.55,
        directions=((0.0, 1.0), (0.0, -1.0), (1.0, 0.0)),
    )
    previous = gravel_junctions._RQ
    gravel_junctions._RQ = quality
    try:
        assert gravel_junctions._overlap_for(junction) == (
            infrastructure.GENERATED_PAVED_JUNCTION_APPROACH_OVERLAP_METRES
        )
    finally:
        gravel_junctions._RQ = previous

def test_generated_paved_junction_texture_routing_reuses_stock_road_artwork() -> None:
    stock_texture = r"o\road\sil_new.paa"
    library = infrastructure.ProceduralInfrastructureLibrary(
        "junction_world",
        paved_texture_path=stock_texture,
        paved_t_junction_texture_path=r"o\road\wrong_t_atlas.paa",
        paved_asf_t_junction_texture_path=r"o\road\wrong_asf_atlas.paa",
        paved_x_junction_texture_path=r"o\road\wrong_x_atlas.paa",
        cache_enabled=False,
    )

    wide_t = infrastructure.InfrastructureModelKey(
        "road", "paved_j3_w091_h000_090_180", 91, 125
    )
    narrow_t = infrastructure.InfrastructureModelKey(
        "road", "paved_j3_w070_h000_090_180", 70, 125
    )
    wide_x = infrastructure.InfrastructureModelKey(
        "road", "paved_j4_w091_h000_090_180_270", 91, 125
    )
    straight = infrastructure.InfrastructureModelKey(
        "road", "paved_w091_l0062", 91, 62
    )

    # Generated hub geometry has its own UV layout. It must therefore use the
    # same verified in-game road surface as the approaches, not a kr_new model's
    # texture dependency/atlas.
    assert library._texture_path(wide_t) == stock_texture
    assert library._texture_path(narrow_t) == stock_texture
    assert library._texture_path(wide_x) == stock_texture
    assert library._texture_path(straight) == stock_texture

def test_stock_junction_texture_preference_avoids_straight_road_artwork() -> None:
    selected = generator._preferred_stock_junction_texture(
        r"o\road\kr_new_sil_sil_t.p3d",
        (
            r"o\road\sil_new.paa",
            r"o\road\kr_new_sil_sil_t.paa",
        ),
        paved_fallback=r"o\road\sil_new.paa",
    )
    assert selected == r"o\road\kr_new_sil_sil_t.paa"

