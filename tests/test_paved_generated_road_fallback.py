from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen import playability
from cwr_worldgen import paved_junction_policy as junctions
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


def test_stock_junction_approach_reserve_never_uses_generated_paved_microsegment() -> None:
    spec = _spec()
    pieces = playability.road_model_variants(
        spec.paved_road_model,
        spec.road_segment_length,
    )
    piece = next(item for item in pieces if item.nominal_length == 6)
    measure = playability._PolylineMeasure.create(
        ((0.0, 0.0), (2.0, 3.0), (5.0, 4.0), (7.0, 1.0))
    )
    key = playability._road_node_key(measure.points[0])
    plan_token = junctions._PLANS.set({
        key: SimpleNamespace(
            model_path=r"o\road\kr_new_sil_sil_t.p3d"
        )
    })
    try:
        upgraded = _upgrade(measure, pieces, piece, spec)
    finally:
        junctions._PLANS.reset(plan_token)

    assert upgraded[0][0].model_path == piece.model_path
    assert not infrastructure.is_generated_paved_road_model(
        upgraded[0][0].model_path
    )


def test_generated_junction_approach_still_allows_generated_paved_fallback() -> None:
    spec = _spec()
    pieces = playability.road_model_variants(
        spec.paved_road_model,
        spec.road_segment_length,
    )
    piece = next(item for item in pieces if item.nominal_length == 6)
    measure = playability._PolylineMeasure.create(
        ((0.0, 0.0), (2.0, 3.0), (5.0, 4.0), (7.0, 1.0))
    )
    key = playability._road_node_key(measure.points[0])
    plan_token = junctions._PLANS.set({
        key: SimpleNamespace(
            model_path=(
                "paved_fallback_test\\i\\"
                r"paved_j3_w091_h000_095_190.p3d"
            )
        )
    })
    try:
        upgraded = _upgrade(measure, pieces, piece, spec)
    finally:
        junctions._PLANS.reset(plan_token)

    assert infrastructure.is_generated_paved_road_model(
        upgraded[0][0].model_path
    )


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
        "reuse_world", 9.12, 6.23, 21.0
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

    stock_texture = r"o\road\sil_new.paa"
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
    assert stock_texture in summary.texture_paths
    assert r"o\road\sil_konec.paa" not in summary.texture_paths
    assert any(
        math.isclose(value, infrastructure._ROADWAY_LOD, rel_tol=1.0e-7)
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


def test_generated_paved_junction_signature_writes_exact_heading_asset(
    tmp_path: Path,
) -> None:
    directions = tuple(
        (
            math.sin(math.radians(heading)),
            math.cos(math.radians(heading)),
        )
        for heading in (5.0, 190.0, 270.0)
    )
    headings, axis = infrastructure.paved_junction_signature_for_directions(
        directions
    )
    assert headings == (0, 80, 175)
    assert math.isclose(math.hypot(*axis), 1.0, abs_tol=1.0e-9)

    model = infrastructure.paved_junction_signature_model_path(
        "reuse_world",
        9.10,
        headings,
    )
    assert model.endswith(
        r"\paved_j3_w091_h000_080_175.p3d"
    )
    assert infrastructure.is_generated_paved_junction_model(model)

    library = infrastructure.ProceduralInfrastructureLibrary(
        "reuse_world",
        paved_texture_path=r"o\road\sil_new.paa",
        cache_enabled=False,
    )
    library.register_model_usage(model)
    result = library.write_assets(
        tmp_path,
        tmp_path / "infrastructure.json",
    )
    assert result.generated_variants == 1

    key = infrastructure.InfrastructureModelKey(
        "road",
        "paved_j3_w091_h000_080_175",
        91,
        int(round(
            infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES * 20.0
        )),
    )
    visual, _map_geometry, roadway, _land = infrastructure._road_lods(
        key,
        r"o\road\sil_new.paa",
    )
    textures = [face.texture for face in visual.faces]
    assert textures.count(r"o\road\sil_new.paa") == 16
    assert r"o\road\sil_konec.paa" not in textures
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


def test_generated_curved_paved_turn_restores_terrtest39_seam_geometry() -> None:
    curved_key = infrastructure.InfrastructureModelKey(
        "road",
        "paved_w091_l0059_r45",
        91,
        59,
    )
    visual, _map_geometry, roadway, _land = infrastructure._road_lods(
        curved_key,
        r"o\road\sil_new.paa",
    )

    # terrtest39 used tangent-aligned turn mouths. terrtest44's square ends made
    # both edge points share the same Z plane and exposed large triangular grass
    # wedges between consecutive turn pieces.
    first_left, first_right = roadway.points[0], roadway.points[1]
    last_left, last_right = roadway.points[-2], roadway.points[-1]
    assert not math.isclose(first_left[2], first_right[2], abs_tol=0.10)
    assert not math.isclose(last_left[2], last_right[2], abs_tol=0.10)

    # The visual LOD extends 0.18 m beyond the nominal tangent mouth and drops
    # only that overlap below the paved surface, matching the seam treatment in
    # terrtest39 without changing the Roadway extent.
    assert infrastructure.GENERATED_PAVED_TURN_VISUAL_OVERLAP_METRES == 0.18
    expected_first_visual = (
        (-4.2943, -0.0150, -1.4354),
        (4.1612, -0.0150, -4.7991),
    )
    for actual, expected in zip(visual.points[:2], expected_first_visual):
        assert math.isclose(actual[0], expected[0], abs_tol=5.0e-4)
        assert math.isclose(actual[1], expected[1], abs_tol=5.0e-4)
        assert math.isclose(actual[2], expected[2], abs_tol=5.0e-4)

    assert min(point[1] for point in visual.points) < (
        infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES - 0.03
    )
    assert all(
        math.isclose(point[1], infrastructure.GENERATED_GRAVEL_ROADWAY_HEIGHT_METRES, abs_tol=1.0e-9)
        for point in roadway.points
    )

    # Straight generated paved pieces keep square mouths for junction approach
    # alignment. This fix is deliberately turn-only.
    straight_key = infrastructure.InfrastructureModelKey(
        "road",
        "paved_w091_l0059",
        91,
        59,
    )
    straight_visual, _geometry, straight_roadway, _land = infrastructure._road_lods(
        straight_key,
        r"o\road\sil_new.paa",
    )
    assert math.isclose(
        straight_roadway.points[0][2],
        straight_roadway.points[1][2],
        abs_tol=1.0e-9,
    )
    assert math.isclose(
        straight_roadway.points[-2][2],
        straight_roadway.points[-1][2],
        abs_tol=1.0e-9,
    )
    assert min(point[1] for point in straight_visual.points) >= (
        infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES - 1.0e-9
    )


def test_generated_paved_junction_overlap_stays_above_terrain() -> None:
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
    assert math.isclose(
        visual_reach,
        infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES,
        abs_tol=0.01,
    )
    assert math.isclose(
        infrastructure.GENERATED_JUNCTION_OVERLAP_TONGUE_START_METRES,
        (
            infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES
            - infrastructure.GENERATED_PAVED_JUNCTION_APPROACH_OVERLAP_METRES
        ),
        abs_tol=1.0e-9,
    )
    assert infrastructure.GENERATED_JUNCTION_OVERLAP_VISUAL_RISE_METRES > 0.0
    assert max(point[1] for point in visual.points) > (
        infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES
    )
    assert min(point[1] for point in visual.points) > 0.0


def test_generated_paved_junction_uses_donor_stock_texture_topology() -> None:
    key = infrastructure.InfrastructureModelKey(
        "road",
        "paved_j3_m091_b091_a265",
        91,
        int(round(
            infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES * 20.0
        )),
    )
    # Wide generated T hubs use the selected donor surface consistently.
    visual = infrastructure._road_lods(
        key,
        r"o\road\sil_new.paa",
    )[0]

    textures = [face.texture for face in visual.faces]
    assert len(visual.faces) == 16
    assert textures.count(r"o\road\sil_new.paa") == 16
    assert r"o\road\sil_konec.paa" not in textures
    assert r"landtext\silnice.pac" not in textures
    # Poseidon/CWA culls the single generated triangle winding in game. Keep
    # every junction quad paired with its exact reverse winding, just like the
    # generated road ribbons that remain visible in terrtest116.
    for offset in range(0, len(visual.faces), 4):
        assert visual.faces[offset + 2].vertices == tuple(
            reversed(visual.faces[offset].vertices)
        )
        assert visual.faces[offset + 3].vertices == tuple(
            reversed(visual.faces[offset + 1].vertices)
        )

    # CWA resolves these intersecting donor-texture faces by MLOD face order more
    # reliably than by their tiny Y separation. The branch quad must be emitted
    # first (points 0..3), with the straight-through carriageway after it
    # (points 4..7), so the main road visually prints over the branch.
    assert all(
        0 <= vertex[0] <= 3
        for vertex in visual.faces[0].vertices
    )
    assert all(
        4 <= vertex[0] <= 7
        for vertex in visual.faces[4].vertices
    )
    assert min(point[1] for point in visual.points[:4]) < (
        infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES
    )
    assert all(
        math.isclose(
            point[1],
            infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES,
            abs_tol=1.0e-9,
        )
        for point in visual.points[4:8]
    )

    ys = [point[1] for point in visual.points]
    assert math.isclose(
        min(ys),
        (
            infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES
            - infrastructure.GENERATED_JUNCTION_INTERSECTING_VISUAL_DROP_METRES
        ),
        abs_tol=1.0e-7,
    )
    assert math.isclose(
        max(ys),
        (
            infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES
            + infrastructure.GENERATED_JUNCTION_OVERLAP_VISUAL_RISE_METRES
        ),
        abs_tol=1.0e-7,
    )
    assert all(
        flag == infrastructure._ROAD_SURFACE_POINT_FLAG
        for flag in visual.point_flags
    )


def test_generated_paved_junction_branch_mouth_layers_over_approach() -> None:
    # The incoming generated approach is authored at VISUAL_TOP. The T-junction
    # branch starts lower in the hub so the through road wins at the centre, but
    # must climb above the approach before the 0.22 m overlap begins. Otherwise
    # CWA draws the square end of the connecting road on top of the junction.
    y = infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES
    inner_y = y - infrastructure.GENERATED_JUNCTION_INTERSECTING_VISUAL_DROP_METRES
    outer_y = y + infrastructure.GENERATED_JUNCTION_OVERLAP_VISUAL_RISE_METRES
    extent = infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES
    overlap_start = infrastructure.GENERATED_JUNCTION_OVERLAP_TONGUE_START_METRES
    fraction = overlap_start / extent
    branch_y_at_overlap_start = inner_y + (outer_y - inner_y) * fraction

    assert branch_y_at_overlap_start > y
    assert outer_y > y


def test_exact_heading_quantization_fits_inside_approach_overlap() -> None:
    maximum_heading_error = (
        infrastructure.GENERATED_PAVED_JUNCTION_HEADING_STEP_DEGREES * 0.5
    )
    corner_sweep = (
        infrastructure.GENERATED_PAVED_HALF_WIDTH_METRES
        * math.sin(math.radians(maximum_heading_error))
    )
    overlap = infrastructure.GENERATED_PAVED_JUNCTION_APPROACH_OVERLAP_METRES

    assert maximum_heading_error == 2.5
    assert corner_sweep < 0.20
    assert math.isclose(overlap, 0.22, abs_tol=1.0e-9)
    assert overlap > corner_sweep


def test_generated_paved_asset_reuses_stock_texture_and_has_roadway_lod(
    tmp_path: Path,
) -> None:
    model = infrastructure.paved_fallback_model_path(
        "reuse_world", 4.55, 6.24, -19.0
    )
    stale = tmp_path / "i" / "pv.paa"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"obsolete-generated-asphalt")
    stock_texture = r"o\road\sil_new.paa"
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
    assert stock_texture in model_summary.texture_paths

    document = json.loads(
        (tmp_path / "infrastructure.json").read_text(encoding="utf-8")
    )
    assert document["models"][0]["usage_count"] == 3
    assert document["paved_texture_source"] == {
        "type": "external-road-texture",
        "texture": stock_texture,
        "generated_texture": False,
    }
    assert any(
        math.isclose(value, infrastructure._ROADWAY_LOD, rel_tol=1.0e-7)
        for value in document["models"][0]["lod_resolutions"]
    )



def test_generated_models_keep_visual_style_and_exact_donor_roadway_style() -> None:
    visual_texture = r"o\road\sil_new.paa"
    roadway_texture = r"o\road\donor_contact.paa"
    style = infrastructure.RoadSurfaceStyle(
        point_flag=0x13F,
        face_flag=0x24102,
        roadway_texture=roadway_texture,
        roadway_point_flag=0x55AA,
        roadway_face_flag=0x12345678,
    )
    keys = (
        infrastructure.InfrastructureModelKey(
            "road", "road_paved_w091_l0063", 91, 63
        ),
        infrastructure.InfrastructureModelKey(
            "road", "road_j3_paved_w091_h000_095_190", 91, 125
        ),
    )

    for key in keys:
        visual, _geometry, roadway, _land = infrastructure._road_lods(
            key,
            visual_texture,
            style,
        )
        assert visual.faces
        assert roadway.faces
        assert all(face.texture == visual_texture for face in visual.faces)
        assert all(face.texture == roadway_texture for face in roadway.faces)
        assert all(
            face.flags == style.roadway_face_flag
            for face in roadway.faces
        )
        assert roadway.point_flags == (
            style.roadway_point_flag,
        ) * len(roadway.points)


def test_unmodded_gravel_and_dirt_can_share_exact_donor_groundtype() -> None:
    dirt_contact = r"o\road\dirt_contact.paa"
    gravel_visual = r"reuse_world\i\g.paa"
    dirt_visual = r"o\road\ces_hned.paa"
    shared = dict(
        roadway_texture=dirt_contact,
        roadway_point_flag=0x77,
        roadway_face_flag=0x8811,
    )
    gravel_style = infrastructure.RoadSurfaceStyle(**shared)
    dirt_style = infrastructure.RoadSurfaceStyle(**shared)

    cases = (
        (
            infrastructure.InfrastructureModelKey(
                "road", "road_gravel_w046_l0063", 46, 63
            ),
            gravel_visual,
            gravel_style,
        ),
        (
            infrastructure.InfrastructureModelKey(
                "road", "road_dirt_w035_l0063", 35, 63
            ),
            dirt_visual,
            dirt_style,
        ),
    )

    for key, visual_texture, style in cases:
        visual, _geometry, roadway, _land = infrastructure._road_lods(
            key,
            visual_texture,
            style,
        )
        assert all(face.texture == visual_texture for face in visual.faces)
        assert all(face.texture == dirt_contact for face in roadway.faces)
        assert roadway.point_flags == (
            style.roadway_point_flag,
        ) * len(roadway.points)
        assert all(
            face.flags == style.roadway_face_flag
            for face in roadway.faces
        )


def test_generated_junction_intersecting_branches_stay_below_through_road() -> None:
    branch_drop = infrastructure.GENERATED_JUNCTION_INTERSECTING_VISUAL_DROP_METRES
    overlap_rise = infrastructure.GENERATED_JUNCTION_OVERLAP_VISUAL_RISE_METRES
    top = infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES
    keys = (
        infrastructure.InfrastructureModelKey(
            "road", "road_j3_paved_w091_h000_095_190", 91, 125
        ),
        infrastructure.InfrastructureModelKey(
            "road", "road_j4_gravel_w046_h000_090_180_270", 46, 125
        ),
        infrastructure.InfrastructureModelKey(
            "road", "road_j3_dirt_w035_h000_090_180", 35, 125
        ),
    )

    for key in keys:
        visual, _geometry, _roadway, _land = infrastructure._road_lods(
            key,
            r"roads\surface.paa",
        )
        ys = tuple(point[1] for point in visual.points)
        assert math.isclose(max(ys), top + overlap_rise, abs_tol=1.0e-9)
        assert math.isclose(min(ys), top - branch_drop, abs_tol=1.0e-9)
        assert any(
            math.isclose(value, top, abs_tol=1.0e-9)
            for value in ys
        )
        assert min(ys) >= -1.0e-9


def test_generated_roads_and_junctions_inherit_donor_surface_metadata() -> None:
    texture = r"myroads\surface.paa"
    style = infrastructure.RoadSurfaceStyle(
        point_flag=0x0040013F,
        face_flag=0x0007C142,
    )
    keys = (
        infrastructure.InfrastructureModelKey(
            "road", "road_paved_w060_l0125_r015", 60, 125
        ),
        infrastructure.InfrastructureModelKey(
            "road", "road_j3_paved_w060_h000_090_180", 60, 125
        ),
        infrastructure.InfrastructureModelKey(
            "road", "road_gravel_w046_l0060", 46, 60
        ),
        infrastructure.InfrastructureModelKey(
            "road", "road_j3_gravel_w046_h000_090_180", 46, 125
        ),
    )

    for key in keys:
        visual, _geometry, roadway, _land = infrastructure._road_lods(
            key,
            texture,
            style,
        )
        assert visual.faces
        assert roadway.faces
        assert visual.point_flags == (style.point_flag,) * len(visual.points)
        assert roadway.point_flags == (
            style.roadway_point_flag,
        ) * len(roadway.points)
        assert all(face.flags == style.face_flag for face in visual.faces)
        assert all(
            face.flags == style.roadway_face_flag
            for face in roadway.faces
        )
        # CfgSurfaces matches the Roadway face texture. Leaving this blank makes
        # a road look like its donor while driving like the default surface.
        assert all(face.texture == texture for face in roadway.faces)


def test_generated_roadway_lod_sits_above_visual_skin() -> None:
    key = infrastructure.InfrastructureModelKey(
        "road", "road_paved_w091_l0063", 91, 63
    )
    contact_texture = r"o\road\donor_contact.paa"
    visual, _geometry, roadway, _land = infrastructure._road_lods(
        key,
        r"o\road\sil_new.paa",
        infrastructure.RoadSurfaceStyle(
            roadway_texture=contact_texture,
            roadway_point_flag=0x44,
            roadway_face_flag=0x5500,
        ),
    )

    assert math.isclose(
        infrastructure.GENERATED_GRAVEL_ROADWAY_HEIGHT_METRES
        - infrastructure.GENERATED_GRAVEL_VISUAL_TOP_METRES,
        0.01,
        abs_tol=1.0e-9,
    )
    assert min(point[1] for point in roadway.points) > max(
        point[1] for point in visual.points
    )
    assert all(face.texture == contact_texture for face in roadway.faces)
    assert roadway.point_flags == (0x44,) * len(roadway.points)
    assert all(face.flags == 0x5500 for face in roadway.faces)


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
        assert roadway.point_flags == (0,) * len(roadway.points)
        assert visual.normals == (infrastructure._ROAD_SURFACE_NORMAL,)
        assert roadway.normals == (infrastructure._ROAD_SURFACE_NORMAL,)
        assert visual.faces
        assert roadway.faces
        assert all(
            face.flags == infrastructure._ROAD_SURFACE_FACE_FLAG
            for face in visual.faces
        )
        assert all(face.flags == 0 for face in roadway.faces)
        assert all(
            face.texture == r"o\road\sil_new.paa"
            for face in roadway.faces
        )
