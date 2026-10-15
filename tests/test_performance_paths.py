from __future__ import annotations

from dataclasses import asdict, dataclass
import struct
from hashlib import sha256
import random
from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen.cache import (
    float_sequence_sha256, int_sequence_sha256, streaming_hash,
)
from cwr_worldgen.generator import (
    _small_report_dict,
    _assemble_world_objects,
    _expand_cwa_generated_vegetation,
    _trusted_legacy_asset_paths,
)
from cwr_worldgen.model import WorldObject
from cwr_worldgen.playability import (
    RoadFitReport, TerrainGradeReport, TransitionReport,
)
from cwr_worldgen.osm import (
    BboxProjection, CompactOrientedRectangle, ObjectGenerationResult,
    _forest_candidate_mask_ok, _mask_at,
    _compact_support_polygon, _oriented_rectangle,
)
from cwr_worldgen.procedural_buildings import (
    _polygon_with_footprint,
    _simple_rectangle_footprint,
)
from cwr_worldgen.procedural_forests import (
    ProceduralForestClusterLibrary,
    cluster_model_path,
    generated_cluster_variant,
    is_generated_cluster_model,
    VIETNAM_PROXY_MODELS,
    VIETNAM_BORDER_PROXY_MODELS,
    VIETNAM_DITCH_PROXY_MODELS,
)
from cwr_worldgen.wrp import _height_grid_bytes, inspect_rvw4, quantize_height, write_rvw4


def _object(object_id: int, model: str) -> WorldObject:
    return WorldObject(object_id, model, float(object_id), 1.25, float(object_id * 2), 90.0)


def test_simple_rectangle_fast_path_matches_shapely_dimensions_and_centroid() -> None:
    points = ((10.0, 20.0), (22.0, 20.0), (22.0, 38.0), (10.0, 38.0))
    fast = _simple_rectangle_footprint(points)
    assert fast is not None
    footprint, centre_x, centre_z = fast
    polygon, reference = _polygon_with_footprint(points)
    assert footprint.width_m == reference.width_m
    assert footprint.length_m == reference.length_m
    assert centre_x == float(polygon.centroid.x)
    assert centre_z == float(polygon.centroid.y)

    # Anything genuinely irregular remains on the mature Shapely/native path.
    assert _simple_rectangle_footprint(((0.0, 0.0), (12.0, 0.0), (10.0, 8.0), (0.0, 8.0))) is None


def test_vectorized_height_bytes_match_scalar_rvw4_quantization() -> None:
    values = tuple(index * 0.0125 for index in range(-80, 81))
    encoded = _height_grid_bytes(values, 0.05)
    unpacked = struct.unpack(f"<{len(values)}h", encoded)
    assert unpacked == tuple(quantize_height(value, 0.05) for value in values)


def test_large_world_ordering_can_skip_clone_and_writer_renumbers(tmp_path: Path) -> None:
    road = _object(1, r"o\road\sil25.p3d")
    buildings = (_object(2, r"world\g\house_a.p3d"), _object(3, r"world\g\house_b.p3d"))
    forests = (_object(4, r"data3d\les ctverec pruchozi_T1.p3d"),)
    rural = (_object(5, r"data3d\str borovice.p3d"),)
    semantic = _object(6, r"world\s\site.p3d")
    nonroads = ObjectGenerationResult(
        objects=buildings + forests + rural,
        road_objects=0,
        building_objects=2,
        forest_objects=1,
        road_objects_truncated=False,
        building_objects_truncated=False,
        forest_objects_truncated=False,
        tree_row_objects=1,
    )

    ordered = _assemble_world_objects((road,), nonroads, (semantic,), renumber=False)
    assert ordered == (road, forests[0], rural[0], buildings[0], buildings[1], semantic)
    # The fast path reuses the original frozen objects instead of cloning them.
    assert ordered[1] is forests[0]
    assert ordered[3] is buildings[0]
    assert [obj.object_id for obj in ordered] == [1, 4, 5, 2, 3, 6]

    wrp = tmp_path / "ordered.wrp"
    write_rvw4(
        wrp,
        16,
        16,
        (0.0,) * 256,
        (0,) * 256,
        (r"world\data\g.paa",),
        ordered,
        height_scale=0.05,
        renumber_object_ids=True,
    )
    summary = inspect_rvw4(wrp, height_scale=0.05)
    assert summary.object_ids == (1, 2, 3, 4, 5, 6)
    assert summary.object_models == tuple(obj.model_path for obj in ordered)


def test_vietnam_generated_forest_carriers_use_sebnam_obj_vegetation() -> None:
    interior = generated_cluster_variant(
        "testworld",
        cluster_model_path("testworld", "pine", 0.0),
        proxy_profile="vietnam",
    )
    border = generated_cluster_variant(
        "testworld",
        cluster_model_path("testworld", "border_thicket", 0.0),
        proxy_profile="vietnam",
    )
    ditch = generated_cluster_variant(
        "testworld",
        cluster_model_path("testworld", "ditch_reeds", 0.0),
        proxy_profile="vietnam",
    )

    assert interior is not None
    assert border is not None
    assert ditch is not None
    assert {item[0] for item in interior[0].proxy_layout} <= set(VIETNAM_PROXY_MODELS)
    assert {item[0] for item in border[0].proxy_layout} <= set(VIETNAM_BORDER_PROXY_MODELS)
    assert {item[0] for item in ditch[0].proxy_layout} <= set(VIETNAM_DITCH_PROXY_MODELS)
    assert all(
        item[0].casefold().startswith("sebnam_obj\\")
        for variant, _grade in (interior, border, ditch)
        for item in variant.proxy_layout
    )


def test_vietnam_procedural_forest_library_accepts_profile() -> None:
    library = ProceduralForestClusterLibrary(
        "testworld",
        proxy_profile="vietnam",
    )
    model = cluster_model_path("testworld", "border_thicket", 0.0)
    library.register_model_usage(model, 2)

    required = set(library.required_proxy_models())
    assert required
    assert required <= set(VIETNAM_BORDER_PROXY_MODELS)
    assert all(path.casefold().startswith("sebnam_obj\\") for path in required)


def test_vietnam_runtime_dependency_trust_includes_reeds() -> None:
    spec = SimpleNamespace(
        paved_road_model=r"o\road\sil25.p3d",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
        stock_road_piece_fitting=False,
        forest_tree_model=r"sebnam_obj\sebles_su_ctver_pruhozi.p3d",
        forest_profile="vietnam",
        forest_everon_steep_model=r"sebnam_obj\sebles_su_ctver_pruhozi.p3d",
        forest_single_tree_model=r"sebnam_obj\sebstr borovice horska.p3d",
        forest_roadside_tree_model=r"sebnam_obj\sebstr borovice horska.p3d",
        forest_roadside_tree_models=VIETNAM_PROXY_MODELS,
        forest_roadside_bush_models=VIETNAM_BORDER_PROXY_MODELS,
        steep_hill_bush_models=VIETNAM_BORDER_PROXY_MODELS,
        forest_hillside_fallback=False,
        wetland_reed_models=VIETNAM_DITCH_PROXY_MODELS,
        semantic_landmarks=False,
        barriers_enabled=False,
        street_furniture_enabled=False,
    )

    trusted = set(_trusted_legacy_asset_paths(spec, 9))
    assert {
        path.casefold() for path in VIETNAM_DITCH_PROXY_MODELS
    } <= trusted


def test_cwa_flattens_generated_vegetation_carrier_into_direct_wrp_objects() -> None:
    parent = WorldObject(
        40,
        cluster_model_path("testworld", "border_thicket", 0.15),
        100.0,
        20.0,
        200.0,
        90.0,
    )
    nonroads = ObjectGenerationResult(
        objects=(parent,),
        road_objects=0,
        building_objects=0,
        forest_objects=0,
        road_objects_truncated=False,
        building_objects_truncated=False,
        forest_objects_truncated=False,
        forest_border_objects=1,
        forest_cluster_objects=0,
        model_usage=((parent.model_path, 1),),
    )
    spec = SimpleNamespace(
        profile="cwa",
        name="testworld",
        forest_profile="everon",
        forest_tree_model=r"data3d\les ctverec pruchozi_T1.p3d",
    )

    expanded = _expand_cwa_generated_vegetation(nonroads, spec)
    parsed = generated_cluster_variant(
        "testworld",
        parent.model_path,
        proxy_profile="everon",
    )
    assert parsed is not None
    variant, grade = parsed

    assert len(expanded.objects) == len(variant.proxy_layout)
    assert expanded.forest_border_objects == len(variant.proxy_layout)
    assert expanded.forest_cluster_objects == 0
    assert [obj.object_id for obj in expanded.objects] == list(
        range(parent.object_id, parent.object_id + len(variant.proxy_layout))
    )
    assert not any(
        is_generated_cluster_model("testworld", model)
        for model, _count in expanded.model_usage
    )

    first = expanded.objects[0]
    model, local_x, local_z, _marker_scale, proxy_heading = variant.proxy_layout[0]
    assert first.model_path == model
    assert abs(first.x - (parent.x + local_z)) < 1.0e-6
    assert abs(first.z - (parent.z - local_x)) < 1.0e-6
    expected_local_y = grade * (
        local_x if variant.slope_axis == "width" else local_z
    )
    assert abs(first.y - (parent.y + expected_local_y)) < 1.0e-6
    assert abs(first.heading_degrees - ((parent.heading_degrees + proxy_heading) % 360.0)) < 1.0e-6


def test_cwa_flattening_keeps_category_counts_assembly_consistent() -> None:
    carriers = (
        WorldObject(1, cluster_model_path("testworld", "pine", 0.15), 100.0, 10.0, 100.0),
        WorldObject(2, cluster_model_path("testworld", "undergrowth_patch", 0.15), 130.0, 10.0, 100.0),
        WorldObject(3, cluster_model_path("testworld", "border_thicket", 0.15), 160.0, 10.0, 100.0),
        WorldObject(4, cluster_model_path("testworld", "ditch_grass", 0.15), 190.0, 10.0, 100.0),
        WorldObject(5, cluster_model_path("testworld", "orchard_row", 0.15), 220.0, 10.0, 100.0),
    )
    nonroads = ObjectGenerationResult(
        objects=carriers,
        road_objects=0,
        building_objects=0,
        forest_objects=1,
        road_objects_truncated=False,
        building_objects_truncated=False,
        forest_objects_truncated=False,
        forest_undergrowth_objects=1,
        forest_border_objects=1,
        ditch_grass_objects=1,
        orchard_objects=1,
        forest_cluster_objects=1,
        model_usage=tuple((obj.model_path, 1) for obj in carriers),
    )
    spec = SimpleNamespace(
        profile="cwa",
        name="testworld",
        forest_profile="everon",
        forest_tree_model=r"data3d\les ctverec pruchozi_T1.p3d",
    )

    expanded = _expand_cwa_generated_vegetation(nonroads, spec)
    assembled = _assemble_world_objects((), expanded, (), renumber=False)

    assert len(assembled) == len(expanded.objects)
    assert expanded.forest_cluster_objects == 0
    assert not any(
        is_generated_cluster_model("testworld", obj.model_path)
        for obj in expanded.objects
    )

    expected_counts = {}
    for variant_name, field in (
        ("pine", "forest_objects"),
        ("undergrowth_patch", "forest_undergrowth_objects"),
        ("border_thicket", "forest_border_objects"),
        ("ditch_grass", "ditch_grass_objects"),
        ("orchard_row", "orchard_objects"),
    ):
        parsed = generated_cluster_variant(
            "testworld",
            cluster_model_path("testworld", variant_name, 0.15),
            proxy_profile="everon",
        )
        assert parsed is not None
        expected_counts[field] = len(parsed[0].proxy_layout)

    for field, expected in expected_counts.items():
        assert getattr(expanded, field) == expected


def test_cwr_ce_keeps_generated_vegetation_carrier_compact() -> None:
    parent = WorldObject(
        10,
        cluster_model_path("testworld", "undergrowth_patch", 0.30),
        50.0,
        12.0,
        75.0,
        17.0,
    )
    nonroads = ObjectGenerationResult(
        objects=(parent,),
        road_objects=0,
        building_objects=0,
        forest_objects=0,
        road_objects_truncated=False,
        building_objects_truncated=False,
        forest_objects_truncated=False,
        forest_undergrowth_objects=1,
        model_usage=((parent.model_path, 1),),
    )
    spec = SimpleNamespace(
        profile="cwr-ce",
        name="testworld",
        forest_profile="everon",
        forest_tree_model=r"data3d\les ctverec pruchozi_T1.p3d",
    )

    assert _expand_cwa_generated_vegetation(nonroads, spec) is nonroads


def test_streaming_hash_consumes_large_style_iterables_once_and_deterministically() -> None:
    class OneShot:
        def __init__(self, values):
            self.values = values
            self.used = False

        def __iter__(self):
            assert not self.used, "streaming hash attempted a preliminary/counting pass"
            self.used = True
            yield from self.values

    first = OneShot(range(10_000))
    second = OneShot(range(10_000))
    assert streaming_hash("one-shot-v1", first) == streaming_hash("one-shot-v1", second)
    assert first.used and second.used


def test_compact_support_rectangle_preserves_polygon_coordinates() -> None:
    original = _oriented_rectangle(123.5, 456.25, 11.0, 17.5, 37.0)
    compact = _compact_support_polygon(original, 123.5, 456.25, 37.0)
    assert isinstance(compact, CompactOrientedRectangle)
    materialized = tuple(compact)
    for actual, expected in zip(materialized, original):
        assert abs(actual[0] - expected[0]) < 1.0e-12
        assert abs(actual[1] - expected[1]) < 1.0e-12
    assert abs(compact[-1][0] - original[-1][0]) < 1.0e-12
    assert len(compact[1:3]) == 2


def test_bbox_projection_hot_path_is_affine_and_round_trips() -> None:
    projection = BboxProjection.create((55.0, 12.0, 55.25, 12.5), 20_000.0)
    point = (55.1, 12.2)
    x, z = projection.to_world(point)
    assert abs(x - 8_000.0) < 1.0e-9
    assert abs(z - 8_000.0) < 1.0e-9
    latitude, longitude = projection.to_latlon((x, z))
    assert abs(latitude - point[0]) < 1.0e-12
    assert abs(longitude - point[1]) < 1.0e-12



def test_chunked_float_terrain_hash_preserves_legacy_digest_across_boundaries() -> None:
    """Byte-for-byte compatibility is required for existing road/terrain caches."""
    values = [0.0, -0.0, float("inf"), float("-inf"), 0.125, -1.125]
    values.extend((index % 257) * 0.05 for index in range(20_003))
    baseline = sha256()
    legacy_pack = struct.Struct("<d").pack
    for value in values:
        baseline.update(legacy_pack(float(value)))

    assert float_sequence_sha256(values) == baseline.hexdigest()
    assert float_sequence_sha256(iter(values)) == baseline.hexdigest()


def test_chunked_int_terrain_hash_preserves_legacy_digest_across_boundaries() -> None:
    values = [-(2**31), 2**31 - 1, 0, -1, 1]
    values.extend((index % 250) - 125 for index in range(20_003))
    baseline = sha256()
    legacy_pack = struct.Struct("<i").pack
    for value in values:
        baseline.update(legacy_pack(int(value)))

    assert int_sequence_sha256(values) == baseline.hexdigest()
    assert int_sequence_sha256((value for value in values)) == baseline.hexdigest()


def test_chunked_terrain_hash_handles_empty_inputs_and_single_use_iterators() -> None:
    expected = sha256(b"").hexdigest()
    assert float_sequence_sha256(()) == expected
    assert int_sequence_sha256(iter(())) == expected

    class Once:
        def __init__(self):
            self.reads = 0

        def __iter__(self):
            self.reads += 1
            if self.reads != 1:
                raise AssertionError("terrain fingerprint generator was reused")
            yield 3
            yield -2

    assert float_sequence_sha256(Once()) == sha256(
        struct.pack("<dd", 3.0, -2.0)
    ).hexdigest()
    assert int_sequence_sha256(Once()) == sha256(
        struct.pack("<ii", 3, -2)
    ).hexdigest()


def test_chunked_terrain_hash_keeps_legacy_integer_overflow_semantics() -> None:
    import pytest

    with pytest.raises(struct.error):
        int_sequence_sha256([0] * 8_192 + [2**31])



def test_large_world_report_compaction_preserves_existing_json_data() -> None:
    road = RoadFitReport(
        objects=(WorldObject(1, "world\\road.p3d", 1.0, 2.0, 3.0, 45.0),),
        chain_count=1,
        connection_count=3,
        failed_connections=0,
        maximum_connection_gap=0.0,
        maximum_chain_gap=0.0,
        truncated=False,
    )
    legacy_road = asdict(road)
    legacy_road["objects"] = len(road.objects)
    assert _small_report_dict(road, "objects") | {
        "objects": len(road.objects),
    } == legacy_road

    grading = TerrainGradeReport(
        elevations=(3.25, 4.5, 5.75),
        changed_cells=2,
        road_seed_cells=1,
        building_pad_cells=0,
        maximum_cut=0.5,
        maximum_fill=0.25,
        maximum_road_slope_before_percent=6.0,
        maximum_road_slope_after_percent=2.0,
        building_roughness_before=0.9,
        building_roughness_after=0.5,
    )
    transitions = TransitionReport(indices=(1, 2, 3), shoreline_cells=1, softened_landuse_cells=2)
    legacy_grading = asdict(grading)
    legacy_grading.pop("elevations")
    legacy_grading["transitions"] = asdict(transitions)
    legacy_grading["transitions"].pop("indices")
    compact = _small_report_dict(grading, "elevations")
    compact["transitions"] = _small_report_dict(transitions, "indices")
    assert compact == legacy_grading


def test_compact_reports_never_deepcopy_omitted_heavy_arrays() -> None:
    @dataclass(frozen=True)
    class Nested:
        count: int

    @dataclass(frozen=True)
    class Report:
        heavy: tuple[object, ...]
        nested: Nested

    class RefuseCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("Unneeded report array was deep-copied")

    result = _small_report_dict(Report((RefuseCopy(),), Nested(7)), "heavy")
    assert result == {"nested": {"count": 7}}



def test_forest_mask_fusion_matches_four_original_probes_at_edges() -> None:
    """The same trees pass mask checks as before, including clipped cell edges."""
    rng = random.Random(1942)
    cells, world_size = 64, 3200.0
    length = cells * cells
    raster = SimpleNamespace(
        forest=tuple(rng.random() < 0.82 for _ in range(length)),
        water=tuple(rng.random() < 0.14 for _ in range(length)),
        roads=tuple(rng.random() < 0.20 for _ in range(length)),
        buildings=tuple(rng.random() < 0.17 for _ in range(length)),
    )
    points = [
        (-0.001, 100.0), (0.0, 0.0),
        (0.0, world_size - 0.001),
        (world_size - 0.001, world_size - 0.001),
        (world_size, 100.0), (100.0, world_size),
        (world_size + 1.0, world_size),
    ]
    points.extend(
        (rng.uniform(-80.0, world_size + 80.0),
         rng.uniform(-80.0, world_size + 80.0))
        for _ in range(3000)
    )
    for x, z in points:
        previous = (
            0 <= x < world_size
            and 0 <= z < world_size
            and _mask_at(raster.forest, cells, world_size, x, z)
            and not _mask_at(raster.water, cells, world_size, x, z)
            and not _mask_at(raster.roads, cells, world_size, x, z)
            and not _mask_at(raster.buildings, cells, world_size, x, z)
        )
        assert _forest_candidate_mask_ok(raster, cells, world_size, x, z) == previous
