from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen.milestone9 import Milestone9Spec, _Milestone9PlayabilitySpec
from cwr_worldgen.model import TerrainShearedWorldObject, WorldObject
from cwr_worldgen.osm import (
    BboxProjection,
    ObjectGenerationResult,
    OsmDataset,
    OsmRaster,
    _audit_vegetation_grounding,
    _rooted_tree_fit,
    apply_water_elevations,
    conservative_water_interior_mask,
    generate_world_objects,
    refine_iterative_grounding_terrain,
    renderable_water_mask,
)


def _raster(cells: int, *, water=(), interior=()) -> OsmRaster:
    size = cells * cells
    water_values = tuple(water) if water else (False,) * size
    return OsmRaster(
        cells=cells,
        water=water_values,
        forest=(False,) * size,
        farmland=(False,) * size,
        urban=(False,) * size,
        roads=(False,) * size,
        buildings=(False,) * size,
        high_resolution=cells * 4,
        coastline_seed_count=0,
        water_interior=tuple(interior),
    )


def test_conservative_water_does_not_excavate_high_mixed_shoreline_cell() -> None:
    cells = 3
    water = (True,) * 9
    interior = (False, False, False, False, True, False, False, False, False)
    raster = _raster(cells, water=water, interior=interior)
    elevations = (25.0, 25.0, 25.0, 25.0, 0.0, 25.0, 25.0, 25.0, 25.0)

    active = renderable_water_mask(elevations, raster, sea_level=0.0, water_depth=5.0)
    assert active[4]
    assert not active[0]
    assert conservative_water_interior_mask(raster)[4]

    result = apply_water_elevations(
        elevations,
        raster,
        sea_level=0.0,
        water_depth=5.0,
        beach_height=4.0,
        blend_cells=2,
        cell_size=25.0,
        maximum_shore_slope_percent=8.0,
    )
    assert result[4] == -5.0
    # It can become part of the adaptive bank, but it must never be treated as
    # the -5 m lake bed merely because the coarse raster contains some water.
    assert result[0] > -1.0


def test_iterative_grounding_ignores_vegetation_supports() -> None:
    cells = 4
    raster = _raster(cells)
    elevations = (3.0,) * 16
    spec = _Milestone9PlayabilitySpec(
        name="grounding_test",
        heightmap_path=Path("unused.png"),
        bbox=(0.0, 0.0, 1.0, 1.0),
        cells=cells,
        cell_size=25.0,
        strict_assets=False,
    )
    provisional = ObjectGenerationResult(
        objects=(WorldObject(1, spec.forest_single_tree_model, 25.0, 9.0, 25.0),),
        road_objects=0,
        building_objects=0,
        forest_objects=1,
        road_objects_truncated=False,
        building_objects_truncated=False,
        forest_objects_truncated=False,
    )
    refined, report = refine_iterative_grounding_terrain(
        elevations, provisional, (), raster, spec
    )
    assert refined == elevations
    assert report.tree_supports == 0
    assert report.adjusted_cells == 0


def test_milestone9_forest_defaults_prefer_ground_contact_and_small_models() -> None:
    spec = Milestone9Spec(source_dir=Path("unused"))
    assert spec.forest_ground_clearance == 0.02
    assert spec.forest_maximum_block_relief == 3.0
    assert spec.forest_everon_steep_maximum_relief == 8.0
    assert spec.forest_polygon_sink_fraction == 0.0
    assert spec.forest_single_tree_root_sink == 0.05
    assert spec.forest_single_tree_maximum_burial == 1.50
    assert spec.forest_single_tree_maximum_float == 0.15
    assert spec.forest_gap_infill_enabled
    assert spec.forest_gap_infill_spacing == 25.0
    assert spec.forest_cluster_tree_maximum_float == 0.20
    assert spec.forest_cluster_bush_maximum_float == 0.60


def test_final_vegetation_audit_detects_floating_individual_tree() -> None:
    cells = 4
    elevations = (0.0,) * 16
    spec = SimpleNamespace(
        cells=cells,
        cell_size=25.0,
        name="audit_world",
        forest_single_tree_maximum_float=0.15,
        forest_cluster_tree_maximum_float=0.20,
        forest_cluster_bush_maximum_float=0.60,
        forest_single_tree_model=r"data3d\str smrk_medium.p3d",
        forest_hillside_tree_model=r"data3d\str smrk_medium.p3d",
        forest_roadside_tree_model=r"data3d\str smrk_medium.p3d",
        forest_roadside_tree_models=(r"data3d\str smrk_medium.p3d",),
    )
    result = _audit_vegetation_grounding(
        (WorldObject(1, r"data3d\str smrk_medium.p3d", 25.0, 0.5, 25.0),),
        elevations,
        spec,
    )
    tree_objects, _cluster_trees, _cluster_bushes, violations, maximum_tree_float, _ = result
    assert tree_objects == 1
    assert violations == 1
    assert maximum_tree_float >= 0.49


def test_rooted_tree_fit_buries_root_instead_of_rejecting_terrain_diagonal_gap() -> None:
    fit = _rooted_tree_fit((10.0, 10.8), root_sink=0.05, maximum_burial=1.50)
    assert fit is not None
    anchor, burial = fit
    assert anchor == 9.95
    assert abs(burial - 0.85) < 1.0e-9
    # The same surface ambiguity is rejected when it would require an absurdly
    # deep trunk burial.
    assert _rooted_tree_fit((10.0, 12.0), root_sink=0.05, maximum_burial=1.0) is None


def test_nogova_leaf_forests_use_stock_paired_sheared_blocks() -> None:
    cells = 8
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 200.0)
    dataset = OsmDataset(
        source_generator="nogova-leaf-grounding", element_count=0, coastlines=(),
        water=(), forests=(), farmland=(), urban=(), roads=(),
    )
    raster = OsmRaster(
        cells=cells,
        water=(False,) * (cells * cells),
        forest=(True,) * (cells * cells),
        farmland=(False,) * (cells * cells),
        urban=(False,) * (cells * cells),
        roads=(False,) * (cells * cells),
        buildings=(False,) * (cells * cells),
        high_resolution=cells,
        coastline_seed_count=0,
    )
    leaf_block = r"o\tree\les_nw_ctver_pruhozi_T1.p3d"
    leaf_triangle = r"o\tree\les_nw_trojuhelnik.p3d"
    spec = _Milestone9PlayabilitySpec(
        name="nogova_leaf_grounding",
        heightmap_path=Path("unused.png"),
        bbox=(0.0, 0.0, 1.0, 1.0),
        cells=cells,
        cell_size=25.0,
        max_road_objects=0,
        max_buildings=0,
        max_forest_objects=200,
        forest_tree_spacing=50.0,
        forest_tree_model=leaf_block,
        forest_everon_steep_model=leaf_triangle,
        forest_individual_objects_only=False,
        forest_single_tree_enabled=True,
        forest_gap_infill_enabled=False,
        forest_undergrowth_enabled=False,
        forest_border_enabled=False,
        steep_hill_bushes_enabled=False,
        ditch_grass_enabled=False,
        rocky_forest_fallback_enabled=False,
        strict_assets=False,
    )

    result = generate_world_objects(
        dataset, projection, raster, (10.0,) * (cells * cells), spec,
        include_roads=False,
    )

    blocks = [
        obj for obj in result.objects
        if obj.model_path.casefold() == leaf_block.casefold()
    ]
    triangles = [
        obj for obj in result.objects
        if obj.model_path.casefold() == leaf_triangle.casefold()
    ]
    assert blocks
    assert len(blocks) == len(triangles)
    assert all(isinstance(obj, TerrainShearedWorldObject) for obj in (*blocks, *triangles))
    assert all(abs(obj.y - 19.0) < 1.0e-9 for obj in (*blocks, *triangles))
    assert all(abs(obj.terrain_shear_x) < 1.0e-12 for obj in (*blocks, *triangles))
    assert all(abs(obj.terrain_shear_z) < 1.0e-12 for obj in (*blocks, *triangles))
    assert {
        (int(obj.x // 50.0), int(obj.z // 50.0)) for obj in blocks
    } == {
        (int(obj.x // 50.0), int(obj.z // 50.0)) for obj in triangles
    }
    assert all(abs((obj.x % 50.0) - 20.353600907) < 1.0e-6 for obj in blocks)
    assert all(abs((obj.z % 50.0) - 17.894533926) < 1.0e-6 for obj in blocks)
    assert all(abs((obj.x % 50.0) - 29.344323452) < 1.0e-6 for obj in triangles)
    assert all(abs((obj.z % 50.0) - 32.222971961) < 1.0e-6 for obj in triangles)
    assert result.forest_cluster_objects == 0
    assert result.forest_single_tree_objects == 0
    assert result.vegetation_audit_violations == 0


def test_nogova_leaf_stock_transform_keeps_vertical_axis_upright() -> None:
    obj = TerrainShearedWorldObject(
        1,
        r"o\tree\les_nw_ctver_pruhozi_T1.p3d",
        100.0,
        25.0,
        200.0,
        0.0,
        0.0,
        terrain_shear_x=0.125,
        terrain_shear_z=-0.075,
    )

    matrix = obj.matrix_4x3()

    assert matrix[1] == 0.125
    assert matrix[3:6] == (0.0, 1.0, 0.0)
    assert matrix[7] == -0.075


def test_uncovered_mapped_forest_gets_rooted_gap_infill_trees() -> None:
    cells = 4
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
    dataset = OsmDataset(
        source_generator="gap-infill", element_count=0, coastlines=(), water=(),
        forests=(), farmland=(), urban=(), roads=(),
    )
    raster = OsmRaster(
        cells=cells,
        water=(False,) * 16,
        forest=tuple(
            (row, col) in {(0, 0), (0, 2), (2, 0), (2, 2)}
            for row in range(cells) for col in range(cells)
        ),
        farmland=(False,) * 16,
        urban=(False,) * 16,
        roads=(False,) * 16,
        buildings=(False,) * 16,
        high_resolution=8,
        coastline_seed_count=0,
    )
    spec = _Milestone9PlayabilitySpec(
        name="gap_infill_test",
        heightmap_path=Path("unused.png"),
        bbox=(0.0, 0.0, 1.0, 1.0),
        cells=cells,
        cell_size=25.0,
        max_road_objects=0,
        max_buildings=0,
        max_forest_objects=200,
        forest_tree_spacing=50.0,
        forest_block_maximum_float=0.0,
        forest_everon_steep_maximum_float=0.0,
        forest_cluster_fallback=False,
        forest_severe_hill_fallback=False,
        forest_hillside_fallback=False,
        forest_single_tree_enabled=False,
        forest_gap_infill_enabled=True,
        forest_gap_infill_spacing=25.0,
        forest_undergrowth_enabled=False,
        forest_border_enabled=False,
        steep_hill_bushes_enabled=False,
        ditch_grass_enabled=False,
        rocky_forest_fallback_enabled=False,
        strict_assets=False,
    )
    result = generate_world_objects(
        dataset, projection, raster, (0.0,) * 16, spec, include_roads=False
    )
    assert result.forest_gap_infill_tree_objects >= 4
    assert result.forest_single_tree_objects >= result.forest_gap_infill_tree_objects
    assert result.vegetation_audit_violations == 0
    assert result.vegetation_audit_maximum_tree_float == 0.0
