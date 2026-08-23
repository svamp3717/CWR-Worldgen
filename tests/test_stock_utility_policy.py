from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cwr_worldgen.generator as generator
from cwr_worldgen.asset_mapping import default_osm_asset_mapping
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import (
    BboxProjection,
    OsmDataset,
    OsmLineFeature,
    OsmPointFeature,
    OsmRaster,
    generate_world_objects,
)
from cwr_worldgen.stock_utility_policy import (
    STOCK_POWER_POLE_MODELS,
    STOCK_POWER_TOWER_MODELS,
    _rewrite_stock_utilities,
)


def _empty_raster(cells: int) -> OsmRaster:
    empty = (False,) * (cells * cells)
    return OsmRaster(cells, empty, empty, empty, empty, empty, empty, cells, 0)


def _spec(bbox, cells):
    return _Milestone9PlayabilitySpec(
        name="stock_utility_policy",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=cells,
        cell_size=25.0,
        max_road_objects=0,
        max_buildings=0,
        max_forest_objects=0,
        street_furniture_enabled=False,
        rural_vegetation_enabled=False,
        meadow_grass_enabled=False,
        wetland_reeds_enabled=False,
        barriers_enabled=False,
        bridges_enabled=False,
        forest_undergrowth_enabled=False,
        forest_border_enabled=False,
        rocky_forest_fallback_enabled=False,
        steep_hill_bushes_enabled=False,
        strict_assets=False,
    )


def test_build_loader_is_wrapped_by_stock_utility_policy() -> None:
    assert getattr(generator._load_nonroad_objects, "_cwr_stock_utility_policy", False)


def test_default_asset_mapping_uses_stock_power_models() -> None:
    mapping = default_osm_asset_mapping(SimpleNamespace(name="mapping_test"), 9)
    rule = next(rule for rule in mapping.rules if rule.rule_id == "osm-power-utilities")
    assert set(rule.models) == set((*STOCK_POWER_POLE_MODELS, *STOCK_POWER_TOWER_MODELS))
    assert not any("util_power_" in model.casefold() for model in rule.models)


def test_rewrite_replaces_procedural_pole_and_moves_full_footprint_off_road() -> None:
    bbox = (0.0, 0.0, 1.0, 1.0)
    cells = 8
    projection = BboxProjection.create(bbox, cells * 25.0)
    pole = OsmPointFeature(
        "node/pole", {"utility": "power_pole"}, projection.to_latlon((40.0, 75.0))
    )
    base_dataset = OsmDataset(
        source_generator="policy-base", element_count=1,
        coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
        utility_points=(pole,),
    )
    spec = _spec(bbox, cells)
    elevations = (5.0,) * (cells * cells)
    result = generate_world_objects(
        base_dataset, projection, _empty_raster(cells), elevations, spec,
        include_roads=False, building_placement_plans=(),
    )
    assert result.utility_objects == 1
    assert result.objects[0].model_path.casefold().endswith(r"\i\util_power_pole.p3d")

    road = OsmLineFeature(
        "way/road", {"highway": "residential"},
        (projection.to_latlon((100.0, 0.0)), projection.to_latlon((100.0, 199.0))),
    )
    dataset = replace(base_dataset, roads=(road,))
    on_road = replace(
        result,
        objects=(replace(result.objects[0], x=100.0, z=75.0),),
        model_usage=((result.objects[0].model_path, 1),),
    )
    rewritten = _rewrite_stock_utilities(
        on_road, dataset, projection, _empty_raster(cells), elevations, spec
    )
    obj = rewritten.objects[0]
    assert obj.model_path in STOCK_POWER_POLE_MODELS
    assert not obj.model_path.casefold().endswith(r"\i\util_power_pole.p3d")
    assert abs(obj.x - 100.0) >= 7.0 - 1e-6
    assert dict(rewritten.model_usage) == {obj.model_path: 1}


def test_settlement_high_voltage_mast_is_normalized_to_ordinary_stock_pole() -> None:
    bbox = (0.0, 0.0, 1.0, 1.0)
    cells = 8
    projection = BboxProjection.create(bbox, cells * 25.0)
    base_dataset = OsmDataset(
        source_generator="settlement-pole", element_count=0,
        coastlines=(), water=(), forests=(), farmland=(), urban=(), roads=(),
    )
    spec = _spec(bbox, cells)
    elevations = (5.0,) * (cells * cells)
    pole = OsmPointFeature(
        "node/pole", {"utility": "power_pole"}, projection.to_latlon((40.0, 75.0))
    )
    shell_dataset = replace(base_dataset, utility_points=(pole,), element_count=1)
    shell = generate_world_objects(
        shell_dataset, projection, _empty_raster(cells), elevations, spec,
        include_roads=False, building_placement_plans=(),
    )
    mast = replace(shell.objects[0], model_path=STOCK_POWER_TOWER_MODELS[0])
    result = replace(shell, objects=(mast,), model_usage=((mast.model_path, 1),))
    rewritten = _rewrite_stock_utilities(
        result, base_dataset, projection, _empty_raster(cells), elevations, spec
    )
    assert rewritten.objects[0].model_path in STOCK_POWER_POLE_MODELS
