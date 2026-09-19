from __future__ import annotations

import json
from pathlib import Path

from cwr_worldgen import generator
from cwr_worldgen.stock_building_policy import (
    STOCK_BUILDING_PRESET,
    StockBuildingLibrary,
    _load_catalogue,
)


def _library() -> StockBuildingLibrary:
    return StockBuildingLibrary(world_name="wg_stock_test", house_style_preset=STOCK_BUILDING_PRESET)


def test_reviewed_catalogue_contains_only_curated_stock_pool() -> None:
    models = _load_catalogue()
    paths = {model.model_path.casefold() for model in models}
    assert len(models) == 130
    assert r"data3d\kostel.p3d" in paths
    assert r"data3d\hangar.p3d" in paths
    assert r"data3d\helfenburk.p3d" in paths
    assert r"o\hous\kostelin.p3d" in paths
    assert r"o\hous\skola.p3d" in paths
    assert r"o\hous\hangar_2.p3d" in paths
    assert r"o\hous\vysilac_fm.p3d" in paths
    assert r"o\misc\leseni2x.p3d" not in paths
    assert r"o\misc\leseni4x.p3d" not in paths
    assert r"data3d\zvonice.p3d" not in paths
    assert r"data3d\hospital.p3d" not in paths
    assert all(model.categories for model in models)
    assert all(model.placement in {"Urban", "Rural", "Both"} for model in models)
    assert all(min(model.width_m, model.length_m, model.height_m) > 0.0 for model in models)


def test_stock_polygon_selection_uses_original_game_model_and_measured_dimensions() -> None:
    library = _library()
    placement = library.plan_polygon(
        {"building": "house"},
        ((0.0, 0.0), (10.0, 0.0), (10.0, 7.0), (0.0, 7.0)),
    )
    selected = placement.selected
    assert isinstance(placement.model_path, str)
    assert placement.model_path.casefold().endswith(".p3d")
    assert "\\g\\" not in placement.model_path.casefold()
    assert selected.stock_model_path == placement.model_path
    assert selected.width_m > 0.0
    assert selected.length_m > 0.0
    assert selected.foundation_depth_m == 0.0
    assert not selected.interiors


def test_stock_special_buildings_stay_in_stock_family() -> None:
    library = _library()
    church = library.plan_polygon(
        {"building": "church", "amenity": "place_of_worship", "religion": "christian"},
        ((0.0, 0.0), (12.0, 0.0), (12.0, 17.0), (0.0, 17.0)),
    )
    school = library.plan_polygon(
        {"building": "school", "amenity": "school"},
        ((0.0, 0.0), (24.0, 0.0), (24.0, 15.0), (0.0, 15.0)),
    )
    catalogue = {model.model_path.casefold(): model for model in library.models}
    assert "church" in catalogue[church.model_path.casefold()].families
    assert "school" in catalogue[school.model_path.casefold()].families



def test_stock_generic_rural_footprints_reuse_procedural_size_classification() -> None:
    library = _library()
    small = library.plan_polygon(
        {"building": "yes"},
        ((0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)),
    )
    large = library.plan_polygon(
        {"building": "yes"},
        ((0.0, 0.0), (40.0, 0.0), (40.0, 20.0), (0.0, 20.0)),
    )

    catalogue = {model.model_path.casefold(): model for model in library.models}
    assert small.selected.family == "outbuilding"
    assert large.selected.family == "agricultural"
    assert large.selected.building_class == "barn"
    assert "outbuilding" in catalogue[small.model_path.casefold()].families
    assert "agricultural" in catalogue[large.model_path.casefold()].families


def test_stock_generic_city_footprint_reuses_procedural_settlement_classification() -> None:
    library = _library()
    library._settlements = ((0.0, 0.0, "city"),)
    placement = library.plan_polygon(
        {"building": "yes"},
        ((-20.0, -10.0), (20.0, -10.0), (20.0, 10.0), (-20.0, 10.0)),
    )

    catalogue = {model.model_path.casefold(): model for model in library.models}
    selected_model = catalogue[placement.model_path.casefold()]
    assert placement.selected.family == "urban"
    assert selected_model.placement in {"Urban", "Both"}
    assert "urban" in selected_model.families


def test_stock_outbuilding_subtype_reuses_procedural_dimension_inference() -> None:
    library = _library()
    shed = library.plan_point(
        {"building": "outbuilding"},
        3.0,
        0.0,
        x=5000.0,
        z=5000.0,
    )
    garage = library.plan_point(
        {"building": "outbuilding"},
        6.0,
        0.0,
        x=5100.0,
        z=5000.0,
    )

    assert shed.selected.family == "outbuilding"
    assert shed.selected.building_class == "shed"
    assert shed.selected.outbuilding_kind == "shed"
    assert garage.selected.family == "outbuilding"
    assert garage.selected.building_class == "garage"
    assert garage.selected.outbuilding_kind == "garage"


def test_stock_settlement_boundary_matches_procedural_one_kilometre_rule() -> None:
    library = _library()
    library._settlements = ((0.0, 0.0, "city"),)

    near = library.plan_point(
        {"building": "yes"},
        10.0,
        0.0,
        x=999.0,
        z=0.0,
    )
    far = library.plan_point(
        {"building": "yes"},
        10.0,
        0.0,
        x=1001.0,
        z=0.0,
    )

    assert near.selected.family == "townhouse"
    assert far.selected.family == "residential"


def test_stock_mode_factory_never_instantiates_procedural_library() -> None:
    library = generator.ProceduralBuildingLibrary(
        world_name="wg_stock_factory",
        house_style_preset=STOCK_BUILDING_PRESET,
    )
    assert isinstance(library, StockBuildingLibrary)


def test_stock_asset_report_writes_catalogue_but_no_generated_models(tmp_path: Path) -> None:
    library = _library()
    placement = library.plan_point({"building": "house"}, 10.0, 0.0, x=10.0, z=20.0)
    library.register_placement(placement, foundation_depth_m=2.0)
    catalogue = tmp_path / "building-asset-catalogue.json"
    result = library.write_assets(tmp_path / "source", catalogue)
    document = json.loads(catalogue.read_text(encoding="utf-8"))
    assert document["mode"] == STOCK_BUILDING_PRESET
    assert document["generated_models"] == 0
    assert document["placements"] == 1
    assert result.generated_variants == 0
    assert result.model_assets == ()


def test_rural_residential_uses_house_pool_while_city_prefers_dense_stock() -> None:
    rural = _library()
    rural_placement = rural.plan_point(
        {"building": "house"},
        10.0,
        0.0,
        x=5000.0,
        z=5000.0,
    )
    rural_catalogue = {model.model_path.casefold(): model for model in rural.models}
    rural_model = rural_catalogue[rural_placement.model_path.casefold()]
    assert rural_model.placement in {"Rural", "Both"}
    assert "residential" in rural_model.families
    assert "urban" not in rural_model.families
    assert "townhouse" not in rural_model.families

    city = _library()
    city._settlements = ((0.0, 0.0, "city"),)
    city_placement = city.plan_point(
        {"building": "house"},
        10.0,
        0.0,
        x=0.0,
        z=0.0,
    )
    city_catalogue = {model.model_path.casefold(): model for model in city.models}
    city_model = city_catalogue[city_placement.model_path.casefold()]
    assert city_model.placement in {"Urban", "Both"}
    assert "townhouse" in city_model.families or "urban" in city_model.families
