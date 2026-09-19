from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen import generator
from cwr_worldgen.gui import default_gui_values
from cwr_worldgen.model import WorldObject
from cwr_worldgen.stock_building_policy import STOCK_BUILDING_PRESET, StockBuildingLibrary
from cwr_worldgen.stock_building_extensions import (
    STOCK_BUILDING_COMBINED_LABEL,
    STOCK_BUILDING_HAUS_COMBINED_LABEL,
    STOCK_BUILDING_HAUS_COMBINED_PRESET,
    STOCK_BUILDING_OPTIONS,
    STOCK_BUILDING_PRESETS,
    STOCK_BUILDING_RESISTANCE_PRESET,
    STOCK_BUILDING_VANILLA_PRESET,
    _PROCEDURAL_BRIDGES_CHECKBOX_TEXT,
    _lift_stock_objects,
    _stock_options_first,
    stock_disabled_gui_option_keys,
    stock_model_source,
)


MIXED_STOCK_LABEL = "Stock combined (non-Resistance + Resistance) buildings"


def _library(preset: str) -> StockBuildingLibrary:
    return StockBuildingLibrary(world_name="wg_stock_ext_test", house_style_preset=preset)


def test_split_catalogue_files_partition_combined_catalogue() -> None:
    data_dir = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "data"
    combined = json.loads((data_dir / "stock_building_models.json").read_text(encoding="utf-8"))
    non_resistance = json.loads(
        (data_dir / "stock_building_models_non_resistance.json").read_text(encoding="utf-8")
    )
    resistance = json.loads(
        (data_dir / "stock_building_models_resistance.json").read_text(encoding="utf-8")
    )
    haus_combined = json.loads(
        (data_dir / "haus.pbo + resistance and vanilla.json").read_text(encoding="utf-8")
    )

    combined_paths = {row["model_path"].casefold() for row in combined["models"]}
    non_resistance_paths = {row["model_path"].casefold() for row in non_resistance["models"]}
    resistance_paths = {row["model_path"].casefold() for row in resistance["models"]}
    haus_combined_paths = {row["model_path"].casefold() for row in haus_combined["models"]}

    assert combined["schema"] == 5
    assert non_resistance["schema"] == 5
    assert resistance["schema"] == 5
    assert haus_combined["schema"] == 5
    assert combined["display_name"] == STOCK_BUILDING_COMBINED_LABEL
    assert non_resistance["display_name"] == "Stock non-Resistance buildings only"
    assert resistance["display_name"] == "Stock Resistance buildings only"
    assert haus_combined["display_name"] == STOCK_BUILDING_HAUS_COMBINED_LABEL
    assert len(combined_paths) == 130
    assert len(non_resistance_paths) == 77
    assert len(resistance_paths) == 53
    assert len(haus_combined_paths) == 172
    assert combined_paths.issubset(haus_combined_paths)
    assert sum(path.startswith("haus\\") for path in haus_combined_paths) == 42
    assert r"data3d\hospital.p3d" not in combined_paths
    assert r"data3d\hospital.p3d" not in non_resistance_paths
    assert non_resistance_paths.isdisjoint(resistance_paths)
    assert combined_paths == non_resistance_paths | resistance_paths
    assert all(not path.startswith("o\\") for path in non_resistance_paths)
    assert all(path.startswith("o\\") for path in resistance_paths)
    assert all(
        all(
            field in row
            for field in (
                "categories",
                "placement",
                "width_m",
                "length_m",
                "height_m",
                "aspect_ratio",
                "origin_to_bottom_m",
            )
        )
        for row in combined["models"]
    )


def test_stock_presets_keep_mixed_and_split_vanilla_from_resistance() -> None:
    mixed = _library(STOCK_BUILDING_PRESET)
    vanilla = _library(STOCK_BUILDING_VANILLA_PRESET)
    resistance = _library(STOCK_BUILDING_RESISTANCE_PRESET)
    haus_combined = _library(STOCK_BUILDING_HAUS_COMBINED_PRESET)

    assert mixed.models
    assert vanilla.models
    assert resistance.models
    assert haus_combined.models
    assert len(mixed.models) == len(vanilla.models) + len(resistance.models)
    assert len(haus_combined.models) == 172
    assert {stock_model_source(model.model_path) for model in vanilla.models} == {"vanilla"}
    assert {stock_model_source(model.model_path) for model in resistance.models} == {"resistance"}
    assert {stock_model_source(model.model_path) for model in haus_combined.models} == {
        "vanilla",
        "resistance",
        "haus",
    }


def test_all_stock_presets_use_stock_library_factory() -> None:
    for preset in STOCK_BUILDING_PRESETS:
        library = generator.ProceduralBuildingLibrary(
            world_name="wg_stock_factory_ext",
            house_style_preset=preset,
        )
        assert isinstance(library, StockBuildingLibrary)
        assert library.house_style_preset == preset


def test_stock_options_are_directly_below_automatic() -> None:
    options, labels = _stock_options_first(
        (("se", "Sweden"), ("stock", MIXED_STOCK_LABEL), ("de", "Germany")),
        ("Automatic (area / country)", "Sweden", "Germany", MIXED_STOCK_LABEL),
        auto_label="Automatic (area / country)",
    )
    assert options[:4] == STOCK_BUILDING_OPTIONS
    assert labels[0] == "Automatic (area / country)"
    assert labels[1:5] == tuple(label for _identifier, label in STOCK_BUILDING_OPTIONS)
    assert MIXED_STOCK_LABEL in labels
    assert "Stock CWA/OFP buildings only" not in labels


def test_stock_presets_disable_procedural_building_gui_options() -> None:
    expected = {
        "procedural_building_interiors",
        "high_quality_building_textures",
        "match_nearby_building_textures",
    }
    for preset in STOCK_BUILDING_PRESETS:
        assert set(stock_disabled_gui_option_keys(preset)) == expected
    assert stock_disabled_gui_option_keys("auto") == ()


def test_procedural_bridges_remain_default_while_gui_switch_is_hidden() -> None:
    assert default_gui_values()["procedural_bridges"] is True
    assert _PROCEDURAL_BRIDGES_CHECKBOX_TEXT == "Procedural bridges (instead of Nogova)"


def test_measured_origin_lift_is_added_to_stock_building_object() -> None:
    library = _library(STOCK_BUILDING_PRESET)
    model = next(model for model in library.models if model.origin_to_bottom_m > 1.0)
    plan = SimpleNamespace(model_path=model.model_path, x=100.0, z=200.0)
    obj = WorldObject(1, model.model_path, 100.0, 12.5, 200.0, 35.0)

    lifted = _lift_stock_objects((obj,), library, (plan,))

    assert len(lifted) == 1
    assert lifted[0].y == obj.y + model.origin_to_bottom_m
    assert lifted[0].x == obj.x
    assert lifted[0].z == obj.z


def test_stock_asset_catalogue_records_source_and_origin_without_generated_p3ds(tmp_path: Path) -> None:
    library = _library(STOCK_BUILDING_RESISTANCE_PRESET)
    placement = library.plan_point({"building": "house"}, 10.0, 0.0, x=20.0, z=30.0)
    library.register_placement(placement, foundation_depth_m=2.0)
    catalogue = tmp_path / "building-asset-catalogue.json"

    result = library.write_assets(tmp_path / "source", catalogue)
    document = json.loads(catalogue.read_text(encoding="utf-8"))

    assert document["mode"] == STOCK_BUILDING_RESISTANCE_PRESET
    assert document["generated_models"] == 0
    assert document["generated_variants"] == 0
    assert document["models"][0]["source_set"] == "resistance"
    assert document["models"][0]["origin_lift_m"] > 0.0
    assert result.generated_variants == 0
    assert result.model_assets == ()
