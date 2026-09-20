from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen import generator
from cwr_worldgen.gui import default_gui_values
from cwr_worldgen.model import WorldObject
from cwr_worldgen.osm import BuildingPlacementPlan, ObjectGenerationResult
from cwr_worldgen.stock_building_policy import STOCK_BUILDING_PRESET, StockBuildingLibrary
from cwr_worldgen.stock_building_extensions import (
    STOCK_BUILDING_AGS_COMBINED_PRESET,
    STOCK_BUILDING_AGS_ONLY_LABEL,
    STOCK_BUILDING_AGS_ONLY_PRESET,
    STOCK_BUILDING_HAUS_COMBINED_PRESET,
    STOCK_BUILDING_HAUS_ONLY_LABEL,
    STOCK_BUILDING_HAUS_ONLY_PRESET,
    STOCK_BUILDING_MULTI_PREFIX,
    STOCK_BUILDING_OPTIONS,
    STOCK_BUILDING_PRESETS,
    STOCK_BUILDING_RESISTANCE_PRESET,
    STOCK_BUILDING_VANILLA_PRESET,
    _PROCEDURAL_BRIDGES_CHECKBOX_TEXT,
    _remove_stock_buildings_overlapping_final_roads,
    _lift_stock_objects,
    _stock_options_first,
    encode_stock_building_presets,
    stock_building_preset_ids,
    stock_disabled_gui_option_keys,
    stock_model_source,
)


MIXED_STOCK_LABEL = "Stock combined (non-Resistance + Resistance) buildings"


def _library(preset: str) -> StockBuildingLibrary:
    return StockBuildingLibrary(world_name="wg_stock_ext_test", house_style_preset=preset)


def test_multi_stock_preset_encoding_is_canonical_and_round_trips() -> None:
    encoded = encode_stock_building_presets(
        (STOCK_BUILDING_AGS_ONLY_PRESET, STOCK_BUILDING_VANILLA_PRESET)
    )
    assert encoded.startswith(STOCK_BUILDING_MULTI_PREFIX)
    assert stock_building_preset_ids(encoded) == (
        STOCK_BUILDING_VANILLA_PRESET,
        STOCK_BUILDING_AGS_ONLY_PRESET,
    )


def test_multi_stock_library_merges_and_deduplicates_selected_catalogues() -> None:
    combined = encode_stock_building_presets(
        (
            STOCK_BUILDING_VANILLA_PRESET,
            STOCK_BUILDING_RESISTANCE_PRESET,
            STOCK_BUILDING_AGS_ONLY_PRESET,
        )
    )
    library = _library(combined)

    paths = {model.model_path.casefold() for model in library.models}
    assert len(paths) == 145
    assert any(path.startswith("data3d\\") for path in paths)
    assert any(path.startswith("o\\") for path in paths)
    assert any(path.startswith(("ags_inds\\", "ags_port\\")) for path in paths)
    assert library.house_style_preset == encode_stock_building_presets(
        (
            STOCK_BUILDING_VANILLA_PRESET,
            STOCK_BUILDING_RESISTANCE_PRESET,
            STOCK_BUILDING_AGS_ONLY_PRESET,
        )
    )


def test_source_catalogues_replace_precombined_json_files() -> None:
    data_dir = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "data"
    non_resistance = json.loads(
        (data_dir / "stock_building_models_non_resistance.json").read_text(encoding="utf-8")
    )
    resistance = json.loads(
        (data_dir / "stock_building_models_resistance.json").read_text(encoding="utf-8")
    )
    haus_only = json.loads(
        (data_dir / "haus.pbo buildings only.json").read_text(encoding="utf-8")
    )
    ags_only = json.loads(
        (data_dir / "ags inds+port.json").read_text(encoding="utf-8")
    )

    non_resistance_paths = {row["model_path"].casefold() for row in non_resistance["models"]}
    resistance_paths = {row["model_path"].casefold() for row in resistance["models"]}
    haus_only_paths = {row["model_path"].casefold() for row in haus_only["models"]}
    ags_only_paths = {row["model_path"].casefold() for row in ags_only["models"]}

    assert non_resistance["schema"] == 5
    assert resistance["schema"] == 5
    assert haus_only["schema"] == 5
    assert ags_only["schema"] == 5
    assert ags_only["display_name"] == "ags_inds.pbo and ags_port.pbo buildings"
    assert STOCK_BUILDING_AGS_ONLY_LABEL == "ags_inds.pbo and ags_port.pbo buildings"
    assert len(non_resistance_paths) == 77
    assert len(resistance_paths) == 53
    assert len(haus_only_paths) == 42
    assert len(ags_only_paths) == 15
    assert non_resistance_paths.isdisjoint(resistance_paths)
    assert all(not path.startswith("o\\") for path in non_resistance_paths)
    assert all(path.startswith("o\\") for path in resistance_paths)
    assert all(path.startswith("haus\\") for path in haus_only_paths)
    assert all(path.startswith(("ags_inds\\", "ags_port\\")) for path in ags_only_paths)

    for removed in (
        "stock_building_models.json",
        "haus.pbo + resistance and vanilla.json",
        "ags inds+port and combined stock.json",
    ):
        assert not (data_dir / removed).exists()


def test_legacy_combined_ids_expand_to_source_catalogues() -> None:
    mixed = _library(STOCK_BUILDING_PRESET)
    vanilla = _library(STOCK_BUILDING_VANILLA_PRESET)
    resistance = _library(STOCK_BUILDING_RESISTANCE_PRESET)
    haus_legacy = _library(STOCK_BUILDING_HAUS_COMBINED_PRESET)
    haus_only = _library(STOCK_BUILDING_HAUS_ONLY_PRESET)
    ags_only = _library(STOCK_BUILDING_AGS_ONLY_PRESET)
    ags_legacy = _library(STOCK_BUILDING_AGS_COMBINED_PRESET)

    assert len(vanilla.models) == 77
    assert len(resistance.models) == 53
    assert len(haus_only.models) == 42
    assert len(ags_only.models) == 15
    assert len(mixed.models) == 130
    assert len(haus_legacy.models) == 172
    assert len(ags_legacy.models) == 145

    assert stock_building_preset_ids(STOCK_BUILDING_PRESET) == (
        STOCK_BUILDING_VANILLA_PRESET,
        STOCK_BUILDING_RESISTANCE_PRESET,
    )
    assert stock_building_preset_ids(STOCK_BUILDING_HAUS_COMBINED_PRESET) == (
        STOCK_BUILDING_VANILLA_PRESET,
        STOCK_BUILDING_RESISTANCE_PRESET,
        STOCK_BUILDING_HAUS_ONLY_PRESET,
    )
    assert stock_building_preset_ids(STOCK_BUILDING_AGS_COMBINED_PRESET) == (
        STOCK_BUILDING_VANILLA_PRESET,
        STOCK_BUILDING_RESISTANCE_PRESET,
        STOCK_BUILDING_AGS_ONLY_PRESET,
    )

    assert {stock_model_source(model.model_path) for model in vanilla.models} == {"vanilla"}
    assert {stock_model_source(model.model_path) for model in resistance.models} == {"resistance"}
    assert {stock_model_source(model.model_path) for model in haus_only.models} == {"haus"}
    assert {stock_model_source(model.model_path) for model in ags_only.models} == {"ags"}
    assert {stock_model_source(model.model_path) for model in haus_legacy.models} == {
        "vanilla", "resistance", "haus",
    }
    assert {stock_model_source(model.model_path) for model in ags_legacy.models} == {
        "vanilla", "resistance", "ags",
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
    assert MIXED_STOCK_LABEL not in labels
    assert "Stock CWA/OFP buildings" not in labels
    assert all(not label.casefold().endswith(" only") for label in labels)


def test_stock_presets_disable_procedural_building_gui_options() -> None:
    expected = {
        "procedural_building_interiors",
        "high_quality_building_textures",
        "match_nearby_building_textures",
    }
    for preset in STOCK_BUILDING_PRESETS:
        assert set(stock_disabled_gui_option_keys(preset)) == expected
    combined = encode_stock_building_presets(
        (STOCK_BUILDING_HAUS_ONLY_PRESET, STOCK_BUILDING_AGS_ONLY_PRESET)
    )
    assert set(stock_disabled_gui_option_keys(combined)) == expected
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
    assert document["selected_building_jsons"] == [
        "data/stock_building_models_resistance.json"
    ]
    assert document["generated_models"] == 0
    assert document["generated_variants"] == 0
    assert document["models"][0]["source_set"] == "resistance"
    assert document["models"][0]["origin_lift_m"] > 0.0
    assert result.generated_variants == 0
    assert result.model_assets == ()


def test_nonroad_cache_fingerprint_tracks_final_building_position_and_model() -> None:
    base = BuildingPlacementPlan(
        osm_key="way/airtest",
        geometry_index=0,
        geometry_kind="polygon",
        x=3170.53955078125,
        z=3193.103515625,
        heading_degrees=110.31077571034629,
        model_path=r"o\hous\hangar_2.p3d",
        support_polygon=(
            (3165.600234862171, 3211.80350857504),
            (3154.6156662242006, 3182.1255158666754),
            (3175.478866700329, 3174.40352267496),
            (3186.4634353382994, 3204.0815153833246),
        ),
        building_family="agricultural",
    )
    moved = replace(
        base,
        x=base.x + 12.0,
        support_polygon=tuple((x + 12.0, z) for x, z in base.support_polygon),
    )
    changed_model = replace(base, model_path=r"o\hous\stodola.p3d")

    assert generator._building_plan_fingerprint((base,)) != generator._building_plan_fingerprint((moved,))
    assert generator._building_plan_fingerprint((base,)) != generator._building_plan_fingerprint((changed_model,))


def test_airtest10_hangar_is_rejected_if_cached_transform_still_crosses_final_road() -> None:
    library = _library(STOCK_BUILDING_RESISTANCE_PRESET)
    hangar = WorldObject(
        55640,
        r"o\hous\hangar_2.p3d",
        3170.53955078125,
        18.24526023864746,
        3193.103515625,
        110.31077571034629,
    )
    clear_house = WorldObject(
        55641,
        r"o\hous\domek03.p3d",
        3300.0,
        18.0,
        3300.0,
        15.0,
    )
    result = ObjectGenerationResult(
        objects=(hangar, clear_house),
        road_objects=0,
        building_objects=2,
        forest_objects=0,
        road_objects_truncated=False,
        building_objects_truncated=False,
        forest_objects_truncated=False,
        model_usage=(
            (hangar.model_path, 1),
            (clear_house.model_path, 1),
        ),
    )
    road_report = SimpleNamespace(
        objects=(
            WorldObject(
                8782,
                r"o\road\sil12.p3d",
                3168.87255859375,
                11.318740844726562,
                3175.530029296875,
                276.5677121713669,
            ),
        )
    )
    spec = SimpleNamespace(
        road_segment_length=24.5,
        cells=128,
        cell_size=50.0,
        world_size=6400.0,
    )
    elevations = (11.4,) * (spec.cells * spec.cells)

    revised, removed = _remove_stock_buildings_overlapping_final_roads(
        result,
        library,
        road_report,
        elevations,
        spec,
    )

    assert removed == (hangar,)
    assert revised.objects == (clear_house,)
    assert revised.building_objects == 1
    assert dict(revised.model_usage) == {clear_house.model_path: 1}


def test_stock_fit_revision_is_final_active_placement_cache_salt() -> None:
    from cwr_worldgen import final_building_road_clearance_policy as clearance
    from cwr_worldgen import stock_building_extensions as extensions

    assert (
        clearance._CACHE_REVISION
        == extensions._BUILDING_PLACEMENT_CACHE_REVISION
    )
    assert "serialized-stock-audit" in clearance._CACHE_REVISION
