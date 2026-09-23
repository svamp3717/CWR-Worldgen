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
    STOCK_BUILDING_AFGANO_LABEL,
    STOCK_BUILDING_AFGANO_PRESET,
    STOCK_BUILDING_AGS_BUILD_LABEL,
    STOCK_BUILDING_AGS_BUILD_PRESET,
    STOCK_BUILDING_SEB_ILO_LABEL,
    STOCK_BUILDING_SEB_ILO_PRESET,
    STOCK_BUILDING_AGS_ONLY_LABEL,
    STOCK_BUILDING_AGS_ONLY_PRESET,
    STOCK_BUILDING_ART_BD_LABEL,
    STOCK_BUILDING_ART_BD_PRESET,
    STOCK_BUILDING_BAS_O_AFRICAHUT_LABEL,
    STOCK_BUILDING_BAS_O_AFRICAHUT_PRESET,
    STOCK_BUILDING_BAS_O_GENERAL_LABEL,
    STOCK_BUILDING_BAS_O_GENERAL_PRESET,
    STOCK_BUILDING_BAS_O_MIDDLEAST_LABEL,
    STOCK_BUILDING_BAS_O_MIDDLEAST_PRESET,
    STOCK_BUILDING_BAS_O_SHANTY_LABEL,
    STOCK_BUILDING_BAS_O_SHANTY_PRESET,
    STOCK_BUILDING_CAF_KKK_BUILDINGS2_LABEL,
    STOCK_BUILDING_CAF_KKK_BUILDINGS2_PRESET,
    STOCK_BUILDING_DMA_LIBYA_O_LABEL,
    STOCK_BUILDING_DMA_LIBYA_O_PRESET,
    STOCK_BUILDING_CATINTRO_LABEL,
    STOCK_BUILDING_CATINTRO_PRESET,
    STOCK_BUILDING_FDF_LABEL,
    STOCK_BUILDING_FDF_PRESET,
    STOCK_BUILDING_SFP4_LABEL,
    STOCK_BUILDING_SFP4_PRESET,
    STOCK_BUILDING_HAUS_COMBINED_PRESET,
    STOCK_BUILDING_HAUS_ONLY_LABEL,
    STOCK_BUILDING_HAUS_ONLY_PRESET,
    STOCK_BUILDING_MULTI_PREFIX,
    STOCK_BUILDING_OPTIONS,
    STOCK_BUILDING_PRESETS,
    STOCK_BUILDING_RESISTANCE_PRESET,
    STOCK_BUILDING_VANILLA_PRESET,
    _PROCEDURAL_BRIDGES_CHECKBOX_TEXT,
    _catalogue_hover_text,
    _building_preset_grid_row,
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


def test_building_preset_controls_follow_current_gui_grid_row() -> None:
    class _Label:
        def __init__(self, row):
            self.row = row

        def grid_info(self):
            return {"row": self.row}

    assert _building_preset_grid_row(_Label(3)) == 3
    assert _building_preset_grid_row(_Label("7")) == 7
    assert _building_preset_grid_row(object(), default=5) == 5



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


def test_bas_o_catalogues_are_separate_named_source_presets() -> None:
    data_dir = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "data"
    cases = (
        (
            "general",
            STOCK_BUILDING_BAS_O_GENERAL_PRESET,
            STOCK_BUILDING_BAS_O_GENERAL_LABEL,
            "BAS_O.pbo General buildings",
            72,
        ),
        (
            "middleast",
            STOCK_BUILDING_BAS_O_MIDDLEAST_PRESET,
            STOCK_BUILDING_BAS_O_MIDDLEAST_LABEL,
            "BAS_O.pbo Middle East buildings",
            11,
        ),
        (
            "shanty",
            STOCK_BUILDING_BAS_O_SHANTY_PRESET,
            STOCK_BUILDING_BAS_O_SHANTY_LABEL,
            "BAS_O.pbo Shanty buildings",
            5,
        ),
        (
            "africahut",
            STOCK_BUILDING_BAS_O_AFRICAHUT_PRESET,
            STOCK_BUILDING_BAS_O_AFRICAHUT_LABEL,
            "BAS_O.pbo African hut buildings",
            2,
        ),
    )
    seen_paths: set[str] = set()

    for suffix, preset, label, display_name, expected_count in cases:
        document = json.loads(
            (data_dir / f"BAS_O.pbo {suffix}.json").read_text(encoding="utf-8")
        )
        paths = {row["model_path"].casefold() for row in document["models"]}

        assert document["schema"] == 5
        assert document["source_set"] == f"BAS_O.pbo {suffix}"
        assert document["display_name"] == display_name
        assert label == display_name
        assert len(paths) == expected_count
        assert all(path.startswith("bas_o\\") for path in paths)
        assert paths.isdisjoint(seen_paths)
        seen_paths.update(paths)

        library = _library(preset)
        assert len(library.models) == expected_count
        assert {stock_model_source(model.model_path) for model in library.models} == {"bas_o"}



def test_art_bd_and_caf_kkk_catalogues_are_selectable_source_presets() -> None:
    data_dir = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "data"
    cases = (
        (
            "art_bd.json",
            STOCK_BUILDING_ART_BD_PRESET,
            STOCK_BUILDING_ART_BD_LABEL,
            "ART_BD.pbo buildings",
            "art_bd\\",
            "art_bd",
            37,
        ),
        (
            "caf_kkk_buildings2.json",
            STOCK_BUILDING_CAF_KKK_BUILDINGS2_PRESET,
            STOCK_BUILDING_CAF_KKK_BUILDINGS2_LABEL,
            "CAF_KKK_Buildings2.pbo buildings",
            "caf_kkk_buildings2\\",
            "caf_kkk_buildings2",
            23,
        ),
    )

    for filename, preset, label, expected_label, prefix, source, expected_count in cases:
        document = json.loads((data_dir / filename).read_text(encoding="utf-8"))
        paths = {row["model_path"].casefold() for row in document["models"]}

        assert document["schema"] == 5
        assert document["kind"] == "completed_model_classifications"
        assert document["complete_count"] == expected_count
        assert document["reviewed_count"] == expected_count
        assert label == expected_label
        assert len(paths) == expected_count
        assert all(path.startswith(prefix) for path in paths)

        library = _library(preset)
        assert len(library.models) == expected_count
        assert {stock_model_source(model.model_path) for model in library.models} == {source}


def test_art_bd_and_caf_kkk_catalogues_are_recorded_in_build_metadata(
    tmp_path: Path,
) -> None:
    combined = encode_stock_building_presets(
        (
            STOCK_BUILDING_ART_BD_PRESET,
            STOCK_BUILDING_CAF_KKK_BUILDINGS2_PRESET,
        )
    )
    library = _library(combined)
    library.ground_texture_profile = "everon"
    catalogue = tmp_path / "building-asset-catalogue.json"

    library.write_assets(tmp_path / "source", catalogue)
    document = json.loads(catalogue.read_text(encoding="utf-8"))

    assert document["selected_building_jsons"] == [
        "data/art_bd.json",
        "data/caf_kkk_buildings2.json",
    ]


def test_dma_libya_o_catalogue_is_selectable_source_preset() -> None:
    data_dir = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "data"
    document = json.loads((data_dir / "DMA_libya_o.json").read_text(encoding="utf-8"))
    paths = {row["model_path"].casefold() for row in document["models"]}

    assert document["schema"] == 5
    assert document["source_set"] == "DMA_libya_o.pbo"
    assert document["display_name"] == "DMA_libya_o.pbo buildings"
    assert document["complete_count"] == 14
    assert document["reviewed_count"] == 14
    assert STOCK_BUILDING_DMA_LIBYA_O_LABEL == "DMA_libya_o.pbo buildings"
    assert len(paths) == 14
    assert all(path.startswith("dma_libya_o\\") for path in paths)

    library = _library(STOCK_BUILDING_DMA_LIBYA_O_PRESET)
    assert len(library.models) == 14
    assert {stock_model_source(model.model_path) for model in library.models} == {"dma_libya_o"}


def test_catintro_and_finmod_catalogues_are_selectable_source_presets() -> None:
    data_dir = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "data"
    cases = (
        (
            "catintro.json",
            STOCK_BUILDING_CATINTRO_PRESET,
            STOCK_BUILDING_CATINTRO_LABEL,
            "catintro.pbo buildings",
            "catintro\\",
            "catintro",
            33,
        ),
        (
            "fdf.json",
            STOCK_BUILDING_FDF_PRESET,
            STOCK_BUILDING_FDF_LABEL,
            "finmod buildings",
            "fdf_s\\",
            "fdf",
            18,
        ),
    )

    for filename, preset, label, expected_label, prefix, source, expected_count in cases:
        document = json.loads((data_dir / filename).read_text(encoding="utf-8"))
        paths = {row["model_path"].casefold() for row in document["models"]}

        assert document["schema"] == 5
        assert document["kind"] == "completed_model_classifications"
        assert document["complete_count"] == expected_count
        assert document["reviewed_count"] == expected_count
        assert document["display_name"] == expected_label
        assert label == expected_label
        assert len(paths) == expected_count
        assert all(path.startswith(prefix) for path in paths)

        library = _library(preset)
        assert len(library.models) == expected_count
        assert {stock_model_source(model.model_path) for model in library.models} == {source}


def test_sfp4_catalogue_is_selectable_source_preset() -> None:
    data_dir = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "data"
    document = json.loads((data_dir / "sfp4.json").read_text(encoding="utf-8"))
    paths = {row["model_path"].casefold() for row in document["models"]}

    assert document["schema"] == 5
    assert document["kind"] == "completed_model_classifications"
    assert document["complete_count"] == 20
    assert document["reviewed_count"] == 20
    assert document["display_name"] == "sfp4 buildings"
    assert STOCK_BUILDING_SFP4_LABEL == "sfp4 buildings"
    assert len(paths) == 20
    assert all(
        path.startswith(("sfp_objects\\", "sfp_skaro\\"))
        for path in paths
    )

    library = _library(STOCK_BUILDING_SFP4_PRESET)
    assert len(library.models) == 20
    assert {stock_model_source(model.model_path) for model in library.models} == {"sfp4"}


def test_afgano_and_ags_build_catalogues_are_selectable_source_presets() -> None:
    data_dir = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "data"
    cases = (
        (
            "afgano.json",
            STOCK_BUILDING_AFGANO_PRESET,
            STOCK_BUILDING_AFGANO_LABEL,
            "afgano.pbo buildings",
            "afgano\\",
            "afgano",
            50,
        ),
        (
            "ags_build.json",
            STOCK_BUILDING_AGS_BUILD_PRESET,
            STOCK_BUILDING_AGS_BUILD_LABEL,
            "ags_build.pbo buildings",
            "ags_build\\",
            "ags_build",
            9,
        ),
    )

    for filename, preset, label, expected_label, prefix, source, expected_count in cases:
        document = json.loads((data_dir / filename).read_text(encoding="utf-8"))
        paths = {row["model_path"].casefold() for row in document["models"]}

        assert document["schema"] == 5
        assert document["kind"] == "completed_model_classifications"
        assert document["complete_count"] == expected_count
        assert document["reviewed_count"] == expected_count
        assert document["display_name"] == expected_label
        assert label == expected_label
        assert len(paths) == expected_count
        assert all(path.startswith(prefix) for path in paths)

        library = _library(preset)
        assert len(library.models) == expected_count
        assert {stock_model_source(model.model_path) for model in library.models} == {source}



def test_seb_ilo_catalogue_is_selectable_source_preset_and_recorded_in_metadata(
    tmp_path: Path,
) -> None:
    data_dir = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "data"
    document = json.loads((data_dir / "seb_ilo.json").read_text(encoding="utf-8"))
    paths = {row["model_path"].casefold() for row in document["models"]}

    assert document["schema"] == 5
    assert document["kind"] == "completed_model_classifications"
    assert document["source_set"] == "seb_ilo.pbo"
    assert document["display_name"] == "seb_ilo.pbo buildings"
    assert document["complete_count"] == 4
    assert document["reviewed_count"] == 4
    assert STOCK_BUILDING_SEB_ILO_LABEL == "seb_ilo.pbo buildings"
    assert len(paths) == 4
    assert all(path.startswith("seb_ilo\\") for path in paths)
    assert all(row["placement"] == "Rural" for row in document["models"])

    library = _library(STOCK_BUILDING_SEB_ILO_PRESET)
    assert len(library.models) == 4
    assert {stock_model_source(model.model_path) for model in library.models} == {"seb_ilo"}

    catalogue = tmp_path / "building-asset-catalogue.json"
    library.write_assets(tmp_path / "source", catalogue)
    metadata = json.loads(catalogue.read_text(encoding="utf-8"))
    assert metadata["selected_building_jsons"] == ["data/seb_ilo.json"]

    hover = _catalogue_hover_text(data_dir / "seb_ilo.json").splitlines()
    assert hover[0] == "Models: 4"
    assert "Residential: 4" in hover
    assert "Industrial: 4" in hover
    assert "Agricultural: 4" in hover
    assert "Rural: 4" in hover


def test_mod_catalogue_hover_text_reports_each_category_count() -> None:
    data_dir = Path(__file__).parents[1] / "src" / "cwr_worldgen" / "data"

    assert _catalogue_hover_text(data_dir / "afgano.json").splitlines() == [
        "Models: 50",
        "Residential: 39",
        "Commercial: 1",
        "Industrial: 7",
        "Agricultural: 7",
        "Military: 0",
        "Civic / Public: 0",
        "Religious: 3",
        "Infrastructure: 0",
        "Ruins: 0",
        "Prop / Misc: 0",
        "",
        "Placement:",
        "Urban: 1",
        "Rural: 45",
        "Both: 4",
    ]
    assert _catalogue_hover_text(data_dir / "ags_build.json").splitlines() == [
        "Models: 9",
        "Residential: 7",
        "Commercial: 7",
        "Industrial: 0",
        "Agricultural: 0",
        "Military: 0",
        "Civic / Public: 7",
        "Religious: 0",
        "Infrastructure: 0",
        "Ruins: 0",
        "Prop / Misc: 0",
        "",
        "Placement:",
        "Urban: 9",
        "Rural: 0",
        "Both: 0",
    ]


def test_afgano_and_ags_build_catalogues_are_recorded_in_build_metadata(
    tmp_path: Path,
) -> None:
    combined = encode_stock_building_presets(
        (STOCK_BUILDING_AFGANO_PRESET, STOCK_BUILDING_AGS_BUILD_PRESET)
    )
    library = _library(combined)
    catalogue = tmp_path / "building-asset-catalogue.json"

    library.write_assets(tmp_path / "source", catalogue)
    document = json.loads(catalogue.read_text(encoding="utf-8"))

    assert document["selected_building_jsons"] == [
        "data/afgano.json",
        "data/ags_build.json",
    ]


def test_catintro_and_finmod_catalogues_are_recorded_in_build_metadata(
    tmp_path: Path,
) -> None:
    combined = encode_stock_building_presets(
        (STOCK_BUILDING_CATINTRO_PRESET, STOCK_BUILDING_FDF_PRESET)
    )
    library = _library(combined)
    catalogue = tmp_path / "building-asset-catalogue.json"

    library.write_assets(tmp_path / "source", catalogue)
    document = json.loads(catalogue.read_text(encoding="utf-8"))

    assert document["selected_building_jsons"] == [
        "data/catintro.json",
        "data/fdf.json",
    ]


def test_dma_libya_o_catalogue_is_recorded_in_build_metadata(tmp_path: Path) -> None:
    library = _library(STOCK_BUILDING_DMA_LIBYA_O_PRESET)
    catalogue = tmp_path / "building-asset-catalogue.json"

    library.write_assets(tmp_path / "source", catalogue)
    document = json.loads(catalogue.read_text(encoding="utf-8"))

    assert document["selected_building_jsons"] == ["data/DMA_libya_o.json"]


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
    stock_count = len(STOCK_BUILDING_OPTIONS)
    assert options[:stock_count] == STOCK_BUILDING_OPTIONS
    assert labels[0] == "Automatic (area / country)"
    assert labels[1:1 + stock_count] == tuple(
        label for _identifier, label in STOCK_BUILDING_OPTIONS
    )
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


def test_stock_building_catalogue_paths_do_not_overlap_settlement_prop_models() -> None:
    from cwr_worldgen import osm

    catalogue_paths = {
        model.model_path.replace("/", "\\").casefold()
        for preset in STOCK_BUILDING_PRESETS
        for model in _library(preset).models
    }
    prop_paths = {
        str(model).replace("/", "\\").casefold()
        for model in (
            *osm.STOCK_SETTLEMENT_DETAIL_MODELS,
            *osm.STOCK_STREET_BUS_SHELTER_MODELS,
            *osm.STOCK_STREET_TREE_SURROUND_MODELS,
            *osm.STOCK_STREET_TREE_MODELS,
        )
    }

    assert catalogue_paths.isdisjoint(prop_paths)


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


def test_stock_asset_catalogue_records_source_origin_and_terrain_without_generated_p3ds(tmp_path: Path) -> None:
    library = _library(STOCK_BUILDING_RESISTANCE_PRESET)
    library.ground_texture_profile = "nogova"
    placement = library.plan_point({"building": "house"}, 10.0, 0.0, x=20.0, z=30.0)
    library.register_placement(placement, foundation_depth_m=2.0)
    catalogue = tmp_path / "building-asset-catalogue.json"

    result = library.write_assets(tmp_path / "source", catalogue)
    document = json.loads(catalogue.read_text(encoding="utf-8"))

    assert document["mode"] == STOCK_BUILDING_RESISTANCE_PRESET
    assert document["ground_texture_profile"] == "nogova"
    assert document["selected_building_jsons"] == [
        "data/stock_building_models_resistance.json"
    ]
    assert document["generated_models"] == 0
    assert document["generated_variants"] == 0
    assert document["models"][0]["source_set"] == "resistance"
    assert document["models"][0]["origin_lift_m"] > 0.0
    assert result.generated_variants == 0
    assert result.model_assets == ()


def test_stock_asset_catalogue_saves_terrain_with_multiple_bas_o_building_sets(
    tmp_path: Path,
) -> None:
    combined = encode_stock_building_presets(
        (
            STOCK_BUILDING_BAS_O_GENERAL_PRESET,
            STOCK_BUILDING_BAS_O_SHANTY_PRESET,
        )
    )
    library = _library(combined)
    library.ground_texture_profile = "desert"
    catalogue = tmp_path / "building-asset-catalogue.json"

    library.write_assets(tmp_path / "source", catalogue)
    document = json.loads(catalogue.read_text(encoding="utf-8"))

    assert document["ground_texture_profile"] == "desert"
    assert document["selected_building_jsons"] == [
        "data/BAS_O.pbo general.json",
        "data/BAS_O.pbo shanty.json",
    ]



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


def _airtest10_stock_road_fixture():
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
    plan = BuildingPlacementPlan(
        osm_key="way/airtest",
        geometry_index=0,
        geometry_kind="polygon",
        x=hangar.x,
        z=hangar.z,
        heading_degrees=hangar.heading_degrees,
        model_path=hangar.model_path,
        support_polygon=(),
        building_family="agricultural",
    )
    return hangar, clear_house, result, road_report, spec, elevations, plan


def test_airtest10_hangar_is_replaced_by_smaller_agricultural_model() -> None:
    library = _library(STOCK_BUILDING_RESISTANCE_PRESET)
    hangar, clear_house, result, road_report, spec, elevations, plan = (
        _airtest10_stock_road_fixture()
    )

    revised, removed = _remove_stock_buildings_overlapping_final_roads(
        result,
        library,
        road_report,
        elevations,
        spec,
        building_plans=(plan,),
    )

    replacement = revised.objects[0]
    assert removed == ()
    assert replacement.object_id == hangar.object_id
    assert replacement.model_path == r"o\hous\stodola3.p3d"
    assert replacement.x == hangar.x
    assert replacement.z == hangar.z
    replacement_model = next(
        model for model in library.models
        if model.model_path.casefold() == replacement.model_path.casefold()
    )
    assert abs(
        replacement.y - (11.4 + 0.10 + replacement_model.origin_to_bottom_m)
    ) < 1.0e-6
    assert revised.objects[1] == clear_house
    assert revised.building_objects == 2
    assert dict(revised.model_usage) == {
        replacement.model_path: 1,
        clear_house.model_path: 1,
    }


def test_stock_road_rescue_uses_residential_when_same_family_has_no_safe_smaller_model() -> None:
    library = _library(STOCK_BUILDING_RESISTANCE_PRESET)
    hangar, clear_house, result, road_report, spec, elevations, plan = (
        _airtest10_stock_road_fixture()
    )
    hangar_model = next(
        model for model in library.models
        if model.model_path.casefold() == hangar.model_path.casefold()
    )
    residential = next(
        model for model in library.models
        if model.model_path.casefold() == r"o\hous\domek02.p3d"
    )
    library.models = (hangar_model, residential)

    revised, removed = _remove_stock_buildings_overlapping_final_roads(
        result,
        library,
        road_report,
        elevations,
        spec,
        building_plans=(plan,),
    )

    assert removed == ()
    assert revised.objects[0].model_path == residential.model_path
    assert revised.objects[1] == clear_house
    assert revised.building_objects == 2


def test_stock_road_rescue_rejects_when_no_smaller_selected_model_can_clear() -> None:
    library = _library(STOCK_BUILDING_RESISTANCE_PRESET)
    hangar, clear_house, result, road_report, spec, elevations, plan = (
        _airtest10_stock_road_fixture()
    )
    hangar_model = next(
        model for model in library.models
        if model.model_path.casefold() == hangar.model_path.casefold()
    )
    library.models = (hangar_model,)

    revised, removed = _remove_stock_buildings_overlapping_final_roads(
        result,
        library,
        road_report,
        elevations,
        spec,
        building_plans=(plan,),
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
    assert extensions.STOCK_PLACEMENT_CACHE_NAMESPACE == (
        "nonroad-object-placement-v101-stock-road-model-rescue"
    )
    assert "stock-road-model-rescue" in clearance._CACHE_REVISION
