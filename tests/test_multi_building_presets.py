from __future__ import annotations

from pathlib import Path
import json
import pickle

from cwr_worldgen import generator
from cwr_worldgen import osm_house_modeler_runtime as runtime
from cwr_worldgen.building_country_policy import building_country_options
from cwr_worldgen.osm import BboxProjection, OsmDataset
from cwr_worldgen.procedural_buildings import BuildingGenerationResult
from cwr_worldgen.multi_building_presets import (
    BUILDING_MULTI_PREFIX,
    PROCEDURAL_AUTO_PRESET,
    MultiBuildingLibrary,
    building_preset_ids,
    encode_building_presets,
)
from cwr_worldgen.stock_building_extensions import (
    STOCK_BUILDING_VANILLA_PRESET,
    STOCK_BUILDING_RESISTANCE_PRESET,
    STOCK_BUILDING_HAUS_ONLY_PRESET,
    encode_stock_building_presets,
    stock_model_source,
)
from cwr_worldgen.stock_building_policy import StockBuildingLibrary


def _countries(count: int = 2) -> tuple[str, ...]:
    options = building_country_options()
    assert len(options) >= count
    return tuple(identifier for identifier, _label in options[:count])


def test_mixed_building_preset_encoding_round_trips_canonically() -> None:
    country = _countries(1)[0]
    encoded = encode_building_presets(
        (country, STOCK_BUILDING_VANILLA_PRESET, PROCEDURAL_AUTO_PRESET)
    )

    assert encoded.startswith(BUILDING_MULTI_PREFIX)
    assert building_preset_ids(encoded) == (
        STOCK_BUILDING_VANILLA_PRESET,
        PROCEDURAL_AUTO_PRESET,
        country,
    )


def test_stock_only_encoding_keeps_existing_stock_multi_transport() -> None:
    selected = (
        STOCK_BUILDING_RESISTANCE_PRESET,
        STOCK_BUILDING_VANILLA_PRESET,
    )
    assert encode_building_presets(selected) == encode_stock_building_presets(selected)


def test_mixed_factory_builds_stock_and_procedural_children() -> None:
    encoded = encode_building_presets(
        (STOCK_BUILDING_VANILLA_PRESET, PROCEDURAL_AUTO_PRESET)
    )
    library = generator.ProceduralBuildingLibrary(
        world_name="wg_mixed_preset_test",
        house_style_preset=encoded,
        maximum_variants=8,
    )

    assert isinstance(library, MultiBuildingLibrary)
    assert isinstance(library.stock_library, StockBuildingLibrary)
    assert not isinstance(library.procedural_library, StockBuildingLibrary)
    assert library.stock_presets == (STOCK_BUILDING_VANILLA_PRESET,)
    assert library.procedural_presets == (PROCEDURAL_AUTO_PRESET,)
    assert library.house_style_preset == encoded


def test_multiple_procedural_country_presets_use_one_procedural_library() -> None:
    countries = _countries(2)
    encoded = encode_building_presets(countries)
    library = generator.ProceduralBuildingLibrary(
        world_name="wg_multi_procedural_test",
        house_style_preset=encoded,
        maximum_variants=8,
    )

    assert encoded.startswith(BUILDING_MULTI_PREFIX)
    assert not isinstance(library, MultiBuildingLibrary)
    assert not isinstance(library, StockBuildingLibrary)
    assert library.house_style_preset == encoded


def test_no_checked_presets_keeps_automatic_transport() -> None:
    assert encode_building_presets(()) == "auto"
    assert building_preset_ids("auto") == ()


def test_explicit_procedural_auto_keeps_distinct_checkbox_transport() -> None:
    assert encode_building_presets((PROCEDURAL_AUTO_PRESET,)) == PROCEDURAL_AUTO_PRESET
    assert building_preset_ids(PROCEDURAL_AUTO_PRESET) == (PROCEDURAL_AUTO_PRESET,)

    library = generator.ProceduralBuildingLibrary(
        world_name="wg_explicit_procedural_auto_test",
        house_style_preset=PROCEDURAL_AUTO_PRESET,
        maximum_variants=8,
    )
    assert not isinstance(library, MultiBuildingLibrary)
    assert library.house_style_preset == "auto"


def _empty_dataset() -> OsmDataset:
    return OsmDataset(
        source_generator="multi-building-presets-test",
        element_count=0,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(),
    )


def test_mixed_place_point_registers_usage_for_both_child_libraries() -> None:
    encoded = encode_building_presets(
        (STOCK_BUILDING_VANILLA_PRESET, PROCEDURAL_AUTO_PRESET)
    )
    library = generator.ProceduralBuildingLibrary(
        world_name="wg_mixed_usage_test",
        house_style_preset=encoded,
        maximum_variants=16,
    )
    assert isinstance(library, MultiBuildingLibrary)

    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 1000.0)
    library.prepare(_empty_dataset(), projection, 12.0)

    placements = [
        library.place_point(
            {"building": "house", "name": f"House {index}"},
            10.0,
            0.0,
            x=20.0 + index * 7.0,
            z=100.0 + (index % 7) * 9.0,
        )
        for index in range(64)
    ]

    procedural_count = sum(library.procedural_library._usage.values())
    stock_count = sum(library.stock_library._usage.values())
    assert procedural_count > 0
    assert stock_count > 0
    assert procedural_count + stock_count == len(placements)


def test_mixed_cache_state_updates_both_child_libraries(tmp_path: Path) -> None:
    encoded = encode_building_presets(
        (STOCK_BUILDING_VANILLA_PRESET, PROCEDURAL_AUTO_PRESET)
    )
    library = generator.ProceduralBuildingLibrary(
        world_name="wg_mixed_cache_test",
        house_style_preset=encoded,
        maximum_variants=8,
    )
    assert isinstance(library, MultiBuildingLibrary)

    cache_dir = tmp_path / "cache"
    library.cache_dir = cache_dir
    library.cache_enabled = False
    library.cache_refresh = True
    library.cache_hits = 0
    library.cache_misses = 0

    for child in (library.stock_library, library.procedural_library):
        assert child.cache_dir == cache_dir
        assert child.cache_enabled is False
        assert child.cache_refresh is True
        assert child.cache_hits == 0
        assert child.cache_misses == 0


def test_mixed_library_survives_pickle_round_trip() -> None:
    encoded = encode_building_presets(
        (STOCK_BUILDING_VANILLA_PRESET, PROCEDURAL_AUTO_PRESET)
    )
    library = generator.ProceduralBuildingLibrary(
        world_name="wg_mixed_pickle_test",
        house_style_preset=encoded,
        maximum_variants=8,
    )
    assert isinstance(library, MultiBuildingLibrary)

    restored = pickle.loads(pickle.dumps(library))

    assert isinstance(restored, MultiBuildingLibrary)
    assert restored.house_style_preset == encoded
    assert restored.stock_presets == library.stock_presets
    assert restored.procedural_presets == library.procedural_presets
    assert restored.world_name == "wg_mixed_pickle_test"


def test_mixed_pool_accepts_stock_modded_and_procedural_sources_together() -> None:
    country = "se_sweden"
    encoded = encode_building_presets(
        (
            STOCK_BUILDING_VANILLA_PRESET,
            STOCK_BUILDING_HAUS_ONLY_PRESET,
            country,
        )
    )
    library = generator.ProceduralBuildingLibrary(
        world_name="wg_stock_mod_proc_test",
        house_style_preset=encoded,
        maximum_variants=8,
    )

    assert isinstance(library, MultiBuildingLibrary)
    assert library.stock_presets == (
        STOCK_BUILDING_VANILLA_PRESET,
        STOCK_BUILDING_HAUS_ONLY_PRESET,
    )
    assert library.procedural_presets == (country,)
    sources = {stock_model_source(model.model_path) for model in library.stock_library.models}
    assert "vanilla" in sources
    assert "haus" in sources


def test_multiple_procedural_presets_drive_multiple_country_styles() -> None:
    countries = ("se_sweden", "jp_japan")
    encoded = encode_building_presets(countries)

    class Library:
        house_style_preset = encoded

    marker = runtime._regional_preset(Library())
    assert marker.startswith("cwr-procedural-multi:")

    selected = {
        runtime.resolve_style(
            tags={"building": "house", "name": f"House {index}"},
            latitude=35.0 + index * 0.01,
            longitude=18.0 + index * 0.01,
            width_m=8.0 + (index % 5),
            length_m=12.0 + (index % 7),
            settlement_context="rural",
            regional_preset=marker,
            seed="multi-procedural-test",
        ).country_profile_identifier
        for index in range(96)
    }
    assert selected == set(countries)


def test_generalized_stock_only_composite_routes_to_stock_library() -> None:
    raw = (
        BUILDING_MULTI_PREFIX
        + STOCK_BUILDING_VANILLA_PRESET
        + ","
        + STOCK_BUILDING_RESISTANCE_PRESET
    )
    library = generator.ProceduralBuildingLibrary(
        world_name="wg_generalized_stock_multi_test",
        house_style_preset=raw,
        maximum_variants=8,
    )

    assert isinstance(library, StockBuildingLibrary)
    assert library.house_style_preset == encode_stock_building_presets(
        (STOCK_BUILDING_VANILLA_PRESET, STOCK_BUILDING_RESISTANCE_PRESET)
    )


def test_mixed_asset_catalogue_merges_procedural_and_stock_rows(tmp_path: Path) -> None:
    class FakeProcedural:
        cache_dir = None
        cache_enabled = True
        cache_refresh = False
        cache_hits = 2
        cache_misses = 3

        @staticmethod
        def is_generated_model(model_path: str) -> bool:
            return str(model_path).startswith("wg_catalogue\\g\\")

        def write_assets(self, source_dir: Path, catalogue_path: Path) -> BuildingGenerationResult:
            document = {
                "schema": 18,
                "generator": "fake procedural",
                "placements": 3,
                "models": [
                    {
                        "model_path": r"wg_catalogue\g\b_generated.p3d",
                        "placements": 3,
                        "stock": False,
                    }
                ],
            }
            catalogue_path.parent.mkdir(parents=True, exist_ok=True)
            catalogue_path.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return BuildingGenerationResult(
                enabled=True,
                placements=3,
                unique_requested_variants=1,
                generated_variants=1,
                reused_placements=2,
                reuse_ratio=2 / 3,
                capped_variants=0,
                model_assets=(),
                texture_files=("d/generated.paa",),
                catalogue_sha256="procedural",
                cache_hits=2,
                cache_misses=3,
            )

    class FakeStock:
        cache_dir = None
        cache_enabled = True
        cache_refresh = False
        cache_hits = 0
        cache_misses = 0

        def write_assets(self, source_dir: Path, catalogue_path: Path) -> BuildingGenerationResult:
            del source_dir
            document = {
                "schema": 2,
                "mode": STOCK_BUILDING_HAUS_ONLY_PRESET,
                "placements": 2,
                "models": [
                    {
                        "model_path": r"haus\h1.p3d",
                        "placements": 2,
                        "stock": True,
                        "source_set": "haus",
                    }
                ],
            }
            catalogue_path.parent.mkdir(parents=True, exist_ok=True)
            catalogue_path.write_text(
                json.dumps(document, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return BuildingGenerationResult(
                enabled=True,
                placements=2,
                unique_requested_variants=1,
                generated_variants=0,
                reused_placements=1,
                reuse_ratio=0.5,
                capped_variants=0,
                model_assets=(),
                texture_files=(),
                catalogue_sha256="stock",
            )

    encoded = encode_building_presets(
        (STOCK_BUILDING_HAUS_ONLY_PRESET, "se_sweden")
    )
    library = MultiBuildingLibrary(
        stock_library=FakeStock(),
        procedural_library=FakeProcedural(),
        house_style_preset=encoded,
        stock_presets=(STOCK_BUILDING_HAUS_ONLY_PRESET,),
        procedural_presets=("se_sweden",),
    )
    source_dir = tmp_path / "source"
    catalogue = tmp_path / "building-asset-catalogue.json"

    result = library.write_assets(source_dir, catalogue)
    document = json.loads(catalogue.read_text(encoding="utf-8"))
    embedded = json.loads((source_dir / "g" / "buildings.json").read_text(encoding="utf-8"))

    assert result.placements == 5
    assert result.generated_variants == 1
    assert result.cache_hits == 2
    assert result.cache_misses == 3
    assert document == embedded
    assert document["selected_stock_presets"] == [STOCK_BUILDING_HAUS_ONLY_PRESET]
    assert document["selected_procedural_presets"] == ["se_sweden"]
    assert document["stock_models"] == 1
    assert {row["model_path"] for row in document["models"]} == {
        r"wg_catalogue\g\b_generated.p3d",
        r"haus\h1.p3d",
    }
    assert not catalogue.with_name("building-asset-catalogue-stock.json").exists()
