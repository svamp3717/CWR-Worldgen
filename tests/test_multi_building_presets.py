from __future__ import annotations

from pathlib import Path
import pickle

from cwr_worldgen import generator
from cwr_worldgen.building_country_policy import building_country_options
from cwr_worldgen.osm import BboxProjection, OsmDataset
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
    encode_stock_building_presets,
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
