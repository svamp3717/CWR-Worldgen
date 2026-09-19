from __future__ import annotations

from cwr_worldgen import generator
from cwr_worldgen.building_preset_mix import (
    BUILDING_MULTI_PREFIX,
    MixedBuildingLibrary,
    building_preset_selection,
    encode_building_presets,
)
from cwr_worldgen.osm import BboxProjection, OsmDataset
from cwr_worldgen.stock_building_extensions import (
    STOCK_BUILDING_HAUS_ONLY_PRESET,
    STOCK_BUILDING_VANILLA_PRESET,
)


def _empty_dataset() -> OsmDataset:
    return OsmDataset(
        source_generator="mixed-building-presets-test",
        element_count=0,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(),
    )


def test_mixed_preset_encoding_round_trips_procedural_and_modded_sources() -> None:
    encoded = encode_building_presets(
        ("auto",),
        (STOCK_BUILDING_HAUS_ONLY_PRESET, STOCK_BUILDING_VANILLA_PRESET),
    )

    assert encoded.startswith(BUILDING_MULTI_PREFIX)
    procedural, stock_presets = building_preset_selection(encoded)
    assert procedural == ("auto",)
    assert stock_presets == (
        STOCK_BUILDING_VANILLA_PRESET,
        STOCK_BUILDING_HAUS_ONLY_PRESET,
    )


def test_generator_factory_builds_mixed_library_for_procedural_plus_stock() -> None:
    encoded = encode_building_presets(("auto",), (STOCK_BUILDING_VANILLA_PRESET,))
    library = generator.ProceduralBuildingLibrary(
        world_name="wg_mixed_buildings",
        maximum_variants=16,
        house_style_preset=encoded,
    )

    assert isinstance(library, MixedBuildingLibrary)
    assert library.house_style_preset == encoded


def test_mixed_library_places_both_generated_and_external_buildings() -> None:
    encoded = encode_building_presets(("auto",), (STOCK_BUILDING_VANILLA_PRESET,))
    library = generator.ProceduralBuildingLibrary(
        world_name="wg_mixed_buildings",
        maximum_variants=32,
        house_style_preset=encoded,
    )
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 1000.0)
    library.prepare(_empty_dataset(), projection, 12.0)

    generated_flags = set()
    placements = []
    for index in range(64):
        placement = library.plan_point(
            {"building": "house", "name": f"House {index}"},
            10.0,
            0.0,
            x=20.0 + index * 7.0,
            z=100.0 + (index % 5) * 11.0,
        )
        generated_flags.add(library.is_generated_model(placement.model_path))
        placements.append(placement)

    assert generated_flags == {False, True}

    for placement in placements:
        library.register_placement(placement, foundation_depth_m=0.75)

    assert sum(library._procedural._usage.values()) > 0
    assert sum(library._stock._usage.values()) > 0


def test_mixed_library_exposes_stock_origin_lift_for_external_models() -> None:
    encoded = encode_building_presets(("auto",), (STOCK_BUILDING_VANILLA_PRESET,))
    library = generator.ProceduralBuildingLibrary(
        world_name="wg_mixed_grounding",
        house_style_preset=encoded,
    )

    model = next(
        model for model in library._stock.models
        if model.origin_to_bottom_m > 0.0
    )
    assert library.origin_lift_for_model(model.model_path) == model.origin_to_bottom_m
