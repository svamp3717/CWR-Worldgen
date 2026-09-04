from __future__ import annotations

from dataclasses import replace
import base64
import json
from cwr_worldgen import parallel_assets
from cwr_worldgen import procedural_buildings as pb
from cwr_worldgen.building_asset_budget_policy import (
    _polygon_variant_budget,
    _texture_cache_tasks,
    _write_modeler_texture_cache_task,
)
from cwr_worldgen.paa import inspect_paa


def _variant(*, overhang: float = 0.2, interiors: bool = False) -> pb.BuildingVariantKey:
    return pb.BuildingVariantKey(
        family="residential",
        roof_style="gabled",
        width_m=10.0,
        length_m=8.0,
        height_m=6.0,
        interiors=interiors,
        second_storey=False,
        regional_style="western_stucco",
        texture_style_token="western_stucco|stucco~|#d8d0c0",
        wall_material="stucco",
        roof_material="tile",
        eave_overhang_m=overhang,
        window_width_m=1.2,
        window_height_m=1.35,
        window_sill_height_m=0.85,
        door_width_m=0.95,
        door_height_m=2.1,
    )


def _encoded_token(*, door_width_m: float) -> str:
    metadata = {
        "texture_renderer_revision": 5,
        "window": {
            "width_m": 1.2,
            "height_m": 1.35,
            "sill_height_m": 0.85,
            "target_bay_spacing_m": 3.6,
            "density_multiplier": 1.0,
            "type": "paired casement",
            "frame_material": "painted timber",
        },
        "door": {
            "width_m": door_width_m,
            "height_m": 2.05,
            "type": "panel",
            "material": "timber",
        },
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).decode("ascii").rstrip("=")
    return f"western_stucco|stucco~{encoded}|cream"


def _placement(key: pb.BuildingVariantKey) -> pb.BuildingPlacement:
    return pb.BuildingPlacement("", 0.0, key, key)


def test_default_polygon_budget_no_longer_bypasses_regular_variant_cap() -> None:
    library = pb.ProceduralBuildingLibrary(world_name="BudgetTest")
    assert library.maximum_variants == 128
    assert library.maximum_polygon_variants == 64
    assert library.maximum_polygon_variants < pb.POLYGON_NATIVE_MAXIMUM_VARIANTS


def test_polygon_budget_scales_down_with_regular_variant_budget() -> None:
    library = pb.ProceduralBuildingLibrary(
        world_name="BudgetTestSmall",
        maximum_variants=40,
    )
    assert library.maximum_polygon_variants == 20


def test_explicit_polygon_budget_is_preserved() -> None:
    library = pb.ProceduralBuildingLibrary(
        world_name="BudgetTestExplicit",
        maximum_variants=40,
        maximum_polygon_variants=300,
    )
    assert library.maximum_polygon_variants == 300


def test_polygon_budget_has_sane_floor_and_ceiling() -> None:
    assert _polygon_variant_budget(8) == 16
    assert _polygon_variant_budget(128) == 64
    assert _polygon_variant_budget(1000) == 96


def test_detailed_building_batches_parallelize_before_sixty_four_misses() -> None:
    assert parallel_assets._BUILDING_PARALLEL_MINIMUM <= 16


def test_modeler_cosmetic_texture_variants_default_to_one() -> None:
    library = pb.ProceduralBuildingLibrary(world_name="TextureBudget")
    assert library.texture_variants == 1


def test_modeler_texture_variant_budget_can_be_explicitly_raised(monkeypatch) -> None:
    monkeypatch.setenv("CWR_WORLDGEN_BUILDING_TEXTURE_VARIANTS", "2")
    library = pb.ProceduralBuildingLibrary(world_name="TextureBudgetOverride")
    assert library.texture_variants == 2


def test_final_rectangular_p3d_budget_is_enforced_after_registration() -> None:
    library = pb.ProceduralBuildingLibrary(
        world_name="FinalBudget",
        maximum_variants=2,
    )
    first = library.register_placement(_placement(_variant(overhang=0.10)))
    second = library.register_placement(_placement(_variant(overhang=0.20)))
    third = library.register_placement(_placement(_variant(overhang=0.30)))

    standard = [key for key in library._usage if not key.footprint_vertices]
    assert len(standard) == 2
    assert first.selected in standard
    assert second.selected in standard
    assert third.selected in standard
    assert sum(library._usage.values()) == 3


def test_386_registration_fanout_is_bounded_to_128_final_rectangular_p3ds() -> None:
    library = pb.ProceduralBuildingLibrary(
        world_name="ReleaseSizedBudget",
        maximum_variants=128,
    )
    for index in range(386):
        library.register_placement(
            _placement(_variant(overhang=0.100 + index * 0.001))
        )

    standard = [key for key in library._usage if not key.footprint_vertices]
    assert len(standard) == 128
    assert sum(library._usage.values()) == 386


def test_final_budget_never_reuses_exterior_for_first_interior() -> None:
    library = pb.ProceduralBuildingLibrary(
        world_name="FinalBudgetModes",
        maximum_variants=1,
    )
    exterior = library.register_placement(_placement(_variant(interiors=False)))
    interior = library.register_placement(_placement(_variant(interiors=True)))

    assert not exterior.selected.interiors
    assert interior.selected.interiors
    # Semantic safety is allowed to exceed the numerical cap by a first member
    # of a distinct engine mode rather than silently deleting enterability.
    assert len(library._usage) == 2


def test_modeler_texture_cache_misses_can_be_prewarmed(tmp_path) -> None:
    library = pb.ProceduralBuildingLibrary(
        world_name="TexturePrewarm",
        cache_dir=tmp_path,
        cache_enabled=True,
    )
    key = _variant()
    library._usage[key] = 1

    total, misses = _texture_cache_tasks(library, pb)
    assert total >= 4
    assert len(misses) == total

    wall = next(task for task in misses if task.kind == "wall")
    _write_modeler_texture_cache_task(wall)
    summary = inspect_paa(wall.cache_path)
    assert summary.width == library.texture_size
    assert summary.height == library.texture_size

    total_after, misses_after = _texture_cache_tasks(library, pb)
    assert total_after == total
    assert len(misses_after) == total - 1


def test_final_budget_never_changes_building_class_or_primary_material_group() -> None:
    library = pb.ProceduralBuildingLibrary(
        world_name="FinalBudgetAppearance",
        maximum_variants=1,
    )
    house_key = replace(
        _variant(),
        building_class="residential",
        colour_palette=("cream", "falun red"),
    )
    cabin_key = replace(
        _variant(overhang=0.33),
        building_class="cabin",
        wall_material="painted vertical timber cladding",
        roof_material="standing-seam metal",
        colour_palette=("ochre yellow", "falun red"),
        texture_style_token="swedish_wood|painted vertical timber cladding~|ochre yellow,falun red",
    )
    first = library.register_placement(_placement(house_key))
    second = library.register_placement(_placement(cabin_key))

    assert first.selected.building_class == "residential"
    assert second.selected.building_class == "cabin"
    assert second.selected.wall_material == "painted vertical timber cladding"
    assert second.selected.colour_palette[0] == "ochre yellow"
    # Fidelity groups may exceed a tiny synthetic numerical cap rather than\n    # silently changing architecture/material identity.
    assert len(library._usage) == 2


def test_texture_cache_collapses_only_pixel_equivalent_outputs(tmp_path) -> None:
    library = pb.ProceduralBuildingLibrary(
        world_name="PixelEquivalentCache",
        cache_dir=tmp_path,
        cache_enabled=True,
    )
    narrow = replace(_variant(), texture_style_token=_encoded_token(door_width_m=0.90))
    wide = replace(_variant(overhang=0.21), texture_style_token=_encoded_token(door_width_m=1.40))
    library._usage[narrow] = 1
    library._usage[wide] = 1

    _total, misses = _texture_cache_tasks(library, pb)
    kinds = [task.kind for task in misses]
    # Door width is invisible to wall/open-wall pixels, so those cache entries
    # collapse. The front layout uses door width and must remain two entries.
    assert kinds.count("wall") == 1
    assert kinds.count("open_wall") == 1
    assert kinds.count("front") == 2
    # Roof rendering ignores facade palette/opening metadata.
    assert kinds.count("roof") == 1
