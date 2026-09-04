from __future__ import annotations

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
