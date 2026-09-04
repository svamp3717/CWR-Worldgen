from __future__ import annotations

from cwr_worldgen import parallel_assets
from cwr_worldgen import procedural_buildings as pb
from cwr_worldgen.building_asset_budget_policy import _polygon_variant_budget


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
