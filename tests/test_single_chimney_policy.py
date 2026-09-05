from __future__ import annotations

from dataclasses import replace

from cwr_worldgen import osm_house_modeler_upgrade as upgrade
from cwr_worldgen.procedural_buildings import BuildingVariantKey


def _key(*, width_m: float = 16.0, texture_variant: int = 0) -> BuildingVariantKey:
    return BuildingVariantKey(
        family="residential",
        roof_style="gabled",
        width_m=width_m,
        length_m=18.0,
        height_m=7.0,
        regional_style="swedish_wood",
        texture_variant=texture_variant,
    )


def test_style_planner_never_requests_multiple_chimneys() -> None:
    # Exercise enough deterministic seeds and large widths to cover the old
    # width>=12 m / 18% second-chimney branch.
    counts = {
        upgrade.detail_plan_for_key(_key(width_m=width, texture_variant=variant)).chimney_count
        for width in (6.0, 12.0, 18.0, 30.0)
        for variant in range(256)
    }
    assert counts <= {0, 1}
    assert 1 in counts


def test_non_house_families_still_do_not_gain_chimneys() -> None:
    key = _key()
    assert upgrade.detail_plan_for_key(replace(key, family="industrial")).chimney_count == 0
    assert upgrade.detail_plan_for_key(replace(key, family="agricultural")).chimney_count == 0


def test_geometry_source_has_hard_single_chimney_clamp() -> None:
    source = (upgrade.__file__ and open(upgrade.__file__, encoding="utf-8").read())
    assert 'chimney_count = min(1, max(0, int(plan.chimney_count)))' in source
    assert 'chimney-second' not in source
    assert 'chimney_count = 2' not in source
