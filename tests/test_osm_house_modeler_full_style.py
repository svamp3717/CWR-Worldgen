from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen.osm import BboxProjection
from cwr_worldgen.osm_house_modeler_full_style import (
    detail_spec_from_key,
    key_fields,
    resolve_style,
)
from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary
from cwr_worldgen import procedural_buildings as pb
from cwr_worldgen import osm_house_modeler_upgrade as upgrade


def _empty_dataset():
    return SimpleNamespace(
        places=(), building_polygons=(), building_points=(), place_areas=(),
    )


def test_detailed_style_choice_exposes_modeler_architecture_fields() -> None:
    choice = resolve_style(
        tags={"building": "cottage", "addr:country": "SE"},
        latitude=59.33,
        longitude=18.07,
        width_m=9.0,
        length_m=13.0,
        settlement_context="rural",
        seed="full-style-test",
    )
    fields = key_fields(choice)
    assert choice.country_code == "SE"
    assert choice.country_profile_identifier.startswith("se_")
    assert fields["wall_material"]
    assert fields["roof_material"]
    assert fields["foundation_type"]
    assert fields["storey_height_m"] >= 2.4
    assert fields["wall_thickness_m"] >= 0.08
    assert fields["roof_pitch_degrees"] >= 5.0
    assert fields["window_width_m"] > 0.0
    assert fields["window_height_m"] > 0.0
    assert fields["door_width_m"] > 0.0
    assert fields["door_height_m"] > 0.0
    assert fields["exterior_detail_spec_json"]
    assert fields["texture_style_token"]


def test_library_resolves_country_per_building_not_world_centre() -> None:
    projection = BboxProjection.create((35.0, -10.0, 70.0, 30.0), 20_000.0)
    library = ProceduralBuildingLibrary(world_name="WGStyle", maximum_variants=16)
    library.prepare(_empty_dataset(), projection, 8.0)

    sx, sz = projection.to_world((59.33, 18.07))
    fx, fz = projection.to_world((48.86, 2.35))
    sweden = library.plan_point({"building": "house"}, 9.0, 0.0, x=sx, z=sz).requested
    france = library.plan_point({"building": "house"}, 9.0, 0.0, x=fx, z=fz).requested

    assert sweden.country_style_identifier.startswith("se_")
    assert france.country_style_identifier.startswith("fr_")
    assert sweden.country_style_identifier != france.country_style_identifier
    assert sweden.texture_style_token
    assert france.texture_style_token
    assert sweden.texture_style_token != france.texture_style_token


def test_country_tags_override_building_coordinate_country() -> None:
    projection = BboxProjection.create((35.0, -10.0, 70.0, 30.0), 20_000.0)
    library = ProceduralBuildingLibrary(world_name="WGTags", maximum_variants=8)
    library.prepare(_empty_dataset(), projection, 8.0)
    fx, fz = projection.to_world((48.86, 2.35))
    key = library.plan_point(
        {"building": "house", "addr:country": "SE"}, 9.0, 0.0, x=fx, z=fz
    ).requested
    assert key.country_style_identifier.startswith("se_")


def test_style_values_drive_enterable_opening_and_wall_helpers() -> None:
    projection = BboxProjection.create((58.0, 17.0, 60.0, 19.0), 8_000.0)
    library = ProceduralBuildingLibrary(
        world_name="WGInterior", generate_interiors=True, maximum_variants=8
    )
    library.prepare(_empty_dataset(), projection, 8.0)
    x, z = projection.to_world((59.33, 18.07))
    key = library.plan_point(
        {"building": "house", "addr:country": "SE", "building:levels": "2"},
        10.0,
        0.0,
        x=x,
        z=z,
    ).requested
    assert key.interiors
    assert key.wall_thickness_m > 0.0
    assert pb._interior_wall_thickness(key) >= key.wall_thickness_m
    door_half, door_height, _ = pb._door_dimensions(key)
    assert door_half > 0.0
    assert door_height > 1.7
    if key.door_width_m > 0.0:
        assert abs(door_half * 2.0 - key.door_width_m) < 0.25
    if key.door_height_m > 0.0:
        assert abs(door_height - key.door_height_m) < 0.25


def test_country_exterior_detail_spec_drives_adapter_plan() -> None:
    projection = BboxProjection.create((58.0, 17.0, 60.0, 19.0), 8_000.0)
    library = ProceduralBuildingLibrary(world_name="WGDetails", maximum_variants=8)
    library.prepare(_empty_dataset(), projection, 8.0)
    x, z = projection.to_world((59.33, 18.07))
    key = library.plan_point(
        {"building": "cottage", "addr:country": "SE", "building:levels": "2"},
        10.0,
        0.0,
        x=x,
        z=z,
    ).requested
    spec = detail_spec_from_key(key)
    assert {"stairs", "porches", "chimneys", "balconies", "rainwater"} <= set(spec)
    plan = upgrade.detail_plan_for_key(key, foundation_depth=max(0.8, key.foundation_depth_m))
    assert plan.chimney_count == (
        int(spec["chimneys"].get("count", 1))
        if spec["chimneys"].get("enabled") and key.family in {"residential", "townhouse"}
        else 0
    )
    assert plan.balcony_count == (
        int(spec["balconies"].get("count", 1))
        if spec["balconies"].get("enabled") and key.family in {"residential", "townhouse", "urban"}
        else 0
    )


def test_roof_material_token_gets_independent_texture_slot() -> None:
    library = ProceduralBuildingLibrary(world_name="WGRoof")
    plain = library._roof_texture("gabled", 0)
    tile = library._roof_texture("gabled|clay tile|#994433", 0)
    metal = library._roof_texture("gabled|painted metal|#555555", 0)
    assert len({plain, tile, metal}) == 3


def test_modeler_can_select_roof_storey_and_carries_spec() -> None:
    selected = None
    for width in range(7, 24):
        choice = resolve_style(
            tags={"building": "cottage", "addr:country": "SE", "building:levels": "2"},
            latitude=59.33,
            longitude=18.07,
            width_m=float(width),
            length_m=float(width + 4),
            settlement_context="rural",
            seed=f"roof-storey-{width}",
        )
        if choice.roof_storey:
            selected = choice
            break
    assert selected is not None
    assert selected.roof_style == "gabled"
    assert selected.roof_storey_spec
