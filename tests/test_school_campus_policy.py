from __future__ import annotations

from pathlib import Path

from PIL import Image
from shapely.geometry import box

from cwr_worldgen import normalization
from cwr_worldgen import osm_house_modeler_fidelity as fidelity
from cwr_worldgen import osm_house_modeler_texture_bridge as texture_bridge
from cwr_worldgen import procedural_buildings as buildings_module
from cwr_worldgen.cache import cache_key as raw_cache_key
from cwr_worldgen.osm import BboxProjection
from cwr_worldgen.osm_house_modeler_styles import classify_building
from cwr_worldgen.procedural_buildings import BuildingVariantKey, ProceduralBuildingLibrary
from cwr_worldgen.school_campus_policy import install_school_campus_policy


def _closed_ring(x0: float, z0: float, width: float, length: float):
    return (
        (x0, z0),
        (x0 + width, z0),
        (x0 + width, z0 + length),
        (x0, z0 + length),
        (x0, z0),
    )


def test_school_campus_marks_only_generic_physical_buildings() -> None:
    install_school_campus_policy()
    projection = BboxProjection.create((0.0, 0.0, 0.01, 0.01), 1000.0)

    def ll(point: tuple[float, float]) -> dict[str, float]:
        latitude, longitude = projection.to_latlon(point)
        return {"lat": latitude, "lon": longitude}

    elements = [
        {
            "type": "way",
            "id": 900,
            "tags": {
                "amenity": "school",
                "building": "no",
                "name": "Regression School Campus",
            },
            "geometry": [ll(point) for point in _closed_ring(100.0, 100.0, 400.0, 300.0)],
        },
        {
            "type": "way",
            "id": 901,
            "tags": {"building": "yes"},
            # 1,200 m2: without school semantics this comfortably trips the
            # oversized-rural agricultural/barn fallback.
            "geometry": [ll(point) for point in _closed_ring(150.0, 160.0, 40.0, 30.0)],
        },
        {
            "type": "way",
            "id": 902,
            "tags": {"building": "barn"},
            "geometry": [ll(point) for point in _closed_ring(260.0, 160.0, 40.0, 30.0)],
        },
        {
            "type": "way",
            "id": 903,
            "tags": {"building": "garage"},
            "geometry": [ll(point) for point in _closed_ring(360.0, 160.0, 12.0, 9.0)],
        },
        {
            "type": "way",
            "id": 904,
            "tags": {"building": "yes"},
            "geometry": [ll(point) for point in _closed_ring(650.0, 160.0, 40.0, 30.0)],
        },
    ]

    normalized_buildings, statistics = normalization._normalize_buildings(
        elements,
        projection,
        box(0.0, 0.0, 1000.0, 1000.0),
        [],
        normalization.NormalizationSpec(source_dir=Path("unused")),
    )

    by_source = {
        source_id: candidate
        for candidate in normalized_buildings
        for source_id in candidate.source_ids
        if source_id in {"way/901", "way/902", "way/903", "way/904"}
    }
    classroom = by_source["way/901"]
    barn = by_source["way/902"]
    garage = by_source["way/903"]
    outside = by_source["way/904"]

    # Keep the old normalization contract: campus membership is a use hint, not
    # a destructive rewrite of the physical OSM building tag or amenity.
    assert classroom.properties["building_kind"] == "yes"
    assert "amenity" not in classroom.properties
    assert classroom.properties.get("name") != "Regression School Campus"
    assert classroom.properties["building:use"] == "school"
    assert "way/900" in classroom.source_ids

    # Explicit auxiliary structures remain authoritative even when they sit on
    # school grounds. A school with a barn or garage still has a barn or garage.
    assert barn.properties["building_kind"] == "barn"
    assert barn.properties.get("building:use") != "school"
    assert garage.properties["building_kind"] == "garage"
    assert garage.properties.get("building:use") != "school"

    # Campus semantics are spatial, not a map-wide antidote to the barn rule.
    assert outside.properties["building_kind"] == "yes"
    assert outside.properties.get("building:use") != "school"
    assert statistics["school_campus_hints"] == 1


def test_school_use_hint_beats_oversized_rural_barn_fallback() -> None:
    install_school_campus_policy()
    tags = {"building": "yes", "building:use": "school"}

    style = classify_building(tags, 30.0, 40.0, settlement="rural")
    assert (style.family, style.building_class) == ("school", "school")

    library = ProceduralBuildingLibrary(world_name="school-campus-regression")
    key = library.key_for(
        tags,
        30.0,
        40.0,
        settlement_context="rural",
    )
    assert key.family == "school"


def test_short_closed_school_keeps_window_facade_below_generic_height_cutoff() -> None:
    install_school_campus_policy()
    key = BuildingVariantKey(
        "school",
        "gabled",
        20.0,
        30.0,
        3.0,
        interiors=False,
        facade_storeys=1,
    )

    # Test30's 3 m gabled schools have roughly 1.95 m of wall below the eaves.
    # That is below the generic 2.55 m anti-tiny-window threshold but still tall
    # enough for one recognisable school window band.
    bands = buildings_module._closed_facade_bands(
        key,
        1.95,
        span_m=20.0,
        ground_texture="school-windows.paa",
        upper_texture="school-windows.paa",
        plain_texture="school-plain.paa",
    )
    assert bands == ((0.0, 1.95, "school-windows.paa", True),)


def test_closed_school_front_keeps_windows_on_both_sides_of_central_door() -> None:
    install_school_campus_policy()
    metadata = {
        "window": {
            "width_m": 1.8,
            "height_m": 1.65,
            "sill_height_m": 0.9,
            "target_bay_spacing_m": 4.0,
            "density_multiplier": 1.0,
            "frame_material": "painted timber",
            "type": "casement",
        },
        "door": {
            "width_m": 1.0,
            "height_m": 2.1,
            "material": "timber",
            "type": "panel",
        },
    }
    image = fidelity.render_modeler_facade_texture(
        Image.new("RGB", (64, 64), (150, 150, 150)),
        metadata,
        family="school",
        front=True,
    )

    glass = (48, 63, 66)
    assert image.getpixel((10, 30)) == glass
    assert image.getpixel((54, 30)) == glass


def test_school_visual_change_invalidates_only_relevant_model_and_front_texture_caches() -> None:
    install_school_campus_policy()
    namespace = "procedural-building-model-v49-robust-polygon-roof-triangulation"
    school_payload = {"variant": {"family": "school", "width_m": 20.0}}
    house_payload = {"variant": {"family": "residential", "width_m": 20.0}}

    assert buildings_module.cache_key(namespace, school_payload) != raw_cache_key(
        namespace, school_payload
    )
    assert buildings_module.cache_key(namespace, house_payload) == raw_cache_key(
        namespace, house_payload
    )

    school_front_identity = texture_bridge.modeler_texture_cache_identity(
        "front",
        family="school",
        style_token="default",
    )
    house_front_identity = texture_bridge.modeler_texture_cache_identity(
        "front",
        family="residential",
        style_token="default",
    )
    assert "school-closed-windows-v1" in school_front_identity
    assert "school-closed-windows-v1" not in house_front_identity


def test_school_policy_invalidates_pre_hint_normalized_bundles() -> None:
    install_school_campus_policy()
    assert normalization.NORMALIZED_SCHEMA_VERSION >= 21
