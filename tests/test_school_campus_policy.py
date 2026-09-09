from __future__ import annotations

from pathlib import Path

from shapely.geometry import box

from cwr_worldgen import normalization
from cwr_worldgen.osm import BboxProjection
from cwr_worldgen.osm_house_modeler_styles import classify_building
from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary
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

    buildings, statistics = normalization._normalize_buildings(
        elements,
        projection,
        box(0.0, 0.0, 1000.0, 1000.0),
        [],
        normalization.NormalizationSpec(source_dir=Path("unused")),
    )

    by_source = {
        source_id: candidate
        for candidate in buildings
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


def test_school_policy_invalidates_pre_hint_normalized_bundles() -> None:
    install_school_campus_policy()
    assert normalization.NORMALIZED_SCHEMA_VERSION >= 21
