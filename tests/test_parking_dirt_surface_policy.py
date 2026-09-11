from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen.parking_dirt_surface_policy import (
    force_dirt_parking_to_gravel,
    is_dirt_parking_surface,
)
from cwr_worldgen.parking_surface_policy import _ParkingGeometry


def _geometry(key: str, surface: str = "paved") -> _ParkingGeometry:
    return _ParkingGeometry(
        osm_key=key,
        outer=((0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)),
        holes=(),
        centre_x=10.0,
        centre_z=5.0,
        half_length=10.0,
        half_width=5.0,
        ux=1.0,
        uz=0.0,
        px=0.0,
        pz=1.0,
        surface=surface,
    )


def _site(key: str, surface: str):
    return SimpleNamespace(
        osm_key=key,
        tags={"site": "parking", "surface": surface},
    )


def test_dirt_like_osm_parking_surfaces_are_recognized() -> None:
    for surface in (
        "dirt",
        "earth",
        "ground",
        "mud",
        "sand",
        "unpaved",
        "compacted",
        "gravel",
        "fine_gravel",
        "pebblestone",
    ):
        assert is_dirt_parking_surface({"surface": surface})

    assert is_dirt_parking_surface({"surface": "asphalt;dirt"})
    assert not is_dirt_parking_surface({"surface": "asphalt"})
    assert not is_dirt_parking_surface({"surface": "paving_stones"})


def test_explicit_dirt_parking_overrides_nearest_road_paved_result() -> None:
    dataset = SimpleNamespace(sites=(
        _site("way/dirt", "dirt"),
        _site("way/asphalt", "asphalt"),
    ))
    geometries = (
        _geometry("way/dirt", "paved"),
        _geometry("way/asphalt", "paved"),
    )

    revised = force_dirt_parking_to_gravel(geometries, dataset)

    assert revised[0].surface == "gravel"
    assert revised[1].surface == "paved"
