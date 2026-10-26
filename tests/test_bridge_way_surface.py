# SPDX-License-Identifier: GPL-3.0-or-later
"""Buried bridge surface fixes are explicit road P3Ds, not terrain ribbons."""
from __future__ import annotations

from types import SimpleNamespace

from cwr_worldgen.bridge_way_surface import (
    _cover_lengths, bridge_road_surface_covers,
)
from cwr_worldgen.bridge_render_policy import (
    _visible_deck_point, _STOCK_ROADWAY_HALF_LENGTH_METRES,
)
from cwr_worldgen.model import WorldObject


def _spec():
    return SimpleNamespace(cells=256, cell_size=5.0, world_size=1280.0)


def _bridge(*, x=400.0, z=400.0, deck=9.0, heading=90.0):
    # Stock visible-deck local Y is +13.04933 metres.
    return WorldObject(
        10, r"O\Hous\most_stred30.p3d",
        x, deck - 13.049331665039062, z, heading,
    )


def _elevations(value=6.0):
    return [value] * (256 * 256)


def _buried_left_bank():
    # The left 20 m of a bridge are buried by its grassy bank. On the
    # right the bridge deck stands safely above terrain.
    terrain = _elevations()
    for z in range(256):
        for x in range(256):
            if x * 5.0 < 398.0:
                terrain[z * 256 + x] = 10.0
    return terrain


def test_stock_cover_lengths_prefer_25_and_6_not_long_overshoots():
    parts = _cover_lengths(31.0)
    assert [length for length, _ in parts] == [25.0, 6.0]
    assert _cover_lengths(5.9) == [(6.0, r"O\road\sil6.p3d")]


def test_clear_elevated_bridge_needs_no_duplicate_road_p3ds():
    road = _bridge()
    assert bridge_road_surface_covers(
        [road], _elevations(2), _spec(),
    ) == ()


def test_grass_buried_at_bridge_abutment_gets_visible_drivable_road_p3d():
    road = _bridge()
    objects = bridge_road_surface_covers(
        [road], _buried_left_bank(), _spec(), starting_id=500,
    )
    assert objects
    assert all(obj.model_path.startswith("O\\road\\sil") for obj in objects)
    assert [obj.object_id for obj in objects] == list(
        range(500, 500 + len(objects))
    )
    left_edge = _visible_deck_point(road, -_STOCK_ROADWAY_HALF_LENGTH_METRES)
    assert any(abs(obj.z - left_edge[2]) < 1 for obj in objects)
    assert all(obj.y > 9.0 for obj in objects)
    assert all(-13 < obj.pitch_degrees < 13 for obj in objects)
    assert all(0 < obj.x < 1280 and 0 < obj.z < 1280 for obj in objects)
    # Repeated builds use exactly the same objects and no randomness.
    assert objects == bridge_road_surface_covers(
        [road], _buried_left_bank(), _spec(), starting_id=500,
    )


def test_fully_buried_single_bridge_can_get_road_on_both_halves():
    road = _bridge(deck=9.0)
    objects = bridge_road_surface_covers(
        [road], _elevations(10.0), _spec(),
    )
    assert len(objects) >= 2
    left = [o for o in objects if o.x < road.x]
    right = [o for o in objects if o.x > road.x]
    assert left and right
    assert all(o.y > 10 for o in objects)


def test_unrelated_objects_do_not_get_bridge_overlays():
    debris = WorldObject(3, r"O\misc\something.p3d", 400, 0, 400)
    assert bridge_road_surface_covers(
        [debris], _elevations(20), _spec(),
    ) == ()
