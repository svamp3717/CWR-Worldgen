# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cwr_worldgen.model import validate_world_identity
from cwr_worldgen.osm import (
    BboxProjection, GeoPolygon, OsmLineFeature, OsmPointFeature, OsmPolygonFeature,
)
from cwr_worldgen.road_ways import is_osm_ways_profile, write_osm_road_ways


TEXTURES = {
    "paved": r"data\silnice.pac",
    "dirt": r"data\cesta.pac",
    "gravel": r"data\asfaltka.pac",
}


def _export(tmp_path, roads, *, include_minor=True, world_size=100.0, **features):
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), world_size)
    path = tmp_path / "test_world.ways"
    info = write_osm_road_ways(
        path, SimpleNamespace(roads=tuple(roads), **features), projection,
        include_minor_roads=include_minor, surface_textures=TEXTURES,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    nodes = {
        int(parts[1]): (float(parts[2]), float(parts[3]))
        for line in lines if (parts := line.split())[0] == "node"
    }
    ways = [
        (int(parts[1]), float(parts[2]), parts[3], [int(node) for node in parts[4:]])
        for line in lines if (parts := line.split())[0] == "way"
    ]
    return info, nodes, ways, path.read_bytes()


def test_ways_share_at_grade_junction_nodes_and_are_deterministic(tmp_path):
    first = OsmLineFeature(
        "way/100", {"highway": "residential", "width": "6.25"},
        ((0.5, 0.15), (0.5, 0.5), (0.5, 0.85)),
    )
    second = OsmLineFeature(
        "way/200", {"highway": "track", "surface": "gravel"},
        ((0.2, 0.5), (0.5, 0.5), (0.8, 0.5)),
    )
    info, nodes, ways, original = _export(tmp_path, [first, second])
    assert info["ways"] == 2
    assert info["nodes"] == 5
    assert {way[2] for way in ways} == {TEXTURES["paved"], TEXTURES["gravel"]}
    assert ways[0][1] == 6.25
    assert len(set(ways[0][3]) & set(ways[1][3])) == 1
    assert all(0 <= x < 100 and 0 <= z < 100 for x, z in nodes.values())
    assert _export(tmp_path, [second, first])[3] == original


def test_ways_skip_bridge_and_tunnel_and_clip_to_terrain(tmp_path):
    ground = OsmLineFeature(
        "way/1", {"highway": "residential", "surface": "dirt"},
        ((0.4, -0.2), (0.4, 0.4), (0.4, 1.2)),
    )
    bridge = OsmLineFeature(
        "way/2", {"highway": "primary", "bridge": "yes"},
        ((0.6, 0.1), (0.6, 0.9)),
    )
    tunnel = OsmLineFeature(
        "way/3", {"highway": "primary", "tunnel": "yes"},
        ((0.8, 0.1), (0.8, 0.9)),
    )
    info, nodes, ways, _ = _export(tmp_path, [ground, bridge, tunnel])
    assert info["excluded_bridge_features"] == 1
    assert len(ways) == 1
    assert ways[0][2] == TEXTURES["dirt"]
    assert all(0 <= x < 100 and 0 <= z < 100 for x, z in nodes.values())


def test_native_ways_profile_is_opt_in_and_validated():
    validate_world_identity(
        name="test_world", display_name="Test World", profile="cwr-ce-ways",
    )
    assert is_osm_ways_profile("cwr-ce-ways")
    assert not is_osm_ways_profile("cwr-ce")
    assert not is_osm_ways_profile("cwa")
    with pytest.raises(ValueError, match="profile"):
        validate_world_identity(
            name="test_world", display_name="Test World", profile="invalid",
        )


def test_ways_reject_missing_surface_texture(tmp_path):
    with pytest.raises(ValueError, match="Roadway texture"):
        write_osm_road_ways(
            tmp_path / "bad.ways",
            SimpleNamespace(roads=()),
            BboxProjection.create((0., 0., 1., 1.), 100.),
            include_minor_roads=True,
            surface_textures={"paved": "has spaces.pac", "dirt": "x.pac", "gravel": "y.pac"},
        )


def test_world_pbo_contains_native_ways_and_enforces_layout(tmp_path):
    from cwr_worldgen.generator import _verify_single_world_pbo_layout
    from cwr_worldgen.pbo import pack_directory

    source = tmp_path / "world"
    source.mkdir()
    (source / "config.cpp").write_text("class CfgWorlds {};", encoding="utf-8")
    (source / "world.wrp").write_bytes(b"4WVR" + b"\x00" * 24)
    (source / "world.ways").write_text(
        "node 1 10 20\nnode 2 30 20\nway 1 6 data\\silnice.pac 1 2\n",
        encoding="utf-8",
    )
    pbo = tmp_path / "world.pbo"
    pack_directory(source, pbo, backend="python")
    result = _verify_single_world_pbo_layout(
        pbo, "world", None, include_ways=True,
    )
    assert result["verified"]
    (source / "world.ways").unlink()
    pack_directory(source, pbo, backend="python")
    with pytest.raises(RuntimeError, match="world.ways"):
        _verify_single_world_pbo_layout(
            pbo, "world", None, include_ways=True,
        )


def test_ways_reject_engine_way_limit_before_writing(tmp_path):
    # The engine limits ways to 10,000 even if the file is below 8 MiB.
    many_roads = [
        OsmLineFeature(
            f"way/{i:05d}", {"highway": "residential"},
            ((0.5, 0.25), (0.5, 0.75)),
        )
        for i in range(10_001)
    ]
    with pytest.raises(ValueError, match="10,000 ways"):
        _export(tmp_path, many_roads)



def _ll(x, z, size=1000.0):
    return (z / size, x / size)


def _open_country_road(size=1000.0):
    return OsmLineFeature(
        "way/1", {"highway": "tertiary"},
        tuple(_ll(*point, size) for point in ((200, 200), (400, 200), (400, 400))),
    )


def _way_points(nodes, way):
    return [nodes[node_id] for node_id in way[3]]


def test_rural_bends_become_gradual_curves_when_space_is_available(tmp_path):
    road = _open_country_road()
    info, nodes, ways, source = _export(tmp_path, [road], world_size=1000)
    points = _way_points(nodes, ways[0])
    assert info["smoothed_rural_bends"] == 1
    assert points[0] == (200, 200)
    assert points[-1] == (400, 400)
    assert (400, 200) not in points
    assert any(380 < x < 400 and 200 < z < 220 for x, z in points)
    assert len(points) >= 7
    assert _export(tmp_path, [road], world_size=1000)[3] == source


def test_urban_place_protects_road_bend(tmp_path):
    village = OsmPointFeature(
        "node/7", {"place": "village"}, _ll(400, 200),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [_open_country_road()], world_size=1000, places=(village,),
    )
    assert info["smoothed_rural_bends"] == 0
    assert _way_points(nodes, ways[0]) == [(200, 200), (400, 200), (400, 400)]


def test_urban_polygon_protects_road_bend(tmp_path):
    outline = [_ll(*p) for p in (
        (375, 190), (430, 190), (430, 230), (375, 230), (375, 190),
    )]
    housing = OsmPolygonFeature(
        "area/10", {"landuse": "residential"},
        (GeoPolygon(tuple(outline)),),
    )
    info, _, _, _ = _export(
        tmp_path, [_open_country_road()], world_size=1000, urban=(housing,),
    )
    assert info["smoothed_rural_bends"] == 0


def test_buildings_and_other_roads_prevent_unsafe_shortcuts(tmp_path):
    outline = [_ll(*p) for p in (
        (385, 205), (399, 205), (399, 218), (385, 218), (385, 205),
    )]
    building = OsmPolygonFeature(
        "building/15", {"building": "yes"},
        (GeoPolygon(tuple(outline)),),
    )
    info, _, _, _ = _export(
        tmp_path, [_open_country_road()], world_size=1000,
        building_polygons=(building,),
    )
    assert info["smoothed_rural_bends"] == 0

    neighbor = OsmLineFeature(
        "way/2", {"highway": "residential"},
        (_ll(380, 212), _ll(420, 212)),
    )
    info, _, _, _ = _export(
        tmp_path, [_open_country_road(), neighbor], world_size=1000,
    )
    assert info["smoothed_rural_bends"] == 0


def test_connected_junction_is_not_rounded(tmp_path):
    spur = OsmLineFeature(
        "way/2", {"highway": "residential"},
        (_ll(400, 200), _ll(500, 100)),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [_open_country_road(), spur], world_size=1000,
    )
    assert info["smoothed_rural_bends"] == 0
    assert (400, 200) in _way_points(nodes, ways[0])
    assert set(ways[0][3]).intersection(ways[1][3])


def test_tight_or_short_bends_keep_original_geometry(tmp_path):
    # Too little approach distance, even in open countryside.
    road = OsmLineFeature(
        "way/short", {"highway": "residential"},
        (_ll(200, 200), _ll(213, 200), _ll(213, 213)),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [road], world_size=1000,
    )
    assert info["smoothed_rural_bends"] == 0
    assert _way_points(nodes, ways[0]) == [
        (200, 200), (213, 200), (213, 213),
    ]



def test_bridge_approach_does_not_round_away_from_bridge_endpoint(tmp_path):
    bridge = OsmLineFeature(
        "way/bridge", {"highway": "tertiary", "bridge": "yes"},
        (_ll(400, 200), _ll(550, 200)),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [_open_country_road(), bridge], world_size=1000,
    )
    assert info["smoothed_rural_bends"] == 0
    assert info["excluded_bridge_features"] == 1
    assert _way_points(nodes, ways[0]) == [(200, 200), (400, 200), (400, 400)]


def test_watercourse_near_corner_blocks_rural_smoothing(tmp_path):
    creek = OsmLineFeature(
        "way/creek", {"waterway": "stream"},
        (_ll(388, 207), _ll(400, 220)),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [_open_country_road()], world_size=1000,
        watercourses=(creek,),
    )
    assert info["smoothed_rural_bends"] == 0
    assert _way_points(nodes, ways[0]) == [(200, 200), (400, 200), (400, 400)]
