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


def _export(tmp_path, roads, *, include_minor=True, world_size=100.0,
            surface_donors=None, surface_widths=None, require_donor_widths=False,
            bridge_objects=(), **features):
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), world_size)
    path = tmp_path / "test_world.ways"
    info = write_osm_road_ways(
        path, SimpleNamespace(roads=tuple(roads), **features), projection,
        include_minor_roads=include_minor, surface_textures=TEXTURES,
        surface_donors=surface_donors,
        surface_widths=surface_widths,
        require_donor_widths=require_donor_widths,
        bridge_objects=bridge_objects,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    nodes = {
        int(parts[1]): (float(parts[2]), float(parts[3]))
        for line in lines if (parts := line.split()) and parts[0] == "node"
    }
    ways = [
        (int(parts[1]), float(parts[2]), parts[3], [int(node) for node in parts[4:]])
        for line in lines if (parts := line.split()) and parts[0] == "way"
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


def test_urban_place_allows_compact_bend_when_no_building_is_in_the_way(tmp_path):
    from shapely.geometry import LineString
    village = OsmPointFeature(
        "node/7", {"place": "village"}, _ll(400, 200),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [_open_country_road()], world_size=1000, places=(village,),
    )
    points = _way_points(nodes, ways[0])
    assert info["smoothed_rural_bends"] == 1
    assert points[0] == (200, 200) and points[-1] == (400, 400)
    assert (400, 200) not in points
    original = LineString(((200, 200), (400, 200), (400, 400)))
    assert LineString(points).hausdorff_distance(original) <= 5.01


def test_urban_polygon_permits_small_safe_bend_without_obstacles(tmp_path):
    from shapely.geometry import LineString
    outline = [_ll(*p) for p in (
        (375, 190), (430, 190), (430, 230), (375, 230), (375, 190),
    )]
    housing = OsmPolygonFeature(
        "area/10", {"landuse": "residential"},
        (GeoPolygon(tuple(outline)),),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [_open_country_road()], world_size=1000, urban=(housing,),
    )
    points = _way_points(nodes, ways[0])
    assert info["smoothed_rural_bends"] == 1
    assert (400, 200) not in points
    assert LineString(points).hausdorff_distance(
        LineString(((200, 200), (400, 200), (400, 400)))
    ) <= 5.01


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


def test_turn_before_t_junction_smooths_but_shared_node_stays_fixed(tmp_path):
    # The old 24m corner-based guard discarded this bend even though the
    # proposed curve ends more than a road-width away from the junction.
    main = OsmLineFeature(
        "way/main", {"highway": "residential"},
        (_ll(200, 300), _ll(400, 300),
         _ll(400, 318), _ll(400, 460)),
    )
    spur = OsmLineFeature(
        "way/spur", {"highway": "residential"},
        (_ll(400, 318), _ll(490, 318)),
    )
    village = OsmPointFeature(
        "node/near_intersection", {"place": "village"}, _ll(400, 300),
    )
    info, nodes, ways, data = _export(
        tmp_path, [main, spur], world_size=1000, places=(village,),
    )
    assert info["smoothed_rural_bends"] == 1
    assert (400, 300) not in _way_points(nodes, ways[0])
    joined = [(way[1], _way_points(nodes, way)) for way in ways]
    assert sum((400, 318) in pts for _, pts in joined) == 2
    assert _export(
        tmp_path, [spur, main], world_size=1000, places=(village,),
    )[3] == data


def test_settlement_corner_near_building_remains_unmodified(tmp_path):
    village = OsmPointFeature(
        "node/hamlet", {"place": "hamlet"}, _ll(400, 200),
    )
    building = OsmPolygonFeature(
        "building/close", {"building": "yes"},
        (GeoPolygon(tuple(_ll(x, z) for x, z in (
            (394, 202), (406, 202), (406, 218),
            (394, 218), (394, 202),
        ))),),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [_open_country_road()], world_size=1000,
        places=(village,), building_polygons=(building,),
    )
    assert info["smoothed_rural_bends"] == 0
    assert (400, 200) in _way_points(nodes, ways[0])


def test_short_isolated_road_bend_gets_rounded_if_its_radius_fits(tmp_path):
    # Short, isolated corners should no longer be forced into a square turn.
    road = OsmLineFeature(
        "way/short", {"highway": "residential"},
        (_ll(200, 200), _ll(213, 200), _ll(213, 213)),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [road], world_size=1000,
    )
    points = _way_points(nodes, ways[0])
    assert info["smoothed_rural_bends"] == 1
    assert points[0] == (200, 200) and points[-1] == (213, 213)
    assert (213, 200) not in points



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



def test_compatible_end_to_end_ways_merge_and_round_joint_as_one_road(tmp_path):
    first = OsmLineFeature(
        "way/001", {"highway": "tertiary"},
        (_ll(200, 200), _ll(400, 200)),
    )
    second = OsmLineFeature(
        "way/002", {"highway": "tertiary"},
        (_ll(400, 200), _ll(400, 400)),
    )
    info, nodes, ways, content = _export(
        tmp_path, [first, second], world_size=1000,
    )
    pts = _way_points(nodes, ways[0])
    assert info["merged_source_runs"] == 1
    assert info["merged_road_chains"] == 1
    assert info["ways"] == 1
    assert info["smoothed_rural_bends"] == 1
    assert (400, 200) not in pts
    assert {pts[0], pts[-1]} == {(200, 200), (400, 400)}
    assert _export(tmp_path, [second, first], world_size=1000)[3] == content
    # Heading changes gradually, not as a right angle between adjacent
    # segments. At the endpoints the cubic is tangent to the source road.
    directions = []
    for a, b in zip(pts, pts[1:]):
        length = __import__("math").dist(a, b)
        directions.append(((b[0]-a[0])/length, (b[1]-a[1])/length))
    assert min(a[0]*b[0] + a[1]*b[1] for a, b in zip(
        directions, directions[1:])) > 0.94


def test_reversed_ways_can_merge_without_an_unintentional_spur(tmp_path):
    west = OsmLineFeature(
        "way/west", {"highway": "residential"},
        (_ll(400, 200), _ll(200, 200)),
    )
    east = OsmLineFeature(
        "way/east", {"highway": "residential"},
        (_ll(600, 200), _ll(400, 200)),
    )
    info, nodes, ways, _ = _export(tmp_path, [west, east], world_size=1000)
    assert info["ways"] == 1
    assert info["merged_source_runs"] == 1
    merged_points = _way_points(nodes, ways[0])
    assert len(merged_points) > 3
    assert {merged_points[0], merged_points[-1]} == {(200, 200), (600, 200)}


def test_t_junction_remains_three_ways_with_a_shared_node(tmp_path):
    roads = [
        OsmLineFeature("way/a", {"highway": "residential"}, (
            _ll(200, 200), _ll(400, 200),
        )),
        OsmLineFeature("way/b", {"highway": "residential"}, (
            _ll(400, 200), _ll(600, 200),
        )),
        OsmLineFeature("way/c", {"highway": "residential"}, (
            _ll(400, 200), _ll(400, 400),
        )),
    ]
    info, nodes, ways, _ = _export(tmp_path, roads, world_size=1000)
    assert info["ways"] == 3
    assert info["merged_source_runs"] == 0
    shared = {nid for nid, xz in nodes.items() if xz == (400, 200)}
    assert len(shared) == 1
    assert all(shared.issubset(set(w[3])) for w in ways)


def test_width_and_surface_changes_prevent_merging(tmp_path):
    primary = OsmLineFeature(
        "way/1", {"highway": "residential", "width": "6.0"},
        (_ll(200, 200), _ll(400, 200)),
    )
    different_width = OsmLineFeature(
        "way/2", {"highway": "residential", "width": "8.0"},
        (_ll(400, 200), _ll(600, 200)),
    )
    info, _, _, _ = _export(
        tmp_path, [primary, different_width], world_size=1000,
    )
    assert info["ways"] == 2
    dirt = OsmLineFeature(
        "way/3", {"highway": "residential", "width": "6.0",
                 "surface": "dirt"},
        (_ll(400, 200), _ll(600, 200)),
    )
    info, _, _, _ = _export(tmp_path, [primary, dirt], world_size=1000)
    assert info["ways"] == 2


def test_near_coincident_endpoints_remain_separate(tmp_path):
    roads = [
        OsmLineFeature("way/1", {"highway": "residential"}, (
            _ll(200, 200), _ll(400, 200),
        )),
        OsmLineFeature("way/2", {"highway": "residential"}, (
            _ll(400.02, 200), _ll(600, 200),
        )),
    ]
    info, nodes, ways, _ = _export(tmp_path, roads, world_size=1000)
    assert info["ways"] == 2
    assert not set(ways[0][3]).intersection(ways[1][3])


def test_ring_ways_preserve_edges_without_self_joining(tmp_path):
    vertices = [(200, 200), (400, 200), (400, 400), (200, 400)]
    roads = [
        OsmLineFeature(f"way/{i}", {"highway": "residential"}, (
            _ll(*vertices[i]), _ll(*vertices[(i+1)%4]),
        ))
        for i in range(4)
    ]
    info, nodes, ways, _ = _export(tmp_path, roads, world_size=1000)
    assert info["ways"] >= 2
    assert info["merged_source_runs"] >= 1
    endpoints = [set((way[3][0], way[3][-1])) for way in ways]
    assert len(endpoints) == len(ways)


def test_isolated_engine_donor_model_sources_preserve_texture_fallback(tmp_path):
    road = _open_country_road()
    info, nodes, ways, _ = _export(
        tmp_path, [road], world_size=1000,
        surface_donors={"paved": r"o\road\sil25.p3d"},
    )
    assert info["surface_sources"]["paved"] == r"donor=o\road\sil25.p3d"
    assert ways[0][2] == r"donor=o\road\sil25.p3d"
    assert info["surface_sources"]["gravel"] == TEXTURES["gravel"]


def test_invalid_donor_is_rejected_without_writing_file(tmp_path):
    with pytest.raises(ValueError, match="Invalid CWR-CE road donor"):
        _export(
            tmp_path, [_open_country_road()], world_size=1000,
            surface_donors={"paved": r"outside map.p3d"},
        )



def test_long_open_approaches_get_a_larger_curve_radius(tmp_path):
    short_approaches = OsmLineFeature(
        "way/short", {"highway": "residential"},
        (_ll(200, 200), _ll(280, 200), _ll(280, 280)),
    )
    long_approaches = OsmLineFeature(
        "way/long", {"highway": "residential"},
        (_ll(200, 200), _ll(400, 200), _ll(400, 400)),
    )
    short_info, short_nodes, short_ways, _ = _export(
        tmp_path, [short_approaches], world_size=1000,
    )
    long_info, long_nodes, long_ways, _ = _export(
        tmp_path, [long_approaches], world_size=1000,
    )
    assert short_info["smoothed_rural_bends"] == 1
    assert long_info["smoothed_rural_bends"] == 1
    short_pts = _way_points(short_nodes, short_ways[0])
    long_pts = _way_points(long_nodes, long_ways[0])
    short_trim = 280 - short_pts[1][0]
    long_trim = 400 - long_pts[1][0]
    assert long_trim > short_trim + 5.0
    assert long_trim <= 52.0 + 0.01
    assert short_pts[0] == long_pts[0] == (200, 200)
    assert short_pts[-1] == (280, 280)
    assert long_pts[-1] == (400, 400)



def test_tunnel_approach_is_not_rounded_or_merged_across(tmp_path):
    tunnel = OsmLineFeature(
        "way/tunnel", {"highway": "tertiary", "tunnel": "yes"},
        (_ll(400, 200), _ll(530, 200)),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [_open_country_road(), tunnel], world_size=1000,
    )
    assert info["ways"] == 1
    assert info["smoothed_rural_bends"] == 0
    assert (400, 200) in _way_points(nodes, ways[0])



def test_native_ways_remove_unused_generated_road_p3ds_but_keep_other_assets(tmp_path):
    from cwr_worldgen.generator import _prune_unused_native_way_road_assets
    source = tmp_path / "source" / "world"
    infrastructure = source / "i"
    infrastructure.mkdir(parents=True)
    road_models = (
        "gravel25.p3d",
        "gravel_j3.p3d",
        "road_paved_w060_l0250.p3d",
        "road_j3_dirt_w060_h000_090_180.p3d",
    )
    retained_bridge = "br_span_w030_l0250.p3d"
    retained_other = "rock_rock_w020_l020.p3d"
    for name in (*road_models, retained_bridge, retained_other):
        (infrastructure / name).write_bytes(b"test")
    for name in ("g.paa", "gj.paa", "dt.paa", "pv.paa", "ge.paa",
                 "gravel-source-surfaces.txt"):
        (infrastructure / name).write_bytes(b"old")
    (infrastructure / "b.paa").write_bytes(b"needed for bridge")

    removed = _prune_unused_native_way_road_assets(source, "world", ())
    assert all(not (infrastructure / name).exists() for name in road_models)
    assert all(not (infrastructure / name).exists()
               for name in ("g.paa", "gj.paa", "dt.paa", "pv.paa",
                            "ge.paa", "gravel-source-surfaces.txt"))
    assert (infrastructure / retained_bridge).is_file()
    assert (infrastructure / retained_other).is_file()
    assert (infrastructure / "b.paa").is_file()
    assert "i/gravel25.p3d" in removed
    assert _prune_unused_native_way_road_assets(source, "world", ()) == ()


def test_native_ways_keep_generated_road_models_actually_placed_by_nonroads(tmp_path):
    from cwr_worldgen.generator import _prune_unused_native_way_road_assets
    source = tmp_path / "source" / "world"
    infrastructure = source / "i"
    infrastructure.mkdir(parents=True)
    (infrastructure / "gravel25.p3d").write_bytes(b"required")
    (infrastructure / "gravel12.p3d").write_bytes(b"unused")
    (infrastructure / "g.paa").write_bytes(b"keep material")
    removed = _prune_unused_native_way_road_assets(
        source, "world", (r"world\i\gravel25.p3d",),
    )
    assert (infrastructure / "gravel25.p3d").is_file()
    assert (infrastructure / "g.paa").is_file()
    assert not (infrastructure / "gravel12.p3d").exists()
    assert removed == ("i/gravel12.p3d",)


def test_native_pbo_layout_rejects_unreferenced_generated_road_p3ds(tmp_path):
    from cwr_worldgen.generator import _verify_single_world_pbo_layout
    from cwr_worldgen.pbo import pack_directory
    src = tmp_path / "world"
    (src / "i").mkdir(parents=True)
    (src / "config.cpp").write_text("class CfgWorlds {};")
    (src / "world.wrp").write_bytes(b"4WVR" + b"\0" * 24)
    (src / "world.ways").write_text("node 1 1 1\nnode 2 2 2\n"
                                    r"way 1 6 o\road\sil_new.paa 1 2" + "\n")
    (src / "i" / "gravel25.p3d").write_bytes(b"old unused road")
    (src / "i" / "br_span_w030_l0250.p3d").write_bytes(b"needed bridge")
    pbo = tmp_path / "world.pbo"
    pack_directory(src, pbo, backend="python")
    with pytest.raises(RuntimeError, match="unused generated road P3Ds"):
        _verify_single_world_pbo_layout(
            pbo, "world", None, include_ways=True,
        )
    (src / "i" / "gravel25.p3d").unlink()
    pack_directory(src, pbo, backend="python")
    assert _verify_single_world_pbo_layout(
        pbo, "world", None, include_ways=True,
    )["verified"]



def test_long_rural_straights_gain_smooth_wandering_centerlines(tmp_path):
    road = OsmLineFeature(
        "way/long_track", {"highway": "track", "surface": "gravel"},
        (_ll(100, 400), _ll(600, 400)),
    )
    info, nodes, ways, first_bytes = _export(
        tmp_path, [road], world_size=1000,
    )
    points = _way_points(nodes, ways[0])
    assert info["softened_rural_straights"] == 1
    assert info["smoothed_rural_bends"] == 0
    assert points[0] == (100, 400)
    assert points[-1] == (600, 400)
    offsets = [z - 400 for _, z in points[1:-1]]
    assert max(offsets) > 0.7
    assert min(offsets) < -0.7
    assert max(abs(v) for v in offsets) <= 4.0
    # No jagged line segments and no accidental duplicate nodes.
    directions = []
    for p, q in zip(points, points[1:]):
        distance = __import__("math").dist(p, q)
        assert distance > 0
        directions.append(((q[0]-p[0])/distance, (q[1]-p[1])/distance))
    assert min(
        a[0]*b[0] + a[1]*b[1] for a, b in zip(directions, directions[1:])
    ) > 0.97
    assert _export(tmp_path, [road], world_size=1000)[3] == first_bytes


def test_reversing_rural_road_keeps_the_exact_same_wandering_geometry(tmp_path):
    vertices = (_ll(100, 430), _ll(600, 430))
    fwd = OsmLineFeature("way/1", {"highway": "track"}, vertices)
    rev = OsmLineFeature("way/1", {"highway": "track"}, vertices[::-1])
    _, nodes, ways, _ = _export(tmp_path, [fwd], world_size=1000)
    a = _way_points(nodes, ways[0])
    _, nodes, ways, _ = _export(tmp_path, [rev], world_size=1000)
    b = _way_points(nodes, ways[0])
    assert a == list(reversed(b))


def test_short_or_major_engineered_roads_remain_straight(tmp_path):
    for highway, length in (
        ("track", 120), ("primary", 500), ("trunk", 500),
        ("motorway", 500),
    ):
        road = OsmLineFeature(
            "way/road", {"highway": highway},
            (_ll(100, 400), _ll(100 + length, 400)),
        )
        info, nodes, ways, _ = _export(
            tmp_path, [road], world_size=1000,
        )
        assert info["softened_rural_straights"] == 0
        assert _way_points(nodes, ways[0]) == [
            (100, 400), (100 + length, 400),
        ]


def test_nearly_collinear_osm_vertices_form_one_natural_curve(tmp_path):
    road = OsmLineFeature(
        "way/digitized", {"highway": "track"},
        tuple(_ll(x, 450) for x in (100, 170, 240, 310, 380, 450, 520)),
    )
    info, nodes, ways, _ = _export(tmp_path, [road], world_size=1000)
    points = _way_points(nodes, ways[0])
    assert info["softened_rural_straights"] == 1
    assert points[0] == (100, 450)
    assert points[-1] == (520, 450)
    assert len(points) > 6
    assert (240, 450) not in points


def test_rural_flow_stays_out_of_settlements_and_nearby_buildings(tmp_path):
    road = OsmLineFeature(
        "way/long", {"highway": "track"},
        (_ll(100, 400), _ll(600, 400)),
    )
    village = OsmPointFeature(
        "node/village", {"place": "village"}, _ll(350, 400),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [road], world_size=1000, places=(village,),
    )
    assert info["softened_rural_straights"] == 0
    assert _way_points(nodes, ways[0]) == [(100, 400), (600, 400)]
    building = OsmPolygonFeature(
        "building/b", {"building": "yes"},
        (GeoPolygon(tuple(_ll(x, z) for x, z in (
            (350, 403), (360, 403), (360, 413), (350, 413), (350, 403),
        ))),),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [road], world_size=1000,
        building_polygons=(building,),
    )
    assert info["softened_rural_straights"] == 0
    assert _way_points(nodes, ways[0]) == [(100, 400), (600, 400)]


def test_parallel_road_and_watercourse_protect_open_straight(tmp_path):
    road = OsmLineFeature(
        "way/main", {"highway": "track"}, (_ll(100, 400), _ll(600, 400)),
    )
    parallel = OsmLineFeature(
        "way/other", {"highway": "track"}, (_ll(100, 407), _ll(600, 407)),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [road, parallel], world_size=1000,
    )
    assert info["softened_rural_straights"] == 0
    assert all(len(way[3]) == 2 for way in ways)
    stream = OsmLineFeature(
        "way/stream", {"waterway": "stream"},
        (_ll(250, 405), _ll(450, 405)),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [road], world_size=1000, watercourses=(stream,),
    )
    assert info["softened_rural_straights"] == 0


def test_flow_keeps_bridges_tunnels_and_junction_points_fixed(tmp_path):
    for tags in ({"bridge": "yes"}, {"tunnel": "yes"}):
        main = OsmLineFeature(
            "way/main", {"highway": "track"},
            (_ll(100, 400), _ll(500, 400)),
        )
        omitted = OsmLineFeature(
            "way/omitted", {"highway": "track", **tags},
            (_ll(500, 400), _ll(650, 400)),
        )
        info, nodes, ways, _ = _export(
            tmp_path, [main, omitted], world_size=1000,
        )
        assert info["softened_rural_straights"] == 0
        assert _way_points(nodes, ways[0]) == [(100, 400), (500, 400)]



def _stock_bridge(x=400.0, z=400.0, heading=90.0):
    """A stock 50.19-m bridge with real measured deck endpoints."""
    return SimpleNamespace(
        model_path=r"O\Hous\most_stred30.p3d",
        x=x, y=-7.0, z=z,
        heading_degrees=heading, pitch_degrees=0.0,
    )


def test_bridge_tagged_osm_way_restores_both_dry_approaches(tmp_path):
    # A typical OSM bridge=yes feature is 600m long, even though the
    # actual stock Roadway deck over water is only 50.19m.
    bridge_tagged = OsmLineFeature(
        "way/bridge", {"highway": "tertiary", "bridge": "yes"},
        (_ll(100, 400), _ll(700, 400)),
    )
    info, nodes, ways, data = _export(
        tmp_path, [bridge_tagged], world_size=1000,
        bridge_objects=(_stock_bridge(),),
    )
    assert info["stock_bridge_modules"] == 1
    assert info["excluded_bridge_features"] == 1
    assert info["restored_bridge_approach_ways"] == 2
    assert info["unmatched_bridge_features"] == 0
    assert len(ways) == 2
    first, second = [_way_points(nodes, way) for way in ways]
    assert first[0] == (100, 400)
    assert second[-1] == (700, 400)
    # The stock bridge deck joins these ends near x=375 and x=425.
    assert 374 < first[-1][0] < 378
    assert 422 < second[0][0] < 426
    assert first[-1][1] == second[0][1] == 400
    assert all(not 380 < x < 420 for x, z in first + second)
    assert _export(
        tmp_path, [bridge_tagged], world_size=1000,
        bridge_objects=(_stock_bridge(),),
    )[3] == data


def test_bridge_endpoints_stay_connected_with_reversed_osm_geometry(tmp_path):
    points = (_ll(100, 400), _ll(700, 400))
    for source in (points, points[::-1]):
        feature = OsmLineFeature(
            "way/bridge", {"highway": "tertiary", "bridge": "yes"}, source,
        )
        info, nodes, ways, _ = _export(
            tmp_path, [feature], world_size=1000,
            bridge_objects=(_stock_bridge(),),
        )
        assert len(ways) == 2
        terminal_points = [
            pts
            for way in ways
            for pts in (_way_points(nodes, way)[0], _way_points(nodes, way)[-1])
        ]
        assert len([p for p in terminal_points if 374 < p[0] < 378]) == 1
        assert len([p for p in terminal_points if 422 < p[0] < 426]) == 1


def test_bridge_way_with_multiple_osm_vertices_retains_road_on_dry_banks(tmp_path):
    feature = OsmLineFeature(
        "way/bridge", {"highway": "residential", "bridge": "yes"},
        tuple(_ll(x, 400) for x in (100, 200, 300, 400, 500, 600, 700)),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [feature], world_size=1000,
        bridge_objects=(_stock_bridge(),),
    )
    assert info["restored_bridge_approach_ways"] == 2
    assert len(ways) == 2
    left = _way_points(nodes, ways[0])
    right = _way_points(nodes, ways[1])
    assert (200, 400) in left and (300, 400) in left
    assert (500, 400) in right and (600, 400) in right


def test_unrelated_bridge_or_missing_stock_deck_does_not_create_water_road(tmp_path):
    feature = OsmLineFeature(
        "way/bridge", {"highway": "tertiary", "bridge": "yes"},
        (_ll(100, 400), _ll(700, 400)),
    )
    for decks in ((), (_stock_bridge(z=500.0),),
                  (_stock_bridge(heading=0.0),)):
        info, nodes, ways, _ = _export(
            tmp_path, [feature], world_size=1000, bridge_objects=decks,
        )
        assert info["restored_bridge_approach_ways"] == 0
        assert len(ways) == 0



def test_two_connected_bridge_modules_have_only_two_bank_roads(tmp_path):
    feature = OsmLineFeature(
        "way/bridge", {"highway": "tertiary", "bridge": "yes"},
        (_ll(100, 400), _ll(800, 400)),
    )
    spacing = 50.190284729003906
    bridges = (_stock_bridge(x=400), _stock_bridge(x=400 + spacing))
    info, nodes, ways, _ = _export(
        tmp_path, [feature], world_size=1000, bridge_objects=bridges,
    )
    assert info["stock_bridge_modules"] == 2
    assert info["restored_bridge_approach_ways"] == 2
    assert len(ways) == 2
    left, right = (_way_points(nodes, ways[0]),
                   _way_points(nodes, ways[1]))
    assert left[0] == (100, 400)
    assert right[-1] == (800, 400)
    assert 374 < left[-1][0] < 378
    assert 472 < right[0][0] < 477
    # No third native road segment is added over the seam of stock modules.
    assert all(not 385 < x < 465 for x, z in left + right)


def test_alternative_bridge_tags_receive_bank_connections(tmp_path):
    for bridge_tags in ({"man_made": "bridge"}, {"special": "bridge"}):
        feature = OsmLineFeature(
            "way/bridge", {"highway": "tertiary", **bridge_tags},
            (_ll(100, 400), _ll(700, 400)),
        )
        info, nodes, ways, _ = _export(
            tmp_path, [feature], world_size=1000,
            bridge_objects=(_stock_bridge(),),
        )
        assert info["restored_bridge_approach_ways"] == 2
        assert len(ways) == 2



def test_opt_in_donor_width_overrides_osm_width_and_merges_compatible_runs(tmp_path):
    a = OsmLineFeature(
        "way/a", {"highway": "residential", "width": "6"},
        (_ll(100, 300), _ll(200, 300)),
    )
    b = OsmLineFeature(
        "way/b", {"highway": "residential", "width": "8"},
        (_ll(200, 300), _ll(300, 300)),
    )
    ordinary, _, ordinary_ways, _ = _export(
        tmp_path, [a, b], world_size=1000,
    )
    assert ordinary["ways"] == 2
    assert {way[1] for way in ordinary_ways} == {6.0, 8.0}

    donor, _, donor_ways, data = _export(
        tmp_path, [a, b], world_size=1000,
        surface_donors={"paved": r"mod\roads\wide25.p3d"},
        surface_widths={"paved": 7.425},
        require_donor_widths=True,
    )
    assert donor["keep_donor_road_width"] is True
    assert donor["surface_widths"] == {"paved": 7.425}
    assert donor["merged_source_runs"] == 1
    assert donor["ways"] == 1
    assert donor_ways[0][1] == 7.425
    assert donor_ways[0][2] == r"donor=mod\roads\wide25.p3d"
    assert _export(
        tmp_path, [b, a], world_size=1000,
        surface_donors={"paved": r"mod\roads\wide25.p3d"},
        surface_widths={"paved": 7.425},
        require_donor_widths=True,
    )[3] == data


def test_native_ways_require_measured_width_for_only_used_surfaces(tmp_path):
    paved = OsmLineFeature(
        "way/p", {"highway": "residential", "width": "11"},
        (_ll(100, 250), _ll(230, 250)),
    )
    info, _, ways, _ = _export(
        tmp_path, [paved], world_size=1000,
        surface_widths={"paved": 5.375},
        require_donor_widths=True,
    )
    assert info["ways"] == 1
    assert ways[0][1] == 5.375
    gravel = OsmLineFeature(
        "way/g", {"highway": "track", "surface": "gravel"},
        (_ll(300, 250), _ll(420, 250)),
    )
    with pytest.raises(ValueError, match="gravel"):
        _export(
            tmp_path, [paved, gravel], world_size=1000,
            surface_widths={"paved": 5.375},
            require_donor_widths=True,
        )


@pytest.mark.parametrize("value", [0.0, -3.0, 51.0, float("nan"), float("inf"), "n/a"])
def test_native_ways_reject_invalid_donor_width(tmp_path, value):
    with pytest.raises(ValueError, match="donor width"):
        _export(
            tmp_path, [_open_country_road()], world_size=1000,
            surface_widths={"paved": value},
            require_donor_widths=True,
        )



def test_donor_width_measurement_prefers_actual_p3d_connector_span(monkeypatch):
    from cwr_worldgen import generator
    monkeypatch.setattr(generator, "read_asset_record_bytes", lambda record: b"P3D")
    monkeypatch.setattr(generator, "inspect_visual_model_dimensions", lambda data:
        SimpleNamespace(connector_width_metres=7.375, width_metres=9.8))
    assert generator._native_way_donor_width(object(), "paved") == 7.375

    monkeypatch.setattr(generator, "inspect_visual_model_dimensions", lambda data:
        SimpleNamespace(connector_width_metres=None, width_metres=4.125))
    assert generator._native_way_donor_width(object(), "gravel") == 4.125


def test_donor_width_measurement_rejects_unreadable_or_invalid_p3ds(monkeypatch):
    from cwr_worldgen import generator
    monkeypatch.setattr(generator, "read_asset_record_bytes", lambda record: b"P3D")
    monkeypatch.setattr(generator, "inspect_visual_model_dimensions", lambda data:
        SimpleNamespace(connector_width_metres=90.0, width_metres=9.0))
    with pytest.raises(ValueError, match="Invalid paved native-way P3D donor width"):
        generator._native_way_donor_width(object(), "paved")

    def missing(record):
        raise OSError("corrupt PBO")
    monkeypatch.setattr(generator, "read_asset_record_bytes", missing)
    with pytest.raises(ValueError, match="Could not measure dirt"):
        generator._native_way_donor_width(object(), "dirt")



def test_sharp_rural_turn_around_145_degrees_is_rounded_when_open(tmp_path):
    import math
    from shapely.geometry import LineString, Point

    b = (500, 350)
    c = (
        b[0] + 220 * math.cos(math.radians(145)),
        b[1] + 220 * math.sin(math.radians(145)),
    )
    road = OsmLineFeature(
        "way/sharp", {"highway": "tertiary"},
        (_ll(200, 350), _ll(*b), _ll(*c)),
    )
    info, nodes, ways, original = _export(
        tmp_path, [road], world_size=1000,
    )
    points = _way_points(nodes, ways[0])
    assert info["smoothed_rural_bends"] == 1
    assert info["ways"] == 1
    assert points[0] == (200, 350)
    assert points[-1] == pytest.approx(c, abs=0.01)
    assert b not in points
    assert LineString(points).is_simple
    source = LineString(((200, 350), b, c))
    assert max(source.distance(Point(p)) for p in points) <= 14.01
    # There is no abrupt 145-degree turn between adjacent road segments.
    heading_changes = []
    for p, q, r in zip(points, points[1:], points[2:]):
        u = (q[0] - p[0], q[1] - p[1])
        v = (r[0] - q[0], r[1] - q[1])
        dot = (u[0] * v[0] + u[1] * v[1]) / math.dist(p, q) / math.dist(q, r)
        heading_changes.append(dot)
    assert min(heading_changes) > 0.92
    assert _export(tmp_path, [road], world_size=1000)[3] == original


def test_short_but_usable_country_turn_is_rounded_without_moving_ends(tmp_path):
    road = OsmLineFeature(
        "way/local", {"highway": "residential"},
        (_ll(300, 300), _ll(322, 300), _ll(322, 322)),
    )
    info, nodes, ways, _ = _export(tmp_path, [road], world_size=1000)
    points = _way_points(nodes, ways[0])
    assert info["smoothed_rural_bends"] == 1
    assert points[0] == (300, 300)
    assert points[-1] == (322, 322)
    assert (322, 300) not in points


def test_sharp_turn_with_adjacent_building_respects_clearance(tmp_path):
    import math
    from shapely.geometry import LineString, Polygon

    b = (500, 350)
    c = (
        b[0] + 220 * math.cos(math.radians(145)),
        b[1] + 220 * math.sin(math.radians(145)),
    )
    road = OsmLineFeature(
        "way/sharp", {"highway": "tertiary"},
        (_ll(200, 350), _ll(*b), _ll(*c)),
    )
    # A building on the inside of the bend must never be covered by the
    # exported road ribbon, even when the adaptive fitter can still find a
    # broader radius that goes around it.
    building = OsmPolygonFeature(
        "building/near_bend", {"building": "yes"},
        (GeoPolygon(tuple(_ll(x, z) for x, z in (
            (488, 351), (500, 351), (500, 360),
            (488, 360), (488, 351),
        ))),),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [road], world_size=1000,
        building_polygons=(building,),
    )
    path = LineString(_way_points(nodes, ways[0]))
    footprint = path.buffer(3.0 + 3.0)
    building_shape = Polygon(tuple((x, z) for x, z in (
        (488, 351), (500, 351), (500, 360),
        (488, 360), (488, 351),
    ))).buffer(4.0)
    if info["smoothed_rural_bends"]:
        assert not footprint.intersects(building_shape)
    else:
        assert (500, 350) in _way_points(nodes, ways[0])



def test_native_ways_reuse_p3d_width_aware_repair_for_short_osm_zigzag(tmp_path):
    """The P3D fitter's curvature check removes a tiny impossible 45° kink."""
    from shapely.geometry import LineString

    road = OsmLineFeature(
        "way/tiny_kink", {"highway": "residential", "width": "10"},
        tuple(_ll(*point) for point in (
            (200, 300), (240, 300), (243, 303), (283, 303),
        )),
    )
    info, nodes, ways, contents = _export(tmp_path, [road], world_size=1000)
    points = _way_points(nodes, ways[0])
    assert info["repaired_rural_kinks"] == 1
    assert info["ways"] == 1
    assert points[0] == (200, 300)
    assert points[-1] == (283, 303)
    assert (240, 300) not in points
    original = LineString((
        (200, 300), (240, 300), (243, 303), (283, 303),
    ))
    assert LineString(points).hausdorff_distance(original) <= 4.5
    assert LineString(points).is_simple
    assert _export(tmp_path, [road], world_size=1000)[3] == contents


def test_p3d_kink_repair_adapts_to_optional_native_donor_width(tmp_path):
    road = OsmLineFeature(
        "way/kink", {"highway": "residential", "width": "6"},
        tuple(_ll(*point) for point in (
            (200, 300), (240, 300), (243, 303), (283, 303),
        )),
    )
    narrow, nodes, ways, _ = _export(tmp_path, [road], world_size=1000)
    assert narrow["repaired_rural_kinks"] == 0
    assert _way_points(nodes, ways[0])[0] == (200, 300)
    assert _way_points(nodes, ways[0])[-1] == (283, 303)

    wide, nodes, ways, _ = _export(
        tmp_path, [road], world_size=1000,
        surface_widths={"paved": 10.0}, require_donor_widths=True,
    )
    assert wide["repaired_rural_kinks"] == 1
    assert (240, 300) not in _way_points(nodes, ways[0])
    assert ways[0][1] == 10.0


def test_p3d_kink_repair_does_not_remove_connected_junction_node(tmp_path):
    main = OsmLineFeature(
        "way/main", {"highway": "residential", "width": "10"},
        tuple(_ll(*point) for point in (
            (200, 300), (240, 300), (243, 303), (283, 303),
        )),
    )
    spur = OsmLineFeature(
        "way/spur", {"highway": "residential"},
        (_ll(240, 300), _ll(240, 400)),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [main, spur], world_size=1000,
    )
    assert info["repaired_rural_kinks"] == 0
    main_nodes = next(_way_points(nodes, way) for way in ways if way[1] == 10.0)
    assert (240, 300) in main_nodes
    assert sum((240, 300) in _way_points(nodes, way) for way in ways) == 2


def test_p3d_kink_repair_respects_obstacles_and_urban_exclusion(tmp_path):
    road = OsmLineFeature(
        "way/obstructed", {"highway": "residential", "width": "10"},
        tuple(_ll(*point) for point in (
            (200, 300), (240, 300), (243, 303), (283, 303),
        )),
    )
    building = OsmPolygonFeature(
        "building/1", {"building": "yes"},
        (GeoPolygon(tuple(_ll(*point) for point in (
            (239, 300), (244, 300), (244, 307), (239, 307), (239, 300),
        ))),),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [road], world_size=1000,
        building_polygons=(building,),
    )
    assert info["repaired_rural_kinks"] == 0
    pts = _way_points(nodes, ways[0])
    assert pts[0] == (200, 300) and pts[-1] == (283, 303)

    village = OsmPointFeature(
        "node/nearby", {"place": "village"}, _ll(240, 300),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [road], world_size=1000,
        places=(village,),
    )
    assert info["repaired_rural_kinks"] == 0
    pts = _way_points(nodes, ways[0])
    assert pts[0] == (200, 300) and pts[-1] == (283, 303)


def test_p3d_kink_repair_does_not_flatten_true_hairpin(tmp_path):
    road = OsmLineFeature(
        "way/hairpin", {"highway": "residential", "width": "10"},
        tuple(_ll(*point) for point in (
            (200, 300), (240, 300), (235, 306), (230, 325),
        )),
    )
    info, nodes, ways, _ = _export(tmp_path, [road], world_size=1000)
    from shapely.geometry import LineString
    assert info["repaired_rural_kinks"] == 0
    pts = _way_points(nodes, ways[0])
    source = LineString((
        (200, 300), (240, 300), (235, 306), (230, 325),
    ))
    assert pts[0] == (200, 300) and pts[-1] == (230, 325)
    assert max(x for x, _ in pts) > 233
    assert LineString(pts).is_simple
    assert LineString(pts).hausdorff_distance(source) <= 14.01



def test_short_multi_corner_s_bend_fairs_without_sharp_kinks(tmp_path):
    """Two neighboring digitized corners should become a flowing transition."""
    import math
    from shapely.geometry import LineString

    original = (
        (200, 350), (268, 350), (274, 354),
        (280, 357), (356, 357),
    )
    road = OsmLineFeature(
        "way/short_s", {"highway": "residential", "width": "6"},
        tuple(_ll(*p) for p in original),
    )
    info, nodes, ways, first = _export(
        tmp_path, [road], world_size=1000,
    )
    result = _way_points(nodes, ways[0])
    assert info["ways"] == 1
    assert info["repaired_rural_kinks"] == 0
    assert info["eased_residual_turns"] >= 1
    assert result[0] == original[0] and result[-1] == original[-1]
    assert len(result) > len(original)
    assert LineString(result).is_simple
    assert LineString(result).hausdorff_distance(LineString(original)) <= 14.01

    def largest_turn(points):
        values = []
        for a, b, c in zip(points, points[1:], points[2:]):
            u, v = (b[0]-a[0], b[1]-a[1]), (c[0]-b[0], c[1]-b[1])
            dot = (u[0]*v[0] + u[1]*v[1])/math.hypot(*u)/math.hypot(*v)
            values.append(math.degrees(math.acos(max(-1.0, min(1.0, dot)))))
        return max(values)

    assert largest_turn(result) < largest_turn(original) * 0.65
    assert _export(tmp_path, [road], world_size=1000)[3] == first


def test_residual_smoothing_keeps_t_junction_connections(tmp_path):
    main = OsmLineFeature(
        "way/short_s", {"highway": "residential"},
        tuple(_ll(*p) for p in (
            (200, 350), (268, 350), (274, 354),
            (280, 357), (356, 357), (450, 357),
        )),
    )
    spur = OsmLineFeature(
        "way/side", {"highway": "residential"},
        (_ll(356, 357), _ll(356, 440)),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [main, spur], world_size=1000,
    )
    assert info["ways"] == 2
    assert sum((356, 357) in _way_points(nodes, way) for way in ways) == 2
    assert all(
        len(way[3]) == len(set(way[3]))
        for way in ways
    )


def test_residual_smoothing_is_blocked_by_nearby_building(tmp_path):
    from shapely.geometry import LineString, Polygon
    original = (
        (200, 350), (268, 350), (274, 354),
        (280, 357), (356, 357),
    )
    road = OsmLineFeature(
        "way/short_s", {"highway": "residential", "width": "6"},
        tuple(_ll(*p) for p in original),
    )
    footprint = ((271, 344), (286, 344), (286, 364),
                 (271, 364), (271, 344))
    building = OsmPolygonFeature(
        "building/near", {"building": "yes"},
        (GeoPolygon(tuple(_ll(*p) for p in footprint)),),
    )
    info, nodes, ways, _ = _export(
        tmp_path, [road], world_size=1000,
        building_polygons=(building,),
    )
    assert info["eased_residual_turns"] == 0
    assert LineString(_way_points(nodes, ways[0])).is_simple
