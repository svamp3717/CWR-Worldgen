# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from types import SimpleNamespace

import pytest

from cwr_worldgen.model import validate_world_identity
from cwr_worldgen.osm import BboxProjection, OsmLineFeature
from cwr_worldgen.road_ways import is_osm_ways_profile, write_osm_road_ways


TEXTURES = {
    "paved": r"data\silnice.pac",
    "dirt": r"data\cesta.pac",
    "gravel": r"data\asfaltka.pac",
}


def _export(tmp_path, roads, *, include_minor=True):
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 100.0)
    path = tmp_path / "test_world.ways"
    info = write_osm_road_ways(
        path, SimpleNamespace(roads=tuple(roads)), projection,
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
