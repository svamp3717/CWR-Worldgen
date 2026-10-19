"""LandGrid and optional map-name overlays from OFP/CWA config.cpp."""
from __future__ import annotations

import io
from pathlib import Path
import struct

import pytest

from cwr_island_studio import pbo_worlds
from cwr_island_studio.oprw import open_island_document
from cwr_island_studio.pbo_worlds import PboWorldArchive
from cwr_island_studio.world_metadata import (
    MapLocation, load_loose_world_metadata, parse_world_config,
)


CONFIG = r"""
class CfgPatches { class world_addon { units[] = {}; }; };
class CfgWorlds
{
    class DefaultWorld;
    class island_a: DefaultWorld
    {
        landGrid = 25;
        class Names
        {
            class Place1 { name = "Old Town"; position[] = {45.0, 62.5, 25}; };
            class Place2 { name = "St. John's"; position[] = {90, 120, 25}; };
        };
    };
    class island_b: DefaultWorld
    {
        landGrid = 12.5;
        class Names { class IslandPort { name = "Harbour"; position[] = {100, 60, 25}; }; };
    };
};
"""


def wrp() -> bytes:
    return struct.pack("<4sii", b"4WVR", 4, 4) + bytes(4 * 4 * 4 + 512 * 32 + 128)


def pbo(path: Path, members: list[tuple[str, bytes]]) -> None:
    stream = io.BytesIO()
    for name, content in members:
        stream.write(name.encode("ascii") + b"\0")
        stream.write(struct.pack("<IIIII", 0, 0, 0, 0, len(content)))
    stream.write(b"\0" + struct.pack("<IIIII", 0, 0, 0, 0, 0))
    for _, content in members:
        stream.write(content)
    path.write_bytes(stream.getvalue())


def test_world_config_extracts_world_specific_grid_and_labels() -> None:
    a = parse_world_config(CONFIG, "island_a")
    assert a.land_grid == 25.0
    assert a.locations == (
        MapLocation("Old Town", 45.0, 62.5),
        MapLocation("St. John's", 90.0, 120.0),
    )
    b = parse_world_config(CONFIG, "island_b")
    assert b.land_grid == 12.5
    assert [label.name for label in b.locations] == ["Harbour"]
    assert parse_world_config(CONFIG, "not_here").land_grid is None


def test_comments_and_unrelated_world_names_are_ignored() -> None:
    payload = (r"""
class CfgWorlds {
    class some_world {
        // landGrid=99;
        /* class Names { class Fake { name="Fake"; position[]={0,0}; }; }; */
        landGrid = 50;
        class Names { class Marker { name="Real"; position[]={50,100,0}; }; };
    };
};
""")
    actual = parse_world_config(payload, "some_world")
    assert actual.land_grid == 50
    assert actual.locations == (MapLocation("Real", 50, 100),)
    assert parse_world_config("class CfgVehicles { class some_world { landGrid=3; }; };", "some_world").source == ""


def test_loose_wrp_reads_sibling_config_without_altering_source(tmp_path: Path) -> None:
    world = tmp_path / "island_a.wrp"
    world.write_bytes(wrp())
    (tmp_path / "config.cpp").write_text(CONFIG)
    meta = load_loose_world_metadata(world)
    assert meta.land_grid == 25
    assert len(meta.locations) == 2
    with open_island_document(world, cell_size=meta.land_grid) as doc:
        assert doc.world_width == 100
    assert world.read_bytes() == wrp()


def test_second_world_in_same_pbo_reuses_directory_and_loads_own_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "islands.pbo"
    pbo(source, [
        ("config.cpp", CONFIG.encode("ascii")),
        (r"maps\island_a.wrp", wrp()),
        (r"maps\island_b.wrp", wrp()),
    ])
    pbo_worlds._DIRECTORY_CACHE.clear()
    calls = []
    scanner = pbo_worlds._scan

    def counted_scan(*args, **kwargs):
        calls.append(args[0])
        return scanner(*args, **kwargs)

    monkeypatch.setattr(pbo_worlds, "_scan", counted_scan)
    first = PboWorldArchive(source)
    one = first.open_world(r"islands\maps\island_a.wrp")
    try:
        assert one.cell_size == 25
        assert len(one.map_locations) == 2
        second = PboWorldArchive(source)
        two = second.open_world(r"islands\maps\island_b.wrp")
        try:
            assert two.cell_size == 12.5
            assert [n.name for n in two.map_locations] == ["Harbour"]
            assert len(calls) == 1, "reopening the same PBO must not rescan the header"
            assert one.path.exists() and two.path.exists()
        finally:
            two.close()
    finally:
        one.close()
        pbo_worlds._DIRECTORY_CACHE.clear()


def test_manual_grid_override_keeps_place_names(tmp_path: Path) -> None:
    source = tmp_path / "islands.pbo"
    pbo(source, [
        ("config.cpp", CONFIG.encode("ascii")),
        ("island_a.wrp", wrp()),
    ])
    with PboWorldArchive(source) as archive:
        with archive.open_world(
            r"islands\island_a.wrp", cell_size=60, auto_grid=False,
        ) as doc:
            assert doc.cell_size == 60
            assert doc.map_locations[0].name == "Old Town"
