"""RAP config.bin regression coverage for CWA LandGrid and Names."""
from __future__ import annotations

from pathlib import Path
import struct

from cwr_island_studio.world_metadata import (
    load_loose_world_metadata, parse_world_config,
)
from cwr_island_studio.pbo_worlds import PboWorldArchive
from test_island_studio_world_metadata import pbo, wrp


def _rap(source: dict) -> bytes:
    """Independent RAP fixture encoder: root body, offset classes, typed values."""
    output = bytearray(b"\x00raP" + struct.pack("<III", 0, 8, 0))

    def string(value: str) -> bytes:
        return value.encode("latin-1") + b"\0"

    def array(values: list) -> None:
        assert len(values) < 128
        output.append(len(values))
        for item in values:
            if isinstance(item, list):
                output.append(3)
                array(item)
            elif isinstance(item, str):
                output.append(0)
                output.extend(string(item))
            elif isinstance(item, float):
                output.append(1)
                output.extend(struct.pack("<f", item))
            else:
                output.append(2)
                output.extend(struct.pack("<i", item))

    def body(items: dict, base: str = "") -> None:
        output.extend(string(base))
        assert len(items) < 128
        output.append(len(items))
        refs: list[tuple[int, object]] = []
        for name, value in items.items():
            if isinstance(value, tuple) and value[0] == "class":
                output.append(0)
                output.extend(string(name))
                pos = len(output)
                output.extend(b"\0" * 4)
                refs.append((pos, value))
            elif isinstance(value, list):
                output.append(2)
                output.extend(string(name))
                array(value)
            else:
                output.append(1)
                if isinstance(value, str):
                    output.append(0)
                    output.extend(string(name))
                    output.extend(string(value))
                elif isinstance(value, float):
                    output.append(1)
                    output.extend(string(name))
                    output.extend(struct.pack("<f", value))
                else:
                    output.append(2)
                    output.extend(string(name))
                    output.extend(struct.pack("<i", value))
        for where, item in refs:
            struct.pack_into("<I", output, where, len(output))
            body(item[1], item[2] if len(item) > 2 else "")

    body(source)
    struct.pack_into("<I", output, 12, len(output))
    return bytes(output)


def _world_data() -> bytes:
    return _rap({
        "CfgPatches": ("class", {
            "test": ("class", {"noise": [1, "x", [0.1, 1]]}),
        }),
        "CfgWorlds": ("class", {
            "BaseLand": ("class", {"LandGrid": 25}),
            "island": ("class", {
                "Names": ("class", {
                    "Harbour": ("class", {
                        "name": "Small Harbour", "position": [10.5, 16.0, 25],
                    }),
                    "Village": ("class", {
                        "name": "Woodbridge", "position": [105, 81.25],
                    }),
                }),
            }, "BaseLand"),
            "other": ("class", {"LandGrid": 12.5}),
        }),
        "CfgVehicles": ("class", {"Example": ("class", {"foo": "bar"})}),
    })


def test_binary_cfgworlds_inheritance_and_locations() -> None:
    data = _world_data()
    assert data[:4] == b"\x00raP"
    result = parse_world_config(data, "island", label="config.bin")
    assert result.land_grid == 25.0
    assert result.source == "config.bin"
    assert [(p.name, p.x, p.z) for p in result.locations] == [
        ("Small Harbour", 10.5, 16.0), ("Woodbridge", 105.0, 81.25),
    ]
    other = parse_world_config(data, "other", label="config.bin")
    assert other.land_grid == 12.5 and other.locations == ()
    assert parse_world_config(data, "unknown").source == ""


def test_loose_world_reads_binary_config_and_prefers_cpp(tmp_path: Path) -> None:
    world = tmp_path / "island.wrp"
    world.write_bytes(wrp())
    binary = tmp_path / "config.bin"
    binary.write_bytes(_world_data())
    meta = load_loose_world_metadata(world)
    assert meta.land_grid == 25.0
    assert meta.locations[0].name == "Small Harbour"
    (tmp_path / "config.cpp").write_text(
        "class CfgWorlds { class island { landGrid=10; }; };",
    )
    assert load_loose_world_metadata(world).land_grid == 10
    binary.write_bytes(b"\x00raP" + bytes(14))
    (tmp_path / "config.cpp").unlink()
    assert load_loose_world_metadata(world).source == ""


def test_pbo_world_uses_rap_config_bin_and_preserves_source(tmp_path: Path) -> None:
    archive_file = tmp_path / "islands.pbo"
    pbo(archive_file, [
        ("config.bin", _world_data()),
        ("island.wrp", wrp()),
    ])
    with PboWorldArchive(archive_file) as archive:
        with archive.open_world(
            "islands\\island.wrp",
        ) as world:
            assert world.cell_size == 25.0
            assert len(world.map_locations) == 2
            assert "config.bin" in world.grid_source
            assert world.read_only_forced is True


def test_corrupt_rap_offsets_and_oversized_strings_do_not_crash() -> None:
    from cwr_island_studio.rap_config import parse_rap_world
    data = bytearray(_world_data())
    # Root CfgWorlds reference is valid, but an intentionally corrupt offset
    # must be rejected rather than seeking arbitrary archive bytes.
    at = data.index(b"CfgWorlds\0") + len(b"CfgWorlds\0")
    struct.pack_into("<I", data, at, 0x7FFFFFFF)
    assert parse_world_config(bytes(data), "island").source == ""
    try:
        parse_rap_world(bytes(data), "island")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid RAP pointer must fail")
