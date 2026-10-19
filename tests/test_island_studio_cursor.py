"""Live terrain and texture name inspector tests, including sparse brush edits."""
from __future__ import annotations

from pathlib import Path
import struct

from cwr_island_studio.core import IslandDocument
from cwr_island_studio.cursor_info import cursor_readout


def _world(path: Path) -> None:
    # 4 x 4 terrain vertices, 25 metres apart; no placed objects.
    terrain = [0, 8, -4, 2] + [4] * 12
    slots = [0, 1, 2, 0] + [1] * 12
    table = (
        b"o\\data\\grass.paa".ljust(32, b"\0")
        + b"o\\data\\rock.paa".ljust(32, b"\0")
        + bytes(510 * 32)
    )
    path.write_bytes(
        struct.pack("<4sii", b"4WVR", 4, 4)
        + struct.pack("<16h", *terrain)
        + struct.pack("<16h", *slots)
        + table
        + bytes(128)
    )


def test_hover_elevation_uses_world_grid_and_height_scale(tmp_path: Path) -> None:
    source = tmp_path / "cursor.wrp"
    _world(source)
    with IslandDocument(source, cell_size=25.0, height_scale=0.5) as world:
        assert cursor_readout(world, 30.0, 3.0, "Elevation") == (
            "X 30.0 m  Z 3.0 m  |  Height 4.00 m"
        )
        assert "Height -2.00 m" in cursor_readout(world, 50, 0, "Elevation")
        world.paint(25, 0, radius=1, raise_metres=1.0)
        assert "Height 5.00 m" in cursor_readout(world, 30, 0, "Elevation")
        assert "Height" not in cursor_readout(world, 30, 0, "Road preview")


def test_texture_modes_show_name_slot_and_unnamed_fallback(tmp_path: Path) -> None:
    source = tmp_path / "cursor.wrp"
    _world(source)
    with IslandDocument(source, cell_size=25.0) as world:
        expected = r"Texture 1: o\data\rock.paa"
        for mode in ("Texture slots", "Actual textures"):
            assert expected in cursor_readout(world, 30, 0, mode)
            assert "Texture 2: (unnamed)" in cursor_readout(world, 55, 0, mode)
        world.paint(25, 0, radius=1, texture=0)
        assert r"Texture 0: o\data\grass.paa" in cursor_readout(
            world, 30, 0, "Actual textures"
        )
        world.paint(25, 0, radius=1, texture=511)
        assert "Texture 511: (unnamed)" in cursor_readout(world, 30, 0, "Texture slots")


def test_hover_does_not_clamp_outside_world(tmp_path: Path) -> None:
    source = tmp_path / "cursor.wrp"
    _world(source)
    with IslandDocument(source, cell_size=25.0) as world:
        for x, z in ((-1, 0), (0, -1), (100, 0), (0, 100), (float("nan"), 0)):
            assert cursor_readout(world, x, z, "Elevation") == "Outside map"
            assert cursor_readout(world, x, z, "Actual textures") == "Outside map"
        assert "Height 4.00 m" in cursor_readout(world, 99.99, 99.99, "Elevation")
        assert "X 30.0 m" in cursor_readout(world, 30, 0, "Road preview")
