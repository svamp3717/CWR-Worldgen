"""Road painting, water shading and actual texture viewport tests."""
from __future__ import annotations

import math
from pathlib import Path
import struct

import numpy as np
import pytest
from PIL import Image

from cwr_island_studio.core import IslandDocument
from cwr_island_studio.roads import is_road_model, plan_straight_road, road_segment, straight_length
from cwr_island_studio.textures import decode_paa_preview, render_real_textures

_HEADER = struct.Struct("<4sii")
_OBJECT = struct.Struct("<12fi76s")


def make_world(path: Path, objects: tuple[bytes, ...] = ()) -> None:
    width = height = 8
    terrain = np.full((height, width), 7, dtype="<i2")
    terrain[0, 0] = -8
    terrain[1, 1] = 0
    textures = np.zeros((height, width), dtype="<i2")
    textures[0, 0] = 1
    table = b"textures\\green.paa".ljust(32, b"\0")
    table += b"textures\\red.png".ljust(32, b"\0")
    table += b"\0" * (510 * 32)
    path.write_bytes(
        _HEADER.pack(b"4WVR", width, height)
        + terrain.tobytes() + textures.tobytes() + table
        + b"".join(objects) + bytes(128)
    )


def make_object(object_id: int, model: str, x: float, z: float) -> bytes:
    return _OBJECT.pack(
        1, 0, 0, 0, 1, 0, 0, 0, 1,
        x, 0, z, object_id, model.encode("ascii").ljust(76, b"\0"),
    )


def test_water_overlay_only_colours_submerged_terrain(tmp_path: Path) -> None:
    path = tmp_path / "test.wrp"
    make_world(path)
    with IslandDocument(path, cell_size=50, height_scale=1) as doc:
        rgb = doc.viewport(0, 0, 400, 400, 8, 8, water_level=0)
        assert rgb.shape == (8, 8, 3)
        assert rgb[0, 0, 2] > rgb[0, 0, 0]
        assert rgb[1, 1, 2] > rgb[1, 1, 0]
        assert rgb[2, 2, 1] > rgb[2, 2, 2]
        elevated = doc.viewport(0, 0, 400, 400, 8, 8, water_level=-9)
        assert elevated[0, 0, 1] > elevated[0, 0, 2]
        doc.paint(100.0, 100.0, radius=1, raise_metres=-20)
        altered = doc.viewport(0, 0, 400, 400, 8, 8, water_level=0)
        assert altered[2, 2, 2] > altered[2, 2, 0]
        assert not np.array_equal(altered[2, 2], rgb[2, 2])


def test_straight_road_preview_matches_placed_object_centres() -> None:
    model = r"o\road\sil25.p3d"
    assert is_road_model(model)
    assert straight_length(model) == 25
    assert straight_length(r"o\road\sil12.p3d") == 12.5
    assert not is_road_model(r"data3d\ker deravej.p3d")
    placements = plan_straight_road((0, 50), (100, 50), model)
    assert len(placements) == 4
    for i, piece in enumerate(placements):
        assert piece.x == pytest.approx(12.5 + 25 * i)
        assert piece.z == 50
        assert piece.heading == 90
        a, b = road_segment(piece.x, piece.z, piece.heading, model)
        assert a == pytest.approx((25 * i, 50))
        assert b == pytest.approx((25 * (i + 1), 50))
    assert plan_straight_road((10, 10), (11, 10), model) == ()
    with pytest.raises(ValueError, match="supported straight"):
        plan_straight_road((0, 0), (100, 0), r"o\road\sil10 25.p3d")


def test_generated_straight_length_and_mixed_road_queries(tmp_path: Path) -> None:
    assert straight_length(r"seam_world\i\paved_w091_l0081.p3d") == pytest.approx(8.1)
    assert is_road_model(r"seam_world\i\paved_w091_l0081.p3d")
    path = tmp_path / "test.wrp"
    make_world(path, (
        make_object(1, r"o\road\sil25.p3d", 100, 100),
        make_object(2, r"data3d\ker deravej.p3d", 120, 120),
    ))
    with IslandDocument(path) as doc:
        before = doc.visible_objects(0, 0, 200, 200, roads_only=True)
        assert [o.object_id for o in before] == [1]
        doc.build_object_index()
        assert [o.object_id for o in doc.visible_objects(0, 0, 200, 200, roads_only=True)] == [1]
        doc.move_object(0, x=300, y=0, z=300)
        doc.add_object(r"o\road\sil25.p3d", x=130, y=0, z=140)
        assert [o.object_id for o in doc.visible_objects(0, 0, 200, 200, roads_only=True)] == [3]
        assert [o.object_id for o in doc.visible_objects(275, 275, 325, 325, roads_only=True)] == [1]
        doc.delete_object(2)
        assert doc.visible_objects(0, 0, 200, 200, roads_only=True) == ()


def _dxt1_paa_solid_green() -> bytes:
    block = struct.pack("<HHI", 0x07E0, 0x07DF, 0)
    return (
        struct.pack("<H", 0xFF01)
        + b"GGAT" + b"CGVA" + struct.pack("<I", 4) + bytes((0, 255, 0, 255))
        + struct.pack("<H", 0)
        + struct.pack("<HH", 4, 4) + len(block).to_bytes(3, "little") + block
        + struct.pack("<HH", 0, 0)
    )


def test_paa_decoder_and_actual_texture_sampling(tmp_path: Path) -> None:
    assets = tmp_path / "assets" / "textures"
    assets.mkdir(parents=True)
    (assets / "green.paa").write_bytes(_dxt1_paa_solid_green())
    Image.new("RGB", (8, 8), (255, 0, 0)).save(assets / "red.png")
    raw = decode_paa_preview(_dxt1_paa_solid_green())
    assert raw.getpixel((0, 0)) == (0, 255, 0)
    slots = np.array([[0, 1], [1, 0]], dtype=np.int32)
    fallback = np.full((2, 2, 3), 15, dtype=np.uint8)
    output, count = render_real_textures(
        slots, np.array([0, 50]), np.array([0, 50]),
        ("textures\\green.paa", "textures\\red.png"),
        tmp_path / "assets", fallback, 50.0,
    )
    assert count == 2
    assert output[0, 0].tolist() == [0, 255, 0]
    assert output[0, 1].tolist() == [255, 0, 0]
    output, count = render_real_textures(
        slots, np.array([0, 50]), np.array([0, 50]),
        ("textures\\not_found.paa", "textures\\red.png"),
        tmp_path / "assets", fallback, 50.0,
    )
    assert count == 1
    assert np.all(output[0, 0] == 15)


def test_actual_textures_never_load_traversal_paths(tmp_path: Path) -> None:
    slots = np.array([[0]], dtype=np.int32)
    fallback = np.full((1, 1, 3), 22, dtype=np.uint8)
    rendered, count = render_real_textures(
        slots, np.array([0]), np.array([0]),
        ("..\\outside.png",), tmp_path, fallback, 50.0,
    )
    assert count == 0
    assert np.array_equal(rendered, fallback)
