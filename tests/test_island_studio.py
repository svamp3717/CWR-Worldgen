"""Regression tests for the standalone, large-world RVW4 island editor."""
from __future__ import annotations

from pathlib import Path
import math
import struct

import numpy as np
import pytest

from cwr_island_studio.core import IslandDocument

HEADER = struct.Struct("<4sii")
OBJECT = struct.Struct("<12fi76s")


def record(
    object_id: int, path: str, x: float, z: float, *,
    y: float = 2.0, heading: float = 0.0, shear: float = 0.0,
) -> bytes:
    radians = math.radians(heading)
    c, s = math.cos(radians), math.sin(radians)
    return OBJECT.pack(
        c, shear, -s,
        0.0, 1.0, 0.0,
        s, -shear, c,
        x, y, z, object_id,
        path.encode("ascii").ljust(76, b"\0"),
    )


def make_world(
    path: Path, *,
    width: int = 8, height: int = 8,
    objects: tuple[bytes, ...] | None = None,
    terminator: bool = True,
) -> bytes:
    if objects is None:
        objects = (
            record(1, r"o\road\sil25.p3d", 100.0, 100.0, shear=0.37),
            record(2, r"data3d\ker listnac.p3d", 300.0, 125.0),
            record(3, r"o\road\ces6.p3d", 225.0, 300.0, heading=22.0),
        )
    count = width * height
    terrain = [i % 31 - 8 for i in range(count)]
    textures = [0 if i % 2 == 0 else 1 for i in range(count)]
    table = b"o\\data\\grass.paa".ljust(32, b"\0")
    table += b"o\\data\\rock.paa".ljust(32, b"\0")
    table += bytes(510 * 32)
    data = (
        HEADER.pack(b"4WVR", width, height)
        + struct.pack(f"<{count}h", *terrain)
        + struct.pack(f"<{count}h", *textures)
        + table
        + b"".join(objects)
        + (bytes(128) if terminator else b"")
    )
    path.write_bytes(data)
    return data


def test_open_is_lazy_and_roundtrip_is_byte_identical(tmp_path: Path) -> None:
    source = tmp_path / "source.wrp"
    original = make_world(source)
    with IslandDocument(source, cell_size=50.0, height_scale=0.5) as doc:
        assert (doc.width, doc.height) == (8, 8)
        assert doc.world_width == 400.0
        assert doc.object_count == 3
        assert doc.has_terminator
        assert doc.texture_paths[0] == r"o\data\grass.paa"
        assert doc.terrain_at(0, 0) == -4.0
        assert doc.object_at(0).model_path == r"o\road\sil25.p3d"
        assert not doc.changed
        exported = doc.export(tmp_path / "copy.wrp")
        assert exported.read_bytes() == original
        with pytest.raises(ValueError, match="different output"):
            doc.export(source)


def test_sparse_terrain_and_texture_patch_preserve_every_other_byte(tmp_path: Path) -> None:
    source = tmp_path / "source.wrp"
    original = make_world(source)
    with IslandDocument(source, height_scale=0.5) as doc:
        changed = doc.paint(100.0, 100.0, radius=1.0, raise_metres=1.0, texture=19)
        assert changed == 1
        assert doc.terrain_at(2, 2) == (int(doc.heights[2, 2]) + 2) * 0.5
        assert doc.texture_at(2, 2) == 19
        image = doc.viewport(0, 0, 400, 400, 8, 8, textures=True)
        assert image.dtype == np.uint8 and image.shape == (8, 8, 3)
        assert not np.array_equal(image[2, 2], image[2, 3])
        result = doc.export(tmp_path / "patched.wrp")
        data = result.read_bytes()
        assert len(data) == len(original)
        terrain_off = HEADER.size + (2 * 8 + 2) * 2
        texture_off = HEADER.size + 8 * 8 * 2 + (2 * 8 + 2) * 2
        changed_offsets = {*range(terrain_off, terrain_off + 2), *range(texture_off, texture_off + 2)}
        assert all(
            before == after for i, (before, after) in enumerate(zip(original, data))
            if i not in changed_offsets
        )
        assert source.read_bytes() == original


def test_position_edit_preserves_exact_object_basis_and_path(tmp_path: Path) -> None:
    source = tmp_path / "source.wrp"
    original = make_world(source)
    with IslandDocument(source) as doc:
        old_bytes = doc._record(0)
        doc.move_object(0, x=150.5, y=-2.25, z=189.0)
        assert doc.object_at(0).x == 150.5
        output = doc.export(tmp_path / "moved.wrp")
        data = output.read_bytes()
        offset = doc.object_offset
        assert data[offset:offset + 9 * 4] == old_bytes[:9 * 4]
        assert data[offset + 52:offset + 128] == old_bytes[52:]
        assert source.read_bytes() == original


def test_heading_rotation_preserves_vertical_shear(tmp_path: Path) -> None:
    source = tmp_path / "source.wrp"
    make_world(source)
    with IslandDocument(source) as doc:
        raw = doc._record(0)
        before = OBJECT.unpack(raw)
        obj = doc.object_at(0)
        doc.move_object(
            0, x=obj.x, y=obj.y, z=obj.z, heading=90.0,
        )
        after = OBJECT.unpack(doc._record(0))
        for offset in (1, 4, 7):
            assert after[offset] == before[offset]
        assert after[0] == pytest.approx(0.0, abs=1e-6)
        assert after[2] == pytest.approx(-1.0, abs=1e-6)
        assert doc.object_at(0).heading == pytest.approx(90.0)


def test_object_add_delete_and_model_edit_preserve_other_records(tmp_path: Path) -> None:
    source = tmp_path / "source.wrp"
    original = make_world(source)
    with IslandDocument(source) as doc:
        original_third = doc._record(2)
        doc.delete_object(0)
        doc.set_model(1, r"data3d\other.p3d")
        idx = doc.add_object(r"o\road\sil6.p3d", x=100, y=3.5, z=220, heading=45)
        assert idx == 3
        doc.move_object(idx, x=125, y=3.5, z=225)
        assert doc.object_at(idx).object_id == 4
        exported = doc.export(tmp_path / "edited.wrp")
        assert source.read_bytes() == original
    with IslandDocument(exported) as saved:
        assert saved.object_count == 3
        assert saved.object_at(0).object_id == 2
        assert saved.object_at(0).model_path == r"data3d\other.p3d"
        assert saved._record(1) == original_third
        assert saved.object_at(2).x == 125.0
        assert saved.object_at(2).object_id == 4


def test_original_objects_stay_in_spatial_query_after_index_and_sparse_moves(tmp_path: Path) -> None:
    path = tmp_path / "world.wrp"
    make_world(path)
    with IslandDocument(path) as doc:
        before = doc.visible_objects(90, 90, 140, 140)
        assert [obj.object_id for obj in before] == [1]
        doc.build_object_index()
        assert [obj.object_id for obj in doc.visible_objects(90, 90, 140, 140)] == [1]
        doc.move_object(0, x=350, y=0, z=350)
        assert doc.visible_objects(90, 90, 140, 140) == ()
        assert [obj.object_id for obj in doc.visible_objects(340, 340, 360, 360)] == [1]
        doc.delete_object(1)
        assert all(o.object_id != 2 for o in doc.visible_objects(0, 0, 400, 400))


def test_single_pixel_preview_and_no_terminator_compatibility(tmp_path: Path) -> None:
    path = tmp_path / "without_terminator.wrp"
    original = make_world(path, terminator=False)
    with IslandDocument(path) as doc:
        assert doc.object_count == 3
        assert doc.viewport(0, 0, 400, 400, 1, 1).shape == (1, 1, 3)
        doc.export(tmp_path / "nochange.wrp")
        assert (tmp_path / "nochange.wrp").read_bytes() == original


def test_invalid_worlds_and_brushes_are_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "other.wrp"
    invalid.write_bytes(b"OPRW" + b"\0" * 200)
    with pytest.raises(ValueError, match="4WVR"):
        IslandDocument(invalid)
    truncated = tmp_path / "truncated.wrp"
    truncated.write_bytes(b"4WVR" + struct.pack("<ii", 1024, 1024) + b"\0" * 100)
    with pytest.raises(ValueError, match="truncated"):
        IslandDocument(truncated)
    valid = tmp_path / "source.wrp"
    make_world(valid)
    with IslandDocument(valid) as doc:
        with pytest.raises(ValueError, match="texture slot"):
            doc.paint(100, 100, radius=50, texture=512)
        with pytest.raises(ValueError, match="elevation limits"):
            doc.paint(100, 100, radius=1, raise_metres=100_000)


def test_large_object_stream_indexes_without_materializing_python_objects(tmp_path: Path) -> None:
    source = tmp_path / "large.wrp"
    one = record(1, r"o\road\sil25.p3d", 1000, 1000)
    many = one * 90_000
    make_world(source, width=64, height=64, objects=(many,))
    with IslandDocument(source) as doc:
        assert doc.object_count == 90_000
        assert doc._records.dtype.itemsize == 128
        assert not doc._records.flags.owndata
        doc.build_object_index()
        assert doc._index_order is not None
        assert len(doc.visible_objects(0, 0, 500, 500)) == 0
        visible = doc.visible_objects(900, 900, 1100, 1100, limit=250)
        assert 0 < len(visible) <= 250
        assert all(o.x == 1000.0 for o in visible)
