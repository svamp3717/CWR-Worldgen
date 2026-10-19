"""Synthetic OFP OPRW v2/v3 test worlds with independently packed grids."""
from __future__ import annotations

from concurrent.futures import CancelledError
from pathlib import Path
import threading
import struct

import numpy as np
import pytest

from cwr_island_studio.app import world_source_kind
from cwr_island_studio.oprw import OprwIslandDocument, open_island_document
from cwr_island_studio.pbo_worlds import PboWorldArchive


def packed(raw: bytes) -> bytes:
    """Encode a valid literal-only OFP LZSS packed block plus checksum."""
    output = bytearray()
    for start in range(0, len(raw), 8):
        chunk = raw[start:start + 8]
        output.append((1 << len(chunk)) - 1)
        output.extend(chunk)
    return bytes(output) + struct.pack("<I", sum(raw) & 0xFFFFFFFF)


def make_oprw(
    *, version: int = 3, map_size: tuple[int, int] = (4, 4),
    layer_size: tuple[int, int] = (2, 2),
    objects: bool = True,
) -> bytes:
    if version == 2:
        map_size = layer_size = (256, 256)
    tx, ty = layer_size
    x, y = map_size
    n_layer, n_map = tx * ty, x * y
    heights = (np.arange(n_map, dtype="<f4").reshape(y, x) * 0.25 - 4)
    textures = np.arange(n_layer, dtype="<u2") % 2
    out = bytearray(b"OPRW" + struct.pack("<I", version))
    if version == 3:
        out.extend(struct.pack("<IIII", tx, ty, x, y))
    out.extend(packed(bytes(n_layer * 4)))  # cell flags
    out.extend(packed(bytes(n_layer)))  # environment sounds
    out.extend(struct.pack("<I", 0))  # no peaks
    out.extend(packed(textures.tobytes()))
    out.extend(packed(bytes(n_layer * 4)))  # extended flags
    out.extend(packed(heights.tobytes()))
    out.extend(struct.pack("<I", 2))
    out.extend(b"data\\dummy.pac\0\0")
    out.extend(b"o\\data\\grass.paa\0\1")
    out.extend(struct.pack("<I", 2))
    out.extend(b"o\\road\\sil25.p3d\0")
    out.extend(b"data3d\\tree.p3d\0")
    if objects:
        for uid, model, px, pz in ((100, 0, 25.0, 30.0), (101, 1, 80.0, 75.0)):
            values = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                      0.0, 0.0, 1.0, px, 12.5, pz)
            out.extend(struct.pack("<II12f", uid, model, *values))
    out.extend(struct.pack("<I", 0xFFFFFFFF))
    return bytes(out)


def make_pbo(path: Path, name: str, body: bytes) -> None:
    path.write_bytes(
        name.encode("latin-1") + b"\0"
        + struct.pack("<IIIII", 0, 0, 0, 0, len(body))
        + b"\0" + struct.pack("<IIIII", 0, 0, 0, 0, 0)
        + body
    )


@pytest.mark.parametrize("version", [2, 3])
def test_oprw_terrain_objects_and_spatial_index(
    tmp_path: Path, version: int,
) -> None:
    filename = tmp_path / f"version{version}.wrp"
    filename.write_bytes(make_oprw(version=version))
    doc = open_island_document(filename)
    assert isinstance(doc, OprwIslandDocument)
    with doc:
        assert doc.read_only and doc.read_only_forced
        assert doc.oprw_version == version
        assert doc.object_count == 2
        assert (doc.width, doc.height) == ((256, 256) if version == 2 else (4, 4))
        assert doc.texture_paths[1] == r"o\data\grass.paa"
        assert doc.texture_at(0, 0) == 0
        assert doc.terrain_at(0, 0) == pytest.approx(-4.0)
        assert doc.terrain_at(1, 0) == pytest.approx(-3.75)
        tile, _, _ = doc.sample_viewport(
            0, 0, doc.world_width, doc.world_height, 12, 12,
        )
        assert tile.dtype == np.float32
        assert tile.shape == (12, 12)
        assert doc.viewport(
            0, 0, doc.world_width, doc.world_height, 12, 12,
        ).shape == (12, 12, 3)
        assert doc.object_at(0).object_id == 100
        assert doc.object_at(0).model_path == r"o\road\sil25.p3d"
        assert doc.object_at(1).z == pytest.approx(75.0)
        doc.build_object_index()
        assert [o.object_id for o in doc.visible_objects(0, 0, 150, 150)] == [100, 101]
        assert [o.object_id for o in doc.visible_objects(
            0, 0, 150, 150, roads_only=True,
        )] == [100]
        with pytest.raises(PermissionError):
            doc.paint(0, 0, radius=0, texture=1)
        with pytest.raises(PermissionError):
            doc.export(tmp_path / "output.wrp")
    assert filename.is_file()


def test_oprw_v3_different_layer_and_terrain_dimensions(tmp_path: Path) -> None:
    filename = tmp_path / "different.wrp"
    filename.write_bytes(make_oprw(objects=False))
    with open_island_document(filename) as doc:
        assert doc.object_count == 0
        assert (doc.layer_width, doc.layer_height) == (2, 2)
        assert doc.texture_at(3, 3) == 1
        image, _, _ = doc.sample_viewport(0, 0, 200, 200, 4, 4, textures=True)
        assert image.tolist() == [
            [0, 0, 1, 1], [0, 0, 1, 1],
            [0, 0, 1, 1], [0, 0, 1, 1],
        ]
        doc.build_object_index()
        assert doc.visible_objects(0, 0, 100, 100) == ()


def test_oprw_open_from_pbo_preserves_source_and_cleans_up(tmp_path: Path) -> None:
    archive_path = tmp_path / "world.pbo"
    original = make_oprw(version=3)
    make_pbo(archive_path, r"maps\island.wrp", original)
    assert world_source_kind(archive_path) == "archive"
    archive = PboWorldArchive(archive_path)
    with archive.open_world(r"world\maps\island.wrp") as doc:
        extracted = doc.path
        assert isinstance(doc, OprwIslandDocument)
        assert doc.read_only_forced
        assert doc.object_count == 2
        assert extracted.read_bytes() == original
    assert not extracted.exists()
    assert archive_path.is_file()


def test_reject_later_oprw_versions_without_guessing_offsets(tmp_path: Path) -> None:
    filename = tmp_path / "arma.wrp"
    filename.write_bytes(b"OPRW" + struct.pack("<I", 18) + bytes(64))
    with pytest.raises(ValueError, match="OPRW version 18"):
        open_island_document(filename)


def test_reject_invalid_packed_data_and_invalid_model_index(tmp_path: Path) -> None:
    filename = tmp_path / "bad.wrp"
    data = bytearray(make_oprw())
    data[25] ^= 1  # tamper packed grid checksum or payload
    filename.write_bytes(data)
    with pytest.raises(ValueError, match="checksum"):
        open_island_document(filename)

    data = bytearray(make_oprw())
    data[-(56 + 4) + 4:-(56 + 4) + 8] = b"\xff\xff\xff\x7f"
    filename.write_bytes(data)
    with pytest.raises(ValueError, match="model index"):
        open_island_document(filename)


def test_oprw_index_cancellation_before_build_does_not_complete(
    tmp_path: Path,
) -> None:
    world = tmp_path / "cancel.wrp"
    world.write_bytes(make_oprw(version=3))
    event = threading.Event()
    with open_island_document(world) as doc:
        event.set()
        with pytest.raises(CancelledError):
            doc.build_object_index(cancel_event=event)
        assert doc._index_order is None
