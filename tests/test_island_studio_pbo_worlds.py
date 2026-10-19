"""Read-only WRP archive support with bounded-memory member staging."""
from __future__ import annotations

import io
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import zstandard as zstd

from cwr_island_studio.app import IslandStudio
from cwr_island_studio.pbo_worlds import PboWorldArchive

_ENTRY = struct.Struct("<IIIII")
_CPRS = 0x43707273
_OBJECT = struct.Struct("<12fi76s")


def wrp(width: int = 8) -> bytes:
    object_record = _OBJECT.pack(
        1, 0, 0, 0, 1, 0, 0, 0, 1, 100, 23, 125, 1,
        b"data3d\\tree.p3d".ljust(76, b"\0"),
    )
    return (
        struct.pack("<4sii", b"4WVR", width, width)
        + bytes(width * width * 4 + 512 * 32)
        + object_record + bytes(128)
    )


def lzss_literal(raw: bytes) -> bytes:
    output = bytearray()
    for start in range(0, len(raw), 8):
        chunk = raw[start:start + 8]
        output.append((1 << len(chunk)) - 1)
        output.extend(chunk)
    return bytes(output) + struct.pack("<I", sum(raw) & 0xFFFFFFFF)


def make_pbo(
    path: Path, members: tuple[tuple[str, bytes, bool], ...],
    *, prefix: str | None = None,
) -> None:
    stream = io.BytesIO()
    if prefix is not None:
        stream.write(b"\0" + _ENTRY.pack(0x56657273, 0, 0, 0, 0))
        stream.write(b"prefix\0" + prefix.encode("ascii") + b"\0\0")
    blobs = []
    for name, uncompressed, compressed in members:
        stored = lzss_literal(uncompressed) if compressed else uncompressed
        blobs.append(stored)
        stream.write(name.encode("ascii") + b"\0")
        stream.write(_ENTRY.pack(
            _CPRS if compressed else 0,
            len(uncompressed) if compressed else 0, 0, 0, len(stored),
        ))
    stream.write(b"\0" + _ENTRY.pack(0, 0, 0, 0, 0))
    for blob in blobs:
        stream.write(blob)
    path.write_bytes(stream.getvalue())


def test_uncompressed_pbo_map_is_immutable_and_temp_removed(tmp_path: Path) -> None:
    filename = tmp_path / "forests.pbo"
    original_world = wrp()
    make_pbo(filename, (
        ("config.cpp", b"config", False),
        ("maps\\island.wrp", original_world, False),
        ("texture.paa", b"image", False),
    ), prefix="cwrworld")
    source_bytes = filename.read_bytes()
    archive = PboWorldArchive(filename)
    assert list(archive.members) == [r"cwrworld\maps\island.wrp"]
    doc = archive.open_world(r"cwrworld\maps\island.wrp", cell_size=40)
    temp = doc.path
    try:
        assert doc.read_only and doc.width == 8
        assert doc.world_width == 320
        assert doc.source_label == r"forests.pbo / cwrworld\maps\island.wrp"
        assert doc.object_at(0).model_path == r"data3d\tree.p3d"
        assert temp.read_bytes() == original_world
        doc.build_object_index()
        assert len(doc.visible_objects(80, 80, 150, 150)) == 1
        for modify in (
            lambda: doc.move_object(0, x=2, y=2, z=2),
            lambda: doc.set_model(0, "new.p3d"),
            lambda: doc.add_object("new.p3d", x=2, y=2, z=2),
            lambda: doc.delete_object(0),
            lambda: doc.paint(50, 50, radius=5, raise_metres=2),
            lambda: doc.export(tmp_path / "edited.wrp"),
        ):
            with pytest.raises(PermissionError, match="read-only"):
                modify()
        assert not doc.changed
    finally:
        doc.close()
    assert not temp.exists()
    assert not (tmp_path / "edited.wrp").exists()
    assert filename.read_bytes() == source_bytes
    doc.close()


def test_archive_with_multiple_worlds_and_compressed_member(tmp_path: Path) -> None:
    filename = tmp_path / "multiple.pbo"
    large = wrp(width=16)
    make_pbo(filename, (
        ("one.wrp", wrp(width=4), False),
        ("unrelated.p3d", b"MLOD", False),
        ("two.WRP", large, True),
    ))
    archive = PboWorldArchive(filename)
    assert set(archive.members) == {r"multiple\one.wrp", r"multiple\two.wrp"}
    with archive.open_world(r"multiple\two.wrp") as doc:
        assert doc.width == 16
        assert doc.path.read_bytes() == large
    assert not Path(archive._temporary.name).exists()


def test_zstd_wrapper_and_invalid_checksum(tmp_path: Path) -> None:
    filename = tmp_path / "woods.pbo"
    make_pbo(filename, (("island.wrp", wrp(), True),))
    wrapped = tmp_path / "woods.pbo.zst"
    wrapped.write_bytes(zstd.ZstdCompressor().compress(filename.read_bytes()))
    archive = PboWorldArchive(wrapped)
    with archive.open_world(r"woods\island.wrp") as doc:
        assert doc.width == 8
    assert not Path(archive._temporary.name).exists()

    corrupt = tmp_path / "broken.pbo"
    packed = bytearray(lzss_literal(wrp()))
    packed[-1] ^= 1
    with corrupt.open("wb") as stream:
        stream.write(b"island.wrp\0")
        stream.write(_ENTRY.pack(_CPRS, len(wrp()), 0, 0, len(packed)))
        stream.write(b"\0" + _ENTRY.pack(0, 0, 0, 0, 0))
        stream.write(packed)
    indexed = PboWorldArchive(corrupt)
    temp = Path(indexed._temporary.name)
    with pytest.raises(ValueError, match="checksum"):
        indexed.open_world(r"broken\island.wrp")
    assert not temp.exists()


def test_empty_and_unsupported_archived_map_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty.pbo"
    make_pbo(empty, (("model.p3d", b"MLOD", False),))
    with pytest.raises(ValueError, match="No .wrp"):
        PboWorldArchive(empty)
    invalid = tmp_path / "invalid.pbo"
    make_pbo(invalid, (("broken.wrp", b"OPRW" + bytes(40), False),))
    archive = PboWorldArchive(invalid)
    temp = Path(archive._temporary.name)
    with pytest.raises(ValueError, match="4WVR"):
        archive.open_world(r"invalid\broken.wrp")
    assert not temp.exists()


def test_staging_large_wrp_uses_normal_mmap_document(tmp_path: Path) -> None:
    base = wrp(width=16)
    record = base[-256:-128]
    large = base[:-128] + record * 22000 + base[-128:]
    archive_path = tmp_path / "huge.pbo"
    make_pbo(archive_path, (("huge.wrp", large, False),))
    archive = PboWorldArchive(archive_path)
    with archive.open_world(r"huge\huge.wrp") as doc:
        assert doc.object_count == 22001
        assert doc._records.flags.owndata is False
        assert doc.path.stat().st_size == len(large)
    assert not Path(archive._temporary.name).exists()


def test_readonly_and_editable_toolbar_states() -> None:
    class Widget:
        def __init__(self):
            self.state = "normal"
        def configure(self, *, state):
            self.state = state

    class Var:
        def __init__(self):
            self.value = "Paint texture"
        def set(self, value):
            self.value = value

    buttons, entries = [Widget(), Widget()], [Widget(), Widget()]
    editor = SimpleNamespace(
        document=SimpleNamespace(read_only=True),
        save_button=Widget(), edit_tool_combo=Widget(),
        edit_road_combo=Widget(), tool=Var(),
        _edit_buttons=buttons, _edit_entries=entries,
    )
    IslandStudio._update_edit_access(editor)
    assert editor.tool.value == "Select"
    assert editor.save_button.state == "disabled"
    assert editor.edit_tool_combo.state == "disabled"
    assert all(b.state == "disabled" for b in buttons)
    assert all(e.state == "readonly" for e in entries)

    editor.document = SimpleNamespace(read_only=False)
    IslandStudio._update_edit_access(editor)
    assert editor.save_button.state == "normal"
    assert editor.edit_tool_combo.state == "readonly"
    assert all(b.state == "normal" for b in buttons)
    assert all(e.state == "normal" for e in entries)
