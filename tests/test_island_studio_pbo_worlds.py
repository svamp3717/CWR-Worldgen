"""Read-only WRP archive support with bounded-memory member staging."""
from __future__ import annotations

import io
import struct
import threading
import time
from concurrent.futures import CancelledError
from tkinter import messagebox
from pathlib import Path
from types import SimpleNamespace

import pytest
import zstandard as zstd

from cwr_island_studio.app import IslandStudio
from cwr_island_studio.core import IslandDocument
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
        document=SimpleNamespace(read_only=True, read_only_forced=True),
        save_button=Widget(), readonly_toggle=Widget(), readonly_checked=Var(),
        edit_tool_combo=Widget(), edit_road_combo=Widget(), tool=Var(),
        _busy=False,
        _edit_buttons=buttons, _edit_entries=entries,
    )
    IslandStudio._update_edit_access(editor)
    assert editor.tool.value == "Select"
    assert editor.readonly_checked.value is True
    assert editor.readonly_toggle.state == "disabled"
    assert editor.save_button.state == "disabled"
    assert editor.edit_tool_combo.state == "disabled"
    assert all(b.state == "disabled" for b in buttons)
    assert all(e.state == "readonly" for e in entries)

    editor.document = SimpleNamespace(read_only=True, read_only_forced=False)
    IslandStudio._update_edit_access(editor)
    assert editor.readonly_checked.value is True
    assert editor.readonly_toggle.state == "normal"
    assert editor.save_button.state == "disabled"

    editor.document = SimpleNamespace(read_only=False, read_only_forced=False)
    IslandStudio._update_edit_access(editor)
    assert editor.readonly_checked.value is False
    assert editor.readonly_toggle.state == "normal"
    assert editor.save_button.state == "normal"
    assert editor.edit_tool_combo.state == "readonly"
    assert all(b.state == "normal" for b in buttons)
    assert all(e.state == "normal" for e in entries)



def _fake_editor(document):
    """Exercise actual GUI read-only callbacks without a graphical display."""
    class Widget:
        def __init__(self):
            self.state = "normal"
        def configure(self, *, state):
            self.state = state

    class Var:
        def __init__(self, value=None):
            self.value = value
        def set(self, value):
            self.value = value
        def get(self):
            return self.value

    names = []
    editor = SimpleNamespace(
        document=document,
        _busy=False, _road_anchor=None, _road_cursor=None,
        _index_future=None, _model_future=None,
        selected=None, cursor_info=Var(""),
        readonly_checked=Var(True), readonly_toggle=Widget(),
        save_button=Widget(), edit_tool_combo=Widget(),
        edit_road_combo=Widget(), tool=Var("Paint texture"),
        _edit_buttons=[Widget(), Widget()], _edit_entries=[Widget()],
        status=Var(""), metadata=Var(""), root=SimpleNamespace(
            title=lambda text: names.append(text),
        ),
        _queue_draw=lambda: None,
        _fit=lambda: None,
        _executor=SimpleNamespace(submit=lambda fn: "not-started-test-future"),
    )
    editor._update_edit_access = lambda: IslandStudio._update_edit_access(editor)
    editor._update_document_label = lambda: IslandStudio._update_document_label(editor)
    return editor, names


def test_loose_wrp_opens_with_readonly_checked_by_default(tmp_path: Path) -> None:
    path = tmp_path / "safe-default.wrp"
    path.write_bytes(wrp())
    with IslandDocument(path) as doc:
        editor, titles = _fake_editor(None)
        editor._busy = True
        IslandStudio._accept_document(editor, doc)
        assert editor.document is doc
        assert doc.read_only and not doc.read_only_forced
        assert editor.readonly_checked.get() is True
        assert editor.readonly_toggle.state == "normal"
        assert editor.save_button.state == "disabled"
        assert editor.tool.get() == "Select"
        assert titles[-1].endswith("[read-only]")
        assert "READ-ONLY" in editor.metadata.get()
        with pytest.raises(PermissionError, match="read-only"):
            doc.paint(25, 25, radius=0, texture=1)


def test_loose_wrp_checkbox_disables_mutations_and_can_restore_editing(tmp_path: Path) -> None:
    path = tmp_path / "editable.wrp"
    original = wrp()
    path.write_bytes(original)
    with IslandDocument(path) as doc:
        editor, titles = _fake_editor(doc)
        assert not doc.read_only and not doc.read_only_forced
        editor.readonly_checked.set(True)
        IslandStudio._toggle_read_only(editor)
        assert doc.read_only
        assert editor.save_button.state == "disabled"
        assert editor.readonly_toggle.state == "normal"
        assert "[read-only]" in titles[-1]
        for modify in (
            lambda: doc.move_object(0, x=1, y=2, z=3),
            lambda: doc.set_model(0, "another.p3d"),
            lambda: doc.add_object("another.p3d", x=1, y=1, z=1),
            lambda: doc.delete_object(0),
            lambda: doc.paint(0, 0, radius=1, texture=2),
            lambda: doc.export(tmp_path / "should-not-export.wrp"),
        ):
            with pytest.raises(PermissionError, match="read-only"):
                modify()
        assert not (tmp_path / "should-not-export.wrp").exists()
        editor.readonly_checked.set(False)
        IslandStudio._toggle_read_only(editor)
        assert not doc.read_only
        assert editor.save_button.state == "normal"
        assert editor.edit_tool_combo.state == "readonly"
        assert not titles[-1].endswith("[read-only]")
        doc.paint(0, 0, radius=1, texture=1)
        assert doc.changed
        editor.readonly_checked.set(True)
        IslandStudio._toggle_read_only(editor)
        assert doc.changed
        with pytest.raises(PermissionError, match="read-only"):
            doc.export(tmp_path / "protected.wrp")
        editor.readonly_checked.set(False)
        IslandStudio._toggle_read_only(editor)
        doc.export(tmp_path / "allowed.wrp")
    assert path.read_bytes() == original
    assert (tmp_path / "allowed.wrp").is_file()


def test_pbo_readonly_cannot_be_unchecked_in_gui_or_engine(tmp_path: Path) -> None:
    source = tmp_path / "locked.wrp"
    source.write_bytes(wrp())
    with IslandDocument(source, read_only=True) as doc:
        assert doc.read_only and doc.read_only_forced
        with pytest.raises(PermissionError, match="must remain read-only"):
            doc.read_only = False
        editor, _titles = _fake_editor(doc)
        editor.readonly_checked.set(False)
        IslandStudio._toggle_read_only(editor)
        assert doc.read_only
        assert editor.readonly_checked.get() is True
        assert editor.readonly_toggle.state == "disabled"
        assert editor.save_button.state == "disabled"
        with pytest.raises(PermissionError, match="read-only"):
            doc.delete_object(0)


def test_readonly_checkbox_reverts_while_loading_or_exporting(tmp_path: Path) -> None:
    path = tmp_path / "busy.wrp"
    path.write_bytes(wrp())
    with IslandDocument(path) as doc:
        editor, _titles = _fake_editor(doc)
        editor._busy = True
        editor.readonly_checked.set(True)
        IslandStudio._toggle_read_only(editor)
        assert not doc.read_only
        assert editor.readonly_checked.get() is False
        assert editor.readonly_toggle.state == "disabled"
        editor._busy = False
        editor._update_edit_access()
        assert editor.readonly_toggle.state == "normal"


def test_wrp_directory_scan_reports_progress_and_is_cancellable(tmp_path: Path) -> None:
    source = tmp_path / "directory.pbo"
    members = tuple((f"assets\\entry{i:04}.txt", b"x", False) for i in range(1100))
    make_pbo(source, members + (("maps\\island.wrp", wrp(), False),))
    messages: list[str] = []
    archive = PboWorldArchive(source, progress=messages.append)
    try:
        assert list(archive.members) == [r"directory\maps\island.wrp"]
        assert any("512 directory entries" in message for message in messages)
        assert any("1,101 directory entries" in message for message in messages)
    finally:
        archive.close()

    cancel = threading.Event()

    def stop_at_first_batch(message: str) -> None:
        if "512 directory entries" in message:
            cancel.set()

    with pytest.raises(CancelledError):
        PboWorldArchive(source, progress=stop_at_first_batch, cancel_event=cancel)


def test_wrapped_wrp_scan_cancels_after_unpacked_phase(tmp_path: Path) -> None:
    source = tmp_path / "source.pbo"
    make_pbo(source, (("island.wrp", wrp(), False),))
    wrapped = tmp_path / "source.pbo.zst"
    wrapped.write_bytes(zstd.ZstdCompressor().compress(source.read_bytes()))
    phases: list[str] = []
    cancel = threading.Event()

    def progress(message: str) -> None:
        phases.append(message)
        if message.startswith("Reading PBO directory"):
            cancel.set()

    with pytest.raises(CancelledError):
        PboWorldArchive(wrapped, progress=progress, cancel_event=cancel)
    assert any(message.startswith("Unpacking compressed archive") for message in phases)
    assert any(message.startswith("Reading PBO directory") for message in phases)


def test_quit_during_pbo_indexing_never_shows_wait_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destroyed: list[bool] = []
    shutdown_calls: list[dict] = []
    cancel = threading.Event()
    monkeypatch.setattr(
        messagebox, "showwarning",
        lambda *_args, **_kwargs: pytest.fail("Quit must not show a wait warning"),
    )
    editor = SimpleNamespace(
        _closing=False, _busy=True, _loading_pbos=True,
        _archive_cancel=cancel,
        document=None, _index_future=None, _model_future=None,
        _waiting_pbos=None, pbo_textures=None,
        _executor=SimpleNamespace(shutdown=lambda **kwargs: shutdown_calls.append(kwargs)),
        root=SimpleNamespace(destroy=lambda: destroyed.append(True)),
    )
    IslandStudio._shutdown(editor)
    assert editor._closing and cancel.is_set()
    assert destroyed == [True]
    assert shutdown_calls == [{"wait": False, "cancel_futures": True}]


def test_locked_temp_cleanup_preserves_original_invalid_world_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "bad-world.pbo"
    # Pass the 4WVR magic check but fail the full terrain/table validation.
    make_pbo(source, (("broken.wrp", b"4WVR" + struct.pack("<ii", 16, 16), False),))
    archive = PboWorldArchive(source)
    temp_path = Path(archive._temporary.name)
    cleanup = archive._temporary.cleanup
    cleanup_calls = 0

    def temporary_windows_lock() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise PermissionError(13, "file is being used by another process")
        cleanup()

    monkeypatch.setattr(archive._temporary, "cleanup", temporary_windows_lock)
    with pytest.raises(ValueError, match="truncated terrain or texture table"):
        archive.open_world(r"bad-world\broken.wrp")
    deadline = time.monotonic() + 2.0
    while temp_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not temp_path.exists()
    assert cleanup_calls >= 2
