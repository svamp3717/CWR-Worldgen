"""One Open World picker routes loose RVW4 maps and PBO/IFA containers."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import struct
from tkinter import filedialog, messagebox

import pytest

from cwr_island_studio.app import IslandStudio, world_source_kind


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("island.wrp", "wrp"),
        ("ISLAND.WRP", "wrp"),
        ("objects.pbo", "archive"),
        ("Archive.PBO", "archive"),
        ("world.ifa", "archive"),
        ("world.pbo.zst", "archive"),
        ("WORLD.IFA.ZST", "archive"),
    ],
)
def test_shared_world_picker_recognizes_both_file_kinds(
    filename: str, expected: str,
) -> None:
    assert world_source_kind(Path(filename)) == expected


@pytest.mark.parametrize("filename", ["world.txt", "archive.zip", "world.wrp.bak", ""])
def test_unsupported_file_type_is_rejected(filename: str) -> None:
    with pytest.raises(ValueError, match="Choose a"):
        world_source_kind(filename)


@pytest.mark.parametrize("selection", ["island.wrp", "world.pbo", "WORLD.IFA.ZST"])
def test_single_dialog_routes_file_through_same_open_path(
    monkeypatch: pytest.MonkeyPatch, selection: str,
) -> None:
    opened: list[Path] = []
    args = []
    def choose(**kwargs):
        args.append(kwargs)
        return selection
    monkeypatch.setattr(filedialog, "askopenfilename", choose)
    editor = SimpleNamespace(
        _busy=False,
        open_path=lambda path: opened.append(path),
    )
    IslandStudio._open_dialog(editor)
    assert opened == [Path(selection)]
    assert len(args) == 1
    kinds = args[0]["filetypes"]
    assert ".wrp" in str(kinds) and ".pbo" in str(kinds)
    assert ".ifa" in str(kinds) and ".zst" in str(kinds)


@pytest.mark.parametrize("selection", ["world.pbo", "world.ifa", "world.pbo.zst"])
def test_open_path_dispatches_archive_to_existing_readonly_loader(selection: str) -> None:
    found: list[Path] = []
    editor = SimpleNamespace(
        _busy=False,
        document=None,
        _open_pbo_path=lambda path: found.append(path),
    )
    IslandStudio.open_path(editor, Path(selection))
    assert found == [Path(selection)]


def test_open_path_retains_loose_wrp_loading(tmp_path: Path) -> None:
    world_path = tmp_path / "world.wrp"
    world_path.write_bytes(
        struct.pack("<4sii", b"4WVR", 2, 2)
        + bytes(2 * 2 * 4 + 512 * 32 + 128)
    )
    received = []
    messages = SimpleNamespace(put=lambda item: received.append(item))
    status = SimpleNamespace(set=lambda _item: None)
    editor = SimpleNamespace(
        _busy=False, document=None,
        cell_size=object(), height_scale=object(),
        _num=lambda _value, name: 50.0 if name == "Grid metres" else 1.0,
        status=status,
        _messages=messages,
        _executor=SimpleNamespace(submit=lambda fn: fn()),
    )
    IslandStudio.open_path(editor, world_path)
    assert len(received) == 1 and received[0][0] == "open"
    doc = received[0][1]
    try:
        assert doc.path == world_path.resolve()
        assert not doc.read_only_forced
        assert (doc.width, doc.height) == (2, 2)
    finally:
        doc.close()


def test_unsupported_file_surfaces_error_without_trying_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    errors = []
    monkeypatch.setattr(messagebox, "showerror", lambda title, msg: errors.append((title, msg)))
    archive_calls = []
    editor = SimpleNamespace(
        _busy=False, document=None,
        _open_pbo_path=lambda path: archive_calls.append(path),
    )
    IslandStudio.open_path(editor, Path("world.docx"))
    assert len(errors) == 1
    assert errors[0][0] == "Unsupported world file"
    assert not archive_calls
