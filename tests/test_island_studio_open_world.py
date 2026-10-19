"""One Open World picker routes loose RVW4 maps and PBO/IFA containers."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import struct
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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


def test_multi_world_picker_returns_without_waiting_for_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actions: dict[str, object] = {}
    opened: list[tuple] = []
    status: list[str] = []
    destroyed: list[bool] = []

    class Widget:
        def __init__(self, *_args, **kwargs):
            if "command" in kwargs:
                actions[kwargs["text"]] = kwargs["command"]

        def pack(self, **_kwargs):
            pass

    class Listbox(Widget):
        def insert(self, *_args):
            pass

        def selection_set(self, _index):
            pass

        def curselection(self):
            return (0,)

        def bind(self, *_args):
            pass

        def focus_set(self):
            pass

    class Dialog(Widget):
        def title(self, _value):
            pass

        def geometry(self, _value):
            pass

        def transient(self, _root):
            pass

        def protocol(self, *_args):
            pass

        def lift(self):
            pass

        def destroy(self):
            destroyed.append(True)

    monkeypatch.setattr(tk, "Toplevel", Dialog)
    monkeypatch.setattr(tk, "Listbox", Listbox)
    monkeypatch.setattr(ttk, "Frame", Widget)
    monkeypatch.setattr(ttk, "Label", Widget)
    monkeypatch.setattr(ttk, "Button", Widget)

    archive = SimpleNamespace(
        path=Path("worlds.pbo"),
        members={r"worlds\a.wrp": object(), r"worlds\b.wrp": object()},
    )
    editor = SimpleNamespace(
        root=object(), _closing=False, _busy=True,
        status=SimpleNamespace(set=status.append),
        _open_archive_member=lambda *args: opened.append(args),
    )
    IslandStudio._select_archive_member(editor, archive, 50.0, 1.0)
    assert not opened
    assert "Found 2 WRP entries" in status[-1]
    actions["Open read-only"]()
    assert destroyed == [True]
    assert opened == [(archive, r"worlds\a.wrp", 50.0, 1.0)]
