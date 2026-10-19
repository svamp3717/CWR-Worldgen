"""One Open World picker routes loose RVW4 maps and PBO/IFA containers."""
from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
import threading
import queue
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



def test_open_second_pbo_retires_first_without_waiting_for_its_index(
    tmp_path: Path,
) -> None:
    """No Tk or old mmap operation may block the second archive scan."""
    previous_closed = threading.Event()
    old_cancel = threading.Event()
    pending_index = Future()
    status: list[str] = []
    cleared: list[str] = []
    tasks: list[object] = []

    class PreviousWorld:
        changed = False

        def close(self) -> None:
            previous_closed.set()

    editor = SimpleNamespace(
        _busy=False,
        document=PreviousWorld(),
        _index_future=pending_index,
        _index_cancel=old_cancel,
        _model_stamp=0,
        _model_bounds={"stale": object()},
        selected=3,
        _visible=(object(),),
        _image=object(),
        canvas=SimpleNamespace(delete=cleared.append),
        metadata=SimpleNamespace(set=lambda _v: None),
        status=SimpleNamespace(set=status.append),
        _update_edit_access=lambda: None,
        _reset_model_bounds=lambda: None,
        _num=lambda _v, name: 50.0 if name == "Grid metres" else 1.0,
        cell_size=object(),
        height_scale=object(),
        _archive_cancel=threading.Event(),
        _messages=SimpleNamespace(put=lambda _msg: None),
        _executor=SimpleNamespace(submit=lambda job: tasks.append(job)),
    )
    editor._retire_current_world = lambda: IslandStudio._retire_current_world(editor)
    world = tmp_path / "second.pbo"
    IslandStudio._open_pbo_path(editor, world)

    assert old_cancel.is_set()
    assert editor.document is None
    assert editor._index_future is None
    assert editor._index_cancel is None
    assert editor._visible == ()
    assert editor.selected is None
    assert cleared == ["all"]
    assert not previous_closed.is_set()
    assert len(tasks) == 1
    assert status[-1] == f"Indexing WRP entries in {world.name}…"

    # Finishing the old index must only close the *old* document.
    pending_index.set_result(None)
    assert previous_closed.wait(timeout=2.0)
    assert editor.document is None
    assert status[-1] == f"Indexing WRP entries in {world.name}…"


def test_open_second_loose_world_retires_first_before_new_loader(
    tmp_path: Path,
) -> None:
    old_cancel = threading.Event()
    pending_index = Future()
    closed = threading.Event()
    tasks: list[object] = []
    statuses: list[str] = []

    class PreviousWorld:
        changed = False

        def close(self) -> None:
            closed.set()

    editor = SimpleNamespace(
        _busy=False,
        document=PreviousWorld(),
        _index_future=pending_index,
        _index_cancel=old_cancel,
        _model_bounds={},
        selected=None,
        _visible=(),
        _image=None,
        canvas=SimpleNamespace(delete=lambda _v: None),
        metadata=SimpleNamespace(set=lambda _v: None),
        status=SimpleNamespace(set=statuses.append),
        _update_edit_access=lambda: None,
        _reset_model_bounds=lambda: None,
        _num=lambda _v, name: 50.0 if name == "Grid metres" else 1.0,
        cell_size=object(),
        height_scale=object(),
        _executor=SimpleNamespace(submit=lambda job: tasks.append(job)),
    )
    editor._retire_current_world = lambda: IslandStudio._retire_current_world(editor)
    IslandStudio.open_path(editor, tmp_path / "another.wrp")
    assert editor.document is None
    assert old_cancel.is_set()
    assert not closed.is_set()
    assert len(tasks) == 1
    pending_index.set_result(None)
    assert closed.wait(timeout=2.0)
    assert statuses[-1] == "Opening another.wrp…"


def test_completed_index_keeps_gui_polling_so_second_archive_can_open() -> None:
    """An index-finished status must not crash Tk's recurring event loop."""
    future = Future()
    future.set_result(None)
    messages: queue.Queue = queue.Queue()
    scheduled: list[tuple[int, object]] = []
    statuses: list[str] = []
    opened: list[tuple[object, float, float]] = []
    editor = SimpleNamespace(
        _closing=False,
        _messages=messages,
        _index_future=future,
        _index_cancel=threading.Event(),
        _model_future=None,
        document=SimpleNamespace(
            object_count=5, read_only_forced=True, read_only=True,
        ),
        _queue_draw=lambda: None,
        status=SimpleNamespace(set=statuses.append),
        root=SimpleNamespace(after=lambda ms, callback: scheduled.append((ms, callback))),
        _select_archive_member=lambda archive, grid, scale: opened.append(
            (archive, grid, scale)
        ),
    )
    IslandStudio._poll(editor)
    assert scheduled and scheduled[-1][0] == 150
    assert statuses[-1].startswith("Ready: 5 objects indexed.")
    assert editor._index_future is None
    archive = object()
    messages.put(("archive_catalog", (archive, 50.0, 1.0)))
    # The follow-up callback must still be alive after indexing the first WRP.
    scheduled[-1][1]()
    assert opened == [(archive, 50.0, 1.0)]
    assert len(scheduled) == 2
