"""Navigation by OFP/CWA in-game console coordinates."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tkinter import messagebox, simpledialog

import pytest

from cwr_island_studio.app import IslandStudio
from cwr_island_studio.navigation import center_on_position, parse_game_position


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("[1234.5, 5678.25, 24.3]", (1234.5, 5678.25, 24.3)),
        ("[1234.5,5678.25,0]", (1234.5, 5678.25, 0.0)),
        ("1234.5, 5678.25", (1234.5, 5678.25, None)),
        ("1234.5 5678.25", (1234.5, 5678.25, None)),
        (" 1234.5 5678.25 0 ", (1234.5, 5678.25, 0.0)),
        ("{1234.5,5678.25,-50}", (1234.5, 5678.25, -50.0)),
        ("Position: [1234.5, 5678.25, 0]", (1234.5, 5678.25, 0.0)),
        ("getPos player: [1234.5, 5678.25, 0]", (1234.5, 5678.25, 0.0)),
        ("[-3.1e2, 4E+2, -1.3]", (-310, 400, -1.3)),
    ],
)
def test_game_coordinate_parser_accepts_console_and_xy_formats(
    source: str, expected: tuple[float, float, float | None],
) -> None:
    assert parse_game_position(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "", "[123, 456, 789, 1011]", "[123]", "[123,]", "[123 456, 0]",
        "getPos player", "[5, seven, 0]", "[nan, 123, 0]",
        "[inf, 123, 0]", "[12, 34, -inf]", "[12, 34",
        "12, 34]", "__import__('os').system('echo hi')",
    ],
)
def test_game_coordinate_parser_rejects_unsupported_and_unsafe_input(source: str) -> None:
    with pytest.raises(ValueError):
        parse_game_position(source)


def test_game_xy_center_is_same_screen_center_at_any_zoom() -> None:
    for zoom in (0.025, 0.2, 1.0, 8.0):
        left, top = center_on_position(1630.25, 4820.75, 1080, 780, zoom)
        assert left + 1080 / (2 * zoom) == pytest.approx(1630.25)
        assert top + 780 / (2 * zoom) == pytest.approx(4820.75)
    with pytest.raises(ValueError, match="positive"):
        center_on_position(50, 60, 0, 100, 1)
    with pytest.raises(ValueError, match="finite"):
        center_on_position(float("inf"), 60, 100, 100, 1)


class Variable:
    def __init__(self, value: str = ""):
        self.value = value

    def set(self, value: str) -> None:
        self.value = value


def _editor(*, read_only: bool) -> tuple[SimpleNamespace, list[str]]:
    actions = []
    doc = SimpleNamespace(world_width=10_000.0, world_height=10_000.0, read_only=read_only)
    editor = SimpleNamespace(
        document=doc, _busy=False, zoom=0.5,
        view_x=200.0, view_z=300.0, _last_game_position=None,
        canvas=SimpleNamespace(winfo_width=lambda: 1000, winfo_height=lambda: 800),
        status=Variable(),
        _refresh_hover=lambda: actions.append("hover"),
        _queue_draw=lambda: actions.append("draw"),
    )
    return editor, actions


@pytest.mark.parametrize("read_only", [True, False])
def test_go_to_position_uses_game_y_as_map_north_and_preserves_zoom(
    read_only: bool,
) -> None:
    editor, actions = _editor(read_only=read_only)
    IslandStudio._go_to_position(editor, "[1234.5, 5678.25, 140]")
    assert editor.zoom == 0.5
    assert editor.view_x == pytest.approx(234.5)
    assert editor.view_z == pytest.approx(4878.25)
    assert editor._last_game_position == "[1234.5, 5678.25, 140]"
    assert actions == ["hover", "draw"]
    assert editor.status.value == "Map centered at game X 1,234.50, Y 5,678.25 m"


@pytest.mark.parametrize(
    "source",
    ["[-1, 500, 0]", "[500, -1, 0]", "[10000, 0, 0]", "[0, 10000, 0]", "invalid"],
)
def test_go_to_position_rejects_invalid_or_out_of_bounds_without_pan(
    monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    editor, actions = _editor(read_only=False)
    errors = []
    monkeypatch.setattr(messagebox, "showerror", lambda title, detail: errors.append((title, detail)))
    IslandStudio._go_to_position(editor, source)
    assert (editor.view_x, editor.view_z) == (200, 300)
    assert not actions
    assert len(errors) == 1 and errors[0][0] == "Invalid game position"


def test_go_to_position_dialog_accepts_console_paste(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = []
    def prompt(title, description, **kwargs):
        prompts.append((title, description, kwargs))
        return "[2222, 3333, 2]"
    monkeypatch.setattr(simpledialog, "askstring", prompt)
    editor, _actions = _editor(read_only=True)
    editor.root = object()
    editor._extent = lambda: (300, 600, 900, 1200)
    received = []
    editor._go_to_position = lambda text: received.append(text)
    IslandStudio._go_to_position_dialog(editor)
    assert received == ["[2222, 3333, 2]"]
    assert "[X, Y, altitude]" in prompts[0][1]
    assert prompts[0][2]["initialvalue"] == "[600.0, 900.0, 0]"


def test_go_to_position_dialog_cancel_and_unloaded_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(simpledialog, "askstring", lambda *_a, **_kw: None)
    editor, actions = _editor(read_only=True)
    editor.root = object()
    editor._extent = lambda: (0, 0, 1000, 1000)
    editor._go_to_position = lambda text: actions.append(text)
    IslandStudio._go_to_position_dialog(editor)
    assert actions == []
    editor.document = None
    IslandStudio._go_to_position_dialog(editor)
    assert actions == []
