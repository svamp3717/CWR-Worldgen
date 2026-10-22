"""Live terrain and texture name inspector tests, including sparse brush edits."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import struct

from cwr_island_studio import app
from cwr_island_studio.core import MapObject
from cwr_island_studio.cursor_info import (
    ObjectHoverTarget, object_under_cursor, object_cursor_readout,
    texture_under_cursor,
)

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


def _object(index: int, x: float, z: float, name: str = r"o\road\sil25.p3d") -> MapObject:
    return MapObject(index, index + 100, name, x, 7.0, z, 45.0)


def test_object_hover_reports_visible_marker_and_clears_when_distant() -> None:
    near = _object(3, 100, 200)
    far = _object(4, 150, 240)
    targets = (
        ObjectHoverTarget(near, 25, 30),
        ObjectHoverTarget(far, 100, 100),
    )
    assert object_under_cursor(targets, 29, 31) == near
    assert object_under_cursor(targets, 40, 31) is None
    assert object_cursor_readout(None) == ""
    report = object_cursor_readout(near)
    assert "Object ID 103 (record 3)" in report
    assert near.model_path in report
    assert "Origin X 100.0, Y 7.0, Z 200.0" in report


def test_hover_matches_polygon_interior_and_road_strokes() -> None:
    building = _object(1, 100, 200, r"data3d\building.p3d")
    road = _object(2, 200, 200)
    targets = (
        ObjectHoverTarget(
            building, 10, 10, "outline",
            ((40, 40), (90, 40), (40, 90)),
        ),
        ObjectHoverTarget(
            road, 150, 150, "segment",
            ((100, 100), (210, 100)),
        ),
    )
    assert object_under_cursor(targets, 48, 49) == building
    # Inside the polygon's rectangular bounds but outside its triangle.
    assert object_under_cursor(targets, 88, 88) is None
    assert object_under_cursor(targets, 180, 103) == road
    assert object_under_cursor(targets, 180, 108) is None


def test_hover_prefers_small_markers_over_overlapping_large_footprint() -> None:
    forest = _object(1, 10, 20)
    house = _object(2, 11, 21)
    targets = (
        ObjectHoverTarget(forest, 10, 10, "outline",
                          ((0, 0), (80, 0), (80, 80), (0, 80))),
        ObjectHoverTarget(house, 40, 40),
    )
    assert object_under_cursor(targets, 40, 40) == house
    assert object_under_cursor(targets, 60, 60) == forest


def test_hover_footer_retains_texture_info_and_clears_on_exit(tmp_path: Path) -> None:
    source = tmp_path / "cursor.wrp"
    _world(source)
    with IslandDocument(source, cell_size=25.0) as world:
        fake = SimpleNamespace(
            document=world,
            layer=SimpleNamespace(get=lambda: "Actual textures"),
            debug_layer=SimpleNamespace(get=lambda: "Off"),
            debug_data=None,
            _hover_screen=(30.0, 97.0),
            cursor_info=Mock(),
            object_cursor_info=Mock(),
            _hover_targets=(ObjectHoverTarget(
                _object(5, 30.0, 3.0), 30.0, 97.0,
            ),),
            _to_world=lambda sx, sy: (sx, 100.0 - sy),
        )
        app.IslandStudio._refresh_hover(fake)
        assert r"Texture 1: o\data\rock.paa" in fake.cursor_info.set.call_args.args[0]
        assert "Object ID 105" in fake.object_cursor_info.set.call_args.args[0]
        fake._hover_screen = (30.0, 50.0)
        app.IslandStudio._refresh_hover(fake)
        assert fake.object_cursor_info.set.call_args.args == ("",)
        app.IslandStudio._mouse_leave(fake, None)
        assert fake.cursor_info.set.call_args.args == ("",)
        assert fake.object_cursor_info.set.call_args.args == ("",)


def test_hover_outside_map_does_not_show_object(tmp_path: Path) -> None:
    source = tmp_path / "cursor.wrp"
    _world(source)
    with IslandDocument(source) as world:
        fake = SimpleNamespace(
            document=world,
            layer=SimpleNamespace(get=lambda: "Elevation"),
            debug_layer=SimpleNamespace(get=lambda: "Off"),
            debug_data=None,
            _hover_screen=(12, 12),
            cursor_info=Mock(),
            object_cursor_info=Mock(),
            _hover_targets=(ObjectHoverTarget(_object(1, 25, 25), 12, 12),),
            _to_world=lambda x, y: (-5, -5),
        )
        app.IslandStudio._refresh_hover(fake)
        assert fake.cursor_info.set.call_args.args == ("Outside map",)
        assert fake.object_cursor_info.set.call_args.args == ("",)


def _clipboard_editor(world: IslandDocument) -> app.IslandStudio:
    editor = object.__new__(app.IslandStudio)
    editor.document = world
    editor.root = Mock()
    editor.root.focus_get.return_value = None
    editor.status = Mock()
    editor._hover_screen = (30.0, 97.0)
    editor._hover_targets = (ObjectHoverTarget(
        _object(8, 30, 3, r"data3d\custom_building.p3d"), 30.0, 97.0,
    ),)
    editor._to_world = lambda sx, sy: (sx, 100.0 - sy)
    return editor


def test_copy_model_and_texture_paths_with_distinct_hotkeys(tmp_path: Path) -> None:
    source = tmp_path / "cursor.wrp"
    _world(source)
    with IslandDocument(source, cell_size=25.0) as world:
        editor = _clipboard_editor(world)
        event = SimpleNamespace(widget=editor.root)
        app.IslandStudio._bind_copy_hotkeys(editor)
        bound = {call.args[0]: call.args[1] for call in editor.root.bind.call_args_list}
        assert set(bound) == {
            "<Control-Shift-KeyPress-M>", "<Control-Shift-KeyPress-T>",
            "<Control-Shift-KeyPress-C>",
        }
        assert bound["<Control-Shift-KeyPress-M>"](event) == "break"
        assert editor.root.clipboard_append.call_args.args == (
            r"data3d\custom_building.p3d",
        )
        assert "Copied model" in editor.status.set.call_args.args[0]
        assert bound["<Control-Shift-KeyPress-T>"](event) == "break"
        assert editor.root.clipboard_append.call_args.args == (
            r"o\data\rock.paa",
        )
        assert "Copied texture" in editor.status.set.call_args.args[0]
        assert bound["<Control-Shift-KeyPress-C>"](event) == "break"
        assert editor.root.clipboard_append.call_args.args == (
            "player setPos [30.000, 3.000, 0]",
        )
        assert "Copied coords:" in editor.status.set.call_args.args[0]
        assert editor.root.clipboard_clear.call_count == 3


def test_copy_respects_empty_hover_and_does_not_clear_clipboard(tmp_path: Path) -> None:
    source = tmp_path / "cursor.wrp"
    _world(source)
    with IslandDocument(source, cell_size=25.0) as world:
        editor = _clipboard_editor(world)
        event = SimpleNamespace(widget=editor.root)
        editor._hover_targets = ()
        assert app.IslandStudio._copy_under_cursor(editor, event, "model") == "break"
        editor.root.clipboard_clear.assert_not_called()
        assert "No visible model" in editor.status.set.call_args.args[0]

        editor._hover_screen = None
        assert app.IslandStudio._copy_under_cursor(editor, event, "texture") == "break"
        editor.root.clipboard_clear.assert_not_called()

        editor._hover_screen = (1, 1)
        editor._to_world = lambda *_args: (-1.0, 10.0)
        assert app.IslandStudio._copy_under_cursor(editor, event, "texture") == "break"
        editor.root.clipboard_clear.assert_not_called()

        editor._to_world = lambda *_args: (52.0, 1.0)  # Unnamed slot 2
        assert app.IslandStudio._copy_under_cursor(editor, event, "texture") == "break"
        editor.root.clipboard_clear.assert_not_called()
        assert "No named texture" in editor.status.set.call_args.args[0]
        # Coordinates are independent of whether the texture has a name.
        assert app.IslandStudio._copy_under_cursor(editor, event, "coords") == "break"
        assert editor.root.clipboard_append.call_args.args == (
            "player setPos [52.000, 1.000, 0]",
        )
        editor.root.clipboard_clear.reset_mock()

        editor._hover_screen = None
        assert app.IslandStudio._copy_under_cursor(editor, event, "coords") == "break"
        editor.root.clipboard_clear.assert_not_called()


def test_copy_keeps_keyboard_focus_and_reports_clipboard_errors(
    tmp_path: Path, monkeypatch,
) -> None:
    source = tmp_path / "cursor.wrp"
    _world(source)
    with IslandDocument(source, cell_size=25.0) as world:
        editor = _clipboard_editor(world)
        event = SimpleNamespace(widget=editor.root)
        editor.root.focus_get.return_value = Mock(spec=app.tk.Entry)
        assert app.IslandStudio._copy_under_cursor(editor, event, "model") is None
        editor.root.clipboard_append.assert_not_called()

        # Read-only comboboxes can retain keyboard focus after changing views.
        readonly_combo = Mock(spec=app.ttk.Combobox)
        readonly_combo.cget.return_value = "readonly"
        editor.root.focus_get.return_value = readonly_combo
        assert app.IslandStudio._copy_under_cursor(editor, event, "model") == "break"
        assert editor.root.clipboard_append.call_args.args == (
            r"data3d\custom_building.p3d",
        )

        editable_combo = Mock(spec=app.ttk.Combobox)
        editable_combo.cget.return_value = "normal"
        editor.root.focus_get.return_value = editable_combo
        assert app.IslandStudio._copy_under_cursor(editor, event, "model") is None

        editor.root.focus_get.return_value = None
        editor.root.clipboard_append.side_effect = app.tk.TclError("clipboard busy")
        assert app.IslandStudio._copy_under_cursor(editor, event, "model") == "break"
        assert "Could not copy model" in editor.status.set.call_args.args[0]


def test_copy_texture_matches_oprw_layer_cell_not_terrain_vertex(tmp_path: Path) -> None:
    from cwr_island_studio.oprw import open_island_document
    from test_island_studio_oprw import make_oprw

    source = tmp_path / "mixed_grids.wrp"
    source.write_bytes(make_oprw(
        map_size=(3, 3), layer_size=(2, 2), objects=False,
    ))
    with open_island_document(source, cell_size=50.0) as world:
        # The texture boundary is X=75, not the terrain vertex at X=100.
        assert world.texture_at(1, 0) == 0
        assert texture_under_cursor(world, 80, 20) == (1, r"o\data\grass.paa")
        assert r"Texture 1: o\data\grass.paa" in cursor_readout(
            world, 80, 20, "Actual textures",
        )


def test_copy_game_coords_uses_pointer_not_selected_object(tmp_path: Path) -> None:
    source = tmp_path / "cursor.wrp"
    _world(source)
    with IslandDocument(source, cell_size=25.0) as world:
        editor = _clipboard_editor(world)
        editor._hover_targets = ()
        editor._to_world = lambda sx, sy: (72.375, 45.125)
        event = SimpleNamespace(widget=editor.root)
        assert app.IslandStudio._copy_under_cursor(editor, event, "coords") == "break"
        assert editor.root.clipboard_append.call_args.args == (
            "player setPos [72.375, 45.125, 0]",
        )
