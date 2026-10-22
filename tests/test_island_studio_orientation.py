"""North-up Island Studio map coordinates must match OSM and in-game WRP."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cwr_island_studio import app
from cwr_island_studio.core import IslandDocument
from cwr_island_studio.model_bounds import ModelBounds
from cwr_island_studio.oprw import open_island_document
from cwr_worldgen.model import WorldObject
from cwr_worldgen.wrp import write_rvw4
from cwr_worldgen.wrp_binarize import binarize_rvw4


def _asymmetric_world(path: Path) -> None:
    # WRP row 0 is SOUTH, row 3 is NORTH. Deliberately asymmetric so a
    # mistaken north/south screen flip cannot accidentally pass.
    write_rvw4(
        path, 4, 4,
        [float(row * 10 + col) for row in range(4) for col in range(4)],
        [row * 4 + col for row in range(4) for col in range(4)],
        [rf"landtext\t{i}.paa" for i in range(16)],
        [
            WorldObject(1, r"data3d\jablon.p3d", 50, 0, 350),
            WorldObject(2, r"data3d\ker listnac.p3d", 150, 0, 50),
        ],
        height_scale=1.0,
    )


def test_world_to_screen_and_back_put_north_at_the_top() -> None:
    editor = SimpleNamespace(
        document=SimpleNamespace(world_height=400.0),
        canvas=SimpleNamespace(winfo_width=lambda: 400, winfo_height=lambda: 400),
        view_x=0.0, view_z=0.0, zoom=1.0,
    )
    # True game/OSM positions must not be mutated for display.
    assert app.IslandStudio._to_screen(editor, 50, 350) == (50, 50)
    assert app.IslandStudio._to_screen(editor, 150, 50) == (150, 350)
    for x, z in ((50, 350), (150, 50), (0, 0), (400, 400)):
        sx, sy = app.IslandStudio._to_screen(editor, x, z)
        assert app.IslandStudio._to_world(editor, sx, sy) == pytest.approx((x, z))
    assert app.IslandStudio._extent(editor) == pytest.approx((0, 400, 400, 0))


def test_loose_wrp_terrain_textures_are_displayed_north_up(tmp_path: Path) -> None:
    source = tmp_path / "asymmetric.wrp"
    _asymmetric_world(source)
    with IslandDocument(source, cell_size=100.0, height_scale=1.0) as doc:
        cells, xs, zs = doc.sample_viewport(
            0, doc.world_height, doc.world_width, 0,
            4, 4, textures=True,
        )
        assert xs == pytest.approx([0, 100, 200, 300])
        assert zs == pytest.approx([350, 250, 150, 50])
        np.testing.assert_array_equal(
            cells, np.array([[12, 13, 14, 15],
                             [8, 9, 10, 11],
                             [4, 5, 6, 7],
                             [0, 1, 2, 3]], dtype=np.int32),
        )
        terrain, _, _ = doc.sample_viewport(
            0, 400, 400, 0, 4, 4,
        )
        assert [int(y) for y in terrain[:, 0]] == [30, 20, 10, 0]
        assert doc.terrain_at(0, 0) == 0
        assert doc.terrain_at(0, 3) == 30


def test_oprw_display_uses_same_north_up_coordinate_convention(tmp_path: Path) -> None:
    source = tmp_path / "asymmetric.wrp"
    binary = tmp_path / "asymmetric.oprw.wrp"
    _asymmetric_world(source)
    binarize_rvw4(source, binary, height_scale=1.0)
    with open_island_document(binary, cell_size=100.0) as doc:
        terrain, xs, zs = doc.sample_viewport(
            0, doc.world_height, doc.world_width, 0, 4, 4,
        )
        assert xs == pytest.approx([0, 100, 200, 300])
        assert zs == pytest.approx([350, 250, 150, 50])
        np.testing.assert_array_equal(
            terrain[:, 0], np.array([30., 20., 10., 0.]),
        )
        assert doc.terrain_at(0, 3) == pytest.approx(30.0)


def test_editor_overlays_follow_same_north_up_terrain_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "objects.wrp"
    _asymmetric_world(source)
    with IslandDocument(source, cell_size=100.0, height_scale=1.0) as doc:
        doc.build_object_index()
        editor = object.__new__(app.IslandStudio)
        editor.document = doc
        editor.canvas = Mock()
        editor.canvas.winfo_width.return_value = 400
        editor.canvas.winfo_height.return_value = 400
        editor.view_x = editor.view_z = 0.0
        editor.zoom = 1.0
        editor._render_pending = True
        editor.layer = SimpleNamespace(get=lambda: "Texture slots")
        editor.water_level = SimpleNamespace(get=lambda: "0")
        editor.show_roads = SimpleNamespace(get=lambda: False)
        editor.show_forests = SimpleNamespace(get=lambda: False)
        editor.show_objects = SimpleNamespace(get=lambda: True)
        editor.show_model_footprints = SimpleNamespace(get=lambda: False)
        editor.show_locations = SimpleNamespace(get=lambda: False)
        editor.status = Mock()
        editor.asset_roots = ()
        editor.pbo_textures = None
        editor._map_locations = ()
        editor.selected = None
        editor._road_anchor = editor._road_cursor = None
        editor._model_bounds = {}
        monkeypatch.setattr(app.ImageTk, "PhotoImage", lambda *_a, **_kw: object())
        app.IslandStudio._draw(editor)
        editor.status.set.assert_not_called()
        assert len(editor._visible) == 2
        assert len(editor._hover_targets) == 2
        assert {item.obj.index for item in editor._hover_targets} == {0, 1}
        assert all(item.kind == "marker" for item in editor._hover_targets)
        markers = editor.canvas.create_oval.call_args_list
        centres = sorted(
            ((call.args[0] + call.args[2]) / 2,
             (call.args[1] + call.args[3]) / 2)
            for call in markers
        )
        assert centres == pytest.approx([(50, 50), (150, 350)])


def test_road_overlay_uses_p3d_outline_before_straight_line_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "road.wrp"
    write_rvw4(
        source, 4, 4, [0.0] * 16, [0] * 16, [r"data\grass.paa"],
        (WorldObject(1, r"o\road\sil25.p3d", 150, 0, 250),),
        height_scale=1.0,
    )
    with IslandDocument(source, cell_size=100.0, height_scale=1.0) as doc:
        doc.build_object_index()
        editor = object.__new__(app.IslandStudio)
        editor.document = doc
        editor.canvas = Mock()
        editor.canvas.winfo_width.return_value = 400
        editor.canvas.winfo_height.return_value = 400
        editor.view_x = editor.view_z = 0.0
        editor.zoom = 1.0
        editor._render_pending = True
        editor.layer = SimpleNamespace(get=lambda: "Road preview")
        editor.water_level = SimpleNamespace(get=lambda: "0")
        editor.show_roads = SimpleNamespace(get=lambda: True)
        editor.show_forests = SimpleNamespace(get=lambda: False)
        editor.show_objects = SimpleNamespace(get=lambda: False)
        editor.show_model_footprints = SimpleNamespace(get=lambda: False)
        editor.show_locations = SimpleNamespace(get=lambda: False)
        editor.status = Mock()
        editor.asset_roots = ()
        editor.pbo_textures = None
        editor._map_locations = ()
        editor.selected = None
        editor._road_anchor = editor._road_cursor = None
        editor._model_bounds = {
            r"o\road\sil25.p3d": ModelBounds(
                0, 20, 0, 10,
                outline=((0, 0), (20, 0), (0, 10)),
            ),
        }
        monkeypatch.setattr(app.ImageTk, "PhotoImage", lambda *_a, **_kw: object())
        app.IslandStudio._draw(editor)
        polygon = editor.canvas.create_polygon.call_args
        assert polygon is not None
        assert polygon.args == pytest.approx((150, 150, 170, 150, 150, 140))
        assert polygon.kwargs["outline"] == "#ed974a"
        assert len(editor._hover_targets) == 1
        assert editor._hover_targets[0].kind == "outline"
        assert editor._hover_targets[0].points == ((150, 150), (170, 150), (150, 140))
        editor.canvas.create_line.assert_not_called()
        assert not editor.status.set.called

        # A missing model continues to use the familiar approximate line.
        editor.canvas.reset_mock()
        editor._model_bounds.clear()
        app.IslandStudio._draw(editor)
        editor.canvas.create_polygon.assert_not_called()
        editor.canvas.create_line.assert_called_once()
        assert len(editor._hover_targets) == 1
        assert editor._hover_targets[0].kind == "segment"
        assert not editor.status.set.called
