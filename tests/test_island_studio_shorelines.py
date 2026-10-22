"""Shorelines are optional on Elevation; the world frame is always shown."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import cwr_island_studio.app as app
from cwr_island_studio.core import IslandDocument
from cwr_island_studio.shorelines import highlight_shoreline
from test_island_studio_layers import make_world


def test_shoreline_highlights_only_land_water_transitions() -> None:
    original = np.full((5, 5, 3), (75, 125, 95), dtype=np.uint8)
    terrain = np.asarray([
        [-4, -4, -4, -4, -4],
        [-4, -4, -4, -4, -4],
        [4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4],
        [4, 4, 4, 4, 4],
    ], dtype=np.float32)
    result = highlight_shoreline(
        original, terrain, water_level=0, height_scale=1.0,
    )
    assert (result[2] == (186, 244, 248)).all()
    assert (result[1] == (11, 31, 42)).all()
    np.testing.assert_array_equal(result[4], original[4])
    assert np.all(original == np.array([75, 125, 95], dtype=np.uint8))
    # A changed sea level must move the shoreline; uniform land has none.
    np.testing.assert_array_equal(
        highlight_shoreline(original, terrain, water_level=-8, height_scale=1.0),
        original,
    )


def test_shoreline_scales_heights_and_does_not_draw_world_edges() -> None:
    pixels = np.full((3, 4, 3), 90, dtype=np.uint8)
    all_land = np.ones((3, 4), dtype=np.float32) * 10
    all_sea = np.ones((3, 4), dtype=np.float32) * -10
    np.testing.assert_array_equal(
        highlight_shoreline(pixels, all_land, water_level=0, height_scale=1), pixels,
    )
    np.testing.assert_array_equal(
        highlight_shoreline(pixels, all_sea, water_level=0, height_scale=1), pixels,
    )
    terrain = np.array([
        [-2, -2, 2, 2],
        [-2, -2, 2, 2],
        [-2, -2, 2, 2],
    ], dtype=np.float32)
    result = highlight_shoreline(
        pixels, terrain, water_level=0, height_scale=0.5,
    )
    assert (result[:, 2] == (186, 244, 248)).all()
    assert (result[:, 1] == (11, 31, 42)).all()
    assert (result[:, 3] == (11, 31, 42)).all()
    with pytest.raises(ValueError, match="size"):
        highlight_shoreline(pixels, terrain[:, :3], water_level=0, height_scale=1)
    with pytest.raises(ValueError, match="scale"):
        highlight_shoreline(pixels, terrain, water_level=0, height_scale=0)


@pytest.mark.parametrize("mode", ("Elevation", "Texture slots", "Road preview", "Actual textures"))
def test_world_frame_always_visible_and_shoreline_only_on_elevation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    source = tmp_path / "test.wrp"
    make_world(source, objects=())
    with IslandDocument(source, cell_size=50, height_scale=1) as document:
        document.build_object_index()
        editor = object.__new__(app.IslandStudio)
        editor.document = document
        editor.canvas = Mock()
        editor.canvas.winfo_width.return_value = 400
        editor.canvas.winfo_height.return_value = 400
        editor.view_x = editor.view_z = 0.0
        editor.zoom = 1.0
        editor._render_pending = True
        editor.layer = SimpleNamespace(get=lambda: mode)
        editor.water_level = SimpleNamespace(get=lambda: "0")
        editor.show_shorelines = SimpleNamespace(get=lambda: True)
        editor.show_roads = SimpleNamespace(get=lambda: False)
        editor.show_forests = SimpleNamespace(get=lambda: False)
        editor.show_objects = SimpleNamespace(get=lambda: False)
        editor.show_model_footprints = SimpleNamespace(get=lambda: False)
        editor.show_locations = SimpleNamespace(get=lambda: False)
        editor.status = Mock()
        editor.asset_roots = ()
        editor.pbo_textures = None
        editor._map_locations = ()
        editor.selected = None
        editor._model_bounds = {}
        editor._road_anchor = None
        editor._road_cursor = None

        monkeypatch.setattr(app.ImageTk, "PhotoImage", lambda *_a, **_k: object())
        monkeypatch.setattr(
            app, "render_real_textures",
            lambda _slots, _x, _z, _paths, _roots, fallback, _size, **_kw: (fallback, 0),
        )
        calls = []
        original_overlay = app.highlight_shoreline

        def track(*args, **kwargs):
            calls.append(kwargs["water_level"])
            return original_overlay(*args, **kwargs)

        monkeypatch.setattr(app, "highlight_shoreline", track)
        app.IslandStudio._draw(editor)

        assert calls == ([0.0] if mode == "Elevation" else [])
        assert editor.status.set.call_args is None or not str(
            editor.status.set.call_args.args[0]
        ).startswith("Preview error:")
        borders = editor.canvas.create_rectangle.call_args_list
        assert len(borders) == 2
        assert borders[0].args == pytest.approx((0, 0, 400, 400))
        assert borders[1].args == pytest.approx((0, 0, 400, 400))
        assert borders[0].kwargs == {"outline": "#13252b", "width": 6}
        assert borders[1].kwargs == {"outline": "#fff0aa", "width": 2}

        # Panning and zooming changes the world frame; it must not become a
        # rectangle around the fixed canvas viewport.
        editor.canvas.reset_mock()
        editor.view_x, editor.view_z, editor.zoom = 20, 50, 2
        editor.show_shorelines = SimpleNamespace(get=lambda: False)
        calls.clear()
        app.IslandStudio._draw(editor)
        assert not calls
        rectangles = editor.canvas.create_rectangle.call_args_list
        assert len(rectangles) == 2
        assert rectangles[0].args == pytest.approx((-40, -100, 760, 700))
