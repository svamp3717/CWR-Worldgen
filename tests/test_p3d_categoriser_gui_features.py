"""Categoriser GUI parity with the standalone P3D asset browser.

These tests use light Tk doubles so CI does not need a display; they verify
streaming search, archive-source changes, texture inspection and copy actions
without changing classifications or rescanning the current model list.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import p3d_categoriser_app as ui
from cwr_worldgen.pbo import PboEntry, write_pbo
from p3d_texture_render import MipmapCache
from p3d_texture_sources import TextureResolver


class Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def model(path):
    return SimpleNamespace(model_path=path, source="source.ifa")


def app_for_search(paths, *, index=0, future=()):
    app = ui.CategoriserApp.__new__(ui.CategoriserApp)
    app.search_var = Var("")
    app.status_var = Var("")
    app.models = [model(path) for path in paths]
    app.index = index
    app.current = app.models[index] if index >= 0 else None
    app.exhausted = False
    app.failures = []
    app.state = {}
    app.model_iter = iter(model(path) for path in future)
    app._commit = Mock()
    app.save_state = Mock()
    app._show_model = Mock(side_effect=lambda item: setattr(app, "current", item))
    return app


def test_search_scans_forward_without_rendering_intermediate_models() -> None:
    app = app_for_search(
        ["ca\\old.p3d"], future=["ca\\tree.p3d", "a3\\urban\\house.p3d", "a3\\another.p3d"]
    )
    app.search_var.set("urban/house")
    app.find_next_model()
    assert app.index == 2
    assert len(app.models) == 3
    app._show_model.assert_called_once_with(app.models[2])
    app._commit.assert_called_once_with(True)
    assert app.save_state.call_count == 2
    assert "found model 3" in app.status_var.get().casefold()
    assert app.models[0].model_path == "ca\\old.p3d"


def test_search_uses_scanned_list_before_streaming_and_wraps() -> None:
    app = app_for_search(["a\\found.p3d", "b\\other.p3d", "c\\target.p3d"], index=1)
    app.search_var.set("found")
    app.find_next_model()
    assert app.index == 0
    assert app.current.model_path == "a\\found.p3d"
    assert len(app.models) == 3

    app.search_var.set("target")
    app.find_next_model()
    assert app.index == 2
    assert app.current.model_path == "c\\target.p3d"


def test_search_no_match_preserves_current_index_and_state() -> None:
    app = app_for_search(["ca\\one.p3d"], future=["ca\\two.p3d"])
    original = app.current
    app.search_var.set("does_not_exist")
    app.find_next_model()
    assert app.current is original
    assert app.index == 0
    assert app.exhausted is True
    assert len(app.models) == 2
    app._commit.assert_not_called()
    app.save_state.assert_not_called()
    assert "no matching" in app.status_var.get().casefold()


def test_search_respects_modal_navigation_guard() -> None:
    app = app_for_search(["one.p3d"], future=["two.p3d"])
    app.search_var.set("two")
    app._begin_navigation_loading = Mock(return_value=False)
    app._finish_navigation_loading = Mock()
    app.find_next_model()
    assert len(app.models) == 1
    app._finish_navigation_loading.assert_not_called()


def test_search_empty_input_does_not_scan() -> None:
    app = app_for_search(["one.p3d"], future=["two.p3d"])
    app.find_next_model()
    assert len(app.models) == 1
    assert "enter a model" in app.status_var.get().casefold()


def test_texture_archives_added_without_restart_and_caches_invalidated(tmp_path: Path) -> None:
    initial = tmp_path / "base.pbo"
    second = tmp_path / "textures.ifa"
    write_pbo(initial, (PboEntry("house.p3d", b"fixture"),))
    write_pbo(second, (PboEntry("brick.paa", b"fixture"),))

    app = ui.CategoriserApp.__new__(ui.CategoriserApp)
    app.texture_resolver = TextureResolver((initial,))
    app._base_texture_sources = (initial,)
    app._extra_texture_sources = []
    app.mipmap_cache = MipmapCache()
    app.status_var = Var("")
    app.current = model("base\\house.p3d")
    app.model_iter = iter(["unchanged"])
    app.models = [app.current]
    app.state = {"base\\house.p3d": object()}
    app._redraw = Mock()
    original_iterator = app.model_iter
    original_state = app.state

    app._set_extra_texture_sources((second, second))
    assert app.texture_resolver.inputs == (initial, second)
    assert app._extra_texture_sources == [second]
    assert app.model_iter is original_iterator
    assert app.state is original_state
    assert app.models == [app.current]
    assert app._redraw.call_count == 1

    app._set_extra_texture_sources((second,))
    assert app._redraw.call_count == 1
    app.clear_added_texture_sources()
    assert app.texture_resolver.inputs == (initial,)
    assert app._extra_texture_sources == []
    assert app._redraw.call_count == 2
    assert app.state is original_state


def test_texture_source_selection_accepts_directories_and_rejects_unrelated_file(
    tmp_path: Path,
) -> None:
    app = ui.CategoriserApp.__new__(ui.CategoriserApp)
    app._base_texture_sources = ()
    app._extra_texture_sources = []
    app.mipmap_cache = MipmapCache()
    app.texture_resolver = TextureResolver(())
    app.status_var = Var("")
    app.current = None
    folder = tmp_path / "Addons"
    folder.mkdir()
    invalid = tmp_path / "archive.zip"
    invalid.write_bytes(b"not a PBO")

    app._set_extra_texture_sources((invalid,))
    assert app._extra_texture_sources == []
    assert "not a pbo/ifa" in app.status_var.get().casefold()
    app._set_extra_texture_sources((folder,))
    assert app.texture_resolver.inputs == (folder,)


def test_path_copy_uses_canonical_model_path() -> None:
    app = ui.CategoriserApp.__new__(ui.CategoriserApp)
    app.root = Mock()
    app.status_var = Var("")
    app.current = model("ca\\structures\\house.p3d")
    app.copy_current_path()
    app.root.clipboard_clear.assert_called_once_with()
    app.root.clipboard_append.assert_called_once_with("ca\\structures\\house.p3d")
    assert "copied model path" in app.status_var.get().casefold()


def test_selected_texture_copy_and_missing_selection() -> None:
    app = ui.CategoriserApp.__new__(ui.CategoriserApp)
    app.root = Mock()
    app.status_var = Var("")
    app._related_texture_paths = ("ca\\data\\brick.paa", "lib\\metal.paa")
    app.texture_list = Mock()
    app.texture_list.curselection.return_value = (1,)
    app.copy_selected_texture_path()
    app.root.clipboard_append.assert_called_once_with("lib\\metal.paa")
    app.texture_list.curselection.return_value = ()
    assert app._selected_texture_path() is None
    assert "select a referenced texture" in app.status_var.get().casefold()


def test_preview_texture_reports_missing_archive_without_replacing_current(monkeypatch) -> None:
    app = ui.CategoriserApp.__new__(ui.CategoriserApp)
    app.current = model("ca\\house.p3d")
    app.status_var = Var("")
    app.texture_list = Mock()
    app.texture_list.curselection.return_value = (0,)
    app._related_texture_paths = ("ca\\data\\not_found.paa",)
    app.texture_resolver = SimpleNamespace(load_rgba=Mock(return_value=None))
    popup = Mock()
    monkeypatch.setattr(ui.tk, "Toplevel", popup)
    app.preview_selected_texture()
    popup.assert_not_called()
    app.texture_resolver.load_rgba.assert_called_once_with(
        "ca\\data\\not_found.paa", "source.ifa"
    )
    assert "texture not found" in app.status_var.get().casefold()
    assert app.current.model_path == "ca\\house.p3d"


def test_actual_texture_preview_shows_decoded_rgba(monkeypatch) -> None:
    app = ui.CategoriserApp.__new__(ui.CategoriserApp)
    app.current = model("ca\\house.p3d")
    app.root = object()
    app.status_var = Var("")
    app.texture_list = Mock()
    app.texture_list.curselection.return_value = (0,)
    app._related_texture_paths = ("ca\\data\\brick.paa",)
    app.texture_resolver = SimpleNamespace(
        load_rgba=Mock(return_value=np.full((8, 4, 4), 255, dtype=np.uint8))
    )

    popup = Mock()
    monkeypatch.setattr(ui.tk, "Toplevel", Mock(return_value=popup))
    label = Mock()
    monkeypatch.setattr(ui.ttk, "Label", Mock(return_value=label))
    canvas = Mock()
    monkeypatch.setattr(ui, "FigureCanvasTkAgg", Mock(return_value=canvas))

    app.preview_selected_texture()
    assert "brick.paa" in popup.title.call_args.args[0]
    assert "4 × 8" in ui.ttk.Label.call_args.kwargs["text"]
    canvas.draw_idle.assert_called_once_with()
    app.texture_resolver.load_rgba.assert_called_once()
    assert "texture loaded" in app.status_var.get().casefold()


def test_reset_view_restores_camera_and_zoom() -> None:
    app = ui.CategoriserApp.__new__(ui.CategoriserApp)
    app.current = model("building.p3d")
    app.azim, app.elev, app.zoom = 190, 65, 3.5
    app._redraw = Mock()
    app._reset_view()
    assert (app.azim, app.elev, app.zoom) == (35.0, 25.0, 1.0)
    app._redraw.assert_called_once()



def _output_app(output: Path):
    from p3d_categoriser_split import SplitStateCategoriserApp
    from p3d_categoriser_app import Classification
    app = SplitStateCategoriserApp.__new__(SplitStateCategoriserApp)
    app.root = object()
    app.output = output
    app.output_var = Var(str(output))
    app.categories = ["Residential"]
    app.status_var = Var("")
    app.current = model(r"ca\house.p3d")
    app.state = {
        r"ca\house.p3d": Classification(
            ["Residential"], "Urban", reviewed=True
        ),
        r"ca\shed.p3d": Classification(
            ["Residential"], "", reviewed=True
        ),
    }
    app._commit = Mock()
    return app


def test_change_output_json_saves_completed_and_incomplete_companion(
    tmp_path: Path, monkeypatch,
) -> None:
    import json
    from p3d_categoriser_storage import incomplete_state_path
    old = tmp_path / "old.json"
    new = tmp_path / "fresh" / "output.json"
    app = _output_app(old)
    monkeypatch.setattr(
        ui.filedialog, "asksaveasfilename", Mock(return_value=str(new))
    )
    answer = Mock(return_value=True)
    monkeypatch.setattr(ui.messagebox, "askyesno", answer)
    app.change_json_output()

    assert app.output == new
    assert app.output_var.get() == str(new)
    assert not old.exists()
    assert new.exists()
    assert incomplete_state_path(new).exists()
    payload = json.loads(new.read_text(encoding="utf-8"))
    assert payload["resume_model_path"] == r"ca\house.p3d"
    assert [x["model_path"] for x in payload["models"]] == [r"ca\house.p3d"]
    assert json.loads(incomplete_state_path(new).read_text())["models"][0]["model_path"] == r"ca\shed.p3d"
    app._commit.assert_called_once_with(False)
    answer.assert_not_called()


def test_save_as_existing_json_asks_before_overwriting(
    tmp_path: Path, monkeypatch,
) -> None:
    old, existing = tmp_path / "old.json", tmp_path / "existing.json"
    existing.write_text('{"leave": "untouched"}', encoding="utf-8")
    app = _output_app(old)
    monkeypatch.setattr(
        ui.filedialog, "asksaveasfilename", Mock(return_value=str(existing))
    )
    question = Mock(return_value=False)
    monkeypatch.setattr(ui.messagebox, "askyesno", question)
    app.change_json_output()
    question.assert_called_once()
    assert app.output == old
    assert app.output_var.get() == str(old)
    assert existing.read_text() == '{"leave": "untouched"}'
    assert not old.exists()
    app._commit.assert_not_called()


def test_save_as_companion_file_also_warns_before_overwriting(
    tmp_path: Path, monkeypatch,
) -> None:
    from p3d_categoriser_storage import incomplete_state_path
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    companion = incomplete_state_path(new)
    companion.write_text('{"previous": true}', encoding="utf-8")
    app = _output_app(old)
    monkeypatch.setattr(
        ui.filedialog, "asksaveasfilename", Mock(return_value=str(new))
    )
    monkeypatch.setattr(ui.messagebox, "askyesno", Mock(return_value=False))
    app.change_json_output()
    assert app.output == old
    assert companion.read_text() == '{"previous": true}'


def test_save_as_cancel_and_invalid_extension_preserve_original_output(
    tmp_path: Path, monkeypatch,
) -> None:
    old = tmp_path / "old.json"
    app = _output_app(old)
    choose = Mock(return_value="")
    monkeypatch.setattr(ui.filedialog, "asksaveasfilename", choose)
    app.change_json_output()
    assert app.output == old
    choose.return_value = str(tmp_path / "categories.txt")
    error = Mock()
    monkeypatch.setattr(ui.messagebox, "showerror", error)
    app.change_json_output()
    error.assert_called_once()
    assert not old.exists()


def test_save_as_failure_rolls_back_active_json_destination(
    tmp_path: Path, monkeypatch,
) -> None:
    old = tmp_path / "old.json"
    new = tmp_path / "new.json"
    app = _output_app(old)
    monkeypatch.setattr(
        ui.filedialog, "asksaveasfilename", Mock(return_value=str(new))
    )
    app.save_state = Mock(side_effect=OSError("write denied"))
    error = Mock()
    monkeypatch.setattr(ui.messagebox, "showerror", error)
    app.change_json_output()
    assert app.output == old
    assert app.output_var.get() == str(old)
    assert error.call_count == 1
