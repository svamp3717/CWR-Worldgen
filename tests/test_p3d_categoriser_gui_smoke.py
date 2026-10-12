"""Real Tk widget smoke test for categoriser browsing controls.

Run on a virtual display: xvfb-run -a pytest -q tests/test_p3d_categoriser_gui_smoke.py
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import measure_p3d_models as measure
from p3d_categoriser_app import CategoriserApp
from p3d_preview_geometry import PreviewModel
from p3d_texture_sources import TextureResolver


def _buttons(widget) -> list[str]:
    found = [widget.cget("text")] if isinstance(widget, ttk.Button) else []
    for child in widget.winfo_children():
        found.extend(_buttons(child))
    return found


@pytest.mark.skipif(
    sys.platform.startswith("linux") and not os.environ.get("DISPLAY"),
    reason="Tk smoke test requires DISPLAY (use xvfb-run)",
)
def test_categoriser_actual_tk_layout_search_and_texture_tools(tmp_path: Path) -> None:
    root = tk.Tk()
    try:
        points = np.asarray([
            (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0),
        ], dtype=np.float32)
        measurement = measure._measurement(
            model_path="ca\\structures\\shed.p3d",
            source="structures.ifa",
            format_name="ODOL",
            version=49,
            lod="first",
            points=points,
        )
        fixture = PreviewModel(
            "ca\\structures\\shed.p3d",
            "structures.ifa",
            measurement,
            points,
            (),
            np.empty((0, 2), dtype=np.int32),
            2,
            ("ca\\structures\\data\\wall_co.paa",),
            "vertex fallback",
        )
        app = CategoriserApp(
            root,
            model_iter=iter([fixture]),
            texture_resolver=TextureResolver(()),
            output=tmp_path / "categories.json",
            categories=("Buildings", "Nature"),
            state={},
            total_models=1,
        )
        root.update_idletasks()
        buttons = _buttons(root)
        for label in (
            "Find next", "Scanned models…", "Preview texture",
            "Copy texture path", "Add PBO / IFA…", "Add folder…",
            "Clear added texture sources", "Reset view",
            "Copy model path", "Zoom +", "Zoom −",
        ):
            assert label in buttons

        assert app.search_entry.winfo_exists()
        assert app.texture_list.size() == 1
        assert app._related_texture_paths == (
            "ca\\structures\\data\\wall_co.paa",
        )
        app.search_var.set("structures/shed")
        app.find_next_model()
        assert app.index == 0
        assert "Found model 1" in app.status_var.get()

        app.show_scanned_models()
        root.update_idletasks()
        assert app._model_list_popup.winfo_exists()
        app._model_list_popup.destroy()
        app._reset_view()
        assert (app.azim, app.elev, app.zoom) == (35.0, 25.0, 1.0)
    finally:
        root.destroy()
