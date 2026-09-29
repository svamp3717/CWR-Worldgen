from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import p3d_categoriser_session as session


class _Var:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Axes:
    def clear(self):
        pass

    def imshow(self, *_args, **_kwargs):
        pass

    def set_title(self, value):
        self.title = value

    def axis(self, *_args):
        pass


class _Figure:
    def tight_layout(self, **_kwargs):
        pass


class _Canvas:
    def draw_idle(self):
        pass


def test_session_renderer_honors_skip_textures(monkeypatch) -> None:
    captured = {}

    def fake_render(*_args, **kwargs):
        captured.update(kwargs)
        return np.zeros((2, 2, 3), dtype=np.uint8), 0, 0

    monkeypatch.setattr(session, "render_textured_model", fake_render)

    app = session.SessionCategoriserApp.__new__(session.SessionCategoriserApp)
    app.ax_preview = _Axes()
    app.figure = _Figure()
    app.canvas = _Canvas()
    app.status_var = _Var("")
    app.skip_textures_var = _Var(True)
    app.texture_resolver = object()
    app.azim = 35.0
    app.elev = 25.0
    app.zoom = 1.0

    model = SimpleNamespace(
        points=np.zeros((3, 3), dtype=np.float32),
        faces=(object(),),
        source="mod.pbo.zst!addons\\inner.pbo!inner\\house.p3d",
    )

    app._draw_model(model)

    assert captured["load_textures"] is False
    assert "textures skipped" in app.ax_preview.title.casefold()
    assert "texture lookup skipped" in app.status_var.get().casefold()
