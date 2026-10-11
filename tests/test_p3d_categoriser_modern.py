"""Arma-era P3D categoriser parity: streaming archives, previews and textures."""
from __future__ import annotations

from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import zstandard

from cwr_worldgen.pbo import PboEntry, write_pbo

TESTS = Path(__file__).resolve().parent
TOOLS = TESTS.parent / "tools"
for root in (TESTS, TOOLS):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import categorise_p3d_models as cli
import measure_p3d_models as measure
import p3d_categoriser_app as ui
import p3d_categoriser_session as session
from p3d_preview_geometry import preview_models
from p3d_texture_render import MipmapCache
from p3d_texture_sources import TextureResolver
from test_p3d_modern_odol import _modern_odol
from test_p3d_modern_odol_mesh import full_odol


def _paa_red() -> bytes:
    return (
        struct.pack("<H", 0xFF01) + struct.pack("<H", 0)
        + struct.pack("<HH", 4, 4) + b"\x08\x00\x00"
        + struct.pack("<HHI", 0xF800, 0, 0)
    )


@pytest.mark.parametrize("version", [40, 49, 58, 64, 75])
@pytest.mark.parametrize("wrapped", [False, True])
def test_categoriser_reads_real_mesh_and_texture_from_arma_archive(
    tmp_path: Path, version: int, wrapped: bool,
) -> None:
    archive = tmp_path / "ca.ifa"
    write_pbo(
        archive,
        (PboEntry("building.p3d", full_odol(version)),
         PboEntry(r"data\brick.paa", _paa_red())),
    )
    if wrapped:
        zipped = tmp_path / "ca.ifa.zst"
        zipped.write_bytes(zstandard.ZstdCompressor(level=1).compress(archive.read_bytes()))
        archive.unlink()
        archive = zipped

    original = archive.read_bytes()
    assert measure.count_models((archive,)) == 1
    previews = list(preview_models((archive,), (), 1000))
    assert len(previews) == 1
    model = previews[0]
    assert not isinstance(model, measure.ModelFailure)
    assert model.model_path == r"ca\building.p3d"
    assert model.measurement.version == version
    assert model.render_mode == "textured mesh"
    assert len(model.faces) == 2
    assert model.original_vertex_count == 4
    assert model.faces[0].texture_path == r"ca\data\brick.paa"

    resolver = cli.TextureResolver((archive,))
    assert isinstance(resolver, TextureResolver)
    image = resolver.load_rgba(r"ca\data\brick.paa", model.source)
    assert image is not None and image.shape == (4, 4, 4)
    assert tuple(image[0, 0]) == (255, 0, 0, 255)
    assert archive.read_bytes() == original


def test_categoriser_counts_nested_ifa_and_reads_its_texture(tmp_path: Path) -> None:
    inner = tmp_path / "lib_models.ifa"
    write_pbo(inner, (
        PboEntry("bunker.p3d", full_odol(49)),
        PboEntry(r"data\brick.paa", _paa_red()),
    ))
    outer = tmp_path / "iron_front_package.pbo"
    write_pbo(outer, (PboEntry(r"addons\lib_models.ifa", inner.read_bytes()),))
    original = outer.read_bytes()

    assert measure.count_models((outer,)) == 1
    model, = list(preview_models((outer,), (), 1000))
    assert not isinstance(model, measure.ModelFailure)
    assert model.model_path == r"lib_models\bunker.p3d"
    assert r"!addons\lib_models.ifa!" in model.source
    assert len(model.faces) == 2
    assert outer.read_bytes() == original


def test_categoriser_falls_back_to_labeled_modern_bounds_only(tmp_path: Path) -> None:
    archive = tmp_path / "a3.pbo"
    write_pbo(archive, (PboEntry("model.p3d", _modern_odol(49)),))
    model, = list(preview_models((archive,), (), 100))
    assert not isinstance(model, measure.ModelFailure)
    assert model.render_mode == "bounds only"
    assert model.original_vertex_count == 0
    assert model.points.shape == (8, 3)
    assert model.edges.shape == (12, 2)
    assert model.faces == ()


def _app_mock(*, width: int = 640, height: int = 400):
    app = ui.CategoriserApp.__new__(ui.CategoriserApp)
    app.ax_preview = Mock()
    app.figure = Mock()
    app.canvas = Mock()
    app.canvas.get_tk_widget.return_value.winfo_width.return_value = width
    app.canvas.get_tk_widget.return_value.winfo_height.return_value = height
    app.skip_textures_var = SimpleNamespace(get=lambda: False)
    app.status_var = Mock()
    app.texture_resolver = object()
    app.mipmap_cache = MipmapCache(max_bytes=1024)
    app.azim = 35.0
    app.elev = 25.0
    return app


def test_categoriser_uses_adaptive_resolution_filtered_renderer_and_cache() -> None:
    app = _app_mock()
    captured = {}

    def render(*args, **kwargs):
        captured.update(kwargs)
        return np.zeros((500, 800, 3), dtype=np.uint8), 2, 0

    fake = SimpleNamespace(
        points=np.zeros((4, 3), dtype=np.float32),
        faces=(object(), object()),
        source="lib_models.ifa!lib_models\\house.p3d",
    )
    app._render_model(fake, renderer=render, zoom=1.75)
    assert captured["width"] == 800
    assert captured["height"] == 500
    assert captured["zoom"] == 1.75
    assert captured["load_textures"] is True
    assert captured["mipmap_cache"] is app.mipmap_cache
    assert app.ax_preview.imshow.call_args.kwargs == {
        "interpolation": "hanning", "resample": True,
    }
    app.status_var.set.assert_any_call("Texture sampling: 2 textured triangle(s).")


def test_categoriser_uses_wireframe_not_fake_faces_for_unsupported_modern_lod() -> None:
    app = _app_mock()
    from p3d_modern_odol import OdolBounds, BOX_EDGES, box_vertices

    bounds = OdolBounds(49, (-1, -2, -3), (4, 5, 6), 1)
    fake = SimpleNamespace(
        render_mode="bounds only",
        points=np.asarray(box_vertices(bounds), dtype=np.float32),
        edges=np.asarray(BOX_EDGES, dtype=np.int32),
        measurement=SimpleNamespace(version=49),
        faces=(),
    )
    def never_render(*args, **kwargs):
        raise AssertionError("must not render a bounding box as a textured mesh")

    app._render_model(fake, renderer=never_render)
    assert app.ax_preview.plot.call_count == 12
    assert "bounds only" in app.ax_preview.set_title.call_args.args[0]
    assert "NOT model geometry" in app.ax_preview.set_title.call_args.args[0]
    app.ax_preview.imshow.assert_not_called()
    assert app.canvas.draw_idle.call_count == 1


def test_categoriser_keeps_zoom_and_reuses_renderer_in_session(monkeypatch) -> None:
    app = session.SessionCategoriserApp.__new__(session.SessionCategoriserApp)
    captures = []

    def fake_shared(model, *, renderer, zoom):
        captures.append((model, renderer, zoom))

    app._render_model = fake_shared
    app.zoom = 2.25
    fake = object()
    app._draw_model(fake)
    assert captures == [(fake, session.render_textured_model, 2.25)]


def test_categoriser_cli_describes_multiple_engines_and_ifa_archives() -> None:
    help_text = cli._parser().format_help()
    assert "Iron Front" in help_text
    assert "Arma" in help_text
    assert "IFA" in help_text
    assert "--texture-source" in help_text
