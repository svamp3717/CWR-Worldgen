"""Navigation and on-demand PBO texture indexing regression coverage."""
from __future__ import annotations

from pathlib import Path
import io
import struct

import numpy as np
import pytest
from PIL import Image
import zstandard as zstd

from cwr_island_studio.navigation import zoom_around
from cwr_island_studio.pbo_textures import PboTextures, canonical_path
from cwr_island_studio.textures import (
    _pbo_image_tile, clear_archive_texture_cache, render_real_textures,
)


def _make_pbo(path: Path, items: list[tuple[str, bytes]],
              *, prefix: str | None = None) -> None:
    raw = io.BytesIO()
    if prefix:
        raw.write(b"\0")
        raw.write(struct.pack("<IIIII", 0x56657273, 0, 0, 0, 0))
        raw.write(b"prefix\0" + prefix.encode("ascii") + b"\0\0")
    for name, payload in items:
        raw.write(name.encode("ascii") + b"\0")
        raw.write(struct.pack("<IIIII", 0, 0, 0, 0, len(payload)))
    raw.write(b"\0")
    raw.write(struct.pack("<IIIII", 0, 0, 0, 0, 0))
    for _, payload in items:
        raw.write(payload)
    path.write_bytes(raw.getvalue())


def _png(red: int, green: int, blue: int) -> bytes:
    image = Image.new("RGB", (8, 8), (red, green, blue))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_pointer_zoom_preserves_exact_world_coordinate() -> None:
    left, top, zoom = 1270.0, 250.0, 0.30
    mouse_x, mouse_y = 511.0, 263.0
    old_world = left + mouse_x / zoom, top + mouse_y / zoom
    new_left, new_top, new_zoom = zoom_around(
        left, top, zoom, mouse_x, mouse_y, 1.25, .05, 8.0,
    )
    assert new_zoom == pytest.approx(0.375)
    assert (new_left + mouse_x / new_zoom, new_top + mouse_y / new_zoom) == pytest.approx(old_world)
    back = zoom_around(new_left, new_top, new_zoom, mouse_x, mouse_y, .8, .05, 8.0)
    assert back == pytest.approx((left, top, zoom))
    clamped = zoom_around(left, top, zoom, mouse_x, mouse_y, 50, .05, 1)
    assert clamped[2] == 1
    assert clamped[0] + mouse_x / clamped[2] == pytest.approx(old_world[0])


def test_invalid_zoom_rejected() -> None:
    with pytest.raises(ValueError):
        zoom_around(0, 0, 0, 10, 20, 1.25, .01, 8)
    with pytest.raises(ValueError):
        zoom_around(0, 0, 1, 10, 20, float("nan"), .01, 8)


def test_indexed_pbo_and_fallback_colours(tmp_path: Path) -> None:
    archive = tmp_path / "o.pbo"
    red = _png(244, 20, 25)
    _make_pbo(archive, [(r"data\grass.png", red), ("not_texture.txt", b"unrelated")])
    reader = PboTextures((archive,))
    try:
        assert set(reader.members) == {r"o\data\grass.png"}
        assert reader.read(r"O/DATA/GRASS.PNG") == red
        assert reader.read(r"o\data\nope.png") is None
        assert reader.read(r"..\unrelated.png") is None
        slots = np.array([[0, 1]], dtype=np.int32)
        fallback = np.full((1, 2, 3), 9, dtype=np.uint8)
        result, count = render_real_textures(
            slots, np.array([0.0, 50.0]), np.array([0.0]),
            (r"o\data\grass.png", r"o\data\missing.png"),
            None, fallback, 50.0, archives=reader,
        )
        assert count == 1
        assert result[0, 0].tolist() == [244, 20, 25]
        assert result[0, 1].tolist() == [9, 9, 9]
    finally:
        clear_archive_texture_cache()
        reader.close()


def test_pbo_prefix_override_and_multiple_archive_priority(tmp_path: Path) -> None:
    older = tmp_path / "old.pbo"
    newer = tmp_path / "new.pbo"
    first, second = _png(10, 11, 12), _png(210, 211, 212)
    _make_pbo(older, [(r"textures\field.png", first)], prefix="lands")
    _make_pbo(newer, [(r"textures\field.png", second)], prefix="lands")
    reader = PboTextures((older, newer))
    try:
        assert reader.read(r"lands\textures\field.png") == second
        assert len(reader.members) == 1
    finally:
        reader.close()


def test_zstd_wrapped_pbo_and_invalid_header(tmp_path: Path) -> None:
    basic = tmp_path / "land.pbo"
    _make_pbo(basic, [(r"data\grass.png", _png(2, 3, 4))])
    wrapped = tmp_path / "land.pbo.zst"
    wrapped.write_bytes(zstd.ZstdCompressor().compress(basic.read_bytes()))
    reader = PboTextures((wrapped,))
    try:
        assert reader.read(r"land\data\grass.png") == _png(2, 3, 4)
    finally:
        reader.close()
    corrupt = tmp_path / "corrupt.pbo"
    corrupt.write_bytes(b"\0\0\0")
    with pytest.raises(ValueError):
        PboTextures((corrupt,))


def test_reject_archive_path_traversal() -> None:
    for unsafe in (r"..\evil.png", r".\evil.png", r"c:\win.png", r"a\\b.png"):
        with pytest.raises(ValueError):
            canonical_path(unsafe)
