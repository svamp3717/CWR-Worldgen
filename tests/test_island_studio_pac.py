"""PAC texture previews for classic OFP/CWA island terrain."""
from __future__ import annotations

import io
from pathlib import Path
import struct

import numpy as np
import pytest

from cwr_island_studio.pbo_textures import PboTextures
from cwr_island_studio.textures import (
    _image_tile, _pbo_image_tile,
    clear_archive_texture_cache, clear_loose_texture_cache,
    decode_pac_preview, render_real_textures,
)


def _pac(*, lzw: bool = False) -> bytes:
    """Make an original, paletted PAC with a green/red checkerboard."""
    # PAC palette bytes are stored blue, green, red.
    palette = bytes((0, 0, 255, 0, 255, 0))
    indices = bytes((0, 1, 1, 0, 0, 1, 1, 0))
    if lzw:
        # Legacy 4K-window LZW block: eight literal indices + signed checksum.
        payload = bytes((0xFF,)) + indices + struct.pack("<i", sum(indices))
        dimensions = struct.pack("<HHHH", 1234, 8765, 4, 2)
    else:
        # PAC RLE control 7: 8 literal palette indices.
        payload = bytes((7,)) + indices
        dimensions = struct.pack("<HH", 4, 2)
    return (
        struct.pack("<H", 2) + palette + dimensions
        + len(payload).to_bytes(3, "little") + payload
        + struct.pack("<HH", 0, 0)
    )


def _pbo(path: Path, name: str, data: bytes) -> None:
    raw = io.BytesIO()
    raw.write(name.encode("ascii") + b"\0")
    raw.write(struct.pack("<IIIII", 0, 0, 0, 0, len(data)))
    raw.write(b"\0" + struct.pack("<IIIII", 0, 0, 0, 0, 0))
    raw.write(data)
    path.write_bytes(raw.getvalue())


@pytest.mark.parametrize("compressed", (False, True))
def test_decodes_pac_palette_and_legacy_lzw(compressed: bool) -> None:
    image = decode_pac_preview(_pac(lzw=compressed))
    assert image.mode == "RGB"
    assert image.size == (4, 2)
    assert image.getpixel((0, 0)) == (255, 0, 0)
    assert image.getpixel((1, 0)) == (0, 255, 0)
    assert image.getpixel((2, 1)) == (0, 255, 0)


def test_loose_pac_is_visible_in_actual_textures_mode(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    directory = root / "Eden" / "bak"
    directory.mkdir(parents=True)
    (directory / "bah.pac").write_bytes(_pac())
    slots = np.zeros((1, 4), dtype=np.int32)
    fallback = np.full((1, 4, 3), 11, dtype=np.uint8)
    try:
        result, loaded = render_real_textures(
            slots, np.array([0, 1, 2, 3]), np.array([0]),
            (r"Eden\bak\bah.pac",), root, fallback, 4.0,
        )
        assert loaded == 1
        assert result[0, 0].tolist() == [255, 0, 0]
        assert result[0, 1].tolist() == [0, 255, 0]
        assert result[0, 3].tolist() == [255, 0, 0]
    finally:
        clear_loose_texture_cache()


def test_pac_inside_pbo_is_visible_without_extraction(tmp_path: Path) -> None:
    archive = tmp_path / "eden.pbo"
    _pbo(archive, r"bak\bah.pac", _pac(lzw=True))
    reader = PboTextures((archive,))
    try:
        assert reader.read(r"eden\bak\bah.pac") == _pac(lzw=True)
        slots = np.array([[0, 0]], dtype=np.int32)
        fallback = np.full((1, 2, 3), 11, dtype=np.uint8)
        result, loaded = render_real_textures(
            slots, np.array([0, 1]), np.array([0]),
            (r"eden\bak\bah.pac",), None, fallback, 4.0,
            archives=reader,
        )
        assert loaded == 1
        assert result[0, 0].tolist() == [255, 0, 0]
        assert result[0, 1].tolist() == [0, 255, 0]
    finally:
        clear_archive_texture_cache()
        reader.close()


def test_invalid_pac_falls_back_to_texture_slot_colour(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    (root / "bad.pac").write_bytes(b"\x02\x00\x01")
    slots = np.zeros((1, 1), dtype=np.int32)
    fallback = np.full((1, 1, 3), 91, dtype=np.uint8)
    try:
        result, loaded = render_real_textures(
            slots, np.array([0]), np.array([0]),
            ("bad.pac",), root, fallback, 4.0,
        )
        assert loaded == 0
        np.testing.assert_array_equal(result, fallback)
    finally:
        clear_loose_texture_cache()


def test_untrusted_pac_paths_do_not_escape_asset_roots(tmp_path: Path) -> None:
    slots = np.zeros((1, 1), dtype=np.int32)
    fallback = np.full((1, 1, 3), 45, dtype=np.uint8)
    result, loaded = render_real_textures(
        slots, np.array([0]), np.array([0]),
        (r"..\bad.pac",), tmp_path, fallback, 4.0,
    )
    assert loaded == 0
    np.testing.assert_array_equal(result, fallback)


def _dxt1_paa(red: bool = True) -> bytes:
    colour = 0xF800 if red else 0x07E0
    payload = struct.pack("<HHI", colour, colour, 0)
    return (
        struct.pack("<H", 0xFF01) + struct.pack("<H", 0)
        + struct.pack("<HH", 4, 4) + len(payload).to_bytes(3, "little")
        + payload + struct.pack("<HH", 0, 0)
    )


def test_typed_dxt_pac_uses_same_decoder_as_paa() -> None:
    # Not every .pac has the original P8 palette header.
    from cwr_island_studio.textures import decode_paa_preview
    pac = _dxt1_paa()
    assert decode_pac_preview(pac).getpixel((0, 0)) == (255, 0, 0)
    assert decode_paa_preview(pac).getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("wrp_suffix,archive_suffix", [
    (".pac", ".paa"), (".paa", ".pac"),
])
def test_pbo_texture_extension_alias_resolves_both_ways(
    tmp_path: Path, wrp_suffix: str, archive_suffix: str,
) -> None:
    archive = tmp_path / "eden.pbo"
    raw = _pac() if archive_suffix == ".pac" else _dxt1_paa()
    _pbo(archive, "bak/grass" + archive_suffix, raw)
    reader = PboTextures((archive,))
    try:
        fallback = np.full((1, 1, 3), 27, dtype=np.uint8)
        diagnostics = {}
        pixels, count = render_real_textures(
            np.zeros((1, 1), dtype=np.int32),
            np.array([0]), np.array([0]),
            ("eden\\bak\\grass" + wrp_suffix,),
            None, fallback, 4.0, archives=reader,
            diagnostics=diagnostics,
        )
        assert count == 1 and diagnostics == {}
        assert pixels[0, 0].tolist() == [255, 0, 0]
    finally:
        clear_archive_texture_cache()
        reader.close()


def test_newest_loose_asset_wins_over_pbo_even_with_extension_alias(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "eden.pbo"
    _pbo(archive, r"bak\\field.pac", _pac())
    directory = tmp_path / "loose" / "eden" / "bak"
    directory.mkdir(parents=True)
    (directory / "field.paa").write_bytes(_dxt1_paa(red=False))
    reader = PboTextures((archive,))
    try:
        result, count = render_real_textures(
            np.zeros((1, 1), dtype=np.int32),
            np.array([0]), np.array([0]), (r"eden\\bak\\field.pac",),
            tmp_path / "loose", np.zeros((1, 1, 3), dtype=np.uint8),
            4.0, archives=reader,
        )
        assert count == 1
        assert result[0, 0].tolist() == [0, 255, 0]
    finally:
        clear_loose_texture_cache()
        clear_archive_texture_cache()
        reader.close()


def test_texture_diagnostics_distinguish_missing_from_undecodable(
    tmp_path: Path,
) -> None:
    (tmp_path / "unreadable.pac").write_bytes(b"broken")
    paths = ("unreadable.pac", "missing.pac")
    fallback = np.full((1, 2, 3), 42, dtype=np.uint8)
    diagnostics = {}
    try:
        result, count = render_real_textures(
            np.array([[0, 1]], dtype=np.int32),
            np.array([0, 4]), np.array([0]), paths,
            tmp_path, fallback, 4.0, diagnostics=diagnostics,
        )
        assert count == 0
        np.testing.assert_array_equal(result, fallback)
        assert diagnostics == {
            "unreadable.pac": "unreadable", "missing.pac": "missing",
        }
    finally:
        clear_loose_texture_cache()


def test_missing_texture_is_diagnosed_even_without_selected_assets() -> None:
    diagnostics = {}
    fallback = np.full((1, 1, 3), 12, dtype=np.uint8)
    result, count = render_real_textures(
        np.zeros((1, 1), dtype=np.int32),
        np.array([0]), np.array([0]), ("o\\missing.pac",),
        None, fallback, 4.0, diagnostics=diagnostics,
    )
    assert count == 0
    assert diagnostics == {"o\\missing.pac": "missing"}
    np.testing.assert_array_equal(result, fallback)
