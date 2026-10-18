"""File-format metadata regression tests for P3D Browser models and textures."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import struct
import sys

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from cwr_worldgen import assets
from cwr_worldgen.pbo import PboEntry, write_pbo
import p3d_asset_browser_app as browser
from p3d_asset_browser_catalog import TexturePreviewCache, scan_catalogue
from p3d_texture_io import inspect_paa_format


def _paa(format_code: int, width: int, height: int, payload: bytes) -> bytes:
    return (
        struct.pack("<HHHH", format_code, 0, width, height)
        + len(payload).to_bytes(3, "little")
        + payload
    )


def test_dxt1_format_reports_real_stored_mips() -> None:
    first = _paa(0xFF01, 4, 4, b"\x00" * 8)
    second = struct.pack("<HH", 2, 2) + (8).to_bytes(3, "little") + b"\x00" * 8
    info = inspect_paa_format(first + second + b"\x00" * 4)
    assert info.container == "PAA"
    assert info.encoding == "DXT1"
    assert info.compression == "DXT1 blocks"
    assert info.mip_sizes == ((4, 4), (2, 2))
    assert "Mipmaps: 2" in info.readout()
    assert not info.inferred_encoding


def test_modern_lzo_flag_is_identified_without_decompressing_payload() -> None:
    info = inspect_paa_format(_paa(0xFF05, 0x8004, 4, b"compressed"))
    assert info.encoding == "DXT5"
    assert info.size == (4, 4)
    assert info.compression == "DXT5 blocks + LZO"


def test_classic_pac_palette_format_is_reported() -> None:
    # Legacy PAC has no leading recognized format marker. Its first two bytes
    # are the palette size, followed by RGB palette entries and stored mips.
    pac = struct.pack("<H", 2) + bytes((0, 0, 0, 255, 255, 255))
    pac += struct.pack("<HH", 2, 2) + (4).to_bytes(3, "little")
    pac += b"\x00\x01\x00\x01" + b"\x00" * 4
    info = inspect_paa_format(pac, is_paa=False)
    assert info.container == "PAC"
    assert info.encoding == "P8"
    assert info.palette_entries == 2
    assert info.inferred_encoding
    assert "Palette: 2 colours" in info.readout()


def test_preview_cache_keeps_format_details_without_second_read(
    tmp_path: Path, monkeypatch,
) -> None:
    pbo = tmp_path / "textures.pbo"
    write_pbo(pbo, (PboEntry("wall.paa", _paa(0xFF01, 4, 4, b"\x00" * 8)),))
    catalogue = scan_catalogue((pbo,), use_cache=False)
    item, = catalogue.textures
    cache = TexturePreviewCache()
    reads: list[str] = []
    original_read = assets.read_asset_record_bytes

    def counted(record):
        reads.append(record.path)
        return original_read(record)

    monkeypatch.setattr(assets, "read_asset_record_bytes", counted)
    preview, info = cache.get(item, catalogue, with_format=True)
    second, cached_info = cache.get(item, catalogue, with_format=True)
    assert preview is second
    assert info is cached_info
    assert info is not None
    assert info.encoding == "DXT1"
    assert len(reads) == 1
    cache.clear()
    assert not cache._formats


def test_model_details_show_binary_format_lods_and_texture_types() -> None:
    app = browser.AssetBrowserApp.__new__(browser.AssetBrowserApp)
    app._busy, app._unbusy = Mock(), Mock()
    app._update_lod_picker, app._draw_model = Mock(), Mock()
    app._set_related_textures = Mock()
    app.info_var = SimpleNamespace(set=Mock())
    app._lod_model_key = None
    app._lod_default_quality = "fast"
    app._lod_current_quality = "fast"
    item = SimpleNamespace(source="assets.pbo", path="town.p3d", size=2500)
    measurement = SimpleNamespace(
        format="ODOL", version=7, width_m=3.0,
        length_m=4.0, height_m=5.0, lod="first",
    )
    app.model_cache = SimpleNamespace(get=lambda *_args, **_kw: SimpleNamespace(
        measurement=measurement,
        available_lods=(SimpleNamespace(visual=True), SimpleNamespace(visual=False)),
        texture_paths=(r"town\wall.paa", r"town\roof.pac"),
        render_mode="mesh", faces=(object(),), original_vertex_count=40,
    ))
    app.catalogue = object()
    app._show_model(item)
    info = app.info_var.set.call_args.args[0]
    assert "Format: ODOL v7" in info
    assert "Storage: ODOL (binarized game model)" in info
    assert "LOD records: 2 (1 visual, 1 special)" in info
    assert "Texture references: 1 PAA, 1 PAC" in info


def test_texture_details_show_codec_and_selected_preview_mip(
    tmp_path: Path,
) -> None:
    pbo = tmp_path / "textures.pbo"
    write_pbo(pbo, (PboEntry("wall.paa", _paa(0xFF01, 4, 4, b"\x00" * 8)),))
    catalogue = scan_catalogue((pbo,), use_cache=False)
    item, = catalogue.textures
    app = browser.AssetBrowserApp.__new__(browser.AssetBrowserApp)
    app._busy, app._unbusy = Mock(), Mock()
    app.ax_preview, app.figure, app.canvas = Mock(), Mock(), Mock()
    app._set_related_models = Mock()
    app.catalogue = catalogue
    app.info_var = SimpleNamespace(set=Mock())
    app.load_timing_var = SimpleNamespace(set=Mock())
    app.texture_preview_cache = TexturePreviewCache()
    app._mipmap_choices = {"Full (mip 0)": "auto"}
    app.texture_mipmap_var = SimpleNamespace(get=lambda: "Full (mip 0)")
    app._show_texture(item)
    info = app.info_var.set.call_args.args[0]
    assert "Format: PAA / DXT1" in info
    assert "Compression: DXT1 blocks" in info
    assert "Stored resolution: 4 × 4" in info
    assert "Preview mip: 4 × 4" in info
