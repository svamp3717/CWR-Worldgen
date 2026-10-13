"""Regression tests for fast P3D model and archive loading paths."""
from __future__ import annotations

import io
from pathlib import Path
import struct
import sys

import pytest

from cwr_worldgen import assets
from cwr_worldgen.pbo import PboEntry, write_pbo

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import measure_p3d_models as measure
from p3d_asset_browser_catalog import scan_catalogue


_ENTRY = struct.Struct("<IIIII")


def _archive(members: list[tuple[str, int, int, bytes]]) -> bytes:
    header = bytearray()
    payload = bytearray()
    for name, packing, original_size, stored in members:
        header.extend(name.encode("latin-1") + b"\0")
        header.extend(_ENTRY.pack(packing, original_size, 0, 0, len(stored)))
        payload.extend(stored)
    header.extend(b"\0" + _ENTRY.pack(0, 0, 0, 0, 0))
    return bytes(header + payload)


@pytest.mark.parametrize(
    "reader", [assets._decompress_lzss_stream, measure._decompress_lzss_stream]
)
def test_lzss_overlapping_backrefs_and_prefilled_window(reader) -> None:
    # One literal 'A', followed by a 5-byte offset-one overlapping reference.
    compressed = b"\x01A\x01\x02" + struct.pack("<I", 6 * ord("A"))
    stream = io.BytesIO(compressed + b"next block")
    assert reader(stream, 6) == b"AAAAAA"
    assert stream.read() == b"next block"

    # Reference to the space-filled dictionary before any literal is output.
    prefilled = b"\0\x01\0" + struct.pack("<I", 3 * 32)
    assert reader(io.BytesIO(prefilled), 3) == b"   "


@pytest.mark.parametrize(
    "reader", [assets._decompress_lzss_stream, measure._decompress_lzss_stream]
)
def test_lzss_rejects_checksum_mismatch_and_truncation(reader) -> None:
    compressed = b"\x01A\x01\x02"
    with pytest.raises(ValueError, match="checksum mismatch"):
        reader(io.BytesIO(compressed + struct.pack("<I", 1)), 6)
    with pytest.raises(ValueError, match="truncated LZSS reference"):
        reader(io.BytesIO(b"\0\x01"), 4)


def test_selecting_late_pbo_asset_seeks_past_unrelated_payloads() -> None:
    huge_texture = b"x" * (4 * 1024 * 1024)
    archive = _archive([
        ("textures\\huge.paa", 0, len(huge_texture), huge_texture),
        ("target.p3d", 0, 6, b"target"),
    ])

    class CountingStream(io.BytesIO):
        read_bytes = 0

        def read(self, size: int = -1) -> bytes:
            data = super().read(size)
            self.read_bytes += len(data)
            return data

    stream = CountingStream(archive)
    actual = assets._read_asset_from_pbo_stream(
        stream, target=r"models\target.p3d", fallback_prefix="models"
    )
    assert actual == b"target"
    assert stream.read_bytes < 1024  # Headers and target only, not 4 MiB.


def test_pbo_seek_still_rejects_truncated_skipped_payload() -> None:
    archive = _archive([
        ("texture.paa", 0, 9999, b"short"),
    ])
    # The stored-length field claims a payload past EOF.
    broken = archive.replace(_ENTRY.pack(0, 9999, 0, 0, 5), _ENTRY.pack(0, 9999, 0, 0, 9999), 1)
    with pytest.raises(ValueError, match="truncated"):
        assets._read_asset_from_pbo_stream(
            io.BytesIO(broken), target="models\\absent.p3d", fallback_prefix="models"
        )


def test_include_pattern_skips_unpacking_unwanted_p3ds(tmp_path: Path) -> None:
    source = tmp_path / "models.pbo"
    source.write_bytes(_archive([
        ("invalid.p3d", measure._PBO_COMPRESSED, 100, b"not-lzss"),
        ("target.p3d", 0, 6, b"target"),
    ]))
    assert measure.count_models((source,)) == 2
    matches = list(measure._iter_models((source,), (r"models\target.p3d",)))
    assert [(path, data) for path, _source, data in matches] == [
        (r"models\target.p3d", b"target")
    ]


def test_asset_browser_reuses_persistent_disk_catalogue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "cache"))
    source = tmp_path / "models.pbo"
    write_pbo(source, [PboEntry("house.p3d", b"ODOL" + b"\0" * 32)])

    first = scan_catalogue((source,))
    assert len(first.models) == 1
    cache_root = (
        (tmp_path / "cache" / "CWR-Worldgen" / "p3d-browser")
        if sys.platform == "win32" else
        (tmp_path / "cache" / "cwr-worldgen" / "p3d-browser")
    )
    assert list((cache_root / "assets").glob("*.json"))

    # Simulate a separate process: prove a disk snapshot is loaded without
    # touching the expensive full PBO scan.
    assets._CATALOGUE_MEMORY.clear()

    def no_rescan(_roots):
        raise AssertionError("unchanged source should use the disk cache")

    monkeypatch.setattr(assets, "_scan_catalogue", no_rescan)
    second = scan_catalogue((source,))
    assert second.models == first.models
    assets._CATALOGUE_MEMORY.clear()
