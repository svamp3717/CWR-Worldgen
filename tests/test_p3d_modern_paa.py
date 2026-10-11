"""Modern Arma PAA: LZO flag in DXT mip widths and genuine LZO matches.

The LZO fixtures below are independently produced by liblzo2's lzo1x_1_compress,
not by our decoder, and represent repeated DXT blocks.
"""
from __future__ import annotations

from pathlib import Path
import struct
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from p3d_lzo import LzoReadError, decode_lzo_bytes
from p3d_texture_io import decode_paa


_RED_DXT1 = struct.pack("<HHI", 0xF800, 0x0000, 0)
_GREEN_DXT5 = b"\xff\x00" + b"\x00" * 6 + struct.pack("<HHI", 0x07E0, 0x0000, 0)

# 16x16 DXT1 compressed with liblzo2, including literal and backreference runs.
_LZO_RED_DXT1 = bytes.fromhex(
    "0500f800000000000020431c0000020000000000f800000000000000f8000000000000110000"
)
_LZO_GREEN_DXT5 = bytes.fromhex(
    "08ff00000000000000e00700800020bb3c00000200000000ff00000000000000"
    "e007000000000000110000"
)
# One DXT1 block independently LZO-compressed (compression can increase size).
_LZO_TINY_DXT1 = bytes.fromhex("1900f8000000000000110000")


def _paa(format_code: int, width_field: int, height: int, payload: bytes, *, tag: bool = False) -> bytes:
    data = bytearray(struct.pack("<H", format_code))
    if tag:
        data.extend(b"GGATCGVA" + struct.pack("<I", 4) + b"\x11\x22\x33\x44")
        data.extend(b"GGATSFFO" + struct.pack("<I", 64) + b"\x00" * 64)
    data.extend(struct.pack("<H", 0))  # DXT palette is empty
    data.extend(struct.pack("<HH", width_field, height))
    data.extend(len(payload).to_bytes(3, "little"))
    data.extend(payload)
    return bytes(data)


def test_reference_lzo_decompresses_real_backreferences() -> None:
    assert decode_lzo_bytes(_LZO_RED_DXT1, 128) == _RED_DXT1 * 16
    assert decode_lzo_bytes(_LZO_GREEN_DXT5, 256) == _GREEN_DXT5 * 16


def test_arma2_lzo_compressed_dxt1_mip_with_width_flag() -> None:
    # 0x8010 is 16 pixels wide, not 32784 pixels wide.
    image = decode_paa(_paa(0xFF01, 0x8010, 16, _LZO_RED_DXT1, tag=True))
    assert image.shape == (16, 16, 4)
    assert np.array_equal(image[0, 0], [255, 0, 0, 255])
    assert np.array_equal(image[-1, -1], [255, 0, 0, 255])


def test_arma3_lzo_compressed_dxt5_mip_with_width_flag() -> None:
    image = decode_paa(_paa(0xFF05, 0x8010, 16, _LZO_GREEN_DXT5))
    assert image.shape == (16, 16, 4)
    assert np.array_equal(image[0, 0], [0, 255, 0, 255])


def test_lzo_compressed_small_texture_and_legacy_uncompressed_dxt() -> None:
    flagged = decode_paa(_paa(0xFF01, 0x8004, 4, _LZO_TINY_DXT1))
    legacy = decode_paa(_paa(0xFF01, 4, 4, _RED_DXT1))
    assert np.array_equal(flagged, legacy)


@pytest.mark.parametrize("format_code", [0xFF01, 0xFF03, 0xFF05])
def test_oversized_width_is_recognized_as_modern_compression(format_code: int) -> None:
    # A modern 0x8400 width must reach the LZO decoder, not fail as 33792.
    with pytest.raises(ValueError, match="LZO"):
        decode_paa(_paa(format_code, 0x8400, 1024, b"\x11"))


@pytest.mark.parametrize("payload", [
    b"",
    _LZO_RED_DXT1[:-1],
    _LZO_RED_DXT1 + b"\x00",
    b"\x12\x00\x00",
])
def test_malformed_lzo_paa_blocks_are_rejected(payload: bytes) -> None:
    with pytest.raises(ValueError, match="LZO"):
        decode_paa(_paa(0xFF01, 0x8010, 16, payload))


def test_non_dxt_formats_cannot_use_dxt_lzo_width_flag() -> None:
    with pytest.raises(ValueError, match="invalid PAA dimensions"):
        decode_paa(_paa(0x4444, 0x8004, 4, b"\x00"))


def test_raw_dxt_payload_length_is_validated() -> None:
    with pytest.raises(ValueError, match="raw DXT1 PAA mip length"):
        decode_paa(_paa(0xFF01, 16, 16, _RED_DXT1))
