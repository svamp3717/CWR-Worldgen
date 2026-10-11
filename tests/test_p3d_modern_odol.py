"""Regression tests for Arma-era ODOL metadata and bounds-only previews."""
from __future__ import annotations

from pathlib import Path
import struct
import sys

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import p3d_modern_odol as modern
import p3d_preview_geometry as preview


def _modern_odol(
    version: int,
    *,
    bounds_min=(-1.0, -2.0, -3.0),
    bounds_max=(4.0, 5.0, 6.0),
    lod_count=2,
    encrypted=False,
) -> bytes:
    out = bytearray(b"ODOL" + struct.pack("<I", version))
    if version >= 75:
        out.extend(struct.pack("<II", int(encrypted), 0))
    if version >= 59:
        out.extend(struct.pack("<I", 1234))  # AppID
    if version >= 58:
        out.extend(b"muzzle\\flash.p3d\\0")
    out.extend(struct.pack("<I", lod_count))
    out.extend(struct.pack("<" + "f" * lod_count, *([1.0] * lod_count)))
    out.extend(b"\\0" * 48)  # ModelInfo preceding BboxMin
    out.extend(struct.pack("<6f", *bounds_min, *bounds_max))
    return bytes(out)


@pytest.mark.parametrize("version", [40, 43, 47, 48, 49, 50, 52, 56, 58, 60, 64, 69, 73, 74, 75])
def test_modern_odol_bounds_support_arma_versions(version: int) -> None:
    data = _modern_odol(version)
    bounds = modern.read_model_bounds(data)
    assert bounds.version == version
    assert bounds.lod_count == 2
    assert bounds.minimum == (-1.0, -2.0, -3.0)
    assert bounds.maximum == (4.0, 5.0, 6.0)

    measurement, points, faces, textures = preview.extract_first_lod_geometry(
        data, model_path=r"a3\\test.p3d", source="fixture",
    )
    assert measurement.format == "ODOL"
    assert measurement.version == version
    assert measurement.lod == "model bounds only"
    assert measurement.vertex_count == 0
    assert (measurement.width_m, measurement.height_m, measurement.length_m) == (5.0, 7.0, 9.0)
    assert points.shape == (8, 3)
    assert len(faces) == 0
    assert textures == ()
    assert np.array_equal(points[0], (-1.0, -2.0, -3.0))
    assert np.array_equal(points[6], (4.0, 5.0, 6.0))


def test_bounding_box_has_twelve_edges() -> None:
    assert len(modern.BOX_EDGES) == 12
    assert all(0 <= a < 8 and 0 <= b < 8 for a, b in modern.BOX_EDGES)


@pytest.mark.parametrize("bad", [
    b"BAD!" + b"\\0" * 80,
    b"ODOL" + struct.pack("<I", 49),
    _modern_odol(49)[:-6],
    _modern_odol(49, bounds_min=(10.0, 0.0, 0.0)),
    _modern_odol(49, bounds_max=(float("nan"), 1.0, 2.0)),
    _modern_odol(49, lod_count=0),
    _modern_odol(49, lod_count=129),
    _modern_odol(75, encrypted=True),
])
def test_modern_odol_rejects_invalid_headers(bad: bytes) -> None:
    with pytest.raises(modern.ModernOdolError):
        modern.read_model_bounds(bad)


def test_v7_reader_stays_on_legacy_path() -> None:
    with pytest.raises(Exception, match="truncated|implausible"):
        preview.extract_first_lod_geometry(
            b"ODOL" + struct.pack("<II", 7, 1),
            model_path="legacy.p3d", source="fixture",
        )


def test_v49_model_header_is_not_modified() -> None:
    data = _modern_odol(49)
    original = bytes(data)
    modern.read_model_bounds(data)
    assert data == original
