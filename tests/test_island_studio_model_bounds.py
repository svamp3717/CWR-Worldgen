"""Object footprint sizing from actual P3D geometry and WRP transforms."""
from __future__ import annotations

from pathlib import Path
import math
import struct

import pytest

from cwr_island_studio.model_bounds import (
    ModelBounds, read_model_bounds, load_model_bounds,
    footprint_corners, footprint_contains,
)
from cwr_island_studio.pbo_textures import PboTextures


def _mlod(points: tuple[tuple[float, float, float], ...]) -> bytes:
    return (
        b"MLOD" + struct.pack("<II", 1, 1)
        + b"SP3X" + struct.pack("<6I", 28, 0, len(points), 0, 0, 0)
        + b"".join(struct.pack("<fffI", *point, 0) for point in points)
    )


def _literal_lzss(raw: bytes) -> bytes:
    encoded = bytearray()
    for start in range(0, len(raw), 8):
        group = raw[start:start + 8]
        encoded.append((1 << len(group)) - 1)
        encoded.extend(group)
    return bytes(encoded) + struct.pack("<I", sum(raw) & 0xFFFFFFFF)


def _odol(points: tuple[tuple[float, float, float], ...], version: int = 7) -> bytes:
    count = len(points)
    flags = b"\0" * (count * 4)
    uv = b"\0" * (count * 8)
    return (
        b"ODOL" + struct.pack("<II", version, 1)
        + struct.pack("<I", count)
        + (_literal_lzss(flags) if len(flags) >= 1024 else flags)
        + struct.pack("<I", count)
        + (_literal_lzss(uv) if len(uv) >= 1024 else uv)
        + struct.pack("<I", count)
        + b"".join(struct.pack("<fff", *point) for point in points)
    )


_POINTS = (
    (-2.0, -5.0, -6.0),
    (8.0, 2.0, -6.0),
    (8.0, 4.0, 4.0),
    (-2.0, 4.0, 4.0),
)


@pytest.mark.parametrize("payload", [_mlod(_POINTS), _odol(_POINTS), _odol(_POINTS, 6)])
def test_native_p3d_formats_provide_origin_relative_world_metres(payload: bytes) -> None:
    actual = read_model_bounds(payload)
    assert actual == ModelBounds(-2.0, 8.0, -6.0, 4.0)
    assert actual.width == 10.0
    assert actual.length == 10.0


def test_compressed_odol_first_lod_measurement() -> None:
    points = _POINTS * 70  # Larger UV/flag arrays use legacy compressed blocks.
    assert read_model_bounds(_odol(points)) == ModelBounds(-2.0, 8.0, -6.0, 4.0)


def test_footprint_obeys_orientation_origin_and_original_scaling() -> None:
    bounds = ModelBounds(-2.0, 8.0, -6.0, 4.0)
    rotated = footprint_corners(bounds, 100.0, 200.0, 90.0)
    assert rotated[0] == pytest.approx((94.0, 202.0))
    assert rotated[1] == pytest.approx((94.0, 192.0))
    assert rotated[2] == pytest.approx((104.0, 192.0))
    assert footprint_contains(rotated, 100.0, 200.0)
    assert not footprint_contains(rotated, 150.0, 200.0)

    # Match a WRP object with a local X scale of 2 and local Z scale of 0.5,
    # irrespective of the heading display value.
    scaled = footprint_corners(bounds, 100.0, 200.0, 0.0,
                               basis=(2.0, 0.0, 0.0, 0.5))
    assert scaled[0] == pytest.approx((96.0, 197.0))
    assert scaled[2] == pytest.approx((116.0, 202.0))
    assert footprint_contains(scaled, 110.0, 200.0)
    assert not footprint_contains(scaled, 120.0, 200.0)
    line = footprint_corners(ModelBounds(0, 0, -5, 5), 100, 200, 0)
    assert not footprint_contains(line, 5000, 5000)
    assert not footprint_contains(line, 100, 200)


def _pbo(path: Path, name: str, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(name.encode("ascii") + b"\0")
        stream.write(struct.pack("<5I", 0, 0, 0, 0, len(payload)))
        stream.write(b"\0")
        stream.write(struct.pack("<5I", 0, 0, 0, 0, 0))
        stream.write(payload)


def test_pbo_footprints_are_loaded_on_demand_without_affecting_textures(tmp_path: Path) -> None:
    model = _mlod(_POINTS)
    archive = tmp_path / "o.pbo"
    _pbo(archive, r"tree\large.p3d", model)
    resources = PboTextures([archive])
    try:
        assert resources.members == {}
        assert len(resources.models) == 1
        assert resources.read_model(r"o\tree\large.p3d") == model
        assert resources.read(r"o\tree\large.p3d") is None
        assert load_model_bounds(r"o\tree\large.p3d", None, resources) == ModelBounds(-2.0, 8.0, -6.0, 4.0)
        assert load_model_bounds(r"o\tree\missing.p3d", None, resources) is None
    finally:
        resources.close()


def test_loose_models_and_unsupported_formats_keep_safe_fallback(tmp_path: Path) -> None:
    target = tmp_path / "data3d" / "house.p3d"
    target.parent.mkdir(parents=True)
    target.write_bytes(_mlod(_POINTS))
    assert load_model_bounds(r"data3d\house.p3d", tmp_path, None) == ModelBounds(-2.0, 8.0, -6.0, 4.0)
    assert load_model_bounds(r"data3d\missing.p3d", tmp_path, None) is None
    assert load_model_bounds(r"..\data3d\house.p3d", tmp_path, None) is None
    target.write_bytes(b"UNKN" + b"\0" * 32)
    assert load_model_bounds(r"data3d\house.p3d", tmp_path, None) is None
