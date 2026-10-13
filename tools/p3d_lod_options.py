"""Enumerate real visual and special-purpose LODs from P3D model bytes.

ODOL v6/v7 stores sequential LOD bodies followed by a resolution table.
MLOD/SP3X stores a resolution after each LOD's TAGG terminator.
Modern ODOL v40+ has an indexed LOD offset table.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import math
import struct

import measure_p3d_models as measure
import p3d_modern_odol_mesh as modern


@dataclass(frozen=True)
class LodOption:
    index: int
    resolution: float
    start: int
    visual: bool

    @property
    def value(self) -> str:
        return f"lod:{self.index}"

    @property
    def label(self) -> str:
        name = f"LOD {self.index + 1} ({self.resolution:g})"
        return name if self.visual else name + " [special]"


def is_visual(resolution: float) -> bool:
    return math.isfinite(resolution) and 0 <= resolution < 10000


def select_lod(options: tuple[LodOption, ...], quality: str) -> tuple[LodOption, ...]:
    """Return the ordered LOD candidates for fast/full or one explicit LOD."""
    if quality.startswith("lod:"):
        try:
            index = int(quality[4:])
        except ValueError as exc:
            raise measure.ModelReadError(f"invalid LOD selection {quality!r}") from exc
        selected = tuple(entry for entry in options if entry.index == index)
        if not selected:
            raise measure.ModelReadError(f"LOD {index + 1} is unavailable")
        return selected
    if quality not in {"fast", "full"}:
        raise measure.ModelReadError(f"invalid LOD selection {quality!r}")
    visual = sorted((entry for entry in options if entry.visual), key=lambda entry: entry.resolution)
    if not visual:
        raise measure.ModelReadError("model has no visual resolution LODs")
    return tuple(reversed(visual)) if quality == "fast" else tuple(visual)


def _bounded_skip(stream: io.BytesIO, size: int, limit: int, label: str) -> None:
    if size < 0 or stream.tell() + size > limit:
        raise measure.ModelReadError(f"truncated or invalid {label}")
    stream.seek(size, io.SEEK_CUR)


def _count(stream: io.BytesIO, label: str, cap: int = 2_000_000) -> int:
    n = measure._read_u32(stream, label)
    if n > cap:
        raise measure.ModelReadError(f"implausible {label}: {n}")
    return n


def _legacy_odol_lods(data: bytes) -> tuple[LodOption, ...]:
    stream = io.BytesIO(data)
    stream.seek(8)
    count = _count(stream, "legacy ODOL LOD count", 128)
    starts = []
    length = len(data)

    def array(width: int, label: str) -> None:
        n = _count(stream, label, 4_000_000)
        if n * width > 256 * 1024 * 1024:
            raise measure.ModelReadError(f"implausible {label} bytes")
        measure._read_odol_array(stream, n, width, label)

    for index in range(count):
        starts.append(stream.tell())
        label = f"ODOL LOD {index + 1}"
        array(4, label + " point flags")
        array(8, label + " UVs")
        n = _count(stream, label + " point count", 4_000_000)
        _bounded_skip(stream, n * 12, length, label + " points")
        n = _count(stream, label + " normal count", 4_000_000)
        _bounded_skip(stream, n * 12 + 48, length, label + " normals and bounds")
        for _ in range(_count(stream, label + " texture count", 100_000)):
            measure._read_cstring(stream, label + " texture")
        array(2, label + " point-to-vertex")
        array(2, label + " vertex-to-point")
        n = _count(stream, label + " face count")
        _bounded_skip(stream, 4, length, label + " face allocation size")
        for _ in range(n):
            _bounded_skip(stream, 6, length, label + " face header")
            count_vertices = stream.read(1)
            if not count_vertices:
                raise measure.ModelReadError(f"truncated {label} face vertices")
            _bounded_skip(stream, count_vertices[0] * 2, length, label + " face indices")
        n = _count(stream, label + " section count")
        _bounded_skip(stream, 18 * n, length, label + " sections")
        for _ in range(_count(stream, label + " selection count", 100_000)):
            measure._read_cstring(stream, label + " selection name")
            for width in (2, 1, 4):
                array(width, label + " selection array")
            _bounded_skip(stream, 1, length, label + " selection flag")
            for width in (4, 2, 1):
                array(width, label + " selection array")
        for _ in range(_count(stream, label + " property count", 100_000)):
            measure._read_cstring(stream, label + " property name")
            measure._read_cstring(stream, label + " property value")
        for _ in range(_count(stream, label + " frame count", 100_000)):
            _bounded_skip(stream, 4, length, label + " frame time")
            n = _count(stream, label + " frame point count", 4_000_000)
            _bounded_skip(stream, n * 12, length, label + " frame points")
        _bounded_skip(stream, 12, length, label + " colours")
        for _ in range(_count(stream, label + " proxy count", 100_000)):
            measure._read_cstring(stream, label + " proxy name")
            _bounded_skip(stream, 48 + 8, length, label + " proxy matrix and indices")

    # ODOL7 ModelInfo begins with the resolutions, not at the file start.
    resolutions = struct.unpack("<" + "f" * count, measure._read_exact(
        stream, count * 4, "ODOL LOD resolutions"
    ))
    if not all(math.isfinite(value) for value in resolutions):
        raise measure.ModelReadError("invalid legacy ODOL LOD resolution")
    return tuple(LodOption(i, resolution, start, is_visual(resolution))
                 for i, (resolution, start) in enumerate(zip(resolutions, starts)))


def _mlod_lods(data: bytes) -> tuple[LodOption, ...]:
    stream = io.BytesIO(data)
    stream.seek(8)
    count = _count(stream, "MLOD LOD count", 128)
    length = len(data)
    result = []
    for index in range(count):
        start = stream.tell()
        if measure._read_exact(stream, 4, "SP3X signature") != b"SP3X":
            raise measure.ModelReadError(f"MLOD LOD {index + 1} is not SP3X")
        head_size, _version, points, normals, faces, _flags = struct.unpack(
            "<6i", measure._read_exact(stream, 24, "SP3X header")
        )
        if not 28 <= head_size <= 4096 or min(points, normals, faces) < 0:
            raise measure.ModelReadError("invalid SP3X header")
        if max(points, normals, faces) > 4_000_000:
            raise measure.ModelReadError("implausible SP3X counts")
        _bounded_skip(stream, head_size - 28 + points * 16 + normals * 12
                      + faces * 152, length, "SP3X geometry")
        if measure._read_exact(stream, 4, "MLOD TAGG") != b"TAGG":
            raise measure.ModelReadError("MLOD LOD is missing TAGG")
        for _ in range(100_000):
            raw_name, size = struct.unpack(
                "<64si", measure._read_exact(stream, 68, "MLOD TAGG header")
            )
            _bounded_skip(stream, size, length, "MLOD TAGG data")
            if raw_name.split(b"\0", 1)[0] == b"#EndOfFile#":
                break
        else:
            raise measure.ModelReadError("too many MLOD TAGG entries")
        resolution = struct.unpack("<f", measure._read_exact(
            stream, 4, "MLOD resolution"
        ))[0]
        if not math.isfinite(resolution):
            raise measure.ModelReadError("non-finite MLOD resolution")
        result.append(LodOption(index, resolution, start, is_visual(resolution)))
    return tuple(result)


def list_lods(data: bytes) -> tuple[LodOption, ...]:
    """Read available LODs without constructing face/UV meshes."""
    if len(data) < 12:
        raise measure.ModelReadError("truncated P3D header")
    signature = data[:4]
    version = struct.unpack_from("<I", data, 4)[0]
    if signature == b"MLOD":
        return _mlod_lods(data)
    if signature != b"ODOL":
        raise measure.ModelReadError("unsupported P3D signature")
    if version in {6, 7}:
        return _legacy_odol_lods(data)
    if version < 40:
        raise measure.ModelReadError(f"unsupported ODOL version {version}")
    bounds = modern.read_model_bounds(data)
    r = modern._Reader(data, bounds.version)
    r.seek(8)
    if r.version >= 75:
        r.take(8)
    if r.version >= 59:
        r.take(4)
    if r.version >= 58:
        r.string()
    count = r.count(128)
    if count != bounds.lod_count:
        raise measure.ModelReadError("ODOL LOD count mismatch")
    r._lod_count = count
    r._resolutions = struct.unpack("<" + "f" * count, r.take(count * 4))
    r.take(72)
    entries = modern._lod_table(r)
    # The ODOL resolution table and offset table share the same index order.
    return tuple(LodOption(index, resolution, start, is_visual(resolution))
                 for index, resolution, start, _end in sorted(entries, key=lambda item: item[0]))
