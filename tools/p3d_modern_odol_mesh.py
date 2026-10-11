"""Decode Arma-era ODOL resolution LODs for the P3D asset browser.

Read-only, bounded implementation of the v40-v75 ODOL fields needed for
vertices, polygons, UV0 and diffuse texture selection. Later animations,
skeleton weights and nonvisual LODs do not affect a static model preview.

The compressed array formats and version gates follow the BI P3D formats:
https://community.bistudio.com/wiki/P3D_File_Format_-_ODOLV4x
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import io
import math
import struct

import numpy as np

import measure_p3d_models as measure
from p3d_modern_odol import ModernOdolError, read_model_bounds

_MAX_VERTICES = 2_000_000
_MAX_FACES = 2_000_000
_MAX_ITEMS = 2_000_000
_MAX_BYTES = 192 * 1024 * 1024
_MAX_STRING = 16_384


class MeshReadError(ModernOdolError):
    """An ODOL header was readable but its resolution LOD could not be decoded."""


@dataclass(slots=True)
class ModernMesh:
    version: int
    resolution: float
    points: np.ndarray
    # Each face is (vertex indices, per-vertex UV coordinates, texture name).
    faces: tuple[tuple[tuple[int, ...], tuple[tuple[float, float], ...], str], ...]
    textures: tuple[str, ...]


class _Reader:
    """Bounds-checked little-endian ODOL reader, including LZO/LZSS arrays."""

    def __init__(self, data: bytes, version: int) -> None:
        self.stream = io.BytesIO(data)
        self.size = len(data)
        self.limit = len(data)
        self.version = version

    def tell(self) -> int:
        return self.stream.tell()

    def seek(self, offset: int) -> None:
        if not 0 <= offset <= self.limit:
            raise MeshReadError(f"ODOL offset {offset} is outside its data")
        self.stream.seek(offset)

    def take(self, size: int, what: str = "ODOL data") -> bytes:
        if size < 0 or size > self.limit - self.tell():
            raise MeshReadError(f"truncated or implausible {what}: {size} bytes")
        return self.stream.read(size)

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def f32(self) -> float:
        return struct.unpack("<f", self.take(4))[0]

    def count(self, limit: int = _MAX_ITEMS) -> int:
        value = self.i32()
        if not 0 <= value <= limit:
            raise MeshReadError(f"implausible ODOL array count {value}")
        return value

    def string(self) -> str:
        start = self.tell()
        available = min(_MAX_STRING, self.limit - start)
        raw = self.take(available, "ODOL string")
        end = raw.find(b"\0")
        if end < 0:
            raise MeshReadError("unterminated or excessively long ODOL string")
        self.seek(start + end + 1)
        return raw[:end].decode("latin-1", errors="replace")

    def fixed_array(self, width: int, limit: int = _MAX_ITEMS) -> bytes:
        return self.take(self.count(limit) * width, "ODOL array")

    def compressed(self, expected: int) -> bytes:
        if expected < 0 or expected > _MAX_BYTES:
            raise MeshReadError(f"implausible decompressed ODOL array size {expected}")
        if expected == 0:
            return b""
        if self.version >= 44:
            compressed = expected >= 1024
            if self.version >= 64:
                compressed = bool(self.u8())
            if compressed:
                return _decode_lzo(self, expected)
            return self.take(expected, "uncompressed ODOL array")
        if expected < 1024:
            return self.take(expected, "uncompressed ODOL array")
        try:
            # Before v44, ODOL arrays use LZSS with a trailing checksum.
            result = measure._decompress_lzss_stream(self.stream, expected)
        except (measure.ModelReadError, ValueError, struct.error) as exc:
            raise MeshReadError(f"invalid legacy ODOL LZSS block: {exc}") from exc
        if self.tell() > self.limit:
            raise MeshReadError("compressed data exceeds LOD boundary")
        return result

    def compressed_array(self, width: int, limit: int = _MAX_ITEMS) -> tuple[int, bytes]:
        count = self.count(limit)
        return count, self.compressed(count * width)

    def condensed(self, width: int) -> tuple[int, bytes]:
        count = self.count()
        same = bool(self.u8())
        if same:
            return count, self.take(width, "condensed array default")
        return count, self.compressed(count * width)


def _decode_lzo(reader: _Reader, expected: int) -> bytes:
    """Expand an unframed BI LZO1X array, consuming its end marker.

    The compressed size is not stored in the ODOL array header, so decoding
    must consume the stream through the LZO end-of-stream instruction.
    """
    out = bytearray()

    def byte() -> int:
        return reader.u8()

    def literals(n: int) -> None:
        if n < 0 or n > expected - len(out):
            raise MeshReadError("LZO literal run exceeds expected output")
        out.extend(reader.take(n, "LZO literals"))

    def copy(position: int, n: int) -> None:
        if position < 0 or position >= len(out) or n < 0 or n > expected - len(out):
            raise MeshReadError("invalid LZO match offset or output size")
        for offset in range(n):
            out.append(out[position + offset])

    def extended(base: int) -> int:
        amount = base
        while True:
            value = byte()
            if value:
                return amount + value
            amount += 255
            if amount > expected:
                raise MeshReadError("excessively long LZO literal/match run")

    def after_match() -> int:
        # The two last bytes of a match encode its following literal count.
        pos = reader.tell()
        reader.seek(pos - 2)
        value = reader.u8() & 3
        reader.seek(pos)
        return value

    state = "start"
    token = 0
    if reader.tell() < reader.limit:
        first = reader.take(1)[0]
        reader.seek(reader.tell() - 1)
        if first > 17:
            token = byte() - 17
            if token < 4:
                state = "match_next"
            else:
                literals(token)
                state = "first_literal"
        else:
            state = "literal"
    while True:
        if state == "literal":
            token = byte()
            if token >= 16:
                state = "match"
                continue
            if token == 0:
                token = extended(15)
            literals(token + 3)
            state = "first_literal"
            continue
        if state == "first_literal":
            token = byte()
            if token >= 16:
                state = "match"
                continue
            offset = 1 + 0x800 + (token >> 2) + (byte() << 2)
            copy(len(out) - offset, 3)
            state = "after_match"
            continue
        if state == "match_next":
            literals(token)
            token = byte()
            state = "match"
            continue
        if state == "match":
            if token >= 64:
                offset = 1 + ((token >> 2) & 7) + (byte() << 3)
                copy(len(out) - offset, (token >> 5) + 1)
            elif token >= 32:
                length = token & 31
                if not length:
                    length = extended(31)
                first, second = byte(), byte()
                offset = 1 + (first >> 2) + (second << 6)
                copy(len(out) - offset, length + 2)
            elif token >= 16:
                high = (token & 8) << 11
                length = token & 7
                if not length:
                    length = extended(7)
                first, second = byte(), byte()
                distance = high + (first >> 2) + (second << 6)
                if distance == 0:
                    if len(out) != expected:
                        raise MeshReadError("LZO end marker before expected output size")
                    return bytes(out)
                copy(len(out) - distance - 0x4000, length + 2)
            else:
                offset = 1 + (token >> 2) + (byte() << 2)
                copy(len(out) - offset, 2)
            state = "after_match"
            continue
        if state == "after_match":
            token = after_match()
            state = "match_next" if token else "literal"
            continue
        raise MeshReadError(f"unknown LZO decoder state {state}")


def _skip_skeleton(r: _Reader) -> None:
    name = r.string()
    if name:
        if r.version >= 23:
            r.take(1)
        for _ in range(r.count(100_000)):
            r.string()
            r.string()
        if r.version > 40:
            r.string()


def _skip_model_info(r: _Reader) -> None:
    """Continue after ModelInfo's two bounding-box triples."""
    version = r.version
    if version >= 70:
        r.take(4)
    if version >= 71:
        r.take(4)
    if version >= 52:
        r.take(24)
    r.take(72)  # three centres plus the 3x3 inverse inertia matrix
    r.take(4)  # centering and occlusion booleans
    if version >= 73:
        r.take(1)
    if version >= 42:
        r.take(16)  # bounding height/factor fields
    if version >= 43:
        r.take(8)
    r.take(1)  # ForceNotAlphaModel (v33+)
    if version >= 37:
        r.take(5)
    if version >= 48:
        r.take(4)
    r.take(1)  # Animated
    _skip_skeleton(r)
    r.take(1)  # MapType
    r.compressed_array(4)  # mass array
    r.take(16)  # mass, inverse mass, armor, inverse armor
    if version >= 72:
        r.take(4)
    if version >= 53:
        r.take(1)
    if version >= 54:
        r.take(1)
    r.take(12 + 4)  # geometry flags and MinShadow
    if version >= 38:
        r.take(1)
    r.string()  # model class
    r.string()  # damage material
    r.take(1 + 4)  # frequent + unused v31
    if version >= 57:
        lods = r._lod_count
        r.take(lods * 4 * 3)  # preferred shadow LOD indices


def _skip_animations(r: _Reader) -> None:
    if not r.u8():
        return
    types = []
    for _ in range(r.count(100_000)):
        kind = r.u32()
        types.append(kind)
        r.string()
        r.string()
        r.take(16)
        if r.version >= 56:
            r.take(8)
        r.take(4)
        if kind <= 7:
            r.take(8)
        elif kind == 8:
            r.take(32)
        elif kind == 9:
            r.take(8 if r.version >= 55 else 4)
        else:
            raise MeshReadError(f"unknown ODOL animation type {kind}")

    bones = r.count(100_000)
    for _ in range(bones):
        for _ in range(r.count(100_000)):
            r.fixed_array(4, 100_000)
    for _ in range(bones):
        for kind in types:
            linked = r.i32()
            if linked != -1 and kind not in (8, 9):
                r.take(24)


def _lod_table(r: _Reader) -> tuple[tuple[float, int, int], ...]:
    count = r._lod_count
    _skip_model_info(r)
    if r.version >= 30:
        _skip_animations(r)
    starts = struct.unpack("<" + "I" * count, r.take(count * 4))
    ends = struct.unpack("<" + "I" * count, r.take(count * 4))
    permanent = r.take(count)
    for is_permanent in permanent:
        if not is_permanent:
            r.take(16)  # face count, colour, hints
            if r.version >= 39:
                r.take(1)
            if r.version >= 51:
                r.take(8)
    lods = []
    for resolution, start, end in zip(r._resolutions, starts, ends):
        if not math.isfinite(resolution):
            continue
        if not 0 < start < end <= r.size:
            raise MeshReadError("invalid ODOL LOD offsets")
        lods.append((resolution, start, end))
    # Prefer actual visual resolution LODs to roadway, geometry and shadow.
    lods.sort(key=lambda item: (not 0 <= item[0] < 10000, item[0] if item[0] < 10000 else 1e20))
    return tuple(lods)


def _material(r: _Reader) -> str:
    """Skip embedded RVMAT, capturing a diffuse texture when it has one."""
    r.string()  # material path
    version = r.u32()
    if version < 1 or version > 100:
        raise MeshReadError(f"unexpected embedded material version {version}")
    r.take(6 * 16)  # six float4 colours
    r.take(4 + 4 * 4)  # specular power and four shader/light/fog integers
    if version == 3:
        r.take(1)
    if version >= 6:
        r.string()
    if version >= 4:
        r.take(8)
    stages = r.count(1024) if version > 6 else 0
    transforms = r.count(1024) if version > 8 else stages
    candidate = ""

    def stage() -> str:
        if version >= 5:
            r.take(4)
        texture = r.string()
        if version >= 8:
            r.take(4)
        if version >= 11:
            r.take(1)
        return texture

    def transform() -> None:
        r.take(4 + 48)

    if version < 8:
        for _ in range(stages):
            transform()
            texture = stage()
            if texture.casefold().endswith((".paa", ".pac")) and not candidate:
                candidate = texture
    else:
        for _ in range(stages):
            texture = stage()
            if texture.casefold().endswith((".paa", ".pac")) and not candidate:
                candidate = texture
        for _ in range(transforms):
            transform()
    if version >= 10:
        texture = stage()
        if texture.casefold().endswith((".paa", ".pac")) and not candidate:
            candidate = texture
    return candidate


def _uvset(r: _Reader) -> np.ndarray:
    v = r.version
    if v >= 45:
        minimum_u, minimum_v, maximum_u, maximum_v = struct.unpack("<4f", r.take(16))
    n = r.u32()
    if n > _MAX_VERTICES:
        raise MeshReadError(f"implausible ODOL UV count {n}")
    width = 4 if v >= 45 else 8
    if r.u8():
        raw = r.take(width, "default UV")
        raw *= n
    else:
        raw = r.compressed(n * width)
    if v >= 45:
        uv = np.frombuffer(raw, dtype="<i2").reshape((-1, 2)).astype(np.float32)
        uv[:, 0] = minimum_u + (uv[:, 0] + 32767) / 65536 * (maximum_u - minimum_u)
        uv[:, 1] = minimum_v + (uv[:, 1] + 32767) / 65536 * (maximum_v - minimum_v)
    else:
        uv = np.frombuffer(raw, dtype="<f4").reshape((-1, 2)).copy()
    if not np.isfinite(uv).all():
        raise MeshReadError("invalid non-finite ODOL texture coordinates")
    return uv


def _read_resolution_lod(r: _Reader, resolution: float, start: int, end: int) -> ModernMesh:
    r.limit = end
    r.seek(start)
    v = r.version
    for _ in range(r.count(100_000)):
        r.string()
        r.take(48 + 12 + (4 if v >= 40 else 0))  # proxy transform and indices
    r.fixed_array(4, 200_000)
    for _ in range(r.count(200_000)):
        r.fixed_array(4, 200_000)
    if v >= 50:
        r.take(4)
    else:
        r.condensed(4)
    if v >= 51:
        r.take(4)
    r.take(8 + 48)  # LOD hints, bounds, centre, radius

    texture_list = [r.string() for _ in range(r.count(100_000))]
    materials = [_material(r) for _ in range(r.count(20_000))]
    index_width = 4 if v >= 69 else 2
    r.compressed_array(index_width)  # PointToVertex
    r.compressed_array(index_width)  # VertexToPoint

    face_count = r.count(_MAX_FACES)
    r.take(6)  # polygon metadata
    polygons: list[tuple[int, ...]] = []
    face_offsets: list[int] = []
    offset = 0
    for _ in range(face_count):
        length = r.u8()
        if length not in (3, 4):
            raise MeshReadError(f"unexpected ODOL polygon with {length} vertices")
        raw = r.take(length * index_width, "polygon vertex indices")
        fmt = "I" if index_width == 4 else "H"
        polygons.append(struct.unpack("<" + fmt * length, raw))
        face_offsets.append(offset)
        offset += (16 if index_width == 4 else 8) + (index_width if length == 4 else 0)

    sections: list[tuple[int, int, str]] = []
    for _ in range(r.count(100_000)):
        lower, upper, _min_bone, _bone_count, _unused = struct.unpack("<5i", r.take(20))
        texture_index = r.i16()
        r.take(4)
        material_index = r.i32()
        if material_index == -1:
            r.string()
        direct = texture_list[texture_index] if 0 <= texture_index < len(texture_list) else ""
        by_material = materials[material_index] if 0 <= material_index < len(materials) else ""
        texture = direct or by_material
        area_count = r.count(1024)
        r.take(4 * area_count)
        if v >= 67:
            flag = r.i32()
            if flag >= 1:
                r.take(44)
        if 0 <= lower < upper <= offset:
            sections.append((lower, upper, texture))

    for _ in range(r.count(100_000)):  # named selections
        r.string()
        r.compressed_array(index_width)
        r.take(5)
        r.compressed_array(4)
        r.compressed_array(index_width)
        r.compressed(r.count(_MAX_BYTES))
    for _ in range(r.count(100_000)):  # named properties
        r.string()
        r.string()
    for _ in range(r.count(100_000)):  # keyframe animations
        r.take(4)
        r.fixed_array(12, _MAX_VERTICES)
    r.take(13)  # top colour, colour, special, simple-bone flag
    r.take(4)   # SizeOfRestData
    if v >= 50:
        r.condensed(4)
    uv = _uvset(r)
    total_uvsets = r.count(64)
    if total_uvsets < 1:
        raise MeshReadError("ODOL resolution LOD has no UV sets")
    for _ in range(total_uvsets - 1):
        _uvset(r)
    vertex_count, raw_vertices = r.compressed_array(12, _MAX_VERTICES)
    if vertex_count == 0 or uv.shape[0] != vertex_count:
        raise MeshReadError(f"ODOL UV/vertex count mismatch ({len(uv)} vs {vertex_count})")
    points = np.frombuffer(raw_vertices, dtype="<f4").reshape((-1, 3)).copy()
    if not np.isfinite(points).all():
        raise MeshReadError("ODOL LOD contains non-finite vertices")

    paths = [""] * face_count
    for lower, upper, texture in sections:
        for i in range(bisect_left(face_offsets, lower), bisect_left(face_offsets, upper)):
            paths[i] = texture

    faces = []
    used_textures: dict[str, None] = {}
    for indices, texture in zip(polygons, paths):
        if any(index >= vertex_count for index in indices):
            continue
        if texture:
            used_textures.setdefault(texture, None)
        faces.append((
            tuple(indices),
            tuple((float(uv[i, 0]), float(uv[i, 1])) for i in indices),
            texture,
        ))
    if not faces:
        raise MeshReadError("ODOL LOD contains no valid triangles or quads")
    return ModernMesh(v, resolution, points, tuple(faces), tuple(used_textures))


def read_model_mesh(data: bytes) -> ModernMesh:
    """Return real first visual LOD vertices, faces, UVs and texture references.

    Raises MeshReadError if the file has no decodable visual LOD; callers may
    still show the (separately parsed) ModelInfo bounds as a fallback.
    """
    bounds = read_model_bounds(data)
    r = _Reader(data, bounds.version)
    r.seek(8)
    if r.version >= 75:
        r.take(8)
    if r.version >= 59:
        r.take(4)
    if r.version >= 58:
        r.string()
    lod_count = r.count(128)
    if lod_count != bounds.lod_count:
        raise MeshReadError("ODOL LOD header changed unexpectedly")
    r._lod_count = lod_count
    r._resolutions = struct.unpack("<" + "f" * lod_count, r.take(4 * lod_count))
    r.take(48 + 24)  # ModelInfo prefix and BboxMin/BboxMax
    entries = _lod_table(r)
    if not entries:
        raise MeshReadError("ODOL has no resolution LODs")
    failures: list[str] = []
    for resolution, start, end in entries:
        try:
            return _read_resolution_lod(r, resolution, start, end)
        except (MeshReadError, struct.error, OverflowError, ValueError) as exc:
            failures.append(f"LOD {resolution:g}: {exc}")
    raise MeshReadError("no readable ODOL visual mesh (" + "; ".join(failures[:3]) + ")")
