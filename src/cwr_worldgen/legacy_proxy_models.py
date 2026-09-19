# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

"""Build visual-only vegetation clones safe to use as CWA 1.99 proxies.

The original engine deforms ClipLandKeep/ClipLandOn models in Object::Animate()
using Object::Transform(). For an object nested behind a P3D proxy that transform
is parent-local rather than world-space, so terrain samples come from the wrong
map coordinates and the shared vegetation shape can be stretched into spikes or
slabs. A proxy-safe clone preserves stock visual geometry and rendering flags but
clears the land-interaction bits so the already-grounded generated carrier owns
terrain fitting instead.
"""

from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
import io
import struct

from .assets import _decompress_lzss_stream, canonical_asset_path
from .procedural_buildings import _Face, _Lod, _MLOD_HEADER, _write_lod

_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_I16 = struct.Struct("<h")
_U8 = struct.Struct("<B")
_VEC2 = struct.Struct("<ff")
_VEC3 = struct.Struct("<fff")
_POINT = struct.Struct("<fffi")
_FACE_VERTEX = struct.Struct("<iiff")

_LAND_DEFORM_MASK = 0x0900  # ClipLandOn | ClipLandKeep
_MAX_VERTEX_COUNT = 2_000_000
_MAX_FACE_COUNT = 2_000_000
_MAX_TEXTURE_COUNT = 100_000


class ProxyCloneError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProxyCloneInfo:
    source_model: str
    generated_model: str
    source_format: str
    point_count: int
    face_count: int
    land_flagged_points: int
    texture_paths: tuple[str, ...]


def proxy_safe_model_path(world_name: str, source_model: str) -> str:
    canonical = canonical_asset_path(source_model)
    digest = sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{world_name}\\f\\p\\{digest}.p3d"


def _read_exact(stream: io.BytesIO, size: int, label: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ProxyCloneError(
            f"truncated {label}: wanted {size} bytes, got {len(data)}"
        )
    return data


def _read_u32(stream: io.BytesIO, label: str) -> int:
    return _U32.unpack(_read_exact(stream, 4, label))[0]


def _read_i32(stream: io.BytesIO, label: str) -> int:
    return _I32.unpack(_read_exact(stream, 4, label))[0]


def _read_cstring(stream: io.BytesIO, label: str) -> str:
    value = bytearray()
    while True:
        byte = stream.read(1)
        if not byte:
            raise ProxyCloneError(f"truncated {label}")
        if byte == b"\0":
            return value.decode("latin-1")
        value.extend(byte)


def _read_odol_array(
    stream: io.BytesIO, count: int, item_size: int, label: str
) -> bytes:
    if count < 0 or count > _MAX_VERTEX_COUNT:
        raise ProxyCloneError(f"implausible {label} count {count}")
    size = count * item_size
    if size < 1024:
        return _read_exact(stream, size, label)
    try:
        return _decompress_lzss_stream(stream, size)
    except ValueError as exc:
        raise ProxyCloneError(f"{label}: {exc}") from exc


def _proxy_lod(
    *,
    points: tuple[tuple[float, float, float], ...],
    normals: tuple[tuple[float, float, float], ...],
    faces: tuple[_Face, ...],
    point_flags: tuple[int, ...],
) -> _Lod:
    if not points or not faces:
        raise ProxyCloneError("stock visual LOD contains no drawable geometry")
    return _Lod(
        points,
        normals,
        faces,
        1.0,
        (),
        (),
        (("autocenter", "0"), ("class", "bushsoft")),
        point_flags,
    )


def _read_odol_visual(data: bytes) -> tuple[_Lod, int, tuple[str, ...]]:
    stream = io.BytesIO(data)
    if _read_exact(stream, 4, "ODOL signature") != b"ODOL":
        raise ProxyCloneError("not an ODOL P3D")
    version = _read_u32(stream, "ODOL version")
    lod_count = _read_u32(stream, "ODOL LOD count")
    if version not in {6, 7}:
        raise ProxyCloneError(f"unsupported ODOL version {version}; expected 6 or 7")
    if lod_count <= 0 or lod_count > 128:
        raise ProxyCloneError(f"implausible ODOL LOD count {lod_count}")

    flag_count = _read_u32(stream, "ODOL point flag count")
    raw_flags = _read_odol_array(stream, flag_count, 4, "ODOL point flags")
    flags = tuple(
        struct.unpack_from("<I", raw_flags, offset)[0]
        for offset in range(0, len(raw_flags), 4)
    )

    uv_count = _read_u32(stream, "ODOL UV count")
    raw_uvs = _read_odol_array(stream, uv_count, 8, "ODOL UV coordinates")
    uvs = tuple(
        _VEC2.unpack_from(raw_uvs, offset)
        for offset in range(0, len(raw_uvs), _VEC2.size)
    )

    point_count = _read_u32(stream, "ODOL point count")
    if point_count <= 0 or point_count > _MAX_VERTEX_COUNT:
        raise ProxyCloneError(f"implausible ODOL point count {point_count}")
    if flag_count != point_count or uv_count != point_count:
        raise ProxyCloneError(
            "ODOL first-LOD table mismatch "
            f"(flags={flag_count}, uv={uv_count}, points={point_count})"
        )
    raw_points = _read_exact(stream, point_count * _VEC3.size, "ODOL points")
    points = tuple(
        _VEC3.unpack_from(raw_points, offset)
        for offset in range(0, len(raw_points), _VEC3.size)
    )

    normal_count = _read_u32(stream, "ODOL normal count")
    if normal_count <= 0 or normal_count > _MAX_VERTEX_COUNT:
        raise ProxyCloneError(f"implausible ODOL normal count {normal_count}")
    raw_normals = _read_exact(
        stream, normal_count * _VEC3.size, "ODOL normals"
    )
    normals = tuple(
        _VEC3.unpack_from(raw_normals, offset)
        for offset in range(0, len(raw_normals), _VEC3.size)
    )
    if normal_count != point_count:
        raise ProxyCloneError(
            f"ODOL first-LOD normals={normal_count}, points={point_count}"
        )

    _read_exact(stream, 48, "ODOL LOD bounds")
    texture_count = _read_u32(stream, "ODOL texture count")
    if texture_count > _MAX_TEXTURE_COUNT:
        raise ProxyCloneError(f"implausible ODOL texture count {texture_count}")
    textures = tuple(
        _read_cstring(stream, f"ODOL texture {index}")
        for index in range(texture_count)
    )

    for label in ("ODOL MLOD edge indices", "ODOL vertex edge indices"):
        count = _read_u32(stream, f"{label} count")
        _read_odol_array(stream, count, 2, label)

    face_count = _read_u32(stream, "ODOL face count")
    _read_u32(stream, "ODOL section offset")
    if face_count > _MAX_FACE_COUNT:
        raise ProxyCloneError(f"implausible ODOL face count {face_count}")

    faces: list[_Face] = []
    for face_index in range(face_count):
        face_flags = _read_u32(stream, f"ODOL face {face_index} flags")
        texture_index = _I16.unpack(
            _read_exact(stream, 2, f"ODOL face {face_index} texture index")
        )[0]
        vertex_count = _U8.unpack(
            _read_exact(stream, 1, f"ODOL face {face_index} vertex count")
        )[0]
        if vertex_count > 64:
            raise ProxyCloneError(
                f"implausible ODOL face {face_index} vertex count {vertex_count}"
            )
        raw_indices = _read_exact(
            stream, vertex_count * 2, f"ODOL face {face_index} indices"
        )
        indices = (
            struct.unpack("<" + "H" * vertex_count, raw_indices)
            if vertex_count
            else ()
        )
        if vertex_count not in {3, 4}:
            continue
        if not all(index < point_count for index in indices):
            continue

        # ODOL stores the already-compiled winding. MLOD's legacy loader reverses
        # face vertices while importing, so reverse here to preserve the final
        # stock orientation after that loader step.
        ordered = tuple(reversed(indices))
        vertices = tuple(
            (
                int(index),
                int(index),
                float(uvs[index][0]),
                float(uvs[index][1]),
            )
            for index in ordered
        )
        texture = (
            textures[texture_index]
            if 0 <= texture_index < len(textures)
            else ""
        )
        faces.append(_Face(texture, vertices, int(face_flags)))

    land_count = sum(bool(flag & _LAND_DEFORM_MASK) for flag in flags)
    safe_flags = tuple(int(flag & ~_LAND_DEFORM_MASK) for flag in flags)
    lod = _proxy_lod(
        points=points,
        normals=normals,
        faces=tuple(faces),
        point_flags=safe_flags,
    )
    return lod, land_count, tuple(
        sorted({face.texture for face in faces if face.texture})
    )


def _read_mlod_visual(data: bytes) -> tuple[_Lod, int, tuple[str, ...]]:
    stream = io.BytesIO(data)
    if _read_exact(stream, 4, "MLOD signature") != b"MLOD":
        raise ProxyCloneError("not an MLOD P3D")
    _read_exact(stream, 4, "MLOD version")
    lod_count = _read_u32(stream, "MLOD LOD count")
    if lod_count <= 0 or lod_count > 128:
        raise ProxyCloneError(f"implausible MLOD LOD count {lod_count}")
    if _read_exact(stream, 4, "SP3X signature") != b"SP3X":
        raise ProxyCloneError("first MLOD LOD is not SP3X")
    head_size = _read_i32(stream, "SP3X header size")
    _read_i32(stream, "SP3X version")
    point_count = _read_i32(stream, "SP3X point count")
    normal_count = _read_i32(stream, "SP3X normal count")
    face_count = _read_i32(stream, "SP3X face count")
    _read_i32(stream, "SP3X flags")
    if head_size < 28 or head_size > 4096:
        raise ProxyCloneError(f"implausible SP3X header size {head_size}")
    if head_size > 28:
        _read_exact(stream, head_size - 28, "SP3X extra header")
    if point_count <= 0 or point_count > _MAX_VERTEX_COUNT:
        raise ProxyCloneError(f"implausible SP3X point count {point_count}")
    if normal_count <= 0 or normal_count > _MAX_VERTEX_COUNT:
        raise ProxyCloneError(f"implausible SP3X normal count {normal_count}")
    if face_count < 0 or face_count > _MAX_FACE_COUNT:
        raise ProxyCloneError(f"implausible SP3X face count {face_count}")

    points: list[tuple[float, float, float]] = []
    flags: list[int] = []
    for index in range(point_count):
        x, y, z, flag = _POINT.unpack(
            _read_exact(stream, _POINT.size, f"SP3X point {index}")
        )
        points.append((x, y, z))
        flags.append(flag & 0xFFFFFFFF)

    raw_normals = _read_exact(
        stream, normal_count * _VEC3.size, "SP3X normals"
    )
    normals = tuple(
        _VEC3.unpack_from(raw_normals, offset)
        for offset in range(0, len(raw_normals), _VEC3.size)
    )

    faces: list[_Face] = []
    for face_index in range(face_count):
        texture = _read_exact(
            stream, 32, f"SP3X face {face_index} texture"
        ).split(b"\0", 1)[0].decode("latin-1")
        vertex_count = _read_i32(stream, f"SP3X face {face_index} vertex count")
        raw_vertices = [
            _FACE_VERTEX.unpack(
                _read_exact(
                    stream,
                    _FACE_VERTEX.size,
                    f"SP3X face {face_index} vertex {vertex_index}",
                )
            )
            for vertex_index in range(4)
        ]
        face_flags = _read_i32(stream, f"SP3X face {face_index} flags")
        if vertex_count not in {3, 4}:
            continue
        vertices = tuple(
            (int(p), int(n), float(u), float(v))
            for p, n, u, v in raw_vertices[:vertex_count]
            if 0 <= p < point_count and 0 <= n < normal_count
        )
        if len(vertices) != vertex_count:
            continue
        faces.append(_Face(texture, vertices, face_flags))

    land_count = sum(bool(flag & _LAND_DEFORM_MASK) for flag in flags)
    safe_flags = tuple(int(flag & ~_LAND_DEFORM_MASK) for flag in flags)
    lod = _proxy_lod(
        points=tuple(points),
        normals=normals,
        faces=tuple(faces),
        point_flags=safe_flags,
    )
    return lod, land_count, tuple(
        sorted({face.texture for face in faces if face.texture})
    )


def write_proxy_safe_visual_clone(
    path: Path,
    data: bytes,
    *,
    source_model: str,
    generated_model: str,
) -> ProxyCloneInfo:
    if data.startswith(b"ODOL"):
        lod, land_count, textures = _read_odol_visual(data)
        source_format = "ODOL"
    elif data.startswith(b"MLOD"):
        lod, land_count, textures = _read_mlod_visual(data)
        source_format = "MLOD"
    else:
        raise ProxyCloneError(
            f"{source_model}: unsupported P3D signature {data[:4]!r}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, 1))
        _write_lod(stream, lod)

    return ProxyCloneInfo(
        source_model=source_model,
        generated_model=generated_model,
        source_format=source_format,
        point_count=len(lod.points),
        face_count=len(lod.faces),
        land_flagged_points=land_count,
        texture_paths=textures,
    )
