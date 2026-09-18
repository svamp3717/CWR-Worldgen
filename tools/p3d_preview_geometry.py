"""Parse first-LOD P3D geometry, UVs and material references for previewing."""
from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import struct
import sys
from typing import Iterator, Sequence

import numpy as np

import measure_p3d_models as measure
from p3d_texture_render import RenderFace

_MAX_FACE_COUNT = 2_000_000
_MAX_TEXTURE_COUNT = 100_000
_I16 = struct.Struct("<h")
_I32 = struct.Struct("<i")
_U8 = struct.Struct("<B")
_MLOD_FACE = struct.Struct("<32si" + "iiff" * 4 + "i")


@dataclass(slots=True)
class PreviewModel:
    model_path: str
    source: str
    measurement: measure.ModelMeasurement
    points: np.ndarray
    faces: tuple[RenderFace, ...]
    edges: np.ndarray
    original_vertex_count: int
    texture_paths: tuple[str, ...]
    render_mode: str


def _read_i16(stream: io.BytesIO, label: str) -> int:
    return _I16.unpack(measure._read_exact(stream, 2, label))[0]


def _read_i32(stream: io.BytesIO, label: str) -> int:
    return _I32.unpack(measure._read_exact(stream, 4, label))[0]


def _read_u8(stream: io.BytesIO, label: str) -> int:
    return _U8.unpack(measure._read_exact(stream, 1, label))[0]


def _read_odol_faces(stream: io.BytesIO, point_count: int, uv_coords: np.ndarray):
    measure._read_exact(stream, 48, "ODOL LOD bounds")
    texture_count = measure._read_u32(stream, "ODOL texture count")
    if texture_count > _MAX_TEXTURE_COUNT:
        raise measure.ModelReadError(f"implausible ODOL texture count {texture_count}")
    textures = tuple(measure._read_cstring(stream, f"ODOL texture {i}") for i in range(texture_count))
    for label in ("ODOL MLOD edge indices", "ODOL vertex edge indices"):
        count = measure._read_u32(stream, f"{label} count")
        measure._read_odol_array(stream, count, 2, label)
    face_count = measure._read_u32(stream, "ODOL face count")
    measure._read_u32(stream, "ODOL section offset")
    if face_count > _MAX_FACE_COUNT:
        raise measure.ModelReadError(f"implausible ODOL face count {face_count}")
    faces: list[RenderFace] = []
    for face_index in range(face_count):
        measure._read_u32(stream, f"ODOL face {face_index} flags")
        texture_index = _read_i16(stream, f"ODOL face {face_index} texture index")
        vertex_count = _read_u8(stream, f"ODOL face {face_index} vertex count")
        if vertex_count > 64:
            raise measure.ModelReadError(f"implausible ODOL face {face_index} vertex count {vertex_count}")
        raw = measure._read_exact(stream, vertex_count * 2, f"ODOL face {face_index} vertex indices")
        indices = struct.unpack("<" + "H" * vertex_count, raw) if vertex_count else ()
        if vertex_count not in (3, 4) or not all(index < point_count for index in indices):
            continue
        uvs = tuple((float(uv_coords[i, 0]), float(uv_coords[i, 1])) for i in indices)
        texture = textures[texture_index] if 0 <= texture_index < len(textures) else ""
        faces.append(RenderFace(tuple(int(i) for i in indices), uvs, texture))
    return tuple(faces), textures


def _read_mlod_faces(stream: io.BytesIO, point_count: int, face_count: int):
    if face_count < 0 or face_count > _MAX_FACE_COUNT:
        raise measure.ModelReadError(f"implausible SP3X face count {face_count}")
    faces: list[RenderFace] = []
    textures: dict[str, None] = {}
    for face_index in range(face_count):
        values = _MLOD_FACE.unpack(measure._read_exact(stream, _MLOD_FACE.size, f"SP3X face {face_index}"))
        texture = values[0].split(b"\0", 1)[0].decode("latin-1", errors="replace")
        vertex_count = int(values[1])
        if vertex_count not in (3, 4):
            continue
        indices = tuple(int(values[2 + i * 4]) for i in range(vertex_count))
        if not all(0 <= i < point_count for i in indices):
            continue
        uvs = tuple((float(values[4 + i * 4]), float(values[5 + i * 4])) for i in range(vertex_count))
        if texture:
            textures.setdefault(texture, None)
        faces.append(RenderFace(indices, uvs, texture))
    return tuple(faces), tuple(textures)


def extract_first_lod_geometry(data: bytes, *, model_path: str, source: str):
    if len(data) < 12:
        raise measure.ModelReadError("P3D is too small to contain a valid header")
    signature = data[:4]
    stream = io.BytesIO(data)
    faces: tuple[RenderFace, ...] = ()
    textures: tuple[str, ...] = ()

    if signature == b"ODOL":
        measure._read_exact(stream, 4, "ODOL signature")
        version = measure._read_u32(stream, "ODOL version")
        lod_count = measure._read_u32(stream, "ODOL LOD count")
        if version not in {6, 7}:
            raise measure.ModelReadError(f"unsupported ODOL version {version}; expected OFP/CWA v6/v7")
        if lod_count <= 0 or lod_count > 128:
            raise measure.ModelReadError(f"implausible ODOL LOD count {lod_count}")
        flag_count = measure._read_u32(stream, "ODOL point-flag count")
        measure._read_odol_array(stream, flag_count, 4, "ODOL point flags")
        uv_count = measure._read_u32(stream, "ODOL UV count")
        raw_uvs = measure._read_odol_array(stream, uv_count, 8, "ODOL UV coordinates")
        point_count = measure._read_u32(stream, "ODOL point count")
        if point_count <= 0 or point_count > measure._MAX_VERTEX_COUNT:
            raise measure.ModelReadError(f"implausible ODOL point count {point_count}")
        if flag_count != point_count or uv_count != point_count:
            raise measure.ModelReadError(
                f"ODOL first-LOD vertex table count mismatch (flags={flag_count}, uv={uv_count}, points={point_count})"
            )
        uv_coords = np.frombuffer(raw_uvs, dtype="<f4").reshape((-1, 2)).copy()
        raw_points = measure._read_exact(stream, point_count * measure._VEC3.size, "ODOL points")
        points = np.frombuffer(raw_points, dtype="<f4").reshape((-1, 3)).copy()
        normal_count = measure._read_u32(stream, "ODOL normal count")
        if normal_count != point_count:
            raise measure.ModelReadError(
                f"ODOL first-LOD normal count {normal_count} does not match {point_count} vertices"
            )
        measure._read_exact(stream, normal_count * measure._VEC3.size, "ODOL normals")
        try:
            faces, textures = _read_odol_faces(stream, point_count, uv_coords)
        except (measure.ModelReadError, struct.error) as exc:
            if version == 7:
                raise
            print(f"[preview warning] {model_path}: ODOL v6 topology unreadable: {exc}; vertex fallback", file=sys.stderr)
        format_name = "ODOL"

    elif signature == b"MLOD":
        measure._read_exact(stream, 4, "MLOD signature")
        version_raw = measure._read_u32(stream, "MLOD version")
        version = (version_raw & 0xFF) * 10 + ((version_raw >> 8) & 0xFF)
        lod_count = measure._read_u32(stream, "MLOD LOD count")
        if lod_count <= 0 or lod_count > 128:
            raise measure.ModelReadError(f"implausible MLOD LOD count {lod_count}")
        if measure._read_exact(stream, 4, "MLOD first-LOD signature") != b"SP3X":
            raise measure.ModelReadError("unsupported MLOD LOD format; expected SP3X")
        head_size = _read_i32(stream, "SP3X header size")
        _read_i32(stream, "SP3X version")
        point_count = _read_i32(stream, "SP3X point count")
        normal_count = _read_i32(stream, "SP3X normal count")
        face_count = _read_i32(stream, "SP3X face count")
        _read_i32(stream, "SP3X flags")
        if head_size < 28 or head_size > 4096:
            raise measure.ModelReadError(f"implausible SP3X header size {head_size}")
        if head_size > 28:
            measure._read_exact(stream, head_size - 28, "SP3X extra header")
        if point_count <= 0 or point_count > measure._MAX_VERTEX_COUNT:
            raise measure.ModelReadError(f"implausible SP3X point count {point_count}")
        if normal_count < 0 or normal_count > measure._MAX_VERTEX_COUNT:
            raise measure.ModelReadError(f"implausible SP3X normal count {normal_count}")
        raw = measure._read_exact(stream, point_count * 16, "SP3X points")
        records = np.frombuffer(raw, dtype=np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),("flags","<i4")]))
        points = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float32, copy=False)
        measure._read_exact(stream, normal_count * measure._VEC3.size, "SP3X normals")
        faces, textures = _read_mlod_faces(stream, point_count, face_count)
        format_name = "MLOD/SP3X"
    else:
        raise measure.ModelReadError(f"unsupported P3D signature {signature!r}")

    measurement = measure._measurement(
        model_path=model_path, source=source, format_name=format_name,
        version=version, lod="first", points=points,
    )
    return measurement, points, faces, textures


def collect_edges(faces: Sequence[RenderFace]) -> np.ndarray:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        for i in range(len(face.indices)):
            a, b = int(face.indices[i]), int(face.indices[(i + 1) % len(face.indices)])
            if a != b:
                edges.add((a, b) if a < b else (b, a))
    return np.asarray(sorted(edges), dtype=np.int32) if edges else np.empty((0, 2), dtype=np.int32)


def preview_models(inputs: Sequence[Path], patterns: Sequence[str], max_points: int) -> Iterator[PreviewModel | measure.ModelFailure]:
    for model_path, source, data in measure._iter_models(inputs, patterns):
        try:
            measurement, points, faces, textures = extract_first_lod_geometry(data, model_path=model_path, source=source)
            edges = collect_edges(faces)
            original_vertex_count = len(points)
            if faces:
                preview_points, mode = points.astype(np.float32, copy=False), "textured mesh"
            else:
                if len(points) > max_points:
                    picks = np.linspace(0, len(points)-1, max_points, dtype=np.int64)
                    points = points[picks]
                preview_points, mode = points.astype(np.float32, copy=False), "vertex fallback"
            yield PreviewModel(model_path, source, measurement, preview_points, faces, edges, original_vertex_count, textures, mode)
        except (measure.ModelReadError, OSError, struct.error, OverflowError, IndexError, ValueError) as exc:
            failure = measure.ModelFailure(model_path=model_path, source=source, error=str(exc))
            print(f"[model error] {failure.model_path}\n  source: {failure.source}\n  error:  {failure.error}", file=sys.stderr, flush=True)
            yield failure
