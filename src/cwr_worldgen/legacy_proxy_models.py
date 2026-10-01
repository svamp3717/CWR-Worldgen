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

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
import io
import math
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


@dataclass(frozen=True, slots=True)
class VisualModelDimensions:
    """Plan-view dimensions and straightness of the first drawable visual LOD."""

    source_format: str
    width_metres: float
    length_metres: float
    height_metres: float
    texture_paths: tuple[str, ...]
    connector_width_metres: float | None = None
    lateral_center_shift_metres: float = 0.0
    is_straight_road_candidate: bool = True


@dataclass(frozen=True, slots=True)
class VisualSurfaceStyle:
    """Dominant render metadata for one donor road surface texture."""

    source_format: str
    texture_path: str
    point_flag: int
    face_flag: int


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


def _boundary_connector_width(
    lod: _Lod,
    *,
    visual_width: float,
    visual_length: float,
) -> float | None:
    """Measure curved modular-road mouths from visual boundary topology.

    A curved road's far connector is not generally located at global max-Z.
    When the ordinary terminal-Z bands cannot see both mouths, find straight,
    collinear boundary-edge groups instead. Multi-lane/divided roads commonly
    expose several lane-width boundary edges on each terminal line, so the full
    mouth span remains measurable even when a median separates carriageways.
    """

    edge_counts: dict[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
        ],
        int,
    ] = {}

    def point_key(index: int) -> tuple[float, float, float]:
        point = lod.points[index]
        return (
            round(float(point[0]), 5),
            round(float(point[1]), 5),
            round(float(point[2]), 5),
        )

    for face in lod.faces:
        indices = tuple(
            int(vertex[0])
            for vertex in face.vertices
            if 0 <= int(vertex[0]) < len(lod.points)
        )
        if len(indices) < 3:
            continue
        for first, second in zip(indices, (*indices[1:], indices[0])):
            a = point_key(first)
            b = point_key(second)
            if a == b:
                continue
            edge = tuple(sorted((a, b)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary_edges = tuple(
        edge for edge, count in edge_counts.items() if count == 1
    )
    if len(boundary_edges) < 4:
        return None

    angle_cosine = math.cos(math.radians(1.0))
    groups: list[dict[str, object]] = []
    for a, b in boundary_edges:
        dx = float(b[0] - a[0])
        dz = float(b[2] - a[2])
        edge_length = math.hypot(dx, dz)
        if edge_length <= 1.0e-4:
            continue
        ux, uz = dx / edge_length, dz / edge_length
        if ux < 0.0 or (abs(ux) <= 1.0e-9 and uz < 0.0):
            ux, uz = -ux, -uz
        nx, nz = -uz, ux
        midpoint_x = (float(a[0]) + float(b[0])) * 0.5
        midpoint_z = (float(a[2]) + float(b[2])) * 0.5
        offset = nx * midpoint_x + nz * midpoint_z

        selected = None
        for group in groups:
            gux, guz = group["direction"]  # type: ignore[misc]
            dot = abs(ux * float(gux) + uz * float(guz))
            if (
                dot >= angle_cosine
                and abs(offset - float(group["offset"])) <= 0.08
            ):
                selected = group
                break
        if selected is None:
            selected = {
                "direction": (ux, uz),
                "offset": offset,
                "edges": [],
            }
            groups.append(selected)
        selected["edges"].append((a, b))  # type: ignore[index]

    minimum_span = max(0.50, float(visual_width) * 0.15)
    maximum_span = max(minimum_span, float(visual_width) * 1.25)
    candidates: list[tuple[int, float, tuple[float, float]]] = []
    for group in groups:
        edges = tuple(group["edges"])  # type: ignore[arg-type]
        if len(edges) < 2:
            continue
        ux, uz = group["direction"]  # type: ignore[misc]
        projections = tuple(
            float(point[0]) * float(ux) + float(point[2]) * float(uz)
            for edge in edges
            for point in edge
        )
        span = max(projections) - min(projections)
        if not minimum_span <= span <= maximum_span:
            continue
        midpoint_projection = (min(projections) + max(projections)) * 0.5
        nx, nz = -float(uz), float(ux)
        offset = float(group["offset"])
        centre = (
            float(ux) * midpoint_projection + nx * offset,
            float(uz) * midpoint_projection + nz * offset,
        )
        candidates.append((len(edges), float(span), centre))

    if len(candidates) < 2:
        return None

    minimum_separation = max(
        0.50,
        math.hypot(float(visual_width), float(visual_length)) * 0.15,
    )
    best = None
    for first_index, first in enumerate(candidates[:-1]):
        for second in candidates[first_index + 1:]:
            separation = math.dist(first[2], second[2])
            if separation < minimum_separation:
                continue
            width_delta = abs(first[1] - second[1])
            tolerance = max(0.25, min(first[1], second[1]) * 0.10)
            if width_delta > tolerance:
                continue
            score = (
                min(first[0], second[0]),
                -width_delta,
                separation,
                min(first[1], second[1]),
            )
            if best is None or score > best[0]:
                best = score, min(first[1], second[1])
    return None if best is None else float(best[1])


def inspect_visual_model_dimensions(data: bytes) -> VisualModelDimensions:
    """Measure a conventional +Z road/object visual from ODOL or MLOD bytes.

    CWA modular roads are authored with width on local X and travel direction on
    local Z.  Keeping those axes explicit is useful: accepting a sideways donor
    would make the fitter place the original mod model incorrectly even before
    generated fallback geometry entered the picture.
    """

    if data.startswith(b"ODOL"):
        lod, _land_count, textures = _read_odol_visual(data)
        source_format = "ODOL"
    elif data.startswith(b"MLOD"):
        lod, _land_count, textures = _read_mlod_visual(data)
        source_format = "MLOD"
    else:
        raise ProxyCloneError(
            f"unsupported P3D signature {data[:4]!r}"
        )

    if not lod.points:
        raise ProxyCloneError("visual LOD contains no points")
    xs = tuple(float(point[0]) for point in lod.points)
    ys = tuple(float(point[1]) for point in lod.points)
    zs = tuple(float(point[2]) for point in lod.points)
    width = max(xs) - min(xs)
    length = max(zs) - min(zs)
    height = max(ys) - min(ys)
    if width <= 1.0e-4 or length <= 1.0e-4:
        raise ProxyCloneError(
            f"visual LOD has degenerate plan dimensions {width:g} x {length:g}"
        )

    # A modular road connects at its terminal mouths, not at the widest point
    # anywhere in the visual LOD. Mod roads can carry shoulders, marker meshes,
    # or other mid-span detail outside the actual road surface. Preserve the
    # complete visual width above for diagnostics, but separately measure a
    # narrow band at both ends for generated-road matching.
    connector_band = max(0.05, min(0.75, length * 0.04))
    lower_connector_x = tuple(
        float(point[0]) for point in lod.points
        if float(point[2]) <= min(zs) + connector_band
    )
    upper_connector_x = tuple(
        float(point[0]) for point in lod.points
        if float(point[2]) >= max(zs) - connector_band
    )
    connector_width = width
    connector_centers: tuple[float, float] | None = None
    if len(lower_connector_x) >= 2 and len(upper_connector_x) >= 2:
        lower_width = max(lower_connector_x) - min(lower_connector_x)
        upper_width = max(upper_connector_x) - min(upper_connector_x)
        candidate_width = min(lower_width, upper_width)
        # Ignore degenerate end markers. A real modular road mouth should still
        # occupy a meaningful fraction of the complete visual footprint.
        if candidate_width >= max(0.50, width * 0.15):
            connector_width = candidate_width
            connector_centers = (
                (min(lower_connector_x) + max(lower_connector_x)) * 0.5,
                (min(upper_connector_x) + max(upper_connector_x)) * 0.5,
            )

    if connector_centers is None:
        boundary_width = _boundary_connector_width(
            lod,
            visual_width=width,
            visual_length=length,
        )
        if boundary_width is not None:
            connector_width = boundary_width

    # Classify straightness from the same terminal mouths used for connector
    # width whenever possible. Wider shoulders or asymmetric mid-span detail in
    # mod roads must not make a genuinely straight donor look curved.
    if connector_centers is not None:
        lateral_shift = abs(connector_centers[1] - connector_centers[0])
    else:
        band = max(length * 0.20, 1.0e-4)
        lower_x = tuple(
            float(point[0]) for point in lod.points
            if float(point[2]) <= min(zs) + band
        )
        upper_x = tuple(
            float(point[0]) for point in lod.points
            if float(point[2]) >= max(zs) - band
        )
        if lower_x and upper_x:
            lower_center = (min(lower_x) + max(lower_x)) * 0.5
            upper_center = (min(upper_x) + max(upper_x)) * 0.5
            lateral_shift = abs(upper_center - lower_center)
        else:
            lateral_shift = 0.0
    straight_tolerance = max(0.08, connector_width * 0.025)

    return VisualModelDimensions(
        source_format=source_format,
        width_metres=width,
        length_metres=length,
        height_metres=height,
        texture_paths=textures,
        connector_width_metres=connector_width,
        lateral_center_shift_metres=lateral_shift,
        is_straight_road_candidate=lateral_shift <= straight_tolerance,
    )



def inspect_visual_surface_style(
    data: bytes,
    *,
    texture_path: str | None = None,
) -> VisualSurfaceStyle:
    """Read donor visual flags that affect road alpha/render behaviour.

    Land-deformation point bits are intentionally stripped by the shared visual
    reader before we choose a point flag. Generated road geometry is already
    terrain-fitted and must not inherit donor mesh deformation behaviour.
    """

    if data.startswith(b"ODOL"):
        lod, _land_count, _textures = _read_odol_visual(data)
        source_format = "ODOL"
    elif data.startswith(b"MLOD"):
        lod, _land_count, _textures = _read_mlod_visual(data)
        source_format = "MLOD"
    else:
        raise ProxyCloneError(
            f"unsupported P3D signature {data[:4]!r}"
        )

    if not lod.faces:
        raise ProxyCloneError("visual LOD contains no drawable faces")

    target = canonical_asset_path(texture_path or "")
    target_base = target.rsplit("\\", 1)[-1] if target else ""
    target_stem = target_base.rsplit(".", 1)[0] if "." in target_base else target_base

    def matches(face: _Face) -> bool:
        if not target:
            return bool(face.texture)
        value = canonical_asset_path(face.texture)
        if value == target:
            return True
        base = value.rsplit("\\", 1)[-1]
        stem = base.rsplit(".", 1)[0] if "." in base else base
        return stem == target_stem and bool(stem)

    selected = tuple(face for face in lod.faces if matches(face))
    if not selected:
        selected = tuple(face for face in lod.faces if face.texture)
    if not selected:
        selected = lod.faces

    face_flag = Counter(int(face.flags) for face in selected).most_common(1)[0][0]

    point_indices = tuple(
        int(vertex[0])
        for face in selected
        for vertex in face.vertices
        if 0 <= int(vertex[0]) < len(lod.point_flags)
    )
    point_values = tuple(
        int(lod.point_flags[index])
        for index in point_indices
    ) or tuple(int(value) for value in lod.point_flags)
    if not point_values:
        raise ProxyCloneError("visual LOD contains no point flags")
    point_flag = Counter(point_values).most_common(1)[0][0]

    chosen_texture = next(
        (face.texture for face in selected if face.texture),
        texture_path or "",
    )
    return VisualSurfaceStyle(
        source_format=source_format,
        texture_path=chosen_texture,
        point_flag=point_flag,
        face_flag=face_flag,
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
