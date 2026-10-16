# SPDX-License-Identifier: GPL-3.0-or-later
"""Experimental pure-Python OFP/CWA ODOL v7 building binarizer.

Wire layout follows CWR-CE ShapeFile.cpp/SerializeBin.cpp and its Python
animated_morph_odol fixture, not the newer Arma ODOL variants. This is a
deliberately limited, opt-in writer: unsupported model features fail loudly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from pathlib import Path

P = struct.Struct("<I")
SPECIAL_LODS = {
    1.0e13: 1,      # Geometry
    1.0e15: 0,      # Memory
    2.0e15: 8,      # LandContact
    3.0e15: 9,      # Roadway
    4.0e15: 10,     # Paths
}


def _pack(fmt: str, *args: object) -> bytes:
    return struct.pack("<" + fmt, *args)


def _string(value: str) -> bytes:
    return value.encode("ascii") + b"\0"


def _array(fmt: str, elements: list[object], *, compressed: bool = False) -> bytes:
    data = b"".join(
        _pack(fmt, *item) if isinstance(item, tuple) else _pack(fmt, item)
        for item in elements
    )
    return P.pack(len(elements)) + (_compress_literals(data) if compressed else data)


def _compress_literals(data: bytes) -> bytes:
    """SSCompress-compatible, literal-only LZSS encoding with sum checksum.

    No attempt at size reduction: this is a correctness-first format writer.
    ODOL arrays are LZSS-compressed at 1024 bytes, including checksum.
    """
    if len(data) < 1024:
        return data
    out = bytearray()
    for offset in range(0, len(data), 8):
        part = data[offset:offset + 8]
        out.append((1 << len(part)) - 1)
        out.extend(part)
    out.extend(_pack("I", sum(data) & 0xFFFFFFFF))
    return bytes(out)


@dataclass
class _Lod:
    resolution: float
    points: list[tuple[float, float, float, int]]
    normals: list[tuple[float, float, float]]
    faces: list[tuple[str, int, list[tuple[int, int, float, float]]]]
    selections: list[tuple[str, bytes, bytes]]
    properties: list[tuple[str, str]]
    mass: list[float]


class _Reader:
    def __init__(self, payload: bytes):
        self.data, self.pos = payload, 0

    def take(self, size: int) -> bytes:
        if size < 0 or size > len(self.data) - self.pos:
            raise ValueError("truncated MLOD input")
        result = self.data[self.pos:self.pos + size]
        self.pos += size
        return result

    def unpack(self, fmt: str):
        st = struct.Struct("<" + fmt)
        return st.unpack(self.take(st.size))


def read_mlod(source: bytes) -> list[_Lod]:
    stream = _Reader(source)
    magic, major, minor, reserved, count = stream.unpack("4sBBHI")
    if magic != b"MLOD" or major != 1 or reserved or not 1 <= count <= 100:
        raise ValueError("expected OFP MLOD with valid header")
    lods: list[_Lod] = []
    for _ in range(count):
        signature, head_size, version, np, nn, nf, flags = stream.unpack("4s6I")
        if signature != b"SP3X" or head_size != 28 or version not in (1, 0x99):
            raise ValueError("unsupported MLOD SP3X header")
        if not (0 <= np <= 65535 and 0 <= nn <= 65535 and 0 <= nf <= 65535):
            raise ValueError("building LOD exceeds ODOL v7 index limits")
        points = [stream.unpack("fffI") for _ in range(np)]
        normals = [stream.unpack("fff") for _ in range(nn)]
        faces = []
        for _ in range(nf):
            texture = stream.take(32).split(b"\0", 1)[0].decode("ascii")
            (nverts,) = stream.unpack("I")
            verts = [stream.unpack("iiff") for _ in range(4)]
            (flag,) = stream.unpack("I")
            if nverts not in (3, 4):
                raise ValueError("ODOL only supports triangles and quads")
            face_vertices = verts[:nverts]
            if any(not (0 <= p < np and 0 <= n < nn) for p, n, _, _ in face_vertices):
                raise ValueError("MLOD face references a missing point/normal")
            faces.append((texture, flag, face_vertices))
        if stream.take(4) != b"TAGG":
            raise ValueError("missing MLOD TAGG section")
        selections = []
        properties = []
        mass = []
        while True:
            rawname, size = stream.unpack("64sI")
            name = rawname.split(b"\0", 1)[0].decode("ascii")
            payload = stream.take(size)
            if name == "#EndOfFile#":
                break
            if name == "#Property#":
                if size != 128:
                    raise ValueError("invalid model property")
                properties.append(tuple(x.split(b"\0", 1)[0].decode("ascii") for x in (payload[:64], payload[64:])))
            elif name == "#Mass#":
                if size != np * 4:
                    raise ValueError("invalid mass table")
                mass = [v[0] for v in struct.iter_unpack("<f", payload)]
            elif name == "#Animation#":
                raise ValueError("ODOL experiment does not yet support morph animation")
            elif name.startswith("proxy:"):
                raise ValueError("ODOL experiment does not yet support P3D proxy selections")
            elif not name.startswith(("#", "-", ".")):
                if size != np + nf:
                    raise ValueError("invalid named selection length")
                selections.append((name, payload[:np], payload[np:]))
        (resolution,) = stream.unpack("f")
        lods.append(_Lod(resolution, points, normals, faces, selections, properties, mass))
    if stream.pos != len(source):
        raise ValueError("trailing data in MLOD")
    return lods


def _bounds(points: list[tuple[float, float, float]]):
    if not points:
        return ((0., 0., 0.),) * 3 + (0.,)
    low = tuple(min(p[i] for p in points) for i in range(3))
    high = tuple(max(p[i] for p in points) for i in range(3))
    center = tuple((a + b) * .5 for a, b in zip(low, high))
    radius = max(math.dist(p, center) for p in points)
    return low, high, center, radius


def _point_flag(flag: int) -> int:
    """CWA ODOL ClipFlags encoding corresponding to MLOD point surface bits."""
    value = int(flag)
    # The MLOD low nibble stores the surface class; ODOL material occupies
    # bits 8..11. Higher rendering flags follow legacy conversion mapping.
    return ((value & 0xF) << 8) | ((value & 0x300) << 4) | ((value & 0x3000) << 2) | ((value & 0xFF0000) << 4)


def _face_flag(flag: int) -> int:
    value = int(flag)
    result = 0
    if value & 0x8:
        result |= 0x40
    if value & 0x10:
        result |= 0x20
    if value & 0x100:
        result |= 0x04000000
    if value & 0x200:
        result |= 0x08000000
    if value & 0x01000000:
        result |= 0x20000000
    return result


def _write_lod(lod: _Lod) -> bytes:
    output = bytearray()
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []
    flags: list[int] = []
    vertex_points: list[int] = []
    # ODOL Memory, LandContact and selection-only LODs can have no faces.
    # Preserve their unreferenced points rather than deriving all vertices from
    # face corners, which would silently delete door axes and path endpoints.
    for index, (x, y, z, flag) in enumerate(lod.points):
        verts.append((x, y, z))
        normals.append((0., 0., 0.))
        uvs.append((0., 0.))
        flags.append(_point_flag(flag))
        vertex_points.append(index)
    faces = []
    texture_names = list(dict.fromkeys(t for t, _, _ in lod.faces if t))
    for texture, face_flag, corners in lod.faces:
        indices = []
        for pi, ni, u, v in corners:
            if len(verts) >= 65535:
                raise ValueError("ODOL vertex limit exceeded: 65535 vertices per LOD")
            x, y, z, raw_flag = lod.points[pi]
            verts.append((x, y, z))
            normals.append(lod.normals[ni])
            uvs.append((u, v))
            flags.append(_point_flag(raw_flag))
            vertex_points.append(pi)
            indices.append(len(verts) - 1)
        # ODOL and MLOD polygon corner order differ for OFP.
        faces.append((texture, _face_flag(face_flag), [indices[1], indices[0], *reversed(indices[2:])]))
    bounds_points = [p[:3] for p in lod.points]
    low, high, center, radius = _bounds(bounds_points)
    output += _array("I", flags, compressed=True)
    output += _array("ff", uvs)
    output += _array("fff", verts)
    output += _array("fff", normals)
    output += _pack("ii", 0, 0)       # per-LOD hints
    output += _pack("fff", *low) + _pack("fff", *high)
    output += _pack("fff", *center) + _pack("f", radius)
    output += _pack("I", len(texture_names))
    for name in texture_names:
        output += _string(name)
    # Point/vertex lookup. Point entries refer to the first corresponding
    # (possibly UV-split) vertex; vertex entries map back to source points.
    first = {}
    for index, point in enumerate(vertex_points):
        first.setdefault(point, index)
    output += _array("H", [first.get(i, 0) for i in range(len(lod.points))], compressed=True)
    output += _array("H", vertex_points, compressed=True)
    # The in-memory Poly entries use 4-byte alignment. The stock sil25
    # example has two 16-byte quad records, despite only 15 wire bytes each.
    output += _pack("II", len(faces), 16 * len(faces))
    for texture, special, indices in faces:
        output += _pack("IhB", special, texture_names.index(texture) if texture else -1, len(indices))
        output += _pack("H" * len(indices), *indices)
    # Contiguous render sections are required by the OFP renderer.
    sections = []
    for i, (texture, special, _indices) in enumerate(faces):
        tex_index = texture_names.index(texture) if texture else -1
        if sections and sections[-1][2:] == (0, tex_index, special):
            beg, _end, mat, tex, spec = sections[-1]
            sections[-1] = (beg, (i + 1) * 16, mat, tex, spec)
        else:
            sections.append((i * 16, (i + 1) * 16, 0, tex_index, special))
    output += _pack("I", len(sections))
    for beg, end, mat, tex, special in sections:
        output += _pack("IIihI", beg, end, mat, tex, special)
    output += _pack("I", len(lod.selections))
    for name, point_weights, face_flags in lod.selections:
        face_ids = [i for i, flag in enumerate(face_flags) if flag]
        selected_vertices = [i for i, point in enumerate(vertex_points) if point_weights[point]]
        output += _string(name)
        output += _array("H", face_ids, compressed=True)
        output += _array("B", [255] * len(face_ids))
        output += _array("I", [], compressed=True)
        output += _pack("B", 0)
        output += _array("I", [], compressed=True)
        output += _array("H", selected_vertices, compressed=True)
        output += _array("B", [point_weights[vertex_points[i]] for i in selected_vertices])
    output += _pack("I", len(lod.properties))
    for name, value in lod.properties:
        output += _string(name) + _string(value)
    output += _pack("I", 0)        # morph animation count
    output += _pack("III", 0xFFFFFFFF, 0xFFFFFFFF, 0) # colors, special
    output += _pack("I", 0)        # proxy count
    return bytes(output)


def mlod_to_odol(source: bytes) -> bytes:
    lods = read_mlod(source)
    all_points = [p[:3] for lod in lods for p in lod.points]
    low, high, center, radius = _bounds(all_points)
    payload = bytearray(b"ODOL")
    payload += _pack("II", 7, len(lods))
    for lod in lods:
        payload += _write_lod(lod)
    for lod in lods:
        payload += _pack("f", lod.resolution)
    # ODOL model metadata, according to CWR-CE ShapeFile.cpp and fixture.
    payload += _pack("i", 0)
    payload += _pack("ff", radius, radius)
    payload += _pack("iii", 0, 0, 0)
    payload += _pack("fff", *center)
    payload += _pack("II", 0xFFFFFFFF, 0xFFFFFFFF)
    payload += _pack("f", 1.0)
    payload += _pack("fff", *low) + _pack("fff", *high)
    payload += _pack("fff", *center) * 2
    payload += _pack("fff", 0., 0., 0.)
    payload += _pack("fffffffff", 1., 0., 0., 0., 1., 0., 0., 0., 1.)
    payload += _pack("BBBBB", 0, 0, 0, 0, 1)
    payload += _pack("b", 0)
    # Geometry mass: indexed by actual Geometry LOD point order.
    geometry_index = next(
        (i for i, lod in enumerate(lods) if math.isclose(lod.resolution, 1e13, rel_tol=1e-6)), None,
    )
    payload += _array(
        "f", lods[geometry_index].mass if geometry_index is not None else [],
        compressed=True,
    )
    payload += _pack("ffff", 0., 0., 0., 0.)
    indices = [-1] * 12
    for i, lod in enumerate(lods):
        for resolution, index in SPECIAL_LODS.items():
            if math.isclose(lod.resolution, resolution, rel_tol=1e-6):
                indices[index] = i
                break
    payload += _pack("b" * 12, *indices)
    return bytes(payload)


def binarize_building(path: Path) -> None:
    """Atomically replace an MLOD P3D with ODOL v7, or preserve original on error."""
    original = path.read_bytes()
    odol = mlod_to_odol(original)
    if odol[:8] != b"ODOL" + _pack("I", 7):
        raise ValueError("invalid ODOL version")
    temporary = path.with_suffix(".odol.tmp")
    try:
        temporary.write_bytes(odol)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
