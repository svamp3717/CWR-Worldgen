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

# CWA MapType enum in CWR-CE's World/MapTypes.hpp. Unlike MLOD,
# binarized ODOL needs this model-level value explicitly written.
MAP_TYPES = {
    "tree": 0, "small tree": 1, "bush": 2,
    "building": 3, "house": 4,
    "forest border": 5, "forest triangle": 6, "forest square": 7,
    "church": 8, "chapel": 9, "cross": 10,
    "rock": 11, "bunker": 12, "fortress": 13, "fountain": 14,
    "view-tower": 15, "lighthouse": 16, "quay": 17,
    "fuelstation": 18, "hospital": 19, "fence": 20, "wall": 21,
    "hide": 22, "busstop": 23,
}


def _model_property(lods: list["_Lod"], key: str) -> str:
    """CWA uses Geometry LOD properties before the first visual LOD."""
    geom = next(
        (lod for lod in lods if math.isclose(lod.resolution, 1e13, rel_tol=1e-6)),
        None,
    )
    if geom is not None:
        found = dict((name.casefold(), value) for name, value in geom.properties).get(key)
        if found:
            return found
    if lods:
        return dict((name.casefold(), value) for name, value in lods[0].properties).get(key, "")
    return ""


def _map_type(lods: list["_Lod"]) -> int:
    value = _model_property(lods, "map").strip().casefold()
    if value not in MAP_TYPES and value:
        raise ValueError(f"unsupported CWA map classification {value!r}")
    return MAP_TYPES.get(value, MAP_TYPES["hide"])


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
    for lod_index in range(count):
        signature, head_size, version, np, nn, nf, flags = stream.unpack("4s6I")
        if signature != b"SP3X" or head_size != 28 or version not in (1, 0x99):
            raise ValueError("unsupported MLOD SP3X header")
        if not (0 <= np <= 65535 and 0 <= nn <= 65535 and 0 <= nf <= 65535):
            raise ValueError("building LOD exceeds ODOL v7 index limits")
        points = [stream.unpack("fffI") for _ in range(np)]
        normals = [stream.unpack("fff") for _ in range(nn)]
        faces = []
        for face_index in range(nf):
            texture = stream.take(32).split(b"\0", 1)[0].decode("ascii")
            (nverts,) = stream.unpack("I")
            verts = [stream.unpack("iiff") for _ in range(4)]
            (flag,) = stream.unpack("I")
            if nverts not in (3, 4):
                raise ValueError("ODOL only supports triangles and quads")
            face_vertices = verts[:nverts]
            for corner_index, (point, normal, _u, _v) in enumerate(face_vertices):
                if not 0 <= point < np:
                    raise ValueError(
                        f"MLOD LOD {lod_index} face {face_index} corner {corner_index} "
                        f"references point {point} (point count {np})"
                    )
                # Worldgen deliberately writes -1 for faces without a normal,
                # especially collision Geometry LODs. Other generated collision
                # LODs use 0 even with an empty normals table.
                if normal != -1 and not (0 <= normal < nn or (nn == 0 and normal == 0)):
                    raise ValueError(
                        f"MLOD LOD {lod_index} face {face_index} corner {corner_index} "
                        f"references normal {normal} (normal count {nn})"
                    )
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
    """Convert raw Oxygen MLOD POINT_* flags into CWA ODOL ClipFlags.

    Matches CWR-CE MLODLoader::convertPointFlagsToClipFlags.
    """
    value = int(flag) & 0xFFFFFFFF
    all_flags = 0x0F | 0xF0 | 0x300 | 0x3000 | 0xFF0000 | 0x0F000000
    if value & ~all_flags:
        value = 0
    result = 0x3F  # ClipAll: clipping-plane flags are always present.
    if value & 0x1:
        result |= 0x100
    elif value & 0x2:
        result |= 0x200
    elif value & 0x4:
        result |= 0x400
    elif value & 0x8:
        result |= 0x800
    if value & 0x100:
        result |= 0x1000
    elif value & 0x200:
        result |= 0x2000
    if value & 0x10:
        result |= 200 * 0x100000  # MSShining
    elif value & 0x40:
        result |= 203 * 0x100000  # MSFullLighted
    elif value & 0x80:
        result |= 202 * 0x100000  # MSHalfLighted
    elif value & 0x20:
        result |= 201 * 0x100000  # MSInShadow
    if value & 0x1000:
        result |= 0x4000
    elif value & 0x2000:
        result |= 0x8000
    if value & 0xFF0000:
        result |= ((value >> 16) & 0xFF) * 0x100000
    return result & 0xFFFFFFFF


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
            # A -1 MLOD normal is the intentional no-normal sentinel, not
            # Python's "last element" indexing. Preserve its absence as a
            # zero normal instead of accidentally inheriting another face's.
            normals.append(lod.normals[ni] if ni >= 0 and lod.normals else (0., 0., 0.))
            uvs.append((u, v))
            flags.append(_point_flag(raw_flag))
            vertex_points.append(pi)
            indices.append(len(verts) - 1)
        # ODOL and MLOD polygon corner order differ for OFP.
        faces.append((texture, _face_flag(face_flag), [indices[1], indices[0], *reversed(indices[2:])]))
    bounds_points = [p[:3] for p in lod.points]
    low, high, center, radius = _bounds(bounds_points)
    output += _array("I", flags, compressed=True)
    # CWA Shape::SerializeBin uses TransferBinaryArray(_tex), which switches
    # UV payloads to SSCompress LZSS at 1024 bytes (128 UV pairs). Leaving
    # them raw desynchronizes the ODOL decoder on detailed buildings.
    output += _array("ff", uvs, compressed=True)
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
        output += _array("B", [255] * len(face_ids), compressed=True)
        output += _array("I", [], compressed=True)
        output += _pack("B", 0)
        output += _array("I", [], compressed=True)
        output += _array("H", selected_vertices, compressed=True)
        output += _array("B", [point_weights[vertex_points[i]] for i in selected_vertices], compressed=True)
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
    # MLOD properties such as map=house are evaluated by the engine when
    # loading an editable model. ODOL is preprocessed, so the equivalent
    # MapType enum must be saved in the model header explicitly.
    payload += _pack("BBBBB", 0, 0, 0, 0, 0)
    payload += _pack("b", _map_type(lods))
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
    try:
        odol = mlod_to_odol(original)
        if odol[:8] != b"ODOL" + _pack("I", 7):
            raise ValueError("invalid ODOL version")
        # Independent decoder smoke test before the source file is replaced.
        # This checks the first visual LOD, but it is not in-game validation.
        from .legacy_proxy_models import (
            _read_odol_surface_lods, inspect_visual_model_dimensions,
        )
        report = inspect_visual_model_dimensions(odol)
        if report.source_format != "ODOL" or report.width_metres <= 0:
            raise ValueError("converted ODOL failed independent model inspection")
        # Check every LOD's arrays, face records, sections and selections.
        # Inspecting only the first visual LOD can miss format mistakes in
        # collision, Memory, Paths or the distant visual LODs.
        parsed_lods, resolutions = _read_odol_surface_lods(odol)
        expected_lods = struct.unpack_from("<I", odol, 8)[0]
        if len(parsed_lods) != expected_lods or len(resolutions) != expected_lods:
            raise ValueError("converted ODOL LOD table is incomplete")
    except (ValueError, IndexError, struct.error) as exc:
        raise ValueError(f"Cannot binarize procedural building {path}: {exc}") from exc
    temporary = path.with_suffix(".odol.tmp")
    try:
        temporary.write_bytes(odol)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
