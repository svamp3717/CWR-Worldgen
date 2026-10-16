# SPDX-License-Identifier: GPL-3.0-or-later
"""Experimental pure-Python OFP/CWA ODOL v7 building binarizer.

Wire layout follows CWR-CE ShapeFile.cpp/SerializeBin.cpp and its Python
animated_morph_odol fixture, not the newer Arma ODOL variants. This is a
deliberately limited, opt-in writer: unsupported model features fail loudly.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
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


def _lod_colors(
    lod: _Lod, texture_root: Path | None,
) -> tuple[int, int]:
    """Calculate Shape::_color and _colorTop as CWA does for source MLOD.

    Shape::CalculateColor averages PAA texture colors weighted by polygon
    surface area (and separately by positive Y projected area for colorTop).
    A source MLOD has neither of these packed ODOL color fields.
    """
    if texture_root is None:
        return (0xFF000000, 0xFF000000)

    sum_rgb = [0.0, 0.0, 0.0]
    sum_rgb_top = [0.0, 0.0, 0.0]
    sum_alpha = sum_alpha_top = 0.0
    area_total = area_top_total = 0.0
    for texture, _flags, corners in lod.faces:
        if len(corners) < 3:
            continue
        # CWA's MLOD loader reverses face winding before CalculateColor.
        order = [corners[1], corners[0], *reversed(corners[2:])]
        xyz = [lod.points[corner[0]][:3] for corner in order]
        area = projected = 0.0
        origin = xyz[0]
        for i in range(2, len(xyz)):
            a, b = xyz[i - 1], xyz[i]
            ab = tuple(a[j] - origin[j] for j in range(3))
            ac = tuple(b[j] - origin[j] for j in range(3))
            cross = (
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            )
            area += math.sqrt(sum(value * value for value in cross)) * 0.5
            projected += max(0.0, cross[1]) * 0.5
        if area <= 0.0:
            continue

        red, green, blue, alpha_byte = _texture_average_rgba(
            texture, texture_root,
        )
        alpha = alpha_byte / 255.0
        area_total += area
        area_top_total += projected
        weighted = area * alpha
        weighted_top = projected * alpha
        sum_alpha += weighted
        sum_alpha_top += weighted_top
        for channel, value in enumerate((red, green, blue)):
            sum_rgb[channel] += value * weighted
            sum_rgb_top[channel] += value * weighted_top

    def packed(rgb: list[float], weighted_alpha: float, total: float) -> int:
        if total <= 0.0 or weighted_alpha <= 0.0:
            return 0xFF000000  # no contributing faces: engine's PackedBlack
        channels = [
            max(0, min(255, round(value / weighted_alpha)))
            for value in rgb
        ]
        a = max(0, min(255, round(255.0 * weighted_alpha / total)))
        return ((a << 24) | (channels[0] << 16)
                | (channels[1] << 8) | channels[2])

    return (
        packed(sum_rgb, sum_alpha, area_total),
        packed(sum_rgb_top, sum_alpha_top, area_top_total),
    )


@lru_cache(maxsize=8192)
def _paa_avgc(path: Path) -> tuple[int, int, int, int]:
    """Read PAA's existing BGRA AVGC tag, without decompressing every mip."""
    data = path.read_bytes()
    if len(data) < 2 or struct.unpack_from("<H", data)[0] not in (0xFF01,):
        raise ValueError(f"not a supported CWA DXT1 PAA texture: {path}")
    offset = 2
    while data[offset:offset + 4] == b"GGAT":
        if len(data) < offset + 12:
            break
        name = data[offset + 4:offset + 8]
        length = struct.unpack_from("<I", data, offset + 8)[0]
        offset += 12
        if length > len(data) - offset:
            break
        if name == b"CGVA" and length == 4:
            blue, green, red, alpha = data[offset:offset + 4]
            return (red, green, blue, alpha)
        offset += length
    raise ValueError(f"missing/invalid AVGC color tag in {path}")


@lru_cache(maxsize=16384)
def _texture_average_rgba(texture: str, texture_root: Path) -> tuple[int, int, int, int]:
    if not texture:
        # CWA uses opaque white for faces without a bound texture.
        return (255, 255, 255, 255)
    parts = texture.replace("/", "\\").strip("\\").split("\\")
    if any(part in ("", ".", "..") or ":" in part for part in parts):
        raise ValueError(f"unsafe CWA texture reference: {texture!r}")
    root = texture_root
    # Texture paths inside MLOD are addon-prefixed, e.g.
    # wg_town\\b\\wall.paa. The source root is the addon folder.
    if parts[0].casefold() == root.name.casefold():
        parts = parts[1:]
    candidate = root.joinpath(*parts)
    if not candidate.is_file():
        raise ValueError(f"cannot compute ODOL map color; missing texture {texture!r} at {candidate}")
    return _paa_avgc(candidate)


def _write_lod(lod: _Lod, *, color: int = 0xFF000000, color_top: int = 0xFF000000) -> bytes:
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
    # FaceArray::RawSize()/ShapeSection::beg/end are offsets in the
    # *in-memory* CWA Poly stream, not bytes occupied by the ODOL wire face.
    # On 32-bit CWA, PolyProperties is 8 bytes; PolyVertices occupies
    # 2 bytes for the count plus 2 bytes per index. Thus triangles use
    # 16 bytes and quads use 18. Assuming 16 for every quad causes section
    # boundaries to drift and can select incorrect textures/material flags.
    poly_offsets = [0]
    for _texture, _special, indices in faces:
        poly_offsets.append(poly_offsets[-1] + 8 + 2 * (1 + len(indices)))
    output += _pack("II", len(faces), poly_offsets[-1])
    for texture, special, indices in faces:
        output += _pack("IhB", special, texture_names.index(texture) if texture else -1, len(indices))
        output += _pack("H" * len(indices), *indices)
    # Contiguous render sections are required by the OFP renderer.
    sections = []
    for i, (texture, special, _indices) in enumerate(faces):
        tex_index = texture_names.index(texture) if texture else -1
        beg_offset, end_offset = poly_offsets[i], poly_offsets[i + 1]
        if sections and sections[-1][2:] == (0, tex_index, special):
            beg, _end, mat, tex, spec = sections[-1]
            sections[-1] = (beg, end_offset, mat, tex, spec)
        else:
            sections.append((beg_offset, end_offset, 0, tex_index, special))
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
    output += _pack("III", color, color_top, 0)  # LOD's source-derived average colors
    output += _pack("I", 0)        # proxy count
    return bytes(output)


def mlod_to_odol(source: bytes, *, texture_root: Path | None = None) -> bytes:
    lods = read_mlod(source)
    lod_colors = [_lod_colors(lod, texture_root) for lod in lods]
    all_points = [p[:3] for lod in lods for p in lod.points]
    low, high, center, _radius_about_center = _bounds(all_points)
    geometry_lod = next(
        (lod for lod in lods if math.isclose(lod.resolution, 1e13, rel_tol=1e-6)),
        None,
    )
    geometry_points = [p[:3] for p in geometry_lod.points] if geometry_lod else []
    _, _, geometry_center, _ = _bounds(geometry_points or all_points)
    geometry_min, geometry_max, _, _ = _bounds(geometry_points or all_points)
    geometry_radius = math.dist(geometry_min, geometry_max) * 0.5
    # These authored buildings carry autocenter=0 in their Geometry LOD.
    # With that setting, the engine retains vertices relative to (0,0,0)
    # and expects an ODOL boundingCenter of *zero*. Supplying the bbox
    # midpoint as boundingCenter would shift the placed objects vertically,
    # even though every serialized LOD vertex was perfectly positioned.
    if _model_property(lods, "autocenter").strip() != "0":
        raise ValueError("ODOL procedural buildings require Geometry LOD autocenter=0")
    bounding_center = (0.0, 0.0, 0.0)
    bounding_radius = max((math.dist(p, bounding_center) for p in all_points), default=0.0)
    payload = bytearray(b"ODOL")
    payload += _pack("II", 7, len(lods))
    for lod, (color, color_top) in zip(lods, lod_colors):
        payload += _write_lod(lod, color=color, color_top=color_top)
    for lod in lods:
        payload += _pack("f", lod.resolution)
    # ODOL model metadata, according to CWR-CE ShapeFile.cpp and fixture.
    payload += _pack("i", 0)
    payload += _pack("ff", bounding_radius, geometry_radius)
    payload += _pack("iii", 0, 0, 0)
    payload += _pack("fff", *geometry_center)
    # CWA's model color is the first visual LOD's averaged texture color.
    # This is what the in-game map uses for buildings/houses.
    model_color, model_color_top = lod_colors[0]
    payload += _pack("II", model_color, model_color_top)
    payload += _pack("f", 1.0)
    payload += _pack("fff", *low) + _pack("fff", *high)
    payload += _pack("fff", *bounding_center)
    payload += _pack("fff", *geometry_center)
    payload += _pack("fff", 0., 0., 0.)
    payload += _pack("fffffffff", 1., 0., 0., 0., 1., 0., 0., 0., 1.)
    # MLOD properties such as map=house are evaluated by the engine when
    # loading an editable model. ODOL is preprocessed, so the equivalent
    # MapType enum must be saved in the model header explicitly.
    # Keep allowAnimation enabled: interior door selections may be animated
    # even when the point flags do not explicitly request land deformation.
    payload += _pack("BBBBB", 0, 0, 0, 0, 1)
    payload += _pack("b", _map_type(lods))
    # Geometry mass: indexed by actual Geometry LOD point order.
    payload += _array(
        "f", geometry_lod.mass if geometry_lod is not None else [],
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


def binarize_building(path: Path, *, texture_root: Path | None = None) -> None:
    """Atomically replace an MLOD P3D with ODOL v7, or preserve original on error."""
    original = path.read_bytes()
    try:
        odol = mlod_to_odol(original, texture_root=texture_root)
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
        source_lods = read_mlod(original)
        expected_lods = len(source_lods)
        if len(parsed_lods) != expected_lods or len(resolutions) != expected_lods:
            raise ValueError("converted ODOL LOD table is incomplete")
        for index, (source_lod, target_lod) in enumerate(zip(source_lods, parsed_lods)):
            # A structurally decodable ODOL can still silently lose a material
            # reference. Check every face's texture against its MLOD donor.
            source_textures = [tex for tex, _flags, _corners in source_lod.faces]
            target_textures = [tex for tex, _flags, _indices in target_lod.faces]
            if source_textures != target_textures:
                raise ValueError(
                    f"converted ODOL LOD {index} lost or reassigned face textures"
                )
    except (ValueError, IndexError, struct.error) as exc:
        raise ValueError(f"Cannot binarize procedural building {path}: {exc}") from exc
    temporary = path.with_suffix(".odol.tmp")
    try:
        temporary.write_bytes(odol)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
