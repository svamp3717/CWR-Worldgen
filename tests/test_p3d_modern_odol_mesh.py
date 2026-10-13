"""Synthetic ODOL v40-v75 resolution LOD fixtures for real mesh previews."""
from __future__ import annotations

from pathlib import Path
import struct
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import p3d_modern_odol_mesh as odol
import p3d_preview_geometry as geometry


def pack(value: int) -> bytes:
    return struct.pack("<I", value)


def _lzo_literals(payload: bytes) -> bytes:
    """One LZO1X literal run with the format's normal end marker."""
    n = len(payload)
    assert n >= 1024
    length = n - 3 - 15
    zeros, last = divmod(length - 1, 255)
    # Extended LZO lengths add 255 for each zero, then base + final byte.
    return b"\x00" + b"\x00" * zeros + bytes((last + 1,)) + payload + b"\x11\x00\x00"


def _lzss_literals(payload: bytes) -> bytes:
    out = bytearray()
    for start in range(0, len(payload), 8):
        part = payload[start:start + 8]
        out.append((1 << len(part)) - 1)
        out.extend(part)
    out.extend(pack(sum(payload) & 0xFFFFFFFF))
    return bytes(out)


def _array(version: int, data: bytes) -> bytes:
    if not data:
        return b""
    if version >= 64:
        # Modern ODOL explicitly flags compressed blocks.
        return (b"\x01" + _lzo_literals(data)) if len(data) >= 1024 else (b"\x00" + data)
    if version >= 44:
        return _lzo_literals(data) if len(data) >= 1024 else data
    return _lzss_literals(data) if len(data) >= 1024 else data


def _lod(version: int, vertices: int) -> bytes:
    result = bytearray()
    result.extend(pack(0))  # proxies
    result.extend(pack(0))  # subskeleton indices
    result.extend(pack(0))  # skeleton-to-subskeleton sets
    if version >= 50:
        result.extend(pack(vertices))
    else:
        result.extend(pack(0) + b"\x00")  # condensed ClipOldFormat
    if version >= 51:
        result.extend(struct.pack("<f", 2.0))  # face area
    result.extend(b"\x00" * (8 + 40))  # hints, LOD bounds and radius
    result.extend(pack(1) + b"ca\\data\\brick.paa\x00")  # LOD texture table
    result.extend(pack(0))  # embedded materials
    result.extend(pack(0) + pack(0))  # PointToVertex / VertexToPoint
    result.extend(pack(2) + pack(0) + struct.pack("<H", 0))  # face table
    idxfmt = "<3I" if version >= 69 else "<3H"
    result.extend(b"\x03" + struct.pack(idxfmt, 0, 1, 2))
    result.extend(b"\x03" + struct.pack(idxfmt, 0, 2, 3))

    result.extend(pack(1))  # sections
    extent = 32 if version >= 69 else 16
    result.extend(struct.pack("<5i", 0, extent, 0, 0, 0))
    result.extend(struct.pack("<h", 0))  # texture index
    result.extend(pack(0) + struct.pack("<i", 0))  # flags and material
    result.extend(pack(0))  # area over texture array
    if version >= 67:
        result.extend(pack(0))  # flag67

    result.extend(pack(0) * 3)  # named selections, properties, keyframes
    result.extend(pack(0) * 3 + b"\x00")  # colors and simple bone refs
    result.extend(pack(0))  # rest-of-LOD data size
    if version >= 50:
        result.extend(pack(0) + b"\x00")  # clip condensed array
    if version >= 45:
        result.extend(struct.pack("<4f", 0.0, 0.0, 1.0, 1.0))

    result.extend(pack(vertices) + b"\x00")  # UVSet0: no default fill
    if version >= 45:
        uv = b"".join(struct.pack("<hh", int(-32767 + (i % 4) * 20000), -32767) for i in range(vertices))
    else:
        uv = b"".join(struct.pack("<ff", (i % 4) / 4, 0.0) for i in range(vertices))
    result.extend(_array(version, uv))
    result.extend(pack(1))  # UV set count

    points = [(float(i % 4), float((i // 4) % 3), float(i % 3)) for i in range(vertices)]
    raw = b"".join(struct.pack("<3f", *p) for p in points)
    result.extend(pack(vertices))
    result.extend(_array(version, raw))
    return bytes(result)


def full_odol(
    version: int, *, vertices: int = 4,
    lod_specs: tuple[tuple[float, int], ...] | None = None,
) -> bytes:
    """Build plaintext ODOL with selectable visual and nonvisual LOD fixtures."""
    specs = lod_specs if lod_specs is not None else ((1.0, vertices),)
    lods = tuple(_lod(version, count) for _resolution, count in specs)
    lod_count = len(specs)
    out = bytearray(b"ODOL" + pack(version))
    if version >= 75:
        out.extend(pack(0) + pack(0))
    if version >= 59:
        out.extend(pack(123))
    if version >= 58:
        out.extend(b"muzzle\x00")
    out.extend(pack(lod_count))
    out.extend(struct.pack("<" + "f" * lod_count, *(res for res, _ in specs)))
    out.extend(b"\x00" * 48 + struct.pack("<6f", -1, -2, -3, 4, 5, 6))
    if version >= 70:
        out.extend(pack(0))
    if version >= 71:
        out.extend(pack(0))
    if version >= 52:
        out.extend(b"\x00" * 24)
    out.extend(b"\x00" * 72)  # remaining centres and inertia
    out.extend(b"\x00" * 4)  # boolean flags
    if version >= 73:
        out.extend(b"\x00")
    if version >= 42:
        out.extend(b"\x00" * 16)
    if version >= 43:
        out.extend(b"\x00" * 8)
    out.extend(b"\x00")  # force not alpha
    if version >= 37:
        out.extend(b"\x00" * 5)
    if version >= 48:
        out.extend(b"\x00" * 4)
    out.extend(b"\x00\x00")  # animated + empty skeleton name
    out.extend(b"\x00")  # map type
    out.extend(pack(0))  # mass array count
    out.extend(b"\x00" * 16)  # mass and armor
    if version >= 72:
        out.extend(b"\x00" * 4)
    if version >= 53:
        out.extend(b"\x00")
    if version >= 54:
        out.extend(b"\x00")
    out.extend(b"\x00" * 16)  # geometry flags and minShadow
    if version >= 38:
        out.extend(b"\x00")
    out.extend(b"\x00\x00\x00")  # class and damage strings, frequent
    out.extend(pack(0))  # unused31
    if version >= 57:
        out.extend(pack(0) * 3 * lod_count)  # shadow LOD preferences
    out.extend(b"\x00")  # animations absent
    address_pos = len(out)
    out.extend(pack(0) * lod_count * 2 + b"\x01" * lod_count)
    starts = []
    ends = []
    for lod in lods:
        starts.append(len(out))
        out.extend(lod)
        ends.append(len(out))
    for index, offset in enumerate(starts):
        struct.pack_into("<I", out, address_pos + 4 * index, offset)
    for index, offset in enumerate(ends):
        struct.pack_into("<I", out, address_pos + 4 * (lod_count + index), offset)
    return bytes(out)


@pytest.mark.parametrize("version", [40, 43, 44, 45, 49, 50, 52, 58, 59, 60, 64, 67, 69, 73, 74, 75])
def test_actual_faces_render_for_arma_odol_versions(version: int) -> None:
    data = full_odol(version)
    before = bytes(data)
    model = odol.read_model_mesh(data)
    assert data == before
    assert model.version == version
    assert model.points.shape == (4, 3)
    assert len(model.faces) == 2
    assert model.faces[0][0] == (0, 1, 2)
    assert model.faces[1][0] == (0, 2, 3)
    assert model.faces[0][2] == r"ca\data\brick.paa"
    assert model.textures == (r"ca\data\brick.paa",)
    assert len(model.faces[0][1]) == 3

    measured, points, faces, textures = geometry.extract_first_lod_geometry(
        data, model_path="ca\\building.p3d", source="sample.pbo",
    )
    assert measured.version == version
    assert measured.vertex_count == 4
    assert measured.lod.startswith("resolution ")
    assert points.shape == (4, 3)
    assert len(faces) == 2
    assert textures == (r"ca\data\brick.paa",)
    assert faces[0].texture_path == r"ca\data\brick.paa"


@pytest.mark.parametrize("version", [40, 49, 64, 75])
def test_real_mesh_decodes_large_compressed_vertex_block(version: int) -> None:
    model = odol.read_model_mesh(full_odol(version, vertices=100))
    assert len(model.points) == 100
    assert len(model.faces) == 2
    assert model.points[99, 0] == pytest.approx(3.0)


def test_corrupt_modern_mesh_still_uses_explicit_bounds_fallback() -> None:
    data = full_odol(49)[:-10]
    m, points, faces, _ = geometry.extract_first_lod_geometry(
        data, model_path="bad.p3d", source="fixture",
    )
    assert m.lod == "model bounds only"
    assert m.vertex_count == 0
    assert points.shape == (8, 3)
    assert not faces


def test_real_mesh_uses_textured_renderer(monkeypatch) -> None:
    import p3d_asset_browser_app as browser

    captured = {}

    def fake_render(points, faces, resolver, source, **kwargs):
        captured["points"] = points
        captured["faces"] = faces
        captured["source"] = source
        captured["args"] = kwargs
        return np.zeros((2, 3, 3), dtype=np.uint8), 2, 0

    monkeypatch.setattr(browser, "render_textured_model", fake_render)
    app = browser.AssetBrowserApp.__new__(browser.AssetBrowserApp)
    app.ax_preview = Mock()
    app.canvas = Mock()
    app.figure = Mock()
    app.texture_resolver = object()
    app.skip_textures_var = SimpleNamespace(get=lambda: False)
    app.azim, app.elev, app.zoom = 35.0, 25.0, 1.0
    measurement, points, faces, _ = geometry.extract_first_lod_geometry(
        full_odol(49), model_path="test.p3d", source="roads.pbo",
    )
    model = geometry.PreviewModel(
        "test.p3d", "roads.pbo", measurement, points, faces,
        geometry.collect_edges(faces), len(points), (r"ca\data\brick.paa",), "textured mesh",
    )
    app._draw_model(model)
    assert len(captured["faces"]) == 2
    assert captured["faces"][0].texture_path == r"ca\data\brick.paa"
    assert captured["source"] == "roads.pbo"
    assert "bounds only" not in app.ax_preview.set_title.call_args.args[0]



def _synthetic_rvmat(stage_textures: tuple[str, ...], *, version: int = 9) -> bytes:
    """Tiny embedded RVMAT with texture stages but no transforms."""
    out = bytearray(b"ca\\data\\test.rvmat\x00")
    out.extend(pack(version))
    out.extend(bytes(6 * 16))  # emissive, ambient, diffuse, specular etc.
    out.extend(bytes(20))  # specular power, shader IDs, lighting and fog
    out.extend(b"\x00")  # surface file (v6+)
    out.extend(bytes(8))  # render flags (v4+)
    out.extend(pack(len(stage_textures)))
    out.extend(pack(0))  # transform count (v9+)
    for texture in stage_textures:
        out.extend(pack(0))  # filter
        out.extend(texture.encode("latin-1") + b"\x00")
        out.extend(pack(0))  # stage ID
    return bytes(out)


def test_material_selects_diffuse_after_normal_and_specular_stages() -> None:
    materials = (
        r"ca\structures_e\data\wood\ground_wood01_nohq.paa",
        r"ca\structures_e\data\wood\ground_wood01_smdi.paa",
        r"ca\structures_e\data\wood\ground_wood01_co.paa",
    )
    r = odol._Reader(_synthetic_rvmat(materials), 49)
    assert odol._material(r) == materials[2]


def test_material_does_not_paint_normal_map_as_a_colour_texture() -> None:
    materials = (
        r"ca\structures_e\data\stone\wall_nohq.paa",
        r"ca\structures_e\data\stone\wall_smdi.paa",
    )
    assert odol._material_diffuse_texture(list(materials)) == ""
    r = odol._Reader(_synthetic_rvmat(materials), 49)
    assert odol._material(r) == ""



@pytest.mark.parametrize("wrapped", [False, True])
def test_actual_odol_v49_mesh_and_paa_open_inside_ironfront_ifa(
    tmp_path: Path, wrapped: bool,
) -> None:
    """Opening a real decoded model through .ifa must be equivalent to .pbo."""
    import zstandard
    from cwr_worldgen.pbo import PboEntry, write_pbo
    from p3d_asset_browser_catalog import load_model, model_source, scan_catalogue
    from p3d_texture_sources import TextureResolver

    # LOD texture path in full_odol(49) is ca\data\brick.paa.
    brick = (
        struct.pack("<H", 0xFF01)  # DXT1
        + struct.pack("<H", 0)     # palette entries
        + struct.pack("<HH", 4, 4)
        + b"\x08\x00\x00"
        + struct.pack("<HHI", 0xF800, 0, 0)
    )
    archive = tmp_path / "ca.ifa"
    write_pbo(
        archive,
        (
            PboEntry("building.p3d", full_odol(49)),
            PboEntry(r"data\brick.paa", brick),
        ),
    )
    if wrapped:
        compressed = tmp_path / "ca.ifa.zst"
        compressed.write_bytes(zstandard.ZstdCompressor().compress(archive.read_bytes()))
        archive.unlink()
        archive = compressed
    before = archive.read_bytes()

    catalogue = scan_catalogue((archive,), use_cache=False)
    assert [item.path for item in catalogue.models] == [r"ca\building.p3d"]
    assert [item.path for item in catalogue.textures] == [r"ca\data\brick.paa"]
    model = load_model(catalogue.models[0], catalogue)
    assert model.measurement.version == 49
    assert model.points.shape == (4, 3)
    assert len(model.faces) == 2
    assert model.faces[0].texture_path == r"ca\data\brick.paa"

    resolver = TextureResolver((archive,))
    image = resolver.load_rgba(r"ca\data\brick.paa", model_source(catalogue.models[0]))
    assert image is not None
    assert image.shape == (4, 4, 4)
    assert tuple(image[0, 0]) == (255, 0, 0, 255)
    assert archive.read_bytes() == before



def test_fast_lod_uses_lower_detail_mesh_without_decoding_full_lod(monkeypatch) -> None:
    # Stored LOD order need not match numerical resolution order.
    data = full_odol(49, lod_specs=(
        (3.0, 50), (1.0, 200), (1.0e13, 10), (5.0, 4), (2.0, 100),
    ))
    observed = []
    original = odol._read_resolution_lod

    def tracked(reader, resolution, start, end):
        observed.append(resolution)
        return original(reader, resolution, start, end)

    monkeypatch.setattr(odol, "_read_resolution_lod", tracked)
    full = odol.read_model_mesh(data, lod_quality="full")
    assert full.resolution == pytest.approx(1.0)
    assert len(full.points) == 200
    assert observed == [1.0]

    observed.clear()
    fast = odol.read_model_mesh(data, lod_quality="fast")
    assert fast.resolution == pytest.approx(3.0)
    assert len(fast.points) == 50
    assert len(fast.faces) == len(full.faces)
    assert observed == [3.0]  # No high-detail, geometry, or shadow parsing.

    measured, points, faces, textures = geometry.extract_first_lod_geometry(
        data, model_path="example.p3d", source="models.pbo", lod_quality="fast",
    )
    assert measured.lod == "resolution 3"
    assert len(points) == 50
    assert len(faces) == 2
    assert textures == (r"ca\data\brick.paa",)


def test_fast_lod_falls_back_to_another_visual_lod(monkeypatch) -> None:
    data = full_odol(49, lod_specs=((1.0, 100), (2.0, 25), (3.0, 4)))
    original = odol._read_resolution_lod
    visited = []

    def failed_low_detail(reader, resolution, start, end):
        visited.append(resolution)
        if resolution == 3.0:
            raise odol.MeshReadError("test invalid low-detail geometry")
        return original(reader, resolution, start, end)

    monkeypatch.setattr(odol, "_read_resolution_lod", failed_low_detail)
    mesh = odol.read_model_mesh(data, lod_quality="fast")
    assert mesh.resolution == pytest.approx(2.0)
    assert visited == [3.0, 2.0]


def test_fast_lod_keeps_one_lod_models_and_rejects_unknown_mode() -> None:
    data = full_odol(75)
    assert odol.read_model_mesh(data, lod_quality="fast").resolution == 1.0
    with pytest.raises(ValueError, match="unsupported preview LOD"):
        odol.read_model_mesh(data, lod_quality="ultra")
