from __future__ import annotations

from pathlib import Path
import struct
import sys

import zstandard

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import p3d_texture_io as texture_io
from p3d_texture_io import NestedPBOAssetRef, PBOAssetRef
from p3d_texture_sources import TextureResolver, sibling_namespace_pbo, sibling_namespace_pbos


_PBO_ENTRY = struct.Struct("<IIIII")


def _write_pbo(path: Path, entries: dict[str, bytes]) -> None:
    header = bytearray()
    payload = bytearray()
    for name, data in entries.items():
        encoded = name.encode("latin-1") + b"\0"
        header += encoded
        header += _PBO_ENTRY.pack(0, len(data), 0, 0, len(data))
        payload += data
    header += b"\0" + _PBO_ENTRY.pack(0, 0, 0, 0, 0)
    path.write_bytes(bytes(header + payload))


def test_sibling_namespace_pbo_matches_texture_prefix(tmp_path: Path) -> None:
    source = tmp_path / "Data3D.pbo"
    texture_pbo = tmp_path / "Data.pbo"
    source.touch()
    texture_pbo.touch()

    assert sibling_namespace_pbo(source, r"data\strecha_af.pac") == texture_pbo
    assert sibling_namespace_pbo(source, r"o\hous\wall.pac") is None


def test_resolver_loads_texture_from_sibling_namespace_pbo(tmp_path: Path) -> None:
    source = tmp_path / "Data3D.pbo"
    texture_pbo = tmp_path / "Data.pbo"
    _write_pbo(source, {})
    payload = b"synthetic-pac-bytes"
    _write_pbo(texture_pbo, {"strecha_af.pac": payload})

    resolver = TextureResolver([source])
    model_source = f"{source}!data3d\\afdum_mesto3.p3d"

    ref = resolver._find_ref(r"data\strecha_af.pac", model_source)
    assert isinstance(ref, PBOAssetRef)
    assert ref.pbo_path == texture_pbo
    assert resolver.load_bytes(r"data\strecha_af.pac", model_source) == payload


def test_explicit_texture_source_can_live_elsewhere(tmp_path: Path) -> None:
    models = tmp_path / "models"
    textures = tmp_path / "textures"
    models.mkdir()
    textures.mkdir()

    source = models / "Data3D.pbo"
    texture_pbo = textures / "Data.pbo"
    _write_pbo(source, {})
    payload = b"external-pac-bytes"
    _write_pbo(texture_pbo, {"domek1_fr.pac": payload})

    resolver = TextureResolver([source, texture_pbo])
    model_source = f"{source}!data3d\\afdum_mesto3.p3d"

    assert resolver.load_bytes(r"data\domek1_fr.pac", model_source) == payload

def test_resolver_loads_texture_from_nested_pbo_in_zstd_mod_package(tmp_path: Path) -> None:
    inner = tmp_path / "inner.pbo"
    payload = b"nested-texture-bytes"
    _write_pbo(inner, {"wall.paa": payload})

    outer = tmp_path / "outer.pbo"
    _write_pbo(
        outer,
        {
            r"addons\inner.pbo": inner.read_bytes(),
            r"lib_models\house.p3d": b"synthetic-model",
        },
    )
    wrapped = tmp_path / "mod-package.pbo.zst"
    wrapped.write_bytes(zstandard.ZstdCompressor(level=1).compress(outer.read_bytes()))

    resolver = TextureResolver([wrapped])
    model_source = f"{wrapped}!lib_models\\house.p3d"

    ref = resolver._find_ref(r"inner\wall.paa", model_source)
    assert isinstance(ref, NestedPBOAssetRef)
    assert ref.source.endswith(r"!addons\inner.pbo")
    assert resolver.load_bytes(r"inner\wall.paa", model_source) == payload


def test_nested_texture_lookup_works_from_nested_model_source(tmp_path: Path) -> None:
    inner = tmp_path / "inner.pbo"
    payload = b"same-addon-texture"
    _write_pbo(
        inner,
        {
            "house.p3d": b"synthetic-model",
            "wall.paa": payload,
        },
    )

    outer = tmp_path / "outer.pbo"
    _write_pbo(outer, {r"addons\inner.pbo": inner.read_bytes()})
    wrapped = tmp_path / "mod-package.pbo.zst"
    wrapped.write_bytes(zstandard.ZstdCompressor(level=1).compress(outer.read_bytes()))

    resolver = TextureResolver([wrapped])
    model_source = f"{wrapped}!addons\\inner.pbo!inner\\house.p3d"

    assert resolver.load_bytes(r"inner\wall.paa", model_source) == payload

def test_nested_model_texture_lookup_targets_current_addon_and_reuses_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = tmp_path / "target.pbo"
    _write_pbo(
        target,
        {
            "house.p3d": b"synthetic-model",
            "wall1.paa": b"texture-one",
            "wall2.paa": b"texture-two",
        },
    )

    outer = tmp_path / "outer.pbo"
    _write_pbo(
        outer,
        {
            r"addons\target.pbo": target.read_bytes(),
            r"addons\broken.pbo": b"this is deliberately not a PBO",
        },
    )
    wrapped = tmp_path / "fast-textures.pbo.zst"
    wrapped.write_bytes(zstandard.ZstdCompressor(level=1).compress(outer.read_bytes()))

    real_open = texture_io.open_pbo_stream
    opens = 0

    from contextlib import contextmanager

    @contextmanager
    def counted_open(path):
        nonlocal opens
        opens += 1
        with real_open(path) as handle:
            yield handle

    monkeypatch.setattr(texture_io, "open_pbo_stream", counted_open)

    resolver = TextureResolver([wrapped])
    source = f"{wrapped}!addons\\target.pbo!target\\house.p3d"

    assert resolver.load_bytes(r"target\wall1.paa", source) == b"texture-one"
    assert resolver.load_bytes(r"target\wall2.paa", source) == b"texture-two"

    assert opens == 1
    assert resolver.indexed_all is False
    assert f"{wrapped}!addons\\target.pbo" in resolver.indexed_nested_sources
    assert f"{wrapped}!addons\\broken.pbo" not in resolver.indexed_nested_sources



def _write_prefixed_pbo(path: Path, prefix: str, entries: dict[str, bytes]) -> None:
    """Minimal real PBO header with a BI prefix property."""
    header = bytearray(b"\0" + _PBO_ENTRY.pack(0x56657273, 0, 0, 0, 0))
    header.extend(b"prefix\0" + prefix.encode("latin-1") + b"\0\0")
    data = bytearray()
    for name, payload in entries.items():
        header.extend(name.encode("latin-1") + b"\0")
        header.extend(_PBO_ENTRY.pack(0, len(payload), 0, 0, len(payload)))
        data.extend(payload)
    header.extend(b"\0" + _PBO_ENTRY.pack(0, 0, 0, 0, 0))
    path.write_bytes(bytes(header + data))


def test_arma_structures_and_oa_structures_e_sibling_pbo_lookup(tmp_path: Path) -> None:
    """A ca/structures texture lives in Structures.pbo, not in CA.pbo."""
    model_pbo = tmp_path / "buildings.pbo"
    general_pbo = tmp_path / "ca.pbo"
    structures = tmp_path / "Structures.pbo"
    structures_e = tmp_path / "structures_E.pbo"
    _write_pbo(model_pbo, {"model.p3d": b"placeholder"})
    _write_pbo(general_pbo, {"unrelated.paa": b"not the desired archive"})
    _write_prefixed_pbo(
        structures, r"ca\structures",
        {r"data\bricks\bricks_broken_dirt_03_co.paa": b"arma2-texture"},
    )
    _write_prefixed_pbo(
        structures_e, r"ca\structures_e",
        {r"data\wood\ground_wood01_co.paa": b"oa-texture"},
    )

    name_a2 = r"ca\structures\data\bricks\bricks_broken_dirt_03_co.paa"
    name_oa = r"ca\structures_e\data\wood\ground_wood01_co.paa"
    assert sibling_namespace_pbo(model_pbo, name_a2) == structures
    assert sibling_namespace_pbo(model_pbo, name_oa) == structures_e
    assert sibling_namespace_pbos(model_pbo, name_a2) == (structures, general_pbo)

    resolver = TextureResolver([model_pbo])
    model_source = f"{model_pbo}!ca\\buildings\\house.p3d"
    assert resolver.load_bytes(name_a2, model_source) == b"arma2-texture"
    assert resolver.load_bytes(name_oa, model_source) == b"oa-texture"
    assert structures.resolve() in resolver.auto_sibling_pbos
    assert structures_e.resolve() in resolver.auto_sibling_pbos
    assert general_pbo.resolve() not in resolver.auto_sibling_pbos
    assert resolver.indexed_all is False


def test_dayz_dz_namespace_automatically_loads_sibling_texture_pbo(tmp_path: Path) -> None:
    model_archive = tmp_path / "residential.pbo"
    texture_archive = tmp_path / "structures.pbo"
    generic_dz_archive = tmp_path / "dz.pbo"
    _write_prefixed_pbo(
        model_archive, r"dz\structures\residential",
        {"house.p3d": b"placeholder"},
    )
    _write_prefixed_pbo(
        texture_archive, r"dz\structures",
        {r"data\wall_co.paa": b"dayz-paa-bytes"},
    )
    _write_pbo(generic_dz_archive, {"unrelated.paa": b"not-the-target"})
    texture = r"dz\structures\data\wall_co.paa"
    assert sibling_namespace_pbo(model_archive, texture) == texture_archive

    resolver = TextureResolver([model_archive])
    source = f"{model_archive}!dz\\structures\\residential\\house.p3d"
    assert resolver.load_bytes(texture, source) == b"dayz-paa-bytes"
    assert texture_archive.resolve() in resolver.auto_sibling_pbos
    assert generic_dz_archive.resolve() not in resolver.auto_sibling_pbos


def test_take_on_helicopters_sibling_texture_pbo_and_prefix(tmp_path: Path) -> None:
    model_pbo = tmp_path / "air_h.pbo"
    texture_pbo = tmp_path / "data_h.pbo"
    generic_hsim = tmp_path / "hsim.pbo"
    _write_prefixed_pbo(
        model_pbo, r"HSim\air_h",
        {"heli_medium.p3d": b"model"},
    )
    _write_prefixed_pbo(
        texture_pbo, r"HSim\data_h",
        {r"data\heli_ext_co.paa": b"toh-paa"},
    )
    _write_pbo(generic_hsim, {"unrelated.paa": b"wrong-archive"})
    texture = r"HSim\data_h\data\heli_ext_co.paa"
    assert sibling_namespace_pbo(model_pbo, texture) == texture_pbo
    assert sibling_namespace_pbos(model_pbo, texture) == (texture_pbo, generic_hsim)

    resolver = TextureResolver([model_pbo])
    source = f"{model_pbo}!HSim\\air_h\\heli_medium.p3d"
    assert resolver.load_bytes(texture, source) == b"toh-paa"
    assert texture_pbo.resolve() in resolver.auto_sibling_pbos
    assert generic_hsim.resolve() not in resolver.auto_sibling_pbos


def test_missing_sibling_does_not_return_incorrect_namespace_pbo(tmp_path: Path) -> None:
    source = tmp_path / "models.pbo"
    _write_pbo(source, {})
    ca = tmp_path / "ca.pbo"
    _write_pbo(ca, {"random.paa": b"unrelated"})
    resolver = TextureResolver([source])
    assert resolver.load_bytes(
        r"ca\structures\data\plastic\plastic_grey_old_co.paa",
        f"{source}!models\\house.p3d",
    ) is None


def test_real_modern_paa_decodes_after_sibling_texture_lookup(tmp_path: Path) -> None:
    """End-to-end: model PBO -> sibling CA/Structures PBO -> Arma LZO DXT1 RGBA."""
    source = tmp_path / "buildings2.pbo"
    structures = tmp_path / "Structures.pbo"
    _write_pbo(source, {"building.p3d": b"synthetic-model"})
    lzo_red_16 = bytes.fromhex(
        "0500f800000000000020431c0000020000000000f800000000000000f8000000000000110000"
    )
    paa = (
        struct.pack("<H", 0xFF01)
        + struct.pack("<H", 0)  # empty DXT palette
        + struct.pack("<HH", 0x8010, 16)  # Arma 2 LZO width flag
        + len(lzo_red_16).to_bytes(3, "little")
        + lzo_red_16
    )
    _write_prefixed_pbo(
        structures, r"ca\structures",
        {r"data\bricks\bricks_broken_dirt_03_co.paa": paa},
    )
    resolver = TextureResolver([source])
    rgba = resolver.load_rgba(
        r"ca\structures\data\bricks\bricks_broken_dirt_03_co.paa",
        f"{source}!ca\\buildings2\\house.p3d",
    )
    assert rgba is not None
    assert rgba.shape == (16, 16, 4)
    assert tuple(int(x) for x in rgba[0, 0]) == (255, 0, 0, 255)
    assert structures.resolve() in resolver.auto_sibling_pbos
