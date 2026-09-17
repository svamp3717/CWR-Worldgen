from __future__ import annotations

from pathlib import Path
import struct
import sys

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from p3d_texture_io import PBOAssetRef
from p3d_texture_sources import TextureResolver, sibling_namespace_pbo


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
