from __future__ import annotations

from pathlib import Path
import sys

import zstandard

from cwr_worldgen import assets
from cwr_worldgen.fast_asset_scan_policy import _parse_pbo_index
from cwr_worldgen.pbo import PboEntry, read_pbo, write_pbo
from cwr_worldgen.road_inspector import _wrp
import cwr_worldgen.runway_exact_background_policy as exact
from cwr_worldgen.runway_performance_policy import _extract_pbo_asset_streaming
from cwr_worldgen.wrp_mod_dependency_scanner import scan_dependencies

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import measure_p3d_models as measure
from p3d_texture_io import TextureResolver


def _wrapped_fixture(tmp_path: Path, name: str) -> tuple[Path, dict[str, bytes]]:
    payloads = {
        "thing.p3d": b"ODOL synthetic model bytes",
        "thing.paa": b"synthetic texture bytes",
        "world.wrp": b"legacy world string o\\hous\\dum01.p3d end",
    }
    raw = tmp_path / "raw.pbo"
    write_pbo(raw, (PboEntry(entry, data) for entry, data in payloads.items()))
    wrapped = tmp_path / name
    wrapped.write_bytes(zstandard.ZstdCompressor(level=1).compress(raw.read_bytes()))
    return wrapped, payloads


def _assert_wrapped_archive_works(wrapped: Path, payloads: dict[str, bytes]) -> None:
    logical_stem = wrapped.name
    if logical_stem.casefold().endswith(".zst"):
        logical_stem = logical_stem[:-4]
    if logical_stem.casefold().endswith(".pbo"):
        logical_stem = logical_stem[:-4]
    prefix = logical_stem.casefold()

    assert {entry.name: entry.data for entry in read_pbo(wrapped)} == payloads

    records, error = assets._pbo_records(wrapped)
    assert error is None
    by_path = {record.path: record for record in records}
    assert set(by_path) == {
        rf"{prefix}\thing.p3d",
        rf"{prefix}\thing.paa",
    }
    assert assets.read_asset_record_bytes(by_path[rf"{prefix}\thing.paa"]) == payloads["thing.paa"]

    index = _parse_pbo_index(wrapped)
    assert rf"{prefix}\thing.p3d" in index.by_path()
    assert rf"{prefix}\thing.paa" in index.by_path()

    assert exact._extract_pbo_asset(wrapped, rf"{prefix}\thing.paa") == payloads["thing.paa"]
    assert (
        _extract_pbo_asset_streaming(exact, wrapped, rf"{prefix}\thing.paa")
        == payloads["thing.paa"]
    )

    dependency_scan = scan_dependencies(wrapped)
    assert dependency_scan.wrp_count == 1
    assert any(row.model_path == r"o\hous\dum01.p3d" for row in dependency_scan.rows)

    wrp_data, _wrp_name = _wrp(wrapped)
    assert wrp_data == payloads["world.wrp"]

    assert measure.count_models((wrapped,)) == 1
    assert list(measure._pbo_model_paths(wrapped)) == [rf"{prefix}\thing.p3d"]
    assert list(measure._pbo_entries(wrapped)) == [
        (rf"{prefix}\thing.p3d", payloads["thing.p3d"])
    ]

    resolver = TextureResolver([wrapped])
    assert resolver.load_bytes(rf"{prefix}\thing.paa", "") == payloads["thing.paa"]


def test_pbo_zst_extension_is_supported(tmp_path: Path) -> None:
    wrapped, payloads = _wrapped_fixture(tmp_path, "bundle.pbo.zst")
    _assert_wrapped_archive_works(wrapped, payloads)


def test_zstd_magic_is_detected_even_when_file_is_named_pbo(tmp_path: Path) -> None:
    wrapped, payloads = _wrapped_fixture(tmp_path, "misnamed.pbo")
    _assert_wrapped_archive_works(wrapped, payloads)

def test_wrapped_mod_package_indexes_nested_addon_pbos(tmp_path: Path) -> None:
    inner = tmp_path / "inner.pbo"
    inner_payloads = {
        "thing.p3d": b"ODOL nested synthetic model bytes",
        "thing.paa": b"nested synthetic texture bytes",
    }
    write_pbo(inner, (PboEntry(name, data) for name, data in inner_payloads.items()))

    outer = tmp_path / "outer.pbo"
    write_pbo(
        outer,
        (
            PboEntry(r"addons\inner.pbo", inner.read_bytes()),
            PboEntry(r"lib_models\direct.p3d", b"ODOL direct synthetic model bytes"),
            PboEntry(r"lib_models\direct.paa", b"direct synthetic texture bytes"),
        ),
    )
    wrapped = tmp_path / "package.pbo.zst"
    wrapped.write_bytes(zstandard.ZstdCompressor(level=1).compress(outer.read_bytes()))

    records, error = assets._pbo_records(wrapped)
    assert error is None
    by_path = {record.path: record for record in records}
    assert {
        r"inner\thing.p3d",
        r"inner\thing.paa",
        r"lib_models\direct.p3d",
        r"lib_models\direct.paa",
    }.issubset(by_path)

    nested_record = by_path[r"inner\thing.paa"]
    assert nested_record.source.endswith(r"!addons\inner.pbo")
    assert assets.read_asset_record_bytes(nested_record) == inner_payloads["thing.paa"]

    direct_record = by_path[r"lib_models\direct.paa"]
    assert direct_record.source == str(wrapped)
    assert assets.read_asset_record_bytes(direct_record) == b"direct synthetic texture bytes"

    assert measure.count_models((wrapped,)) == 2
    assert set(measure._pbo_model_paths(wrapped)) == {
        r"inner\thing.p3d",
        r"lib_models\direct.p3d",
    }
    discovered = list(measure._iter_models((wrapped,), ()))
    by_model = {model_path: (source, data) for model_path, source, data in discovered}
    assert set(by_model) == {
        r"inner\thing.p3d",
        r"lib_models\direct.p3d",
    }
    nested_source, nested_data = by_model[r"inner\thing.p3d"]
    assert nested_source == f"{wrapped}!addons\\inner.pbo!inner\\thing.p3d"
    assert nested_data == inner_payloads["thing.p3d"]
    direct_source, direct_data = by_model[r"lib_models\direct.p3d"]
    assert direct_source == f"{wrapped}!lib_models\\direct.p3d"
    assert direct_data == b"ODOL direct synthetic model bytes"


def test_wrapped_mod_package_combines_models_from_multiple_nested_pbos(tmp_path: Path) -> None:
    vehicles = tmp_path / "vehicles.pbo"
    buildings = tmp_path / "buildings.pbo"
    write_pbo(
        vehicles,
        (
            PboEntry("tank.p3d", b"ODOL tank"),
            PboEntry("truck.p3d", b"ODOL truck"),
        ),
    )
    write_pbo(
        buildings,
        (
            PboEntry("house.p3d", b"ODOL house"),
            PboEntry("barn.p3d", b"ODOL barn"),
        ),
    )

    outer = tmp_path / "outer.pbo"
    write_pbo(
        outer,
        (
            PboEntry(r"addons\vehicles.pbo", vehicles.read_bytes()),
            PboEntry(r"addons\buildings.pbo", buildings.read_bytes()),
            PboEntry(r"lib_models\crate.p3d", b"ODOL crate"),
        ),
    )
    wrapped = tmp_path / "combined.pbo.zst"
    wrapped.write_bytes(zstandard.ZstdCompressor(level=1).compress(outer.read_bytes()))

    assert measure.count_models((wrapped,)) == 5
    assert set(measure._pbo_model_paths(wrapped)) == {
        r"vehicles\tank.p3d",
        r"vehicles\truck.p3d",
        r"buildings\house.p3d",
        r"buildings\barn.p3d",
        r"lib_models\crate.p3d",
    }

    discovered = list(measure._iter_models((wrapped,), ()))
    assert [model_path for model_path, _source, _data in discovered] == [
        r"vehicles\tank.p3d",
        r"vehicles\truck.p3d",
        r"buildings\house.p3d",
        r"buildings\barn.p3d",
        r"lib_models\crate.p3d",
    ]
    sources = {model_path: source for model_path, source, _data in discovered}
    assert sources[r"vehicles\tank.p3d"] == (
        f"{wrapped}!addons\\vehicles.pbo!vehicles\\tank.p3d"
    )
    assert sources[r"buildings\house.p3d"] == (
        f"{wrapped}!addons\\buildings.pbo!buildings\\house.p3d"
    )
    assert sources[r"lib_models\crate.p3d"] == (
        f"{wrapped}!lib_models\\crate.p3d"
    )

