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
