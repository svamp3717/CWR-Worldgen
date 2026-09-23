from __future__ import annotations

from pathlib import Path
import struct
import sys

import cwr_worldgen.runway_exact_background_policy as exact
from cwr_worldgen import assets
from cwr_worldgen.fast_asset_scan_policy import _parse_pbo_index
from cwr_worldgen.pbo import read_pbo
from cwr_worldgen.runway_exact_background_policy import _extract_pbo_asset
from cwr_worldgen.runway_performance_policy import _extract_pbo_asset_streaming
from cwr_worldgen.wrp_mod_dependency_scanner import scan_dependencies

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import measure_p3d_models as measure
from p3d_texture_io import TextureResolver


_ENTRY = struct.Struct("<IIIII")
_UINT32_MAX = 0xFFFFFFFF


def _write_legacy_sentinel_pbo(path: Path) -> dict[str, bytes]:
    entries = {
        "thing.p3d": b"ODOL synthetic model bytes",
        "thing.paa": b"synthetic texture bytes",
    }
    header = bytearray()
    payload = bytearray()
    for name, data in entries.items():
        header += name.encode("ascii") + b"\0"
        # Match real legacy/addon archives: original size populated even for
        # uncompressed entries and an ordinary non-zero member timestamp.
        header += _ENTRY.pack(0, len(data), 0, 1113147368, len(data))
        payload += data
    # CATIntro.pbo and similar archives use UINT32_MAX in the timestamp field
    # of the otherwise-empty end-of-header record.
    header += b"\0" + _ENTRY.pack(0, 0, 0, _UINT32_MAX, 0)
    path.write_bytes(bytes(header + payload))
    return entries


def test_uint32_max_pbo_header_terminator_is_accepted_by_every_parser(tmp_path: Path) -> None:
    pbo = tmp_path / "legacy.pbo"
    payloads = _write_legacy_sentinel_pbo(pbo)

    # General archive reader.
    assert {entry.name: entry.data for entry in read_pbo(pbo)} == payloads

    # Full and targeted asset scanners, including the independent record reader.
    records, error = assets._pbo_records(pbo)
    assert error is None
    by_path = {record.path: record for record in records}
    assert set(by_path) == {r"legacy\thing.p3d", r"legacy\thing.paa"}
    assert assets.read_asset_record_bytes(by_path[r"legacy\thing.paa"]) == payloads["thing.paa"]

    index = _parse_pbo_index(pbo)
    assert {entry.canonical_path for entry in index.entries} == {
        r"legacy\thing.p3d",
        r"legacy\thing.paa",
    }

    # Runway texture readers have separate in-memory and streaming PBO parsers.
    assert _extract_pbo_asset(pbo, r"legacy\thing.paa") == payloads["thing.paa"]
    assert (
        _extract_pbo_asset_streaming(exact, pbo, r"legacy\thing.paa")
        == payloads["thing.paa"]
    )

    # WRP dependency scanner should parse the archive and merely report that
    # this particular fixture contains no WRP member.
    scan = scan_dependencies(pbo)
    assert scan.wrp_count == 0
    assert scan.warnings == ("PBO contains no .wrp entries",)

    # Standalone P3D measurement and texture tools maintain their own parsers.
    assert measure.count_models((pbo,)) == 1
    assert list(measure._pbo_model_paths(pbo)) == [r"legacy\thing.p3d"]
    assert list(measure._pbo_entries(pbo)) == [
        (r"legacy\thing.p3d", payloads["thing.p3d"])
    ]

    resolver = TextureResolver([pbo])
    assert resolver.load_bytes(r"legacy\thing.paa", "") == payloads["thing.paa"]
