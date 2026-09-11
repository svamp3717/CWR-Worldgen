from __future__ import annotations

from pathlib import Path

import cwr_worldgen.runway_exact_background_policy as exact
from cwr_worldgen.pbo import PboEntry, write_pbo
from cwr_worldgen.runway_performance_policy import _extract_pbo_asset_streaming


def test_runway_texture_extraction_does_not_read_entire_pbo(tmp_path, monkeypatch) -> None:
    pbo = tmp_path / "O.pbo"
    expected = b"single-runway-background"
    write_pbo(
        pbo,
        (
            PboEntry("t1.paa", expected),
            PboEntry("large-unused.bin", b"x" * (2 * 1024 * 1024)),
        ),
    )

    def whole_file_read_forbidden(self):
        raise AssertionError("runway texture lookup must not read the entire PBO")

    monkeypatch.setattr(Path, "read_bytes", whole_file_read_forbidden)

    assert _extract_pbo_asset_streaming(exact, pbo, r"o\t1.paa") == expected
