from __future__ import annotations

from pathlib import Path

import numpy as np

import cwr_worldgen.runway_exact_background_policy as exact
from cwr_worldgen.paa import _compress_dxt1_rgb_bytes
from cwr_worldgen.pbo import PboEntry, write_pbo
from cwr_worldgen.runway_performance_policy import (
    _compress_dxt1_blocks_numpy,
    _extract_pbo_asset_streaming,
)


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


def test_batched_runway_dxt1_matches_scalar_encoder_byte_for_byte() -> None:
    rng = np.random.default_rng(3717)
    blocks = rng.integers(0, 256, size=(257, 16, 3), dtype=np.uint8)
    blocks = np.concatenate(
        (
            blocks,
            np.zeros((1, 16, 3), dtype=np.uint8),
            np.full((1, 16, 3), 255, dtype=np.uint8),
            np.full((1, 16, 3), (58, 66, 45), dtype=np.uint8),
        ),
        axis=0,
    )

    batched = _compress_dxt1_blocks_numpy(blocks)
    expected = b"".join(
        _compress_dxt1_rgb_bytes(block.tobytes()) for block in blocks
    )

    assert batched.tobytes(order="C") == expected
