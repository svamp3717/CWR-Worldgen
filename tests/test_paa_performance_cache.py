from __future__ import annotations

from cwr_worldgen.paa import _compress_dxt1_rgb_bytes


def test_dxt1_block_compression_is_memoized_without_changing_bytes() -> None:
    raw = bytes((index * 17) % 256 for index in range(48))
    _compress_dxt1_rgb_bytes.cache_clear()
    first = _compress_dxt1_rgb_bytes(raw)
    second = _compress_dxt1_rgb_bytes(raw)
    assert first == second
    info = _compress_dxt1_rgb_bytes.cache_info()
    assert info.misses == 1
    assert info.hits == 1
