# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from cwr_worldgen import stock_road_long_s_bend_policy as _long
from cwr_worldgen import stock_road_s_bend_exact_policy as _s_exact
from cwr_worldgen import stock_road_s_bend_policy as _s_bend


def test_long_s_bend_policy_extends_exact_search_limit() -> None:
    assert _long._INSTALLED
    assert _s_exact.MAXIMUM_EXACT_S_BEND_RUN_METRES >= 1200.0


def test_exact_s_bend_beam_temporarily_receives_long_search_limit(monkeypatch) -> None:
    original_limit = float(_s_bend._MAXIMUM_S_BEND_SPAN_METRES)
    seen = []
    sentinel = ((0.0, 0.0), (1.0, 1.0))

    def fake_beam(source_points, entry_heading, exit_heading, pieces):
        seen.append(float(_s_bend._MAXIMUM_S_BEND_SPAN_METRES))
        return sentinel

    monkeypatch.setattr(_s_bend, "_beam_s_bend_path", fake_beam)

    result = _s_exact._long_exact_s_bend_path(
        ((0.0, 0.0), (1.0, 1.0)),
        0.0,
        0.0,
        (),
    )

    assert result == sentinel
    assert seen == [_long.MAXIMUM_LONG_EXACT_S_BEND_RUN_METRES]
    assert _s_bend._MAXIMUM_S_BEND_SPAN_METRES == original_limit
