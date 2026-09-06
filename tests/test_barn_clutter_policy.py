# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import unittest

from cwr_worldgen import generator, osm
from cwr_worldgen.barn_clutter_policy import STOH_MODEL


class BarnClutterPolicyTests(unittest.TestCase):
    def test_stoh_is_barn_only_settlement_clutter(self) -> None:
        self.assertIn(STOH_MODEL, osm.STOCK_SETTLEMENT_BARN_CLUTTER_MODELS)
        self.assertIn(STOH_MODEL, osm.STOCK_SETTLEMENT_DETAIL_MODELS)
        self.assertIn(STOH_MODEL, generator.STOCK_SETTLEMENT_DETAIL_MODELS)
        self.assertNotIn(STOH_MODEL, osm.STOCK_SETTLEMENT_GENERIC_DETAIL_MODELS)


if __name__ == "__main__":
    unittest.main()
