from __future__ import annotations

from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


policy_path = Path("src/cwr_worldgen/building_asset_budget_policy.py")
policy = policy_path.read_text(encoding="utf-8")
policy = replace_once(
    policy,
    '''        kwargs["texture_variants"] = _texture_variant_budget(
            int(kwargs.get("texture_variants", buildings.DEFAULT_BUILDING_TEXTURE_VARIANTS))
        )
''',
    '''        # Normal world generation leaves texture_variants unspecified and gets
        # the modeler-optimized single cosmetic variant. Explicit library callers
        # retain the public constructor override; the environment variable remains
        # the highest-priority runtime tuning knob.
        if (
            "texture_variants" not in kwargs
            or os.environ.get(_ENV_TEXTURE_VARIANTS, "").strip()
        ):
            kwargs["texture_variants"] = _texture_variant_budget(
                int(kwargs.get("texture_variants", buildings.DEFAULT_BUILDING_TEXTURE_VARIANTS))
            )
''',
    "explicit texture variant override",
)
policy_path.write_text(policy, encoding="utf-8", newline="\n")


test_path = Path("tests/test_milestone8.py")
test = test_path.read_text(encoding="utf-8")
test = replace_once(
    test,
    '            library = ProceduralBuildingLibrary(world_name=world_name)\n',
    '            library = ProceduralBuildingLibrary(world_name=world_name, texture_variants=10)\n',
    "explicit ten-variant filename stress test",
)
test = replace_once(
    test,
    '    def test_building_positions_select_all_ten_texture_variants_deterministically(self) -> None:\n',
    '    def test_building_positions_use_single_modeler_texture_variant_deterministically(self) -> None:\n',
    "single modeler texture variant test name",
)
test = replace_once(
    test,
    '        self.assertEqual(set(variants_first), set(range(10)))\n',
    '        self.assertEqual(set(variants_first), {0})\n',
    "single modeler texture variant expectation",
)
test = replace_once(
    test,
    '''            self.assertEqual(result.generated_variants, 3)
            self.assertEqual(result.reused_placements, 0)
            self.assertEqual(result.reuse_ratio, 0.0)
            self.assertEqual(len(result.model_assets), 3)
''',
    '''            self.assertEqual(result.generated_variants, 1)
            self.assertEqual(result.reused_placements, 2)
            self.assertAlmostEqual(result.reuse_ratio, 2.0 / 3.0)
            self.assertEqual(len(result.model_assets), 1)
''',
    "final P3D variant cap result",
)
test = replace_once(
    test,
    '            self.assertEqual(len({asset.key.texture_variant for asset in result.model_assets}), 3)\n',
    '            self.assertEqual({asset.key.texture_variant for asset in result.model_assets}, {0})\n',
    "final P3D texture variant set",
)
test = replace_once(
    test,
    '''            # Only texture variants referenced by generated P3Ds are emitted.
            # This test produces three selected palettes, so writing all ten
            # configured variants would only bloat the addon and load time.
            self.assertEqual(len(walls), 3)
            self.assertEqual(len(fronts), 3)
            self.assertEqual(len(roofs), 3)
            self.assertEqual(len({(root / relative).read_bytes() for relative in walls}), 3)
            self.assertEqual(len({(root / relative).read_bytes() for relative in fronts}), 3)
            self.assertEqual(len({(root / relative).read_bytes() for relative in roofs}), 3)
''',
    '''            # OSM House Modeler country/material/opening choices now provide
            # the visible variety. The normal generator therefore emits one
            # cosmetic weather variant instead of multiplying every selected P3D.
            self.assertEqual(len(walls), 1)
            self.assertEqual(len(fronts), 1)
            self.assertEqual(len(roofs), 1)
            self.assertEqual(len({(root / relative).read_bytes() for relative in walls}), 1)
            self.assertEqual(len({(root / relative).read_bytes() for relative in fronts}), 1)
            self.assertEqual(len({(root / relative).read_bytes() for relative in roofs}), 1)
''',
    "single emitted modeler texture variant",
)
test_path.write_text(test, encoding="utf-8", newline="\n")
