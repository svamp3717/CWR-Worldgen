"""Ensure Windows asset-tool loading screens can be covered by other windows."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from configure_tool_splash_spec import disable_splash_always_on_top


class SplashSpecTests(unittest.TestCase):
    _SPEC = (
        "a = Analysis(['tools/browse_p3d_assets.py'])\n"
        "splash = Splash(\n"
        "    'build/splash/Browse-P3D-Assets.png',\n"
        "    binaries=a.binaries,\n"
        "    datas=a.datas,\n"
        ")\n"
        "exe = EXE(pyz, a.scripts, splash, splash.binaries)\n"
    )

    def test_keeps_splash_and_disables_foreground_pin(self) -> None:
        updated = disable_splash_always_on_top(self._SPEC)
        self.assertIn("    always_on_top=False,", updated)
        self.assertIn("    'build/splash/Browse-P3D-Assets.png',", updated)
        self.assertIn("exe = EXE(pyz, a.scripts, splash, splash.binaries)", updated)
        compile(updated, "test.spec", "exec")
        self.assertEqual(disable_splash_always_on_top(updated), updated)

    def test_replaces_explicit_topmost_true(self) -> None:
        source = self._SPEC.replace(
            "    binaries=a.binaries,", "    always_on_top=True,\n    binaries=a.binaries,"
        )
        updated = disable_splash_always_on_top(source)
        self.assertIn("    always_on_top=False,", updated)
        self.assertNotIn("always_on_top=True", updated)
        self.assertEqual(updated.count("always_on_top="), 1)

    def test_refuses_unknown_or_duplicate_splash_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected one"):
            disable_splash_always_on_top("a = Analysis([])\n")
        with self.assertRaisesRegex(ValueError, "expected one"):
            disable_splash_always_on_top(self._SPEC + "\n" + self._SPEC)


if __name__ == "__main__":
    unittest.main()
