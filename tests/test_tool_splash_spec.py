"""Ensure Windows asset-tool bootloader splashes stay visible but not pinned."""
from __future__ import annotations

from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import sys
import unittest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from configure_tool_splash_spec import configure_splash_visibility


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

    def test_initially_visible_then_unpins_and_keeps_splash_build(self) -> None:
        updated = configure_splash_visibility(self._SPEC)
        self.assertIn("    always_on_top=True,", updated)
        self.assertIn("    'build/splash/Browse-P3D-Assets.png',", updated)
        self.assertIn("exe = EXE(pyz, a.scripts, splash, splash.binaries)", updated)
        self.assertLess(updated.index("_cwr_splash_templates.raise_window"), updated.index("splash = Splash"))
        self.assertIn("after 1500 {wm attributes . -topmost 0; raise .}", updated)
        compile(updated, "test.spec", "exec")
        self.assertEqual(configure_splash_visibility(updated), updated)

    def test_replaces_explicit_non_topmost_setting(self) -> None:
        source = self._SPEC.replace(
            "    binaries=a.binaries,", "    always_on_top=False,\n    binaries=a.binaries,"
        )
        updated = configure_splash_visibility(source)
        self.assertIn("    always_on_top=True,", updated)
        self.assertNotIn("always_on_top=False", updated)
        self.assertEqual(updated.count("always_on_top="), 1)

    def test_generated_spec_changes_actual_splash_tcl_template(self) -> None:
        # No PyInstaller dependency in the unit-test environment. Emulate the
        # stable public Splash spec and its internal bootloader Tcl template.
        pyinstaller = ModuleType("PyInstaller")
        building = ModuleType("PyInstaller.building")
        templates = ModuleType("PyInstaller.building.splash_templates")
        templates.raise_window = "raise .\n"
        building.splash_templates = templates
        pyinstaller.building = building
        captured: dict[str, object] = {}

        def splash(*_args, **kwargs):
            captured["kwargs"] = kwargs
            captured["tcl"] = templates.raise_window

        minimal_spec = "splash = Splash(\n    'splash.png',\n)\n"
        with patch.dict(sys.modules, {
            "PyInstaller": pyinstaller,
            "PyInstaller.building": building,
            "PyInstaller.building.splash_templates": templates,
        }):
            exec(configure_splash_visibility(minimal_spec), {"Splash": splash})
        self.assertEqual(captured["kwargs"], {"always_on_top": True})
        self.assertEqual(
            captured["tcl"],
            "raise .\n\nafter 1500 {wm attributes . -topmost 0; raise .}\n",
        )

    def test_rejects_spec_without_one_splash_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected one"):
            configure_splash_visibility("a = Analysis([])\n")
        with self.assertRaisesRegex(ValueError, "expected one"):
            configure_splash_visibility(self._SPEC + "\n" + self._SPEC)


if __name__ == "__main__":
    unittest.main()
