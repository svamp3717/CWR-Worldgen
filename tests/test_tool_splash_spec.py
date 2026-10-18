"""Verify the Windows PyInstaller splash is visible in the taskbar."""
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

    def test_generates_managed_splash_and_preserves_build_inputs(self) -> None:
        updated = configure_splash_visibility(
            self._SPEC, title="Browse P3D Assets - Loading",
        )
        self.assertIn("    always_on_top=True,", updated)
        self.assertIn("    'build/splash/Browse-P3D-Assets.png',", updated)
        self.assertIn("exe = EXE(pyz, a.scripts, splash, splash.binaries)", updated)
        self.assertIn("'wm overrideredirect . 0'", updated)
        self.assertIn("wm title . {Browse P3D Assets - Loading}", updated)
        self.assertIn("wm resizable . 0 0", updated)
        self.assertIn("wm protocol . WM_DELETE_WINDOW {}", updated)
        self.assertIn("after 1500 {wm attributes . -topmost 0}", updated)
        self.assertLess(
            updated.index("_cwr_splash_templates.position_window_on_top"),
            updated.index("splash = Splash"),
        )
        compile(updated, "test.spec", "exec")
        self.assertEqual(
            configure_splash_visibility(updated, title="Browse P3D Assets - Loading"),
            updated,
        )

    def test_non_topmost_option_is_restored_for_initial_visibility(self) -> None:
        source = self._SPEC.replace(
            "    binaries=a.binaries,", "    always_on_top=False,\n    binaries=a.binaries,"
        )
        updated = configure_splash_visibility(source)
        self.assertIn("    always_on_top=True,", updated)
        self.assertNotIn("always_on_top=False", updated)
        self.assertEqual(updated.count("always_on_top="), 1)

    def test_generated_spec_changes_actual_tcl_template(self) -> None:
        pyinstaller = ModuleType("PyInstaller")
        building = ModuleType("PyInstaller.building")
        templates = ModuleType("PyInstaller.building.splash_templates")
        templates.raise_window = "raise .\n"
        templates.position_window_on_top = (
            "wm overrideredirect . 1\n"
            "wm geometry . +0+0\n"
            "wm attributes . -topmost 1\n"
        )
        building.splash_templates = templates
        pyinstaller.building = building
        captured: dict[str, object] = {}

        def splash(*_args, **kwargs):
            captured["kwargs"] = kwargs
            captured["window_tcl"] = templates.position_window_on_top
            captured["raise_tcl"] = templates.raise_window

        minimal_spec = "splash = Splash(\n    'splash.png',\n)\n"
        with patch.dict(sys.modules, {
            "PyInstaller": pyinstaller,
            "PyInstaller.building": building,
            "PyInstaller.building.splash_templates": templates,
        }):
            exec(configure_splash_visibility(minimal_spec, title="P3D Browser Loading"), {
                "Splash": splash,
            })
        self.assertEqual(captured["kwargs"], {"always_on_top": True})
        self.assertIn("wm overrideredirect . 0", captured["window_tcl"])
        self.assertNotIn("wm overrideredirect . 1", captured["window_tcl"])
        self.assertEqual(
            captured["raise_tcl"],
            "raise .\n"
            "\nwm title . {P3D Browser Loading}\n"
            "wm resizable . 0 0\n"
            "wm protocol . WM_DELETE_WINDOW {}\n"
            "after 1500 {wm attributes . -topmost 0}\n",
        )

    def test_fails_clearly_when_pyinstaller_changes_splash_template(self) -> None:
        pyinstaller = ModuleType("PyInstaller")
        building = ModuleType("PyInstaller.building")
        templates = ModuleType("PyInstaller.building.splash_templates")
        templates.raise_window = "raise .\n"
        templates.position_window_on_top = "unsupported new window script"
        building.splash_templates = templates
        pyinstaller.building = building
        minimal_spec = "splash = Splash(\n    'splash.png',\n)\n"
        with patch.dict(sys.modules, {
            "PyInstaller": pyinstaller,
            "PyInstaller.building": building,
            "PyInstaller.building.splash_templates": templates,
        }):
            with self.assertRaisesRegex(RuntimeError, "template has changed"):
                exec(configure_splash_visibility(minimal_spec), {"Splash": lambda *a, **k: None})

    def test_installed_pyinstaller_template_uses_managed_window(self) -> None:
        # CI installs PyInstaller before running this test. With the real
        # package available, also exercise its current template module instead
        # of relying solely on our lightweight fake.
        try:
            from PyInstaller.building import splash_templates
        except ImportError:
            self.skipTest("PyInstaller not installed")
        original_window = splash_templates.position_window_on_top
        original_raise = splash_templates.raise_window
        captured = {}

        def capture_splash(*_args, **kwargs):
            captured["kwargs"] = kwargs
            captured["window"] = splash_templates.position_window_on_top
            captured["raise"] = splash_templates.raise_window

        try:
            spec = configure_splash_visibility(
                "splash = Splash(\\n    'splash.png',\\n)\\n",
                title="Windows Splash Loading",
            )
            exec(spec, {"Splash": capture_splash})
            self.assertEqual(captured["kwargs"], {"always_on_top": True})
            self.assertIn("wm overrideredirect . 0", captured["window"])
            self.assertIn("wm title . {Windows Splash Loading}", captured["raise"])
            self.assertIn("after 1500 {wm attributes . -topmost 0}", captured["raise"])
        finally:
            splash_templates.position_window_on_top = original_window
            splash_templates.raise_window = original_raise

    def test_rejects_bad_titles_and_ambiguous_specs(self) -> None:
        for title in ("", "bad } title", "bad\n title"):
            with self.assertRaisesRegex(ValueError, "title"):
                configure_splash_visibility(self._SPEC, title=title)
        with self.assertRaisesRegex(ValueError, "expected one"):
            configure_splash_visibility("a = Analysis([])\n")
        with self.assertRaisesRegex(ValueError, "expected one"):
            configure_splash_visibility(self._SPEC + "\n" + self._SPEC)


if __name__ == "__main__":
    unittest.main()
