from __future__ import annotations

from pathlib import Path
import unittest


class WindowsPackagingWorkflowTests(unittest.TestCase):
    def test_frozen_gui_build_is_windowed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "build-windows-exe.yml").read_text(encoding="utf-8")
        self.assertIn("--windowed `", workflow)
        self.assertNotIn("--console `", workflow)
        self.assertNotIn("--hide-console hide-early `", workflow)
        self.assertIn("build_gui.py", workflow)


if __name__ == "__main__":
    unittest.main()
