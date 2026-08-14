from __future__ import annotations

from pathlib import Path
import unittest


class WindowsPackagingWorkflowTests(unittest.TestCase):
    def test_frozen_build_preserves_hidden_cli_stdio_and_smoke_tests_dispatch(self) -> None:
        root = Path(__file__).resolve().parents[1]
        workflow = (root / ".github" / "workflows" / "build-windows-exe.yml").read_text(encoding="utf-8")
        self.assertIn("--console `", workflow)
        self.assertIn("--hide-console hide-early `", workflow)
        self.assertNotIn("--windowed `", workflow)
        self.assertIn("--cwr-cli --version", workflow)


if __name__ == "__main__":
    unittest.main()
