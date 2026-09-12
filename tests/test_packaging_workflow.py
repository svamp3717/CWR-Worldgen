from __future__ import annotations

from pathlib import Path
import unittest


class PackagingWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_frozen_entry_dispatches_workers_before_importing_package(self) -> None:
        source = (self.root / "tools" / "pyinstaller_gui_entry.py").read_text(encoding="utf-8")
        freeze_index = source.index("_multiprocessing.freeze_support()")
        package_index = source.index("from cwr_worldgen.gui_entry import main")
        self.assertLess(freeze_index, package_index)

    def test_all_pyinstaller_workflows_use_safe_entry_script(self) -> None:
        entry = "tools/pyinstaller_gui_entry.py"
        workflows = (
            ".github/workflows/build-windows-exe.yml",
            ".github/workflows/build-windows-loose.yml",
            ".github/workflows/build-linux.yml",
            ".github/workflows/build-macos.yml",
            ".github/workflows/release.yml",
        )
        for relative in workflows:
            with self.subTest(workflow=relative):
                text = (self.root / relative).read_text(encoding="utf-8")
                self.assertIn(entry, text)
                self.assertNotIn("build_gui.py", text)

    def test_pyinstaller_collects_stock_building_catalogue(self) -> None:
        catalogue = self.root / "src" / "cwr_worldgen" / "data" / "stock_building_models.json"
        self.assertTrue(catalogue.is_file())

        pyproject = (self.root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"data/*.json"', pyproject)

        workflows = (
            ".github/workflows/build-windows-exe.yml",
            ".github/workflows/build-windows-loose.yml",
            ".github/workflows/build-linux.yml",
            ".github/workflows/build-macos.yml",
            ".github/workflows/release.yml",
        )
        for relative in workflows:
            with self.subTest(workflow=relative):
                text = (self.root / relative).read_text(encoding="utf-8")
                self.assertIn("--collect-all cwr_worldgen", text)

    def test_frozen_windows_gui_build_is_windowed(self) -> None:
        workflow = (self.root / ".github" / "workflows" / "build-windows-exe.yml").read_text(encoding="utf-8")
        self.assertIn("--windowed `", workflow)
        self.assertNotIn("--console `", workflow)
        self.assertNotIn("--hide-console hide-early `", workflow)


if __name__ == "__main__":
    unittest.main()
