from __future__ import annotations

from pathlib import Path
import unittest


class PackagingWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_frozen_entry_dispatches_workers_before_importing_package(self) -> None:
        source = (self.root / "tools" / "pyinstaller_gui_entry.py").read_text(encoding="utf-8")
        freeze_index = source.index("_multiprocessing.freeze_support()")
        package_index = source.index("from cwr_worldgen.debug_entry import main")
        self.assertLess(freeze_index, package_index)

    def test_frozen_entry_uses_guarded_debug_launcher(self) -> None:
        source = (self.root / "tools" / "pyinstaller_gui_entry.py").read_text(encoding="utf-8")
        self.assertIn("from cwr_worldgen.debug_entry import main", source)
        self.assertNotIn("from cwr_worldgen.gui_entry import main", source)

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

    def test_pyinstaller_collects_stock_building_catalogues(self) -> None:
        data_dir = self.root / "src" / "cwr_worldgen" / "data"
        for name in (
            "stock_building_models_non_resistance.json",
            "stock_building_models_resistance.json",
            "haus.pbo buildings only.json",
            "ags inds+port.json",
            "art_bd.json",
            "caf_kkk_buildings2.json",
            "DMA_libya_o.json",
            "catintro.json",
            "fdf.json",
            "sfp4.json",
            "afgano.json",
            "ags_build.json",
        ):
            self.assertTrue((data_dir / name).is_file(), name)

        for removed in (
            "stock_building_models.json",
            "haus.pbo + resistance and vanilla.json",
            "ags inds+port and combined stock.json",
        ):
            self.assertFalse((data_dir / removed).exists(), removed)

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

    def test_frozen_debug_log_defaults_beside_executable(self) -> None:
        source = (self.root / "src" / "cwr_worldgen" / "debug_entry.py").read_text(encoding="utf-8")
        self.assertIn("Path(sys.executable).resolve().parent / CRASH_LOG_FILENAME", source)
        self.assertIn("CWR_WORLDGEN_STARTUP_SMOKE", source)
        self.assertIn("_install_frozen_log_streams()", source)

    def test_frozen_faulthandler_uses_real_log_file(self) -> None:
        source = (self.root / "src" / "cwr_worldgen" / "debug_entry.py").read_text(encoding="utf-8")
        self.assertIn("_FAULT_LOG_STREAM", source)
        self.assertIn(
            "faulthandler.enable(file=_FAULT_LOG_STREAM, all_threads=True)",
            source,
        )
        self.assertIn("_enable_faulthandler()", source)


if __name__ == "__main__":
    unittest.main()
