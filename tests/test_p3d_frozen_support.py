"""Tests for the standalone P3D GUI PyInstaller startup support."""
from __future__ import annotations

import io
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from p3d_frozen_support import configure_windowed_logging, dismiss_splash


class FrozenWindowedLoggingTests(unittest.TestCase):
    def test_python_source_run_does_not_change_streams(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(sys, "frozen", False, create=True), \
             patch.object(sys, "stdout", stdout), \
             patch.object(sys, "stderr", stderr):
            configure_windowed_logging("browse-p3d-assets")
            self.assertIs(sys.stdout, stdout)
            self.assertIs(sys.stderr, stderr)

    def test_frozen_console_run_does_not_change_streams(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(sys, "frozen", True, create=True), \
             patch.object(sys, "stdout", stdout), \
             patch.object(sys, "stderr", stderr):
            configure_windowed_logging("browse-p3d-assets")
            self.assertIs(sys.stdout, stdout)
            self.assertIs(sys.stderr, stderr)

    def test_dismiss_splash_after_gui_ready(self) -> None:
        root = Mock()
        splash = SimpleNamespace(close=Mock())
        with patch.dict(sys.modules, {"pyi_splash": splash}):
            dismiss_splash(root)
        root.update_idletasks.assert_called_once_with()
        splash.close.assert_called_once_with()

    def test_dismiss_splash_without_pyinstaller_module(self) -> None:
        root = Mock()
        with patch.dict(sys.modules, {"pyi_splash": None}):
            dismiss_splash(root)
        root.update_idletasks.assert_called_once_with()

    def test_frozen_windowed_run_redirects_prints_to_persistent_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LOCALAPPDATA": tmp}), \
                 patch.object(sys, "frozen", True, create=True), \
                 patch.object(sys, "stdout", None), \
                 patch.object(sys, "stderr", None):
                configure_windowed_logging("categorise-p3d-models")
                self.assertIsNotNone(sys.stdout)
                self.assertIs(sys.stdout, sys.stderr)
                print("scan progress")
                print("scan warning", file=sys.stderr)
                sys.stdout.close()

            log = Path(tmp) / "CWR-Worldgen" / "logs" / "categorise-p3d-models.log"
            self.assertEqual(log.read_text(encoding="utf-8"), "scan progress\nscan warning\n")


if __name__ == "__main__":
    unittest.main()
