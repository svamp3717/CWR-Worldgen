"""Checks for generated bootloader loading screens."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from generate_pyinstaller_splash import SIZE, TOOLS, render_splash


class SplashImageTests(unittest.TestCase):
    def test_renders_png_for_every_standalone_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for name, (_title, _subtitle, accent) in TOOLS.items():
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.png"
                    render_splash(name, path)
                    self.assertTrue(path.is_file())
                    with Image.open(path) as image:
                        self.assertEqual(image.format, "PNG")
                        self.assertEqual(image.mode, "RGB")
                        self.assertEqual(image.size, SIZE)
                        self.assertEqual(image.getpixel((4, 100)), accent)
                        self.assertNotEqual(
                            image.getpixel((70, 323)),
                            image.getpixel((40, 350)),
                        )

    def test_refuses_unknown_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unknown standalone tool"):
                render_splash("not-a-tool", Path(directory) / "unexpected.png")


if __name__ == "__main__":
    unittest.main()
