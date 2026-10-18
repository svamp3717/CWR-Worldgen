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

from generate_pyinstaller_splash import (
    SIZE, TOOLS, WORLDGEN_ACCENT, WORLDGEN_NAME, render_splash,
)


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

    def test_worldgen_splash_uses_separate_forest_palette_and_single_wordmark(self) -> None:
        self.assertNotIn(WORLDGEN_NAME, TOOLS)
        self.assertNotIn(WORLDGEN_ACCENT, [tool[2] for tool in TOOLS.values()])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "CWR-Worldgen.png"
            render_splash(WORLDGEN_NAME, path)
            with Image.open(path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, SIZE)
                self.assertEqual(image.getpixel((4, 100)), WORLDGEN_ACCENT)
                self.assertEqual(image.getpixel((15, 11)), (19, 27, 22))
                # The reserved second-headline region must contain only the
                # dark background and technical grid, not the old tagline.
                subtitle_space = image.crop((40, 172, 598, 202))
                self.assertTrue(
                    set(subtitle_space.getdata()).issubset({(19, 27, 22), (35, 47, 38)})
                )
                # Distinctive bright wordmark at the top and a green bar.
                self.assertIn(
                    (246, 248, 243),
                    image.crop((40, 105, 530, 164)).getdata(),
                )
                self.assertEqual(image.getpixel((100, 323)), WORLDGEN_ACCENT)

    def test_wrp_splash_only_uses_white_for_wrp_headline(self) -> None:
        name = "WRP-Mod-Dependency-Scanner"
        self.assertEqual(TOOLS[name][:2], ("WRP", "MOD DEPENDENCY SCANNER"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrp.png"
            render_splash(name, path)
            with Image.open(path) as image:
                white = (244, 248, 252)
                accent = TOOLS[name][2]
                # The word WRP is white; the subtitle and other text are amber.
                self.assertIn(white, image.crop((40, 90, 330, 149)).getdata())
                self.assertIn(accent, image.crop((40, 148, 620, 202)).getdata())
                self.assertNotIn(white, image.crop((0, 149, 640, 360)).getdata())
                self.assertNotIn(white, image.crop((35, 29, 200, 70)).getdata())

    def test_refuses_unknown_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unknown standalone tool"):
                render_splash("not-a-tool", Path(directory) / "unexpected.png")


if __name__ == "__main__":
    unittest.main()
