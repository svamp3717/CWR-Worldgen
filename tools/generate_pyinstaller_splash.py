"""Generate branded PNG splash screens for PyInstaller's onefile bootloader.

Keeping the artwork generator in source control avoids checking binary images
into the repository and allows local builds to reproduce the workflow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SIZE = (640, 360)
TOOLS: dict[str, tuple[str, str, tuple[int, int, int]]] = {
    "Categorise-P3D-Models": (
        "P3D MODEL", "CATEGORISER", (76, 198, 191),
    ),
    "Browse-P3D-Assets": (
        "P3D ASSET", "BROWSER", (114, 169, 246),
    ),
    "WRP-Mod-Dependency-Scanner": (
        "WRP MOD", "DEPENDENCY SCANNER", (250, 186, 101),
    ),
}


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    """Choose installed Windows or cross-platform fonts, with a safe fallback."""
    candidates = (
        ("C:/Windows/Fonts/seguisb.ttf", "DejaVuSans-Bold.ttf")
        if bold else
        ("C:/Windows/Fonts/segoeui.ttf", "DejaVuSans.ttf")
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_splash(name: str, destination: Path) -> None:
    """Create a compact non-transparent PNG for the specified standalone tool."""
    if name not in TOOLS:
        raise ValueError(f"unknown standalone tool: {name}")
    title, subtitle, accent = TOOLS[name]
    image = Image.new("RGB", SIZE, (17, 25, 39))
    draw = ImageDraw.Draw(image)

    # Subtle map-grid theme, with an accent bar identifying each utility.
    grid_color = (27, 41, 57)
    for x in range(32, SIZE[0], 48):
        draw.line((x, 0, x, SIZE[1]), fill=grid_color, width=1)
    for y in range(22, SIZE[1], 48):
        draw.line((0, y, SIZE[0], y), fill=grid_color, width=1)
    draw.rectangle((0, 0, 9, SIZE[1]), fill=accent)
    draw.rounded_rectangle(
        (36, 29, 183, 56), radius=8, fill=(33, 48, 63)
    )
    draw.text((48, 35), "CWR WORLDGEN", fill=(213, 229, 240),
              font=_font(13, bold=True))

    draw.text((40, 98), title, font=_font(35, bold=True), fill=(244, 248, 252))
    draw.text((40, 150), subtitle, font=_font(25, bold=True), fill=accent)
    draw.line((40, 209, 598, 209), fill=(61, 80, 96), width=1)

    draw.text((40, 243), "Starting application", font=_font(19),
              fill=(218, 229, 238))
    draw.text((40, 276), "Unpacking required components. Please wait...",
              font=_font(15), fill=(150, 174, 192))

    draw.rounded_rectangle((40, 320, 600, 326), radius=3, fill=(51, 65, 83))
    # Decorative indicator, not a fake progress percentage.
    draw.rounded_rectangle((40, 320, 155, 326), radius=3, fill=accent)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=sorted(TOOLS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_splash(args.name, args.output)


if __name__ == "__main__":
    main()
