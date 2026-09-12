# SPDX-License-Identifier: GPL-3.0-or-later
"""Stage editable runtime JSON for PyInstaller distribution packages."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

JSON_DIRECTORIES = ("data", "house_styles", "country_styles")


def stage_external_json(destination: Path, *, source_root: Path | None = None) -> int:
    """Copy all package runtime JSON into ``destination`` preserving directories."""

    root = source_root or Path(__file__).resolve().parents[1] / "src" / "cwr_worldgen"
    destination = destination.resolve()
    copied = 0

    for relative in JSON_DIRECTORIES:
        source_dir = root / relative
        files = sorted(path for path in source_dir.glob("*.json") if path.is_file())
        if not files:
            raise RuntimeError(f"no runtime JSON found in {source_dir}")

        target_dir = destination / relative
        if target_dir.exists():
            for old in target_dir.glob("*.json"):
                old.unlink()
        target_dir.mkdir(parents=True, exist_ok=True)

        for source in files:
            shutil.copy2(source, target_dir / source.name)
            copied += 1

    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="Destination config directory")
    args = parser.parse_args()
    count = stage_external_json(args.destination)
    print(f"Staged {count} editable JSON files in {args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
