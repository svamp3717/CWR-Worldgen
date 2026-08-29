# SPDX-License-Identifier: GPL-3.0-or-later
"""Run Road Inspector after a successful GUI world build.

This helper is deliberately a separate process from world generation. Inspector
runtime patches stay isolated, and a diagnostic failure cannot retroactively turn
a successfully generated world into a failed build.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
from typing import Sequence

REPORT_DIRNAME = "road-inspector"
ERROR_FILENAME = "error.txt"


def road_inspector_report_dir(build_dir: Path | str) -> Path:
    return Path(build_dir).expanduser() / REPORT_DIRNAME


def generated_world_pbo(build_dir: Path | str, world_name: str) -> Path | None:
    """Find the generated world PBO by its exact internal world name."""

    root = Path(build_dir).expanduser()
    name = str(world_name).strip()
    if not name:
        return None

    preferred = (
        root / f"{name}.pbo",
        root / "Addons" / f"{name}.pbo",
        root / "CWR-Worldgen" / "Addons" / f"{name}.pbo",
    )
    for candidate in preferred:
        if candidate.is_file():
            return candidate.resolve()

    if not root.is_dir():
        return None
    target = name.casefold()
    try:
        matches = [
            candidate.resolve()
            for candidate in root.rglob("*")
            if candidate.is_file()
            and candidate.suffix.casefold() == ".pbo"
            and candidate.stem.casefold() == target
        ]
    except OSError:
        return None
    if not matches:
        return None

    def score(path: Path) -> tuple[int, int, str]:
        parts = tuple(part.casefold() for part in path.parts)
        return (0 if "addons" in parts else 1, len(parts), str(path).casefold())

    return min(matches, key=score)


def normalized_roads_geojson(
    build_dir: Path | str,
    source_dir: Path | str | None,
) -> Path | None:
    """Return normalized roads context when the build/source bundle contains it."""

    candidates: list[Path] = []
    if source_dir is not None and str(source_dir).strip():
        candidates.append(Path(source_dir).expanduser() / "normalized" / "roads.geojson")
    candidates.append(Path(build_dir).expanduser() / "normalized" / "roads.geojson")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _fresh_report_dir(build_dir: Path | str) -> Path:
    report_dir = road_inspector_report_dir(build_dir)
    if report_dir.exists():
        shutil.rmtree(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    return report_dir


def _record_failure(report_dir: Path, message: str) -> None:
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / ERROR_FILENAME).write_text(message.rstrip() + "\n", encoding="utf-8")
    except OSError:
        pass


def run_postbuild_road_inspector(
    build_dir: Path | str,
    world_name: str,
    *,
    source_dir: Path | str | None = None,
) -> int:
    """Run Road Inspector and always preserve a successful world-build result.

    On inspector failure the warning is printed and ``road-inspector/error.txt``
    is written when possible. Returning zero is intentional: this is an optional
    post-build diagnostic, not part of world generation correctness.
    """

    build_root = Path(build_dir).expanduser().resolve()
    try:
        report_dir = _fresh_report_dir(build_root)
    except OSError as exc:
        print(f"Road Inspector warning: could not prepare report folder: {exc}")
        return 0

    pbo_path = generated_world_pbo(build_root, world_name)
    if pbo_path is None:
        message = (
            "Road Inspector warning: generated world PBO was not found for "
            f"{world_name!r} under {build_root}."
        )
        print(message)
        _record_failure(report_dir, message)
        return 0

    roads_path = normalized_roads_geojson(build_root, source_dir)
    argv = [str(pbo_path), "--output", str(report_dir)]
    if roads_path is not None:
        argv.extend(("--roads", str(roads_path)))

    try:
        # Keep inspector monkeypatches out of the worldgen GUI process until this
        # dedicated child process actually executes the diagnostic.
        from . import road_inspector_entry

        code = int(road_inspector_entry.main(argv))
    except SystemExit as exc:
        code = int(exc.code or 0) if isinstance(exc.code, int | type(None)) else 1
    except Exception as exc:
        message = f"Road Inspector warning: {type(exc).__name__}: {exc}"
        print(message)
        _record_failure(report_dir, message)
        return 0

    if code != 0:
        message = f"Road Inspector warning: inspector exited with code {code}."
        print(message)
        _record_failure(report_dir, message)
        return 0

    error_path = report_dir / ERROR_FILENAME
    try:
        error_path.unlink(missing_ok=True)
    except OSError:
        pass
    print(f"Road Inspector report: {report_dir / 'report.html'}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run nonfatal Road Inspector after a world build.")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--world-name", required=True)
    parser.add_argument("--source-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run_postbuild_road_inspector(
        args.build_dir,
        args.world_name,
        source_dir=args.source_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
