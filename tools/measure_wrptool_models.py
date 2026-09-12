#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure the P3D models referenced by a WrpTool object catalogue.

WrpTool's objects.ini / objects.xml files are useful as a curated list of OFP
objects, but they do not need to contain model dimensions.  This utility reads
all P3D references from those catalogue files, locates the corresponding loose
models in a WrpTool/BIS data tree, then delegates the actual geometry measuring
to ``measure_p3d_models.py``.

Typical usage::

    python tools/measure_wrptool_models.py "D:\\WrpTool" \
        --data-root "D:\\BIS" \
        --include "o\\hous\\*.p3d" \
        --include "data3d\\dum*.p3d" \
        --output stock-houses.json

If WrpTool and its unpacked object data live under the same root, ``--data-root``
can be omitted.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fnmatch
import json
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence

# A script executed from tools/ has that directory on sys.path, so importing the
# sibling scanner remains deliberately simple and keeps this utility standalone.
from measure_p3d_models import ModelFailure, ModelMeasurement, scan_models


_P3D_REFERENCE = re.compile(
    r"(?i)([a-z0-9_.$@+()\-][a-z0-9_.$@+() /\\\-]{0,300}?\.p3d)"
)
_WRPTOOL_CONFIG_NAMES = frozenset({"objects.ini", "objects.xml"})


def _canonical(value: str) -> str:
    value = value.replace("/", "\\").strip().strip('"\'').lstrip("\\")
    while "\\\\" in value:
        value = value.replace("\\\\", "\\")
    return value.casefold()


def _matches(path: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    folded = _canonical(path)
    return any(fnmatch.fnmatchcase(folded, _canonical(pattern)) for pattern in patterns)


def _discover_catalogues(root: Path) -> tuple[Path, ...]:
    if root.is_file():
        if root.name.casefold() not in _WRPTOOL_CONFIG_NAMES:
            raise ValueError(
                f"WrpTool catalogue must be objects.ini or objects.xml, got {root.name}"
            )
        return (root,)

    preferred: list[Path] = []
    fallback: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name.casefold() not in _WRPTOOL_CONFIG_NAMES:
            continue
        # Prefer files close to the supplied WrpTool root.  Old installations
        # occasionally contain backups/examples deeper in subdirectories.
        try:
            depth = len(path.relative_to(root).parts)
        except ValueError:
            depth = 99
        (preferred if depth <= 2 else fallback).append(path)
    result = sorted(preferred, key=lambda p: p.as_posix().casefold())
    if not result:
        result = sorted(fallback, key=lambda p: p.as_posix().casefold())
    return tuple(result)


def _read_catalogue_references(path: Path) -> set[str]:
    raw = path.read_bytes()
    # WrpTool-era configs are normally ASCII/ANSI. latin-1 preserves every byte
    # and is therefore safer than failing because somebody named a category with
    # an accented character in 2004.
    text = raw.decode("latin-1", errors="strict")
    result: set[str] = set()
    for match in _P3D_REFERENCE.finditer(text):
        value = _canonical(match.group(1))
        if value.endswith(".p3d"):
            result.add(value)
    return result


def read_wrptool_references(
    wrptool_path: Path,
    *,
    include: Sequence[str] = (),
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    catalogues = _discover_catalogues(wrptool_path)
    if not catalogues:
        raise ValueError(f"no objects.ini or objects.xml found under {wrptool_path}")

    references: set[str] = set()
    for catalogue in catalogues:
        references.update(_read_catalogue_references(catalogue))
    if include:
        references = {item for item in references if _matches(item, include)}
    return catalogues, tuple(sorted(references))


def _suffix_candidates(path: Path, root: Path) -> tuple[str, ...]:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        relative = path.name
    canonical = _canonical(relative)
    parts = canonical.split("\\")
    # WrpTool/BIS data trees are not perfectly standardized.  Index every useful
    # suffix, so D:\BIS\data\O\Hous\foo.p3d can still satisfy o\hous\foo.p3d.
    return tuple("\\".join(parts[index:]) for index in range(len(parts)))


def _index_loose_models(roots: Sequence[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            raise ValueError(f"data root does not exist: {root}")
        if root.is_file():
            continue
        for path in sorted(root.rglob("*.p3d"), key=lambda p: p.as_posix().casefold()):
            if not path.is_file():
                continue
            for candidate in _suffix_candidates(path, root):
                # Prefer the first path found for a canonical reference.  The
                # sorted walk keeps this deterministic when somebody has two
                # unpacked copies of the same addon lying around, as humans do.
                index.setdefault(candidate, path)
    return index


def measure_wrptool_models(
    wrptool_path: Path,
    *,
    data_roots: Sequence[Path] = (),
    include: Sequence[str] = (),
) -> tuple[
    tuple[Path, ...],
    tuple[str, ...],
    list[ModelMeasurement],
    list[ModelFailure],
    tuple[str, ...],
]:
    catalogues, references = read_wrptool_references(wrptool_path, include=include)

    roots: list[Path] = []
    if wrptool_path.is_dir():
        roots.append(wrptool_path)
    else:
        roots.append(wrptool_path.parent)
    roots.extend(Path(item) for item in data_roots)

    index = _index_loose_models(roots)
    resolved: list[Path] = []
    missing: list[str] = []
    for reference in references:
        path = index.get(reference)
        if path is None:
            missing.append(reference)
        else:
            resolved.append(path)

    # Feed exact resolved files to the sibling scanner.  It gives us one JSON
    # shape for loose/PBO/WRPTool sources instead of maintaining two P3D parsers.
    unique_resolved = tuple(dict.fromkeys(path.resolve() for path in resolved))
    measurements, failures = scan_models(unique_resolved)

    # scan_models sees a loose file by basename. Restore WrpTool's canonical
    # virtual path where possible so this output can become a placement catalogue.
    by_source: dict[str, str] = {}
    reverse_index = {path.resolve(): ref for ref, path in index.items() if ref in references}
    for path in unique_resolved:
        ref = reverse_index.get(path)
        if ref is not None:
            by_source[str(path)] = ref

    repaired_measurements: list[ModelMeasurement] = []
    for item in measurements:
        canonical = by_source.get(item.source)
        if canonical is None:
            repaired_measurements.append(item)
            continue
        repaired_measurements.append(
            ModelMeasurement(
                model_path=canonical,
                source=item.source,
                format=item.format,
                version=item.version,
                lod=item.lod,
                vertex_count=item.vertex_count,
                min_x=item.min_x,
                min_y=item.min_y,
                min_z=item.min_z,
                max_x=item.max_x,
                max_y=item.max_y,
                max_z=item.max_z,
                width_m=item.width_m,
                height_m=item.height_m,
                length_m=item.length_m,
                footprint_area_m2=item.footprint_area_m2,
                aspect_ratio=item.aspect_ratio,
                origin_to_bottom_m=item.origin_to_bottom_m,
            )
        )
    repaired_measurements.sort(key=lambda item: item.model_path)

    return (
        catalogues,
        references,
        repaired_measurements,
        failures,
        tuple(sorted(missing)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure loose OFP/CWA P3Ds referenced by WrpTool objects.ini/objects.xml."
    )
    parser.add_argument(
        "wrptool",
        type=Path,
        help="WrpTool directory, objects.ini, or objects.xml",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        default=[],
        type=Path,
        metavar="DIR",
        help=(
            "Additional unpacked OFP/CWA model tree. May be repeated. "
            "Use this when WrpTool's objects live outside its installation directory."
        ),
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help=(
            "Only measure WrpTool references matching this canonical P3D glob. "
            "May be repeated; example: --include 'o\\hous\\*.p3d'."
        ),
    )
    parser.add_argument("-o", "--output", type=Path, help="Write JSON report to this file")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if a referenced model is missing or cannot be measured",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        catalogues, references, measurements, failures, missing = measure_wrptool_models(
            args.wrptool,
            data_roots=args.data_root,
            include=args.include,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = {
        "schema": 1,
        "source": "wrptool-object-catalogue",
        "catalogues": [str(path) for path in catalogues],
        "reference_count": len(references),
        "model_count": len(measurements),
        "failure_count": len(failures),
        "missing_count": len(missing),
        "models": [asdict(item) for item in measurements],
        "failures": [asdict(item) for item in failures],
        "missing_models": list(missing),
    }
    text = json.dumps(report, indent=2, sort_keys=False) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(
            f"WrpTool references: {len(references):,}; measured: {len(measurements):,}; "
            f"missing: {len(missing):,}; failed: {len(failures):,}; wrote {args.output}",
            file=sys.stderr,
        )

    if args.strict and (failures or missing):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
