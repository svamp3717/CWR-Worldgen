#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure P3Ds referenced by WrpTool's objects.ini / objects.xml catalogue.

WrpTool's catalogue is used as the curated model list. Geometry is still read
from the real loose P3D files via ``measure_p3d_models.py``.

The INI contains human-maintained comment-delimited sections such as::

    ;
    ; houses
    ;
    data3d\\dum01.p3d=houses
    data3d\\kostelik.p3d=houses

and later::

    ;
    ; old houses
    ;
    Data3D\\AFbarabizna.p3d=houses

Use ``--houses`` to read both of those exact WrpTool sections. ``--section`` may
also be repeated for any other comment-delimited section name.

Example::

    python tools/measure_wrptool_models.py "D:\\WrpTool" \\
        --data-root "D:\\BIS" \\
        --houses \\
        --output stock-houses.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fnmatch
import json
from pathlib import Path
import re
import sys
from typing import Sequence

from measure_p3d_models import ModelMeasurement, scan_models


_CONFIG_NAMES = frozenset({"objects.ini", "objects.xml"})
_HOUSE_SECTIONS = ("houses", "old houses")
_P3D_REFERENCE = re.compile(
    r"(?i)([a-z0-9_.$@+()\-][a-z0-9_.$@+() /\\\-]{0,300}?\.p3d)"
)
_COMMENT_TEXT = re.compile(r"^\s*;\s*([^;].*?)\s*$")


def canonical(value: str) -> str:
    value = value.replace("/", "\\").strip().strip("\"'").lstrip("\\")
    while "\\\\" in value:
        value = value.replace("\\\\", "\\")
    return value.casefold()


def canonical_section(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def matches(path: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return True
    folded = canonical(path)
    return any(fnmatch.fnmatchcase(folded, canonical(pattern)) for pattern in patterns)


def find_catalogues(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        if path.name.casefold() not in _CONFIG_NAMES:
            raise ValueError("WrpTool input file must be objects.ini or objects.xml")
        return (path,)
    if not path.is_dir():
        raise ValueError(f"WrpTool path does not exist: {path}")
    found = tuple(
        sorted(
            (item for item in path.rglob("*") if item.is_file() and item.name.casefold() in _CONFIG_NAMES),
            key=lambda item: item.as_posix().casefold(),
        )
    )
    if not found:
        raise ValueError(f"no objects.ini or objects.xml found below {path}")
    return found


def comment_section_headers(text: str) -> dict[int, str]:
    """Return line-index -> section name for ``; / ; name / ;`` INI headers."""
    lines = text.splitlines()
    headers: dict[int, str] = {}
    for index in range(1, len(lines) - 1):
        if lines[index - 1].strip() != ";" or lines[index + 1].strip() != ";":
            continue
        match = _COMMENT_TEXT.match(lines[index])
        if match is None:
            continue
        name = canonical_section(match.group(1))
        if name:
            headers[index] = name
    return headers


def references_from_ini_sections(
    text: str,
    include: Sequence[str],
    sections: Sequence[str],
) -> tuple[str, ...]:
    """Read only P3Ds belonging to named WrpTool comment sections."""
    wanted = {canonical_section(value) for value in sections if canonical_section(value)}
    if not wanted:
        return ()

    lines = text.splitlines()
    headers = comment_section_headers(text)
    active_section: str | None = None
    refs: set[str] = set()
    for index, line in enumerate(lines):
        section = headers.get(index)
        if section is not None:
            active_section = section
            continue
        if active_section not in wanted:
            continue
        for hit in _P3D_REFERENCE.finditer(line):
            ref = canonical(hit.group(1))
            if matches(ref, include):
                refs.add(ref)
    return tuple(sorted(refs))


def read_references(
    catalogues: Sequence[Path],
    include: Sequence[str],
    sections: Sequence[str] = (),
) -> tuple[str, ...]:
    refs: set[str] = set()
    selected_sections = tuple(
        dict.fromkeys(canonical_section(value) for value in sections if canonical_section(value))
    )
    for catalogue in catalogues:
        text = catalogue.read_bytes().decode("latin-1")
        if selected_sections:
            # The XML defines WrpTool classes/shapes but does not preserve the
            # INI's comment-delimited object groups. Section selection therefore
            # intentionally comes from objects.ini only.
            if catalogue.suffix.casefold() != ".ini":
                continue
            refs.update(references_from_ini_sections(text, include, selected_sections))
            continue
        for hit in _P3D_REFERENCE.finditer(text):
            ref = canonical(hit.group(1))
            if matches(ref, include):
                refs.add(ref)
    return tuple(sorted(refs))


def suffixes(path: Path, root: Path) -> tuple[str, ...]:
    try:
        relative = canonical(path.relative_to(root).as_posix())
    except ValueError:
        relative = canonical(path.name)
    parts = relative.split("\\")
    return tuple("\\".join(parts[index:]) for index in range(len(parts)))


def index_loose_p3ds(roots: Sequence[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in roots:
        root = root.expanduser().resolve()
        if not root.exists():
            raise ValueError(f"data root does not exist: {root}")
        if root.is_file():
            continue
        for model in sorted(root.rglob("*.p3d"), key=lambda item: item.as_posix().casefold()):
            if not model.is_file():
                continue
            for candidate in suffixes(model, root):
                index.setdefault(candidate, model)
    return index


def measure_from_wrptool(
    wrptool: Path,
    data_roots: Sequence[Path],
    include: Sequence[str],
    sections: Sequence[str] = (),
) -> dict[str, object]:
    catalogues = find_catalogues(wrptool)
    references = read_references(catalogues, include, sections)

    roots: list[Path] = [wrptool if wrptool.is_dir() else wrptool.parent]
    roots.extend(data_roots)
    model_index = index_loose_p3ds(roots)

    resolved: dict[Path, str] = {}
    missing: list[str] = []
    for ref in references:
        model = model_index.get(ref)
        if model is None:
            missing.append(ref)
        else:
            resolved.setdefault(model.resolve(), ref)

    measurements, failures = scan_models(tuple(resolved))
    repaired: list[ModelMeasurement] = []
    for measurement in measurements:
        ref = resolved.get(Path(measurement.source).resolve())
        repaired.append(replace(measurement, model_path=ref or measurement.model_path))
    repaired.sort(key=lambda item: item.model_path)

    return {
        "schema": 1,
        "source": "wrptool-object-catalogue",
        "catalogues": [str(item) for item in catalogues],
        "sections": [canonical_section(value) for value in sections],
        "reference_count": len(references),
        "model_count": len(repaired),
        "missing_count": len(missing),
        "failure_count": len(failures),
        "models": [asdict(item) for item in repaired],
        "missing_models": sorted(missing),
        "failures": [asdict(item) for item in failures],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Measure OFP/CWA P3Ds listed in a WrpTool object catalogue."
    )
    result.add_argument("wrptool", type=Path, help="WrpTool directory, objects.ini, or objects.xml")
    result.add_argument(
        "--data-root",
        type=Path,
        action="append",
        default=[],
        metavar="DIR",
        help="Additional unpacked OFP/CWA data tree; may be repeated",
    )
    result.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="GLOB",
        help="Only include canonical P3D paths matching this glob; may be repeated",
    )
    result.add_argument(
        "--section",
        action="append",
        default=[],
        metavar="NAME",
        help="Only include an exact '; / ; NAME / ;' objects.ini section; may be repeated",
    )
    result.add_argument(
        "--houses",
        action="store_true",
        help="Include both WrpTool '; houses ;' and '; old houses ;' comment sections",
    )
    result.add_argument("-o", "--output", type=Path, help="Write JSON report to this file")
    result.add_argument("--strict", action="store_true", help="Fail if any model is missing/unreadable")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    sections = list(args.section)
    if args.houses:
        sections = [*_HOUSE_SECTIONS, *sections]
    sections = list(dict.fromkeys(canonical_section(value) for value in sections))

    try:
        report = measure_from_wrptool(args.wrptool, args.data_root, args.include, sections)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        section_note = f"; sections {', '.join(report['sections'])}" if report["sections"] else ""
        print(
            f"WrpTool refs {report['reference_count']:,}; measured {report['model_count']:,}; "
            f"missing {report['missing_count']:,}; failed {report['failure_count']:,}{section_note}",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(text)

    if args.strict and (report["missing_count"] or report["failure_count"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
