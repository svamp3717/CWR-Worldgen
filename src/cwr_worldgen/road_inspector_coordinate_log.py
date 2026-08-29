# SPDX-License-Identifier: GPL-3.0-or-later
"""Write upload-friendly in-game coordinates for every inspected road object."""
from __future__ import annotations

import csv
from pathlib import Path

from . import road_inspector as _core


_FILENAME = "ingame-coordinates.csv"
_ORIGINAL_WRITE_INSPECTION_REPORT = None
_INSTALLED = False


def _coord(value: float) -> str:
    text = f"{float(value):.4f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _teleport(x: float, z: float) -> str:
    return f"player setPos [{_coord(x)}, {_coord(z)}, 0]"


def _write_coordinate_log(result, path: Path) -> None:
    issues_by_object: dict[int, list[str]] = {}
    for issue in result.issues:
        for object_id in issue.object_ids:
            issues_by_object.setdefault(int(object_id), []).append(str(issue.issue_id))

    fields = [
        "record_type",
        "issue_id",
        "severity",
        "category",
        "object_id",
        "family",
        "kind",
        "model",
        "world_x",
        "world_z",
        "teleport_command",
        "related_issue_ids",
        "source_road_ids",
        "source_highways",
        "source_surfaces",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        # Findings come first so an uploaded file immediately exposes the places
        # the inspector considers suspicious, including source-road context.
        for issue in result.issues:
            metrics = issue.metrics or {}
            writer.writerow(
                {
                    "record_type": "finding",
                    "issue_id": issue.issue_id,
                    "severity": issue.severity,
                    "category": issue.category,
                    "world_x": _coord(issue.x),
                    "world_z": _coord(issue.z),
                    "teleport_command": _teleport(issue.x, issue.z),
                    "related_issue_ids": issue.issue_id,
                    "source_road_ids": metrics.get("source_road_ids", ""),
                    "source_highways": metrics.get("source_highways", ""),
                    "source_surfaces": metrics.get("source_surfaces", ""),
                }
            )

        # Then include every road object, not merely roads mentioned by findings.
        # This makes one uploaded CSV sufficient to inspect any coordinate later.
        for road in sorted(result.road_objects, key=lambda value: int(value.object_id)):
            x, z = road.logical_center
            related = ";".join(issues_by_object.get(int(road.object_id), ()))
            writer.writerow(
                {
                    "record_type": "road",
                    "object_id": road.object_id,
                    "family": road.family,
                    "kind": road.kind,
                    "model": road.model_path,
                    "world_x": _coord(x),
                    "world_z": _coord(z),
                    "teleport_command": _teleport(x, z),
                    "related_issue_ids": related,
                }
            )


def write_inspection_report(result, output_dir: Path):
    if _ORIGINAL_WRITE_INSPECTION_REPORT is None:
        raise RuntimeError("Road Inspector coordinate logger is not installed")
    paths = _ORIGINAL_WRITE_INSPECTION_REPORT(result, output_dir)
    _write_coordinate_log(result, Path(output_dir) / _FILENAME)
    # Keep the existing return dictionary stable for callers that expect the
    # original four report keys. The coordinate CSV is an additional sidecar.
    return paths


def install() -> None:
    global _ORIGINAL_WRITE_INSPECTION_REPORT, _INSTALLED
    if _INSTALLED:
        return
    _ORIGINAL_WRITE_INSPECTION_REPORT = _core.write_inspection_report
    _core.write_inspection_report = write_inspection_report
    _INSTALLED = True
