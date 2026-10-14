#!/usr/bin/env python3
"""Verify donor contact groundtypes in the actual generated world PBO.

The authoritative contact properties are the texture and flags on the P3D
Roadway LOD, NOT the visible road artwork. This audit reads both the embedded
road-donors.json provenance and the serialized Roadway faces after PBO packing.

Usage:
    python tools/audit_generated_road_groundtypes.py build/Addons/wg_lundby5.pbo

An audit match confirms encoded contact metadata, not the CWA engine's runtime
surface choice; terrain clipping can still make a vehicle touch the ground.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from cwr_worldgen.legacy_proxy_models import (
    ProxyCloneError, inspect_roadway_face_styles,
    inspect_roadway_surface_style,
)
from cwr_worldgen.pbo import read_pbo


def _family(name: str) -> str | None:
    leaf = name.replace("\\", "/").rsplit("/", 1)[-1].casefold()
    if not leaf.endswith(".p3d"):
        return None
    if leaf.startswith(("road_paved_", "paved_w", "paved_j", "road_j3_paved_", "road_j4_paved_")):
        return "paved"
    if leaf.startswith(("road_dirt_", "road_j3_dirt_", "road_j4_dirt_")):
        return "dirt"
    if leaf.startswith((
        "road_gravel_", "gravel25", "gravel12", "gravel6", "gravel3",
        "gravel_j", "road_j3_gravel_", "road_j4_gravel_",
    )):
        return "gravel"
    return None


def audit_pbo(path: Path) -> dict[str, object]:
    entries = read_pbo(path)
    donor_entry = next(
        (entry for entry in entries if entry.name.replace("/", "\\").casefold() == "i\\road-donors.json"),
        None,
    )
    if donor_entry is None:
        raise ValueError("the generated PBO has no i/road-donors.json to compare")
    donors = json.loads(donor_entry.data.decode("utf-8"))
    styles = {
        family: dict(donors.get(family, {}).get("generated_surface_style") or {})
        for family in ("paved", "dirt", "gravel")
    }
    counts = {family: 0 for family in styles}
    problems: list[str] = []
    notes: list[str] = []
    verified = 0
    for entry in entries:
        if not entry.name.replace("/", "\\").casefold().startswith("i\\"):
            continue
        family = _family(entry.name)
        if family is None:
            continue
        counts[family] += 1
        style = styles[family]
        texture = style.get("roadway_texture")
        point_flag = style.get("roadway_point_flag")
        face_flag = style.get("roadway_face_flag")
        if texture is None or point_flag is None or face_flag is None:
            problems.append(f"{entry.name}: missing {family} donor contact metadata")
            continue
        try:
            contact = inspect_roadway_surface_style(entry.data)
            roadway_faces = inspect_roadway_face_styles(entry.data)
        except (ProxyCloneError, ValueError, OSError) as exc:
            problems.append(f"{entry.name}: unreadable Roadway LOD: {exc}")
            continue
        wanted = (
            str(texture).replace("/", "\\").casefold(),
            int(str(point_flag), 16),
            int(str(face_flag), 16),
        )
        actual = (
            contact.texture_path.replace("/", "\\").casefold(),
            int(contact.point_flag),
            int(contact.face_flag),
        )
        mismatched_faces = [
            index
            for index, (texture, flags, points) in enumerate(roadway_faces)
            if (
                texture.replace("/", "\\").casefold() != wanted[0]
                or flags != wanted[2]
                or not points
                or any(point_flag != wanted[1] for point_flag in points)
            )
        ]
        if mismatched_faces or actual != wanted:
            problems.append(
                f"{entry.name}: Roadway does not match donor style on "
                f"faces {mismatched_faces[:12]!r}: expected={wanted!r}, "
                f"dominant actual={actual!r}"
            )
        else:
            verified += 1

    for family, count in counts.items():
        if count:
            source_format = styles[family].get("roadway_source_format")
            if source_format not in ("ODOL", "MLOD"):
                notes.append(
                    f"{family}: contact style is {source_format or 'unknown'}; "
                    "it was not directly verified against a readable donor Roadway LOD"
                )
    if not sum(counts.values()):
        problems.append("the PBO contains no recognized generated road P3Ds")
    return {
        "pbo": str(path),
        "generated_road_models_by_family": counts,
        "verified_models": verified,
        "mismatches": problems,
        "provenance_warnings": notes,
        "runtime_groundtype_verified": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pbo", type=Path, help="generated world .pbo")
    args = parser.parse_args(argv)
    try:
        report = audit_pbo(args.pbo)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 1 if report["mismatches"] else 0


if __name__ == "__main__":
    sys.exit(main())
