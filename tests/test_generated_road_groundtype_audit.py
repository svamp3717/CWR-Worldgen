"""Read packed generated road contact LODs, not just metadata claims."""
from __future__ import annotations

import json
from pathlib import Path

from cwr_worldgen.pbo import PboEntry, write_pbo
from cwr_worldgen.procedural_infrastructure import (
    InfrastructureModelKey,
    RoadSurfaceStyle,
    write_infrastructure_mlod,
)
from tools.audit_generated_road_groundtypes import audit_pbo


def _fixture_pbo(
    root: Path, *, corrupt_dirt: bool = False,
    corrupt_one_dirt_face: bool = False, no_gravel_donor: bool = False,
) -> Path:
    surface_info = {
        "paved": {
            "roadway_texture": r"roads\\asphalt_contact.paa",
            "roadway_point_flag": "0x0000013f",
            "roadway_face_flag": "0x00024102",
            "roadway_source_format": "ODOL",
        },
        "dirt": {
            "roadway_texture": r"roads\\earth_contact.paa",
            "roadway_point_flag": "0x00005678",
            "roadway_face_flag": "0x00021234",
            "roadway_source_format": "ODOL",
        },
        "gravel": {
            "roadway_texture": r"roads\\gravel_contact.paa",
            "roadway_point_flag": "0x00000345",
            "roadway_face_flag": "0x00024321",
            "roadway_source_format": (
                "borrowed-dirt-donor" if no_gravel_donor else "MLOD"
            ),
        },
    }
    entries = [
        PboEntry(
            r"i\\road-donors.json",
            json.dumps({
                key: {"generated_surface_style": style}
                for key, style in surface_info.items()
            }).encode(),
        ),
    ]
    for family, subtype in (
        ("paved", "road_paved_w091_l0063"),
        ("dirt", "road_dirt_w035_l0063"),
        ("gravel", "road_gravel_w046_l0063"),
    ):
        style = surface_info[family]
        texture = style["roadway_texture"]
        contact_style = RoadSurfaceStyle(
            roadway_texture=(
                "roads\\wrong_contact.paa"
                if corrupt_dirt and family == "dirt"
                else texture
            ),
            roadway_point_flag=int(style["roadway_point_flag"], 16),
            roadway_face_flag=int(style["roadway_face_flag"], 16),
        )
        model = root / f"{subtype}.p3d"
        write_infrastructure_mlod(
            model,
            InfrastructureModelKey("road", subtype, 91, 63),
            r"roads\\visual.paa",
            road_surface_style=contact_style,
        )
        payload = model.read_bytes()
        if family == "dirt" and corrupt_one_dirt_face:
            # Alter only the LAST Roadway face texture. The dominant Roadway
            # texture stays correct, so a majority-only check would miss this.
            needle = texture.encode("ascii")
            last = payload.rfind(needle)
            assert last >= 0
            payload = payload[:last] + b"x" + payload[last + 1:]
        entries.append(PboEntry(f"i/{subtype}.p3d", payload))
    pbo = root / "generated.pbo"
    write_pbo(pbo, entries)
    return pbo


def test_archive_audit_checks_serialized_paved_dirt_gravel_contact(tmp_path):
    report = audit_pbo(_fixture_pbo(tmp_path))
    assert report["mismatches"] == []
    assert report["generated_road_models_by_family"] == {
        "paved": 1, "dirt": 1, "gravel": 1,
    }
    assert report["verified_models"] == 3
    assert report["provenance_warnings"] == []
    assert report["runtime_groundtype_verified"] is False


def test_archive_audit_detects_wrong_dirt_contact_texture(tmp_path):
    report = audit_pbo(_fixture_pbo(tmp_path, corrupt_dirt=True))
    assert report["verified_models"] == 2
    assert len(report["mismatches"]) == 1
    assert "road_dirt_w035_l0063" in report["mismatches"][0]


def test_archive_audit_distinguishes_borrowed_gravel_from_direct_donor(tmp_path):
    report = audit_pbo(_fixture_pbo(tmp_path, no_gravel_donor=True))
    assert report["mismatches"] == []
    assert report["verified_models"] == 3
    assert any("gravel" in message and "borrowed-dirt-donor" in message
               for message in report["provenance_warnings"])



def test_archive_audit_catches_one_bad_face_despite_correct_dominant_style(tmp_path):
    report = audit_pbo(_fixture_pbo(tmp_path, corrupt_one_dirt_face=True))
    assert report["verified_models"] == 2
    assert len(report["mismatches"]) == 1
    assert "road_dirt_w035_l0063" in report["mismatches"][0]
    assert "faces" in report["mismatches"][0]
