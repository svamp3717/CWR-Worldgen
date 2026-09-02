from __future__ import annotations

from pathlib import Path
import math
import struct

from cwr_worldgen.pbo import PboEntry, write_pbo
from cwr_worldgen.road_inspector import inspect_road_geometry, write_inspection_report

_HEADER = struct.Struct("<4sii")
_OBJECT = struct.Struct("<12fi76s")
_TEXTURE_TABLE_BYTES = 512 * 32


def _matrix(x: float, y: float, z: float, heading: float, pitch: float) -> tuple[float, ...]:
    h = math.radians(heading)
    p = math.radians(pitch)
    ch, sh = math.cos(h), math.sin(h)
    cp, sp = math.cos(p), math.sin(p)
    return (
        ch, 0.0, -sh,
        -sh * sp, cp, -ch * sp,
        sh * cp, sp, ch * cp,
        x, y, z,
    )


def _wrp_bytes(objects: tuple[tuple[int, str, float, float, float, float, float], ...]) -> bytes:
    data = bytearray()
    data.extend(_HEADER.pack(b"4WVR", 1, 1))
    data.extend(struct.pack("<h", 0))
    data.extend(struct.pack("<h", 0))
    data.extend(bytes(_TEXTURE_TABLE_BYTES))
    for object_id, model, x, y, z, heading, pitch in objects:
        data.extend(
            _OBJECT.pack(
                *_matrix(x, y, z, heading, pitch),
                object_id,
                model.encode("ascii").ljust(76, b"\0"),
            )
        )
    data.extend(bytes(_OBJECT.size))
    return bytes(data)


def _write_wrp(tmp_path: Path, name: str, objects) -> Path:
    wrp = tmp_path / name
    wrp.write_bytes(_wrp_bytes(tuple(objects)))
    return wrp


def test_clean_straights_have_no_findings(tmp_path: Path) -> None:
    wrp = _write_wrp(tmp_path, "clean.wrp", (
        (1, r"o\road\sil25.p3d", 0.0, 0.0, 0.0, 0.0, 0.0),
        (2, r"o\road\sil25.p3d", 0.0, 0.0, 25.0, 0.0, 0.0),
    ))
    result = inspect_road_geometry(wrp)
    assert result.road_object_count == 2
    assert result.issues == ()


def test_misaligned_straights_are_reported(tmp_path: Path) -> None:
    wrp = _write_wrp(tmp_path, "bad.wrp", (
        (1, r"o\road\sil25.p3d", 0.0, 0.0, 0.0, 0.0, 0.0),
        (2, r"o\road\sil25.p3d", 0.0, 0.0, 25.0, 5.0, 0.0),
    ))
    result = inspect_road_geometry(wrp)
    assert len(result.issues) == 1
    assert result.issues[0].category in {"connector_gap", "straight_miter"}


def test_pitch_uses_rvw4_horizontal_projection(tmp_path: Path) -> None:
    pitch = 10.0
    spacing = 25.0 * math.cos(math.radians(pitch))
    wrp = _write_wrp(tmp_path, "graded.wrp", (
        (1, r"o\road\sil25.p3d", 0.0, 0.0, 0.0, 0.0, pitch),
        (2, r"o\road\sil25.p3d", 0.0, 0.0, spacing, 0.0, pitch),
    ))
    assert inspect_road_geometry(wrp).issues == ()


def test_complete_t_junction_has_no_junction_findings(tmp_path: Path) -> None:
    wrp = _write_wrp(tmp_path, "junction-ok.wrp", (
        (1, r"o\road\kr_new_sil_sil_t.p3d", 0.0, 0.0, 0.0, 0.0, 0.0),
        (2, r"o\road\sil25.p3d", 0.85, 0.0, 18.75, 0.0, 0.0),
        (3, r"o\road\sil25.p3d", 0.85, 0.0, -18.75, 0.0, 0.0),
        (4, r"o\road\sil25.p3d", -17.90, 0.0, 0.0, 90.0, 0.0),
    ))
    result = inspect_road_geometry(wrp)
    assert not [issue for issue in result.issues if issue.category in {"bad_junction", "junction_connector_mismatch"}]


def test_t_junction_missing_arm_is_reported(tmp_path: Path) -> None:
    wrp = _write_wrp(tmp_path, "junction-missing.wrp", (
        (1, r"o\road\kr_new_sil_sil_t.p3d", 0.0, 0.0, 0.0, 0.0, 0.0),
        (2, r"o\road\sil25.p3d", 0.85, 0.0, 18.75, 0.0, 0.0),
        (3, r"o\road\sil25.p3d", 0.85, 0.0, -18.75, 0.0, 0.0),
    ))
    issue = next(issue for issue in inspect_road_geometry(wrp).issues if issue.category == "bad_junction")
    assert issue.metrics["missing_connectors"] == 1.0


def test_t_junction_extra_arm_is_reported(tmp_path: Path) -> None:
    wrp = _write_wrp(tmp_path, "junction-extra.wrp", (
        (1, r"o\road\kr_new_sil_sil_t.p3d", 0.0, 0.0, 0.0, 0.0, 0.0),
        (2, r"o\road\sil25.p3d", 0.85, 0.0, 18.75, 0.0, 0.0),
        (3, r"o\road\sil25.p3d", 0.85, 0.0, -18.75, 0.0, 0.0),
        (4, r"o\road\sil25.p3d", -17.90, 0.0, 0.0, 90.0, 0.0),
        (5, r"o\road\sil25.p3d", 12.50, 0.0, 0.0, 90.0, 0.0),
    ))
    issue = next(issue for issue in inspect_road_geometry(wrp).issues if issue.category == "bad_junction")
    assert issue.metrics["extra_approaches"] >= 1.0


def test_three_way_intersection_without_junction_is_reported(tmp_path: Path) -> None:
    wrp = _write_wrp(tmp_path, "intersection.wrp", (
        (1, r"o\road\sil25.p3d", 0.0, 0.0, 12.5, 0.0, 0.0),
        (2, r"o\road\sil25.p3d", 0.0, 0.0, -12.5, 0.0, 0.0),
        (3, r"o\road\sil25.p3d", -12.5, 0.0, 0.0, 90.0, 0.0),
    ))
    result = inspect_road_geometry(wrp)
    assert any(issue.category == "intersection_without_junction" for issue in result.issues)


def test_paved_interior_crossing_is_reported(tmp_path: Path) -> None:
    wrp = _write_wrp(tmp_path, "crossing.wrp", (
        (1, r"o\road\sil6.p3d", 0.0, 0.0, 0.0, 0.0, 0.0),
        (2, r"o\road\sil6.p3d", 0.0, 0.0, 0.0, 90.0, 0.0),
    ))
    issue = next(issue for issue in inspect_road_geometry(wrp).issues if issue.category == "paved_crossing_without_junction")
    assert issue.object_ids == (1, 2)


def test_generated_gravel_is_mapped_but_not_seam_scored(tmp_path: Path) -> None:
    wrp = _write_wrp(tmp_path, "gravel.wrp", (
        (1, r"wg_demo\i\gravel12.p3d", 0.0, 0.0, 0.0, 0.0, 0.0),
    ))
    result = inspect_road_geometry(wrp)
    assert result.road_object_count == 1
    assert result.road_objects[0].road_type == "gravel"
    assert result.issues == ()


def test_pbo_input_and_reports_are_read_only(tmp_path: Path) -> None:
    world = _wrp_bytes(((1, r"o\road\sil25.p3d", 0.0, 0.0, 0.0, 0.0, 0.0),))
    pbo = tmp_path / "sample.pbo"
    write_pbo(pbo, (PboEntry("sample.wrp", world),))
    before = pbo.read_bytes()

    result = inspect_road_geometry(pbo)
    paths = write_inspection_report(result, tmp_path / "report")

    assert pbo.read_bytes() == before
    assert result.wrp_entry == "sample.wrp"
    assert {path.name for path in paths.values()} == {
        "issues.json", "issues.csv", "summary.json", "ingame-coordinates.csv", "report.html"
    }


def test_inspector_has_no_generator_or_policy_hooks() -> None:
    import cwr_worldgen.road_inspector as inspector

    source = Path(inspector.__file__).read_text(encoding="utf-8")
    for forbidden in ("generator", "playability", "_policy", "fit_road_objects"):
        assert forbidden not in source
