from __future__ import annotations

from pathlib import Path
import math
import struct

from cwr_worldgen import road_inspector as inspector
from cwr_worldgen.model import WorldObject
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
    assert result.paved_stock_repairs == ()
    assert result.paved_replacements == ()




def test_stock_repair_search_prefers_clean_vanilla_curve_sequence() -> None:
    start = inspector.RoadEndpoint(
        1,
        r"o\road\sil25.p3d",
        "sil",
        "straight",
        0,
        (0.0, 0.0),
        0.0,
        180.0,
        4.55,
    )
    curve_end, curve_heading = inspector._stock_arc_step(
        start.point, 0.0, 1, 25
    )
    radians = math.radians(curve_heading)
    end_point = (
        curve_end[0] + math.sin(radians) * 25.0,
        curve_end[1] + math.cos(radians) * 25.0,
    )
    end = inspector.RoadEndpoint(
        2,
        r"o\road\sil25.p3d",
        "sil",
        "straight",
        1,
        end_point,
        curve_heading,
        curve_heading,
        4.55,
    )
    expected = (1, 1, 25, 0, 0, 25, 25)
    reference = inspector._stock_repair_samples(start, end, expected)

    result = inspector._stock_repair_choice(
        "sil", reference, start, end
    )

    assert result is not None
    choice, deviation, length_error, angle_error = result
    assert choice == expected
    assert deviation <= inspector._STOCK_REPAIR_MAXIMUM_PATH_DEVIATION_METRES
    assert length_error <= inspector._STOCK_REPAIR_POSITION_TOLERANCE_METRES
    assert angle_error <= inspector._stock_repair_angle_limit(4.55)
    assert inspector._stock_repair_models("sil", choice) == (
        r"o\road\sil10 25.p3d",
        r"o\road\sil25.p3d",
    )


def test_in_memory_inspection_matches_final_world_object_geometry() -> None:
    result = inspector.inspect_road_objects(
        (
            WorldObject(1, r"o\road\sil25.p3d", 0.0, 0.0, 0.0, 0.0, 0.0),
            WorldObject(2, r"o\road\sil25.p3d", 0.0, 0.0, 25.0, 5.0, 0.0),
        ),
        world_name="memory_world",
    )

    assert result.input_path == "<memory>"
    assert result.wrp_entry == "memory_world.wrp"
    assert len(result.issues) == 1
    assert result.issues[0].category in {"connector_gap", "straight_miter"}
    assert result.paved_stock_repairs == ()
    assert len(result.paved_replacements) == 1
    assert result.paved_replacements[0].model_path.startswith(
        r"memory_world\i\paved_w091_"
    )


def test_aligned_axial_overlap_is_reported_but_does_not_request_generated_pavement(
    tmp_path: Path,
) -> None:
    wrp = _write_wrp(tmp_path, "overlap.wrp", (
        (1, r"o\road\sil25.p3d", 0.0, 0.0, 0.0, 0.0, 0.0),
        # Physical 25 m slabs placed 24.5 m apart overlap by 0.5 m.
        (2, r"o\road\sil25.p3d", 0.0, 0.0, 24.5, 0.0, 0.0),
    ))

    result = inspect_road_geometry(wrp)

    assert any(issue.category == "connector_gap" for issue in result.issues)
    assert result.paved_stock_repairs == ()
    assert result.paved_replacements == ()


def test_misaligned_straights_are_reported(tmp_path: Path) -> None:
    wrp = _write_wrp(tmp_path, "bad.wrp", (
        (1, r"o\road\sil25.p3d", 0.0, 0.0, 0.0, 0.0, 0.0),
        (2, r"o\road\sil25.p3d", 0.0, 0.0, 25.0, 5.0, 0.0),
    ))
    result = inspect_road_geometry(wrp)
    assert len(result.issues) == 1
    assert result.issues[0].category in {"connector_gap", "straight_miter"}
    assert result.paved_stock_repairs == ()
    assert len(result.paved_replacements) == 1
    plan = result.paved_replacements[0]
    assert plan.action == "generate"
    assert plan.replace_object_ids == (1, 2)
    assert plan.issue_ids == (result.issues[0].issue_id,)
    assert plan.model_path.startswith(r"bad\i\paved_w091_")
    assert plan.width_metres == 9.1
    assert 49.0 < plan.length_metres < 51.0


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
    result = inspect_road_geometry(wrp)
    issue = next(issue for issue in result.issues if issue.category == "bad_junction")
    assert issue.metrics["missing_connectors"] == 1.0
    assert result.paved_replacements == ()


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



def test_existing_generated_paved_model_is_reused_by_replacement_plan(
    tmp_path: Path,
) -> None:
    base_objects = (
        (1, r"o\road\sil25.p3d", 0.0, 0.0, 0.0, 0.0, 0.0),
        (2, r"o\road\sil25.p3d", 0.0, 0.0, 25.0, 5.0, 0.0),
    )
    first = inspect_road_geometry(
        _write_wrp(tmp_path, "reuse.wrp", base_objects)
    )
    assert len(first.paved_replacements) == 1
    model = first.paved_replacements[0].model_path

    wrp = _write_wrp(
        tmp_path,
        "reuse.wrp",
        base_objects + (
            (99, model, 500.0, 0.0, 500.0, 0.0, 0.0),
        ),
    )
    second = inspect_road_geometry(wrp)

    assert len(second.paved_replacements) == 1
    assert second.paved_replacements[0].model_path == model
    assert second.paved_replacements[0].action == "reuse"



def test_unused_generated_paved_asset_in_pbo_is_reused(
    tmp_path: Path,
) -> None:
    base_objects = (
        (1, r"o\road\sil25.p3d", 0.0, 0.0, 0.0, 0.0, 0.0),
        (2, r"o\road\sil25.p3d", 0.0, 0.0, 25.0, 5.0, 0.0),
    )
    wrp_bytes = _wrp_bytes(base_objects)
    probe = tmp_path / "packed-reuse.wrp"
    probe.write_bytes(wrp_bytes)
    first = inspect_road_geometry(probe)
    assert len(first.paved_replacements) == 1
    model = first.paved_replacements[0].model_path
    filename = model.rsplit("\\", 1)[-1]

    pbo = tmp_path / "packed-reuse.pbo"
    write_pbo(
        pbo,
        (
            PboEntry("packed-reuse.wrp", wrp_bytes),
            PboEntry(rf"i\{filename}", b"already-packed"),
        ),
    )
    second = inspect_road_geometry(pbo)

    assert len(second.paved_replacements) == 1
    assert second.paved_replacements[0].model_path == model
    assert second.paved_replacements[0].action == "reuse"


def test_generated_gravel_is_mapped_but_not_seam_scored(tmp_path: Path) -> None:
    wrp = _write_wrp(tmp_path, "gravel.wrp", (
        (1, r"wg_demo\i\gravel12.p3d", 0.0, 0.0, 0.0, 0.0, 0.0),
    ))
    result = inspect_road_geometry(wrp)
    assert result.road_object_count == 1
    assert result.road_objects[0].road_type == "gravel"
    assert result.issues == ()


def test_generated_paved_straight_is_inspected_as_paved(tmp_path: Path) -> None:
    wrp = _write_wrp(tmp_path, "generated-paved-straight.wrp", (
        (1, r"wg_demo\i\paved_w091_l0062.p3d", 0.0, -0.025, 0.0, 0.0, 0.0),
    ))
    result = inspect_road_geometry(wrp)

    assert result.road_object_count == 1
    road = result.road_objects[0]
    assert road.road_type == "paved"
    assert road.family == "sil"
    assert road.kind == "straight"
    assert math.isclose(road.endpoints[0].point[1], -3.1, abs_tol=1.0e-9)
    assert math.isclose(road.endpoints[1].point[1], 3.1, abs_tol=1.0e-9)
    assert result.issues == ()


def test_generated_paved_curve_keeps_curved_endpoint_tangents(tmp_path: Path) -> None:
    wrp = _write_wrp(tmp_path, "generated-paved-curve.wrp", (
        (1, r"wg_demo\i\paved_w070_l0062_r20.p3d", 0.0, -0.025, 0.0, 0.0, 0.0),
    ))
    result = inspect_road_geometry(wrp)

    assert result.road_object_count == 1
    road = result.road_objects[0]
    assert road.road_type == "paved"
    assert road.family == "asf"
    assert road.kind == "curve"
    assert not math.isclose(
        road.endpoints[0].tangent,
        road.endpoints[1].tangent,
        abs_tol=1.0e-6,
    )
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
        "issues.json", "issues.csv", "summary.json", "ingame-coordinates.csv",
        "paved-stock-repairs.json", "paved-stock-repairs.csv",
        "paved-replacements.json", "paved-replacements.csv", "report.html",
    }
    assert paths["stock_repairs_json"].read_text(encoding="utf-8").startswith("[")
    assert paths["replacements_json"].read_text(encoding="utf-8").startswith("[")


def test_inspector_has_no_generator_or_policy_hooks() -> None:
    import cwr_worldgen.road_inspector as inspector

    source = Path(inspector.__file__).read_text(encoding="utf-8")
    for forbidden in ("generator", "playability", "_policy", "fit_road_objects"):
        assert forbidden not in source
