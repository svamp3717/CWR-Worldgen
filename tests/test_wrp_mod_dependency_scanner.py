from __future__ import annotations

from pathlib import Path
import struct

from cwr_worldgen.model import WorldObject
from cwr_worldgen.pbo import PboEntry, write_pbo
from cwr_worldgen.wrp import write_rvw4
from cwr_worldgen.wrp_mod_dependency_scanner import (
    DEFAULT_STOCK_ROOTS,
    model_namespace,
    normalize_stock_roots,
    result_document,
    rows_for_display,
    scan_dependencies,
    scan_wrp_bytes,
    write_csv_report,
    write_json_report,
)


def _write_wrp(path: Path, models: tuple[str, ...]) -> bytes:
    objects = tuple(
        WorldObject(index + 1, model, float(index), 0.0, float(index), 0.0)
        for index, model in enumerate(models)
    )
    write_rvw4(
        path,
        1,
        1,
        (0.0,),
        (0,),
        (r"data\g.paa",),
        objects,
        height_scale=0.05,
    )
    return path.read_bytes()


def test_rvw4_scanner_reports_mod_models_and_exact_reference_counts(tmp_path: Path) -> None:
    wrp = tmp_path / "sample.wrp"
    _write_wrp(
        wrp,
        (
            r"data3d\stock_house.p3d",
            r"o\hous\stock_resistance.p3d",
            r"bas_o\house1.p3d",
            r"bas_o\house1.p3d",
            r"art_bd\apartment.p3d",
        ),
    )

    result = scan_dependencies(wrp)
    rows = {row.model_path: row for row in result.rows}

    assert result.wrp_names == ("sample.wrp",)
    assert result.wrp_count == 1
    assert result.unique_models == 4
    assert result.unique_mod_models == 2
    assert rows[r"bas_o\house1.p3d"].references == 2
    assert rows[r"bas_o\house1.p3d"].parser == "RVW4 object records"
    assert rows[r"bas_o\house1.p3d"].is_stock is False
    assert rows[r"art_bd\apartment.p3d"].is_stock is False
    assert rows[r"data3d\stock_house.p3d"].is_stock is True
    assert rows[r"o\hous\stock_resistance.p3d"].is_stock is True

    visible = rows_for_display(result)
    assert {row.model_path for row in visible} == {
        r"art_bd\apartment.p3d",
        r"bas_o\house1.p3d",
    }


def test_rvw4_scanner_accepts_eof_terminated_object_list(tmp_path: Path) -> None:
    wrp = tmp_path / "eof-terminated.wrp"
    data = _write_wrp(
        wrp,
        (
            r"o\\tree\\stock_tree.p3d",
            r"ceeb_signs\\sign.p3d",
            r"ceeb_signs\\sign.p3d",
        ),
    )

    # Some real-world RVW4 files omit the final empty 128-byte object record.
    rows = scan_wrp_bytes(data[:-128], wrp_name=wrp.name)
    by_path = {row.model_path: row for row in rows}

    assert by_path[r"o\\tree\\stock_tree.p3d"].references == 1
    assert by_path[r"ceeb_signs\\sign.p3d"].references == 2
    assert by_path[r"ceeb_signs\\sign.p3d"].parser == "RVW4 object records"


def test_pbo_scanner_finds_every_wrp_member(tmp_path: Path) -> None:
    first_path = tmp_path / "one.wrp"
    second_path = tmp_path / "two.wrp"
    first = _write_wrp(first_path, (r"dma_libya_o\house.p3d",))
    second = _write_wrp(second_path, (r"data3d\tree.p3d", r"caf_kkk_buildings2\tower.p3d"))
    pbo = tmp_path / "worlds.pbo"
    write_pbo(
        pbo,
        (
            PboEntry("worlds\\one.wrp", first),
            PboEntry("worlds\\two.wrp", second),
            PboEntry("readme.txt", b"not a world"),
        ),
    )

    result = scan_dependencies(pbo)

    assert result.input_kind == "pbo"
    assert result.wrp_names == (r"worlds\one.wrp", r"worlds\two.wrp")
    assert result.wrp_count == 2
    assert {row.model_path for row in rows_for_display(result)} == {
        r"caf_kkk_buildings2\tower.p3d",
        r"dma_libya_o\house.p3d",
    }


def _literal_lzss(data: bytes) -> bytes:
    encoded = bytearray()
    for offset in range(0, len(data), 8):
        chunk = data[offset : offset + 8]
        encoded.append((1 << len(chunk)) - 1)
        encoded.extend(chunk)
    encoded.extend(struct.pack("<I", sum(data) & 0xFFFFFFFF))
    return bytes(encoded)


def test_pbo_scanner_reads_cprs_compressed_wrp_members(tmp_path: Path) -> None:
    wrp_path = tmp_path / "compressed-source.wrp"
    world = _write_wrp(wrp_path, (r"bas_o\\compressed_house.p3d",))
    stored = _literal_lzss(world)
    fields = struct.Struct("<IIIII")
    pbo = tmp_path / "compressed.pbo"
    pbo.write_bytes(
        b"world.wrp\0"
        + fields.pack(0x43707273, len(world), 0, 0, len(stored))
        + b"\0"
        + fields.pack(0, 0, 0, 0, 0)
        + stored
    )

    result = scan_dependencies(pbo)

    assert result.wrp_names == ("world.wrp",)
    assert {row.model_path for row in rows_for_display(result)} == {
        r"bas_o\compressed_house.p3d"
    }


def test_pbo_without_wrp_reports_warning_instead_of_crashing(tmp_path: Path) -> None:
    pbo = tmp_path / "models-only.pbo"
    write_pbo(pbo, (PboEntry("thing.p3d", b"ODOL"),))

    result = scan_dependencies(pbo)

    assert result.wrp_count == 0
    assert result.rows == ()
    assert result.warnings == ("PBO contains no .wrp entries",)


def test_legacy_wrp_fallback_extracts_namespaced_p3d_paths() -> None:
    data = (
        b"OPRW\x00"
        b"dma_libya_o\\ags_middleeasternhouse3.p3d\x00"
        b"data3d\\str smrk.p3d\x00"
        b"not_a_model.txt\x00"
        b"bas_o\\foo bar\\house.p3d\x00"
    )

    rows = scan_wrp_bytes(data, wrp_name="legacy.wrp")
    by_path = {row.model_path: row for row in rows}

    assert set(by_path) == {
        r"bas_o\foo bar\house.p3d",
        r"data3d\str smrk.p3d",
        r"dma_libya_o\ags_middleeasternhouse3.p3d",
    }
    assert by_path[r"dma_libya_o\ags_middleeasternhouse3.p3d"].parser == "legacy P3D string scan"
    assert by_path[r"data3d\str smrk.p3d"].is_stock is True
    assert by_path[r"bas_o\foo bar\house.p3d"].is_stock is False


def test_stock_roots_are_user_configurable(tmp_path: Path) -> None:
    wrp = tmp_path / "custom-stock.wrp"
    _write_wrp(wrp, (r"bas_o\house.p3d", r"my_mod\thing.p3d"))

    result = scan_dependencies(wrp, stock_roots=("data3d", "o", "bas_o"))

    assert {row.model_path for row in rows_for_display(result)} == {r"my_mod\thing.p3d"}
    assert normalize_stock_roots(("data3d; O, BAS_O",)) == ("data3d", "o", "bas_o")
    assert DEFAULT_STOCK_ROOTS == ("data3d", "o")
    assert model_namespace(r"my_mod\folder\thing.p3d") == "my_mod"


def test_reports_include_mod_summary_and_full_dependency_rows(tmp_path: Path) -> None:
    wrp = tmp_path / "report.wrp"
    _write_wrp(wrp, (r"data3d\stock.p3d", r"art_bd\store1.p3d"))
    result = scan_dependencies(wrp)

    document = result_document(result)
    assert document["unique_mod_dependencies"] == [r"art_bd\store1.p3d"]
    assert len(document["dependencies"]) == 1

    full_document = result_document(result, include_stock=True)
    assert len(full_document["dependencies"]) == 2

    json_path = write_json_report(result, tmp_path / "report.json")
    csv_path = write_csv_report(result, tmp_path / "report.csv")
    full_csv_path = write_csv_report(
        result,
        tmp_path / "report-all.csv",
        include_stock=True,
    )

    assert '"unique_mod_models": 1' in json_path.read_text(encoding="utf-8")
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "art_bd\\store1.p3d" in csv_text
    assert "data3d\\stock.p3d" not in csv_text
    assert "data3d\\stock.p3d" in full_csv_path.read_text(encoding="utf-8")


def test_cli_and_gui_entrypoints_and_source_launcher_are_registered() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")

    assert 'cwr-wrp-mod-scan = "cwr_worldgen.wrp_mod_dependency_scanner:main"' in pyproject
    assert 'cwr-wrp-mod-scan-gui = "cwr_worldgen.wrp_mod_dependency_scanner:gui_main"' in pyproject
    assert (root / "tools" / "wrp_mod_dependency_scanner.py").is_file()
