from __future__ import annotations

from pathlib import Path
import struct
from unittest.mock import Mock

from cwr_worldgen.model import WorldObject
from cwr_worldgen.pbo import PboEntry, write_pbo
from cwr_worldgen.wrp import write_rvw4
from cwr_worldgen.wrp_mod_dependency_scanner import (
    DEFAULT_STOCK_ROOTS,
    DependencyRow,
    ScanResult,
    ScannerGui,
    _sort_display_rows,
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
            r"o\tree\stock_tree.p3d",
            r"ceeb_signs\sign.p3d",
            r"ceeb_signs\sign.p3d",
        ),
    )

    # Some real-world RVW4 files omit the final empty 128-byte object record.
    rows = scan_wrp_bytes(data[:-128], wrp_name=wrp.name)
    by_path = {row.model_path: row for row in rows}

    assert by_path[r"o\tree\stock_tree.p3d"].references == 1
    assert by_path[r"ceeb_signs\sign.p3d"].references == 2
    assert by_path[r"ceeb_signs\sign.p3d"].parser == "RVW4 object records"


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
    assert by_path[r"dma_libya_o\ags_middleeasternhouse3.p3d"].parser == "OPRW P3D string occurrences"
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


def _write_8wvr(path: Path, models: tuple[str, ...]) -> bytes:
    """Minimal unbinarized Arma WRP with one material and a dummy object."""
    material = b"ca\\data\\grass.rvmat"
    data = bytearray(b"8WVR" + struct.pack("<IIIIf", 1, 1, 1, 1, 10.0))
    data.extend(struct.pack("<fH", 2.5, 1))  # elevation, material index
    data.extend(struct.pack("<II", 2, len(material)))
    data.extend(material)
    data.extend(struct.pack("<I", 0))  # end of material's concat strings
    for idx, model in enumerate(models):
        encoded = model.encode("ascii")
        data.extend(struct.pack("<12fII", *([0.0] * 12), idx + 1, len(encoded)))
        data.extend(encoded)
    data.extend(struct.pack("<12fII", *([0.0] * 12), 0, 0))
    path.write_bytes(data)
    return bytes(data)


def test_8wvr_scans_arma_object_records_with_exact_counts(tmp_path: Path) -> None:
    data = _write_8wvr(
        tmp_path / "arma.wrp",
        (
            r"ca\buildings\house.p3d",
            r"addon\house.p3d",
            r"addon\house.p3d",
        ),
    )
    rows = scan_wrp_bytes(data, wrp_name="arma.wrp", stock_roots=("ca",))
    by_path = {row.model_path: row for row in rows}
    assert by_path[r"ca\buildings\house.p3d"].is_stock
    assert not by_path[r"addon\house.p3d"].is_stock
    assert by_path[r"addon\house.p3d"].references == 2
    assert by_path[r"addon\house.p3d"].parser == "8WVR object records"


def test_dayz_0wzd_scans_exact_object_records_and_stock_dz_roots(tmp_path: Path) -> None:
    data = _write_8wvr(
        tmp_path / "dayz.wrp",
        (
            r"dz\structures\residential\house.p3d",
            r"my_dayz_mod\hut.p3d",
            r"my_dayz_mod\hut.p3d",
        ),
    )
    dayz_data = b"0WZD" + data[4:]
    wrp = tmp_path / "dayz.wrp"
    wrp.write_bytes(dayz_data)

    result = scan_dependencies(wrp, game="dayz")
    rows = {row.model_path: row for row in result.rows}
    assert result.stock_roots == ("dz",)
    assert rows[r"dz\structures\residential\house.p3d"].is_stock is True
    assert rows[r"my_dayz_mod\hut.p3d"].is_stock is False
    assert rows[r"my_dayz_mod\hut.p3d"].references == 2
    assert rows[r"my_dayz_mod\hut.p3d"].parser == "0WZD object records"
    assert {row.model_path for row in rows_for_display(result)} == {
        r"my_dayz_mod\hut.p3d"
    }


def test_dayz_0wzd_in_pbo_and_malformed_0wzd(tmp_path: Path) -> None:
    import pytest

    raw = _write_8wvr(tmp_path / "source.wrp", (r"dz\plants\tree.p3d",))
    data = b"0WZD" + raw[4:]
    pbo = tmp_path / "dayz-world.pbo"
    write_pbo(pbo, (PboEntry(r"world\dayz.wrp", data),))
    result = scan_dependencies(pbo, game="dayz")
    assert result.wrp_names == (r"world\dayz.wrp",)
    assert result.unique_models == 1
    assert result.unique_mod_models == 0
    assert result.rows[0].parser == "0WZD object records"
    with pytest.raises(ValueError, match="0WZD"):
        scan_wrp_bytes(data[:-4], wrp_name="broken.wrp")


def test_dayz_binarized_wrp_reports_string_scan_limit(tmp_path: Path) -> None:
    wrp = tmp_path / "dayz-binarized.wrp"
    wrp.write_bytes(b"OPRW" + b"\\dz\\structures\\house.p3d\x00"
                    + b"addon\\custom.p3d\x00")
    result = scan_dependencies(wrp, game="dayz")
    assert any("not object placements" in warning for warning in result.warnings)
    assert all("string occurrences" in row.parser for row in result.rows)


def test_take_on_helicopters_unbinarized_wrp_profile(tmp_path: Path) -> None:
    wrp = tmp_path / "seattle.wrp"
    _write_8wvr(
        wrp,
        (
            r"HSim\structures_h\building.p3d",
            r"HSim\structures_h\building.p3d",
            r"othermod\tower.p3d",
            r"ca\structures\house.p3d",
        ),
    )
    result = scan_dependencies(wrp, game="tkoh")
    assert result.stock_roots == ("hsim",)
    assert result.unique_mod_models == 2
    by_path = {row.model_path: row for row in result.rows}
    assert by_path[r"hsim\structures_h\building.p3d"].is_stock
    assert by_path[r"hsim\structures_h\building.p3d"].references == 2
    assert by_path[r"hsim\structures_h\building.p3d"].parser == "8WVR object records"
    assert {row.model_path for row in rows_for_display(result)} == {
        r"ca\structures\house.p3d",
        r"othermod\tower.p3d",
    }
    # Rearmed can use external Arma 2 data, without reclassifying it as
    # part of the stock Take On Helicopters installation.
    with_rearmed = scan_dependencies(wrp, game="tkoh", stock_roots=("hsim", "ca"))
    assert with_rearmed.unique_mod_models == 1


def test_take_on_helicopters_binarized_wrp_warning_and_pbo(tmp_path: Path) -> None:
    data = b"OPRW" + struct.pack("<I", 25) + (
        b"HSim\\air_h\\heli.p3d\0"
        b"custom_air\\hanger.p3d\0"
    )
    wrp = tmp_path / "seattle-binarized.wrp"
    wrp.write_bytes(data)
    loose = scan_dependencies(wrp, game="tkoh")
    assert loose.unique_mod_models == 1
    assert any("not object placements" in msg for msg in loose.warnings)
    assert all("string occurrences" in row.parser for row in loose.rows)

    pbo = tmp_path / "world.pbo"
    write_pbo(pbo, (PboEntry("seattle.wrp", data),))
    archived = scan_dependencies(pbo, game="tkoh")
    assert archived.wrp_count == 1
    assert archived.unique_mod_models == 1
    assert any("not object placements" in msg for msg in archived.warnings)


def test_8wvr_truncated_record_is_rejected(tmp_path: Path) -> None:
    import pytest

    data = _write_8wvr(tmp_path / "truncated.wrp", (r"addon\house.p3d",))
    with pytest.raises(ValueError, match="8WVR"):
        scan_wrp_bytes(data[:-4], wrp_name="truncated.wrp")


def test_game_profiles_filter_stock_models_but_keep_addons(tmp_path: Path) -> None:
    data = (
        b"OPRW" + struct.pack("<I", 25)
        + b"\0ca\\buildings\\arma2.p3d\0"
        + b"ca_e\\buildings_e\\oa.p3d\0"
        + b"a3\\structures_f\\house.p3d\0"
        + b"custom_mod\\building.p3d\0"
    )
    wrp = tmp_path / "binarized.wrp"
    wrp.write_bytes(data)
    by_game = {
        game: {r.model_path for r in rows_for_display(scan_dependencies(wrp, game=game))}
        for game in ("cwa", "arma2", "arma2oa", "arma3")
    }
    assert r"ca\buildings\arma2.p3d" not in by_game["arma2"]
    assert r"ca_e\buildings_e\oa.p3d" in by_game["arma2"]
    assert r"ca_e\buildings_e\oa.p3d" not in by_game["arma2oa"]
    assert r"a3\structures_f\house.p3d" not in by_game["arma3"]
    assert r"custom_mod\building.p3d" in by_game["arma3"]
    assert r"a3\structures_f\house.p3d" in by_game["cwa"]
    assert scan_dependencies(wrp, game="arma3").rows[0].parser == "OPRW P3D string occurrences"
    assert scan_dependencies(wrp, game="arma3", stock_roots=("custom_mod",)).stock_roots == ("custom_mod",)


def test_game_profile_does_not_modify_input_files(tmp_path: Path) -> None:
    wrp = tmp_path / "input.wrp"
    before = _write_8wvr(wrp, (r"a3\structures_f\house.p3d",))
    assert scan_dependencies(wrp, game="arma3").unique_mod_models == 0
    assert wrp.read_bytes() == before


def _sortable_rows() -> tuple[DependencyRow, ...]:
    return (
        DependencyRow("Zoo.wrp", r"z\\Alpha.p3d", "Z", 2, False, "RVW4"),
        DependencyRow("alpha.wrp", r"a\\Zulu.p3d", "A", 10, True, "OPRW"),
        DependencyRow("Beta.wrp", r"b\\beta.p3d", "b", 1, False, "legacy"),
    )


def test_table_sort_handles_all_columns_and_numeric_reference_counts() -> None:
    rows = _sortable_rows()
    expected = {
        "wrp": (1, 2, 0),
        "namespace": (1, 2, 0),
        "model": (1, 2, 0),
        "refs": (2, 0, 1),
        "type": (0, 2, 1),
        "parser": (2, 1, 0),
    }
    for column, order in expected.items():
        ascending = _sort_display_rows(rows, column)
        assert ascending == tuple(rows[index] for index in order)
        assert _sort_display_rows(rows, column, descending=True) == tuple(reversed(ascending))
    assert rows[1].references == 10  # Numeric, not lexicographic ordering.


def test_dayz_gui_scan_worker_passes_selected_game_profile(tmp_path: Path) -> None:
    import queue

    wrp = tmp_path / "dayz-oprw.wrp"
    wrp.write_bytes(b"OPRW\\0dz\\\\plants\\\\tree.p3d\\0")
    gui = ScannerGui.__new__(ScannerGui)
    gui._queue = queue.Queue()
    gui._scan_worker(wrp, ("dz",), "dayz")
    kind, result = gui._queue.get_nowait()
    assert kind == "ok"
    assert result.stock_roots == ("dz",)
    assert result.unique_mod_models == 0
    assert any("not object placements" in warning for warning in result.warnings)


def test_gui_heading_click_toggles_direction_and_changes_column() -> None:
    gui = ScannerGui.__new__(ScannerGui)
    gui.tree = Mock()
    gui._populate = Mock()
    gui._column_headings = {"wrp": "WRP", "refs": "Refs"}
    gui._sort_column = None
    gui._sort_descending = False

    gui._sort_by("refs")
    assert (gui._sort_column, gui._sort_descending) == ("refs", False)
    gui.tree.heading.assert_any_call("refs", text="Refs ▲")

    gui._sort_by("refs")
    assert (gui._sort_column, gui._sort_descending) == ("refs", True)
    gui.tree.heading.assert_any_call("refs", text="Refs ▼")

    gui._sort_by("wrp")
    assert (gui._sort_column, gui._sort_descending) == ("wrp", False)
    gui.tree.heading.assert_any_call("refs", text="Refs")
    assert gui._populate.call_count == 3


def test_gui_reapplies_sort_after_stock_visibility_changes() -> None:
    gui = ScannerGui.__new__(ScannerGui)
    gui.tree = Mock()
    gui.tree.get_children.return_value = ()
    gui.status_var = Mock()
    gui.include_stock_var = Mock()
    gui._result = ScanResult("example.wrp", "wrp", ("a",), ("example.wrp",), _sortable_rows())
    gui._sort_column = "refs"
    gui._sort_descending = True

    gui.include_stock_var.get.return_value = False
    gui._populate()
    assert [call.kwargs["values"][3] for call in gui.tree.insert.call_args_list] == [2, 1]

    gui.tree.insert.reset_mock()
    gui.include_stock_var.get.return_value = True
    gui._populate()
    assert [call.kwargs["values"][3] for call in gui.tree.insert.call_args_list] == [10, 2, 1]


def test_arma1_queens_gambit_and_ironfront_stock_profiles(tmp_path: Path) -> None:
    data = (
        b"OPRW" + struct.pack("<I", 25)
        + b"\0ca\\buildings\\wall.p3d\0"
        + b"dbe1\\buildings\\palace.p3d\0"
        + b"models_dbe1\\objects\\tree.p3d\0"
        + b"sara_dbe1\\plants\\bush.p3d\0"
        + b"lib\\buildings\\farm.p3d\0"
        + b"LIB_HOUSES\\structures\\barn.p3d\0"
        + b"lib_customtool\\objects\\car.p3d\0"
        + b"customaddon\\objects\\addon.p3d\0"
    )
    wrp = tmp_path / "mixed.wrp"
    wrp.write_bytes(data)
    def visible(game: str) -> set[str]:
        return {row.model_path for row in rows_for_display(scan_dependencies(wrp, game=game))}

    arma1 = visible("arma1")
    queens_gambit = visible("arma1qg")
    iron_front = visible("ironfront")
    assert r"ca\buildings\wall.p3d" not in arma1
    assert r"dbe1\buildings\palace.p3d" in arma1
    assert r"dbe1\buildings\palace.p3d" not in queens_gambit
    assert r"models_dbe1\objects\tree.p3d" not in queens_gambit
    assert r"sara_dbe1\plants\bush.p3d" not in queens_gambit
    assert r"lib\buildings\farm.p3d" not in iron_front
    assert r"lib_houses\structures\barn.p3d" not in iron_front
    assert r"lib_customtool\objects\car.p3d" not in iron_front
    assert r"dbe1\buildings\palace.p3d" in iron_front
    assert r"customaddon\objects\addon.p3d" in iron_front
    assert wrp.read_bytes() == data


def test_ironfront_ifa_archive_scans_without_renaming(tmp_path: Path) -> None:
    archive = tmp_path / "staszow.ifa"
    original = _write_wrp(tmp_path / "staszow.wrp", (
        r"lib_houses\buildings\farm.p3d",
        r"ownmod\props\object.p3d",
    ))
    write_pbo(archive, (PboEntry("staszow.wrp", original),))
    archive_bytes = archive.read_bytes()

    result = scan_dependencies(archive, game="ironfront")
    assert result.input_kind == "ifa"
    assert result.wrp_names == ("staszow.wrp",)
    assert result.unique_models == 2
    assert {row.model_path for row in rows_for_display(result)} == {
        r"ownmod\props\object.p3d",
    }
    assert archive.read_bytes() == archive_bytes


def test_ironfront_namespace_pattern_does_not_match_every_lib_prefix(tmp_path: Path) -> None:
    wrp = tmp_path / "addon.wrp"
    _write_wrp(wrp, (
        r"lib_objects\bridge.p3d",
        r"libraryaddon\bridge.p3d",
    ))
    visible = {row.model_path for row in rows_for_display(scan_dependencies(wrp, game="ironfront"))}
    assert visible == {r"libraryaddon\bridge.p3d"}


def test_sort_behavior_retained_when_adding_game_profiles() -> None:
    gui = ScannerGui.__new__(ScannerGui)
    gui.tree = Mock()
    gui._populate = Mock()
    gui._column_headings = {"wrp": "WRP", "refs": "Refs"}
    gui._sort_column = None
    gui._sort_descending = False
    gui._sort_by("refs")
    gui._sort_by("refs")
    assert gui._sort_descending is True
    gui.tree.heading.assert_any_call("refs", text="Refs ▼")
