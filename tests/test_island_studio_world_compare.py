"""World comparison across editable 4WVR and CWA OPRW v3."""
from __future__ import annotations

from concurrent.futures import CancelledError
from pathlib import Path

import pytest

from cwr_island_studio.oprw import open_island_document
from cwr_island_studio.world_compare import compare_worlds
from cwr_worldgen.model import WorldObject
from cwr_worldgen.wrp import write_rvw4
from cwr_worldgen.wrp_binarize import binarize_rvw4


def make_world(
    path: Path, *, change: bool = False, reorder: bool = False,
    duplicate: bool = False,
) -> None:
    heights = [5.0 + n * 0.05 for n in range(64)]
    textures = [0 if n % 2 else 1 for n in range(64)]
    if change:
        heights[18] += 1.25
        textures[20] = 1 - textures[20]
    items = [
        WorldObject(10, r"data3d\strom.p3d", 50, 10, 75, 0),
        WorldObject(11, r"o\road\sil25.p3d", 120, 12, 90, 90),
        WorldObject(12, r"data3d\ker.p3d", 140, 8, 90, 60),
    ]
    if change:
        items[0] = WorldObject(10, r"data3d\strom.p3d", 50, 8, 75, 0)
        items[1] = WorldObject(11, r"data3d\other.p3d", 120, 12, 90, 25)
        items = items[:-1] + [
            WorldObject(99, r"data3d\ker.p3d", 140, 8, 90, 60),
        ]
    if reorder:
        items.reverse()
    if duplicate:
        items.append(WorldObject(13, r"data3d\ker.p3d", 130, 0, 80))
    write_rvw4(
        path, 8, 8, heights, textures,
        [r"landtext\main.pac", r"data\sand.paa"],
        items, height_scale=0.05,
    )
    if duplicate:
        import struct
        # The generator intentionally prohibits duplicate IDs. Patch the
        # fourth record on disk to emulate a real-world malformed WRP.
        raw = bytearray(path.read_bytes())
        struct.pack_into("<i", raw, 12 + 64 * 4 + 512 * 32 + 3 * 128 + 48, 11)
        path.write_bytes(raw)


def test_4wvr_oprw_round_trip_has_no_terrain_object_or_texture_changes(
    tmp_path: Path,
) -> None:
    loose = tmp_path / "island.wrp"
    binary = tmp_path / "island.binary.wrp"
    make_world(loose)
    binarize_rvw4(loose, binary, height_scale=0.05)
    before = loose.read_bytes()
    with open_island_document(loose, height_scale=0.05) as reference:
        with open_island_document(binary, height_scale=0.05) as candidate:
            report = compare_worlds(reference, candidate)
            assert report.terrain_changes == 0
            assert report.max_height_difference == pytest.approx(0, abs=0.001)
            assert report.texture_changes == 0
            assert report.texture_table_changes == 0
            assert report.moved_objects == 0
            assert report.rotated_or_scaled_objects == 0
            assert report.changed_models == 0
            assert report.unique_ids_reference_only == 0
            assert report.unique_ids_candidate_only == 0
            assert report.ambiguous_ids == 0
            assert "0" in report.report()
    assert loose.read_bytes() == before


def test_reports_terrain_texture_model_object_movements_and_missing_ids(
    tmp_path: Path,
) -> None:
    a, b = tmp_path / "a.wrp", tmp_path / "b.wrp"
    make_world(a)
    make_world(b, change=True, reorder=True)
    with open_island_document(a, height_scale=0.05) as left:
        with open_island_document(b, height_scale=0.05) as right:
            result = compare_worlds(left, right)
    assert result.terrain_changes == 1
    assert result.max_height_difference == pytest.approx(1.25)
    assert result.texture_changes == 1
    assert result.moved_objects == 1
    assert result.rotated_or_scaled_objects == 1
    assert result.changed_models == 1
    assert result.unique_ids_reference_only == 1
    assert result.unique_ids_candidate_only == 1
    assert set(p.category for p in result.positions) >= {"Terrain", "Texture", "Object"}
    assert any(p.category == "Missing object" for p in result.positions)


def test_object_order_is_irrelevant_and_duplicate_ids_reported(
    tmp_path: Path,
) -> None:
    a, b = tmp_path / "a.wrp", tmp_path / "b.wrp"
    make_world(a)
    make_world(b, reorder=True)
    with open_island_document(a, height_scale=0.05) as left:
        with open_island_document(b, height_scale=0.05) as right:
            report = compare_worlds(left, right)
    assert (report.moved_objects, report.rotated_or_scaled_objects, report.changed_models) == (0, 0, 0)

    make_world(b, duplicate=True)
    with open_island_document(a, height_scale=0.05) as left:
        with open_island_document(b, height_scale=0.05) as right:
            result = compare_worlds(left, right)
    assert result.ambiguous_ids == 2
    assert result.moved_objects == 0


def test_cancel_and_differently_sized_grids(tmp_path: Path) -> None:
    a, b = tmp_path / "a.wrp", tmp_path / "b.wrp"
    make_world(a)
    write_rvw4(
        b, 4, 4, [10.] * 16, [1] * 16,
        ["a.paa", "b.paa"], [], height_scale=0.05,
    )
    with open_island_document(a, height_scale=0.05) as left:
        with open_island_document(b, height_scale=0.05) as right:
            with pytest.raises(CancelledError):
                compare_worlds(left, right, cancel=lambda: True)
            result = compare_worlds(left, right)
    assert result.terrain_changes is None
    assert result.texture_changes is None
    assert "N/A" in result.report()



def test_gui_compare_loose_wrp_runs_background_and_preserves_open_world(
    tmp_path: Path,
) -> None:
    from concurrent.futures import Future
    from types import SimpleNamespace
    import queue
    from cwr_island_studio.app import IslandStudio

    a, b = tmp_path / "a.wrp", tmp_path / "b.wrp"
    make_world(a)
    make_world(b, change=True)

    def immediate(task):
        future = Future()
        try:
            future.set_result(task())
        except BaseException as exc:
            future.set_exception(exc)
        return future

    with open_island_document(a, height_scale=0.05) as reference:
        reports = []
        scheduled = []
        statuses = []
        fake = SimpleNamespace(
            document=reference, _busy=False,
            cell_size=SimpleNamespace(get=lambda: "50"),
            height_scale=SimpleNamespace(get=lambda: "0.05"),
            auto_grid=SimpleNamespace(get=lambda: False),
            _num=lambda item, _name: float(item.get()),
            _executor=SimpleNamespace(submit=immediate),
            _messages=queue.Queue(), status=SimpleNamespace(set=statuses.append),
            _index_future=None, _model_future=None, _waiting_pbos=None,
            _closing=False,
            root=SimpleNamespace(after=lambda ms, cb: scheduled.append((ms, cb))),
            _show_comparison=reports.append,
        )
        fake._poll = lambda: IslandStudio._poll(fake)
        IslandStudio._compare_path(fake, b)
        assert fake._busy is True
        assert not reports
        fake._poll()
        assert fake._busy is False
        assert len(reports) == 1
        assert reports[0].moved_objects == 1
        assert fake.document is reference
        assert reference.path.exists()
        assert scheduled[-1][0] == 150
        assert "complete" in statuses[-1].lower()


def test_gui_compare_a_world_in_pbo_without_replacing_current_world(
    tmp_path: Path,
) -> None:
    from concurrent.futures import Future
    from types import SimpleNamespace
    import queue
    from cwr_island_studio.app import IslandStudio
    from cwr_worldgen.pbo import pack_directory

    source = tmp_path / "source"
    source.mkdir()
    first, second = tmp_path / "reference.wrp", source / "b.wrp"
    make_world(first)
    make_world(second, change=True)
    archive_path = tmp_path / "archive.pbo"
    pack_directory(source, archive_path, backend="python")

    def immediate(task):
        future = Future()
        future.set_result(task())
        return future

    with open_island_document(first, height_scale=0.05) as reference:
        reports = []
        fake = SimpleNamespace(
            document=reference, _busy=False,
            cell_size=SimpleNamespace(get=lambda: "50"),
            height_scale=SimpleNamespace(get=lambda: "0.05"),
            auto_grid=SimpleNamespace(get=lambda: False),
            _num=lambda item, _name: float(item.get()),
            _executor=SimpleNamespace(submit=immediate),
            _messages=queue.Queue(), status=SimpleNamespace(set=lambda msg: None),
            _index_future=None, _model_future=None, _waiting_pbos=None,
            _closing=False, root=SimpleNamespace(after=lambda ms, fn: None),
            _show_comparison=reports.append,
        )
        fake._compare_archive_member = (
            lambda *args: IslandStudio._compare_archive_member(fake, *args)
        )
        fake._select_comparison_member = (
            lambda *args: IslandStudio._select_comparison_member(fake, *args)
        )
        fake._poll = lambda: IslandStudio._poll(fake)
        IslandStudio._compare_path(fake, archive_path)
        fake._poll()
        assert len(reports) == 1
        assert reports[0].terrain_changes == 1
        assert reports[0].reference_objects == 3
        assert fake.document is reference
        assert fake._busy is False
