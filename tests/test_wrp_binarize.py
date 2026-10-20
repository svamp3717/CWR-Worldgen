"""Experimental OFP OPRW v3 binarization and compatibility regression tests."""
from __future__ import annotations

from pathlib import Path
import struct

import pytest

from cwr_worldgen.model import WorldObject
from cwr_worldgen.wrp import write_rvw4
from cwr_worldgen.wrp_binarize import CWA_RVW4_HEIGHT_SCALE, _packed, binarize_rvw4
from cwr_island_studio.oprw import OprwIslandDocument, open_island_document


def _source(path: Path, cells: int = 16) -> None:
    elevations = [((i % cells) + i // cells) * 0.5 - 15 for i in range(cells*cells)]
    textures = [1 if i % 3 else 2 for i in range(cells*cells)]
    write_rvw4(
        path, cells, cells, elevations, textures,
        [r"landtext\mo.pac", r"o\data\s.paa", r"o\data\w.paa"],
        (
            WorldObject(10, r"o\road\sil25.p3d", 120, 4, 200, 90),
            WorldObject(11, r"data3d\strom.p3d", 200, 8, 120, 35),
            WorldObject(12, r"o\road\sil25.p3d", 100, 9, 50, 0),
        ),
        height_scale=0.5,
    )


def test_binarize_generated_rvw4_oprw_v3_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "world.wrp"
    target = tmp_path / "world.binarized.wrp"
    _source(source)
    original = source.read_bytes()
    object_count = binarize_rvw4(source, target, height_scale=0.5)
    assert object_count == 3
    assert source.read_bytes() == original
    assert target.read_bytes()[:8] == b"OPRW\x03\x00\x00\x00"
    with open_island_document(target) as doc:
        assert isinstance(doc, OprwIslandDocument)
        assert (doc.width, doc.height) == (16, 16)
        assert doc.oprw_version == 3
        assert doc.terrain_at(0, 0) == pytest.approx(-15.0)
        assert doc.terrain_at(3, 2) == pytest.approx(-12.5)
        assert doc.texture_at(0, 0) == 2
        assert doc.texture_at(1, 0) == 1
        assert doc.texture_paths[2] == r"o\data\w.paa"
        assert doc.model_names == (
            r"o\road\sil25.p3d", r"data3d\strom.p3d",
        )
        assert [doc.object_at(i).object_id for i in range(3)] == [10, 11, 12]
        assert [doc.object_at(i).x for i in range(3)] == pytest.approx([120, 200, 100])
        doc.build_object_index()
        assert doc.object_count == 3
        with pytest.raises(PermissionError):
            doc.export(tmp_path / "no.wrp")


def test_binarize_uses_cwa_game_4wvr_scale_not_writer_scale(tmp_path: Path) -> None:
    """CWR-CE always multiplies 4WVR height samples by 0.045 metres.

    Simulate a legacy Worldgen file written with the old 0.05 quantizer.
    Passing that obsolete scale to the OPRW converter would raise positive
    ground 11.1% while leaving tree Y unchanged (sunk trees). New Worldgen
    output instead quantizes at the engine's fixed 0.045 scale.
    """
    source = tmp_path / "same_seed.wrp"
    binary = tmp_path / "same_seed.binarized.wrp"
    assert CWA_RVW4_HEIGHT_SCALE == pytest.approx(0.03 * 1.5)
    tree = WorldObject(999, r"data3d\\str_fikovnik.p3d", 75, 20.0, 75)
    write_rvw4(
        source, 16, 16, [20.0] * 256, [0] * 256,
        [r"landtext\\pi.pac"], [tree], height_scale=0.05,
    )
    raw_sample = struct.unpack_from("<h", source.read_bytes(), 12)[0]
    assert raw_sample == 400
    assert binarize_rvw4(source, binary) == 1  # use engine-correct default
    with open_island_document(binary) as doc:
        assert doc.terrain_at(0, 0) == pytest.approx(
            raw_sample * CWA_RVW4_HEIGHT_SCALE, abs=1e-5,
        )
        assert doc.terrain_at(0, 0) == pytest.approx(18.0)
        assert doc.object_at(0).y == pytest.approx(20.0)
        assert doc.object_at(0).object_id == 999


def test_generated_world_specs_keep_trees_on_game_ground_in_both_formats(
    tmp_path: Path,
) -> None:
    """All generator milestones use the same CWA 4WVR terrain unit as OPRW."""
    from cwr_worldgen.model import HeightmapSpec, WorldSpec
    from cwr_island_studio.world_compare import compare_worlds

    world = WorldSpec(profile="cwr-ce")
    raster = HeightmapSpec(heightmap_path=tmp_path / "height.png")
    assert world.height_scale == raster.height_scale == CWA_RVW4_HEIGHT_SCALE

    source = tmp_path / "test.wrp"
    output = tmp_path / "test.oprw"
    y = 9.0  # 200 units at the actual game height scale
    write_rvw4(
        source, 16, 16, [y] * (16 * 16), [0] * (16 * 16),
        [r"landtext\\main.paa"],
        [
            WorldObject(13, r"data3d\\strom.p3d", 150, y, 200),
            WorldObject(14, r"o\\road\\sil25.p3d", 400, y, 450),
        ], height_scale=world.height_scale,
    )
    assert struct.unpack_from("<h", source.read_bytes(), 12)[0] == 200
    binarize_rvw4(source, output)
    with open_island_document(
        source, height_scale=world.height_scale,
    ) as editable, open_island_document(
        output, height_scale=world.height_scale, cwa_runtime_scale=True,
    ) as runtime:
        assert editable.terrain_at(3, 5) == pytest.approx(y)
        assert runtime.terrain_at(3, 5) == pytest.approx(y)
        assert editable.object_at(0).y == runtime.object_at(0).y == pytest.approx(y)
        report = compare_worlds(editable, runtime)
        assert report.terrain_changes == 0, report.report()
        assert report.moved_objects == 0, report.report()
        assert report.rotated_or_scaled_objects == 0, report.report()


def test_slope_and_coast_heights_ground_objects_identically_in_both_formats(
    tmp_path: Path,
) -> None:
    """CWA sees the same signed terrain and grounded objects in 4WVR and OPRW.

    A non-flat grid catches unit conversion bugs that flat-ground tests can
    miss. Read against the engine's fixed 0.045 m/int16 contract rather than
    passing through the scale chosen by the converter itself.
    """
    from cwr_island_studio.world_compare import compare_worlds
    from cwr_worldgen.model import WorldSpec
    from cwr_worldgen.wrp import quantize_elevations

    scale = WorldSpec(profile="cwr-ce").height_scale
    assert scale == pytest.approx(0.045)
    cells = 16
    grid_metres = 50.0
    raw_units = [
        -440 + 70 * row + 13 * column
        for row in range(cells) for column in range(cells)
    ]
    metres = [unit * scale for unit in raw_units]
    assert min(metres) < 0 < max(metres)
    assert quantize_elevations(metres, scale) == pytest.approx(metres)

    # Different heights and slopes, with tree, road, building and fence
    # anchors coinciding with known terrain vertices.
    anchors = [
        (4, 7, r"data3d\strom.p3d", 0.0),
        (8, 10, r"o\road\sil25.p3d", 7.0),
        (13, 12, r"data3d\house.p3d", 0.0),
        (0, 10, r"data3d\fence.p3d", 0.0),
    ]
    objects = [
        WorldObject(
            400 + index, model, column * grid_metres,
            metres[row * cells + column], row * grid_metres,
            pitch_degrees=pitch,
        )
        for index, (column, row, model, pitch) in enumerate(anchors)
    ]

    source = tmp_path / "sloped.wrp"
    binary = tmp_path / "sloped.oprw"
    write_rvw4(
        source, cells, cells, metres, [0] * (cells * cells),
        [r"landtext\pi.pac"], objects, height_scale=scale,
    )
    raw_file = source.read_bytes()
    assert [
        struct.unpack_from("<h", raw_file, 12 + index * 2)[0]
        for index in range(cells * cells)
    ] == raw_units

    assert binarize_rvw4(source, binary) == len(objects)
    assert source.read_bytes() == raw_file
    with open_island_document(
        source, cell_size=grid_metres, height_scale=scale,
        cwa_runtime_scale=True,
    ) as loose, open_island_document(
        binary, cell_size=grid_metres, height_scale=scale,
        cwa_runtime_scale=True,
    ) as packed:
        for row in range(cells):
            for column in range(cells):
                engine_height = raw_units[row * cells + column] * 0.045
                assert loose.terrain_at(column, row) == pytest.approx(
                    engine_height, abs=1e-5,
                )
                assert packed.terrain_at(column, row) == pytest.approx(
                    engine_height, abs=1e-5,
                )

        for index, (column, row, _model, _pitch) in enumerate(anchors):
            engine_height = raw_units[row * cells + column] * 0.045
            assert loose.object_at(index).y == pytest.approx(
                engine_height, abs=1e-5,
            )
            assert packed.object_at(index).y == pytest.approx(
                engine_height, abs=1e-5,
            )

        report = compare_worlds(loose, packed)
        assert report.terrain_changes == 0, report.report()
        assert report.moved_objects == 0, report.report()
        assert report.rotated_or_scaled_objects == 0, report.report()


def test_binarization_supports_large_compressed_grids(tmp_path: Path) -> None:
    source = tmp_path / "large.wrp"
    target = tmp_path / "large.oprw.wrp"
    _source(source, cells=64)
    assert binarize_rvw4(source, target, height_scale=0.5) == 3
    with open_island_document(target) as doc:
        assert doc.terrain_at(63, 63) == pytest.approx(48.0)
        assert doc.texture_at(63, 63) == 2
        assert doc.object_count == 3


def test_binarize_rejects_overwriting_source_or_truncated_objects(tmp_path: Path) -> None:
    source = tmp_path / "test.wrp"
    _source(source)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        binarize_rvw4(source, source)
    assert source.read_bytes() == before
    source.write_bytes(before[:-1])
    target = tmp_path / "out.wrp"
    with pytest.raises(ValueError, match="misaligned"):
        binarize_rvw4(source, target)
    assert not target.exists()


def test_literal_and_repeated_sscompress_blocks_are_readable(tmp_path: Path) -> None:
    # A large all-zero grid collapses into backward references.
    raw = bytes(4096)
    encoded = _packed(raw)
    assert len(encoded) < 1500
    from cwr_island_studio.oprw import _read_packed
    import io
    assert _read_packed(io.BytesIO(encoded), len(raw), "grid") == raw
    assert _packed(b"test") == b"test"
    # Repeating four-byte patterns uses a different reference distance.
    raw = (b"ABCD" * 2048)
    assert _read_packed(io.BytesIO(_packed(raw)), len(raw), "grid") == raw


def test_repacked_game_pbo_contains_oprw_but_source_stays_rvw4(
    tmp_path: Path,
) -> None:
    from cwr_worldgen.pbo import pack_directory, read_pbo
    from cwr_worldgen.wrp_binarize import repack_game_pbo_with_oprw

    source_dir = tmp_path / "source" / "island"
    source_dir.mkdir(parents=True)
    source_wrp = source_dir / "island.wrp"
    source_config = source_dir / "config.cpp"
    source_config.write_text("class CfgWorlds { class island {}; };")
    _source(source_wrp)
    editable = source_wrp.read_bytes()
    game_pbo = tmp_path / "island.pbo"
    pack_directory(source_dir, game_pbo, backend="python")
    assert game_pbo.read_bytes().count(b"4WVR") >= 1

    binary = tmp_path / "island.binarized.wrp"
    binarize_rvw4(source_wrp, binary, height_scale=0.5)
    repack_game_pbo_with_oprw(
        source_dir, source_wrp, binary, game_pbo, backend="python",
    )
    assert source_wrp.read_bytes() == editable
    entries = {e.name.replace("\\", "/").casefold(): e.data for e in read_pbo(game_pbo)}
    assert entries["island.wrp"][:8] == b"OPRW\x03\x00\x00\x00"
    assert entries["config.cpp"] == source_config.read_bytes()
    assert binary.read_bytes() == entries["island.wrp"]


def test_failed_repack_restores_previous_game_pbo_and_editable_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cwr_worldgen.pbo as pack_module
    from cwr_worldgen.wrp_binarize import repack_game_pbo_with_oprw

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_wrp = source_dir / "world.wrp"
    _source(source_wrp)
    original = source_wrp.read_bytes()
    game_pbo = tmp_path / "game.pbo"
    game_pbo.write_bytes(b"previous game archive")
    binary = tmp_path / "new.wrp"
    binarize_rvw4(source_wrp, binary, height_scale=0.5)

    def fail_pack(*_args, **_kwargs):
        raise RuntimeError("simulated pack failure")

    monkeypatch.setattr(pack_module, "pack_directory", fail_pack)
    with pytest.raises(RuntimeError, match="simulated"):
        repack_game_pbo_with_oprw(source_dir, source_wrp, binary, game_pbo)
    assert source_wrp.read_bytes() == original
    assert game_pbo.read_bytes() == b"previous game archive"
    assert not list(tmp_path.glob(".wrp-binarize-*"))


def test_advanced_checkbox_flows_through_gui_command_without_default_enabling() -> None:
    from cwr_worldgen.gui import default_gui_values, build_milestone9_command
    defaults = default_gui_values()
    assert defaults["binarize_world"] is False
    defaults.update({
        "source_dir": "source", "output": "output",
        "name": "island", "display_name": "Island",
    })
    argv = build_milestone9_command(defaults, python="python")
    assert "--binarize-world" not in argv
    defaults["binarize_world"] = True
    assert "--binarize-world" in build_milestone9_command(defaults, python="python")


def test_binarize_world_option_is_saved_and_restored(tmp_path: Path) -> None:
    from cwr_worldgen.gui import (
        WorldgenGui, default_gui_values, defaults_with_recent_source,
        load_gui_state, save_gui_state,
    )

    class Checkbox:
        def __init__(self):
            self.value = False
            self.callbacks = []
        def get(self):
            return self.value
        def set(self, value):
            self.value = value
            for callback in self.callbacks:
                callback()
        def trace_add(self, mode, callback):
            self.callbacks.append(callback)

    state_path = tmp_path / "state.json"
    save_gui_state(state_path, {"ground_textures": "malden"})
    gui = object.__new__(WorldgenGui)
    gui.state_path = state_path
    checkbox = Checkbox()
    gui.vars = {"binarize_world": checkbox}
    WorldgenGui._install_binarized_world_state_persistence(gui)
    checkbox.set(True)
    assert load_gui_state(state_path)["binarize_world"] is True
    restored = defaults_with_recent_source(default_gui_values(), load_gui_state(state_path))
    assert restored["binarize_world"] is True
    checkbox.set(False)
    assert load_gui_state(state_path)["binarize_world"] is False
    assert load_gui_state(state_path)["ground_textures"] == "malden"


def test_binarize_world_cli_accepts_advanced_command_option() -> None:
    from cwr_worldgen.gui import build_milestone9_command, default_gui_values
    from cwr_worldgen.cli import _parser
    values = default_gui_values()
    values.update({
        "source_dir": "source", "output": "build",
        "name": "island", "display_name": "Island",
        "binarize_world": True,
    })
    argv = build_milestone9_command(values, python="python")
    parsed = _parser().parse_args(argv[argv.index("milestone9"):])
    assert parsed.binarize_world is True



def test_milestone9_postbuild_binarization_uses_cwa_engine_height_scale(
    tmp_path: Path,
) -> None:
    """Regress the AttributeError after a completed Milestone 9 generation."""
    from types import SimpleNamespace
    from cwr_worldgen.milestone9 import Milestone9Spec, _binarize_generated_world
    from cwr_worldgen.pbo import pack_directory, read_pbo

    world_dir = tmp_path / "source" / "testworld"
    world_dir.mkdir(parents=True)
    original_wrp = world_dir / "testworld.wrp"
    write_rvw4(
        original_wrp, 16, 16, [3.25] * (16 * 16), [1] * (16 * 16),
        [r"testworld\data\d.paa", r"testworld\data\g.paa"],
        [WorldObject(100, r"data3d\strom.p3d", 40, 3.25, 80)],
        height_scale=0.05,
    )
    original_bytes = original_wrp.read_bytes()
    game_pbo = tmp_path / "testworld.pbo"
    pack_directory(world_dir, game_pbo, backend="python")

    # The top-level Milestone9Spec intentionally has no height_scale.
    # CWA's *runtime* 4WVR scale is fixed at 0.045, even though Worldgen
    # quantizes height samples using playability.height_scale (0.05).
    spec = Milestone9Spec(
        source_dir=world_dir, name="testworld",
        binarize_world=True, pbo_backend="python",
    )
    assert not hasattr(spec, "height_scale")
    stage = SimpleNamespace(
        output_dir=tmp_path, source_dir=world_dir,
        wrp_path=original_wrp, pbo_path=game_pbo,
    )
    playability = SimpleNamespace(height_scale=0.05)
    binary = _binarize_generated_world(stage, spec, playability)

    assert binary.exists()
    assert original_wrp.read_bytes() == original_bytes
    with open_island_document(binary) as document:
        # 3.25 / 0.05 = 65 raw 4WVR units; the game loads them as 65×0.045.
        assert document.terrain_at(0, 0) == pytest.approx(2.925)
        assert document.object_at(0).y == pytest.approx(3.25)
        assert document.object_count == 1
    packed = {item.name: item.data for item in read_pbo(game_pbo)}
    assert packed["testworld.wrp"] == binary.read_bytes()
