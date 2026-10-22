"""Worldgen sidecar diagnostic layers and Island Studio viewport alignment."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from cwr_island_studio.app import IslandStudio
from cwr_worldgen.debug_layers import (
    LAYERS, find_worldgen_debug_layers, load_worldgen_debug_layers,
    write_worldgen_debug_layers,
)


def _sample(tmp_path: Path):
    wrp = tmp_path / "testworld.wrp"
    wrp.write_bytes(b"synthetic-test-wrp")
    cells = 8
    false = (False,) * (cells * cells)
    forest = list(false)
    forest[7 * 8 + 1] = True  # north-west part of island
    raster = SimpleNamespace(
        water=false, forest=tuple(forest), roads=false,
        farmland=false, urban=false, buildings=false,
    )
    materials = (
        SimpleNamespace(code="g", name="grass", colour=(80, 130, 75)),
        SimpleNamespace(code="f", name="forest", colour=(30, 70, 35)),
    )
    debug = tmp_path / "testworld-worldgen-debug.npz"
    write_worldgen_debug_layers(
        debug, wrp_path=wrp, world_name="testworld",
        cells=cells, cell_size=50., raster=raster,
        base_material=[0] * 64,
        final_material=[1 if bit else 0 for bit in forest],
        wrp_slots=[2 if bit else 1 for bit in forest],
        materials=materials,
        forest_objects=[SimpleNamespace(x=75., z=375.), SimpleNamespace(x=75., z=375.)],
    )
    return wrp, debug


def test_companion_contains_true_worldgen_stage_masks_and_placements(tmp_path: Path) -> None:
    wrp, debug_path = _sample(tmp_path)
    data = load_worldgen_debug_layers(
        debug_path, world_width=400, world_height=400, wrp_path=wrp,
    )
    assert set(data.rasters) == set(LAYERS)
    assert data.metadata["format"] == "cwr-worldgen-debug-layers-v1"
    assert data.metadata["wrp_sha256"] == hashlib.sha256(wrp.read_bytes()).hexdigest()
    assert data.sample("osm_forest", 75, 375) == 1
    assert data.sample("osm_forest", 75, 25) == 0
    assert data.sample("vegetation_placements", 75, 375) == 2
    assert data.sample("semantic_final", 75, 375) == 1
    assert data.sample("wrp_slots", 75, 375) == 2
    assert data.sample("osm_forest", -5, 375) is None
    assert "OSM forest" in data.choices
    assert data.name_key("OSM forest") == "osm_forest"


def test_debug_composite_is_north_up_and_preserves_background_off_mask(tmp_path: Path) -> None:
    wrp, debug_path = _sample(tmp_path)
    data = load_worldgen_debug_layers(
        debug_path, world_width=400, world_height=400, wrp_path=wrp,
    )
    background = np.full((8, 8, 3), (45, 54, 61), dtype=np.uint8)
    result = data.composite(
        background, "osm_forest", left=0, top=400, right=400, bottom=0,
    )
    assert not np.array_equal(result[0, 1], background[0, 1])
    np.testing.assert_array_equal(result[7, 1], background[7, 1])
    np.testing.assert_array_equal(result[0, 2], background[0, 2])
    # Semantic and WRP layers have different actual source data.
    semantic = data.composite(
        background, "semantic_final", left=0, top=400, right=400, bottom=0,
    )
    slots = data.composite(
        background, "wrp_slots", left=0, top=400, right=400, bottom=0,
    )
    assert not np.array_equal(semantic, slots)


def test_debug_sidecar_rejects_wrong_world_and_dimensions(tmp_path: Path) -> None:
    wrp, path = _sample(tmp_path)
    with pytest.raises(ValueError, match="different world size"):
        load_worldgen_debug_layers(path, world_width=800, world_height=400, wrp_path=wrp)
    wrp.write_bytes(b"a different WRP")
    with pytest.raises(ValueError, match="do not match"):
        load_worldgen_debug_layers(path, world_width=400, world_height=400, wrp_path=wrp)


def test_sidecar_discovery_finds_source_wrp_and_pbo_output_layout(tmp_path: Path) -> None:
    _, debug = _sample(tmp_path)
    source = tmp_path / "source" / "testworld" / "testworld.wrp"
    source.parent.mkdir(parents=True)
    assert find_worldgen_debug_layers(source, "testworld") == debug
    pbo = tmp_path / "@CWR-Milestone9" / "Addons" / "testworld.pbo"
    pbo.parent.mkdir(parents=True)
    assert find_worldgen_debug_layers(pbo, "testworld") == debug
    assert find_worldgen_debug_layers(pbo, "another") is None


def test_island_studio_attach_debug_layers_updates_selector(tmp_path: Path) -> None:
    wrp, debug = _sample(tmp_path)
    options = []
    chosen = []
    editor = SimpleNamespace(
        document=SimpleNamespace(path=wrp, world_width=400., world_height=400.),
        _busy=False, status=Mock(), debug_data=None,
        debug_combo=SimpleNamespace(configure=lambda **kw: options.append(kw)),
        debug_layer=SimpleNamespace(set=lambda label: chosen.append(label)),
        _queue_draw=Mock(),
    )
    IslandStudio._attach_worldgen_debug(editor, debug)
    assert editor.debug_data is not None
    assert options[-1]["values"][0] == "Off"
    assert "OSM forest" in options[-1]["values"]
    assert chosen == ["OSM forest"]
    editor._queue_draw.assert_called_once()


def test_debug_file_never_becomes_a_game_asset(tmp_path: Path) -> None:
    _, debug = _sample(tmp_path)
    assert debug.name == "testworld-worldgen-debug.npz"
    assert not (tmp_path / "source" / "testworld" / debug.name).exists()
