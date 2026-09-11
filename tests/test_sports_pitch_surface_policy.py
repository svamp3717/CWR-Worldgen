from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
from PIL import Image

from cwr_worldgen import generator
from cwr_worldgen import runway_exact_background_policy as exact
from cwr_worldgen import runway_surface_policy as runway
from cwr_worldgen.sports_pitch_surface_policy import (
    _pitch_cells,
    _pitch_geometries,
    _render_pitch_cell,
    apply_sports_pitch_textures,
)


class _IdentityProjection:
    @staticmethod
    def to_world(point):
        return float(point[0]), float(point[1])


def _dataset():
    polygon = SimpleNamespace(
        outer=((20.0, 20.0), (90.0, 20.0), (90.0, 60.0), (20.0, 60.0), (20.0, 20.0))
    )
    feature = SimpleNamespace(
        osm_key="way/football-1",
        tags={"site": "sports_pitch", "sport": "soccer"},
        polygons=(polygon,),
    )
    return SimpleNamespace(sites=(feature,))


def _spec(tmp_path=None):
    return SimpleNamespace(
        name="pitchworld",
        cells=4,
        cell_size=50.0,
        ground_texture_profile="generated",
        deterministic_seed="pitch-test",
        cache_enabled=True,
        cache_refresh=False,
        cache_dir=(tmp_path / "cache") if tmp_path is not None else None,
    )


def test_pitch_geometry_and_cells_follow_osm_rectangle() -> None:
    geometries = _pitch_geometries(_dataset(), _IdentityProjection())
    assert len(geometries) == 1
    pitch = geometries[0]
    assert pitch.soccer
    assert abs(pitch.half_length - 35.0) < 1.0e-6
    assert abs(pitch.half_width - 20.0) < 1.0e-6

    cells = _pitch_cells(geometries, 4, 50.0)
    assert set(cells) == {0, 1, 4, 5}


def test_pitch_render_keeps_background_outside_markings() -> None:
    spec = _spec()
    geometries = _pitch_geometries(_dataset(), _IdentityProjection())
    base = Image.new("RGB", (128, 128), (62, 78, 49))

    rendered = _render_pitch_cell(
        cell_index=0,
        geometries=geometries,
        spec=spec,
        base=base,
    )
    source = np.asarray(base)
    actual = np.asarray(rendered)

    assert np.array_equal(actual[0, 0], source[0, 0])
    assert np.count_nonzero(np.any(actual != source, axis=2)) > 0
    assert int(actual.max()) > 180


def test_pitch_texture_generation_reuses_persistent_cache(tmp_path, monkeypatch) -> None:
    spec = _spec(tmp_path)
    dataset = _dataset()
    projection = _IdentityProjection()
    base = Image.new("RGB", (128, 128), (62, 78, 49))

    monkeypatch.setattr(
        generator,
        "_material_definitions",
        lambda _spec: (SimpleNamespace(code="g", name="grass", colour=(62, 78, 49)),),
    )
    monkeypatch.setattr(
        generator,
        "_ground_texture_paths",
        lambda _spec: (r"pitchworld\data\g.paa",),
    )
    monkeypatch.setattr(exact, "_load_exact_texture", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        runway,
        "_background_texture",
        lambda *_args, **_kwargs: base.copy(),
    )

    source_dir = tmp_path / "pitchworld"
    source_dir.mkdir()
    texture_indices = (1,) * (spec.cells * spec.cells)
    texture_paths = (r"pitchworld\data\d.paa", r"pitchworld\data\g.paa")

    first_indices, first_paths, first_generated = apply_sports_pitch_textures(
        source_dir,
        dataset,
        projection,
        spec,
        texture_indices,
        texture_paths,
    )
    assert first_generated
    assert len(first_paths) > len(texture_paths)
    assert any(first_indices[index] >= len(texture_paths) for index in (0, 1, 4, 5))
    first_report = json.loads((source_dir / "sports-pitch-textures.json").read_text())
    assert first_report["cache_misses"] == 4
    assert first_report["background_path"] == r"pitchworld\data\g.paa"

    second_indices, second_paths, second_generated = apply_sports_pitch_textures(
        source_dir,
        dataset,
        projection,
        spec,
        texture_indices,
        texture_paths,
    )
    second_report = json.loads((source_dir / "sports-pitch-textures.json").read_text())
    assert second_report["cache_hits"] == 4
    assert second_report["cache_misses"] == 0
    assert second_indices == first_indices
    assert second_paths == first_paths
    assert second_generated == first_generated
