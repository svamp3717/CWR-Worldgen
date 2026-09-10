from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from cwr_worldgen import generator, surface_pass
import cwr_worldgen.runway_surface_policy as runway_policy
from cwr_worldgen.cache import cache_key as raw_cache_key
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature
from cwr_worldgen.paa import inspect_paa
from cwr_worldgen.runway_model_policy import (
    DESERT_RUNWAY_END_TEXTURE,
    DESERT_RUNWAY_MIDDLE_TEXTURE,
    DESERT_RUNWAY_START_TEXTURE,
    GRASS_RUNWAY_END_TEXTURE,
    GRASS_RUNWAY_MIDDLE_TEXTURE,
    GRASS_RUNWAY_START_TEXTURE,
    runway_model_path,
    runway_texture_triplet,
)
from cwr_worldgen.runway_surface_policy import (
    RUNWAY_TEXTURE_PREFIX,
    RUNWAY_TEXTURE_SIZE,
    RVW4_TEXTURE_LIMIT,
    _profile_surface_colour,
    _render_runway_cell,
    _runway_geometries,
    _runway_texture_budget,
    apply_generated_runway_texture_table,
    install_runway_surface_policy,
    runway_overlay_objects,
    runway_texture_cell_indices,
)


def _runway_dataset(
    projection: BboxProjection,
    start: tuple[float, float] = (85.0, 20.0),
    end: tuple[float, float] = (85.0, 140.0),
    *,
    width: float = 30.0,
) -> OsmDataset:
    runway = OsmLineFeature(
        "way/49810543",
        {"aeroway": "runway", "surface": "asphalt", "width": str(width)},
        tuple(projection.to_latlon(point) for point in (start, end)),
    )
    return OsmDataset(
        source_generator="runway-test",
        element_count=1,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(),
        aeroway_lines=(runway,),
    )


def _spec(profile: str = "everon", *, name: str = "wg_runway"):
    return SimpleNamespace(
        name=name,
        ground_texture_profile=profile,
        surface_pass_enabled=True,
        surface_ground_mode="milestone9",
        deterministic_seed="runway-tests",
        cells=16,
        cell_size=10.0,
        world_size=160.0,
        sea_level=0.0,
    )


def _base_texture_table(profile: str = "everon") -> tuple[str, ...]:
    return (
        r"wg_runway\data\d.paa",
        *surface_pass.surface_texture_wire_paths("wg_runway", profile),
    )


def test_stock_runway_family_names_remain_verified_for_reference_and_fallback() -> None:
    assert runway_texture_triplet("everon") == (
        r"o\runtr_z.paa",
        r"o\runtr_d.paa",
        r"o\runtr_k.paa",
    )
    assert runway_texture_triplet("desert") == (
        r"o\runpi_z.paa",
        r"o\runpi_d.paa",
        r"o\runpi_k.paa",
    )
    assert GRASS_RUNWAY_START_TEXTURE == r"o\runtr_z.paa"
    assert GRASS_RUNWAY_MIDDLE_TEXTURE == r"o\runtr_d.paa"
    assert GRASS_RUNWAY_END_TEXTURE == r"o\runtr_k.paa"
    assert DESERT_RUNWAY_START_TEXTURE == r"o\runpi_z.paa"
    assert DESERT_RUNWAY_MIDDLE_TEXTURE == r"o\runpi_d.paa"
    assert DESERT_RUNWAY_END_TEXTURE == r"o\runpi_k.paa"


def test_runway_cells_are_selected_from_the_actual_mapped_width() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    spec = _spec()
    narrow = runway_texture_cell_indices(
        _runway_dataset(projection, width=10.0), projection, spec
    )
    wide = runway_texture_cell_indices(
        _runway_dataset(projection, width=40.0), projection, spec
    )
    assert narrow
    assert wide
    assert len(wide) > len(narrow)
    assert all(0 <= index < spec.cells * spec.cells for index in wide)


def test_nogova_runway_background_uses_selected_stock_ground_family() -> None:
    material_index = surface_pass.MATERIAL_INDEX["g"]
    material = surface_pass.MILESTONE9_MATERIALS[material_index]
    ground_path = surface_pass.surface_texture_wire_paths("wg_runway", "nogova")[material_index]
    assert ground_path == r"o\t1.paa"
    assert _profile_surface_colour(
        material, "nogova", ground_path=ground_path
    ) == (58, 66, 45)


def test_generated_runway_table_reuses_identical_cell_textures(tmp_path) -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(projection)
    spec = _spec("nogova")
    base_paths = _base_texture_table("nogova")
    grass_index = surface_pass.MATERIAL_INDEX["g"] + 1
    base_indices = (grass_index,) * (spec.cells * spec.cells)
    touched = runway_texture_cell_indices(dataset, projection, spec)

    revised_indices, revised_paths, generated = apply_generated_runway_texture_table(
        tmp_path,
        dataset,
        projection,
        spec,
        base_indices,
        base_paths,
    )

    assert 0 < len(generated) < len(touched)
    assert len(revised_paths) == len(base_paths) + len(generated)
    assert len(revised_paths) <= RVW4_TEXTURE_LIMIT
    assert all(path.startswith(r"wg_runway\rw") for path in generated)
    assert len(set(generated)) == len(generated)
    assert all(revised_indices[index] >= len(base_paths) for index in touched)
    assert len({revised_indices[index] for index in touched}) == len(generated)

    untouched = next(
        index for index in range(spec.cells * spec.cells)
        if index not in set(touched)
    )
    assert revised_indices[untouched] == grass_index

    for path in generated:
        filename = path.rsplit("\\", 1)[-1]
        local = tmp_path / filename
        assert local.is_file()
        summary = inspect_paa(local)
        assert summary.width == RUNWAY_TEXTURE_SIZE
        assert summary.height == RUNWAY_TEXTURE_SIZE

    report = json.loads((tmp_path / "runway-textures.json").read_text(encoding="utf-8"))
    assert report["mode"] == "generated-terrain-textures"
    assert report["runway_cells"] == len(touched)
    assert report["generated_runway_textures"] == len(generated)
    assert report["reused_runway_cell_assignments"] == len(touched) - len(generated)
    assert report["reuse_ratio"] > 0.0
    assert report["nogova_background_mode"] == "selected-stock-path-colour-match"


def test_generated_texture_path_fits_rvw4_even_for_maximum_world_name(tmp_path) -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(projection)
    spec = _spec(name="abcdefghijklmnopqrst")
    base_paths = (
        rf"{spec.name}\data\d.paa",
        *surface_pass.surface_texture_wire_paths(spec.name, "everon"),
    )
    grass_index = surface_pass.MATERIAL_INDEX["g"] + 1
    _indices, _paths, generated = apply_generated_runway_texture_table(
        tmp_path,
        dataset,
        projection,
        spec,
        (grass_index,) * (spec.cells * spec.cells),
        base_paths,
    )
    assert generated
    assert max(len(path.encode("ascii")) for path in generated) <= 31
    assert generated[0].rsplit("\\", 1)[-1].startswith(RUNWAY_TEXTURE_PREFIX)


def test_vertical_runway_is_drawn_along_texture_v_not_sideways() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(projection, start=(85.0, 20.0), end=(85.0, 140.0))
    spec = _spec("everon")
    geometries = _runway_geometries(dataset, projection, "everon")
    assert len(geometries) == 1
    cell_index = 8 * spec.cells + 8
    image = _render_runway_cell(
        cell_index=cell_index,
        original_wrp_texture_index=surface_pass.MATERIAL_INDEX["g"] + 1,
        geometries=geometries,
        materials=surface_pass.MILESTONE9_MATERIALS,
        spec=spec,
    )
    pixels = np.asarray(image)
    marking = np.asarray((224, 221, 207), dtype=np.uint8)
    marked = np.all(pixels == marking, axis=2)
    assert int(marked.sum(axis=0).max()) > int(marked.sum(axis=1).max()) * 4


def test_horizontal_generated_runway_rotates_the_marking_with_world_bearing() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(
        projection,
        start=(20.0, 85.0),
        end=(140.0, 85.0),
    )
    spec = _spec("everon")
    geometries = _runway_geometries(dataset, projection, "everon")
    cell_index = 8 * spec.cells + 8
    image = _render_runway_cell(
        cell_index=cell_index,
        original_wrp_texture_index=surface_pass.MATERIAL_INDEX["g"] + 1,
        geometries=geometries,
        materials=surface_pass.MILESTONE9_MATERIALS,
        spec=spec,
    )
    pixels = np.asarray(image)
    marking = np.asarray((224, 221, 207), dtype=np.uint8)
    marked = np.all(pixels == marking, axis=2)
    assert int(marked.sum(axis=1).max()) > int(marked.sum(axis=0).max()) * 4


def test_generated_runway_deck_is_neutral_not_green() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(projection, start=(85.0, 20.0), end=(85.0, 140.0))
    spec = _spec("nogova")
    image = _render_runway_cell(
        cell_index=8 * spec.cells + 8,
        original_wrp_texture_index=surface_pass.MATERIAL_INDEX["g"] + 1,
        geometries=_runway_geometries(dataset, projection, "nogova"),
        materials=surface_pass.MILESTONE9_MATERIALS,
        spec=spec,
    )
    pixels = np.asarray(image)
    # Sample inside the deck but away from the centre stripe and wheel tracks.
    sample = pixels[64, 20].astype(int)
    assert max(sample) - min(sample) < 20


def test_texture_budget_prefers_generated_paas_until_512_slots(monkeypatch) -> None:
    spec = _spec()
    monkeypatch.setattr(
        runway_policy,
        "_ORIGINAL_GROUND_TEXTURE_PATHS",
        lambda _spec: tuple(f"base{i}" for i in range(20)),
    )
    fits, base, final = _runway_texture_budget(spec, 100)
    assert fits
    assert base == 21
    assert final == 121

    fits, base, final = _runway_texture_budget(spec, 492)
    assert not fits
    assert base == 21
    assert final == 513


def test_p3d_fallback_still_has_rotatable_z_d_k_tiles() -> None:
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(projection, start=(20.0, 85.0), end=(140.0, 85.0))
    spec = _spec("everon")
    objects = runway_overlay_objects(
        dataset,
        projection,
        (10.0,) * (spec.cells * spec.cells),
        spec,
        starting_id=700,
    )
    assert [obj.object_id for obj in objects] == [700, 701, 702]
    assert [obj.model_path for obj in objects] == [
        runway_model_path("wg_runway", "everon", "z"),
        runway_model_path("wg_runway", "everon", "d"),
        runway_model_path("wg_runway", "everon", "k"),
    ]
    assert [obj.heading_degrees for obj in objects] == pytest.approx(
        [90.0, 90.0, 90.0], abs=1.0e-6
    )


def test_line_runway_is_removed_from_generic_paved_aeroway_mask() -> None:
    install_runway_surface_policy()
    projection = BboxProjection.create((0.0, 0.0, 1.0, 1.0), 160.0)
    dataset = _runway_dataset(projection)
    mask = surface_pass._aeroway_mask(dataset, projection, 16)
    assert not mask.any()


def test_runway_policy_invalidates_previous_surface_representations() -> None:
    install_runway_surface_policy()
    payload = {"world": "runway-test"}
    assert generator.cache_key(
        "surface-pipeline-v11-vectorized-material-pass",
        payload,
    ) == raw_cache_key(
        "surface-pipeline-v18-nogova-runway-blend-dedup",
        payload,
    )
