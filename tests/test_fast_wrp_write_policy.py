from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cwr_worldgen import wrp
from cwr_worldgen.fast_wrp_write_policy import _fast_write_rvw4, _install_writer_binding
from cwr_worldgen.model import WorldObject


def _objects(count: int) -> tuple[WorldObject, ...]:
    models = (
        r"o\road\sil25.p3d",
        r"data3d\ker listnac.p3d",
        r"world\g\house_a.p3d",
    )
    return tuple(
        WorldObject(
            object_id=index + 17,
            model_path=models[index % len(models)],
            x=float(index % 257) * 7.25,
            y=1.0 + float(index % 19) * 0.05,
            z=float(index // 257) * 11.5,
            heading_degrees=((index * 17.25) % 720.0) - 360.0,
            pitch_degrees=0.0 if index % 4 else ((index % 31) - 15) * 0.5,
        )
        for index in range(count)
    )


def test_vectorized_generator_writer_matches_scalar_rvw4_bytes(tmp_path: Path) -> None:
    width = height = 64
    cells = width * height
    elevations = tuple(((index % 101) - 50) * 0.05 for index in range(cells))
    texture_indices = tuple(index % 3 for index in range(cells))
    texture_paths = (
        r"world\data\g.paa",
        r"o\t1.paa",
        r"world\data\r.paa",
    )
    objects = _objects(4097)
    scalar = tmp_path / "scalar.wrp"
    fast = tmp_path / "fast.wrp"

    wrp.write_rvw4(
        scalar,
        width,
        height,
        elevations,
        texture_indices,
        texture_paths,
        objects,
        height_scale=0.05,
        renumber_object_ids=True,
    )
    _fast_write_rvw4(
        wrp.write_rvw4,
        wrp,
        fast,
        width,
        height,
        elevations,
        texture_indices,
        texture_paths,
        objects,
        height_scale=0.05,
        renumber_object_ids=True,
    )

    assert fast.read_bytes() == scalar.read_bytes()


def test_vectorized_writer_keeps_scalar_fallback_for_general_iterables(tmp_path: Path) -> None:
    width = height = 16
    cells = width * height
    objects = _objects(5)
    expected = tmp_path / "expected.wrp"
    actual = tmp_path / "actual.wrp"

    wrp.write_rvw4(
        expected,
        width,
        height,
        (0.0,) * cells,
        (0,) * cells,
        (r"world\data\g.paa",),
        iter(objects),
        height_scale=0.05,
        renumber_object_ids=True,
    )
    _fast_write_rvw4(
        wrp.write_rvw4,
        wrp,
        actual,
        width,
        height,
        (0.0,) * cells,
        (0,) * cells,
        (r"world\data\g.paa",),
        iter(objects),
        height_scale=0.05,
        renumber_object_ids=True,
    )

    assert actual.read_bytes() == expected.read_bytes()


def test_fast_writer_binds_beneath_existing_runway_wrapper() -> None:
    calls: list[str] = []

    def scalar_writer(*args, **kwargs):
        calls.append("scalar")

    runway = SimpleNamespace(_INSTALLED=True, _ORIGINAL_WRITE_RVW4=scalar_writer)

    def runway_wrapper(*args, **kwargs):
        calls.append("runway")
        return runway._ORIGINAL_WRITE_RVW4(*args, **kwargs)

    generator = SimpleNamespace(write_rvw4=runway_wrapper)
    result = _install_writer_binding(generator, wrp, runway)

    assert result == "runway-inner"
    assert generator.write_rvw4 is runway_wrapper
    assert runway._ORIGINAL_WRITE_RVW4 is not scalar_writer

    # renumber=False takes the scalar fallback, but it must still pass through
    # the runway-owned outer hook before reaching that inner writer in real use.
    generator.write_rvw4(
        Path("unused.wrp"),
        1,
        1,
        (0.0,),
        (0,),
        (r"world\data\g.paa",),
        (),
        height_scale=0.05,
        renumber_object_ids=False,
    )
    assert calls == ["runway", "scalar"]
