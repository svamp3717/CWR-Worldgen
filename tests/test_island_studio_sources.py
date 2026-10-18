"""Multiple loose asset roots and selected archives resolve predictably."""
from __future__ import annotations

import io
from pathlib import Path
import struct

import numpy as np
from PIL import Image

from cwr_island_studio.asset_sources import append_unique, remove_selected, roots
from cwr_island_studio.model_bounds import ModelBounds, load_model_bounds
from cwr_island_studio.pbo_textures import PboTextures
from cwr_island_studio.textures import (
    clear_archive_texture_cache, clear_loose_texture_cache,
    render_real_textures,
)


def _png(rgb: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (8, 8), rgb)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _model(width: float) -> bytes:
    positions = ((0.0, 0.0, -2.0), (width, 0.0, 6.0))
    return (
        b"MLOD"
        + struct.pack("<II", 1, 1)
        + b"SP3X"
        + struct.pack("<6I", 28, 0, len(positions), 0, 0, 0)
        + b"".join(struct.pack("<fffI", *xyz, 0) for xyz in positions)
    )


def _pbo(path: Path, members: tuple[tuple[str, bytes], ...]) -> None:
    stream = io.BytesIO()
    for name, content in members:
        stream.write(name.encode("ascii") + b"\0")
        stream.write(struct.pack("<IIIII", 0, 0, 0, 0, len(content)))
    stream.write(b"\0" + struct.pack("<IIIII", 0, 0, 0, 0, 0))
    for _, content in members:
        stream.write(content)
    path.write_bytes(stream.getvalue())


def _render(
    directories: tuple[Path, ...],
    archives: PboTextures | None = None,
) -> tuple[list[int], list[int], int]:
    slots = np.array([[0, 1]], dtype=np.int32)
    background = np.full((1, 2, 3), 17, dtype=np.uint8)
    image, count = render_real_textures(
        slots, np.array([0.0, 50.0]), np.array([0.0]),
        (r"o\data\grass.png", r"o\data\missing.png"),
        directories, background, 50.0, archives=archives,
    )
    return image[0, 0].tolist(), image[0, 1].tolist(), count


def test_add_and_remove_sources_preserves_order_and_skips_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    third = tmp_path / "three"
    chosen = append_unique((first,), (second, first, third, second))
    assert chosen == (first.resolve(), second.resolve(), third.resolve())
    assert append_unique(chosen, (first, second)) == chosen
    assert remove_selected(chosen, [1]) == (first.resolve(), third.resolve())
    assert remove_selected(chosen, [0, 2]) == (second.resolve(),)
    assert roots(first) == (first.resolve(),)
    assert roots(None) == ()


def test_multiple_texture_folders_and_ordered_override(tmp_path: Path) -> None:
    one, two, three = (tmp_path / name for name in ("one", "two", "three"))
    for folder in (one, two, three):
        (folder / "o" / "data").mkdir(parents=True)
    (one / "o" / "data" / "grass.png").write_bytes(_png((1, 20, 30)))
    (three / "o" / "data" / "grass.png").write_bytes(_png((201, 210, 220)))

    try:
        assert _render((one, two))[0] == [1, 20, 30]
        assert _render((one, two, three))[0] == [201, 210, 220]
        assert _render((one, two, three))[1] == [17, 17, 17]
        assert _render((one, two, three))[2] == 1
        assert _render((one, two))[0] == [1, 20, 30]
    finally:
        clear_loose_texture_cache()


def test_multiple_model_folders_share_priority_and_fallback(tmp_path: Path) -> None:
    one, two, three = (tmp_path / n for n in ("a", "b", "c"))
    for folder in (one, two, three):
        (folder / "o" / "buildings").mkdir(parents=True)
    (one / "o" / "buildings" / "shed.p3d").write_bytes(_model(12.0))
    (three / "o" / "buildings" / "shed.p3d").write_bytes(_model(36.0))
    name = r"o\buildings\shed.p3d"
    assert load_model_bounds(name, (one, two), None) == ModelBounds(0, 12, -2, 6)
    assert load_model_bounds(name, (one, two, three), None) == ModelBounds(0, 36, -2, 6)
    assert load_model_bounds(name, (one, two), None) == ModelBounds(0, 12, -2, 6)


def test_added_and_removed_pbos_rebuild_virtual_asset_search(tmp_path: Path) -> None:
    older, newer = tmp_path / "o.pbo", tmp_path / "new.pbo"
    grass_old, grass_new = _png((11, 22, 33)), _png((101, 122, 133))
    _pbo(older, (
        (r"data\grass.png", grass_old),
        (r"buildings\shed.p3d", _model(12.0)),
    ))
    # A PBO named new has a different virtual namespace; give it the
    # original namespace via prefix metadata so it overrides the earlier one.
    with newer.open("wb") as stream:
        stream.write(b"\0" + struct.pack("<IIIII", 0x56657273, 0, 0, 0, 0))
        stream.write(b"prefix\0o\0\0")
        for name, body in ((r"data\grass.png", grass_new),
                           (r"buildings\shed.p3d", _model(30.0))):
            stream.write(name.encode("ascii") + b"\0")
            stream.write(struct.pack("<IIIII", 0, 0, 0, 0, len(body)))
        stream.write(b"\0" + struct.pack("<IIIII", 0, 0, 0, 0, 0))
        stream.write(grass_new)
        stream.write(_model(30.0))

    first = PboTextures((older,))
    try:
        assert _render((), first)[0] == [11, 22, 33]
        assert load_model_bounds(r"o\buildings\shed.p3d", (), first) == ModelBounds(0, 12, -2, 6)
    finally:
        clear_archive_texture_cache()
        first.close()

    both = PboTextures((older, newer))
    try:
        assert _render((), both)[0] == [101, 122, 133]
        assert load_model_bounds(r"o\buildings\shed.p3d", (), both) == ModelBounds(0, 30, -2, 6)
    finally:
        clear_archive_texture_cache()
        both.close()

    # After removing the overriding archive, the original becomes visible.
    remaining = PboTextures(remove_selected((older, newer), (1,)))
    try:
        assert _render((), remaining)[0] == [11, 22, 33]
    finally:
        clear_archive_texture_cache()
        remaining.close()


def test_loose_sources_have_precedence_over_selected_archives(tmp_path: Path) -> None:
    archive = tmp_path / "o.pbo"
    _pbo(archive, ((r"data\grass.png", _png((25, 25, 25))),
                   (r"buildings\shed.p3d", _model(10.0))))
    folder = tmp_path / "loose"
    (folder / "o" / "data").mkdir(parents=True)
    (folder / "o" / "buildings").mkdir(parents=True)
    (folder / "o" / "data" / "grass.png").write_bytes(_png((99, 99, 99)))
    (folder / "o" / "buildings" / "shed.p3d").write_bytes(_model(42.0))
    archives = PboTextures((archive,))
    try:
        assert _render((folder,), archives)[0] == [99, 99, 99]
        assert load_model_bounds(r"o\buildings\shed.p3d", (folder,), archives) == ModelBounds(0, 42, -2, 6)
        assert _render((), archives)[0] == [25, 25, 25]
    finally:
        clear_archive_texture_cache()
        clear_loose_texture_cache()
        archives.close()
