from __future__ import annotations

from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cwr_worldgen.paa import write_rgb_dxt1_paa, write_solid_dxt1_paa
from cwr_worldgen.runway_exact_background_policy import (
    _canonical,
    _extract_pbo_asset,
    _load_exact_texture,
    _parse_dxt1_paa,
    _write_with_preserved_background,
    decode_dxt1_paa,
    install_runway_exact_background_policy,
)
from cwr_worldgen import runway_surface_policy as runway


def _spec(profile: str, *, asset_roots=(), deploy_mod_dir=None):
    return SimpleNamespace(
        name="wg_runway",
        ground_texture_profile=profile,
        asset_roots=tuple(asset_roots),
        deploy_mod_dir=deploy_mod_dir,
    )


def _write_test_texture(path: Path, colour=(42, 67, 31), size=128) -> bytes:
    write_solid_dxt1_paa(path, width=size, height=size, colour=colour)
    return path.read_bytes()


def _write_uncompressed_pbo(path: Path, entry_name: str, data: bytes) -> None:
    fields = struct.Struct("<IIIII")
    path.parent.mkdir(parents=True, exist_ok=True)
    output = bytearray()
    output += entry_name.encode("ascii") + b"\0"
    output += fields.pack(0, len(data), 0, 0, len(data))
    output += b"\0" + fields.pack(0, 0, 0, 0, 0)
    output += data
    path.write_bytes(bytes(output))


def test_dxt1_decoder_round_trips_generated_paa(tmp_path) -> None:
    source = tmp_path / "grass.paa"
    _write_test_texture(source, (41, 72, 33))
    image = decode_dxt1_paa(source.read_bytes())
    assert image.size == (128, 128)
    sample = np.asarray(image)[20, 20].astype(int)
    assert np.max(np.abs(sample - np.asarray((41, 72, 33)))) <= 8


def test_exact_loader_reads_world_local_generated_and_malden_textures(tmp_path) -> None:
    for profile in ("generated", "malden"):
        source_dir = tmp_path / profile / "wg_runway"
        local = source_dir / "data" / "g.paa"
        _write_test_texture(local, (51, 78, 36))
        exact = _load_exact_texture(
            source_dir,
            _spec(profile),
            r"wg_runway\data\g.paa",
        )
        assert exact is not None
        assert exact.top_image.size == (128, 128)
        assert Path(exact.source) == local


@pytest.mark.parametrize(
    ("profile", "wire_path"),
    (
        ("nogova", r"o\t1.paa"),
        ("everon", r"eden\zbh.paa"),
        ("desert", r"o\ps.paa"),
    ),
)
def test_exact_loader_skips_external_asset_roots_while_texture_scanning_is_disabled(
    tmp_path, profile: str, wire_path: str
) -> None:
    root = tmp_path / "game"
    local = root.joinpath(*wire_path.split("\\"))
    _write_test_texture(local, (37, 59, 29))

    exact = _load_exact_texture(root / "unused-world", _spec(profile, asset_roots=(root,)), wire_path)

    assert exact is None


def test_exact_loader_skips_game_pbo_discovery_while_texture_scanning_is_disabled(tmp_path) -> None:
    game = tmp_path / "CWA"
    deploy = game / "@generated"
    deploy.mkdir(parents=True)
    texture_file = tmp_path / "source.paa"
    texture = _write_test_texture(texture_file, (33, 52, 25))
    pbo = game / "Res" / "AddOns" / "O.pbo"
    _write_uncompressed_pbo(pbo, "t1.paa", texture)

    exact = _load_exact_texture(
        tmp_path / "world",
        _spec("nogova", deploy_mod_dir=deploy),
        r"o\t1.paa",
    )

    assert exact is None


def test_pbo_asset_reader_respects_pbo_name_as_virtual_prefix(tmp_path) -> None:
    texture_file = tmp_path / "texture.paa"
    texture = _write_test_texture(texture_file)
    pbo = tmp_path / "O.pbo"
    _write_uncompressed_pbo(pbo, "trava2.paa", texture)
    assert _extract_pbo_asset(pbo, r"o\trava2.paa") == texture
    assert _extract_pbo_asset(pbo, r"o\trava3.paa") is None


def test_unchanged_edge_dxt1_blocks_are_copied_verbatim(tmp_path) -> None:
    size = 128
    yy, xx = np.mgrid[0:size, 0:size]
    pixels = np.empty((size, size, 3), dtype=np.uint8)
    pixels[:, :, 0] = 25 + (xx % 37)
    pixels[:, :, 1] = 45 + (yy % 51)
    pixels[:, :, 2] = 20 + ((xx + yy) % 29)
    base_image = Image.fromarray(pixels, mode="RGB")
    source_path = tmp_path / "base.paa"
    write_rgb_dxt1_paa(source_path, base_image)
    source_data = source_path.read_bytes()
    source_mips = _parse_dxt1_paa(source_data)
    source_top = decode_dxt1_paa(source_data)
    exact = SimpleNamespace(
        wire_path=r"o\t1.paa",
        source=str(source_path),
        data=source_data,
        mips=source_mips,
        top_image=source_top,
    )

    modified = source_top.copy()
    array = np.asarray(modified).copy()
    array[40:88, 54:74] = (112, 113, 108)
    modified = Image.fromarray(array, mode="RGB")
    output = tmp_path / "runway.paa"

    # The writer normally receives its fallback through policy installation.
    # This test exercises the exact path, so provide the normal writer only for
    # the defensive size-mismatch branch that is not taken here.
    import cwr_worldgen.runway_exact_background_policy as exact_policy
    old = exact_policy._ORIGINAL_WRITE_RUNWAY_PAA
    exact_policy._ORIGINAL_WRITE_RUNWAY_PAA = write_rgb_dxt1_paa
    try:
        _write_with_preserved_background(output, modified, exact)
    finally:
        exact_policy._ORIGINAL_WRITE_RUNWAY_PAA = old

    output_mips = _parse_dxt1_paa(output.read_bytes())
    # Top-left 4x4 block is nowhere near the runway and must remain literally the
    # stock compressed block. A central block intersects the runway and must not.
    assert output_mips[0].payload[:8] == source_mips[0].payload[:8]
    blocks_x = size // 4
    central_offset = ((16 * blocks_x) + 16) * 8
    assert (
        output_mips[0].payload[central_offset : central_offset + 8]
        != source_mips[0].payload[central_offset : central_offset + 8]
    )


def test_exact_policy_bumps_runway_cache_after_nogova_calibration() -> None:
    install_runway_exact_background_policy()
    assert runway._SURFACE_CACHE_V18 == "surface-pipeline-v20-exact-runway-background-textures"
