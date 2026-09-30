from __future__ import annotations

import io
import json
import math
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import zstandard

from cwr_worldgen import assets as asset_module
from cwr_worldgen import cli
from cwr_worldgen import generator
from cwr_worldgen import paved_junction_policy
from cwr_worldgen import paved_junction_fallback_policy as junction_fallback
from cwr_worldgen import paved_road_generated_fallback_policy as fallback
from cwr_worldgen import playability
from cwr_worldgen import procedural_infrastructure as infrastructure
from cwr_worldgen import road_chain_parallel_policy
from cwr_worldgen import road_quality_policy
from cwr_worldgen import gravel_junction_policy
from cwr_worldgen.assets import model_texture_dependencies, scan_assets
from cwr_worldgen.osm import (
    BboxProjection,
    OsmDataset,
    OsmLineFeature,
    road_model_for_tags,
)
from cwr_worldgen.pbo import PboEntry, write_pbo
from cwr_worldgen.procedural_buildings import (
    _Face,
    _Lod,
    _MLOD_HEADER,
    _write_lod,
    inspect_mlod,
)
from cwr_worldgen.gui import (
    WorldgenGui,
    build_milestone9_command,
    default_gui_values,
    defaults_with_recent_source,
)
from cwr_worldgen.source_pipeline import Milestone5Spec
from cwr_worldgen.milestone6 import Milestone6Spec
from cwr_worldgen.milestone7 import Milestone7Spec
from cwr_worldgen.milestone8 import Milestone8Spec
from cwr_worldgen.milestone9 import Milestone9Spec, _Milestone9PlayabilitySpec
from cwr_worldgen.legacy_proxy_models import inspect_visual_model_dimensions


def test_missing_asset_pbo_is_rejected_before_world_generation(tmp_path: Path) -> None:
    missing = tmp_path / "missing-roads.pbo"
    spec = SimpleNamespace(asset_roots=(missing,))

    with pytest.raises(ValueError, match="asset root does not exist"):
        generator._validate_asset_roots_exist(spec)


def _write_fake_mod_asset(root: Path, relative: str, payload: bytes) -> Path:
    path = root / relative.replace("\\", "/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _literal_lzss(payload: bytes) -> bytes:
    """Encode a valid BIS LZSS stream using literals only for test fixtures."""
    packed = bytearray()
    for offset in range(0, len(payload), 8):
        chunk = payload[offset:offset + 8]
        packed.append((1 << len(chunk)) - 1)
        packed.extend(chunk)
    packed.extend(struct.pack("<I", sum(payload) & 0xFFFFFFFF))
    return bytes(packed)


def _write_compressed_member_pbo(
    path: Path,
    member: str,
    payload: bytes,
) -> None:
    stored = _literal_lzss(payload)
    header = bytearray()
    header.extend(member.replace("/", "\\").encode("ascii") + b"\0")
    header.extend(
        struct.pack(
            "<IIIII",
            asset_module._PBO_COMPRESSED,
            len(payload),
            0,
            0,
            len(stored),
        )
    )
    header.extend(b"\0" + struct.pack("<IIIII", 0, 0, 0, 0, 0))
    path.write_bytes(bytes(header) + stored)


def _mlod_road(
    width: float,
    length: float,
    texture: str,
) -> bytes:
    half_width = width * 0.5
    half_length = length * 0.5
    lod = _Lod(
        (
            (-half_width, 0.025, -half_length),
            (-half_width, 0.025, half_length),
            (half_width, 0.025, half_length),
            (half_width, 0.025, -half_length),
        ),
        ((0.0, -1.0, 0.0),),
        (
            _Face(
                texture,
                (
                    (0, 0, 0.0, 0.0),
                    (1, 0, 0.0, 1.0),
                    (2, 0, 1.0, 1.0),
                    (3, 0, 1.0, 0.0),
                ),
                0,
            ),
        ),
        1.0,
        properties=(("autocenter", "0"), ("class", "road"), ("map", "road")),
        point_flags=(0x13F,) * 4,
    )
    stream = io.BytesIO()
    stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, 1))
    _write_lod(stream, lod)
    return stream.getvalue()


def _mlod_road_with_midspan_detail(
    width: float,
    length: float,
    visual_width: float,
    texture: str,
) -> bytes:
    """Road mouth width stays narrow while decorative mid-span mesh is wider."""
    half_width = width * 0.5
    half_visual_width = visual_width * 0.5
    half_length = length * 0.5
    detail_half_length = min(1.0, length * 0.10)
    lod = _Lod(
        (
            (-half_width, 0.025, -half_length),
            (-half_width, 0.025, half_length),
            (half_width, 0.025, half_length),
            (half_width, 0.025, -half_length),
            (-half_visual_width, 0.035, -detail_half_length),
            (-half_visual_width, 0.035, detail_half_length),
            (half_visual_width, 0.035, detail_half_length),
            (half_visual_width, 0.035, -detail_half_length),
        ),
        ((0.0, -1.0, 0.0),),
        (
            _Face(
                texture,
                (
                    (0, 0, 0.0, 0.0),
                    (1, 0, 0.0, 1.0),
                    (2, 0, 1.0, 1.0),
                    (3, 0, 1.0, 0.0),
                ),
                0,
            ),
            _Face(
                texture,
                (
                    (4, 0, 0.0, 0.0),
                    (5, 0, 0.0, 1.0),
                    (6, 0, 1.0, 1.0),
                    (7, 0, 1.0, 0.0),
                ),
                0,
            ),
        ),
        1.0,
        properties=(("autocenter", "0"), ("class", "road"), ("map", "road")),
        point_flags=(0x13F,) * 8,
    )
    stream = io.BytesIO()
    stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, 1))
    _write_lod(stream, lod)
    return stream.getvalue()


def _mlod_curve_sample(
    width: float,
    length: float,
    lateral_shift: float,
    texture: str,
) -> bytes:
    half_width = width * 0.5
    half_length = length * 0.5
    lower_center = -lateral_shift * 0.5
    upper_center = lateral_shift * 0.5
    lod = _Lod(
        (
            (lower_center - half_width, 0.025, -half_length),
            (upper_center - half_width, 0.025, half_length),
            (upper_center + half_width, 0.025, half_length),
            (lower_center + half_width, 0.025, -half_length),
        ),
        ((0.0, -1.0, 0.0),),
        (
            _Face(
                texture,
                (
                    (0, 0, 0.0, 0.0),
                    (1, 0, 0.0, 1.0),
                    (2, 0, 1.0, 1.0),
                    (3, 0, 1.0, 0.0),
                ),
                0,
            ),
        ),
        1.0,
        properties=(("autocenter", "0"), ("class", "road"), ("map", "road")),
        point_flags=(0x13F,) * 4,
    )
    stream = io.BytesIO()
    stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, 1))
    _write_lod(stream, lod)
    return stream.getvalue()


def _mlod_skewed_road(
    width: float,
    chord_length: float,
    lateral_shift: float,
    texture: str,
) -> bytes:
    half_width = width * 0.5
    half_length = chord_length * 0.5
    half_shift = lateral_shift * 0.5
    lod = _Lod(
        (
            (-half_width - half_shift, 0.025, -half_length),
            (half_width - half_shift, 0.025, -half_length),
            (half_width + half_shift, 0.025, half_length),
            (-half_width + half_shift, 0.025, half_length),
        ),
        ((0.0, -1.0, 0.0),),
        (
            _Face(
                texture,
                (
                    (0, 0, 0.0, 0.0),
                    (1, 0, 1.0, 0.0),
                    (2, 0, 1.0, 1.0),
                    (3, 0, 0.0, 1.0),
                ),
                0,
            ),
        ),
        1.0,
        properties=(("autocenter", "0"), ("class", "road"), ("map", "road")),
        point_flags=(0x13F,) * 4,
    )
    stream = io.BytesIO()
    stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, 1))
    _write_lod(stream, lod)
    return stream.getvalue()


def _mlod_divided_curved_highway(
    texture: str,
) -> bytes:
    """Synthetic AGS-style divided highway with a rotated far connector mouth."""

    start_center = (0.0, -20.0)
    end_center = (6.0, 12.0)
    end_cross = (0.8191520443, -0.5735764364)

    def point(
        center: tuple[float, float],
        cross: tuple[float, float],
        offset: float,
    ) -> tuple[float, float, float]:
        return (
            center[0] + cross[0] * offset,
            0.025,
            center[1] + cross[1] * offset,
        )

    # Two 12 m carriageways separated by a 4 m median: total mouth width 28 m.
    points = (
        point(start_center, (1.0, 0.0), -14.0),
        point(start_center, (1.0, 0.0), -2.0),
        point(end_center, end_cross, -2.0),
        point(end_center, end_cross, -14.0),
        point(start_center, (1.0, 0.0), 2.0),
        point(start_center, (1.0, 0.0), 14.0),
        point(end_center, end_cross, 14.0),
        point(end_center, end_cross, 2.0),
    )
    lod = _Lod(
        points,
        ((0.0, -1.0, 0.0),),
        (
            _Face(
                texture,
                (
                    (0, 0, 0.0, 0.0),
                    (1, 0, 1.0, 0.0),
                    (2, 0, 1.0, 1.0),
                    (3, 0, 0.0, 1.0),
                ),
                0,
            ),
            _Face(
                texture,
                (
                    (4, 0, 0.0, 0.0),
                    (5, 0, 1.0, 0.0),
                    (6, 0, 1.0, 1.0),
                    (7, 0, 0.0, 1.0),
                ),
                0,
            ),
        ),
        1.0,
        properties=(("autocenter", "0"), ("class", "road"), ("map", "road")),
        point_flags=(0x13F,) * len(points),
    )
    stream = io.BytesIO()
    stream.write(_MLOD_HEADER.pack(b"MLOD", 1, 1, 0, 1))
    _write_lod(stream, lod)
    return stream.getvalue()


def test_modded_road_texture_is_discovered_from_donor_p3d(tmp_path: Path) -> None:
    root = tmp_path / "mod"
    _write_fake_mod_asset(
        root,
        r"myroads\asphalt25.p3d",
        b"MLOD synthetic myroads\\textures\\asphalt_main.paa\x00",
    )
    _write_fake_mod_asset(
        root,
        r"myroads\textures\asphalt_main.paa",
        b"synthetic-paa",
    )

    scan = scan_assets(
        (root,),
        (r"myroads\asphalt25.p3d",),
        use_cache=False,
    )
    dependencies = model_texture_dependencies(
        scan.records,
        r"myroads\asphalt25.p3d",
    )

    assert dependencies == (r"myroads\textures\asphalt_main.paa",)
    assert generator._preferred_road_texture(
        r"myroads\asphalt25.p3d",
        dependencies,
    ) == r"myroads\textures\asphalt_main.paa"


def test_divided_curved_highway_uses_rotated_connector_mouth_width(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mod"
    texture = r"ags_roads\auto_1.paa"
    straight = r"ags_roads\hway75.p3d"
    curve = r"ags_roads\hway50c2.p3d"
    _write_fake_mod_asset(root, straight, _mlod_road(28.0, 75.0, texture))
    _write_fake_mod_asset(root, curve, _mlod_divided_curved_highway(texture))
    _write_fake_mod_asset(root, texture, b"synthetic-paa")

    curve_info = inspect_visual_model_dimensions(
        (root / curve.replace("\\", "/")).read_bytes()
    )
    assert curve_info.width_metres > 30.0
    assert curve_info.connector_width_metres == pytest.approx(28.0, abs=0.02)

    spec = SimpleNamespace(
        paved_road_model=straight,
        paved_road_curve_model=curve,
        gravel_road_model="",
        gravel_road_curve_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        dirt_road_curve_model="",
        road_segment_length=75.0,
        asset_roots=(root,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )
    dimensions = generator._modded_road_model_dimensions(spec, {})
    assert dimensions[playability._road_model_key(straight)][0] == pytest.approx(28.0)
    assert dimensions[playability._road_model_key(curve)][0] == pytest.approx(28.0)


def test_custom_paved_highway_width_and_axis_survive_junction_stitching() -> None:
    model = infrastructure.custom_road_model_path(
        "agsworld",
        "paved",
        width_metres=28.0,
        length_metres=75.0,
        curve_degrees=44.0,
    )
    assert model.endswith(("road_paved_w280_l0750_l044.p3d", "road_paved_w280_l0750_r044.p3d"))
    assert playability._generated_paved_half_width(model) == pytest.approx(14.0)

    obj = playability.WorldObject(1, model, 100.0, 0.0, 200.0, 0.0)
    axis = junction_fallback._generated_paved_axis(obj, SimpleNamespace())
    assert axis is not None
    assert math.dist(*axis) == pytest.approx(75.0)


def test_wide_generated_paved_junction_keeps_mod_texture(tmp_path: Path) -> None:
    texture = r"ags_roads\auto_1.paa"
    model = infrastructure.paved_junction_signature_model_path(
        "agsworld",
        28.0,
        (0, 90, 180),
    )
    library = infrastructure.ProceduralInfrastructureLibrary(
        "agsworld",
        paved_texture_path=texture,
        cache_enabled=False,
    )
    library.register_model(model)
    library.write_assets(tmp_path, tmp_path / "infrastructure.json")

    relative = model.split("\\", 1)[1].replace("\\", "/")
    summary = inspect_mlod(tmp_path / relative)
    textures = {value.casefold() for value in summary.texture_paths}
    assert texture.casefold() in textures
    assert r"o\road\sil_new.paa" not in textures
    assert r"o\road\sil_konec.paa" not in textures


@pytest.mark.parametrize(
    ("surface", "width", "texture"),
    (
        ("paved", 5.2, r"modroads\paved.paa"),
        ("gravel", 4.6, r"modroads\gravel.paa"),
        ("dirt", 3.8, r"modroads\dirt.paa"),
    ),
)
def test_measured_custom_donor_builds_matching_three_way_junction_plan(
    surface: str,
    width: float,
    texture: str,
) -> None:
    donors = {
        "paved": r"modroads\paved25.p3d",
        "gravel": r"modroads\gravel25.p3d",
        "dirt": r"modroads\dirt25.p3d",
    }
    spec = SimpleNamespace(
        name="donorworld",
        paved_road_model=donors["paved"],
        gravel_road_model=donors["gravel"],
        dirt_road_model=donors["dirt"],
        road_segment_length=25.0,
        custom_road_shapes=True,
    )
    model = donors[surface]
    values = (
        ((0.0, 1.0), surface == "dirt", model, "north", "way-a"),
        ((0.0, -1.0), surface == "dirt", model, "south", "way-a"),
        ((1.0, 0.0), surface == "dirt", model, "east", "way-b"),
    )
    token = playability._ROAD_MODEL_DIMENSIONS.set(
        {playability._road_model_key(model): (width, 25.0)}
    )
    try:
        plan = playability._generated_custom_road_junction_cap_plan(values, spec)
    finally:
        playability._ROAD_MODEL_DIMENSIONS.reset(token)

    assert plan is not None
    model_path, axis = plan
    signature = infrastructure.custom_road_junction_signature(model_path)
    assert signature is not None
    assert signature[0] == surface
    assert signature[1] == pytest.approx(width)
    assert len(signature[2]) == 3
    assert 0 in signature[2]
    assert math.hypot(*axis) == pytest.approx(1.0)


def test_parallel_fitter_emits_custom_donor_junction_cap() -> None:
    bbox = (59.40, 16.82, 59.41, 16.83)
    projection = BboxProjection.create(bbox, 1000.0)
    centre = (500.0, 500.0)
    donor = r"modroads\paved25.p3d"
    dataset = OsmDataset(
        source_generator="custom-donor-junction",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(
            OsmLineFeature(
                "way/main",
                {"highway": "residential", "surface": "asphalt"},
                tuple(
                    projection.to_latlon(point)
                    for point in ((500.0, 250.0), centre, (500.0, 750.0))
                ),
            ),
            OsmLineFeature(
                "way/branch",
                {"highway": "residential", "surface": "asphalt"},
                tuple(
                    projection.to_latlon(point)
                    for point in (centre, (750.0, 500.0))
                ),
            ),
        ),
    )
    spec = _Milestone9PlayabilitySpec(
        name="donorworld",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=40,
        cell_size=25.0,
        max_road_objects=10000,
        strict_assets=False,
        paved_road_model=donor,
    )
    donor_key = playability._road_model_key(donor)
    dimensions_token = playability._ROAD_MODEL_DIMENSIONS.set(
        {donor_key: (5.2, 25.0)}
    )
    availability_token = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(
        {donor_key: frozenset({donor_key})}
    )
    effective_token = playability._ROAD_MODEL_EFFECTIVE_DONORS.set(
        {donor_key: donor}
    )
    try:
        report = road_chain_parallel_policy._fit_stock_piece_road_objects_parallel(
            dataset,
            projection,
            [0.0] * (spec.cells * spec.cells),
            spec,
        )
    finally:
        playability._ROAD_MODEL_EFFECTIVE_DONORS.reset(effective_token)
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(availability_token)
        playability._ROAD_MODEL_DIMENSIONS.reset(dimensions_token)

    assert report.junction_cap_objects == 1
    cap = report.objects[0]
    signature = infrastructure.custom_road_junction_signature(cap.model_path)
    assert signature is not None
    assert signature[0] == "paved"
    assert signature[1] == pytest.approx(5.2)
    assert len(signature[2]) == 3
    assert cap.model_path.startswith(r"donorworld\i\road_j3_paved_w052_")


def test_mixed_modded_paved_gravel_t_has_no_open_connector_gap() -> None:
    bbox = (59.40, 16.82, 59.41, 16.83)
    projection = BboxProjection.create(bbox, 1000.0)
    centre = (500.0, 500.0)
    paved = r"bas_o\_road\bas_asf25.p3d"
    dataset = OsmDataset(
        source_generator="mixed-modded-junction",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(
            OsmLineFeature(
                "way/paved-main",
                {"highway": "residential", "surface": "asphalt"},
                tuple(
                    projection.to_latlon(point)
                    for point in ((500.0, 250.0), centre, (500.0, 750.0))
                ),
            ),
            OsmLineFeature(
                "way/gravel-branch",
                {"highway": "track", "surface": "gravel"},
                tuple(
                    projection.to_latlon(point)
                    for point in (centre, (750.0, 500.0))
                ),
            ),
        ),
    )
    spec = _Milestone9PlayabilitySpec(
        name="mixedworld",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=40,
        cell_size=25.0,
        max_road_objects=10000,
        strict_assets=False,
        paved_road_model=paved,
    )
    paved_key = playability._road_model_key(paved)
    dimensions_token = playability._ROAD_MODEL_DIMENSIONS.set(
        {paved_key: (7.0, 25.0)}
    )
    availability_token = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(
        {paved_key: frozenset({paved_key})}
    )
    effective_token = playability._ROAD_MODEL_EFFECTIVE_DONORS.set(
        {paved_key: paved}
    )
    try:
        report = road_chain_parallel_policy._fit_stock_piece_road_objects_parallel(
            dataset,
            projection,
            [0.0] * (spec.cells * spec.cells),
            spec,
        )
    finally:
        playability._ROAD_MODEL_EFFECTIVE_DONORS.reset(effective_token)
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(availability_token)
        playability._ROAD_MODEL_DIMENSIONS.reset(dimensions_token)

    assert report.junction_cap_objects == 1
    cap = report.objects[0]
    junction = infrastructure.custom_road_junction_signature(cap.model_path)
    assert junction is not None
    assert junction[0] == "paved"
    assert junction[1] == pytest.approx(7.0)
    assert len(junction[2]) == 3

    yaw = math.radians(cap.heading_degrees)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    arm_extent = infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES

    connectors = []
    for heading in junction[2]:
        radians = math.radians(float(heading))
        local_x = math.sin(radians) * arm_extent
        local_z = math.cos(radians) * arm_extent
        connectors.append((
            float(cap.x) + local_x * cosine + local_z * sine,
            float(cap.z) - local_x * sine + local_z * cosine,
        ))

    approach_endpoints = []
    endpoint_details = []
    surfaces = set()
    for obj in report.objects[report.junction_cap_objects:]:
        signature = infrastructure.custom_road_model_signature(obj.model_path)
        surface = playability.road_model_surface(spec, obj.model_path)
        if surface is None:
            continue
        surfaces.add(surface)
        if signature is not None:
            length = float(signature[2])
        else:
            measured = playability.road_model_dimensions(obj.model_path)
            length = (
                float(measured[1])
                if measured is not None
                else road_quality_policy._piece_length(
                    obj.model_path,
                    spec.road_segment_length,
                )
            )
        radians = math.radians(float(obj.heading_degrees))
        dx = math.sin(radians) * length * 0.5
        dz = math.cos(radians) * length * 0.5
        object_endpoints = (
            (float(obj.x) - dx, float(obj.z) - dz),
            (float(obj.x) + dx, float(obj.z) + dz),
        )
        approach_endpoints.extend(object_endpoints)
        endpoint_details.extend(
            (
                endpoint,
                surface,
                obj.model_path,
                int(obj.object_id),
            )
            for endpoint in object_endpoints
        )

    assert {"paved", "gravel"} <= surfaces
    nearest_details = [
        min(
            (
                math.dist(connector, endpoint),
                surface,
                model_path,
                object_id,
                endpoint,
            )
            for endpoint, surface, model_path, object_id in endpoint_details
        )
        for connector in connectors
    ]
    connector_gaps = [value[0] for value in nearest_details]
    assert max(connector_gaps) <= (
        infrastructure.GENERATED_PAVED_JUNCTION_APPROACH_CLEARANCE_METRES + 0.08
    ), {
        "connector_gaps": connector_gaps,
        "nearest": nearest_details,
        "connectors": connectors,
        "objects": [
            (
                obj.model_path,
                round(float(obj.x), 3),
                round(float(obj.z), 3),
                round(float(obj.heading_degrees), 3),
            )
            for obj in report.objects
        ],
    }


def test_custom_gravel_donor_junction_keeps_directional_arm_reach() -> None:
    arm_extent = infrastructure.GENERATED_PAVED_JUNCTION_ARM_EXTENT_METRES
    junction = road_quality_policy._Junction(
        point=(0.0, 0.0),
        axis=(0.0, 1.0),
        half_length=arm_extent,
        half_width=infrastructure.GENERATED_GRAVEL_HALF_WIDTH_METRES,
        directions=((0.0, 1.0), (0.0, -1.0), (1.0, 0.0)),
        directional_exit_distances=(
            ((0.0, 1.0), arm_extent),
            ((0.0, -1.0), arm_extent),
            ((1.0, 0.0), arm_extent),
        ),
    )

    assert road_quality_policy._exit_distance(
        junction, (1.0, 0.0)
    ) == pytest.approx(arm_extent)
    assert gravel_junction_policy._is_gravel_junction(junction) is False


@pytest.mark.parametrize(
    ("surface", "source_model", "expected_width"),
    (
        ("paved", r"o\road\sil25.p3d", 9.10),
        ("dirt", r"o\road\ces25.p3d", 3.50),
        ("gravel", r"unified\i\gravel25.p3d", 4.60),
    ),
)
def test_unified_shape_mode_converts_straight_roads_to_generated_ribbons(
    surface: str,
    source_model: str,
    expected_width: float,
) -> None:
    spec = SimpleNamespace(
        name="unified",
        paved_road_model=r"o\road\sil25.p3d",
        gravel_road_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
        custom_road_shapes=True,
        procedural_paved_road_fallback=True,
    )
    pieces = playability.road_model_variants(
        source_model,
        25.0,
        donor_only=True,
    )
    piece = next(
        (value for value in pieces if value.nominal_length == 6),
        pieces[-1],
    )
    measure = playability._PolylineMeasure.create(
        ((0.0, 0.0), (0.0, float(piece.length_metres)))
    )
    result = ((piece, (0.0, 0.0), (0.0, float(piece.length_metres))),)
    token = road_quality_policy._CONTEXT.set(
        road_quality_policy._Context((), spec, {})
    )
    try:
        upgraded = fallback._upgrade_stock_result(
            result,
            measure,
            pieces,
            start_distance=0.0,
            preferred_end_distance=measure.total,
            minimum_end_distance=0.0,
            maximum_end_distance=measure.total,
        )
    finally:
        road_quality_policy._CONTEXT.reset(token)

    assert len(upgraded) == 1
    signature = infrastructure.custom_road_model_signature(
        upgraded[0][0].model_path
    )
    assert signature is not None
    assert signature[0] == surface
    assert signature[1] == pytest.approx(expected_width)
    assert signature[3] == 0


@pytest.mark.parametrize(
    ("surface", "model", "expected_width"),
    (
        ("paved", r"o\road\sil25.p3d", 9.10),
        ("dirt", r"o\road\ces25.p3d", 3.50),
        ("gravel", r"unified\i\gravel25.p3d", 4.60),
    ),
)
def test_unified_shape_mode_builds_generated_junctions_for_stock_surfaces(
    surface: str,
    model: str,
    expected_width: float,
) -> None:
    spec = SimpleNamespace(
        name="unified",
        paved_road_model=r"o\road\sil25.p3d",
        gravel_road_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
        custom_road_shapes=True,
    )
    values = (
        ((0.0, 1.0), surface == "dirt", model, "north", "way-a"),
        ((0.0, -1.0), surface == "dirt", model, "south", "way-a"),
        ((1.0, 0.0), surface == "dirt", model, "east", "way-b"),
    )

    plan = playability._generated_custom_road_junction_cap_plan(values, spec)

    assert plan is not None
    model_path, _axis = plan
    signature = infrastructure.custom_road_junction_signature(model_path)
    assert signature is not None
    assert signature[0] == surface
    assert signature[1] == pytest.approx(expected_width)
    assert len(signature[2]) == 3


def test_stock_paved_junction_policy_uses_unified_generated_hub() -> None:
    bbox = (59.40, 16.82, 59.41, 16.83)
    projection = BboxProjection.create(bbox, 1000.0)
    centre = (500.0, 500.0)
    dataset = OsmDataset(
        source_generator="stock-unified-junction",
        element_count=2,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(
            OsmLineFeature(
                "way/main",
                {"highway": "residential", "surface": "asphalt"},
                tuple(
                    projection.to_latlon(point)
                    for point in ((500.0, 250.0), centre, (500.0, 750.0))
                ),
            ),
            OsmLineFeature(
                "way/branch",
                {"highway": "residential", "surface": "asphalt"},
                tuple(
                    projection.to_latlon(point)
                    for point in (centre, (750.0, 500.0))
                ),
            ),
        ),
    )
    spec = _Milestone9PlayabilitySpec(
        name="unified",
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=40,
        cell_size=25.0,
        max_road_objects=10000,
        strict_assets=False,
    )

    plans = paved_junction_policy._plans(dataset, projection, spec)

    key = playability._road_node_key(centre)
    assert key in plans
    signature = infrastructure.custom_road_junction_signature(
        plans[key].model_path
    )
    assert signature is not None
    assert signature[0] == "paved"
    assert signature[1] == pytest.approx(9.10)


def test_paved_postpass_preserves_existing_unified_hub() -> None:
    model = infrastructure.custom_road_junction_model_path(
        "unified",
        "paved",
        9.10,
        (0, 90, 180),
    )
    cap = playability.WorldObject(
        1,
        model,
        100.0,
        0.06,
        100.0,
        0.0,
    )
    report = SimpleNamespace(
        objects=(cap,),
        junction_cap_objects=1,
    )
    plan = SimpleNamespace(
        model_path=model,
        point=(100.0, 100.0),
    )

    applied = paved_junction_policy._apply_plans(
        report,
        {(1000, 1000): plan},
        (),
        SimpleNamespace(),
    )

    assert applied is report
    assert applied.objects == (cap,)


def test_unified_gravel_gap_fillers_use_custom_ribbon_family() -> None:
    spec = SimpleNamespace(
        name="unified",
        gravel_road_model="",
        road_segment_length=25.0,
        custom_road_shapes=True,
    )

    piece = playability.gravel_filler_piece(spec, 6)

    signature = infrastructure.custom_road_model_signature(piece.model_path)
    assert signature is not None
    assert signature[0] == "gravel"
    assert signature[1] == pytest.approx(4.60)
    assert signature[2] == pytest.approx(6.0)


def test_stock_curve_donor_texture_fallbacks_support_unified_generation() -> None:
    assert generator._resolved_road_donor_texture(
        (),
        surface="paved",
        donor_model=r"o\road\sil10 25.p3d",
    ) == r"landtext\silnice.pac"
    assert generator._resolved_road_donor_texture(
        (),
        surface="dirt",
        donor_model=r"o\road\ces10 25.p3d",
    ) == r"o\road\ces_hned.paa"


def test_stock_paved_family_does_not_create_donor_junction_plan() -> None:
    model = r"o\road\sil25.p3d"
    spec = SimpleNamespace(
        name="stockworld",
        paved_road_model=model,
        gravel_road_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
    )
    values = (
        ((0.0, 1.0), False, model, "north", "way-a"),
        ((0.0, -1.0), False, model, "south", "way-a"),
        ((1.0, 0.0), False, model, "east", "way-b"),
    )
    token = playability._ROAD_MODEL_DIMENSIONS.set(
        {playability._road_model_key(model): (9.1, 25.0)}
    )
    try:
        assert playability._generated_custom_road_junction_cap_plan(values, spec) is None
    finally:
        playability._ROAD_MODEL_DIMENSIONS.reset(token)


@pytest.mark.parametrize(
    ("surface", "texture"),
    (
        ("paved", r"modroads\textures\paved.paa"),
        ("gravel", r"modroads\textures\gravel.paa"),
        ("dirt", r"modroads\textures\dirt.paa"),
    ),
)
def test_custom_donor_junction_model_uses_surface_donor_texture(
    tmp_path: Path,
    surface: str,
    texture: str,
) -> None:
    model = infrastructure.custom_road_junction_model_path(
        "donorworld",
        surface,
        5.2,
        (0, 90, 180),
    )
    library = infrastructure.ProceduralInfrastructureLibrary(
        "donorworld",
        paved_texture_path=(
            texture if surface == "paved" else r"o\road\sil_new.paa"
        ),
        gravel_texture_path=texture if surface == "gravel" else None,
        dirt_texture_path=texture if surface == "dirt" else None,
        cache_enabled=False,
    )
    library.register_model(model)
    library.write_assets(tmp_path, tmp_path / "infrastructure.json")

    relative = model.split("\\", 1)[1].replace("\\", "/")
    summary = inspect_mlod(tmp_path / relative)
    assert texture.casefold() in {
        value.casefold() for value in summary.texture_paths
    }
    assert any(
        math.isclose(value, infrastructure._ROADWAY_LOD, rel_tol=1e-6)
        for value in summary.resolutions
    )


def test_unified_stock_and_modded_roads_use_one_straight_donor() -> None:
    stock = r"o\road\sil25.p3d"
    modded = r"bas_o\_road\bas_asf25.p3d"
    modded_key = playability._road_model_key(modded)

    dimensions_token = playability._ROAD_MODEL_DIMENSIONS.set(
        {modded_key: (7.0, 25.0)}
    )
    availability_token = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(
        {
            modded_key: frozenset({
                modded_key,
                playability._road_model_key(r"bas_o\_road\bas_asf12.p3d"),
                playability._road_model_key(r"bas_o\_road\bas_asf6.p3d"),
            })
        }
    )
    try:
        stock_pieces = playability.road_model_variants(
            stock,
            24.5,
            donor_only=True,
        )
        modded_pieces = playability.road_model_variants(
            modded,
            24.5,
            donor_only=True,
        )
    finally:
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(availability_token)
        playability._ROAD_MODEL_DIMENSIONS.reset(dimensions_token)

    assert [piece.model_path for piece in stock_pieces] == [stock]
    assert [piece.model_path for piece in modded_pieces] == [modded]
    assert [piece.length_metres for piece in stock_pieces] == pytest.approx([25.0])
    assert [piece.length_metres for piece in modded_pieces] == pytest.approx([25.0])


def test_unified_short_run_fallback_generates_exact_ribbon() -> None:
    donor = playability._RoadPiece(
        r"unified\i\gravel25.p3d",
        25.0,
        25,
    )
    measure = playability._PolylineMeasure.create(
        ((0.0, 0.0), (6.0, 0.0))
    )

    fitted = playability._short_run_fallback_piece(
        measure,
        (donor,),
        start_trim=0.0,
        end_trim=0.0,
        generated_world_name="unified",
        generated_surface="gravel",
        generated_width_metres=4.6,
    )

    assert len(fitted) == 1
    piece, start, end = fitted[0]
    signature = infrastructure.custom_road_model_signature(piece.model_path)
    assert signature is not None
    assert signature[0] == "gravel"
    assert signature[1] == pytest.approx(4.6)
    assert signature[2] == pytest.approx(6.0)
    assert piece.model_path != donor.model_path
    assert start == pytest.approx((0.0, 0.0))
    assert end == pytest.approx((6.0, 0.0))


def test_unified_builtin_gravel_uses_single_25m_donor() -> None:
    pieces = playability.road_model_variants(
        r"unified\i\gravel25.p3d",
        24.5,
        donor_only=True,
    )
    assert [piece.model_path for piece in pieces] == [
        r"unified\i\gravel25.p3d"
    ]
    assert [piece.length_metres for piece in pieces] == pytest.approx([25.0])


def test_unified_variant_paths_ignore_stock_and_modded_siblings() -> None:
    assert playability.road_model_variant_paths(
        r"o\road\sil25.p3d",
        25.0,
        donor_only=True,
    ) == (r"o\road\sil25.p3d",)
    assert playability.road_model_variant_paths(
        r"bas_o\_road\bas_asf25.p3d",
        25.0,
        donor_only=True,
    ) == (r"bas_o\_road\bas_asf25.p3d",)


def test_modded_family_reuses_only_existing_sibling_models(tmp_path: Path) -> None:
    root = tmp_path / "mod"
    _write_fake_mod_asset(root, r"myroads\asphalt25.p3d", b"donor")
    _write_fake_mod_asset(root, r"myroads\asphalt6.p3d", b"short")
    # asphalt12.p3d is deliberately absent.

    spec = SimpleNamespace(
        paved_road_model=r"myroads\asphalt25.p3d",
        gravel_road_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
        asset_roots=(root,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )
    availability = generator._modded_road_variant_availability(spec)
    token = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(availability)
    try:
        variants = playability.road_model_variants(
            spec.paved_road_model,
            spec.road_segment_length,
        )
    finally:
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(token)

    assert [piece.model_path for piece in variants] == [
        r"myroads\asphalt25.p3d",
        r"myroads\asphalt6.p3d",
    ]


def test_modded_family_without_asset_roots_never_invents_siblings() -> None:
    spec = SimpleNamespace(
        paved_road_model=r"myroads\asphalt25.p3d",
        gravel_road_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
        asset_roots=(),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )
    availability = generator._modded_road_variant_availability(spec)
    token = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(availability)
    try:
        variants = playability.road_model_variants(
            spec.paved_road_model,
            spec.road_segment_length,
        )
    finally:
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(token)

    assert [piece.model_path for piece in variants] == [
        r"myroads\asphalt25.p3d",
    ]


def test_configured_modded_gravel_model_is_selected() -> None:
    spec = SimpleNamespace(
        name="world",
        paved_road_model=r"myroads\asphalt25.p3d",
        gravel_road_model=r"myroads\gravel25.p3d",
        dirt_road_model=r"myroads\dirt25.p3d",
        procedural_gravel_roads=True,
    )

    assert road_model_for_tags(
        spec,
        {"highway": "track", "surface": "gravel"},
    ) == r"myroads\gravel25.p3d"


@pytest.mark.parametrize(
    ("surface", "texture"),
    (
        ("paved", r"modroads\textures\paved.paa"),
        ("gravel", r"modroads\textures\gravel.paa"),
        ("dirt", r"modroads\textures\dirt.paa"),
    ),
)
def test_generated_missing_shapes_reuse_mod_texture(
    tmp_path: Path,
    surface: str,
    texture: str,
) -> None:
    model = infrastructure.custom_road_model_path(
        "donorworld",
        surface,
        width_metres=5.8,
        length_metres=11.7,
        curve_degrees=27.0,
    )
    library = infrastructure.ProceduralInfrastructureLibrary(
        "donorworld",
        paved_texture_path=r"modroads\textures\paved.paa",
        gravel_texture_path=r"modroads\textures\gravel.paa",
        dirt_texture_path=r"modroads\textures\dirt.paa",
        cache_enabled=False,
    )
    library.register_model(model)
    catalogue = tmp_path / "infrastructure.json"
    result = library.write_assets(tmp_path, catalogue)

    assert result.generated_variants == 1
    assert result.texture_files == ()
    relative = model.split("\\", 1)[1].replace("\\", "/")
    summary = infrastructure.inspect_mlod(tmp_path / relative)
    assert texture.casefold() in {
        value.casefold() for value in summary.texture_paths
    }

    document = json.loads(catalogue.read_text(encoding="utf-8"))
    source = document[f"{surface}_texture_source"]
    assert source["type"] == "external-road-texture"
    assert source["texture"].casefold() == texture.casefold()


def test_generated_paved_junction_accepts_modded_width() -> None:
    incidents = (
        ((0.0, 1.0), "sil"),
        ((0.0, -1.0), "sil"),
        ((1.0, 0.0), "sil"),
    )
    plan = paved_junction_policy._generated_plan(
        (100.0, 100.0),
        incidents,
        world_name="donorworld",
        width_override=6.0,
    )

    assert plan is not None
    assert plan.model_path.endswith(
        r"\i\paved_j3_w060_h000_090_180.p3d"
    )


def test_generated_custom_paved_is_recognized_by_junction_policy() -> None:
    model = infrastructure.custom_road_model_path(
        "donorworld", "paved", 6.0, 10.0, 27.0
    )
    assert paved_junction_policy._family(model) == "sil"


def test_gui_road_defaults_include_stock_curve_donors() -> None:
    values = default_gui_values()
    assert values["paved_road_model"] == r"o\road\sil25.p3d"
    assert values["paved_road_curve_model"] == r"o\road\sil10 25.p3d"
    assert values["gravel_road_model"] == ""
    assert values["gravel_road_curve_model"] == ""
    assert values["dirt_road_model"] == r"o\road\ces25.p3d"
    assert values["dirt_road_curve_model"] == r"o\road\ces10 25.p3d"


def test_saved_blank_stock_curves_upgrade_without_touching_custom_families() -> None:
    defaults = default_gui_values()
    state = {
        "paved_road_model": r"o\road\sil25.p3d",
        "paved_road_curve_model": "",
        "gravel_road_model": r"mods\gravel25.p3d",
        "gravel_road_curve_model": "",
        "dirt_road_model": r"mods\dirt25.p3d",
        "dirt_road_curve_model": "",
    }

    values = defaults_with_recent_source(defaults, state)

    assert values["paved_road_curve_model"] == r"o\road\sil10 25.p3d"
    assert values["gravel_road_curve_model"] == ""
    assert values["dirt_road_model"] == r"mods\dirt25.p3d"
    assert values["dirt_road_curve_model"] == ""


def test_restore_road_defaults_resets_straight_and_curve_together() -> None:
    class FakeVar:
        def __init__(self, value: str = "") -> None:
            self.value = value

        def get(self) -> str:
            return self.value

        def set(self, value: object) -> None:
            self.value = str(value)

    fake = SimpleNamespace(
        vars={
            "paved_road_model": FakeVar(r"mods\custom25.p3d"),
            "paved_road_curve_model": FakeVar(r"mods\custom_curve.p3d"),
            "gravel_road_model": FakeVar(r"mods\gravel25.p3d"),
            "gravel_road_curve_model": FakeVar(r"mods\gravel_curve.p3d"),
            "dirt_road_model": FakeVar(r"mods\dirt25.p3d"),
            "dirt_road_curve_model": FakeVar(r"mods\dirt_curve.p3d"),
        },
        footer_status_var=FakeVar(),
    )

    WorldgenGui._restore_road_defaults(fake, "paved")
    assert fake.vars["paved_road_model"].get() == r"o\road\sil25.p3d"
    assert fake.vars["paved_road_curve_model"].get() == r"o\road\sil10 25.p3d"
    assert fake.vars["gravel_road_model"].get() == r"mods\gravel25.p3d"

    WorldgenGui._restore_road_defaults(fake, "gravel")
    assert fake.vars["gravel_road_model"].get() == ""
    assert fake.vars["gravel_road_curve_model"].get() == ""

    WorldgenGui._restore_road_defaults(fake, "dirt")
    assert fake.vars["dirt_road_model"].get() == r"o\road\ces25.p3d"
    assert fake.vars["dirt_road_curve_model"].get() == r"o\road\ces10 25.p3d"
    assert "default dirt straight and curve" in fake.footer_status_var.get()


def test_gui_command_exposes_all_three_modded_road_donors() -> None:
    values = default_gui_values()
    values.update({
        "source_dir": "source",
        "output": "build",
        "name": "donorworld",
        "display_name": "Donor World",
        "paved_road_model": r"myroads\asphalt25.p3d",
        "paved_road_curve_model": r"myroads\asphalt_curve.p3d",
        "gravel_road_model": r"myroads\gravel25.p3d",
        "gravel_road_curve_model": r"myroads\gravel_curve.p3d",
        "dirt_road_model": r"myroads\track25.p3d",
        "dirt_road_curve_model": r"myroads\track_curve.p3d",
    })
    command = build_milestone9_command(values, python="python")

    for option, model in (
        ("--paved-road-model", r"myroads\asphalt25.p3d"),
        ("--paved-road-curve-model", r"myroads\asphalt_curve.p3d"),
        ("--gravel-road-model", r"myroads\gravel25.p3d"),
        ("--gravel-road-curve-model", r"myroads\gravel_curve.p3d"),
        ("--dirt-road-model", r"myroads\track25.p3d"),
        ("--dirt-road-curve-model", r"myroads\track_curve.p3d"),
    ):
        index = command.index(option)
        assert command[index + 1] == model


def test_modded_road_family_and_texture_are_discovered_inside_pbo(
    tmp_path: Path,
) -> None:
    pbo = tmp_path / "myroads.pbo"
    write_pbo(
        pbo,
        (
            PboEntry(
                "asphalt25.p3d",
                b"MLOD donor myroads\\textures\\asphalt_main.paa\x00",
            ),
            PboEntry(
                "asphalt12.p3d",
                b"MLOD short myroads\\textures\\asphalt_main.paa\x00",
            ),
            PboEntry("textures/asphalt_main.paa", b"synthetic-paa"),
        ),
    )

    spec = SimpleNamespace(
        paved_road_model=r"myroads\asphalt25.p3d",
        gravel_road_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
        asset_roots=(pbo,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )

    availability = generator._modded_road_variant_availability(spec)
    token = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(availability)
    try:
        variants = playability.road_model_variants(
            spec.paved_road_model,
            spec.road_segment_length,
        )
    finally:
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(token)

    assert [piece.model_path for piece in variants] == [
        r"myroads\asphalt25.p3d",
        r"myroads\asphalt12.p3d",
    ]

    scan = scan_assets(
        (pbo,),
        (spec.paved_road_model,),
        use_cache=False,
    )
    dependencies = model_texture_dependencies(
        scan.records,
        spec.paved_road_model,
    )
    assert dependencies == (r"myroads\textures\asphalt_main.paa",)


def test_modded_donor_geometry_controls_generated_width_and_length(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mod"
    donor = r"myroads\asphalt_long.p3d"
    texture = r"myroads\tex\road.paa"
    _write_fake_mod_asset(root, donor, _mlod_road(7.2, 21.4, texture))
    _write_fake_mod_asset(root, texture, b"synthetic-paa")

    spec = SimpleNamespace(
        name="donorworld",
        paved_road_model=donor,
        gravel_road_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
        asset_roots=(root,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
        custom_road_shapes=True,
    )
    dimensions = generator._modded_road_model_dimensions(spec)
    key = playability._road_model_key(donor)
    assert dimensions[key] == pytest.approx((7.2, 21.4))

    dimension_token = playability._ROAD_MODEL_DIMENSIONS.set(dimensions)
    tag_token = playability._ACTIVE_ROAD_TAGS.set(
        {"highway": "residential", "surface": "asphalt"}
    )
    try:
        variants = playability.road_model_variants(donor, spec.road_segment_length)
        assert len(variants) == 1
        assert variants[0].model_path == donor
        assert variants[0].length_metres == pytest.approx(21.4)

        # Residential OSM width is 6 m, but a generated bend joining this donor
        # must stay 7.2 m wide or it would visibly neck down at the seam.
        assert fallback._generated_width(
            variants,
            spec,
            "paved",
        ) == pytest.approx(7.2)
    finally:
        playability._ACTIVE_ROAD_TAGS.reset(tag_token)
        playability._ROAD_MODEL_DIMENSIONS.reset(dimension_token)


def test_modded_donor_geometry_is_measured_inside_pbo(tmp_path: Path) -> None:
    texture = r"myroads\textures\track.paa"
    pbo = tmp_path / "myroads.pbo"
    write_pbo(
        pbo,
        (
            PboEntry(
                "track_long.p3d",
                _mlod_road(3.8, 18.6, texture),
            ),
            PboEntry("textures/track.paa", b"synthetic-paa"),
        ),
    )
    spec = SimpleNamespace(
        paved_road_model=r"o\road\sil25.p3d",
        gravel_road_model="",
        dirt_road_model=r"myroads\track_long.p3d",
        road_segment_length=25.0,
        asset_roots=(pbo,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )

    dimensions = generator._modded_road_model_dimensions(spec)
    assert dimensions[
        playability._road_model_key(spec.dirt_road_model)
    ] == pytest.approx((3.8, 18.6))


def test_milestone9_cli_accepts_modded_road_donor_flags() -> None:
    args = cli._parser().parse_args(
        [
            "milestone9",
            "--source-dir",
            "source",
            "--output",
            "build",
            "--paved-road-model",
            r"myroads\paved25.p3d",
            "--paved-road-curve-model",
            r"myroads\paved_curve.p3d",
            "--gravel-road-model",
            r"myroads\gravel25.p3d",
            "--gravel-road-curve-model",
            r"myroads\gravel_curve.p3d",
            "--dirt-road-model",
            r"myroads\track25.p3d",
            "--dirt-road-curve-model",
            r"myroads\track_curve.p3d",
        ]
    )

    assert args.paved_road_model == r"myroads\paved25.p3d"
    assert args.paved_road_curve_model == r"myroads\paved_curve.p3d"
    assert args.gravel_road_model == r"myroads\gravel25.p3d"
    assert args.gravel_road_curve_model == r"myroads\gravel_curve.p3d"
    assert args.dirt_road_model == r"myroads\track25.p3d"
    assert args.dirt_road_curve_model == r"myroads\track_curve.p3d"


@pytest.mark.parametrize("command", ("milestone5", "milestone6", "milestone7", "milestone8", "milestone9"))
def test_source_bundle_cli_exposes_road_donor_flags(command: str) -> None:
    args = cli._parser().parse_args(
        [
            command,
            "--source-dir",
            "source",
            "--output",
            "build",
            "--paved-road-model",
            r"myroads\paved25.p3d",
        ]
    )
    assert args.paved_road_model == r"myroads\paved25.p3d"


@pytest.mark.parametrize(
    "spec_type",
    (Milestone5Spec, Milestone6Spec, Milestone7Spec, Milestone8Spec, Milestone9Spec),
)
def test_source_bundle_specs_accept_road_donor_models(spec_type) -> None:
    spec = spec_type(
        source_dir=Path("source"),
        paved_road_model=r"myroads\paved25.p3d",
        paved_road_curve_model=r"myroads\paved_curve.p3d",
        gravel_road_model=r"myroads\gravel25.p3d",
        gravel_road_curve_model=r"myroads\gravel_curve.p3d",
        dirt_road_model=r"myroads\track25.p3d",
        dirt_road_curve_model=r"myroads\track_curve.p3d",
    )

    assert spec.paved_road_model == r"myroads\paved25.p3d"
    assert spec.paved_road_curve_model == r"myroads\paved_curve.p3d"
    assert spec.gravel_road_model == r"myroads\gravel25.p3d"
    assert spec.gravel_road_curve_model == r"myroads\gravel_curve.p3d"
    assert spec.dirt_road_model == r"myroads\track25.p3d"
    assert spec.dirt_road_curve_model == r"myroads\track_curve.p3d"


def test_milestone9_spec_accepts_gui_road_arguments_exactly() -> None:
    args = cli._parser().parse_args(
        [
            "milestone9",
            "--source-dir",
            r"G:\cwa_worldgen\source-data\tiny_bjorsund",
            "--output",
            r"G:\cwa_worldgen\build\terrtest82",
            "--name",
            "wg_terrtest82",
            "--display-name",
            "terrtest82",
            "--paved-road-model",
            r"o\road\sil25.p3d",
            "--gravel-road-model",
            r"o\road\sil25.p3d",
            "--dirt-road-model",
            r"o\road\ces25.p3d",
        ]
    )

    spec = Milestone9Spec(
        source_dir=args.source_dir,
        name=args.name,
        display_name=args.display_name,
        paved_road_model=args.paved_road_model,
        gravel_road_model=args.gravel_road_model,
        dirt_road_model=args.dirt_road_model,
    )

    assert spec.paved_road_model == r"o\road\sil25.p3d"
    assert spec.gravel_road_model == r"o\road\sil25.p3d"
    assert spec.dirt_road_model == r"o\road\ces25.p3d"


def test_pre_zstd_asset_cache_is_invalidated_for_nested_mod_donor(
    tmp_path: Path,
) -> None:
    inner = tmp_path / "sebnam_obj.pbo"
    texture = r"sebnam_obj\trail.paa"
    donor = r"sebnam_obj\sebtrailpath10 25.p3d"
    write_pbo(
        inner,
        (
            PboEntry("sebtrailpath10 25.p3d", _mlod_road(4.8, 24.5, texture)),
            PboEntry("trail.paa", b"synthetic-paa"),
        ),
    )
    outer = tmp_path / "mod-package.pbo"
    write_pbo(
        outer,
        (PboEntry(r"addons\sebnam_obj.pbo", inner.read_bytes()),),
    )
    wrapped = tmp_path / "mod-package.pbo.zst"
    wrapped.write_bytes(
        zstandard.ZstdCompressor(level=1).compress(outer.read_bytes())
    )
    cache_dir = tmp_path / "cache"

    first = scan_assets(
        (wrapped,),
        (donor,),
        cache_dir=cache_dir,
        use_cache=True,
        refresh=False,
    )
    assert donor in {record.path for record in first.records}
    assert first.cache_path is not None

    cache_path = Path(first.cache_path)
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    document["asset_schema"] = 1
    document["records"] = []
    cache_path.write_text(json.dumps(document), encoding="utf-8")
    asset_module._CATALOGUE_MEMORY.clear()

    second = scan_assets(
        (wrapped,),
        (donor,),
        cache_dir=cache_dir,
        use_cache=True,
        refresh=False,
    )

    assert donor in {record.path for record in second.records}
    assert second.cache_hit is False
    deps = model_texture_dependencies(second.records, donor)
    assert deps == (texture,)


def test_unresolved_mod_gravel_donor_never_silently_uses_generic_texture() -> None:
    donor = r"sebnam_obj\sebtrailpath10 25.p3d"
    with pytest.raises(ValueError, match="could not resolve the gravel road donor"):
        generator._resolved_road_donor_texture(
            (),
            surface="gravel",
            donor_model=donor,
        )


def test_stock_road_donors_keep_known_texture_fallbacks() -> None:
    assert generator._resolved_road_donor_texture(
        (),
        surface="paved",
        donor_model=r"o\road\sil25.p3d",
    ) == r"landtext\silnice.pac"
    assert generator._resolved_road_donor_texture(
        (),
        surface="dirt",
        donor_model=r"o\road\ces25.p3d",
    ) == r"o\road\ces_hned.paa"


def test_sebnam_curve_donor_resolves_to_straight_family(tmp_path: Path) -> None:
    texture = r"sebnam_obj\p\sebtrailpath.paa"
    curved = r"sebnam_obj\sebtrailpath10 25.p3d"
    pbo = tmp_path / "sebnam_obj.pbo"
    write_pbo(
        pbo,
        (
            PboEntry(
                "sebtrailpath10 25.p3d",
                _mlod_skewed_road(3.5, 4.36, 0.40, texture),
            ),
            PboEntry("sebtrailpath25.p3d", _mlod_road(3.5, 25.0, texture)),
            PboEntry("sebtrailpath12.p3d", _mlod_road(3.5, 12.5, texture)),
            PboEntry("sebtrailpath6.p3d", _mlod_road(3.5, 6.25, texture)),
            PboEntry("p/sebtrailpath.paa", b"synthetic-paa"),
        ),
    )
    spec = SimpleNamespace(
        name="sebworld",
        paved_road_model=r"o\road\sil25.p3d",
        gravel_road_model=curved,
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=24.5,
        asset_roots=(pbo,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
        custom_road_shapes=True,
        procedural_gravel_roads=True,
    )

    scan = scan_assets((pbo,), (curved,), use_cache=False)
    by_path = {record.path: record for record in scan.records}
    curved_shape = inspect_visual_model_dimensions(
        asset_module.read_asset_record_bytes(by_path[curved])
    )
    straight_shape = inspect_visual_model_dimensions(
        asset_module.read_asset_record_bytes(
            by_path[r"sebnam_obj\sebtrailpath25.p3d"]
        )
    )
    assert curved_shape.is_straight_road_candidate is False
    assert straight_shape.is_straight_road_candidate is True
    assert model_texture_dependencies(scan.records, curved) == (texture,)
    assert model_texture_dependencies(
        scan.records, r"sebnam_obj\sebtrailpath25.p3d"
    ) == (texture,)
    match = generator._CURVED_ROAD_DONOR_NAME.fullmatch(
        r"sebtrailpath10 25.p3d"
    )
    assert match is not None
    assert match.group("prefix") == "sebtrailpath"
    preferred = rf"sebnam_obj\{match.group('prefix')}25.p3d"
    assert preferred == r"sebnam_obj\sebtrailpath25.p3d"
    assert preferred in by_path

    effective = generator._modded_road_effective_donors(spec)
    assert effective[playability._road_model_key(curved)] == (
        r"sebnam_obj\sebtrailpath25.p3d"
    )

    availability = generator._modded_road_variant_availability(spec, effective)
    dimensions = generator._modded_road_model_dimensions(spec, effective)
    effective_token = playability._ROAD_MODEL_EFFECTIVE_DONORS.set(effective)
    availability_token = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(availability)
    dimensions_token = playability._ROAD_MODEL_DIMENSIONS.set(dimensions)
    try:
        assert playability.road_model_for_tags(
            spec,
            {"highway": "track", "surface": "gravel"},
        ) == r"sebnam_obj\sebtrailpath25.p3d"

        variants = playability.road_model_variants(
            curved,
            spec.road_segment_length,
        )
        assert [piece.model_path for piece in variants] == [
            r"sebnam_obj\sebtrailpath25.p3d",
        ]
        assert [piece.length_metres for piece in variants] == pytest.approx(
            [25.0]
        )
        assert fallback._generated_width(
            variants,
            spec,
            "gravel",
        ) == pytest.approx(3.5)
        filler = playability.gravel_filler_piece(spec, 6)
        assert filler.model_path == infrastructure.custom_road_model_path(
            "sebworld",
            "gravel",
            3.5,
            spec.road_segment_length * 6.0 / 25.0,
        )
        filler_signature = infrastructure.custom_road_model_signature(
            filler.model_path
        )
        assert filler_signature is not None
        assert filler_signature[0] == "gravel"
        assert filler_signature[1] == pytest.approx(3.5)
    finally:
        playability._ROAD_MODEL_DIMENSIONS.reset(dimensions_token)
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(availability_token)
        playability._ROAD_MODEL_EFFECTIVE_DONORS.reset(effective_token)


def test_curved_mod_donor_without_straight_sibling_is_rejected(tmp_path: Path) -> None:
    texture = r"sebnam_obj\p\sebtrailpath.paa"
    pbo = tmp_path / "sebnam_obj.pbo"
    write_pbo(
        pbo,
        (
            PboEntry(
                "sebtrailpath10 25.p3d",
                _mlod_skewed_road(3.5, 4.36, 0.40, texture),
            ),
            PboEntry("p/sebtrailpath.paa", b"synthetic-paa"),
        ),
    )
    spec = SimpleNamespace(
        paved_road_model=r"o\road\sil25.p3d",
        gravel_road_model=r"sebnam_obj\sebtrailpath10 25.p3d",
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=24.5,
        asset_roots=(pbo,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )

    with pytest.raises(ValueError, match="curved road piece"):
        generator._modded_road_effective_donors(spec)


def test_sebnam_style_uses_straight_family_and_explicit_curve_reference(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mod"
    texture = r"sebnam_obj\p\sebtrailpath.paa"
    straight25 = r"sebnam_obj\sebtrailpath25.p3d"
    straight12 = r"sebnam_obj\sebtrailpath12.p3d"
    straight6 = r"sebnam_obj\sebtrailpath6.p3d"
    curve = r"sebnam_obj\sebtrailpath10 25.p3d"

    _write_fake_mod_asset(root, straight25, _mlod_road(3.5, 25.0, texture))
    _write_fake_mod_asset(root, straight12, _mlod_road(3.5, 12.5, texture))
    _write_fake_mod_asset(root, straight6, _mlod_road(3.5, 6.25, texture))
    _write_fake_mod_asset(root, curve, _mlod_curve_sample(3.5, 4.65, 0.38, texture))
    _write_fake_mod_asset(root, texture, b"synthetic-paa")

    spec = SimpleNamespace(
        name="sebworld",
        paved_road_model=r"o\road\sil25.p3d",
        paved_road_curve_model="",
        gravel_road_model=straight25,
        gravel_road_curve_model=curve,
        dirt_road_model=r"o\road\ces25.p3d",
        dirt_road_curve_model="",
        road_segment_length=25.0,
        asset_roots=(root,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
        custom_road_shapes=True,
    )

    effective = generator._modded_road_effective_donors(spec)
    assert effective[playability._road_model_key(straight25)] == straight25
    assert generator._road_style_donor(spec, "gravel") == curve

    availability = generator._modded_road_variant_availability(spec, effective)
    dimensions = generator._modded_road_model_dimensions(spec, effective)
    effective_token = playability._ROAD_MODEL_EFFECTIVE_DONORS.set(effective)
    availability_token = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(availability)
    dimensions_token = playability._ROAD_MODEL_DIMENSIONS.set(dimensions)
    try:
        pieces = playability.road_model_variants(straight25, 25.0)
    finally:
        playability._ROAD_MODEL_DIMENSIONS.reset(dimensions_token)
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(availability_token)
        playability._ROAD_MODEL_EFFECTIVE_DONORS.reset(effective_token)

    assert availability == {
        playability._road_model_key(straight25): frozenset({
            playability._road_model_key(straight25)
        })
    }
    assert playability._road_model_key(straight12) not in dimensions
    assert playability._road_model_key(straight6) not in dimensions
    assert [piece.model_path for piece in pieces] == [straight25]
    assert [piece.length_metres for piece in pieces] == pytest.approx([25.0])


def test_legacy_curved_sebnam_selection_auto_resolves_to_straight_sibling(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mod"
    texture = r"sebnam_obj\p\sebtrailpath.paa"
    curve = r"sebnam_obj\sebtrailpath10 25.p3d"
    straight = r"sebnam_obj\sebtrailpath25.p3d"
    _write_fake_mod_asset(root, curve, _mlod_curve_sample(3.5, 4.65, 0.38, texture))
    _write_fake_mod_asset(root, straight, _mlod_road(3.5, 25.0, texture))
    _write_fake_mod_asset(root, r"sebnam_obj\sebtrailpath12.p3d", _mlod_road(3.5, 12.5, texture))
    _write_fake_mod_asset(root, r"sebnam_obj\sebtrailpath6.p3d", _mlod_road(3.5, 6.25, texture))
    _write_fake_mod_asset(root, texture, b"synthetic-paa")

    spec = SimpleNamespace(
        paved_road_model=r"o\road\sil25.p3d",
        gravel_road_model=curve,
        dirt_road_model=r"o\road\ces25.p3d",
        road_segment_length=25.0,
        asset_roots=(root,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )

    effective = generator._modded_road_effective_donors(spec)
    assert effective[playability._road_model_key(curve)] == straight


def test_legacy_curve_resolution_never_walks_unrelated_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "mod"
    texture = r"sebnam_obj\p\sebtrailpath.paa"
    curve = r"sebnam_obj\sebtrailpath10 25.p3d"
    straight = r"sebnam_obj\sebtrailpath25.p3d"
    _write_fake_mod_asset(root, curve, _mlod_curve_sample(3.5, 4.65, 0.38, texture))
    _write_fake_mod_asset(root, straight, _mlod_road(3.5, 25.0, texture))
    _write_fake_mod_asset(root, texture, b"synthetic-paa")

    # Hundreds of unrelated invalid P3Ds reproduce the old pathological case.
    # The resolver must not open any of them merely to identify one road family.
    for index in range(300):
        _write_fake_mod_asset(
            root,
            rf"unrelated\junk{index:03d}.p3d",
            b"not-a-p3d",
        )

    inspected: list[bytes] = []
    real_inspect = generator.inspect_visual_model_dimensions

    def counted_inspect(data: bytes):
        inspected.append(data[:8])
        return real_inspect(data)

    monkeypatch.setattr(
        generator,
        "inspect_visual_model_dimensions",
        counted_inspect,
    )

    spec = SimpleNamespace(
        paved_road_model="",
        paved_road_curve_model="",
        gravel_road_model=curve,
        gravel_road_curve_model="",
        dirt_road_model="",
        dirt_road_curve_model="",
        road_segment_length=25.0,
        asset_roots=(root,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )

    effective = generator._modded_road_effective_donors(spec)

    assert effective[playability._road_model_key(curve)] == straight
    # The filename + exact sibling lookup is enough; no model mesh walk occurs.
    assert inspected == []


def test_exact_road_family_lookup_ignores_missing_texture_dependencies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mod"
    donor = r"myroads\road25.p3d"
    _write_fake_mod_asset(
        root,
        donor,
        _mlod_road(4.0, 25.0, r"myroads\missing.paa"),
    )

    from cwr_worldgen.fast_asset_scan_policy import locate_assets_fast

    result = locate_assets_fast(
        (root,),
        (donor,),
        use_cache=False,
    )

    assert result.missing_models == ()
    assert result.missing_dependencies == ()
    assert [record.path for record in result.records] == [donor]


def test_bas_o_straight_curve_pair_keeps_width_and_texture(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mod"
    texture = r"bas_o\_tobj\bas_road2.paa"
    straight25 = r"bas_o\_road\bas_asf25.p3d"
    straight12 = r"bas_o\_road\bas_asf12.p3d"
    straight6 = r"bas_o\_road\bas_asf6.p3d"
    curve = r"bas_o\_road\bas_asf10 25.p3d"

    _write_fake_mod_asset(
        root,
        straight25,
        _mlod_road_with_midspan_detail(5.2, 25.0, 8.4, texture),
    )
    _write_fake_mod_asset(root, straight12, _mlod_road(5.2, 12.5, texture))
    _write_fake_mod_asset(root, straight6, _mlod_road(5.2, 6.25, texture))
    _write_fake_mod_asset(root, curve, _mlod_curve_sample(5.2, 4.36, 0.38, texture))
    _write_fake_mod_asset(root, texture, b"synthetic-paa")

    spec = SimpleNamespace(
        name="basworld",
        paved_road_model=straight25,
        paved_road_curve_model=curve,
        gravel_road_model="",
        gravel_road_curve_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        dirt_road_curve_model="",
        road_segment_length=25.0,
        asset_roots=(root,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
        custom_road_shapes=True,
    )

    effective = generator._modded_road_effective_donors(spec)
    availability = generator._modded_road_variant_availability(spec, effective)
    dimensions = generator._modded_road_model_dimensions(spec, effective)

    straight_info = inspect_visual_model_dimensions(
        (root / straight25.replace("\\\\", "/")).read_bytes()
    )
    curve_info = inspect_visual_model_dimensions(
        (root / curve.replace("\\\\", "/")).read_bytes()
    )
    assert straight_info.width_metres == pytest.approx(8.4)
    assert straight_info.connector_width_metres == pytest.approx(5.2)
    assert curve_info.connector_width_metres == pytest.approx(5.2)
    assert dimensions[playability._road_model_key(straight25)] == pytest.approx(
        (5.2, 25.0)
    )
    assert dimensions[playability._road_model_key(curve)] == pytest.approx(
        (5.2, 4.36)
    )

    effective_token = playability._ROAD_MODEL_EFFECTIVE_DONORS.set(effective)
    availability_token = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set(availability)
    dimensions_token = playability._ROAD_MODEL_DIMENSIONS.set(dimensions)
    tag_token = playability._ACTIVE_ROAD_TAGS.set(
        {"highway": "primary", "surface": "asphalt"}
    )
    try:
        pieces = playability.road_model_variants(straight25, 25.0)
        generated_width = fallback._generated_width(pieces, spec, "paved")
    finally:
        playability._ACTIVE_ROAD_TAGS.reset(tag_token)
        playability._ROAD_MODEL_DIMENSIONS.reset(dimensions_token)
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(availability_token)
        playability._ROAD_MODEL_EFFECTIVE_DONORS.reset(effective_token)

    assert availability == {
        playability._road_model_key(straight25): frozenset({
            playability._road_model_key(straight25)
        })
    }
    assert playability._road_model_key(straight12) not in dimensions
    assert playability._road_model_key(straight6) not in dimensions
    assert [piece.model_path for piece in pieces] == [straight25]
    assert [piece.length_metres for piece in pieces] == pytest.approx([25.0])
    assert generated_width == pytest.approx(5.2)
    assert generated_width != pytest.approx(9.1)

    assert generator._road_style_donor(spec, "paved") == curve
    scan = scan_assets((root,), (curve,), use_cache=False)
    assert generator._resolved_road_donor_texture(
        scan.records,
        surface="paved",
        donor_model=curve,
    ) == texture


def test_fast_mod_road_measurement_reads_legacy_compressed_p3d(
    tmp_path: Path,
) -> None:
    texture = r"bas_o\_tobj\bas_road2.paa"
    donor = r"bas_o\_road\bas_asf25.p3d"
    payload = _mlod_road(6.4, 25.0, texture)
    pbo = tmp_path / "bas_o.pbo"
    _write_compressed_member_pbo(
        pbo,
        r"_road\bas_asf25.p3d",
        payload,
    )

    scan = generator.locate_assets_fast(
        (pbo,),
        (donor,),
        use_cache=False,
    )
    record = next(record for record in scan.records if record.path == donor)

    assert record.readable is True
    assert asset_module.read_asset_record_bytes(record) == payload
    measured = inspect_visual_model_dimensions(
        asset_module.read_asset_record_bytes(record)
    )
    assert measured.width_metres == pytest.approx(6.4)
    assert measured.length_metres == pytest.approx(25.0)


def test_unmeasured_mod_road_width_fails_instead_of_using_generic_width() -> None:
    donor = r"bas_o\_road\bas_asf25.p3d"
    spec = SimpleNamespace(
        paved_road_model=r"o\road\sil25.p3d",
        paved_road_curve_model="",
        gravel_road_model=donor,
        gravel_road_curve_model=r"bas_o\_road\bas_asf10 25.p3d",
        dirt_road_model=r"o\road\ces25.p3d",
        dirt_road_curve_model="",
        custom_road_shapes=True,
    )
    pieces = (playability._RoadPiece(donor, 25.0, 25),)

    dimensions_token = playability._ROAD_MODEL_DIMENSIONS.set(None)
    try:
        with pytest.raises(ValueError, match="could not measure the gravel road donor"):
            fallback._generated_width(pieces, spec, "gravel")
    finally:
        playability._ROAD_MODEL_DIMENSIONS.reset(dimensions_token)


def test_unmeasured_pbo_donor_reports_actual_p3d_failure(tmp_path: Path) -> None:
    donor = r"bas_o\_road\bas_asf25.p3d"
    pbo = tmp_path / "BAS_O.pbo"
    write_pbo(
        pbo,
        (
            PboEntry(
                r"_road\bas_asf25.p3d",
                b"ODOL" + struct.pack("<II", 99, 1),
            ),
        ),
    )
    spec = SimpleNamespace(
        paved_road_model=r"o\road\sil25.p3d",
        paved_road_curve_model="",
        gravel_road_model=donor,
        gravel_road_curve_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        dirt_road_curve_model="",
        road_segment_length=25.0,
        asset_roots=(pbo,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
        custom_road_shapes=True,
    )

    errors: dict[str, str] = {}
    dimensions = generator._modded_road_model_dimensions(
        spec,
        {},
        measurement_errors=errors,
    )
    key = playability._road_model_key(donor)
    assert key not in dimensions
    assert pbo.name in errors[key]
    assert "unsupported ODOL version 99" in errors[key]

    pieces = (playability._RoadPiece(donor, 25.0, 25),)
    dimensions_token = playability._ROAD_MODEL_DIMENSIONS.set(dimensions or None)
    errors_token = playability._ROAD_MODEL_MEASUREMENT_ERRORS.set(errors)
    try:
        with pytest.raises(
            ValueError,
            match="unsupported ODOL version 99",
        ):
            fallback._generated_width(pieces, spec, "gravel")
    finally:
        playability._ROAD_MODEL_MEASUREMENT_ERRORS.reset(errors_token)
        playability._ROAD_MODEL_DIMENSIONS.reset(dimensions_token)


def test_road_donor_diagnostics_records_measured_straight_and_curve_style() -> None:
    straight = r"bas_o\_road\bas_asf25.p3d"
    curve = r"bas_o\_road\bas_asf10 25.p3d"
    spec = SimpleNamespace(
        paved_road_model=r"o\road\sil25.p3d",
        paved_road_curve_model="",
        gravel_road_model=straight,
        gravel_road_curve_model=curve,
        dirt_road_model=r"o\road\ces25.p3d",
        dirt_road_curve_model="",
    )
    diagnostics = generator._road_donor_diagnostics(
        spec,
        {playability._road_model_key(straight): straight},
        {
            playability._road_model_key(straight): (5.2, 25.0),
            playability._road_model_key(curve): (5.2, 4.36),
        },
    )

    gravel = diagnostics["gravel"]
    assert gravel["straight"] == straight
    assert gravel["curve"] == curve
    assert gravel["style_donor"] == curve
    assert gravel["effective_straight"] == straight
    assert gravel["measured_straight_width_metres"] == pytest.approx(5.2)
    assert gravel["measured_straight_length_metres"] == pytest.approx(25.0)
    assert gravel["straight_geometry_measured"] is True
    assert gravel["measured_curve_connector_width_metres"] == pytest.approx(5.2)
    assert gravel["measured_curve_chord_length_metres"] == pytest.approx(4.36)
    assert gravel["curve_geometry_measured"] is True
    assert gravel["connector_width_delta_metres"] == pytest.approx(0.0)


def test_generated_mod_road_rejects_different_straight_and_curve_mouths() -> None:
    straight = r"bas_o\_road\bas_asf25.p3d"
    curve = r"bas_o\_road\bas_asf10 25.p3d"
    spec = SimpleNamespace(
        paved_road_model=straight,
        paved_road_curve_model=curve,
        gravel_road_model="",
        gravel_road_curve_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        dirt_road_curve_model="",
        custom_road_shapes=True,
    )
    pieces = (playability._RoadPiece(straight, 25.0, 25),)
    dimensions_token = playability._ROAD_MODEL_DIMENSIONS.set(
        {
            playability._road_model_key(straight): (5.2, 25.0),
            playability._road_model_key(curve): (6.0, 4.36),
        }
    )
    try:
        with pytest.raises(ValueError, match="connector widths do not match"):
            fallback._generated_width(pieces, spec, "paved")
    finally:
        playability._ROAD_MODEL_DIMENSIONS.reset(dimensions_token)


def test_curve_path_cannot_bypass_straight_donor_validation(tmp_path: Path) -> None:
    root = tmp_path / "mod"
    texture = r"bas_o\_tobj\bas_road2.paa"
    curve = r"bas_o\_road\bas_asf10 25.p3d"
    _write_fake_mod_asset(root, curve, _mlod_curve_sample(5.2, 4.36, 0.38, texture))

    spec = SimpleNamespace(
        paved_road_model=curve,
        paved_road_curve_model=curve,
        gravel_road_model="",
        gravel_road_curve_model="",
        dirt_road_model=r"o\road\ces25.p3d",
        dirt_road_curve_model="",
        road_segment_length=25.0,
        asset_roots=(root,),
        cache_dir=None,
        cache_enabled=False,
        cache_refresh=False,
    )

    dimensions = generator._modded_road_model_dimensions(spec, {})
    assert playability._road_model_key(curve) not in dimensions
