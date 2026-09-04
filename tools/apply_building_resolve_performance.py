from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_between(text: str, start: str, end: str, replacement: str, *, label: str) -> str:
    start_index = text.find(start)
    if start_index < 0:
        raise RuntimeError(f"{label}: start marker not found")
    end_index = text.find(end, start_index)
    if end_index < 0:
        raise RuntimeError(f"{label}: end marker not found")
    return text[:start_index] + replacement + text[end_index:]


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


runtime_path = ROOT / "src" / "cwr_worldgen" / "osm_house_modeler_runtime.py"
runtime = runtime_path.read_text(encoding="utf-8")

helpers = '''\n\ndef _tag_signature(tags: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Return a stable hashable signature for one mapped building tag set."""
    return tuple(sorted((str(key), str(value)) for key, value in tags.items()))


def _polygon_request_signature(tags, points, holes=()):
    return (
        _tag_signature(tags),
        tuple((float(x), float(z)) for x, z in points),
        tuple(tuple((float(x), float(z)) for x, z in ring) for ring in holes),
    )


def _point_request_signature(tags, footprint, x, z):
    return (
        _tag_signature(tags),
        round(float(footprint), 9),
        round(float(x), 9) if x is not None else None,
        round(float(z), 9) if z is not None else None,
    )


@contextmanager
def _prepared_style_request(value):
    """Expose one already-resolved prepare-time style key to placement."""
    previous = getattr(_STYLE_STATE, "prepared_request", None)
    _STYLE_STATE.prepared_request = value
    try:
        yield
    finally:
        _STYLE_STATE.prepared_request = previous
'''
runtime = runtime.replace("\n\ndef _prepare_geographic_context(self, dataset, projection):", helpers + "\n\ndef _prepare_geographic_context(self, dataset, projection):", 1)

runtime = replace_once(
    runtime,
    '''def _prepare_geographic_context(self, dataset, projection):\n    self._modeler_projection = projection\n    result = _ORIGINAL_PREPARE_GEO(self, dataset, projection)\n''',
    '''def _prepare_geographic_context(self, dataset, projection):\n    self._modeler_projection = projection\n    # ``prepare()`` already resolves the full modeler style for every mapped\n    # footprint. Keep those exact requests so the later placement pass can reuse\n    # them instead of repeating country/material/window/door selection.\n    self._modeler_prepared_polygon_keys = {}\n    self._modeler_prepared_point_keys = {}\n    result = _ORIGINAL_PREPARE_GEO(self, dataset, projection)\n''',
    label="prepare cache initialization",
)

runtime = replace_once(
    runtime,
    '''    base = _ORIGINAL_KEY_FOR(\n        self,\n        tags,\n        width_m,\n        length_m,\n        foundation_depth_m=foundation_depth_m,\n        settlement_context=settlement_context,\n    )\n''',
    '''    prepared = getattr(_STYLE_STATE, "prepared_request", None)\n    if prepared is not None and foundation_depth_m is None:\n        prepared_key, prepared_width, prepared_length, prepared_context = prepared\n        requested_dimensions = sorted((float(width_m), float(length_m)))\n        prepared_dimensions = sorted((float(prepared_width), float(prepared_length)))\n        if (\n            str(settlement_context) == str(prepared_context)\n            and math.isclose(requested_dimensions[0], prepared_dimensions[0], rel_tol=0.0, abs_tol=1.0e-7)\n            and math.isclose(requested_dimensions[1], prepared_dimensions[1], rel_tol=0.0, abs_tol=1.0e-7)\n        ):\n            return prepared_key\n\n    base = _ORIGINAL_KEY_FOR(\n        self,\n        tags,\n        width_m,\n        length_m,\n        foundation_depth_m=foundation_depth_m,\n        settlement_context=settlement_context,\n    )\n''',
    label="prepared style fast path",
)

new_iter_and_plan = '''def _iter_dataset_keys(self, dataset, projection, point_footprint):
    polygon_cache = self._modeler_prepared_polygon_keys
    point_cache = self._modeler_prepared_point_keys
    for feature in dataset.building_polygons:
        for polygon in feature.polygons:
            projected = [projection.to_world(point) for point in polygon.outer[:-1]]
            if len(projected) < 3:
                continue
            projected_holes = tuple(
                tuple(projection.to_world(point) for point in ring[:-1])
                for ring in polygon.holes
                if len(ring) >= 4
            )
            rectangle = _pb._simple_rectangle_footprint(projected, projected_holes)
            if rectangle is not None:
                footprint, centre_x, centre_z = rectangle
            else:
                polygon_geometry, footprint = _pb._polygon_with_footprint(projected, projected_holes)
                centre = polygon_geometry.centroid
                centre_x, centre_z = float(centre.x), float(centre.y)
            context = self._settlement_context(centre_x, centre_z)
            with _style_location(self, centre_x, centre_z):
                key = self.key_for(
                    feature.tags,
                    footprint.width_m,
                    footprint.length_m,
                    settlement_context=context,
                )
            polygon_cache[_polygon_request_signature(
                feature.tags, projected, projected_holes
            )] = (key, footprint.width_m, footprint.length_m, context)
            yield key
    for feature in dataset.building_points:
        x, z = projection.to_world(feature.point)
        context = self._settlement_context(x, z)
        with _style_location(self, x, z):
            key = self.key_for(
                feature.tags,
                point_footprint,
                point_footprint,
                settlement_context=context,
            )
        point_cache[_point_request_signature(
            feature.tags, point_footprint, x, z
        )] = (key, point_footprint, point_footprint, context)
        yield key


def _plan_polygon(self, tags, points: Sequence[tuple[float, float]], **kwargs):
    holes = kwargs.get("holes", ()) or ()
    prepared = getattr(self, "_modeler_prepared_polygon_keys", {}).get(
        _polygon_request_signature(tags, points, holes)
    )
    if points:
        centre_x = sum(float(point[0]) for point in points) / len(points)
        centre_z = sum(float(point[1]) for point in points) / len(points)
    else:
        centre_x = centre_z = None
    with _style_location(self, centre_x, centre_z), _prepared_style_request(prepared):
        return _ORIGINAL_PLAN_POLYGON(self, tags, points, **kwargs)


def _plan_point(self, tags, footprint, heading_degrees, *, x=None, z=None, **kwargs):
    prepared = getattr(self, "_modeler_prepared_point_keys", {}).get(
        _point_request_signature(tags, footprint, x, z)
    )
    with _style_location(self, x, z), _prepared_style_request(prepared):
        return _ORIGINAL_PLAN_POINT(
            self,
            tags,
            footprint,
            heading_degrees,
            x=x,
            z=z,
            **kwargs,
        )


'''
runtime = replace_between(
    runtime,
    "def _iter_dataset_keys(self, dataset, projection, point_footprint):",
    "def _register_placement(self, placement, *, foundation_depth_m=None):",
    new_iter_and_plan,
    label="dataset key and placement wrappers",
)
runtime_path.write_text(runtime, encoding="utf-8", newline="\n")


osm_path = ROOT / "src" / "cwr_worldgen" / "osm.py"
osm = osm_path.read_text(encoding="utf-8")

new_priority = '''    # Classify each feature once. The old lazy implementation rescanned both
    # building collections for every one of the five priority bands, repeating
    # semantic tag work five times before actual placement even started.
    polygon_features_by_priority: list[list[OsmPolygonFeature]] = [[] for _ in range(5)]
    point_features_by_priority: list[list[OsmPointFeature]] = [[] for _ in range(5)]
    for feature in dataset.building_polygons:
        priority = _building_placement_priority(feature.tags)
        if 0 <= priority < 5:
            polygon_features_by_priority[priority].append(feature)
    for feature in dataset.building_points:
        priority = _building_placement_priority(feature.tags)
        if 0 <= priority < 5:
            point_features_by_priority[priority].append(feature)

    def candidates_for_priority(priority: int):
        # Input feature groups are already sorted by OSM key, so bucketing above
        # preserves the previous deterministic ordering inside each band.
        polygons = (
            (feature.osm_key, polygon_index, "polygon", feature, polygon)
            for feature in polygon_features_by_priority[priority]
            for polygon_index, polygon in enumerate(feature.polygons)
        )
        points = (
            (feature.osm_key, 0, "point", feature, feature.point)
            for feature in point_features_by_priority[priority]
        )
        yield from heapq.merge(
            polygons, points, key=lambda item: (item[0], item[1], item[2])
        )

    candidates = (
        (priority, *candidate)
        for priority in range(5)
        for candidate in candidates_for_priority(priority)
    )
'''
osm = replace_between(
    osm,
    "    def candidates_for_priority(priority: int):",
    "    progress(4, f\"Streaming {candidate_total:,} mapped building candidates by priority\")",
    new_priority,
    label="single priority classification pass",
)

osm = replace_once(
    osm,
    '''    progress_interval = max(1, candidate_total // 40)\n    for candidate_number, (_priority, osm_key, geometry_index, geometry_kind, feature, geometry) in enumerate(candidates, start=1):\n''',
    '''    progress_interval = max(1, candidate_total // 40)\n    polygon_planner = (\n        getattr(building_asset_library, "plan_polygon", None)\n        if building_asset_library is not None else None\n    )\n    point_planner = (\n        getattr(building_asset_library, "plan_point", None)\n        if building_asset_library is not None else None\n    )\n    for candidate_number, (_priority, osm_key, geometry_index, geometry_kind, feature, geometry) in enumerate(candidates, start=1):\n''',
    label="planner binding",
)

osm = replace_once(
    osm,
    '''                planner = getattr(building_asset_library, "plan_polygon", None)\n                procedural_placement = (\n                    planner(\n''',
    '''                procedural_placement = (\n                    polygon_planner(\n''',
    label="polygon planner binding",
)
osm = replace_once(
    osm,
    '''                    if planner is not None\n                    else building_asset_library.place_polygon(\n''',
    '''                    if polygon_planner is not None\n                    else building_asset_library.place_polygon(\n''',
    label="polygon planner condition",
)
osm = replace_once(
    osm,
    '''                planner = getattr(building_asset_library, "plan_point", None)\n                procedural_placement = (\n                    planner(\n''',
    '''                procedural_placement = (\n                    point_planner(\n''',
    label="point planner binding",
)
osm = replace_once(
    osm,
    '''                    if planner is not None\n                    else building_asset_library.place_point(\n''',
    '''                    if point_planner is not None\n                    else building_asset_library.place_point(\n''',
    label="point planner condition",
)
osm_path.write_text(osm, encoding="utf-8", newline="\n")


test_path = ROOT / "tests" / "test_building_resolve_performance.py"
test_path.write_text('''from __future__ import annotations

from pathlib import Path

from cwr_worldgen import osm as osm_module
from cwr_worldgen import osm_house_modeler_runtime as runtime
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import (
    BboxProjection,
    GeoPolygon,
    OsmDataset,
    OsmPointFeature,
    OsmPolygonFeature,
    OsmRaster,
    plan_building_placements,
)
from cwr_worldgen.procedural_buildings import ProceduralBuildingLibrary


def _empty_dataset(**overrides):
    values = dict(
        source_generator="building-resolve-performance",
        element_count=0,
        coastlines=(),
        water=(),
        forests=(),
        farmland=(),
        urban=(),
        roads=(),
    )
    values.update(overrides)
    return OsmDataset(**values)


def _empty_raster(cells: int) -> OsmRaster:
    empty = (False,) * (cells * cells)
    return OsmRaster(cells, empty, empty, empty, empty, empty, empty, cells, 0)


def _rectangle_feature(projection, index: int, x: float, z: float) -> OsmPolygonFeature:
    width = 8.0 + (index % 4) * 2.0
    length = 10.0 + (index % 5) * 2.0
    half_w = width * 0.5
    half_l = length * 0.5
    world = (
        (x - half_w, z - half_l),
        (x + half_w, z - half_l),
        (x + half_w, z + half_l),
        (x - half_w, z + half_l),
        (x - half_w, z - half_l),
    )
    outer = tuple(projection.to_latlon(point) for point in world)
    return OsmPolygonFeature(
        f"way/resolve-{index}",
        {"building": "house", "addr:country": "SE"},
        (GeoPolygon(outer),),
    )


def test_prepared_modeler_style_keys_are_reused_during_rectangular_placement(monkeypatch) -> None:
    bbox = (59.0, 18.0, 59.03, 18.05)
    projection = BboxProjection.create(bbox, 2400.0)
    buildings = tuple(
        _rectangle_feature(
            projection,
            index,
            120.0 + (index % 6) * 260.0,
            120.0 + (index // 6) * 330.0,
        )
        for index in range(24)
    )
    dataset = _empty_dataset(element_count=len(buildings), building_polygons=buildings)

    calls = 0
    original = runtime.resolve_style

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime, "resolve_style", counted)
    library = ProceduralBuildingLibrary(
        world_name="resolve_perf",
        maximum_variants=128,
        texture_variants=1,
    )
    library.prepare(dataset, projection, 8.0)
    prepare_calls = calls
    assert prepare_calls >= len(buildings)

    projected_requests = []
    for feature in buildings:
        polygon = feature.polygons[0]
        points = tuple(projection.to_world(point) for point in polygon.outer[:-1])
        projected_requests.append((feature, points))
        library.plan_polygon(feature.tags, points)

    # These exact rectangular footprints were fully resolved during prepare(), so
    # placement must not repeat country/material/window/door style resolution.
    assert calls == prepare_calls

    # The cache is an optimization only. Removing one prepared entry must produce
    # the exact same requested/selected model while proving the fallback still
    # performs a real style resolution when no prepared request is available.
    feature, points = projected_requests[0]
    cached = library.plan_polygon(feature.tags, points)
    library._modeler_prepared_polygon_keys.clear()
    before_fallback = calls
    fallback = library.plan_polygon(feature.tags, points)
    assert calls == before_fallback + 1
    assert fallback.requested == cached.requested
    assert fallback.selected == cached.selected
    assert fallback.heading_degrees == cached.heading_degrees


def test_building_priority_is_classified_once_per_feature(monkeypatch) -> None:
    bbox = (0.0, 0.0, 0.01, 0.01)
    cells = 8
    projection = BboxProjection.create(bbox, cells * 25.0)
    points = tuple(
        OsmPointFeature(
            f"node/priority-{index}",
            {"building": ("house", "shed", "school", "apartments", "barn")[index % 5]},
            projection.to_latlon((20.0 + (index % 5) * 32.0, 20.0 + (index // 5) * 32.0)),
        )
        for index in range(25)
    )
    dataset = _empty_dataset(element_count=len(points), building_points=points)
    spec = _Milestone9PlayabilitySpec(
        heightmap_path=Path("unused.png"),
        bbox=bbox,
        cells=cells,
        cell_size=25.0,
        max_buildings=100,
        max_forest_objects=0,
        max_road_objects=0,
        residential_infill_enabled=False,
        strict_assets=False,
    )

    calls = 0
    original = osm_module._building_placement_priority

    def counted(tags):
        nonlocal calls
        calls += 1
        return original(tags)

    monkeypatch.setattr(osm_module, "_building_placement_priority", counted)
    plans, truncated = plan_building_placements(
        dataset, projection, _empty_raster(cells), spec
    )
    assert not truncated
    assert len(plans) == len(points)
    assert calls == len(points)
''', encoding="utf-8", newline="\n")

print("Applied mapped building resolve performance optimization")
