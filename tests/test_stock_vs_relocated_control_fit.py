"""Does changing only the stock road PBO model prefix change WRP placement?

The real, unmodified stock ODOL sil25 P3D works for the user when its path
is cwr_paved_stock_control\roads\sil25.p3d. The stock o\road\sil25.p3d
is special-cased in the generator. Isolate source-model selection from the
generated WRP object geometry before trying any visual flag alterations.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from cwr_worldgen import playability
from cwr_worldgen.milestone9 import _Milestone9PlayabilitySpec
from cwr_worldgen.osm import BboxProjection, OsmDataset, OsmLineFeature


STOCK = r"o\road\sil25.p3d"
RELOCATED = r"cwr_paved_stock_control\roads\sil25.p3d"


def _scene():
    bbox = (59.40, 16.82, 59.41, 16.83)
    projection = BboxProjection.create(bbox, 1000.0)
    junction = (500.0, 500.0)
    dataset = OsmDataset(
        source_generator="stock-versus-identical-ODOL-namespace",
        element_count=2,
        coastlines=(), water=(), forests=(), farmland=(), urban=(),
        roads=(
            OsmLineFeature(
                "way/asphalt", {"highway": "residential", "surface": "asphalt"},
                tuple(projection.to_latlon(point) for point in (
                    (500., 200.), junction, (500., 800.),
                )),
            ),
            OsmLineFeature(
                "way/gravel", {"highway": "track", "surface": "gravel"},
                tuple(projection.to_latlon(point) for point in (
                    junction, (800., 550.),
                )),
            ),
        ),
    )
    return bbox, projection, dataset


def _fit(model: str, *, custom_shapes: bool):
    bbox, projection, dataset = _scene()
    spec = _Milestone9PlayabilitySpec(
        name="stockcontrolfit",
        heightmap_path=Path("unused.png"),
        bbox=bbox, cells=40, cell_size=25.0,
        max_road_objects=10000, strict_assets=False,
        paved_road_model=model,
        custom_road_shapes=custom_shapes,
    )
    key = playability._road_model_key(model)
    dimensions = {
        key: (9.1, 25.0),
        key[:-6] + "12.p3d": (9.1, 12.0),
        key[:-6] + "6.p3d": (9.1, 6.0),
    } if model == RELOCATED else {}
    paths = frozenset((key, key[:-6] + "12.p3d", key[:-6] + "6.p3d"))
    t_dims = playability._ROAD_MODEL_DIMENSIONS.set(dimensions or None)
    t_available = playability._ROAD_MODEL_VARIANTS_AVAILABLE.set({key: paths})
    t_effective = playability._ROAD_MODEL_EFFECTIVE_DONORS.set({key: model})
    try:
        # Reproduce the uphill/downhill slope that caused gravel/paved
        # projection problems in real CWA instead of testing only flat land.
        elevations = [
            9.0 + col * 0.015 + row * 0.03
            for row in range(spec.cells)
            for col in range(spec.cells)
        ]
        return playability.fit_road_objects(
            dataset, projection, elevations, spec,
        )
    finally:
        playability._ROAD_MODEL_EFFECTIVE_DONORS.reset(t_effective)
        playability._ROAD_MODEL_VARIANTS_AVAILABLE.reset(t_available)
        playability._ROAD_MODEL_DIMENSIONS.reset(t_dims)


def _fingerprint(report):
    return tuple(sorted((
        obj.model_path.replace("cwr_paved_stock_control\\roads\\", "o\\road\\").casefold(),
        round(obj.x, 5), round(obj.y, 5), round(obj.z, 5),
        round(obj.heading_degrees, 5), round(obj.pitch_degrees, 5),
    ) for obj in report.objects))


@pytest.mark.parametrize("custom_shapes", (False, True))
def test_identical_stock_and_relocated_donors_do_not_move_the_junction(
    custom_shapes: bool,
):
    stock = _fit(STOCK, custom_shapes=custom_shapes)
    control = _fit(RELOCATED, custom_shapes=custom_shapes)
    before = _fingerprint(stock)
    after = _fingerprint(control)
    print(
        "PATH_DIAGNOSTIC", {
            "custom_shapes": custom_shapes,
            "stock_objects": len(before),
            "control_objects": len(after),
            "only_stock": list(sorted(set(before) - set(after)))[:12],
            "only_control": list(sorted(set(after) - set(before)))[:12],
        },
    )
    assert before == after
