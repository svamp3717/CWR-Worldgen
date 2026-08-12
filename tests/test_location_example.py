# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from PIL import Image

from cwr_worldgen.location_example import (
    HgtTile,
    LocationExampleSpec,
    _write_heightmap,
    parse_opentopomap_link,
    required_hgt_tiles,
    reference_zoom_for_bbox,
    square_bbox,
)


class LocationExampleTests(unittest.TestCase):
    def test_location_example_defaults_use_25m_cells_for_a_6_4km_world(self) -> None:
        spec = LocationExampleSpec(
            map_url="https://opentopomap.org/#map=13/59.42766/16.89457",
            output_dir=Path("build"),
            cache_dir=Path("cache"),
        )
        self.assertEqual(spec.cells, 256)
        self.assertEqual(spec.cell_size, 25.0)
        self.assertEqual(spec.world_size, 6400.0)

    def test_parses_supplied_link_and_exact_square_bbox(self) -> None:
        link = parse_opentopomap_link("https://opentopomap.org/#map=13/59.42766/16.89457")
        self.assertEqual(link.zoom, 13)
        self.assertAlmostEqual(link.latitude, 59.42766)
        self.assertAlmostEqual(link.longitude, 16.89457)
        bbox = square_bbox(link.latitude, link.longitude, 6400.0)
        expected = (
            59.398881748360814,
            16.837989602089053,
            59.45643825163919,
            16.95115039791095,
        )
        for actual, wanted in zip(bbox, expected):
            self.assertAlmostEqual(actual, wanted, places=12)
        self.assertEqual(required_hgt_tiles(bbox), ("N59E016",))

    def test_hgt_bilinear_sampling_and_north_up_tiff(self) -> None:
        # 3x3 HGT grid, north row first. Values rise eastward and southward.
        values = (100, 110, 120, 200, 210, 220, 300, 310, 320)
        payload = b"".join(value.to_bytes(2, "big", signed=True) for value in values)
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "N59E016.hgt.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("N59E016.hgt", payload)
            tile = HgtTile.from_zip(archive_path, "N59E016")
            self.assertAlmostEqual(tile.sample(59.5, 16.5), 210.0)
            path = Path(directory) / "height.tif"
            minimum, maximum = _write_heightmap(
                path,
                (59.0, 16.0, 60.0, 17.0),
                3,
                {"N59E016": tile},
            )
            self.assertAlmostEqual(minimum, 166.66666666666666)
            self.assertAlmostEqual(maximum, 313.3333333333333)
            with Image.open(path) as image:
                self.assertEqual(image.mode, "F")
                self.assertEqual(image.size, (3, 3))
                # Samples are stored WRP vertices and rows remain north-up.
                self.assertAlmostEqual(float(image.getpixel((0, 0))), 166.66666666666666, places=4)
                self.assertAlmostEqual(float(image.getpixel((2, 2))), 313.3333333333333, places=4)


    def test_reference_zoom_can_be_derived_from_bbox(self) -> None:
        bbox = square_bbox(59.42766, 16.89457, 12800.0)
        zoom = reference_zoom_for_bbox(bbox)
        self.assertGreaterEqual(zoom, 2)
        self.assertLessEqual(zoom, 16)

    def test_rejects_non_opentopomap_link(self) -> None:
        with self.assertRaises(ValueError):
            parse_opentopomap_link("https://example.com/#map=13/59/16")


if __name__ == "__main__":
    unittest.main()
