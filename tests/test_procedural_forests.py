# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from pathlib import Path
import math
import struct
import tempfile
import unittest

from cwr_worldgen.procedural_buildings import inspect_mlod
from cwr_worldgen.procedural_forests import (
    DEFAULT_BORDER_PROXY_MODELS,
    DEFAULT_PROXY_MODELS,
    DEFAULT_UNDERGROWTH_PROXY_MODELS,
    EVERON_SAFE_BORDER_PROXY_MODELS,
    EVERON_SAFE_UNDERGROWTH_PROXY_MODELS,
    NOGOVA_LEAF_BORDER_PROXY_MODELS,
    NOGOVA_LEAF_PROXY_MODELS,
    NOGOVA_PINE_BORDER_PROXY_MODELS,
    NOGOVA_PINE_PROXY_MODELS,
    NOGOVA_BORDER_PROXY_MODELS,
    NOGOVA_PROXY_MODELS,
    ALL_FOREST_CLUSTER_VARIANTS,
    FOREST_CLUSTER_GRADES,
    FOREST_CLUSTER_VARIANTS,
    ProceduralForestClusterLibrary,
    cluster_model_path,
    write_forest_cluster_mlod,
)


class ProceduralForestClusterTests(unittest.TestCase):
    def test_interior_clusters_use_individual_everon_tree_models(self) -> None:
        self.assertEqual(
            DEFAULT_PROXY_MODELS,
            (
                r"data3d\str smrk_medium.p3d",
                r"data3d\str smrk vysoky.p3d",
            ),
        )
        self.assertFalse(any("\\les " in path.casefold() for path in DEFAULT_PROXY_MODELS))

    def test_border_and_undergrowth_use_original_data3d_vegetation(self) -> None:
        self.assertEqual(DEFAULT_UNDERGROWTH_PROXY_MODELS, DEFAULT_BORDER_PROXY_MODELS)
        self.assertTrue(DEFAULT_BORDER_PROXY_MODELS)
        self.assertTrue(all(path.casefold().startswith("data3d" + "\\") for path in DEFAULT_BORDER_PROXY_MODELS))
        self.assertFalse(any(path.casefold().startswith("o\\tree" + "\\") for path in DEFAULT_BORDER_PROXY_MODELS))

    def test_everon_safe_proxy_profile_excludes_suspect_bushes(self) -> None:
        library = ProceduralForestClusterLibrary(
            "cwr_cluster", proxy_profile="everon-safe"
        )
        library.register_models((
            cluster_model_path("cwr_cluster", "pine", 0.30),
            cluster_model_path("cwr_cluster", "border_thicket", 0.15),
            cluster_model_path("cwr_cluster", "undergrowth_patch", 0.15),
        ))
        models = {path.casefold() for path in library.required_proxy_models()}

        self.assertTrue(
            {path.casefold() for path in EVERON_SAFE_BORDER_PROXY_MODELS}
            .intersection(models)
        )
        self.assertTrue(
            {path.casefold() for path in EVERON_SAFE_UNDERGROWTH_PROXY_MODELS}
            .intersection(models)
        )
        self.assertNotIn(r"data3d\ker pichlavej.p3d", models)
        self.assertNotIn(r"data3d\ker deravej.p3d", models)
        self.assertTrue(any(path.startswith("data3d\\str ") for path in models))
        self.assertFalse(any(path.startswith("data3d\\les ") for path in models))

    def test_nogova_proxy_profile_remaps_forest_and_bush_clusters(self) -> None:
        library = ProceduralForestClusterLibrary("cwr_cluster", proxy_profile="nogova")
        library.register_models((
            cluster_model_path("cwr_cluster", "pine", 0.30),
            cluster_model_path("cwr_cluster", "border_thicket", 0.15),
        ))
        models = library.required_proxy_models()
        self.assertTrue(set(NOGOVA_PROXY_MODELS).intersection(models))
        self.assertTrue(set(NOGOVA_BORDER_PROXY_MODELS).intersection(models))
        self.assertFalse(any(path.casefold().startswith("data3d\\les ") for path in models))
        self.assertFalse(any(path.casefold().startswith("data3d\\ker ") for path in models))

    def test_nogova_pine_proxy_profile_uses_individual_resistance_pines(self) -> None:
        library = ProceduralForestClusterLibrary("cwr_cluster", proxy_profile="nogova_pine")
        library.register_models((
            cluster_model_path("cwr_cluster", "pine", 0.30),
            cluster_model_path("cwr_cluster", "border_thicket", 0.15),
        ))
        models = library.required_proxy_models()
        self.assertTrue(set(NOGOVA_PINE_PROXY_MODELS).intersection(models))
        self.assertTrue(set(NOGOVA_PINE_BORDER_PROXY_MODELS).intersection(models))
        self.assertFalse(any(path.casefold().startswith("data3d" + "\\") for path in models))

    def test_nogova_leaf_proxy_profile_uses_individual_resistance_trees(self) -> None:
        library = ProceduralForestClusterLibrary("cwr_cluster", proxy_profile="nogova_leaf")
        library.register_models((
            cluster_model_path("cwr_cluster", "pine", 0.30),
            cluster_model_path("cwr_cluster", "border_thicket", 0.15),
        ))
        models = library.required_proxy_models()
        self.assertTrue(set(NOGOVA_LEAF_PROXY_MODELS).intersection(models))
        self.assertTrue(set(NOGOVA_LEAF_BORDER_PROXY_MODELS).intersection(models))
        self.assertFalse(any(path.casefold().startswith("data3d" + "\\") for path in models))

    def test_serialized_proxy_markers_are_unambiguous_to_cwa_199(self) -> None:
        def squared_distance(a, b):
            return sum((a[index] - b[index]) ** 2 for index in range(3))

        def legacy_proxy_axes(p0, p1, p2):
            dist01 = squared_distance(p0, p1)
            dist02 = squared_distance(p0, p2)
            dist12 = squared_distance(p1, p2)

            if dist01 > dist02:
                p1, p2 = p2, p1
                dist01, dist02 = dist02, dist01
            if dist01 > dist12:
                p0, p2 = p2, p0
                dist01, dist12 = dist12, dist01
            if dist02 > dist12:
                p0, p1 = p1, p0

            direction = tuple(p1[index] - p0[index] for index in range(3))
            up_candidate = tuple(p2[index] - p0[index] for index in range(3))

            direction_length = math.sqrt(sum(value * value for value in direction))
            direction = tuple(value / direction_length for value in direction)
            projection = sum(direction[index] * up_candidate[index] for index in range(3))
            up = tuple(
                up_candidate[index] - direction[index] * projection
                for index in range(3)
            )
            up_length = math.sqrt(sum(value * value for value in up))
            up = tuple(value / up_length for value in up)
            return direction, up

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for variant in ALL_FOREST_CLUSTER_VARIANTS:
                for grade in FOREST_CLUSTER_GRADES:
                    path = root / f"{variant.name}_{int(round(grade * 100)):02d}.p3d"
                    write_forest_cluster_mlod(path, variant, grade)
                    data = path.read_bytes()

                    self.assertEqual(data[:4], b"MLOD")
                    offset = 12
                    self.assertEqual(data[offset:offset + 4], b"SP3X")
                    head_size = struct.unpack_from("<i", data, offset + 4)[0]
                    point_count = struct.unpack_from("<i", data, offset + 12)[0]
                    self.assertEqual(point_count, len(variant.proxy_layout) * 3)
                    offset += head_size

                    points = [
                        struct.unpack_from("<3f", data, offset + point_index * 16)
                        for point_index in range(point_count)
                    ]
                    for proxy_index in range(len(variant.proxy_layout)):
                        p0, p1, p2 = points[proxy_index * 3:proxy_index * 3 + 3]
                        direction, up = legacy_proxy_axes(p0, p1, p2)

                        # Exercise the float32 coordinates actually consumed by
                        # CWA 1.99. Direction must remain horizontal and the
                        # orthogonalized Up vector must remain vertical.
                        self.assertAlmostEqual(
                            direction[1],
                            0.0,
                            places=5,
                            msg=f"{variant.name} grade={grade} proxy={proxy_index}",
                        )
                        self.assertAlmostEqual(
                            abs(up[1]),
                            1.0,
                            places=5,
                            msg=f"{variant.name} grade={grade} proxy={proxy_index}",
                        )
                        self.assertAlmostEqual(
                            up[0],
                            0.0,
                            places=5,
                            msg=f"{variant.name} grade={grade} proxy={proxy_index}",
                        )
                        self.assertAlmostEqual(
                            up[2],
                            0.0,
                            places=5,
                            msg=f"{variant.name} grade={grade} proxy={proxy_index}",
                        )

    def test_cluster_model_contains_reusable_stock_proxies_and_support_lods(self) -> None:
        variant = FOREST_CLUSTER_VARIANTS[0]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cluster.p3d"
            write_forest_cluster_mlod(path, variant, 0.30)
            summary = inspect_mlod(path)
            self.assertEqual(summary.lod_count, 3)
            self.assertEqual(summary.resolutions[0], 1.0)
            self.assertGreater(summary.resolutions[1], 1.0e12)
            self.assertGreater(summary.resolutions[2], 1.0e15)
            proxy_names = tuple(
                name
                for lod_names in summary.selection_names
                for name in lod_names
                if name.casefold().startswith("proxy:")
            )
            self.assertEqual(len(proxy_names), len(variant.proxy_layout))
            self.assertTrue(all("data3d" in name.casefold() for name in proxy_names))
            self.assertTrue(all("af str" not in name.casefold() for name in proxy_names))
            self.assertTrue(all("\\str " in name.casefold() for name in proxy_names))
            self.assertFalse(any("\\les " in name.casefold() for name in proxy_names))
            # Generated proxy carriers must stay ordinary ObjectPlain-style
            # vegetation containers. Interior carriers now proxy only individual
            # trees, never another special forest-block object.
            self.assertIn(("class", "bushsoft"), summary.named_properties[1])
            self.assertNotIn(("class", "forest"), summary.named_properties[1])

    def test_content_addressed_cluster_assets_are_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = cluster_model_path("cwr_cluster", "pine", 0.30)

            first = ProceduralForestClusterLibrary(
                "cwr_cluster", cache_dir=root / "cache"
            )
            first.register_model(model)
            first_result = first.write_assets(root / "one", root / "one.json")
            self.assertEqual(first_result.cache_hits, 0)
            self.assertEqual(first_result.cache_misses, 1)

            second = ProceduralForestClusterLibrary(
                "cwr_cluster", cache_dir=root / "cache"
            )
            second.register_model(model)
            second_result = second.write_assets(root / "two", root / "two.json")
            self.assertEqual(second_result.cache_hits, 1)
            self.assertEqual(second_result.cache_misses, 0)
            self.assertEqual(
                (root / "one" / "f" / "c_pine_30.p3d").read_bytes(),
                (root / "two" / "f" / "c_pine_30.p3d").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
