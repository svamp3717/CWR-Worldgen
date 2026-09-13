from cwr_worldgen import bridge_source_water_policy
from cwr_worldgen import final_building_road_clearance_policy
from cwr_worldgen import forest_vector_performance_policy
from cwr_worldgen import playability
from cwr_worldgen import road_chain_parallel_policy
from cwr_worldgen import road_quality_parallel_compat_policy
from cwr_worldgen import shared_object_index_policy
from cwr_worldgen import stock_utility_policy
from cwr_worldgen import vegetation_clearance_policy


def test_parallel_road_fitter_sits_under_source_water_bridge_wrapper() -> None:
    assert (
        bridge_source_water_policy._ORIGINAL_STOCK_FIT
        is road_chain_parallel_policy._fit_stock_piece_road_objects_parallel
    )
    assert (
        playability._stock_piece_chain
        is road_quality_parallel_compat_policy._batched_quality_chain
    )


def test_vector_forest_context_survives_late_building_wrapper() -> None:
    # final_building_road_clearance_policy is installed later in package startup.
    # Its captured core must already contain the forest ContextVar wrapper or the
    # primary-grid STRtree batch would never activate in real GUI/CLI builds.
    assert (
        final_building_road_clearance_policy._ORIGINAL_GENERATE_WORLD_OBJECTS
        is forest_vector_performance_policy._generate_with_vector_forest_context
    )


def test_shared_postprocess_functions_are_live_after_package_startup() -> None:
    assert (
        stock_utility_policy._rewrite_stock_utilities
        is shared_object_index_policy._indexed_rewrite_stock_utilities
    )
    assert (
        vegetation_clearance_policy.filter_vegetation_objects
        is shared_object_index_policy._indexed_filter_vegetation_objects
    )
