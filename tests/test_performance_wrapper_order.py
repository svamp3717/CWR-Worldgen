from cwr_worldgen import bridge_source_water_policy
from cwr_worldgen import bridge_underlay_cleanup_policy
from cwr_worldgen import bridge_underlay_spatial_policy
from cwr_worldgen import final_building_road_clearance_policy
from cwr_worldgen import forest_primary_parallel_policy
from cwr_worldgen import forest_vector_performance_policy
from cwr_worldgen import object_stage_parallel_policy
from cwr_worldgen import playability
from cwr_worldgen import road_chain_parallel_policy
from cwr_worldgen import road_finish_parallel_policy
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


def test_bridge_underlay_cleanup_uses_spatial_candidate_pruning() -> None:
    assert (
        bridge_underlay_cleanup_policy._remove_bridge_underlays
        is bridge_underlay_spatial_policy._remove_bridge_underlays
    )
    assert (
        bridge_underlay_cleanup_policy._add_terminal_underlays
        is bridge_underlay_spatial_policy._add_terminal_underlays
    )


def test_multicore_object_context_survives_late_building_wrapper() -> None:
    # The generic object-stage precompute stays immediately above vector forest,
    # while the primary-forest planner wraps it before the late building policy
    # captures the complete non-road generation chain.
    assert (
        object_stage_parallel_policy._BASE_GENERATE
        is forest_vector_performance_policy._generate_with_vector_forest_context
    )
    assert (
        forest_primary_parallel_policy._BASE_GENERATE
        is object_stage_parallel_policy._parallel_generate_world_objects
    )
    assert (
        final_building_road_clearance_policy._ORIGINAL_GENERATE_WORLD_OBJECTS
        is forest_primary_parallel_policy._parallel_generate
    )
    assert object_stage_parallel_policy._MAX_PRIMARY_JOBS == 0


def test_final_road_chain_finalizer_supersedes_only_road_half() -> None:
    assert playability._fit_stock_piece_road_objects is road_finish_parallel_policy._fit
    assert road_chain_parallel_policy._execute_run_jobs is road_finish_parallel_policy._execute
    assert playability._curved_gravel_model_for_run is road_finish_parallel_policy._cached_curved_model
    assert playability._road_object_on_slope is road_finish_parallel_policy._cached_road_object
    assert playability._model_axis is road_finish_parallel_policy._cached_model_axis
    assert road_finish_parallel_policy._BASE_STOCK_FIT is object_stage_parallel_policy._BASE_STOCK_FIT
    # bridge_source_water_policy used functools.wraps around the parallel base.
    assert (
        getattr(road_finish_parallel_policy._BASE_STOCK_FIT, "__wrapped__", None)
        is road_chain_parallel_policy._fit_stock_piece_road_objects_parallel
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
