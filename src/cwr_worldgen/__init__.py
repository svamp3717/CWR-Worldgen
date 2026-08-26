# SPDX-License-Identifier: GPL-3.0-or-later
"""OFP/CWA world generation utilities."""

import os as _os
import sys as _sys
import warnings as _warnings

# CWR deliberately reuses dem-stitcher's persistent tile cache. The upstream
# library warns whenever that managed directory already exists, which is the
# expected state on every run after the first. Suppress only that exact warning
# from dem_stitcher.stitcher; all other DEM/network/raster warnings stay visible.
_warnings.filterwarnings(
    "ignore",
    message=r"^The directory.* exists; We are writing new files to this directory$",
    category=UserWarning,
    module=r"^dem_stitcher\.stitcher$",
)

from ._version import __version__
from .network import install_network_compatibility as _install_network_compatibility

_install_network_compatibility()
from .generator import BuildResult, build_milestone1, build_milestone2, build_milestone3, build_milestone4
from .model import HeightmapSpec, OsmSpec, PlayabilitySpec, WorldSpec
from .stock_utility_policy import install_stock_utility_policy as _install_stock_utility_policy

_install_stock_utility_policy()
from .road_quality_policy import install_road_quality_policy as _install_road_quality_policy

_install_road_quality_policy()
from .stock_road_curve_policy import install_stock_road_curve_policy as _install_stock_road_curve_policy

_install_stock_road_curve_policy()
from .stock_road_geometry_policy import install_stock_road_geometry_policy as _install_stock_road_geometry_policy

_install_stock_road_geometry_policy()
from .stock_road_transform_policy import install_stock_road_transform_policy as _install_stock_road_transform_policy

_install_stock_road_transform_policy()
# Fit rigid stock connectors in 3D after their measured planar geometry is known.
from .stock_road_3d_connector_policy import (
    install_stock_road_3d_connector_policy as _install_stock_road_3d_connector_policy,
)

_install_stock_road_3d_connector_policy()
from .gravel_junction_policy import install_gravel_junction_policy as _install_gravel_junction_policy

_install_gravel_junction_policy()
from .gravel_gap_policy import install_gravel_gap_policy as _install_gravel_gap_policy

_install_gravel_gap_policy()
from .gravel_family_policy import install_gravel_family_policy as _install_gravel_family_policy

_install_gravel_family_policy()
from .stock_road_junction_policy import install_stock_road_junction_policy as _install_stock_road_junction_policy

_install_stock_road_junction_policy()
from .stock_road_measured_junction_policy import (
    install_stock_road_measured_junction_policy as _install_stock_road_measured_junction_policy,
)

_install_stock_road_measured_junction_policy()
from .stock_road_skew_policy import install_stock_road_skew_policy as _install_stock_road_skew_policy

_install_stock_road_skew_policy()
# Generated gravel borrows stock dirt connector geometry, but not its brown surface.
from .gravel_asphalt_transition_policy import (
    install_gravel_asphalt_transition_policy as _install_gravel_asphalt_transition_policy,
)

_install_gravel_asphalt_transition_policy()
from .stock_road_connector_policy import install_stock_road_connector_policy as _install_stock_road_connector_policy

_install_stock_road_connector_policy()
# Measure the final fitted branch/cap geometry and bridge only connector gaps
# that still remain in the actual road-object report.
from .stock_road_surface_overlap_policy import (
    install_stock_road_surface_overlap_policy as _install_stock_road_surface_overlap_policy,
)

_install_stock_road_surface_overlap_policy()
from .raceway_policy import install_raceway_policy as _install_raceway_policy

_install_raceway_policy()

# Public Overpass servers sometimes all return transient 5xx/timeout errors at
# once. Install the bounded retry wrapper before milestone modules import the
# shared source pipeline so every GUI/CLI source fetch receives the same policy.
from .overpass_retry import install_overpass_retries as _install_overpass_retries

_install_overpass_retries()

from .milestone6 import Milestone6Spec, build_milestone6
from .milestone7 import Milestone7Spec, build_milestone7
from .milestone8 import Milestone8Spec, build_milestone8
from .milestone9 import Milestone9Spec, build_milestone9
from .milestone9_advisory_policy import (
    install_milestone9_advisory_policy as _install_milestone9_advisory_policy,
)

_install_milestone9_advisory_policy()
from .grid_default_policy import install_default_grid_policy as _install_default_grid_policy

_install_default_grid_policy()
from .network import install_overture_release_resolution as _install_overture_release_resolution

_install_overture_release_resolution()
from .procedural_buildings import (
    BuildingGenerationResult,
    BuildingVariantKey,
    ProceduralBuildingLibrary,
    inspect_mlod,
    write_building_mlod,
)
from .normalization import (
    NormalizationSpec,
    NormalizedBundle,
    load_normalized_dataset,
    normalize_source_bundle,
    validate_normalized_bundle,
)
from .source_pipeline import (
    FrozenSourceBundle,
    Milestone5Spec,
    SourceFetchSpec,
    build_milestone5,
    fetch_sources,
    load_source_bundle,
    validate_source_bundle,
)

# Milestone 9 historically used an @-prefixed staging directory. The GUI/tool
# launches set CWR_WORLDGEN_RUNTIME_DIR so generated files use a neutral build
# folder instead. Direct library calls retain the historical default for API
# compatibility unless their caller opts into the new runtime directory.
_milestone9_module = _sys.modules[__name__ + ".milestone9"]
_original_milestone9_build_milestone4 = _milestone9_module.build_milestone4


def _build_milestone9_with_configured_runtime(*args, **kwargs):
    runtime_dir = _os.environ.get("CWR_WORLDGEN_RUNTIME_DIR", "").strip()
    if runtime_dir and kwargs.get("mod_directory_name") == "@CWR-Milestone9":
        kwargs["mod_directory_name"] = runtime_dir
    return _original_milestone9_build_milestone4(*args, **kwargs)


_milestone9_module.build_milestone4 = _build_milestone9_with_configured_runtime

# Final worlds get a small human-readable reproduction note beside the PBO.
# Install this after the runtime-folder compatibility wrapper so the ReadMe sees
# the final generated Addons path and can mirror itself into an optional mod.
from .terrain_readme import install_milestone9_terrain_readme as _install_milestone9_terrain_readme

_install_milestone9_terrain_readme()

build_milestone9 = _milestone9_module.build_milestone9

# The coordinate-aware picker is GUI-only. Importing tkinter can legitimately
# fail on headless/library-only installations, so leave non-GUI use untouched.
try:
    from .map_picker_coords import (
        install_osm_area_picker_coordinate_controls as _install_osm_area_picker_coordinate_controls,
    )

    _install_osm_area_picker_coordinate_controls()
except ImportError:
    pass

__all__ = [
    "__version__",
    "BuildResult",
    "HeightmapSpec",
    "OsmSpec",
    "PlayabilitySpec",
    "WorldSpec",
    "FrozenSourceBundle",
    "Milestone5Spec",
    "SourceFetchSpec",
    "Milestone6Spec",
    "Milestone7Spec",
    "Milestone8Spec",
    "Milestone9Spec",
    "BuildingGenerationResult",
    "BuildingVariantKey",
    "ProceduralBuildingLibrary",
    "NormalizationSpec",
    "NormalizedBundle",
    "build_milestone1",
    "build_milestone2",
    "build_milestone3",
    "build_milestone4",
    "build_milestone5",
    "build_milestone6",
    "build_milestone7",
    "build_milestone8",
    "build_milestone9",
    "inspect_mlod",
    "write_building_mlod",
    "fetch_sources",
    "load_source_bundle",
    "validate_source_bundle",
    "normalize_source_bundle",
    "validate_normalized_bundle",
    "load_normalized_dataset",
]
