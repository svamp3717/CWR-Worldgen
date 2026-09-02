# SPDX-License-Identifier: GPL-3.0-or-later
"""OFP/CWA world generation utilities."""

from dataclasses import replace as _dataclass_replace
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
from . import paved_junction_policy as _paved_junction_policy
from .paved_junction_policy import install_paved_junction_policy as _install_paved_junction_policy

_install_paved_junction_policy()

# Keep the selected merge endpoint, but remove any older paved slab that survives
# between the fixed 30 m clear zone and that endpoint. This is intentionally a
# thin wrapper so the previous junction implementation remains one-commit
# rollbackable while Lundby testing settles the final cleanup rule.
_original_paved_junction_apply = _paved_junction_policy._apply_plans


def _apply_paved_junctions_without_premerge_slabs(report, plans, elevations, spec):
    applied = _original_paved_junction_apply(report, plans, elevations, spec)
    if applied is report or not plans:
        return applied

    applications = []
    used_caps = set()
    protected_cap_ids = set()
    for key in sorted(plans):
        plan = plans[key]
        cap_index = _paved_junction_policy._cap_index(report, plan, used_caps)
        if cap_index is None:
            continue
        choices = _paved_junction_policy._plan_application(report, plan, spec)
        if choices is None:
            continue
        used_caps.add(cap_index)
        protected_cap_ids.add(report.objects[cap_index].object_id)
        applications.append((plan, choices))
    if not applications:
        return applied

    protected_ids = protected_cap_ids | {
        target.object_id
        for _plan, choices in applications
        for _score, target, _choice in choices
    }
    remove_ids = set()
    # Scan caps as well as ordinary chain pieces. A mixed/nearby base junction
    # can leave a plain sil6 cap inside the paved approach corridor; Lundby68's
    # object 94 is exactly that case. The actual T/X caps are protected above.
    for obj in report.objects:
        if obj.object_id in protected_ids:
            continue
        axis = _paved_junction_policy._object_axis(obj, spec)
        if axis is None:
            continue
        midpoint = (
            (axis[0][0] + axis[1][0]) * 0.5,
            (axis[0][1] + axis[1][1]) * 0.5,
        )
        for plan, choices in applications:
            for arm, (_score, target, _choice) in zip(plan.arms, choices):
                sx, sz = arm.source_direction
                target_along = (
                    (target.point[0] - plan.point[0]) * sx
                    + (target.point[1] - plan.point[1]) * sz
                )
                if target_along <= _paved_junction_policy._CLEAR_RADIUS + 0.20:
                    continue
                samples = (axis[0], midpoint, axis[1])
                along = tuple(
                    (point[0] - plan.point[0]) * sx
                    + (point[1] - plan.point[1]) * sz
                    for point in samples
                )
                if (
                    max(along) <= _paved_junction_policy._CLEAR_RADIUS - 0.20
                    or min(along) >= target_along - 0.20
                ):
                    continue
                lateral = min(
                    abs(
                        (point[0] - plan.point[0]) * sz
                        - (point[1] - plan.point[1]) * sx
                    )
                    for point in samples
                )
                if lateral <= 6.0:
                    remove_ids.add(obj.object_id)
                    break
            if obj.object_id in remove_ids:
                break

    if not remove_ids:
        return applied
    return _dataclass_replace(
        applied,
        objects=tuple(
            obj for obj in applied.objects if obj.object_id not in remove_ids
        ),
    )


_paved_junction_policy._apply_plans = _apply_paved_junctions_without_premerge_slabs

from .gravel_junction_policy import install_gravel_junction_policy as _install_gravel_junction_policy

_install_gravel_junction_policy()
from .gravel_gap_policy import install_gravel_gap_policy as _install_gravel_gap_policy

_install_gravel_gap_policy()
from .gravel_family_policy import install_gravel_family_policy as _install_gravel_family_policy

_install_gravel_family_policy()
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
from .postbuild_cleanup import install_postbuild_cleanup as _install_postbuild_cleanup

_install_postbuild_cleanup()
from .network import install_overture_release_resolution as _install_overture_release_resolution

_install_overture_release_resolution()
from .procedural_buildings import (
    BuildingGenerationResult,
    BuildingVariantKey,
    ProceduralBuildingLibrary,
    inspect_mlod,
    write_building_mlod,
)
from .osm_house_modeler_runtime import (
    install_osm_house_modeler_upgrade as _install_osm_house_modeler_upgrade,
)

_install_osm_house_modeler_upgrade()
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
