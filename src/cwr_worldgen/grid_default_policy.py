# SPDX-License-Identifier: GPL-3.0-or-later
"""Application-wide default terrain grid for OFP/CWA world generation."""
from __future__ import annotations

import inspect

DEFAULT_TERRAIN_CELL_SIZE_METRES = 50.0


def _set_constructor_default(cls: type, parameter_name: str, value: object) -> None:
    """Replace one generated dataclass ``__init__`` default in-place."""
    init = cls.__init__
    signature = inspect.signature(init)
    positional = [
        parameter
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    names = [parameter.name for parameter in positional]
    if parameter_name not in names:
        return
    defaults = list(init.__defaults__ or ())
    first_default = len(positional) - len(defaults)
    index = names.index(parameter_name)
    if index >= first_default:
        defaults[index - first_default] = value
        init.__defaults__ = tuple(defaults)
        return
    kwdefaults = dict(init.__kwdefaults__ or {})
    if parameter_name in kwdefaults:
        kwdefaults[parameter_name] = value
        init.__kwdefaults__ = kwdefaults


def _patch_dataclass_defaults() -> None:
    from . import model
    from . import source_pipeline

    for name in (
        "WorldSpec",
        "HeightmapSpec",
        "OsmSpec",
        "PlayabilitySpec",
        "ConstraintPlayabilitySpec",
    ):
        cls = getattr(model, name, None)
        if isinstance(cls, type):
            _set_constructor_default(cls, "cell_size", DEFAULT_TERRAIN_CELL_SIZE_METRES)

    _set_constructor_default(
        source_pipeline.SourceFetchSpec,
        "cell_size",
        DEFAULT_TERRAIN_CELL_SIZE_METRES,
    )
    _set_constructor_default(
        source_pipeline.SourceRegridSpec,
        "cell_size",
        DEFAULT_TERRAIN_CELL_SIZE_METRES,
    )


def _patch_cli_parser_defaults() -> None:
    from . import cli as cli_module

    original = cli_module._parser
    if getattr(original, "_cwr_50m_default", False):
        return

    def parser_with_50m_default():
        parser = original()
        pending = [parser]
        while pending:
            current = pending.pop()
            for action in current._actions:
                if action.dest == "cell_size" and action.default == 25.0:
                    action.default = DEFAULT_TERRAIN_CELL_SIZE_METRES
                choices = getattr(action, "choices", None)
                if isinstance(choices, dict):
                    pending.extend(choices.values())
        return parser

    parser_with_50m_default._cwr_50m_default = True
    cli_module._parser = parser_with_50m_default


def _patch_gui_default() -> None:
    try:
        from . import gui as gui_module
    except ImportError:
        return
    gui_module.DEFAULT_GUI_CELL_SIZE_METRES = DEFAULT_TERRAIN_CELL_SIZE_METRES


def install_default_grid_policy() -> None:
    """Make 50 m the default while preserving every explicit cell-size choice."""
    _patch_dataclass_defaults()
    _patch_cli_parser_defaults()
    _patch_gui_default()
