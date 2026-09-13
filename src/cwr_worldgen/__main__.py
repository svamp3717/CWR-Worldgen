# SPDX-License-Identifier: GPL-3.0-or-later
import multiprocessing as _multiprocessing

_multiprocessing.freeze_support()

from . import cli as _cli
from . import milestone9 as _milestone9

# Package initialization installs several Milestone 9 policy wrappers. cli.py
# imports build_milestone9 by value, so refresh that cached reference only after
# package initialization has finished. This keeps `python -m cwr_worldgen` on the
# final wrapped build path, including terrain ReadMe generation/deployment.
_cli.build_milestone9 = _milestone9.build_milestone9

raise SystemExit(_cli.main())
