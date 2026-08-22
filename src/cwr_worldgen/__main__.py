# SPDX-License-Identifier: GPL-3.0-or-later
import multiprocessing as _multiprocessing

_multiprocessing.freeze_support()

from .cli import main

raise SystemExit(main())
