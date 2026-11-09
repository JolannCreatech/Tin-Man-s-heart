"""Single place where Magenta RealTime is imported.

The environment variable must be set *before* ``magenta_rt.paths`` is imported,
therefore every DIASTOLE module imports Magenta through this shim instead of
importing ``magenta_rt`` directly.
"""

from __future__ import annotations

from diastole.config import ensure_magenta_home

ensure_magenta_home()

import mlx.core as mx  # noqa: E402
from magenta_rt import paths  # noqa: E402

__all__ = ["mx", "paths"]
