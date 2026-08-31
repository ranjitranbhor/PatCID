#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility shims for third-party engines on modern runtimes.

Currently one shim, for DECIMER-Segmentation under numpy >= 2.

The problem
-----------
``decimer_segmentation/optimized_complete_structure.py`` does, at import time::

    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

``np.VisibleDeprecationWarning`` was removed in numpy 2.0, so merely importing
the package raises ``AttributeError``.  Its metadata sidesteps this by capping
``tensorflow<=2.15.1`` (which in turn pins ``numpy<2``) — but TensorFlow
publishes no wheel for that range on CPython 3.12+, so on a modern runtime the
cap is unsatisfiable and pip reports ``ResolutionImpossible``.

On Colab (CPython 3.13) it is worse: numpy only supports 3.13 from 2.1, so
``numpy<2`` cannot be installed as a wheel at all.

The fix
-------
Restore the attribute before the package is imported.  The only use is the
``filterwarnings`` call above — a cosmetic suppression of numba warnings — so a
stand-in class is behaviourally equivalent: the filter simply never matches.
This is the whole of the numpy-2 incompatibility; a scan of the package finds
no other removed-in-numpy-2 attribute.

:func:`apply_numpy2_shims` is called automatically by the DECIMER segmenter
before it imports ``decimer_segmentation``, so callers normally need do
nothing.  It is idempotent and a no-op on numpy 1.x.
"""

from __future__ import annotations

_APPLIED = False


class VisibleDeprecationWarning(UserWarning):
    """Stand-in for the class numpy 2.0 removed.

    numpy's own ``VisibleDeprecationWarning`` also derived from ``UserWarning``,
    so code that filters or catches it behaves the same way.
    """


def apply_numpy2_shims() -> bool:
    """Restore numpy attributes that numpy 2.0 removed but engines still use.

    Returns:
        True if a shim was installed, False if none was needed (numpy 1.x, or
        already applied).
    """
    global _APPLIED
    if _APPLIED:
        return False

    import numpy as np

    if hasattr(np, "VisibleDeprecationWarning"):
        _APPLIED = True
        return False

    np.VisibleDeprecationWarning = VisibleDeprecationWarning  # type: ignore[attr-defined]
    _APPLIED = True
    return True


def numpy_major_version() -> int:
    import numpy as np

    return int(np.__version__.split(".")[0])
