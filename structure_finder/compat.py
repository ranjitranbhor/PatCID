#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility shims for third-party engines on modern runtimes.

Two shims, both for DECIMER-Segmentation on a modern runtime: numpy >= 2,
and Keras 3.

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

import os

_APPLIED = False
_LEGACY_KERAS_ENV = "TF_USE_LEGACY_KERAS"
_LEGACY_KERAS_ACTIVE = False


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


# ---------------------------------------------------------------------------
# Keras 3
# ---------------------------------------------------------------------------


class LegacyKerasUnavailable(RuntimeError):
    """Raised when Keras 2 is needed but cannot be activated any more."""


def enable_legacy_keras() -> bool:
    """Make ``tf.keras`` resolve to Keras 2, which mrcnn's weight loader needs.

    ``mrcnn/model.py`` loads the Mask R-CNN checkpoint with::

        from tensorflow.python.keras.saving import hdf5_format
        hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)

    That code requires every layer weight to be a ``tf.Variable``.  Under Keras 3
    (the default from TensorFlow 2.16) layer weights are ``keras.Variable``
    instead, so loading the published ``.h5`` fails with::

        NotImplementedError: Save or restore weights that is not an instance of
        `tf.Variable` is not supported in h5, use `save_format='tf'` instead.

    Setting ``TF_USE_LEGACY_KERAS=1`` (with the ``tf-keras`` package installed)
    points ``tf.keras`` back at Keras 2, where that path works.  Verified on
    TensorFlow 2.19.1: ``load_weights(by_name=True)`` succeeds under legacy Keras
    and fails under Keras 3.

    TensorFlow reads the variable when it is first imported, so this must run
    **before** ``import tensorflow``.

    Returns:
        True if legacy Keras is active (or was already), False if TensorFlow is
        not installed at all.

    Raises:
        LegacyKerasUnavailable: TensorFlow was already imported under Keras 3
            (the session must be restarted), or ``tf-keras`` is missing.
    """
    global _LEGACY_KERAS_ACTIVE
    import sys

    if _LEGACY_KERAS_ACTIVE:
        return True

    already_imported = "tensorflow" in sys.modules
    os.environ[_LEGACY_KERAS_ENV] = "1"

    try:
        import tensorflow as tf
    except ImportError:
        return False

    # Two signals, because neither alone is reliable:
    #
    #  * tf.keras.__name__ says "tf_keras..." only until something does
    #    `import tensorflow.keras` - which mrcnn/model.py does. That binds a
    #    real tensorflow.keras module and the name reverts to
    #    "tensorflow.keras" even though Keras 2 is still underneath.
    #  * tf_keras in sys.modules is what TensorFlow itself imports when it
    #    honours TF_USE_LEGACY_KERAS, so it survives that rebinding.
    #
    # The result is then memoised: once legacy Keras is in force for this
    # interpreter it cannot silently revert, and re-deriving it would hit the
    # name problem above.
    name = getattr(getattr(tf, "keras", None), "__name__", "")
    if name.startswith("tf_keras") or "tf_keras" in sys.modules:
        _LEGACY_KERAS_ACTIVE = True
        return True

    # tf.keras is still Keras 3, so the switch did not take effect.
    try:
        import tf_keras  # noqa: F401
    except ImportError as error:
        raise LegacyKerasUnavailable(
            "DECIMER-Segmentation needs Keras 2 to load its .h5 weights, but "
            "the 'tf-keras' package is not installed.\n"
            "  pip install tf-keras\n"
            "then restart the Python session."
        ) from error

    raise LegacyKerasUnavailable(
        "DECIMER-Segmentation needs Keras 2 to load its .h5 weights. "
        "'tf-keras' is installed, but TensorFlow was "
        + ("already imported before TF_USE_LEGACY_KERAS could be set"
           if already_imported else "imported under Keras 3")
        + ".\nTensorFlow reads that variable at import time, so restart the "
        "session (in Colab: Runtime -> Restart session) and run again. "
        "Importing structure_finder before tensorflow sets it early enough."
    )
