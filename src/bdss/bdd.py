"""BDD backend: the only module that imports `dd`.

`dd.cudd` is the fast backend; when it cannot be imported (build failure, missing library) we
fall back to the pure-Python `dd.autoref`, which has the same high-level API.
"""

from bdss.config import BDD_BACKEND


def _import_backend(name):
    if name == "autoref":
        from dd import autoref

        return autoref
    from dd import cudd

    return cudd


if BDD_BACKEND == "autoref":
    _backend = _import_backend("autoref")
    BACKEND = "autoref"
else:
    try:
        _backend = _import_backend("cudd")
        BACKEND = "cudd"
    except Exception:  # CUDD may fail at import time even when the wheel installed
        _backend = _import_backend("autoref")
        BACKEND = "autoref"


def new_manager():
    """Return a fresh BDD manager of the selected backend."""
    return _backend.BDD()
