"""ctypes loader for the optional native ETC1 core.

Lookup order:

1. a prebuilt shared object shipped next to this file (``_etc1*.so`` /
   ``.dylib`` / ``.dll``);
2. a previously compiled copy in the user cache directory
   (``$ETC1_NATIVE_CACHE`` or ``$XDG_CACHE_HOME/etc1_codec``);
3. an on-the-fly compile of ``_etc1.c`` with the system C compiler.

If none of the steps succeeds :func:`load` returns ``None`` and
:mod:`etc1.core` silently uses its pure-Python implementation instead.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

BACKEND = "python-fallback"

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "_etc1.c")


def _candidate_prebuilt():
    try:
        for name in sorted(os.listdir(_HERE)):
            if name.startswith("_etc1") and (
                name.endswith(".so") or name.endswith(".dylib") or name.endswith(".dll")
            ):
                return os.path.join(_HERE, name)
    except OSError:
        pass
    return None


def _cache_paths():
    cdir = os.environ.get("ETC1_NATIVE_CACHE")
    if cdir:
        base = cdir
    else:
        root = os.environ.get("XDG_CACHE_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache"
        )
        base = os.path.join(root, "etc1_codec")
    try:
        os.makedirs(base, exist_ok=True)
    except OSError:
        base = tempfile.gettempdir()
    try:
        with open(_SRC, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        digest = "nosrc"
    so = os.path.join(
        base, "_etc1_%d%d_%s.so" % (sys.version_info[0], sys.version_info[1], digest)
    )
    return so


def _compile(so):
    cc = (
        os.environ.get("CC")
        or shutil.which("cc")
        or shutil.which("gcc")
        or shutil.which("clang")
    )
    if not cc or not os.path.exists(_SRC):
        return False
    cmd = [cc, "-O3", "-fPIC", "-shared", "-std=c11", "-lm", _SRC, "-o", so]
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and os.path.exists(so)


def _configure(lib):
    pu8 = ctypes.POINTER(ctypes.c_ubyte)
    pd = ctypes.POINTER(ctypes.c_double)
    lib.etc1_init.argtypes = []
    lib.etc1_init.restype = None
    lib.etc1_encode_u64.argtypes = [ctypes.c_char_p]
    lib.etc1_encode_u64.restype = ctypes.c_uint64
    lib.etc1_decode_rgb.argtypes = [ctypes.c_char_p, pu8]
    lib.etc1_decode_rgb.restype = None
    lib.etc1_diff_valid.argtypes = [ctypes.c_char_p]
    lib.etc1_diff_valid.restype = ctypes.c_int
    lib.etc1_encode_values.argtypes = [pd, ctypes.c_char_p]
    lib.etc1_encode_values.restype = None
    lib.etc1_decode_values.argtypes = [ctypes.c_char_p, pd]
    lib.etc1_decode_values.restype = None
    return lib


def _try(path):
    if not path or not os.path.exists(path):
        return None
    try:
        lib = ctypes.CDLL(path)
        _configure(lib)
        lib.etc1_init()
        return lib
    except (OSError, AttributeError):
        return None


def _load():
    global BACKEND
    lib = _try(_candidate_prebuilt())
    if lib is None:
        so = _cache_paths()
        lib = _try(so)
        if lib is None and _compile(so):
            lib = _try(so)
    if lib is not None:
        BACKEND = "native"
    return lib


_NATIVE = _load()


def load():
    """Return the loaded ctypes library, or ``None`` when unavailable."""
    return _NATIVE


def backend():
    """Human readable name of the active backend."""
    return BACKEND
