from __future__ import annotations

import os
import subprocess

from cffi import FFI
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "vollib.mojo")
BUILD_SCRIPT = os.path.join(ROOT, "build", "build.sh")
LIB = os.environ.get("MOJO_VOLLIB_LIB") or os.path.join(
    ROOT, "dist", "libmojo-vollib.so"
)

_CDEF = """
double mv_black_scholes(
    long long, double, double, double, double, double);
double mv_delta(long long, double, double, double, double, double);
double mv_gamma(long long, double, double, double, double, double);
double mv_theta(long long, double, double, double, double, double);
double mv_vega(long long, double, double, double, double, double);
double mv_rho(long long, double, double, double, double, double);
void mv_black_scholes_batch(
    long long, long long, long long, long long,
    long long, long long, long long, long long);
void mv_greeks_batch(
    long long, long long, long long, long long,
    long long, long long, long long, long long,
    long long, long long, long long, long long);
"""


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if os.environ.get("MOJO_VOLLIB_LIB"):
        if not os.path.exists(LIB):
            raise BuildError(f"MOJO_VOLLIB_LIB does not exist: {LIB}")
        return LIB
    stale = not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC)
    if force or stale:
        proc = subprocess.run(
            ["bash", BUILD_SCRIPT],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if proc.returncode != 0 or not os.path.exists(LIB):
            raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_loaded = None


def lib():
    global _loaded
    if _loaded is None:
        ffi = FFI()
        ffi.cdef(_CDEF)
        _loaded = ffi.dlopen(build())
    return _loaded


def addr(array: np.ndarray) -> int:
    """Return an address only for buffers that satisfy the Mojo ABI contract."""
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    if not array.flags.aligned:
        raise ValueError("FFI buffers must be aligned")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise ValueError("non-empty FFI buffers must have a non-null address")
    return address
