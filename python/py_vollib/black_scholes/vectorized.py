"""Broadcasting array APIs for high-throughput pricing and Greeks."""

from __future__ import annotations

import numpy as np

from py_vollib._lib import addr, lib


def _float64_input(array):
    array = np.asarray(array)
    if array.dtype.kind not in "iuf":
        raise TypeError("S, K, t, r, and sigma must be real numeric values")
    if array.dtype.kind == "f" and array.dtype.itemsize > np.dtype(np.float64).itemsize:
        raise TypeError("floating-point inputs wider than float64 are not supported")
    return np.ascontiguousarray(array, dtype=np.float64)


def _inputs(flag, S, K, t, r, sigma):
    arrays = np.broadcast_arrays(
        np.asarray(flag), np.asarray(S), np.asarray(K), np.asarray(t), np.asarray(r),
        np.asarray(sigma),
    )
    raw_flags = arrays[0]
    if raw_flags.size:
        flags = np.equal(raw_flags, "c")
        if not np.all(flags | np.equal(raw_flags, "p")):
            raise ValueError("flag must contain only 'c' or 'p'")
        flags = np.ascontiguousarray(flags, dtype=np.uint8)
    else:
        flags = np.empty(raw_flags.shape, dtype=np.uint8)
    values = [_float64_input(array) for array in arrays[1:]]
    return flags, values, arrays[0].shape


def black_scholes_batch(flag, S, K, t, r, sigma):
    """Price broadcastable arrays of European calls and puts."""
    flags, values, shape = _inputs(flag, S, K, t, r, sigma)
    result = np.empty(shape, dtype=np.float64)
    if result.size:
        lib().mv_black_scholes_batch(
            addr(flags),
            *(addr(value) for value in values),
            addr(result),
            result.size,
        )
    return result


def greeks_batch(flag, S, K, t, r, sigma):
    """Return delta, gamma, theta, vega, and rho arrays in one kernel pass."""
    flags, values, shape = _inputs(flag, S, K, t, r, sigma)
    results = {
        name: np.empty(shape, dtype=np.float64)
        for name in ("delta", "gamma", "theta", "vega", "rho")
    }
    if results["delta"].size:
        lib().mv_greeks_batch(
            addr(flags),
            *(addr(value) for value in values),
            *(addr(result) for result in results.values()),
            results["delta"].size,
        )
    return results


__all__ = ["black_scholes_batch", "greeks_batch"]
