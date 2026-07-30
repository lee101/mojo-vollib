"""Black-Scholes pricing with the historical py-vollib signature."""

from __future__ import annotations

from py_vollib._lib import lib

_black_scholes = lib().mv_black_scholes


def _flag_value(flag: str) -> int:
    if flag == "c":
        return 1
    if flag == "p":
        return 0
    raise ValueError("flag must be 'c' or 'p'")


def black_scholes(flag, S, K, t, r, sigma):
    """Return the Black-Scholes price of a European call or put."""
    return _black_scholes(
        _flag_value(flag), float(S), float(K), float(t), float(r), float(sigma)
    )


from py_vollib.black_scholes.vectorized import (  # noqa: E402
    black_scholes_batch,
    greeks_batch,
)

__all__ = ["black_scholes", "black_scholes_batch", "greeks_batch"]
