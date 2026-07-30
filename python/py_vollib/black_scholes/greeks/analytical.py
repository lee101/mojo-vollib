"""Analytical Black-Scholes Greeks with py-vollib scaling conventions."""

from __future__ import annotations

from py_vollib._lib import lib
from py_vollib.black_scholes import _flag_value


def _call(name, flag, S, K, t, r, sigma):
    return getattr(lib(), name)(
        _flag_value(flag), float(S), float(K), float(t), float(r), float(sigma)
    )


def delta(flag, S, K, t, r, sigma):
    return _call("mv_delta", flag, S, K, t, r, sigma)


def gamma(flag, S, K, t, r, sigma):
    return _call("mv_gamma", flag, S, K, t, r, sigma)


def theta(flag, S, K, t, r, sigma):
    return _call("mv_theta", flag, S, K, t, r, sigma)


def vega(flag, S, K, t, r, sigma):
    return _call("mv_vega", flag, S, K, t, r, sigma)


def rho(flag, S, K, t, r, sigma):
    return _call("mv_rho", flag, S, K, t, r, sigma)


__all__ = ["delta", "gamma", "theta", "vega", "rho"]
