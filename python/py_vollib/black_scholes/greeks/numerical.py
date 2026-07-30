"""Finite-difference Greeks matching py-vollib's conventions."""

from __future__ import annotations

from py_vollib.black_scholes import black_scholes

_DS = 0.01


def delta(flag, S, K, t, r, sigma):
    if S == 0 and K == 0:
        raise ZeroDivisionError("delta undefined for S=0, K=0")
    if S == 0:
        return 0.0 if flag == "c" else -1.0
    if t == 0.0:
        if S == K:
            return {"c": 0.5, "p": -0.5}[flag]
        if S > K:
            return {"c": 1.0, "p": 0.0}[flag]
        return {"c": 0.0, "p": -1.0}[flag]
    upper = black_scholes(flag, S * (1 + _DS), K, t, r, sigma)
    lower = black_scholes(flag, S * (1 - _DS), K, t, r, sigma)
    return (upper - lower) / (2 * S * _DS)


def theta(flag, S, K, t, r, sigma):
    if t <= 1.0 / 365.0:
        return black_scholes(flag, S, K, 0.00001, r, sigma) - black_scholes(
            flag, S, K, t, r, sigma
        )
    return black_scholes(
        flag, S, K, t - 1.0 / 365.0, r, sigma
    ) - black_scholes(flag, S, K, t, r, sigma)


def vega(flag, S, K, t, r, sigma):
    return (
        black_scholes(flag, S, K, t, r, sigma + 0.01)
        - black_scholes(flag, S, K, t, r, sigma - 0.01)
    ) / 2.0


def rho(flag, S, K, t, r, sigma):
    return (
        black_scholes(flag, S, K, t, r + 0.01, sigma)
        - black_scholes(flag, S, K, t, r - 0.01, sigma)
    ) / 2.0


def gamma(flag, S, K, t, r, sigma):
    if S == 0 and K == 0:
        raise ZeroDivisionError("gamma undefined for S=0, K=0")
    if S == 0:
        return 0.0
    if t == 0:
        return float("inf") if S == K else 0.0
    upper = black_scholes(flag, S * (1 + _DS), K, t, r, sigma)
    center = black_scholes(flag, S, K, t, r, sigma)
    lower = black_scholes(flag, S * (1 - _DS), K, t, r, sigma)
    return (upper - 2.0 * center + lower) / (S * _DS) ** 2


__all__ = ["delta", "gamma", "theta", "vega", "rho"]
