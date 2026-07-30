import numpy as np
import pytest

from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks import analytical, numerical
from vollib.black_scholes import black_scholes as upstream_black_scholes
from vollib.black_scholes.greeks import analytical as upstream_analytical
from vollib.black_scholes.greeks import numerical as upstream_numerical


def cases(seed=7, count=250):
    rng = np.random.default_rng(seed)
    flags = np.where(rng.random(count) < 0.5, "c", "p")
    spots = rng.uniform(5.0, 500.0, count)
    strikes = rng.uniform(5.0, 500.0, count)
    times = rng.uniform(0.005, 5.0, count)
    rates = rng.uniform(-0.1, 0.25, count)
    sigmas = rng.uniform(0.02, 1.5, count)
    return list(zip(flags, spots, strikes, times, rates, sigmas))


@pytest.mark.parametrize("flag", ["c", "p"])
def test_price_matches_upstream(flag):
    for _, S, K, t, r, sigma in cases():
        got = black_scholes(flag, S, K, t, r, sigma)
        expected = upstream_black_scholes(flag, S, K, t, r, sigma)
        assert got == pytest.approx(expected, rel=5e-10, abs=5e-12)


@pytest.mark.parametrize("name", ["delta", "gamma", "theta", "vega", "rho"])
def test_analytical_greek_matches_upstream(name):
    ours = getattr(analytical, name)
    upstream = getattr(upstream_analytical, name)
    for args in cases(count=200):
        assert ours(*args) == pytest.approx(
            upstream(*args), rel=5e-6, abs=2e-9
        )


@pytest.mark.parametrize("name", ["delta", "gamma", "theta", "vega", "rho"])
def test_numerical_greek_matches_upstream(name):
    ours = getattr(numerical, name)
    upstream = getattr(upstream_numerical, name)
    for args in cases(count=80):
        assert ours(*args) == pytest.approx(upstream(*args), rel=1e-10, abs=1e-11)


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("c", 12.11158143496968), ("p", 1.6627045623120258)],
)
def test_published_upstream_price_vector(flag, expected):
    assert black_scholes(flag, 100, 90, 0.5, 0.01, 0.2) == pytest.approx(
        expected, abs=1e-12
    )


@pytest.mark.parametrize(
    ("name", "expected", "tolerance"),
    [
        ("delta", 0.522, 1e-3),
        ("gamma", 0.066, 1e-3),
        ("theta", -4.31 / 365.0, 2e-5),
        ("vega", 0.121, 1e-3),
        ("rho", 0.0891, 1e-4),
    ],
)
def test_hull_textbook_vectors(name, expected, tolerance):
    got = getattr(analytical, name)("c", 49, 50, 0.3846, 0.05, 0.2)
    assert got == pytest.approx(expected, abs=tolerance)


def test_put_call_parity():
    S, K, t, r, sigma = 123.0, 117.0, 1.7, 0.035, 0.41
    call = black_scholes("c", S, K, t, r, sigma)
    put = black_scholes("p", S, K, t, r, sigma)
    assert call - put == pytest.approx(S - K * np.exp(-r * t), abs=2e-14)


@pytest.mark.parametrize(
    ("flag", "S", "K", "expected"),
    [
        ("c", 110.0, 100.0, 10.0),
        ("p", 90.0, 100.0, 10.0),
        ("c", 90.0, 100.0, 0.0),
    ],
)
def test_expiry_price_matches_upstream(flag, S, K, expected):
    assert black_scholes(flag, S, K, 0.0, 0.05, 0.2) == expected
    assert upstream_black_scholes(flag, S, K, 0.0, 0.05, 0.2) == expected


def test_invalid_scalar_flag_is_rejected():
    with pytest.raises(ValueError, match="flag"):
        black_scholes("x", 100, 100, 1, 0.01, 0.2)


def test_numerical_expiry_limits():
    assert numerical.delta("c", 110, 100, 0, 0.01, 0.2) == 1.0
    assert numerical.delta("p", 90, 100, 0, 0.01, 0.2) == -1.0
    assert numerical.gamma("c", 100, 100, 0, 0.01, 0.2) == float("inf")
