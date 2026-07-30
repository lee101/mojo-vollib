import numpy as np
import pytest

from py_vollib.black_scholes import black_scholes_batch, greeks_batch
from py_vollib.black_scholes.vectorized import _inputs
from vollib.black_scholes import black_scholes as upstream_black_scholes
from vollib.black_scholes.greeks import analytical as upstream_analytical


@pytest.fixture(scope="module")
def portfolio():
    rng = np.random.default_rng(42)
    n = 40_003
    return (
        np.where(rng.random(n) < 0.5, "c", "p"),
        rng.uniform(10.0, 400.0, n),
        rng.uniform(10.0, 400.0, n),
        rng.uniform(0.01, 4.0, n),
        rng.uniform(-0.05, 0.2, n),
        rng.uniform(0.03, 1.2, n),
    )


def test_batch_price_matches_upstream(portfolio):
    got = black_scholes_batch(*portfolio)
    expected = np.array(
        [upstream_black_scholes(*args) for args in zip(*portfolio)]
    )
    assert np.allclose(got, expected, rtol=5e-10, atol=5e-11)


@pytest.mark.parametrize("name", ["delta", "gamma", "theta", "vega", "rho"])
def test_batch_greeks_match_upstream(portfolio, name):
    got = greeks_batch(*portfolio)[name]
    function = getattr(upstream_analytical, name)
    expected = np.array([function(*args) for args in zip(*portfolio)])
    assert np.allclose(got, expected, rtol=5e-6, atol=2e-9)


def test_batch_broadcasting():
    strikes = np.array([80.0, 100.0, 120.0])
    got = black_scholes_batch("c", 100.0, strikes, 0.5, 0.02, 0.25)
    expected = np.array(
        [upstream_black_scholes("c", 100, K, 0.5, 0.02, 0.25) for K in strikes]
    )
    assert got.shape == (3,)
    assert np.allclose(got, expected, rtol=5e-10, atol=5e-12)


def test_batch_multidimensional_shape_and_mixed_flags():
    flags = np.array([["c"], ["p"]])
    strikes = np.array([[90.0, 100.0, 110.0]])
    got = black_scholes_batch(flags, 100, strikes, 1.0, 0.01, 0.2)
    assert got.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            assert got[i, j] == pytest.approx(
                upstream_black_scholes(
                    flags[i, 0], 100, strikes[0, j], 1.0, 0.01, 0.2
                ),
                rel=5e-10,
            )


def test_greeks_batch_returns_all_named_arrays():
    result = greeks_batch(["c", "p"], 100, [90, 110], 0.5, 0.01, 0.2)
    assert tuple(result) == ("delta", "gamma", "theta", "vega", "rho")
    assert all(value.shape == (2,) for value in result.values())
    assert all(value.dtype == np.float64 for value in result.values())


def test_batch_accepts_noncontiguous_inputs():
    base = np.linspace(60, 140, 20)
    spots = base[::2]
    got = black_scholes_batch("c", spots, 100, 1, 0.01, 0.2)
    expected = np.array(
        [upstream_black_scholes("c", S, 100, 1, 0.01, 0.2) for S in spots]
    )
    assert np.allclose(got, expected, rtol=5e-10, atol=5e-12)


@pytest.mark.parametrize("n", range(1, 34))
def test_simd_tail_matches_upstream(n):
    data = (
        np.where(np.arange(n) % 2 == 0, "c", "p"),
        np.linspace(80.0, 140.0, n),
        np.linspace(85.0, 115.0, n),
        np.linspace(0.1, 2.0, n),
        np.linspace(-0.01, 0.08, n),
        np.linspace(0.1, 0.7, n),
    )
    got = greeks_batch(*data)
    for name, values in got.items():
        function = getattr(upstream_analytical, name)
        expected = np.array([function(*args) for args in zip(*data)])
        assert np.allclose(values, expected, rtol=5e-6, atol=2e-9)


@pytest.mark.parametrize("n", [65_535, 65_536])
def test_parallel_threshold_paths_match_upstream(n):
    indices = np.array([0, n // 3, n // 2, n - 1])
    flags = np.where(np.arange(n) % 2 == 0, "c", "p")
    spots = np.linspace(50.0, 150.0, n)
    strikes = np.linspace(70.0, 130.0, n)
    times = np.linspace(0.05, 2.0, n)
    rates = np.linspace(-0.01, 0.08, n)
    sigmas = np.linspace(0.1, 0.8, n)
    data = (flags, spots, strikes, times, rates, sigmas)
    prices = black_scholes_batch(*data)
    greeks = greeks_batch(*data)

    for i in indices:
        args = tuple(value[i] for value in data)
        assert prices[i] == pytest.approx(
            upstream_black_scholes(*args), rel=5e-10, abs=5e-12
        )
        for name, values in greeks.items():
            assert values[i] == pytest.approx(
                getattr(upstream_analytical, name)(*args),
                rel=5e-6,
                abs=2e-9,
            )


def test_contiguous_float64_inputs_remain_zero_copy():
    n = 8
    flags = np.array(["c", "p"] * 4)
    inputs = [
        np.linspace(80.0 + offset, 120.0 + offset, n)
        for offset in range(5)
    ]
    encoded_flags, values, shape = _inputs(flags, *inputs)
    assert shape == (n,)
    assert encoded_flags.dtype == np.uint8
    assert all(np.shares_memory(value, source) for value, source in zip(values, inputs))


def test_batch_scalar_shape():
    price = black_scholes_batch("c", 100, 100, 0.5, 0.01, 0.2)
    greeks = greeks_batch("c", 100, 100, 0.5, 0.01, 0.2)
    assert price.shape == ()
    assert all(value.shape == () for value in greeks.values())


def test_batch_empty_input():
    result = black_scholes_batch([], [], [], [], [], [])
    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_invalid_batch_flag_is_rejected():
    with pytest.raises(ValueError, match="flag"):
        black_scholes_batch(["c", "x"], 100, 100, 1, 0.01, 0.2)


@pytest.mark.parametrize(
    "value",
    [
        np.array([100 + 2j], dtype=np.complex128),
        np.array([100], dtype=np.longdouble),
        np.array(["100"]),
    ],
)
def test_batch_rejects_inputs_that_would_narrow_or_coerce_silently(value):
    with pytest.raises(TypeError, match="real numeric|wider than float64"):
        black_scholes_batch("c", value, 100, 1, 0.01, 0.2)


def test_temporary_broadcast_buffers_live_through_the_ffi_call():
    prices = black_scholes_batch(
        np.array([[["c"]], [["p"]]]),
        np.arange(12, dtype=np.float32).reshape(1, 3, 4)[:, ::-1] + 100,
        np.array([[90.0, 100.0, 110.0, 120.0]]),
        0.5,
        0.01,
        0.2,
    )
    assert prices.shape == (2, 3, 4)
    assert np.isfinite(prices).all()
