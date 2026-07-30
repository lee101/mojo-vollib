import inspect

from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks import analytical, numerical


def test_upstream_scalar_signatures_are_preserved():
    expected = "(flag, S, K, t, r, sigma)"
    assert str(inspect.signature(black_scholes)) == expected
    for module in (analytical, numerical):
        for name in ("delta", "gamma", "theta", "vega", "rho"):
            assert str(inspect.signature(getattr(module, name))) == expected
