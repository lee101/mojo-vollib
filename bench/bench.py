"""Benchmark mojo-vollib against upstream vollib on identical portfolios."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

from py_vollib.black_scholes import (  # noqa: E402
    black_scholes,
    black_scholes_batch,
    greeks_batch,
)
from vollib.black_scholes import black_scholes as upstream_price  # noqa: E402
from vollib.black_scholes.greeks import analytical as upstream_greeks  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def portfolio(n, seed=123):
    rng = np.random.default_rng(seed)
    return (
        np.where(rng.random(n) < 0.5, "c", "p"),
        rng.uniform(20.0, 250.0, n),
        rng.uniform(20.0, 250.0, n),
        rng.uniform(0.01, 3.0, n),
        rng.uniform(-0.03, 0.15, n),
        rng.uniform(0.05, 0.9, n),
    )


def upstream_price_batch(data):
    return [upstream_price(*args) for args in zip(*data)]


def upstream_all_greeks(data):
    return {
        name: [getattr(upstream_greeks, name)(*args) for args in zip(*data)]
        for name in ("delta", "gamma", "theta", "vega", "rho")
    }


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    scalar_args = ("c", 100.0, 105.0, 0.75, 0.03, 0.25)
    scalar_iterations = 100_000

    def mojo_scalar():
        total = 0.0
        for _ in range(scalar_iterations):
            total += black_scholes(*scalar_args)
        return total

    def upstream_scalar():
        total = 0.0
        for _ in range(scalar_iterations):
            total += upstream_price(*scalar_args)
        return total

    price_data = portfolio(250_000)
    greek_data = tuple(value[:100_000] for value in price_data)

    black_scholes(*scalar_args)
    black_scholes_batch(*price_data)
    greeks_batch(*greek_data)

    rows = []
    mojo_time = timeit(mojo_scalar)
    upstream_time = timeit(upstream_scalar)
    rows.append(("scalar price x 100k", mojo_time, upstream_time))

    mojo_time = timeit(lambda: black_scholes_batch(*price_data))
    upstream_time = timeit(lambda: upstream_price_batch(price_data))
    rows.append(("batch price x 250k", mojo_time, upstream_time))

    mojo_time = timeit(lambda: greeks_batch(*greek_data))
    upstream_time = timeit(lambda: upstream_all_greeks(greek_data))
    rows.append(("all 5 Greeks x 100k", mojo_time, upstream_time))

    print(f"Machine: {cpu_name()} | {platform.system()} {platform.machine()}")
    print()
    print("| benchmark | mojo-vollib | upstream py-vollib | speedup |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_time, upstream_time in rows:
        speedup = upstream_time / mojo_time
        print(
            f"| {name} | {mojo_time * 1e3:.2f} ms | "
            f"{upstream_time * 1e3:.2f} ms | {speedup:.2f}x |"
        )

if __name__ == "__main__":
    main()
