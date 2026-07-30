# mojo-vollib

`mojo-vollib` is a Mojo port of the Black-Scholes pricing and Greeks subset of
[`py-vollib`](https://github.com/vollib/py_vollib). It keeps the historical scalar
import paths and function signatures, while adding broadcasting batch APIs for
portfolios where compiled, parallel execution matters.

The covered scalar API is:

- `py_vollib.black_scholes.black_scholes`
- `py_vollib.black_scholes.greeks.analytical.{delta,gamma,theta,vega,rho}`
- `py_vollib.black_scholes.greeks.numerical.{delta,gamma,theta,vega,rho}`

The analytical functions preserve py-vollib's conventions: theta is per calendar
day, and vega and rho are per one-percentage-point change. The numerical module
preserves upstream's finite-difference steps and expiry limits.

This release does not cover implied volatility, Black-76, Black-Scholes-Merton,
dividend yield, or py-vollib's helper and LetsBeRational modules. The vectorized
functions are useful extensions, not upstream API.

## Install

The repository pins the tested Mojo nightly and installs the real upstream
`py-vollib` package for parity testing:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` creates `dist/libmojo-vollib.so`.

## Usage

The upstream-compatible scalar API:

```python
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.greeks.analytical import delta, gamma

price = black_scholes("c", 100, 90, 0.5, 0.01, 0.2)
call_delta = delta("c", 100, 90, 0.5, 0.01, 0.2)
call_gamma = gamma("c", 100, 90, 0.5, 0.01, 0.2)

print(price)       # 12.111581434969676
print(call_delta)  # 0.8026368118270925
print(call_gamma)  # 0.019638495951377066
```

For portfolios, inputs follow NumPy broadcasting rules. Calls and puts can be
mixed in the same batch:

```python
import numpy as np
from py_vollib.black_scholes import black_scholes_batch, greeks_batch

flags = np.array(["c", "p", "c"])
strikes = np.array([90.0, 100.0, 110.0])

prices = black_scholes_batch(flags, 100.0, strikes, 0.5, 0.01, 0.2)
greeks = greeks_batch(flags, 100.0, strikes, 0.5, 0.01, 0.2)
print(prices)
print(greeks["delta"])
```

## Correctness

The test suite compares every covered scalar function and both batch kernels
against the installed canonical `vollib` implementation behind current
`py-vollib`. It uses randomized valid-domain portfolios, published py-vollib
vectors, Hull textbook values, put-call parity, expiry behavior, broadcasting,
mixed flags, empty arrays, and non-contiguous inputs.

On the current pinned environment:

```text
79 passed
```

The Mojo normal CDF uses `erfc`, while upstream analytical Greeks use their own
CDF approximation. Small last-digit differences are therefore expected; tested
absolute and relative tolerances cover that approximation gap without masking
formula errors.

## Benchmarks

Measured with `pixi run bench` on:

```text
Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz | Linux x86_64
```

| benchmark | mojo-vollib | upstream py-vollib | speedup |
| --- | ---: | ---: | ---: |
| scalar price x 100k | 76.72 ms | 550.04 ms | 7.17x |
| batch price x 250k | 14.70 ms | 1279.90 ms | 87.07x |
| all 5 Greeks x 100k | 12.36 ms | 2629.10 ms | 212.78x |

These are best-of-three wall-clock measurements on identical inputs. The scalar
row includes one CFFI crossing per price. The batch rows compare a single Mojo
batch call with upstream's scalar API applied across the same portfolio; the
large gain comes from removing Python call overhead, SIMD and parallel
execution, and computing shared Greek intermediates once.

No GPU path is shipped.

Run the benchmark on another machine with:

```bash
pixi run bench
```

## How it works

All kernels live in one Mojo compilation unit. Scalar exports calculate a single
price or Greek. Batch exports process contiguous `float64` arrays, parallelizing
portfolios of at least 65,536 options across physical CPU cores. The five-Greek
kernel uses the native `float64` SIMD width with unaligned-safe loads and stores,
then handles the remainder in a scalar tail. It shares `d1`, `d2`, the normal
density, the discount factor, and square root of time in one pass.

Python owns every allocation. The wrapper broadcasts inputs with NumPy, converts
them to C-contiguous `float64` arrays plus a one-byte call/put flag array, and
passes their addresses through a CFFI ABI as integers. Already contiguous
`float64` inputs remain zero-copy. Complex, non-numeric, and wider-than-`float64`
inputs are rejected instead of being narrowed silently. Mojo reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values inside the exports and writes
into NumPy-owned output arrays. No Mojo allocation or ownership crosses the FFI.

Set `MOJO_VOLLIB_LIB=/absolute/path/to/libmojo-vollib.so` to load a prebuilt
library. Otherwise the wrapper rebuilds `dist/libmojo-vollib.so` when the Mojo
source is newer.

## License

MIT.
