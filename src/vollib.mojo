from std.algorithm import parallelize
from std.math import abs, erfc, exp, log, sqrt
from std.sys.info import num_physical_cores, simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime INV_SQRT_TWO = 0.707106781186547524400844362104849039
comptime INV_SQRT_TWO_PI = 0.398942280401432677939946059934381868
comptime PARALLEL_THRESHOLD = 65536


@always_inline
def norm_cdf(x: Float64) -> Float64:
    return 0.5 * erfc(-x * INV_SQRT_TWO)


@always_inline
def norm_pdf(x: Float64) -> Float64:
    return INV_SQRT_TWO_PI * exp(-0.5 * x * x)


@always_inline
def norm_cdf_vector[W: Int](
    x: SIMD[DType.float64, W]
) -> SIMD[DType.float64, W]:
    var y = abs(x)
    var exponential = exp(-y * y / 2.0)
    var numerator = 0.0352624965998911 * y + 0.700383064443688
    numerator = numerator * y + 6.37396220353165
    numerator = numerator * y + 33.912866078383
    numerator = numerator * y + 112.079291497871
    numerator = numerator * y + 221.213596169931
    numerator = numerator * y + 220.206867912376
    var denominator = 0.0883883476483184 * y + 1.75566716318264
    denominator = denominator * y + 16.064177579207
    denominator = denominator * y + 86.7807322029461
    denominator = denominator * y + 296.564248779674
    denominator = denominator * y + 637.333633378831
    denominator = denominator * y + 793.826512519948
    denominator = denominator * y + 440.413735824752
    var central_tail = exponential * numerator / denominator
    var outer_denominator = y + 0.65
    outer_denominator = y + 4.0 / outer_denominator
    outer_denominator = y + 3.0 / outer_denominator
    outer_denominator = y + 2.0 / outer_denominator
    outer_denominator = y + 1.0 / outer_denominator
    var outer_tail = exponential / (outer_denominator * 2.506628274631)
    var tail = y.lt(7.07106781186547).select(central_tail, outer_tail)
    tail = y.gt(37.0).select(0.0, tail)
    return x.gt(0.0).select(1.0 - tail, tail)


@always_inline
def d_one(S: Float64, K: Float64, t: Float64, r: Float64, sigma: Float64) -> Float64:
    return (log(S / K) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt(t))


@always_inline
def price_value(
    is_call: Bool, S: Float64, K: Float64, t: Float64, r: Float64, sigma: Float64
) -> Float64:
    if t <= 0.0 or sigma <= 0.0:
        var discounted_strike = K * exp(-r * max(t, 0.0))
        if is_call:
            return max(S - discounted_strike, 0.0)
        return max(discounted_strike - S, 0.0)
    var root_t = sqrt(t)
    var d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * t) / (sigma * root_t)
    var d2 = d1 - sigma * root_t
    var discounted_strike = K * exp(-r * t)
    if is_call:
        return S * norm_cdf(d1) - discounted_strike * norm_cdf(d2)
    return discounted_strike * norm_cdf(-d2) - S * norm_cdf(-d1)


@always_inline
def delta_value(
    is_call: Bool, S: Float64, K: Float64, t: Float64, r: Float64, sigma: Float64
) -> Float64:
    var value = norm_cdf(d_one(S, K, t, r, sigma))
    return value if is_call else value - 1.0


@always_inline
def gamma_value(
    S: Float64, K: Float64, t: Float64, r: Float64, sigma: Float64
) -> Float64:
    var d1 = d_one(S, K, t, r, sigma)
    return norm_pdf(d1) / (S * sigma * sqrt(t))


@always_inline
def theta_value(
    is_call: Bool, S: Float64, K: Float64, t: Float64, r: Float64, sigma: Float64
) -> Float64:
    var root_t = sqrt(t)
    var d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * t) / (sigma * root_t)
    var d2 = d1 - sigma * root_t
    var first = -S * norm_pdf(d1) * sigma / (2.0 * root_t)
    var second = r * K * exp(-r * t)
    if is_call:
        return (first - second * norm_cdf(d2)) / 365.0
    return (first + second * norm_cdf(-d2)) / 365.0


@always_inline
def vega_value(
    S: Float64, K: Float64, t: Float64, r: Float64, sigma: Float64
) -> Float64:
    return S * norm_pdf(d_one(S, K, t, r, sigma)) * sqrt(t) * 0.01


@always_inline
def rho_value(
    is_call: Bool, S: Float64, K: Float64, t: Float64, r: Float64, sigma: Float64
) -> Float64:
    var d2 = d_one(S, K, t, r, sigma) - sigma * sqrt(t)
    var scale = t * K * exp(-r * t) * 0.01
    return scale * norm_cdf(d2) if is_call else -scale * norm_cdf(-d2)


@export("mv_black_scholes")
def mv_black_scholes(
    flag: Int, S: Float64, K: Float64, t: Float64, r: Float64, sigma: Float64
) abi("C") -> Float64:
    return price_value(flag == 1, S, K, t, r, sigma)


@export("mv_delta")
def mv_delta(
    flag: Int, S: Float64, K: Float64, t: Float64, r: Float64, sigma: Float64
) abi("C") -> Float64:
    return delta_value(flag == 1, S, K, t, r, sigma)


@export("mv_gamma")
def mv_gamma(
    flag: Int, S: Float64, K: Float64, t: Float64, r: Float64, sigma: Float64
) abi("C") -> Float64:
    return gamma_value(S, K, t, r, sigma)


@export("mv_theta")
def mv_theta(
    flag: Int, S: Float64, K: Float64, t: Float64, r: Float64, sigma: Float64
) abi("C") -> Float64:
    return theta_value(flag == 1, S, K, t, r, sigma)


@export("mv_vega")
def mv_vega(
    flag: Int, S: Float64, K: Float64, t: Float64, r: Float64, sigma: Float64
) abi("C") -> Float64:
    return vega_value(S, K, t, r, sigma)


@export("mv_rho")
def mv_rho(
    flag: Int, S: Float64, K: Float64, t: Float64, r: Float64, sigma: Float64
) abi("C") -> Float64:
    return rho_value(flag == 1, S, K, t, r, sigma)


def price_range(
    flags: BPtr,
    spots: FPtr,
    strikes: FPtr,
    times: FPtr,
    rates: FPtr,
    sigmas: FPtr,
    dst: FPtr,
    begin: Int,
    end: Int,
):
    for i in range(begin, end):
        dst[i] = price_value(
            flags[i] == 1, spots[i], strikes[i], times[i], rates[i], sigmas[i]
        )


def greeks_range(
    flags: BPtr,
    spots: FPtr,
    strikes: FPtr,
    times: FPtr,
    rates: FPtr,
    sigmas: FPtr,
    deltas: FPtr,
    gammas: FPtr,
    thetas: FPtr,
    vegas: FPtr,
    rhos: FPtr,
    begin: Int,
    end: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var i = begin
    var vector_end = end - (end - begin) % W
    while i < vector_end:
        var flag_values = flags.load[width=W](i)
        var spot_values = spots.load[width=W](i)
        var strike_values = strikes.load[width=W](i)
        var time_values = times.load[width=W](i)
        var rate_values = rates.load[width=W](i)
        var sigma_values = sigmas.load[width=W](i)
        var root_times = sqrt(time_values)
        var d1 = (
            log(spot_values / strike_values)
            + (rate_values + 0.5 * sigma_values * sigma_values) * time_values
        ) / (sigma_values * root_times)
        var d2 = d1 - sigma_values * root_times
        var pdf_d1 = INV_SQRT_TWO_PI * exp(-0.5 * d1 * d1)
        var discounts = exp(-rate_values * time_values)
        var cdf_d1 = norm_cdf_vector(d1)
        var call_mask = flag_values.eq(1)

        deltas.store(i, call_mask.select(cdf_d1, cdf_d1 - 1.0))
        gammas.store(i, pdf_d1 / (spot_values * sigma_values * root_times))
        var first = -spot_values * pdf_d1 * sigma_values / (2.0 * root_times)
        var rho_scales = time_values * strike_values * discounts * 0.01
        var cdf_d2 = norm_cdf_vector(d2)
        var cdf_negative_d2 = norm_cdf_vector(-d2)
        thetas.store(
            i,
            call_mask.select(
                (
                    first
                    - rate_values * strike_values * discounts * cdf_d2
                ) / 365.0,
                (
                    first
                    + rate_values * strike_values * discounts * cdf_negative_d2
                ) / 365.0,
            ),
        )
        vegas.store(i, spot_values * pdf_d1 * root_times * 0.01)
        rhos.store(
            i,
            call_mask.select(
                rho_scales * cdf_d2,
                -rho_scales * cdf_negative_d2,
            ),
        )
        i += W

    for scalar_i in range(i, end):
        var is_call = flags[scalar_i] == 1
        var S = spots[scalar_i]
        var K = strikes[scalar_i]
        var t = times[scalar_i]
        var r = rates[scalar_i]
        var sigma = sigmas[scalar_i]
        var root_t = sqrt(t)
        var d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * t) / (sigma * root_t)
        var d2 = d1 - sigma * root_t
        var pdf_d1 = norm_pdf(d1)
        var discount = exp(-r * t)
        var cdf_d1 = norm_cdf(d1)

        deltas[scalar_i] = cdf_d1 if is_call else cdf_d1 - 1.0
        gammas[scalar_i] = pdf_d1 / (S * sigma * root_t)
        var first = -S * pdf_d1 * sigma / (2.0 * root_t)
        var rho_scale = t * K * discount * 0.01
        if is_call:
            thetas[scalar_i] = (first - r * K * discount * norm_cdf(d2)) / 365.0
            rhos[scalar_i] = rho_scale * norm_cdf(d2)
        else:
            thetas[scalar_i] = (
                first + r * K * discount * norm_cdf(-d2)
            ) / 365.0
            rhos[scalar_i] = -rho_scale * norm_cdf(-d2)
        vegas[scalar_i] = S * pdf_d1 * root_t * 0.01


@export("mv_black_scholes_batch")
def mv_black_scholes_batch(
    flags_addr: Int,
    spots_addr: Int,
    strikes_addr: Int,
    times_addr: Int,
    rates_addr: Int,
    sigmas_addr: Int,
    dst_addr: Int,
    n: Int,
) abi("C"):
    if n <= 0:
        return
    var flags = BPtr(unsafe_from_address=flags_addr)
    var spots = FPtr(unsafe_from_address=spots_addr)
    var strikes = FPtr(unsafe_from_address=strikes_addr)
    var times = FPtr(unsafe_from_address=times_addr)
    var rates = FPtr(unsafe_from_address=rates_addr)
    var sigmas = FPtr(unsafe_from_address=sigmas_addr)
    var dst = FPtr(unsafe_from_address=dst_addr)

    if n < PARALLEL_THRESHOLD:
        price_range(flags, spots, strikes, times, rates, sigmas, dst, 0, n)
        return

    var workers = min(n, num_physical_cores())

    @parameter
    def work(worker: Int):
        var begin = worker * n // workers
        var end = (worker + 1) * n // workers
        price_range(flags, spots, strikes, times, rates, sigmas, dst, begin, end)

    parallelize[work](workers, workers)


@export("mv_greeks_batch")
def mv_greeks_batch(
    flags_addr: Int,
    spots_addr: Int,
    strikes_addr: Int,
    times_addr: Int,
    rates_addr: Int,
    sigmas_addr: Int,
    deltas_addr: Int,
    gammas_addr: Int,
    thetas_addr: Int,
    vegas_addr: Int,
    rhos_addr: Int,
    n: Int,
) abi("C"):
    if n <= 0:
        return
    var flags = BPtr(unsafe_from_address=flags_addr)
    var spots = FPtr(unsafe_from_address=spots_addr)
    var strikes = FPtr(unsafe_from_address=strikes_addr)
    var times = FPtr(unsafe_from_address=times_addr)
    var rates = FPtr(unsafe_from_address=rates_addr)
    var sigmas = FPtr(unsafe_from_address=sigmas_addr)
    var deltas = FPtr(unsafe_from_address=deltas_addr)
    var gammas = FPtr(unsafe_from_address=gammas_addr)
    var thetas = FPtr(unsafe_from_address=thetas_addr)
    var vegas = FPtr(unsafe_from_address=vegas_addr)
    var rhos = FPtr(unsafe_from_address=rhos_addr)

    if n < PARALLEL_THRESHOLD:
        greeks_range(
            flags,
            spots,
            strikes,
            times,
            rates,
            sigmas,
            deltas,
            gammas,
            thetas,
            vegas,
            rhos,
            0,
            n,
        )
        return

    var workers = min(n, num_physical_cores())

    @parameter
    def work(worker: Int):
        var begin = worker * n // workers
        var end = (worker + 1) * n // workers
        greeks_range(
            flags,
            spots,
            strikes,
            times,
            rates,
            sigmas,
            deltas,
            gammas,
            thetas,
            vegas,
            rhos,
            begin,
            end,
        )

    parallelize[work](workers, workers)
