"""Link-level performance metrics: BER, SER, outage and spectral efficiency.

The rest of the repository scores a phase design by the *received SNR* it
delivers. That is the right quantity to optimize, but it is not the quantity a
communications paper reports: reviewers expect error-rate waterfalls, outage
curves and ergodic spectral efficiency against transmit SNR. This module turns
a set of realized channel gains into exactly those curves.

Convention used throughout
--------------------------
``rho``      transmit SNR, ``P_t / sigma^2`` (linear, unitless).
``gain``     channel power gain ``|h_eff|^2`` of the composite
             BS -> (RIS) -> user link, already including path loss.
``gamma``    instantaneous received SNR, ``rho * gain``.

Error rates are computed *semi-analytically*: conditioned on ``gamma`` the link
is an AWGN channel with coherent detection, so the conditional error
probability is known in closed form and only the average over channel
realizations needs sampling. This is exact (not an approximation of the fading
average) and reaches 1e-6 without the 1e8 symbol draws a pure Monte-Carlo run
would need. :func:`monte_carlo_ber` reproduces the same numbers by actually
transmitting symbols, and ``test_link_metrics.py`` checks that the two agree.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

# Reported error rates below this are indistinguishable from zero at any
# simulation size we could run, and plotting log10(0) is not useful.
BER_FLOOR = 1e-7

SUPPORTED_MODULATIONS = ("BPSK", "QPSK", "16QAM", "64QAM")


def qfunc(x: np.ndarray | float) -> np.ndarray:
    """Gaussian tail probability ``Q(x) = P(N(0,1) > x)``."""
    return 0.5 * erfc(np.asarray(x, dtype=np.float64) / np.sqrt(2.0))


def _qam_order(modulation: str) -> int:
    return {"16QAM": 16, "64QAM": 64}[modulation]


def awgn_ber(gamma: np.ndarray | float, modulation: str = "QPSK") -> np.ndarray:
    """Bit-error probability on an AWGN channel at symbol SNR ``gamma``.

    Args:
        gamma: Received SNR ``E_s/N_0`` in linear scale (any shape).
        modulation: One of :data:`SUPPORTED_MODULATIONS`.

    Returns:
        Array of bit-error probabilities, same shape as ``gamma``.

    Notes:
        BPSK and QPSK are exact. Square Gray-mapped QAM uses the standard
        ``BER ~ SER / log2(M)`` relation, which is tight above ~1e-2 and is what
        the QAM curves in the RIS literature report.
    """
    g = np.maximum(np.asarray(gamma, dtype=np.float64), 0.0)
    mod = modulation.upper()

    if mod == "BPSK":
        # E_s = E_b, so Q(sqrt(2 E_b/N_0)).
        return qfunc(np.sqrt(2.0 * g))
    if mod == "QPSK":
        # Gray-mapped QPSK: two orthogonal BPSK streams, E_b/N_0 = gamma/2.
        return qfunc(np.sqrt(g))
    if mod in ("16QAM", "64QAM"):
        m = _qam_order(mod)
        k = np.log2(m)
        c = 1.0 - 1.0 / np.sqrt(m)
        q = qfunc(np.sqrt(3.0 * g / (m - 1.0)))
        ser = 4.0 * c * q - 4.0 * (c ** 2) * (q ** 2)
        return np.clip(ser / k, 0.0, 0.5)

    raise ValueError(
        f"unsupported modulation {modulation!r}; expected one of {SUPPORTED_MODULATIONS}"
    )


def awgn_ser(gamma: np.ndarray | float, modulation: str = "QPSK") -> np.ndarray:
    """Symbol-error probability on an AWGN channel at symbol SNR ``gamma``."""
    g = np.maximum(np.asarray(gamma, dtype=np.float64), 0.0)
    mod = modulation.upper()

    if mod == "BPSK":
        return qfunc(np.sqrt(2.0 * g))
    if mod == "QPSK":
        q = qfunc(np.sqrt(g))
        return 2.0 * q - q ** 2
    if mod in ("16QAM", "64QAM"):
        m = _qam_order(mod)
        c = 1.0 - 1.0 / np.sqrt(m)
        q = qfunc(np.sqrt(3.0 * g / (m - 1.0)))
        return np.clip(4.0 * c * q - 4.0 * (c ** 2) * (q ** 2), 0.0, 1.0)

    raise ValueError(f"unsupported modulation {modulation!r}")


def average_ber(gains: np.ndarray, rho: float, modulation: str = "QPSK") -> float:
    """BER averaged over channel realizations at transmit SNR ``rho``.

    Args:
        gains: Channel power gains ``|h_eff|^2``, one per realization.
        rho: Transmit SNR ``P_t/sigma^2``, linear.
        modulation: See :func:`awgn_ber`.
    """
    gamma = rho * np.asarray(gains, dtype=np.float64)
    return float(np.mean(awgn_ber(gamma, modulation)))


def ber_curve(
    gains: np.ndarray,
    rho_db: np.ndarray,
    modulation: str = "QPSK",
    floor: float = BER_FLOOR,
) -> np.ndarray:
    """BER against a sweep of transmit SNR points, clipped at ``floor``."""
    gains = np.asarray(gains, dtype=np.float64)
    rho = 10.0 ** (np.asarray(rho_db, dtype=np.float64) / 10.0)
    # (R, S): one row per SNR point, one column per channel realization.
    gamma = rho[:, None] * gains[None, :]
    return np.maximum(awgn_ber(gamma, modulation).mean(axis=1), floor)


def ser_curve(
    gains: np.ndarray,
    rho_db: np.ndarray,
    modulation: str = "QPSK",
    floor: float = BER_FLOOR,
) -> np.ndarray:
    """SER against a sweep of transmit SNR points, clipped at ``floor``."""
    gains = np.asarray(gains, dtype=np.float64)
    rho = 10.0 ** (np.asarray(rho_db, dtype=np.float64) / 10.0)
    gamma = rho[:, None] * gains[None, :]
    return np.maximum(awgn_ser(gamma, modulation).mean(axis=1), floor)


def spectral_efficiency_curve(gains: np.ndarray, rho_db: np.ndarray) -> np.ndarray:
    """Ergodic spectral efficiency ``E[log2(1 + rho |h|^2)]`` in bits/s/Hz."""
    gains = np.asarray(gains, dtype=np.float64)
    rho = 10.0 ** (np.asarray(rho_db, dtype=np.float64) / 10.0)
    gamma = rho[:, None] * gains[None, :]
    return np.log2(1.0 + gamma).mean(axis=1)


def outage_curve(
    gains: np.ndarray, rho_db: np.ndarray, rate_threshold_bps_hz: float
) -> np.ndarray:
    """Outage probability ``P(log2(1 + gamma) < R_th)`` over the SNR sweep."""
    gains = np.asarray(gains, dtype=np.float64)
    rho = 10.0 ** (np.asarray(rho_db, dtype=np.float64) / 10.0)
    gamma = rho[:, None] * gains[None, :]
    rate = np.log2(1.0 + gamma)
    return (rate < rate_threshold_bps_hz).mean(axis=1)


def snr_at_target(
    rho_db: np.ndarray, metric: np.ndarray, target: float, decreasing: bool = True
) -> float:
    """Transmit SNR at which ``metric`` first crosses ``target``.

    Linear interpolation in (dB, log10 metric) space, which is where BER and
    outage curves are close to straight. Returns ``nan`` when the sweep never
    reaches the target, so a scheme that cannot hit 1e-4 within the swept range
    is reported as "not achieved" rather than silently extrapolated.
    """
    rho_db = np.asarray(rho_db, dtype=np.float64)
    m = np.asarray(metric, dtype=np.float64)
    ok = m > 0
    if ok.sum() < 2:
        return float("nan")
    x, y = rho_db[ok], np.log10(m[ok])
    t = np.log10(target)

    for i in range(len(x) - 1):
        lo, hi = y[i], y[i + 1]
        if (decreasing and lo >= t >= hi) or (not decreasing and lo <= t <= hi):
            if hi == lo:
                return float(x[i])
            frac = (t - lo) / (hi - lo)
            return float(x[i] + frac * (x[i + 1] - x[i]))
    return float("nan")


def monte_carlo_ber(
    gains: np.ndarray,
    rho: float,
    modulation: str = "QPSK",
    symbols_per_realization: int = 2000,
    rng: np.random.Generator | None = None,
) -> float:
    """BER measured by actually transmitting Gray-mapped symbols.

    Cross-check for the semi-analytic path. Coherent detection is modelled as
    perfect phase/amplitude equalization of the composite channel, so each
    realization is an AWGN link at SNR ``rho * gain``.
    """
    rng = rng or np.random.default_rng(0)
    mod = modulation.upper()
    gains = np.asarray(gains, dtype=np.float64)

    if mod == "BPSK":
        bits_per_axis, axes = 1, 1
    elif mod == "QPSK":
        bits_per_axis, axes = 1, 2
    elif mod in ("16QAM", "64QAM"):
        bits_per_axis, axes = int(np.log2(_qam_order(mod)) // 2), 2
    else:
        raise ValueError(f"unsupported modulation {modulation!r}")

    levels = 2 ** bits_per_axis
    pam = 2 * np.arange(levels) - (levels - 1)          # {-L+1, ..., L-1}
    # Unit average symbol energy over both axes.
    scale = np.sqrt(axes * np.mean(pam ** 2))
    # Gray code: index -> constellation position, so neighbours differ in 1 bit.
    gray = np.arange(levels) ^ (np.arange(levels) >> 1)
    inv_gray = np.argsort(gray)

    # Hamming weight of every possible per-axis label XOR.
    popcount = np.array([bin(v).count("1") for v in range(levels)], dtype=np.int64)

    bit_errors = 0
    bit_total = 0
    for gain in gains:
        sigma2 = 1.0 / (rho * gain)                      # noise var per symbol
        n = symbols_per_realization
        idx = rng.integers(0, levels, size=(n, axes))    # Gray label per axis
        tx = pam[inv_gray[idx]] / scale
        # Complex baseband noise has variance sigma2/2 on each real axis;
        # BPSK simply discards the quadrature axis rather than folding its
        # noise into the in-phase one.
        noise = rng.normal(0.0, np.sqrt(sigma2 / 2.0), size=(n, axes))
        rx = tx + noise
        # Nearest-PAM decision per axis.
        hard = np.argmin(np.abs(rx[..., None] * scale - pam[None, None, :]), axis=-1)
        decoded = gray[hard]
        bit_errors += int(popcount[(idx ^ decoded).astype(np.int64)].sum())
        bit_total += n * axes * bits_per_axis

    return bit_errors / bit_total
