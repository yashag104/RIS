"""Checks for the link-level metric layer (src/link_metrics.py).

The BER/outage/spectral-efficiency curves in the paper are produced by closed
forms rather than by transmitting symbols, so the closed forms are the thing
that has to be right. These tests pin them against Monte-Carlo simulation and
against the textbook values.
"""

import numpy as np
import pytest

from src.link_metrics import (
    average_ber,
    awgn_ber,
    awgn_ser,
    ber_curve,
    monte_carlo_ber,
    outage_curve,
    qfunc,
    snr_at_target,
    spectral_efficiency_curve,
)


def test_qfunc_known_values():
    assert qfunc(0.0) == pytest.approx(0.5)
    assert qfunc(1.0) == pytest.approx(0.158655, abs=1e-6)
    assert qfunc(2.0) == pytest.approx(0.0227501, abs=1e-7)


def test_bpsk_and_qpsk_are_textbook():
    # BPSK at 0 dB: Q(sqrt(2)).
    assert awgn_ber(1.0, "BPSK") == pytest.approx(qfunc(np.sqrt(2.0)))
    # Gray QPSK has the same BER as BPSK at equal Eb/N0, i.e. at twice Es/N0.
    assert awgn_ber(2.0, "QPSK") == pytest.approx(awgn_ber(1.0, "BPSK"))


def test_ber_is_monotone_decreasing_in_snr():
    gamma = np.logspace(-1, 2, 40)
    for mod in ("BPSK", "QPSK", "16QAM", "64QAM"):
        ber = awgn_ber(gamma, mod)
        assert np.all(np.diff(ber) <= 1e-12), mod
        assert np.all(ber <= 0.5) and np.all(ber >= 0.0), mod


def test_ser_at_least_ber():
    gamma = np.logspace(-1, 2, 30)
    for mod in ("BPSK", "QPSK", "16QAM"):
        assert np.all(awgn_ser(gamma, mod) + 1e-12 >= awgn_ber(gamma, mod)), mod


def test_higher_order_needs_more_snr():
    # At a fixed SNR, denser constellations must do worse.
    g = 10.0
    assert awgn_ber(g, "QPSK") < awgn_ber(g, "16QAM") < awgn_ber(g, "64QAM")


@pytest.mark.parametrize("mod", ["BPSK", "QPSK"])
@pytest.mark.parametrize("snr_db", [2.0, 6.0, 10.0])
def test_semi_analytic_matches_monte_carlo(mod, snr_db):
    """The closed form must reproduce what actually transmitting symbols gives."""
    rng = np.random.default_rng(7)
    gains = np.ones(10)
    rho = 10 ** (snr_db / 10)
    closed = average_ber(gains, rho, mod)
    measured = monte_carlo_ber(gains, rho, mod, symbols_per_realization=40000, rng=rng)
    assert measured == pytest.approx(closed, rel=0.15, abs=2e-4)


def test_16qam_monte_carlo_matches_at_high_snr():
    """The Gray BER ~ SER/log2(M) relation is tight where the curves are read."""
    rng = np.random.default_rng(11)
    gains = np.ones(10)
    rho = 10 ** (16.0 / 10)
    closed = average_ber(gains, rho, "16QAM")
    measured = monte_carlo_ber(gains, rho, "16QAM", symbols_per_realization=60000, rng=rng)
    assert measured == pytest.approx(closed, rel=0.15)


def test_averaging_over_fading_is_not_the_ber_of_the_average():
    """Jensen: BER is convex in gain here, so fading must hurt."""
    gains = np.array([0.1, 1.9])
    rho = 10.0
    assert average_ber(gains, rho, "QPSK") > awgn_ber(rho * gains.mean(), "QPSK")


def test_curves_have_expected_shape_and_direction():
    rho_db = np.linspace(0, 30, 16)
    gains = np.full(50, 1.0)
    ber = ber_curve(gains, rho_db, "QPSK")
    se = spectral_efficiency_curve(gains, rho_db)
    out = outage_curve(gains, rho_db, 2.0)
    assert ber.shape == se.shape == out.shape == rho_db.shape
    assert np.all(np.diff(ber) <= 1e-12)
    assert np.all(np.diff(se) >= -1e-12)
    assert np.all(np.diff(out) <= 1e-12)


def test_outage_endpoints():
    gains = np.ones(100)
    # 0 dB gives log2(1+1) = 1 bit/s/Hz exactly, below a 2 bit/s/Hz threshold.
    assert outage_curve(gains, np.array([0.0]), 2.0)[0] == pytest.approx(1.0)
    assert outage_curve(gains, np.array([40.0]), 2.0)[0] == pytest.approx(0.0)


def test_spectral_efficiency_matches_shannon():
    gains = np.ones(5)
    se = spectral_efficiency_curve(gains, np.array([10.0]))[0]
    assert se == pytest.approx(np.log2(1 + 10), rel=1e-12)


def test_snr_at_target_interpolates_and_reports_misses():
    rho_db = np.array([0.0, 10.0, 20.0])
    ber = np.array([1e-1, 1e-3, 1e-5])
    # log10 BER is linear in dB here, so 1e-2 sits exactly halfway.
    assert snr_at_target(rho_db, ber, 1e-2) == pytest.approx(5.0)
    # A target the sweep never reaches must be reported as missing, not extrapolated.
    assert np.isnan(snr_at_target(rho_db, ber, 1e-9))


def test_ber_floor_is_applied():
    gains = np.ones(4)
    assert ber_curve(gains, np.array([60.0]), "QPSK", floor=1e-7)[0] == 1e-7


def test_unknown_modulation_raises():
    with pytest.raises(ValueError):
        awgn_ber(1.0, "8PSK")
