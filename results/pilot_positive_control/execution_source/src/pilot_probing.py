"""Pilot-limited observation of a passive RIS, and the classical estimators.

A passive RIS has no receive chain, so it cannot measure per-element CSI. What
it *can* do is cycle through ``M`` probing configurations while the BS sends a
pilot; the user measures one complex sample per configuration and feeds the
``M`` values back. That is the only channel information any scheme below gets,
except the ones explicitly labelled as oracles.

Signal model, normalised by ``sqrt(P_t)`` so the noise variance is ``1/rho``
with ``rho = P_t / sigma^2``::

    y_m = h_d + sum_n c_n exp(j phi_{m,n}) + w_m,     w_m ~ CN(0, 1/rho)
        = a_m^T x + w_m,   x = [h_d, c_1 .. c_N],  a_m = [1, exp(j phi_m)]

The probe codebook ``phi`` is fixed and known to every tile (it is a seeded
constant of the design, not a per-block secret).

Estimators provided:

* :func:`random_max_sampling` -- apply whichever probe the user reported as
  strongest. No channel model at all; the classical low-overhead baseline.
* :class:`LMMSEEstimator` -- linear MMSE estimate of ``x`` from the ``M``
  samples, using the channel covariance learned from training scenes, followed
  by MRC. The strongest *linear* scheme at the same pilot budget.
* :func:`ls_full_estimate` -- least squares with ``M = N + 1`` orthogonal
  (DFT) probes, i.e. full per-element CSI at the full pilot cost.
"""

from __future__ import annotations

import numpy as np


def probe_codebook(num_probes: int, num_elements: int, seed: int = 2024) -> np.ndarray:
    """Fixed random probing phases, shape (M, N), radians.

    Random phases are the standard choice for compressive RIS probing: every
    probe illuminates every element, so each measurement carries information
    about the whole channel.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2 * np.pi, size=(num_probes, num_elements))


def probe_matrix(phases: np.ndarray) -> np.ndarray:
    """Measurement matrix ``A`` (M, N+1): a leading 1 for the direct path."""
    m = phases.shape[0]
    return np.concatenate([np.ones((m, 1), dtype=complex), np.exp(1j * phases)], axis=1)


def stack_channel(h_direct: np.ndarray, cascade: np.ndarray) -> np.ndarray:
    """``x = [h_d, c_flat]``, shape (S, N+1)."""
    s = h_direct.shape[0]
    return np.concatenate([h_direct.reshape(s, 1), cascade.reshape(s, -1)], axis=1).astype(complex)


def noiseless_observations(h_direct: np.ndarray, cascade: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """Noise-free probe responses, shape (S, M)."""
    x = stack_channel(h_direct, cascade)
    return x @ probe_matrix(phases).T


def add_pilot_noise(y_clean: np.ndarray, rho_lin, rng: np.random.Generator) -> np.ndarray:
    """Add CN(0, 1/rho) noise. ``rho_lin`` may be scalar or per-scene (S,)."""
    rho = np.asarray(rho_lin, dtype=float)
    std = np.sqrt(1.0 / rho)
    if std.ndim == 1:
        std = std[:, None]
    w = (rng.standard_normal(y_clean.shape) + 1j * rng.standard_normal(y_clean.shape)) / np.sqrt(2)
    return y_clean + std * w


def mrc_from_estimate(x_hat: np.ndarray, num_tiles: int) -> np.ndarray:
    """Closed-form alignment onto the (estimated) direct path, shape (S, T, Ne)."""
    theta = np.angle(x_hat[:, :1]) - np.angle(x_hat[:, 1:])
    s = x_hat.shape[0]
    return np.mod(theta, 2 * np.pi).reshape(s, num_tiles, -1)


def random_max_sampling(y: np.ndarray, phases: np.ndarray, num_tiles: int) -> np.ndarray:
    """Pick the probe with the largest reported power, shape (S, T, Ne)."""
    best = np.argmax(np.abs(y), axis=1)
    s = y.shape[0]
    return phases[best].reshape(s, num_tiles, -1)


class LMMSEEstimator:
    """Linear MMSE channel estimate from ``M`` probe samples.

    The covariance ``R = E[x x^H]`` (and mean) are estimated from training
    scenes, i.e. this baseline is given the same data the learned schemes are
    trained on, but pooled centrally. With ``M < N + 1`` it can only recover the
    part of ``x`` that the covariance makes predictable from ``M`` projections.
    """

    def __init__(self, x_train: np.ndarray, phases: np.ndarray, shrinkage: float = 1e-3):
        self.A = probe_matrix(phases)
        self.mu = x_train.mean(axis=0)
        xc = x_train - self.mu
        # Samples are rows: R_ij = E[(x_i-mu_i) conj(x_j-mu_j)].
        # xc.conj().T @ xc would conjugate R and reverse spatial phases.
        R = (xc.T @ xc.conj()) / max(x_train.shape[0] - 1, 1)
        # Light diagonal loading: the sample covariance from a few thousand
        # scenes is rank-limited at N+1 = 1025.
        R = R + shrinkage * np.real(np.trace(R)) / R.shape[0] * np.eye(R.shape[0])
        self.R = R
        self.RAh = R @ self.A.conj().T                  # (N+1, M)
        self.ARAh = self.A @ self.RAh                   # (M, M)

    def estimate(self, y: np.ndarray, rho_lin: float) -> np.ndarray:
        """``x_hat`` of shape (S, N+1) for observations ``y`` (S, M) at SNR ``rho``."""
        m = self.ARAh.shape[0]
        G = np.linalg.solve(self.ARAh + np.eye(m) / rho_lin, (y - self.mu @ self.A.T).T)
        return (self.RAh @ G).T + self.mu


class LocalProbeRegressor:
    """Local linear estimator fitted from feedback and this tile's labels.

    Only M-dimensional probe responses and the tile's own (N_tile+1) channel
    estimates are needed. The same computation can run independently per tile;
    federation is not necessary to fit this baseline. Regularization is relative
    to input power, avoiding the tiny absolute scale of mmWave coefficients.
    This is empirical linear regression, not an oracle-covariance LMMSE claim.
    """

    def __init__(self, y_train, x_local, ridge=1e-3):
        self.y_mean = y_train.mean(0)
        self.x_mean = x_local.mean(0)
        yc, xc = y_train - self.y_mean, x_local - self.x_mean
        gram = yc.conj().T @ yc / len(yc)
        cross = yc.conj().T @ xc / len(yc)
        loading = ridge * np.trace(gram).real / gram.shape[0]
        self.weights = np.linalg.solve(gram + max(loading, 1e-30) * np.eye(gram.shape[0]), cross)

    def estimate(self, y):
        return (y - self.y_mean) @ self.weights + self.x_mean


def ls_full_estimate(x_true: np.ndarray, rho_lin: float, rng: np.random.Generator) -> np.ndarray:
    """Least-squares estimate with ``N + 1`` orthogonal DFT probes.

    A DFT matrix of size ``N + 1`` has unit-modulus entries (so it is a valid
    set of RIS configurations) and an all-ones first column (the direct path).
    With ``A^H A = (N+1) I`` the LS estimate is exactly ``x + A^H w / (N+1)``,
    whose error is CN(0, 1 / (rho (N+1))) per coefficient -- sampled directly
    here instead of multiplying by a 1025 x 1025 matrix per scene.
    """
    n1 = x_true.shape[1]
    std = np.sqrt(1.0 / (rho_lin * n1))
    w = (rng.standard_normal(x_true.shape) + 1j * rng.standard_normal(x_true.shape)) / np.sqrt(2)
    return x_true + std * w


def pilot_features(y: np.ndarray, rho_lin, tile_coords: np.ndarray) -> np.ndarray:
    """Network input for every (scene, tile), shape (T, S, 2M + 3), float32.

    * the ``M`` samples, rotated so probe 0 is real-positive (the optimal phases
      are invariant to a common rotation of every channel, so the input should
      be too) and scaled to unit RMS;
    * the per-probe pilot SNR in dB / 20, so the network knows how much to trust
      the samples;
    * the tile's normalised (row, col) on the surface -- the only thing that
      differs between tiles, since every tile hears the same feedback.
    """
    y = np.asarray(y)
    s, m = y.shape
    rot = np.exp(-1j * np.angle(y[:, :1]))
    yr = y * rot
    rms = np.sqrt(np.mean(np.abs(yr) ** 2, axis=1, keepdims=True))
    yn = yr / np.maximum(rms, 1e-30)
    rho = np.broadcast_to(np.asarray(rho_lin, dtype=float), (s,))
    snr_db = 10 * np.log10(np.maximum(rms[:, 0] ** 2 * rho, 1e-12))
    base = np.concatenate([yn.real, yn.imag, (snr_db / 20.0)[:, None]], axis=1).astype(np.float32)
    T = tile_coords.shape[0]
    out = np.empty((T, s, 2 * m + 3), dtype=np.float32)
    out[:, :, : 2 * m + 1] = base[None]
    out[:, :, 2 * m + 1:] = tile_coords[:, None, :]
    return out


def location_aided_design(
    cascade_los: np.ndarray, h_direct_los: np.ndarray
) -> np.ndarray:
    """Phases from geometry alone: focus the LoS wavefront, shape (S, T, Ne).

    ``cascade_los`` must be computed from the (possibly erroneous) user position
    the controller believes; see :func:`los_cascade_at`.
    """
    th = np.angle(h_direct_los)[:, None, None] - np.angle(cascade_los)
    return np.mod(th, 2 * np.pi)


def los_cascade_at(user_pos: np.ndarray, geometry, scene) -> tuple[np.ndarray, np.ndarray]:
    """LoS cascade and LoS direct phase a controller would compute for ``user_pos``.

    Only phases matter for the design, so amplitudes are left unnormalised.
    """
    lam = geometry.wavelength
    k = 2 * np.pi / lam
    pos = geometry.element_positions()
    T, Ne = pos.shape[:2]
    pf = pos.reshape(-1, 3)
    bs = np.asarray(scene.bs_position, dtype=float)
    d_bs = np.linalg.norm(pf - bs, axis=1)
    d_u = np.linalg.norm(pf[None] - user_pos[:, None, :], axis=2)
    casc = np.exp(-1j * k * (d_bs[None, :] + d_u)).reshape(-1, T, Ne)
    d_dir = np.linalg.norm(user_pos - bs, axis=1)
    return casc, np.exp(-1j * k * d_dir)
