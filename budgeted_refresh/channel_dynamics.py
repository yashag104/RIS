"""Time-varying channels for the tiled RIS: a user walking in a straight line.

Reuses the static geometry and link budget of :mod:`src.surface_channel` (same
32x32 surface at 28 GHz, same room, BS, path-loss exponents, element gain,
Rician K and direct-link blockage) and adds motion:

* BS -> surface is static (neither moves).
* Surface -> user LoS is recomputed exactly from the element-to-user distance
  every slot, so the phase drift across the aperture is the true near-field
  wavefront change, not an i.i.d. perturbation.
* Surface -> user NLoS is a fixed set of ``num_paths`` scatterer paths. Each path
  keeps its spatial signature across the aperture (as in the static model) and
  picks up a Doppler phase ``k * u_p . displacement`` at the user, with ``u_p``
  its arrival direction there. Paths therefore decorrelate at the rate set by
  speed and wavelength -- Jakes-like, but deterministic given the path set.
* The direct link gets the same treatment, blocked by the usual 30 dB.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.surface_channel import (  # noqa: E402
    SceneConfig,
    SurfaceGeometry,
    _nlos_field,
    _random_front_directions,
    close_in_path_loss,
)


def _unit_sphere(rng, shape):
    v = rng.standard_normal(shape + (3,))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


@dataclass
class Trajectories:
    """``R`` users, each walking at ``speed`` m/s for ``num_slots`` slots."""

    geometry: SurfaceGeometry
    scene: SceneConfig
    num_traj: int
    speed: float
    slot_s: float
    num_slots: int
    seed: int = 0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        g, sc = self.geometry, self.scene
        lam = g.wavelength
        self.k = 2 * np.pi / lam
        pos = g.element_positions().reshape(-1, 3)
        self.pos = pos
        self.N = pos.shape[0]
        self.T, self.Ne = g.num_tiles, g.elements_per_tile
        center = pos.mean(axis=0)
        self.center = center
        bs = np.asarray(sc.bs_position, float)
        K = 10 ** (sc.k_factor_db / 10)
        self.w_los, self.w_nlos = np.sqrt(K / (K + 1)), np.sqrt(1 / (K + 1))
        self.g2 = (4 * np.pi * g.spacing ** 2 / lam ** 2) if sc.element_gain_enabled else 1.0
        self.blockage = 10 ** (-sc.direct_link_blockage_db / 10)
        ple = sc.ris_path_loss_exponent
        R, P = self.num_traj, sc.num_paths

        # Static BS -> surface hop, one independent NLoS draw per trajectory.
        d_bs = np.linalg.norm(pos - bs, axis=1)
        h_bs_los = np.sqrt(close_in_path_loss(d_bs, lam, ple)) * np.exp(-1j * self.k * d_bs)
        pl_bs_c = close_in_path_loss(np.linalg.norm(center - bs), lam, ple)
        h_bs_nlos = np.sqrt(pl_bs_c) * _nlos_field(rng, pos, lam, R, P)
        self.h_bs = self.w_los * h_bs_los[None, :] + self.w_nlos * h_bs_nlos     # (R, N)

        # Walk: start inside the room with a margin, random horizontal heading.
        dist = self.speed * self.slot_s * self.num_slots
        lo = np.asarray(sc.user_low, float) + [dist, dist, 0]
        hi = np.asarray(sc.user_high, float) - [dist, dist, 0]
        lo = np.minimum(lo, hi)
        self.start = rng.uniform(lo, hi, size=(R, 3))
        ang = rng.uniform(0, 2 * np.pi, R)
        self.heading = np.stack([np.cos(ang), np.sin(ang), np.zeros(R)], axis=-1)
        self.bs = bs

        # User-side NLoS paths: spatial signature across the aperture + Doppler.
        rel = pos - center
        amp = np.exp(-0.5 * np.arange(P))
        amp = amp / np.sqrt(np.sum(amp ** 2))
        self.u_gain = amp * (rng.standard_normal((R, P)) + 1j * rng.standard_normal((R, P))) / np.sqrt(2)
        u_dirs_ap = _random_front_directions(rng, (R, P))                        # at the surface
        self.u_sig = np.exp(1j * self.k * np.einsum("rpd,nd->rpn", u_dirs_ap, rel))  # (R, P, N)
        self.u_arr = _unit_sphere(rng, (R, P))                                   # at the user
        # Direct-link NLoS paths.
        self.d_gain = (rng.standard_normal((R, P)) + 1j * rng.standard_normal((R, P))) / np.sqrt(2 * P)
        self.d_arr = _unit_sphere(rng, (R, P))

    def user_pos(self, t: int) -> np.ndarray:
        return self.start + self.heading * (self.speed * self.slot_s * t)

    def at(self, t: int):
        """Channels at slot ``t``: ``h_d`` (R,) and ``cascade`` (R, N)."""
        lam, ple, k = self.geometry.wavelength, self.scene.ris_path_loss_exponent, self.k
        disp = self.heading * (self.speed * self.slot_s * t)                      # (R, 3)
        user = self.start + disp

        d_u = np.linalg.norm(self.pos[None] - user[:, None], axis=2)             # (R, N)
        h_u_los = np.sqrt(close_in_path_loss(d_u, lam, ple)) * np.exp(-1j * k * d_u)
        pl_u_c = close_in_path_loss(np.linalg.norm(user - self.center, axis=1), lam, ple)
        dop = np.exp(1j * k * np.einsum("rpd,rd->rp", self.u_arr, disp))         # (R, P)
        h_u_nlos = np.sqrt(pl_u_c)[:, None] * np.einsum("rp,rpn->rn", self.u_gain * dop, self.u_sig)
        h_u = self.w_los * h_u_los + self.w_nlos * h_u_nlos
        cascade = self.g2 * self.h_bs * h_u

        d_dir = np.linalg.norm(user - self.bs, axis=1)
        pl_d = close_in_path_loss(d_dir, lam, self.scene.direct_path_loss_exponent) * self.blockage
        dop_d = np.exp(1j * k * np.einsum("rpd,rd->rp", self.d_arr, disp))
        h_d = np.sqrt(pl_d) * (self.w_los * np.exp(-1j * k * d_dir)
                               + self.w_nlos * np.sum(self.d_gain * dop_d, axis=1))
        return h_d, cascade
