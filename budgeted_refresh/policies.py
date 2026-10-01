"""Refresh policies for an RIS whose control link carries ``B`` bits per slot.

Every policy computes new phases with the same rule (co-phase each element to
the current received signal), so they differ only in *which* elements they
refresh and *when* -- never in how good a single refresh is.

Timing convention, identical for every policy: at slot ``t`` the controller
knows the fresh channel ``h(t)`` and sends at most ``B`` bits; whatever it sends
takes effect from slot ``t+1`` and is scored against ``h(t+1)``. The controller
has perfect, fresh CSI, so the control link is the only bottleneck -- that is
the question under study, deliberately isolated from estimation.

Bit costs (``b`` = phase bits per element, ``N`` elements, ``T`` tiles):

* full refresh     -- the whole configuration, ``N*b`` bits, streamed over
                      ``ceil(N*b/B)`` slots and applied atomically at the end.
* round-robin      -- elements in a fixed cyclic order, so no addresses:
                      ``floor(B/b)`` elements per slot.
* greedy, indexed  -- the elements with the largest SNR gain, each sent with
                      its index: ``b + ceil(log2 N)`` bits per element (or a
                      ``N``-bit bitmap plus ``b`` per element, whichever fits more).
* greedy, tile     -- whole tiles with the largest summed gain:
                      ``ceil(log2 T) + Ne*b`` bits per tile.
"""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2 * np.pi


def quantize(phase, bits):
    step = TWO_PI / (2 ** bits)
    return np.round(np.asarray(phase) / step) * step


def received(h_d, casc, theta):
    return h_d + np.sum(casc * np.exp(1j * theta), axis=1)


def best_config(h_d, casc, bits, n_ref=16):
    """Quantized co-phasing, best over ``n_ref`` common reference phases.

    With coarse phases the choice of common reference matters; searching a few
    makes the bound tight instead of arbitrarily penalising 1-bit surfaces.
    """
    best_theta, best_p = None, None
    base = np.angle(h_d)[:, None] - np.angle(casc)
    for r in range(n_ref):
        th = quantize(base + TWO_PI * r / n_ref, bits)
        p = np.abs(received(h_d, casc, th)) ** 2
        if best_p is None:
            best_theta, best_p = th, p
        else:
            m = p > best_p
            best_theta[m], best_p[m] = th[m], p[m]
    return best_theta


def element_gain(h_d, casc, theta, bits):
    """Candidate new phase per element and its first-order gain.

    Each element is re-aligned to the phase of the *current* received signal
    (coordinate ascent), so a partial update never fights the rest of the
    surface. Gain = increase of that element's projection on the resultant.
    """
    psi = np.angle(received(h_d, casc, theta))[:, None]
    cand = quantize(psi - np.angle(casc), bits)
    rot = np.exp(-1j * psi)
    gain = np.real(casc * np.exp(1j * cand) * rot) - np.real(casc * np.exp(1j * theta) * rot)
    return cand, gain


class Policy:
    name = "base"

    def __init__(self, N, T, bits, budget):
        self.N, self.T, self.Ne, self.b, self.B = N, T, N // T, bits, budget

    def reset(self, theta0):
        self.theta = theta0.copy()

    def step(self, t, h_d, casc):
        """Use h(t); leave self.theta as the configuration for slot t+1."""
        raise NotImplementedError


class Genie(Policy):
    """Unlimited control bits, still one slot of delay: the latency floor."""

    name = "genie_delayed"

    def step(self, t, h_d, casc):
        self.theta, _ = element_gain(h_d, casc, self.theta, self.b)


class Static(Policy):
    name = "static"

    def step(self, t, h_d, casc):
        pass


class FullRefresh(Policy):
    name = "full_refresh"

    def reset(self, theta0):
        super().reset(theta0)
        self.L = max(1, math.ceil(self.N * self.b / self.B))
        self.next_start, self.pending, self.apply_after = 0, None, None

    def step(self, t, h_d, casc):
        if t == self.next_start:
            self.pending, _ = element_gain(h_d, casc, self.theta, self.b)
            self.apply_after = t + self.L - 1          # last slot of transmission
            self.next_start = t + self.L
        if t == self.apply_after:
            self.theta = self.pending


class RoundRobin(Policy):
    name = "round_robin"

    def reset(self, theta0):
        super().reset(theta0)
        self.m = min(self.N, self.B // self.b)
        self.ptr = 0

    def step(self, t, h_d, casc):
        if self.m == 0:
            return
        idx = (self.ptr + np.arange(self.m)) % self.N
        self.ptr = (self.ptr + self.m) % self.N
        cand, _ = element_gain(h_d, casc, self.theta, self.b)
        self.theta[:, idx] = cand[:, idx]


class GreedyIndexed(Policy):
    name = "greedy_indexed"

    def reset(self, theta0):
        super().reset(theta0)
        a = math.ceil(math.log2(self.N))
        m_idx = self.B // (self.b + a)
        m_map = (self.B - self.N) // self.b if self.B > self.N else 0
        self.m = min(self.N, max(m_idx, m_map))

    def step(self, t, h_d, casc):
        if self.m == 0:
            return
        cand, gain = element_gain(h_d, casc, self.theta, self.b)
        if self.m >= self.N:
            self.theta = cand
            return
        top = np.argpartition(-gain, self.m - 1, axis=1)[:, :self.m]
        rows = np.arange(casc.shape[0])[:, None]
        self.theta[rows, top] = cand[rows, top]


class GreedyTile(Policy):
    name = "greedy_tile"

    def reset(self, theta0):
        super().reset(theta0)
        cost = math.ceil(math.log2(max(self.T, 2))) + self.Ne * self.b
        self.m = min(self.T, self.B // cost)

    def step(self, t, h_d, casc):
        if self.m == 0:
            return
        cand, gain = element_gain(h_d, casc, self.theta, self.b)
        R = casc.shape[0]
        tile_gain = gain.reshape(R, self.T, self.Ne).sum(axis=2)
        top = np.argpartition(-tile_gain, self.m - 1, axis=1)[:, :self.m]
        mask = np.zeros((R, self.T), bool)
        mask[np.arange(R)[:, None], top] = True
        mask = np.repeat(mask, self.Ne, axis=1)
        self.theta = np.where(mask, cand, self.theta)


ALL = [Genie, Static, FullRefresh, RoundRobin, GreedyIndexed, GreedyTile]
