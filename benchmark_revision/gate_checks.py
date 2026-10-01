"""Day-1 gate checks for the benchmark-paper revision.

Run before any rewriting, because each one can change the paper's story:

A. Theory: does full-probe LS land where closed-form theory says it must?
B. LMMSE: is the M = 1025 shortfall (0.253 < 0.320 of oracle) caused by the
   rank-limited covariance (600 noisy labels, 1025 unknowns) or by the random
   probe codebook? Variants: (i) paper setting, (ii) covariance from 5000
   *true* channels (what a good prior could give), (iii) orthogonal DFT probes.
C. Structured probing: do beam-type probes (LoS focusing on a grid of candidate
   user positions; 2D-DFT beams) beat the paper's best scheme at M = 16, 64?
   Plus the geometry ceiling: focusing on the TRUE user position.

Same channel generator, SNR (Pt = 30 dBm, noise -90 dBm) and scoring as
run_pilot_limited.py. Usage: .venv/bin/python benchmark_revision/gate_checks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import i0e, i1e

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pilot_probing import (  # noqa: E402
    LMMSEEstimator, add_pilot_noise, los_cascade_at, ls_full_estimate,
    mrc_from_estimate, noiseless_observations, probe_codebook, stack_channel,
)
from src.surface_channel import SceneConfig, SurfaceGeometry, generate_surface_channels  # noqa: E402

L_BLOCK = 2048
RHO = 10 ** ((30 - (-90)) / 10)


def score(ch, theta, pilots):
    h = ch["h_direct"] + (ch["cascade"] * np.exp(1j * theta)).sum((1, 2))
    g = np.abs(h) ** 2
    bound = (np.abs(ch["h_direct"]) + np.abs(ch["cascade"]).sum((1, 2))) ** 2
    rate = np.log2(1 + RHO * g)
    return {"frac_oracle": float(np.mean(g / bound)),
            "snr_db": float(10 * np.log10(RHO * g.mean())),
            "net_se": float(max(0.0, 1 - pilots / L_BLOCK) * rate.mean()),
            "pilots": int(pilots)}


def dft_probes(M, N1):
    """First M rows of the (N+1)-point DFT: unit modulus, first column all ones."""
    m, n = np.arange(M)[:, None], np.arange(N1)[None, :]
    return np.exp(-2j * np.pi * m * n / N1)


def lmmse_general(x_train, A, y, shrink=1e-3):
    mu = x_train.mean(0)
    xc = x_train - mu
    R = xc.T @ xc.conj() / (len(xc) - 1)
    R = R + shrink * np.real(np.trace(R)) / R.shape[0] * np.eye(R.shape[0])
    RAh = R @ A.conj().T
    G = np.linalg.solve(A @ RAh + np.eye(A.shape[0]) / RHO, (y - mu @ A.T).T)
    return (RAh @ G).T + mu


def ls_theory_fraction(x_true):
    """E[fraction] for MRC on LS estimates with per-coefficient noise 1/(rho(N+1)).

    For |c| = a and CN(0, s2) error, E[cos(phase error)] has the Rician closed
    form sqrt(pi*g/4) e^{-g/2} [I0(g/2) + I1(g/2)], g = a^2/s2. Coherent part
    only (incoherent residual is O(1/N) and ignored).
    """
    s2 = 1.0 / (RHO * x_true.shape[1])
    a = np.abs(x_true[:, 1:])
    g = a ** 2 / s2
    ecos = np.sqrt(np.pi * g / 4) * (i0e(g / 2) + i1e(g / 2))   # i*e already include e^{-g/2}
    num = (np.abs(x_true[:, 0]) + (a * ecos).sum(1)) ** 2
    den = (np.abs(x_true[:, 0]) + a.sum(1)) ** 2
    return float(np.mean(num / den))


def focus_grid(M, scene, z=1.5):
    lo, hi = np.array(scene.user_low), np.array(scene.user_high)
    nx = int(np.ceil(np.sqrt(M * (hi[0] - lo[0]) / (hi[1] - lo[1]))))
    ny = int(np.ceil(M / nx))
    xs = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx
    ys = lo[1] + (np.arange(ny) + 0.5) * (hi[1] - lo[1]) / ny
    pts = np.array([(x, y, z) for x in xs for y in ys])[:M]
    return pts


def best_probe_design(ch, phases, rng):
    """Apply each probe, user reports all M samples, pick the strongest."""
    y = add_pilot_noise(noiseless_observations(ch["h_direct"], ch["cascade"], phases), RHO, rng)
    best = np.argmax(np.abs(y), axis=1)
    S = y.shape[0]
    return phases[best].reshape(S, *ch["cascade"].shape[1:])


def main():
    geo, scene = SurfaceGeometry(), SceneConfig()
    N1 = geo.total_elements + 1
    out = {}
    for seed in (42, 123):
        tr = generate_surface_channels(600, geo, scene, np.random.default_rng(seed))
        te = generate_surface_channels(600, geo, scene, np.random.default_rng(seed + 2))
        big = generate_surface_channels(5000, geo, scene, np.random.default_rng(seed + 99))
        x_tr, x_te, x_big = (stack_channel(c["h_direct"], c["cascade"]) for c in (tr, te, big))
        labels = ls_full_estimate(x_tr, RHO, np.random.default_rng(seed + 30))
        T = geo.num_tiles
        r = {}

        # A. theory vs simulation for full-probe LS
        ls = ls_full_estimate(x_te, RHO, np.random.default_rng(seed + 32))
        r["A_ls_sim"] = score(te, mrc_from_estimate(ls, T), N1)
        r["A_ls_theory_frac"] = ls_theory_fraction(x_te)
        r["oracle"] = score(te, mrc_from_estimate(x_te, T), 0)

        for M in (16, 64, 1025):
            rng = np.random.default_rng(seed + 20 + M)
            ph = probe_codebook(M, geo.total_elements, 2024)
            A_rand = np.concatenate([np.ones((M, 1)), np.exp(1j * ph)], axis=1)
            y_rand = add_pilot_noise(x_te @ A_rand.T, RHO, rng)
            # (i) paper setting
            est = LMMSEEstimator(labels, ph).estimate(y_rand, RHO)
            r[f"B_lmmse_paper_M{M}"] = score(te, mrc_from_estimate(est, T), M)
            # (ii) covariance from 5000 true channels, same random probes
            est = lmmse_general(x_big, A_rand, y_rand)
            r[f"B_lmmse_truecov5000_M{M}"] = score(te, mrc_from_estimate(est, T), M)
            # (iii) DFT probes, paper covariance (noisy labels)
            A_dft = dft_probes(M, N1)
            y_dft = add_pilot_noise(x_te @ A_dft.T, RHO, np.random.default_rng(seed + 50 + M))
            est = lmmse_general(labels, A_dft, y_dft)
            r[f"B_lmmse_dft_M{M}"] = score(te, mrc_from_estimate(est, T), M)

        # C. structured probes, best-of-M selection (no channel model, no training)
        for M in (16, 64, 256):
            pts = focus_grid(M, scene)
            casc, hd = los_cascade_at(pts, geo, scene)
            ph = np.mod(np.angle(hd)[:, None] - np.angle(casc.reshape(M, -1)), 2 * np.pi)
            th = best_probe_design(te, ph, np.random.default_rng(seed + 70 + M))
            r[f"C_focusgrid_best_M{M}"] = score(te, th, M)
            ph_r = probe_codebook(M, geo.total_elements, 2024)
            th = best_probe_design(te, ph_r, np.random.default_rng(seed + 80 + M))
            r[f"C_random_best_M{M}"] = score(te, th, M)
        casc, hd = los_cascade_at(te["user_pos"], geo, scene)
        th = np.mod(np.angle(hd)[:, None, None] - np.angle(casc), 2 * np.pi)
        r["C_true_position_focus_ceiling"] = score(te, th, 0)
        out[str(seed)] = r
        print(f"seed {seed} done", flush=True)

    p = ROOT / "benchmark_revision" / "results" / "gate_checks.json"
    p.write_text(json.dumps(out, indent=1))
    for seed, r in out.items():
        print(f"\n=== seed {seed}")
        for k, v in r.items():
            if isinstance(v, dict):
                print(f"  {k:34s} frac={v['frac_oracle']:.3f}  snr={v['snr_db']:6.2f}  net_se={v['net_se']:.3f}  M={v['pilots']}")
            else:
                print(f"  {k:34s} {v:.3f}")


if __name__ == "__main__":
    main()
