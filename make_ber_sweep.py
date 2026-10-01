#!/usr/bin/env python
"""BER vs received SNR for the passive pilot experiment.

Reuses the models trained by ``run_pilot_limited.py``. No retraining: a phase
configuration does not depend on transmit power, so the sweep is a scoring pass
over already-designed phases.

Curves are plotted against the received SNR of the perfect-CSI reference design,
which is one common axis for every scheme. Plotting each scheme against its own
realized SNR would collapse them all onto one curve, since BER is a function of
realized SNR alone.

  python make_ber_sweep.py --probes 16 --seeds 42 123 456 789 1024
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_pilot_limited import predict, prepare_case
from src.link_metrics import awgn_ber
from src.pilot_probing import (
    LMMSEEstimator,
    LocalProbeRegressor,
    mrc_from_estimate,
    random_max_sampling,
)

SNR_GRID = np.arange(-5.0, 25.01, 0.5)

LABELS = {
    "fed_ris": "FedAvg (federated)",
    "centralized_dl": "Centralized DL",
    "local_only": "Local models (no federation)",
    "lmmse_mrc": "LMMSE + MRC",
    "local_linear_mrc": "Local linear + MRC",
    "random_max": "Best observed probe",
    "oracle_mrc": "Perfect-CSI MRC (bound)",
    "no_ris": "No RIS (blocked)",
}
STYLE = {
    "fed_ris": dict(color="#c0392b", ls="-", marker="o"),
    "centralized_dl": dict(color="#e67e22", ls="--", marker="s"),
    "local_only": dict(color="#8e44ad", ls="-.", marker="^"),
    "lmmse_mrc": dict(color="#1f77b4", ls="-", marker="D"),
    "local_linear_mrc": dict(color="#2e8b57", ls="--", marker="v"),
    "random_max": dict(color="#7f7f7f", ls=":", marker="x"),
    "oracle_mrc": dict(color="#000000", ls=":", marker=None),
    "no_ris": dict(color="#999999", ls="-.", marker=None),
}
ORDER = ["oracle_mrc", "lmmse_mrc", "local_linear_mrc", "random_max",
         "local_only", "centralized_dl", "fed_ris", "no_ris"]


class _Args:
    """Minimal stand-in for the runner's argparse namespace."""
    quick = False
    threads = 1
    rounds = 100
    min_rounds = 20
    patience = 15
    min_delta = 1e-4
    train_samples = 600
    validation_samples = 200
    test_samples = 600
    codebook_seed = 2024
    hidden_dim = 128
    skip_training = True
    seeds = None
    pilot_counts = None
    tx_powers_dbm = None
    output = None


def gains_for_case(seed: int, num_probes: int, pt_dbm: float, models_path: Path) -> dict:
    """Per-scene channel power gain |h_eff|^2 for every scheme in one case."""
    from models.ris_net import MLPModel

    prepared = prepare_case(_Args(), seed, num_probes, pt_dbm)
    geometry = prepared["geometry"]
    channels = prepared["channels"][2]          # held-out test scenes
    rho, phases, y = prepared["rho"], prepared["phases"], prepared["y"]
    labels, x = prepared["labels"], prepared["x"]

    def gain(theta):
        h = channels["h_direct"].copy()
        if theta is not None:
            h = h + (channels["cascade"] * np.exp(1j * theta)).sum((1, 2))
        return np.abs(h) ** 2

    out = {"no_ris": gain(None)}

    # ---- classical controls, designed from the same M responses -------------
    out["oracle_mrc"] = gain(mrc_from_estimate(x[2], geometry.num_tiles))
    out["random_max"] = gain(random_max_sampling(y[2], phases, geometry.num_tiles))

    lmmse = LMMSEEstimator(labels[0], phases)
    out["lmmse_mrc"] = gain(mrc_from_estimate(lmmse.estimate(y[2], rho), geometry.num_tiles))

    ne = geometry.elements_per_tile
    local = []
    for t in range(geometry.num_tiles):
        start = 1 + t * ne
        targets = np.concatenate([labels[0][:, :1], labels[0][:, start:start + ne]], axis=1)
        reg = LocalProbeRegressor(y[0], targets)
        est = reg.estimate(y[2])
        local.append(np.mod(np.angle(est[:, :1]) - np.angle(est[:, 1:]), 2 * np.pi))
    out["local_linear_mrc"] = gain(np.stack(local, axis=1))

    # ---- learned arms, loaded from the completed training run ---------------
    blob = torch.load(models_path, weights_only=False, map_location="cpu")
    features = [d.features for d in prepared["datasets"][0]]      # placeholder shape
    from src.pilot_probing import pilot_features
    feats = pilot_features(y[2], rho, geometry.tile_grid_coords())

    for arm, state in blob["weights"].items():
        if arm not in LABELS:
            continue

        def fresh():
            return MLPModel(2 * num_probes + 3, ne, _Args.hidden_dim, 2, 0.0)

        if isinstance(state, list):
            models = []
            for sd in state:
                m = fresh()
                m.load_state_dict(sd)
                models.append(m)
        else:
            m = fresh()
            m.load_state_dict(state)
            models = m
        out[arm] = gain(predict(models, feats, "cpu"))

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", type=int, default=16)
    ap.add_argument("--pt-dbm", type=float, default=30.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    ap.add_argument("--pilot-dir", default="results/pilot_limited")
    ap.add_argument("--output", default="results/ber_sweep")
    ap.add_argument("--modulation", default="QPSK", choices=["QPSK", "16QAM"])
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    per_seed = {}
    for seed in args.seeds:
        mp = Path(args.pilot_dir) / f"seed_{seed}" / f"M{args.probes}_Pt{args.pt_dbm:g}" / "models.pt"
        if not mp.exists():
            print(f"[skip] no trained models at {mp}")
            continue
        print(f"[seed {seed}] scoring {mp}")
        per_seed[seed] = gains_for_case(seed, args.probes, args.pt_dbm, mp)

    if not per_seed:
        raise SystemExit("No cases scored; check --pilot-dir and --probes.")

    # Common axis: received SNR of the perfect-CSI reference design.
    curves, seeds = {}, sorted(per_seed)
    for scheme in ORDER:
        rows = []
        for s in seeds:
            g = per_seed[s]
            if scheme not in g:
                continue
            ref = g["oracle_mrc"].mean()
            # rho such that the reference design sits at each grid SNR
            rho_grid = 10 ** (SNR_GRID / 10) / ref
            ber = [awgn_ber(r * g[scheme], args.modulation).mean() for r in rho_grid]
            rows.append(ber)
        if rows:
            curves[scheme] = np.array(rows)

    payload = {
        "schema": "ber-sweep-v1",
        "modulation": args.modulation,
        "num_probes": args.probes,
        "tx_power_dbm_of_training": args.pt_dbm,
        "seeds": seeds,
        "x_axis": "received SNR of the perfect-CSI reference design (dB)",
        "snr_db": SNR_GRID.tolist(),
        "curves": {k: v.mean(0).tolist() for k, v in curves.items()},
        "curves_per_seed": {k: v.tolist() for k, v in curves.items()},
    }
    write = out / f"ber_sweep_M{args.probes}_{args.modulation}.json"
    write.write_text(json.dumps(payload, indent=1))
    print(f"[done] {write}")

    plot(curves, args, out)


def plot(curves, args, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.3, "grid.linestyle": "--", "grid.linewidth": 0.4})
    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    floor = 1e-6
    for scheme in ORDER:
        if scheme not in curves:
            continue
        y = np.maximum(curves[scheme].mean(0), floor)
        st = STYLE[scheme]
        ax.semilogy(SNR_GRID, y, label=LABELS[scheme], lw=1.6,
                    markevery=8, markersize=4.5, **st)

    ax.axhline(1e-3, color="#aaaaaa", lw=0.7)
    ax.text(-4.6, 1.25e-3, r"BER $=10^{-3}$", fontsize=7, color="#888888")
    ax.set_xlim(-5, 25)
    ax.set_ylim(floor, 1.0)
    ax.set_xlabel("Reference received SNR (dB)")
    ax.set_ylabel(f"Bit error rate ({args.modulation})")
    ax.set_title(f"Passive pilot experiment, M = {args.probes} probes\n"
                 f"{len(args.seeds)} seeds, 600 held-out scenes", fontsize=9, loc="left")
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.92)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        p = out / f"ber_vs_snr_M{args.probes}_{args.modulation}.{ext}"
        fig.savefig(p, dpi=220, bbox_inches="tight")
        print(f"[done] {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
