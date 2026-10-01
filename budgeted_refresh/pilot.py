"""Gate-1 pilot: does the choice of refresh policy matter under a bit budget?

Kill criterion (fixed before running): if the best budget-aware policy does
not beat the naive ones (full refresh, round-robin) by >= 1.5 dB at some
realistic budget and speed, the scheduling question is not worth a paper.

Output: results/pilot.json and results/pilot_loss_vs_budget.png
Usage:  .venv/bin/python budgeted_refresh/pilot.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from channel_dynamics import Trajectories
from policies import ALL, best_config, received
from src.surface_channel import SceneConfig, SurfaceGeometry

HERE = Path(__file__).resolve().parent


def run(speed, bits, budgets, args):
    geo, scene = SurfaceGeometry(), SceneConfig(k_factor_db=args.k_db)
    tr = Trajectories(geo, scene, args.traj, speed, args.slot_ms * 1e-3, args.slots, seed=args.seed)
    h_d0, c0 = tr.at(0)
    theta0 = best_config(h_d0, c0, bits)

    pols = {}
    for B in budgets:
        for cls in ALL:
            if cls.name in ("genie_delayed", "static") and B != budgets[0]:
                continue
            p = cls(tr.N, tr.T, bits, B)
            p.reset(theta0)
            pols[(cls.name, B)] = p
    power = {key: [] for key in pols}
    ref = []

    h_d, c = h_d0, c0
    for t in range(args.slots - 1):
        for p in pols.values():
            p.step(t, h_d, c)
        h_d, c = tr.at(t + 1)
        ref.append(np.abs(received(h_d, c, best_config(h_d, c, bits))) ** 2)
        for key, p in pols.items():
            power[key].append(np.abs(received(h_d, c, p.theta)) ** 2)

    # Score only after every policy has completed at least one full cycle, so
    # the common perfect start does not flatter the slow policies.
    warm = args.warmup
    ref = np.array(ref)[warm:]                             # (slots-1-warm, R)
    out = {}
    for (name, B), pw in power.items():
        pw = np.array(pw)[warm:]
        loss = 10 * np.log10(ref.mean() / pw.mean())       # average-SNR loss
        p10 = float(np.percentile(10 * np.log10(ref / np.maximum(pw, 1e-300)), 90))
        out.setdefault(name, {})[str(B)] = {"loss_db": float(loss), "p90_loss_db": p10}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--traj", type=int, default=32)
    ap.add_argument("--slots", type=int, default=400)
    ap.add_argument("--slot-ms", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--k-db", type=float, default=10.0)
    args = ap.parse_args()
    speeds, bit_list = [1.0, 3.0, 10.0], [1, 2]
    budgets = [16, 32, 64, 128, 256, 512, 1024, 2048]
    if args.quick:
        args.traj, args.slots, args.warmup, speeds, bit_list = 8, 200, 80, [10.0], [1]

    results = {"config": {**vars(args), "speeds_mps": speeds, "phase_bits": bit_list,
                          "budgets_bits_per_slot": budgets, "carrier_hz": 28e9,
                          "elements": 1024, "tiles": 16}, "runs": {}}
    for v in speeds:
        for b in bit_list:
            t0 = time.time()
            results["runs"][f"v{v:g}_b{b}"] = run(v, b, budgets, args)
            print(f"speed {v} m/s, {b}-bit: {time.time() - t0:.0f}s", flush=True)

    tag = "quick" if args.quick else f"K{args.k_db:g}dB"
    out = HERE / "results" / f"pilot_{tag}.json"
    out.write_text(json.dumps(results, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
