#!/usr/bin/env python
"""Matched-update architecture diagnostic using the primary pilot harness.

This is centralized pilot-feedback training, not corrected supplied-CSI GAT
retraining or a convergence study. The budget and compact model profiles are
fixed before testing. Width/steps are disclosed; FLOPs are not matched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from models.ris_net import create_model
from run_link_level import provenance, seed_everything, seed_statistics, write_json
from run_pilot_limited import prepare_case, predict, score
from src.controlled_training import tensor_data, train_steps, validation_loss
from src.pilot_probing import pilot_features

PROFILES = dict(HIDDEN_DIM=128, NUM_LAYERS=2, DROPOUT=0.0,
                PIXEL_GRID_ROWS=8, PIXEL_GRID_COLS=8,
                GNN_HIDDEN_DIM=128, GNN_NUM_LAYERS=2, GNN_NUM_HEADS=4,
                CNN_HIDDEN_CHANNELS=32, CNN_SE_REDUCTION=16,
                TRANSFORMER_D_MODEL=128, TRANSFORMER_NUM_HEADS=4,
                TRANSFORMER_NUM_LAYERS=2, TRANSFORMER_FF_DIM=256)


def run(args):
    out = Path(args.output)
    results = []
    for seed in args.seeds:
        p = prepare_case(args, seed, args.probes, args.tx_power_dbm)
        td, vd = [tensor_data(d, "cpu") for d in p["datasets"]]
        pooled = tuple(torch.cat([d[i] for d in td]) for i in range(4))
        features = pilot_features(p["y"][2], p["rho"], p["geometry"].tile_grid_coords())
        prov = provenance(args, p["geometry"], p["scene"])
        prov.update(schema="pilot-architecture-v1", seed=seed,
                    observation_model="passive receiver feedback",
                    sample_counts_train_validation_test=p["ns"],
                    codebook_sha256=hashlib.sha256(p["phases"].tobytes()).hexdigest(),
                    training_features_sha256=hashlib.sha256(pooled[0].numpy().tobytes()).hexdigest(),
                    architecture_profile=PROFILES, optimizer="Adam", learning_rate=1e-3,
                    batch_size=64, absolute_phases=True,
                    supplied_csi_gat_retraining=False, convergence_claim=False,
                    selection="minimum validation loss at fixed evaluation checkpoints",
                    budget_scope="equal optimizer updates and batch size; unequal FLOPs and parameters")
        case = {"meta": prov, "architectures": {}}
        for arch in args.architectures:
            seed_everything(seed, args.threads)
            model = create_model(arch, 2*args.probes+3, p["geometry"].elements_per_tile,
                                 config=SimpleNamespace(**PROFILES))
            # These inputs are receiver feedback, never elementwise CSI.
            if arch == "GNN" and model.element_proj is not None:
                raise ValueError("Pilot feature width incorrectly interpreted as supplied CSI")
            rng, opt = np.random.default_rng(seed + 200), None
            history, best, selected, weights = [], float("inf"), 0, None
            for start in range(0, args.steps, args.validation_interval):
                count = min(args.validation_interval, args.steps-start)
                loss, opt = train_steps(model, pooled, count, 64, 1e-3, rng, p["rho"],
                                        absolute_phases=True, optimizer=opt)
                val = validation_loss(model, vd, p["rho"], absolute_phases=True)
                history.append(dict(step=start+count, train_loss=loss, validation_loss=val))
                if val < best:
                    best, selected, weights = val, start+count, copy.deepcopy(model.state_dict())
                print(f"seed={seed} {arch} step={start+count} val={val:.6f}", flush=True)
            model.load_state_dict(weights)
            metrics = score(p["channels"][2], predict(model, features, "cpu"), p["rho"],
                            args.probes, args.coherence_symbols)
            case["architectures"][arch] = dict(parameters=model.count_parameters(),
                optimizer_steps=args.steps, selected_step=selected, history=history, scores=metrics)
            write_json(out / f"seed_{seed}.json", case)
        results.append(case)
    summary = {"schema": "pilot-architecture-v1", "seeds": args.seeds,
               "architectures": {}}
    for arch in args.architectures:
        rows = [r["architectures"][arch] for r in results]
        summary["architectures"][arch] = dict(parameters=rows[0]["parameters"],
            optimizer_steps=args.steps, scores={metric: seed_statistics([r["scores"][metric] for r in rows])
            for metric in ("mean_received_snr_db", "net_spectral_efficiency", "ber_qpsk")})
        for metric in ("mean_received_snr_db", "net_spectral_efficiency"):
            summary["architectures"][arch]["paired_"+metric+"_vs_mlp"] = seed_statistics([
                r["architectures"][arch]["scores"][metric] -
                r["architectures"]["MLP"]["scores"][metric] for r in results])
    write_json(out / "summary.json", summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    ap.add_argument("--architectures", nargs="+", default=["MLP", "GNN", "CNN", "Transformer"])
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--validation-interval", type=int, default=100)
    ap.add_argument("--probes", type=int, default=16)
    ap.add_argument("--tx-power-dbm", type=float, default=30.0)
    ap.add_argument("--train-samples", type=int, default=600)
    ap.add_argument("--validation-samples", type=int, default=200)
    ap.add_argument("--test-samples", type=int, default=600)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--output", default="results/pilot_architecture")
    ap.set_defaults(quick=False, rounds=300, min_rounds=20, patience=15, min_delta=1e-4,
                    codebook_seed=2024, coherence_symbols=2048)
    args = ap.parse_args()
    if min(args.steps, args.validation_interval, args.train_samples, args.validation_samples,
           args.test_samples, args.probes, args.threads) < 1:
        ap.error("Budgets and sample counts must be positive")
    if len(set(args.seeds)) != len(args.seeds) or "MLP" not in args.architectures:
        ap.error("Use distinct seeds and include MLP as the paired reference")
    run(args)


if __name__ == "__main__":
    main()
