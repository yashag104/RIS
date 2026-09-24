#!/usr/bin/env python
"""Passive RIS experiment: train and infer using receiver probe feedback.

No learned method receives per-element test CSI, test user positions, or the
true direct-path phase. Training labels come from explicitly charged N+1 DFT
probes. Full-probe LS/MRC is a higher-pilot-cost control, not an equal-budget
competitor. This is a research diagnostic; a learning advantage is not assumed.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from run_link_level import provenance, seed_everything, seed_statistics, write_json
from src.controlled_training import train_comparison
from src.link_metrics import awgn_ber
from src.pilot_probing import (
    LMMSEEstimator,
    LocalProbeRegressor,
    add_pilot_noise,
    ls_full_estimate,
    mrc_from_estimate,
    noiseless_observations,
    pilot_features,
    probe_codebook,
    random_max_sampling,
    stack_channel,
)
from src.surface_channel import (
    generate_surface_channels,
    geometry_from_config,
    scene_from_config,
    surface_datasets,
)

SCHEMA = "passive-feedback-v1"


def pilot_datasets(channels, geometry, y, rho, label_estimates):
    """Targets use full training probes; features use limited probes only."""
    measured = {"h_direct": label_estimates[:, 0],
                "cascade": label_estimates[:, 1:].reshape(channels["cascade"].shape)}
    datasets = surface_datasets(measured, geometry)
    features = pilot_features(y, rho, geometry.tile_grid_coords())
    for t, ds in enumerate(datasets):
        ds.features = features[t]
        ds.phase_offset = np.zeros(len(y), dtype=np.float32)
    return datasets


def predict(models, features, device):
    """Absolute phases: no oracle/global phase offset is applied."""
    models = models if isinstance(models, list) else [models] * len(features)
    theta = []
    with torch.no_grad():
        for t, model in enumerate(models):
            model.to(device).eval()
            theta.append(np.concatenate([model(torch.tensor(features[t][i:i+64], device=device)).cpu().numpy()
                                         for i in range(0, len(features[t]), 64)]))
    return np.stack(theta, axis=1)


def score(channels, theta, rho, pilots, coherence):
    h = channels["h_direct"].copy()
    if theta is not None:
        h += (channels["cascade"] * np.exp(1j * theta)).sum((1, 2))
    gains = np.abs(h) ** 2
    gamma = rho * gains
    bound = (np.abs(channels["h_direct"]) + np.abs(channels["cascade"]).sum((1, 2))) ** 2
    rate = np.log2(1 + gamma)
    return {"mean_received_snr_db": float(10 * np.log10(gamma.mean())),
            "spectral_efficiency": float(rate.mean()),
            "net_spectral_efficiency": float(max(0, 1 - pilots / coherence) * rate.mean()),
            "ber_qpsk": float(awgn_ber(gamma, "QPSK").mean()),
            "outage_2bps_hz": float(np.mean(rate < 2)),
            "fraction_of_oracle_power": float(np.mean(gains / bound)),
            "pilots_per_block": pilots}


def prepare_case(args, seed, num_probes, pt_dbm):
    """Shared acquisition/splits for headline and architecture experiments."""
    from config import Config
    Config.SEED, Config.MODEL_TYPE = seed, "MLP"
    Config.NUM_USERS = 1
    Config.TILE_GRID_ROWS = Config.TILE_GRID_COLS = 2 if args.quick else 4
    Config.NUM_TILES = Config.TILE_GRID_ROWS * Config.TILE_GRID_COLS
    Config.TX_POWER_DBM = pt_dbm
    Config.FL_ROUNDS = 2 if args.quick else args.rounds
    Config.MIN_ROUNDS = min(args.min_rounds, Config.FL_ROUNDS)
    Config.PATIENCE, Config.MIN_DELTA = args.patience, args.min_delta
    Config.LOCAL_EPOCHS, Config.BATCH_SIZE = 1, 64
    Config.LEARNING_RATE = 1e-3
    seed_everything(seed, args.threads)
    geometry, scene = geometry_from_config(Config), scene_from_config(Config)
    ns = [64, 32, 32] if args.quick else [args.train_samples, args.validation_samples, args.test_samples]
    channels = [generate_surface_channels(n, geometry, scene, np.random.default_rng(seed + i))
                for i, n in enumerate(ns)]
    rho = 10 ** ((pt_dbm - Config.NOISE_POWER_DBM) / 10)
    phases = probe_codebook(num_probes, geometry.total_elements, args.codebook_seed)
    y = [add_pilot_noise(noiseless_observations(c["h_direct"], c["cascade"], phases), rho,
                         np.random.default_rng(seed + 20 + i)) for i, c in enumerate(channels)]
    x = [stack_channel(c["h_direct"], c["cascade"]) for c in channels]
    labels = [ls_full_estimate(x[i], rho, np.random.default_rng(seed + 30 + i)) for i in (0, 1)]
    datasets = [pilot_datasets(channels[i], geometry, y[i], rho, labels[i]) for i in (0, 1)]
    return {"config": Config, "geometry": geometry, "scene": scene, "ns": ns,
            "channels": channels, "rho": rho, "phases": phases, "y": y,
            "x": x, "labels": labels, "datasets": datasets}


def run_case(args, seed, num_probes, pt_dbm, out):
    from models.ris_net import MLPModel
    prepared = prepare_case(args, seed, num_probes, pt_dbm)
    Config, geometry, scene, ns, channels, rho, phases, y, x, labels, datasets = (
        prepared[k] for k in ("config", "geometry", "scene", "ns", "channels", "rho",
                              "phases", "y", "x", "labels", "datasets"))
    def fresh():
        return MLPModel(2 * num_probes + 3, geometry.elements_per_tile, args.hidden_dim, 2, 0.0)
    case_dir = out / f"seed_{seed}" / f"M{num_probes}_Pt{pt_dbm:g}"
    case_dir.mkdir(parents=True, exist_ok=True)
    prov = provenance(args, geometry, scene)
    prov.update({"schema": SCHEMA, "observation_model": "passive receiver feedback",
                 "seed": seed, "num_probes": num_probes, "tx_power_dbm": pt_dbm,
                 "noise_power_dbm": Config.NOISE_POWER_DBM,
                 "label_acquisition": "N+1 orthogonal DFT probes at training SNR",
                 "input_dim": 2 * num_probes + 3, "hidden_dim": args.hidden_dim,
                 "num_layers": 2, "model_type": "MLP", "dropout": 0.0,
                 "sample_counts_train_validation_test": ns,
                 "codebook_sha256": hashlib.sha256(phases.tobytes()).hexdigest()})
    if args.skip_training:
        models, meta = {}, {"status": "not_run"}
    else:
        models, meta = train_comparison(fresh, datasets[0], datasets[1], Config,
            absolute_phases=True, progress=lambda p: write_json(case_dir / "training_progress.json", p))
        torch.save({"provenance": prov,
                    "weights": {k: [m.state_dict() for m in v] if isinstance(v, list) else v.state_dict()
                                for k, v in models.items()}}, case_dir / "models.pt")
    # Every estimator gets exactly the same noisy training-label acquisition.
    lmmse = LMMSEEstimator(labels[0], phases)
    lmmse_theta = mrc_from_estimate(lmmse.estimate(y[2], rho), geometry.num_tiles)
    local_theta = []
    for t in range(geometry.num_tiles):
        start = 1 + t * geometry.elements_per_tile
        local_targets = np.concatenate([labels[0][:, :1],
                                        labels[0][:, start:start + geometry.elements_per_tile]], axis=1)
        estimator = LocalProbeRegressor(y[0], local_targets)
        local_theta.append(mrc_from_estimate(estimator.estimate(y[2]), 1)[:, 0])
    full_ls = ls_full_estimate(x[2], rho, np.random.default_rng(seed + 32))
    designs = {"no_ris": (None, 0),
               "random_max": (random_max_sampling(y[2], phases, geometry.num_tiles), num_probes),
               "lmmse_mrc": (lmmse_theta, num_probes),
               "local_linear_mrc": (np.stack(local_theta, axis=1), num_probes),
               "full_probe_ls_mrc": (mrc_from_estimate(full_ls, geometry.num_tiles), geometry.total_elements + 1),
               "oracle_mrc": (mrc_from_estimate(x[2], geometry.num_tiles), 0)}
    features = pilot_features(y[2], rho, geometry.tile_grid_coords())
    for k, model in models.items():
        designs[k] = (predict(model, features, Config.DEVICE), num_probes)
    scores = {k: score(channels[2], theta, rho, m, args.coherence_symbols)
              for k, (theta, m) in designs.items()}
    for k, v in scores.items():
        v["snr_gap_to_local_linear_mrc_db"] = (v["mean_received_snr_db"] -
                                               scores["local_linear_mrc"]["mean_received_snr_db"])
        v["net_rate_gap_to_local_linear_mrc"] = (v["net_spectral_efficiency"] -
                                                scores["local_linear_mrc"]["net_spectral_efficiency"])
        v["deployable"] = k != "oracle_mrc"
    accounting = {
        "coherence_symbols": args.coherence_symbols,
        "training_and_validation_probe_symbols": (ns[0] + ns[1]) *
                                                     (num_probes + geometry.total_elements + 1),
        "training_probe_symbols": ns[0] * (num_probes + geometry.total_elements + 1),
        "training_label_pilots_per_scene": geometry.total_elements + 1,
        "pilot_feedback_bytes_per_block_to_controller": num_probes * 8,
        "pilot_feedback_endpoint_bytes_per_block_to_tiles": num_probes * 8 * geometry.num_tiles,
        "phase_command_bytes_if_centralized_float32": geometry.total_elements * 4,
        "centralized_training_label_bytes": ns[0] * (geometry.total_elements + 1) * 8,
        "centralized_training_feedback_bytes": ns[0] * num_probes * 8,
        "acquisition_location": "receiver; feedback and DFT reconstruction at controller",
        "additional_centralized_training_upload_bytes": 0,
        "full_label_feedback_bytes_train_and_validation":
            (ns[0] + ns[1]) * (geometry.total_elements + 1) * 8,
        "limited_feedback_bytes_train_and_validation_to_controller":
            (ns[0] + ns[1]) * num_probes * 8,
        "label_delivery_endpoint_bytes_train_and_validation_to_tiles":
            (ns[0] + ns[1]) * geometry.num_tiles * (geometry.elements_per_tile + 1) * 8,
        "limited_feedback_endpoint_bytes_train_and_validation_to_tiles":
            (ns[0] + ns[1]) * geometry.num_tiles * num_probes * 8,
        "privacy_claim": False,
        "local_linear_training_communication_bytes": 0,
        "zero_communication_scope": "model exchange only; data delivery is counted separately",
        "scope": "payload bytes; no topology, energy or cycle-accurate claim",
    }
    result = {"meta": prov, "training": meta, "accounting": accounting, "scores": scores}
    write_json(case_dir / "results.json", result)
    return result


def summarize(results):
    groups = {}
    for r in results:
        key = f"M{r['meta']['num_probes']}_Pt{r['meta']['tx_power_dbm']:g}"
        groups.setdefault(key, []).append(r)
    summary = {"schema": SCHEMA, "groups": {}}
    for key, rs in groups.items():
        summary["groups"][key] = {"seeds": [r["meta"]["seed"] for r in rs], "scores": {
            scheme: {metric: seed_statistics([r["scores"][scheme][metric] for r in rs])
                     for metric in ("mean_received_snr_db", "spectral_efficiency", "net_spectral_efficiency",
                                    "ber_qpsk", "fraction_of_oracle_power", "snr_gap_to_local_linear_mrc_db",
                                    "net_rate_gap_to_local_linear_mrc")}
            for scheme in rs[0]["scores"]}}
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1024])
    ap.add_argument("--pilot-counts", nargs="+", type=int, default=[16, 64])
    ap.add_argument("--tx-powers-dbm", nargs="+", type=float, default=[30.0])
    ap.add_argument("--train-samples", type=int, default=600)
    ap.add_argument("--validation-samples", type=int, default=200)
    ap.add_argument("--test-samples", type=int, default=600)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--min-rounds", type=int, default=20)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--codebook-seed", type=int, default=2024)
    ap.add_argument("--coherence-symbols", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if min(args.pilot_counts + [args.coherence_symbols]) <= 0:
        ap.error("Pilot counts and coherence length must be positive")
    if min(args.train_samples, args.test_samples, args.validation_samples,
           args.rounds, args.patience, args.hidden_dim, args.threads) < 1:
        ap.error("Sample counts and training budgets must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        ap.error("Seeds must be distinct")
    out = Path(args.output or ("results/pilot_limited_quick" if args.quick else "results/pilot_limited"))
    seeds = args.seeds[:1] if args.quick else args.seeds
    results = []
    for m in args.pilot_counts:
        for pt in args.tx_powers_dbm:
            for seed in seeds:
                results.append(run_case(args, seed, m, pt, out))
                write_json(out / "summary.json", summarize(results))
    print(f"[done] passive feedback results -> {out}")


if __name__ == "__main__":
    main()
