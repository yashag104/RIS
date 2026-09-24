#!/usr/bin/env python
"""Corrected contiguous-aperture SISO audit (supplied-CSI diagnostic).

python run_link_level.py --skip-training          # classical controls, five seeds
python run_link_level.py --quick                  # isolated smoke output
python run_link_level.py --rounds 100 --model GNN # validation-controlled study

Passive receiver-feedback experiments live in run_pilot_limited.py. Existing
legacy results are never silently reused. Each run stores its configuration,
source hashes, seeds and individual results; cached models must match all of it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

RESULTS_SUBDIR = "link_level_corrected"
QUICK_RESULTS_SUBDIR = "link_level_corrected_quick"
SCHEMA = "contiguous-siso-v1"
_SOURCE_HASHES = None


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (tuple, list)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.generic):
        return _jsonable(o.item())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def seed_everything(seed, threads=1):
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure(args, Config):
    Config.SEED = args.seed
    Config.NUM_USERS = 1
    Config.SHARED_SCENE_TILES = True
    Config.TRAINING_OBJECTIVE = "sumrate"
    Config.SLEEP_SCHEDULING_ENABLED = False
    Config.TILE_GRID_ROWS = Config.TILE_GRID_COLS = 2 if args.quick else 4
    Config.NUM_TILES = Config.TILE_GRID_ROWS * Config.TILE_GRID_COLS
    Config.TRAIN_SAMPLES = 64 if args.quick else args.train_samples
    Config.TEST_SAMPLES = 32 if args.quick else args.test_samples
    Config.VALIDATION_SAMPLES = 32 if args.quick else args.validation_samples
    Config.FL_ROUNDS = 2 if args.quick else args.rounds
    Config.LOCAL_EPOCHS = 1 if args.quick else args.local_epochs
    Config.BATCH_SIZE = 32 if args.quick else args.batch_size
    Config.MIN_ROUNDS = min(args.min_rounds, Config.FL_ROUNDS)
    Config.PATIENCE, Config.MIN_DELTA = args.patience, args.min_delta
    Config.MODEL_TYPE = args.model
    Config.TOTAL_RIS_ELEMENTS = Config.NUM_TILES * Config.ELEMENTS_PER_TILE
    return Config


def provenance(args, geometry, scene):
    # Pin the manifest at the start of this process's study. Later edits on
    # disk do not change the Python modules already loaded by a running study.
    global _SOURCE_HASHES
    if _SOURCE_HASHES is None:
        paths = [Path("run_link_level.py"), Path("run_pilot_limited.py"), Path("run_pilot_ablation.py"), Path("config.py")]
        for folder in ("src", "models", "baselines", "experiments"):
            paths += sorted(Path(folder).glob("*.py"))
        _SOURCE_HASHES = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in paths if p.exists()}
    source = dict(_SOURCE_HASHES)
    return {"schema": SCHEMA, "arguments": vars(args).copy(),
            "geometry": asdict(geometry), "scene": asdict(scene),
            "aperture_m": geometry.aperture_m, "source_sha256": source,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "numpy": np.__version__, "torch": torch.__version__,
            "observation_model": "supplied noisy cascaded CSI; not a passive sensing implementation"}


def build_channels(Config, cache_path=None, force=False):
    """Generate full-aperture fields before slicing; never read legacy caches."""
    from src.surface_channel import (
        generate_surface_channels,
        geometry_from_config,
        scene_from_config,
        surface_datasets,
    )
    geometry, scene = geometry_from_config(Config), scene_from_config(Config)
    sets = [generate_surface_channels(n, geometry, scene, np.random.default_rng(Config.SEED + s))
            for n, s in ((Config.TRAIN_SAMPLES, 0), (Config.VALIDATION_SAMPLES, 1),
                         (Config.TEST_SAMPLES, 2))]
    train = surface_datasets(sets[0], geometry, Config.CSI_ERROR_VARIANCE, Config.SEED + 10)
    validation = surface_datasets(sets[1], geometry, Config.CSI_ERROR_VARIANCE, Config.SEED + 11)
    return train, validation, sets[2], geometry, scene


def train_models(Config, train, validation, cache_path, force, fingerprint):
    from models.ris_net import create_model
    from src.controlled_training import train_comparison
    def fresh():
        return create_model(Config.MODEL_TYPE, train[0].features.shape[1], Config.ELEMENTS_PER_TILE,
                            Config.HIDDEN_DIM, Config.NUM_LAYERS, Config.DROPOUT, config=Config)
    if Path(cache_path).exists() and not force:
        blob = torch.load(cache_path, weights_only=False, map_location="cpu")
        if blob.get("fingerprint") == fingerprint:
            models = {}
            for k, v in blob["weights"].items():
                seq = v if isinstance(v, list) else [v]
                ms = []
                for state in seq:
                    m = fresh()
                    m.load_state_dict(state)
                    ms.append(m)
                models[k] = ms if isinstance(v, list) else ms[0]
            return models, blob["training_meta"]
    models, meta = train_comparison(fresh, train, validation, Config,
        progress=lambda p: write_json(Path(cache_path).with_suffix(".progress.json"), p))
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"fingerprint": fingerprint, "training_meta": meta,
                "weights": {k: [m.state_dict() for m in v] if isinstance(v, list) else v.state_dict()
                            for k, v in models.items()}}, cache_path)
    return models, meta


def seed_statistics(values):
    """Student-t interval across independent seed runs, not across correlated tiles."""
    from scipy.stats import t
    a = np.asarray(values, dtype=float)
    mean = a.mean(axis=0)
    half = t.ppf(.975, len(a) - 1) * a.std(axis=0, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else None
    return {"mean": _jsonable(mean), "ci95_half_width": _jsonable(half), "n_seeds": len(a)}


def summarize_seeds(runs):
    summary = {"seeds": [r["meta"]["seed"] for r in runs],
               "independent_unit": "training and channel seed", "metrics": {}}
    for k in runs[0]["mean_gain_db"]:
        values = [r["mean_gain_db"][k] for r in runs]
        summary["metrics"][k] = {"mean_gain_db": seed_statistics(values),
            "paired_gain_db_vs_local_mrc": seed_statistics([
                r["mean_gain_db"][k] - r["mean_gain_db"]["local_mrc"] for r in runs])}
    for block, metric in (("csi_robustness", "mean_snr_db"),
                          ("csi_robustness", "gap_to_local_mrc_db"),
                          ("array_scaling", "reflected_only_snr_db")):
        summary[f"{block}.{metric}"] = {
            k: seed_statistics([r[block][metric][k] for r in runs])
            for k in runs[0][block][metric]}
    return summary


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-training", action="store_true")
    ap.add_argument("--train-samples", type=int, default=600)
    ap.add_argument("--test-samples", type=int, default=600)
    ap.add_argument("--validation-samples", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--min-rounds", type=int, default=20)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--model", choices=["GNN", "MLP"], default="GNN")
    ap.add_argument("--seed", type=int, default=None, help="one independent seed (default: five seeds)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--output", default=None)
    return ap


def main():
    args = parser().parse_args()
    if min(args.train_samples, args.test_samples, args.validation_samples,
           args.rounds, args.local_epochs, args.batch_size, args.threads, args.patience) < 1:
        raise ValueError("Sample counts, budgets and thread count must be positive")
    if args.seed is not None and args.seeds is not None:
        raise ValueError("Use --seed or --seeds, not both")
    from config import Config
    from experiments.link_level import LinkLevelSuite, build_scene_set
    from src.surface_channel import surface_datasets
    from utils.plotting_link import render_all
    subdir = QUICK_RESULTS_SUBDIR if args.quick else (RESULTS_SUBDIR if args.skip_training else
                                                     RESULTS_SUBDIR + "_" + args.model.lower())
    out = Path(args.output or Path(Config.RESULTS_DIR) / subdir)
    if args.plot_only:
        for p in sorted(out.glob("seed_*/link_level_results.json")):
            result = json.loads(p.read_text())
            if result["meta"].get("schema") != SCHEMA:
                raise ValueError("Refusing to render obsolete geometry results")
            render_all(result, str(p.parent))
        return
    seeds = args.seeds or ([args.seed] if args.seed is not None else
                          ([42] if args.quick else [42, 123, 456, 789, 1024]))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct for independent-run intervals")
    runs = []
    for seed in seeds:
        args.seed = seed
        configure(args, Config)
        seed_everything(seed, args.threads)
        train, validation, test, geometry, scene = build_channels(Config)
        prov = provenance(args, geometry, scene)
        fingerprint = hashlib.sha256(json.dumps(prov, sort_keys=True).encode()).hexdigest()
        seed_dir = out / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_training:
            models, meta = {}, {"status": "not_run"}
        else:
            models, meta = train_models(Config, train, validation, seed_dir / "models.pt",
                                         args.force, fingerprint)
        def builder(epsilon, test=test, geometry=geometry, seed=seed):
            return surface_datasets(test, geometry, epsilon, seed + 12)
        suite = LinkLevelSuite(Config, models, scene_builder=builder)
        result = suite.run(build_scene_set(builder(Config.CSI_ERROR_VARIANCE)))
        result["meta"].update({"schema": SCHEMA, "seed": seed, "training": meta,
                                "provenance": prov, "is_quick_run": args.quick,
                                "channel_fingerprint": fingerprint})
        result["meta"]["tile_mean_cascade_power"] = np.mean(np.abs(test["cascade"]) ** 2, axis=(0, 2))
        write_json(seed_dir / "link_level_results.json", result)
        render_all(result, str(seed_dir))
        runs.append(result)
        write_json(out / "summary.json", summarize_seeds(runs))
    print(f"[done] {len(runs)} independent seeds -> {out}")


if __name__ == "__main__":
    main()
