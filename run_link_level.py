#!/usr/bin/env python
"""Run the six link-level (physical-layer) results and render the paper figures.

    python run_link_level.py              # paper scale (16 tiles x 64 elements)
    python run_link_level.py --quick      # minutes, for checking the pipeline
    python run_link_level.py --plot-only  # re-render figures from saved JSON

Produces ``results/link_level/link_level_results.json`` plus six IEEE-style
figures. Channels and trained weights are cached under ``data/`` so a re-run
only repeats the parts that changed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch

RESULTS_SUBDIR = "link_level"


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def seed_everything(seed: int) -> None:
    # One process, no dataloader workers: let torch use every core.
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure(args, Config):
    """Apply the requested experiment scale to the shared Config object."""
    Config.SHARED_SCENE_TILES = True
    Config.TRAINING_OBJECTIVE = "sumrate"
    # Sleep scheduling would drop tiles from individual rounds, which changes the
    # trained model for reasons unrelated to anything measured here.
    Config.SLEEP_SCHEDULING_ENABLED = False

    if args.quick:
        Config.TILE_GRID_ROWS, Config.TILE_GRID_COLS = 2, 2
        Config.NUM_TILES = 4
        Config.TRAIN_SAMPLES = 120
        Config.TEST_SAMPLES = 120
        Config.FL_ROUNDS = 3
        Config.LOCAL_EPOCHS = 1
        Config.BATCH_SIZE = 32
    else:
        Config.TILE_GRID_ROWS, Config.TILE_GRID_COLS = 4, 4
        Config.NUM_TILES = 16
        Config.TRAIN_SAMPLES = args.train_samples
        Config.TEST_SAMPLES = args.test_samples
        Config.FL_ROUNDS = args.rounds
        Config.LOCAL_EPOCHS = args.local_epochs
        Config.BATCH_SIZE = 64

    Config.TOTAL_RIS_ELEMENTS = Config.NUM_TILES * Config.ELEMENTS_PER_TILE
    return Config


def build_channels(Config, cache_path: str, force: bool):
    """Generate (or load) the shared-scene train and test channels."""
    from src.channel_model import generate_multi_tile_channels
    from src.dataset_utils import create_non_iid_datasets

    if os.path.exists(cache_path) and not force:
        print(f"[cache] loading channels from {cache_path}")
        blob = torch.load(cache_path, weights_only=False)
        return blob["train_datasets"], blob["test_channels"], blob["tile_positions"]

    t0 = time.time()
    print(f"[gen] train channels: {Config.NUM_TILES} tiles x {Config.TRAIN_SAMPLES} samples")
    train_datasets, tile_positions = create_non_iid_datasets(Config, Config.NUM_TILES)

    print(f"[gen] test channels: {Config.NUM_TILES} tiles x {Config.TEST_SAMPLES} samples")
    test_channels = generate_multi_tile_channels(
        num_samples=Config.TEST_SAMPLES,
        tile_positions=tile_positions,
        elements_per_tile=Config.ELEMENTS_PER_TILE,
        num_users=Config.NUM_USERS,
        room_size=Config.ROOM_SIZE,
        frequency=Config.FREQUENCY,
        k_factor_db=Config.RICIAN_K_FACTOR_DB,
        num_paths=Config.NUM_PATHS,
        spatial_corr_rho=Config.SPATIAL_CORRELATION_RHO,
        scenario=Config.CHANNEL_SCENARIO,
        grid_rows=Config.PIXEL_GRID_ROWS,
        grid_cols=Config.PIXEL_GRID_COLS,
        direct_link_blockage_db=Config.DIRECT_LINK_BLOCKAGE_DB,
        element_gain_enabled=Config.RIS_ELEMENT_GAIN_ENABLED,
    )
    print(f"[gen] done in {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {"train_datasets": train_datasets, "test_channels": test_channels,
         "tile_positions": tile_positions},
        cache_path,
    )
    return train_datasets, test_channels, tile_positions


def train_models(Config, train_datasets, cache_path: str, force: bool):
    """Train the federated model and the pooled-data centralized control.

    Both use :class:`src.client.RISClient` -- the same architecture, objective,
    optimizer and schedule -- so the only difference is whether the data stayed
    on the tiles.
    """
    from models.ris_net import create_model
    from src.client import RISClient
    from src.dataset_utils import RISChannelDataset
    from src.server import FederatedServer

    input_dim = train_datasets[0].features.shape[1]

    def fresh_model():
        return create_model(
            model_type=Config.MODEL_TYPE,
            input_dim=input_dim,
            num_elements=Config.ELEMENTS_PER_TILE,
            hidden_dim=Config.HIDDEN_DIM,
            num_layers=Config.NUM_LAYERS,
            dropout=Config.DROPOUT,
            config=Config,
        )

    if os.path.exists(cache_path) and not force:
        print(f"[cache] loading trained weights from {cache_path}")
        blob = torch.load(cache_path, weights_only=False)
        fed, cen = fresh_model(), fresh_model()
        fed.load_state_dict(blob["fed_ris"])
        cen.load_state_dict(blob["centralized_dl"])
        return {"fed_ris": fed, "centralized_dl": cen}, blob["training_meta"]

    # ---- federated ----
    t0 = time.time()
    server = FederatedServer(fresh_model(), Config)
    clients = [RISClient(i, fresh_model(), ds, Config) for i, ds in enumerate(train_datasets)]
    round_metrics = []
    for r in range(Config.FL_ROUNDS):
        m = server.aggregate_round(clients, r)
        round_metrics.append(m)
        print(f"  [FL] round {r + 1}/{Config.FL_ROUNDS} "
              f"loss={m.get('avg_client_loss', float('nan')):.5f}")
    fed_time = time.time() - t0

    fed_model = fresh_model()
    fed_model.load_state_dict(server.get_global_weights())

    # ---- centralized control: identical learner, pooled data ----
    t0 = time.time()
    pooled = RISChannelDataset.concat(train_datasets)
    cen_client = RISClient(-1, fresh_model(), pooled, Config)
    total_epochs = Config.FL_ROUNDS * Config.LOCAL_EPOCHS
    cen_hist = cen_client.train_local_model(epochs=total_epochs)
    cen_time = time.time() - t0
    cen_model = cen_client.model

    comm = server.get_communication_summary()
    meta = {
        "fl_rounds": Config.FL_ROUNDS,
        "local_epochs": Config.LOCAL_EPOCHS,
        "train_samples_per_tile": Config.TRAIN_SAMPLES,
        "num_tiles": Config.NUM_TILES,
        "model_type": Config.MODEL_TYPE,
        "aggregation": Config.AGGREGATION_METHOD,
        "fl_wall_clock_s": fed_time,
        "centralized_wall_clock_s": cen_time,
        "fl_total_communication_kb": comm["total_kilobytes"],
        "fl_round_losses": [m.get("avg_client_loss") for m in round_metrics],
        "centralized_final_loss": cen_hist.get("avg_loss"),
        "model_parameters": sum(p.numel() for p in fed_model.parameters()),
        # What a centralized controller would have had to ship instead: every
        # tile's raw CSI feature vector, every round.
        "raw_csi_upload_kb": (
            Config.NUM_TILES * Config.TRAIN_SAMPLES * input_dim * 4 / 1024
        ),
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(
        {"fed_ris": fed_model.state_dict(), "centralized_dl": cen_model.state_dict(),
         "training_meta": meta},
        cache_path,
    )
    return {"fed_ris": fed_model, "centralized_dl": cen_model}, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="tiny scale, for pipeline checks")
    ap.add_argument("--plot-only", action="store_true", help="re-render from saved JSON")
    ap.add_argument("--force", action="store_true", help="ignore caches and regenerate")
    ap.add_argument("--train-samples", type=int, default=600)
    ap.add_argument("--test-samples", type=int, default=600)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from config import Config

    configure(args, Config)
    results_dir = os.path.join(Config.RESULTS_DIR, RESULTS_SUBDIR)
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "link_level_results.json")

    if args.plot_only:
        with open(json_path) as fh:
            results = json.load(fh)
        from utils.plotting_link import render_all
        render_all(results, results_dir)
        print(f"[done] figures re-rendered into {results_dir}")
        return

    seed_everything(args.seed)
    tag = "quick" if args.quick else "full"
    ch_cache = os.path.join(Config.DATA_DIR, f"link_level_channels_{tag}.pt")
    md_cache = os.path.join(Config.MODELS_DIR, f"link_level_models_{tag}.pt")
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)

    train_datasets, test_channels, _tile_positions = build_channels(Config, ch_cache, args.force)

    from src.dataset_utils import RISChannelDataset

    def scene_builder(csi_error_variance: float):
        """Per-tile test datasets over the same true channels at a given CSI quality."""
        return [
            RISChannelDataset.from_channels(
                tile_channels,
                num_ris_elements=Config.ELEMENTS_PER_TILE,
                num_users=Config.NUM_USERS,
                csi_error_variance=csi_error_variance,
            )
            for tile_channels in test_channels
        ]

    test_tile_datasets = scene_builder(Config.CSI_ERROR_VARIANCE)

    models, training_meta = train_models(Config, train_datasets, md_cache, args.force)

    from experiments.link_level import LinkLevelSuite, build_scene_set

    scenes = build_scene_set(test_tile_datasets)
    suite = LinkLevelSuite(Config, models, scene_builder=scene_builder)
    t0 = time.time()
    results = suite.run(scenes)
    results["meta"]["training"] = training_meta
    results["meta"]["is_quick_run"] = bool(args.quick)
    results["meta"]["seed"] = args.seed
    results["meta"]["wall_clock_s"] = time.time() - t0

    with open(json_path, "w") as fh:
        json.dump(_jsonable(results), fh, indent=1)
    print(f"[done] results -> {json_path}")

    from utils.plotting_link import render_all
    render_all(results, results_dir)
    print(f"[done] figures -> {results_dir}")


if __name__ == "__main__":
    main()
