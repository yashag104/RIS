"""
Main training script for Federated Learning on RIS Tiles
Orchestrates the complete FL process and evaluation
"""

import os
import sys
import torch
import numpy as np
import json
import pickle
from torch.utils.data import DataLoader
from datetime import datetime

# Import project modules
from config import Config
from models.ris_net import create_model
from src.dataset_utils import create_non_iid_datasets, create_test_dataset, save_datasets
from src.client import RISClient
from src.server import FederatedServer
from utils.metrics import *
from utils.metrics import dbm_to_watts
from utils.plotting import *


def setup_directories(config):
    """Create necessary directories for saving results"""
    directories = [
        config.RESULTS_DIR,
        config.MODELS_DIR,
        config.PLOTS_DIR,
        config.METRICS_DIR,
        config.DATA_DIR
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    print(f"[OK] Directories created")


def initialize_datasets(config):
    """Create or load datasets"""
    print("\n" + "=" * 60)
    print("DATASET PREPARATION")
    print("=" * 60)

    dataset_path = os.path.join(config.DATA_DIR, 'datasets.pkl')

    # Check if datasets exist
    if os.path.exists(dataset_path):
        print("Loading existing datasets...")
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        train_datasets = data['train_datasets']
        test_dataset = data['test_dataset']
        tile_positions = data.get('tile_positions', None)
    else:
        print("Generating new datasets...")
        train_datasets, tile_positions = create_non_iid_datasets(config, config.NUM_TILES)
        test_dataset = create_test_dataset(config)

        # Save datasets
        data = {
            'train_datasets': train_datasets,
            'test_dataset': test_dataset,
            'tile_positions': tile_positions
        }
        with open(dataset_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[OK] Datasets saved to {dataset_path}")

    # Print dataset info
    print(f"\nDataset Summary:")
    print(f"  Number of RIS Tiles: {len(train_datasets)}")
    print(f"  Samples per Tile: {len(train_datasets[0])}")
    print(f"  Test Samples: {len(test_dataset)}")
    print(f"  Input Dimension: {train_datasets[0].get_input_dim()}")
    print(f"  RIS Elements per Tile: {config.ELEMENTS_PER_TILE}")

    return train_datasets, test_dataset, tile_positions


def initialize_models(config, input_dim):
    """Initialize global model and client models"""
    print("\n" + "=" * 60)
    print("MODEL INITIALIZATION")
    print("=" * 60)

    # Create global model using factory
    global_model = create_model(
        model_type=config.MODEL_TYPE,
        input_dim=input_dim,
        num_elements=config.ELEMENTS_PER_TILE,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
        config=config
    )

    print(f"[OK] Global model ({config.MODEL_TYPE}) created")
    total_params = global_model.count_parameters()
    model_size_bytes = total_params * getattr(config, 'COMM_BYTES_PER_PARAM', 1)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size (FP32): {total_params * 4 / 1024:.2f} KB")
    print(f"  Model size (INT8 for FL comm): {model_size_bytes / 1024:.2f} KB")
    per_round_bytes = model_size_bytes * 2 * config.NUM_TILES  # upload + download
    total_fl_bytes = per_round_bytes * config.FL_ROUNDS
    print(f"  Expected comm per round: {per_round_bytes / 1024:.2f} KB "
          f"(= {model_size_bytes} B x 2 x {config.NUM_TILES} tiles)")
    print(f"  Expected total comm ({config.FL_ROUNDS} rounds): "
          f"{total_fl_bytes / 1024:.2f} KB = {total_fl_bytes / (1024*1024):.2f} MB")

    return global_model


def train_federated(config, train_datasets, test_dataset):
    """Main federated learning training loop"""
    print("\n" + "=" * 60)
    print("FEDERATED LEARNING TRAINING")
    print("=" * 60)

    # Initialize datasets and models
    input_dim = train_datasets[0].get_input_dim()
    global_model = initialize_models(config, input_dim)

    # Create server
    server = FederatedServer(global_model, config)

    # Create clients
    clients = []
    for i, dataset in enumerate(train_datasets):
        # Each client gets a copy of the model
        client_model = create_model(
            model_type=config.MODEL_TYPE,
            input_dim=input_dim,
            num_elements=config.ELEMENTS_PER_TILE,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            config=config
        )
        client = RISClient(i, client_model, dataset, config)
        clients.append(client)

    print(f"[OK] Created {len(clients)} RIS tile clients")

    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # Training loop
    print(f"\nStarting Federated Learning for {config.FL_ROUNDS} rounds...")
    print(f"Local epochs per round: {config.LOCAL_EPOCHS}")

    all_round_metrics = []

    for round_num in range(config.FL_ROUNDS):
        # Execute one FL round
        round_metric = server.aggregate_round(clients, round_num)

        # Evaluate global model periodically
        if round_num % 5 == 0 or round_num == config.FL_ROUNDS - 1:
            print(f"\n[Evaluation] Testing global model...")

            # Set global model to first client for evaluation
            clients[0].set_model_weights(server.get_global_weights())
            eval_metrics = clients[0].evaluate(test_loader)
            snr_metrics = clients[0].compute_snr_improvement(test_dataset)

            round_metric['eval_metrics'] = eval_metrics
            round_metric['snr_metrics'] = snr_metrics

            print(f"  Test Loss: {eval_metrics['loss']:.6f}")
            print(f"  Phase Error: {eval_metrics['phase_error_mean']:.4f} rad "
                  f"({np.rad2deg(eval_metrics['phase_error_mean']):.2f}°)")
            print(f"  SNR (Optimized): {snr_metrics['snr_optimized_ris_mean']:.2f} dB")
            print(f"  SNR Gain: {snr_metrics['snr_gain_over_no_ris']:.2f} dB")

        all_round_metrics.append(round_metric)

        # Save checkpoint periodically
        if round_num % config.SAVE_EVERY_N_ROUNDS == 0:
            checkpoint_path = os.path.join(config.MODELS_DIR, f'checkpoint_round_{round_num}.pt')
            torch.save({
                'round': round_num,
                'global_model_state': server.get_global_weights(),
                'round_metrics': all_round_metrics
            }, checkpoint_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Communication sanity check
    comm_summary = server.get_communication_summary()
    model_params = sum(p.numel() for p in server.global_model.parameters())
    expected_total = (model_params * config.COMM_BYTES_PER_PARAM *
                      2 * len(clients) * config.FL_ROUNDS)
    actual_total = comm_summary['total_bytes']
    print(f"\n  Communication check:")
    print(f"    Model parameters: {model_params:,}")
    print(f"    Expected total: {expected_total / (1024*1024):.2f} MB")
    print(f"    Actual total:   {actual_total / (1024*1024):.2f} MB")
    if abs(actual_total - expected_total) > expected_total * 0.01:
        print(f"    WARNING: Communication mismatch! "
              f"Ratio = {actual_total / expected_total:.2f}x")

    return server, clients, all_round_metrics


def evaluate_baselines(config, test_dataset):
    """Evaluate baseline methods for comparison"""
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION")
    print("=" * 60)

    baselines = {}
    num_samples = min(100, len(test_dataset))

    # No RIS baseline
    print("Evaluating: No RIS (direct link)")
    snr_no_ris = []
    for i in range(num_samples):
        metadata = test_dataset.metadata[i]
        h_direct = metadata['H_direct'][0]

        noise_power = dbm_to_watts(config.NOISE_POWER_DBM)
        tx_power = dbm_to_watts(config.TX_POWER_DBM)
        signal = tx_power * np.abs(h_direct) ** 2
        snr = 10 * np.log10(signal / noise_power)
        snr_no_ris.append(snr)

    baselines['no_ris'] = {
        'snr': np.mean(snr_no_ris),
        'rate': calculate_achievable_rate(np.mean(snr_no_ris))
    }
    print(f"  SNR: {baselines['no_ris']['snr']:.2f} dB")
    print(f"  Rate: {baselines['no_ris']['rate']:.2f} bps/Hz")

    # Random RIS baseline
    print("\nEvaluating: Random RIS")
    snr_random = []
    for i in range(num_samples):
        metadata = test_dataset.metadata[i]
        h_direct = metadata['H_direct'][0]
        h_ris = metadata['H_ris'][0]
        h_bs_ris = metadata['h_bs_ris']
        h_cascade = h_ris * h_bs_ris

        random_phases = np.random.uniform(0, 2 * np.pi, len(h_ris))
        h_total = h_direct + np.sum(h_cascade * np.exp(1j * random_phases))

        noise_power = dbm_to_watts(config.NOISE_POWER_DBM)
        tx_power = dbm_to_watts(config.TX_POWER_DBM)
        signal = tx_power * np.abs(h_total) ** 2
        snr = 10 * np.log10(signal / noise_power)
        snr_random.append(snr)

    baselines['random_ris'] = {
        'snr': np.mean(snr_random),
        'rate': calculate_achievable_rate(np.mean(snr_random))
    }
    print(f"  SNR: {baselines['random_ris']['snr']:.2f} dB")
    print(f"  Rate: {baselines['random_ris']['rate']:.2f} bps/Hz")

    # Optimal (Genie-aided) baseline
    print("\nEvaluating: Optimal (Genie-aided)")
    snr_optimal = []
    for i in range(num_samples):
        metadata = test_dataset.metadata[i]
        h_direct = metadata['H_direct'][0]
        h_ris = metadata['H_ris'][0]
        h_bs_ris = metadata['h_bs_ris']
        h_cascade = h_ris * h_bs_ris

        # Recompute true MRC-optimal phases from raw channels
        true_optimal_phases = np.mod(
            np.angle(h_direct) - np.angle(h_cascade), 2 * np.pi
        )
        h_total = h_direct + np.sum(h_cascade * np.exp(1j * true_optimal_phases))

        noise_power = dbm_to_watts(config.NOISE_POWER_DBM)
        tx_power = dbm_to_watts(config.TX_POWER_DBM)
        signal = tx_power * np.abs(h_total) ** 2
        snr = 10 * np.log10(signal / noise_power)
        snr_optimal.append(snr)

    baselines['optimal'] = {
        'snr': np.mean(snr_optimal),
        'rate': calculate_achievable_rate(np.mean(snr_optimal))
    }
    print(f"  SNR: {baselines['optimal']['snr']:.2f} dB")
    print(f"  Rate: {baselines['optimal']['rate']:.2f} bps/Hz")

    return baselines


def save_results(config, server, clients, all_round_metrics, baselines, test_dataset):
    """Save all results, metrics, and generate plots"""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config.RESULTS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Save configuration
    config_dict = Config.get_config_dict()
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config_dict, f, indent=4, default=str)
    print("[OK] Configuration saved")

    # 2. Save final model
    torch.save(server.get_global_weights(), os.path.join(run_dir, 'final_model.pt'))
    print("[OK] Final model saved")

    # 3. Compile comprehensive metrics
    # Communication summary
    comm_summary = server.get_communication_summary()

    # Convergence metrics (may be partial)
    convergence = server.get_convergence_metrics() or {}

    # Final evaluation
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    clients[0].set_model_weights(server.get_global_weights())
    final_eval = clients[0].evaluate(test_loader)
    final_snr = clients[0].compute_snr_improvement(test_dataset, num_samples=200)

    # ---- Ensure reduction_percentage exists in convergence ----
    # Try to obtain initial and final loss from available places
    initial_loss = convergence.get('initial_loss')
    final_loss = convergence.get('final_loss')

    # fallback: use first round loss from all_round_metrics if present
    try:
        if initial_loss is None and isinstance(all_round_metrics, (list, tuple)) and len(all_round_metrics) > 0:
            initial_loss = all_round_metrics[0].get('loss', None)
    except Exception:
        initial_loss = None

    # fallback: use final_eval['loss'] if convergence didn't provide final
    try:
        if final_loss is None:
            final_loss = final_eval.get('loss', None)
    except Exception:
        final_loss = None

    # compute percentage reduction safely
    reduction_percentage = None
    try:
        if initial_loss is not None and final_loss is not None:
            init_f = float(initial_loss)
            fin_f = float(final_loss)
            if init_f != 0:
                reduction_percentage = (init_f - fin_f) / init_f * 100.0
            else:
                # initial_loss is zero: cannot compute meaningful reduction
                reduction_percentage = None
    except (ValueError, TypeError):
        reduction_percentage = None

    # set (or overwrite) into convergence dict so downstream code can rely on it
    convergence['initial_loss'] = initial_loss
    convergence['final_loss'] = final_loss
    convergence['reduction_percentage'] = reduction_percentage
    # keep existing converged_round if present, otherwise None
    convergence.setdefault('converged_round', None)

    # Compile all metrics
    all_metrics = {
        'round_metrics': all_round_metrics,
        'comm_summary': comm_summary,
        'convergence': convergence,
        'final_evaluation': final_eval,
        'snr_metrics': final_snr,
        'baselines': baselines,
        'achievable_rate_mean': calculate_achievable_rate(final_snr.get('snr_optimized_ris_mean'))
    }

    # Save metrics
    with open(os.path.join(run_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(all_metrics, f)

    # Save metrics as JSON (excluding non-serializable data)
    metrics_json = {
        'convergence': {
            'converged_round': convergence.get('converged_round'),
            'initial_loss': convergence.get('initial_loss'),
            'final_loss': convergence.get('final_loss'),
            'reduction_percentage': convergence.get('reduction_percentage')
        },
        'communication': {
            'total_mb': comm_summary.get('total_megabytes'),
            'avg_latency_ms': comm_summary.get('avg_packet_latency_ms'),
            'energy_j': comm_summary.get('energy_communication_joules')
        },
        'performance': {
            'final_loss': final_eval.get('loss'),
            'phase_error_deg': float(final_eval.get('phase_error_mean', 0.0) * 180 / np.pi) if final_eval.get('phase_error_mean') is not None else None,
            'snr_optimized_db': float(final_snr.get('snr_optimized_ris_mean')) if final_snr.get('snr_optimized_ris_mean') is not None else None,
            'snr_gain_db': float(final_snr.get('snr_gain_over_no_ris')) if final_snr.get('snr_gain_over_no_ris') is not None else None,
            'achievable_rate_bps_hz': float(all_metrics.get('achievable_rate_mean')) if all_metrics.get('achievable_rate_mean') is not None else None
        },
        'baselines': baselines
    }

    with open(os.path.join(run_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_json, f, indent=4, default=float)

    print("[OK] Metrics saved")

    # 4. Generate plots
    print("\nGenerating plots...")

    # Convergence curve
    plot_convergence_curve(all_round_metrics, plots_dir)
    print("  [OK] Convergence curve")

    # SNR comparison
    plot_snr_comparison(final_snr, plots_dir)
    print("  [OK] SNR comparison")

    # Communication overhead
    plot_communication_overhead(all_round_metrics, plots_dir)
    print("  [OK] Communication overhead")

    # Energy consumption
    plot_energy_consumption(all_round_metrics, plots_dir)
    print("  [OK] Energy consumption")

    # Trade-off curves
    plot_tradeoff_curves(all_round_metrics, None, plots_dir)
    print("  [OK] Trade-off curves")

    # Client performance
    plot_client_performance(all_round_metrics, plots_dir)
    print("  [OK] Per-client performance")

    # NoC metrics
    plot_noc_metrics(comm_summary, plots_dir, round_metrics=all_round_metrics)
    print("  [OK] NoC metrics")

    # Beam pattern (sample)
    sample_idx = 0
    features, optimal_phases = test_dataset[sample_idx]
    features = features.unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        predicted_phases = clients[0].model(features).squeeze().cpu().numpy()
    plot_beam_pattern(predicted_phases, test_dataset.metadata[sample_idx], plots_dir)
    print("  [OK] Beam pattern")

    # Phase heatmap
    plot_phase_heatmap(predicted_phases, optimal_phases.numpy(), save_path=plots_dir)
    print("  [OK] Phase heatmap")

    # Summary dashboard
    # create_summary_dashboard relied on 'reduction_percentage' earlier; we ensured it above
    create_summary_dashboard(all_metrics, plots_dir)
    print("  [OK] Summary dashboard")

    # 5. Create comparison table
    fl_metrics = {
        'snr': final_snr.get('snr_optimized_ris_mean'),
        'rate': all_metrics.get('achievable_rate_mean'),
        'communication_kb': comm_summary.get('total_kilobytes'),
        'energy_mj': sum([m.get('total_energy', 0) * 1000 for m in all_round_metrics]) if all_round_metrics else 0,
        'rounds': convergence.get('converged_round')
    }

    comparison = create_comparison_table(fl_metrics, baselines)

    # Save comparison table
    import pandas as pd
    df = pd.DataFrame(comparison)
    df.to_csv(os.path.join(run_dir, 'comparison_table.csv'), index=False)
    print("[OK] Comparison table saved")

    # Print comparison table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(df.to_string(index=False))

    print(f"\n[OK] All results saved to: {run_dir}")

    return run_dir

def main():
    """Main execution function"""
    # Set random seeds for reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print("\n" + "=" * 60)
    print("FEDERATED LEARNING FOR DISTRIBUTED RIS TILES")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Number of Tiles: {Config.NUM_TILES}")
    print(f"Elements per Tile: {Config.ELEMENTS_PER_TILE}")
    print(f"FL Rounds: {Config.FL_ROUNDS}")

    # Setup
    setup_directories(Config)

    # Prepare datasets
    train_datasets, test_dataset, tile_positions = initialize_datasets(Config)

    # Evaluate baselines
    baselines = evaluate_baselines(Config, test_dataset)

    # Train federated model
    server, clients, all_round_metrics = train_federated(Config, train_datasets, test_dataset)

    # Save results and generate visualizations
    run_dir = save_results(Config, server, clients, all_round_metrics, baselines, test_dataset)

    print("\n" + "=" * 60)
    print("EXECUTION COMPLETE!")
    print("=" * 60)
    print(f"\nResults directory: {run_dir}")


if __name__ == "__main__":
    main()
