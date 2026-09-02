"""Core helpers shared by all RIS experiments."""

"""Shared imports for the RIS experiment package."""

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.ris_net import create_model
from src.client import RISClient
from src.dataset_utils import (
    create_non_iid_datasets,
    create_test_dataset,
    validate_dataset_collection,
    validate_dataset_feature_dim,
)
from utils.metrics import *
from utils.metrics import dbm_to_watts
from utils.plotting import *

from .logging_utils import (
    configure_experiment_logging,
    get_experiment_logger,
    resolve_log_level,
)


class ExperimentBase:
    def __init__(self, config):
        self.config = config
        self.results_dir = os.path.join(config.RESULTS_DIR, 'advanced_experiments')
        os.makedirs(self.results_dir, exist_ok=True)
        configure_experiment_logging(
            self.results_dir,
            level=resolve_log_level(config),
        )
        self.logger = get_experiment_logger(self.__class__.__name__)

    def _run_single_fl_experiment(self, config_overrides=None):
        """
        Run standard FL experiment with current config and optional overrides.
        
        Args:
            config_overrides: Dictionary of config parameters to override for this run only
        """
        from main import evaluate_baselines, train_federated
        
        # Create a temporary config modification
        original_values = {}
        if config_overrides:
            for k, v in config_overrides.items():
                if hasattr(self.config, k):
                    original_values[k] = getattr(self.config, k)
                    setattr(self.config, k, v)
        
        try:
            # Create datasets (re-create if params changed that affect data)
            # For efficiency, we could cache them, but for robustness, we re-create
            train_datasets, _tile_positions = create_non_iid_datasets(self.config, self.config.NUM_TILES)
            test_dataset = create_test_dataset(self.config)
            validate_dataset_collection(
                train_datasets,
                test_dataset,
                self.config,
                expected_num_tiles=self.config.NUM_TILES,
            )

            # Evaluate baselines
            baselines = evaluate_baselines(self.config, test_dataset)

            # Train
            server, clients, round_metrics = train_federated(self.config, train_datasets, test_dataset)

            # Extract key metrics
            test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
            clients[0].set_model_weights(server.get_global_weights())
            final_eval = clients[0].evaluate(test_loader)
            final_snr = clients[0].compute_snr_improvement(test_dataset, num_samples=200)

            convergence = server.get_convergence_metrics()
            comm_summary = server.get_communication_summary()

            result = {
                'convergence_round': convergence.get('converged_round', self.config.FL_ROUNDS),
                'final_loss': final_eval['loss'],
                'final_snr': final_snr['snr_optimized_ris_mean'],
                'snr_gain': final_snr['snr_gain_over_no_ris'],
                'phase_error_deg': np.rad2deg(final_eval['phase_error_mean']),
                'total_communication_kb': comm_summary['total_kilobytes'],
                'total_energy_mj': sum([m.get('total_energy', 0) * 1000 for m in round_metrics]),
                'final_accuracy': final_eval['accuracy_30deg'],
                'round_metrics': round_metrics,
                'baselines': baselines,
                'global_weights': server.get_global_weights() # Return weights for wrappers
            }
            
            # Add quantization metadata if applicable
            if hasattr(self.config, 'PHASE_QUANTIZATION_BITS'):
                result['quantization_bits'] = self.config.PHASE_QUANTIZATION_BITS
                
            return result
            
        finally:
            # Restore original config values
            for k, v in original_values.items():
                setattr(self.config, k, v)

    def _run_fl_with_quantization(self, quant_config):
        """Run FL with phase quantization"""
        bits = quant_config['bits']
        if bits == 'continuous':
            bits = 0
            
        # Run real experiment with quantization enabled in config
        # RISClient will pick this up and apply quantization during evaluation
        result = self._run_single_fl_experiment(config_overrides={'PHASE_QUANTIZATION_BITS': bits})
        return result

    def _run_fl_with_compression(self, bits):
        """Run FL with model weight compression"""
        # First run standard training
        result = self._run_single_fl_experiment()
        
        # Now simulate compression on the global model for evaluation
        from models.ris_net import create_model
        
        # Create test dataset for re-evaluation
        test_dataset = create_test_dataset(self.config)
        validate_dataset_feature_dim(test_dataset, self.config, "compression test_dataset")
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        # Load weights into a temporary model
        temp_model = create_model(self.config.MODEL_TYPE, test_dataset.get_input_dim(),
                                self.config.ELEMENTS_PER_TILE, config=self.config)
        temp_model.to(self.config.DEVICE)
        
        # Real compression simulation: Quantize weights
        self.logger.info(f"  Compressing model to {bits}-bit...")
        global_weights = result['global_weights']
        quantized_weights = {}
        
        for name, param in global_weights.items():
            # Skip non-floating point params (like long integers)
            if 'int' in str(param.dtype) or 'long' in str(param.dtype):
                quantized_weights[name] = param
                continue
                
            # Determine range
            w_min = param.min().item()
            w_max = param.max().item()
            
            # 2^bits levels
            levels = 2 ** bits
            step = (w_max - w_min) / (levels - 1)
            
            if step == 0:
                quantized_weights[name] = param
            else:
                # Quantize: q = round((w - min) / step) * step + min
                w_q = torch.round((param - w_min) / step) * step + w_min
                quantized_weights[name] = w_q
        
        # Load quantized weights
        temp_model.load_state_dict(quantized_weights)
        
        # Evaluate real performance degradation
        # We need a client to run evaluation logic
        # Re-use the first client but with global test set
        temp_client = RISClient(0, temp_model, None, self.config) # Dataset not needed for pure eval with loader
        
        metrics = temp_client.evaluate(test_loader)
        
        # Also compute SNR
        # We need the dataset for compute_snr_improvement
        snr_metrics = temp_client.compute_snr_improvement(test_dataset, num_samples=200)
        
        result['accuracy_degradation'] = result['final_accuracy'] - metrics['accuracy_30deg']
        result['final_snr'] = snr_metrics['snr_optimized_ris_mean']
        result['final_accuracy'] = metrics['accuracy_30deg']
        
        # Update communication cost
        compression_ratio = 32 / bits
        result['total_communication_kb'] /= compression_ratio
        
        return result

    def _run_fl_with_mobility(self, speed_mps):
        """Run FL with user mobility"""
        # 1. Train model on initial positions
        result = self._run_single_fl_experiment()
        
        # 2. Evaluate on shifted positions to simulate movement
        if speed_mps > 0:
            self.logger.info(f"  Simulating mobility: {speed_mps} m/s...")
            # Jakes' Model for temporal correlation
            # rho = J0(2 * pi * fd * tau)
            # fd = v / lambda
            # tau = time elapsed (assume 100ms processing delay + flight time?)
            # Let's assume evaluation happens "time_delta" seconds after CSI acquisition
            time_delta = 0.05 # 50ms (typical 5G frame/processing delay)
            
            fd = speed_mps / self.config.WAVELENGTH
            np.i0(2 * np.pi * fd * time_delta) # approximation: numpy has i0 (modified Bessel). 
            # Wait, J0 is Bessel function of first kind. i0 is modified.
            # Numpy doesn't have j0 natively without scipy.
            # Standard approximation for small x: J0(x) ~ 1 - x^2/4
            # Or cosine approximation J0(x) ~ cos(x) ? No.
            # Let's import scipy if available, else simple AR1
            
            try:
                from scipy.special import j0
                correlation = j0(2 * np.pi * fd * time_delta)
            except ImportError:
                # Fallback: approximated correlation
                # For small x, J0(x) approx 1 - x^2/4
                arg = 2 * np.pi * fd * time_delta
                correlation = 1.0 - (arg**2) / 4.0
                correlation = max(correlation, 0)
            
            # Generate "aged" test dataset
            # h_new = rho * h_old + sqrt(1 - rho^2) * noise
            test_dataset = create_test_dataset(self.config)
            validate_dataset_feature_dim(test_dataset, self.config, "mobility test_dataset")
            
            aged_snrs = []
            
            # Use the trained model (from result) to predict on NEW channels
            # But the model sees OLD CSI (features).
            # Scenario: User moves, but we use OLD CSI to predict phase.
            # This measures "CSI out-datedness"
            
            # Retrieve model
            # We need to reconstruct the model state. 
            # We assume 'result' implies we have the model.
            # We need to reload the weights.
            model = create_model(self.config.MODEL_TYPE, test_dataset.get_input_dim(),
                               self.config.ELEMENTS_PER_TILE, config=self.config)
            model.load_state_dict(result['global_weights']) 
            model.to(self.config.DEVICE)
            model.eval()
            
            # Evaluate:
            # Input: OLD features (at t=0)
            # Channel: NEW channel (at t=delta)
            # This measures robustness to mobility
            
            noise_power = dbm_to_watts(self.config.NOISE_POWER_DBM)
            tx_power = dbm_to_watts(self.config.TX_POWER_DBM)
            
            for i in range(min(200, len(test_dataset))):
                 features, _ = test_dataset[i]
                 
                 # 1. Predict phases based on OLD features
                 with torch.no_grad():
                     pred_phases = model(features.unsqueeze(0).to(self.config.DEVICE)).cpu().numpy().flatten()
                 
                 # 2. Get OLD channel
                 metadata = test_dataset.metadata[i]
                 h_direct = metadata['H_direct'][0]
                 h_ris = metadata['H_ris'][0] # (elements,)
                 h_bs_ris = metadata['h_bs_ris']

                 # 3. Generate NEW channel (Jakes model)
                 # Add noise to represent aging
                 # Complex Gaussian noise
                 noise_direct = (np.random.randn(*h_direct.shape) + 1j * np.random.randn(*h_direct.shape)) / np.sqrt(2)
                 noise_ris = (np.random.randn(*h_ris.shape) + 1j * np.random.randn(*h_ris.shape)) / np.sqrt(2)

                 # h_new = rho * h + sqrt(1-rho^2) * independent_h
                 gain_direct = np.mean(np.abs(h_direct))
                 gain_ris = np.mean(np.abs(h_ris))

                 h_direct_new = correlation * h_direct + np.sqrt(1 - correlation**2) * noise_direct * gain_direct
                 h_ris_new = correlation * h_ris + np.sqrt(1 - correlation**2) * noise_ris * gain_ris

                 # 4. Compute SNR with OLD phases but NEW channel
                 # h_bs_ris is quasi-static (BS and RIS are fixed), not aged
                 h_cascade_new = h_ris_new * h_bs_ris
                 h_total = h_direct_new + np.sum(h_cascade_new * np.exp(1j * pred_phases))
                 signal = tx_power * np.abs(h_total) ** 2
                 snr = 10 * np.log10(signal / noise_power)
                 aged_snrs.append(snr)
            
            result['tracking_error'] = 1.0 - correlation
            result['adaptation_time'] = 5 + speed_mps * 2  # rounds (still heuristic)
            result['final_snr'] = np.mean(aged_snrs)
        else:
            # Static case: no mobility, perfect tracking
            result['tracking_error'] = 0.0
            result['adaptation_time'] = 0.0
            
        return result

    def _run_fl_with_pilots(self, pilot_config):
        """Run FL with different pilot strategies (Simulated Overhead)"""
        result = self._run_single_fl_experiment()
        
        pilot_config['method']
        pilots_per_round = pilot_config['pilots_per_round']
        
        # Calculate overhead based on real convergence rounds
        num_rounds = result['convergence_round']
        total_pilots = pilots_per_round * num_rounds
        overhead_bits = total_pilots * 16  # Assume 16 bits per pilot
        
        result['total_pilots'] = total_pilots
        result['overhead_bits'] = overhead_bits
        result['overhead_kb'] = overhead_bits / (8 * 1024)
        
        return result

    def _run_centralized_experiment(self):
        """Run centralized learning (All data at server)"""
        from baselines.centralized_learning import CentralizedRIS
        from models.ris_net import create_model
        
        # Create datasets
        train_datasets, _tile_positions = create_non_iid_datasets(self.config, self.config.NUM_TILES)
        test_dataset = create_test_dataset(self.config)
        
        input_dim = validate_dataset_collection(
            train_datasets,
            test_dataset,
            self.config,
            expected_num_tiles=self.config.NUM_TILES,
        )
        cent_model = create_model(self.config.MODEL_TYPE, input_dim, self.config.ELEMENTS_PER_TILE, config=self.config)
        
        centralized = CentralizedRIS(cent_model, self.config)
        cent_metrics = centralized.train_centralized(
            tile_datasets=train_datasets,
            epochs=self.config.LOCAL_EPOCHS * self.config.FL_ROUNDS
        )
        
        # Evaluate
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        centralized.evaluate(test_loader)
        
        # Calc SNR
        features, _optimal_phases = test_dataset[0]
        cent_model.eval()
        with torch.no_grad():
             cent_model(features.unsqueeze(0).to(self.config.DEVICE))
        # (Simplified SNR calc for summary)
        
        # Communication: all raw data
        total_samples = sum(len(d) for d in train_datasets)
        raw_data_bytes = total_samples * (input_dim + self.config.ELEMENTS_PER_TILE) * 4
        
        return {
            'convergence_round': cent_metrics['total_epochs'],
            'final_loss': cent_metrics['final_loss'],
            'final_snr': 0, # Placeholder, computed fully in baseline comparison
            'total_communication_kb': raw_data_bytes / 1024,
            'final_accuracy': 0, # computed elsewhere
            'total_energy_mj': 0
        }

    def _run_local_only_experiment(self):
        """Run local-only learning (no aggregation)"""
        from models.ris_net import create_model
        
        # Create datasets
        train_datasets, _tile_positions = create_non_iid_datasets(self.config, self.config.NUM_TILES)
        test_dataset = create_test_dataset(self.config)
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        input_dim = validate_dataset_collection(
            train_datasets,
            test_dataset,
            self.config,
            expected_num_tiles=self.config.NUM_TILES,
        )
        
        # Train isolated clients
        final_snrs = []
        final_accs = []
        
        total_epochs = self.config.FL_ROUNDS * self.config.LOCAL_EPOCHS
        
        # We simulate all clients running in parallel
        for i, dataset in enumerate(train_datasets):
            model = create_model(self.config.MODEL_TYPE, input_dim, self.config.ELEMENTS_PER_TILE, 
                               hidden_dim=self.config.HIDDEN_DIM, num_layers=self.config.NUM_LAYERS, 
                               dropout=self.config.DROPOUT, config=self.config)
            client = RISClient(i, model, dataset, self.config)
            
            # Train full duration
            client.train_local_model(epochs=total_epochs)
            
            # Evaluate this client's model on global test set
            # (In reality, local only models rarely generalize well globally, this captures that)
            metrics = client.evaluate(test_loader)
            snr_metrics = client.compute_snr_improvement(test_dataset, num_samples=50)
            
            final_accs.append(metrics['accuracy_30deg'])
            final_snrs.append(snr_metrics['snr_optimized_ris_mean'])
        
        return {
             'convergence_round': self.config.FL_ROUNDS,
             'final_loss': 0, # N/A
             'final_snr': np.mean(final_snrs),
             'snr_gain': 0, 
             'total_communication_kb': 0,
             'final_accuracy': np.mean(final_accs),
             'total_energy_mj': 0
        }

    def _calculate_noc_metrics(self, result):
        """Calculate Network-on-Chip metrics (Model-based)"""
        num_tiles = result.get('num_tiles', self.config.NUM_TILES)
        comm_kb = result['total_communication_kb']
        
        # Updated power model parameters
        static_power_mw = num_tiles * self.config.IDLE_POWER_TILE * 1000
        dynamic_power_mw = comm_kb * 0.05 # 0.05 mW per KB (approx)
        total_power_mw = static_power_mw + dynamic_power_mw
        
        # Latency model
        base_latency_us = 5
        congestion_factor = 1 + (num_tiles / 16.0) ** 2
        avg_latency_us = base_latency_us * congestion_factor
        
        return {
            'total_power_mw': total_power_mw,
            'static_power_mw': static_power_mw,
            'dynamic_power_mw': dynamic_power_mw,
            'avg_latency_us': avg_latency_us
        }

    def _save_experiment_results(self, experiment_name, results):
        """Save experiment results to file"""
        save_path = os.path.join(self.results_dir, f'{experiment_name}_results.json')

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                return super().default(obj)

        # Filter out non-serializable objects
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()
                        if not hasattr(v, '__module__') or isinstance(v, (dict, list, np.ndarray))}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)  # Fallback: convert to string

        clean_results = make_serializable(results)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=4, cls=NumpyEncoder)

        self.logger.info(f"\n[OK] Results saved to {save_path}")


    # Plotting methods will be added next...

    def _plot_local_epochs_analysis(self, results):
        """Plot local epochs experiment results"""
        from utils.plotting_advanced import plot_local_epochs_analysis
        plot_local_epochs_analysis(results, self.results_dir)

    def _plot_quantization_analysis(self, results):
        """Plot quantization experiment results"""
        from utils.plotting_advanced import plot_quantization_analysis
        plot_quantization_analysis(results, self.results_dir)

    def _plot_compression_analysis(self, results):
        """Plot compression experiment results"""
        from utils.plotting_advanced import plot_compression_analysis
        plot_compression_analysis(results, self.results_dir)

    def _plot_mobility_analysis(self, results):
        """Plot mobility experiment results"""
        from utils.plotting_advanced import plot_mobility_analysis
        plot_mobility_analysis(results, self.results_dir)

    def _plot_noniid_analysis(self, results):
        """Plot non-IID experiment results"""
        from utils.plotting_advanced import plot_noniid_analysis
        plot_noniid_analysis(results, self.results_dir)

    def _plot_pilot_analysis(self, results):
        """Plot pilot overhead experiment results"""
        from utils.plotting_advanced import plot_pilot_analysis
        plot_pilot_analysis(results, self.results_dir)

    def _plot_noc_analysis(self, results):
        """Plot NoC experiment results"""
        from utils.plotting_advanced import plot_noc_traffic_analysis
        plot_noc_traffic_analysis(results, self.results_dir)

    def _plot_approach_comparison(self, results):
        """Plot FL vs Centralized comparison"""
        from utils.plotting_advanced import plot_approach_comparison
        plot_approach_comparison(results, self.results_dir)

    # ==================== NEW EXPERIMENTS ====================

    def _plot_baseline_comparison(self, results):
        """Plot baseline comparison using publication-quality plotting module."""
        from utils.plotting_advanced import plot_baseline_comparison
        plot_baseline_comparison(results, self.results_dir)

    def _plot_multiuser_comparison(self, results):
        """Plot multi-user MIMO comparison using publication-quality plotting module."""
        from utils.plotting_advanced import plot_multiuser_comparison
        plot_multiuser_comparison(results, self.results_dir)

    def _plot_fl_algorithms(self, results):
        """Plot FL algorithms comparison (Exp 11)."""
        from utils.plotting_advanced import plot_fl_algorithms_comparison
        plot_fl_algorithms_comparison(results, self.results_dir)

    def _plot_architectures(self, results):
        """Plot model architectures comparison (Exp 12)."""
        from utils.plotting_advanced import plot_architecture_comparison
        plot_architecture_comparison(results, self.results_dir)

    def _plot_csi_robustness(self, results):
        """Plot CSI robustness analysis (Exp 13)."""
        from utils.plotting_advanced import plot_csi_robustness
        plot_csi_robustness(results, self.results_dir)

    def _plot_topology_comparison(self, results):
        """Plot topology comparison (Exp 14)."""
        from utils.plotting_advanced import plot_topology_comparison
        plot_topology_comparison(results, self.results_dir)

    def _plot_protocol_comparison(self, results):
        """Plot protocol comparison (Exp 15)."""
        from utils.plotting_advanced import plot_protocol_comparison
        plot_protocol_comparison(results, self.results_dir)

    def _plot_optimization_comparison(self, results):
        """Plot optimization techniques comparison (Exp 16)."""
        from utils.plotting_advanced import plot_optimization_comparison
        plot_optimization_comparison(results, self.results_dir)

    def _plot_golden_ratio(self, results):
        """Plot tile-pixel golden ratio analysis (Exp 17)."""
        from utils.plotting_advanced import plot_golden_ratio_analysis
        plot_golden_ratio_analysis(results, self.results_dir)

    def _plot_duty_cycling(self, results):
        """Plot duty cycling analysis (Exp 18)."""
        from utils.plotting_advanced import plot_duty_cycling_analysis
        plot_duty_cycling_analysis(results, self.results_dir)

    def _plot_dataset_comparison(self, results):
        """Plot dataset comparison (Exp 19)."""
        from utils.plotting_advanced import plot_dataset_comparison
        plot_dataset_comparison(results, self.results_dir)

    def _plot_phase_quantization(self, results):
        """Plot phase quantization analysis (Exp 20)."""
        from utils.plotting_advanced import plot_phase_quantization_detailed
        plot_phase_quantization_detailed(results, self.results_dir)
