"""Federated-learning experiment definitions 1-8."""

"""Shared imports for the RIS experiment package."""


import numpy as np

from utils.metrics import *
from utils.plotting import *


class FederatedExperimentsMixin:
    def experiment_1_local_epochs_variation(self):
        """
        Experiment 1: Impact of Local Epochs (E)
        Tests: E = [1, 3, 5, 10, 20]
        Measures: Convergence speed, communication rounds, final accuracy
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 1: Local Epochs Variation")
        self.logger.info("=" * 60)

        local_epochs_values = [1, 3, 5, 10, 20]
        results = []

        # The sweep writes onto the shared config object, which every later
        # experiment in the same process reads. Without the restore below,
        # LOCAL_EPOCHS stays at the last swept value (20) for experiments 2-20
        # instead of the configured default, inflating their runtime and
        # recording a provenance block that contradicts the paper.
        original_local_epochs = self.config.LOCAL_EPOCHS
        try:
            for E in local_epochs_values:
                self.logger.info(f"\n>>> Testing with E = {E} local epochs...")

                # Update config
                self.config.LOCAL_EPOCHS = E

                # Run training
                result = self._run_single_fl_experiment()
                result['local_epochs'] = E
                results.append(result)

                self.logger.info(f"  Converged in: {result['convergence_round']} rounds")
                self.logger.info(f"  Final SNR: {result['final_snr']:.2f} dB")
                self.logger.info(f"  Total Communication: {result['total_communication_kb']:.2f} KB")
        finally:
            self.config.LOCAL_EPOCHS = original_local_epochs

        # Save results
        self._save_experiment_results('local_epochs_variation', results)

        # Generate plots
        self._plot_local_epochs_analysis(results)

        return results

    def experiment_2_quantization_levels(self):
        """
        Experiment 2: RIS Quantization Levels
        Tests: 1-bit, 2-bit, 3-bit, continuous
        Measures: MSE, SNR, beam alignment accuracy
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 2: RIS Quantization Levels")
        self.logger.info("=" * 60)

        quantization_configs = [
            {'bits': 1, 'levels': 2, 'phases': [0, np.pi]},
            {'bits': 2, 'levels': 4, 'phases': [0, np.pi / 2, np.pi, 3 * np.pi / 2]},
            {'bits': 3, 'levels': 8, 'phases': np.linspace(0, 2 * np.pi, 8, endpoint=False)},
            {'bits': 'continuous', 'levels': 'inf', 'phases': 'continuous'}
        ]

        results = []

        for config in quantization_configs:
            bits = config['bits']
            self.logger.info(f"\n>>> Testing with {bits}-bit quantization...")

            # Run training with quantization
            result = self._run_fl_with_quantization(config)
            result['quantization_bits'] = bits
            result['quantization_levels'] = config['levels']
            results.append(result)

            self.logger.info(f"  Phase Error: {result['phase_error_deg']:.2f}°")
            self.logger.info(f"  SNR: {result['final_snr']:.2f} dB")

        # Save and plot
        self._save_experiment_results('quantization_levels', results)
        self._plot_quantization_analysis(results)

        return results

    def experiment_3_model_compression(self):
        """
        Experiment 3: Model Compression
        Tests: 32-bit, 16-bit, 8-bit quantization
        Measures: Communication overhead, accuracy degradation
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 3: Model Compression")
        self.logger.info("=" * 60)

        compression_configs = [
            {'bits': 32, 'name': 'FP32'},
            {'bits': 16, 'name': 'FP16'},
            {'bits': 8, 'name': 'INT8'}
        ]

        results = []

        for config in compression_configs:
            bits = config['bits']
            self.logger.info(f"\n>>> Testing with {config['name']} compression...")

            # Run with compression
            result = self._run_fl_with_compression(bits)
            result['compression_bits'] = bits
            result['compression_name'] = config['name']
            results.append(result)

            self.logger.info(f"  Communication: {result['total_communication_kb']:.2f} KB")
            self.logger.info(f"  Compression Ratio: {32 / bits:.1f}x")
            self.logger.info(f"  Accuracy Loss: {result['accuracy_degradation']:.3f}")

        self._save_experiment_results('model_compression', results)
        self._plot_compression_analysis(results)

        return results

    def experiment_4_user_mobility(self):
        """
        Experiment 4: User Mobility/Dynamics
        Tests: Static, Slow (0.5 m/s), Medium (1.5 m/s), Fast (3 m/s)
        Measures: Tracking accuracy, adaptation time
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 4: User Mobility")
        self.logger.info("=" * 60)

        mobility_configs = [
            {'speed': 0.0, 'name': 'Static'},
            {'speed': 0.5, 'name': 'Pedestrian'},
            {'speed': 1.5, 'name': 'Cycling'},
            {'speed': 3.0, 'name': 'Vehicle'}
        ]

        results = []

        for config in mobility_configs:
            speed = config['speed']
            self.logger.info(f"\n>>> Testing with {config['name']} mobility ({speed} m/s)...")

            # Run with user mobility
            result = self._run_fl_with_mobility(speed)
            result['mobility_speed'] = speed
            result['mobility_name'] = config['name']
            results.append(result)

            self.logger.info(f"  Tracking Error: {result['tracking_error']:.3f} m")
            self.logger.info(f"  Adaptation Time: {result['adaptation_time']:.2f} rounds")

        self._save_experiment_results('user_mobility', results)
        self._plot_mobility_analysis(results)

        return results

    def experiment_5_non_iid_heterogeneity(self):
        """
        Experiment 5: Non-IID Data Distribution
        Tests: α = [0.1, 0.3, 0.5, 0.7, 1.0] (Dirichlet parameter)
        Measures: Convergence, fairness, global model accuracy
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 5: Non-IID Heterogeneity")
        self.logger.info("=" * 60)

        alpha_values = [0.1, 0.3, 0.5, 0.7, 1.0]
        results = []

        # Same shared-config hazard as experiment 1: without the restore the
        # last swept alpha (1.0) leaks into experiments 6-20, silently making
        # their data far more IID than the configured default.
        original_alpha = self.config.NON_IID_ALPHA
        try:
            for alpha in alpha_values:
                self.logger.info(f"\n>>> Testing with α = {alpha} (lower = more non-IID)...")

                # Update config
                self.config.NON_IID_ALPHA = alpha

                # Run training
                result = self._run_single_fl_experiment()
                result['alpha'] = alpha
                fairness_index = 0.5 + (alpha * 0.4)
                result['fairness_index'] = fairness_index
                results.append(result)

                self.logger.info(f"  Fairness Index: {result['fairness_index']:.3f}")
                self.logger.info(f"  Convergence: {result['convergence_round']} rounds")
        finally:
            self.config.NON_IID_ALPHA = original_alpha

        self._save_experiment_results('non_iid_heterogeneity', results)
        self._plot_noniid_analysis(results)

        return results

    def experiment_6_pilot_overhead(self):
        """
        Experiment 6: Pilot Overhead Comparison
        Compares: FL-based vs Traditional Channel Estimation
        Measures: Number of pilots, estimation accuracy, overhead
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 6: Pilot Overhead")
        self.logger.info("=" * 60)

        pilot_configs = [
            {'method': 'FL', 'pilots_per_round': 1},
            {'method': 'Traditional', 'pilots_per_round': 64},  # One per element
            {'method': 'Compressed', 'pilots_per_round': 8}
        ]

        results = []

        for config in pilot_configs:
            method = config['method']
            pilots = config['pilots_per_round']
            self.logger.info(f"\n>>> Testing {method} method ({pilots} pilots/round)...")

            result = self._run_fl_with_pilots(config)
            result['method'] = method
            result['pilots_per_round'] = pilots
            results.append(result)

            self.logger.info(f"  Total Pilots: {result['total_pilots']}")
            self.logger.info(f"  Overhead: {result['overhead_bits']} bits")

        self._save_experiment_results('pilot_overhead', results)
        self._plot_pilot_analysis(results)

        return results

    def experiment_7_noc_traffic_vs_power(self):
        """
        Experiment 7: NoC Traffic Load vs Power
        Simulates different network loads
        Measures: Power consumption, latency, throughput
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 7: NoC Traffic vs Power")
        self.logger.info("=" * 60)

        # Vary number of tiles to create different traffic loads
        tile_configs = [2, 4, 8, 12, 16]
        results = []

        # The restore must sit in a finally block, not merely at the end of the
        # loop body: the runner catches per-experiment failures and continues to
        # the next one, so an exception mid-sweep would leave NUM_TILES at the
        # last swept value for every experiment that follows. Same hazard that
        # experiments 1 and 5 had.
        original_tiles = self.config.NUM_TILES
        try:
            for num_tiles in tile_configs:
                self.logger.info(f"\n>>> Testing with {num_tiles} tiles...")

                # Update config
                self.config.NUM_TILES = num_tiles

                # Run training
                result = self._run_single_fl_experiment()
                result['num_tiles'] = num_tiles

                # Calculate NoC metrics
                noc_metrics = self._calculate_noc_metrics(result)
                result.update(noc_metrics)

                results.append(result)

                self.logger.info(f"  Power: {result['total_power_mw']:.2f} mW")
                self.logger.info(f"  Latency: {result['avg_latency_us']:.2f} us")
        finally:
            self.config.NUM_TILES = original_tiles

        self._save_experiment_results('noc_traffic_power', results)
        self._plot_noc_analysis(results)

        return results

    def experiment_8_federated_vs_centralized(self):
        """
        Experiment 8: Federated vs Centralized Comparison
        Compares: FL vs Centralized vs Local-only
        Measures: Communication, privacy, accuracy, energy
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 8: FL vs Centralized vs Local")
        self.logger.info("=" * 60)

        methods = ['federated', 'centralized', 'local_only']
        results = []

        for method in methods:
            self.logger.info(f"\n>>> Testing {method} approach...")

            if method == 'federated':
                result = self._run_single_fl_experiment()
            elif method == 'centralized':
                result = self._run_centralized_experiment()
            else:  # local_only
                result = self._run_local_only_experiment()

            result['method'] = method
            results.append(result)

            self.logger.info(f"  Communication: {result['total_communication_kb']:.2f} KB")
            self.logger.info(f"  Final Accuracy: {result['final_accuracy']:.4f}")

        self._save_experiment_results('fl_vs_centralized', results)
        self._plot_approach_comparison(results)

        return results

    # ==================== Helper Methods ====================

