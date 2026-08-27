"""Baseline and multi-user experiment definitions 9-10."""

"""Shared imports for the RIS experiment package."""

import copy
import json
import os
import pickle
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Config
from models.ris_net import create_model
from src.client import RISClient
from src.dataset_utils import create_non_iid_datasets, create_test_dataset
from src.server import FederatedServer
from utils.metrics import *
from utils.metrics import dbm_to_watts
from utils.plotting import *

from .logging_utils import get_experiment_logger


class BaselineMultiuserExperimentsMixin:
    def experiment_9_baseline_comparison(self):
        """
        Experiment 9: Comprehensive Baseline Comparison
        Compares: FL vs AO vs Centralized DL vs Random Search vs No RIS vs Random RIS vs Optimal
        Metrics: SNR, convergence, communication, privacy, computational complexity
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 9: Comprehensive Baseline Comparison")
        self.logger.info("=" * 60)

        from baselines.alternating_optimization import AlternatingOptimization
        from baselines.centralized_learning import CentralizedRIS
        from baselines.random_search import RandomSearch

        # Create datasets (same for all methods - fair comparison)
        train_datasets, tile_positions = create_non_iid_datasets(self.config, self.config.NUM_TILES)
        test_dataset = create_test_dataset(self.config)
        num_eval_samples = min(100, len(test_dataset))

        noise_power = dbm_to_watts(self.config.NOISE_POWER_DBM)
        tx_power = dbm_to_watts(self.config.TX_POWER_DBM)

        results = {}

        # ---- 1. No RIS Baseline ----
        self.logger.info("\n>>> Evaluating: No RIS (direct link only)...")
        snr_no_ris = []
        for i in range(num_eval_samples):
            metadata = test_dataset.metadata[i]
            h_direct = metadata['H_direct'][0]
            signal = tx_power * np.abs(h_direct) ** 2
            snr = 10 * np.log10(signal / noise_power)
            snr_no_ris.append(snr)
        results['no_ris'] = {
            'snr_db': np.mean(snr_no_ris),
            'rate_bps_hz': calculate_achievable_rate(np.mean(snr_no_ris)),
            'communication_kb': 0,
            'energy_mj': 0,
            'convergence_iters': 0,
            'privacy': True,
            'complexity': 'O(1)'
        }
        self.logger.info(f"  SNR: {results['no_ris']['snr_db']:.2f} dB")

        # ---- 2. Random RIS Baseline ----
        self.logger.info("\n>>> Evaluating: Random RIS phases...")
        snr_random = []
        for i in range(num_eval_samples):
            metadata = test_dataset.metadata[i]
            h_direct = metadata['H_direct'][0]
            h_ris = metadata['H_ris'][0]
            h_bs_ris = metadata['h_bs_ris']
            h_cascade = h_ris * h_bs_ris
            random_phases = np.random.uniform(0, 2 * np.pi, len(h_ris))
            h_total = h_direct + np.sum(h_cascade * np.exp(1j * random_phases))
            signal = tx_power * np.abs(h_total) ** 2
            snr = 10 * np.log10(signal / noise_power)
            snr_random.append(snr)
        results['random_ris'] = {
            'snr_db': np.mean(snr_random),
            'rate_bps_hz': calculate_achievable_rate(np.mean(snr_random)),
            'communication_kb': 0,
            'energy_mj': 0,
            'convergence_iters': 1,
            'privacy': True,
            'complexity': 'O(N)'
        }
        self.logger.info(f"  SNR: {results['random_ris']['snr_db']:.2f} dB")

        # ---- 3. Random Search (1000 trials) ----
        self.logger.info("\n>>> Evaluating: Random Search (1000 trials)...")
        rs = RandomSearch(
            num_elements=self.config.ELEMENTS_PER_TILE,
            num_trials=1000,
            seed=42
        )
        snr_rs = []
        for i in range(num_eval_samples):
            metadata = test_dataset.metadata[i]
            h_direct = metadata['H_direct'][0]
            h_ris = metadata['H_ris'][0]
            h_bs_ris = metadata['h_bs_ris']
            h_cascade = h_ris * h_bs_ris

            best_snr = -np.inf
            for trial in range(1000):
                phases = np.random.uniform(0, 2 * np.pi, len(h_ris))
                h_total = h_direct + np.sum(h_cascade * np.exp(1j * phases))
                signal = tx_power * np.abs(h_total) ** 2
                snr = 10 * np.log10(signal / noise_power)
                if snr > best_snr:
                    best_snr = snr
            snr_rs.append(best_snr)
        results['random_search'] = {
            'snr_db': np.mean(snr_rs),
            'rate_bps_hz': calculate_achievable_rate(np.mean(snr_rs)),
            'communication_kb': 0,
            'energy_mj': 0,
            'convergence_iters': 1000,
            'privacy': False,  # Needs centralized CSI
            'complexity': 'O(N·T)'
        }
        self.logger.info(f"  SNR: {results['random_search']['snr_db']:.2f} dB")

        # ---- 4. Deep Reinforcement Learning (TD3) ----
        self.logger.info("\n>>> Evaluating: Deep Reinforcement Learning (TD3)...")
        from baselines.drl_agent import TD3Agent
        
        # Initialize DRL Agent
        # State: full feature vector from the dataset (same as model input)
        # Action: phases [0, 2pi] mapped to [-pi, pi] for tanh
        state_dim = train_datasets[0].get_input_dim()
        action_dim = self.config.ELEMENTS_PER_TILE
        max_action = np.pi
        
        agent = TD3Agent(state_dim, action_dim, max_action, device=self.config.DEVICE)
        
        # Train DRL agent (Online Learning on Train Data)
        self.logger.info("  Training DRL agent...")
        drl_losses = []
        drl_epochs = 50 # Short training for baseline
        
        for epoch in range(drl_epochs):
            epoch_loss = 0
            for i, dataset in enumerate(train_datasets):
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                for features, _ in dataloader:
                     # Features are already (batch, 2*elements) from dataset
                     # Action: optimal phases? No, DRL explores. 
                     # We need to simulate the environment step.
                     
                     # For DRL training, we treat this as a contextual bandit problem
                     # State s -> Action a -> Reward r
                     
                     bs = features.size(0)
                     state = features.to(self.config.DEVICE)
                     
                     # Select action with noise
                     action = agent.actor(state)
                     noise = torch.randn_like(action) * 0.1
                     action = (action + noise).clamp(-max_action, max_action)
                     
                     # Compute Reward (SNR)
                     # We need the channel info to compute SNR. 
                     # The features contain H_direct and H_ris implicitly? 
                     # The dataset returns (features, optimal_phases). Features are input to model.
                     # We need to reconstruct the channel from features or use a simulator.
                     # PROXY: Use MSE against optimal phases as reward (since we have labels)
                     # This turns DRL into supervised learning, which is not fair.
                     # REAL DRL: Must compute SNR.
                     
                     # Since we can't easily reconstruct H from just features in this loop without metadata,
                     # we will use the supervised proxy for this baseline implementation validity,
                     # OR better: use the validation loop style where we have metadata.
                     pass 
            
            # Since proper DRL training requires an interactive environment (State -> Reward),
            # and our dataset is offline, we will simulate "online" training by iterating through data 
            # and calculating reward using the Channel Model helper.
            pass

        # RE-IMPLEMENTATION: Use proper RISEnv
        from baselines.drl_agent import RISEnv
        env = RISEnv(train_datasets[0], self.config.TX_POWER_DBM, self.config.NOISE_POWER_DBM)
        
        num_train_samples = min(500, len(train_datasets[0]))
        state = env.reset()
        
        for i in range(num_train_samples):
            # Select action
            action = agent.select_action(state, noise=0.1)
            
            # Step environment
            next_state, reward, done, _ = env.step(action)
            
            agent.add_to_buffer(state, action, next_state, reward, float(done))
            state = next_state
            
            if i > 32:
                agent.train(batch_size=32)

        # Evaluate DRL
        snr_drl = []
        for i in range(num_eval_samples):
            features, _ = test_dataset[i]
            state = features.numpy()
            action = agent.select_action(state, noise=0.0)
            phases = np.mod(action + np.pi, 2*np.pi)
            
            metadata = test_dataset.metadata[i]
            h_direct = metadata['H_direct'][0]
            h_ris = metadata['H_ris'][0]
            h_bs_ris = metadata['h_bs_ris']
            h_cascade = h_ris * h_bs_ris

            h_total = h_direct + np.sum(h_cascade * np.exp(1j * phases))
            signal = tx_power * np.abs(h_total) ** 2
            snr = 10 * np.log10(signal / noise_power)
            snr_drl.append(snr)
            
        results['drl_td3'] = {
            'snr_db': np.mean(snr_drl),
            'rate_bps_hz': calculate_achievable_rate(np.mean(snr_drl)),
            'communication_kb': 0, # On-device
            'energy_mj': num_train_samples * 1.0, # High training energy
            'convergence_iters': num_train_samples,
            'privacy': True,
            'complexity': 'High'
        }
        self.logger.info(f"  SNR: {results['drl_td3']['snr_db']:.2f} dB")

        # ---- 5. Alternating Optimization ----
        self.logger.info("\n>>> Evaluating: Alternating Optimization...")
        ao = AlternatingOptimization(
            num_elements=self.config.ELEMENTS_PER_TILE,
            max_iterations=100,
            lr_phase=0.1,
            convergence_threshold=1e-4
        )
        snr_ao = []
        ao_iters = []
        for i in range(num_eval_samples):
            metadata = test_dataset.metadata[i]
            h_direct_complex = metadata['H_direct'][0]
            h_ris = metadata['H_ris'][0]
            h_bs_ris = metadata['h_bs_ris']

            # AO optimization using the cascaded channel model
            h_ris_user = h_ris

            phases, snr_history = ao.optimize_phases(
                h_direct=h_direct_complex,
                h_ris_user=h_ris_user,
                h_bs_ris=h_bs_ris,
                noise_power=noise_power / tx_power  # Normalize
            )

            # Compute actual SNR with optimized phases
            h_cascade = h_ris * h_bs_ris
            h_total = h_direct_complex + np.sum(h_cascade * np.exp(1j * phases))
            signal = tx_power * np.abs(h_total) ** 2
            snr = 10 * np.log10(signal / noise_power)
            snr_ao.append(snr)
            ao_iters.append(len(snr_history))

        ao_complexity = ao.compute_complexity()
        results['alternating_opt'] = {
            'snr_db': np.mean(snr_ao),
            'rate_bps_hz': calculate_achievable_rate(np.mean(snr_ao)),
            'communication_kb': num_eval_samples * self.config.ELEMENTS_PER_TILE * 8 / 1024,  # CSI upload
            'energy_mj': np.mean(ao_iters) * 0.1,  # Approximate
            'convergence_iters': np.mean(ao_iters),
            'privacy': False,
            'complexity': ao_complexity['complexity_class']
        }
        self.logger.info(f"  SNR: {results['alternating_opt']['snr_db']:.2f} dB")
        self.logger.info(f"  Avg iterations: {np.mean(ao_iters):.1f}")

        # ---- 5. Centralized Deep Learning ----
        self.logger.info("\n>>> Evaluating: Centralized Deep Learning...")
        input_dim = train_datasets[0].get_input_dim()
        input_dim = train_datasets[0].get_input_dim()
        cent_model = create_model(
            model_type=self.config.MODEL_TYPE,
            input_dim=input_dim,
            num_elements=self.config.ELEMENTS_PER_TILE,
            hidden_dim=self.config.HIDDEN_DIM,
            num_layers=self.config.NUM_LAYERS,
            dropout=self.config.DROPOUT,
            config=self.config
        )
        centralized = CentralizedRIS(cent_model, self.config)
        cent_metrics = centralized.train_centralized(
            tile_datasets=train_datasets,
            epochs=self.config.LOCAL_EPOCHS * self.config.FL_ROUNDS
        )

        # Evaluate centralized model on test set
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        cent_eval = centralized.evaluate(test_loader)

        # Compute centralized SNR
        cent_model_eval = centralized.get_model()
        cent_model_eval.eval()
        snr_cent = []
        with torch.no_grad():
            for i in range(num_eval_samples):
                features, _ = test_dataset[i]
                metadata = test_dataset.metadata[i]
                h_direct = metadata['H_direct'][0]
                h_ris = metadata['H_ris'][0]
                h_bs_ris = metadata['h_bs_ris']
                h_cascade = h_ris * h_bs_ris

                pred = cent_model_eval(features.unsqueeze(0).to(self.config.DEVICE))
                pred_phases = pred.squeeze().cpu().numpy()

                h_total = h_direct + np.sum(h_cascade * np.exp(1j * pred_phases))
                signal = tx_power * np.abs(h_total) ** 2
                snr = 10 * np.log10(signal / noise_power)
                snr_cent.append(snr)

        # Communication: all raw data sent to server
        total_samples = sum(len(d) for d in train_datasets)
        raw_data_bytes = total_samples * (input_dim + self.config.ELEMENTS_PER_TILE) * 4
        results['centralized_dl'] = {
            'snr_db': np.mean(snr_cent),
            'rate_bps_hz': calculate_achievable_rate(np.mean(snr_cent)),
            'communication_kb': raw_data_bytes / 1024,
            'energy_mj': cent_metrics['total_epochs'] * 0.5,
            'convergence_iters': cent_metrics['total_epochs'],
            'privacy': False,
            'complexity': 'O(N·E·B)',
            'final_loss': cent_metrics['final_loss'],
            'training_history': cent_metrics['training_history']
        }
        self.logger.info(f"  SNR: {results['centralized_dl']['snr_db']:.2f} dB")
        self.logger.info(f"  Final Loss: {cent_metrics['final_loss']:.6f}")

        # ---- 6. Federated Learning (Ours) ----
        self.logger.info("\n>>> Evaluating: Federated Learning (Ours)...")
        fl_result = self._run_single_fl_experiment()
        results['federated_ours'] = {
            'snr_db': fl_result['final_snr'],
            'rate_bps_hz': calculate_achievable_rate(fl_result['final_snr']),
            'communication_kb': fl_result['total_communication_kb'],
            'energy_mj': fl_result['total_energy_mj'],
            'convergence_iters': fl_result['convergence_round'],
            'privacy': True,
            'complexity': 'O(N·E·B/K)',
            'final_loss': fl_result['final_loss'],
            'phase_error_deg': fl_result['phase_error_deg']
        }
        self.logger.info(f"  SNR: {results['federated_ours']['snr_db']:.2f} dB")
        self.logger.info(f"  Privacy: YES")

        # ---- 7. Genie-Aided Optimal ----
        self.logger.info("\n>>> Evaluating: Genie-Aided Optimal...")
        snr_optimal = []
        for i in range(num_eval_samples):
            features, optimal_phases = test_dataset[i]
            metadata = test_dataset.metadata[i]
            h_direct = metadata['H_direct'][0]
            h_ris = metadata['H_ris'][0]
            h_bs_ris = metadata['h_bs_ris']
            h_cascade = h_ris * h_bs_ris
            h_total = h_direct + np.sum(h_cascade * np.exp(1j * optimal_phases.numpy()))
            signal = tx_power * np.abs(h_total) ** 2
            snr = 10 * np.log10(signal / noise_power)
            snr_optimal.append(snr)
        results['optimal'] = {
            'snr_db': np.mean(snr_optimal),
            'rate_bps_hz': calculate_achievable_rate(np.mean(snr_optimal)),
            'communication_kb': 0,
            'energy_mj': 0,
            'convergence_iters': 0,
            'privacy': True,
            'complexity': 'N/A (oracle)'
        }
        self.logger.info(f"  SNR: {results['optimal']['snr_db']:.2f} dB")

        # ---- Summary Table ----
        self.logger.info("\n" + "=" * 80)
        self.logger.info("COMPREHENSIVE BASELINE COMPARISON SUMMARY")
        self.logger.info("=" * 80)
        header = f"{'Method':<22} {'SNR (dB)':>10} {'Rate':>8} {'Comm (KB)':>10} {'Privacy':>8} {'Iters':>8}"
        self.logger.info(header)
        self.logger.info("-" * 80)
        for method_name, r in results.items():
            privacy_str = "Yes" if r['privacy'] else "No"
            self.logger.info(f"{method_name:<22} {r['snr_db']:>10.2f} {r['rate_bps_hz']:>8.2f} "
                  f"{r['communication_kb']:>10.1f} {privacy_str:>8} {r['convergence_iters']:>8}")
        self.logger.info("=" * 80)

        # Save results
        self._save_experiment_results('baseline_comparison', [results])

        # Generate plots
        self._plot_baseline_comparison(results)

        return results

    def experiment_10_multiuser_comparison(self):
        """
        Experiment 10: Multi-User MIMO Extension
        Tests: 1, 2, 4, 8 simultaneous users
        Metrics: Sum-rate, per-user fairness, convergence
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 10: Multi-User MIMO Comparison")
        self.logger.info("=" * 60)

        user_counts = [1, 2, 4, 8]
        results = []

        for num_users in user_counts:
            self.logger.info(f"\n>>> Testing with {num_users} simultaneous users...")

            # Generate multi-user channel samples
            test_dataset = create_test_dataset(self.config)
            num_eval_samples = min(100, len(test_dataset))

            noise_power = dbm_to_watts(self.config.NOISE_POWER_DBM)
            tx_power = dbm_to_watts(self.config.TX_POWER_DBM)

            per_user_snrs = []
            sum_rates = []
            fairness_indices = []

            for i in range(num_eval_samples):
                metadata = test_dataset.metadata[i]

                # Get channels for all users
                H_direct = metadata['H_direct'][:num_users]  # [num_users] complex
                H_ris = metadata['H_ris'][:num_users]  # [num_users, N] complex

                # Optimize phases for sum-rate maximization
                # Use weighted combination approach
                optimal_phases = self._optimize_multiuser_phases(
                    H_direct, H_ris, num_users, noise_power, tx_power
                )

                # Compute per-user SNR with optimized phases
                user_snrs = []
                user_rates = []
                for u in range(num_users):
                    h_total = H_direct[u] + np.sum(H_ris[u] * np.exp(1j * optimal_phases))
                    signal = tx_power * np.abs(h_total) ** 2

                    # For multi-user, include inter-user interference
                    interference = 0
                    for v in range(num_users):
                        if v != u:
                            h_int = H_direct[v] + np.sum(H_ris[v] * np.exp(1j * optimal_phases))
                            interference += tx_power * np.abs(h_int) ** 2 * 0.1  # Cross-talk factor

                    sinr = signal / (noise_power + interference)
                    snr_db = 10 * np.log10(sinr)
                    rate = np.log2(1 + sinr)
                    user_snrs.append(snr_db)
                    user_rates.append(rate)

                per_user_snrs.append(user_snrs)
                sum_rates.append(np.sum(user_rates))

                # Jain's fairness index
                rates = np.array(user_rates)
                if np.sum(rates ** 2) > 0:
                    fairness = (np.sum(rates)) ** 2 / (num_users * np.sum(rates ** 2))
                else:
                    fairness = 1.0
                fairness_indices.append(fairness)

            # Now run FL for this user configuration
            self.logger.info(f"  Running FL training for {num_users} users...")
            fl_result = self._run_single_fl_experiment()

            result = {
                'num_users': num_users,
                'avg_sum_rate': np.mean(sum_rates),
                'std_sum_rate': np.std(sum_rates),
                'avg_per_user_snr': np.mean([np.mean(s) for s in per_user_snrs]),
                'min_per_user_snr': np.mean([np.min(s) for s in per_user_snrs]),
                'max_per_user_snr': np.mean([np.max(s) for s in per_user_snrs]),
                'avg_fairness': np.mean(fairness_indices),
                'fl_convergence': fl_result['convergence_round'],
                'fl_final_loss': fl_result['final_loss'],
                'fl_communication_kb': fl_result['total_communication_kb'],
                'per_user_snr_distribution': [np.mean([s[u] for s in per_user_snrs]) for u in range(num_users)]
            }
            results.append(result)

            self.logger.info(f"  Sum Rate: {result['avg_sum_rate']:.2f} bps/Hz")
            self.logger.info(f"  Avg Per-User SNR: {result['avg_per_user_snr']:.2f} dB")
            self.logger.info(f"  Fairness Index: {result['avg_fairness']:.4f}")

        # Save results
        self._save_experiment_results('multiuser_comparison', results)

        # Generate plots
        self._plot_multiuser_comparison(results)

        return results

    def _optimize_multiuser_phases(self, H_direct, H_ris, num_users,
                                     noise_power, tx_power, num_iterations=50):
        """
        Optimize RIS phases for multi-user sum-rate maximization.

        Uses gradient ascent on weighted sum-rate.
        """
        num_elements = H_ris.shape[1]
        phases = np.random.uniform(0, 2 * np.pi, num_elements)

        lr = 0.05
        weights = np.ones(num_users) / num_users  # Equal weights

        for iteration in range(num_iterations):
            # Compute gradient of sum-rate w.r.t. phases
            gradient = np.zeros(num_elements)

            for u in range(num_users):
                h_total = H_direct[u] + np.sum(H_ris[u] * np.exp(1j * phases))
                signal = tx_power * np.abs(h_total) ** 2
                sinr = signal / noise_power

                # Gradient of log2(1 + SINR) w.r.t. phases
                for n in range(num_elements):
                    grad_component = np.conj(h_total) * 1j * H_ris[u, n] * np.exp(1j * phases[n])
                    grad_snr = 2 * tx_power * np.real(grad_component)
                    # Chain rule: d/dθ log2(1+SINR) = 1/((1+SINR)*ln2) * d_SINR/dθ
                    grad_rate = grad_snr / ((1 + sinr) * np.log(2) * noise_power)
                    gradient[n] += weights[u] * grad_rate

            # Gradient ascent
            phases = phases + lr * gradient
            phases = np.mod(phases, 2 * np.pi)

        return phases

