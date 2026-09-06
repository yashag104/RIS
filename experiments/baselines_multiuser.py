"""Baseline and multi-user experiment definitions 9-10."""

"""Shared imports for the RIS experiment package."""


import numpy as np
import torch
from torch.utils.data import DataLoader

from models.ris_net import create_model
from src.dataset_utils import (
    create_non_iid_datasets,
    create_test_dataset,
    validate_dataset_collection,
    validate_dataset_feature_dim,
)
from utils.metrics import *
from utils.metrics import dbm_to_watts
from utils.plotting import *


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
        train_datasets, _tile_positions = create_non_iid_datasets(self.config, self.config.NUM_TILES)
        test_dataset = create_test_dataset(self.config)
        input_dim = validate_dataset_collection(
            train_datasets,
            test_dataset,
            self.config,
            expected_num_tiles=self.config.NUM_TILES,
        )
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
        RandomSearch(
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
                best_snr = max(best_snr, snr)
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
        state_dim = input_dim
        action_dim = self.config.ELEMENTS_PER_TILE
        max_action = np.pi
        
        agent = TD3Agent(state_dim, action_dim, max_action, device=self.config.DEVICE)
        
        # Train DRL agent (Online Learning on Train Data)
        self.logger.info("  Training DRL agent...")
        drl_epochs = 50 # Short training for baseline
        
        for epoch in range(drl_epochs):
            for i, dataset in enumerate(train_datasets):
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                for features, _ in dataloader:
                     # Features are already (batch, 2*elements) from dataset
                     # Action: optimal phases? No, DRL explores. 
                     # We need to simulate the environment step.
                     
                     # For DRL training, we treat this as a contextual bandit problem
                     # State s -> Action a -> Reward r
                     
                     features.size(0)
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
            
            # Since proper DRL training requires an interactive environment (State -> Reward),
            # and our dataset is offline, we will simulate "online" training by iterating through data 
            # and calculating reward using the Channel Model helper.

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
        centralized.evaluate(test_loader)

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
        self.logger.info("  Privacy: YES")

        # ---- 7. Genie-Aided Optimal ----
        self.logger.info("\n>>> Evaluating: Genie-Aided Optimal...")
        snr_optimal = []
        for i in range(num_eval_samples):
            metadata = test_dataset.metadata[i]
            h_direct = metadata['H_direct'][0]
            h_ris = metadata['H_ris'][0]
            h_bs_ris = metadata['h_bs_ris']
            h_cascade = h_ris * h_bs_ris

            # Recompute the MRC optimum from the raw channels rather than reading
            # the dataset label. The label deliberately omits the global
            # angle(h_direct) offset (it lives in metadata['phase_offset']), so
            # applying it directly produces a MISALIGNED configuration that is not
            # optimal at all -- which let both federated_ours and random_search
            # score above this supposed upper bound. Recomputing here matches
            # client.compute_snr_improvement and main.py, and stays correct
            # regardless of how the label is parameterised.
            true_optimal_phases = np.mod(
                np.angle(h_direct) - np.angle(h_cascade), 2 * np.pi
            )
            h_total = h_direct + np.sum(h_cascade * np.exp(1j * true_optimal_phases))
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
            original_num_users = self.config.NUM_USERS
            self.config.NUM_USERS = num_users
            try:
                test_dataset = create_test_dataset(self.config)
                validate_dataset_feature_dim(
                    test_dataset,
                    self.config,
                    f"multiuser test_dataset[{num_users}]",
                )
                num_eval_samples = min(100, len(test_dataset))

                self.logger.info(f"  Running FL training for {num_users} users...")
                fl_result = self._run_single_fl_experiment()
            finally:
                self.config.NUM_USERS = original_num_users

            noise_power = dbm_to_watts(self.config.NOISE_POWER_DBM)
            tx_power = dbm_to_watts(self.config.TX_POWER_DBM)

            per_user_snrs = []
            sum_rates = []
            fairness_indices = []

            # Federated-model results, evaluated under the same SINR model
            fl_per_user_snrs = []
            fl_sum_rates = []
            fl_fairness = []
            fl_phase_fn = self._make_fl_phase_provider(fl_result, test_dataset)
            if fl_phase_fn is None:
                self.logger.warning(
                    "  FL global weights unavailable; reporting classical optimizer only."
                )

            for i in range(num_eval_samples):
                metadata = test_dataset.metadata[i]

                # Get channels for all users. metadata['H_ris'] is the
                # RIS->user hop only; the channel the RIS actually controls is
                # the BS->RIS->user cascade, so the BS->RIS hop has to be
                # included. Omitting it drops one path loss and inflates the
                # reflected path by tens of dB.
                H_direct = metadata['H_direct'][:num_users]  # [num_users] complex
                h_bs_ris = metadata['h_bs_ris']              # [N] complex
                H_casc = metadata['H_ris'][:num_users] * h_bs_ris[None, :]

                # One shared RIS configuration must serve every user at once.
                # This is the classical optimizer: a reference bound, NOT the
                # federated model. Reported separately so the two are never confused.
                optimal_phases = self._optimize_multiuser_phases(
                    H_direct, H_casc, num_users, noise_power, tx_power
                )
                user_snrs, user_rates = self._multiuser_rates(
                    H_direct, H_casc, optimal_phases, num_users, noise_power, tx_power
                )

                per_user_snrs.append(user_snrs)
                sum_rates.append(np.sum(user_rates))
                fairness_indices.append(self._jain_fairness(user_rates))

                # The federated model's own phases, evaluated under the identical
                # SINR model. Previously the multi-user results came entirely from
                # the classical optimizer, so they said nothing about what FL learned.
                if fl_phase_fn is not None:
                    fl_phases = fl_phase_fn(i)
                    fl_snrs, fl_rates = self._multiuser_rates(
                        H_direct, H_casc, fl_phases, num_users, noise_power, tx_power
                    )
                    fl_per_user_snrs.append(fl_snrs)
                    fl_sum_rates.append(np.sum(fl_rates))
                    fl_fairness.append(self._jain_fairness(fl_rates))

            result = {
                'num_users': num_users,
                # NOTE: the 'avg_*' fields below come from the classical
                # gradient-ascent optimizer and act as a reference bound.
                # The federated model's own numbers are the 'fl_*' fields.
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

            if fl_sum_rates:
                result.update({
                    'fl_avg_sum_rate': np.mean(fl_sum_rates),
                    'fl_std_sum_rate': np.std(fl_sum_rates),
                    'fl_avg_per_user_snr': np.mean([np.mean(s) for s in fl_per_user_snrs]),
                    'fl_min_per_user_snr': np.mean([np.min(s) for s in fl_per_user_snrs]),
                    'fl_avg_fairness': np.mean(fl_fairness),
                    'fl_sum_rate_gap': np.mean(sum_rates) - np.mean(fl_sum_rates),
                })

            results.append(result)

            self.logger.info(f"  [classical] Sum Rate: {result['avg_sum_rate']:.2f} bps/Hz")
            self.logger.info(f"  [classical] Avg Per-User SNR: {result['avg_per_user_snr']:.2f} dB")
            self.logger.info(f"  [classical] Fairness Index: {result['avg_fairness']:.4f}")
            if fl_sum_rates:
                self.logger.info(f"  [FL model]  Sum Rate: {result['fl_avg_sum_rate']:.2f} bps/Hz")
                self.logger.info(f"  [FL model]  Avg Per-User SNR: {result['fl_avg_per_user_snr']:.2f} dB")
                self.logger.info(f"  [FL model]  Fairness Index: {result['fl_avg_fairness']:.4f}")

        # Save results
        self._save_experiment_results('multiuser_comparison', results)

        # Generate plots
        self._plot_multiuser_comparison(results)

        return results

    @staticmethod
    def _multiuser_rates(H_direct, H_casc, phases, num_users, noise_power, tx_power):
        """Per-user SNR (dB) and achieved rate under orthogonal time sharing.

        A single-antenna BS with one shared RIS serves users on orthogonal
        resources, so there is no co-channel interference term. An earlier
        version added a fabricated 0.1 cross-talk factor, which by itself
        produced a ~40 dB cliff between one and two users. The genuine
        multi-user costs are that one phase configuration cannot align to every
        user at once, and that each user receives only 1/U of the airtime.

        Shared by the classical optimizer and the federated model so the two are
        always scored by the identical rule.

        Args:
            H_direct: Direct BS->user channel, ``[num_users]`` complex.
            H_casc: Cascade ``h_ris_user * h_bs_ris``, ``[num_users, N]`` complex.
            phases: RIS phase shifts to apply, ``[N]``.

        Returns:
            ``(snr_db_list, rate_list)``, both length ``num_users``.
        """
        reflect = np.exp(1j * phases)
        snr_db, rates = [], []
        for u in range(num_users):
            h_total = H_direct[u] + np.sum(H_casc[u] * reflect)
            snr = tx_power * np.abs(h_total) ** 2 / noise_power
            snr_db.append(10 * np.log10(max(snr, 1e-20)))
            rates.append(np.log2(1 + snr) / num_users)
        return snr_db, rates

    @staticmethod
    def _jain_fairness(rates) -> float:
        """Jain's fairness index over per-user rates."""
        rates = np.asarray(rates, dtype=float)
        denom = len(rates) * np.sum(rates ** 2)
        return float(np.sum(rates) ** 2 / denom) if denom > 0 else 1.0

    def _make_fl_phase_provider(self, fl_result, test_dataset):
        """Return ``fn(sample_index) -> phases`` from the trained global model.

        Returns None if the FL run exposed no weights, in which case the
        multi-user experiment reports only the classical optimizer.
        """
        weights = fl_result.get('global_weights') if fl_result else None
        if weights is None:
            return None

        import torch

        from models.ris_net import create_model

        model = create_model(
            getattr(self.config, 'MODEL_TYPE', 'MLP'),
            input_dim=test_dataset.get_input_dim(),
            num_elements=self.config.ELEMENTS_PER_TILE,
            config=self.config,
        )
        model.load_state_dict(weights)
        model.eval()
        device = next(model.parameters()).device

        def provider(index):
            features, _ = test_dataset[index]
            with torch.no_grad():
                predicted = model(features.unsqueeze(0).to(device)).squeeze().cpu().numpy()
            # The label omits the global MRC offset; add it back from CSI, exactly
            # as the single-user evaluation in client.py does.
            offset = test_dataset.metadata[index].get('phase_offset', 0.0)
            return np.mod(predicted + offset, 2 * np.pi)

        return provider

    def _optimize_multiuser_phases(self, H_direct, H_casc, num_users,
                                     noise_power, tx_power, num_iterations=100):
        """
        Optimize one shared RIS phase configuration for weighted sum-rate.

        Args:
            H_direct: Direct BS->user channels, shape [num_users]
            H_casc: Cascaded BS->RIS->user channels, shape [num_users, N]

        Uses normalised gradient ascent with backtracking, initialised at the
        single-user optimum so the search starts on a useful part of the
        sum-rate surface.
        """
        num_elements = H_casc.shape[1]
        weights = np.ones(num_users) / num_users  # Equal weights

        # Initialise by aligning to user 0, the single-user optimum. Starting
        # from random phases leaves the search in a flat region of the sum-rate
        # surface and the run finishes no better than random.
        phases = np.mod(np.angle(H_direct[0]) - np.angle(H_casc[0]), 2 * np.pi)

        def sum_rate(theta):
            total = 0.0
            for u in range(num_users):
                h = H_direct[u] + np.sum(H_casc[u] * np.exp(1j * theta))
                total += weights[u] * np.log2(
                    1 + tx_power * np.abs(h) ** 2 / noise_power)
            return total

        best_phases, best_rate = phases.copy(), sum_rate(phases)
        step = 0.5  # radians; scale-free because the gradient is normalised

        for _ in range(num_iterations):
            gradient = np.zeros(num_elements)
            for u in range(num_users):
                h_total = H_direct[u] + np.sum(H_casc[u] * np.exp(1j * phases))
                snr = tx_power * np.abs(h_total) ** 2 / noise_power
                # d|h|^2/dtheta_n = 2 Re{conj(h) * j * a_n * exp(j theta_n)}
                d_abs2 = 2 * np.real(
                    np.conj(h_total) * 1j * H_casc[u] * np.exp(1j * phases))
                grad_rate = (tx_power * d_abs2 / noise_power) / ((1 + snr) * np.log(2))
                gradient += weights[u] * grad_rate

            # Normalise: the raw gradient scales with the channel power, which
            # is around 1e-20 here, so an absolute learning rate moves the
            # phases not at all. A unit-norm step makes progress independent of
            # that scale.
            norm = np.linalg.norm(gradient)
            if norm < 1e-30:
                break
            phases = np.mod(phases + step * gradient / norm, 2 * np.pi)

            rate = sum_rate(phases)
            if rate > best_rate:
                best_phases, best_rate = phases.copy(), rate
            else:
                step *= 0.7  # backtrack when the step overshoots
                if step < 1e-3:
                    break

        return best_phases
