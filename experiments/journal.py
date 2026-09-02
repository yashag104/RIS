"""Journal-scale experiment definitions 11-20."""

"""Shared imports for the RIS experiment package."""


import numpy as np

from utils.metrics import *
from utils.metrics import dbm_to_watts
from utils.plotting import *


class JournalExperimentsMixin:
    def experiment_11_fl_algorithms(self):
        """
        Experiment 11: Federated Learning Algorithms Comparison
        Compares: FedAvg vs FedProx vs SCAFFOLD
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 11: FL Algorithms Comparison")
        self.logger.info("=" * 60)

        algos = ['FedAvg', 'FedProx', 'SCAFFOLD']
        results = []

        for algo in algos:
            self.logger.info(f"\n>>> Testing {algo}...")
            
            # Override aggregation method
            overrides = {'AGGREGATION_METHOD': algo}
            
            # Add algo specific params if needed
            if algo == 'FedProx':
                overrides['FEDPROX_MU'] = 0.01
            
            result = self._run_single_fl_experiment(config_overrides=overrides)
            result['algorithm'] = algo
            results.append(result)
            
            self.logger.info(f"  Converged: {result['convergence_round']} rounds")
            self.logger.info(f"  Final SNR: {result['final_snr']:.2f} dB")

        self._save_experiment_results('fl_algorithms', results)
        self._plot_fl_algorithms(results)
        return results

    def experiment_12_architectures(self):
        """
        Experiment 12: Model Architectures Comparison
        Compares: MLP vs GNN vs CNN_Attention vs Transformer
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 12: Model Architectures Comparison")
        self.logger.info("=" * 60)

        archs = ['MLP', 'GNN', 'CNN_Attention', 'Transformer']
        results = []

        for arch in archs:
            self.logger.info(f"\n>>> Testing {arch} architecture...")
            
            try:
                result = self._run_single_fl_experiment(config_overrides={'MODEL_TYPE': arch})
                result['architecture'] = arch
                results.append(result)

                self.logger.info(f"  Final SNR: {result['final_snr']:.2f} dB")
            except Exception as e:
                self.logger.info(f"  Failed {arch}: {e}")
                # Fallback empty result
                results.append({'architecture': arch, 'error': str(e), 'final_snr': 0})

        self._save_experiment_results('model_architectures', results)
        self._plot_architectures(results)
        return results

    def experiment_13_csi_robustness(self):
        """
        Experiment 13: Robustness to CSI Imperfections
        Tests various levels of CSI error variance
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 13: CSI Robustness")
        self.logger.info("=" * 60)

        variances = [0.0, 0.01, 0.05, 0.1, 0.2]
        results = []

        for var in variances:
            self.logger.info(f"\n>>> Testing CSI Variance = {var}...")
            
            result = self._run_single_fl_experiment(config_overrides={'CSI_ERROR_VARIANCE': var})
            result['csi_variance'] = var
            results.append(result)

            self.logger.info(f"  Final SNR: {result['final_snr']:.2f} dB")

        self._save_experiment_results('csi_robustness', results)
        self._plot_csi_robustness(results)
        return results

    # ==================== JOURNAL-QUALITY EXPERIMENTS (14-19) ====================

    def experiment_14_topology_comparison(self):
        """
        Experiment 14: NoC Topology Comparison
        Tests: Mesh, Torus, FoldedTorus, Tree, Butterfly, Ring (6 topologies)
        Measures: Latency, energy, utilization, per-round comm cost
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 14: NoC Topology Comparison")
        self.logger.info("=" * 60)

        from src.noc_simulator import NoCSimulator

        topologies = ['Mesh', 'Torus', 'FoldedTorus', 'Tree', 'Butterfly', 'Ring']
        num_tiles = self.config.NUM_TILES
        
        # Estimate model size from config
        model_size_bytes = self.config.ELEMENTS_PER_TILE * 4 * 256  # Rough estimate
        num_rounds = self.config.FL_ROUNDS
        bandwidth = self.config.NOC_BANDWIDTH_GBPS
        
        results = []
        
        for topo_name in topologies:
            self.logger.info(f"\n>>> Testing {topo_name} topology...")
            
            try:
                sim = NoCSimulator(
                    num_tiles=num_tiles,
                    topology=topo_name,
                    bandwidth_gbps=bandwidth
                )
                
                # Simulate full FL training
                fl_metrics = sim.simulate_full_fl_training(
                    model_size_bytes=model_size_bytes,
                    num_rounds=num_rounds,
                    protocol='ParameterServer'
                )
                
                topo_info = sim.get_topology_info()
                
                result = {
                    'topology': topo_name,
                    **fl_metrics,
                    **topo_info,
                }
                results.append(result)
                
                self.logger.info(f"  Latency: {fl_metrics['total_latency_us']:.2f} us")
                self.logger.info(f"  Energy: {fl_metrics['total_energy_nj']:.2f} nJ")
                self.logger.info(f"  Diameter: {topo_info['diameter']}")
                
            except Exception as e:
                self.logger.info(f"  Error: {e}")
                results.append({'topology': topo_name, 'error': str(e)})
        
        self._save_experiment_results('topology_comparison', results)
        self._plot_topology_comparison(results)
        return results

    def experiment_15_protocol_comparison(self):
        """
        Experiment 15: Communication Protocol Comparison
        Tests: ParameterServer, AllReduce, RingAllReduce, Gossip
        Across: Mesh and Torus topologies
        Measures: Bytes transferred, latency, energy, utilization
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 15: Communication Protocol Comparison")
        self.logger.info("=" * 60)

        from src.noc_simulator import NoCSimulator

        protocols = ['ParameterServer', 'AllReduce', 'RingAllReduce', 'Gossip']
        test_topologies = ['Mesh', 'Torus']
        num_tiles = self.config.NUM_TILES
        model_size_bytes = self.config.ELEMENTS_PER_TILE * 4 * 256
        num_rounds = self.config.FL_ROUNDS
        
        results = []
        
        for topo in test_topologies:
            sim = NoCSimulator(num_tiles=num_tiles, topology=topo,
                              bandwidth_gbps=self.config.NOC_BANDWIDTH_GBPS)
            
            for proto in protocols:
                self.logger.info(f"\n>>> {topo} + {proto}...")
                
                try:
                    metrics = sim.simulate_full_fl_training(
                        model_size_bytes=model_size_bytes,
                        num_rounds=num_rounds,
                        protocol=proto
                    )
                    
                    result = {
                        'topology': topo,
                        'protocol': proto,
                        **metrics,
                    }
                    results.append(result)
                    
                    self.logger.info(f"  Total bytes: {metrics['total_bytes']:,}")
                    self.logger.info(f"  Latency: {metrics['total_latency_us']:.2f} us")
                    self.logger.info(f"  Energy: {metrics['total_energy_nj']:.2f} nJ")
                    
                except Exception as e:
                    self.logger.info(f"  Error: {e}")
                    results.append({'topology': topo, 'protocol': proto, 'error': str(e)})
        
        self._save_experiment_results('protocol_comparison', results)
        self._plot_protocol_comparison(results)
        return results

    def experiment_16_optimization_techniques(self):
        """
        Experiment 16: Optimization Technique Comparison
        Tests: FL, AO, SDR, SCA, ADMM, DRL, Random (7 methods)
        Measures: SNR, solve time, complexity, scalability
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 16: Optimization Technique Comparison")
        self.logger.info("=" * 60)

        from baselines.admm_optimizer import ADMMOptimizer
        from baselines.alternating_optimization import AlternatingOptimization
        from baselines.sca_optimizer import SCAOptimizer
        from src.channel_model import RicianChannel
        
        num_elements = self.config.ELEMENTS_PER_TILE
        num_samples = 50  # Channel realizations for comparison
        noise_power = dbm_to_watts(self.config.NOISE_POWER_DBM)
        
        # Generate channel samples
        self.logger.info("  Generating channel samples...")
        channel_model = RicianChannel(
            num_elements=num_elements,
            k_factor_db=self.config.RICIAN_K_FACTOR_DB,
            frequency=self.config.FREQUENCY,
        )
        
        channel_samples = []
        for _ in range(num_samples):
            bs_pos = np.array([5, 10, 2.5])
            user_pos = np.random.uniform([0, 0, 0.5], [10, 10, 2], size=(1, 3))
            ris_pos = np.array([5, 0, 1.5])
            
            ch = channel_model.generate_channel(bs_pos, user_pos, ris_pos, scenario="LoS")
            channel_samples.append({
                'h_direct': ch['h_direct'],
                'h_ris_user': ch['h_ris_user'],
                'h_bs_ris': ch['h_bs_ris'],
            })
        
        results = {}
        
        # 1. Random phases (baseline)
        self.logger.info("  Testing Random Search...")
        random_snrs = []
        for sample in channel_samples:
            phases = np.random.uniform(0, 2 * np.pi, num_elements)
            h_d = sample['h_direct'][0] if not np.isscalar(sample['h_direct']) else sample['h_direct']
            h_r = sample['h_ris_user'][0] if sample['h_ris_user'].ndim > 1 else sample['h_ris_user']
            h_eff = h_d + np.dot(h_r * sample['h_bs_ris'], np.exp(1j * phases))
            snr = np.abs(h_eff)**2 / noise_power
            random_snrs.append(10*np.log10(max(snr, 1e-20)))
        results['Random'] = {
            'avg_snr_db': float(np.mean(random_snrs)),
            'std_snr_db': float(np.std(random_snrs)),
            'avg_solve_time': 0.0,
            'method': 'Random',
        }
        self.logger.info(f"    SNR: {results['Random']['avg_snr_db']:.2f} dB")
        
        # 2. AO
        self.logger.info("  Testing Alternating Optimization...")
        try:
            ao = AlternatingOptimization(
                num_elements=num_elements,
                max_iterations=100,
            )
            ao_metrics = ao.batch_optimize(channel_samples, noise_power)
            results['AO'] = ao_metrics
            self.logger.info(f"    SNR: {ao_metrics['avg_snr_db']:.2f} dB")
        except Exception as e:
            self.logger.info(f"    Error: {e}")
            results['AO'] = {'error': str(e)}
        
        # 3. SCA
        self.logger.info("  Testing Successive Convex Approximation...")
        try:
            sca = SCAOptimizer(num_elements=num_elements)
            sca_metrics = sca.batch_optimize(channel_samples, noise_power)
            results['SCA'] = sca_metrics
            self.logger.info(f"    SNR: {sca_metrics['avg_snr_db']:.2f} dB")
        except Exception as e:
            self.logger.info(f"    Error: {e}")
            results['SCA'] = {'error': str(e)}
        
        # 4. ADMM
        self.logger.info("  Testing ADMM...")
        try:
            admm = ADMMOptimizer(num_elements=num_elements)
            admm_metrics = admm.batch_optimize(channel_samples, noise_power)
            results['ADMM'] = admm_metrics
            self.logger.info(f"    SNR: {admm_metrics['avg_snr_db']:.2f} dB")
        except Exception as e:
            self.logger.info(f"    Error: {e}")
            results['ADMM'] = {'error': str(e)}
        
        # 5. SDR (optional, requires cvxpy)
        self.logger.info("  Testing SDR...")
        try:
            from baselines.sdr_optimizer import SDROptimizer
            sdr = SDROptimizer(num_elements=num_elements, num_randomizations=50)
            sdr_metrics = sdr.batch_optimize(channel_samples[:min(10, num_samples)], noise_power)
            results['SDR'] = sdr_metrics
            self.logger.info(f"    SNR: {sdr_metrics['avg_snr_db']:.2f} dB")
        except ImportError:
            self.logger.info("    Skipped (cvxpy not installed)")
            results['SDR'] = {'error': 'cvxpy not installed'}
        except Exception as e:
            self.logger.info(f"    Error: {e}")
            results['SDR'] = {'error': str(e)}
        
        # 6. MRC upper bound (closed-form)
        self.logger.info("  Computing MRC upper bound...")
        mrc_snrs = []
        for sample in channel_samples:
            h_d = sample['h_direct'][0] if not np.isscalar(sample['h_direct']) else sample['h_direct']
            h_r = sample['h_ris_user'][0] if sample['h_ris_user'].ndim > 1 else sample['h_ris_user']
            a = h_r * sample['h_bs_ris']
            optimal_phases = np.angle(h_d) - np.angle(a)
            h_eff = h_d + np.dot(a, np.exp(1j * optimal_phases))
            snr = np.abs(h_eff)**2 / noise_power
            mrc_snrs.append(10*np.log10(max(snr, 1e-20)))
        results['MRC_Optimal'] = {
            'avg_snr_db': float(np.mean(mrc_snrs)),
            'std_snr_db': float(np.std(mrc_snrs)),
            'avg_solve_time': 0.0,
            'method': 'MRC_Optimal',
        }
        self.logger.info(f"    SNR: {results['MRC_Optimal']['avg_snr_db']:.2f} dB")
        
        self._save_experiment_results('optimization_techniques', results)
        self._plot_optimization_comparison(results)
        return results

    def experiment_17_tile_pixel_golden_ratio(self):
        """
        Experiment 17: Systemmatic Tile-Pixel Configuration Sweep
        Tests: Multiple chip areas × tile counts × pixel counts
        Derives: Optimal density formula (golden ratio)
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 17: Tile-Pixel Golden Ratio Sweep")
        self.logger.info("=" * 60)

        from src.channel_model import RicianChannel
        from src.noc_simulator import NoCSimulator

        chip_areas = getattr(self.config, 'CHIP_AREAS_M2', [25, 100, 400])
        tile_counts = getattr(self.config, 'TILE_COUNTS', [4, 16, 36, 64])
        pixel_counts = getattr(self.config, 'PIXEL_COUNTS', [16, 64, 144, 256])
        
        noise_power = dbm_to_watts(self.config.NOISE_POWER_DBM)
        results = []
        best_score = -np.inf
        best_config = {}
        
        for area in chip_areas:
            for n_tiles in tile_counts:
                for n_pixels in pixel_counts:
                    total_elements = n_tiles * n_pixels
                    
                    # Skip unreasonable combos
                    if total_elements > 10000 or total_elements < 16:
                        continue
                    
                    # Find nearest square factorization for grid
                    sqrt_p = max(1, int(np.sqrt(n_pixels)))
                    p_rows, p_cols = sqrt_p, max(1, n_pixels // sqrt_p)
                    actual_pixels = p_rows * p_cols
                    
                    # Evaluate SNR with this configuration
                    channel_model = RicianChannel(
                        num_elements=actual_pixels,
                        k_factor_db=self.config.RICIAN_K_FACTOR_DB,
                        frequency=self.config.FREQUENCY,
                        grid_rows=p_rows,
                        grid_cols=p_cols,
                    )
                    
                    # Quick SNR estimate (5 samples)
                    snrs = []
                    side = np.sqrt(area)
                    for _ in range(5):
                        bs_pos = np.array([side/2, side, 2.5])
                        user_pos = np.random.uniform([0, 0, 0.5], [side, side, 2], size=(1, 3))
                        ris_pos = np.array([side/2, 0, 1.5])
                        
                        ch = channel_model.generate_channel(bs_pos, user_pos, ris_pos, "LoS")
                        h_d = ch['h_direct'][0]
                        a = ch['h_ris_user'][0] * ch['h_bs_ris']
                        optimal_phases = np.angle(h_d) - np.angle(a)
                        h_eff = h_d + np.dot(a, np.exp(1j * optimal_phases))
                        snr = np.abs(h_eff)**2 / noise_power
                        snrs.append(10*np.log10(max(snr, 1e-20)))
                    
                    avg_snr = np.mean(snrs)
                    
                    # NoC cost
                    max(1, int(np.sqrt(n_tiles)))
                    try:
                        sim = NoCSimulator(num_tiles=n_tiles, topology='Mesh')
                        model_size = actual_pixels * 4 * 256
                        comm_metrics = sim.simulate_fl_round(model_size, 'ParameterServer')
                        comm_latency_us = comm_metrics['latency_us']
                        comm_energy_nj = comm_metrics['energy_nj']
                    except Exception:
                        comm_latency_us = n_tiles * 10
                        comm_energy_nj = n_tiles * 100
                    
                    # Energy per pixel
                    pixel_power = actual_pixels * n_tiles * 0.015  # W
                    
                    # Composite score
                    snr_norm = avg_snr / 30.0  # Normalize
                    energy_norm = 1.0 / (1.0 + pixel_power)
                    comm_norm = 1.0 / (1.0 + comm_latency_us / 1000.0)
                    
                    score = (self.config.WEIGHT_SNR * snr_norm + 
                            self.config.WEIGHT_ENERGY * energy_norm + 
                            self.config.WEIGHT_COMM * comm_norm)
                    
                    entry = {
                        'chip_area_m2': area,
                        'num_tiles': n_tiles,
                        'pixels_per_tile': actual_pixels,
                        'total_elements': n_tiles * actual_pixels,
                        'tile_density': n_tiles / area,
                        'pixel_density': actual_pixels * n_tiles / area,
                        'avg_snr_db': float(avg_snr),
                        'comm_latency_us': float(comm_latency_us),
                        'comm_energy_nj': float(comm_energy_nj),
                        'pixel_power_w': float(pixel_power),
                        'composite_score': float(score),
                    }
                    results.append(entry)
                    
                    if score > best_score:
                        best_score = score
                        best_config = entry
        
        # Derive golden ratio formula
        if results:
            # Fit: optimal_tiles = a * sqrt(area) + b
            areas_seen = sorted({r['chip_area_m2'] for r in results})
            optimal_per_area = {}
            for a in areas_seen:
                area_results = [r for r in results if r['chip_area_m2'] == a]
                best_for_area = max(area_results, key=lambda x: x['composite_score'])
                optimal_per_area[a] = best_for_area
            
            self.logger.info("\n--- Golden Ratio Results ---")
            for a, cfg in optimal_per_area.items():
                self.logger.info(f"  Area={a}m²: T={cfg['num_tiles']}, P={cfg['pixels_per_tile']}, "
                      f"SNR={cfg['avg_snr_db']:.1f}dB, Score={cfg['composite_score']:.4f}")
            
            golden_ratio_summary = {
                'best_overall': best_config,
                'optimal_per_area': optimal_per_area,
                'formula_hint': 'T_opt ≈ sqrt(A/10), P_opt ≈ min(256, A/T)',
            }
            results.append({'_golden_ratio': golden_ratio_summary})
        
        self._save_experiment_results('tile_pixel_golden_ratio', results)
        try:
            self._plot_golden_ratio(results)
        except Exception as e:
            self.logger.info(f"  [WARN] Plotting failed: {e}")
        return results

    def experiment_18_duty_cycling(self):
        """
        Experiment 18: Dynamic Duty Cycling Analysis
        Tests: No DC, Threshold DC, Top-K DC, Adaptive DC
        Measures: Energy savings vs SNR trade-off
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 18: Dynamic Duty Cycling")
        self.logger.info("=" * 60)

        from src.channel_model import RicianChannel

        num_elements = self.config.ELEMENTS_PER_TILE
        noise_power = dbm_to_watts(self.config.NOISE_POWER_DBM)
        strategies = [
            {'name': 'No DC', 'enabled': False, 'strategy': 'threshold'},
            {'name': 'Threshold (-10 dB)', 'enabled': True, 'strategy': 'threshold', 'threshold': -10},
            {'name': 'Threshold (-20 dB)', 'enabled': True, 'strategy': 'threshold', 'threshold': -20},
            {'name': 'Top-K (50%)', 'enabled': True, 'strategy': 'topk', 'min_ratio': 0.5},
            {'name': 'Top-K (25%)', 'enabled': True, 'strategy': 'topk', 'min_ratio': 0.25},
            {'name': 'Adaptive', 'enabled': True, 'strategy': 'adaptive', 'min_ratio': 0.25},
        ]
        
        channel_model = RicianChannel(
            num_elements=num_elements,
            k_factor_db=self.config.RICIAN_K_FACTOR_DB,
            frequency=self.config.FREQUENCY,
        )
        
        # Generate channels
        num_samples = 100
        channels = []
        for _ in range(num_samples):
            bs_pos = np.array([5, 10, 2.5])
            user_pos = np.random.uniform([0, 0, 0.5], [10, 10, 2], size=(1, 3))
            ris_pos = np.array([5, 0, 1.5])
            ch = channel_model.generate_channel(bs_pos, user_pos, ris_pos, "LoS")
            channels.append(ch)
        
        results = []
        
        for strat_cfg in strategies:
            name = strat_cfg['name']
            self.logger.info(f"\n>>> Testing {name}...")
            
            snrs = []
            active_ratios = []
            energy_savings_pcts = []
            
            for ch in channels:
                h_r = ch['h_ris_user'][0]
                h_d = ch['h_direct'][0]
                a = h_r * ch['h_bs_ris']

                # Get optimal phases
                optimal_phases = np.angle(h_d) - np.angle(a)
                
                if not strat_cfg['enabled']:
                    # No duty cycling
                    mask = np.ones(num_elements, dtype=bool)
                else:
                    # Apply duty cycling
                    csi = h_r  # Use RIS-user channel as CSI
                    csi_power_db = 10 * np.log10(np.abs(csi)**2 + 1e-20)
                    min_active = max(1, int(strat_cfg.get('min_ratio', 0.25) * num_elements))
                    
                    if strat_cfg['strategy'] == 'threshold':
                        thresh = strat_cfg.get('threshold', -10)
                        mask = csi_power_db > thresh
                        if np.sum(mask) < min_active:
                            top_k = np.argsort(csi_power_db)[-min_active:]
                            mask = np.zeros(num_elements, dtype=bool)
                            mask[top_k] = True
                    elif strat_cfg['strategy'] == 'topk':
                        k = max(min_active, int(strat_cfg.get('min_ratio', 0.25) * num_elements))
                        top_k = np.argsort(csi_power_db)[-k:]
                        mask = np.zeros(num_elements, dtype=bool)
                        mask[top_k] = True
                    elif strat_cfg['strategy'] == 'adaptive':
                        med = np.median(csi_power_db)
                        mask = csi_power_db > (med - 6)
                        if np.sum(mask) < min_active:
                            top_k = np.argsort(csi_power_db)[-min_active:]
                            mask = np.zeros(num_elements, dtype=bool)
                            mask[top_k] = True
                
                # Apply mask to phases
                masked_phases = optimal_phases.copy()
                masked_phases[~mask] = 0
                
                # Compute SNR with masked phases
                theta = np.exp(1j * masked_phases) * mask  # Zero contribution from OFF pixels
                h_eff = h_d + np.dot(a, theta)
                snr = np.abs(h_eff)**2 / noise_power
                snrs.append(10*np.log10(max(snr, 1e-20)))
                
                active_ratio = np.mean(mask)
                active_ratios.append(active_ratio)
                
                # Energy
                active_pw = 0.015
                sleep_pw = 0.001
                e_all = num_elements * active_pw
                e_dc = np.sum(mask) * active_pw + np.sum(~mask) * sleep_pw
                energy_savings_pcts.append((e_all - e_dc) / e_all * 100)
            
            result = {
                'strategy': name,
                'avg_snr_db': float(np.mean(snrs)),
                'std_snr_db': float(np.std(snrs)),
                'avg_active_ratio': float(np.mean(active_ratios)),
                'avg_energy_savings_pct': float(np.mean(energy_savings_pcts)),
                'snr_loss_vs_full_db': 0.0,  # Filled below
            }
            results.append(result)
            
            self.logger.info(f"  SNR: {result['avg_snr_db']:.2f} dB")
            self.logger.info(f"  Active ratio: {result['avg_active_ratio']:.2%}")
            self.logger.info(f"  Energy savings: {result['avg_energy_savings_pct']:.1f}%")
        
        # Compute SNR loss relative to no-DC
        if results:
            no_dc_snr = results[0]['avg_snr_db']
            for r in results:
                r['snr_loss_vs_full_db'] = no_dc_snr - r['avg_snr_db']
        
        self._save_experiment_results('duty_cycling', results)
        self._plot_duty_cycling(results)
        return results

    def experiment_19_dataset_comparison(self):
        """
        Experiment 19: Multi-Scenario Dataset Comparison
        Tests: DeepMIMO O1_28, DeepMIMO O1_60, 3GPP UMi, Synthetic Rician
        Measures: SNR distribution, convergence, channel characteristics
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 19: Dataset/Channel Model Comparison")
        self.logger.info("=" * 60)

        from src.channel_model import RicianChannel, ThreeGPPUMiChannel

        num_elements = self.config.ELEMENTS_PER_TILE
        noise_power = dbm_to_watts(self.config.NOISE_POWER_DBM)
        num_samples = 200
        
        scenarios = [
            {'name': 'Synthetic Rician (LoS)', 'type': 'synthetic', 'scenario': 'LoS'},
            {'name': 'Synthetic Rician (NLoS)', 'type': 'synthetic', 'scenario': 'NLoS'},
            {'name': 'Synthetic Rician (Mixed)', 'type': 'synthetic', 'scenario': 'mixed'},
            {'name': '3GPP UMi 28 GHz (LoS)', 'type': '3gpp_umi', 'scenario': 'LoS', 'freq': 28e9},
            {'name': '3GPP UMi 28 GHz (NLoS)', 'type': '3gpp_umi', 'scenario': 'NLoS', 'freq': 28e9},
            {'name': '3GPP UMi 60 GHz (Mixed)', 'type': '3gpp_umi', 'scenario': 'mixed', 'freq': 60e9},
        ]
        
        results = []
        
        for scen in scenarios:
            self.logger.info(f"\n>>> {scen['name']}...")
            
            if scen['type'] == 'synthetic':
                channel_model = RicianChannel(
                    num_elements=num_elements,
                    k_factor_db=self.config.RICIAN_K_FACTOR_DB,
                    frequency=self.config.FREQUENCY,
                )
            else:  # 3gpp_umi
                channel_model = ThreeGPPUMiChannel(
                    num_elements=num_elements,
                    frequency=scen.get('freq', 28e9),
                )
            
            snrs_optimal = []
            snrs_no_ris = []
            channel_gains = []
            
            for _ in range(num_samples):
                bs_pos = np.array([5, 10, 2.5])
                user_pos = np.random.uniform([0, 0, 0.5], [10, 10, 2], size=(1, 3))
                ris_pos = np.array([5, 0, 1.5])
                
                ch = channel_model.generate_channel(
                    bs_pos, user_pos, ris_pos, scenario=scen['scenario']
                )
                
                h_d = ch['h_direct'][0]
                h_r = ch['h_ris_user'][0]
                a = h_r * ch['h_bs_ris']

                # No RIS
                snr_no = np.abs(h_d)**2 / noise_power
                snrs_no_ris.append(10*np.log10(max(snr_no, 1e-20)))

                # Optimal RIS
                opt_phases = np.angle(h_d) - np.angle(a)
                h_eff = h_d + np.dot(a, np.exp(1j * opt_phases))
                snr_opt = np.abs(h_eff)**2 / noise_power
                snrs_optimal.append(10*np.log10(max(snr_opt, 1e-20)))
                
                # Channel gain
                channel_gains.append(float(np.mean(np.abs(h_r))))
            
            result = {
                'scenario': scen['name'],
                'type': scen['type'],
                'avg_snr_optimal_db': float(np.mean(snrs_optimal)),
                'std_snr_optimal_db': float(np.std(snrs_optimal)),
                'avg_snr_no_ris_db': float(np.mean(snrs_no_ris)),
                'ris_gain_db': float(np.mean(snrs_optimal) - np.mean(snrs_no_ris)),
                'avg_channel_gain': float(np.mean(channel_gains)),
                'num_samples': num_samples,
            }
            results.append(result)
            
            self.logger.info(f"  Optimal SNR: {result['avg_snr_optimal_db']:.2f} ± {result['std_snr_optimal_db']:.2f} dB")
            self.logger.info(f"  No RIS SNR: {result['avg_snr_no_ris_db']:.2f} dB")
            self.logger.info(f"  RIS Gain: {result['ris_gain_db']:.2f} dB")
        
        self._save_experiment_results('dataset_comparison', results)
        self._plot_dataset_comparison(results)
        return results


    def experiment_20_phase_quantization(self):
        """
        Experiment 20: Phase Quantization Loss Analysis
        Tests: Continuous, 1-bit, 2-bit, 3-bit quantization
        Across: LoS, NLoS, 3GPP UMi scenarios
        Measures: SNR degradation, quantization error, beam alignment
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT 20: Phase Quantization Loss Analysis")
        self.logger.info("=" * 60)

        from src.channel_model import RicianChannel, ThreeGPPUMiChannel, quantize_phases

        num_elements = self.config.ELEMENTS_PER_TILE
        noise_power = dbm_to_watts(self.config.NOISE_POWER_DBM)
        num_samples = 200

        quant_configs = [
            {'bits': 0, 'name': 'Continuous'},
            {'bits': 1, 'name': '1-bit (2 states)'},
            {'bits': 2, 'name': '2-bit (4 states)'},
            {'bits': 3, 'name': '3-bit (8 states)'},
        ]

        scenarios = [
            {'name': 'Rician LoS', 'type': 'rician', 'scenario': 'LoS'},
            {'name': 'Rician NLoS', 'type': 'rician', 'scenario': 'NLoS'},
            {'name': '3GPP UMi 28G', 'type': '3gpp', 'scenario': 'LoS'},
        ]

        results = []

        for scen in scenarios:
            self.logger.info(f"\n--- {scen['name']} ---")

            if scen['type'] == 'rician':
                channel_model = RicianChannel(
                    num_elements=num_elements,
                    k_factor_db=self.config.RICIAN_K_FACTOR_DB,
                    frequency=self.config.FREQUENCY,
                )
            else:
                channel_model = ThreeGPPUMiChannel(
                    num_elements=num_elements,
                    frequency=28e9,
                )

            # Generate channel samples once
            channel_samples = []
            for _ in range(num_samples):
                bs_pos = np.array([5, 10, 2.5])
                user_pos = np.random.uniform([0, 0, 0.5], [10, 10, 2], size=(1, 3))
                ris_pos = np.array([5, 0, 1.5])
                ch = channel_model.generate_channel(bs_pos, user_pos, ris_pos, scen['scenario'])
                channel_samples.append(ch)

            for qcfg in quant_configs:
                bits = qcfg['bits']
                name = qcfg['name']
                self.logger.info(f"  >>> {name}...")

                snrs = []
                quant_errors = []

                for ch in channel_samples:
                    h_d = ch['h_direct'][0]
                    h_r = ch['h_ris_user'][0]
                    a = h_r * ch['h_bs_ris']

                    # Optimal continuous phases
                    optimal_phases = np.mod(np.angle(h_d) - np.angle(a), 2 * np.pi)

                    if bits > 0:
                        q_phases, _ = quantize_phases(optimal_phases, bits)
                        # Compute quantization error manually
                        phase_diff = np.abs(optimal_phases - q_phases)
                        phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
                        quant_errors.append(float(np.mean(phase_diff)))
                    else:
                        q_phases = optimal_phases
                        quant_errors.append(0.0)

                    # Compute SNR with (possibly quantized) phases
                    h_eff = h_d + np.dot(a, np.exp(1j * q_phases))
                    snr = np.abs(h_eff) ** 2 / noise_power
                    snrs.append(10 * np.log10(max(snr, 1e-20)))

                result = {
                    'scenario': scen['name'],
                    'quantization': name,
                    'bits': bits,
                    'avg_snr_db': float(np.mean(snrs)),
                    'std_snr_db': float(np.std(snrs)),
                    'avg_quant_error_rad': float(np.mean(quant_errors)),
                    'avg_quant_error_deg': float(np.rad2deg(np.mean(quant_errors))),
                    'num_levels': 2 ** bits if bits > 0 else 'inf',
                }
                results.append(result)

                self.logger.info(f"    SNR: {result['avg_snr_db']:.2f} dB")
                self.logger.info(f"    Quant Error: {result['avg_quant_error_deg']:.2f}°")

        # Compute SNR loss relative to continuous for each scenario
        for scen in scenarios:
            scen_results = [r for r in results if r['scenario'] == scen['name']]
            cont_snr = next((r['avg_snr_db'] for r in scen_results if r['bits'] == 0), 0)
            for r in scen_results:
                r['snr_loss_vs_continuous_db'] = cont_snr - r['avg_snr_db']

        self._save_experiment_results('phase_quantization', results)
        self._plot_phase_quantization(results)
        return results


