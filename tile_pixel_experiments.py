"""
Tile-Pixel Optimization Experiments for RIS Federated Learning
Finds the "Golden Ratio" between tiles, pixels, and chip area
Includes NoC topology comparison and sleep scheduling analysis
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import copy
import os
import json
import pickle
from datetime import datetime

from config import Config
from models.ris_net import RISNet
from src.dataset_utils import create_non_iid_datasets, create_test_dataset
from src.client import RISClient
from src.server import FederatedServer
from utils.metrics import (
    calculate_noc_topology_metrics,
    compare_all_topologies,
    calculate_composite_score,
    calculate_tile_efficiency,
    calculate_area_coverage,
    calculate_optimal_tiles_formula,
    calculate_sleep_energy_savings
)


class TilePixelExperiments:
    """
    Comprehensive experiment suite for tile-pixel optimization
    """

    def __init__(self, config):
        self.config = config
        self.results_dir = os.path.join(config.RESULTS_DIR, 'tile_pixel_experiments')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Store original config values
        self.original_num_tiles = config.NUM_TILES
        self.original_elements_per_tile = config.ELEMENTS_PER_TILE
        self.original_room_size = config.ROOM_SIZE

    def _reset_config(self):
        """Reset config to original values"""
        Config.update_tile_config(
            int(np.sqrt(self.original_num_tiles)),
            int(np.sqrt(self.original_num_tiles)),
            int(np.sqrt(self.original_elements_per_tile)),
            int(np.sqrt(self.original_elements_per_tile))
        )
        Config.ROOM_SIZE = self.original_room_size
        Config.CHIP_AREA_M2 = self.original_room_size[0] * self.original_room_size[1]

    def experiment_1_tile_optimization(self, quick=False):
        """
        Experiment 1: Find optimal number of tiles for fixed chip area.
        
        Tests: T = [4, 9, 16, 25, 36, 49, 64] tiles (perfect squares)
        Fixed: P = 64 pixels/tile, A = 100 m²
        Measures: SNR, Energy, Communication, NoC utilization, Composite score
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 1: TILE OPTIMIZATION")
        print("=" * 60)
        
        # Test configurations (perfect squares for grid layout)
        tile_configs = [4, 9, 16, 25, 36, 49, 64] if not quick else [4, 9, 16]
        pixels_per_tile = 64
        
        results = []
        
        for num_tiles in tile_configs:
            grid_size = int(np.sqrt(num_tiles))
            print(f"\nTesting {num_tiles} tiles ({grid_size}x{grid_size} grid)...")
            
            # Update config
            Config.update_tile_config(grid_size, grid_size, 8, 8)
            
            # Run FL experiment
            result = self._run_fl_experiment(quick=quick)
            
            # Calculate composite score
            composite = calculate_composite_score(
                result.get('snr_gain', 0),
                result.get('total_energy_mj', 0),
                result.get('total_communication_kb', 0),
                weight_snr=Config.WEIGHT_SNR,
                weight_energy=Config.WEIGHT_ENERGY,
                weight_comm=Config.WEIGHT_COMM
            )
            
            # Calculate NoC metrics
            noc_metrics = calculate_noc_topology_metrics(
                num_tiles, Config.NOC_TOPOLOGY,
                result.get('total_communication_kb', 0) * 1024,
                Config.NOC_BANDWIDTH_GBPS,
                Config.FL_ROUNDS
            )
            
            # Calculate tile efficiency
            efficiency = calculate_tile_efficiency(
                result.get('snr_gain', 0),
                result.get('total_energy_mj', 0) / 1000,  # Convert to J
                num_tiles
            )
            
            result.update({
                'num_tiles': num_tiles,
                'grid_size': grid_size,
                'pixels_per_tile': pixels_per_tile,
                'composite_score': composite,
                'noc_metrics': noc_metrics,
                'tile_efficiency': efficiency
            })
            
            results.append(result)
            
            print(f"  SNR Gain: {result.get('snr_gain', 0):.2f} dB")
            print(f"  Energy: {result.get('total_energy_mj', 0):.2f} mJ")
            print(f"  Composite Score: {composite['composite_score']:.4f}")
            print(f"  NoC Utilization: {noc_metrics['bandwidth_utilization']*100:.1f}%")
        
        # Find optimal configuration
        best_idx = np.argmax([r['composite_score']['composite_score'] for r in results])
        best_config = results[best_idx]
        
        print(f"\n[OPTIMAL] {best_config['num_tiles']} tiles (score: {best_config['composite_score']['composite_score']:.4f})")
        
        # Save results
        self._save_results('tile_optimization', results)
        self._plot_tile_optimization(results)
        
        self._reset_config()
        return results

    def experiment_2_pixel_optimization(self, quick=False):
        """
        Experiment 2: Find optimal pixels per tile.
        
        Fixed: T = 9 tiles (3x3)
        Tests: P = [16, 36, 64, 100, 144, 196] pixels/tile
        Measures: SNR, Energy, Communication, Convergence
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: PIXEL OPTIMIZATION")
        print("=" * 60)
        
        num_tiles = 9
        grid_size = 3
        pixel_configs = [16, 36, 64, 100, 144, 196] if not quick else [16, 36, 64]
        
        results = []
        
        for pixels_per_tile in pixel_configs:
            pixel_grid = int(np.sqrt(pixels_per_tile))
            print(f"\nTesting {pixels_per_tile} pixels/tile ({pixel_grid}x{pixel_grid})...")
            
            # Update config
            Config.update_tile_config(grid_size, grid_size, pixel_grid, pixel_grid)
            
            # Run FL experiment
            result = self._run_fl_experiment(quick=quick)
            
            # Calculate area coverage
            coverage = calculate_area_coverage(
                num_tiles, pixels_per_tile,
                Config.CHIP_AREA_M2,
                Config.WAVELENGTH
            )
            
            # Calculate composite score
            composite = calculate_composite_score(
                result.get('snr_gain', 0),
                result.get('total_energy_mj', 0),
                result.get('total_communication_kb', 0)
            )
            
            result.update({
                'num_tiles': num_tiles,
                'pixels_per_tile': pixels_per_tile,
                'pixel_grid': pixel_grid,
                'area_coverage': coverage,
                'composite_score': composite
            })
            
            results.append(result)
            
            print(f"  SNR Gain: {result.get('snr_gain', 0):.2f} dB")
            print(f"  Coverage: {coverage['coverage_percentage']:.4f}%")
            print(f"  Composite Score: {composite['composite_score']:.4f}")
        
        # Find optimal
        best_idx = np.argmax([r['composite_score']['composite_score'] for r in results])
        best_config = results[best_idx]
        
        print(f"\n[OPTIMAL] {best_config['pixels_per_tile']} pixels/tile (score: {best_config['composite_score']['composite_score']:.4f})")
        
        self._save_results('pixel_optimization', results)
        self._plot_pixel_optimization(results)
        
        self._reset_config()
        return results

    def experiment_3_area_scaling(self, quick=False):
        """
        Experiment 3: How should tiles scale with chip area?
        
        Tests: A = [25, 100, 225, 400] m² (5x5 to 20x20 rooms)
        Tests: T = [4, 9, 16, 25] tiles for each area
        Derives: Golden ratio formula
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: AREA SCALING (GOLDEN RATIO)")
        print("=" * 60)
        
        # Area configurations (room sizes)
        if quick:
            area_configs = [(5, 5), (10, 10)]
            tile_configs = [4, 9]
        else:
            area_configs = [(5, 5), (10, 10), (15, 15), (20, 20)]
            tile_configs = [4, 9, 16, 25]
        
        results = []
        
        for room_x, room_y in area_configs:
            area_m2 = room_x * room_y
            Config.update_room_size(room_x, room_y)
            
            print(f"\n--- Room: {room_x}x{room_y}m ({area_m2} m²) ---")
            
            for num_tiles in tile_configs:
                grid_size = int(np.sqrt(num_tiles))
                print(f"  Testing {num_tiles} tiles...")
                
                Config.update_tile_config(grid_size, grid_size, 8, 8)
                
                # Predict optimal using formula
                predicted = calculate_optimal_tiles_formula(
                    area_m2, 64, Config.NOC_BANDWIDTH_GBPS, Config.FL_ROUNDS
                )
                
                # Run experiment
                result = self._run_fl_experiment(quick=True)  # Quick for scaling
                
                composite = calculate_composite_score(
                    result.get('snr_gain', 0),
                    result.get('total_energy_mj', 0),
                    result.get('total_communication_kb', 0)
                )
                
                result.update({
                    'room_size': (room_x, room_y),
                    'chip_area_m2': area_m2,
                    'num_tiles': num_tiles,
                    'predicted_optimal_tiles': predicted['optimal_tiles_grid'],
                    'composite_score': composite
                })
                
                results.append(result)
                print(f"    Score: {composite['composite_score']:.4f}")
        
        # Derive golden ratio
        self._derive_golden_ratio(results)
        
        self._save_results('area_scaling', results)
        self._plot_area_scaling(results)
        
        self._reset_config()
        return results

    def experiment_4_noc_topology_comparison(self, quick=False):
        """
        Experiment 4: Compare NoC topologies.
        
        Tests: Mesh, Torus, FoldedTorus, Tree, Butterfly
        Fixed: 9 tiles, 64 pixels/tile
        Measures: Latency, Utilization, Power
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 4: NoC TOPOLOGY COMPARISON")
        print("=" * 60)
        
        # First run FL to get communication data
        Config.update_tile_config(3, 3, 8, 8)
        fl_result = self._run_fl_experiment(quick=quick)
        
        bytes_transmitted = fl_result.get('total_communication_kb', 0) * 1024
        
        # Compare all topologies
        comparison = compare_all_topologies(
            Config.NUM_TILES,
            bytes_transmitted,
            Config.NOC_BANDWIDTH_GBPS,
            Config.FL_ROUNDS
        )
        
        print("\nTopology Comparison Results:")
        print("-" * 50)
        
        for topology, metrics in comparison['results'].items():
            print(f"\n{topology}:")
            print(f"  Avg Hops: {metrics['avg_hops']:.2f}")
            print(f"  Latency: {metrics['avg_latency_ms']:.3f} ms")
            print(f"  Utilization: {metrics['bandwidth_utilization']*100:.1f}%")
            print(f"  Power: {metrics['power_w']:.2f} W")
            print(f"  Congested: {'YES [CONGESTED]' if metrics['is_congested'] else 'NO [OK]'}")
        
        print(f"\n[BEST] By Latency: {comparison['latency_ranking'][0]}")
        print(f"[BEST] By Utilization: {comparison['utilization_ranking'][0]}")
        print(f"[BEST] By Power: {comparison['power_ranking'][0]}")
        
        self._save_results('noc_topology_comparison', comparison)
        self._plot_topology_comparison(comparison)
        
        self._reset_config()
        return comparison

    def experiment_5_sleep_scheduling(self, quick=False):
        """
        Experiment 5: Compare always-on vs dynamic sleep scheduling.
        
        Tests: Sleep enabled/disabled
        Measures: Energy savings, SNR impact
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 5: SLEEP SCHEDULING IMPACT")
        print("=" * 60)
        
        Config.update_tile_config(3, 3, 8, 8)
        results = {}
        
        # Test without sleep
        print("\n--- Always-On Mode ---")
        Config.SLEEP_SCHEDULING_ENABLED = False
        result_no_sleep = self._run_fl_experiment(quick=quick)
        results['always_on'] = result_no_sleep
        
        # Test with sleep scheduling
        print("\n--- Dynamic Sleep Mode ---")
        Config.SLEEP_SCHEDULING_ENABLED = True
        result_with_sleep = self._run_fl_experiment(quick=quick)
        results['dynamic_sleep'] = result_with_sleep
        
        # Calculate theoretical savings
        sleep_savings = calculate_sleep_energy_savings(
            Config.NUM_TILES,
            Config.FL_ROUNDS,
            Config.ACTIVE_POWER_TILE,
            Config.SLEEP_POWER_TILE,
            sleep_ratio=0.3  # Estimated
        )
        
        results['theoretical_savings'] = sleep_savings
        
        print("\nComparison:")
        print("-" * 40)
        print(f"Always-On Energy: {result_no_sleep.get('total_energy_mj', 0):.2f} mJ")
        print(f"Dynamic Sleep Energy: {result_with_sleep.get('total_energy_mj', 0):.2f} mJ")
        print(f"Theoretical Max Savings: {sleep_savings['savings_percentage']:.1f}%")
        
        snr_no_sleep = result_no_sleep.get('snr_gain', 0)
        snr_with_sleep = result_with_sleep.get('snr_gain', 0)
        print(f"\nSNR (Always-On): {snr_no_sleep:.2f} dB")
        print(f"SNR (Dynamic Sleep): {snr_with_sleep:.2f} dB")
        print(f"SNR Penalty: {snr_no_sleep - snr_with_sleep:.2f} dB")
        
        self._save_results('sleep_scheduling', results)
        self._plot_sleep_comparison(results)
        
        Config.SLEEP_SCHEDULING_ENABLED = True  # Reset to enabled
        self._reset_config()
        return results

    def experiment_6_aspect_ratio(self, quick=False):
        """
        Experiment 6: Compare square vs rectangular tile configurations.
        
        Tests: 4x4, 2x8, 8x2 (all 16 tiles)
        Tests: 8x8, 4x16, 16x4 pixels per tile (all 64 pixels)
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 6: ASPECT RATIO ANALYSIS")
        print("=" * 60)
        
        # Tile aspect ratios (all 16 tiles total)
        tile_configs = [(4, 4), (2, 8), (8, 2)]
        # Pixel aspect ratios (all 64 pixels total)  
        pixel_configs = [(8, 8), (4, 16), (16, 4)]
        
        results = []
        
        for tile_rows, tile_cols in tile_configs:
            for pixel_rows, pixel_cols in pixel_configs:
                print(f"\nTiles: {tile_rows}x{tile_cols}, Pixels: {pixel_rows}x{pixel_cols}")
                
                Config.update_tile_config(tile_rows, tile_cols, pixel_rows, pixel_cols)
                
                result = self._run_fl_experiment(quick=quick)
                
                composite = calculate_composite_score(
                    result.get('snr_gain', 0),
                    result.get('total_energy_mj', 0),
                    result.get('total_communication_kb', 0)
                )
                
                result.update({
                    'tile_config': (tile_rows, tile_cols),
                    'pixel_config': (pixel_rows, pixel_cols),
                    'tile_aspect_ratio': tile_cols / tile_rows,
                    'pixel_aspect_ratio': pixel_cols / pixel_rows,
                    'composite_score': composite
                })
                
                results.append(result)
                print(f"  Composite Score: {composite['composite_score']:.4f}")
        
        # Find best
        best_idx = np.argmax([r['composite_score']['composite_score'] for r in results])
        best = results[best_idx]
        
        print(f"\n[OPTIMAL] Tiles {best['tile_config']}, Pixels {best['pixel_config']}")
        
        self._save_results('aspect_ratio', results)
        self._plot_aspect_ratio(results)
        
        self._reset_config()
        return results

    def _run_fl_experiment(self, quick=False):
        """
        Run a single FL experiment with current config.
        Returns metrics dictionary.
        """
        # Create datasets
        train_datasets, tile_positions = create_non_iid_datasets(
            self.config, self.config.NUM_TILES
        )
        test_dataset = create_test_dataset(self.config)
        
        # Get input dimension
        input_dim = train_datasets[0].get_input_dim()
        
        # Create global model
        global_model = RISNet(
            input_dim=input_dim,
            num_elements=self.config.ELEMENTS_PER_TILE,
            hidden_dim=self.config.HIDDEN_DIM,
            num_layers=self.config.NUM_LAYERS,
            dropout=self.config.DROPOUT
        )
        
        # Create server
        server = FederatedServer(global_model, self.config)
        
        # Create clients
        clients = []
        for i, dataset in enumerate(train_datasets):
            client_model = RISNet(
                input_dim=input_dim,
                num_elements=self.config.ELEMENTS_PER_TILE,
                hidden_dim=self.config.HIDDEN_DIM,
                num_layers=self.config.NUM_LAYERS,
                dropout=self.config.DROPOUT
            )
            client = RISClient(i, client_model, dataset, self.config)
            clients.append(client)
        
        # Training
        rounds = 10 if quick else self.config.FL_ROUNDS
        all_metrics = []
        
        for round_num in range(rounds):
            # Check sleep state for each client
            for client in clients:
                if self.config.SLEEP_SCHEDULING_ENABLED:
                    # Simulate signal strength based on client position
                    signal = 0.5 + 0.5 * np.random.random()
                    client.update_sleep_state(signal)
            
            # Aggregate round
            round_metric = server.aggregate_round(clients, round_num)
            all_metrics.append(round_metric)
        
        # Final evaluation
        test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        clients[0].set_model_weights(server.get_global_weights())
        eval_metrics = clients[0].evaluate(test_loader)
        snr_metrics = clients[0].compute_snr_improvement(test_dataset, num_samples=50)
        
        # Get sleep metrics if enabled
        sleep_metrics = {}
        if self.config.SLEEP_SCHEDULING_ENABLED:
            sleep_metrics = {
                f'client_{c.client_id}': c.get_sleep_metrics() 
                for c in clients
            }
        
        # Communication summary
        comm_summary = server.get_communication_summary()
        
        return {
            'final_loss': eval_metrics.get('loss', 0),
            'phase_error_deg': np.rad2deg(eval_metrics.get('phase_error_mean', 0)),
            'snr_gain': snr_metrics.get('snr_gain_over_no_ris', 0),
            'final_snr': snr_metrics.get('snr_optimized_ris_mean', 0),
            'total_communication_kb': comm_summary.get('total_kilobytes', 0),
            'total_energy_mj': comm_summary.get('energy_communication_joules', 0) * 1000,
            'convergence_rounds': rounds,
            'sleep_metrics': sleep_metrics
        }

    def _derive_golden_ratio(self, results):
        """Derive the golden ratio formula from experimental results."""
        print("\n--- DERIVING GOLDEN RATIO ---")
        
        # Group by area and find optimal tiles for each
        area_optimal = {}
        for r in results:
            area = r['chip_area_m2']
            score = r['composite_score']['composite_score']
            if area not in area_optimal or score > area_optimal[area]['score']:
                area_optimal[area] = {
                    'tiles': r['num_tiles'],
                    'score': score
                }
        
        # Fit T = k * sqrt(A)
        areas = np.array(list(area_optimal.keys()))
        tiles = np.array([v['tiles'] for v in area_optimal.values()])
        
        if len(areas) > 1:
            # Simple linear regression on sqrt(area)
            sqrt_areas = np.sqrt(areas)
            k = np.sum(tiles * sqrt_areas) / np.sum(sqrt_areas ** 2)
            
            print(f"Derived formula: T_optimal ≈ {k:.2f} × √A")
            print(f"For A=100m²: T_optimal ≈ {k * 10:.0f} tiles")
        else:
            print("Not enough data points to derive formula")

    def _save_results(self, experiment_name, results):
        """Save experiment results to JSON file."""
        filepath = os.path.join(self.results_dir, f'{experiment_name}_results.json')
        
        # Convert any non-serializable types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(convert(results), f, indent=2)
        
        print(f"Results saved to: {filepath}")

    # ============ Plotting Functions ============
    
    def _plot_tile_optimization(self, results):
        """Plot tile optimization results."""
        try:
            import matplotlib.pyplot as plt
            
            tiles = [r['num_tiles'] for r in results]
            snr = [r['snr_gain'] for r in results]
            energy = [r['total_energy_mj'] for r in results]
            scores = [r['composite_score']['composite_score'] for r in results]
            utilization = [r['noc_metrics']['bandwidth_utilization'] * 100 for r in results]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            axes[0, 0].plot(tiles, snr, 'bo-', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Number of Tiles')
            axes[0, 0].set_ylabel('SNR Gain (dB)')
            axes[0, 0].set_title('SNR Gain vs Tiles')
            axes[0, 0].grid(True)
            
            axes[0, 1].plot(tiles, energy, 'ro-', linewidth=2, markersize=8)
            axes[0, 1].set_xlabel('Number of Tiles')
            axes[0, 1].set_ylabel('Energy (mJ)')
            axes[0, 1].set_title('Energy Consumption vs Tiles')
            axes[0, 1].grid(True)
            
            axes[1, 0].plot(tiles, utilization, 'go-', linewidth=2, markersize=8)
            axes[1, 0].axhline(y=80, color='r', linestyle='--', label='Target (80%)')
            axes[1, 0].set_xlabel('Number of Tiles')
            axes[1, 0].set_ylabel('NoC Utilization (%)')
            axes[1, 0].set_title('NoC Bandwidth Utilization vs Tiles')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            axes[1, 1].plot(tiles, scores, 'mo-', linewidth=2, markersize=8)
            best_idx = np.argmax(scores)
            axes[1, 1].plot(tiles[best_idx], scores[best_idx], 'g*', markersize=20, label='Optimal')
            axes[1, 1].set_xlabel('Number of Tiles')
            axes[1, 1].set_ylabel('Composite Score')
            axes[1, 1].set_title('Composite Optimization Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'tile_optimization.png'), dpi=300)
            plt.savefig(os.path.join(self.results_dir, 'tile_optimization.pdf'))
            plt.close()
            
            print("Plot saved: tile_optimization.png")
        except Exception as e:
            print(f"Plotting failed: {e}")

    def _plot_pixel_optimization(self, results):
        """Plot pixel optimization results."""
        try:
            import matplotlib.pyplot as plt
            
            pixels = [r['pixels_per_tile'] for r in results]
            snr = [r['snr_gain'] for r in results]
            coverage = [r['area_coverage']['coverage_percentage'] for r in results]
            scores = [r['composite_score']['composite_score'] for r in results]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].plot(pixels, snr, 'bo-', linewidth=2, markersize=8)
            axes[0].set_xlabel('Pixels per Tile')
            axes[0].set_ylabel('SNR Gain (dB)')
            axes[0].set_title('SNR Gain vs Pixels')
            axes[0].grid(True)
            
            axes[1].plot(pixels, coverage, 'go-', linewidth=2, markersize=8)
            axes[1].set_xlabel('Pixels per Tile')
            axes[1].set_ylabel('Area Coverage (%)')
            axes[1].set_title('RIS Coverage vs Pixels')
            axes[1].grid(True)
            
            axes[2].plot(pixels, scores, 'mo-', linewidth=2, markersize=8)
            best_idx = np.argmax(scores)
            axes[2].plot(pixels[best_idx], scores[best_idx], 'g*', markersize=20)
            axes[2].set_xlabel('Pixels per Tile')
            axes[2].set_ylabel('Composite Score')
            axes[2].set_title('Optimization Score')
            axes[2].grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'pixel_optimization.png'), dpi=300)
            plt.close()
            
            print("Plot saved: pixel_optimization.png")
        except Exception as e:
            print(f"Plotting failed: {e}")

    def _plot_area_scaling(self, results):
        """Plot area scaling heatmap."""
        try:
            import matplotlib.pyplot as plt
            
            # Create heatmap data
            areas = sorted(set(r['chip_area_m2'] for r in results))
            tiles = sorted(set(r['num_tiles'] for r in results))
            
            scores = np.zeros((len(tiles), len(areas)))
            for r in results:
                i = tiles.index(r['num_tiles'])
                j = areas.index(r['chip_area_m2'])
                scores[i, j] = r['composite_score']['composite_score']
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(scores, cmap='viridis', aspect='auto')
            
            ax.set_xticks(range(len(areas)))
            ax.set_xticklabels([f'{a}m²' for a in areas])
            ax.set_yticks(range(len(tiles)))
            ax.set_yticklabels([f'{t} tiles' for t in tiles])
            
            ax.set_xlabel('Chip Area')
            ax.set_ylabel('Number of Tiles')
            ax.set_title('Optimal Configuration Heatmap')
            
            plt.colorbar(im, label='Composite Score')
            
            # Mark optimal per column
            for j in range(len(areas)):
                best_i = np.argmax(scores[:, j])
                ax.plot(j, best_i, 'r*', markersize=15)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'area_scaling_heatmap.png'), dpi=300)
            plt.close()
            
            print("Plot saved: area_scaling_heatmap.png")
        except Exception as e:
            print(f"Plotting failed: {e}")

    def _plot_topology_comparison(self, comparison):
        """Plot NoC topology comparison."""
        try:
            import matplotlib.pyplot as plt
            
            topologies = list(comparison['results'].keys())
            latencies = [comparison['results'][t]['avg_latency_ms'] for t in topologies]
            utilizations = [comparison['results'][t]['bandwidth_utilization'] * 100 for t in topologies]
            powers = [comparison['results'][t]['power_w'] for t in topologies]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            colors = ['blue', 'green', 'orange', 'red', 'purple']
            
            axes[0].bar(topologies, latencies, color=colors)
            axes[0].set_ylabel('Latency (ms)')
            axes[0].set_title('Average Latency')
            axes[0].tick_params(axis='x', rotation=45)
            
            axes[1].bar(topologies, utilizations, color=colors)
            axes[1].axhline(y=80, color='r', linestyle='--', label='Target')
            axes[1].set_ylabel('Utilization (%)')
            axes[1].set_title('Bandwidth Utilization')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].legend()
            
            axes[2].bar(topologies, powers, color=colors)
            axes[2].set_ylabel('Power (W)')
            axes[2].set_title('Power Consumption')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'noc_topology_comparison.png'), dpi=300)
            plt.close()
            
            print("Plot saved: noc_topology_comparison.png")
        except Exception as e:
            print(f"Plotting failed: {e}")

    def _plot_sleep_comparison(self, results):
        """Plot sleep scheduling comparison."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            modes = ['Always-On', 'Dynamic Sleep']
            energy = [
                results['always_on'].get('total_energy_mj', 0),
                results['dynamic_sleep'].get('total_energy_mj', 0)
            ]
            snr = [
                results['always_on'].get('snr_gain', 0),
                results['dynamic_sleep'].get('snr_gain', 0)
            ]
            
            axes[0].bar(modes, energy, color=['red', 'green'])
            axes[0].set_ylabel('Energy (mJ)')
            axes[0].set_title('Energy Consumption')
            
            axes[1].bar(modes, snr, color=['blue', 'cyan'])
            axes[1].set_ylabel('SNR Gain (dB)')
            axes[1].set_title('Signal Quality')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'sleep_scheduling_comparison.png'), dpi=300)
            plt.close()
            
            print("Plot saved: sleep_scheduling_comparison.png")
        except Exception as e:
            print(f"Plotting failed: {e}")

    def _plot_aspect_ratio(self, results):
        """Plot aspect ratio analysis."""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            labels = [f"T:{r['tile_config']}\nP:{r['pixel_config']}" for r in results]
            scores = [r['composite_score']['composite_score'] for r in results]
            
            bars = ax.bar(range(len(labels)), scores, color='steelblue')
            
            # Highlight best
            best_idx = np.argmax(scores)
            bars[best_idx].set_color('green')
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel('Composite Score')
            ax.set_title('Aspect Ratio Comparison\n(Tiles x Pixels configurations)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'aspect_ratio_comparison.png'), dpi=300)
            plt.close()
            
            print("Plot saved: aspect_ratio_comparison.png")
        except Exception as e:
            print(f"Plotting failed: {e}")


def run_all_experiments(quick=False):
    """Run all tile-pixel optimization experiments."""
    print("\n" + "=" * 70)
    print("TILE-PIXEL OPTIMIZATION EXPERIMENT SUITE")
    print("=" * 70)
    
    experiments = TilePixelExperiments(Config)
    
    all_results = {}
    
    try:
        all_results['tile_optimization'] = experiments.experiment_1_tile_optimization(quick)
    except Exception as e:
        print(f"Experiment 1 failed: {e}")
    
    try:
        all_results['pixel_optimization'] = experiments.experiment_2_pixel_optimization(quick)
    except Exception as e:
        print(f"Experiment 2 failed: {e}")
    
    try:
        all_results['area_scaling'] = experiments.experiment_3_area_scaling(quick)
    except Exception as e:
        print(f"Experiment 3 failed: {e}")
    
    try:
        all_results['noc_topology'] = experiments.experiment_4_noc_topology_comparison(quick)
    except Exception as e:
        print(f"Experiment 4 failed: {e}")
    
    try:
        all_results['sleep_scheduling'] = experiments.experiment_5_sleep_scheduling(quick)
    except Exception as e:
        print(f"Experiment 5 failed: {e}")
    
    try:
        all_results['aspect_ratio'] = experiments.experiment_6_aspect_ratio(quick)
    except Exception as e:
        print(f"Experiment 6 failed: {e}")
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {experiments.results_dir}")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Tile-Pixel Optimization Experiments')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick mode')
    parser.add_argument('--experiment', '-e', type=int, choices=[1, 2, 3, 4, 5, 6],
                        help='Run specific experiment (1-6)')
    
    args = parser.parse_args()
    
    if args.experiment:
        experiments = TilePixelExperiments(Config)
        exp_funcs = {
            1: experiments.experiment_1_tile_optimization,
            2: experiments.experiment_2_pixel_optimization,
            3: experiments.experiment_3_area_scaling,
            4: experiments.experiment_4_noc_topology_comparison,
            5: experiments.experiment_5_sleep_scheduling,
            6: experiments.experiment_6_aspect_ratio
        }
        exp_funcs[args.experiment](quick=args.quick)
    else:
        run_all_experiments(quick=args.quick)
