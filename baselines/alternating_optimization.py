"""
Alternating Optimization (AO) Baseline for RIS Phase Configuration
Based on: Wu & Zhang, "Intelligent Reflecting Surface Enhanced Wireless Network 
via Joint Active and Passive Beamforming," IEEE TWC 2019

Algorithm:
1. Initialize random phase shifts
2. Repeat:
   - Fix RIS phases, optimize beamformer (closed-form)
   - Fix beamformer, optimize RIS phases (gradient ascent)
3. Until convergence

This serves as a strong baseline - achieves near-optimal performance but:
- Requires centralized CSI collection (privacy violation)
- High computational cost (many iterations per channel realization)
- Not scalable to large-scale deployments
"""

import numpy as np
import torch
from typing import Tuple, Dict, List


class AlternatingOptimization:
    """
    Alternating Optimization for RIS phase configuration.
    
    This is a model-based optimization approach that iteratively optimizes
    the RIS phase shifts and beamformer to maximize received SNR.
    """
    
    def __init__(
        self,
        num_elements: int,
        max_iterations: int = 100,
        lr_phase: float = 0.1,
        convergence_threshold: float = 1e-4,
        verbose: bool = False
    ):
        """
        Args:
            num_elements: Number of RIS reflecting elements
            max_iterations: Maximum AO iterations
            lr_phase: Learning rate for phase gradient ascent
            convergence_threshold: Stop when SNR improvement < threshold (dB)
            verbose: Print iteration progress
        """
        self.num_elements = num_elements
        self.max_iterations = max_iterations
        self.lr_phase = lr_phase
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        
    def optimize_phases(
        self,
        h_direct: np.ndarray,
        h_ris_user: np.ndarray,
        h_bs_ris: np.ndarray,
        noise_power: float,
        initial_phases: np.ndarray = None
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Optimize RIS phase shifts using alternating optimization.
        
        Args:
            h_direct: BS-User direct channel (complex scalar or vector)
            h_ris_user: RIS-User channel (complex vector, shape: [N])
            h_bs_ris: BS-RIS channel (complex vector, shape: [N])
            noise_power: Noise power
            initial_phases: Initial phase configuration (if None, random init)
            
        Returns:
            optimal_phases: Optimized phase shifts in [0, 2π], shape: [N]
            snr_history: SNR value at each iteration (in dB)
        """
        N = self.num_elements
        
        # Initialize phases randomly if not provided
        if initial_phases is None:
            phases = np.random.uniform(0, 2 * np.pi, N)
        else:
            phases = initial_phases.copy()
        
        snr_history = []
        prev_snr = -np.inf
        
        for iteration in range(self.max_iterations):
            # Step 1: Fix phases, optimize beamformer
            # For single-antenna BS, optimal beamformer is just phase alignment
            # (In multi-antenna case, this would be MRT or ZF beamforming)
            Theta = np.diag(np.exp(1j * phases))
            
            # Effective channel: h_eff = h_direct + h_ris_user^H @ Theta @ h_bs_ris
            h_cascade = np.conj(h_ris_user) @ Theta @ h_bs_ris
            h_eff = h_direct + h_cascade
            
            # Compute SNR
            signal_power = np.abs(h_eff) ** 2
            snr_linear = signal_power / noise_power
            snr_db = 10 * np.log10(snr_linear + 1e-10)
            snr_history.append(snr_db)
            
            # Check convergence
            if iteration > 0:
                snr_improvement = snr_db - prev_snr
                if snr_improvement < self.convergence_threshold:
                    if self.verbose:
                        print(f"AO converged at iteration {iteration}, SNR = {snr_db:.2f} dB")
                    break
            
            prev_snr = snr_db
            
            # Step 2: Fix beamformer (implicit), optimize phases via gradient ascent
            # Gradient of SNR w.r.t. phase θ_n:
            # ∂SNR/∂θ_n ∝ 2 * Re{conj(y) * j * exp(jθ_n) * conj(h_ris_user[n]) * h_bs_ris[n]}
            # where y = h_direct + Σ exp(jθ_k) * conj(h_ris_user[k]) * h_bs_ris[k]
            
            # Compute gradient for each phase
            gradient = np.zeros(N)
            for n in range(N):
                # Contribution from other elements
                h_other = sum([
                    np.exp(1j * phases[k]) * np.conj(h_ris_user[k]) * h_bs_ris[k]
                    for k in range(N) if k != n
                ])
                received_signal = h_direct + h_other
                
                # Gradient component
                grad_component = np.conj(received_signal) * 1j * np.exp(1j * phases[n]) * \
                                np.conj(h_ris_user[n]) * h_bs_ris[n]
                gradient[n] = 2 * np.real(grad_component)
            
            # Gradient ascent update (maximize SNR)
            phases = phases + self.lr_phase * gradient
            
            # Project back to [0, 2π]
            phases = np.mod(phases, 2 * np.pi)
            
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: SNR = {snr_db:.2f} dB")
        
        return phases, snr_history
    
    def batch_optimize(
        self,
        channel_samples: List[Dict],
        noise_power: float
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run AO on multiple channel realizations.
        
        Args:
            channel_samples: List of channel sample dicts with keys:
                - 'h_direct': Direct channel
                - 'h_ris_user': RIS-user channel
                - 'h_bs_ris': BS-RIS channel
            noise_power: Noise power
            
        Returns:
            all_phases: Array of optimized phases, shape: [num_samples, N]
            metrics: Dictionary with performance metrics
        """
        num_samples = len(channel_samples)
        all_phases = np.zeros((num_samples, self.num_elements))
        all_snrs = []
        convergence_iters = []
        
        for i, sample in enumerate(channel_samples):
            phases, snr_history = self.optimize_phases(
                h_direct=sample['h_direct'],
                h_ris_user=sample['h_ris_user'],
                h_bs_ris=sample['h_bs_ris'],
                noise_power=noise_power
            )
            
            all_phases[i] = phases
            all_snrs.append(snr_history[-1])
            convergence_iters.append(len(snr_history))
        
        metrics = {
            'avg_snr_db': np.mean(all_snrs),
            'std_snr_db': np.std(all_snrs),
            'min_snr_db': np.min(all_snrs),
            'max_snr_db': np.max(all_snrs),
            'avg_convergence_iters': np.mean(convergence_iters),
            'total_iterations': sum(convergence_iters)
        }
        
        return all_phases, metrics
    
    def compute_complexity(self) -> Dict[str, float]:
        """
        Estimate computational complexity.
        
        Returns:
            complexity: Dictionary with complexity metrics
        """
        N = self.num_elements
        
        # Per iteration complexity:
        # - Beamformer optimization: O(N) for single-antenna BS
        # - Phase gradient computation: O(N²) for naive implementation
        # - Total per iteration: O(N²)
        # - Expected iterations: ~50-100
        
        flops_per_iteration = N ** 2  # Dominant term
        expected_iterations = self.max_iterations * 0.5  # Assume 50% convergence
        total_flops = flops_per_iteration * expected_iterations
        
        return {
            'flops_per_iteration': flops_per_iteration,
            'expected_iterations': expected_iterations,
            'total_flops': total_flops,
            'complexity_class': f"O(N²·I) where N={N}, I={expected_iterations}"
        }


def compare_with_random_init(
    ao: AlternatingOptimization,
    h_direct: np.ndarray,
    h_ris_user: np.ndarray,
    h_bs_ris: np.ndarray,
    noise_power: float,
    num_trials: int = 10
) -> Dict:
    """
    Test sensitivity to random initialization.
    
    AO can get stuck in local minima. This function runs multiple trials
    with different random initializations and reports statistics.
    """
    all_results = []
    
    for trial in range(num_trials):
        phases, snr_history = ao.optimize_phases(
            h_direct, h_ris_user, h_bs_ris, noise_power
        )
        all_results.append({
            'final_snr': snr_history[-1],
            'iterations': len(snr_history),
            'phases': phases
        })
    
    final_snrs = [r['final_snr'] for r in all_results]
    
    # Find best trial
    best_idx = np.argmax(final_snrs)
    worst_idx = np.argmin(final_snrs)
    
    return {
        'best_snr_db': final_snrs[best_idx],
        'worst_snr_db': final_snrs[worst_idx],
        'avg_snr_db': np.mean(final_snrs),
        'std_snr_db': np.std(final_snrs),
        'snr_range_db': final_snrs[best_idx] - final_snrs[worst_idx],
        'best_phases': all_results[best_idx]['phases'],
        'num_trials': num_trials
    }
