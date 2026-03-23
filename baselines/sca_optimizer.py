"""
Successive Convex Approximation (SCA) Optimizer for RIS Phase Configuration
Based on: Pan et al., "Multicell MIMO Communications Relying on Intelligent
Reflecting Surfaces," IEEE TWC 2020

Algorithm:
1. Initialize phase shifts (random or from previous solution)
2. At iteration t, linearize the non-convex objective around θ^(t)
3. Solve the resulting convex subproblem (SOCP or QP)
4. Update θ^(t+1) and repeat until convergence

Advantages:
- Guaranteed convergence to KKT stationary point
- Lower per-iteration complexity than SDR: O(N^2) vs O(N^3.5)
- Can incorporate additional constraints easily

Disadvantages:
- May converge to local optima (depends on initialization)
- Requires centralized CSI
- Multiple iterations needed
"""

import numpy as np
from typing import Dict, List, Optional


class SCAOptimizer:
    """
    Successive Convex Approximation for RIS phase optimization.
    
    Maximizes received SNR by iteratively solving convex approximations
    of the original non-convex unit-modulus constrained problem.
    
    At each iteration, the objective |h_d + a^H θ|^2 is lower-bounded
    by a concave quadratic (first-order Taylor expansion of the concave part),
    and the unit-modulus constraint is handled via penalty or projection.
    """
    
    def __init__(
        self,
        num_elements: int,
        max_iterations: int = 200,
        convergence_threshold: float = 1e-5,
        step_size: float = 0.5,
        penalty_rho: float = 10.0,
        verbose: bool = False
    ):
        """
        Args:
            num_elements: Number of RIS reflecting elements
            max_iterations: Maximum SCA iterations
            convergence_threshold: Convergence threshold (relative change)
            step_size: Step size for convex combination (0, 1]
            penalty_rho: Penalty parameter for unit-modulus constraint
            verbose: Print iteration progress
        """
        self.num_elements = num_elements
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.step_size = step_size
        self.penalty_rho = penalty_rho
        self.verbose = verbose
        self.iteration_counts = []
        
    def optimize_phases(
        self,
        h_direct: np.ndarray,
        h_ris_user: np.ndarray,
        h_bs_ris: np.ndarray,
        noise_power: float,
        initial_phases: np.ndarray = None
    ) -> Dict:
        """
        Optimize RIS phase shifts using SCA.
        
        The objective: max_θ |h_d + a^H θ|^2  s.t. |θ_n| = 1
        where a_n = conj(h_ris_user_n) * h_bs_ris_n.
        
        SCA approach:
        - Write f(θ) = |h_d + a^H θ|^2
        - At point θ^k, compute gradient ∇f(θ^k)
        - Surrogate: f̃(θ; θ^k) = f(θ^k) + 2 Re{∇f(θ^k)^H (θ - θ^k)}
        - Maximize surrogate subject to |θ_n| = 1
        - Solution: θ_n^{k+1} = exp(j * angle(∇f_n(θ^k)))
        
        Args:
            h_direct: BS-User direct channel
            h_ris_user: RIS-User channel (N,) or (K, N)
            h_bs_ris: BS-RIS channel (N,)
            noise_power: Noise power (linear)
            initial_phases: Optional initial phase shifts
            
        Returns:
            Dictionary with optimized phases and metrics
        """
        import time
        start_time = time.time()
        
        N = self.num_elements
        
        # Construct the composite channel vector a
        if h_ris_user.ndim > 1:
            h_r = h_ris_user[0]
        else:
            h_r = h_ris_user
            
        h_d = h_direct[0] if not np.isscalar(h_direct) else h_direct
        a = h_r * h_bs_ris  # (N,)
        
        # Initialize
        if initial_phases is not None:
            theta = np.exp(1j * initial_phases)
        else:
            theta = np.exp(1j * np.random.uniform(0, 2 * np.pi, N))
        
        # Track convergence
        prev_obj = -np.inf
        obj_history = []
        
        for iteration in range(self.max_iterations):
            # Current objective value
            h_eff = h_d + np.dot(a, theta)
            obj = np.abs(h_eff) ** 2
            obj_history.append(float(obj))
            
            # Check convergence
            if iteration > 0:
                rel_change = abs(obj - prev_obj) / max(abs(prev_obj), 1e-10)
                if rel_change < self.convergence_threshold:
                    break
            prev_obj = obj
            
            # Compute gradient of f(θ) = |h_d + a^H θ|^2
            # ∇_θ f = a * conj(h_d + a^H θ)
            gradient = a * np.conj(h_eff)
            
            # SCA update: maximize linear surrogate subject to |θ_n| = 1
            # Optimal solution: θ_n = exp(j * angle(gradient_n))
            theta_new = np.exp(1j * np.angle(gradient))
            
            # Convex combination for stability
            alpha = self.step_size
            theta_combined = alpha * theta_new + (1 - alpha) * theta
            
            # Project back to unit modulus
            theta = theta_combined / np.abs(theta_combined)
        
        # Extract final phases
        phases = np.angle(theta) % (2 * np.pi)
        num_iters = iteration + 1
        self.iteration_counts.append(num_iters)
        
        solve_time = time.time() - start_time
        snr_linear = obj / noise_power
        snr_db = 10 * np.log10(max(snr_linear, 1e-20))
        
        return {
            'phases': phases,
            'snr_db': snr_db,
            'snr_linear': float(snr_linear),
            'solve_time': solve_time,
            'iterations': num_iters,
            'converged': num_iters < self.max_iterations,
            'obj_history': obj_history,
            'num_elements': self.num_elements,
            'method': 'SCA',
        }
    
    def batch_optimize(
        self,
        channel_samples: List[Dict],
        noise_power: float
    ) -> Dict:
        """
        Run SCA on multiple channel realizations.
        
        Args:
            channel_samples: List of channel dicts with keys:
                'h_direct', 'h_ris_user', 'h_bs_ris'
            noise_power: Noise power (linear)
            
        Returns:
            Aggregated metrics dictionary
        """
        snrs = []
        times = []
        iterations = []
        
        for i, sample in enumerate(channel_samples):
            if self.verbose and i % 10 == 0:
                print(f"  SCA: Processing sample {i+1}/{len(channel_samples)}")
                
            result = self.optimize_phases(
                h_direct=sample['h_direct'],
                h_ris_user=sample['h_ris_user'],
                h_bs_ris=sample['h_bs_ris'],
                noise_power=noise_power
            )
            snrs.append(result['snr_db'])
            times.append(result['solve_time'])
            iterations.append(result['iterations'])
        
        return {
            'method': 'SCA',
            'avg_snr_db': float(np.mean(snrs)),
            'std_snr_db': float(np.std(snrs)),
            'median_snr_db': float(np.median(snrs)),
            'avg_solve_time': float(np.mean(times)),
            'total_time': float(np.sum(times)),
            'avg_iterations': float(np.mean(iterations)),
            'num_samples': len(channel_samples),
            'all_snrs': snrs,
        }
    
    def compute_complexity(self) -> Dict:
        """
        Estimate computational complexity.
        
        Returns:
            Dictionary with complexity metrics
        """
        N = self.num_elements
        avg_iters = np.mean(self.iteration_counts) if self.iteration_counts else self.max_iterations
        return {
            'method': 'SCA',
            'per_iteration_complexity': f'O(N) = O({N})',
            'avg_iterations': avg_iters,
            'total_complexity': f'O({avg_iters:.0f} * {N})',
            'total_flops_estimate': avg_iters * N * 10,  # ~10 ops per element per iter
            'requires_centralized_csi': True,
            'online_capable': True,  # Can warm-start from previous solution
        }
