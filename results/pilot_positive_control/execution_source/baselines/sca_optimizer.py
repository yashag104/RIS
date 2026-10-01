"""Linear-surrogate phase iteration for supplied-channel SISO power.

This is not a reproduction of a multiuser weighted-sum-rate or multicell MIMO
algorithm. It uses a per-element linear-surrogate alignment and damped unit-circle
projection, with no SOCP/QP solver and no asserted KKT guarantee. The exact SISO
phase-alignment rule remains the meaningful training-free baseline.
"""


import numpy as np

from utils.logger import logger


class SISOPhaseSurrogate:
    """Damped unit-circle projection of the SISO linear surrogate maximizer."""
    
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
            penalty_rho: Deprecated compatibility argument; unused by this routine
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
    ) -> dict:
        """
        Optimize RIS phase shifts using SCA.
        
        The objective: max_θ |h_d + a^T θ|^2  s.t. |θ_n| = 1
        where a_n = h_ris_user_n * h_bs_ris_n.
        
        SCA approach:
        - Write f(θ) = |h_d + a^H θ|^2
        - At point θ^k, compute gradient ∇f(θ^k)
        - Surrogate: f̃(θ; θ^k) = f(θ^k) + 2 Re{∇f(θ^k)^H (θ - θ^k)}
        - Maximize surrogate subject to |θ_n| = 1
        - Solution: θ_n^{k+1} = exp(-j * angle(a_n * conj(h_eff)))
        
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
            
            # SCA update: maximize the linear surrogate subject to |θ_n| = 1.
            # The surrogate is Re{conj(h_eff) * sum_n a_n θ_n}, so the term for
            # element n is maximised at θ_n = exp(-j * angle(a_n * conj(h_eff))),
            # which rotates a_n θ_n onto h_eff. The sign matters: taking
            # +angle(gradient) rotates each contribution away from h_eff and
            # minimises the objective instead, leaving SCA at the level of
            # random phases.
            theta_new = np.exp(-1j * np.angle(gradient))
            
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
            'method': 'SISO linear-surrogate control',
        }
    
    def batch_optimize(
        self,
        channel_samples: list[dict],
        noise_power: float
    ) -> dict:
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
                logger.info(f"  SCA: Processing sample {i+1}/{len(channel_samples)}")
                
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
            'method': 'SISO linear-surrogate control',
            'avg_snr_db': float(np.mean(snrs)),
            'std_snr_db': float(np.std(snrs)),
            'median_snr_db': float(np.median(snrs)),
            'avg_solve_time': float(np.mean(times)),
            'total_time': float(np.sum(times)),
            'avg_iterations': float(np.mean(iterations)),
            'num_samples': len(channel_samples),
            'all_snrs': snrs,
        }
    
    def compute_complexity(self) -> dict:
        """
        Estimate computational complexity.
        
        Returns:
            Dictionary with complexity metrics
        """
        N = self.num_elements
        avg_iters = np.mean(self.iteration_counts) if self.iteration_counts else self.max_iterations
        return {
            'method': 'SISO linear-surrogate control',
            'per_iteration_complexity': f'O(N) = O({N})',
            'avg_iterations': avg_iters,
            'total_complexity': f'O({avg_iters:.0f} * {N})',
            'total_flops_estimate': avg_iters * N * 10,  # ~10 ops per element per iter
            'requires_centralized_csi': False,
            'online_capable': True,  # Can warm-start from previous solution
        }


# Compatibility alias for historical scripts/result keys.
SCAOptimizer = SISOPhaseSurrogate
