"""
Alternating Direction Method of Multipliers (ADMM) Optimizer for RIS Phase Configuration
Based on: Boyd et al., "Distributed Optimization and Statistical Learning via the
Alternating Direction Method of Multipliers," Foundations and Trends in ML, 2011
Applied to RIS: unit-modulus constrained quadratic optimization

Algorithm:
1. Split variable: θ (unconstrained) and z (unit-modulus constrained)
2. Iteratively:
   a) θ-update: gradient step on augmented Lagrangian (unconstrained)
   b) z-update: project onto unit-modulus manifold
   c) dual variable update: u ← u + θ - z
3. Penalty parameter ρ auto-tuned via residual balancing

Advantages:
- Naturally decomposable across tiles (distributable)
- Handles unit-modulus constraint via simple projection
- Converges for convex problems; empirically good for non-convex
- Suitable for on-chip distributed implementation

Disadvantages:
- May converge slowly for ill-conditioned problems
- Penalty parameter tuning affects convergence speed
"""

import numpy as np
from typing import Dict, List, Optional


class ADMMOptimizer:
    """
    ADMM for RIS phase optimization with unit-modulus constraints.
    
    Solves: max_θ |h_d + a^H θ|^2  s.t. |θ_n| = 1
    
    Reformulation as consensus ADMM:
        min_{θ, z}  -|h_d + a^H θ|^2
        s.t.  θ = z, |z_n| = 1
    
    Augmented Lagrangian:
        L_ρ(θ, z, u) = -|h_d + a^H θ|^2 + (ρ/2)||θ - z + u||^2
    """
    
    def __init__(
        self,
        num_elements: int,
        max_iterations: int = 300,
        rho: float = 1.0,
        rho_min: float = 0.1,
        rho_max: float = 100.0,
        convergence_threshold: float = 1e-5,
        adaptive_rho: bool = True,
        mu_adapt: float = 10.0,
        tau_adapt: float = 2.0,
        verbose: bool = False
    ):
        """
        Args:
            num_elements: Number of RIS reflecting elements
            max_iterations: Maximum ADMM iterations
            rho: Initial penalty parameter
            rho_min: Minimum penalty parameter
            rho_max: Maximum penalty parameter
            convergence_threshold: Primal/dual residual threshold
            adaptive_rho: Whether to auto-tune rho via residual balancing
            mu_adapt: Threshold ratio for rho adaptation
            tau_adapt: Scaling factor for rho adaptation
            verbose: Print iteration progress
        """
        self.num_elements = num_elements
        self.max_iterations = max_iterations
        self.rho = rho
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.convergence_threshold = convergence_threshold
        self.adaptive_rho = adaptive_rho
        self.mu_adapt = mu_adapt
        self.tau_adapt = tau_adapt
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
        Optimize RIS phase shifts using ADMM.
        
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
        
        # Construct composite channel vector
        if h_ris_user.ndim > 1:
            h_r = h_ris_user[0]
        else:
            h_r = h_ris_user
            
        h_d = h_direct[0] if not np.isscalar(h_direct) else h_direct
        a = h_r * h_bs_ris  # (N,)
        
        # Initialize variables
        if initial_phases is not None:
            theta = np.exp(1j * initial_phases)
        else:
            theta = np.exp(1j * np.random.uniform(0, 2 * np.pi, N))
        
        z = theta.copy()
        u = np.zeros(N, dtype=complex)  # Scaled dual variable
        rho = self.rho
        
        # Precompute: H = a a^H is rank-1, so (H + ρI)^{-1} has closed form
        # via Sherman-Morrison: (ρI + a a^H)^{-1} = (1/ρ)(I - a a^H / (ρ + ||a||^2))
        a_norm_sq = np.dot(a, np.conj(a)).real
        
        obj_history = []
        primal_residuals = []
        dual_residuals = []
        
        for iteration in range(self.max_iterations):
            z_old = z.copy()
            
            # === θ-update: minimize -|h_d + a^H θ|^2 + (ρ/2)||θ - z + u||^2 ===
            # Gradient of -|h_d + a^H θ|^2 w.r.t. θ = -a * conj(h_d + a^H θ)
            # Setting gradient of augmented Lagrangian to zero:
            # -a * conj(h_d) - a * (a^H θ) * conj(...) + ρ(θ - z + u) = 0
            # This is complex, so we use an iterative approach on θ:
            # (a a^H + ρ I) θ = a * conj(h_d) + ρ(z - u)
            
            rhs = a * np.conj(h_d) + rho * (z - u)
            
            # Sherman-Morrison inverse: (ρI + a a^H)^{-1} b = (1/ρ)(b - a(a^H b)/(ρ + ||a||^2))
            a_dot_rhs = np.dot(np.conj(a), rhs)
            theta = (rhs - a * a_dot_rhs / (rho + a_norm_sq)) / rho
            
            # === z-update: project onto unit-modulus ===
            # min (ρ/2)||θ + u - z||^2  s.t. |z_n| = 1
            # Solution: z_n = (θ_n + u_n) / |θ_n + u_n|
            v = theta + u
            z = v / np.abs(v)
            
            # Handle any zero entries
            zero_mask = np.abs(v) < 1e-10
            if np.any(zero_mask):
                z[zero_mask] = np.exp(1j * np.random.uniform(0, 2 * np.pi, 
                                                              np.sum(zero_mask)))
            
            # === Dual variable update ===
            u = u + theta - z
            
            # === Compute residuals ===
            primal_res = np.linalg.norm(theta - z)
            dual_res = rho * np.linalg.norm(z - z_old)
            primal_residuals.append(float(primal_res))
            dual_residuals.append(float(dual_res))
            
            # Objective (evaluated at z which satisfies constraints)
            h_eff = h_d + np.dot(a, z)
            obj = np.abs(h_eff) ** 2
            obj_history.append(float(obj))
            
            # === Adaptive ρ ===
            if self.adaptive_rho:
                if primal_res > self.mu_adapt * dual_res:
                    rho *= self.tau_adapt
                    u /= self.tau_adapt  # Scale dual variable
                elif dual_res > self.mu_adapt * primal_res:
                    rho /= self.tau_adapt
                    u *= self.tau_adapt
                rho = np.clip(rho, self.rho_min, self.rho_max)
                # Update precomputation
                a_norm_sq = np.dot(a, np.conj(a)).real
            
            # === Convergence check ===
            eps_pri = np.sqrt(N) * 1e-4 + self.convergence_threshold * max(
                np.linalg.norm(theta), np.linalg.norm(z))
            eps_dual = np.sqrt(N) * 1e-4 + self.convergence_threshold * np.linalg.norm(u)
            
            if primal_res < eps_pri and dual_res < eps_dual:
                break
        
        # Final phases from z (which satisfies unit-modulus)
        phases = np.angle(z) % (2 * np.pi)
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
            'final_primal_residual': float(primal_residuals[-1]),
            'final_dual_residual': float(dual_residuals[-1]),
            'final_rho': float(rho),
            'obj_history': obj_history,
            'num_elements': self.num_elements,
            'method': 'ADMM',
        }
    
    def optimize_phases_distributed(
        self,
        h_direct: np.ndarray,
        h_ris_user: np.ndarray,
        h_bs_ris: np.ndarray,
        noise_power: float,
        num_tiles: int,
        initial_phases: np.ndarray = None
    ) -> Dict:
        """
        Distributed ADMM: each tile optimizes its own subset of phases.
        
        This demonstrates ADMM's natural decomposability for on-chip FL.
        Each tile solves a local subproblem, then consensus is enforced.
        
        Args:
            h_direct: BS-User direct channel
            h_ris_user: RIS-User channel
            h_bs_ris: BS-RIS channel
            noise_power: Noise power
            num_tiles: Number of tiles (determines partition)
            initial_phases: Optional initial phases
            
        Returns:
            Dictionary with optimized phases and per-tile metrics
        """
        import time
        start_time = time.time()
        
        N = self.num_elements
        
        if h_ris_user.ndim > 1:
            h_r = h_ris_user[0]
        else:
            h_r = h_ris_user
        h_d = h_direct[0] if not np.isscalar(h_direct) else h_direct
        a = h_r * h_bs_ris
        
        # Partition elements across tiles
        elements_per_tile = N // num_tiles
        tile_indices = []
        for t in range(num_tiles):
            start = t * elements_per_tile
            end = start + elements_per_tile if t < num_tiles - 1 else N
            tile_indices.append(list(range(start, end)))
        
        # Initialize
        if initial_phases is not None:
            z_global = np.exp(1j * initial_phases)
        else:
            z_global = np.exp(1j * np.random.uniform(0, 2 * np.pi, N))
        
        u_tiles = [np.zeros(len(idx), dtype=complex) for idx in tile_indices]
        rho = self.rho
        
        obj_history = []
        
        for iteration in range(self.max_iterations):
            # === Local θ-updates (parallelizable across tiles) ===
            theta_global = np.zeros(N, dtype=complex)
            
            for t, idx in enumerate(tile_indices):
                a_t = a[idx]
                z_t = z_global[idx]
                u_t = u_tiles[t]
                
                # Local update using gradient of global objective restricted to tile
                # Simplified: align local phases with gradient direction
                h_eff = h_d + np.dot(a, z_global)
                grad_t = a_t * np.conj(h_eff)
                rhs = grad_t + rho * (z_t - u_t)
                
                a_t_norm_sq = np.dot(a_t, np.conj(a_t)).real
                a_t_dot_rhs = np.dot(np.conj(a_t), rhs)
                theta_t = (rhs - a_t * a_t_dot_rhs / (rho + a_t_norm_sq + 1e-10)) / rho
                
                theta_global[idx] = theta_t
            
            # === Global z-update: project to unit modulus ===
            z_old = z_global.copy()
            for t, idx in enumerate(tile_indices):
                v = theta_global[idx] + u_tiles[t]
                z_global[idx] = v / (np.abs(v) + 1e-10)
            
            # === Dual updates ===
            for t, idx in enumerate(tile_indices):
                u_tiles[t] = u_tiles[t] + theta_global[idx] - z_global[idx]
            
            # Objective
            h_eff = h_d + np.dot(a, z_global)
            obj = np.abs(h_eff) ** 2
            obj_history.append(float(obj))
            
            # Convergence
            primal_res = np.linalg.norm(theta_global - z_global)
            dual_res = rho * np.linalg.norm(z_global - z_old)
            
            eps_tol = np.sqrt(N) * self.convergence_threshold
            if primal_res < eps_tol and dual_res < eps_tol:
                break
        
        phases = np.angle(z_global) % (2 * np.pi)
        solve_time = time.time() - start_time
        snr_linear = obj / noise_power
        snr_db = 10 * np.log10(max(snr_linear, 1e-20))
        
        return {
            'phases': phases,
            'snr_db': snr_db,
            'snr_linear': float(snr_linear),
            'solve_time': solve_time,
            'iterations': iteration + 1,
            'converged': iteration + 1 < self.max_iterations,
            'num_tiles': num_tiles,
            'elements_per_tile': elements_per_tile,
            'obj_history': obj_history,
            'num_elements': self.num_elements,
            'method': 'ADMM-Distributed',
        }
    
    def batch_optimize(
        self,
        channel_samples: List[Dict],
        noise_power: float
    ) -> Dict:
        """
        Run ADMM on multiple channel realizations.
        """
        snrs = []
        times = []
        iterations = []
        
        for i, sample in enumerate(channel_samples):
            if self.verbose and i % 10 == 0:
                print(f"  ADMM: Processing sample {i+1}/{len(channel_samples)}")
                
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
            'method': 'ADMM',
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
        """Estimate computational complexity."""
        N = self.num_elements
        avg_iters = np.mean(self.iteration_counts) if self.iteration_counts else self.max_iterations
        return {
            'method': 'ADMM',
            'per_iteration_complexity': f'O(N) = O({N})',
            'avg_iterations': avg_iters,
            'total_complexity': f'O({avg_iters:.0f} * {N})',
            'total_flops_estimate': avg_iters * N * 15,
            'requires_centralized_csi': False,  # Can be distributed!
            'online_capable': True,
            'distributable': True,
        }
