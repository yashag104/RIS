"""
Semidefinite Relaxation (SDR) Optimizer for RIS Phase Configuration
Based on: Wu & Zhang, "Intelligent Reflecting Surface Enhanced Wireless Network
via Joint Active and Passive Beamforming," IEEE TWC 2019

Algorithm:
1. Formulate SNR maximization as QCQP with unit-modulus constraints
2. Lift to SDP via Φ = v v^H, relax rank-1 constraint
3. Solve SDP with cvxpy
4. Extract rank-1 solution via Gaussian randomization + SVD

Advantages:
- Provides theoretical upper bound on achievable SNR
- Globally optimal under SDR relaxation (often tight)
- Well-established in IEEE TWC/JSAC literature

Disadvantages:
- O(N^3.5) complexity per solve (SDP)
- Centralized: requires full CSI
- Not online / adaptive
"""

import numpy as np
from typing import Dict, List, Optional

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False


class SDROptimizer:
    """
    Semidefinite Relaxation for RIS phase optimization.
    
    Maximizes received SNR by solving:
        max_θ |h_d + h_r^H Θ g|^2
    where Θ = diag(e^{jθ_1}, ..., e^{jθ_N}) and g = h_bs_ris.
    
    The SDR reformulation:
        max_v  v^H Q v
        s.t.   |v_n|^2 = 1, n = 1, ..., N+1
    where v = [θ_1 e^{j}, ..., θ_N e^{j}, 1]^T and Q is constructed
    from the channel matrices.
    """
    
    def __init__(
        self,
        num_elements: int,
        num_randomizations: int = 100,
        verbose: bool = False
    ):
        """
        Args:
            num_elements: Number of RIS reflecting elements
            num_randomizations: Number of Gaussian randomizations for rank-1 extraction
            verbose: Print optimization progress
        """
        if not HAS_CVXPY:
            raise ImportError(
                "cvxpy is required for SDR optimizer. Install with: pip install cvxpy"
            )
        self.num_elements = num_elements
        self.num_randomizations = num_randomizations
        self.verbose = verbose
        self.solve_times = []
        
    def _construct_q_matrix(
        self,
        h_direct: np.ndarray,
        h_ris_user: np.ndarray,
        h_bs_ris: np.ndarray
    ) -> np.ndarray:
        """
        Construct the Q matrix for the SDR formulation.
        
        The effective channel is: h_eff = h_d + h_r^H Θ g
        where h_r = h_ris_user (N,), g = h_bs_ris (N,), h_d = h_direct (scalar).
        
        We define v = [θ_1, ..., θ_N, t]^T where |v_n| = 1 and t = 1.
        Then h_eff = h_d*t + sum_n h_r_n^* g_n θ_n
                   = [diag(h_r^*) g; h_d]^H v
                   = a^H v
        
        So |h_eff|^2 = v^H (a a^H) v = v^H Q v.
        
        Returns:
            Q: (N+1) x (N+1) Hermitian PSD matrix
        """
        N = self.num_elements
        
        # a = [h_r^* ⊙ g; h_d]  (N+1,)
        # For single user: h_ris_user is (N,), h_bs_ris is (N,)
        if h_ris_user.ndim > 1:
            h_ris_user = h_ris_user[0]  # Take first user if multi-user
            
        a = np.zeros(N + 1, dtype=complex)
        a[:N] = np.conj(h_ris_user) * h_bs_ris
        
        if np.isscalar(h_direct):
            a[N] = h_direct
        else:
            a[N] = h_direct[0] if len(h_direct) > 0 else 0
            
        Q = np.outer(a, np.conj(a))
        return Q
    
    def optimize_phases(
        self,
        h_direct: np.ndarray,
        h_ris_user: np.ndarray,
        h_bs_ris: np.ndarray,
        noise_power: float,
        initial_phases: np.ndarray = None
    ) -> Dict:
        """
        Optimize RIS phase shifts using SDR.
        
        Args:
            h_direct: BS-User direct channel (complex scalar or vector)
            h_ris_user: RIS-User channel (complex vector, shape: [N] or [K, N])
            h_bs_ris: BS-RIS channel (complex vector, shape: [N])
            noise_power: Noise power (linear)
            initial_phases: Ignored (SDR finds global optimum)
            
        Returns:
            Dictionary with optimized phases and metrics
        """
        import time
        start_time = time.time()
        
        N = self.num_elements
        Q = self._construct_q_matrix(h_direct, h_ris_user, h_bs_ris)
        
        # SDR: max trace(Q @ V) s.t. V_nn = 1, V >> 0
        V = cp.Variable((N + 1, N + 1), hermitian=True)
        
        objective = cp.Maximize(cp.real(cp.trace(Q @ V)))
        constraints = [V >> 0]  # PSD constraint
        
        # Unit-modulus: diagonal elements = 1
        for n in range(N + 1):
            constraints.append(V[n, n] == 1)
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        except cp.SolverError:
            try:
                problem.solve(solver=cp.ECOS, verbose=False)
            except Exception:
                # Fallback: return random phases
                phases = np.random.uniform(0, 2 * np.pi, N)
                return self._build_result(phases, h_direct, h_ris_user, 
                                         h_bs_ris, noise_power, time.time() - start_time,
                                         converged=False)
        
        if V.value is None:
            phases = np.random.uniform(0, 2 * np.pi, N)
            return self._build_result(phases, h_direct, h_ris_user,
                                     h_bs_ris, noise_power, time.time() - start_time,
                                     converged=False)
        
        V_opt = V.value
        
        # Gaussian randomization to extract rank-1 solution
        best_phases = None
        best_snr = -np.inf
        
        # Eigendecomposition of V_opt
        eigvals, eigvecs = np.linalg.eigh(V_opt)
        eigvals = np.maximum(eigvals, 0)  # Ensure non-negative
        V_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
        
        for _ in range(self.num_randomizations):
            # Generate random vector
            r = (np.random.randn(N + 1) + 1j * np.random.randn(N + 1)) / np.sqrt(2)
            v = V_sqrt @ r
            
            # Project to unit modulus
            v = v / np.abs(v)
            
            # Extract phases (relative to last element which should be 1)
            theta = v[:N] / v[N]
            phases = np.angle(theta) % (2 * np.pi)
            
            # Compute SNR
            snr = self._compute_snr(phases, h_direct, h_ris_user, h_bs_ris, noise_power)
            
            if snr > best_snr:
                best_snr = snr
                best_phases = phases
        
        # Also try the rank-1 approximation (SVD)
        U, S, Vh = np.linalg.svd(V_opt)
        v_svd = U[:, 0] * np.sqrt(S[0])
        v_svd = v_svd / np.abs(v_svd)
        theta_svd = v_svd[:N] / v_svd[N]
        phases_svd = np.angle(theta_svd) % (2 * np.pi)
        snr_svd = self._compute_snr(phases_svd, h_direct, h_ris_user, h_bs_ris, noise_power)
        
        if snr_svd > best_snr:
            best_snr = snr_svd
            best_phases = phases_svd
        
        solve_time = time.time() - start_time
        self.solve_times.append(solve_time)
        
        # Upper bound from SDR relaxation
        sdr_upper_bound = 10 * np.log10(max(problem.value / noise_power, 1e-20))
        
        return self._build_result(best_phases, h_direct, h_ris_user,
                                 h_bs_ris, noise_power, solve_time,
                                 converged=True, sdr_upper_bound=sdr_upper_bound)
    
    def _compute_snr(self, phases, h_direct, h_ris_user, h_bs_ris, noise_power):
        """Compute SNR for given phases."""
        theta = np.exp(1j * phases)
        
        if h_ris_user.ndim > 1:
            h_ris_user_1 = h_ris_user[0]
        else:
            h_ris_user_1 = h_ris_user
            
        h_d = h_direct[0] if not np.isscalar(h_direct) else h_direct
        
        h_eff = h_d + np.dot(np.conj(h_ris_user_1) * h_bs_ris, theta)
        signal_power = np.abs(h_eff) ** 2
        return signal_power / noise_power
    
    def _build_result(self, phases, h_direct, h_ris_user, h_bs_ris,
                     noise_power, solve_time, converged=True, sdr_upper_bound=None):
        """Build result dictionary."""
        snr_linear = self._compute_snr(phases, h_direct, h_ris_user, h_bs_ris, noise_power)
        snr_db = 10 * np.log10(max(snr_linear, 1e-20))
        
        result = {
            'phases': phases,
            'snr_db': snr_db,
            'snr_linear': snr_linear,
            'solve_time': solve_time,
            'converged': converged,
            'num_elements': self.num_elements,
            'method': 'SDR',
        }
        if sdr_upper_bound is not None:
            result['sdr_upper_bound_db'] = sdr_upper_bound
        return result
    
    def batch_optimize(
        self,
        channel_samples: List[Dict],
        noise_power: float
    ) -> Dict:
        """
        Run SDR on multiple channel realizations.
        
        Args:
            channel_samples: List of channel dicts with keys:
                'h_direct', 'h_ris_user', 'h_bs_ris'
            noise_power: Noise power (linear)
            
        Returns:
            Aggregated metrics dictionary
        """
        snrs = []
        times = []
        upper_bounds = []
        
        for i, sample in enumerate(channel_samples):
            if self.verbose and i % 10 == 0:
                print(f"  SDR: Processing sample {i+1}/{len(channel_samples)}")
                
            result = self.optimize_phases(
                h_direct=sample['h_direct'],
                h_ris_user=sample['h_ris_user'],
                h_bs_ris=sample['h_bs_ris'],
                noise_power=noise_power
            )
            snrs.append(result['snr_db'])
            times.append(result['solve_time'])
            if 'sdr_upper_bound_db' in result:
                upper_bounds.append(result['sdr_upper_bound_db'])
        
        metrics = {
            'method': 'SDR',
            'avg_snr_db': float(np.mean(snrs)),
            'std_snr_db': float(np.std(snrs)),
            'median_snr_db': float(np.median(snrs)),
            'avg_solve_time': float(np.mean(times)),
            'total_time': float(np.sum(times)),
            'num_samples': len(channel_samples),
            'all_snrs': snrs,
        }
        if upper_bounds:
            metrics['avg_upper_bound_db'] = float(np.mean(upper_bounds))
            metrics['tightness_gap_db'] = float(np.mean(upper_bounds) - np.mean(snrs))
        
        return metrics
    
    def compute_complexity(self) -> Dict:
        """
        Estimate computational complexity.
        
        Returns:
            Dictionary with complexity metrics
        """
        N = self.num_elements
        return {
            'method': 'SDR',
            'sdp_complexity': f'O(N^3.5) = O({N**3.5:.0f})',
            'randomization_complexity': f'O({self.num_randomizations} * N^2)',
            'total_flops_estimate': N**3.5 + self.num_randomizations * N**2,
            'requires_centralized_csi': True,
            'online_capable': False,
        }
