"""
Realistic Channel Model for RIS Federated Learning

Implements:
- Rician fading with configurable K-factor (LoS/NLoS)
- Spatial correlation between adjacent RIS elements
- Multi-path channel with configurable number of paths
- CSI estimation error model
- Phase noise model
- DeepMIMO dataset integration

Reference channel model follows 3GPP TR 38.901 conventions for mmWave.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import os


# ============================================================================
# Phase Quantization Utilities
# ============================================================================

def quantize_phases(phases: np.ndarray, bits: int) -> tuple:
    """
    Quantize continuous phase shifts to discrete levels.
    
    Maps each phase to the nearest level in a uniform quantization grid:
        - 1-bit: {0, π}  (2 states)
        - 2-bit: {0, π/2, π, 3π/2}  (4 states)
        - 3-bit: {0, π/4, π/2, ..., 7π/4}  (8 states)
    
    Args:
        phases: Continuous phase shifts in [0, 2π], shape (N,)
        bits: Number of quantization bits (1, 2, or 3)
    
    Returns:
        quantized_phases: Quantized phases, same shape as input
        error_stats: Dict with quantization error statistics
    """
    if bits <= 0:
        return phases.copy(), {'mean_error_rad': 0.0, 'max_error_rad': 0.0, 'snr_loss_factor': 1.0}
    
    num_levels = 2 ** bits
    step = 2 * np.pi / num_levels
    levels = np.arange(num_levels) * step  # Quantization grid
    
    # Normalize to [0, 2π)
    phases_norm = np.mod(phases, 2 * np.pi)
    
    # Find nearest level for each phase
    # Compute circular distance to each level
    diffs = np.abs(phases_norm[:, None] - levels[None, :])
    diffs = np.minimum(diffs, 2 * np.pi - diffs)  # Circular wrap
    nearest_idx = np.argmin(diffs, axis=1)
    quantized = levels[nearest_idx]
    
    # Compute quantization error (circular)
    error = np.abs(phases_norm - quantized)
    error = np.minimum(error, 2 * np.pi - error)
    
    # SNR loss factor from quantization: |E[e^{j*error}]|^2
    # For uniform quantization: sinc(π/2^B)^2
    snr_loss_factor = (np.sinc(1.0 / num_levels)) ** 2
    
    error_stats = {
        'mean_error_rad': float(np.mean(error)),
        'max_error_rad': float(np.max(error)),
        'mean_error_deg': float(np.rad2deg(np.mean(error))),
        'max_error_deg': float(np.rad2deg(np.max(error))),
        'rmse_rad': float(np.sqrt(np.mean(error ** 2))),
        'num_levels': num_levels,
        'step_rad': step,
        'step_deg': float(np.rad2deg(step)),
        'snr_loss_factor': float(snr_loss_factor),
        'snr_loss_db': float(10 * np.log10(snr_loss_factor)) if snr_loss_factor > 0 else -np.inf,
    }
    
    return quantized, error_stats


# ============================================================================
# Spatial Correlation
# ============================================================================

def generate_spatial_correlation_matrix(
    num_elements: int,
    rho: float = 0.7,
    grid_rows: int = 8,
    grid_cols: int = 8
) -> np.ndarray:
    """
    Generate spatial correlation matrix for RIS elements using exponential model.
    
    Elements arranged in a 2D grid. Correlation between elements m and n:
        R[m,n] = rho^(d(m,n))
    where d(m,n) is the Manhattan distance on the grid.
    
    Args:
        num_elements: Total number of RIS elements
        rho: Correlation coefficient between adjacent elements (0-1)
        grid_rows: Number of rows in the element grid
        grid_cols: Number of columns in the element grid
    
    Returns:
        R: Spatial correlation matrix (num_elements x num_elements)
    """
    R = np.zeros((num_elements, num_elements))
    
    for m in range(num_elements):
        row_m, col_m = m // grid_cols, m % grid_cols
        for n in range(num_elements):
            row_n, col_n = n // grid_cols, n % grid_cols
            # Manhattan distance on 2D grid
            distance = abs(row_m - row_n) + abs(col_m - col_n)
            R[m, n] = rho ** distance
    
    return R


def apply_spatial_correlation(
    channel: np.ndarray,
    correlation_matrix: np.ndarray
) -> np.ndarray:
    """
    Apply spatial correlation to a channel vector.
    
    h_correlated = R^(1/2) @ h_uncorrelated
    
    Args:
        channel: Uncorrelated channel vector (num_elements,) complex
        correlation_matrix: Spatial correlation matrix (num_elements x num_elements)
    
    Returns:
        Spatially correlated channel vector
    """
    # Cholesky decomposition for matrix square root
    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # Add small regularization if not positive definite
        eps = 1e-6 * np.eye(correlation_matrix.shape[0])
        L = np.linalg.cholesky(correlation_matrix + eps)
    
    return L @ channel


# ============================================================================
# Rician Channel Generation
# ============================================================================

class RicianChannel:
    """
    Rician fading channel model for RIS systems.
    
    The channel consists of:
    - LoS (deterministic) component with free-space path loss
    - NLoS (scattered) component with Rayleigh fading
    - K-factor controls the ratio of LoS to NLoS power
    
    h = sqrt(K/(K+1)) * h_LoS + sqrt(1/(K+1)) * h_NLoS
    """
    
    def __init__(
        self,
        num_elements: int,
        k_factor_db: float = 10.0,
        num_paths: int = 5,
        frequency: float = 28e9,
        spatial_corr_rho: float = 0.7,
        grid_rows: int = 8,
        grid_cols: int = 8,
        path_loss_exponent: float = 2.5,
        element_spacing_factor: float = 0.5,  # in wavelengths
    ):
        """
        Args:
            num_elements: Number of RIS reflecting elements
            k_factor_db: Rician K-factor in dB (higher = stronger LoS)
            num_paths: Number of multi-path components for NLoS
            frequency: Operating frequency in Hz
            spatial_corr_rho: Spatial correlation coefficient (0-1)
            grid_rows: RIS grid rows
            grid_cols: RIS grid columns
            path_loss_exponent: Path loss exponent
            element_spacing_factor: Element spacing in wavelengths
        """
        self.num_elements = num_elements
        self.k_factor_db = k_factor_db
        self.k_factor_linear = 10 ** (k_factor_db / 10)
        self.num_paths = num_paths
        self.frequency = frequency
        self.wavelength = 3e8 / frequency
        self.spatial_corr_rho = spatial_corr_rho
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.path_loss_exponent = path_loss_exponent
        self.element_spacing = element_spacing_factor * self.wavelength
        
        # Pre-compute spatial correlation matrix
        self.R = generate_spatial_correlation_matrix(
            num_elements, spatial_corr_rho, grid_rows, grid_cols
        )
    
    def _compute_steering_vector(
        self,
        azimuth: float,
        elevation: float
    ) -> np.ndarray:
        """
        Compute array steering vector for a UPA (Uniform Planar Array).
        
        Args:
            azimuth: Azimuth angle in radians
            elevation: Elevation angle in radians
        
        Returns:
            Steering vector (num_elements,) complex
        """
        d = self.element_spacing
        k = 2 * np.pi / self.wavelength
        
        a = np.zeros(self.num_elements, dtype=complex)
        for idx in range(self.num_elements):
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            phase = k * d * (
                col * np.sin(azimuth) * np.cos(elevation) +
                row * np.sin(elevation)
            )
            a[idx] = np.exp(1j * phase)
        return a
    
    def _compute_path_loss(self, distance: float, override_exponent: float = None) -> float:
        """
        Compute free-space path loss.
        
        Args:
            distance: Distance in meters
            override_exponent: Optional custom path loss exponent
        
        Returns:
            Path loss (linear scale, < 1)
        """
        if distance < 0.1:
            distance = 0.1  # Minimum distance to avoid singularity
            
        exp = override_exponent if override_exponent is not None else self.path_loss_exponent
        pl = (self.wavelength / (4 * np.pi * distance)) ** exp
        return pl
    
    def generate_los_component(
        self,
        tx_pos: np.ndarray,
        rx_pos: np.ndarray,
        ris_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate LoS channel component for BS-RIS and RIS-User links.
        
        Args:
            tx_pos: BS position (3,)
            rx_pos: User position (3,)
            ris_pos: RIS center position (3,)
        
        Returns:
            h_bs_ris_los: BS-RIS LoS channel (num_elements,)
            h_ris_user_los: RIS-User LoS channel (num_elements,)
        """
        # BS -> RIS
        d_bs_ris = tx_pos - ris_pos
        dist_bs_ris = np.linalg.norm(d_bs_ris)
        az_bs = np.arctan2(d_bs_ris[1], d_bs_ris[0])
        el_bs = np.arcsin(np.clip(d_bs_ris[2] / max(dist_bs_ris, 1e-10), -1, 1))
        
        a_bs = self._compute_steering_vector(az_bs, el_bs)
        pl_bs_ris = self._compute_path_loss(dist_bs_ris)
        phase_bs_ris = -2 * np.pi * dist_bs_ris / self.wavelength
        h_bs_ris_los = np.sqrt(pl_bs_ris) * np.exp(1j * phase_bs_ris) * a_bs
        
        # RIS -> User
        d_ris_user = rx_pos - ris_pos
        dist_ris_user = np.linalg.norm(d_ris_user)
        az_user = np.arctan2(d_ris_user[1], d_ris_user[0])
        el_user = np.arcsin(np.clip(d_ris_user[2] / max(dist_ris_user, 1e-10), -1, 1))
        
        a_user = self._compute_steering_vector(az_user, el_user)
        pl_ris_user = self._compute_path_loss(dist_ris_user)
        phase_ris_user = -2 * np.pi * dist_ris_user / self.wavelength
        h_ris_user_los = np.sqrt(pl_ris_user) * np.exp(1j * phase_ris_user) * a_user
        
        return h_bs_ris_los, h_ris_user_los
    
    def generate_nlos_component(
        self,
        tx_pos: np.ndarray,
        rx_pos: np.ndarray,
        ris_pos: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate NLoS (scattered) channel components with multiple paths.
        
        Each path has random AoA/AoD and complex gain.
        
        Args:
            tx_pos: BS position (3,)
            rx_pos: User position (3,)
            ris_pos: RIS center position (3,)
        
        Returns:
            h_bs_ris_nlos: BS-RIS NLoS channel (num_elements,)
            h_ris_user_nlos: RIS-User NLoS channel (num_elements,)
        """
        dist_bs_ris = np.linalg.norm(tx_pos - ris_pos)
        dist_ris_user = np.linalg.norm(rx_pos - ris_pos)
        
        h_bs_ris_nlos = np.zeros(self.num_elements, dtype=complex)
        h_ris_user_nlos = np.zeros(self.num_elements, dtype=complex)
        
        for p in range(self.num_paths):
            # Random AoA/AoD for each path
            az_bs = np.random.uniform(-np.pi, np.pi)
            el_bs = np.random.uniform(-np.pi / 4, np.pi / 4)
            az_user = np.random.uniform(-np.pi, np.pi)
            el_user = np.random.uniform(-np.pi / 4, np.pi / 4)
            
            # Path gain (exponentially decaying with path index)
            gain_bs = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            gain_bs *= np.exp(-0.5 * p)  # Later paths are weaker
            
            gain_user = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            gain_user *= np.exp(-0.5 * p)
            
            a_bs = self._compute_steering_vector(az_bs, el_bs)
            a_user = self._compute_steering_vector(az_user, el_user)
            
            h_bs_ris_nlos += gain_bs * a_bs
            h_ris_user_nlos += gain_user * a_user
        
        # Normalize NLoS power
        pl_bs_ris = self._compute_path_loss(dist_bs_ris)
        pl_ris_user = self._compute_path_loss(dist_ris_user)
        
        h_bs_ris_nlos *= np.sqrt(pl_bs_ris) / np.sqrt(self.num_paths)
        h_ris_user_nlos *= np.sqrt(pl_ris_user) / np.sqrt(self.num_paths)
        
        return h_bs_ris_nlos, h_ris_user_nlos
    
    def generate_channel(
        self,
        tx_pos: np.ndarray,
        rx_pos: np.ndarray,
        ris_pos: np.ndarray,
        scenario: str = "LoS"
    ) -> Dict[str, np.ndarray]:
        """
        Generate complete realistic RIS channel.
        
        Combines Rician fading (LoS + NLoS) with spatial correlation.
        
        Args:
            tx_pos: BS position (3,)
            rx_pos: User position(s) - single (3,) or multiple (num_users, 3)
            ris_pos: RIS center position (3,)
            scenario: "LoS", "NLoS", or "mixed"
        
        Returns:
            Dictionary with channel components:
            - h_direct: Direct BS-User channel (num_users,) or scalar
            - h_bs_ris: BS-RIS channel (num_elements,)
            - h_ris_user: RIS-User channel per user (num_users, num_elements)
        """
        if rx_pos.ndim == 1:
            rx_pos = rx_pos.reshape(1, 3)
        
        num_users = rx_pos.shape[0]
        k = self.k_factor_linear
        
        # Determine effective K-factor based on scenario
        if scenario == "NLoS":
            k_eff = 0.0  # Pure Rayleigh
        elif scenario == "mixed":
            k_eff = k * 0.5  # Reduced LoS
        else:  # "LoS"
            k_eff = k
        
        # ---- Direct channel (BS -> User) ----
        h_direct = np.zeros(num_users, dtype=complex)
        for u in range(num_users):
            dist = np.linalg.norm(tx_pos - rx_pos[u])
            # Direct link is typically blocked when RIS is needed, so use higher path loss exponent (e.g., 3.5)
            pl = self._compute_path_loss(dist, override_exponent=3.5)
            phase = -2 * np.pi * dist / self.wavelength
            
            h_los = np.sqrt(pl) * np.exp(1j * phase)
            h_nlos = np.sqrt(pl) * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
            
            if k_eff > 0:
                h_direct[u] = np.sqrt(k_eff / (k_eff + 1)) * h_los + \
                              np.sqrt(1 / (k_eff + 1)) * h_nlos
            else:
                h_direct[u] = h_nlos
        
        # ---- BS -> RIS channel ----
        h_bs_ris_los, _ = self.generate_los_component(tx_pos, rx_pos[0], ris_pos)
        h_bs_ris_nlos, _ = self.generate_nlos_component(tx_pos, rx_pos[0], ris_pos)
        
        if k_eff > 0:
            h_bs_ris = np.sqrt(k_eff / (k_eff + 1)) * h_bs_ris_los + \
                       np.sqrt(1 / (k_eff + 1)) * h_bs_ris_nlos
        else:
            h_bs_ris = h_bs_ris_nlos
        
        # Apply spatial correlation to BS-RIS channel
        h_bs_ris = apply_spatial_correlation(h_bs_ris, self.R)
        
        # ---- RIS -> User channels ----
        h_ris_user = np.zeros((num_users, self.num_elements), dtype=complex)
        for u in range(num_users):
            _, h_ris_u_los = self.generate_los_component(tx_pos, rx_pos[u], ris_pos)
            _, h_ris_u_nlos = self.generate_nlos_component(tx_pos, rx_pos[u], ris_pos)
            
            if k_eff > 0:
                h_ris_u = np.sqrt(k_eff / (k_eff + 1)) * h_ris_u_los + \
                          np.sqrt(1 / (k_eff + 1)) * h_ris_u_nlos
            else:
                h_ris_u = h_ris_u_nlos
            
            # Apply spatial correlation
            h_ris_user[u] = apply_spatial_correlation(h_ris_u, self.R)
        
        return {
            'h_direct': h_direct,
            'h_bs_ris': h_bs_ris,
            'h_ris_user': h_ris_user,
            'tx_pos': tx_pos,
            'rx_pos': rx_pos,
            'ris_pos': ris_pos,
            'scenario': scenario,
            'k_factor_db': self.k_factor_db,
        }


# ============================================================================
# CSI Estimation Error
# ============================================================================

def apply_csi_error(
    channel: np.ndarray,
    error_variance: float = 0.01
) -> np.ndarray:
    """
    Add CSI estimation error to channel.
    
    h_estimated = h_true + N(0, sigma^2_e)
    
    Args:
        channel: True channel (complex array, any shape)
        error_variance: Variance of estimation error
    
    Returns:
        Estimated channel with added error
    """
    if error_variance <= 0:
        return channel
    
    noise = np.sqrt(error_variance / 2) * (
        np.random.randn(*channel.shape) + 1j * np.random.randn(*channel.shape)
    )
    return channel + noise


# ============================================================================
# Phase Noise
# ============================================================================

def apply_phase_noise(
    phases: np.ndarray,
    noise_std_deg: float = 5.0
) -> np.ndarray:
    """
    Add phase noise to commanded phase shifts.
    
    θ_actual = θ_commanded + δ, where δ ~ N(0, σ²_phase)
    
    Args:
        phases: Commanded phase shifts in radians
        noise_std_deg: Standard deviation of phase noise in degrees
    
    Returns:
        Actual phase shifts with noise (wrapped to [0, 2π])
    """
    if noise_std_deg <= 0:
        return phases
    
    noise_std_rad = np.deg2rad(noise_std_deg)
    noise = np.random.normal(0, noise_std_rad, phases.shape)
    noisy_phases = phases + noise
    
    return np.mod(noisy_phases, 2 * np.pi)


# ============================================================================
# Phase Quantization
# ============================================================================

def quantize_phases(
    phases: np.ndarray,
    num_bits: int
) -> np.ndarray:
    """
    Quantize continuous phase shifts to discrete levels.
    
    Args:
        phases: Continuous phase shifts in radians [0, 2π]
        num_bits: Number of quantization bits
            1-bit: {0, π}
            2-bit: {0, π/2, π, 3π/2}
            3-bit: {0, π/4, π/2, ..., 7π/4}
    
    Returns:
        Quantized phase shifts
    """
    if num_bits <= 0:
        return phases  # Continuous (no quantization)
    
    num_levels = 2 ** num_bits
    step = 2 * np.pi / num_levels
    quantized = np.round(phases / step) * step
    return np.mod(quantized, 2 * np.pi)


# ============================================================================
# 3GPP TR 38.901 Urban Micro (UMi) Channel Model
# ============================================================================

class ThreeGPPUMiChannel:
    """
    3GPP TR 38.901 Urban Micro (UMi) street canyon channel model.
    
    Implements:
    - Distance-dependent LoS probability
    - Dual-slope path loss (LoS and NLoS)
    - Log-normal shadow fading
    - Spatial consistency via correlated large-scale parameters
    
    Supports frequencies from 0.5 GHz to 100 GHz.
    
    Reference: 3GPP TR 38.901 V16.1.0, Table 7.4.1-1
    """
    
    def __init__(
        self,
        num_elements: int,
        frequency: float = 28e9,
        grid_rows: int = 8,
        grid_cols: int = 8,
        element_spacing_factor: float = 0.5,
        bs_height: float = 10.0,
        ue_height: float = 1.5,
    ):
        """
        Args:
            num_elements: Number of RIS elements
            frequency: Carrier frequency in Hz (0.5-100 GHz)
            grid_rows: RIS element grid rows
            grid_cols: RIS element grid columns
            element_spacing_factor: Element spacing in wavelengths
            bs_height: BS antenna height in meters
            ue_height: UE height in meters
        """
        self.num_elements = num_elements
        self.frequency = frequency
        self.frequency_ghz = frequency / 1e9
        self.wavelength = 3e8 / frequency
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.element_spacing = element_spacing_factor * self.wavelength
        self.bs_height = bs_height
        self.ue_height = ue_height
        
        # Shadow fading std (dB) from 3GPP Table 7.4.1-1
        self.sf_std_los = 4.0
        self.sf_std_nlos = 7.82
        
        # Spatial correlation
        self.R = generate_spatial_correlation_matrix(
            num_elements, 0.7, grid_rows, grid_cols
        )
    
    def los_probability(self, distance_2d: float) -> float:
        """
        LoS probability per 3GPP TR 38.901 Table 7.4.2-1 (UMi-Street Canyon).
        
        P_LoS(d) = min(18/d, 1) * (1 - exp(-d/36)) + exp(-d/36)
        """
        if distance_2d <= 0:
            return 1.0
        p = min(18.0 / distance_2d, 1.0) * (1 - np.exp(-distance_2d / 36.0)) + \
            np.exp(-distance_2d / 36.0)
        return float(np.clip(p, 0, 1))
    
    def path_loss_los(self, distance_3d: float, distance_2d: float) -> float:
        """
        LoS path loss (dB) per 3GPP TR 38.901 Table 7.4.1-1.
        
        PL_UMi-LOS = 32.4 + 21 log10(d_3D) + 20 log10(f_c [GHz])
        Valid for 10 m ≤ d_2D ≤ 5 km
        """
        d3d = max(distance_3d, 1.0)
        fc = self.frequency_ghz
        
        pl = 32.4 + 21.0 * np.log10(d3d) + 20.0 * np.log10(fc)
        
        # Add shadow fading
        sf = np.random.normal(0, self.sf_std_los)
        
        return pl + sf
    
    def path_loss_nlos(self, distance_3d: float, distance_2d: float) -> float:
        """
        NLoS path loss (dB) per 3GPP TR 38.901 Table 7.4.1-1.
        
        PL_UMi-NLOS = 32.4 + 31.9 log10(d_3D) + 20 log10(f_c [GHz])
        """
        d3d = max(distance_3d, 1.0)
        fc = self.frequency_ghz
        
        pl_nlos = 32.4 + 31.9 * np.log10(d3d) + 20.0 * np.log10(fc)
        
        # Take max with LoS path loss (3GPP requirement)
        pl_los = 32.4 + 21.0 * np.log10(d3d) + 20.0 * np.log10(fc)
        pl = max(pl_nlos, pl_los)
        
        # Add shadow fading
        sf = np.random.normal(0, self.sf_std_nlos)
        
        return pl + sf
    
    def _compute_steering_vector(
        self, azimuth: float, elevation: float
    ) -> np.ndarray:
        """Compute UPA steering vector."""
        d = self.element_spacing
        k = 2 * np.pi / self.wavelength
        
        a = np.zeros(self.num_elements, dtype=complex)
        for idx in range(self.num_elements):
            row = idx // self.grid_cols
            col = idx % self.grid_cols
            phase = k * d * (
                col * np.sin(azimuth) * np.cos(elevation) +
                row * np.sin(elevation)
            )
            a[idx] = np.exp(1j * phase)
        
        return a / np.sqrt(self.num_elements)
    
    def generate_channel(
        self,
        tx_pos: np.ndarray,
        rx_pos: np.ndarray,
        ris_pos: np.ndarray,
        scenario: str = "LoS",
        num_clusters: int = 12,
        rays_per_cluster: int = 20,
    ) -> Dict[str, np.ndarray]:
        """
        Generate 3GPP UMi channel realization.
        
        Args:
            tx_pos: BS position (3,)
            rx_pos: User position(s) (3,) or (K, 3)
            ris_pos: RIS position (3,)
            scenario: "LoS", "NLoS", or "mixed" (probabilistic)
            num_clusters: Number of multipath clusters
            rays_per_cluster: Rays per cluster
            
        Returns:
            Channel dictionary matching RicianChannel interface
        """
        if rx_pos.ndim == 1:
            rx_pos = rx_pos.reshape(1, 3)
        
        num_users = rx_pos.shape[0]
        
        # Determine LoS/NLoS per user
        h_direct = np.zeros(num_users, dtype=complex)
        h_ris_user = np.zeros((num_users, self.num_elements), dtype=complex)
        
        for u in range(num_users):
            # Direct BS-User
            d_direct = tx_pos - rx_pos[u]
            dist_3d_direct = np.linalg.norm(d_direct)
            dist_2d_direct = np.linalg.norm(d_direct[:2])
            
            # Determine LoS
            if scenario == "mixed":
                is_los = np.random.rand() < self.los_probability(dist_2d_direct)
            elif scenario == "LoS":
                is_los = True
            else:
                is_los = False
            
            # Path loss
            if is_los:
                pl_db = self.path_loss_los(dist_3d_direct, dist_2d_direct)
            else:
                pl_db = self.path_loss_nlos(dist_3d_direct, dist_2d_direct)
            
            pl_linear = 10 ** (-pl_db / 20)  # Voltage domain
            phase_direct = -2 * np.pi * dist_3d_direct / self.wavelength
            h_direct[u] = pl_linear * np.exp(1j * phase_direct)
            
            # RIS-User channel
            d_ris_user = rx_pos[u] - ris_pos
            dist_3d_ru = np.linalg.norm(d_ris_user)
            dist_2d_ru = np.linalg.norm(d_ris_user[:2])
            
            if is_los:
                pl_ru_db = self.path_loss_los(dist_3d_ru, dist_2d_ru)
            else:
                pl_ru_db = self.path_loss_nlos(dist_3d_ru, dist_2d_ru)
            
            pl_ru = 10 ** (-pl_ru_db / 20)
            
            # LoS component
            az_ru = np.arctan2(d_ris_user[1], d_ris_user[0])
            el_ru = np.arcsin(np.clip(d_ris_user[2] / max(dist_3d_ru, 1e-10), -1, 1))
            a_ru = self._compute_steering_vector(az_ru, el_ru)
            
            h_los = pl_ru * np.exp(-1j * 2 * np.pi * dist_3d_ru / self.wavelength) * a_ru
            
            # NLoS clusters
            h_nlos = np.zeros(self.num_elements, dtype=complex)
            cluster_powers = np.exp(-np.arange(num_clusters) * 0.3)
            cluster_powers /= np.sum(cluster_powers)
            
            for c in range(num_clusters):
                az_c = np.random.uniform(-np.pi, np.pi)
                el_c = np.random.uniform(-np.pi / 6, np.pi / 6)
                a_c = self._compute_steering_vector(az_c, el_c)
                gain_c = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                h_nlos += np.sqrt(cluster_powers[c]) * gain_c * a_c
            
            h_nlos *= pl_ru
            
            if is_los:
                k_factor = 9.0  # Rician K for UMi LoS (dB)
                k_lin = 10 ** (k_factor / 10)
                h_ris_user[u] = np.sqrt(k_lin / (k_lin + 1)) * h_los + \
                                np.sqrt(1 / (k_lin + 1)) * h_nlos
            else:
                h_ris_user[u] = h_nlos
            
            # Apply spatial correlation
            h_ris_user[u] = apply_spatial_correlation(h_ris_user[u], self.R)
        
        # BS-RIS channel
        d_bs_ris = tx_pos - ris_pos
        dist_3d_br = np.linalg.norm(d_bs_ris)
        dist_2d_br = np.linalg.norm(d_bs_ris[:2])
        pl_br_db = self.path_loss_los(dist_3d_br, dist_2d_br)  # BS-RIS usually LoS
        pl_br = 10 ** (-pl_br_db / 20)
        
        az_br = np.arctan2(d_bs_ris[1], d_bs_ris[0])
        el_br = np.arcsin(np.clip(d_bs_ris[2] / max(dist_3d_br, 1e-10), -1, 1))
        a_br = self._compute_steering_vector(az_br, el_br)
        h_bs_ris = pl_br * np.exp(-1j * 2 * np.pi * dist_3d_br / self.wavelength) * a_br
        h_bs_ris = apply_spatial_correlation(h_bs_ris, self.R)
        
        return {
            'h_direct': h_direct,
            'h_bs_ris': h_bs_ris,
            'h_ris_user': h_ris_user,
            'tx_pos': tx_pos,
            'rx_pos': rx_pos,
            'ris_pos': ris_pos,
            'scenario': scenario,
            'channel_model': '3GPP_UMi',
            'frequency_ghz': self.frequency_ghz,
        }


# ============================================================================
# DeepMIMO Integration
# ============================================================================

class DeepMIMODatasetLoader:
    """
    Wrapper for DeepMIMO dataset generation and loading.
    
    Requires: pip install DeepMIMO
    
    Usage:
        loader = DeepMIMODatasetLoader(scenario='O1_28')
        channels = loader.generate_ris_channels(num_samples=2000, num_ris_elements=64)
    """
    
    def __init__(
        self,
        scenario: str = 'O1_28',
        data_dir: str = 'data/deepmimo',
        active_bs: List[int] = None,
        num_paths: int = 5,
        frequency_band: int = 0,
    ):
        """
        Args:
            scenario: DeepMIMO scenario name (e.g., 'O1_28' for outdoor 28GHz)
            data_dir: Directory containing downloaded scenario data
                      (scenario subfolder must exist: data_dir/scenario/)
            active_bs: List of active base station indices (default: [1])
            num_paths: Number of channel paths to use
            frequency_band: Subband index (0 for first subband)
        """
        self.scenario = scenario
        self.data_dir = data_dir
        self.active_bs = active_bs or [1]  # DeepMIMOv3 is 1-indexed
        self.num_paths = num_paths
        self.frequency_band = frequency_band

        # Try importing local DeepMIMOv3 package first, then installed package
        try:
            import sys
            # Add parent dir so local DeepMIMOv3/ folder is importable
            parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if parent not in sys.path:
                sys.path.insert(0, parent)
            import DeepMIMOv3 as DeepMIMO
            self.dm = DeepMIMO
            self.is_installed = True
        except ImportError:
            print("WARNING: DeepMIMOv3 not found. Place the DeepMIMOv3 folder in the project root.")
            self.is_installed = False

    @property
    def is_available(self) -> bool:
        return self.is_installed

    def _check_scenario_present(self):
        """Raise a clear error if scenario data files are missing."""
        if not self.is_installed:
            raise RuntimeError("DeepMIMOv3 package not found.")
        scenario_path = os.path.join(os.path.abspath(self.data_dir), self.scenario)
        if not os.path.exists(scenario_path):
            raise RuntimeError(
                f"DeepMIMO scenario data not found at: {scenario_path}\n"
                f"Download the '{self.scenario}' scenario from https://deepmimo.net/ "
                f"and place it inside '{self.data_dir}/'."
            )

    def generate_dataset(
        self,
        num_user_rows: Tuple[int, int] = (1, 100),
        num_antennas_bs: Tuple[int, ...] = (1, 1, 1),
        num_antennas_user: Tuple[int, ...] = (1, 1, 1),
    ) -> dict:
        """
        Generate DeepMIMO dataset with specified parameters.

        Args:
            num_user_rows: (first_row, last_row) of user grid to activate (1-indexed)
            num_antennas_bs: antenna shape at BS (e.g. (1,1,1) for single antenna)
            num_antennas_user: antenna shape at UE

        Returns:
            DeepMIMO dataset list (one entry per active BS)
        """
        self._check_scenario_present()

        # Build parameter dict using DeepMIMOv3 defaults + overrides
        dataset_params = self.dm.default_params()
        dataset_params['scenario'] = self.scenario
        dataset_params['dataset_folder'] = os.path.abspath(self.data_dir)
        dataset_params['active_BS'] = np.array(self.active_bs)
        dataset_params['user_rows'] = np.arange(num_user_rows[0], num_user_rows[1] + 1)
        dataset_params['num_paths'] = self.num_paths
        # DeepMIMOv3 uses 'bs_antenna' and 'ue_antenna' (lowercase)
        dataset_params['bs_antenna']['shape'] = np.array(num_antennas_bs)
        dataset_params['ue_antenna']['shape'] = np.array(num_antennas_user)

        dataset = self.dm.generate_data(dataset_params)
        return dataset

    
    def generate_ris_channels(
        self,
        num_samples: int = 2000,
        num_ris_elements: int = 64,
        num_users: int = 4,
        ris_grid_rows: int = 8,
        ris_grid_cols: int = 8,
    ) -> List[Dict]:
        """
        Generate RIS channel samples from DeepMIMO data.
        
        Maps DeepMIMO BS-User channels to BS-RIS-User cascaded channels
        by treating the RIS as an intermediary with spatial response.
        
        Args:
            num_samples: Number of channel samples to generate
            num_ris_elements: Number of RIS elements
            num_users: Number of users per sample
            ris_grid_rows: RIS grid rows
            ris_grid_cols: RIS grid columns
        
        Returns:
            List of channel sample dictionaries
        """
        if not self.is_installed:
            raise RuntimeError("DeepMIMOv3 not found. Use synthetic channel model.")
        
        dataset = self.generate_dataset(
            num_user_rows=(1, max(502, num_samples // 10 + 2))
        )
        
        channels = []
        bs_data = dataset[0]  # First active BS
        total_users = len(bs_data['user']['channel'])
        
        for i in range(num_samples):
            # Sample random users
            user_indices = np.random.choice(total_users, num_users, replace=False)
            
            # Extract channels
            h_direct = np.zeros(num_users, dtype=complex)
            h_ris_user = np.zeros((num_users, num_ris_elements), dtype=complex)
            
            for u, uid in enumerate(user_indices):
                user_channel = bs_data['user']['channel'][uid]
                
                # Direct channel: use first antenna element
                if user_channel.ndim >= 2:
                    h_direct[u] = user_channel.flatten()[0]
                else:
                    h_direct[u] = user_channel[0] if len(user_channel) > 0 else 0
                
                # RIS channel: map channel to RIS elements
                # Use subcarrier channels as RIS element responses
                ch_flat = user_channel.flatten()
                if len(ch_flat) >= num_ris_elements:
                    h_ris_user[u] = ch_flat[:num_ris_elements]
                else:
                    # Repeat and tile to fill all elements
                    repeats = int(np.ceil(num_ris_elements / max(len(ch_flat), 1)))
                    h_ris_user[u] = np.tile(ch_flat, repeats)[:num_ris_elements]
            
            # BS-RIS channel (use first user's reverse channel as proxy)
            h_bs_ris = h_ris_user[0].copy()
            
            # User locations
            user_locs = np.array([
                bs_data['user']['location'][uid] for uid in user_indices
            ])
            
            channels.append({
                'h_direct': h_direct,
                'h_bs_ris': h_bs_ris,
                'h_ris_user': h_ris_user,
                'user_positions': user_locs,
                'user_indices': user_indices,
                'source': 'DeepMIMO',
                'scenario': self.scenario,
            })
        
        return channels


# ============================================================================
# Unified Channel Generator
# ============================================================================

def generate_ris_channel_dataset(
    num_samples: int,
    num_ris_elements: int,
    num_users: int,
    room_size: Tuple[float, float, float],
    frequency: float = 28e9,
    tile_position: Optional[np.ndarray] = None,
    non_iid_bias: Optional[Tuple[float, float]] = None,
    k_factor_db: float = 10.0,
    num_paths: int = 5,
    spatial_corr_rho: float = 0.7,
    scenario: str = "LoS",
    csi_error_variance: float = 0.0,
    grid_rows: int = 8,
    grid_cols: int = 8,
    use_deepmimo: bool = False,
    deepmimo_scenario: str = 'O1_28',
    deepmimo_data_dir: str = 'data/deepmimo',
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Generate a complete RIS channel dataset with realistic models.
    
    Can use either synthetic Rician channel or DeepMIMO data.
    
    Args:
        num_samples: Number of samples to generate
        num_ris_elements: Number of RIS elements
        num_users: Number of users
        room_size: Environment dimensions (x, y, z) in meters
        frequency: Operating frequency in Hz
        tile_position: Position of this RIS tile (3,) or None for default
        non_iid_bias: Spatial bias for non-IID (bias_x, bias_y) or None
        k_factor_db: Rician K-factor in dB
        num_paths: Number of multi-path components
        spatial_corr_rho: Spatial correlation coefficient
        scenario: "LoS", "NLoS", or "mixed"
        csi_error_variance: CSI estimation error variance (0 = perfect CSI)
        grid_rows: RIS element grid rows
        grid_cols: RIS element grid columns
        use_deepmimo: Whether to use DeepMIMO data
        deepmimo_scenario: DeepMIMO scenario name
        deepmimo_data_dir: Path for DeepMIMO data files
    
    Returns:
        features: Input features array (num_samples, feature_dim)
        labels: Optimal phase shifts (num_samples, num_elements)
        metadata: List of dicts with channel details per sample
    """
    
    # ---- Try DeepMIMO first if requested ----
    if use_deepmimo:
        loader = DeepMIMODatasetLoader(
            scenario=deepmimo_scenario,
            data_dir=deepmimo_data_dir,
        )
        if loader.is_available:
            try:
                dm_channels = loader.generate_ris_channels(
                    num_samples=num_samples,
                    num_ris_elements=num_ris_elements,
                    num_users=num_users,
                    ris_grid_rows=grid_rows,
                    ris_grid_cols=grid_cols,
                )
                return _channels_to_dataset(dm_channels, num_ris_elements, csi_error_variance)
            except Exception as e:
                print(f"DeepMIMO generation failed: {e}")
                print("Falling back to synthetic Rician channel model.")
    
    # ---- Synthetic Rician channel ----
    channel_model = RicianChannel(
        num_elements=num_ris_elements,
        k_factor_db=k_factor_db,
        num_paths=num_paths,
        frequency=frequency,
        spatial_corr_rho=spatial_corr_rho,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    
    # Positions
    bs_position = np.array([room_size[0] / 2, room_size[1], room_size[2] / 2])
    
    if tile_position is not None:
        ris_position = np.array(tile_position)
    else:
        ris_position = np.array([room_size[0] / 2, 0, room_size[2] / 2])
    
    all_channels = []
    
    for i in range(num_samples):
        # Generate user positions
        user_pos = np.random.uniform(
            low=[0, 0, 0.5],
            high=room_size,
            size=(num_users, 3)
        )
        
        # Apply non-IID bias
        if non_iid_bias is not None:
            bias_x, bias_y = non_iid_bias
            user_pos[:, 0] = np.clip(user_pos[:, 0] + bias_x, 0, room_size[0])
            user_pos[:, 1] = np.clip(user_pos[:, 1] + bias_y, 0, room_size[1])
        
        # Generate channel
        ch = channel_model.generate_channel(
            tx_pos=bs_position,
            rx_pos=user_pos,
            ris_pos=ris_position,
            scenario=scenario,
        )
        ch['bs_position'] = bs_position
        ch['user_positions'] = user_pos
        
        all_channels.append(ch)
    
    return _channels_to_dataset(all_channels, num_ris_elements, csi_error_variance)


def _channels_to_dataset(
    channels: List[Dict],
    num_ris_elements: int,
    csi_error_variance: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Convert channel dictionaries to features/labels arrays.
    
    Features include: user positions (normalized), channel magnitudes, channel phases
    Labels: optimal phase shifts [0, 2π]
    """
    features_list = []
    labels_list = []
    metadata_list = []
    
    for ch in channels:
        h_direct = ch['h_direct']
        h_ris_user = ch['h_ris_user']
        h_bs_ris = ch.get('h_bs_ris', h_ris_user[0])
        
        # Apply CSI error if configured
        if csi_error_variance > 0:
            h_direct_est = apply_csi_error(h_direct, csi_error_variance)
            h_ris_user_est = apply_csi_error(h_ris_user, csi_error_variance)
            h_bs_ris_est = apply_csi_error(h_bs_ris, csi_error_variance)
        else:
            h_direct_est = h_direct
            h_ris_user_est = h_ris_user
            h_bs_ris_est = h_bs_ris
        
        # Compute optimal phase shifts (MRC-based)
        # For first user (extend to multi-user as needed)
        # θ_n = ∠(h_direct) − ∠(h_cascade_n) aligns all reflected paths with direct path
        target_user = 0
        h_cascade = h_ris_user_est[target_user] * h_bs_ris_est
        optimal_phases = np.angle(h_direct_est[target_user]) - np.angle(h_cascade)
        optimal_phases = np.mod(optimal_phases, 2 * np.pi)
        
        # Build feature vector
        features = []
        
        # User positions (normalized to [0, 1])
        user_pos = ch.get('user_positions', ch.get('rx_pos', np.zeros((4, 3))))
        if hasattr(user_pos, 'flatten'):
            features.extend(user_pos.flatten() / 10.0)  # Normalize
        
        # Channel magnitudes (estimated)
        # BUG FIX: Scale up microscopic channel magnitudes to O(1) for neural network stability
        scale = 1e5
        
        # BUG FIX: Use full cascaded channel so network knows BS-RIS phase
        h_cascade_all = h_ris_user_est * h_bs_ris_est
        
        # BUG FIX: Use continuous Real/Imaginary components instead of Abs/Angle 
        # to prevent discontinuous Phase Wrapping gradient failures.
        features.extend(h_direct_est.real.flatten() * scale)
        features.extend(h_cascade_all.real.flatten() * scale)
        
        features.extend(h_direct_est.imag.flatten() * scale)
        features.extend(h_cascade_all.imag.flatten() * scale)
        
        features = np.array(features, dtype=np.float32)
        labels = optimal_phases.astype(np.float32)
        
        features_list.append(features)
        labels_list.append(labels)
        
        # Metadata
        metadata_list.append({
            'H_direct': h_direct,
            'H_ris': h_ris_user,
            'h_bs_ris': h_bs_ris,
            'user_positions': user_pos,
            'bs_position': ch.get('bs_position', np.zeros(3)),
            'ris_position': ch.get('ris_pos', np.zeros(3)),
            'scenario': ch.get('scenario', 'unknown'),
        })
    
    return np.array(features_list), np.array(labels_list), metadata_list
