"""
Random Search Baseline for RIS Phase Configuration

Simple baseline that tries random phase configurations and returns the best.
Serves as a sanity check - any intelligent method should beat this.
"""

import numpy as np
from typing import Dict, Tuple, List


class RandomSearch:
    """
    Random search baseline for RIS optimization.
    
    Tries num_trials random phase configurations and returns the one
    with the highest SNR.
    """
    
    def __init__(self, num_elements: int, num_trials: int = 1000, seed: int = None):
        """
        Args:
            num_elements: Number of RIS elements
            num_trials: Number of random configurations to try
            seed: Random seed for reproducibility
        """
        self.num_elements = num_elements
        self.num_trials = num_trials
        if seed is not None:
            np.random.seed(seed)
    
    def optimize_phases(
        self,
        h_direct: np.ndarray,
        h_ris_user: np.ndarray,
        h_bs_ris: np.ndarray,
        noise_power: float
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Find best phase configuration via random search.
        
        Args:
            h_direct: BS-User direct channel
            h_ris_user: RIS-User channel (complex vector, shape: [N])
            h_bs_ris: BS-RIS channel (complex vector, shape: [N])
            noise_power: Noise power
            
        Returns:
            best_phases: Best phase configuration found
            best_snr_db: SNR achieved with best phases
            all_snrs: SNR values for all trials (for analysis)
        """
        best_snr = -np.inf
        best_phases = None
        all_snrs = []
        
        for trial in range(self.num_trials):
            # Generate random phases
            phases = np.random.uniform(0, 2 * np.pi, self.num_elements)
            
            # Compute SNR with these phases
            snr = self._compute_snr(phases, h_direct, h_ris_user, h_bs_ris, noise_power)
            all_snrs.append(snr)
            
            # Track best
            if snr > best_snr:
                best_snr = snr
                best_phases = phases.copy()
        
        best_snr_db = 10 * np.log10(best_snr + 1e-10)
        
        return best_phases, best_snr_db, all_snrs
    
    def _compute_snr(
        self,
        phases: np.ndarray,
        h_direct: np.ndarray,
        h_ris_user: np.ndarray,
        h_bs_ris: np.ndarray,
        noise_power: float
    ) -> float:
        """Compute SNR for given phase configuration."""
        Theta = np.diag(np.exp(1j * phases))
        h_cascade = np.conj(h_ris_user) @ Theta @ h_bs_ris
        h_eff = h_direct + h_cascade
        signal_power = np.abs(h_eff) ** 2
        snr_linear = signal_power / noise_power
        return snr_linear
    
    def batch_optimize(
        self,
        channel_samples: List[Dict],
        noise_power: float
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run random search on multiple channel realizations.
        
        Args:
            channel_samples: List of channel sample dicts
            noise_power: Noise power
            
        Returns:
            all_phases: Array of best phases for each sample
            metrics: Performance metrics
        """
        num_samples = len(channel_samples)
        all_phases = np.zeros((num_samples, self.num_elements))
        all_snrs = []
        
        for i, sample in enumerate(channel_samples):
            phases, snr_db, _ = self.optimize_phases(
                h_direct=sample['h_direct'],
                h_ris_user=sample['h_ris_user'],
                h_bs_ris=sample['h_bs_ris'],
                noise_power=noise_power
            )
            all_phases[i] = phases
            all_snrs.append(snr_db)
        
        metrics = {
            'avg_snr_db': np.mean(all_snrs),
            'std_snr_db': np.std(all_snrs),
            'min_snr_db': np.min(all_snrs),
            'max_snr_db': np.max(all_snrs),
            'total_trials': self.num_trials * num_samples
        }
        
        return all_phases, metrics


def random_ris_single_trial(
    num_elements: int,
    h_direct: np.ndarray,
    h_ris_user: np.ndarray,
    h_bs_ris: np.ndarray,
    noise_power: float
) -> Tuple[np.ndarray, float]:
    """
    Single random RIS configuration (for quick baseline).
    
    This is the "Random RIS" baseline from the original experiments.
    """
    phases = np.random.uniform(0, 2 * np.pi, num_elements)
    Theta = np.diag(np.exp(1j * phases))
    h_cascade = np.conj(h_ris_user) @ Theta @ h_bs_ris
    h_eff = h_direct + h_cascade
    signal_power = np.abs(h_eff) ** 2
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear + 1e-10)
    
    return phases, snr_db
