"""
RIS Tile Client - Local Training Logic
Each RIS tile trains its local model on its data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import numpy as np

from utils.metrics import dbm_to_watts


class RISClient:
    """
    Represents a single RIS tile in the federated learning system
    Supports dynamic sleep scheduling for energy optimization
    """

    # Sleep state constants
    STATE_ACTIVE = "ACTIVE"
    STATE_SLEEP = "SLEEP"

    def __init__(self, client_id, model, dataset, config):
        self.client_id = client_id
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = config.DEVICE

        # Move model to device
        self.model.to(self.device)

        # Create data loader
        if dataset is not None:
            self.train_loader = DataLoader(
                dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                drop_last=True
            )
        else:
            self.train_loader = None

        # Optimizer — use lower LR for GNN models
        model_type = getattr(config, 'MODEL_TYPE', 'MLP')
        if model_type == 'GNN':
            lr = getattr(config, 'GNN_LEARNING_RATE', 0.0003)
        else:
            lr = config.LEARNING_RATE
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr
        )

        # Learning rate scheduler (CosineAnnealing over FL rounds * local epochs)
        total_steps = getattr(config, 'FL_ROUNDS', 20) * getattr(config, 'LOCAL_EPOCHS', 3)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-5
        )

        # Phase-aware circular loss function (Bug 9 fix part 2)
        self.criterion = self._phase_mse_loss

        # Metrics tracking
        self.train_losses = []
        self.local_epochs_completed = 0
        self.samples_processed = 0
        self.energy_consumed = 0.0
        self.computation_time = 0.0

        # ---- FedProx support ----
        self.aggregation_method = getattr(config, 'AGGREGATION_METHOD', 'FedAvg')
        self.fedprox_mu = getattr(config, 'FEDPROX_MU', 0.01)
        self.global_model_weights = None  # Set by server before training

        # ---- SCAFFOLD support ----
        self.scaffold_c_local = None   # Local control variate
        self.scaffold_c_global = None  # Global control variate

        # Sleep scheduling state
        self.sleep_enabled = getattr(config, 'SLEEP_SCHEDULING_ENABLED', False)
        self.sleep_state = self.STATE_ACTIVE
        self.sleep_threshold = getattr(config, 'SLEEP_SIGNAL_THRESHOLD', 0.1)
        self.sleep_check_interval = getattr(config, 'SLEEP_CHECK_INTERVAL', 5)
        self.active_power = getattr(config, 'ACTIVE_POWER_TILE', 1.0)
        self.sleep_power = getattr(config, 'SLEEP_POWER_TILE', 0.05)
        self.rounds_since_check = 0
        self.total_sleep_rounds = 0
        self.total_active_rounds = 0
        self.last_signal_strength = 1.0  # Default to high (awake)

        # ---- Pixel-Level Duty Cycling ----
        self.duty_cycle_enabled = getattr(config, 'DUTY_CYCLE_ENABLED', False)
        self.dc_threshold_db = getattr(config, 'DUTY_CYCLE_THRESHOLD_DB', -10)
        self.dc_min_active_ratio = getattr(config, 'DUTY_CYCLE_MIN_ACTIVE_RATIO', 0.25)
        self.dc_strategy = getattr(config, 'DUTY_CYCLE_STRATEGY', 'threshold')
        self.active_power_pixel = getattr(config, 'ACTIVE_POWER_PIXEL', 0.015)
        self.sleep_power_pixel = getattr(config, 'SLEEP_POWER_PIXEL', 0.001)
        self.num_pixels = getattr(config, 'ELEMENTS_PER_TILE', 64)
        self.pixel_mask = np.ones(self.num_pixels, dtype=bool)  # All ON initially
        self.dc_history = []  # Track active ratio per round
        
    def _phase_mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Circular MSE for phase angles (handles the 2*pi wrap-around).
        A prediction of 0.01 and target of 6.27 should yield an error of 0.02, not 6.26.
        """
        # diff in [-pi, pi]
        diff = torch.remainder(pred - target + torch.pi, 2 * torch.pi) - torch.pi
        return torch.mean(diff ** 2)

    def train_local_model(self, epochs):
        """
        Train the local model for specified epochs

        Args:
            epochs: Number of local training epochs

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        epoch_losses = []
        total_samples = 0
        total_flops = 0
        grad_norms = []

        for epoch in range(epochs):
            batch_losses = []

            for batch_idx, (features, labels) in enumerate(self.train_loader):
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(features)
                loss = self.criterion(predictions, labels)

                # ---- FedProx: Add proximal term ----
                if self.aggregation_method == 'FedProx' and self.global_model_weights is not None:
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        if name in self.global_model_weights:
                            global_param = self.global_model_weights[name].to(self.device)
                            proximal_term += torch.sum((param - global_param) ** 2)
                    loss = loss + (self.fedprox_mu / 2.0) * proximal_term

                # Backward pass
                loss.backward()

                # ---- SCAFFOLD: Apply gradient correction ----
                if (self.aggregation_method == 'SCAFFOLD' and
                    self.scaffold_c_local is not None and
                    self.scaffold_c_global is not None):
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and name in self.scaffold_c_local:
                            correction = (self.scaffold_c_global[name].to(self.device) -
                                        self.scaffold_c_local[name].to(self.device))
                            param.grad.data.add_(correction)

                # Gradient clipping to stabilize GNN training
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Log gradient norms for debugging
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norms.append(total_norm ** 0.5)

                self.optimizer.step()

                batch_losses.append(loss.item())
                total_samples += features.size(0)

                # Estimate FLOPs (simplified)
                model_params = self.model.count_parameters()
                flops_per_sample = 3 * model_params  # Rough estimate
                total_flops += flops_per_sample * features.size(0)

            # Step the LR scheduler after each epoch
            self.scheduler.step()

            epoch_loss = np.mean(batch_losses)
            epoch_losses.append(epoch_loss)
            self.train_losses.append(epoch_loss)

            if self.config.VERBOSE and epoch % 2 == 0:
                print(f"  Client {self.client_id} | Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.6f}")

        self.local_epochs_completed += epochs
        self.samples_processed += total_samples

        # Calculate energy consumption
        energy_computation = total_flops * self.config.ENERGY_PER_FLOP
        energy_idle = self.config.IDLE_POWER_TILE * epochs * 0.1  # Assume 0.1s per epoch
        self.energy_consumed += (energy_computation + energy_idle)

        metrics = {
            'client_id': self.client_id,
            'avg_loss': np.mean(epoch_losses),
            'final_loss': epoch_losses[-1],
            'samples_processed': total_samples,
            'energy_consumed': energy_computation + energy_idle,
            'flops': total_flops,
            'avg_grad_norm': np.mean(grad_norms) if grad_norms else 0.0,
            'max_grad_norm': np.max(grad_norms) if grad_norms else 0.0,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }

        return metrics

    def set_global_reference(self, global_weights):
        """Store global model weights for FedProx proximal term."""
        import copy
        self.global_model_weights = copy.deepcopy(global_weights)

    def set_scaffold_controls(self, c_global, c_local=None):
        """Set SCAFFOLD control variates."""
        import copy
        self.scaffold_c_global = copy.deepcopy(c_global)
        if c_local is not None:
            self.scaffold_c_local = copy.deepcopy(c_local)
        elif self.scaffold_c_local is None:
            # Initialize local control to zeros
            self.scaffold_c_local = {name: torch.zeros_like(param) 
                                     for name, param in self.model.named_parameters()}

    def compute_scaffold_update(self, old_weights, new_weights, lr, num_steps):
        """Compute SCAFFOLD control variate update after local training."""
        import copy
        c_new = {}
        c_delta = {}
        for name in old_weights:
            if name in new_weights:
                c_local_old = self.scaffold_c_local.get(name, torch.zeros_like(old_weights[name]))
                c_global = self.scaffold_c_global.get(name, torch.zeros_like(old_weights[name]))
                
                option_ii = (c_local_old - c_global + 
                           (old_weights[name] - new_weights[name]) / (num_steps * lr))
                c_new[name] = option_ii
                c_delta[name] = option_ii - c_local_old
        
        self.scaffold_c_local = c_new
        return c_delta

    def get_model_weights(self):
        """Return current model weights"""
        return copy.deepcopy(self.model.state_dict())

    def set_model_weights(self, weights):
        """Update model with new weights from server"""
        self.model.load_state_dict(weights)

    def evaluate(self, test_loader):
        """
        Evaluate the local model

        Args:
            test_loader: DataLoader for test data

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        num_batches = 0

        predictions_list = []
        labels_list = []

        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(features)

                loss = self.criterion(predictions, labels)
                total_loss += loss.item()

                # Additional metrics
                mse = torch.mean((predictions - labels) ** 2).item()
                mae = torch.mean(torch.abs(predictions - labels)).item()

                total_mse += mse
                total_mae += mae
                num_batches += 1

                predictions_list.append(predictions.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_mae = total_mae / num_batches

        predictions_array = np.concatenate(predictions_list, axis=0)
        labels_array = np.concatenate(labels_list, axis=0)

        # Apply quantization if enabled
        quant_bits = getattr(self.config, 'PHASE_QUANTIZATION_BITS', 0)
        if quant_bits > 0:
            from src.channel_model import quantize_phases
            # Apply to flat array
            predictions_array = quantize_phases(predictions_array, quant_bits)

        # Phase prediction accuracy (within threshold)
        phase_error = np.abs(predictions_array - labels_array)
        phase_error = np.minimum(phase_error, 2 * np.pi - phase_error)  # Circular error
        accuracy_10deg = np.mean(phase_error < np.deg2rad(10))
        accuracy_30deg = np.mean(phase_error < np.deg2rad(30))

        metrics = {
            'loss': avg_loss,
            'mse': avg_mse,
            'mae': avg_mae,
            'phase_error_mean': np.mean(phase_error),
            'phase_error_std': np.std(phase_error),
            'accuracy_10deg': accuracy_10deg,
            'accuracy_30deg': accuracy_30deg,
            'predictions': predictions_array,
            'labels': labels_array
        }

        return metrics

    def compute_snr_improvement(self, test_dataset, num_samples=100):
        """
        Compute SNR improvement using predicted phase shifts

        Args:
            test_dataset: Test dataset with metadata
            num_samples: Number of samples to evaluate

        Returns:
            SNR metrics
        """
        self.model.eval()

        snr_no_ris = []
        snr_random_ris = []
        snr_optimized_ris = []
        snr_optimal = []

        num_samples = min(num_samples, len(test_dataset))

        with torch.no_grad():
            for i in range(num_samples):
                features, optimal_phases = test_dataset[i]
                features = features.unsqueeze(0).to(self.device)

                # Predict phase shifts
                predicted_phases = self.model(features).squeeze().cpu().numpy()
                
                # Apply quantization if enabled
                quant_bits = getattr(self.config, 'PHASE_QUANTIZATION_BITS', 0)
                if quant_bits > 0:
                    from src.channel_model import quantize_phases
                    predicted_phases = quantize_phases(predicted_phases, quant_bits)
                
                optimal_phases = optimal_phases.numpy()

                # Get channel information from metadata
                metadata = test_dataset.metadata[i]
                H_direct = metadata['H_direct']
                H_ris = metadata['H_ris']
                h_bs_ris = metadata['h_bs_ris']

                # Target user (first user)
                target_user = 0
                h_direct = H_direct[target_user]
                h_ris = H_ris[target_user]
                h_cascade = h_ris * h_bs_ris

                # Noise power and Tx power in Watts
                noise_power = dbm_to_watts(self.config.NOISE_POWER_DBM)
                tx_power = dbm_to_watts(self.config.TX_POWER_DBM)

                # SNR without RIS (direct path only)
                signal_no_ris = tx_power * np.abs(h_direct) ** 2
                snr_no_ris.append(10 * np.log10(signal_no_ris / noise_power))

                # SNR with random RIS
                random_phases = np.random.uniform(0, 2 * np.pi, len(h_ris))
                h_total_random = h_direct + np.sum(h_cascade * np.exp(1j * random_phases))
                signal_random = tx_power * np.abs(h_total_random) ** 2
                snr_random_ris.append(10 * np.log10(signal_random / noise_power))

                # SNR with optimized RIS (predicted)
                h_total_optimized = h_direct + np.sum(h_cascade * np.exp(1j * predicted_phases))
                signal_optimized = tx_power * np.abs(h_total_optimized) ** 2
                snr_optimized_ris.append(10 * np.log10(signal_optimized / noise_power))

                # SNR with optimal RIS (recompute true MRC-optimal phases from raw channels)
                # theta_n = angle(h_direct) - angle(h_cascade_n) for coherent combining
                true_optimal_phases = np.mod(
                    np.angle(h_direct) - np.angle(h_cascade), 2 * np.pi
                )
                h_total_optimal = h_direct + np.sum(h_cascade * np.exp(1j * true_optimal_phases))
                signal_optimal = tx_power * np.abs(h_total_optimal) ** 2
                snr_opt_db = 10 * np.log10(signal_optimal / noise_power)
                snr_optimal.append(snr_opt_db)

                # Per-sample assertion: FL SNR must not exceed genie-aided optimal
                snr_fl_db = snr_optimized_ris[-1]
                if snr_fl_db > snr_opt_db + 0.1:  # 0.1 dB tolerance for float precision
                    print(f"  WARNING: Sample {i}: FL SNR ({snr_fl_db:.2f} dB) > "
                          f"genie-aided ({snr_opt_db:.2f} dB) by "
                          f"{snr_fl_db - snr_opt_db:.2f} dB. "
                          f"Check channel/phase computation.")

        # Sanity check: genie-aided optimal must be an upper bound
        mean_optimal = np.mean(snr_optimal)
        mean_predicted = np.mean(snr_optimized_ris)
        if mean_predicted > mean_optimal + 0.01:
            print(f"  WARNING: FL SNR ({mean_predicted:.2f} dB) exceeds genie-aided optimal "
                  f"({mean_optimal:.2f} dB). Check SNR computation.")

        metrics = {
            'snr_no_ris_mean': np.mean(snr_no_ris),
            'snr_random_ris_mean': np.mean(snr_random_ris),
            'snr_optimized_ris_mean': np.mean(snr_optimized_ris),
            'snr_optimal_mean': np.mean(snr_optimal),
            'snr_gain_over_no_ris': np.mean(snr_optimized_ris) - np.mean(snr_no_ris),
            'snr_gain_over_random': np.mean(snr_optimized_ris) - np.mean(snr_random_ris),
            'optimality_gap': np.mean(snr_optimal) - np.mean(snr_optimized_ris),
            'snr_no_ris_all': snr_no_ris,
            'snr_random_ris_all': snr_random_ris,
            'snr_optimized_ris_all': snr_optimized_ris,
            'snr_optimal_all': snr_optimal
        }

        return metrics

    def get_communication_cost(self):
        """
        Calculate communication cost (model size in bytes, INT8 quantized)
        """
        model_params = sum(p.numel() for p in self.model.parameters())
        comm_bytes_per_param = getattr(self.config, 'COMM_BYTES_PER_PARAM', 1)
        bytes_transmitted = model_params * comm_bytes_per_param

        return {
            'params_count': model_params,
            'bytes': bytes_transmitted,
            'kilobytes': bytes_transmitted / 1024,
            'megabytes': bytes_transmitted / (1024 * 1024)
        }

    # ============ Sleep Scheduling Methods ============

    def update_sleep_state(self, signal_strength=None):
        """
        Update sleep state based on signal strength.
        
        Args:
            signal_strength: Normalized signal strength (0-1). If None, uses last value.
        
        Returns:
            Current sleep state
        """
        if not self.sleep_enabled:
            return self.STATE_ACTIVE
        
        if signal_strength is not None:
            self.last_signal_strength = signal_strength
        
        self.rounds_since_check += 1
        
        # Only check sleep state periodically
        if self.rounds_since_check >= self.sleep_check_interval:
            self.rounds_since_check = 0
            
            if self.last_signal_strength < self.sleep_threshold:
                self.sleep_state = self.STATE_SLEEP
            else:
                self.sleep_state = self.STATE_ACTIVE
        
        # Track rounds in each state
        if self.sleep_state == self.STATE_SLEEP:
            self.total_sleep_rounds += 1
        else:
            self.total_active_rounds += 1
        
        return self.sleep_state

    def should_participate(self):
        """
        Check if this tile should participate in the current round.
        
        Returns:
            True if tile is active and should participate
        """
        if not self.sleep_enabled:
            return True
        return self.sleep_state == self.STATE_ACTIVE

    def get_current_power(self):
        """
        Get current power consumption based on sleep state.
        
        Returns:
            Power consumption in Watts
        """
        if self.sleep_state == self.STATE_SLEEP:
            return self.sleep_power
        return self.active_power

    def get_sleep_metrics(self):
        """
        Get sleep scheduling metrics.
        
        Returns:
            Dictionary with sleep scheduling statistics
        """
        total_rounds = self.total_sleep_rounds + self.total_active_rounds
        sleep_ratio = self.total_sleep_rounds / total_rounds if total_rounds > 0 else 0
        
        # Calculate energy savings
        energy_if_always_active = total_rounds * self.active_power
        actual_energy = (self.total_active_rounds * self.active_power + 
                        self.total_sleep_rounds * self.sleep_power)
        energy_saved = energy_if_always_active - actual_energy
        
        return {
            'sleep_enabled': self.sleep_enabled,
            'current_state': self.sleep_state,
            'total_active_rounds': self.total_active_rounds,
            'total_sleep_rounds': self.total_sleep_rounds,
            'sleep_ratio': sleep_ratio,
            'last_signal_strength': self.last_signal_strength,
            'energy_if_always_active_j': energy_if_always_active,
            'actual_energy_j': actual_energy,
            'energy_saved_j': energy_saved,
            'savings_percentage': (energy_saved / energy_if_always_active * 100) if energy_if_always_active > 0 else 0
        }

    def force_wake(self):
        """Force tile to wake up (useful for aggregation rounds)"""
        self.sleep_state = self.STATE_ACTIVE
        self.rounds_since_check = 0

    def force_sleep(self):
        """Force tile to sleep (for testing)"""
        if self.sleep_enabled:
            self.sleep_state = self.STATE_SLEEP

    # ============ Pixel-Level Duty Cycling Methods ============

    def compute_pixel_mask(self, csi_vector):
        """
        Compute pixel ON/OFF mask based on CSI and strategy.
        
        Args:
            csi_vector: Complex CSI vector for this tile's pixels (num_pixels,)
            
        Returns:
            Boolean mask (num_pixels,) — True = pixel ON
        """
        if not self.duty_cycle_enabled:
            self.pixel_mask = np.ones(self.num_pixels, dtype=bool)
            return self.pixel_mask
        
        N = len(csi_vector)
        csi_power_db = 10 * np.log10(np.abs(csi_vector) ** 2 + 1e-20)
        min_active = max(1, int(self.dc_min_active_ratio * N))
        
        if self.dc_strategy == 'threshold':
            # Pixels ON if CSI power > threshold
            mask = csi_power_db > self.dc_threshold_db
            # Ensure minimum active ratio
            if np.sum(mask) < min_active:
                # Turn on the top-k strongest pixels
                top_k_idx = np.argsort(csi_power_db)[-min_active:]
                mask = np.zeros(N, dtype=bool)
                mask[top_k_idx] = True
                
        elif self.dc_strategy == 'topk':
            # Always keep top-k pixels active (k based on min_active_ratio)
            k = max(min_active, int(self.dc_min_active_ratio * N))
            top_k_idx = np.argsort(csi_power_db)[-k:]
            mask = np.zeros(N, dtype=bool)
            mask[top_k_idx] = True
            
        elif self.dc_strategy == 'adaptive':
            # Adaptive: adjust threshold based on median CSI
            median_db = np.median(csi_power_db)
            adaptive_threshold = median_db - 6  # 6 dB below median
            mask = csi_power_db > adaptive_threshold
            if np.sum(mask) < min_active:
                top_k_idx = np.argsort(csi_power_db)[-min_active:]
                mask = np.zeros(N, dtype=bool)
                mask[top_k_idx] = True
        else:
            mask = np.ones(N, dtype=bool)
        
        self.pixel_mask = mask
        self.dc_history.append(float(np.mean(mask)))
        return mask
    
    def apply_duty_cycle_to_phases(self, phases, csi_vector=None):
        """
        Apply duty cycling mask to phase shifts.
        Inactive pixels get phase 0 (effectively disconnected).
        
        Args:
            phases: Phase shift array (num_pixels,)
            csi_vector: Optional CSI to recompute mask
            
        Returns:
            Masked phases, pixel mask
        """
        if csi_vector is not None:
            self.compute_pixel_mask(csi_vector)
        
        masked_phases = phases.copy()
        if self.duty_cycle_enabled:
            masked_phases[~self.pixel_mask] = 0.0
        
        return masked_phases, self.pixel_mask
    
    def get_duty_cycle_metrics(self):
        """
        Get pixel-level duty cycling metrics.
        
        Returns:
            Dictionary with duty cycling statistics
        """
        active_count = int(np.sum(self.pixel_mask))
        total_pixels = self.num_pixels
        active_ratio = active_count / total_pixels if total_pixels > 0 else 1.0
        
        # Energy calculation
        energy_all_active = total_pixels * self.active_power_pixel
        energy_with_dc = (active_count * self.active_power_pixel + 
                         (total_pixels - active_count) * self.sleep_power_pixel)
        energy_savings = energy_all_active - energy_with_dc
        
        return {
            'duty_cycle_enabled': self.duty_cycle_enabled,
            'strategy': self.dc_strategy,
            'active_pixels': active_count,
            'total_pixels': total_pixels,
            'active_ratio': active_ratio,
            'energy_all_active_w': energy_all_active,
            'energy_with_dc_w': energy_with_dc,
            'energy_savings_w': energy_savings,
            'savings_pct': (energy_savings / energy_all_active * 100) if energy_all_active > 0 else 0,
            'avg_active_ratio_history': float(np.mean(self.dc_history)) if self.dc_history else 1.0,
        }