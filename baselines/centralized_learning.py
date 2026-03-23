"""
Centralized Deep Learning Baseline for RIS

Trains a single global model on pooled data from all tiles.
This represents the upper bound on learning performance but violates privacy.

Comparison with Federated Learning:
- Centralized: Best accuracy, no privacy, high communication cost
- Federated: Comparable accuracy, privacy-preserving, lower latency
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import copy
from typing import List, Dict
import numpy as np


class CentralizedRIS:
    """
    Centralized training baseline.
    
    Collects all data from all tiles and trains a single global model.
    """
    
    def __init__(self, model: nn.Module, config):
        """
        Args:
            model: Neural network model (e.g., RISNet)
            config: Configuration object
        """
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.model.to(self.device)
        
        # Track metrics
        self.training_history = []
        
    def train_centralized(
        self,
        tile_datasets: List,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None
    ) -> Dict:
        """
        Train on pooled dataset from all tiles.
        
        Args:
            tile_datasets: List of datasets from each tile
            epochs: Number of training epochs (default: config.LOCAL_EPOCHS * config.FL_ROUNDS)
            batch_size: Batch size (default: config.BATCH_SIZE)
            learning_rate: Learning rate (default: config.LEARNING_RATE)
            
        Returns:
            metrics: Training metrics
        """
        # Use defaults from config if not specified
        if epochs is None:
            # Train for same total epochs as FL (LOCAL_EPOCHS × FL_ROUNDS)
            epochs = self.config.LOCAL_EPOCHS * self.config.FL_ROUNDS
        if batch_size is None:
            batch_size = self.config.BATCH_SIZE
        if learning_rate is None:
            learning_rate = self.config.LEARNING_RATE
        
        # Pool all datasets
        combined_dataset = ConcatDataset(tile_datasets)
        dataloader = DataLoader(
            combined_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        
        # Setup optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        epoch_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (features, targets) in enumerate(dataloader):
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(features)
                loss = criterion(predictions, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Record epoch metrics
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            self.training_history.append({
                'epoch': epoch,
                'loss': avg_epoch_loss
            })
            
            if (epoch + 1) % 10 == 0 and getattr(self.config, 'VERBOSE', False):
                print(f"Centralized Training - Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
        
        # Compute final metrics
        metrics = {
            'final_loss': epoch_losses[-1],
            'initial_loss': epoch_losses[0],
            'total_epochs': epochs,
            'total_samples': len(combined_dataset),
            'loss_reduction_pct': (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] * 100,
            'training_history': epoch_losses
        }
        
        return metrics
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Evaluate the trained model.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            eval_metrics: Evaluation metrics
        """
        self.model.eval()
        criterion = nn.MSELoss()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(features)
                loss = criterion(predictions, targets)
                
                total_loss += loss.item()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Concatenate all predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Compute phase error
        phase_errors = np.abs(all_predictions - all_targets)
        phase_errors = np.minimum(phase_errors, 2 * np.pi - phase_errors)  # Circular error
        phase_error_deg = np.mean(phase_errors) * 180 / np.pi
        
        return {
            'test_loss': total_loss / len(test_loader),
            'phase_error_deg': phase_error_deg,
            'num_samples': len(all_predictions)
        }
    
    def get_model(self) -> nn.Module:
        """Return the trained model."""
        return self.model
    
    def compute_communication_cost(self, tile_datasets: List) -> Dict:
        """
        Estimate communication cost for centralized approach.
        
        In centralized learning, all raw data must be transmitted to the server.
        
        Args:
            tile_datasets: List of tile datasets
            
        Returns:
            comm_cost: Communication cost metrics
        """
        # Each tile must send all its data samples to the central server
        total_samples = sum(len(dataset) for dataset in tile_datasets)
        
        # Assume each sample has:
        # - Features: input_dim × 4 bytes (float32)
        # - Target: num_elements × 4 bytes
        input_dim = self.model.input_dim
        num_elements = self.model.num_elements
        bytes_per_sample = (input_dim + num_elements) * 4
        
        total_bytes = total_samples * bytes_per_sample
        total_kb = total_bytes / 1024
        total_mb = total_kb / 1024
        
        # In FL, only model weights are transmitted (much smaller)
        model_size_bytes = sum(p.numel() * 4 for p in self.model.parameters())
        fl_bytes_per_round = model_size_bytes * len(tile_datasets) * 2  # Upload + download
        fl_total_bytes = fl_bytes_per_round * self.config.FL_ROUNDS
        
        return {
            'centralized_total_bytes': total_bytes,
            'centralized_total_mb': total_mb,
            'fl_total_bytes': fl_total_bytes,
            'fl_total_mb': fl_total_bytes / (1024**2),
            'overhead_ratio': total_bytes / fl_total_bytes,
            'privacy': 'Centralized: NO, FL: YES'
        }


def compare_centralized_vs_federated(
    centralized_model: nn.Module,
    federated_model: nn.Module,
    test_loader: DataLoader,
    config
) -> Dict:
    """
    Direct comparison between centralized and federated models.
    
    Args:
        centralized_model: Trained centralized model
        federated_model: Trained federated model
        test_loader: Test data
        config: Configuration
        
    Returns:
        comparison: Comparison metrics
    """
    # Evaluate both models
    criterion = nn.MSELoss()
    
    def evaluate_model(model):
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for features, targets in test_loader:
                features = features.to(config.DEVICE)
                targets = targets.to(config.DEVICE)
                predictions = model(features)
                loss = criterion(predictions, targets)
                total_loss += loss.item()
        return total_loss / len(test_loader)
    
    cent_loss = evaluate_model(centralized_model)
    fed_loss = evaluate_model(federated_model)
    
    return {
        'centralized_loss': cent_loss,
        'federated_loss': fed_loss,
        'accuracy_gap': (fed_loss - cent_loss) / cent_loss * 100,  # % worse
        'interpretation': 'Federated is comparable if gap < 5%'
    }
