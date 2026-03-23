"""
Base Station Server - Federated Aggregation Logic
Coordinates federated learning across RIS tiles
"""

import torch
import numpy as np
import copy
from collections import OrderedDict


class FederatedServer:
    """
    Base Station that aggregates models from RIS tiles
    """

    def __init__(self, global_model, config):
        self.global_model = global_model
        self.config = config
        self.device = config.DEVICE

        # Move model to device
        self.global_model.to(self.device)

        # Tracking metrics
        self.round_metrics = []
        self.aggregation_method = getattr(config, 'AGGREGATION_METHOD', 'FedAvg')

        # Communication metrics
        self.total_bytes_received = 0
        self.total_bytes_sent = 0
        self.round_latencies = []

        # SCAFFOLD: global control variate
        self.scaffold_c_global = {
            name: torch.zeros_like(param)
            for name, param in global_model.named_parameters()
        }

    def aggregate_weights_fedavg(self, client_weights, client_sizes):
        """
        FedAvg: Weighted average of client models

        Args:
            client_weights: List of client model state dicts
            client_sizes: List of dataset sizes for each client

        Returns:
            Aggregated model weights
        """
        # Calculate weights based on dataset sizes
        total_size = sum(client_sizes)
        weights = [size / total_size for size in client_sizes]

        # Initialize aggregated weights with first client
        aggregated_weights = OrderedDict()

        for key in client_weights[0].keys():
            # Skip non-floating-point buffers (e.g. GNN edge_index)
            if client_weights[0][key].is_floating_point():
                aggregated_weights[key] = sum(
                    w * client_weights[i][key] for i, w in enumerate(weights)
                )
            else:
                # Non-float buffers (topology, indices) are identical across
                # clients — copy unchanged from the first client.
                aggregated_weights[key] = client_weights[0][key].clone()

        return aggregated_weights

    def aggregate_weights_fedprox(self, client_weights, client_sizes, mu=0.01):
        """
        FedProx: FedAvg with proximal term (useful for non-IID)

        Args:
            client_weights: List of client model state dicts
            client_sizes: List of dataset sizes
            mu: Proximal term coefficient

        Returns:
            Aggregated model weights
        """
        # Get current global weights
        global_weights = self.global_model.state_dict()

        # FedAvg aggregation
        aggregated_weights = self.aggregate_weights_fedavg(client_weights, client_sizes)

        # Add proximal term (only for floating-point parameters)
        for key in aggregated_weights.keys():
            if aggregated_weights[key].is_floating_point():
                aggregated_weights[key] = (
                    aggregated_weights[key] + mu * global_weights[key]
                ) / (1 + mu)

        return aggregated_weights

    def aggregate_weights_scaffold(self, client_weights, client_sizes, c_deltas):
        """
        SCAFFOLD: FedAvg + control variate correction
        
        Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging
        for Federated Learning," ICML 2020.

        Args:
            client_weights: List of client model state dicts
            client_sizes: List of dataset sizes
            c_deltas: List of control variate deltas from clients

        Returns:
            Aggregated model weights
        """
        # Standard FedAvg aggregation of model weights
        aggregated_weights = self.aggregate_weights_fedavg(client_weights, client_sizes)

        # Update global control variate
        num_clients = len(c_deltas)
        for name in self.scaffold_c_global:
            delta_sum = torch.zeros_like(self.scaffold_c_global[name])
            for c_delta in c_deltas:
                if name in c_delta:
                    delta_sum += c_delta[name]
            self.scaffold_c_global[name] += delta_sum / num_clients

        return aggregated_weights

    def aggregate_weights(self, client_weights, client_sizes, **kwargs):
        """
        Main aggregation function - routes to specific method

        Args:
            client_weights: List of client model state dicts
            client_sizes: List of dataset sizes for each client
            **kwargs: Additional args (e.g., c_deltas for SCAFFOLD)

        Returns:
            Aggregated model weights
        """
        if self.aggregation_method == "FedAvg":
            return self.aggregate_weights_fedavg(client_weights, client_sizes)
        elif self.aggregation_method == "FedProx":
            return self.aggregate_weights_fedprox(client_weights, client_sizes)
        elif self.aggregation_method == "SCAFFOLD":
            c_deltas = kwargs.get('c_deltas', [])
            return self.aggregate_weights_scaffold(client_weights, client_sizes, c_deltas)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def broadcast_model(self, clients):
        """
        Send global model to all clients.
        Also sets FedProx reference and SCAFFOLD controls.

        Args:
            clients: List of RISClient objects
        """
        global_weights = self.global_model.state_dict()

        # Calculate communication cost (INT8 quantized transmission)
        model_size = sum(p.numel() for p in self.global_model.parameters())
        comm_bytes_per_param = getattr(self.config, 'COMM_BYTES_PER_PARAM', 1)
        bytes_per_client = model_size * comm_bytes_per_param
        total_bytes = bytes_per_client * len(clients)
        self.total_bytes_sent += total_bytes

        # Broadcast to all clients
        for client in clients:
            client.set_model_weights(copy.deepcopy(global_weights))
            
            # FedProx: set global reference for proximal term
            if self.aggregation_method == 'FedProx':
                client.set_global_reference(global_weights)
            
            # SCAFFOLD: send global control variate
            if self.aggregation_method == 'SCAFFOLD':
                client.set_scaffold_controls(self.scaffold_c_global)

        if self.config.VERBOSE:
            print(f"  [Server] Broadcasted model to {len(clients)} clients "
                  f"({total_bytes / 1024:.2f} KB total)")

    def aggregate_round(self, clients, round_num):
        """
        Execute one round of federated learning.
        Supports FedAvg, FedProx, and SCAFFOLD.

        Args:
            clients: List of RISClient objects
            round_num: Current round number

        Returns:
            Dictionary with round metrics
        """
        print(f"\n{'='*60}")
        print(f"Round {round_num + 1}/{self.config.FL_ROUNDS} [{self.aggregation_method}]")
        print(f"{'='*60}")

        # Step 1: Broadcast current global model (+ FedProx/SCAFFOLD state)
        self.broadcast_model(clients)

        # Calculate broadcast size (for metrics)
        model_size = sum(p.numel() for p in self.global_model.parameters())
        comm_bytes_per_param = getattr(self.config, 'COMM_BYTES_PER_PARAM', 1)
        bytes_per_client = model_size * comm_bytes_per_param
        bytes_downloaded_total = bytes_per_client * len(clients)

        # Step 2: Local training on each client
        client_metrics = []
        client_weights = []
        client_sizes = []
        old_weights_list = []  # For SCAFFOLD
        c_deltas = []  # For SCAFFOLD

        for client in clients:
            # Store weights before training (for SCAFFOLD)
            if self.aggregation_method == 'SCAFFOLD':
                old_weights_list.append(client.get_model_weights())

            print(f"\n[Client {client.client_id}] Starting local training...")
            metrics = client.train_local_model(self.config.LOCAL_EPOCHS)
            client_metrics.append(metrics)

            # Collect weights
            new_weights = client.get_model_weights()
            client_weights.append(new_weights)
            client_sizes.append(len(client.dataset))

            # SCAFFOLD: compute control variate update
            if self.aggregation_method == 'SCAFFOLD' and old_weights_list:
                num_steps = self.config.LOCAL_EPOCHS * max(len(client.dataset) // self.config.BATCH_SIZE, 1)
                c_delta = client.compute_scaffold_update(
                    old_weights_list[-1], new_weights,
                    self.config.LEARNING_RATE, num_steps
                )
                c_deltas.append(c_delta)

        # Step 3: Aggregate weights at server
        print(f"\n[Server] Aggregating weights from {len(clients)} clients...")
        aggregated_weights = self.aggregate_weights(
            client_weights, client_sizes, c_deltas=c_deltas
        )

        # Calculate communication cost for uploads (INT8 quantized)
        bytes_uploaded_total = model_size * comm_bytes_per_param * len(clients)
        self.total_bytes_received += bytes_uploaded_total

        # Step 4: Update global model
        self.global_model.load_state_dict(aggregated_weights)

        # Compile round metrics
        round_metric = {
            'round': round_num,
            'aggregation_method': self.aggregation_method,
            'avg_client_loss': np.mean([m['avg_loss'] for m in client_metrics]),
            'max_client_loss': np.max([m['avg_loss'] for m in client_metrics]),
            'min_client_loss': np.min([m['avg_loss'] for m in client_metrics]),
            'total_samples': sum([m['samples_processed'] for m in client_metrics]),
            'total_energy': sum([m['energy_consumed'] for m in client_metrics]),
            'bytes_uploaded': bytes_uploaded_total,
            'bytes_downloaded': bytes_downloaded_total,
            'total_bytes': bytes_uploaded_total + bytes_downloaded_total,
            'client_metrics': client_metrics
        }

        self.round_metrics.append(round_metric)

        # Print summary
        print(f"\n[Round {round_num + 1} Summary]")
        print(f"  Avg Loss: {round_metric['avg_client_loss']:.6f}")
        print(f"  Total Energy: {round_metric['total_energy']:.6f} J")
        print(f"  Communication: {round_metric['total_bytes'] / 1024:.2f} KB")

        return round_metric

    def get_global_model(self):
        """Return the global model"""
        return self.global_model

    def get_global_weights(self):
        """Return global model weights"""
        return self.global_model.state_dict()

    def get_communication_summary(self):
        """
        Get summary of communication costs

        Returns:
            Dictionary with communication metrics
        """
        total_bytes = self.total_bytes_received + self.total_bytes_sent

        # Calculate packet latency (simplified model)
        # Latency = Data Size / Bandwidth
        bandwidth_bytes_per_sec = self.config.NOC_BANDWIDTH_GBPS * 1e9 / 8
        avg_packet_latency = (total_bytes / len(self.round_metrics)) / bandwidth_bytes_per_sec if self.round_metrics else 0

        # Energy for communication
        energy_communication = total_bytes * 8 * self.config.ENERGY_PER_BIT

        num_rounds = max(len(self.round_metrics), 1)
        transmission_time = (total_bytes * 8) / (self.config.NOC_BANDWIDTH_GBPS * 1e9)
        # Each FL round assumed ~1 second; utilization = fraction of available BW used
        total_available_time = num_rounds * 1.0
        bandwidth_utilization = min(transmission_time / total_available_time, 1.0)

        summary = {
            'total_bytes_received': self.total_bytes_received,
            'total_bytes_sent': self.total_bytes_sent,
            'total_bytes': total_bytes,
            'total_kilobytes': total_bytes / 1024,
            'total_megabytes': total_bytes / (1024 * 1024),
            'avg_bytes_per_round': total_bytes / len(self.round_metrics) if self.round_metrics else 0,
            'avg_packet_latency_sec': avg_packet_latency,
            'avg_packet_latency_ms': avg_packet_latency * 1000,
            'energy_communication_joules': energy_communication,
            'bandwidth_utilization': bandwidth_utilization
        }

        return summary

    def get_convergence_metrics(self):
        """
        Analyze convergence behavior

        Returns:
            Convergence metrics
        """
        if not self.round_metrics:
            return {}

        losses = [m['avg_client_loss'] for m in self.round_metrics]

        # Find convergence point (when loss stabilizes)
        convergence_threshold = 0.01  # 1% change
        converged_round = len(losses)

        min_reduction_pct = 0.20  # Must reduce by at least 20% to consider convergence
        for i in range(10, len(losses)):
            # Skip if loss hasn't reduced enough yet
            if losses[i] > losses[0] * (1 - min_reduction_pct):
                continue
            recent_losses = losses[i-10:i]
            if np.std(recent_losses) / np.mean(recent_losses) < convergence_threshold:
                converged_round = i
                break

        metrics = {
            'converged_round': converged_round,
            'initial_loss': losses[0],
            'final_loss': losses[-1],
            'loss_reduction': losses[0] - losses[-1],
            'loss_reduction_percent': ((losses[0] - losses[-1]) / losses[0]) * 100,
            'convergence_rate': (losses[0] - losses[-1]) / len(losses),
            'all_losses': losses
        }

        return metrics