"""
Component Testing Script
Test individual components before full training
"""

import torch
import numpy as np
from config import Config
from models.ris_net import RISNet, RISNetCNN
from src.dataset_utils import RISChannelDataset, create_non_iid_datasets, create_test_dataset
from src.client import RISClient
from src.server import FederatedServer
from torch.utils.data import DataLoader


def test_dataset_generation():
    """Test dataset generation"""
    print("\n" + "=" * 60)
    print("TEST: Dataset Generation")
    print("=" * 60)

    try:
        # Create single dataset
        dataset = RISChannelDataset(
            num_samples=100,
            num_ris_elements=64,
            num_users=4,
            room_size=(10, 10, 3),
            frequency=28e9
        )

        print(f"✓ Dataset created")
        print(f"  Samples: {len(dataset)}")
        print(f"  Input dim: {dataset.get_input_dim()}")
        print(f"  Features shape: {dataset.features.shape}")
        print(f"  Labels shape: {dataset.labels.shape}")

        # Test data loading
        features, labels = dataset[0]
        print(f"\n✓ Sample data loaded")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Features range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  Labels range: [{labels.min():.3f}, {labels.max():.3f}]")

        # Test non-IID datasets
        datasets, positions = create_non_iid_datasets(Config, num_tiles=4)
        print(f"\n✓ Non-IID datasets created")
        print(f"  Number of datasets: {len(datasets)}")
        print(f"  Tile positions: {len(positions)}")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_model_architecture():
    """Test model architectures"""
    print("\n" + "=" * 60)
    print("TEST: Model Architecture")
    print("=" * 60)

    try:
        # Test RISNet
        input_dim = 100
        num_elements = 64

        model = RISNet(
            input_dim=input_dim,
            num_elements=num_elements,
            hidden_dim=256,
            num_layers=3,
            dropout=0.1
        )

        print(f"✓ RISNet created")
        print(f"  Parameters: {model.count_parameters():,}")

        # Test forward pass
        batch_size = 32
        x = torch.randn(batch_size, input_dim)
        output = model(x)

        print(f"\n✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")

        # Test CNN model
        cnn_model = RISNetCNN(
            input_channels=4,
            grid_size=(8, 8),
            hidden_channels=64
        )

        x_grid = torch.randn(batch_size, 4, 8, 8)
        output_grid = cnn_model(x_grid)

        print(f"\n✓ RISNetCNN created")
        print(f"  Input shape: {x_grid.shape}")
        print(f"  Output shape: {output_grid.shape}")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_client():
    """Test RIS client functionality"""
    print("\n" + "=" * 60)
    print("TEST: RIS Client")
    print("=" * 60)

    try:
        # Create dataset
        dataset = RISChannelDataset(
            num_samples=100,
            num_ris_elements=64,
            num_users=4,
            room_size=(10, 10, 3),
            frequency=28e9
        )

        # Create model
        model = RISNet(
            input_dim=dataset.get_input_dim(),
            num_elements=64,
            hidden_dim=128,
            num_layers=2
        )

        # Create client
        client = RISClient(
            client_id=0,
            model=model,
            dataset=dataset,
            config=Config
        )

        print(f"✓ Client created")
        print(f"  Client ID: {client.client_id}")
        print(f"  Dataset size: {len(client.dataset)}")

        # Test training
        print("\n  Testing local training...")
        metrics = client.train_local_model(epochs=2)

        print(f"✓ Local training successful")
        print(f"  Average loss: {metrics['avg_loss']:.6f}")
        print(f"  Energy consumed: {metrics['energy_consumed']:.6f} J")

        # Test evaluation
        test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        eval_metrics = client.evaluate(test_loader)

        print(f"\n✓ Evaluation successful")
        print(f"  Test loss: {eval_metrics['loss']:.6f}")
        print(f"  Phase error: {np.rad2deg(eval_metrics['phase_error_mean']):.2f}°")

        # Test SNR computation
        snr_metrics = client.compute_snr_improvement(dataset, num_samples=20)

        print(f"\n✓ SNR computation successful")
        print(f"  SNR (no RIS): {snr_metrics['snr_no_ris_mean']:.2f} dB")
        print(f"  SNR (optimized): {snr_metrics['snr_optimized_ris_mean']:.2f} dB")
        print(f"  SNR gain: {snr_metrics['snr_gain_over_no_ris']:.2f} dB")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_server():
    """Test federated server"""
    print("\n" + "=" * 60)
    print("TEST: Federated Server")
    print("=" * 60)

    try:
        # Create datasets
        datasets, _ = create_non_iid_datasets(Config, num_tiles=3)

        # Create global model
        input_dim = datasets[0].get_input_dim()
        global_model = RISNet(
            input_dim=input_dim,
            num_elements=64,
            hidden_dim=128,
            num_layers=2
        )

        # Create server
        server = FederatedServer(global_model, Config)

        print(f"✓ Server created")

        # Create clients
        clients = []
        for i, dataset in enumerate(datasets):
            model = RISNet(
                input_dim=input_dim,
                num_elements=64,
                hidden_dim=128,
                num_layers=2
            )
            client = RISClient(i, model, dataset, Config)
            clients.append(client)

        print(f"✓ {len(clients)} clients created")

        # Test one FL round
        print("\n  Testing FL round...")
        round_metric = server.aggregate_round(clients, round_num=0)

        print(f"\n✓ FL round successful")
        print(f"  Avg client loss: {round_metric['avg_client_loss']:.6f}")
        print(f"  Total energy: {round_metric['total_energy']:.6f} J")
        print(f"  Communication: {round_metric['total_bytes'] / 1024:.2f} KB")

        # Test communication summary
        comm_summary = server.get_communication_summary()

        print(f"\n✓ Communication summary")
        print(f"  Total bytes: {comm_summary['total_kilobytes']:.2f} KB")
        print(f"  Avg latency: {comm_summary['avg_packet_latency_ms']:.3f} ms")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics calculation"""
    print("\n" + "=" * 60)
    print("TEST: Metrics Calculation")
    print("=" * 60)

    try:
        from utils.metrics import *

        # Test SNR calculation
        signal_power = 1e-3  # 1 mW
        snr = calculate_snr(signal_power)
        print(f"✓ SNR calculation: {snr:.2f} dB")

        # Test achievable rate
        rate = calculate_achievable_rate(snr)
        print(f"✓ Achievable rate: {rate:.2f} bps/Hz")

        # Test phase error
        predicted = np.random.uniform(0, 2 * np.pi, 64)
        true = np.random.uniform(0, 2 * np.pi, 64)
        phase_metrics = calculate_phase_error(predicted, true)
        print(f"✓ Phase error: {phase_metrics['mean_error_deg']:.2f}°")

        # Test energy efficiency
        energy_metrics = calculate_energy_efficiency(0.1, 5.0)
        print(f"✓ Energy efficiency: {energy_metrics['bits_per_joule']:.2f} bits/J")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_plotting():
    """Test plotting functions"""
    print("\n" + "=" * 60)
    print("TEST: Plotting Functions")
    print("=" * 60)

    try:
        from utils.plotting import *
        import tempfile

        # Create temporary directory
        temp_dir = tempfile.mkdtemp()

        # Create dummy data
        round_metrics = [
            {
                'round': i,
                'avg_client_loss': 0.1 * (1 - i / 100),
                'max_client_loss': 0.12 * (1 - i / 100),
                'min_client_loss': 0.08 * (1 - i / 100),
                'total_bytes': 1024 * 100,
                'total_energy': 0.001,
                'client_metrics': [
                    {'avg_loss': 0.1 * (1 - i / 100)} for _ in range(3)
                ]
            }
            for i in range(20)
        ]

        # Test convergence plot
        plot_convergence_curve(round_metrics, temp_dir)
        print(f"✓ Convergence plot created")

        # Test communication plot
        plot_communication_overhead(round_metrics, temp_dir)
        print(f"✓ Communication plot created")

        # Test energy plot
        plot_energy_consumption(round_metrics, temp_dir)
        print(f"✓ Energy plot created")

        print(f"\n  Plots saved to: {temp_dir}")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all component tests"""
    print("\n" + "=" * 60)
    print("COMPONENT TESTING SUITE")
    print("=" * 60)

    tests = [
        ("Dataset Generation", test_dataset_generation),
        ("Model Architecture", test_model_architecture),
        ("RIS Client", test_client),
        ("Federated Server", test_server),
        ("Metrics Calculation", test_metrics),
        ("Plotting Functions", test_plotting)
    ]

    results = []

    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<30} {status}")

    total = len(results)
    passed = sum(1 for _, r in results if r)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Ready for full training.")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before training.")


if __name__ == "__main__":
    run_all_tests()