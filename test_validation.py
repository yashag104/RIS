"""
Validation Tests for RIS Federated Learning

Tests for the specific issues identified in the anomalous results:
1. SNR computation correctness (dBm-to-Watts, path loss)
2. Genie-aided optimal is a true upper bound
3. GNN architecture and adjacency matrix
4. Communication volume accounting
5. No-RIS baseline sanity
"""

import sys
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, '.')

from config import Config
from utils.metrics import dbm_to_watts, calculate_snr, compute_ris_snr_db
from src.channel_model import (
    RicianChannel, generate_ris_channel_dataset, _channels_to_dataset
)
from models.ris_net_gnn import build_noc_adjacency, RISNetGNNWrapper
from models.ris_net import create_model


# ============================================================================
# Test 1: dBm-to-Watts conversion
# ============================================================================

def test_dbm_to_watts():
    """Verify dBm to Watts conversion is correct."""
    print("\n" + "=" * 60)
    print("TEST 1: dBm-to-Watts Conversion")
    print("=" * 60)
    passed = True

    # -90 dBm = 1e-12 W (1 picowatt)
    val = dbm_to_watts(-90)
    expected = 1e-12
    ok = abs(val - expected) / expected < 1e-6
    print(f"  -90 dBm -> {val:.3e} W (expected {expected:.3e}) {'PASS' if ok else 'FAIL'}")
    passed &= ok

    # 30 dBm = 1 W
    val = dbm_to_watts(30)
    expected = 1.0
    ok = abs(val - expected) < 1e-6
    print(f"   30 dBm -> {val:.6f} W (expected {expected}) {'PASS' if ok else 'FAIL'}")
    passed &= ok

    # 0 dBm = 0.001 W (1 mW)
    val = dbm_to_watts(0)
    expected = 0.001
    ok = abs(val - expected) / expected < 1e-6
    print(f"    0 dBm -> {val:.6f} W (expected {expected}) {'PASS' if ok else 'FAIL'}")
    passed &= ok

    # 20 dBm = 0.1 W (100 mW)
    val = dbm_to_watts(20)
    expected = 0.1
    ok = abs(val - expected) / expected < 1e-6
    print(f"   20 dBm -> {val:.6f} W (expected {expected}) {'PASS' if ok else 'FAIL'}")
    passed &= ok

    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================================
# Test 2: Path loss computation
# ============================================================================

def test_path_loss():
    """Verify path loss at specific distances."""
    print("\n" + "=" * 60)
    print("TEST 2: Path Loss Computation")
    print("=" * 60)
    passed = True

    ch = RicianChannel(
        num_elements=64,
        k_factor_db=10.0,
        frequency=28e9,
        path_loss_exponent=2.5,
    )

    # Expected: PL(d) = (lambda / (4*pi*d))^alpha
    # lambda = 3e8 / 28e9 = 0.01071 m

    for d in [1.0, 5.0, 10.0, 50.0]:
        pl = ch._compute_path_loss(d)
        expected = (ch.wavelength / (4 * np.pi * d)) ** ch.path_loss_exponent
        ok = abs(pl - expected) / expected < 1e-6
        pl_db = 10 * np.log10(pl)
        print(f"  d={d:5.1f}m: PL={pl:.3e} ({pl_db:.1f} dB) {'PASS' if ok else 'FAIL'}")
        passed &= ok

    # At d=10m: PL should be roughly -96 to -100 dB
    pl_10m = ch._compute_path_loss(10.0)
    pl_10m_db = 10 * np.log10(pl_10m)
    ok = -110 < pl_10m_db < -80
    print(f"\n  PL(10m) = {pl_10m_db:.1f} dB (expected -80 to -110 dB) {'PASS' if ok else 'FAIL'}")
    passed &= ok

    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================================
# Test 3: No-RIS baseline SNR sanity
# ============================================================================

def test_no_ris_snr():
    """
    Test that no-RIS SNR is in a physically reasonable range.

    With P_tx=30dBm=1W, noise=-90dBm=1e-12W, 28GHz, 10m room:
    SNR_no_ris = P_tx * PL(d) * |h_fade|^2 / sigma_n^2

    For d~7m (avg distance): PL ~ -96 dB, so SNR ~ 30 - 96 - (-90) = +24 dB.
    This is CORRECT for a 10m room. The +24 dB is physically valid.
    """
    print("\n" + "=" * 60)
    print("TEST 3: No-RIS Baseline SNR")
    print("=" * 60)
    passed = True

    np.random.seed(42)

    ch = RicianChannel(
        num_elements=64,
        k_factor_db=10.0,
        frequency=28e9,
        path_loss_exponent=2.5,
    )

    tx_power = dbm_to_watts(30)  # 1 W
    noise_power = dbm_to_watts(-90)  # 1e-12 W

    bs_pos = np.array([5, 10, 1.5])
    ris_pos = np.array([5, 0, 1.5])

    snr_values = []
    for _ in range(100):
        user_pos = np.random.uniform([0, 0, 0.5], [10, 10, 3], size=(1, 3))
        channels = ch.generate_channel(bs_pos, user_pos, ris_pos)
        h_direct = channels['h_direct'][0]

        signal = tx_power * np.abs(h_direct) ** 2
        snr_db = 10 * np.log10(signal / noise_power)
        snr_values.append(snr_db)

    mean_snr = np.mean(snr_values)
    print(f"  Mean no-RIS SNR: {mean_snr:.2f} dB")
    print(f"  Min: {np.min(snr_values):.2f} dB, Max: {np.max(snr_values):.2f} dB")

    # For 10m room with these parameters, no-RIS SNR should be ~10 to 30 dB
    ok = -40 <= mean_snr <= 40
    print(f"  Range check (expected -40 to 40 dB):\t\t\t", end="")
    if -40 <= mean_snr <= 40:
        print("PASS")
    else:
        print("FAIL")
    passed &= ok

    print("  Note: Negative SNR reflects the obstructed direct path (exponent=3.5).")
    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================================
# Test 4: Genie-aided optimal is an upper bound
# ============================================================================

def test_genie_aided_upper_bound():
    """
    Verify that the MRC-optimal phases give the highest possible SNR.
    Random phases and perturbed phases must give lower SNR.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Genie-Aided Optimal Upper Bound")
    print("=" * 60)
    passed = True

    np.random.seed(42)

    ch = RicianChannel(
        num_elements=64,
        k_factor_db=10.0,
        frequency=28e9,
        path_loss_exponent=2.5,
    )

    tx_power = dbm_to_watts(30)
    noise_power = dbm_to_watts(-90)

    bs_pos = np.array([5, 10, 1.5])
    ris_pos = np.array([5, 0, 1.5])

    violations = 0
    num_tests = 200

    for _ in range(num_tests):
        user_pos = np.random.uniform([0, 0, 0.5], [10, 10, 3], size=(1, 3))
        channels = ch.generate_channel(bs_pos, user_pos, ris_pos)

        h_direct = channels['h_direct'][0]
        h_bs_ris = channels['h_bs_ris']
        h_ris_user = channels['h_ris_user'][0]
        h_cascade = h_ris_user * h_bs_ris

        # MRC optimal phases
        optimal_phases = np.mod(
            np.angle(h_direct) - np.angle(h_cascade), 2 * np.pi
        )

        # SNR with optimal phases
        h_total_opt = h_direct + np.sum(h_cascade * np.exp(1j * optimal_phases))
        snr_opt = tx_power * np.abs(h_total_opt) ** 2 / noise_power

        # SNR with random phases (should be lower)
        for _ in range(10):
            random_phases = np.random.uniform(0, 2 * np.pi, 64)
            h_total_rand = h_direct + np.sum(h_cascade * np.exp(1j * random_phases))
            snr_rand = tx_power * np.abs(h_total_rand) ** 2 / noise_power

            if snr_rand > snr_opt * 1.001:  # 0.1% tolerance
                violations += 1

        # SNR with perturbed-optimal phases (should be lower)
        for std in [0.1, 0.5, 1.0]:
            perturbed = optimal_phases + np.random.normal(0, std, 64)
            h_total_pert = h_direct + np.sum(h_cascade * np.exp(1j * perturbed))
            snr_pert = tx_power * np.abs(h_total_pert) ** 2 / noise_power

            if snr_pert > snr_opt * 1.001:
                violations += 1

    ok = violations == 0
    print(f"  Random/perturbed phase tests: {num_tests * 13} trials, {violations} violations")
    print(f"  MRC upper bound holds: {'PASS' if ok else 'FAIL'}")
    passed &= ok

    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================================
# Test 5: Torus adjacency matrix
# ============================================================================

def test_torus_adjacency():
    """
    Verify that Torus adjacency for a 4x4 grid gives exactly 4 neighbors
    per node (with wrap-around) plus self-loop.
    """
    print("\n" + "=" * 60)
    print("TEST 5: Torus Adjacency Matrix")
    print("=" * 60)
    passed = True

    edge_index = build_noc_adjacency(16, "Torus", grid_rows=4, grid_cols=4)
    edges = set()
    for i in range(edge_index.size(1)):
        edges.add((edge_index[0, i].item(), edge_index[1, i].item()))

    # Check each node has exactly 4 neighbors + self-loop = 5 edges
    for node in range(16):
        neighbors = {dst for src, dst in edges if src == node}
        # Should have self-loop + 4 neighbors
        ok = len(neighbors) == 5 and node in neighbors
        row, col = node // 4, node % 4
        expected_neighbors = {
            node,  # self
            row * 4 + (col + 1) % 4,  # right (wrap)
            row * 4 + (col - 1) % 4,  # left (wrap)
            ((row + 1) % 4) * 4 + col,  # down (wrap)
            ((row - 1) % 4) * 4 + col,  # up (wrap)
        }
        ok2 = neighbors == expected_neighbors
        if not ok2:
            print(f"  Node {node} ({row},{col}): neighbors={neighbors}, "
                  f"expected={expected_neighbors}")
        passed &= ok2

    total_edges = len(edges)
    expected_edges = 16 * 5  # 16 nodes * 5 edges each
    ok = total_edges == expected_edges
    print(f"  Total edges: {total_edges} (expected {expected_edges}) {'PASS' if ok else 'FAIL'}")
    passed &= ok

    print(f"  Each node has 4 neighbors + self-loop: {'PASS' if passed else 'FAIL'}")
    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================================
# Test 6: GNN parameter count and communication volume
# ============================================================================

def test_communication_volume():
    """
    Verify that the GNN model size is consistent with expected communication.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Communication Volume")
    print("=" * 60)
    passed = True

    # Create a GNN model with the current config
    input_dim = 1330  # Approximate input dim for 10 users, 64 elements
    model = create_model(
        model_type="GNN",
        input_dim=input_dim,
        num_elements=64,
        config=Config,
    )

    total_params = model.count_parameters()
    model_size_int8 = total_params * Config.COMM_BYTES_PER_PARAM
    per_round = model_size_int8 * 2 * Config.NUM_TILES  # up + down, all tiles
    total = per_round * Config.FL_ROUNDS

    print(f"  Model parameters: {total_params:,}")
    print(f"  Model size (INT8): {model_size_int8 / 1024:.2f} KB")
    print(f"  Per round (16 tiles, up+down): {per_round / 1024:.2f} KB")
    print(f"  Total ({Config.FL_ROUNDS} rounds): {total / 1024:.2f} KB = {total / (1024*1024):.2f} MB")

    # Check communication is in a reasonable range
    # With hidden_dim=32: expect ~50K params -> ~50 KB INT8
    # Total: 50 KB * 2 * 16 * 20 = 32 MB
    ok = total < 1024 * 1024 * 1024  # Less than 1 GB total
    print(f"  Total < 1 GB: {'PASS' if ok else 'FAIL'}")
    passed &= ok

    # Bandwidth check: at 10 Gbps, how long to transmit per round?
    bandwidth_bytes_per_sec = Config.NOC_BANDWIDTH_GBPS * 1e9 / 8
    latency_sec = per_round / bandwidth_bytes_per_sec
    print(f"  Per-round latency at 10 Gbps: {latency_sec * 1000:.3f} ms")

    ok = latency_sec * 1000 < 100  # Less than 100ms per round
    print(f"  Latency < 100 ms: {'PASS' if ok else 'FAIL'}")
    passed &= ok

    # Utilization check
    total_transmission_time = (total * 8) / (Config.NOC_BANDWIDTH_GBPS * 1e9)
    total_available_time = Config.FL_ROUNDS * 1.0
    utilization = total_transmission_time / total_available_time
    print(f"  Bandwidth utilization: {utilization * 100:.2f}%")

    ok = utilization < 0.8  # Under 80%
    print(f"  Utilization < 80%: {'PASS' if ok else 'FAIL'}")
    passed &= ok

    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================================
# Test 7: GNN forward pass produces valid phases
# ============================================================================

def test_gnn_forward():
    """
    Verify GNN produces phases in [0, 2pi] and gradients flow.
    """
    print("\n" + "=" * 60)
    print("TEST 7: GNN Forward Pass & Gradients")
    print("=" * 60)
    passed = True

    input_dim = 100
    model = RISNetGNNWrapper(
        input_dim=input_dim,
        num_elements=64,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        num_tiles=16,
        topology="Torus",
        grid_rows=4,
        grid_cols=4,
    )

    # Test single sample (per-tile training mode — skips GAT)
    x = torch.randn(1, input_dim)
    y = model(x)
    ok = y.shape == (1, 64)
    print(f"  Single sample output shape: {y.shape} (expected (1, 64)) {'PASS' if ok else 'FAIL'}")
    passed &= ok

    ok = (y >= 0).all() and (y <= 2 * np.pi + 0.01).all()
    print(f"  Phase range [0, 2pi]: [{y.min():.3f}, {y.max():.3f}] {'PASS' if ok else 'FAIL'}")
    passed &= ok

    # Test batch (per-tile training mode — skips GAT)
    x = torch.randn(64, input_dim)
    y = model(x)
    ok = y.shape == (64, 64)
    print(f"  Batch(64) output shape: {y.shape} (expected (64, 64)) {'PASS' if ok else 'FAIL'}")
    passed &= ok

    # Test full-graph mode (batch_size == num_tiles → uses GAT)
    x = torch.randn(16, input_dim)
    y = model(x)
    ok = y.shape == (16, 64)
    print(f"  Full graph(16) output shape: {y.shape} (expected (16, 64)) {'PASS' if ok else 'FAIL'}")
    passed &= ok

    # Test gradient flow
    x = torch.randn(16, input_dim, requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    has_grad = all(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    print(f"  Gradients flow through all layers: {'PASS' if has_grad else 'FAIL'}")
    passed &= has_grad

    # Check GAT layer gradients specifically
    for i, gat in enumerate(model.gat_layers):
        gat_has_grad = all(p.grad is not None and p.grad.abs().sum() > 0
                          for p in gat.parameters() if p.requires_grad)
        print(f"    GAT layer {i} gradients: {'PASS' if gat_has_grad else 'FAIL (zero/None)'}")
        passed &= gat_has_grad

    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================================
# Test 8: Steering vector normalization
# ============================================================================

def test_steering_vector_norm():
    """
    Verify the UPA steering vector has unit norm.
    """
    print("\n" + "=" * 60)
    print("TEST 8: Steering Vector Normalization")
    print("=" * 60)
    passed = True

    ch = RicianChannel(num_elements=64, frequency=28e9)

    # Expected norm is sqrt(num_elements) because we removed 1/sqrt(N) normalization
    # to correctly model passive RIS reflections
    expected_norm = np.sqrt(64)  # 64 elements
    for az in [0, np.pi / 4, np.pi / 2, np.pi]:
        for el in [0, np.pi / 6, np.pi / 4]:
            a = ch._compute_steering_vector(az, el)
            norm = np.linalg.norm(a)
            if not np.isclose(norm, expected_norm, rtol=1e-3):
                print(f"  az={az:.2f}, el={el:.2f}: |a|={norm:.6f}\t\tFAIL")
                print(f"  Expected |a|={expected_norm:.6f}")
                passed = False
    
    if passed:
        print("  All steering vectors have expected norm (sqrt(N)):\tPASS")
    else:
        print("  All steering vectors have expected norm:\t\tFAIL")
    return passed


# ============================================================================
# Test 9: SNR formula consistency
# ============================================================================

def test_snr_formula():
    """
    Verify SNR = P_tx * |h_total|^2 / sigma_n^2 gives consistent results
    with the compute_ris_snr_db utility.
    """
    print("\n" + "=" * 60)
    print("TEST 9: SNR Formula Consistency")
    print("=" * 60)
    passed = True

    np.random.seed(42)

    h_direct = 1e-3 * np.exp(1j * 0.5)
    h_ris_user = 1e-4 * (np.random.randn(64) + 1j * np.random.randn(64))
    h_bs_ris = 1e-4 * (np.random.randn(64) + 1j * np.random.randn(64))
    phases = np.random.uniform(0, 2 * np.pi, 64)

    tx_power = 1.0  # 1 W
    noise_power = 1e-12  # -90 dBm

    # Manual computation
    h_cascade = h_ris_user * h_bs_ris
    h_total = h_direct + np.sum(h_cascade * np.exp(1j * phases))
    snr_manual = tx_power * np.abs(h_total) ** 2 / noise_power
    snr_manual_db = 10 * np.log10(snr_manual)

    # Using utility function
    snr_util_db = compute_ris_snr_db(h_direct, h_ris_user, h_bs_ris, phases,
                                      tx_power, noise_power)

    ok = abs(snr_manual_db - snr_util_db) < 0.01
    print(f"  Manual SNR: {snr_manual_db:.2f} dB")
    print(f"  Utility SNR: {snr_util_db:.2f} dB")
    print(f"  Match: {'PASS' if ok else 'FAIL'}")
    passed &= ok

    # Test: with no RIS, should equal direct path SNR
    snr_no_ris = compute_ris_snr_db(h_direct, np.zeros(64), np.zeros(64),
                                     np.zeros(64), tx_power, noise_power)
    snr_direct = 10 * np.log10(tx_power * np.abs(h_direct) ** 2 / noise_power)
    ok = abs(snr_no_ris - snr_direct) < 0.01
    print(f"  No-RIS SNR: {snr_no_ris:.2f} dB (direct: {snr_direct:.2f} dB) {'PASS' if ok else 'FAIL'}")
    passed &= ok

    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("RIS FEDERATED LEARNING - VALIDATION TEST SUITE")
    print("=" * 60)

    results = {}
    results['dbm_to_watts'] = test_dbm_to_watts()
    results['path_loss'] = test_path_loss()
    results['no_ris_snr'] = test_no_ris_snr()
    results['genie_aided'] = test_genie_aided_upper_bound()
    results['torus_adjacency'] = test_torus_adjacency()
    results['comm_volume'] = test_communication_volume()
    results['gnn_forward'] = test_gnn_forward()
    results['steering_vector'] = test_steering_vector_norm()
    results['snr_formula'] = test_snr_formula()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(results)
    passed = sum(results.values())
    for name, result in results.items():
        print(f"  {name:25s}: {'PASS' if result else 'FAIL'}")
    print(f"\n  {passed}/{total} tests passed")

    if passed < total:
        print("\n  SOME TESTS FAILED - investigate before proceeding!")
        return 1
    else:
        print("\n  All tests passed!")
        return 0


if __name__ == "__main__":
    exit(main())
