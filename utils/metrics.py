"""
Metrics calculation for RIS Federated Learning
Wireless performance, learning metrics, and hardware efficiency
"""

import numpy as np
import torch
from scipy.stats import entropy


def dbm_to_watts(dbm):
    """
    Convert dBm to Watts.

    Formula: P_watts = 10^((P_dBm - 30) / 10)

    The -30 converts dBm -> dBW first, then to linear.
    Example: -90 dBm = -120 dBW = 1e-12 W

    Args:
        dbm: Power in dBm
    Returns:
        Power in Watts
    """
    return 10.0 ** ((dbm - 30.0) / 10.0)


def calculate_snr(signal_power, noise_power_dbm=-90):
    """
    Calculate Signal-to-Noise Ratio

    Args:
        signal_power: Signal power in Watts
        noise_power_dbm: Noise power in dBm

    Returns:
        SNR in dB
    """
    noise_power = dbm_to_watts(noise_power_dbm)
    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    return snr_db


def compute_ris_snr_db(h_direct, h_ris_user, h_bs_ris, phases, tx_power, noise_power):
    """
    Compute SNR for RIS-aided channel with cascaded model.

    h_total = h_direct + SUM(h_ris_user * h_bs_ris * exp(j * phases))
    SNR = tx_power * |h_total|^2 / noise_power

    Args:
        h_direct: Direct channel (scalar complex)
        h_ris_user: RIS-to-user channel (N,) complex
        h_bs_ris: BS-to-RIS channel (N,) complex
        phases: Phase shifts (N,) float in radians
        tx_power: Transmit power in Watts
        noise_power: Noise power in Watts

    Returns:
        SNR in dB
    """
    h_cascade = h_ris_user * h_bs_ris
    h_total = h_direct + np.sum(h_cascade * np.exp(1j * phases))
    signal = tx_power * np.abs(h_total) ** 2
    return 10 * np.log10(signal / noise_power)


def calculate_sinr(signal_power, interference_power, noise_power_dbm=-90):
    """
    Calculate Signal-to-Interference-plus-Noise Ratio

    Args:
        signal_power: Desired signal power
        interference_power: Interference power
        noise_power_dbm: Noise power in dBm

    Returns:
        SINR in dB
    """
    noise_power = dbm_to_watts(noise_power_dbm)
    sinr_linear = signal_power / (interference_power + noise_power)
    sinr_db = 10 * np.log10(sinr_linear)
    return sinr_db


def calculate_achievable_rate(snr_db):
    """
    Calculate achievable data rate using Shannon capacity

    Args:
        snr_db: SNR in dB

    Returns:
        Rate in bits/s/Hz
    """
    snr_linear = 10 ** (snr_db / 10)
    rate = np.log2(1 + snr_linear)
    return rate


def calculate_phase_error(predicted_phases, true_phases):
    """
    Calculate phase prediction error (circular metric)

    Args:
        predicted_phases: Predicted phase shifts (radians)
        true_phases: True optimal phase shifts (radians)

    Returns:
        Dictionary with error metrics
    """
    # Circular error
    error = np.abs(predicted_phases - true_phases)
    error = np.minimum(error, 2 * np.pi - error)  # Take shorter arc

    metrics = {
        'mean_error_rad': np.mean(error),
        'std_error_rad': np.std(error),
        'mean_error_deg': np.rad2deg(np.mean(error)),
        'std_error_deg': np.rad2deg(np.std(error)),
        'max_error_rad': np.max(error),
        'max_error_deg': np.rad2deg(np.max(error)),
        'rmse_rad': np.sqrt(np.mean(error ** 2)),
        'rmse_deg': np.rad2deg(np.sqrt(np.mean(error ** 2)))
    }

    return metrics


def calculate_beam_alignment(predicted_phases, true_phases, h_ris):
    """
    Calculate beam alignment quality

    Args:
        predicted_phases: Predicted phases
        true_phases: Optimal phases
        h_ris: RIS channel coefficients

    Returns:
        Alignment metrics
    """
    # Effective channel with predicted phases
    h_predicted = np.sum(h_ris * np.exp(1j * predicted_phases))

    # Effective channel with optimal phases
    h_optimal = np.sum(h_ris * np.exp(1j * true_phases))

    # Beam alignment (normalized correlation)
    alignment = np.abs(np.vdot(h_predicted, h_optimal)) / (np.abs(h_predicted) * np.abs(h_optimal))

    # Power gain ratio
    power_ratio = np.abs(h_predicted) ** 2 / np.abs(h_optimal) ** 2

    metrics = {
        'beam_alignment': alignment,
        'power_ratio': power_ratio,
        'power_loss_db': 10 * np.log10(power_ratio) if power_ratio > 0 else -np.inf
    }

    return metrics


def calculate_energy_efficiency(total_energy, achievable_rate):
    """
    Calculate energy efficiency (bits per Joule)

    Args:
        total_energy: Total energy consumed (Joules)
        achievable_rate: Data rate (bits/s/Hz)

    Returns:
        Energy efficiency metrics
    """
    if total_energy == 0:
        return {'energy_per_bit': np.inf, 'bits_per_joule': 0}

    metrics = {
        'energy_per_bit': total_energy / achievable_rate if achievable_rate > 0 else np.inf,
        'bits_per_joule': achievable_rate / total_energy if total_energy > 0 else 0,
        'total_energy_joules': total_energy,
        'total_energy_mj': total_energy * 1000
    }

    return metrics


def calculate_convergence_metrics(loss_history, threshold=0.01):
    """
    Analyze convergence behavior

    Args:
        loss_history: List of losses over rounds
        threshold: Convergence threshold (relative change)

    Returns:
        Convergence metrics
    """
    if len(loss_history) < 2:
        return {}

    # Find convergence point
    converged = False
    convergence_round = len(loss_history)

    for i in range(10, len(loss_history)):
        window = loss_history[i - 10:i]
        relative_change = np.std(window) / np.mean(window)
        if relative_change < threshold:
            converged = True
            convergence_round = i
            break

    # Calculate convergence rate
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    convergence_rate = (initial_loss - final_loss) / len(loss_history)

    metrics = {
        'converged': converged,
        'convergence_round': convergence_round,
        'convergence_speed': convergence_rate,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'total_reduction': initial_loss - final_loss,
        'reduction_percentage': ((initial_loss - final_loss) / initial_loss) * 100
    }

    return metrics


def calculate_communication_efficiency(bytes_transmitted, performance_gain):
    """
    Calculate communication efficiency

    Args:
        bytes_transmitted: Total bytes transmitted
        performance_gain: Performance improvement (e.g., SNR gain)

    Returns:
        Efficiency metrics
    """
    metrics = {
        'bytes_per_db_gain': bytes_transmitted / performance_gain if performance_gain > 0 else np.inf,
        'kb_per_db_gain': (bytes_transmitted / 1024) / performance_gain if performance_gain > 0 else np.inf,
        'total_bytes': bytes_transmitted,
        'total_kilobytes': bytes_transmitted / 1024,
        'total_megabytes': bytes_transmitted / (1024 * 1024)
    }

    return metrics


def calculate_noc_metrics(bytes_transmitted, bandwidth_gbps, num_rounds):
    """
    Calculate Network-on-Chip metrics

    Args:
        bytes_transmitted: Total bytes
        bandwidth_gbps: NoC bandwidth in Gbps
        num_rounds: Number of communication rounds

    Returns:
        NoC performance metrics
    """
    # Convert bandwidth to bytes/sec
    bandwidth_bytes_per_sec = bandwidth_gbps * 1e9 / 8

    # Average latency per round
    avg_bytes_per_round = bytes_transmitted / num_rounds if num_rounds > 0 else 0
    avg_latency_sec = avg_bytes_per_round / bandwidth_bytes_per_sec

    # Utilization
    total_time = num_rounds * 1.0  # Assume 1 sec per round
    transmission_time = bytes_transmitted / bandwidth_bytes_per_sec
    utilization = min(transmission_time / total_time, 1.0) if total_time > 0 else 0

    metrics = {
        'avg_packet_latency_sec': avg_latency_sec,
        'avg_packet_latency_ms': avg_latency_sec * 1000,
        'avg_packet_latency_us': avg_latency_sec * 1e6,
        'bandwidth_utilization': utilization,
        'peak_bandwidth_gbps': bandwidth_gbps,
        'avg_throughput_gbps': (bytes_transmitted * 8) / (total_time * 1e9) if total_time > 0 else 0
    }

    return metrics


def calculate_data_heterogeneity(datasets):
    """
    Measure data heterogeneity across clients (non-IID degree)

    Args:
        datasets: List of datasets from different clients

    Returns:
        Heterogeneity metrics
    """
    # Simple approach: compare feature distributions
    feature_means = []
    feature_stds = []

    for dataset in datasets:
        features = dataset.features
        feature_means.append(np.mean(features, axis=0))
        feature_stds.append(np.std(features, axis=0))

    feature_means = np.array(feature_means)
    feature_stds = np.array(feature_stds)

    # KL divergence between client distributions (simplified)
    mean_divergence = np.std(feature_means, axis=0).mean()
    std_divergence = np.std(feature_stds, axis=0).mean()

    metrics = {
        'mean_divergence': mean_divergence,
        'std_divergence': std_divergence,
        'heterogeneity_score': mean_divergence + std_divergence
    }

    return metrics


def calculate_fairness_index(client_performances):
    """
    Calculate Jain's fairness index across clients

    Args:
        client_performances: List of performance values for each client

    Returns:
        Fairness index (0 to 1, 1 is perfectly fair)
    """
    performances = np.array(client_performances)
    n = len(performances)

    numerator = (np.sum(performances)) ** 2
    denominator = n * np.sum(performances ** 2)

    fairness = numerator / denominator if denominator > 0 else 0

    return {
        'jains_index': fairness,
        'mean_performance': np.mean(performances),
        'std_performance': np.std(performances),
        'min_performance': np.min(performances),
        'max_performance': np.max(performances)
    }


def create_comparison_table(fl_metrics, baselines):
    """
    Create comparison table for paper

    Args:
        fl_metrics: Metrics from federated learning approach
        baselines: Dictionary with baseline metrics

    Returns:
        Formatted comparison dictionary
    """
    comparison = {
        'Method': [],
        'SNR (dB)': [],
        'Achievable Rate (bps/Hz)': [],
        'Communication (KB)': [],
        'Energy (mJ)': [],
        'Convergence Rounds': []
    }

    # Add baselines
    if 'no_ris' in baselines:
        comparison['Method'].append('No RIS')
        comparison['SNR (dB)'].append(f"{baselines['no_ris']['snr']:.2f}")
        comparison['Achievable Rate (bps/Hz)'].append(f"{baselines['no_ris']['rate']:.2f}")
        comparison['Communication (KB)'].append('0')
        comparison['Energy (mJ)'].append('0')
        comparison['Convergence Rounds'].append('N/A')

    if 'random_ris' in baselines:
        comparison['Method'].append('Random RIS')
        comparison['SNR (dB)'].append(f"{baselines['random_ris']['snr']:.2f}")
        comparison['Achievable Rate (bps/Hz)'].append(f"{baselines['random_ris']['rate']:.2f}")
        comparison['Communication (KB)'].append('0')
        comparison['Energy (mJ)'].append('0')
        comparison['Convergence Rounds'].append('N/A')

    if 'centralized' in baselines:
        comparison['Method'].append('Centralized Learning')
        comparison['SNR (dB)'].append(f"{baselines['centralized']['snr']:.2f}")
        comparison['Achievable Rate (bps/Hz)'].append(f"{baselines['centralized']['rate']:.2f}")
        comparison['Communication (KB)'].append(f"{baselines['centralized']['communication_kb']:.2f}")
        comparison['Energy (mJ)'].append(f"{baselines['centralized']['energy_mj']:.2f}")
        comparison['Convergence Rounds'].append(f"{baselines['centralized']['rounds']}")

    # Add FL approach
    comparison['Method'].append('Federated RIS (Ours)')
    comparison['SNR (dB)'].append(f"{fl_metrics['snr']:.2f}")
    comparison['Achievable Rate (bps/Hz)'].append(f"{fl_metrics['rate']:.2f}")
    comparison['Communication (KB)'].append(f"{fl_metrics['communication_kb']:.2f}")
    comparison['Energy (mJ)'].append(f"{fl_metrics['energy_mj']:.2f}")
    comparison['Convergence Rounds'].append(f"{fl_metrics['rounds']}")

    if 'optimal' in baselines:
        comparison['Method'].append('Genie-Aided Optimal')
        comparison['SNR (dB)'].append(f"{baselines['optimal']['snr']:.2f}")
        comparison['Achievable Rate (bps/Hz)'].append(f"{baselines['optimal']['rate']:.2f}")
        comparison['Communication (KB)'].append('N/A')
        comparison['Energy (mJ)'].append('N/A')
        comparison['Convergence Rounds'].append('N/A')

    return comparison


# ============ NoC Topology Metrics ============

def calculate_noc_topology_metrics(num_tiles, topology, bytes_transmitted, bandwidth_gbps, num_rounds):
    """
    Calculate NoC metrics for different topologies.
    
    Args:
        num_tiles: Number of tiles in the system
        topology: One of "Mesh", "Torus", "FoldedTorus", "Tree", "Butterfly"
        bytes_transmitted: Total bytes to transmit
        bandwidth_gbps: Available bandwidth in Gbps
        num_rounds: Number of FL rounds
    
    Returns:
        Dictionary with topology-specific metrics
    """
    # Calculate grid dimensions (assuming square-ish layout)
    grid_size = int(np.ceil(np.sqrt(num_tiles)))
    
    # Calculate average hop count based on topology
    if topology == "Mesh":
        # Manhattan distance average for mesh: (2/3) * (grid_size - 1)
        avg_hops = (2/3) * (grid_size - 1) * 2
        bisection_bandwidth_factor = 1.0 / grid_size  # Limited by bisection
        
    elif topology == "Torus":
        # Torus has wrap-around links, reducing average hops
        avg_hops = grid_size / 2  # Half of mesh
        bisection_bandwidth_factor = 2.0 / grid_size  # Better than mesh
        
    elif topology == "FoldedTorus":
        # Folded torus has physical locality advantages
        avg_hops = grid_size / 2.5  # Slightly better than regular torus
        bisection_bandwidth_factor = 2.5 / grid_size
        
    elif topology == "Tree":
        # Tree: O(log N) hops but bottleneck at root
        avg_hops = 2 * np.log2(num_tiles + 1)  # Up and down the tree
        bisection_bandwidth_factor = 1.0 / (num_tiles / 2)  # Root bottleneck
        
    elif topology == "Butterfly":
        # Butterfly: O(log N) stages
        avg_hops = np.log2(num_tiles)
        bisection_bandwidth_factor = 1.0  # Full bisection bandwidth
        
    else:
        raise ValueError(f"Unknown topology: {topology}")
    
    # Convert bandwidth to bytes/sec
    bandwidth_bytes_per_sec = bandwidth_gbps * 1e9 / 8
    
    # Effective bandwidth considering topology
    effective_bandwidth = bandwidth_bytes_per_sec * bisection_bandwidth_factor
    
    # Latency calculation
    bytes_per_round = bytes_transmitted / num_rounds if num_rounds > 0 else 0
    base_latency_sec = bytes_per_round / effective_bandwidth if effective_bandwidth > 0 else np.inf
    hop_latency = avg_hops * 1e-9  # 1ns per hop (typical NoC)
    total_latency_sec = base_latency_sec + hop_latency
    
    # Utilization
    total_transmission_time = bytes_transmitted / effective_bandwidth if effective_bandwidth > 0 else np.inf
    total_time = num_rounds * 1.0  # 1 sec per round assumption
    utilization = min(total_transmission_time / total_time, 1.0) if total_time > 0 else 0
    
    # Power consumption (topology-dependent)
    power_factors = {
        "Mesh": 1.0,
        "Torus": 1.2,  # More links
        "FoldedTorus": 1.3,
        "Tree": 0.7,   # Fewer links but concentrated
        "Butterfly": 1.5  # Complex routing
    }
    base_power_mw = 100 * num_tiles  # Base power in mW
    topology_power_mw = base_power_mw * power_factors.get(topology, 1.0)
    
    metrics = {
        'topology': topology,
        'avg_hops': avg_hops,
        'bisection_bandwidth_factor': bisection_bandwidth_factor,
        'effective_bandwidth_gbps': effective_bandwidth * 8 / 1e9,
        'avg_latency_sec': total_latency_sec,
        'avg_latency_ms': total_latency_sec * 1000,
        'avg_latency_us': total_latency_sec * 1e6,
        'bandwidth_utilization': utilization,
        'is_congested': utilization > 0.8,
        'power_mw': topology_power_mw,
        'power_w': topology_power_mw / 1000
    }
    
    return metrics


def compare_all_topologies(num_tiles, bytes_transmitted, bandwidth_gbps, num_rounds):
    """
    Compare all NoC topologies and rank them.
    
    Returns:
        Dictionary with comparison results and rankings
    """
    topologies = ["Mesh", "Torus", "FoldedTorus", "Tree", "Butterfly"]
    results = {}
    
    for topology in topologies:
        results[topology] = calculate_noc_topology_metrics(
            num_tiles, topology, bytes_transmitted, bandwidth_gbps, num_rounds
        )
    
    # Rank by latency (lower is better)
    latency_ranking = sorted(topologies, key=lambda t: results[t]['avg_latency_ms'])
    
    # Rank by utilization (lower is better for avoiding congestion)
    utilization_ranking = sorted(topologies, key=lambda t: results[t]['bandwidth_utilization'])
    
    # Rank by power (lower is better)
    power_ranking = sorted(topologies, key=lambda t: results[t]['power_mw'])
    
    return {
        'results': results,
        'latency_ranking': latency_ranking,
        'utilization_ranking': utilization_ranking,
        'power_ranking': power_ranking,
        'best_overall': latency_ranking[0]  # Simple heuristic
    }


# ============ Composite Optimization Score ============

def calculate_composite_score(snr_db, energy_mj, comm_kb, 
                              weight_snr=0.4, weight_energy=0.3, weight_comm=0.3,
                              snr_ref=70.0, energy_ref=60000.0, comm_ref=2000000.0):
    """
    Calculate composite optimization score.
    Higher score is better.
    
    Args:
        snr_db: Signal-to-noise ratio in dB
        energy_mj: Energy consumption in millijoules
        comm_kb: Communication overhead in kilobytes
        weight_snr: Weight for SNR component (higher SNR is better)
        weight_energy: Weight for energy component (lower energy is better)
        weight_comm: Weight for communication component (lower comm is better)
        snr_ref, energy_ref, comm_ref: Reference values for normalization
    
    Returns:
        Composite score (higher is better) and breakdown
    """
    # Normalize each component to [0, 1] range
    snr_norm = min(snr_db / snr_ref, 1.5)  # Cap at 1.5 for exceptionally good SNR
    energy_norm = min(energy_mj / energy_ref, 2.0)  # Lower is better, cap at 2
    comm_norm = min(comm_kb / comm_ref, 2.0)  # Lower is better, cap at 2
    
    # Calculate composite score
    # SNR contributes positively (higher is better)
    # Energy and comm contribute negatively (lower is better)
    score = weight_snr * snr_norm - weight_energy * energy_norm - weight_comm * comm_norm
    
    # Shift to positive range for easier interpretation
    score_positive = score + 1.0  # Now ranges roughly from 0 to 2
    
    return {
        'composite_score': score_positive,
        'raw_score': score,
        'snr_contribution': weight_snr * snr_norm,
        'energy_penalty': weight_energy * energy_norm,
        'comm_penalty': weight_comm * comm_norm,
        'snr_normalized': snr_norm,
        'energy_normalized': energy_norm,
        'comm_normalized': comm_norm
    }


# ============ Tile Efficiency Metrics ============

def calculate_tile_efficiency(snr_gain, energy_j, num_tiles):
    """
    Calculate SNR gain per tile per Joule.
    
    Args:
        snr_gain: Total SNR gain in dB
        energy_j: Total energy consumed in Joules
        num_tiles: Number of tiles
    
    Returns:
        Efficiency metrics
    """
    if energy_j <= 0 or num_tiles <= 0:
        return {'efficiency': 0, 'snr_per_tile': 0, 'energy_per_tile': 0}
    
    return {
        'efficiency': snr_gain / (num_tiles * energy_j),
        'snr_per_tile': snr_gain / num_tiles,
        'energy_per_tile_j': energy_j / num_tiles,
        'energy_per_tile_mj': (energy_j * 1000) / num_tiles
    }


def calculate_area_coverage(num_tiles, pixels_per_tile, chip_area_m2, wavelength):
    """
    Calculate RIS coverage metrics.
    
    Args:
        num_tiles: Number of tiles
        pixels_per_tile: Number of pixels (elements) per tile
        chip_area_m2: Chip/room area in square meters
        wavelength: Operating wavelength in meters
    
    Returns:
        Coverage metrics
    """
    # Each pixel is typically λ/2 × λ/2
    pixel_size = wavelength / 2
    pixel_area = pixel_size ** 2
    
    # Total RIS area
    total_pixels = num_tiles * pixels_per_tile
    total_ris_area = total_pixels * pixel_area
    
    # Coverage ratio
    coverage_ratio = total_ris_area / chip_area_m2 if chip_area_m2 > 0 else 0
    
    return {
        'total_pixels': total_pixels,
        'pixel_size_m': pixel_size,
        'pixel_area_m2': pixel_area,
        'total_ris_area_m2': total_ris_area,
        'total_ris_area_cm2': total_ris_area * 1e4,
        'chip_area_m2': chip_area_m2,
        'coverage_ratio': coverage_ratio,
        'coverage_percentage': coverage_ratio * 100
    }


def calculate_optimal_tiles_formula(chip_area_m2, pixels_per_tile, bandwidth_gbps, fl_rounds,
                                     target_utilization=0.8, target_snr_per_tile=8.0):
    """
    Calculate optimal number of tiles based on constraints (Golden Ratio).
    
    This is a heuristic formula derived from balancing:
    1. NoC bandwidth constraint
    2. Beamforming gain requirements
    3. Area coverage
    
    Args:
        chip_area_m2: Chip/room area in square meters
        pixels_per_tile: Pixels per tile
        bandwidth_gbps: Available NoC bandwidth
        fl_rounds: Number of FL rounds
        target_utilization: Target NoC utilization (default 80%)
        target_snr_per_tile: Target SNR contribution per tile (dB)
    
    Returns:
        Optimal tile count recommendations
    """
    # Constraint 1: NoC bandwidth
    # Each tile sends model weights proportional to pixels_per_tile
    # Communication per tile per round ≈ pixels_per_tile * hidden_dim * 4 bytes
    model_size_bytes = pixels_per_tile * 256 * 4  # Assuming hidden_dim=256
    comm_per_tile_per_round = model_size_bytes * 2  # Up and down
    
    available_bytes_per_sec = bandwidth_gbps * 1e9 / 8 * target_utilization
    max_tiles_noc = available_bytes_per_sec / (comm_per_tile_per_round * fl_rounds) if fl_rounds > 0 else 100
    
    # Constraint 2: Area scaling (heuristic)
    # More area needs more tiles, but with diminishing returns
    k_area = 0.5  # Scaling constant (to be tuned by experiments)
    tiles_area = k_area * np.sqrt(chip_area_m2)
    
    # Constraint 3: Beamforming gain
    # Target total SNR gain = num_tiles * target_snr_per_tile
    # This gives a minimum number of tiles needed
    target_total_snr = 60  # dB target
    min_tiles_snr = target_total_snr / target_snr_per_tile
    
    # Optimal is the minimum of the constraints (bottleneck)
    optimal_tiles = min(max_tiles_noc, tiles_area * 4)  # Scale up area estimate
    optimal_tiles = max(optimal_tiles, min_tiles_snr)  # But at least min for SNR
    
    # Round to nearest perfect square for grid layout
    sqrt_opt = np.sqrt(optimal_tiles)
    grid_size = max(2, int(np.round(sqrt_opt)))
    optimal_tiles_grid = grid_size ** 2
    
    return {
        'optimal_tiles_raw': optimal_tiles,
        'optimal_tiles_grid': optimal_tiles_grid,
        'optimal_grid_size': grid_size,
        'max_tiles_noc_constraint': max_tiles_noc,
        'tiles_area_heuristic': tiles_area * 4,
        'min_tiles_snr_constraint': min_tiles_snr,
        'bottleneck': 'noc' if max_tiles_noc < tiles_area * 4 else 'area'
    }


# ============ Sleep Scheduling Metrics ============

def calculate_sleep_energy_savings(num_tiles, num_rounds, active_power_w, sleep_power_w,
                                   sleep_ratio=0.3, round_duration_sec=1.0):
    """
    Calculate energy savings from sleep scheduling.
    
    Args:
        num_tiles: Number of tiles
        num_rounds: Number of FL rounds
        active_power_w: Power consumption when active (W)
        sleep_power_w: Power consumption when sleeping (W)
        sleep_ratio: Fraction of tiles sleeping on average
        round_duration_sec: Duration of each round in seconds
    
    Returns:
        Energy savings metrics
    """
    # Energy without sleep scheduling (all tiles always active)
    energy_no_sleep = num_tiles * active_power_w * num_rounds * round_duration_sec
    
    # Energy with sleep scheduling
    active_tiles = num_tiles * (1 - sleep_ratio)
    sleeping_tiles = num_tiles * sleep_ratio
    energy_with_sleep = (active_tiles * active_power_w + sleeping_tiles * sleep_power_w) * num_rounds * round_duration_sec
    
    # Savings
    energy_saved = energy_no_sleep - energy_with_sleep
    savings_percentage = (energy_saved / energy_no_sleep) * 100 if energy_no_sleep > 0 else 0
    
    return {
        'energy_no_sleep_j': energy_no_sleep,
        'energy_with_sleep_j': energy_with_sleep,
        'energy_saved_j': energy_saved,
        'energy_saved_mj': energy_saved * 1000,
        'savings_percentage': savings_percentage,
        'avg_active_tiles': active_tiles,
        'avg_sleeping_tiles': sleeping_tiles
    }