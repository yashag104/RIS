"""
Plotting utilities for RIS Federated Learning
Creates publication-quality figures
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import os

# Set style for publication-quality plots
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('seaborn-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def plot_convergence_curve(round_metrics, save_path=None):
    """
    Plot loss convergence over FL rounds
    """
    rounds = [m['round'] + 1 for m in round_metrics]
    avg_losses = [m['avg_client_loss'] for m in round_metrics]
    max_losses = [m['max_client_loss'] for m in round_metrics]
    min_losses = [m['min_client_loss'] for m in round_metrics]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(rounds, avg_losses, 'b-', linewidth=2, label='Average Loss')
    ax.fill_between(rounds, min_losses, max_losses, alpha=0.3, label='Min-Max Range')
    
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Convergence of Federated Learning')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'convergence_curve.png'), bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'convergence_curve.pdf'), bbox_inches='tight')
    plt.close()


def plot_snr_comparison(snr_metrics, save_path=None):
    """
    Plot SNR comparison: No RIS, Random RIS, Optimized RIS, Optimal
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart of mean SNR
    methods = ['No RIS', 'Random RIS', 'Optimized RIS', 'Optimal']
    mean_snrs = [
        snr_metrics['snr_no_ris_mean'],
        snr_metrics['snr_random_ris_mean'],
        snr_metrics['snr_optimized_ris_mean'],
        snr_metrics['snr_optimal_mean']
    ]
    
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    bars = ax1.bar(methods, mean_snrs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('SNR (dB)')
    ax1.set_title('Average SNR Comparison')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_snrs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f} dB', ha='center', va='bottom')
    
    # CDF plot
    for i, (method, color) in enumerate(zip(methods, colors)):
        if i == 0:
            data = snr_metrics['snr_no_ris_all']
        elif i == 1:
            data = snr_metrics['snr_random_ris_all']
        elif i == 2:
            data = snr_metrics['snr_optimized_ris_all']
        else:
            data = snr_metrics['snr_optimal_all']
        
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax2.plot(sorted_data, cdf, label=method, linewidth=2, color=color)
    
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('CDF')
    ax2.set_title('SNR Cumulative Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'snr_comparison.png'), bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'snr_comparison.pdf'), bbox_inches='tight')
    plt.close()


def plot_communication_overhead(round_metrics, save_path=None):
    """
    Plot communication overhead over rounds
    """
    rounds = [m['round'] + 1 for m in round_metrics]
    bytes_per_round = [m['total_bytes'] / 1024 for m in round_metrics]  # KB
    cumulative_bytes = np.cumsum(bytes_per_round)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Per-round communication
    ax1.bar(rounds, bytes_per_round, alpha=0.7, color='#3498db', edgecolor='black')
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Data Transmitted (KB)')
    ax1.set_title('Communication per Round')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Cumulative communication
    ax2.plot(rounds, cumulative_bytes, linewidth=2, color='#2ecc71', marker='o', markersize=4)
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Cumulative Data (KB)')
    ax2.set_title('Total Communication Overhead')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'communication_overhead.png'), bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'communication_overhead.pdf'), bbox_inches='tight')
    plt.close()


def plot_energy_consumption(round_metrics, save_path=None):
    """
    Plot energy consumption breakdown
    """
    rounds = [m['round'] + 1 for m in round_metrics]
    energy_per_round = [m['total_energy'] * 1000 for m in round_metrics]  # mJ
    cumulative_energy = np.cumsum(energy_per_round)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Per-round energy
    ax1.plot(rounds, energy_per_round, linewidth=2, color='#e74c3c', marker='s', markersize=4)
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Energy Consumption (mJ)')
    ax1.set_title('Energy per Round')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative energy
    ax2.plot(rounds, cumulative_energy, linewidth=2, color='#9b59b6', marker='o', markersize=4)
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Cumulative Energy (mJ)')
    ax2.set_title('Total Energy Consumption')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'energy_consumption.png'), bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'energy_consumption.pdf'), bbox_inches='tight')
    plt.close()


def plot_tradeoff_curves(round_metrics, snr_metrics_per_round, save_path=None):
    """
    Plot trade-off between accuracy and communication/energy
    """
    rounds = [m['round'] + 1 for m in round_metrics]
    losses = [m['avg_client_loss'] for m in round_metrics]
    comm_kb = np.cumsum([m['total_bytes'] / 1024 for m in round_metrics])
    energy_mj = np.cumsum([m['total_energy'] * 1000 for m in round_metrics])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy vs Communication
    ax1.plot(comm_kb, losses, linewidth=2, color='#3498db', marker='o', markersize=4)
    ax1.set_xlabel('Cumulative Communication (KB)')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Accuracy vs Communication Trade-off')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Lower loss is better
    
    # Accuracy vs Energy
    ax2.plot(energy_mj, losses, linewidth=2, color='#e74c3c', marker='s', markersize=4)
    ax2.set_xlabel('Cumulative Energy (mJ)')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_title('Accuracy vs Energy Trade-off')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'tradeoff_curves.png'), bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'tradeoff_curves.pdf'), bbox_inches='tight')
    plt.close()


def plot_beam_pattern(predicted_phases, metadata, save_path=None, filename='beam_pattern.png'):
    """
    Plot radiation/beam pattern (polar plot)
    """
    # Extract RIS position and user positions
    ris_pos = metadata['ris_position']
    user_pos = metadata['user_positions'][0]  # Target user
    
    # Calculate expected angle to user
    direction = user_pos - ris_pos
    expected_angle = np.arctan2(direction[1], direction[0])
    
    # Simulate beam pattern
    angles = np.linspace(0, 2 * np.pi, 360)
    num_elements = len(predicted_phases)
    
    # Array response (simplified)
    pattern = []
    for angle in angles:
        # Steering vector
        response = 0
        for i in range(num_elements):
            elem_phase = predicted_phases[i]
            # Simplified array factor
            response += np.exp(1j * (elem_phase + i * np.pi * np.cos(angle - expected_angle)))
        pattern.append(np.abs(response) ** 2)
    
    pattern = np.array(pattern)
    pattern = pattern / np.max(pattern)  # Normalize
    pattern_db = 10 * np.log10(pattern + 1e-10)
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
    ax.plot(angles, pattern_db, linewidth=2, color='#2ecc71')
    ax.fill(angles, pattern_db, alpha=0.3, color='#2ecc71')
    
    # Mark expected direction
    ax.plot([expected_angle, expected_angle], [pattern_db.min(), 0], 'r--', linewidth=2, label='Target Direction')
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(pattern_db.min(), 5)
    ax.set_title('RIS Beam Pattern (Normalized)', pad=20)
    ax.legend(loc='upper right')
    ax.grid(True)
    
    if save_path:
        plt.savefig(os.path.join(save_path, filename), bbox_inches='tight')
        plt.savefig(os.path.join(save_path, filename.replace('.png', '.pdf')), bbox_inches='tight')
    plt.close()


def plot_client_performance(client_metrics_per_round, save_path=None):
    """
    Plot per-client performance over rounds
    """
    num_clients = len(client_metrics_per_round[0]['client_metrics'])
    num_rounds = len(client_metrics_per_round)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for client_id in range(num_clients):
        losses = []
        for round_metric in client_metrics_per_round:
            client_loss = round_metric['client_metrics'][client_id]['avg_loss']
            losses.append(client_loss)
        
        rounds = list(range(1, num_rounds + 1))
        ax.plot(rounds, losses, linewidth=2, marker='o', markersize=3, label=f'Tile {client_id}')
    
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Per-Tile Learning Performance')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'client_performance.png'), bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'client_performance.pdf'), bbox_inches='tight')
    plt.close()


def plot_phase_heatmap(predicted_phases, optimal_phases, grid_size=(8, 8), save_path=None):
    """
    Plot phase shift heatmaps for visualization
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Reshape to grid
    pred_grid = predicted_phases[:grid_size[0] * grid_size[1]].reshape(grid_size)
    opt_grid = optimal_phases[:grid_size[0] * grid_size[1]].reshape(grid_size)
    error_grid = np.abs(pred_grid - opt_grid)
    error_grid = np.minimum(error_grid, 2 * np.pi - error_grid)  # Circular error
    
    # Predicted phases
    im1 = ax1.imshow(pred_grid, cmap='hsv', vmin=0, vmax=2*np.pi)
    ax1.set_title('Predicted Phase Shifts')
    ax1.set_xlabel('Element X')
    ax1.set_ylabel('Element Y')
    plt.colorbar(im1, ax=ax1, label='Phase (rad)')
    
    # Optimal phases
    im2 = ax2.imshow(opt_grid, cmap='hsv', vmin=0, vmax=2*np.pi)
    ax2.set_title('Optimal Phase Shifts')
    ax2.set_xlabel('Element X')
    ax2.set_ylabel('Element Y')
    plt.colorbar(im2, ax=ax2, label='Phase (rad)')
    
    # Error
    im3 = ax3.imshow(error_grid, cmap='hot', vmin=0, vmax=np.pi)
    ax3.set_title('Phase Error')
    ax3.set_xlabel('Element X')
    ax3.set_ylabel('Element Y')
    plt.colorbar(im3, ax=ax3, label='Error (rad)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'phase_heatmap.png'), bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'phase_heatmap.pdf'), bbox_inches='tight')
    plt.close()


def plot_noc_metrics(comm_summary, save_path=None):
    """
    Plot Network-on-Chip metrics
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Communication breakdown (pie chart)
    labels = ['Uplink (Tiles→BS)', 'Downlink (BS→Tiles)']
    sizes = [comm_summary['total_bytes_received'], comm_summary['total_bytes_sent']]
    colors = ['#3498db', '#2ecc71']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Communication Breakdown')
    
    # Bandwidth utilization (gauge-style bar)
    utilization = comm_summary['bandwidth_utilization'] * 100
    ax2.barh(['Bandwidth'], [utilization], color='#e74c3c', alpha=0.7)
    ax2.barh(['Bandwidth'], [100 - utilization], left=[utilization], color='#ecf0f1', alpha=0.7)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Utilization (%)')
    ax2.set_title(f'NoC Bandwidth Utilization: {utilization:.2f}%')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Latency
    latency_ms = comm_summary['avg_packet_latency_ms']
    ax3.bar(['Avg Packet\nLatency'], [latency_ms], color='#9b59b6', alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Latency (ms)')
    ax3.set_title('Average Packet Latency')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Total data transferred
    total_mb = comm_summary['total_megabytes']
    ax4.bar(['Total Data\nTransferred'], [total_mb], color='#f39c12', alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Data (MB)')
    ax4.set_title('Total Communication Volume')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, 'noc_metrics.png'), bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'noc_metrics.pdf'), bbox_inches='tight')
    plt.close()


def create_summary_dashboard(all_metrics, save_path=None):
    """
    Create comprehensive dashboard with all key metrics
    """
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Convergence
    ax1 = fig.add_subplot(gs[0, :2])
    rounds = [m['round'] + 1 for m in all_metrics['round_metrics']]
    losses = [m['avg_client_loss'] for m in all_metrics['round_metrics']]
    ax1.plot(rounds, losses, linewidth=2, color='#3498db', marker='o', markersize=4)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.set_title('Learning Convergence')
    ax1.grid(True, alpha=0.3)
    
    # 2. SNR Comparison
    ax2 = fig.add_subplot(gs[0, 2])
    snr_data = all_metrics['snr_metrics']
    methods = ['No RIS', 'Random', 'Learned', 'Optimal']
    snrs = [snr_data['snr_no_ris_mean'], snr_data['snr_random_ris_mean'],
            snr_data['snr_optimized_ris_mean'], snr_data['snr_optimal_mean']]
    ax2.bar(range(len(methods)), snrs, color=['#e74c3c', '#f39c12', '#2ecc71', '#3498db'], alpha=0.7)
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.set_ylabel('SNR (dB)')
    ax2.set_title('SNR Performance')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Communication
    ax3 = fig.add_subplot(gs[1, 0])
    comm_kb = np.cumsum([m['total_bytes'] / 1024 for m in all_metrics['round_metrics']])
    ax3.plot(rounds, comm_kb, linewidth=2, color='#9b59b6', marker='s', markersize=4)
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Cumulative (KB)')
    ax3.set_title('Communication Overhead')
    ax3.grid(True, alpha=0.3)
    
    # 4. Energy
    ax4 = fig.add_subplot(gs[1, 1])
    energy_mj = np.cumsum([m['total_energy'] * 1000 for m in all_metrics['round_metrics']])
    ax4.plot(rounds, energy_mj, linewidth=2, color='#e74c3c', marker='d', markersize=4)
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Cumulative (mJ)')
    ax4.set_title('Energy Consumption')
    ax4.grid(True, alpha=0.3)
    
    # 5. Per-client performance
    ax5 = fig.add_subplot(gs[1, 2])
    num_clients = len(all_metrics['round_metrics'][0]['client_metrics'])
    for i in range(num_clients):
        client_losses = [m['client_metrics'][i]['avg_loss'] for m in all_metrics['round_metrics']]
        ax5.plot(rounds, client_losses, linewidth=1.5, alpha=0.7, label=f'T{i}')
    ax5.set_xlabel('Round')
    ax5.set_ylabel('Loss')
    ax5.set_title('Per-Tile Performance')
    ax5.legend(ncol=2, fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Key metrics summary (text)
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    summary_text = f"""
    FEDERATED LEARNING SUMMARY
    
    Learning:  Converged in {all_metrics['convergence']['converged_round']} rounds | Final Loss: {all_metrics['convergence']['final_loss']:.6f} | Reduction: {all_metrics['convergence']['reduction_percentage']:.1f}%
    
    Wireless:  SNR Gain: {snr_data['snr_gain_over_no_ris']:.2f} dB | Achievable Rate: {all_metrics['achievable_rate_mean']:.2f} bps/Hz | Optimality Gap: {snr_data['optimality_gap']:.2f} dB
    
    Hardware:  Total Comm: {all_metrics['comm_summary']['total_kilobytes']:.2f} KB | Avg Latency: {all_metrics['comm_summary']['avg_packet_latency_ms']:.3f} ms | Energy: {energy_mj[-1]:.2f} mJ
    """
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'summary_dashboard.png'), bbox_inches='tight')
        plt.savefig(os.path.join(save_path, 'summary_dashboard.pdf'), bbox_inches='tight')
    plt.close()