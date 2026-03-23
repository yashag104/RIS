"""
Results Analysis Script
Analyze and compare results from multiple FL runs
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd


def load_run_results(run_dir):
    """Load results from a specific run"""
    metrics_path = os.path.join(run_dir, 'metrics.pkl')
    config_path = os.path.join(run_dir, 'config.json')

    if not os.path.exists(metrics_path):
        return None

    with open(metrics_path, 'rb') as f:
        metrics = pickle.load(f)

    with open(config_path, 'r') as f:
        config = json.load(f)

    return {
        'metrics': metrics,
        'config': config,
        'run_dir': run_dir
    }


def compare_runs(results_dir='results/'):
    """Compare all runs in the results directory"""
    runs = []

    # Find all run directories
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path) and item.startswith('run_'):
            result = load_run_results(item_path)
            if result:
                runs.append(result)

    if not runs:
        print("No runs found!")
        return

    print(f"Found {len(runs)} runs to compare\n")

    # Create comparison dataframe
    comparison_data = []

    for i, run in enumerate(runs):
        metrics = run['metrics']
        config = run['config']

        # Extract key metrics
        row = {
            'Run': i + 1,
            'Timestamp': os.path.basename(run['run_dir']).replace('run_', ''),
            'Tiles': config['NUM_TILES'],
            'Elements/Tile': config['ELEMENTS_PER_TILE'],
            'FL Rounds': config['FL_ROUNDS'],
            'Converged Round': metrics['convergence']['converged_round'],
            'Final Loss': metrics['final_evaluation']['loss'],
            'SNR (dB)': metrics['snr_metrics']['snr_optimized_ris_mean'],
            'SNR Gain (dB)': metrics['snr_metrics']['snr_gain_over_no_ris'],
            'Rate (bps/Hz)': metrics['achievable_rate_mean'],
            'Comm (MB)': metrics['comm_summary']['total_megabytes'],
            'Energy (mJ)': sum([m['total_energy'] * 1000 for m in metrics['round_metrics']]),
            'Latency (ms)': metrics['comm_summary']['avg_packet_latency_ms']
        }

        comparison_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Display comparison
    print("=" * 100)
    print("RUN COMPARISON")
    print("=" * 100)
    print(df.to_string(index=False))
    print("\n")

    # Statistical summary
    print("=" * 100)
    print("STATISTICAL SUMMARY")
    print("=" * 100)

    numeric_cols = ['Final Loss', 'SNR (dB)', 'SNR Gain (dB)', 'Rate (bps/Hz)',
                    'Comm (MB)', 'Energy (mJ)', 'Latency (ms)']

    summary = df[numeric_cols].describe()
    print(summary)
    print("\n")

    # Best performing run
    print("=" * 100)
    print("BEST PERFORMING RUNS")
    print("=" * 100)

    best_snr_idx = df['SNR (dB)'].idxmax()
    best_energy_idx = df['Energy (mJ)'].idxmin()
    best_comm_idx = df['Comm (MB)'].idxmin()
    best_convergence_idx = df['Converged Round'].idxmin()

    print(f"\nBest SNR: Run {df.iloc[best_snr_idx]['Run']} - {df.iloc[best_snr_idx]['SNR (dB)']:.2f} dB")
    print(f"Lowest Energy: Run {df.iloc[best_energy_idx]['Run']} - {df.iloc[best_energy_idx]['Energy (mJ)']:.2f} mJ")
    print(f"Lowest Comm: Run {df.iloc[best_comm_idx]['Run']} - {df.iloc[best_comm_idx]['Comm (MB)']:.2f} MB")
    print(
        f"Fastest Convergence: Run {df.iloc[best_convergence_idx]['Run']} - {df.iloc[best_convergence_idx]['Converged Round']} rounds")

    # Save comparison
    df.to_csv(os.path.join(results_dir, 'runs_comparison.csv'), index=False)
    print(f"\n✓ Comparison saved to {results_dir}/runs_comparison.csv")

    return df, runs


def plot_runs_comparison(df, save_dir='results/'):
    """Create comparison plots across runs"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # SNR comparison
    ax = axes[0, 0]
    ax.bar(df['Run'], df['SNR (dB)'], alpha=0.7, color='#3498db', edgecolor='black')
    ax.set_xlabel('Run')
    ax.set_ylabel('SNR (dB)')
    ax.set_title('SNR Performance Across Runs')
    ax.grid(True, alpha=0.3, axis='y')

    # Energy comparison
    ax = axes[0, 1]
    ax.bar(df['Run'], df['Energy (mJ)'], alpha=0.7, color='#e74c3c', edgecolor='black')
    ax.set_xlabel('Run')
    ax.set_ylabel('Energy (mJ)')
    ax.set_title('Energy Consumption Across Runs')
    ax.grid(True, alpha=0.3, axis='y')

    # Communication comparison
    ax = axes[0, 2]
    ax.bar(df['Run'], df['Comm (MB)'], alpha=0.7, color='#2ecc71', edgecolor='black')
    ax.set_xlabel('Run')
    ax.set_ylabel('Communication (MB)')
    ax.set_title('Communication Overhead Across Runs')
    ax.grid(True, alpha=0.3, axis='y')

    # Convergence comparison
    ax = axes[1, 0]
    ax.bar(df['Run'], df['Converged Round'], alpha=0.7, color='#f39c12', edgecolor='black')
    ax.set_xlabel('Run')
    ax.set_ylabel('Rounds to Convergence')
    ax.set_title('Convergence Speed Across Runs')
    ax.grid(True, alpha=0.3, axis='y')

    # Rate comparison
    ax = axes[1, 1]
    ax.bar(df['Run'], df['Rate (bps/Hz)'], alpha=0.7, color='#9b59b6', edgecolor='black')
    ax.set_xlabel('Run')
    ax.set_ylabel('Rate (bps/Hz)')
    ax.set_title('Achievable Rate Across Runs')
    ax.grid(True, alpha=0.3, axis='y')

    # Latency comparison
    ax = axes[1, 2]
    ax.bar(df['Run'], df['Latency (ms)'], alpha=0.7, color='#1abc9c', edgecolor='black')
    ax.set_xlabel('Run')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Packet Latency Across Runs')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'runs_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(save_dir, 'runs_comparison.pdf'), bbox_inches='tight')
    plt.close()

    print(f"✓ Comparison plots saved to {save_dir}/runs_comparison.png")


def analyze_convergence_patterns(runs, save_dir='results/'):
    """Analyze convergence patterns across runs"""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, run in enumerate(runs):
        metrics = run['metrics']
        rounds = [m['round'] + 1 for m in metrics['round_metrics']]
        losses = [m['avg_client_loss'] for m in metrics['round_metrics']]

        ax.plot(rounds, losses, linewidth=2, marker='o', markersize=3,
                alpha=0.7, label=f"Run {i + 1}")

    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Average Loss')
    ax.set_title('Convergence Patterns Across All Runs')
    ax.legend(ncol=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'convergence_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Convergence comparison saved to {save_dir}/convergence_comparison.png")


def generate_latex_table(df, save_dir='results/'):
    """Generate LaTeX table for paper"""
    latex_table = df.to_latex(
        index=False,
        float_format="%.2f",
        caption="Comparison of Federated Learning Runs for Distributed RIS Tiles",
        label="tab:fl_comparison"
    )

    with open(os.path.join(save_dir, 'comparison_table.tex'), 'w') as f:
        f.write(latex_table)

    print(f"✓ LaTeX table saved to {save_dir}/comparison_table.tex")


def main():
    """Main analysis function"""
    print("\n" + "=" * 100)
    print("FEDERATED LEARNING RESULTS ANALYSIS")
    print("=" * 100 + "\n")

    # Compare all runs
    df, runs = compare_runs()

    if df is None or len(runs) == 0:
        return

    # Generate comparison plots
    plot_runs_comparison(df)

    # Analyze convergence
    analyze_convergence_patterns(runs)

    # Generate LaTeX table
    generate_latex_table(df)

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()