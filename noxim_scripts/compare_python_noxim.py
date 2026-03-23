#!/usr/bin/env python3
"""
Compare Python analytical NoC results vs Noxim cycle-accurate results.
Generates comparison plots for the GlobeCom paper.
"""

import json
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# --- Python analytical results (from experiments) ---
PYTHON_RESULTS = {
    'topology': {
        'Mesh':  {'avg_hops': 2.67, 'latency_us': 4.194, 'energy_nj': 2654.21, 'utilization': 0.9375},
        'Torus': {'avg_hops': 2.00, 'latency_us': 4.194, 'energy_nj': 1769.47, 'utilization': 0.4687},
        'Ring':  {'avg_hops': 4.00, 'latency_us': 4.195, 'energy_nj': 3538.94, 'utilization': 0.9374},
        'Tree':  {'avg_hops': 4.00, 'latency_us': 4.194, 'energy_nj': 2101.25, 'utilization': 0.9375},
    },
    'protocol': {
        'ParameterServer': {'latency_us': 4.194, 'energy_nj': 2654.21, 'bytes': 9830400},
        'AllReduce':       {'latency_us': 2.097, 'energy_nj': 2654.21, 'bytes': 20971520},
        'RingAllReduce':   {'latency_us': 0.494, 'energy_nj': 1555.20, 'bytes': 9830400},
        'Gossip':          {'latency_us': 1.311, 'energy_nj': 5861.38, 'bytes': 26214400},
    },
    'noc_power': {
        2:  {'power_mw': 889.6, 'latency_us': 5.078},
        4:  {'power_mw': 1779.1, 'latency_us': 5.312},
        8:  {'power_mw': 3558.2, 'latency_us': 6.250},
        12: {'power_mw': 5337.4, 'latency_us': 7.812},
        16: {'power_mw': 7116.5, 'latency_us': 10.000},
    }
}


def load_noxim_results(results_dir):
    """Load parsed Noxim results from JSON."""
    json_path = os.path.join(results_dir, 'noxim_results_parsed.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return []


def plot_topology_comparison(noxim_results, output_dir):
    """Compare topology metrics: Python analytical vs Noxim."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Topology Comparison: Python Analytical vs Noxim Cycle-Accurate', fontsize=14, fontweight='bold')
    
    topologies = list(PYTHON_RESULTS['topology'].keys())
    python_latency = [PYTHON_RESULTS['topology'][t]['latency_us'] for t in topologies]
    python_energy = [PYTHON_RESULTS['topology'][t]['energy_nj'] for t in topologies]
    python_hops = [PYTHON_RESULTS['topology'][t]['avg_hops'] for t in topologies]
    
    # Find matching Noxim results
    noxim_latency = []
    noxim_energy = []
    for t in topologies:
        match = [r for r in noxim_results if t.lower() in r.get('file', '').lower()]
        if match:
            noxim_latency.append(match[0].get('global_avg_delay_cycles', 0) * 1e-3)  # cycles to µs at 1GHz
            noxim_energy.append(match[0].get('total_energy_j', 0) * 1e9)  # J to nJ
        else:
            noxim_latency.append(0)
            noxim_energy.append(0)
    
    x = np.arange(len(topologies))
    width = 0.35
    
    # Latency
    axes[0].bar(x - width/2, python_latency, width, label='Python Analytical', color='#2196F3', alpha=0.8)
    if any(v > 0 for v in noxim_latency):
        axes[0].bar(x + width/2, noxim_latency, width, label='Noxim Cycle-Accurate', color='#FF5722', alpha=0.8)
    axes[0].set_xlabel('Topology')
    axes[0].set_ylabel('Latency (µs)')
    axes[0].set_title('Communication Latency')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(topologies, rotation=45)
    axes[0].legend()
    
    # Energy
    axes[1].bar(x - width/2, python_energy, width, label='Python Analytical', color='#2196F3', alpha=0.8)
    if any(v > 0 for v in noxim_energy):
        axes[1].bar(x + width/2, noxim_energy, width, label='Noxim Cycle-Accurate', color='#FF5722', alpha=0.8)
    axes[1].set_xlabel('Topology')
    axes[1].set_ylabel('Energy (nJ)')
    axes[1].set_title('Total Energy')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(topologies, rotation=45)
    axes[1].legend()
    
    # Average Hops
    axes[2].bar(x, python_hops, color='#4CAF50', alpha=0.8)
    axes[2].set_xlabel('Topology')
    axes[2].set_ylabel('Average Hops')
    axes[2].set_title('Average Hop Count')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(topologies, rotation=45)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'topology_comparison_python_vs_noxim.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def plot_protocol_comparison(noxim_results, output_dir):
    """Compare protocol metrics: Python vs Noxim."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('FL Protocol Comparison: Python vs Noxim', fontsize=14, fontweight='bold')
    
    protocols = list(PYTHON_RESULTS['protocol'].keys())
    python_lat = [PYTHON_RESULTS['protocol'][p]['latency_us'] for p in protocols]
    python_energy = [PYTHON_RESULTS['protocol'][p]['energy_nj'] for p in protocols]
    
    noxim_lat = []
    noxim_energy = []
    for p in protocols:
        pname = p.lower().replace('_', '')
        match = [r for r in noxim_results if pname in r.get('file', '').lower().replace('_', '')]
        if match:
            noxim_lat.append(match[0].get('global_avg_delay_cycles', 0) * 1e-3)
            noxim_energy.append(match[0].get('total_energy_j', 0) * 1e9)
        else:
            noxim_lat.append(0)
            noxim_energy.append(0)
    
    x = np.arange(len(protocols))
    width = 0.35
    
    axes[0].bar(x - width/2, python_lat, width, label='Python', color='#2196F3', alpha=0.8)
    if any(v > 0 for v in noxim_lat):
        axes[0].bar(x + width/2, noxim_lat, width, label='Noxim', color='#FF5722', alpha=0.8)
    axes[0].set_ylabel('Latency (µs)')
    axes[0].set_title('Communication Latency per FL Round')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(protocols, rotation=30, ha='right')
    axes[0].legend()
    
    axes[1].bar(x - width/2, python_energy, width, label='Python', color='#2196F3', alpha=0.8)
    if any(v > 0 for v in noxim_energy):
        axes[1].bar(x + width/2, noxim_energy, width, label='Noxim', color='#FF5722', alpha=0.8)
    axes[1].set_ylabel('Energy (nJ)')
    axes[1].set_title('Total Communication Energy')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(protocols, rotation=30, ha='right')
    axes[1].legend()
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'protocol_comparison_python_vs_noxim.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def plot_injection_rate_sweep(noxim_results, output_dir):
    """Plot latency vs injection rate from Noxim sweep."""
    pir_results = [(r, float(r['file'].split('_')[-1].replace('.txt', '')))
                   for r in noxim_results if r.get('file', '').startswith('H_pir_')]
    
    if not pir_results:
        print("  [SKIP] No injection rate sweep results found")
        return
    
    pir_results.sort(key=lambda x: x[1])
    
    pirs = [x[1] for x in pir_results]
    delays = [x[0].get('global_avg_delay_cycles', 0) for x in pir_results]
    throughputs = [x[0].get('network_throughput_flits_per_cycle', 0) for x in pir_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(pirs, delays, 'o-', color='#2196F3', linewidth=2, markersize=8)
    ax1.set_xlabel('Packet Injection Rate')
    ax1.set_ylabel('Average Delay (cycles)')
    ax1.set_title('Latency vs Load')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(pirs, throughputs, 's-', color='#FF5722', linewidth=2, markersize=8)
    ax2.set_xlabel('Packet Injection Rate')
    ax2.set_ylabel('Throughput (flits/cycle)')
    ax2.set_title('Throughput vs Load')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('NoC Load Analysis (4×4 Mesh, Noxim)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'noxim_load_analysis.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


def plot_buffer_sweep(noxim_results, output_dir):
    """Plot effect of buffer depth on performance."""
    buf_results = [(r, int(r['file'].split('_')[-1].replace('.txt', '')))
                   for r in noxim_results if r.get('file', '').startswith('F_buffer_')]
    
    if not buf_results:
        print("  [SKIP] No buffer depth sweep results found")
        return
    
    buf_results.sort(key=lambda x: x[1])
    
    depths = [x[1] for x in buf_results]
    delays = [x[0].get('global_avg_delay_cycles', 0) for x in buf_results]
    energies = [x[0].get('total_energy_j', 0) * 1e9 for x in buf_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.bar(range(len(depths)), delays, tick_label=depths, color='#4CAF50', alpha=0.8)
    ax1.set_xlabel('Buffer Depth (flits)')
    ax1.set_ylabel('Average Delay (cycles)')
    ax1.set_title('Buffer Depth vs Latency')
    
    ax2.bar(range(len(depths)), energies, tick_label=depths, color='#FF9800', alpha=0.8)
    ax2.set_xlabel('Buffer Depth (flits)')
    ax2.set_ylabel('Total Energy (nJ)')
    ax2.set_title('Buffer Depth vs Energy')
    
    plt.suptitle('Buffer Depth Analysis (Noxim)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'noxim_buffer_analysis.pdf')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'results', 'noxim_results')
    else:
        results_dir = sys.argv[1]
    
    output_dir = results_dir
    
    print(f"Loading Noxim results from: {results_dir}")
    noxim_results = load_noxim_results(results_dir)
    
    if not noxim_results:
        print("[WARN] No Noxim results found. Generating Python-only plots...")
        noxim_results = []
    
    print(f"\nGenerating comparison plots...")
    plot_topology_comparison(noxim_results, output_dir)
    plot_protocol_comparison(noxim_results, output_dir)
    plot_injection_rate_sweep(noxim_results, output_dir)
    plot_buffer_sweep(noxim_results, output_dir)
    
    print("\nAll plots generated!")
