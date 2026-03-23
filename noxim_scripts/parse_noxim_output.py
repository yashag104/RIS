#!/usr/bin/env python3
"""
Parse Noxim stdout output files and extract key metrics into JSON/CSV.

Noxim output format includes lines like:
  % Total received packets: 12345
  % Total received flits: 67890
  % Global average delay (cycles): 42.5
  % Network throughput (flits/cycle): 0.123
  % Total energy (J): 1.23e-06
  ...and per-router stats when -detailed is used.
"""

import os
import re
import json
import csv
import sys
from pathlib import Path


def parse_noxim_output(filepath):
    """Parse a single Noxim output file and return metrics dict."""
    metrics = {
        'file': os.path.basename(filepath),
        'total_received_packets': 0,
        'total_received_flits': 0,
        'global_avg_delay_cycles': 0.0,
        'max_delay_cycles': 0.0,
        'network_throughput_flits_per_cycle': 0.0,
        'total_energy_j': 0.0,
        'dynamic_energy_j': 0.0,
        'static_energy_j': 0.0,
        'avg_buffer_utilization': 0.0,
        'max_buffer_utilization': 0.0,
    }

    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        metrics['error'] = str(e)
        return metrics

    # Extract metrics using regex patterns matching Noxim output
    patterns = {
        'total_received_packets': r'Total received packets:\s+(\d+)',
        'total_received_flits': r'Total received flits:\s+(\d+)',
        'global_avg_delay_cycles': r'Global average delay \(cycles\):\s+([\d.eE+-]+)',
        'max_delay_cycles': r'Max delay \(cycles\):\s+([\d.eE+-]+)',
        'network_throughput_flits_per_cycle': r'Network throughput \(flits/cycle\):\s+([\d.eE+-]+)',
        'total_energy_j': r'Total energy \(J\):\s+([\d.eE+-]+)',
        'dynamic_energy_j': r'Dynamic energy \(J\):\s+([\d.eE+-]+)',
        'static_energy_j': r'Static energy \(J\):\s+([\d.eE+-]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            val = match.group(1)
            try:
                metrics[key] = int(val) if '.' not in val and 'e' not in val.lower() else float(val)
            except ValueError:
                metrics[key] = val

    # Extract per-router stats if detailed mode was used
    router_stats = []
    router_pattern = r'Router\[(\d+)\]\[(\d+)\].*?delay.*?([\d.]+).*?throughput.*?([\d.]+)'
    for match in re.finditer(router_pattern, content, re.DOTALL):
        router_stats.append({
            'x': int(match.group(1)),
            'y': int(match.group(2)),
            'delay': float(match.group(3)),
            'throughput': float(match.group(4)),
        })
    
    if router_stats:
        metrics['per_router_stats'] = router_stats
        delays = [r['delay'] for r in router_stats if r['delay'] > 0]
        if delays:
            metrics['min_router_delay'] = min(delays)
            metrics['max_router_delay'] = max(delays)

    # Derive additional metrics
    if metrics['total_received_flits'] > 0 and metrics['global_avg_delay_cycles'] > 0:
        metrics['latency_us'] = metrics['global_avg_delay_cycles'] * 1e-3  # At 1 GHz: 1 cycle = 1 ns
        metrics['energy_per_flit_pj'] = (metrics['total_energy_j'] * 1e12 / 
                                          metrics['total_received_flits']) if metrics['total_energy_j'] > 0 else 0

    return metrics


def parse_all(results_dir):
    """Parse all Noxim output files in a directory."""
    results = []
    
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('.txt'):
            filepath = os.path.join(results_dir, fname)
            print(f"  Parsing: {fname}...", end=' ')
            metrics = parse_noxim_output(filepath)
            results.append(metrics)
            
            if metrics.get('total_received_flits', 0) > 0:
                print(f"OK (flits={metrics['total_received_flits']}, "
                      f"delay={metrics['global_avg_delay_cycles']:.1f}, "
                      f"throughput={metrics['network_throughput_flits_per_cycle']:.4f})")
            else:
                print(f"WARN: No data received ({metrics.get('error', 'check output')})")
    
    return results


def save_results(results, results_dir):
    """Save parsed results as JSON and CSV."""
    # JSON
    json_path = os.path.join(results_dir, 'noxim_results_parsed.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  JSON: {json_path}")
    
    # CSV (exclude per_router_stats for flat format)
    csv_path = os.path.join(results_dir, 'noxim_results_parsed.csv')
    flat_results = [{k: v for k, v in r.items() if k != 'per_router_stats'} for r in results]
    
    if flat_results:
        all_keys = set()
        for r in flat_results:
            all_keys.update(r.keys())
        all_keys = sorted(all_keys)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(flat_results)
        print(f"  CSV:  {csv_path}")
    
    # Print summary table
    print("\n" + "="*100)
    print(f"{'Experiment':<35s} {'Packets':>10s} {'Flits':>10s} {'Delay(cyc)':>12s} "
          f"{'Throughput':>12s} {'Energy(J)':>12s}")
    print("="*100)
    for r in results:
        print(f"{r['file']:<35s} "
              f"{r.get('total_received_packets',0):>10d} "
              f"{r.get('total_received_flits',0):>10d} "
              f"{r.get('global_avg_delay_cycles',0):>12.2f} "
              f"{r.get('network_throughput_flits_per_cycle',0):>12.6f} "
              f"{r.get('total_energy_j',0):>12.2e}")
    print("="*100)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'results', 'noxim_results')
    else:
        results_dir = sys.argv[1]
    
    if not os.path.isdir(results_dir):
        print(f"[ERROR] Results directory not found: {results_dir}")
        print(f"  Run run_all_noxim.sh first to generate results.")
        sys.exit(1)
    
    print(f"Parsing Noxim results from: {results_dir}\n")
    results = parse_all(results_dir)
    save_results(results, results_dir)
