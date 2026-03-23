"""
Advanced Plotting Functions for FL-RIS Research Experiments
============================================================
Publication-quality figures for all 20 experiments.
Each function includes literature reference annotations.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from utils.plotting import (
    C, MARKERS, HATCHES, CMAP_SEQ, IEEE_RC,
    _save, _add_reference_note, _annotate_bars, _style_legend, _plot_with_ci,
)

plt.rcParams.update(IEEE_RC)


# ============================================================
# Experiments 1-8: Core ML Experiments
# ============================================================

def plot_local_epochs_analysis(results, save_path):
    """Exp 1: Impact of local epochs on convergence, communication, SNR, accuracy."""
    epochs = [r['local_epochs'] for r in results]
    conv = [r['convergence_round'] for r in results]
    comm = [r['total_communication_kb'] for r in results]
    snr = [r['final_snr'] for r in results]
    acc = [r['final_accuracy'] * 100 for r in results]

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(7.16, 5.0))

    configs = [
        (a1, conv, 'Convergence Rounds', '(a)', MARKERS[0], C[0]),
        (a2, comm, 'Communication (KB)', '(b)', MARKERS[1], C[1]),
        (a3, snr, 'Final SNR (dB)', '(c)', MARKERS[2], C[2]),
        (a4, acc, 'Accuracy (%)', '(d)', MARKERS[3], C[4]),
    ]
    for ax, data, ylabel, title, mk, c in configs:
        ax.plot(epochs, data, marker=mk, color=c, linewidth=1.5, markersize=5, zorder=3)
        ax.set_xlabel('Local Epochs ($E$)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9, loc='left')

    _add_reference_note(fig, 'local_epochs')
    _save(fig, save_path, 'local_epochs_analysis')


def plot_quantization_analysis(results, save_path):
    """Exp 2: Phase quantization impact on error and SNR."""
    labels, errors, snrs = [], [], []
    for r in results:
        b = r['quantization_bits']
        labels.append('Cont.' if b == 'continuous' else f'{b}-bit')
        errors.append(r['phase_error_deg'])
        snrs.append(r['final_snr'])

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.16, 2.6))
    x = range(len(labels))
    colors = [C[i % len(C)] for i in range(len(labels))]

    bars1 = a1.bar(x, errors, color=colors, edgecolor='black', linewidth=0.5, width=0.55, zorder=3)
    a1.set_xticks(list(x)); a1.set_xticklabels(labels)
    a1.set_ylabel('Phase Error (\u00b0)')
    a1.set_title('(a) Phase Error', fontsize=9, loc='left')
    _annotate_bars(a1, bars1, fmt='{:.1f}\u00b0', fontsize=6.5, offset=0.2)

    bars2 = a2.bar(x, snrs, color=colors, edgecolor='black', linewidth=0.5, width=0.55, zorder=3)
    a2.set_xticks(list(x)); a2.set_xticklabels(labels)
    a2.set_ylabel('SNR (dB)')
    a2.set_title('(b) SNR', fontsize=9, loc='left')
    _annotate_bars(a2, bars2, fmt='{:.1f}', fontsize=6.5, offset=0.2)

    a3.plot(errors, snrs, marker=MARKERS[0], color=C[4], linewidth=1.5, markersize=6, zorder=3)
    for i, lb in enumerate(labels):
        a3.annotate(lb, (errors[i], snrs[i]), textcoords='offset points',
                    xytext=(5, 5), fontsize=6.5,
                    arrowprops=dict(arrowstyle='-', color='#999999', lw=0.5))
    a3.set_xlabel('Phase Error (\u00b0)')
    a3.set_ylabel('SNR (dB)')
    a3.set_title('(c) Trade-off', fontsize=9, loc='left')

    _add_reference_note(fig, 'quantization')
    _save(fig, save_path, 'quantization_analysis')


def plot_compression_analysis(results, save_path):
    """Exp 3: Model compression impact on communication and accuracy."""
    names = [r['compression_name'] for r in results]
    comm = [r['total_communication_kb'] for r in results]
    acc = [r['final_accuracy'] * 100 for r in results]
    deg = [r['accuracy_degradation'] * 100 for r in results]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.16, 2.6))
    colors = [C[i % len(C)] for i in range(len(names))]

    configs = [
        (a1, comm, 'Communication (KB)', '(a) Communication'),
        (a2, acc, 'Accuracy (%)', '(b) Accuracy'),
        (a3, deg, 'Accuracy Loss (%)', '(c) Degradation'),
    ]
    for ax, data, ylabel, title in configs:
        bars = ax.bar(names, data, color=colors, edgecolor='black',
                      linewidth=0.5, width=0.5, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9, loc='left')
        ax.tick_params(axis='x', rotation=15)
        _annotate_bars(ax, bars, fmt='{:.1f}', fontsize=6)

    _add_reference_note(fig, 'compression')
    _save(fig, save_path, 'compression_analysis')


def plot_mobility_analysis(results, save_path):
    """Exp 4: User mobility impact on tracking and SNR."""
    names = [r['mobility_name'] for r in results]
    speeds = [r['mobility_speed'] for r in results]
    terr = [r['tracking_error'] for r in results]
    adapt = [r['adaptation_time'] for r in results]
    snr = [r['final_snr'] for r in results]

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(7.16, 5.0))

    a1.plot(speeds, terr, marker=MARKERS[0], color=C[1], linewidth=1.5, markersize=5, zorder=3)
    a1.set_xlabel('Speed (m/s)'); a1.set_ylabel('Tracking Error')
    a1.set_title('(a) Tracking Error', fontsize=9, loc='left')

    a2.plot(speeds, adapt, marker=MARKERS[1], color=C[4], linewidth=1.5, markersize=5, zorder=3)
    a2.set_xlabel('Speed (m/s)'); a2.set_ylabel('Adaptation (rounds)')
    a2.set_title('(b) Adaptation Time', fontsize=9, loc='left')

    a3.plot(speeds, snr, marker=MARKERS[2], color=C[0], linewidth=1.5, markersize=5, zorder=3)
    a3.set_xlabel('Speed (m/s)'); a3.set_ylabel('SNR (dB)')
    a3.set_title('(c) SNR vs. Speed', fontsize=9, loc='left')

    colors = [C[i % len(C)] for i in range(len(names))]
    bars = a4.bar(names, snr, color=colors, edgecolor='black', linewidth=0.5, width=0.5, zorder=3)
    a4.set_ylabel('SNR (dB)')
    a4.set_title('(d) Per-Scenario SNR', fontsize=9, loc='left')
    a4.tick_params(axis='x', rotation=15)
    _annotate_bars(a4, bars, fmt='{:.1f}', fontsize=6)

    _add_reference_note(fig, 'mobility')
    _save(fig, save_path, 'mobility_analysis')


def plot_noniid_analysis(results, save_path):
    """Exp 5: Non-IID heterogeneity impact on convergence and fairness."""
    alphas = [r['alpha'] for r in results]
    conv = [r['convergence_round'] for r in results]
    fair = [r.get('fairness_index', 0.8) for r in results]
    acc = [r['final_accuracy'] * 100 for r in results]

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.16, 2.6))

    a1.plot(alphas, conv, marker=MARKERS[0], color=C[1], linewidth=1.5, markersize=5, zorder=3)
    a1.set_xlabel(r'$\alpha$ (Dirichlet)'); a1.set_ylabel('Convergence Rounds')
    a1.set_title('(a) Convergence', fontsize=9, loc='left')
    a1.invert_xaxis()

    a2.plot(alphas, fair, marker=MARKERS[1], color=C[2], linewidth=1.5, markersize=5, zorder=3)
    a2.axhline(1.0, color='#999999', linestyle='--', linewidth=0.8, alpha=0.5)
    a2.set_xlabel(r'$\alpha$ (Dirichlet)'); a2.set_ylabel("Jain's Fairness Index")
    a2.set_title('(b) Fairness', fontsize=9, loc='left')
    a2.set_ylim(0, 1.1); a2.invert_xaxis()

    a3.plot(alphas, acc, marker=MARKERS[2], color=C[0], linewidth=1.5, markersize=5, zorder=3)
    a3.set_xlabel(r'$\alpha$ (Dirichlet)'); a3.set_ylabel('Accuracy (%)')
    a3.set_title('(c) Accuracy', fontsize=9, loc='left')
    a3.invert_xaxis()

    _add_reference_note(fig, 'noniid')
    _save(fig, save_path, 'noniid_analysis')


def plot_pilot_analysis(results, save_path):
    """Exp 6: Pilot overhead comparison."""
    methods = [r['method'] for r in results]
    pilots = [r['total_pilots'] for r in results]
    oh_kb = [r['overhead_kb'] for r in results]
    colors = [C[i % len(C)] for i in range(len(methods))]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.6))

    bars1 = a1.bar(methods, pilots, color=colors, edgecolor='black',
                   linewidth=0.5, width=0.5, zorder=3)
    a1.set_ylabel('Total Pilots')
    a1.set_yscale('log')
    a1.set_title('(a) Pilot Count (log scale)', fontsize=9, loc='left')

    bars2 = a2.bar(methods, oh_kb, color=colors, edgecolor='black',
                   linewidth=0.5, width=0.5, zorder=3)
    a2.set_ylabel('Overhead (KB)')
    a2.set_title('(b) Communication Overhead', fontsize=9, loc='left')
    _annotate_bars(a2, bars2, fmt='{:.1f}', fontsize=6)

    _add_reference_note(fig, 'pilot_overhead')
    _save(fig, save_path, 'pilot_overhead_analysis')


def plot_noc_traffic_analysis(results, save_path):
    """Exp 7: NoC traffic load vs power consumption."""
    tiles = [r['num_tiles'] for r in results]
    power = [r['total_power_mw'] for r in results]
    lat = [r['avg_latency_us'] for r in results]
    comm = [r['total_communication_kb'] for r in results]

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(7.16, 5.0))

    a1.plot(tiles, power, marker=MARKERS[0], color=C[1], label='Total', linewidth=1.5, zorder=3)
    a1.plot(tiles, [r['static_power_mw'] for r in results], marker=MARKERS[1], color=C[0],
            linestyle='--', label='Static', linewidth=1.2, zorder=3)
    a1.plot(tiles, [r['dynamic_power_mw'] for r in results], marker=MARKERS[2], color=C[2],
            linestyle='--', label='Dynamic', linewidth=1.2, zorder=3)
    a1.set_xlabel('Number of Tiles'); a1.set_ylabel('Power (mW)')
    a1.set_title('(a) Power Breakdown', fontsize=9, loc='left')
    _style_legend(a1, fontsize=7)

    a2.plot(tiles, lat, marker=MARKERS[1], color=C[4], linewidth=1.5, markersize=5, zorder=3)
    a2.set_xlabel('Number of Tiles'); a2.set_ylabel('Latency (\u03bcs)')
    a2.set_title('(b) Aggregation Latency', fontsize=9, loc='left')

    bars = a3.bar(tiles, comm, color=C[0], edgecolor='black', linewidth=0.4,
                  width=max(1, tiles[0] * 0.3) if tiles else 1, zorder=3)
    a3.set_xlabel('Number of Tiles'); a3.set_ylabel('Communication (KB)')
    a3.set_title('(c) Data Volume', fontsize=9, loc='left')

    ener_j = [p / 1000 * 0.1 for p in power]
    eff = [c * 1024 * 8 / e if e > 0 else 0 for c, e in zip(comm, ener_j)]
    a4.plot(tiles, eff, marker=MARKERS[2], color=C[3], linewidth=1.5, markersize=5, zorder=3)
    a4.set_xlabel('Number of Tiles'); a4.set_ylabel('Efficiency (bits/J)')
    a4.set_yscale('log')
    a4.set_title('(d) Energy Efficiency', fontsize=9, loc='left')

    _add_reference_note(fig, 'noc_traffic')
    _save(fig, save_path, 'noc_traffic_power_analysis')


def plot_approach_comparison(results, save_path):
    """Exp 8: Federated vs Centralized vs Local comparison with radar chart."""
    methods = [r['method'].replace('_', ' ').title() for r in results]
    comm = [r['total_communication_kb'] for r in results]
    acc = [r['final_accuracy'] * 100 for r in results]
    energy = [r['total_energy_mj'] for r in results]
    conv = [r['convergence_round'] for r in results]
    colors = [C[i % len(C)] for i in range(len(methods))]

    fig = plt.figure(figsize=(7.16, 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.45)

    panels = [
        (gs[0, 0], comm, 'Communication (KB)', '(a)', True),
        (gs[0, 1], acc, 'Accuracy (%)', '(b)', False),
        (gs[0, 2], energy, 'Energy (mJ)', '(c)', False),
        (gs[1, 0], conv, 'Convergence Rounds', '(d)', False),
    ]
    for gspec, data, ylabel, title, log in panels:
        ax = fig.add_subplot(gspec)
        bars = ax.bar(methods, data, color=colors, edgecolor='black',
                      linewidth=0.5, width=0.5, zorder=3)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9, loc='left')
        ax.tick_params(axis='x', rotation=15)
        if log:
            ax.set_yscale('log')

    # Radar chart
    ax5 = fig.add_subplot(gs[1, 1:], projection='polar')
    cats = ['Accuracy', 'Comm.\nEfficiency', 'Energy\nEfficiency', 'Convergence']
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist() + [0]

    for i, (m, c) in enumerate(zip(methods, colors)):
        vals = [
            acc[i] / 100,
            1 - comm[i] / max(comm) if max(comm) > 0 else 0,
            1 - energy[i] / max(energy) if max(energy) > 0 else 0,
            1 - conv[i] / max(conv) if max(conv) > 0 else 0,
        ] + [acc[i] / 100]
        ax5.plot(angles, vals, marker=MARKERS[i % len(MARKERS)], color=c,
                 linewidth=1.2, markersize=4, label=m, zorder=3)
        ax5.fill(angles, vals, alpha=0.06, color=c)

    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(cats, fontsize=7)
    ax5.set_ylim(0, 1)
    ax5.set_title('(e) Multi-Criteria Comparison', fontsize=9, pad=15)
    _style_legend(ax5, loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=6.5)

    _add_reference_note(fig, 'fl_vs_centralized')
    _save(fig, save_path, 'approach_comparison')


# ============================================================
# Experiments 9-10: Baseline Comparisons
# ============================================================

def plot_baseline_comparison(results, save_path):
    """Exp 9: Comprehensive baseline comparison (7 methods)."""
    if isinstance(results, list) and len(results) > 0:
        data = results[0] if isinstance(results[0], dict) else results
    else:
        data = results

    method_labels = {
        'no_ris': 'No RIS',
        'random_ris': 'Random RIS',
        'random_search': 'Random Search',
        'alternating_opt': 'Alt. Opt.',
        'centralized_dl': 'Centralized DL',
        'federated_ours': 'FL (Ours)',
    }

    methods = []
    snrs = []
    comms = []
    energies = []
    for key, label in method_labels.items():
        if key in data and isinstance(data[key], dict) and 'error' not in data[key]:
            methods.append(label)
            snrs.append(data[key].get('snr_db', 0))
            comms.append(data[key].get('communication_kb', 0))
            energies.append(data[key].get('energy_mj', 0))

    n = len(methods)
    colors = [C[i % len(C)] for i in range(n)]

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(7.16, 5.5))

    # SNR comparison
    bars1 = a1.bar(range(n), snrs, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    a1.set_xticks(range(n))
    a1.set_xticklabels(methods, rotation=30, ha='right', fontsize=6.5)
    a1.set_ylabel('SNR (dB)')
    a1.set_title('(a) SNR Comparison', fontsize=9, loc='left')
    _annotate_bars(a1, bars1, fmt='{:.1f}', fontsize=6, offset=0.2)

    # Communication cost
    bars2 = a2.bar(range(n), comms, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    a2.set_xticks(range(n))
    a2.set_xticklabels(methods, rotation=30, ha='right', fontsize=6.5)
    a2.set_ylabel('Communication (KB)')
    a2.set_title('(b) Communication Cost', fontsize=9, loc='left')
    if max(comms) > 10 * min(max(comms, default=1), 1):
        a2.set_yscale('symlog')

    # Energy cost
    bars3 = a3.bar(range(n), energies, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    a3.set_xticks(range(n))
    a3.set_xticklabels(methods, rotation=30, ha='right', fontsize=6.5)
    a3.set_ylabel('Energy (mJ)')
    a3.set_title('(c) Energy Cost', fontsize=9, loc='left')

    # Summary table
    a4.axis('off')
    table_data = []
    for i, m in enumerate(methods):
        table_data.append([m, f'{snrs[i]:.1f}', f'{comms[i]:.0f}', f'{energies[i]:.1f}'])
    table = a4.table(cellText=table_data,
                     colLabels=['Method', 'SNR (dB)', 'Comm (KB)', 'Energy (mJ)'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#e6e6e6')
            cell.set_text_props(weight='bold')
        cell.set_edgecolor('#cccccc')
    a4.set_title('(d) Summary', fontsize=9, loc='left')

    _add_reference_note(fig, 'baseline_comparison')
    _save(fig, save_path, 'baseline_comparison')


def plot_multiuser_comparison(results, save_path):
    """Exp 10: Multi-user MIMO scaling analysis."""
    if not results:
        return

    n_users = [r.get('num_users', i + 1) for i, r in enumerate(results)]
    sum_rates = [r.get('sum_rate', r.get('final_snr', 0)) for r in results]
    per_user_snr = [r.get('per_user_snr', r.get('final_snr', 0)) for r in results]
    fairness = [r.get('fairness_index', r.get('final_accuracy', 0.8)) for r in results]
    comms = [r.get('total_communication_kb', 0) for r in results]

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(7.16, 5.0))

    a1.plot(n_users, sum_rates, marker=MARKERS[0], color=C[0], linewidth=1.5, markersize=5, zorder=3)
    a1.set_xlabel('Number of Users')
    a1.set_ylabel('Sum Rate (bps/Hz)')
    a1.set_title('(a) Sum Rate Scaling', fontsize=9, loc='left')

    colors = [C[i % len(C)] for i in range(len(n_users))]
    bars = a2.bar(range(len(n_users)), per_user_snr, color=colors,
                  edgecolor='black', linewidth=0.5, zorder=3)
    a2.set_xticks(range(len(n_users)))
    a2.set_xticklabels([str(u) for u in n_users])
    a2.set_xlabel('Number of Users')
    a2.set_ylabel('Per-User SNR (dB)')
    a2.set_title('(b) Per-User SNR', fontsize=9, loc='left')

    a3.plot(n_users, fairness, marker=MARKERS[1], color=C[2], linewidth=1.5, markersize=5, zorder=3)
    a3.axhline(1.0, color='#999999', linestyle='--', linewidth=0.8, alpha=0.5)
    a3.set_xlabel('Number of Users')
    a3.set_ylabel("Jain's Fairness Index")
    a3.set_ylim(0, 1.1)
    a3.set_title('(c) Fairness', fontsize=9, loc='left')

    a4.plot(n_users, comms, marker=MARKERS[2], color=C[4], linewidth=1.5, markersize=5, zorder=3)
    a4.set_xlabel('Number of Users')
    a4.set_ylabel('Communication (KB)')
    a4.set_title('(d) Communication Cost', fontsize=9, loc='left')

    _add_reference_note(fig, 'multiuser')
    _save(fig, save_path, 'multiuser_comparison')


# ============================================================
# Experiments 11-13: Algorithm & Architecture Comparisons
# ============================================================

def plot_fl_algorithms_comparison(results, save_path):
    """Exp 11: FedAvg vs FedProx vs SCAFFOLD comparison."""
    if not results:
        return

    algo_names = []
    final_losses = []
    final_snrs = []
    total_comms = []
    conv_rounds = []

    for r in results:
        name = r.get('algorithm', r.get('method', f'Algo-{len(algo_names)}'))
        algo_names.append(name)
        final_losses.append(r.get('final_loss', 0))
        final_snrs.append(r.get('final_snr', 0))
        total_comms.append(r.get('total_communication_kb', 0))
        conv_rounds.append(r.get('convergence_round', 0))

    n = len(algo_names)
    colors = [C[i % len(C)] for i in range(n)]

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(7.16, 5.0))

    # Convergence curves (if round_metrics available)
    has_curves = any('round_metrics' in r for r in results)
    if has_curves:
        for i, r in enumerate(results):
            rm = r.get('round_metrics', [])
            if rm:
                rounds = [m['round'] + 1 for m in rm]
                losses = [m['avg_client_loss'] for m in rm]
                a1.plot(rounds, losses, marker=MARKERS[i % len(MARKERS)],
                        color=colors[i], markevery=max(1, len(rounds) // 8),
                        markersize=4, linewidth=1.5, label=algo_names[i], zorder=3)
        a1.set_xlabel('Communication Round')
        a1.set_ylabel('Loss (MSE)')
        _style_legend(a1, fontsize=7)
    else:
        bars = a1.bar(algo_names, final_losses, color=colors, edgecolor='black',
                      linewidth=0.5, zorder=3)
        a1.set_ylabel('Final Loss')
        _annotate_bars(a1, bars, fmt='{:.4f}', fontsize=6)
    a1.set_title('(a) Convergence', fontsize=9, loc='left')

    # SNR bars
    bars2 = a2.bar(algo_names, final_snrs, color=colors, edgecolor='black',
                   linewidth=0.5, zorder=3)
    a2.set_ylabel('Final SNR (dB)')
    a2.set_title('(b) SNR Performance', fontsize=9, loc='left')
    _annotate_bars(a2, bars2, fmt='{:.1f}', fontsize=6)

    # Communication bars
    bars3 = a3.bar(algo_names, total_comms, color=colors, edgecolor='black',
                   linewidth=0.5, zorder=3)
    a3.set_ylabel('Total Communication (KB)')
    a3.set_title('(c) Communication Cost', fontsize=9, loc='left')

    # Convergence rounds
    bars4 = a4.bar(algo_names, conv_rounds, color=colors, edgecolor='black',
                   linewidth=0.5, zorder=3)
    a4.set_ylabel('Convergence Round')
    a4.set_title('(d) Convergence Speed', fontsize=9, loc='left')

    _add_reference_note(fig, 'fl_algorithms')
    _save(fig, save_path, 'fl_algorithms_comparison')


def plot_architecture_comparison(results, save_path):
    """Exp 12: MLP vs GNN vs CNN+Attention vs Transformer comparison."""
    if not results:
        return

    arch_names = []
    final_snrs = []
    final_losses = []
    total_comms = []

    for r in results:
        name = r.get('architecture', r.get('model_type', f'Arch-{len(arch_names)}'))
        arch_names.append(name)
        final_snrs.append(r.get('final_snr', 0))
        final_losses.append(r.get('final_loss', 0))
        total_comms.append(r.get('total_communication_kb', 0))

    n = len(arch_names)
    colors = [C[i % len(C)] for i in range(n)]

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(7.16, 5.0))

    # SNR comparison
    bars1 = a1.bar(range(n), final_snrs, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    a1.set_xticks(range(n))
    a1.set_xticklabels(arch_names, rotation=20, ha='right', fontsize=7)
    a1.set_ylabel('SNR (dB)')
    a1.set_title('(a) SNR Performance', fontsize=9, loc='left')
    _annotate_bars(a1, bars1, fmt='{:.1f}', fontsize=6)

    # Convergence curves
    has_curves = any('round_metrics' in r for r in results)
    if has_curves:
        for i, r in enumerate(results):
            rm = r.get('round_metrics', [])
            if rm:
                rounds = [m['round'] + 1 for m in rm]
                losses = [m['avg_client_loss'] for m in rm]
                a2.plot(rounds, losses, marker=MARKERS[i % len(MARKERS)],
                        color=colors[i], markevery=max(1, len(rounds) // 8),
                        markersize=4, linewidth=1.5, label=arch_names[i], zorder=3)
        a2.set_xlabel('Communication Round')
        a2.set_ylabel('Loss (MSE)')
        _style_legend(a2, fontsize=6.5)
    else:
        bars = a2.bar(arch_names, final_losses, color=colors, edgecolor='black',
                      linewidth=0.5, zorder=3)
        a2.set_ylabel('Final Loss')
    a2.set_title('(b) Convergence', fontsize=9, loc='left')

    # Communication
    bars3 = a3.bar(range(n), total_comms, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    a3.set_xticks(range(n))
    a3.set_xticklabels(arch_names, rotation=20, ha='right', fontsize=7)
    a3.set_ylabel('Communication (KB)')
    a3.set_title('(c) Communication Cost', fontsize=9, loc='left')

    # Summary table
    a4.axis('off')
    table_data = []
    for i in range(n):
        table_data.append([arch_names[i], f'{final_snrs[i]:.1f}',
                           f'{final_losses[i]:.4f}', f'{total_comms[i]:.0f}'])
    table = a4.table(cellText=table_data,
                     colLabels=['Architecture', 'SNR (dB)', 'Loss', 'Comm (KB)'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.4)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#e6e6e6')
            cell.set_text_props(weight='bold')
        cell.set_edgecolor('#cccccc')
    a4.set_title('(d) Summary', fontsize=9, loc='left')

    _add_reference_note(fig, 'best_architecture_gnn')
    _save(fig, save_path, 'architecture_comparison')


def plot_csi_robustness(results, save_path):
    """Exp 13: CSI error variance impact on performance."""
    if not results:
        return

    variances = [r.get('csi_error_variance', r.get('error_variance', 0)) for r in results]
    snrs = [r.get('final_snr', 0) for r in results]
    losses = [r.get('final_loss', 0) for r in results]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.8))

    a1.plot(variances, snrs, marker=MARKERS[0], color=C[0], linewidth=1.5, markersize=6, zorder=3)
    a1.set_xlabel('CSI Error Variance')
    a1.set_ylabel('Final SNR (dB)')
    a1.set_title('(a) SNR vs. CSI Error', fontsize=9, loc='left')
    if variances[0] == 0:
        a1.axhline(snrs[0], color='#999999', linestyle='--', linewidth=0.8, alpha=0.5,
                   label=f'Perfect CSI: {snrs[0]:.1f} dB')
        _style_legend(a1, fontsize=7)

    a2.plot(variances, losses, marker=MARKERS[1], color=C[1], linewidth=1.5, markersize=6, zorder=3)
    a2.set_xlabel('CSI Error Variance')
    a2.set_ylabel('Final Loss (MSE)')
    a2.set_title('(b) Loss Degradation', fontsize=9, loc='left')

    _add_reference_note(fig, 'csi_robustness')
    _save(fig, save_path, 'csi_robustness_analysis')


# ============================================================
# Experiments 14-15: NoC Topology & Protocol
# ============================================================

def plot_topology_comparison(results, save_path):
    """Exp 14: 6 NoC topology comparison — latency, energy, hops, diameter, bisection BW, radar."""
    if not results:
        return

    topos = [r['topology'] if 'topology' in r else r.get('name', '') for r in results]
    latencies = [r.get('total_latency_ms', r.get('total_latency_us', 0) / 1000) for r in results]
    energies = [r.get('total_energy_uj', r.get('total_energy_nj', 0) / 1000) for r in results]
    hops = [r.get('avg_hops', 0) for r in results]
    diameters = [r.get('diameter', r.get('topology_diameter', 0)) for r in results]
    bisection = [r.get('bisection_bandwidth', r.get('topology_bisection_bw', 0)) for r in results]
    degrees = [r.get('degree', 0) for r in results]

    n = len(topos)
    colors = [C[i % len(C)] for i in range(n)]

    fig = plt.figure(figsize=(7.16, 5.5))
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.45)

    # Latency
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(range(n), latencies, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(topos, rotation=35, ha='right', fontsize=6)
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('(a) Total Latency', fontsize=9, loc='left')

    # Energy
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(range(n), energies, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(topos, rotation=35, ha='right', fontsize=6)
    ax2.set_ylabel('Energy (\u03bcJ)')
    ax2.set_title('(b) Total Energy', fontsize=9, loc='left')

    # Average hops
    ax3 = fig.add_subplot(gs[0, 2])
    bars3 = ax3.bar(range(n), hops, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    ax3.set_xticks(range(n))
    ax3.set_xticklabels(topos, rotation=35, ha='right', fontsize=6)
    ax3.set_ylabel('Average Hops')
    ax3.set_title('(c) Avg. Hop Count', fontsize=9, loc='left')
    _annotate_bars(ax3, bars3, fmt='{:.1f}', fontsize=6, offset=0.05)

    # Diameter
    ax4 = fig.add_subplot(gs[1, 0])
    bars4 = ax4.bar(range(n), diameters, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    ax4.set_xticks(range(n))
    ax4.set_xticklabels(topos, rotation=35, ha='right', fontsize=6)
    ax4.set_ylabel('Diameter')
    ax4.set_title('(d) Network Diameter', fontsize=9, loc='left')

    # Bisection bandwidth
    ax5 = fig.add_subplot(gs[1, 1])
    bars5 = ax5.bar(range(n), bisection, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    ax5.set_xticks(range(n))
    ax5.set_xticklabels(topos, rotation=35, ha='right', fontsize=6)
    ax5.set_ylabel('Bisection BW')
    ax5.set_title('(e) Bisection Bandwidth', fontsize=9, loc='left')

    # Radar chart
    ax6 = fig.add_subplot(gs[1, 2], projection='polar')
    cats = ['Low Latency', 'Low Energy', 'Low Hops', 'Low Diameter', 'High BW']
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist() + [0]

    for i in range(n):
        max_lat = max(latencies) if max(latencies) > 0 else 1
        max_eng = max(energies) if max(energies) > 0 else 1
        max_hop = max(hops) if max(hops) > 0 else 1
        max_dia = max(diameters) if max(diameters) > 0 else 1
        max_bis = max(bisection) if max(bisection) > 0 else 1
        vals = [
            1 - latencies[i] / max_lat,
            1 - energies[i] / max_eng,
            1 - hops[i] / max_hop,
            1 - diameters[i] / max_dia,
            bisection[i] / max_bis,
        ] + [1 - latencies[i] / max_lat]
        ax6.plot(angles, vals, marker=MARKERS[i % len(MARKERS)], color=colors[i],
                 linewidth=1.0, markersize=3, label=topos[i], zorder=3)
        ax6.fill(angles, vals, alpha=0.04, color=colors[i])

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(cats, fontsize=5.5)
    ax6.set_ylim(0, 1)
    ax6.set_title('(f) Multi-Criteria', fontsize=8, pad=12)
    _style_legend(ax6, loc='upper right', bbox_to_anchor=(1.5, 1.15), fontsize=5.5)

    _add_reference_note(fig, 'best_topology_torus')
    _save(fig, save_path, 'topology_comparison')


def plot_protocol_comparison(results, save_path):
    """Exp 15: Communication protocol comparison across topologies."""
    if not results:
        return

    # Group by topology
    topo_groups = {}
    for r in results:
        topo = r.get('topology', 'Unknown')
        proto = r.get('protocol', 'Unknown')
        if topo not in topo_groups:
            topo_groups[topo] = {}
        topo_groups[topo][proto] = r

    topologies = sorted(topo_groups.keys())
    all_protocols = sorted(set(p for t in topo_groups.values() for p in t.keys()))

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(7.16, 5.5))

    x = np.arange(len(topologies))
    width = 0.8 / max(len(all_protocols), 1)

    metrics = [
        (a1, 'total_latency_ms', 'Latency (ms)', '(a) Aggregation Latency'),
        (a2, 'total_energy_uj', 'Energy (\u03bcJ)', '(b) Energy Consumption'),
        (a3, 'total_bytes', 'Total Bytes', '(c) Data Transferred'),
        (a4, 'avg_utilization', 'Utilization', '(d) BW Utilization'),
    ]

    for ax, metric_key, ylabel, title in metrics:
        for j, proto in enumerate(all_protocols):
            vals = []
            for topo in topologies:
                r = topo_groups.get(topo, {}).get(proto, {})
                v = r.get(metric_key, 0)
                if metric_key == 'total_energy_uj' and v == 0:
                    v = r.get('total_energy_nj', 0) / 1000
                if metric_key == 'total_latency_ms' and v == 0:
                    v = r.get('total_latency_us', 0) / 1000
                vals.append(v)
            offset = (j - len(all_protocols) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width * 0.9, color=C[j % len(C)],
                   edgecolor='black', linewidth=0.3, label=proto if ax == a1 else '', zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(topologies, rotation=25, ha='right', fontsize=6.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9, loc='left')

    _style_legend(a1, fontsize=6, loc='upper right')

    _add_reference_note(fig, 'best_protocol_ringallreduce')
    _save(fig, save_path, 'protocol_comparison')


# ============================================================
# Experiment 16: Optimization Techniques
# ============================================================

def plot_optimization_comparison(results, save_path):
    """Exp 16: Optimization technique comparison (ADMM, SCA, SDR, DRL, Random, etc.)."""
    if not results:
        return

    # Handle dict format (method_name -> metrics)
    if isinstance(results, dict):
        methods = []
        snrs = []
        stds = []
        times = []
        for name, data in results.items():
            if isinstance(data, dict) and 'error' not in data:
                methods.append(name)
                snrs.append(data.get('avg_snr_db', 0))
                stds.append(data.get('std_snr_db', 0))
                times.append(data.get('avg_solve_time', 0))
    else:
        methods = [r.get('method', f'M{i}') for i, r in enumerate(results)]
        snrs = [r.get('avg_snr_db', r.get('final_snr', 0)) for r in results]
        stds = [r.get('std_snr_db', 0) for r in results]
        times = [r.get('avg_solve_time', 0) for r in results]

    n = len(methods)
    if n == 0:
        return

    colors = [C[i % len(C)] for i in range(n)]

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(7.16, 5.5))

    # SNR with error bars
    bars1 = a1.bar(range(n), snrs, yerr=stds, color=colors, edgecolor='black',
                   linewidth=0.5, capsize=3, error_kw={'linewidth': 0.8}, zorder=3)
    a1.set_xticks(range(n))
    a1.set_xticklabels(methods, rotation=30, ha='right', fontsize=7)
    a1.set_ylabel('Average SNR (dB)')
    a1.set_title('(a) SNR with Std. Dev.', fontsize=9, loc='left')

    # Solve time (log scale)
    times_plot = [max(t, 1e-6) for t in times]
    bars2 = a2.bar(range(n), times_plot, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    a2.set_xticks(range(n))
    a2.set_xticklabels(methods, rotation=30, ha='right', fontsize=7)
    a2.set_ylabel('Solve Time (s)')
    a2.set_yscale('log')
    a2.set_title('(b) Computational Cost (log)', fontsize=9, loc='left')

    # SNR vs solve time scatter
    for i in range(n):
        a3.scatter(times_plot[i], snrs[i], color=colors[i], marker=MARKERS[i % len(MARKERS)],
                   s=60, edgecolor='black', linewidth=0.5, zorder=3, label=methods[i])
        a3.annotate(methods[i], (times_plot[i], snrs[i]), textcoords='offset points',
                    xytext=(5, 5), fontsize=6)
    a3.set_xlabel('Solve Time (s)')
    a3.set_ylabel('SNR (dB)')
    a3.set_xscale('log')
    a3.set_title('(c) SNR vs. Complexity', fontsize=9, loc='left')

    # Summary table
    a4.axis('off')
    best_snr = max(snrs) if snrs else 0
    table_data = []
    for i in range(n):
        gap = best_snr - snrs[i]
        table_data.append([methods[i], f'{snrs[i]:.2f}', f'{stds[i]:.2f}',
                           f'{times[i]:.4f}', f'{gap:.2f}'])
    table = a4.table(cellText=table_data,
                     colLabels=['Method', 'SNR (dB)', 'Std', 'Time (s)', 'Gap (dB)'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.3)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#e6e6e6')
            cell.set_text_props(weight='bold')
        cell.set_edgecolor('#cccccc')
    a4.set_title('(d) Gap to Best', fontsize=9, loc='left')

    _add_reference_note(fig, 'best_optimizer_admm')
    _save(fig, save_path, 'optimization_comparison')


# ============================================================
# Experiments 17-20: System Optimization
# ============================================================

def plot_golden_ratio_analysis(results, save_path):
    """Exp 17: Tile-pixel golden ratio analysis — heatmaps and Pareto frontier."""
    if not results:
        return

    # Separate config results from golden ratio metadata
    configs = [r for r in results if '_golden_ratio' not in r]
    meta = [r for r in results if '_golden_ratio' in r]

    if not configs:
        return

    areas = sorted(set(r.get('chip_area_m2', 0) for r in configs))
    tiles_list = [r.get('num_tiles', 0) for r in configs]
    pixels_list = [r.get('pixels_per_tile', 0) for r in configs]
    snrs = [r.get('avg_snr_db', 0) for r in configs]
    scores = [r.get('composite_score', 0) for r in configs]
    energies = [r.get('comm_energy_nj', 0) for r in configs]

    n = len(configs)
    colors = [C[i % len(C)] for i in range(n)]

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(7.16, 5.5))

    # SNR bars by configuration
    labels = [f'T={tiles_list[i]},P={pixels_list[i]}' for i in range(n)]
    bars1 = a1.bar(range(n), snrs, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    a1.set_xticks(range(n))
    a1.set_xticklabels(labels, rotation=35, ha='right', fontsize=6)
    a1.set_ylabel('Average SNR (dB)')
    a1.set_title('(a) SNR by Configuration', fontsize=9, loc='left')

    # Composite score bars
    bars2 = a2.bar(range(n), scores, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    a2.set_xticks(range(n))
    a2.set_xticklabels(labels, rotation=35, ha='right', fontsize=6)
    a2.set_ylabel('Composite Score')
    a2.set_title('(b) Composite Score', fontsize=9, loc='left')
    _annotate_bars(a2, bars2, fmt='{:.2f}', fontsize=6, offset=0.01)

    # SNR vs Energy scatter (Pareto frontier)
    for i in range(n):
        a3.scatter(energies[i], snrs[i], color=colors[i], marker=MARKERS[i % len(MARKERS)],
                   s=60, edgecolor='black', linewidth=0.5, zorder=3)
        a3.annotate(labels[i], (energies[i], snrs[i]), textcoords='offset points',
                    xytext=(4, 4), fontsize=5.5)
    a3.set_xlabel('Communication Energy (nJ)')
    a3.set_ylabel('SNR (dB)')
    a3.set_title('(c) SNR vs. Energy Trade-off', fontsize=9, loc='left')

    # Summary table
    a4.axis('off')
    table_data = []
    for i in range(n):
        area = configs[i].get('chip_area_m2', '?')
        table_data.append([f'{area}', f'{tiles_list[i]}', f'{pixels_list[i]}',
                           f'{snrs[i]:.1f}', f'{scores[i]:.2f}'])
    table = a4.table(cellText=table_data,
                     colLabels=['Area (m\u00b2)', 'Tiles', 'Pixels', 'SNR (dB)', 'Score'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#e6e6e6')
            cell.set_text_props(weight='bold')
        cell.set_edgecolor('#cccccc')
    a4.set_title('(d) Configuration Summary', fontsize=9, loc='left')

    _add_reference_note(fig, 'golden_ratio')
    _save(fig, save_path, 'golden_ratio_analysis')


def plot_duty_cycling_analysis(results, save_path):
    """Exp 18: Dynamic duty cycling strategies comparison."""
    if not results:
        return

    strategies = [r.get('strategy', f'S{i}') for i, r in enumerate(results)]
    snrs = [r.get('avg_snr_db', 0) for r in results]
    stds = [r.get('std_snr_db', 0) for r in results]
    active = [r.get('avg_active_ratio', 1.0) for r in results]
    savings = [r.get('avg_energy_savings_pct', 0) for r in results]
    snr_loss = [abs(r.get('snr_loss_vs_full_db', 0)) for r in results]

    n = len(strategies)
    colors = [C[i % len(C)] for i in range(n)]

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(7.16, 5.0))

    # SNR with error bars
    bars1 = a1.bar(range(n), snrs, yerr=stds, color=colors, edgecolor='black',
                   linewidth=0.5, capsize=3, error_kw={'linewidth': 0.8}, zorder=3)
    a1.set_xticks(range(n))
    a1.set_xticklabels(strategies, rotation=30, ha='right', fontsize=6)
    a1.set_ylabel('Average SNR (dB)')
    a1.set_title('(a) SNR by Strategy', fontsize=9, loc='left')

    # Energy savings
    bars2 = a2.bar(range(n), savings, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    a2.set_xticks(range(n))
    a2.set_xticklabels(strategies, rotation=30, ha='right', fontsize=6)
    a2.set_ylabel('Energy Savings (%)')
    a2.set_title('(b) Energy Savings', fontsize=9, loc='left')
    _annotate_bars(a2, bars2, fmt='{:.0f}%', fontsize=6, offset=0.5)

    # SNR loss vs savings scatter
    for i in range(n):
        a3.scatter(savings[i], snr_loss[i], color=colors[i],
                   marker=MARKERS[i % len(MARKERS)], s=70, edgecolor='black',
                   linewidth=0.5, zorder=3, label=strategies[i])
    a3.set_xlabel('Energy Savings (%)')
    a3.set_ylabel('SNR Loss (dB)')
    a3.set_title('(c) Savings vs. SNR Loss', fontsize=9, loc='left')
    _style_legend(a3, fontsize=5.5, loc='upper left')

    # Summary table
    a4.axis('off')
    table_data = []
    for i in range(n):
        table_data.append([strategies[i], f'{snrs[i]:.2f}', f'{active[i]:.2f}',
                           f'{savings[i]:.0f}%', f'{snr_loss[i]:.4f}'])
    table = a4.table(cellText=table_data,
                     colLabels=['Strategy', 'SNR (dB)', 'Active', 'Savings', 'Loss (dB)'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.3)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#e6e6e6')
            cell.set_text_props(weight='bold')
        cell.set_edgecolor('#cccccc')
    a4.set_title('(d) Summary', fontsize=9, loc='left')

    _add_reference_note(fig, 'duty_cycling')
    _save(fig, save_path, 'duty_cycling_analysis')


def plot_dataset_comparison(results, save_path):
    """Exp 19: Channel dataset/scenario comparison."""
    if not results:
        return

    scenarios = [r.get('scenario', f'S{i}') for i, r in enumerate(results)]
    snrs = [r.get('avg_snr_optimal_db', 0) for r in results]
    stds = [r.get('std_snr_optimal_db', 0) for r in results]
    snrs_no = [r.get('avg_snr_no_ris_db', 0) for r in results]
    gains = [r.get('ris_gain_db', 0) for r in results]
    ch_gains = [r.get('avg_channel_gain', 0) for r in results]

    n = len(scenarios)
    colors = [C[i % len(C)] for i in range(n)]
    short_labels = [s[:18] + '...' if len(s) > 20 else s for s in scenarios]

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(7.16, 5.5))

    # Optimal SNR with error bars
    bars1 = a1.bar(range(n), snrs, yerr=stds, color=colors, edgecolor='black',
                   linewidth=0.5, capsize=3, error_kw={'linewidth': 0.8}, zorder=3)
    a1.set_xticks(range(n))
    a1.set_xticklabels(short_labels, rotation=35, ha='right', fontsize=5.5)
    a1.set_ylabel('Optimal SNR (dB)')
    a1.set_title('(a) Optimal SNR by Scenario', fontsize=9, loc='left')

    # RIS gain
    bars2 = a2.bar(range(n), gains, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    a2.set_xticks(range(n))
    a2.set_xticklabels(short_labels, rotation=35, ha='right', fontsize=5.5)
    a2.set_ylabel('RIS Gain (dB)')
    a2.set_title('(b) RIS Gain', fontsize=9, loc='left')

    # Channel gain
    bars3 = a3.bar(range(n), ch_gains, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    a3.set_xticks(range(n))
    a3.set_xticklabels(short_labels, rotation=35, ha='right', fontsize=5.5)
    a3.set_ylabel('Average Channel Gain')
    a3.set_title('(c) Channel Gain', fontsize=9, loc='left')

    # Summary table
    a4.axis('off')
    table_data = []
    for i in range(n):
        ds_type = results[i].get('type', 'N/A')
        table_data.append([short_labels[i], ds_type, f'{snrs[i]:.1f}',
                           f'{snrs_no[i]:.1f}', f'{gains[i]:.2f}'])
    table = a4.table(cellText=table_data,
                     colLabels=['Scenario', 'Type', 'Opt SNR', 'No RIS', 'Gain (dB)'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    table.scale(1.0, 1.3)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#e6e6e6')
            cell.set_text_props(weight='bold')
        cell.set_edgecolor('#cccccc')
    a4.set_title('(d) Summary', fontsize=9, loc='left')

    _add_reference_note(fig, 'dataset_comparison')
    _save(fig, save_path, 'dataset_comparison')


def plot_phase_quantization_detailed(results, save_path):
    """Exp 20: Phase quantization analysis across scenarios."""
    if not results:
        return

    # Handle both list and dict formats
    if isinstance(results, dict):
        items = list(results.items())
        labels = [str(k) for k, _ in items]
        snrs = [v.get('avg_snr_db', v.get('final_snr', 0)) for _, v in items]
        errors = [v.get('phase_error_deg', 0) for _, v in items]
    else:
        labels = []
        snrs = []
        errors = []
        for r in results:
            b = r.get('quantization_bits', r.get('bits', '?'))
            scenario = r.get('scenario', '')
            label = f'{b}-bit' if b != 'continuous' else 'Cont.'
            if scenario:
                label = f'{label} ({scenario[:6]})'
            labels.append(label)
            snrs.append(r.get('avg_snr_db', r.get('final_snr', 0)))
            errors.append(r.get('phase_error_deg', 0))

    n = len(labels)
    if n == 0:
        return

    colors = [C[i % len(C)] for i in range(n)]

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(7.16, 5.0))

    # SNR by quantization
    bars1 = a1.bar(range(n), snrs, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    a1.set_xticks(range(n))
    a1.set_xticklabels(labels, rotation=30, ha='right', fontsize=6)
    a1.set_ylabel('SNR (dB)')
    a1.set_title('(a) SNR by Quantization', fontsize=9, loc='left')

    # Phase error
    bars2 = a2.bar(range(n), errors, color=colors, edgecolor='black', linewidth=0.5, zorder=3)
    a2.set_xticks(range(n))
    a2.set_xticklabels(labels, rotation=30, ha='right', fontsize=6)
    a2.set_ylabel('Phase Error (\u00b0)')
    a2.set_title('(b) Phase Error', fontsize=9, loc='left')

    # Error vs SNR scatter
    for i in range(n):
        a3.scatter(errors[i], snrs[i], color=colors[i], marker=MARKERS[i % len(MARKERS)],
                   s=60, edgecolor='black', linewidth=0.5, zorder=3)
        a3.annotate(labels[i], (errors[i], snrs[i]), textcoords='offset points',
                    xytext=(4, 4), fontsize=5.5)
    a3.set_xlabel('Phase Error (\u00b0)')
    a3.set_ylabel('SNR (dB)')
    a3.set_title('(c) Error\u2013SNR Trade-off', fontsize=9, loc='left')

    # SNR loss from continuous
    if snrs:
        best_snr = max(snrs)
        losses = [best_snr - s for s in snrs]
        bars4 = a4.bar(range(n), losses, color=colors, edgecolor='black',
                       linewidth=0.5, zorder=3)
        a4.set_xticks(range(n))
        a4.set_xticklabels(labels, rotation=30, ha='right', fontsize=6)
        a4.set_ylabel('SNR Loss (dB)')
        a4.set_title('(d) SNR Loss vs. Best', fontsize=9, loc='left')

    _add_reference_note(fig, 'phase_quantization')
    _save(fig, save_path, 'phase_quantization_analysis')
