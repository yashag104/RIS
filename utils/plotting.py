"""
Publication-Quality Plotting for FL-RIS Research
=================================================
IEEE-standard figures with ColorBrewer palettes, confidence intervals,
literature reference annotations, and consistent professional styling.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

from utils.references import get_figure_annotation

# ==================== IEEE Publication Style ====================
IEEE_RC = {
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.grid': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.constrained_layout.use': True,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
}
plt.rcParams.update(IEEE_RC)

# ==================== ColorBrewer Palette ====================
# Set1 qualitative palette — colorblind-safe, high contrast
COLORS = {
    'blue':     '#377eb8',
    'red':      '#e41a1c',
    'green':    '#4daf4a',
    'orange':   '#ff7f00',
    'purple':   '#984ea3',
    'brown':    '#a65628',
    'pink':     '#f781bf',
    'gray':     '#666666',
}
C = list(COLORS.values())
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']
HATCHES = ['', '//', '\\\\', 'xx', '..', '++', 'oo', '**']

# Sequential colormaps for heatmaps (perceptually uniform, colorblind-safe)
CMAP_SEQ = 'viridis'
CMAP_DIV = 'RdBu_r'
CMAP_PHASE = 'twilight'


# ==================== Helpers ====================

def _save(fig, save_path, name):
    """Save figure as both PDF (vector) and PNG (raster)."""
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f'{name}.pdf'), format='pdf')
    fig.savefig(os.path.join(save_path, f'{name}.png'), format='png')
    plt.close(fig)


def _add_reference_note(fig, experiment_tag, y_offset=-0.03):
    """Add literature reference annotation as a footnote below the figure."""
    note = get_figure_annotation(experiment_tag)
    if note:
        fig.text(0.5, y_offset, note,
                 ha='center', va='top', fontsize=5.5,
                 style='italic', color='#555555',
                 transform=fig.transFigure)


def _annotate_bars(ax, bars, fmt='{:.1f}', fontsize=6.5, offset=0.3):
    """Add value labels above bar chart bars."""
    for bar in bars:
        height = bar.get_height()
        if np.isfinite(height):
            ax.text(bar.get_x() + bar.get_width() / 2, height + offset,
                    fmt.format(height), ha='center', va='bottom', fontsize=fontsize)


def _style_legend(ax, **kwargs):
    """Apply consistent legend styling."""
    defaults = dict(frameon=True, fancybox=False, edgecolor='#cccccc',
                    framealpha=0.9, borderpad=0.4, handlelength=1.5)
    defaults.update(kwargs)
    return ax.legend(**defaults)


def _plot_with_ci(ax, x, y_mean, y_std=None, color='#377eb8', label=None,
                  marker='o', markevery=None):
    """Plot line with optional shaded confidence band."""
    if markevery is None:
        markevery = max(1, len(x) // 8)
    line, = ax.plot(x, y_mean, color=color, marker=marker, markevery=markevery,
                    linewidth=1.5, markersize=4.5, label=label, zorder=3)
    if y_std is not None:
        y_lo = np.array(y_mean) - np.array(y_std)
        y_hi = np.array(y_mean) + np.array(y_std)
        ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.12, zorder=2)
    return line


# ==================== Core Plot Functions ====================

def plot_convergence_curve(round_metrics, save_path=None):
    """FL convergence: average loss with min-max band."""
    rounds = [m['round'] + 1 for m in round_metrics]
    avg = [m['avg_client_loss'] for m in round_metrics]
    lo = [m['min_client_loss'] for m in round_metrics]
    hi = [m['max_client_loss'] for m in round_metrics]

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(rounds, avg, color=C[0], marker=MARKERS[0], markevery=3,
            markersize=4.5, label='Average Loss', zorder=3)
    ax.fill_between(rounds, lo, hi, color=C[0], alpha=0.12,
                    label='Min\u2013Max Range', zorder=2)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Loss (MSE)')
    _style_legend(ax, loc='upper right')
    _add_reference_note(fig, 'fl_vs_centralized')

    if save_path:
        _save(fig, save_path, 'convergence_curve')


def plot_snr_comparison(snr_metrics, save_path=None):
    """SNR bar chart + CDF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))

    methods = ['No RIS', 'Random RIS', 'FL-Optimized', 'MRC Optimal']
    snrs = [snr_metrics['snr_no_ris_mean'], snr_metrics['snr_random_ris_mean'],
            snr_metrics['snr_optimized_ris_mean'], snr_metrics['snr_optimal_mean']]
    colors = [C[7], C[3], C[0], C[2]]

    bars = ax1.bar(methods, snrs, color=colors, edgecolor='black',
                   linewidth=0.5, width=0.55, zorder=3)
    _annotate_bars(ax1, bars, offset=0.3)
    ax1.set_ylabel('SNR (dB)')
    ax1.set_title('(a) Average SNR', fontsize=9)
    ax1.tick_params(axis='x', rotation=20)

    # CDF subplot
    keys = ['snr_no_ris_all', 'snr_random_ris_all',
            'snr_optimized_ris_all', 'snr_optimal_all']
    for i, (key, label, c) in enumerate(zip(keys, methods, colors)):
        d = np.sort(snr_metrics[key])
        cdf = np.arange(1, len(d) + 1) / len(d)
        ax2.plot(d, cdf, color=c, marker=MARKERS[i],
                 markevery=max(1, len(d) // 8), markersize=4,
                 label=label, linewidth=1.2, zorder=3)
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('CDF')
    ax2.set_title('(b) SNR Distribution', fontsize=9)
    _style_legend(ax2, fontsize=6.5, loc='lower right')
    _add_reference_note(fig, 'baseline_comparison')

    if save_path:
        _save(fig, save_path, 'snr_comparison')


def plot_communication_overhead(round_metrics, save_path=None):
    """Per-round and cumulative communication."""
    rounds = [m['round'] + 1 for m in round_metrics]
    kb = [m['total_bytes'] / 1024 for m in round_metrics]
    cum = np.cumsum(kb).tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))

    bars = ax1.bar(rounds, kb, color=C[0], edgecolor='black',
                   linewidth=0.3, width=0.7, zorder=3)
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Data Transmitted (KB)')
    ax1.set_title('(a) Per-Round', fontsize=9)

    ax2.plot(rounds, cum, color=C[2], marker=MARKERS[1], markevery=3,
             markersize=4.5, linewidth=1.5, zorder=3)
    ax2.fill_between(rounds, 0, cum, color=C[2], alpha=0.08, zorder=2)
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Cumulative Data (KB)')
    ax2.set_title('(b) Cumulative', fontsize=9)
    _add_reference_note(fig, 'compression')

    if save_path:
        _save(fig, save_path, 'communication_overhead')


def plot_energy_consumption(round_metrics, save_path=None):
    """Energy per round and cumulative."""
    rounds = [m['round'] + 1 for m in round_metrics]
    e = [m['total_energy'] * 1000 for m in round_metrics]
    cum = np.cumsum(e).tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))

    ax1.plot(rounds, e, color=C[1], marker=MARKERS[1], markevery=3,
             markersize=4.5, zorder=3)
    ax1.fill_between(rounds, 0, e, color=C[1], alpha=0.08, zorder=2)
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Energy per Round (mJ)')
    ax1.set_title('(a) Per-Round Energy', fontsize=9)

    ax2.plot(rounds, cum, color=C[4], marker=MARKERS[2], markevery=3,
             markersize=4.5, zorder=3)
    ax2.fill_between(rounds, 0, cum, color=C[4], alpha=0.08, zorder=2)
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Cumulative Energy (mJ)')
    ax2.set_title('(b) Cumulative Energy', fontsize=9)
    _add_reference_note(fig, 'duty_cycling')

    if save_path:
        _save(fig, save_path, 'energy_consumption')


def plot_tradeoff_curves(round_metrics, snr_metrics_per_round, save_path=None):
    """Loss vs communication and energy trade-offs."""
    losses = [m['avg_client_loss'] for m in round_metrics]
    comm = np.cumsum([m['total_bytes'] / 1024 for m in round_metrics]).tolist()
    energy = np.cumsum([m['total_energy'] * 1000 for m in round_metrics]).tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.8))

    ax1.plot(comm, losses, color=C[0], marker=MARKERS[0], markevery=3,
             markersize=4.5, zorder=3)
    ax1.set_xlabel('Cumulative Communication (KB)')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('(a) Loss vs. Communication', fontsize=9)

    ax2.plot(energy, losses, color=C[1], marker=MARKERS[1], markevery=3,
             markersize=4.5, zorder=3)
    ax2.set_xlabel('Cumulative Energy (mJ)')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_title('(b) Loss vs. Energy', fontsize=9)
    _add_reference_note(fig, 'fl_vs_centralized')

    if save_path:
        _save(fig, save_path, 'tradeoff_curves')


def plot_beam_pattern(predicted_phases, metadata, save_path=None, filename='beam_pattern'):
    """Polar beam pattern of the RIS."""
    ris_pos = metadata['ris_position']
    user_pos = metadata['user_positions'][0]
    direction = user_pos - ris_pos
    expected_angle = np.arctan2(direction[1], direction[0])

    angles = np.linspace(0, 2 * np.pi, 360)
    N = len(predicted_phases)
    pattern = np.array([
        np.abs(sum(np.exp(1j * (predicted_phases[i] + i * np.pi * np.cos(a - expected_angle)))
                   for i in range(N)))**2
        for a in angles
    ])
    pattern /= pattern.max()
    pattern_db = 10 * np.log10(pattern + 1e-10)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(3.5, 3.5))
    ax.plot(angles, pattern_db, linewidth=1.2, color=C[0], zorder=3)
    ax.fill(angles, pattern_db, alpha=0.10, color=C[0])
    ax.plot([expected_angle] * 2, [pattern_db.min(), 0], '--',
            color=C[1], linewidth=1.2, label='Target Direction', zorder=4)
    ax.set_ylim(pattern_db.min(), 5)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    _style_legend(ax, loc='upper right', fontsize=7)
    _add_reference_note(fig, 'baseline_comparison')

    if save_path:
        _save(fig, save_path, 'beam_pattern')


def plot_client_performance(client_metrics_per_round, save_path=None):
    """Per-tile loss trajectories."""
    n_clients = len(client_metrics_per_round[0]['client_metrics'])
    n_rounds = len(client_metrics_per_round)
    rounds = list(range(1, n_rounds + 1))

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for cid in range(n_clients):
        losses = [rm['client_metrics'][cid]['avg_loss'] for rm in client_metrics_per_round]
        ax.plot(rounds, losses, linewidth=1.0, alpha=0.75,
                marker=MARKERS[cid % len(MARKERS)],
                markevery=max(1, n_rounds // 5), markersize=3,
                color=C[cid % len(C)], label=f'Tile {cid}')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Loss (MSE)')
    _style_legend(ax, ncol=4, fontsize=5.5, loc='upper right')
    _add_reference_note(fig, 'fl_vs_centralized')

    if save_path:
        _save(fig, save_path, 'client_performance')


def plot_phase_heatmap(predicted_phases, optimal_phases, grid_size=(8, 8), save_path=None):
    """Phase shift heatmaps: predicted, optimal, error with RMSE."""
    n = grid_size[0] * grid_size[1]
    pred = predicted_phases[:n].reshape(grid_size)
    opt = optimal_phases[:n].reshape(grid_size)
    err = np.minimum(np.abs(pred - opt), 2 * np.pi - np.abs(pred - opt))
    rmse = np.sqrt(np.mean(err**2))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.16, 2.4))
    configs = [
        (ax1, pred, '(a) FL-Predicted', CMAP_PHASE, (0, 2 * np.pi)),
        (ax2, opt, '(b) MRC-Optimal', CMAP_PHASE, (0, 2 * np.pi)),
        (ax3, err, f'(c) Phase Error (RMSE={rmse:.2f} rad)', 'inferno', (0, np.pi)),
    ]
    for ax, data, title, cmap, vr in configs:
        im = ax.imshow(data, cmap=cmap, vmin=vr[0], vmax=vr[1],
                       aspect='equal', interpolation='nearest')
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Element X')
        ax.set_ylabel('Element Y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _add_reference_note(fig, 'baseline_comparison')

    if save_path:
        _save(fig, save_path, 'phase_heatmap')


def plot_noc_metrics(comm_summary, save_path=None, round_metrics=None):
    """NoC metrics dashboard with per-round breakdowns."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.16, 5.0))

    if round_metrics and len(round_metrics) > 1:
        rounds = [m['round'] + 1 for m in round_metrics]
        per_round_kb = [m['total_bytes'] / 1024 for m in round_metrics]
        cum_kb = np.cumsum(per_round_kb).tolist()

        # (a) Per-round uplink vs downlink stacked bar
        # Each round: server sends model (downlink) and receives updates (uplink)
        n_clients = round_metrics[0].get('num_clients', 16)
        model_bytes = per_round_kb[0] / 2  # approx half up, half down
        uplink_kb = [model_bytes] * len(rounds)
        downlink_kb = [pk - model_bytes for pk in per_round_kb]
        ax1.bar(rounds, downlink_kb, color=C[0], edgecolor='black',
                linewidth=0.3, width=0.7, label='Downlink (Server→Tiles)', zorder=3)
        ax1.bar(rounds, uplink_kb, bottom=downlink_kb, color=C[2], edgecolor='black',
                linewidth=0.3, width=0.7, label='Uplink (Tiles→Server)', zorder=3)
        ax1.set_xlabel('Communication Round')
        ax1.set_ylabel('Data per Round (KB)')
        ax1.set_title('(a) Per-Round Communication', fontsize=9)
        _style_legend(ax1, fontsize=6, loc='upper right')

        # (b) Cumulative communication over rounds
        ax2.plot(rounds, cum_kb, color=C[4], marker=MARKERS[1], markevery=max(1, len(rounds)//8),
                 markersize=4.5, linewidth=1.5, zorder=3)
        ax2.fill_between(rounds, 0, cum_kb, color=C[4], alpha=0.08, zorder=2)
        total_mb = cum_kb[-1] / 1024
        ax2.axhline(y=cum_kb[-1], color=C[1], linestyle='--', linewidth=0.8, alpha=0.6)
        ax2.text(rounds[-1], cum_kb[-1] * 1.02, f'{total_mb:.1f} MB',
                 ha='right', va='bottom', fontsize=7, color=C[1])
        ax2.set_xlabel('Communication Round')
        ax2.set_ylabel('Cumulative Data (KB)')
        ax2.set_title('(b) Cumulative Communication', fontsize=9)

        # (c) Per-round energy breakdown
        energy_per_round = [m['total_energy'] * 1000 for m in round_metrics]
        cum_energy = np.cumsum(energy_per_round).tolist()
        ax3.bar(rounds, energy_per_round, color=C[3], edgecolor='black',
                linewidth=0.3, width=0.7, alpha=0.7, label='Per Round', zorder=3)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(rounds, cum_energy, color=C[1], marker=MARKERS[2],
                      markevery=max(1, len(rounds)//8), markersize=4,
                      linewidth=1.5, label='Cumulative', zorder=4)
        ax3.set_xlabel('Communication Round')
        ax3.set_ylabel('Energy per Round (mJ)')
        ax3_twin.set_ylabel('Cumulative Energy (mJ)', color=C[1])
        ax3_twin.tick_params(axis='y', labelcolor=C[1])
        ax3_twin.spines['right'].set_visible(True)
        ax3_twin.spines['right'].set_color(C[1])
        ax3.set_title('(c) NoC Energy Consumption', fontsize=9)
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        _style_legend(ax3, fontsize=6, loc='upper left',
                      **{'handles': lines1 + lines2, 'labels': labels1 + labels2})

        # (d) Convergence loss vs cumulative communication trade-off
        losses = [m['avg_client_loss'] for m in round_metrics]
        cum_mb = [c / 1024 for c in cum_kb]
        ax4.plot(cum_mb, losses, color=C[0], marker=MARKERS[0],
                 markevery=max(1, len(rounds)//8), markersize=4.5, linewidth=1.5, zorder=3)
        for i in [0, len(rounds)//2, -1]:
            ax4.annotate(f'R{rounds[i]}', (cum_mb[i], losses[i]),
                         textcoords='offset points', xytext=(5, 5), fontsize=6, color=C[7])
        ax4.set_xlabel('Cumulative Communication (MB)')
        ax4.set_ylabel('Training Loss')
        ax4.set_title('(d) Loss vs. Communication Cost', fontsize=9)
    else:
        # Fallback: display summary values when round_metrics not available
        labels = ['Uplink', 'Downlink']
        vals = [comm_summary['total_bytes_received'] / 1e6,
                comm_summary['total_bytes_sent'] / 1e6]
        bars = ax1.bar(labels, vals, color=[C[0], C[2]], edgecolor='black',
                       linewidth=0.5, width=0.45, zorder=3)
        _annotate_bars(ax1, bars, fmt='{:.2f}')
        ax1.set_ylabel('Data (MB)')
        ax1.set_title('(a) Communication Volume', fontsize=9)

        total = comm_summary['total_megabytes']
        bars_tot = ax2.bar(['Total'], [total], color=C[5], edgecolor='black',
                           linewidth=0.5, width=0.35, zorder=3)
        _annotate_bars(ax2, bars_tot, fmt='{:.2f}')
        ax2.set_ylabel('Data (MB)')
        ax2.set_title('(b) Total Communication', fontsize=9)

        lat = comm_summary['avg_packet_latency_ms']
        bars_lat = ax3.bar(['Packet Latency'], [lat], color=C[4], edgecolor='black',
                           linewidth=0.5, width=0.35, zorder=3)
        _annotate_bars(ax3, bars_lat, fmt='{:.3f}')
        ax3.set_ylabel('Latency (ms)')
        ax3.set_title('(c) Average Latency', fontsize=9)

        util = comm_summary['bandwidth_utilization'] * 100
        ax4.barh(['BW Util.'], [100], color='#eeeeee', edgecolor='black',
                 linewidth=0.5, height=0.35, zorder=2)
        ax4.barh(['BW Util.'], [util], color=C[3], edgecolor='black',
                 linewidth=0.5, height=0.35, zorder=3)
        ax4.set_xlim(0, 105)
        ax4.set_xlabel('Utilization (%)')
        ax4.set_title(f'(d) NoC BW Utilization: {util:.1f}%', fontsize=9)

    _add_reference_note(fig, 'noc_traffic')

    if save_path:
        _save(fig, save_path, 'noc_metrics')


def create_summary_dashboard(all_metrics, save_path=None):
    """Summary dashboard combining key metrics across all dimensions."""
    fig = plt.figure(figsize=(7.16, 6.0))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)

    # Convergence
    ax1 = fig.add_subplot(gs[0, :2])
    rounds = [m['round'] + 1 for m in all_metrics['round_metrics']]
    losses = [m['avg_client_loss'] for m in all_metrics['round_metrics']]
    lo = [m['min_client_loss'] for m in all_metrics['round_metrics']]
    hi = [m['max_client_loss'] for m in all_metrics['round_metrics']]
    ax1.plot(rounds, losses, color=C[0], marker='o', markersize=3.5, zorder=3)
    ax1.fill_between(rounds, lo, hi, color=C[0], alpha=0.10, zorder=2)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('(a) FL Convergence', fontsize=9)

    # SNR
    ax2 = fig.add_subplot(gs[0, 2])
    snr = all_metrics['snr_metrics']
    methods = ['No RIS', 'Random', 'FL', 'Optimal']
    vals = [snr['snr_no_ris_mean'], snr['snr_random_ris_mean'],
            snr['snr_optimized_ris_mean'], snr['snr_optimal_mean']]
    bars = ax2.bar(range(4), vals, color=[C[7], C[3], C[0], C[2]],
                   edgecolor='black', linewidth=0.4, zorder=3)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=6)
    ax2.set_ylabel('SNR (dB)')
    ax2.set_title('(b) SNR Comparison', fontsize=9)

    # Communication
    ax3 = fig.add_subplot(gs[1, 0])
    comm = np.cumsum([m['total_bytes'] / 1024 for m in all_metrics['round_metrics']]).tolist()
    ax3.plot(rounds, comm, color=C[4], marker='s', markersize=3, zorder=3)
    ax3.fill_between(rounds, 0, comm, color=C[4], alpha=0.08, zorder=2)
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Cumulative KB')
    ax3.set_title('(c) Communication', fontsize=9)

    # Energy
    ax4 = fig.add_subplot(gs[1, 1])
    eng = np.cumsum([m['total_energy'] * 1000 for m in all_metrics['round_metrics']]).tolist()
    ax4.plot(rounds, eng, color=C[1], marker='d', markersize=3, zorder=3)
    ax4.fill_between(rounds, 0, eng, color=C[1], alpha=0.08, zorder=2)
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Cumulative mJ')
    ax4.set_title('(d) Energy', fontsize=9)

    # Per-tile
    ax5 = fig.add_subplot(gs[1, 2])
    n_c = len(all_metrics['round_metrics'][0]['client_metrics'])
    for i in range(n_c):
        cl = [m['client_metrics'][i]['avg_loss'] for m in all_metrics['round_metrics']]
        ax5.plot(rounds, cl, linewidth=0.8, alpha=0.65, color=C[i % len(C)])
    ax5.set_xlabel('Round')
    ax5.set_ylabel('Loss (MSE)')
    ax5.set_title('(e) Per-Tile Loss', fontsize=9)
    _add_reference_note(fig, 'fl_vs_centralized')

    if save_path:
        _save(fig, save_path, 'summary_dashboard')
