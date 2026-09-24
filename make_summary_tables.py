#!/usr/bin/env python
"""Export Tier 0/1 results from corrected, provenance-carrying JSON only.

Legacy circular-panel results and log-only ablations are intentionally not read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_link_level import write_json

LABELS = {
    'no_ris': 'No RIS', 'random_ris': 'Random phases', 'local_mrc': 'Local noisy-CSI MRC',
    'ao': 'Projected-gradient control', 'sca': 'Surrogate control', 'genie': 'Perfect-CSI bound',
    'random_max': 'Best observed probe', 'lmmse_mrc': 'LMMSE + MRC',
    'local_linear_mrc': 'Local linear estimate + MRC', 'full_probe_ls_mrc': 'Full-probe LS + MRC',
    'oracle_mrc': 'Perfect-CSI bound (oracle)', 'fed_ris': 'FedAvg (validation selected)',
    'centralized_dl': 'Central (total-step budget)', 'centralized_client_budget': 'Central (client-step budget)',
    'local_only': 'Local models', 'fed_1round': 'One-round FedAvg', 'fed_5round': 'Five-round FedAvg',
}


def cell(stat, digits=2):
    mean, ci = stat['mean'], stat['ci95_half_width']
    return f'{mean:.{digits}f}' + (f' $\\pm$ {ci:.{digits}f}' if ci is not None else ' (one seed)')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--link-dir', default='results/link_level_corrected')
    ap.add_argument('--pilot-dir', default='results/pilot_limited')
    ap.add_argument('--pilot-extensions', default='results/pilot_limited_extended')
    ap.add_argument('--output', default='results/tier01_report')
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tex, findings, md = [], [], ['# Tier 0 / Tier 1 corrected results', '']
    manifest = {'inputs': [], 'legacy_results_used': False}
    import hashlib
    def read(p):
        manifest['inputs'].append({'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
        return json.loads(p.read_text())
    link_path = Path(args.link_dir) / 'summary.json'
    if link_path.exists():
        s = read(link_path)
        n = len(s['seeds'])
        tex += [r'\begin{table}[t]\centering\small',
                rf'\caption{{Supplied-CSI diagnostic: channel power gain in dB, {n} seeds. Intervals are 95\% Student-$t$ intervals over seeds.}}',
                r'\begin{tabular}{lr}\toprule Scheme & Gain (dB)\\\midrule']
        md += [f'Supplied-CSI audit: {n} independent seeds.', '', '| Scheme | Channel gain (dB), 95% CI |', '|---|---:|']
        for k, m in s['metrics'].items():
            v = cell(m['mean_gain_db'])
            tex.append(f'{LABELS[k]} & {v} ' + r'\\')
            md.append(f'| {LABELS[k]} | {v.replace("$", "").replace(chr(92)+"pm", "±")} |')
        tex += [r'\bottomrule\end{tabular}\end{table}']
        # Physical diagnostics from every individual result, with provenance.
        records = [read(Path(args.link_dir) / f'seed_{seed}' / 'link_level_results.json')
                   for seed in s['seeds']]
        for r in records:
            if r['meta'].get('schema') != 'contiguous-siso-v1':
                raise ValueError('Refusing legacy link results')
            if r['meta'].get('is_quick_run'):
                raise ValueError('Smoke results cannot be exported as paper evidence')
        spreads = [float(np.ptp(10*np.log10(r['meta']['tile_mean_cascade_power']))) for r in records]
        slopes = [r['array_scaling']['reflected_only_snr_db']['genie'][-1] -
                  r['array_scaling']['reflected_only_snr_db']['genie'][0] for r in records]
        manifest['geometry_diagnostics'] = {'tile_mean_power_spread_db_by_seed': spreads,
                                           'reflected_gain_64_to_1024_db_by_seed': slopes}
        md += ['', f'Tile-average power spread by seed (dB): {spreads}.',
               f'Reflected-only oracle gain from 64 to 1024 elements (dB): {slopes}.', '']
        findings.append(
            f'The corrected aperture has a tile-average channel-power spread of '
            f'{min(spreads):.2f}--{max(spreads):.2f}\\,dB across the independent seeds. '
            f'Its reflected-only oracle gain from 64 to 1024 elements is '
            f'{min(slopes):.2f}--{max(slopes):.2f}\\,dB, compared with '
            r'$20\log_{10}(1024/64)=24.08$\,dB. The previous geometric saturation claim does not survive.')
    pilot_path = Path(args.pilot_dir) / 'summary.json'
    if pilot_path.exists():
        s = read(pilot_path)
        if s['schema'] != 'passive-feedback-v1':
            raise ValueError('Wrong pilot observation model')
        records = []
        for group, g in s['groups'].items():
            for seed in g['seeds']:
                base = read(Path(args.pilot_dir) / f'seed_{seed}' / group / 'results.json')
                extended_path = Path(args.pilot_extensions) / f'seed_{seed}' / group / 'results.json'
                if base['training'].get('stopping_reason') == 'budget_exhausted' and extended_path.exists():
                    ext = read(extended_path)
                    same_fields = ('seed', 'num_probes', 'tx_power_dbm', 'noise_power_dbm', 'geometry',
                                   'scene', 'sample_counts_train_validation_test', 'input_dim',
                                   'hidden_dim', 'num_layers', 'model_type', 'dropout', 'codebook_sha256')
                    if any(base['meta'][k] != ext['meta'][k] for k in same_fields):
                        raise ValueError(f'Extended case changes the experiment: {extended_path}')
                    old = base['training']['validation_losses']
                    new = ext['training']['validation_losses']
                    if len(new) < len(old) or not np.allclose(old, new[:len(old)], rtol=1e-5, atol=1e-7):
                        raise ValueError(f'Extended case does not reproduce the first budget: {extended_path}')
                    if ext['training']['fl_rounds'] > base['training']['fl_rounds']:
                        base = ext  # Selection depends on budget exhaustion, never test scores.
                records.append(base)
        from run_pilot_limited import summarize
        s = summarize(records)
        write_json(out / 'pilot_summary.json', s)
        for group, g in s['groups'].items():
            tex += [r'\begin{table*}[t]\centering\small',
                    rf'\caption{{Passive pilot experiment {group.replace("_", ", ")}: {len(g["seeds"])} seeds; 95\% intervals. Net rate uses the scheme-specific probe count and declared coherence length.}}',
                    r'\begin{tabular}{lrrr}\toprule Scheme & Received SNR (dB) & Net rate (bit/s/Hz) & Paired net-rate gap to local linear\\\midrule']
            md += ['', f'## {group}', '', '| Scheme | SNR (dB) | Net rate | Paired gap to local linear |', '|---|---:|---:|---:|']
            for k, m in g['scores'].items():
                values = [cell(m[x], 3) for x in ('mean_received_snr_db', 'net_spectral_efficiency', 'net_rate_gap_to_local_linear_mrc')]
                tex.append(LABELS[k] + ' & ' + ' & '.join(values) + r'\\')
                md.append('| ' + LABELS[k] + ' | ' + ' | '.join(v.replace('$', '').replace('\\pm','±') for v in values) + ' |')
            tex += [r'\bottomrule\end{tabular}\end{table*}']
            if 'fed_ris' in g['scores']:
                fed = g['scores']['fed_ris']
                gap = fed['net_rate_gap_to_local_linear_mrc']
                findings.append(
                    f'For {group.replace("_", ", ")}, FedAvg achieves '
                    f'{cell(fed["net_spectral_efficiency"], 3)} bit/s/Hz after pilot overhead. '
                    'Its paired difference from the local linear estimator is '
                    f'{cell(gap, 3)} bit/s/Hz. '
                    'These intervals reflect seed variation under the specified model only.')
        if any(r['meta']['arguments'].get('quick') for r in records):
            raise ValueError('Smoke results cannot be exported as paper evidence')
        states = [{"seed": r['meta']['seed'], "M": r['meta']['num_probes'],
                   "pt_dbm": r['meta']['tx_power_dbm'],
                   "rounds": r['training'].get('fl_rounds'),
                   "best_round": r['training'].get('best_round'),
                   "stopping_reason": r['training'].get('stopping_reason'),
                   "fl_bytes": r['training'].get('fl_total_communication_bytes'),
                   "pooled_dataset_payload_reference_bytes":
                       r['accounting']['centralized_training_label_bytes'] +
                       r['accounting']['centralized_training_feedback_bytes'],
                   "additional_centralized_training_upload_bytes": 0}
                  for r in records]
        manifest['training_runs'] = states
        trained = [r for r in states if r['rounds'] is not None]
        if trained:
            rounds = [r['rounds'] for r in trained]
            ratios = [r['fl_bytes']/r['pooled_dataset_payload_reference_bytes'] for r in trained]
            exhausted = sum(r['stopping_reason'] == 'budget_exhausted' for r in trained)
            findings.append(
                f'The recorded FL runs spend {min(rounds)}--{max(rounds)} rounds; '
                f'{exhausted} reach the budget without the declared validation plateau. '
                f'Model-transfer payload is {min(ratios):.1f}--{max(ratios):.1f} times '
                'the size of the pooled training dataset. This dataset size is a reference, '
                'not an extra transfer required by the passive centralized control: the '
                'controller already receives the acquisition feedback. Distributed training '
                'additionally requires data delivery to the tiles. No communication advantage '
                'is established for federation.')
        accounting_audit = []
        for r in records:
            meta = r['meta']
            train, valid, _ = meta['sample_counts_train_validation_test']
            g = meta['geometry']
            tiles = g['tile_rows'] * g['tile_cols']
            ne = g['pixel_rows'] * g['pixel_cols']
            nt = tiles * ne
            m = meta['num_probes']
            accounting_audit.append({
                'seed': meta['seed'], 'num_probes': m, 'tx_power_dbm': meta['tx_power_dbm'],
                'common_receiver_feedback_training_validation_bytes': (train + valid) * (m + nt + 1) * 8,
                'additional_centralized_training_upload_bytes': 0,
                'distributed_training_validation_data_endpoint_bytes': (train + valid) * tiles * (m + ne + 1) * 8,
                'federated_model_endpoint_bytes': r['training'].get('fl_total_communication_bytes', 0),
                'local_model_exchange_bytes': 0,
                'distributed_online_feedback_endpoint_bytes_per_block': tiles * m * 8,
                'central_online_phase_command_bytes_float32': nt * 4,
                'best_probe_index_command_bytes': int(np.ceil(np.log2(max(m, 2)) / 8)),
                'scope': 'endpoint payloads; feedback timing and physical broadcast routing excluded',
            })
        write_json(out / 'accounting_audit.json', accounting_audit)
        md += ['', 'Training stopping and payload accounting:', '', '```json', json.dumps(states, indent=2), '```']
        plot_pilot(s, records, out)
    if not tex:
        raise SystemExit('No corrected results found; run the corrected experiments first.')
    (out / 'tables.tex').write_text('\n'.join(tex) + '\n')
    (out / 'findings.tex').write_text('\n\n'.join(findings) + '\n')
    (out / 'REPORT.md').write_text('\n'.join(md) + '\n')
    write_json(out / 'manifest.json', manifest)
    print(f'Wrote {out / "REPORT.md"} and tables.tex')


def plot_pilot(summary, records, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = list(summary['groups'])
    schemes = ['random_max', 'lmmse_mrc', 'local_linear_mrc', 'full_probe_ls_mrc',
               'centralized_dl', 'centralized_client_budget', 'local_only', 'fed_1round', 'fed_5round', 'fed_ris']
    fig, axes = plt.subplots(1, len(groups), figsize=(7.16, 4.3), squeeze=False, sharey=True, sharex=True)
    keys = [k for k in schemes if any(k in summary['groups'][g]['scores'] for g in groups)]
    for i, (ax, group) in enumerate(zip(axes[0], groups)):
        metrics = summary['groups'][group]['scores']
        vals = [metrics[k]['net_spectral_efficiency']['mean'] if k in metrics else np.nan for k in keys]
        ci = [metrics[k]['net_spectral_efficiency']['ci95_half_width'] or 0 if k in metrics else 0 for k in keys]
        ax.barh(range(len(keys)), vals, xerr=ci, color='#4079a8', capsize=3)
        ax.set_yticks(range(len(keys)), [LABELS[k] for k in keys], fontsize=7.5)
        ax.tick_params(axis='y', labelleft=(i == 0))
        ax.tick_params(axis='x', labelsize=7)
        ax.set_xlabel('Net rate (bit/s/Hz); 95% CI', fontsize=8)
        ax.set_title(group.replace('_', ', '), fontsize=9)
    axes[0, 0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / 'pilot_comparison.pdf')
    fig.savefig(out / 'pilot_comparison.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in records:
        vals = r['training'].get('validation_losses', [])
        if vals:
            ax.plot(np.arange(1,len(vals)+1), vals, label=f"M={r['meta']['num_probes']}, seed={r['meta']['seed']}")
    ax.set_xlabel('Federated round')
    ax.set_ylabel('Held-out validation loss (measured training channels)')
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'validation_curves.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
