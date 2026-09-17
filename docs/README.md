# Documentation

| Document | What it is for |
|---|---|
| [`NOVELTY.md`](NOVELTY.md) | The six contributions, each tied to the file and experiment that backs it, plus the honest limits |
| [`PAPER_RESULTS.md`](PAPER_RESULTS.md) | The six figures that should go in the paper, what each one argues, and how to regenerate them |
| [`BASELINE_COMPARISON.md`](BASELINE_COMPARISON.md) | Measured comparison against every implemented baseline, and a positional comparison with the published literature |
| [`system_flow.md`](system_flow.md) | Narrative walk through the whole pipeline with Mermaid diagrams — the prose companion to the rendered `system_architecture.pdf` |

## Regenerating everything

```bash
python run_link_level.py          # six link-level results + figures + JSON
python -m utils.system_diagram    # the end-to-end system figure
python make_summary_tables.py --write   # refresh every table in these documents
```

The tables in these documents live between `<!-- BEGIN GENERATED TABLES -->` and
`<!-- END GENERATED TABLES -->` markers and are written by
`make_summary_tables.py` from `results/link_level/link_level_results.json`. Edit
the prose freely; do not hand-edit inside the markers, because the next
regeneration will overwrite it.

## Figures

| File | Figure |
|---|---|
| `results/figures/system_architecture.pdf` | Fig. 0 — end-to-end system (vector, for the paper) |
| `results/link_level/fig1_ber_vs_snr.pdf` | Fig. 1 — BER vs transmit SNR, QPSK and 16-QAM |
| `results/link_level/fig2_spectral_efficiency.pdf` | Fig. 2 — ergodic spectral efficiency and power saving |
| `results/link_level/fig3_outage_probability.pdf` | Fig. 3 — outage probability |
| `results/link_level/fig4_hardware_impairments.pdf` | Fig. 4 — quantization and phase jitter |
| `results/link_level/fig5_csi_robustness.pdf` | Fig. 5 — robustness to imperfect CSI |
| `results/link_level/fig6_array_scaling.pdf` | Fig. 6 — array-gain scaling and SNR CDF |
