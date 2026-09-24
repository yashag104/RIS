# Tier 0 / Tier 1 correction audit

Scope: review items 1–9. User selected pilot-limited CSI as the revised research question. This document records items 1–9. Subsequent Tier 2 corrections and venue scope are recorded in [TIER2_CORRECTIONS.md](TIER2_CORRECTIONS.md).

| Review item | Correction | Evidence / remaining limit |
|---|---|---|
| 1. Metres-apart tiles | `run_link_level.py` generates one aperture with `src/surface_channel.py`; the full field is sliced afterwards. No legacy cache is read. | Geometry tests verify pitch, adjacent tile boundaries, aperture, unique positions, spherical LoS phases. Old results are superseded. A 17 cm panel is not asserted to be one silicon die. |
| 2. Separable SISO | Added the exact local-MRC control and proof. Main experiment now observes `M < N+1` noisy receiver probes. | Under-determined acquisition is nontrivial; the known-CSI objective remains separable. This is not proof that FL is useful. |
| 3. Losing / selective baselines | All completed comparisons are exported. Absolute and paired gaps are stored. Withdrawn old headline gains and superiority claims. | The revised method must earn an advantage. Negative results remain in the report and manuscript. |
| 4. Passive CSI sensing | Receiver measures probes; passive aperture only applies configurations. No oracle phase or true test CSI enters learned inference. | Supplied-CSI diagnostic abstracts acquisition and would need receiver feedback or explicit sensing hardware. References: Zheng et al., arXiv:2110.01292; Taha et al., arXiv:1904.10136. |
| 5. False scaling saturation / anchor | Shared contiguous geometry. Reflected-only oracle power anchors the N² line; total power including the direct path is separate. | Model is synthetic; no general deployment saturation conclusion is drawn. |
| 6. Robustness regression to mean | Report fraction of oracle power and gap to local MRC along with absolute SNR/BER/rate. | No claim that a smaller clean-to-noisy drop demonstrates denoising. Epsilon=1 is 0 dB signal-to-error ratio, not pure noise. |
| 7. Unmatched optimization | Save exact per-client and total optimizer steps. Central controls match each budget. Same initialization, Adam, batch size, fixed learning rate, objective, and validation split. | FedAvg resets optimizer state at broadcasts; central/local optimizers persist. The gap is not attributed solely to locality. Uniform minibatch sampling uses ceil(S/B) steps per epoch-equivalent. |
| 8. Unconverged model: pilot MLP addressed; original GAT open | Initial 100-round budget, extended to 300 when exhausted; independent validation, explicit stopping/selection records, local/central validation selection. One/five-round checkpoints are scored too. | A plateau is an operational stopping criterion, not an optimum. Old 20-round GAT numbers are withdrawn. Pilot MLP evidence does not stand in for corrected GAT retraining. |
| 9. Traffic / no advantage | Count float32 exchanges, training-data delivery, zero-round local methods, and fixed-round FL. The passive controller already has receiver feedback: additional centralized training upload is zero. Pooled dataset size is only a reference. Charge training-label probes and per-block feedback separately. | Crossover is analytic only; no measured benefit is claimed there. Same-owner tiles have no demonstrated privacy threat model. |

## Saved artifacts

- `results/link_level_corrected/seed_*/link_level_results.json`: completed classical-only supplied-CSI results (`training.status=not_run`), source hashes, geometry, scene and seed. No neural rows are present.
- `results/pilot_limited/seed_*/M*_Pt*/results.json`: pilot budget, training-label acquisition, all controls, validation histories, step counts, model size and payload accounting.
- `summary.json` in each directory: Student-t 95% intervals across independent runs; paired gaps use the same seed.
- `results/tier01_report/REPORT.md`, `tables.tex`, `manifest.json`: generated result report and manuscript tables with input hashes. The old mixed-scale architecture ablation is excluded.

Quick runs use different directories and are not paper evidence. Main result code does not read old caches or `experiments.log`.

## What must not be inferred

The paper is a working controlled simulation study, not a demonstrated state-of-the-art FL method. Limited probes can make inference hard while the best solution remains a local statistical estimator. All tiles receive the same feedback; shared observations and similar tile statistics can make federation unnecessary. Training labels require expensive full-probe acquisition. Pilot net rate excludes feedback delay, training amortization and common receiver demodulation overhead; those assumptions are stated in the manuscript.

The coherent-combining, channel and sensing regressions validate consistency of the implemented assumptions. They do not validate mutual coupling, angle-dependent element patterns, measured hardware power, a technology node, or a circuit implementation.

## Completed regeneration (2026-09-23)

The supplied-CSI audit covers five independent seeds on the full 1024-element aperture. Local perfect-estimate MRC equals the oracle. Tile-average power spread is 0.06–0.18 dB; reflected-only gain from 64 to 1024 elements is 23.99–24.13 dB (N² prediction: 24.08 dB).

The passive experiment covers five seeds at each of M=16 and M=64, with 600 training, 200 validation and 600 test scenes per seed. The two 64-probe cases that exhausted 100 rounds were extended to a 300-round cap and stopped at 122/123 rounds. The exporter verifies that extensions reproduce the original validation trajectory before replacing an exhausted case. All retained cases meet the declared validation plateau rule; no mathematical convergence claim is made.

**Research blocker remains:** the federated MLP loses in net rate to best-probe selection, local linear estimation, LMMSE, and independent local neural models at both tested pilot budgets. At M=64, net rate is 2.323 ± 0.057 bit/s/Hz for FedAvg, 3.566 ± 0.132 for local linear estimation, and 3.984 ± 0.127 for LMMSE (95% seed-level intervals). The implementation and claims have been corrected; a beneficial federated method has not been demonstrated.

`results/tier01_report/accounting_audit.json` separates common receiver acquisition feedback, data delivery to tiles, model exchange, and online command/feedback payloads. In this acquisition architecture, additional centralized training upload is zero. FL model payload is 26–150 times the pooled dataset size, which is a reference size rather than a required centralized transfer.

Validation: 51 physical/link/regression checks passed; the final focused suite passed all 11 tests; Ruff and whitespace checks passed. No LaTeX compiler is installed in this environment, so the revised TeX source and generated figure PDFs are delivered without claiming a compiled manuscript PDF.

The original full studies ran uninterrupted while source development continued. Their execution source manifests are pinned to the first case in each process. Original disk-at-case-start hashes are retained alongside them; `results/provenance_manifest_correction.json` records the metadata-only correction. Subsequent runners pin source manifests automatically. Extended cases carry their own execution manifest.

The original supplied-CSI GAT has **not** been retrained on corrected geometry. Tier 1 #8 remains open for that model. The new finite-budget pilot architecture diagnostic also does not close it.
