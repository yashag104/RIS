# Contiguous RIS: supplied-CSI audit and pilot-limited control

The active manuscript is [paper.tex](paper.tex). It has been revised around **passive receiver pilot feedback**, not a claim that learning is necessary for known-channel SISO phase alignment.

The old circular-panel simulation, paper source and PDFs are historical artifacts. The manuscript and PDFs are preserved under [docs/archive](docs/archive). Old results under `results/link_level` and the system experiment logs are **not evidence for the corrected paper**. See [the Tier 0/1 audit](docs/TIER01_CORRECTIONS.md) and [Tier 2 corrections](docs/TIER2_CORRECTIONS.md) for scope and remaining research limitations.

## Reproduce

Use the project virtual environment (or install `requirements.txt`). Run from this directory:

```bash
# Full contiguous 16-tile classical audit: five seeds, 600 test scenes per seed
.venv/bin/python run_link_level.py --skip-training --output results/link_level_tier2

# Passive pilot experiment: five seeds, M=16 and 64, up to 300 FL rounds
.venv/bin/python run_pilot_limited.py

# Five-seed pilot architecture diagnostic: shared data, 500 updates/model
.venv/bin/python run_pilot_ablation.py

# Analytical 128/256 Gb/s traffic scenarios, including Butterfly + RingAllReduce
.venv/bin/python run_noc_study.py

# Tables, confidence intervals, plots, and hash manifests from corrected JSON
.venv/bin/python make_summary_tables.py --link-dir results/link_level_tier2
.venv/bin/python make_tier2_report.py

# Focused physical, statistical, and information-boundary regression tests
.venv/bin/python -m pytest -q test_tier01.py test_tier2.py test_noc.py test_link_metrics.py
```

`--quick` on either runner writes separate smoke artifacts. These cannot replace a full run. `--seed` / `--seeds` restrict the seed set on the supplied-CSI runner; `--seeds` does so on the pilot runner. Single-seed output has no confidence interval. Set `MPLCONFIGDIR=/tmp/ris-matplotlib` if your home config directory is read-only.

The original GAT can be retrained under corrected geometry with `run_link_level.py --model GNN --rounds 100`. Both centralized update budgets and unfederated local models are included. This is expensive on CPU and is not required to establish the exact SISO closed form. Pilot results use a separately identified width-128 MLP; they do not validate the old GAT claims.

## What the experiment actually measures

- One 32×32 element aperture at half-wavelength spacing, approximately 17.14 cm per side at 28 GHz. Shared scattering fields are generated before slicing into tiles.
- With supplied CSI, local MRC is exact for perfect estimates and has no model-training traffic. This diagnostic does not implement passive channel acquisition.
- With passive probes, the receiver measures `M` complex responses and sends feedback. Neural inference gets only these responses and tile coordinates, with no true direct-path phase or per-element test CSI.
- Offline training labels cost an additional `N+1` DFT probes per training and validation scene. They are noisy estimates, not free simulator truth.
- Comparisons include best observed probe, LMMSE/MRC, independently fitted local linear estimators, full-probe LS/MRC, one/five-round FL, local neural models, and centralized learning at per-client and total-network update budgets.
- Validation selects checkpoints. A recorded plateau is not proof of convergence. Budget exhaustion remains explicit.
- Float32 model payloads are counted because that is what the numerical averaging uses. The old INT8 discount is not claimed. Payload traffic is not a circuit energy measurement.

There is no guarantee that federation wins. The paired comparisons in `results/tier01_report/REPORT.md` determine whether it has a useful operating point. Improvements from adding an RIS are not attributed to the learning method.

The new architecture diagnostic and analytical interconnect sweep are reported in `results/tier2_report/REPORT.md`. Architecture comparisons use the same physical/pilot scale as the headline experiment, with a separately declared finite update budget. They do not establish convergence of the original supplied-CSI GAT, which remains untrained on corrected geometry. The supplied-CSI table contains no neural rows.

NoC latency is a conditional bottleneck-serialization estimate; technology-node, cycle-accurate, area, power, and energy claims are withdrawn. No Noxim validation or circuit implementation is claimed. The manuscript uses a generic IEEE journal layout and is positioned as wireless-systems simulation, with venue scope notes in the Tier 2 audit. It has not been compiled or checked against a selected venue's page limit.

Other experiment entry points (`main.py`, `run_all_experiments.py`, legacy system-level suites) remain historical research tools and are not manuscript evidence.
